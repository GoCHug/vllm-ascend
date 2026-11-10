# DeepSeek V4.1 多模态（图像）通路详解

> **本文讲解 DeepSeek V4.1-Flash 在 Ascend NPU 上的图像全链路：ViT 塔 → Aligner 空间合并 → 图像 span 拼装 → 嵌入覆盖写 → VL 路由偏置。重点是每个 token 的形状、每行 `input_ids` 的归属、以及视觉与文本子系统在何处相互让路。**

> **代码版本冻结声明**：本文基于本仓库当前 checkout 的 `vllm_ascend/models/deepseek_v41/` 包与根目录 `vllm/`（上游 vLLM main）源码，行号以该 checkout 为准。Ascend 视觉实现只有两个文件：[vision.py](../vision.py)（127 行）与 [vl_model.py](../vl_model.py)（336 行），官方参考在同目录 `reference/`。文中 `Lxxx` 指紧邻链接所标文件的行号。

> **统一模型配置**（[0_DeepSeek-V4.1-Flash-1017_config.json](0_DeepSeek-V4.1-Flash-1017_config.json)；`vision_config` 字段运行时被拍平成 `vision_*` 别名）：

| checkpoint 字段 | 值 | 运行时别名 | 作用 |
|---|---|---|---|
| `num_hidden_layers` | 32 | `vision_n_layers` | ViT 层数；**0 即关闭视觉** |
| `hidden_size` | 1024 | `vision_dim` | 塔内宽度 = 每 patch 输出宽度 |
| `num_attention_heads` | 16 | `vision_n_heads` | 头数，`head_dim = 1024//16 = 64` |
| `intermediate_size` | 2816 | `vision_inter_dim` | MLP 中间宽（`w1` 出 `2×2816`） |
| `patch_size` | 14 | `vision_patch_size` | 每 patch `14×14` 像素 → 展平 `588` |
| `rope_theta` | 10000 | `vision_rope_theta` | 2D RoPE 基座 |
| `downsample_ratio` | 3 | `vision_downsample_ratio` | Aligner 的 `3×3` 合并率 |
| `max_image_tokens` | 1024 | `vision_max_n_token` | **单图** span 预算上限 |
| `min_pixels` | 295936（`544²`） | `vision_min_pixels` | 小图放大阈值 |
| `max_wh_ratio` | `null` | `vision_max_wh_ratio` | 宽高比钳制；空=不裁 |
| `image_token_id` | 129264 | `image_sentinel_base_id` | **span 内每个位置都填它** |
| （派生） | 129265 | `image_pad_token_id` | `base_id + 1` |
| `text_config.hidden_size` | 5120 | `hidden_size` | Aligner 输出宽度 |

> 两个哨兵别名由 [utils.py:L183-L184](../../../utils.py#L183) 注入；缺了它们路由与 Engram 会静默走错分支（§6）。

---

## 一、总览：从 patches 到 hidden states 的五段流水线

V4.1 的图像不是独立编码服务，而是**把图变成一段长度已知的普通 token**，让文本模型无差别吃进去。

> **类比**：像把一张手绘图喂给只读盲文的读者——① **预处理**＝裁成 `14×14` 的邮票贴满整张纸；② **ViT 塔**＝只看邮票的排版员，让**所有格子互相看**（双向、无因果），并用 **2D RoPE** 记住"第几行第几列"；③ **Aligner**＝把 `3×3` 相邻邮票糊成一格（分辨率 ÷3、语义升一档）再投影到 5120 维；④ **span 拼装**＝按阅读顺序排成一行字，每行末尾加"换行符"、首尾加"起止符"；⑤ **嵌入合并**＝把这段"图像句子"的向量**覆盖写**进文本嵌入。即：**图像在 LLM 眼里只是一段"id 相同、向量特殊、路由偏置特殊、Engram 免疫"的 token 区间。**

```
1 预处理（CPU，上游 mm_preprocess.py）
  PIL → max_wh_ratio 钳制 → min_pixels 放大 → patch 对齐 → safe_resize
  → 归一化[-1,1] → 切 patch → patches / vit_grid / llm_grid / types
        ▼
2 ViT 塔  vision.py:DeepseekV41ViT(L91)  整塔复制，无 TP/DP
  PatchEmbed(L31, 588→1024) → 32×Block(L78) → norm(L100, fp32)
  每 Block: norm1 → Attn(L40: wqkv→2D RoPE→MMEncoderAttention 双向)
            → 残差 → norm2 → MLP(L67: 1024→5632, SwiGLU, 2816→1024)
        ▼
3 Aligner  vision.py:DeepseekV41Aligner(L112)
  view→permute(2,0,1)→F.pad 补 r=3 倍数→F.unfold(3,stride=3)
  → (n_llm_h·n_llm_w, 9216) → w1 9216→5120 → GELU → w2 5120→5120
        ▼
4 span 拼装  vl_model.py:_build_image_span(L130)
  span_len = n_llm_h·(n_llm_w+1)+2；START/END/NEW_LINE 填可学习向量，
  IMAGE 槽按升序填 aligner 行
        ▼
5 嵌入合并  vl_model.py:embed_input_ids(L190)
  embed 全表查 → _merge_multimodal_embeddings 按掩码 **覆盖写**
        ▼
下游(§6)：MoE 用 bias_vl 选专家；Engram 把整段 span 从 n-gram 剔除
```

> **全程没有张量并行**：第 2、3 段在每张 NPU 上完整跑一遍（TP 组内是冗余计算、不是切分），视觉权重因此是"每卡一份"的固定开销（§8.3）。这与文本侧 `wq_b`/`wo_a`/`wo_b` 的 TP 切分不同，也是本文与 [3_deepseek_v41_attn.md](3_deepseek_v41_attn.md)"按 TP=8 折算本地 shape"风格有别之处。

---

## 二、ViT 塔逐层拆解

[vision.py](../vision.py) 的 docstring（[L3-L7](../vision.py#L3)）把两条硬约束写死：**replicated (no TP/DP sharding)**；从官方参考移植，**Weight names match the HF checkpoint so no renaming is needed at load time**。后者决定 §7 的写法：`patch_embed.proj`、`blocks.N.attn.wqkv`、`aligner.w1` 等原名保留，一 key 对一 key，可走 `default_weight_loader` 直通。

### 2.1 PatchEmbed：一次矩阵乘吃掉 `14×14×3`

`DeepseekV41PatchEmbed`（[L31-L37](../vision.py#L31)）= `nn.Linear(588, 1024)` 作用于 `x.flatten(1)`：`[N,3,14,14] → [N,588] → [N,1024]`（`N=n_vit_h·n_vit_w`，含 bias）。不用 `Conv2d` 是因为参考实现（[reference/vision.py:L37](reference/vision.py#L37)）就是 `nn.Linear`：数学等价但权重形状不同，必须跟随 checkpoint 的 `(1024,588)`。patch 顺序由预处理 `reshape(3,nh,p,nw,p).permute(1,3,0,2,4)` 决定（[mm_preprocess.py:L156-L160](../../../../../vllm/vllm/models/deepseek_v41/common/mm_preprocess.py)），即**行优先**——§3、§4 的隐含契约。

### 2.2 2D RoPE：只编 32 维，h/w 各 16 个频率

```python
@lru_cache(8)                                              # L21
def get_vision_cos_sin(n_h, n_w, dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)   # 行号
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)   # 列号
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float()
    freqs = (freqs * inv_freq).flatten(1)                   # [N, dim]
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)
```

关键量（[L96](../vision.py#L96)）：`rope_dim = vision_dim // vision_n_heads // 2 = 1024//16//2 = 32`。这个 `//2` **不是"只转一半维度"**——它是传给 `get_vision_cos_sin` 的 `dim`，即 `cos/sin` 的长度；`head_dim=64`，而 `ApplyRotaryEmb` 约定 `cos: [seq_len, head_size//2]`（[common.py:L143](../../../../../vllm/vllm/model_executor/layers/rotary_embedding/common.py#L143)），长度必须是 `64//2=32`，恰好相等。这 32 维 = 前 16 个行频率 `h·ω_k` + 后 16 个列频率 `w·ω_k`（`ω_k = theta^(-2k/32)`），返回 `[N,1,32]`，进 Attention 时被 `cos.squeeze(1)` 还原为 `[N,32]`（[L61](../vision.py#L61)）。

**具体转哪两维**：Neox 风格下 `x1,x2 = chunk(x,2,dim=-1)`（[common.py:L157-L158](../../../../../vllm/vllm/model_executor/layers/rotary_embedding/common.py#L157)），把 64 维 head **前后对半劈开**再配对——前 16 个频率管 `(0,32)…(15,47)`、后 16 个管 `(16,48)…(31,63)`，**不是**"前一半编行、后一半编列"的直观切法。

> **`lru_cache(8)`**：键为 `(n_h,n_w,dim,theta)`，最多记住 8 种图尺寸；淘汰只影响复用、不影响正确性。**全塔共用一组 cos/sin**：`DeepseekV41ViT.forward`（[L102-L109](../vision.py#L102)）在 32 个 block 循环外算一次、`to(device)` 一次再复用——ViT 没有层间位置漂移，每层看的是同一张网格。

### 2.3 双向注意力：一图一次调用，没有因果掩码

`DeepseekV41VisionAttention`（[L40-L64](../vision.py#L40)）：q/k/v 由 `wqkv`（`1024→3072`）一次投影、`chunk(3,-1)` 拆开（L45、L60）；q、k 摞成 `[2,N,16,64]` 一次做融合旋转：

```python
qk = self.apply_rotary_emb(torch.stack((q, k)), cos.squeeze(1), sin.squeeze(1))
q, k = qk.unbind()
o = self.attn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
return self.wo(o.squeeze(0).reshape(n, -1))                # L61-L64
```

旋转算子以 `ApplyRotaryEmb(enforce_enable=True, is_neox_style=True, enable_fp32_compute=True)` 构造（[L47-L51](../vision.py#L47)）：内部升 fp32 再回 bf16，避免乘加掉精度。**一次只算一张图**——`q.unsqueeze(0)` 使前导 batch 维恒为 1。`MMEncoderAttention._forward_sdpa`（[mm_encoder_attention.py:L529](../../../../../vllm/vllm/model_executor/layers/attention/mm_encoder_attention.py#L529)）签名里**没有 mask 参数**，只传 `q/k/v/scale/cu_seqlens`：没有掩码 ⇒ softmax 在全 `N` 个 patch 上归一化 ⇒ **彻底双向**，第 0 个 patch 能看到第 N-1 个——这正是图像与文本最本质的差别。不拼多图 batch 同因：拼 batch 要 padding 与 block-diagonal 掩码，而该算子只接受 `cu_seqlens`；当前选最保守的逐图调用（[vl_model.py:L157-L176](../vl_model.py#L157)），代价见 §8.4。

### 2.4 MLP 与 RMSNorm

`DeepseekV41VisionMLP`（[L67-L75](../vision.py#L67)）：`w1 = Linear(1024, 2×2816, bias=False)` 一次出 gate 与 up，`SiluAndMul` 内部 `chunk(2,-1)` 做 `silu(gate)*up`，`w2 = Linear(2816,1024, bias=False)`；与参考手写版（[reference/vision.py:L69-L71](reference/vision.py#L69)）数值等价，但省一次 HBM 往返。

`DeepseekV41VisionBlock`（[L78-L88](../vision.py#L78)）是标准 pre-norm：`x + attn(norm1(x))`，再 `+ mlp(norm2(...))`。三处 `dtype=torch.float32`（`norm1` L81、`norm2` L83、塔尾 `norm` L100）都是硬要求：`eps=1e-6` 比文本侧 `rms_norm_eps=1e-20` 大 14 个数量级（**视觉塔既有设计，不要"统一"**）；fp32 权重 + vLLM `ir.ops.rms_norm` 内部 `x.to(torch.float32)`（[ops/layernorm.py:L14-L21](../../../../../vllm/vllm/ir/ops/layernorm.py#L14)）与参考的 `x.float()` 一致；只有两处 norm，不是文本侧"norm + fp32 路由输入"的双输出写法。

---

## 三、Aligner：`3×3` 空间合并与 MLP 投影

一句话：**把 9 个相邻 patch 的 1024 维首尾相接成 9216 维，投到 5120，再过一层同宽 MLP**——同时完成降分辨率（token ÷9）与跨模态对齐（塔宽 → LLM 宽）。

```python
    def forward(self, x, n_vit_h, n_vit_w):                # vision.py L122-L127
        r = self.downsample_ratio                          # 3
        x = x.view(n_vit_h, n_vit_w, -1).permute(2, 0, 1)  # [1024, n_vit_h, n_vit_w]
        x = F.pad(x, (0, -n_vit_w % r, 0, -n_vit_h % r))   # 补到 r 的整数倍
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))                 # [n_llm_h*n_llm_w, 5120]
```

权重在 `__init__`（[L115-L120](../vision.py#L115)）：`in_dim = vision_dim * downsample_ratio**2 = 9216`，`w1 = Linear(9216, 5120)`、`w2 = Linear(5120, 5120)`（`hidden_size`）。

三个形状陷阱：`view` 依赖行优先（§2.1 保证）；`permute(2,0,1)` 得 `[C,H,W]` 让 `unfold` 连通道一起 unfold；`F.pad` 作用在 3-D 张量时从最后一维往前配 `(左,右,上,下)` ⇒ 对应 `(W 轴, H 轴)`。

### 3.1 `-n % r` 不是"负 padding"

1. **Python 里 `-n % r` 永远非负**（`-13 % 3 == 2`、`-12 % 3 == 0`），真义是 `pad = (r - n%r) % r`，即"**再补几格才凑成 r 的整数倍**"，整除时补 0、位置在右/下、填 0。
2. **补齐后的窗口数与 `llm_grid` 的 `ceil` 严格一致**：上游算 `n_llm = ceil(n_vit/r)`（[mm_preprocess.py:L71-L76](../../../../../vllm/vllm/models/deepseek_v41/common/mm_preprocess.py)），`pad` 后 `unfold` 窗口数 `= (n_vit+pad)/r = ceil(n_vit/r)`。**两者必须相等**，否则 span 长度与特征行数错位（少一行 → 掩码赋值形状不匹配而抛错；多一行 → 静默错位）。

**不整除例子**（`n_vit_h=10, n_vit_w=13, r=3`）：右补 `-13%3=2` 列、下补 `-10%3=2` 行 → `12×15` → 窗口 `4×5=20`；`llm_grid` 校验 `ceil(10/3)=4`、`ceil(13/3)=5` → `4×5=20` ✅。`F.pad` 补的 0 与预处理 `ImageOps.pad` 的灰边（≈0）语义一致（§9.3）。

> `F.unfold` 输出 `[1, 9216, n_llm_h·n_llm_w]`、窗口按行优先编号，`transpose(0,1)` 后每行是原图一个 `3×3` 单元、按阅读顺序。内部 9216 维是**通道优先**（`c₀ 的 9 个位置, c₁ 的 9 个位置, …`），这正是训练顺序（参考 [reference/vision.py:L116-L118](reference/vision.py#L116) 逐字相同）；改成"9 个 patch 各 1024 维依次排开"等于在 `w1` 前插一个置换矩阵，**不报错、只掉点**。

### 3.2 形状表：`n_vit → n_llm → span_len`

以 `r=3`、`482×482` 像素的图为例（`482/14 = 34.43 → ceil 35`）：

| 阶段 | 网格 | 数值 | 位置数 | 说明 |
|---|---|---|---|---|
| 像素（对齐后） | `H×W` | `490×490`（`35×14`） | — | patch 对齐 |
| ViT patch | `n_vit_h×n_vit_w` | `35×35` | `N=1225` | 塔内序列长度 |
| 补齐后 | `ceil(35/3)×3` | `36×36` | — | 右/下各补 1 |
| 窗口 | `n_llm_h×n_llm_w` | `12×12` | `144` | Aligner 输出行数 |
| Aligner 输出 | — | `[144,5120]` | 144 | 全是 IMAGE 槽 |
| span | — | `12·13+2` | **158** | 144 IMAGE + 12 NL + 2 边界 |

```
span_len = n_llm_h·(n_llm_w + 1) + 2
         = n_llm_h·n_llm_w (IMAGE 槽 = Aligner 行数)
         + n_llm_h         (每行一个 IMAGE_NEW_LINE)
         + 2               (IMAGE_START + IMAGE_END)
```

该式与上游 `types_sizes = n_llm_h*(n_llm_w+1)+2`（[mm_preprocess.py:L342](../../../../../vllm/vllm/models/deepseek_v41/common/mm_preprocess.py)）、参考 `num_image_tokens`（[reference/image_processor.py:L35-L36](reference/image_processor.py#L35)）**三处完全一致**。**数量守恒**：`n_llm_h·n_llm_w` 必须精确等于 span 里 `types==IMAGE` 的位置数；多一个少一个，`_build_image_span` 的掩码赋值会因形状不匹配抛错——这是全链路里**会自己叫出来**的错误。

---

## 四、图像 span 的拼装

### 4.1 四类 token

常量来自上游（[mm_preprocess.py:L43](../../../../../vllm/vllm/models/deepseek_v41/common/mm_preprocess.py)），与参考 [reference/image_processor.py:L22-L23](reference/image_processor.py#L22) 相同：

| 常量 | 值 | 含义 | 填入内容 |
|---|---|---|---|
| `IMAGE_START` | 0 | span 起点 | 可学习 `image_start` |
| `IMAGE` | 1 | Aligner 特征格 | aligner 第 k 行（阅读顺序） |
| `IMAGE_NEW_LINE` | 2 | 行结束标记 | 可学习 `image_newline` |
| `IMAGE_END` | 3 | span 终点 | 可学习 `image_end` |
| `TEXT` | -1 | 普通文本 | 常规词嵌入 |

三个向量在 [vl_model.py:L84-L90](../vl_model.py#L84) 创建为 `nn.Parameter(torch.empty(config.hidden_size, dtype=torch.float32))`——**fp32 存储、用时转 dtype**（`_build_image_span` 里 `.to(dtype)` 才降 bf16，L139-L141）。参考侧是 `nn.Parameter(torch.empty(args.dim))`（[reference/model.py:L1223-L1225](reference/model.py#L1223)）跟随全局默认，Ascend 显式钉 fp32 更稳。注意**没有 `image_pad` 参数**——上游已移除对齐 pad，测试 [test_deepseek_v41_vision.py:L182](../../../../tests/ut/models/test_deepseek_v41_vision.py#L182) 专门断言。

### 4.2 一个 3×4 grid 的逐步例子

取 `n_llm_h=3, n_llm_w=4`（对应 `n_vit_h=9, n_vit_w=12`，`126×168` 像素），`span_len = 3·(4+1)+2 = 17`。

**Step 1｜骨架** `[START] + ([IMAGE]×4 + [NEW_LINE])×3 + [END]`：

```
槽位   0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16
类型   ST   IM   IM   IM   IM   NL   IM   IM   IM   IM   NL   IM   IM   IM   IM   NL   END
       └┬┘  └───────┬────────┘  └┬┘  └───────┬────────┘  └┬┘  └───────┬────────┘  └┬┘  └┬┘
       起        第0行 4格       换行     第1行 4格       换行     第2行 4格       换行  终
```

**Step 2｜数 IMAGE 槽**：`{1,2,3,4} ∪ {6,7,8,9} ∪ {11,12,13,14}` 共 12 个 = `3×4` ✅ 正好等于 Aligner 的 12 行。

**Step 3｜映射**：Aligner 第 `k` 行（行优先，`k=i·n_llm_w+j`）→ 第 `k` 个 IMAGE 槽（下标升序），即 `单元(i,j) → 槽位下标 1 + i·(n_llm_w+1) + j`，于是 `k=0..11` 依次落到槽位 `1,2,3,4, 6,7,8,9, 11,12,13,14`。

**Step 4｜填值**（[vl_model.py:L136-L143](../vl_model.py#L136)）：

```python
span = image_embeds.new_empty(types.numel(), image_embeds.shape[-1])  # [17,5120] 未初始化
span[types == IMAGE_START]    = self.image_start.to(dtype)
span[types == IMAGE_END]      = self.image_end.to(dtype)
span[types == IMAGE_NEW_LINE] = self.image_newline.to(dtype)
span[types == IMAGE]          = image_embeds        # 12 行按升序落位
```

布尔掩码赋值**按目标下标升序**消费右侧张量，所以 `image_embeds` 的 12 行天然落到升序的 12 个槽位——这就是"阅读顺序"在实现层的全部含义。`new_empty` 同时意味着**所有位置都必须被四类之一覆盖**；若某个类型值不在四类中（例如沿用 V4 的其它 `perm`），该位置会拿到上一轮显存垃圾且**一路不报错**。

**Step 5｜`input_ids` 侧**：这 17 个位置**全是 `129264`**，`token_types` 才是 `0,1,1,1,1,2,1,1,1,1,2,1,1,1,1,2,3`。这种"只有类型能区分"的设计让 tokenizer 不用改、位置计算对图像与文本一视同仁；代价是**任何要区分图像的子系统都得自己再判一次 id**——直接引出 §5.4 与 §6。

> 多图时每图独立占一段 `span_len`，**两段直接相邻、无额外分隔 token**（上游 #56554 起）：区分靠 `mm_features` 的 `mm_position`（[model_runner.py:L3932-L3944](../../../worker/v1/model_runner.py#L3932) 构 `req_doc_ranges`）与"每个 span 各算各的双向窗口"（§8.5）。边界登记错了，两图就会在注意力里连成一片，**且不报错**。

---

## 五、嵌入合并与占位符

### 5.1 `get_placeholder_str` 与注册

[L40-L63](../vl_model.py#L40)：`@MULTIMODAL_REGISTRY.register_processor(DeepseekV4VLMultiModalProcessor, info=DeepseekV4VLProcessingInfo, dummy_inputs=DeepseekV4VLDummyInputsBuilder)` 装饰 `AscendDeepseekV41ForCausalLM`；`get_placeholder_str` 对 `modality=="image"` 返回 `IMAGE_PLACEHOLDER`（`"<｜deepseek_image｜>"`），`del i` 表示**所有图共用同一占位符、不靠字符串区分第 i 张**（靠顺序）。该 id 必须等于 `129264`（[mm_preprocess.py:L59-L68](../../../../../vllm/vllm/models/deepseek_v41/common/mm_preprocess.py) 断言），且 1 个占位符会被 `PromptReplacement` 换成 `span_len` 个 `image_token_id`（[mm_preprocess.py:L353-L376](../../../../../vllm/vllm/models/deepseek_v41/common/mm_preprocess.py)）——**用户写 1 个，模型看 158 个**。`get_mm_max_tokens_per_item` 返回 `{"image": vision_max_n_token}`（L261-L266），即调度器给每张图预留 1024 token。

### 5.2 `embed_multimodal`：多模态分支唯一入口

```python
def embed_multimodal(self, **kwargs):                     # L179
    image_input = self._parse_and_validate_image_input(**kwargs)   # L103：按 patches 判空
    if image_input is None or self.vision is None:
        return []                                         # 无图 / 塔未建
    return self._process_image_input(image_input["patches"], image_input["vit_grid"],
                                     image_input["llm_grid"], image_input["types"])
```

`_parse_and_validate_image_input`（[L103-L115](../vl_model.py#L103)）`patches is None` 就返回 `None`，否则打包四个字段。`_process_image_input`（[L145-L177](../vl_model.py#L145)）逐图串行：`patches` 先 `to(aligner.w1.weight.dtype)`（L153），两个游标 `vit_offset`（走 patch 维，步长 `n_vit_h·n_vit_w`）与 `span_offset`（走 types 维，步长 `span_len`）同步推进，每图调一次 `_encode_image`（`aligner(vision(...))`，L117-L128，断言塔存在）与一次 `_build_image_span`；`zip(..., strict=True)` 保证两 grid 长度一致。

### 5.3 `embed_input_ids`：先全表查，再覆盖写

```python
inputs_embeds = self.language_model.embed_input_ids(input_ids)     # 全表查一遍（L202）
return _merge_multimodal_embeddings(inputs_embeds=inputs_embeds,   # L205-L209
    multimodal_embeddings=multimodal_embeddings, is_multimodal=is_multimodal)
```

`_merge_multimodal_embeddings`（[utils.py:L717](../../../../../vllm/vllm/model_executor/models/utils.py#L717)）的语义是**覆盖**而非相加或插值：`inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype)`（L738），注释明确 in-place。后果：`129264` 是合法 in-vocab id，全表查得到的真实词向量若没被覆盖掉就会**静默**成为"占位符字面向量"；覆盖顺序必须与 `multimodal_embeddings` 拼接顺序严格一致；数量校验只在赋值抛 `RuntimeError` 时才反算（L739-L749）。

### 5.4 `requires_raw_input_tokens = True`

```python
# Engram history and vision MoE routing also consume the original token IDs.
requires_raw_input_tokens = True        # L54
```

上游定义：模型按**原始 id** 而非输入嵌入处理输入（[interfaces.py:L206-L210](../../../../../vllm/vllm/model_executor/models/interfaces.py#L206)）。置 True 后，即使走 `inputs_embeds` 通道，runner 也必须把 `input_ids` 一起带进模型，因为三处消费者都要它：**VL MoE 路由**（算子内判断图像行，依据 `id ∈ [base, base+5)`，types 不进 MoE）、**Engram n-gram**（哈希历史基于 id 序列，必须剔除图像哨兵）、**span 识别**（`image_sentinel_mask` 是纯 id 比较）。

> **嵌入被图像覆盖了，id 没有被覆盖**：在下游任何"看 id"的子系统眼里，这段区间永远是 17（或 158）个 `129264`——"嵌入与 id 分离"是全文主线。另：[`set_embedding_num_tokens_padded`](../vl_model.py#L211)（L211-L214）把 padding 后长度转发给词嵌入，供 PCP token 交换与 MRV2 `prepare_inputs_embeds` 使用，保证"有效行被覆盖、填充行不继承上一轮"（测试 [test_deepseek_v41_vision.py:L198-L252](../../../../tests/ut/models/test_deepseek_v41_vision.py#L198)）。

---

## 六、VL 路由偏置与 Engram 的相互排除

### 6.1 三套独立开关

| 开关 | 判据 | 建了什么 | 位置 |
|---|---|---|---|
| **视觉塔 + `bias_vl`** | `vision_n_layers>0` 且 `limit_per_prompt("image")>0` | ViT/Aligner/三向量 + 每层 gate 的 `bias_vl` | [vl_model.py:L69-L92](../vl_model.py#L69)、[model.py:L214-L222](../model.py#L214) |
| **Engram** | `engram_layer_ids` 非空 **且** 运行时 opt-in | 指定层挂 `AscendEngram` | [model.py:L837-L842](../model.py#L837) |
| **mm-prefix** | `vision_n_layers>0` | `is_mm_prefix_lm` / `mm_prefix_clamp_sliding_window` | [vl_model.py:L69-L71](../vl_model.py#L69) |

> `model.py:L837-L842` 是 **Engram 层挂载判断**（`has_engram and not is_draft_layer and layer_idx in config.engram_layer_ids`），**与视觉无关**；视觉路由偏置在 L214-L222 与 L256-L257。两者唯一交集是 §6.3。

### 6.2 `bias_vl`：图像行换一套专家偏好

**参考语义**（[reference/model.py:L811-L829](reference/model.py#L811)）：`bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, self.bias)`，再 `indices = (scores + bias).topk(...)`——**偏置只影响"选谁"，权重仍来自 `scores.gather(...)` 的原分数**；参考侧 `image_mask = token_types >= 0`、`engram_mask = ~image_mask`（[L1252-L1254](reference/model.py#L1252)），一行两用。Ascend 没有 types，用 id 区间重建（[fused_topk_router.py:L32-L77](../../../ops/fused_moe/router/fused_topk_router.py#L32)）：

```python
scores = torch.nn.functional.softplus(router_logits).sqrt()      # sqrtsoftplus
image_mask = (input_ids >= image_sentinel_lo) & (input_ids < image_sentinel_lo + 5)
row_bias = torch.where(image_mask.unsqueeze(-1), bias_vl..., text_bias... or zeros)
dynamic_ids = torch.topk(scores + row_bias, k=top_k, dim=-1, sorted=True).indices
lookup_ids  = torch.where(image_mask, 0, input_ids)             # 图像行不查 tid2eid
topk_ids    = torch.where(image_mask.unsqueeze(-1), dynamic_ids, tid2eid[lookup_ids])
topk_weights = scores.gather(1, topk_ids)                       # 权重来自无偏分数
```

哨兵共 5 个（`DEEPSEEK_V4_IMAGE_SENTINEL_COUNT = 5`），下限来自配置：`image_sentinel_lo=config.image_sentinel_base_id`（[model.py:L257](../model.py#L257)，即 `129264`）；`bias_vl` 以原名进 `FusedMoEFactory`（[model.py:L256](../model.py#L256)）。与参考的差异集中在两处：判据由 types 变成 id 区间；**哈希层的文本行查 `tid2eid`、图像行走动态 topk**。

> **为何用区间而非 types**：MoE 是 NPU 融合算子，入参只有 hidden_states 与 input_ids，types 不在其 ABI 里（这也是 `requires_raw_input_tokens` 的第二个理由）。代价是"span 内所有位置必须同 id"，恰好由 §4.2 Step 5 保证。

**加载侧**：checkpoint 的 key 就叫 `.gate.bias_vl`，Ascend **保留原名**（[model.py:L1907-L1913](../model.py#L1907)）——`if ".gate.bias_vl" in name: pass`（注释："The parameter keeps the checkpoint name on Ascend. It is passed to the hash router as its vision-only correction bias, while text rows continue to use tid2eid."），文本侧 `elif ".gate.bias" in name:` 才改名成 `e_score_correction_bias`；哈希层的该参数是 `None`（[model.py:L234](../model.py#L234)），checkpoint 若仍带会被显式跳过（L1918-L1919）。

### 6.3 Engram：整段图像既"不贡献"也"不领取"

Engram 是 n-gram 历史查表，历史必须是纯文本——若 158 个相同 id 混进 4-gram 哈希，桶会被污染。两头都堵：

```python
def engram_dead_mask(token_ids, image_token_id, image_pad_token_id):        # hash_state.py:L50
    """Positions that must not take part in an n-gram ..."""
    return (token_ids == image_token_id) | (token_ids == image_pad_token_id)

def valid_engram_token_mask(input_ids, image_token_id, image_pad_token_id):  # common.py:L37
    """Exclude the complete V4.1 image region from n-gram history."""
    return (input_ids != image_token_id) & (input_ids != image_pad_token_id)
```

`engram_dead_mask`（[hash_state.py:L50-L62](../engram/hash_state.py#L50)）用在**建哈希/查表前**：当前 chunk 与 lookback 的图像位置都变 dead，**不参与 n-gram 构成**；`valid_engram_token_mask`（[common.py:L37-L43](../engram/common.py#L37)）用在**门控融合时**：图像位置**不被加上** engram 贡献。落地在 `engram_gate` 末步（[common.py:L71](../engram/common.py#L71)）：`gate.masked_fill(~token_mask.unsqueeze(-1), 0)` ⇒ `gate=0` ⇒ 残差只保留原 hidden ⇒ **图像 token 的隐藏状态与 Engram 完全无关**。这正是参考注释"image tokens take no part in an n-gram and get no engram contribution"（[reference/model.py:L1252-L1254](reference/model.py#L1252)）的 Ascend 版；拆成两个函数是因为哈希状态要跨 chunk/lookback 维护（`dead_mask` 与 `lookback_dead_mask` 分开传，[hash_state.py:L79-L89](../engram/hash_state.py#L79)）。

---

## 七、权重加载与文本-only 场景

### 7.1 名字映射：只有一条规则

```python
def _vision_parameter_name(name: str) -> str | None:       # vl_model.py:L31
    if name.startswith("model."):
        name = name.removeprefix("model.")
    if name.startswith(("vision.", "aligner.", "image_")):
        return name
    return None
```

`model.vision.blocks.0.attn.wqkv.weight` → 视觉塔；`model.aligner.w1.weight`、`model.image_start` → Aligner 与三向量；`model.layers.0.self_attn...` → 返回 `None` → 交给语言模型。

### 7.2 `load_weights`：只在塔没建时静默跳过

```python
def language_weights():                                    # L305
    for name, loaded_weight in weights:
        vision_name = _vision_parameter_name(name)
        if vision_name is None:
            yield name, loaded_weight                      # 文本权重透传
            continue
        # Text-only serving intentionally leaves the vision tower
        # unconstructed, although a multimodal checkpoint still
        # contains its tensors. Keep loading strict when the tower is
        # enabled, but do not make disabled modalities loadable state.
        if vision_name not in params and self.vision is None:
            continue                                       # L313-L316：唯一静默跳过点
        param  = params[vision_name]                       # L317：塔启用时缺参数 → KeyError
        loader = getattr(param, "weight_loader", default_weight_loader)
        loader(param, loaded_weight)                       # L298-L323 的骨架
```

| 情形 | 行为 |
|---|---|
| 视觉权重 + 塔已建 + 参数存在 | 正常加载，记入 `loaded_vision` |
| 视觉权重 + **`self.vision is None`** + 参数不存在 | **静默 `continue`**（L313-L316） |
| 视觉权重 + 塔已建 + **参数不存在** | `params[vision_name]` 抛 `KeyError` → 加载失败（有意严格） |
| 文本权重 | `yield` 给语言模型，返回名统一加 `language_model.` 前缀（L323） |

> `vision_name not in params and self.vision is None` 是 `and` 不是 `or`：若写成"参数不存在就跳过"，**塔启用却缺 tensor** 的严重问题会被吞掉。测试 [test_deepseek_v41_vision.py:L255-L277](../../../../tests/ut/models/test_deepseek_v41_vision.py#L255) 用 `wrapper.vision = None` 精确锁住这条边界。

### 7.3 文本-only：塔可以不建，但默认会建

```python
image_enabled = config.vision_n_layers > 0 and self.multimodal_config.get_limit_per_prompt("image") > 0   # L75
```

| 场景 | `vision_n_layers` | `limit_per_prompt("image")` | 塔 | checkpoint 视觉 tensor |
|---|---|---|---|---|
| 正常多模态 | 32 | 默认（≥1） | ✅ 建 | 加载 |
| `--limit-mm-per-prompt '{"image":0}'` | 32 | 0 | ❌ 不建 | 静默跳过 |
| 纯文本 checkpoint | 0 | — | ❌ 不建 | 不存在 |

**只跑文本不等于省显存**：用多模态权重起服务、又没显式把 image 限额设 0，塔照建、权重照载（约 0.9 GiB/卡，§8.3），哪怕一次图都不接。参考侧对应 `args.vision_enabled`，并在"prompt 含图但塔关闭"时直接报错（[image_processor.py:L158-L159](reference/image_processor.py#L158)）——**宁可报错、不静默降级**；Ascend 同语义落在 `_encode_image`/`_process_image_input` 的断言上（[vl_model.py:L123](../vl_model.py#L123)、L152）。

---

## 八、性能与限制

### 8.1 限制清单

| 限制 | 数值/行为 | 出处 |
|---|---|---|
| 单图 token 上限 | `max_image_tokens=1024`，超了注意力侧抛错 | [dsa_v1.py:L712-L715](../../../attention/dsa_v1.py#L712) |
| 小图放大阈值 | `min_pixels=295936=544²`，更小的按面积比放大 | [mm_preprocess.py:L140-L143](../../../../../vllm/vllm/models/deepseek_v41/common/mm_preprocess.py) |
| 宽高比钳制 | `max_wh_ratio=null` ⇒ 不裁只缩 | [mm_preprocess.py:L138-L139](../../../../../vllm/vllm/models/deepseek_v41/common/mm_preprocess.py) |
| 每 LLM token 像素 | `patch×downsample = 42×42` | 配置推导 |
| ViT 前向粒度 | **每图一次**，`for` 串行，不可批 | [vl_model.py:L157-L176](../vl_model.py#L157) |
| 塔的切分 | 无 TP/DP，整塔复制到每卡 | [vision.py:L3-L7](../vision.py#L3) |
| span 与 prefill chunk | **必须完整落在首个 chunk** | [reference/model.py:L1257-L1259](reference/model.py#L1257) |

### 8.2 一张图真实有多大

| 形状取向 | `n_llm_h×n_llm_w` | span | `n_vit_h×n_vit_w` | patch 数 | 像素 |
|---|---|---|---|---|---|
| 接近正方 | `31×31` | `31·32+2=994` | `93×93` | `8649` | `1302×1302` |
| 超宽单行 | `1×1021` | `1·1022+2=1024` | `3×3063` | `9189` | `42×42882` |

**ViT 注意力的二次项**：一次双向注意力 `O(N²)`。正方那档 `N=8649`，每层每头 `8649²≈7.5×10⁷` 个 score，×16 头 ×32 层 ≈ **`3.8×10¹⁰`**——这是"每图一次全塔前向"最贵的部分，也是多图吞吐上不去的根因。

### 8.3 复制塔的显存代价

| 模块 | 参数数 | 占比 |
|---|---|---|
| `patch_embed.proj`（`588→1024`+bias） | `603,136` | 0.12% |
| 单 Block（attn 3,148,800+1,049,600、MLP 5,767,168+2,883,584、2 norm） | `12,851,200` | — |
| × 32 Blocks | `411,238,400` | 84.7% |
| `aligner.w1`（`9216→5120`）+ `w2`（`5120→5120`），均含 bias | `73,410,560` | 15.1% |
| 三向量 + 塔尾 norm / **总计** | `16,384` / **≈485.3 M** | ≈0% |

bf16 下约 **`970 MB ≈ 0.90 GiB`**，**每卡一份**（TP 组内完全冗余，DP/EP 组间各一份）。两个性质要警惕：**与 batch 无关的恒定支出**、**与并行度无关、加卡不减**。

### 8.4 为什么不做 batching / 为什么 span 必须在首个 chunk 内

拼 batch 要 padding（各图 `n_vit_h·n_vit_w` 不同）与 block-diagonal 掩码，而算子只接受 `cu_seqlens`（[mm_encoder_attention.py:L529](../../../../../vllm/vllm/model_executor/layers/attention/mm_encoder_attention.py#L529)）；varlen 打包则需 cos/sin 按图切片，改动面大。上游 MRV2 虽有 `EncoderRunner` 通道（[test_deepseek_v41_vision.py:L29](../../../../tests/ut/models/test_deepseek_v41_vision.py#L29)），塔仍逐图算——现阶段取舍明确：**先保证语义正确（一图一次、绝不串味）**。

span 必须整体在 prefill 首个 chunk 内，有两个原因：① 参考侧 `merge_image_embeddings` 在 `start_pos == 0` 一次性覆盖写，并断言 `image spans must be prefilled in a single chunk`（[reference/model.py:L1257-L1259](reference/model.py#L1257)）；② Ascend 侧图像 span 在 LLM 内**双向可见**，`build_vision_bidirectional_swa_indices`（[dsa_v1.py:L677-L735](../../../attention/dsa_v1.py#L677)）把 span 内 token 的可见范围改成"因果滑窗 ∪ 整个 span"，而该构建**只在 prefill 分支执行**（[dsa_v1.py:L2015](../../../attention/dsa_v1.py#L2015)）：

```python
in_span = (req_ids == req_idx) & (positions >= span_start) & (positions <= span_end)
start_positions = torch.where(in_span, minimum(start_positions, span_start), start_positions)
end_positions   = torch.where(in_span, maximum(end_positions,   span_end),   end_positions)
```

span 区间来自 runner 的 `req_doc_ranges`（[model_runner.py:L3924-L3944](../../../worker/v1/model_runner.py#L3924)），只有图所在的 prefill 步才拿得到。

> **部署推论**：`--max-num-batched-tokens` 必须**大于"prompt 前缀 + 最长单图 span"**，否则 span 被 chunk 切开——轻则报错，重则图像区只有一半是双向的、**且不报错**。`max_image_tokens=1024` 时该下界约为 `1024 + 前缀 token 数`。

---

## 九、常见坑 / FAQ

**9.1 传了图却走了纯文本分支？** 依次查 `vision_n_layers>0`、`limit_per_prompt("image")>0`、请求里 `patches` 是否存在（[L104-L106](../vl_model.py#L104)）。前两条任一为假，`self.vision` 就是 `None`，`embed_multimodal` 返回 `[]`（[L181-L182](../vl_model.py#L181)）——**静默当文本处理，`129264` 退化成普通词嵌入**。

**9.2 为什么 span 比 IMAGE 格多？** 还要加**每行 1 个 `IMAGE_NEW_LINE`**（`n_llm_h` 个）与**首尾各 1 个**边界：`3×4` 图是 `12+3+2=17`，`12×12` 图是 `144+12+2=158`。

**9.3 图片被拉伸或出现灰边？** `max_wh_ratio is None` 时走 `ImageOps.pad(image,(best_w,best_h),color=(127,127,127))`（[mm_preprocess.py:L150-L153](../../../../../vllm/vllm/models/deepseek_v41/common/mm_preprocess.py)）：**等比缩放塞进目标框、空白补灰、不拉伸**。灰值 `127/255≈0.498` 归一化后 `≈-0.004`，几乎就是 0，与 Aligner 里 `F.pad` 补的 0 一致（刻意为之，避免边缘强信号）。想拉伸需设 `max_wh_ratio` 并走进 `image.resize(...)` 分支。

**9.4 文本-only 为什么还占 0.9 GiB？** 见 §7.3：`image_enabled` 是"有塔"**且**"限额>0"，要真省下这 0.9 GiB 必须显式 `--limit-mm-per-prompt '{"image":0}'`。

**9.5 两张图紧挨着会串味吗？** ViT 侧不会（逐图前向、前导 batch 维恒 1），LLM 侧也不会（每图各有一段 `req_doc_ranges`、双向窗口各算各的）。**前提是边界登记正确**——上游自 #56554 起移除对齐 pad，边界错了不报错、只会让两图连成一片；排查先核对 `llm_grid` 之和与 `types.numel()` 是否相等。

**9.6 重跑一步后图像区变乱码？** 因为合并是**原地覆盖**（[utils.py:L738](../../../../../vllm/vllm/model_executor/models/utils.py#L738)），而 `inputs_embeds` buffer 会复用。上游用 `prepare_inputs_embeds` + `set_embedding_num_tokens_padded`（[vl_model.py:L211](../vl_model.py#L211)）保证"每步覆盖有效行、空步不继承残留"，并由 [test_deepseek_v41_vision.py:L198-L252](../../../../tests/ut/models/test_deepseek_v41_vision.py#L198) 的 `(3,5)→(1,2)→(0,0)` 三步断言锁住。**自己写调用链时别假设 `inputs_embeds` 干净。**

**9.7 只改 `get_vision_cos_sin` 的 h/w 顺序行吗？** 不行。`cos/sin` 的 32 维顺序必须与 `ApplyRotaryEmb(is_neox_style=True)` 的"前后对半劈"（[common.py:L157-L158](../../../../../vllm/vllm/model_executor/layers/rotary_embedding/common.py#L157)）配套。**单改一边既不报错也不形状不匹配，只让图像理解变差。**

**9.8 手工构造 `input_ids` 只给第一个位置填 `129264`？** 路由会把那段当文本——`image_mask` 是逐位置的 id 区间判断（[fused_topk_router.py:L52](../../../ops/fused_moe/router/fused_topk_router.py#L52)），**不连续就漏判、且不报错**。这也是 §4.2 Step 5 强调"17 个位置全填"的原因。

**9.9 `dsa_v1` 抛 `Image span exceeds vision_max_n_token`？** 说明某段 `span_end-span_start+1 > 1024`（[dsa_v1.py:L712-L715](../../../attention/dsa_v1.py#L712)）。常见原因：自定义 processor 用了更大的 `max_n_token`，或服务端与 processor 的 `max_image_tokens` 不一致（processor 在上游 `mm_preprocess`，Ascend 只读配置）。

---

## 十、一句话总结

V4.1 图像通路 = **复制的双向 ViT 塔（2D RoPE、每图一次）** + **`3×3` 空间合并的 Aligner（9216→5120）** + **"每行 N 个 IMAGE + 1 个换行、首尾各一边界"的定长 span** + **一次原地覆盖写的嵌入合并**；这段 span 在 LLM 内享受三种特殊待遇：**路由换 `bias_vl`（靠 id 区间识别）、注意力 span 内双向、Engram 整段免疫**。正确性由一条守恒式保证：

```
Aligner 输出行数 (n_llm_h·n_llm_w) == span 里 types==IMAGE 的位置数
  == Σ 各图 (span_len − n_llm_h − 2) == input_ids 里连续 129264 的段长 − n_llm_h − 2
```

任何一环对不上，要么立刻抛形状错误（幸运），要么静默地把图像当成一段普通词嵌入（不幸）。

---

**延伸阅读**：文本主干与 DSA（本文只当它下游黑盒）[3_deepseek_v41_attn.md](3_deepseek_v41_attn.md)；官方参考 [reference/vision.py](reference/vision.py)、[reference/model.py](reference/model.py)、[reference/image_processor.py](reference/image_processor.py)；上游预处理 [mm_preprocess.py](../../../../../vllm/vllm/models/deepseek_v41/common/mm_preprocess.py)；相关测试 [test_deepseek_v41_vision.py](../../../../tests/ut/models/test_deepseek_v41_vision.py)。
