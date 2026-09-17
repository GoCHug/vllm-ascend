# DeepSeek V4 MTP 多 Token 预测详解

> **本文档详细讲解 DeepSeek V4 的 MTP (Multi-Token Prediction) 多 Token 预测机制，包括投影融合、轮询调度、延迟 HC 头压缩、自建 TopK 索引缓冲区与权重加载映射。**

> **代码版本冻结声明**：本文基于 **vllm-ascend v0.26.0rc** 源码整理，以 [deepseek_v4_mtp.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_mtp.py)（共 524 行）为准，文中 `Lxxx` 均指该文件行号；主模型侧入口在 [deepseek_v4.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py)（`get_mtp_target_hidden_states` 在 L1336）。术语约定：`hc_mult=4` 条并行流统一称"**并行残差流**"（Hyper-Connections），`hc_head` 的作用是把 `hc_mult` 条残差流**压缩回单条**（详见文档 0/1/2）。

> **统一模型配置**（W4A8 版，MTP 相关行）：`num_nextn_predict_layers=1`（1 个 MTP 预测层）、`hidden_size=7168`、`hc_mult=4`、`vocab_size=129280`；MTP 层在 `compress_ratios` 数组中对应下标 61（值 **0=dense**，即 draft 注意力不带 Compressor/Indexer），但 FFN 仍是 MoE（384 专家、Top-6，hash 路由关闭）。

---

## 一、MTP 总览

### 1.1 设计理念

MTP 是 DeepSeek V4 的推测解码（Speculative Decoding）草稿模型：一次前向额外预测后续 token 的草稿，再交给主模型并行验证，用一次草稿前向 + 一次批量验证替换多轮串行解码。

> **生活化类比**：主模型像一位严谨但语速慢的主审，MTP 像一位语速快的助理先把"接下来可能要说的 1 个词"写在纸条上；主审一次看完纸条，对的直接采用，错的从错处重写。

```
主模型第 t 步：4 条并行残差流的隐藏状态 (T, 4*7168)
    │  （尚未做 hc_head 压缩，直接存入 _mtp_hidden_buffer）
    ▼
MTP 层（mtp.0）
    ├─ 融合：上一步残差流 h_proj  +  下一 token 嵌入 e_proj
    ├─ 过一个完整 draft DecoderLayer（dense 注意力 + MoE）
    ├─ 输出仍是 4 条残差流（延迟压缩）
    └─ compute_logits 时才 hc_head 压回单流 → norm → LM head
```

### 1.2 模块层级

```
DeepSeekV4MTP (入口类, L202; @support_torch_compile)
└── model: DeepSeekMultiTokenPredictor (L141)
    ├── embed_tokens: VocabParallelEmbedding(129280, 7168)     — 词嵌入
    ├── logits_processor: LogitsProcessor                       — logits 处理
    └── layers: ModuleDict {"0": DeepSeekMultiTokenPredictorLayer (L56), ...}
        ├── e_proj / h_proj: ReplicatedLinear(7168 → 7168)      — 双路投影（走 quant_config）
        ├── enorm / hnorm: RMSNorm(7168)                        — 嵌入 / 残差流归一化
        ├── shared_head: SharedHead (L36)                       — 每层各一份
        │   ├── norm: RMSNorm(7168)
        │   └── head: ParallelLMHead(129280, 7168)
        ├── mtp_block: DeepseekV2DecoderLayer(is_draft_layer=True, L89)
        │   └── dense DSA 注意力 + DeepseekV4MoE（无 hash）
        ├── hc_head_fn/base/scale: Parameter (fp32)             — HC 头门控参数
        ├── hc_norm: RMSNorm(4*7168, has_weight=False, fp32)    — 无权重全宽归一化
        └── topk_indices_buffer: (max_num_batched_tokens, 1024) int32
            └─ is_v32 时【本层自建】，不与主模型共享（L77-L86）
```

### 1.3 与主模型的交互

| 交互点 | 说明 |
|--------|------|
| `_mtp_hidden_buffer` | 主模型 forward 写入的**预 hc_head 残差流缓冲**，形状 `(max_num_batched_tokens, hc_mult*hidden_size)`，每个 target step 后有效（主模型 [L1336-L1340](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L1336)）。MTP 的 `previous_hidden_states` 即来源于此 |
| `topk_indices_buffer` | **v0.26 中主模型与 MTP 不共享**：主模型各层共用一份模型级 buffer；MTP 层在 `__init__` 自建一份传给自己的 `mtp_block`（L79-L93）。原因见文档 5 §6（spec decode 下 impl 层引用会失效） |
| `spec_step_idx` / `spec_step_index` | 当前推测步号；预测器用它对 `num_mtp_layers` 取模轮询（L177） |
| 词表/embedding | MTP 自建 `embed_tokens`，checkpoint 中通过权重映射加载与主模型同源的词嵌入 |

---

## 二、DeepSeekV4MTP — 入口类（L202-L251）

继承 `nn.Module + SupportsPP + DeepseekV2MixtureOfExperts`。构造时创建 `DeepSeekMultiTokenPredictor`（prefix 自动加 `mtp`）并调用 `set_moe_parameters()`（L209）。

### 2.1 set_moe_parameters（L211-L229）

```python
self.expert_weights = []
self.num_expert_groups = getattr(self.config, "n_group", 1)   # V4 配置无 n_group → 1
for layer in self.model.layers.values():
    if isinstance(layer, PPMissingLayer):    # PP 下不在本 rank 的层跳过
        continue
    layer = layer.mtp_block                  # DeepseekV2DecoderLayer
    if isinstance(layer.mlp, DeepseekV4MoE):
        example_moe = layer.mlp              # 注释：取最后的 MoE，前面可能是 dense 层
        self.moe_mlp_layers.append(layer.mlp)
        self.moe_layers.append(layer.mlp.experts)
self.extract_moe_parameters(example_moe)
```

作用与主模型的 MoE mixin 一致：注册 EPLB/EP 通信组、抽取冗余专家等超参，使 draft 层的 384 个专家同样纳入专家并行调度。

### 2.2 forward / compute_logits / embed_input_ids

三个方法都是纯转发（L231-L251）：`forward → model(...)`、`compute_logits → model.compute_logits(...)`、`embed_input_ids → model.embed_input_ids(...)`。

---

## 三、DeepSeekMultiTokenPredictor — 预测器主体（L141-L198）

### 3.1 核心成员

| 成员 | 说明 |
|------|------|
| `mtp_start_layer_idx` | `config.num_hidden_layers` = **61**（L145），MTP 配置数组下标从此开始 |
| `num_mtp_layers` | `getattr(config, "num_nextn_predict_layers", 1)` → **1**（L146） |
| `layers` | ModuleDict，key 为 `"0"`…`"N-1"`（L149-L157） |
| `embed_tokens` | `VocabParallelEmbedding(vocab_size, hidden_size)`（L158） |
| `logits_processor` | `LogitsProcessor(vocab_size)`（L162） |

### 3.2 轮询调度（forward，L167-L184）

```python
if inputs_embeds is None:
    inputs_embeds = self.embed_tokens(input_ids)   # 也允许外部传入已算好的嵌入
current_step_idx = spec_step_idx % self.num_mtp_layers
return self.layers[str(current_step_idx)](
    input_ids, positions, previous_hidden_states, inputs_embeds, current_step_idx,
)
```

当推测步号超过层数时循环复用同一批层。本配置只有 1 个 MTP 层，即每步都用 `layers["0"]`。

---

## 四、DeepSeekMultiTokenPredictorLayer — 单层实现（L56-L138）

### 4.1 核心成员

| 成员 | 形状 | 说明 |
|------|------|------|
| `e_proj` | `7168 → 7168` | ReplicatedLinear，**走 quant_config（W4A8 包中量化）**，嵌入投影 |
| `h_proj` | `7168 → 7168` | ReplicatedLinear，同上，残差流投影（逐流作用） |
| `enorm` / `hnorm` | RMSNorm(7168) | 嵌入归一化 / 上一步残差流归一化 |
| `shared_head` | SharedHead | **每个 MTP 层各持一份**（L88，本配置仅 1 份） |
| `mtp_block` | DeepseekV2DecoderLayer | 完整 draft block，传 `is_draft_layer=True` 与自建 buffer（L89-L95） |
| `is_v32` | bool | `hasattr(config, "index_topk")`（L76），V4 恒为 True |
| `hc_eps` / `hc_mult` | 1e-6 / 4 | 来自配置 |
| `hc_head_fn` | `(4, 4*7168)` fp32 | HC 头门控矩阵（每个目标流一行） |
| `hc_head_base` | `(4,)` fp32 | 门控偏置 |
| `hc_head_scale` | `(1,)` fp32 | 门控温度缩放 |
| `hc_norm` | RMSNorm(`4*7168`, `has_weight=False`, dtype=**fp32**) | **v0.26 新增**：对拼平的全宽残差流做无权重 RMSNorm（L103） |

### 4.2 投影融合（forward，L107-L129）

MTP 输入由两路融合：

1. **下一 token 嵌入** `inputs_embeds`（`(T, 7168)`）：先 mask，再 `enorm`；
2. **上一步 4 条残差流** `previous_hidden_states`：先 view 回 `(T, 4, 7168)`，再 `hnorm`（对最后一维归一化）。

```
inputs_embeds (T, 7168)
    │ where(position==0 → 0)  [L117]
    │ enorm                    [L118]
    │ e_proj                   [L122]
    ▼ (T, 7168) ── unsqueeze(-2) ──▶ (T, 1, 7168)
                                        │ 广播相加
previous_hidden_states (T, 4*7168)     │
    │ view(-1, 4, 7168)        [L119]  ▼
    │ hnorm                     [L120] ▶ (T, 4, 7168)  hidden_states
    │ h_proj（逐流 7168→7168）  [L122]
    ▼ (T, 4, 7168)
```

```python
# L117-L124（顺序：先两路各自归一化，再投影相加）
inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
inputs_embeds = self.enorm(inputs_embeds)
previous_hidden_states = previous_hidden_states.view(-1, self.hc_mult, self.config.hidden_size)
previous_hidden_states = self.hnorm(previous_hidden_states)
hidden_states = self.e_proj(inputs_embeds).unsqueeze(-2) + self.h_proj(previous_hidden_states)
hidden_states, residual = self.mtp_block(positions=positions, hidden_states=hidden_states, residual=None)
```

要点：
- `e_proj` 输出在倒数第 2 维插一个长度 1 的轴，靠广播**同时加到 4 条残差流上**——同一个"下一 token 提示"注入每一条流；
- `h_proj` 对 4 条流共享同一组权重逐流线性变换；
- `enorm/hnorm` 发生在投影**之前**（先归一化再投影）。

### 4.3 Position 0 Mask（L117）

```python
# masking inputs at position 0, as not needed by MTP
inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
```

position 0 是序列首 token，它没有"前一个 token 需要预测"，其嵌入对 MTP 无意义，置零防止污染。

### 4.4 mtp_block — draft DecoderLayer（L89-L95）

`mtp_block` 复用主模型的 `DeepseekV2DecoderLayer`，两个关键开关：

- `is_draft_layer=True`：`DeepseekV4MoE` 中 hash 条件含 `and not is_draft_layer`（主模型 L421），因此即便 draft 层号 < 3 也**不用 hash 路由**，走线性路由；
- `topk_indices_buffer=` 自建 buffer：draft 注意力在 `compress_ratios[61]=0` 下是 **dense**（无 Compressor/Indexer），该 buffer 仅在 is_v32 时按需分配以保持 DSA 模块接口一致。
- FFN 仍是完整的 `DeepseekV4MoE`（384 专家/Top-6/共享专家），**不是稠密 MLP**。

输入 `(T, 4, 7168)` 与 `residual=None`，输出 `(hidden_states, residual)`，4 条残差流的结构与主层完全一致（hc_pre/hc_post 见文档 1/2）。

### 4.5 延迟 HC 头压缩与 hc_head 新实现

Layer.forward **不调用** `hc_head`（L126-L127 调用处被注释），直接返回 4 流隐藏状态。原因：

1. 下一步 MTP（若有多层）需要完整的多流输入；
2. 只有出 logits 时才需要单流，压缩延迟到 `compute_logits`；
3. 避免"压缩→投影→再展开"的往返开销。

v0.26 的 `hc_head`（[L131-L138](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_mtp.py#L131)）改用 `hc_norm` 模块，与主模型 `DeepseekV4Model.hc_head`（deepseek_v4.py L1130 附近）保持一致：

```python
def hc_head(self, x, hc_fn, hc_scale, hc_base):
    shape, dtype = x.size(), x.dtype          # (T, 4, 7168)
    x = x.flatten(1).float()                  # (T, 4*7168)，升 fp32
    x_norm = self.hc_norm(x)                  # ★ 无权重 RMSNorm，对【全宽 4*7168】归一化
    mixes = F.linear(x_norm, hc_fn)           # (T, 4)：每条目标流一个混合门控 logit
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps   # (T, 4)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)         # ★ 加权用的是【原始 x】
    return y.to(dtype)                        # (T, 7168)
```

| 步骤 | 细节 |
|------|------|
| 全宽归一化 | 门控 logit 由 `x_norm`（全宽 RMSNorm 后）算出，而非旧实现里手写 `rsqrt(x.square().mean())` 乘到线性结果上——数学等价但统一收敛为 `hc_norm` 模块（无权重、fp32） |
| 门控 | `sigmoid(温度缩放后的 mixes + base) + hc_eps`，4 个非负权重 |
| 加权 | 权重乘在**未经归一化的原始 x** 上（`x.view(shape)`），沿 `dim=1`（4 条流）求和压成单流 |

---

## 五、SharedHead — 输出归一化与 LM Head（L36-L53）

```python
class SharedHead(nn.Module):
    def __init__(self, config, prefix, quant_config=None):
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(config.vocab_size, config.hidden_size,
                                   quant_config=quant_config, prefix=maybe_prefix(prefix, "head"))

    def forward(self, hidden_states):
        return self.norm(hidden_states)   # 只做 norm；head 矩阵在外部 logits_processor 中使用
```

- 结构上**每个 MTP 层各持有一个** `shared_head`（L88）；本配置只有 1 个 MTP 层，故只有一份；
- 命名为 "shared" 体现在 checkpoint 权重与主模型 LM head 同源（推理时通常 tie embeddings）；
- TP 下 `ParallelLMHead` 按词表维切分，与主模型 logits 路径一致。

---

## 六、完整 Forward 数据流

### 6.1 单步草稿前向

```
输入: input_ids (T,)、positions (T,)、previous_hidden_states (T, 4*7168)、
      inputs_embeds (T, 7168)|None、spec_step_index
  1. 缺省嵌入: inputs_embeds = embed_tokens(input_ids)（Predictor L175-L176）
  2. 轮询选层: layers[str(spec_step_idx % 1)]
  3. mask position0 → enorm → e_proj → (T,1,7168)
     prev.view(T,4,7168) → hnorm → h_proj → (T,4,7168)
     两者广播相加得 (T,4,7168)                           [Layer L117-L122]
  4. mtp_block（dense DSA 注意力 + MoE，含 hc_pre/hc_post）→ (T,4,7168)
输出: 4 条残差流隐藏状态（不压缩）
```

### 6.2 compute_logits（Predictor L186-L198）

```python
mtp_layer = self.layers[str(spec_step_idx % self.num_mtp_layers)]
hidden_states = hidden_states.view(-1, mtp_layer.hc_mult, mtp_layer.config.hidden_size)  # L193
hidden_states = mtp_layer.hc_head(hidden_states,
                                  mtp_layer.hc_head_fn, mtp_layer.hc_head_scale,
                                  mtp_layer.hc_head_base)                                # (T, 7168)
logits = self.logits_processor(mtp_layer.shared_head.head,
                               mtp_layer.shared_head(hidden_states))                    # (T, 129280)
```

即 **view 回 4 流 → hc_head 压单流 → shared_head.norm → 分片 LM head**。注意进入 `hc_head` 前显式 view：若上游传入的是拼平的 `(T, 4*7168)`，这里恢复 `(T, 4, 7168)`。

---

## 七、权重加载与映射（load_weights，L253-L479）

### 7.1 总体流程

1. `get_spec_layer_idx_from_weight_name(config, name)`（L291，主模型文件 L302-L305：`mtp.` 开头返回 0）筛选属于 MTP 的权重，非 MTP 权重直接 `continue`；并断言名字含 `mtp.0.`（L295）；
2. fp8 特殊重命名：`embed.weight → mtp.0.emb.tok_emb.weight`、`head.weight → mtp.0.head.weight`（L284-L289）；
3. 三段式前缀改写（L296-L301）：
   - 含 `.emb.tok_emb.`：`mtp.0.` → `model.`（预测器顶层嵌入）；
   - `no_mtp_block_in_name(name)` 命中（投影/HC 头/norm/head 等）：`mtp.0.` → `model.layers.0.`；
   - 其余：`mtp.0.` → `model.layers.0.mtp_block.`（draft block 内部权重）；
4. 逐片段改名后经 `stacked_params_mapping`（w1/w3 → gate_up_proj 融合）、`expert_params_mapping`（384 专家，EPLB 时含冗余）或默认 loader 加载。

### 7.2 权重名片段映射表

| checkpoint 片段 | 加载名片段 | 行号 |
|----------------|-----------|------|
| `.w1.` / `.w2.` / `.w3.` | `.gate_proj.` / `.down_proj.` / `.up_proj.` | L303-L308 |
| `.scale` 结尾 | `.weight_scale` | L310-L311 |
| `.head.` | `.shared_head.head.` | L313-L314 |
| `.norm.` | `.shared_head.norm.` | L316-L317 |
| `.emb.tok_emb.` | `.embed_tokens.` | L319-L320 |
| `.attn.`（且不含 `self_attn`） | `.self_attn.` | L322-L323 |
| `.ffn.` / `.ffn_norm.` / `.attn_norm.` | `.mlp.` / `.post_attention_layernorm.` / `.input_layernorm.` | L324-L329 |
| `.gate.bias` | `.gate.e_score_correction_bias` | L331-L332 |
| `fused_qkv_a_proj` | 可选 QKV 融合加载；目标参数不存在时回退普通加载 | L362-L367 |

### 7.3 `no_mtp_block_in_name`（L511-L524）

下列名字直接挂在 `model.layers.0.*` 下，**不加** `.mtp_block.`：

```python
[".hc_head_fn", ".hc_head_base", ".hc_head_scale",
 ".e_proj.", ".h_proj.", ".enorm.", ".hnorm.",
 ".norm.", ".head.", ".emb.tok_emb."]
```

另有 `_rewrite_spec_layer_name`（L481-L509）服务通用多层命名：`embed_tokens/enorm/hnorm/eh_proj/shared_head` 视为预测层直属权重，其余加 `.mtp_block.`，嵌入等共享权重提升到 `model.` 顶层。

### 7.4 Attention Sink 的 CP 分支（L334-L343）

```python
if "sink" in name:
    if enable_dsa_cp():
        param.data.copy_(loaded_weight)              # 上下文并行：整份复制（本 rank 持有全头）
    else:
        narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
        param.data.copy_(narrow_weight)               # 普通 TP：只切本 rank 的头段
```

### 7.5 mix_placement 融合共享专家（L345-L450）

`ascend_config.mix_placement=True` 时，checkpoint 中合并的共享专家权重被均分为 `n_shared_experts` 份，改名到 `mlp.experts.{384+j}.*`，再借 `expert_params_mapping` 以专家身份加载：`down_proj` 沿 dim 1 切，gate/up 沿 dim 0 切（L386-L413）。

---

## 八、训练目标：预压缩残差流缓冲

主模型 `get_mtp_target_hidden_states`（deepseek_v4.py [L1336-L1340](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L1336)）返回 `self.model._mtp_hidden_buffer`：

```
Pre-hc_head residual stream buffer
形状: (max_num_batched_tokens, hc_mult * hidden_size) = (…, 4*7168)
生命周期: 主模型 forward 填充，每个 target step 后有效
```

即 MTP 训练时对齐的目标是**主模型未经 hc_head 压缩的 4 流状态**——与推理期 MTP 消费的 `previous_hidden_states` 完全同构，保证训练/推理路径一致。

---

## 九、一句话总结

V4 的 MTP 是一个复用主模型 `DeepseekV2DecoderLayer` 的单预测层草稿模型：把"下一 token 嵌入（enorm+e_proj）"广播加到"上一步 4 条并行残差流（view+hnorm+h_proj）"上，送入 **dense 注意力 + 完整 MoE（无 hash）** 的 draft block；全程保持 4 流结构、只在 `compute_logits` 时用 **hc_norm + sigmoid 门控的 hc_head** 压回单流出 logits；DSSA 的 TopK 索引缓冲区由 MTP 层**自建**；权重加载通过 `mtp.0.*` 三段式前缀改写、w1/w2/w3 与量化 scale 重命名、专家映射与 attn_sink 的 CP/TP 分支完成。
