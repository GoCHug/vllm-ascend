# DeepSeek V4 DSpark 稀疏块草稿模型详解

> **本文档讲解 DeepSeek V4 的 DSpark（DeepSeek Sparse Block Drafter）块草稿模型：Markov n-gram 偏置头、主模型多层隐藏状态融合、上下文 KV 预计算、并行残差流（mHC）头压缩、block 内双向注意力，以及 `mtp.*` 命名空间下的权重加载映射。**

> **代码版本冻结声明**：本文基于 **vllm-ascend v0.26.0rc** 源码整理，模型文件为 [deepseek_v4_dspark.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py)（共 490 行）。关键位置：`DSparkMarkovHead` L70、`DeepseekV4DSparkModel` L91、`DSparkDeepseekV4ForCausalLM` L268、`set_moe_parameters` L287、`load_weights` L353、`_remap_dspark_name` L451。block 双向注意力的运行时设置在 [dspark_proposer.py:L365-L371](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py#L365)，非因果索引构建在 [dsa_v1.py:L474/L486](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L474)。文中 `Lxxx` 均指对应文件行号。

> **统一模型配置**（W4A8 版，本文相关行）：

| 参数 | 值 | 含义 |
|------|-----|------|
| `hidden_size` | 7168 | 残差流宽度 |
| `hc_mult` | 4 | 并行残差流（mHC）条数 |
| `num_hidden_layers` | 61 | 主模型层数，draft 层编号从 61 开始 |
| `vocab_size` | 129280 | 词表大小 |
| `dspark_block_size` | 5 | 一次出一个 5-token 的 draft block |
| `dspark_markov_rank` | 512 | Markov 头低秩瓶颈 |
| `dspark_noise_token_id` | 128799 | 草稿噪声/起始 token id |
| `dspark_target_layer_ids` | `[58, 59, 60]` | 取主模型最后 3 层隐藏状态做融合 |
| draft 层数 | 3（`n_mtp_layers`/`dspark_num_mtp_layers`，缺省 3） | 编号 61/62/63，`compress_ratio` 数组对应 3 个 0（dense） |

---

## 一、DSpark 总览 — 入门篇

### 1.1 DSpark 是什么

> **生活化类比**：
> - **MTP（多 token 预测）** 像**逐字猜**的助手：你说一个字它猜下一个字，你验证后再猜下一个，串行推进。
> - **DSpark（稀疏块草稿）** 像**一口气猜一整句**的助手：它同时翻看你前面写的好几段（主模型多个中间层的隐藏状态），一次写出一整段草稿（一个 block，本文配置 5 个 token），再交给主模型批量验证。

DSpark 权重虽然也存在目标 checkpoint 的 `mtp.*` 命名空间下，但它**不是**普通的串行 MTP 模块，而是一个 **block drafter（块草稿器）**：吃进主模型多个指定层的隐藏状态，一次吐出一整个 draft block，做更激进的推测解码。

```
主模型 forward
    ├─ 在 dspark_target_layer_ids（58/59/60）收集 aux_hidden_states
    └─ DSpark:
        ① combine_hidden_states：main_proj 拼接投影 + main_norm
        ② precompute_and_store_context_kv：预计算并写入各 draft 层 SWA KV
        ③ 3 个 Draft DecoderLayer（复用主模型层，is_draft_layer=True）
        ④ hc_head：把 4 条并行残差流压回 1 条
        ⑤ markov_head：给出 n-gram 加性偏置
        → 一整个 draft block 的 logits
```

### 1.2 模块层级

```
DSparkDeepseekV4ForCausalLM  (L268)
├── lm_head: ParallelLMHead                               词表并行输出头
├── logits_processor: LogitsProcessor                     logits 后处理
└── model: DeepseekV4DSparkModel  (L91)
    ├── embed_tokens: VocabParallelEmbedding              token 嵌入
    ├── main_proj: ColumnParallelLinear(gather_output=True)  3 层隐藏状态拼接投影
    ├── main_norm: RMSNorm(7168)                          融合后归一化（挂第 61 层）
    ├── norm: RMSNorm(7168)                               最终归一化（挂第 63 层）
    ├── markov_head: DSparkMarkovHead  (L70)
    │     ├── markov_w1 = nn.Embedding(129280, 512)       复制版嵌入（串行免通信）
    │     └── markov_w2 = ReplicatedLinear(512, 129280)   复制版投影（非 ParallelLMHead）
    ├── hc_head_fn   Parameter (4, 4*7168) fp32           残差流混合权重
    ├── hc_head_base Parameter (4,)        fp32
    ├── hc_head_scale Parameter (1,)       fp32
    └── layers: ModuleDict{61,62,63 -> DeepseekV2DecoderLayer(is_draft_layer=True)}
          61 层额外挂 main_proj/main_norm；63 层额外挂 norm/markov_head/hc_head_*
```

> **与旧文档的两处关键更正**：`markov_w1` 是普通 **`nn.Embedding`**（不是词表并行 Embedding），`markov_w2` 是 **`ReplicatedLinear`**（不是 `ParallelLMHead`），MarkovHead **也没有** `logits_processor` 成员；`main_proj` 是 **`ColumnParallelLinear(gather_output=True)`**（不是 `ReplicatedLinear`）。原因见第三、四章。

### 1.3 与主模型的交互点

| 交互点 | 说明 |
|--------|------|
| `aux_hidden_states` | 主模型在 58/59/60 层收集的隐藏状态，拼接后供 `main_proj` 融合 |
| `context_states / positions / slot_mapping` | prefill 上下文，用于预计算 draft 层 SWA KV |
| `DeepseekV2DecoderLayer` | 直接复用主模型 Transformer 层（DSA 注意力 + MoE + 并行残差流） |
| `is_draft_layer=True` | 使 MoE 的 hash 路由关闭（[deepseek_v4.py:L421](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L421)：`... and not is_draft_layer`） |
| `mtp.*` 命名空间 | DSpark 权重在 checkpoint 里挂 mtp 前缀，加载时大量重映射 |

---

## 二、工具函数与常量 — 基础篇

### 2.1 `_EXPERT_SCALE_RE`（[L44](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L44)）

```python
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.scale$")
```

专用于匹配**专家**三类投影的量化缩放因子，以区分"专家缩放"与"普通线性层缩放"——两者落盘后缀不同（见 6.3）。

### 2.2 `_apply_dsv4_rope`（[L47-L60](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L47)）

```python
cos, sin = get_cos_and_sin_dsa(positions)       # 取 DSA 全局 cos/sin 表
cos_t, sin_t = cos[rotary_emb.layername], sin[rotary_emb.layername]  # 按层名取
if inverse:
    sin_t = -sin_t                              # 逆 RoPE
return rotary_emb(x, cos_t, sin_t)
```

不同压缩率的层 RoPE 基座不同，因此 cos/sin 按 `layername` 区分。draft 层是 dense（`rope_theta=10000`，rope 组只有 `default`）。

### 2.3 `_get_dspark_num_mtp_layers`（[L63-L67](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L63)）

```python
num_layers = getattr(config, "n_mtp_layers", None)
if num_layers is None:
    num_layers = getattr(config, "dspark_num_mtp_layers", 3)
return int(num_layers or 3)      # 优先级 n_mtp_layers > dspark_num_mtp_layers > 3
```

---

## 三、DSparkMarkovHead — Markov n-gram 偏置头（[L70-L88](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L70)）

### 3.1 是什么 / 干什么用

> **生活化类比**：Markov 头像一个"**语言习惯数据库**"。它记得"我吃饭"后面常接"了"、"今天天"后面常接"气"，用一个二元（n-gram）转移模型给常见搭配加分，让草稿更通顺。

### 3.2 结构（注意全部"复制"，不要词表并行）

```python
class DSparkMarkovHead(nn.Module):
    def __init__(self, config, prefix):
        # Markov 解码对每个 draft 位置都是串行执行的；两个低秩权重都复制整份，
        # 使每一步都不产生跨卡通信。
        self.markov_w1 = nn.Embedding(config.vocab_size, config.dspark_markov_rank)   # (129280, 512)
        self.markov_w2 = ReplicatedLinear(config.dspark_markov_rank, config.vocab_size,
                                          bias=False, prefix=f"{prefix}.markov_w2")    # (512, 129280)
```

| 成员 | 实际类型 | shape | 为什么是复制而不是并行 |
|------|---------|-------|----------------------|
| `markov_w1` | `nn.Embedding` | `(129280, 512)` | Markov 每个 draft 位置串行查表，复制可免 all-gather |
| `markov_w2` | `ReplicatedLinear` | `(512, 129280)` | 同上，串行小算子，整份复制比通信更划算 |

### 3.3 两个方法

```python
def embed(self, token_ids):                 # (T,) -> (T, 512)
    return self.markov_w1(token_ids)

def bias(self, markov_embed):               # (T, 512) -> (T, 129280)
    return self.markov_w2(markov_embed)     # 直接线性投影，没有 logits_processor
```

即 bias 就是一个低秩二元模型 `bias(next) = W2 · W1[prev]`，作为主 logits 的**加性偏置**。旧文档里"`self.logits_processor(self.markov_w2, ...)`"在 v0.26 并不存在。

---

## 四、DeepseekV4DSparkModel — 主体模型（[L91-L265](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L91)）

### 4.1 构造与关键字段（[L92-L163](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L92)）

```python
config = vllm_config.speculative_config.draft_model_config.hf_config
self.hc_mult = config.hc_mult                       # 4
self.hidden_size = config.hidden_size               # 7168
self.block_size = int(config.dspark_block_size)     # 5
self.target_layer_ids = list(config.dspark_target_layer_ids)  # [58,59,60]
self.num_dspark_layers = _get_dspark_num_mtp_layers(config)   # 3
self.mtp_start_layer_idx = config.num_hidden_layers           # 61
```

**词嵌入与 draft 层**：

```python
self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size, ...)        # L104
self.layers = nn.ModuleDict({
    str(61 + idx): DeepseekV2DecoderLayer(vllm_config, prefix=f"mtp.{idx}",
                                          is_draft_layer=True)                  # L110
    for idx in range(3)
})
```

层字典的 key 从 61 开始（延续主模型编号），但模块 prefix 仍是 `mtp.0/1/2`。

### 4.2 多层融合 main_proj（[L121-L133](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L121)）

```python
self.main_proj = ColumnParallelLinear(
    hidden_size * len(target_layer_ids),   # 7168*3 = 21504 输入
    hidden_size,                           # 7168 输出
    bias=False, quant_config=None,
    prefix="...layers.61.main_proj",
    gather_output=True,                    # 列并行但输出全聚合，每层拿到完整 7168
)
self.main_norm = RMSNorm(hidden_size, ...)
first_layer.main_proj, first_layer.main_norm = self.main_proj, self.main_norm
```

> **生活化类比**：猜下一段时不只看最后一段，而是把倒数第 3、2、1 段拼起来一起读。`main_proj` 就是把这三段"揉成一条"的投影。
>
> 数据流：`(T, 21504) → main_proj → (T, 7168) → main_norm → (T, 7168)`（`combine_hidden_states` L167-L168）。

### 4.3 输出端：norm / MarkovHead / hc_head 参数（[L135-L159](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L135)）

`norm`、`markov_head` 以及三个 fp32 参数 `hc_head_fn (4, 4*7168)`、`hc_head_base (4,)`、`hc_head_scale (1,)` 都建在 model 上，再挂到**最后一层**（63 层）上，供层内直接引用。

### 4.4 上下文 KV 预计算（[L170-L219](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L170)）

三个方法配合，把 prompt 上下文的 SWA KV 提前算好写入缓存：

1. `_project_shared_kv`（L170）：`wkv → kv_norm` 得 `(T, 512)`，`split` 成 `k_nope(448)`/`k_pe(64)`，对 `k_pe` 做 RoPE，拼回 `(T, 1, 512)`。
2. `_store_standard_swa_kv`（L182）：1D slot_mapping 先经 `DeviceOperator.format_dsa_slot_mapping` 按 block size 整形，再 `dsa_kv_compress_scatter` 散射进 SWA cache。
3. `precompute_and_store_context_kv`（L205）：对 3 个 draft 层逐层投影 + 写缓存。

> **考前小抄类比**：正式"打草稿"前先把课本知识点抄成小抄（预计算上下文 KV），之后每个 draft 位置直接查缓存，不必对 prompt 重算注意力。

### 4.5 forward（[L221-L231](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L221)）

```python
def forward(self, input_ids, positions):
    hidden_states = self.embed_tokens(input_ids)        # (T, 7168)
    hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)  # (T, 4, 7168)
    residual = None
    for layer in self.layers.values():                 # 顺序过 61→62→63
        hidden_states, residual = layer(positions, hidden_states, residual,
                                        llama_4_scaling=None)
    return self.hc_head(hidden_states, self.hc_head_fn,
                        self.hc_head_scale, self.hc_head_base)   # (T, 7168)
```

嵌入后先复制成 4 条并行残差流，再走与主模型完全一致的 DecoderLayer，最后用 `hc_head` 压回 1 条。

### 4.6 hc_head — 并行残差流压缩（[L233-L240](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L233)）

```python
def hc_head(self, x, hc_fn, hc_scale, hc_base):        # x: (T, 4, 7168)
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()                            # (T, 4*7168)，fp32 计算
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)  # 手写 RMSNorm
    mixes = F.linear(x, hc_fn) * rsqrt                  # (T, 4) 每条流的混合分
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps          # (T, 4) 门控
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)                # (T, 7168)
    return y.to(dtype)
```

> **注意**：DSpark 的 `hc_head` 仍是**旧版手写 `rsqrt` 归一化**（在 flatten 后的全宽上算），并**没有**改用主模型/MTP 文档里那个 v0.26 新增的 `hc_norm`（RMSNorm 模块）。它对 4 条并行残差流各学一个 sigmoid 门控权重，加权求和压成 1 条。

### 4.7 logits / 专家映射

- `compute_logits`（[L248-L254](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L248)）：`logits_processor(lm_head, self.norm(hidden_states))`，先最终 `norm` 再词表投影。
- `get_expert_mapping`（[L256-L264](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L256)）：调用 `fused_moe_make_expert_params_mapping` 生成 `(param_name, weight_name, expert_id, shard_id)` 列表，供加载专家权重。
- `markov_embed/markov_bias`（L242-L246）只是转发给 `markov_head`。

### 4.8 Block 内双向注意力（运行时行为，重要）

> **考试打草稿类比**：正常因果注意力像写正式答卷——写第 3 个字只能看前 2 个字；DSpark 出草稿块时像在草稿纸上写——块内每个字都能看到**同一块的其他字**，大家一起"商量"出最连贯的整句。

DSpark 复用了主模型的 DSA 层，但在 draft block 生成阶段通过运行时 metadata 打开非因果可见性：

| 阶段 | causal | 可见范围 |
|------|:------:|---------|
| 主模型 prefill/decode | True | 当前位 + 滑窗历史（标准因果） |
| 上下文 KV 预计算 | True | 标准因果 mask |
| **draft block 生成** | **False** | 滑窗历史 + **整个 block 互相可见** |

**证据一：proposer 关闭因果**（[dspark_proposer.py:L365-L371](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py#L365)）：

```python
if hasattr(self.model, "get_draft_attn_causal"):
    cad.causal = self.model.get_draft_attn_causal()[0]   # draft 模型给 False
else:
    cad.causal = False
cad.attn_mask = None
cad.attn_state = AscendAttentionState.ChunkedPrefill
```

**证据二：扩大窗口 + 构造非因果索引**：

- `get_dspark_sparse_sas_window`（[dsa_v1.py:L474-L478](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L474)）返回 `(sliding_window + num_speculative_tokens - 1, 0)`，左窗口扩到覆盖整个 block；
- `build_dspark_swa_indices`（[dsa_v1.py:L486](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L486)）按块表构造 slot 索引，让块内每个 query 都能看到"尾部历史窗口 + 整块 draft"，越界列置 -1。

```
 prompt 历史                    draft block（5 个位置）
 ←── sliding_window=128 ──┃   p1  p2  p3  p4  p5
 p1 可见: [历史窗口]       ┃   p1  p2  p3  p4  p5   ← 块内全可见（双向）
 p3 可见: [历史窗口]       ┃   p1  p2  p3  p4  p5   ← 能"偷看"后面
```

**为什么草稿可以"作弊"**：draft 只是提案，最终要过主模型的**因果**验证，概率过低的 token 会被拒绝并从目标分布重采样；双向注意力只提高草稿质量/接受率，不破坏自回归正确性。

---

## 五、入口类 DSparkDeepseekV4ForCausalLM（[L268-L449](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L268)）

```python
@support_torch_compile
class DSparkDeepseekV4ForCausalLM(nn.Module, DeepseekV2MixtureOfExperts):
```

- `__init__`（L269）：`has_own_embed_tokens/has_own_lm_head = (quant_config is not None)`（L273-L274，量化时用独立 embed/head），建 model、`ParallelLMHead`、`LogitsProcessor(vocab_size)`，再 `set_moe_parameters()`。
- `set_moe_parameters`（[L287-L304](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L287)）：遍历 draft 层收集所有 `DeepseekV4MoE`（跳过 `PPMissingLayer`），用**最后一个** MoE 作 example（前面可能是稠密层），再 `extract_moe_parameters`。
- `forward`（L306）：直接转发给 `self.model`。
- `compute_logits`（L317）：`del spec_step_idx`——DSpark 一次出整块，不按 spec 步索引。
- 委托方法：`markov_embed/markov_bias`、`get_draft_kv_cache_layer_names`、`combine_hidden_states`、`precompute_and_store_context_kv`。

---

## 六、权重加载与映射 — 进阶篇（[L353-L449](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L353)）

### 6.1 准备工作

```python
expert_mapping = self.model.get_expert_mapping()
expert_scale_suffix = ".weight_scale" if expert_dtype == "fp4" else ".weight_scale_inv"  # L360-362
stacked_params_mapping = [                                  # L366-371，checkpoint 分开存、模型融合存
    ("mlp.gate_up_proj", "mlp.gate_proj", 0),
    ("mlp.gate_up_proj", "mlp.up_proj", 1),
    ("shared_experts.gate_up_proj", "shared_experts.gate_proj", 0),
    ("shared_experts.gate_up_proj", "shared_experts.up_proj", 1),
]
```

### 6.2 逐权重处理顺序

1. **embed/head 复用**（L383-L386）：非量化时 `embed.weight→model.embed_tokens.weight`、`head.weight→lm_head.weight`，与主模型共享。
2. **DSpark 重命名**：`_remap_dspark_name` 返回 None 就跳过（非 mtp 权重属于目标模型）。
3. **缩放后缀**（L395-L397）：命中专家正则用 `expert_scale_suffix`，其他量化参数统一 `.weight_scale`。
4. **E8M0 专家 scale**（L400-L402）：`dtype == float8_e8m0fnu` 时 `.view(torch.uint8)`，保留原始指数字节；专家权重走 `expert_mapping` 的专用 weight_loader。
5. **堆叠融合**（L424-L432）：仅对 `model.layers.` 下的参数，把 `gate_proj/up_proj` 装进融合的 `gate_up_proj`（shard 0/1）。
6. **attn_sink 切片**（L434-L442，含 CP 分支）：

```python
if "attn_sink" in name:
    if enable_dsa_cp():                 # 上下文并行：整份拷贝（对应全头 attn_sink）
        narrow = loaded_weight
    else:                               # 普通 TP：只切本 rank 负责的头
        narrow = loaded_weight[head_start:head_end]
    params_dict[name].copy_(narrow)
```

### 6.3 `_remap_dspark_name`（[L451-L490](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_dspark.py#L451)）

- 正则 `mtp\.(\d+)\.(.*)` 提取 stage 与 rest，不匹配返回 None；
- `confidence_head.*` 直接返回 None（checkpoint 有，但 DSpark 不用，L458-L459）；
- `mtp.0.embed.weight → model.embed_tokens.weight`；最后一层 `head.weight → lm_head.weight`；`hc_head_* → model.*`；
- 层号路由：`main_proj/main_norm → 61 层`，`norm/markov_head → 63 层`，其余 `61 + stage`；
- 组件名替换：`.attn.→.self_attn.`、`.ffn_norm.→.post_attention_layernorm.`、`.attn_norm.→.input_layernorm.`、`.ffn.→.mlp.`、`.w1/.w2/.w3.→.gate_proj/.down_proj/.up_proj.`、`.mlp.gate.bias→.mlp.gate.e_score_correction_bias`。

| checkpoint 名 | 模型内部名 |
|--------------|-----------|
| `mtp.0.main_proj.*` | `model.layers.61.main_proj.*` |
| `mtp.0.attn.*` | `model.layers.61.self_attn.*` |
| `mtp.1.ffn.*` | `model.layers.62.mlp.*` |
| `mtp.2.norm.*` / `mtp.2.markov_head.*` | `model.layers.63.norm.*` / `.markov_head.*` |
| `mtp.2.head.weight` | `lm_head.weight` |

---

## 七、完整推理流程与 shape 汇总 — 高级篇

```
Prefill
  1. 主模型 forward，在 58/59/60 层收集 aux_hidden_states，并存预 hc_head 残差流缓冲
  2. DSpark 初始化：
     combine_hidden_states(aux) → (T,7168) 融合上下文起点
     precompute_and_store_context_kv(...) → 各 draft 层写好 SWA KV（考前小抄）
Draft（出 5-token block）
  3. embed → 复制 4 条并行残差流 → 顺序过 61/62/63 层（块内双向注意力）
  4. hc_head 压回 1 条 → norm → lm_head 得主 logits
  5. markov_bias(markov_embed(prev)) 作为加性 n-gram 偏置
  6. 逐位置采样出 5 个候选 token
验证
  7. 主模型用标准因果注意力并行验证，接受匹配前缀，拒绝处按目标分布重采样
```

| 张量 | shape |
|------|-------|
| `aux_hidden_states` | `(T, 7168*3)` |
| `combined_hidden` | `(T, 7168)` |
| `hidden_states`（过层时） | `(T, 4, 7168)` |
| `shared_kv` | `(T, 1, 512)` |
| `hc_head_fn` / `base` / `scale` | `(4, 28672)` / `(4,)` / `(1,)` |
| `head_hidden` / `logits` | `(T, 7168)` / `(T, 129280)` |
| `markov_embed` / `markov_bias` | `(T, 512)` / `(T, 129280)` |

---

## 八、DSpark vs MTP — 专家篇

| 特性 | MTP（[6_deepseek_v4_mtp.md](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/dsv4_docs/6_deepseek_v4_mtp.md)） | DSpark（本文） |
|------|------|------|
| 预测方式 | 串行单步 | 一次一个 block（5 token） |
| 层数 | `num_nextn_predict_layers=1` | 3 个 draft 层（61/62/63，dense） |
| 层调度 | 按 `spec_step_idx` 轮询 | 顺序通过全部层 |
| 上下文来源 | 上一步残差流 + 下一 token 嵌入 | 主模型 58/59/60 三层隐藏状态融合 |
| 上下文融合 | `e_proj + h_proj` 相加 | `main_proj`（21504→7168）拼接投影 |
| KV 预计算 | 无 | `precompute_and_store_context_kv` |
| Markov 偏置 | 无 | 有（nn.Embedding + ReplicatedLinear） |
| 块内注意力 | 因果 | **非因果（块内双向）** |
| 残差流头 | 新版 `hc_norm`（RMSNorm 模块） | 旧版手写 `rsqrt` 的 `hc_head` |
| Confidence head | 无 | checkpoint 有，加载时跳过 |
| 目标 | 保守、高接受率 | 激进、高吞吐、长上下文友好 |

---

## 九、设计要点小结

1. **Block drafter 而非串行 MTP**：一次出 5-token 块，配合块内双向注意力提升草稿质量。
2. **多层上下文融合**：`main_proj`（列并行、`gather_output=True`）把主模型最后 3 层隐藏状态揉成一条。
3. **上下文 KV 预计算**：考前小抄，draft 时直接查 SWA 缓存。
4. **Markov 头全复制**：`nn.Embedding + ReplicatedLinear`，串行 n-gram 解码零跨卡通信，输出作加性偏置。
5. **复用主模型层**：`DeepseekV2DecoderLayer(is_draft_layer=True)`，并因此关闭 MoE hash 路由。
6. **参数挂接**：融合模块挂 61 层、输出端挂 63 层；权重经 `_remap_dspark_name` 从 `mtp.*` 重映射，并处理 fp4/E8M0/gate_up 融合/attn_sink CP 切分。

## 十、一句话总结

DSpark 是一个挂在 `mtp.*` 命名空间下、由 **3 个 dense draft 层 + 多层融合 `main_proj` + 手写 `hc_head` + 复制版 Markov 头**组成的块草稿器：它融合主模型 58/59/60 层隐藏状态、预计算上下文 SWA KV，用**块内双向注意力**一次生成 5 个候选 token，再交给主模型用因果注意力批量验证，从而在不损失正确性的前提下换取更高的推测解码吞吐。
