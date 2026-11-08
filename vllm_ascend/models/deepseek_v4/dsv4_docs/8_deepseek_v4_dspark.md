# DeepSeek V4 DSpark 稀疏块草稿模型详解

> **本文档讲解 DeepSeek V4 的 DSpark（DeepSeek Sparse Block Drafter）块草稿模型：Markov n-gram 偏置头、Confidence 接受率头、主模型多层隐藏状态融合、上下文 KV 预计算、并行残差流（mHC）头压缩、block 内非因果注意力，以及 `mtp.*` 命名空间下的权重加载映射。**

> **代码版本冻结声明**：本文基于 **vllm-ascend v0.32.0** 源码整理，模型文件为 [dspark.py](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py)（共 674 行）。关键位置：上游 `DSparkMarkovHead` [qwen3_dspark.py:L48](../../../../../vllm/vllm/model_executor/models/qwen3_dspark.py#L48)、`DSparkConfidenceHead` [qwen3_dspark.py:L164](../../../../../vllm/vllm/model_executor/models/qwen3_dspark.py#L164)、Ascend 导入 L41、`DeepseekV4DSparkModel` L108、`DSparkDeepseekV4ForCausalLM` L400、`set_moe_parameters` L425、`load_weights` L503、`_remap_dspark_name` L631。block 非因果注意力的运行时设置在 [dspark_proposer.py:L342-L354](../../../../../vllm-ascend/vllm_ascend/worker/v1/spec_decode/dspark_proposer.py#L342)，非因果索引构建在 [dsa_v1.py:L542/L554](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L542)。文中 `Lxxx` 均指对应文件行号。

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
> - **DSpark（稀疏块草稿）** 像**一口气猜一整句**的助手：它同时翻看你前面写的好几段（主模型多个中间层的隐藏状态），先用一次并行前向写出一整段草稿（一个 block，本文配置 5 个 token），再用 Markov 头从左到右给块内 token 补上依赖关系，最后交给主模型批量验证。

DSpark 权重虽然也存在目标 checkpoint 的 `mtp.*` 命名空间下，但它**不是**普通的串行 MTP 模块，而是一个 **block drafter（块草稿器）**：吃进主模型多个指定层的隐藏状态，一次前向覆盖整个 draft block，做更激进的推测解码。

```
主模型 forward
    ├─ 在 dspark_target_layer_ids（58/59/60）收集 aux_hidden_states
    └─ DSpark:
        ① combine_hidden_states：main_proj 拼接投影 + main_norm
        ② precompute_and_store_context_kv：预计算并写入各 draft 层 SWA KV
        ③ 3 个 Draft DecoderLayer（复用主模型层，is_draft_layer=True）
        ④ hc_head：把 4 条并行残差流压回 1 条
        ⑤ markov_head：给出 n-gram 加性偏置（采样时逐位置施加）
        ⑥ confidence_head：估计每个 draft 位置的接受概率
        → 一整个 draft block 的 logits / 候选 token / 置信度
```

### 1.2 模块层级

```
DSparkDeepseekV4ForCausalLM  (L400; SupportsEagle3)
├── lm_head: ParallelLMHead                               词表并行输出头（L417-421）
├── logits_processor: LogitsProcessor                     logits 后处理（L422）
└── model: DeepseekV4DSparkModel  (L108)
    ├── embed_tokens: VocabParallelEmbedding              token 嵌入（L122-127）
    ├── context_wkv_proj: MergedColumnParallelLinear      3 层 WKV 合并投影（L156-164）
    ├── main_proj: ColumnParallelLinear(gather_output=True)  3 层隐藏状态拼接投影（L172-180）
    ├── main_norm: RMSNorm(7168)                          融合后归一化（挂第 61 层，L181-183）
    ├── norm: RMSNorm(7168)                               最终归一化（挂第 63 层，L185）
    ├── markov_head: DSparkMarkovHead                     上游实现导入（L41/L188-196）
    │     ├── markov_w1 = nn.Embedding(129280, 512)       复制版嵌入（串行免通信）
    │     └── markov_w2 = ParallelLMHead(129280, 512, disable_tp=True)  复制版投影
    ├── confidence_head: DSparkConfidenceHead             接受概率头（L198-203）
    │     └── proj = ReplicatedLinear(7168+512, 1)         fp32 标量输出
    ├── hc_head_fn   Parameter (4, 4*7168) fp32           残差流混合权重（L209-212）
    ├── hc_head_base Parameter (4,)        fp32           （L213-216）
    ├── hc_head_scale Parameter (1,)       fp32           （L217-220）
    └── layers: ModuleDict{61,62,63 -> DeepseekV4DecoderLayer(is_draft_layer=True)}
          61 层额外挂 main_proj/main_norm；63 层额外挂 norm/markov_head/hc_head_*
          confidence_head 不挂层，保持 model 级（L198-203）
```

> **与早期实现的关键差异**：v0.32.0 的 Markov 头**不再在 Ascend 侧自定义**，而是从上游导入 `DSparkMarkovHead` / `DSparkConfidenceHead`（`dspark.py:41`）。`markov_w1` 仍是普通 `nn.Embedding`，`markov_w2` 改为 `ParallelLMHead(..., disable_tp=True)`（不切分、整份复制），并通过 `logits_processor` 做词表投影；`main_proj` 仍是 `ColumnParallelLinear(gather_output=True)`。

### 1.3 与主模型的交互点

| 交互点 | 说明 |
|--------|------|
| `aux_hidden_states` | 主模型在 58/59/60 层收集的隐藏状态，拼接后供 `main_proj` 融合 |
| `context_states / positions / slot_mapping` | prefill 上下文，用于预计算 draft 层 SWA KV |
| `DeepseekV4DecoderLayer` | 直接复用主模型 Transformer 层（DSA 注意力 + MoE + 并行残差流） |
| `is_draft_layer=True` | 使 MoE 的 hash 路由关闭（[model.py:345-352](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py#L345)：`if is_draft_layer: self.hash = False`） |
| `mtp.*` 命名空间 | DSpark 权重在 checkpoint 里挂 mtp 前缀，加载时大量重映射 |

---

## 二、工具函数与常量 — 基础篇

### 2.1 `_CONTEXT_WKV_RE` / `_duplicate_context_wkv_weights`（[L83-L105](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L83)）

```python
_CONTEXT_WKV_RE = re.compile(r"^mtp\.(\d+)\.attn\.wkv\.(.+)$")

def _duplicate_context_wkv_weights(weights, num_layers):
    for name, weight in weights:
        yield name, weight
        match = _CONTEXT_WKV_RE.fullmatch(name)
        if match is None:
            continue
        layer_idx = int(match.group(1))
        if layer_idx >= num_layers:
            continue
        stacked_weight = weight.detach()
        stacked_weight.shard_id = layer_idx
        yield f"context_wkv_proj.{match.group(2)}", stacked_weight
```

它把每层 `mtp.{i}.attn.wkv.*` 权重**额外**复制一份并打上 `shard_id=i`，供合并的 `context_wkv_proj` 一次性加载；原权重仍保留，用于 draft token 的单层前向。用显存换一次合并投影，避免上下文预计算时逐层启动 3 个线性算子。

### 2.2 `_apply_dsv4_rope`（[L55-L69](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L55)）

```python
cos, sin = rope if rope is not None else get_cos_and_sin_dsa(positions)
layer_name = rotary_emb.layername
cos_t = cos[layer_name]
sin_t = sin[layer_name]
if inverse:
    sin_t = -sin_t
return rotary_emb(x, cos_t, sin_t)
```

不同压缩率的层 RoPE 基座可能不同，因此 cos/sin 按 `layername` 区分。draft 层是 dense，通常只对应 `default` rope 组。`precompute_and_store_context_kv` 会一次性按 3 层 `layername` 取 cos/sin（`dspark.py:311-315`），后续逐层复用。

### 2.3 `_get_dspark_num_mtp_layers`（[L72-L76](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L72)）

```python
num_layers = getattr(config, "n_mtp_layers", None)
if num_layers is None:
    num_layers = getattr(config, "dspark_num_mtp_layers", 3)
return int(num_layers or 3)      # 优先级 n_mtp_layers > dspark_num_mtp_layers > 3
```

---

## 三、Markov 头与 Confidence 头 — 基础篇

### 3.1 DSparkMarkovHead — Markov n-gram 偏置头（上游 [qwen3_dspark.py:L48-L93](../../../../../vllm/vllm/model_executor/models/qwen3_dspark.py#L48)）

> **生活化类比**：Markov 头像一个"**语言习惯数据库**"。它记得"我吃饭"后面常接"了"、"今天天"后面常接"气"，用一个二元（n-gram）转移模型给常见搭配加分，让草稿更通顺。

```python
class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size, draft_vocab_size, markov_rank, prefix, ...):
        # Markov 解码对每个 draft 位置都是串行执行的；两个低秩权重都复制整份，
        # 使每一步都不产生跨卡通信。
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)          # (129280, 512)
        self.markov_w2 = ParallelLMHead(draft_vocab_size, markov_rank,
                                        bias=False, disable_tp=True, ...)  # (129280, 512)

    def embed(self, token_ids):
        return self.markov_w1(token_ids)

    def bias(self, markov_embed, logits_processor):
        return logits_processor(self.markov_w2, markov_embed)
```

| 成员 | 实际类型 | shape | 为什么是复制而不是并行 |
|------|---------|-------|----------------------|
| `markov_w1` | `nn.Embedding` | `(129280, 512)` | Markov 每个 draft 位置串行查表，复制可免 all-gather |
| `markov_w2` | `ParallelLMHead(disable_tp=True)` | `(129280, 512)` | `disable_tp` 使其不按 TP 切分、整份复制；复用 `logits_processor` 的 LM head 接口 |

`bias(next) = W2 · W1[prev]`，作为主 logits 的**加性偏置**；注意 v0.32.0 的 `bias` 需要外部传入 `logits_processor`（`dspark.py:377-378`、入口类 `:477-478`）。

### 3.2 DSparkConfidenceHead — 接受概率头（上游 [qwen3_dspark.py:L164-L189](../../../../../vllm/vllm/model_executor/models/qwen3_dspark.py#L164)）

```python
class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim, prefix, bias=False, with_markov=True):
        self.with_markov = with_markov
        self.proj = ReplicatedLinear(input_dim, 1, bias=bias,
                                     params_dtype=torch.float32, ...)

    def forward(self, hidden, markov_embed):
        x = torch.cat([hidden, markov_embed], dim=-1).float()
        return self.proj(x).squeeze(-1)      # (T,) 标量 logit
```

**作用**：它输入 `[head_hidden || markov_embed]`（`7168 + 512`），输出每个 draft 位置的**接受概率 logit**；入口类 `compute_confidence`（`dspark.py:480-483`）再套 `sigmoid`，得到 `(B, K)` 的逐位置接受概率。

- **不参与草稿 token 生成**：draft logits 只由 `compute_logits + markov_bias` 决定，confidence head 不改 token；
- **用于动态验证预算**：`DynamicSpecScheduler`（[utils.py:151-172](../../../../../vllm-ascend/vllm_ascend/worker/v1/spec_decode/utils.py#L151)）用它把每请求的 `(K,)` 置信度累乘成生存概率，再分配实际验证长度（`_compute_dspark_token_probs`，[utils.py:336-387](../../../../../vllm-ascend/vllm_ascend/worker/v1/spec_decode/utils.py#L336)）。置信度低的块少验证，避免把算力浪费在必被拒绝的草稿上；
- **上游同源行为**：vLLM v2 的 `DSparkSpeculator._sample_sequential` 也会记录 `draft_token_confidence_probs`（[speculator.py:L192-L199](../../../../../vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L192)）。
- **权重不再跳过**：checkpoint 中最后一层的 `confidence_head.*` 会通过 `_remap_dspark_name` 映射到 `model.confidence_head.*` 并真实加载（`dspark.py:641-642`）。

---

## 四、DeepseekV4DSparkModel — 主体模型（[L108-L396](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L108)）

### 4.1 构造与关键字段（[L109-L226](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L109)）

```python
config = vllm_config.speculative_config.draft_model_config.hf_config
self.hc_mult = config.hc_mult                       # 4
self.hidden_size = config.hidden_size               # 7168
self.block_size = int(config.dspark_block_size)     # 5
self.target_layer_ids = list(config.dspark_target_layer_ids)  # [58,59,60]
self.num_dspark_layers = _get_dspark_num_mtp_layers(config)   # 3
self.mtp_start_layer_idx = config.num_hidden_layers           # 61
```

**词嵌入、draft 层与 OProj 缓冲池**：

```python
self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size, ...)        # L122-127
self.layers = nn.ModuleDict({
    str(61 + idx): DeepseekV4DecoderLayer(vllm_config, prefix=f"mtp.{idx}",
                                          is_draft_layer=True)                  # L128-137
    for idx in range(self.num_dspark_layers)
})
self._oproj_buffer_pool = OProjBufferPool()                                     # L138
```

层字典的 key 从 61 开始（延续主模型编号），但模块 prefix 仍是 `mtp.0/1/2`。OProj 缓冲池在 61/62/63 三层间共享（`L138-L144`），减少重复的 o_proj 临时显存。

### 4.2 多层融合 main_proj（[L166-L183](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L166)）

```python
_model_quant_cfg = getattr(config, "quantization_config", None)
_main_proj_qconfig = (vllm_config.quant_config
                      if _model_quant_cfg is not None
                      and _model_quant_cfg.get("quant_method") == "fp8" else None)
self.main_proj = ColumnParallelLinear(
    hidden_size * len(target_layer_ids),   # 7168*3 = 21504 输入
    hidden_size,                           # 7168 输出
    bias=False, return_bias=False,
    quant_config=_main_proj_qconfig,       # 仅 fp8 量化时启用
    prefix="...layers.61.main_proj",
    gather_output=True,                    # 列并行但输出全聚合，每层拿到完整 7168
)
self.main_norm = RMSNorm(hidden_size, ...)
first_layer.main_proj, first_layer.main_norm = self.main_proj, self.main_norm
```

> **生活化类比**：猜下一段时不只看最后一段，而是把倒数第 3、2、1 段拼起来一起读。`main_proj` 就是把这三段"揉成一条"的投影。
>
> 数据流：`(T, 21504) → main_proj → (T, 7168) → main_norm → (T, 7168)`（`combine_hidden_states`，[L231-L232](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L231)）。

### 4.3 输出端：norm / MarkovHead / ConfidenceHead / hc_head 参数（[L185-L226](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L185)）

`norm`、`markov_head` 以及三个 fp32 参数 `hc_head_fn (4, 4*7168)`、`hc_head_base (4,)`、`hc_head_scale (1,)` 都建在 model 上，再挂到**最后一层**（63 层）上，供层内直接引用（`L204-L223`）；`confidence_head` 则保持在 model 级（`L198-L203`），由入口类的 `compute_confidence` 直接调用。`markov_head` 的 `draft_vocab_size` 取 `getattr(config, "draft_vocab_size", None) or config.vocab_size`（`L187`）；本文配置为 full-vocab，因此 draft id 与目标 id 相同。

### 4.4 上下文 KV 预计算（[L228-L327](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L228)）

v0.32.0 新增/调整了以下方法，配合把 prompt 上下文的 SWA KV 提前算好写入缓存：

1. `get_draft_kv_cache_layer_names`（[L228-L229](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L228)）：返回 3 个 draft 层的 `swa_cache_layer.prefix`，供 runner 识别草稿层 KV。
2. `combine_hidden_states`（[L231-L232](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L231)）：`main_norm(main_proj(aux))`，把 58/59/60 三层拼成一条融合上下文。
3. `_project_shared_kv`（[L234-L254](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L234)）：`kv_norm → 在 k_pe 视图上原地做 RoPE → view 成 (T,1,head_dim)`。不再走 `split → rope → cat → contiguous` 的冗余拷贝（`L243-L253`）。
4. `_store_standard_swa_kv`（[L256-L289](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L256)）：1D `slot_mapping` 先经 `get_dsa_attn_kv_plan(...).format_dsa_slot_mapping` 按 block size 整形，再 `dsa_kv_compress_scatter` 散射进 SWA cache。若层带 `turboquant`，先对 KV 做同一旋转基变换再 `write_dsa_cache`（`L276-L283`），保证 draft query 与 context KV 同基。
5. `precompute_and_store_context_kv`（[L291-L327](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L291)）：先用合并的 `context_wkv_proj` 一次算出所有层的 KV（`L303-L307`），再一次性按 3 个 `layername` 取 cos/sin（`L311-L315`），逐层 `_project_shared_kv`、`_store_standard_swa_kv`（`L316-L327`）。
6. `context_wkv_proj`（[L156-L164](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L156)）：`MergedColumnParallelLinear(hidden_size, [head_dim] * 3, disable_tp=True)`；权重由 `_duplicate_context_wkv_weights` 从各层 `attn.wkv` 复制而来（`L86-L105`，加载时 `L534`）。

> **考前小抄类比**：正式"打草稿"前先把课本知识点抄成小抄（预计算上下文 KV），之后每个 draft 位置直接查缓存，不必对 prompt 重算注意力。

### 4.5 forward（[L329-L363](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L329)）

```python
def forward(self, input_ids, positions):
    hidden_states = self.embed_tokens(input_ids).unsqueeze(-2).repeat(1, self.hc_mult, 1)  # (T,4,7168)
    full_num_tokens = positions.shape[0]
    if self.use_sequence_parallel_moe:
        ... sp_padding_mask / sp_shard ...
    residual = None
    for layer in self.layers.values():                 # 顺序过 61→62→63
        hidden_states, residual = layer(positions, hidden_states, residual,
                                        llama_4_scaling=None, input_ids=input_ids)
    if self.use_sequence_parallel_moe:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)[:full_num_tokens]
    head_hidden = self.hc_head(hidden_states, self.hc_head_fn,
                               self.hc_head_scale, self.hc_head_base)   # (T, 7168)
    return head_hidden
```

嵌入后先复制成 4 条并行残差流，再走与主模型完全一致的 DecoderLayer，最后用 `hc_head` 压回 1 条。v0.32.0 新增 SP 分支（`L336-L361`）：`sp_shard` 切 token 维、逐层计算后 all-gather 并截回 `full_num_tokens`；`input_ids` 也随 hidden 一起分片并传给层（`L354`）。

### 4.6 hc_head — 并行残差流压缩（[L365-L372](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L365)）

```python
def hc_head(self, x, hc_fn, hc_scale, hc_base):        # x: (T, 4, 7168)
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()                            # (T, 4*7168)，fp32 计算
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)  # 手写 RMSNorm
    mixes = torch.nn.functional.linear(x, hc_fn) * rsqrt  # (T, 4) 每条流的混合分
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps          # (T, 4) 门控
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)                # (T, 7168)
    return y.to(dtype)
```

> **注意**：DSpark 的 `hc_head` 仍是**手写 `rsqrt` 归一化**（在 flatten 后的全宽上算），并**没有**改用主模型/MTP 使用的 `hc_norm`（RMSNorm 模块）。它对 4 条并行残差流各学一个 sigmoid 门控权重，加权求和压成 1 条。

### 4.7 logits / 专家映射

- `compute_logits`（[L380-L386](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L380)）：`logits_processor(lm_head, self.norm(hidden_states))`，先最终 `norm` 再词表投影。
- `get_expert_mapping`（[L388-L396](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L388)）：调用 `fused_moe_make_expert_params_mapping` 生成 `(param_name, weight_name, expert_id, shard_id)` 列表，供加载专家权重。
- `markov_embed/markov_bias`（[L374-L378](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L374)）转发给 `markov_head`；`markov_bias` 额外接收 `logits_processor`。

### 4.8 Block 内非因果注意力与顺序 Markov 采样（运行时行为，重要）

> **考试打草稿类比**：正常因果注意力像写正式答卷——写第 3 个字只能看前 2 个字；DSpark 出草稿块时像在草稿纸上写——块内每个字都能看到**同一块的其他字**，大家一起"商量"出最连贯的整句；随后再用 Markov 头从左到右把块内 token 的依赖关系补齐。

DSpark 复用了主模型的 DSA 层，但在 draft block 生成阶段通过运行时 metadata 打开非因果可见性：

| 阶段 | causal | 可见范围 |
|------|:------:|---------|
| 主模型 prefill/decode | True | 当前位 + 滑窗历史（标准因果） |
| 上下文 KV 预计算 | True | 标准因果 mask |
| **draft block 前向** | **False** | 滑窗历史 + **整个 block 互相可见** |
| **draft block 采样** | 顺序（从左到右） | 第 i 个 token 依赖第 i-1 个 token 的 Markov 偏置 |

**证据一：proposer 关闭因果**（[dspark_proposer.py:L342-L354](../../../../../vllm-ascend/vllm_ascend/worker/v1/spec_decode/dspark_proposer.py#L342)）：

```python
if hasattr(self.model, "get_draft_attn_causal"):
    cad.causal = self.model.get_draft_attn_causal()[0]
else:
    cad.causal = False          # DeepSeek V4 DSpark 没有该方法 → 走这里
cad.attn_mask = None
cad.attn_state = AscendAttentionState.ChunkedPrefill
```

**证据二：扩大窗口 + 构造非因果索引**：

- `get_dspark_sparse_sas_window`（[dsa_v1.py:L542-L546](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L542)）返回 `(sliding_window + num_speculative_tokens - 1, 0)`，左窗口扩到覆盖整个 block；
- `build_dspark_swa_indices`（[dsa_v1.py:L554-L568 起](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L554)）按块表构造 slot 索引，让块内每个 query 都能看到"尾部历史窗口 + 整块 draft"，越界列置 -1。

**证据三：块内依赖由顺序 Markov 采样注入**（[llm_base_proposer.py:L1575-L1596](../../../../../vllm-ascend/vllm_ascend/worker/v1/spec_decode/llm_base_proposer.py#L1575)）：

```python
logits = raw_logits.view(-1, num_speculative_tokens, vocab)   # 并行前向一次得到整块 logits
draft_token_ids[:, 0].copy_(self._dspark_seed_buffer[:num_blk])  # 首列 = 锚点/种子
for idx in range(num_speculative_tokens):
    markov_emb = self.model.markov_embed(draft_token_ids[:, idx])  # 依赖前一个已采样 token
    logits_bias = self.model.markov_bias(markov_emb)
    logits[:, idx].add_(logits_bias)
    if use_probabilistic:                                          # 概率采样
        token_ids, probs = self._sample_draft_from_logits(logits[:, idx], sampling_metadata)
    else:                                                          # argmax
        token_ids = logits[:, idx].argmax(dim=-1)
        if dspark_has_vocab_mapping:
            token_ids = self.model.map_draft_to_target(token_ids)
    draft_token_ids[:, idx + 1].copy_(token_ids)
```

即：**backbone 前向是块内并行的（非因果），但 token 采样是从左到右的**——第 i 个位置用第 i-1 个已采样 token 的 Markov 嵌入做加性偏置，形成 semi-autoregressive drafting（上游 [speculator.py:L3-L20](../../../../../vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L3) 明确描述了这一差异）。采样完成后，`dynamic_spec.update` 用 confidence head 估计各位置接受率并决定验证长度（`llm_base_proposer.py:1602-1611`）。

**为什么草稿可以"作弊"**：draft 只是提案，最终要过主模型的**因果**验证，概率过低的 token 会被拒绝并从目标分布重采样；双向注意力 + Markov 偏置只提高草稿质量/接受率，不破坏自回归正确性。

---

## 五、入口类 DSparkDeepseekV4ForCausalLM（[L400-L673](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L400)）

```python
@support_torch_compile
class DSparkDeepseekV4ForCausalLM(nn.Module, DeepseekV2MixtureOfExperts, SupportsEagle3):
    draft_id_to_target_id = None
```

- `__init__`（[L403-L423](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L403)）：读 draft config；量化时计算 `rotation_path`（`L409-L411`，用于旋转权重加载）；建 `DeepseekV4DSparkModel`（`L413-L416`）、`ParallelLMHead`（`L417-L421`）、`LogitsProcessor`（`L422`），最后 `set_moe_parameters()`（`L423`）。
- `set_moe_parameters`（[L425-L442](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L425)）：遍历 draft 层收集所有 `DeepseekV4MoE`（跳过 `PPMissingLayer`），用**最后一个** MoE 作 example（前面可能是稠密层），再 `extract_moe_parameters`。
- `forward`（[L444-L453](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L444)）：直接转发给 `self.model`。
- `compute_draft_logits`（[L455-L457](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L455)）：full-vocab 草稿，直接返回基础 logits，无 d2t scatter。
- `map_draft_to_target`（[L459-L460](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L459)）：full-vocab 时草稿 id 即目标 id，原样返回。
- `compute_logits`（[L462-L472](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L462)）：`del spec_step_idx`——DSpark 一次出整块，不按 spec 步索引。
- `markov_embed/markov_bias`（[L474-L478](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L474)）：转发给 model；`markov_bias` 传入 `self.logits_processor`。
- `compute_confidence`（[L480-L483](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L480)）：`sigmoid(confidence_head(head_hidden, markov_embed))`，返回逐位置接受概率。
- 委托方法（[L485-L501](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L485)）：`get_draft_kv_cache_layer_names`、`combine_hidden_states`、`precompute_and_store_context_kv`。
- `load_weights`（[L503-L629](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L503)）与 `_remap_dspark_name`（[L631-L673](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L631)）见第六章。

---

## 六、权重加载与映射 — 进阶篇（[L503-L673](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L503)）

### 6.1 准备工作

```python
expert_mapping = self.model.get_expert_mapping()                       # L509
stacked_params_mapping = [                                              # L513-L518
    ("mlp.gate_up_proj", "mlp.gate_proj", 0),
    ("mlp.gate_up_proj", "mlp.up_proj", 1),
    ("shared_experts.gate_up_proj", "shared_experts.gate_proj", 0),
    ("shared_experts.gate_up_proj", "shared_experts.up_proj", 1),
]
weights = _duplicate_context_wkv_weights(weights, self.model.num_dspark_layers)  # L534
```

### 6.2 逐权重处理顺序

1. **embed/head 复用**（`L537-L540`）：非旋转量化（`not self.rotation_path`）时 `embed.weight → model.embed_tokens.weight`、`head.weight → lm_head.weight`，与主模型共享；带旋转路径时保留原 name 交给 `_remap_dspark_name`。
2. **hc_head 参数直连**（`L541-L542`）：`hc_head_fn/base/scale → model.*`。
3. **DSpark 重命名**（`L544-L547`）：`_remap_dspark_name` 返回 None 就跳过（非 mtp 权重属于目标模型）。
4. **draft 自带 embed/head 探测**（`L549-L551`）：`process_eagle_weight(self, name)` 标记/处理独立 embed/lm_head。
5. **缩放后缀**（`L554-L555`）：所有 `.scale` 统一改 `.weight_scale`（Ascend convention）。v0.32.0 不再区分 fp4/fp8 的 `expert_scale_suffix`。
6. **`context_wkv_proj`**（`L557-L567`）：命中后按合成权重的 `shard_id` 调 `param.weight_loader`，把每层 WKV 送进合并投影的对应 shard。
7. **vision router bias 跳过**（`L569-L576`）：`e_score_correction_bias_vl` 在 DSpark 文本草稿中不存在，直接忽略；不 alias 到文本 correction bias，避免开启投机解码改变文本路由。
8. **专家权重**（`L578-L596`）：走 `expert_mapping` 的 expert-aware `weight_loader`（含冗余专家）。
9. **堆叠融合**（`L598-L608`）：仅对 `model.layers.` 下的参数，把 `gate_proj/up_proj` 装进融合的 `gate_up_proj`（shard 0/1）。
10. **attn_sink 切片**（`L610-L622`）：走参数的 `weight_loader`，按 TP 头段 `loaded_weight[head_start:head_end]` 切片；v0.32.0 已无 `enable_dsa_cp()` 分支。
11. **默认回退**（`L623-L626`）：其余参数用默认/自定义 `weight_loader` 直接加载。

### 6.3 `_remap_dspark_name`（[L631-L673](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dspark.py#L631)）

- `context_wkv_proj.*` 直接加 `model.` 前缀（`L632-L633`）；
- 正则 `mtp\.(\d+)\.(.*)` 提取 stage 与 rest，不匹配返回 None（`L635-L639`）；
- **最后一层的 `confidence_head.*` → `model.confidence_head.*`**（`L641-L642`，真实加载，不再跳过）；
- `mtp.0.embed.weight → model.embed_tokens.weight`；最后一层 `head.weight → lm_head.weight`；`hc_head_* → model.*`（`L644-L649`）；
- 层号路由：`main_proj/main_norm → 61 层`，`norm/markov_head → 63 层`，其余 `61 + stage`（`L651-L659`）；
- 组件名替换：`.attn.→.self_attn.`、`.ffn_norm.→.post_attention_layernorm.`、`.attn_norm.→.input_layernorm.`、`.ffn.→.mlp.`、`.w1/.w2/.w3.→.gate_proj/.down_proj/.up_proj.`、`.mlp.gate.bias→.mlp.gate.e_score_correction_bias`（`L661-L672`）。

| checkpoint 名 | 模型内部名 |
|--------------|-----------|
| `mtp.0.main_proj.*` | `model.layers.61.main_proj.*` |
| `mtp.0.attn.*` | `model.layers.61.self_attn.*` |
| `mtp.1.ffn.*` | `model.layers.62.mlp.*` |
| `mtp.2.norm.*` / `mtp.2.markov_head.*` | `model.layers.63.norm.*` / `.markov_head.*` |
| `mtp.2.confidence_head.*` | `model.confidence_head.*` |
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
  3. embed → 复制 4 条并行残差流 → 顺序过 61/62/63 层（块内非因果注意力）
  4. hc_head 压回 1 条 → norm → lm_head 得整块主 logits
  5. 从左到右逐位置采样：markov_bias(markov_embed(prev)) 作为加性 n-gram 偏置
  6. 采样完成后 compute_confidence → 每位置接受概率（供动态验证预算）
验证
  7. 主模型用标准因果注意力并行验证，接受匹配前缀，拒绝处按目标分布重采样
```

| 张量 | shape |
|------|-------|
| `aux_hidden_states` | `(T, 7168*3)` |
| `combined_hidden` | `(T, 7168)` |
| `hidden_states`（过层时） | `(T, 4, 7168)` |
| `shared_kv` | `(T, 1, head_dim)` |
| `hc_head_fn` / `base` / `scale` | `(4, 28672)` / `(4,)` / `(1,)` |
| `head_hidden` / `logits` | `(T, 7168)` / `(T, 129280)` |
| `markov_embed` / `markov_bias` | `(T, 512)` / `(T, 129280)` |
| `confidence` | `(B, K)`，每位置一个 `[0,1]` 接受概率 |

---

## 八、DSpark vs MTP — 专家篇

| 特性 | MTP（[6_deepseek_v4_mtp.md](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dsv4_docs/6_deepseek_v4_mtp.md)） | DSpark（本文） |
|------|------|------|
| 预测方式 | 串行单步 | 一次前向覆盖一个 block（5 token），采样时按 Markov 顺序补依赖 |
| 层数 | `num_nextn_predict_layers=1` | 3 个 draft 层（61/62/63，dense） |
| 层调度 | 按 `spec_step_idx` 轮询 | 顺序通过全部层 |
| 上下文来源 | 上一步残差流 + 下一 token 嵌入 | 主模型 58/59/60 三层隐藏状态融合 |
| 上下文融合 | `e_proj + h_proj` 相加 | `main_proj`（21504→7168）拼接投影 |
| KV 预计算 | 无 | `precompute_and_store_context_kv` + 合并 `context_wkv_proj` |
| Markov 偏置 | 无 | 有（上游 `DSparkMarkovHead`：`nn.Embedding` + `ParallelLMHead(disable_tp)`） |
| 块内注意力 | 因果 | **非因果（块内双向）**；采样顺序施加 Markov 偏置 |
| 残差流头 | `hc_norm`（RMSNorm 模块） | 手写 `rsqrt` 的 `hc_head` |
| Confidence head | 无 | 有（`DSparkConfidenceHead`，驱动动态验证预算） |
| 目标 | 保守、高接受率 | 激进、高吞吐、长上下文友好 |

---

## 九、设计要点小结

1. **Block drafter 而非串行 MTP**：一次前向覆盖 5-token 块，块内非因果注意力提升草稿质量，采样阶段再用 Markov 头从左到右补依赖。
2. **多层上下文融合**：`main_proj`（列并行、`gather_output=True`）把主模型最后 3 层隐藏状态揉成一条；融合模块挂 61 层。
3. **上下文 KV 预计算**：合并 `context_wkv_proj` 一次算出所有层的 WKV，再逐层写 SWA 缓存，显存换算子启动次数。
4. **Markov 头复用上游实现**：`nn.Embedding + ParallelLMHead(disable_tp=True)` + `logits_processor`，串行小算子零跨卡通信，输出作加性偏置。
5. **Confidence 头估计接受率**：`sigmoid(proj([hidden || markov_embed]))` 给出逐位置接受概率，供 `DynamicSpecScheduler` 分配验证预算，避免浪费验证算力。
6. **复用主模型层**：`DeepseekV4DecoderLayer(is_draft_layer=True)`，并因此关闭 MoE hash 路由；权重经 `_remap_dspark_name` 从 `mtp.*` 重映射，处理 gate_up 融合、`context_wkv_proj` 分片与 attn_sink TP 切片。

## 十、一句话总结

DSpark 是一个挂在 `mtp.*` 命名空间下、由 **3 个 dense draft 层 + 多层融合 `main_proj` + 手写 `hc_head` + 上游 Markov 头 + Confidence 头**组成的块草稿器：它融合主模型 58/59/60 层隐藏状态、预计算上下文 SWA KV，用**块内非因果注意力**一次前向得到整块 logits，再以**顺序 Markov 采样**生成 5 个候选 token 并用 confidence head 估计接受率，最后交给主模型用因果注意力批量验证，从而在不损失正确性的前提下换取更高的推测解码吞吐。
