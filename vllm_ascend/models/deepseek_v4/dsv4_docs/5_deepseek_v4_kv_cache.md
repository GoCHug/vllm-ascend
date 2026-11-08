# DeepSeek V4 KV 缓存体系与 Compressor/Indexer 详解

> **本文档详细讲解 DeepSeek V4 的三级 KV 缓存体系，包括 SWA（滑动窗口）、Compressor（KV 压缩）、Indexer（稀疏索引）的缓存布局、Spec、Block Size 与层间索引复用。**

> **代码版本冻结声明**：本文基于 **vllm-ascend v0.32.0** 源码整理。模型代码已拆包为 [model.py](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py)（1425 行）、[compressor.py](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/compressor.py)（232 行）、[indexer.py](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/indexer.py)（768 行）；文中未特别注明的 `Lxxx` 均指 **model.py** 行号（第二、三节分别指 compressor.py / indexer.py，其余文件单独标注）。block size 表在 [layer/attention/layer.py](../../../../../vllm-ascend/vllm_ascend/models/layer/attention/layer.py)，按层压缩比查询函数 `get_dsv4_compress_ratio` 在 [utils.py:L122](../../../../../vllm-ascend/vllm_ascend/utils.py#L122)。本版本三类缓存类**继承上游 vLLM 基类**（`vllm.models.deepseek_v4.compressor.CompressorStateCache` L139 / `vllm.models.deepseek_v4.attention.DeepseekV4IndexerCache` L882 / `vllm.v1.attention.backends.mla.sparse_swa.DeepseekV4SWACache` L74），只在 Ascend 侧覆写 spec 与后端选择。

> **统一模型配置**（W4A8 版，本文相关行）：`head_dim=512`、`qk_rope_head_dim=64`、`index_head_dim=128`、`index_n_heads=64`、`index_topk=1024`、`sliding_window=128`、`compress_rope_theta=160000`（dense 层用 `rope_theta=10000`）。`compress_ratios` 数组共 **64 项**：

| 下标区间 | 内容 | 计数 |
|---------|------|------|
| `[0:61]` 主模型 61 层 | 仅含 **128**（c128）和 **4**（c4）两种值，序列形如 `128,128,4,128,4,…`，末尾三层为 `4,128,4` | 31 个 c128 + 30 个 c4 |
| `[61:64]` 3 个 draft 层（MTP/DSpark） | **0（dense，无压缩）** | 3 个 0 |

---

## 一、三级 KV 缓存总览

### 1.1 设计理念

DeepSeek V4 采用 **DSA (Deep Sparse Attention)** 架构，用三级 KV 在 1M 上下文下平衡显存与精度：

> **生活化类比**：读一本厚书时，你不会把全书摊在桌上。SWA 是你眼前摊开的**最近 128 个字**（原文、随时可见）；Compressor 是你给每 4 页/128 页写的**章节摘要卡片**（压缩状态，记住梗概）；Indexer 是书后的**索引**——提问时先在摘要上打分，只翻出最相关的 1024 个历史块细看。

```
原始 KV (O(N) 显存)
    │
    ├── SWA（滑动窗口）   ：最近 128 个 token 的完整 KV，每层都有
    │
    ├── Compressor（压缩）：c4 层 / c128 层把历史 KV 压成状态向量
    │
    └── Indexer（稀疏索引）：仅 c4 层，在压缩状态上选 Top-1024 历史块
```

### 1.2 按层压缩配置

压缩比由 `get_dsv4_compress_ratio(config, layer_idx)`（[utils.py:L122-L127](../../../../../vllm-ascend/vllm_ascend/utils.py#L122)）给出：

```python
def get_dsv4_compress_ratio(config, layer_idx) -> int:
    compress_ratios = getattr(config, "compress_ratios", None)
    if compress_ratios is None or layer_idx >= len(compress_ratios):
        return 0                      # 缺省 / 越界一律视为 dense
    return compress_ratios[layer_idx]
```

| `compress_ratio` | Compressor | Indexer | SWA | RoPE 基座 | 本配置中的层 |
|------------------|:-----------:|:-------:|:---:|-----------|-------------|
| **0（dense）** | ✗ | ✗ | ✓ | `rope_theta=10000` | 3 个 draft 层（数组下标 61~63） |
| **4（c4）** | ✓ `overlap=True` | ✓ TopK=1024 | ✓ | `compress_rope_theta=160000`，rope 组 `["default","c4"]` | 30 个主层 |
| **128（c128）** | ✓ `overlap=False` | ✗ | ✓ | `compress_rope_theta=160000`，rope 组 `["default","c128"]` | 31 个主层 |

> **注意**：`compress_ratios` 中 dense 的取值是 **0 不是 1**（早期文档写 1）。Compressor 构造器也只接受 4/128，传入其他值会在 [compressor.py:L181-L183](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/compressor.py#L181) 抛 `ValueError`。
>
> **层号映射**：Attention 初始化时不是直接用运行时层号，而是先经 `extract_dsv4_layer_index`（[utils.py:L99-L112](../../../../../vllm-ascend/vllm_ascend/utils.py#L99)）转换——MTP 层运行时叫 `mtp.0`，查配置数组时映射为 `num_hidden_layers + 本地层号`（即 61、62、63），正好对应数组末尾的 3 个 0（model.py L498、L565）。

### 1.3 模块挂载关系（DeepseekV4Attention，L483）

```
DeepseekV4Attention (L483)
│
├── swa_cache_layer: AscendDeepseekV4SWACache       [所有层，L637-L643]
│
├── compressor: Compressor        (L610-L619)       [compress_ratio > 1，即 c4/c128]
│   ├── wkv / wgate: ReplicatedLinear               — KV / 门控投影（A5 不量化）
│   ├── norm: RMSNorm(head_dim, dtype=fp32)         — 固定 fp32 权重
│   ├── ape: Parameter (ratio, coff*head_dim) fp32  — 压缩位置编码
│   └── state_cache: AscendCompressorStateCache     — 压缩状态 KV
│
└── indexer: DeepseekV4Indexer    (L621-L632)       [仅 compress_ratio == 4]
    ├── wq_b: ReplicatedLinear(1536 → 64*128)       — Indexer Query 投影（走 quant_config）
    ├── weights_proj: ReplicatedLinear(7168 → 64)   — 头间权重投影（不量化）
    ├── k_cache: AscendDeepseekV4IndexerCache       — Indexer 自身 KV
    └── compressor: Compressor(head_dim=128, rotate=True)  — Indexer 专用压缩器
```

三者的投影权重与缓存最终都打包进 `DSAModules`（[ops/dsa.py:L39](../../../../../vllm-ascend/vllm_ascend/ops/dsa.py#L39)，model.py L645-L658），交给 `AscendDeepseekSparseAttention`（model.py L660-L680）在后端 kernel 中执行。与旧版不同，Python 侧的 `Compressor.forward`（compressor.py L201-L232）与 `DeepseekV4Indexer.forward`（indexer.py L437-L497）已是**真实实现**：前者调用融合算子 `torch.ops._C_ascend.compressor`，后者编排 Query 量化、压缩、Hadamard 旋转与 TopK 选择。

---

## 二、Compressor — KV 压缩器

**类定义**: [compressor.py:L105-L232](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/compressor.py#L105)（本节 `Lxxx` 均指 compressor.py）

### 2.1 作用与原理

Compressor 把滑窗"吐出来"的历史 KV 按 `compress_ratio` 个 token 一组压成一个状态向量（`kv_state + score_state`），后续注意力读的是状态而非原始 KV。主注意力的 Compressor `head_dim=512`；Indexer 内部的 Compressor 用 `head_dim=index_head_dim=128` 且 `rotate=True`（[indexer.py:L357-L367](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/indexer.py#L357)）。

### 2.2 核心成员（L106-L183）

| 成员 | 形状 / dtype | 说明 |
|------|-------------|------|
| `dim` | 7168 | 输入隐藏维（L123） |
| `head_dim` | 512（主）/ 128（Indexer 内） | 压缩头维度，构造参数 |
| `rope_head_dim` / `nope_head_dim` | 64 / head_dim−64=448 | RoPE 与非 RoPE 切分（L125-L126） |
| `compress_ratio` | 4 或 128 | 压缩比（L127） |
| `overlap` | bool | `ratio == 4` 时为 True（L128） |
| `coff` | int | `1 + overlap`：c4=2，c128=1（L131） |
| `norm_eps` | float | RMSNorm epsilon（L130） |
| `ape` | `(ratio, coff*head_dim)` **fp32** | 压缩块内位置编码（Absolute Position Embedding，L133） |
| `wkv` / `wgate` | `dim → coff*head_dim` | KV / 门控投影；**A5（`DSV4_COMPRESSED_CACHE`）上 `quant_config=None`（不量化）**，非 A5 才走量化（L134-L153） |
| — | — | 两者都置 `skip_weight_nz_conversion=True`（L156-L157）：定制压缩算子直接吃 ND 权重，不做 NZ 转换 |
| `norm` | `RMSNorm(head_dim)` | **固定 fp32 权重**（L160，kernel 只接受 fp32 norm_weight，并非仅 A5） |
| `state_cache` | `AscendCompressorStateCache` | 状态缓存，state_dtype 固定 **float32**（L162-L179） |

状态维度与 block：

| 层型 | `state_dim = kv_state + score_state` | state block_size（全局 128 档为例） |
|------|--------------------------------------|------------------------------------|
| c4（主注意力） | `2 × coff × head_dim = 4 × 512 = 2048` | `DSV4_BLOCK_SIZES[bs][0][2]` = 8（L170） |
| c4（Indexer 内，head_dim=128） | `4 × 128 = 512` | 同上（共享 c4 列） |
| c128 | `2 × head_dim = 1024` | `DSV4_BLOCK_SIZES[bs][0][3]` = 32（A5 FP8 / A5 BF16 为 16，L178） |

### 2.3 overlap 重叠机制（c4 专属）

c4 压缩时 `coff=2`，每个压缩块同时产出"与上一块重叠"的半区，保证块边界处信息连续。v0.32.0 已把块间拼接**下沉进融合算子**：`Compressor.forward`（L201-L232）只负责组织输入，并给 `torch.ops._C_ascend.compressor` 传 `coff=2 if self.overlap else 1`、`cmp_ratio=self.compress_ratio`、`rotary_mode=2`、`cache_mode=1`（L212-L231）；不再存在 Python 侧的 `overlap_transform`。

```python
compressed_kv = torch.ops._C_ascend.compressor(
    hidden_states, self.wkv.weight, self.wgate.weight,
    state_cache.squeeze(-2), self.ape, self.norm.weight,
    compress_sin.view(-1, compress_sin.shape[-1]),
    compress_cos.view(-1, compress_cos.shape[-1]),
    state_block_table=state_metadata.block_table,
    cu_seqlens=compressor_metadata.query_start_loc,
    start_pos=compressor_metadata.start_pos,
    cmp_ratio=self.compress_ratio,
    coff=2 if self.overlap else 1,
    rotary_mode=2, cache_mode=1,
)
```

### 2.4 RoPE 与状态写回（L185-L232）

压缩状态的位置编码同样不再由 Python 侧 `rope_single` 完成，而是由融合算子在 `rotary_mode=2` 下完成：cos/sin 由 `_compute_metadata`（L185-L199）预计算并复用（可等待 `DeviceMetadataStage.COMPRESSOR` 的就绪信号），算子按 `state_block_table` 与 `cu_seqlens` 直接把压缩结果写入 state cache。`forward` 返回 `(compressed_kv, slot_mapping)` 供上层注意力或 Indexer 使用。

### 2.5 AscendCompressorStateCache（compressor.py L53-L94）

继承上游 `CompressorStateCache`，`state_dim / dtype(fp32) / compress_ratio / block_size` 四个属性。上游还会按 `coff * compress_ratio` 设定 `sliding_window`（c4=8、c128=128），Ascend 侧覆写 `block_size`（L62-L64）。

`get_kv_cache_spec`（L66-L80）返回 `AscendSlidingWindowMLASpec`：

| 参数 | 取值 | 说明 |
|------|------|------|
| `block_size` | 构造时传入（c4→`[0][2]`，c128→`[0][3]`） | 状态块大小 |
| `num_kv_heads` | 1 | MLA 单 KV 头 |
| `head_size` | `state_dim` | 状态宽度 |
| `dtype` | float32 | 压缩状态始终 fp32 |
| `sliding_window` | `coff * compress_ratio`（8 / 128） | 状态窗口 |
| `alignment` | None | 无对齐填充 |
| `page_size_padded` | **精确条件**：`state_dim == 2*256 且 ratio==4` 时取 `pads[0]`，否则 `pads[1]`（L70） | 即 Indexer 的 c4 压缩器（state_dim=512）用 t1 填充页，主注意力 c4/c128 用 t2 |

`get_attn_backend()` 按压缩比分派（L84-L94）：c4 → `AscendDSAC4StateBackend`，c128 → `AscendDSAC128StateBackend`。

`AscendCompressorMetadata`（L98-L102）只是把压缩后的 KV cache 与 state cache 两个 metadata 句柄打包传给 `forward`。

---

## 三、Indexer — 稀疏索引器

**类定义**: [indexer.py:L297-L768](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/indexer.py#L297)（本节 `Lxxx` 均指 indexer.py）

### 3.1 作用

只挂在 c4 层。它用一套独立的低维注意力（64 头 × 128 维）对"压缩后的历史"打分，选出与当前 query 最相关的 `index_topk=1024` 个块，主注意力只对这些块做精细计算。

> 注意：Indexer 的 Query 输入不是原始 hidden，而是 MLA 低秩 Query 压缩向量 `qr`（`q_lora_rank=1536`），因此 `wq_b` 的输入维是 1536。

### 3.2 核心成员

| 成员 | 形状 | 说明 |
|------|------|------|
| `n_heads` / `head_dim` | 64 / 128 | `index_n_heads` / `index_head_dim`（L313-L314） |
| `rope_head_dim` | 64 | 复用 `qk_rope_head_dim`（L315） |
| `index_topk` | **1024** | `config.index_topk`（L316） |
| `q_lora_rank` | 1536 | 与主注意力共享的低秩 query 宽度 |
| `softmax_scale` | `head_dim ** -0.5` | 打分缩放（L318） |
| `wq_b` | `1536 → 64*128=8192` | ReplicatedLinear，**走 quant_config（W8A8 / MXFP8 等）**，`return_bias=False`（L323-L330） |
| `cv_wq_b` | — | `CVLinearWrapper`，把 `wq_b` 拆成 Vector 量化 + Cube 矩阵乘（L332） |
| `topk_indices_buffer` | — | 复用前序层 TopK 时必需，`skip_topk` 为真却为空会直接报错（L333-L335） |
| `ops` | `AscendIndexerOps` | Indexer 算子封装，`index_topk` 构造（L336） |
| `weights_proj` | `7168 → 64` | ReplicatedLinear，**`quant_config=None` 不量化**，把 64 个头的分布汇总成头权重（L337-L344） |
| `k_dtype` | fp8 / int8 | 由 `get_indexer_k_dtype` 解析（L345） |
| `k_cache` | `AscendDeepseekV4IndexerCache` | 仅 c4 时创建（L347-L355） |
| `compressor` | `Compressor(head_dim=128, rotate=True)` | 仅 ratio>1 时创建（L357-L367），与主注意力的 Compressor 是**两个独立实例**，wkv/wgate/norm/state_cache 各自独立 |

### 3.3 AscendDeepseekV4IndexerCache（indexer.py L136-L181）

继承上游 `DeepseekV4IndexerCache`，KV dtype 由 Indexer 构造时的 `k_dtype` 决定（L345）。`get_indexer_k_dtype`（L113-L133）把 `indexer_kv_dtype="auto"` 解析为该平台唯一实现的类型：**A5 → `float8_e4m3fn`（"fp8"），非 A5 → `torch.int8`（"int8"）**；传入其它值会提前报错，避免后续 ACLNN dtype 不匹配。

`get_kv_cache_spec`（L147-L167）返回 `AscendMLAAttentionSpec`：

| 参数 | 取值 | 说明 |
|------|------|------|
| `block_size` | `DSV4_BLOCK_SIZES[bs][0][0] * self.compress_ratio`（128 档：128×4=512） | 调度侧按原始 token 计数；物理页仍由 `storage_block_size` 决定（L151-L156） |
| `num_kv_heads` | 1 | 单 KV 头 |
| `head_size` | 128 | index_head_dim |
| `dtype` | A5：fp8；非 A5：int8 | KV 存储精度 |
| `model_version` | `"deepseek_v4"` | 版本标记 |
| `cache_dtype_str` | `cache_config.cache_dtype` | 原始量化串 |
| `tokens_per_state` | `self.compress_ratio`（4） | 替代旧字段 `compress_ratio`（vLLM #51718 改名） |
| `scale_dim` | `1 if head_dim==128 else 0` → 1 | 逐 token scale 放置维度 |
| `scale_dtype` | A5：`torch.float`；非 A5：`torch.float16` | 量化 scale 精度 |

`get_attn_backend()` 按压缩比分派（L171-L181）：c4 → `AscendDSAIndexerBackend`，c128 → `AscendDSAC128Backend`。

### 3.4 AscendIndexerOps（L198-L294）

新增的算子门面，把一次 Indexer 调用拆成可独立调度的步骤：

| 方法 | 行号 | 作用 |
|------|------|------|
| `unpack_dsa_indexer_kv_cache` | L205 | 拆出 `state_cache / key_cache / scale_cache / full_cache` |
| `quantize_query` | L208 | Query 量化，得到 `query + query_scale` |
| `quantize_key_and_update_cache` | L211 | K 量化并写入 K cache / full cache |
| `update_scale_cache` | L225 | 写回逐 token scale |
| `select_topk` | L237 | 调 `torch.ops._C_ascend.npu_quant_lightning_indexer_v2`，`cmp_ratio=4`、`layout_k="PA_BBND"`、`mask_mode=3` |
| `quantize_update_cache_and_select_topk` | L268 | 上述步骤的融合快捷路径（`indexer_quant_scatter`） |

---

## 四、SWA 滑动窗口缓存

### 4.1 AscendDeepseekV4SWACache（model.py L121-L157）

存储最近 `sliding_window=128` 个 token 的完整未压缩 KV，所有层（含 dense draft 层）都有。构造时基类先按 `torch.uint8` 建，再用 `self.dtype` 覆盖（L130-L133）；`block_size` 取 `DSV4_BLOCK_SIZES[bs][0][1]`（128 档=128，L135）。

| 特性 | A5（FP8 KV） | A5（BF16 KV）/ 非 A5 |
|------|----|-------|
| KV dtype | `float8_e4m3fn` | `bfloat16`（model.py L634-L640 传入） |
| `cached_head_size` | `head_dim + 96 = 608`（L140） | `head_dim = 512` |

`get_kv_cache_spec`（L137-L150）返回 `AscendSlidingWindowMLASpec`，含 `cache_dtype_str`、`model_version="deepseek_v4"`、`sliding_window=self.window_size`（128）、`alignment=None`。

> 多出的 **96 维**用于承载 FP8 KV 的 scale/元数据（rope 128 + nope 448 + bf16 scale 14 + pad 18 = 608 B/token），使 FP8 的 K/V 与缩放因子能放进同一个 paged 缓存。旧版的 `+128` 已不适用。
>
> `get_attn_backend()` 固定返回 `AscendDSASWABackend`（L154-L157）。

---

## 五、Block Size 配置

**定义位置**: [layer.py:L33-L65](../../../../../vllm-ascend/vllm_ascend/models/layer/attention/layer.py#L33)，`get_dsv4_block_sizes()` 按设备与 KV 布局返回三套表之一，模块加载时固化为全局 `DSV4_BLOCK_SIZES`（L57）与 `DSV4_BLOCK_SIZES_A5_BF16`（L58）。模型侧通过 `dsv4_block_sizes(vllm_config)`（L61-L65）懒加载引用，规避循环导入。

### 5.1 非 A5（L35-L39）

```python
_DSV4_BLOCK_SIZES = {
    128: [[128, 128, 8, 32], [16640, 131072]],
    64:  [[64,  64,  4, 16], [8320,  65536]],
    32:  [[32,  32,  2, 8],  [4160,  32768]],
}
```

### 5.2 A5 FP8 压缩缓存（L40-L44）

```python
_DSV4_COMPRESSED_BLOCK_SIZES = {
    128: [[128, 128, 8, 16], [16896, 77824]],
    64:  [[64,  64,  4, 8],  [8448,  38912]],
    32:  [[32,  32,  2, 4],  [4224,  19456]],
}
```

### 5.3 A5 BF16 KV（L45-L49）

```python
_DSV4_BLOCK_SIZES_A5_BF16 = {
    128: [[128, 128, 8, 16], [16896, 131072]],
    64:  [[64,  64,  4, 8],  [8448,  65536]],
    32:  [[32,  32,  2, 4],  [4224,  32768]],
}
```

选择逻辑（L50-L54）：A5（`DSV4_COMPRESSED_CACHE`）下，`use_a5_bf16_kv=True` 返回 BF16 表，否则返回 FP8 压缩表；其它设备返回通用表。因此 **A5 上全局 `DSV4_BLOCK_SIZES` 是 FP8 压缩表**，而 `dsv4_block_sizes(vllm_config)` 在 `cache_dtype` 为 `auto/bfloat16/bf16` 时切到 BF16 表（`is_a5_bf16_kv_enabled`，`dsa_attn_kv_plan.py` L45-L63）。A5 默认（`auto`）即 BF16 布局。

### 5.4 索引含义（注释见 layer.py L34）

第一组 `[0]` 四个 block size（128 档）：

| 索引 | 用途 | 非 A5 | A5 FP8 | A5 BF16 |
|------|------|-------|--------|---------|
| `[0][0]` | mla 主 KV / Indexer KV | 128 | 128 | 128 |
| `[0][1]` | SWA KV | 128 | 128 | 128 |
| `[0][2]` | c4 压缩状态 | 8 | 8 | 8 |
| `[0][3]` | c128 压缩状态 | 32 | **16** | **16** |

第二组 `[1]` 两个填充页大小（128 档）：

| 索引 | 用途 | 非 A5 | A5 FP8 | A5 BF16 |
|------|------|-------|--------|---------|
| `[1][0]` | `page_size_padded_t1`（Indexer c4 state，state_dim=512） | 16640 | 16896 | 16896 |
| `[1][1]` | `page_size_padded_t2`（主 c4/c128 state） | 131072 | **77824** | 131072 |

> A5 的 c128 状态块大小减半（32→16），FP8 下 t2 填充页也显著更小（131072→77824）；切到 BF16 KV 时 t2 恢复 131072。
>
> **取表差异**：主 MLA（`DSAAttention.get_kv_cache_spec`，layer.py L220）与 state 的填充页（`compressor.py` L69）走 `dsv4_block_sizes(vllm_config)`，会随 A5 BF16 切表；而 SWA（model.py L135）、state 的 `block_size`（compressor.py L170/L178）、Indexer 的存储块（indexer.py L149）直接使用全局 `DSV4_BLOCK_SIZES`。

### 5.5 TurboQuant（4bit latent cache）

当 `cache_dtype="turboquant_4bit_nc"`（[turboquant/__init__.py:L4-L7](../../../../../vllm-ascend/vllm_ascend/quantization/methods/kv_cache/turboquant/__init__.py#L4)）时：

- **c4 主 KV**：`dtype=torch.uint8`，`cached_head_size=SLOT_BYTES=512//2+2=258`（layer.py L212-L219），即每 token 258 B 的 4bit 打包页；c128 与 dense 层仍为 bfloat16。
- **SWA**：dtype 固定 `bfloat16`（model.py L635-L636），`cached_head_size=head_dim`，不做 4bit 打包。
- **写入**：`AscendDSAImpl._write_kv_cache`（[dsa_v1.py:L2355-L2373](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L2355)）对 c4 先 `turboquant.compress(kv)` 再按 int8 视图 scatter；SWA 仅做同基底的 `forward(kv)` 旋转，不量化。
- **平台**：仅非 A5 在 `supports_kv_cache_dtype` 中接受该 dtype（dsa_v1.py L303-L305）；A5 只接受 `auto/bfloat16/fp8`。

---

## 六、IndexCache — 层间 TopK 索引复用

代码位于 Attention 初始化 model.py L596-L608。开启后部分 c4 层不自己算 TopK，直接复用前序 c4 层的索引，省掉一次 Indexer 打分（参考论文链接见 L599 注释）。

**触发条件（三者同时）**：
1. 本层 `compress_ratio == 4`（确有 Indexer）；
2. `config.use_index_cache == True`（hf-overrides 显式打开，L596）；
3. 前缀不含 `.mtp.`——**MTP/draft 层被排除**（L605）。

> **MTP 细节**：MTP 层不参与复用。注释（L600-L603）解释：spec decode 只在**主模型层级**共享 `topk_indices_buffer`，impl 层拿不到对 MTP 有效的引用，强行复用会拿到陈旧索引。MTP 预测层改为**自建** TopK buffer（见文档 6 §MTP 部分）。
>
> **PP 细节**：复用决策由 `dsv4_skips_indexer_topk`（[utils.py:L130-L151](../../../../../vllm-ascend/vllm_ascend/utils.py#L130)）统一给出，并接收 `pp_start_layer`：每个 PP stage 的**第一个本地 c4 层必须自己算**（L147-L150），只有同 stage 内后续的 c4 层才允许复用。

### 6.1 频率模式（`index_topk_freq`）

```python
indexer_seq_idx = sum(ratio == 4 for ratio in compress_ratios[:layer_idx])  # utils.py L139
if pattern is None:
    freq = getattr(config, "index_topk_freq", 1)                            # L142
    skip_topk = max(indexer_seq_idx - 1, 0) % freq != 0                     # L143
```

- `indexer_seq_idx`：本层是第几个 c4 层（只数配置数组中本层之前的 `4`，用的是 **config_layer_idx**，因此 draft 层不会干扰计数）；
- 第 0 个 c4 层必算，之后每隔 `freq` 个 c4 层算一次，其余跳过。

### 6.2 模式串模式（`index_topk_pattern`）

```python
pattern = getattr(config, "index_topk_pattern", None)
assert pattern[0] == "F"                                        # 首个 c4 层必须算（L145）
skip_topk = indexer_seq_idx < len(pattern) and pattern[indexer_seq_idx] == "S"   # L146，越界保持 False
```

如 `"FSFS"` 表示第 0、2 个 c4 层计算（F），第 1、3 个复用（S）。

### 6.3 topk_indices_buffer

- 形状 `(max_num_batched_tokens, index_topk_tokens)`、int32，主模型在模型级创建一份并逐层传入 `DeepseekV4Attention(..., topk_indices_buffer=...)`（model.py L849-L863、L876）；
- `skip_topk=False` 的层写入新索引，`skip_topk=True` 的层直接读用（indexer.py L454-L455、L495-L496、L416-L435）；
- `skip_topk` 与 buffer 一起打包进 `DSAModules`（L645-L658），由 DSA 后端 kernel 消费；
- **MTP 预测层不共享此 buffer**，自建自管（见文档 6）。

---

## 七、KV 缓存 Spec 与后端体系

### 7.1 两类 Spec

| Spec 类型 | 定义 | 使用对象 |
|-----------|------|---------|
| `AscendSlidingWindowMLASpec`（[kv_cache_interface.py:L380](../../../../../vllm-ascend/vllm_ascend/core/kv_cache_interface.py#L380)） | 继承 `SlidingWindowMLASpec` | SWA Cache、Compressor State Cache |
| `AscendMLAAttentionSpec`（[kv_cache_interface.py:L208](../../../../../vllm-ascend/vllm_ascend/core/kv_cache_interface.py#L208)） | 继承 `MLAAttentionSpec` | 主 MLA KV、Indexer KV |

两类的 Ascend 专有字段：

| 字段 | 所在 Spec | 含义 |
|------|-----------|------|
| `cache_dtype_str` | 两者 | 原始 `cache_dtype` 字符串 |
| `model_version` | 两者 | DSV4 固定 `"deepseek_v4"` |
| `compress_ratio` | 仅 SlidingWindow | 状态页压缩比（默认 1，`storage_block_size = block_size // compress_ratio`） |
| `sliding_window` / `alignment` / `page_size_padded` | SlidingWindow | 窗口、对齐、填充页 |
| `tokens_per_state` | 仅 MLA | 替代旧 `compress_ratio`（vLLM #51718） |
| `scale_dim` / `scale_dtype` | 仅 MLA | 逐 token FP8/INT8 scale 布局 |
| `num_heads` / `indexes_kv_by_block_stride` | 仅 MLA | 每 rank query 头数、块步长寻址 |

> 同文件还有 `AscendSFAIndexerCacheSpec`（L312），用于 SFA 的 Indexer K/scale 独立张量；DSV4 的 Indexer 不走它，仍由 `AscendMLAAttentionSpec` 描述。

### 7.2 各缓存类的落点

| 缓存类 | 位置 | 基类 | spec | 后端 |
|--------|------|------|------|------|
| `AscendCompressorStateCache` | compressor.py L53 | 上游 `CompressorStateCache` | SlidingWindowMLA(fp32, padded page) | `AscendDSAC4StateBackend` / `AscendDSAC128StateBackend` |
| `AscendDeepseekV4IndexerCache` | indexer.py L136 | 上游 `DeepseekV4IndexerCache` | MLA(fp8/int8 + scale) | `AscendDSAIndexerBackend`（c4）/ `AscendDSAC128Backend`（c128） |
| `AscendDeepseekV4SWACache` | model.py L121 | 上游 `DeepseekV4SWACache` | SlidingWindowMLA(fp8/bf16, window=128) | `AscendDSASWABackend` |
| `DSAAttention` 主 MLA | layer.py L208 | 上游 `AttentionLayerBase` | MLA(tokens_per_state=compress_ratio) | `AscendDSAC4Backend` / `AscendDSAC128Backend` |

三个缓存类的 `forward()` 都是空实现（缓存本身只描述布局）；主 MLA 由 `DSAAttention` 负责布局描述。真正的读写在 `AscendDSAImpl` / 融合算子中随注意力一起完成（`vllm_ascend/attention/dsa_v1.py`，详见文档 3）。

### 7.3 按角色拆分的后端体系（dsa_v1.py）

| 后端类 | 行号 | 名称 | 元数据 Builder | 支持的调度块大小 |
|--------|------|------|----------------|------------------|
| `AscendDSABackend` | L272 | `ASCEND_DSA` | `AscendDSAMetadataBuilder`（L755） | `[2,4,8,16,32,64,128]` |
| `AscendDSAC4Backend` | L354 | `ASCEND_DSA_C4` | `AscendDSAAttentionMetadataBuilder`（L1780） | `[128,256,512]` |
| `AscendDSAC128Backend` | L370 | `ASCEND_DSA_C128` | `AscendDSAAttentionMetadataBuilder` | `[4096,8192,16384]` |
| `AscendDSASWABackend` | L386 | `ASCEND_DSA_SWA` | `AscendDSASWAMetadataBuilder`（L2119） | `[32,64,128]` |
| `AscendDSAC4StateBackend` | L400 | `ASCEND_DSA_C4_STATE` | `AscendDSAStateMetadataBuilder`（L1760） | `[2,4,8]` |
| `AscendDSAC128StateBackend` | L414 | `ASCEND_DSA_C128_STATE` | `AscendDSAStateMetadataBuilder` | A5：`[4,8,16]`；否则 `[8,16,32]` |
| `AscendDSAIndexerBackend` | L430 | `ASCEND_DSA_INDEXER` | `AscendDSAIndexerMetadataBuilder`（L2164） | 继承 C4：`[128,256,512]` |

- C4/C128 的块大小是**调度侧逻辑块**：物理 32/64/128-token 页分别代表 128/256/512 与 4096/8192/16384 个原始 token；Indexer 同样按 `storage_block_size * compress_ratio` 描述。
- Builder 选择由 `_get_dsa_role_builder`（L442-L461）完成；PCP 开启时改走 `vllm_ascend/attention/context_parallel/dsa_cp.py` 的同角色 Builder。
- 实现类 `AscendDSAImpl`（L2270）；metadata 侧也按角色拆成 State / Attention / SWA / Indexer 四种 Builder（L1760 / L1780 / L2119 / L2164），各角色只构建自己需要的 cache 元数据。

---

## 八、A5 设备适配汇总

| 特性 | A5 | 非 A5 |
|------|----|-------|
| SWA KV dtype | FP8 布局：`float8_e4m3fn`；BF16 布局（默认/显式 bf16）：`bfloat16` | `bfloat16` |
| SWA `cached_head_size` | FP8：512 + 96 = 608；BF16：512 | 512 |
| Indexer KV dtype | `float8_e4m3fn` | `int8` |
| Indexer scale dtype | `torch.float` | `torch.float16` |
| Compressor wkv/wgate | 不量化（`quant_config=None`） | 走 quant_config |
| Compressor norm 权重 | fp32（**所有设备均如此**） | fp32 |
| 压缩状态 dtype | float32 | float32 |
| c128 state block_size（128 档） | 16 | 32 |
| t1 / t2 填充页（128 档） | 16896 / 77824（FP8）；16896 / 131072（BF16） | 16640 / 131072 |
| TurboQuant（c4 主 KV 258 B/token） | 不支持 | 支持 |
| 默认 KV 布局（`cache_dtype="auto"`） | BF16 | bfloat16 |

---

## 九、一句话总结

V4 的每一层都带一份 **SWA 全量 KV**（窗长 128，FP8 时行宽 608 B）；30 个 c4 层额外挂 **Compressor（4:1 重叠压缩、fp32 状态）+ Indexer（64×128 低维注意力选 Top-1024）**，31 个 c128 层只挂 **128:1 非重叠压缩器**，3 个 draft 层为 dense（比值 **0**）；三类缓存继承上游基类、通过 `AscendSlidingWindowMLASpec / AscendMLAAttentionSpec` 描述布局，块大小与填充页由 `DSV4_BLOCK_SIZES` 按设备与 KV 布局（非 A5 / A5 FP8 / A5 BF16）查表；可选的 IndexCache 让 c4 层按频率或模式串复用 TopK 索引（每个 PP stage 的本地首个 c4 层必算），MTP 层被刻意排除。
