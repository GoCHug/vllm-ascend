# DeepSeek V4.1 Attention 架构详解（源共享 DSA + 两级索引）

> **本文档讲解 DeepSeek V4.1 的注意力子系统：低秩 MLA 投影、SWA 滑窗环形缓存、跨层共享的压缩 KV（C1/C2）、QLI+QSLI 两级索引选块，以及它们在 Ascend NPU 上如何被打包进一个融合稀疏注意力算子。**

> **代码版本冻结声明**：本文基于本仓库 `vllm_ascend/models/deepseek_v41/` 源码整理。核心文件与关键位置：
> - 模型侧封装 [model.py](../model.py)：`DeepseekV41LayerRole` L449、`DeepseekV41Topology` L468、`build_layer_plan` L510、`AscendDeepseekV41SWACache` L564、`DeepseekV41SWAAttention` L587、`DeepseekV41Attention` L678、`init_attention_projections` L380。
> - 压缩器 [compressor.py](../compressor.py)：`DeepseekV41Compressor` L15、`pool_projected` L55。
> - 索引器 [indexer.py](../indexer.py)：`DeepseekV41Indexer` L37、`update_keys` L116、`select` L137、`select_projected` L191。
> - 执行边界 [attention/dsa_v41.py](../../../attention/dsa_v41.py)：`dsa_v41_forward` L60、`AscendDSAV41Metadata` L140、`AscendDSAV41Impl` L303、`_preprocess` L401、`multistream_preprocess` L491、`_write_compressed_source` L548、`_select_sparse_indices` L618、`_forward_attention` L664、`forward` L749、`AscendDSAV41MetadataBuilder` L794。
> - 外层封装 [ops/dsa.py](../../../ops/dsa.py)：`DSAModules` L39、`AscendDeepseekSparseAttention` L57。
> - A5 打包算子绑定 [mixed_quant_attention.py](../mixed_quant_attention.py)：`run_a5_indexer` L178、`qsmla` L263、`write_attention_cache` L309、`write_index_cache` L337。
>
> 文中 `Lxxx` 未带文件名时指 `model.py`。上游参考实现见 [reference/model.py](reference/model.py) 的 `Attention`（L614）、`Compressor`（L429）、`Indexer`（L488）、`select_candidate_blocks`（L584）——它们用最直观的 PyTorch 写出了同一套语义，**强烈建议对照阅读**。

> **统一模型配置**（`DeepSeek-V4.1-Flash-1017`，TP 举例取 8）：

| 参数 | 值 | 含义 |
|------|-----|------|
| `hidden_size` | 5120 | 残差流宽度（`dim`） |
| `num_attention_heads` | 64 | 总 Query 头数；TP=8 时 `n_local_heads=8` |
| `head_dim` | 512 | 每头维度（MLA latent 宽度也是 512） |
| `qk_rope_head_dim` | 64 | 走 RoPE 的维度；剩余 `nope_head_dim=448` |
| `q_lora_rank` | 1280 | Query 低秩瓶颈 |
| `o_lora_rank` | 1024 | 输出低秩瓶颈 |
| `o_groups` | 8 | 输出分组数；TP=8 时 `n_local_groups=1` |
| `sliding_window` | 128 | SWA 窗口大小 |
| `index_n_heads` / `index_head_dim` | 32 / 128 | Indexer 的头数与头维度 |
| `index_topk` | 512 | 每个 Query 最终保留的压缩位置数 |
| `candidate_topk_blocks` / `candidate_block_size` | 2048 / 8 | 一级索引保留的候选块数 / 每块压缩位数 |
| `rope_theta` / `compress_rope_theta` | 10000 / 160000 | dense 层 / 长上下文层（`compress_ratio>0`）的 RoPE 基座 |
| `compress_ratios` | `0,0,2×18,1×20,0,0,0` | 逐层压缩比 |
| `kv_source_layer_ids` | `[2, 8, 14, 20]` | 真正写压缩 KV 的层 |
| `index_source_layer_ids` | `[2, 8, 14, 20, 24, 28, 32, 36]` | 真正跑索引的层 |
| `candidate_source_layer_id` | 20 | 一级索引（候选块）的产地 |

---

## 一、DSA 在 V4.1 里变成了什么

### 1.1 一句话定位

> **V4.1 的注意力仍然是 MLA + DSA（滑窗原文 + 压缩摘要 + 索引选块），但「谁来做压缩、谁来做索引」被彻底重新分配了：40 层里只有 4 层真正生产压缩 KV，只有 8 层真正跑索引，其余层直接借别人的缓存。**

在 V4 里，30 个 c4 层各建一个 Compressor + Indexer，31 个 c128 层各建一个 Compressor——**每层一份压缩历史**。V4.1 把这个开销砍成「4 份压缩 KV + 4 份索引 K + 8 次索引计算」，代价是同一压缩槽（slot）内的各层必须共用同一个压缩视角。

> **生活化类比**：V4 像**每个人各自做课程笔记**（30 份摘要、30 份索引，互不相通）；V4.1 像**班上只请 4 位速记员**（层 2/8/14/20），每人负责一段课程，全班共用他们的笔记。为了不让笔记写得太细导致来不及，速记员用两种档位：c1「每句都记」和 c2「两句并一句记」；为了翻笔记快，再配两级目录——先用「粗目录」挑 2048 个柜子（层 20 的 QLI），再在柜子里挑 512 份（层 24/28/32/36 的 QSLI）。

### 1.2 五层调用结构

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 第1层  DeepseekV41Attention                         models/deepseek_v41/model.py:L678 │
│   role = build_layer_plan(config).layer(layer_idx)        ← 决定本层是不是源 │
│   建 wq_a/wq_b/wkv/wo_a/wo_b/norm/attn_sink（init_attention_projections L380）│
│   源层才建 long_kv_cache（L717）+ compressor（L726）                            │
│   索引源才建 indexer（L731）                                                   │
│   把 role/topology/源前缀交给 v41_impl（L751），并注册到 static_forward_context │
│   L758-L760                                                                   │
└───────────────────────────────┬────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 第2层  AscendDeepseekSparseAttention                ops/dsa.py:L57                │
│   MultiHeadLatentAttentionWrapper 子类：持有 DSAModules（wq_a…attn_sink、      │
│   indexer、compressor、swa_cache_layer，dsa.py:L39），只做参数装配              │
└───────────────────────────────┬────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 第3层  forward：一个自定义算子边界                  model.py:L762-L765            │
│   torch.ops.vllm.dsa_v41_forward(hidden_states, output, self.v41_layer_name)   │
│   → dsa_v41_forward（dsa_v41.py:L60，@eager_break_during_capture）              │
│   通过 forward_context.no_compile_layers[layer_name] 取回模块                   │
└───────────────────────────────┬────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 第4层  AscendDSAV41Impl.forward                     dsa_v41.py:L749             │
│   ① _prepare_inputs_and_caches → _preprocess（Q/KV 投影 + 写缓存 + 压缩）       │
│   ② _select_sparse_indices（QLI / QSLI，产出 compressed_indices）               │
│   ③ _forward_attention（npu_sparse_flash_mla 或 qsmla）                          │
│   ④ 逆 RoPE → _forward_o_proj（分组 wo_a + wo_b），直接写进 output 缓冲          │
└───────────────────────────────┬────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 第5层  NPU 算子                                                                 │
│   传统设备：torch.ops._C_ascend.npu_sparse_flash_mla（dsa_v41.py:L716）          │
│             torch.ops._C_ascend.npu_quant_lightning_indexer_v3（indexer.py:L285）│
│   A5 设备：ops.mixed_quant_sparse_flash_mla（mixed_quant_attention.py:L285）     │
│             ops.quant_lightning_indexer / quant_sparse_lightning_indexer        │
│             torch.ops._C_ascend.kv_compress_epilog_v2（写打包缓存，L326）        │
└────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 `DeepseekV41LayerRole`：每层拿到的「角色卡」

`build_layer_plan(config)` 只算一次，产出一个不可变的 `DeepseekV41Topology`，每层一张角色卡（[L449-L466](../model.py#L449)）：

| 字段 | 含义 | 本配置下的取值示例 |
|------|------|-------------------|
| `layer_idx` | 层号 | 0..39（draft 层不参与） |
| `compress_ratio` | 压缩比 | 0 / 1 / 2 |
| `kv_source_layer` | 最近的 ≤ 本层的 KV 源 | 层 26 → **20** |
| `index_source_layer` | 最近的 ≤ 本层的索引源 | 层 26 → **24** |
| `is_kv_source` | 自己是不是 KV 源 | 只有 2/8/14/20 为 True |
| `is_index_source` | 自己是不是索引源 | 只有 2/8/14/20/24/28/32/36 为 True |
| `is_candidate_source` | 自己是不是候选块源地 | 只有 20 为 True |
| `uses_candidate_filter` | 二级索引（要过滤候选块） | 索引源 >20 的层为 True |
| `engram_slot` | 该层挂的 Engram 表槽位 | 层 1→0、层 14→1 |
| `has_long_context`（属性） | `compress_ratio > 0` | 决定 RoPE 是否上 YaRN |

三条派生规则（[L532-L551](../model.py#L532)）：

```python
kv_source    = _latest_source(layer_idx, kv_sources)   if ratio else None   # L533
index_source = _latest_source(layer_idx, index_sources) if ratio else None  # L534
uses_candidate_filter = index_source is not None and index_source > candidate_source  # L548
```

`DeepseekV41Topology` 还提供 `kv_consumers(source)` / `index_consumers(source)` 两个反查（[L483-L487](../model.py#L483)），排查「某源层影响了哪些层」时很好用。

---

## 二、投影：MLA 低秩 + 分组输出

### 2.1 完整形状表（TP=8）

`init_attention_projections`（[L380-L446](../model.py#L380)）一次性建好全部投影。TP=8 时：

| 模块 | 类型 | 权重形状（out, in） | 输出形状（T 个 token） | 说明 |
|------|------|---------|----------------------|------|
| `wq_a` | `ReplicatedLinear` | `(1280, 5120)` → 即 out=1280, in=5120 | `(T, 1280)` | 每 rank 一份完整副本（瓶颈本来就窄） |
| `q_norm` | `RMSNorm(1280)` | `(1280,)` | `(T, 1280)` | 带权重 |
| `wq_b` | `ColumnParallelLinear` | `(64×512, 1280)` = `(32768, 1280)` | `(T, 8, 512)` | 按头切；`head_dim=512` |
| `q_norm_without_weight` | `RMSNorm(512, has_weight=False)` | — | — | 无权重版，供特定算子路径使用（[L407](../model.py#L407)） |
| `wkv` | `ReplicatedLinear` | `(512, 5120)` | `(T, 512)` | **单头 latent**（`num_key_value_heads=1`），每 rank 复制 |
| `kv_norm` | `RMSNorm(512)` | `(512,)` | `(T, 512)` | 对 latent 归一化 |
| `wo_a` | `ColumnParallelLinear` | `(8×1024, 64×512/8)` = `(8192, 4096)` | `(T, 8, 1024)` | **分组**：每组只投影自己那 8 个头；`skip_weight_nz_conversion=True`（[L437](../model.py#L437)） |
| `wo_b` | `RowParallelLinear` | `(5120, 8×1024)` = `(5120, 8192)` | `(T, 5120)` | 行并行 + all-reduce（SP 模式下关掉，见 L836） |
| `attn_sink` | `Parameter` | `(8,)` fp32 | — | 注意力 sink，按头切分（加载时 `narrow`，见 L1921-1929） |

> **为什么 `o_groups=8` 而 V4 是 16**：分组输出让 `wo_a` 变成块对角矩阵，参数量和 FLOPs 都随组数线性下降。`o_groups` 必须能被 TP 整除（`n_local_groups = o_groups // tp_size`，[L391](../model.py#L391)）；TP=8 时每组恰好 8 个头、每 rank 1 组，`n_local_groups=1`。

### 2.2 一处必须记住的坑：`wo_a` 保持 ND

[L434-L437](../model.py#L434)：

```python
# Every DSA o_proj path consumes wo_a.weight directly via
# npu_transpose_batchmatmul / npu_transpose_quant_batchmatmul,
# so the weight must remain ND.
self.wo_a.skip_weight_nz_conversion = True
```

Ascend 的量化 GEMM 通常会把权重转成 NZ 格式；但 DSA 的 `o_proj` 是**分组批量转置乘**，直接吃 ND 权重（见 [dsa_v41.py:L483-L489](../../../attention/dsa_v41.py#L483) 里 `_forward_o_proj` 的调用，以及 reference 实现里 `torch.einsum("bsgd,grd->bsgr", o, wo_a)` 的写法）。把 `wo_a` 转成 NZ 会直接算错或崩。

### 2.3 RoPE：只转最后 64 维，且两档基座

- 旋转维度：`rope_head_dim=64`，位于 512 维 latent 的**尾部**（`nope_head_dim=448`）。旋转用 `DeviceOperator.apply_partial_rotary_inplace(..., start=nope_head_dim, end=head_dim)`（[dsa_v41.py:L342-L345](../../../attention/dsa_v41.py#L342)），即只对 `[448, 512)` 这一段做，且是 inplace 的——所以**逆旋转必须用同一段**。
- 基座由 `use_yarn=role.has_long_context` 决定（[L706](../model.py#L706)）传进 `ComplexExpRotaryEmbedding`：
  - `use_yarn=False`（`compress_ratio==0` 的层 0/1）：`base=rope_theta=10000`，不做 YaRN 外推，即纯滑窗注意力只关心最近 128 个位置。
  - `use_yarn=True`（`compress_ratio>0`）：`base=compress_rope_theta=160000`，并用 `factor=16`、`beta_fast=32`、`beta_slow=1`、`original_max_position_embeddings=65536` 做 YaRN。
- `rope_groups=["default"]`（[L624](../model.py#L624)）：`ComplexExpRotaryEmbedding` 用「配置指纹 + 组名」缓存整张 cos/sin 表（[ops/rope_dsv4.py:L233-L270](../../../ops/rope_dsv4.py#L233)），同配置的层共享同一张表，`get_cos_and_sin_dsa` 按 layer name 取用。

> **为什么压缩位用的是「组首位置」的频率**：压缩后的第 `j` 个 entry 代表原始位置 `j * ratio`（c2 时 `j*2`）。因此写压缩 KV 时取 `freqs[j*ratio]`，而不是当前 token 的位置。代码里由 metadata 直接提供 `c2_source_cos/c2_source_sin`（[dsa_v41.py:L584-L591](../../../attention/dsa_v41.py#L584)），传统路径则在 [reference/model.py](reference/model.py) L752-L761 用 `freqs_cis[::ratio]` 体现。

---

## 三、SWA：每层都有的 128-token 环形缓存

### 3.1 结构

每层（含 3 个 draft 层）都有一份 SWA 缓存，由 `AscendDeepseekV41SWACache` 构造（[L564-L584](../model.py#L564)）：

```python
block_size = DSV4_BLOCK_SIZES[cache_config.block_size][0][1]   # 逻辑 128 档 → 物理 128 token
spec = make_swa_cache_spec(block_size=..., window_size=128, head_size=512, ...)
```

| 属性 | 值 | 说明 |
|------|-----|------|
| `head_size` | 512（A5 打包后 544 B） | latent 宽度 |
| `num_kv_heads` | 1 | MLA 单头 latent |
| `sliding_window` | 128 | 窗口 |
| 物理块 | 128 token/块 | 来自 `DSV4_BLOCK_SIZES` 表 |
| `bounded_replay` | 由 `resolve_bounded_replay` 决定 | 开启后 `prefix_cacheable=False`、`prefix_replay_tokens=window`（[core/kv_cache_interface.py:L391-L397](../../../core/kv_cache_interface.py#L391)） |

### 3.2 写与读

- **写**：`_write_swa_cache`（[dsa_v41.py:L347-L359](../../../attention/dsa_v41.py#L347)）。A5 打包缓存走 `write_attention_cache(..., kind="win")`（mxfp8、group 32）；其它设备走 `npu_scatter_nd_update_sk`。
- **读**：不单独读——SWA 作为 `ori_kv` 与压缩 KV 作为 `cmp_kv` **拼在一次融合算子里**。可见行由 `metadata.swa.ori_sparse_indices` 与 `ori_topk_length` 给出；回调放时若 `replay_block_table` 非空，`swa_cache` 会被重定位（[dsa_v41.py:L709-L715](../../../attention/dsa_v41.py#L709)），细节见 [9_deepseek_v41_decoder_replay.md](9_deepseek_v41_decoder_replay.md)。
- **预填充 vs 解码的索引构造**差异，见 reference 实现 `get_window_topk_idxs`（[reference/model.py:L409-L426](reference/model.py#L409)）：prefill 每行是「自己的因果窗口」，decode 是「整个环，从最旧开始」；环未填满的槽位用 -1 标记。

### 3.3 A5 的预填充陷阱

`_write_swa_cache_on_main_stream`（[dsa_v41.py:L361-L368](../../../attention/dsa_v41.py#L361)）：**打包缓存**在「存在 prefill 或不是纯 decode」时，SWA 写必须在**主流**上做，不能挪到辅助流——因为 A5 的打包写算子在 prefill/mixed 场景下不能被独立重叠，而且「无法判定是纯 decode」也不算纯 decode。

---

## 四、压缩 KV：C1 与 C2 两种「速记档位」

### 4.1 两个档位的语义

| 档位 | `compress_ratio` | 池化方式 | 权重 dtype | 状态缓存 | 本配置中的源层 |
|------|:---:|---------|-----------|---------|--------------|
| **C1** | 1 | **不池化**：一个 token 一个压缩 entry，只是一次投影 + RMSNorm | `wkv` 为 **bf16** | **无** | 层 20 |
| **C2** | 2 | 2 个 token 用 **softmax 权重**加权求和成 1 个 | `wkv`/`wgate` 为 **fp32** | **32 行 fp32 环形状态**（`kv_state` + `score_state` 交织，`head_size=2*512=1024`） | 层 2/8/14 |

代码：`DeepseekV41Compressor.__init__`（[compressor.py:L15-L47](../compressor.py#L15)）里 `ratio == 2` 才会建 `wgate` 与 `state_cache`：

```python
self.wkv  = nn.Linear(dim, 512, bias=False, dtype=torch.float32 if ratio == 2 else torch.bfloat16)  # L21
self.norm = RMSNorm(512, eps=..., dtype=torch.bfloat16)                                             # L22
if ratio == 2:
    self.wgate = nn.Linear(dim, 512, dtype=torch.float32)                                           # L24
    ... self.state_cache = DeepseekV41CacheLayer(..., AscendCircularBufferSpec(                     # L36-L47
            block_size=STATE_RING_ROWS(32), num_kv_heads=1, head_size=2*width, dtype=torch.float32, ...))
```

> **为什么 C2 必须 fp32**：softmax 池化的分母/分子都在 fp32 里算才不会掉精度；而 C1 只是线性投影，保持 checkpoint 的 bf16 即可。这与训练端 reference 实现的选择完全一致（[reference/model.py:L444-L448](reference/model.py#L444)）。

### 4.2 C2 的「组未满」处理

一个 C2 组有两个 token：位置 `2j` 与 `2j+1`。只有**组满**（第二个 token 到来）时才产出 1 个压缩 entry。Ascend 侧把这个判断做成设备端掩码：

```python
complete = (input_positions.remainder(2) == 1) & valid      # dsa_v41.py:L1477
```

并把「组内两个 token 的位置」交给环形状态算子（`c2_ring_metadata`，[dsa_v41.py:L1418-L1525](../../../attention/dsa_v41.py#L1418)）去累加 `kv`/`score` 并在满组时池化，输出写到持久缓冲 `_ring_pooled`（[compressor.py:L29-L33](../compressor.py#L29)），再由 `pool_projected` 归一化（[compressor.py:L55-L67](../compressor.py#L55)）。

C1 没有「未满」问题，但**压缩槽位的换算**仍然是 `slot // ratio`：

```python
valid = (slot_mapping >= 0) & ((slot_mapping + 1) % ratio == 0)   # dsa_v41.py:L252
return torch.where(valid, slot_mapping // ratio, -1)               # L253
```

即「原始 token 的物理槽位」→「第几个压缩 entry」。`ratio=1` 时它退化为恒等映射（只过滤负值），`ratio=2` 时只在奇数位置（组尾）产生一次写。

### 4.3 写入顺序：一个不能反的依赖

源层写压缩缓存的顺序是固定死的（[dsa_v41.py:L548-L616](../../../attention/dsa_v41.py#L548)）：

```
kv/score 投影(C2) 或 直投(C1)
        ↓
latent（RoPE 之前！）
        ├──► ① indexer.update_keys(latent, index_slots, cos, sin)   ← 先写索引 K
        │        └─ 内部自己只对最后 64 维做 RoPE，再量化写入 index k_cache
        ↓
   ② 对 latent 的最后 64 维做 RoPE（就地）
        ↓
   ③ 写 long_kv_cache（kind="cmp"）
```

`update_keys` 的注释写得很直接（[indexer.py:L116-L117](../indexer.py#L116)）：

> *Publish source-owned index K before latent is RoPE'd as long KV.*

因为索引 K 是 `wk(latent)` 的投影，而 long KV 那一步会把 `latent` **就地**旋转成另一个东西；顺序一乱，索引 K 就会拿到被旋转过的 latent，TopK 全错但不会报错——这是 V4.1 最容易踩的精度坑之一。

### 4.4 压缩 KV 的读取者

消费者层不建 `compressor`，也不建 `long_kv_cache`，只在构造时记下**源层缓存的名字**（[L744-L747](../model.py#L744)）：

```python
root = prefix.rsplit(".layers.", 1)[0]                    # 例：model
source = f"{root}.layers.{role.kv_source_layer}.self_attn"
self.long_kv_source_prefix = f"{source}.long_kv_cache" if role.has_long_context else None
self.index_k_source_prefix = f"{source}.indexer.k_cache" if role.has_long_context else None
```

运行时 `_get_layer_metadata`（[dsa_v41.py:L325-L339](../../../attention/dsa_v41.py#L325)）用这两个前缀从 `forward_context.attn_metadata` 里取出**源层的 metadata**（块表、序列长度、压缩残余量），所以消费者用的是源层的页表——这就是「源共享」在元数据层的落地。

---

## 五、两级索引：QLI 粗筛 + QSLI 细挑

### 5.1 为什么要两级

1M 上下文 + c2 压缩后仍有 512K 个压缩位置。让 8 个索引源各自在全量位置上做 Top-512，既慢又没必要。V4.1 的做法是：

```
层 20（candidate_source）：
    QLI(mode=1) ──► 在全部压缩位置上打分，保留 2048 个候选【块】，每块 8 个压缩位
                     （= 16384 个位置的上限，实际按需）
         │
         ├──► 层 20 自己：在候选块内直接取 Top-512
         │
         ▼  共享给后面 4 个索引源
层 24/28/32/36：QLI/QSLI(mode=2 / QSLI)
    QSLI ──► 只在 2048×8 个候选位置里打细选 Top-512
```

覆盖率取舍：`2048 × 8 = 16384` 个压缩位 ≈ **32768 个原始 token**（c1 时 16384 个原始 token）的检索视野，另外 SWA 覆盖最近 128 个原始 token。

### 5.2 索引器的模块与形状

`DeepseekV41Indexer`（[indexer.py:L37-L110](../indexer.py#L37)），所有索引头在**每个 TP rank 上复制**（不做 TP 切分，注释 L39-L42 说明：这样每个 rank 独立得到完全相同的稀疏索引，省一次 all-reduce）：

| 模块 | 形状 | 说明 |
|------|------|------|
| `wq_b` | `ReplicatedLinear(1280, 32×128)` → `(T, 32, 128)` | 从 `qr`（Q 的 1280 维瓶颈）再投影，**不占 Q 主路径的 `wq_b`** |
| `weights_proj` | `ReplicatedLinear(5120, 32)` → `(T, 32)` | 每头一个权重，乘 `weights_scale = 128^-0.5 × 32^-0.5`（[L66-L67](../indexer.py#L66)） |
| `wk` | `nn.Linear(512, 128)` bf16 | **仅 `owns_k`**（4 个 KV 源）建 |
| `k_norm` | `RMSNorm(128)` | 同上 |
| `k_cache` | `DeepseekV41CacheLayer` | int8（其它设备）或 mxfp4（A5） |
| `k_cache_folded` | `DeepseekV41CacheLayer` | **仅** A5 + 候选源层：8 行压缩成一页（544 B），供 QSLI 使用 |

**打分公式**（`select_projected` 内，传统路径 [indexer.py:L264-L300](../indexer.py#L264)，A5 路径 [mixed_quant_attention.py:L53-L103](../mixed_quant_attention.py#L53)）：

```
score[t, h, j] = q_index[t, h, :] · k_index[j, :]          # 32 头 × 128 维
score[t, j]     = Σ_h relu(score[t,h,j]) * w[t,h]
topk            = topk(score[t, :], K=index_topk)          # K=512，排序后转位置
```

三个参数决定「走哪条路」（[indexer.py:L284](../indexer.py#L284)）：

```python
mode = 1 if is_candidate_source else 2 if uses_candidate_filter else 3
```

| `mode` | 层 | 行为 |
|:---:|-----|------|
| 1 | 20 | 打分 → **产出候选块** → 再在候选块内取 TopK |
| 2 | 24/28/32/36 | 打分 → **只在给定的候选块掩码内**取 TopK |
| 3 | 2/8/14 | 打分 → 在全量压缩位置取 TopK（无候选） |

### 5.3 候选块是怎么选出来的

语义最清楚的版本在 reference 实现 `select_candidate_blocks`（[reference/model.py:L584-L611](reference/model.py#L584)），四步：

1. **块分 = 块内最大分**：`pad(-inf) → unflatten(-1, (-1, 8)) → amax(-1)`。
2. **锁定最新块**：`last = (compress_lens - 1) // block_size`，把这一块的分数置 `+inf`——它只填了一半，却装着最新 token，否则会被更老但更满的块挤掉（这是训练/推理一致性的关键细节）。
3. **取前 2048 块**，并把 `-inf` 的补位丢掉（`top.values > -inf`）。
4. **展开回位置形状**：`keep.repeat_interleave(block_size, -1)[..., :width]`，得到一个与 `logits` 同形状的 **bool 掩码**——下游层只需 mask，不必理解「块」的概念。

Ascend 侧把这一步压进算子（`candidate_topk_index` / `candidate_mode` / `candidate_topk_blocks` / `candidate_block_size` 四个参数，[indexer.py:L285-L300](../indexer.py#L285)），并额外输出 `candidate_out`（块号）与 `candidate_length`（每 token 实际块数），由 `_select_sparse_indices` 拷进共享缓冲（[dsa_v41.py:L660-L661](../../../attention/dsa_v41.py#L660)）。

### 5.4 共享状态的传递

| 状态 | 生产者 | 消费者 | 缓冲 |
|------|--------|--------|------|
| `topk_indices` | 每个索引源覆写 | 自己 + 后续消费者层（`_select_sparse_indices` 直接返回 `shared.topk_indices[...]`，[dsa_v41.py:L623-L624](../../../attention/dsa_v41.py#L623)） | `(T, 512)` int32 |
| `candidates` | 层 20 | 24/28/32/36 | `(T, 1, 2048)` int32 |
| `candidate_lengths` | 层 20 | QSLI 元数据 | `(T, 1)` int32 |
| `topk_lengths` | 每个索引源 | A5 Q-SMLA 的 `cmp_topk_length` | `(T, 1)` int32 |

> **`reset()` 是空实现**（[L499-L503](../model.py#L499)）不是偷懒：源层总在消费者之前覆写有效行，保留存储地址可以让 ACL Graph 捕获到固定地址，回放时不依赖 Python 状态。

---

## 六、融合稀疏注意力算子

### 6.1 传统设备：`npu_sparse_flash_mla`

一次调用同时完成「滑窗原文 + 压缩摘要」两段 attention（[dsa_v41.py:L716-L741](../../../attention/dsa_v41.py#L716)）：

| 参数 | 取值 / 来源 | 含义 |
|------|------------|------|
| `q` | `(T, 8, 512)` | `layout_q="TND"`，每 rank 8 头 |
| `ori_kv` | SWA 缓存 | 滑窗原文 KV（`layout_kv="PA_BBND"`，分页块表寻址） |
| `cmp_kv` | 源层 `long_kv_cache` | 压缩摘要 KV；`compress_ratio=0` 时为 `None` |
| `ori_sparse_indices` | `metadata.swa.ori_sparse_indices` | 每个 query 可见的窗口槽位（-1 跳过） |
| `ori_topk_length` | `metadata.swa.ori_topk_length` | 每个 query 实际可见数量（band 模式下也用来表达「下界」） |
| `cmp_sparse_indices` | `pad_sparse_indices(compressed_indices, 512)` | 压缩位置 TopK，**补到固定 512** 列（[dsa_v41.py:L296-L300](../../../attention/dsa_v41.py#L296)） |
| `ori_block_table` / `cmp_block_table` | SWA / 源层的块表 | 两套页表可不同 |
| `cu_seqlens_q` | `metadata.swa.query_start_loc` | 变长打包 |
| `seqused_ori_kv` / `seqused_cmp_kv` | SWA / 压缩序列长度 | 解码时逐请求不同 |
| `cmp_residual_kv` | `metadata.attention.cmp_residual` | 压缩组残余（当前 query 落在未满组的偏移） |
| `sinks` | `attn_sink`（fp32） | 注意力 sink，稳定长上下文 softmax |
| `cmp_ratio` | 1 / 2 / 0 | 压缩比，算子内部据此换算压缩位坐标 |
| `ori_mask_mode` / `cmp_mask_mode` | `metadata.swa.ori_mask_mode` / `3` | 3 = 稀疏索引掩码 |
| `ori_win_left` / `ori_win_right` | 窗口左右边界 | 一个标量界定窗口 |
| `topk_value_mode` | `1` | TopK 语义变体（见下） |
| `softmax_scale` | `512^-0.5` | 由 `head_dim` 决定（[L715](../model.py#L715)） |

> **`topk_value_mode=1` 的来历**：V4.1 的 `topk_indices` 是**紧凑的 `[T, K]`**（末尾可能不足 512 列），而算子需要 `[T, 1, topk]` 的定长布局，所以 `pad_sparse_indices` 补 -1（[dsa_v41.py:L296-L300](../../../attention/dsa_v41.py#L296)），`topk_value_mode=1` 让算子把 -1 当作「跳过」。这一点与 V4 的语义差异最容易在对齐数值时出问题。

### 6.2 A5 设备：`qsmla`

A5 的缓存是**打包混合量化**的（FP8/FP4 值 + BF16 scale 交织），所以走另一套算子组合（[mixed_quant_attention.py:L263-L306](../mixed_quant_attention.py#L263)）：

```python
ops.mixed_quant_sparse_flash_mla(
    q, ori_kv=..., cmp_kv=...,
    ori_sparse_indices=ori_indices, cmp_sparse_indices=cmp_indices[:, None, :],
    ori_block_table=..., cmp_block_table=...,
    cu_seqlens_q=..., seqused_ori_kv=..., seqused_cmp_kv=...,
    ori_topk_length=..., cmp_topk_length=compressed_lengths,
    sinks=..., metadata=task_metadata, quant_mode=1, softmax_scale=...,
    layout_q="TND", layout_kv="PA_BBND")
```

与传统的两点差异：

1. **窗口索引**优先取 `metadata.swa.ori_sparse_indices`；没有则用 `build_window_indices_triton` 现算（[L252-L260](../mixed_quant_attention.py#L252)）。
2. **长度通过 `topk_lengths` 缓冲显式传入**（[dsa_v41.py:L671-L672](../../../attention/dsa_v41.py#L671)）——因为 A5 算子不接受 -1 填充的「隐式」长度。

### 6.3 A5 的缓存写入：`kv_compress_epilog_v2`

打包缓存不能用普通 scatter（字节布局不同），统一走 `write_attention_cache`（[mixed_quant_attention.py:L309-L334](../mixed_quant_attention.py#L309)）：

| `kind` | group_size | quant_mode | 说明 |
|--------|:---:|-----------|------|
| `"cmp"` | 16 | `mxfp4_bf16` | 压缩 KV：2 个 FP4/字节 + 每 16 值 1 个 BF16 scale |
| `"win"` | 32 | `mxfp8_bf16` | SWA：1 个 FP8/字节 + 每 32 值 1 个 BF16 scale |

字节换算见 [5_deepseek_v41_kv_cache.md](5_deepseek_v41_kv_cache.md) 第 3 章。

---

## 七、多流重叠：把 Vector 藏在 Cube 后面

`multistream_preprocess`（[dsa_v41.py:L491-L546](../../../attention/dsa_v41.py#L491)）在 `multistream_dsv4_dsa_overlap` 打开时生效，把 Q 与 KV 的投影切成三段互补的 Cube/Vector 交替：

```
主流(Cube)                辅助流(Vector)
─────────────             ───────────────────────────
wq_a 量化                 (等 q_quant)
wq_a matmul  ───────────► kv 量化        ← Part 1: Q Cube ∥ KV Vector
q_norm + wq_b 量化  ────► wkv matmul     ← Part 2: Q Vector ∥ KV Cube
wq_b matmul  ───────────► kv_norm + RoPE ← Part 3: Q Cube ∥ KV Vector
join（等辅助流）
写 SWA 缓存（必要时代主流执行）
```

三个工程细节：

1. **`share_quant`**（[L502-L506](../../../attention/dsa_v41.py#L502)）：当 `wq_a` 与 `wkv` 的量化方法/通信属性一致、且 query/cache 是同一批 hidden 时，**KV 直接复用 Q 的量化结果**，省一次 cast。
2. **A5 预填充的写缓存必须回主流**（`write_cache_on_main`，[L533-L545](../../../attention/dsa_v41.py#L533)），理由同 §3.3。
3. **索引权重的投影也单独上辅助流**（`_prepare_indexer_inputs`，[L445-L460](../../../attention/dsa_v41.py#L445)）：`weights_proj` 是纯 Vector 的 `(T,5120)→(T,32)`，正好填在 Q 的 RoPE 期间。

---

## 八、一次注意力前向的完整时序

```
forward(attn, positions, hidden_states, output)            dsa_v41.py:L749
│
├─(A) forward_context.attn_metadata is None  ← profiling/无元数据
│     output.zero_() + 跑一次 o_proj 以占满 workspace，然后返回        L756-L773
│
├─(B) metadata = _get_layer_metadata(...)      ← 取本层 SWA / 源层 long_kv / 源层 index_k / C2 状态   L325
│
├─(C) _prepare_inputs_and_caches → _preprocess                       L401
│     ├ qr = q_norm(wq_a(x)); q = wq_b(qr).unflatten(-1, (8,512))
│     ├ kv = kv_norm(wkv(x)); 部分 RoPE; 写 SWA 缓存
│     ├ q 部分 RoPE
│     ├ 索引源：投影 q_index / weights（辅助流）
│     └ KV 源：_write_compressed_source（先索引 K，再压缩 KV）
│
├─(D) _select_sparse_indices                                         L618
│     ├ 非长上下文层 → None
│     ├ 非索引源 → 直接返回 shared.topk_indices[:n]（复用！）
│     └ 索引源 → indexer.select_projected（QLI/QSLI）→ 写共享缓冲
│
├─(E) _forward_attention(q, metadata, compressed_indices)           L664
│     ├ 传统：npu_sparse_flash_mla(...)
│     └ A5  ：packed_cache_ops.qsmla(...)
│
├─(F) 逆 RoPE（只转最后 64 维）                                      L786
└─(G) _forward_o_proj 直接写进 output（避免多一次整张量拷贝）          L790
```

> **(G) 的收益**：`dsa_v41_forward` 是自定义算子，`output` 是图稳定的缓冲。V4.1 让 `o_proj` 直接写进这个缓冲（[L483-L489](../../../attention/dsa_v41.py#L483) 的 `_project_output`），省掉一次 `(T, 5120)` 的整张量拷贝——在 decode 的 token 数很小时，这次拷贝的 kernel launch 开销占比反而不低。

---

## 九、正确性检查清单（对齐数值时按这个顺序怀疑）

| # | 检查点 | 症状 | 位置 |
|---|--------|------|------|
| 1 | **索引 K 是否在 latent 被 RoPE 覆盖前写入** | 短上下文完全正常、长上下文答非所问 | [indexer.py:L116-L135](../indexer.py#L116) |
| 2 | **压缩位坐标是「组首」**（`freqs[j*ratio]`） | 长文本位置漂移、RoPE 相位错 | [dsa_v41.py:L584-L591](../../../attention/dsa_v41.py#L584) |
| 3 | **`candidate_block_size` 与 `compress_ratio` 配套** | 候选块覆盖范围错、召回骤降 | config `candidate_block_size=8` |
| 4 | **`pad_sparse_indices` 是否补到 512 且填 -1** | A5/非 A5 结果不一致 | [dsa_v41.py:L296-L300](../../../attention/dsa_v41.py#L296) |
| 5 | **`wo_a` 是否仍是 ND** | 输出全错或算子报格式错 | [model.py:L434-L437](../model.py#L434) |
| 6 | **dense 层是否误开了 YaRN** | 层 0/1 输出与参考差一个数量级 | [model.py:L706](../model.py#L706) |
| 7 | **消费者是否取到了正确的源层元数据** | 某几层正常、某几层崩 | [dsa_v41.py:L325-L339](../../../attention/dsa_v41.py#L325) |
| 8 | **C2 组满判定用 `remainder(2)==1`** | 压缩 KV 少一半或错位 | [dsa_v41.py:L1477](../../../attention/dsa_v41.py#L1477) |

---

## 十、FAQ

**Q1：为什么 40 层共享 4 份压缩 KV 还能不掉点？**
A：压缩 KV 是「历史摘要」，同一段历史在不同层的语义差异远小于注意力本身的差异；V4.1 的训练就是按这个拓扑做的（checkpoint 的 `kv_source_layer_ids` 是训练超参，不是推理优化开关）。反过来说，**不能**随手改这个数组——权重是对着它训出来的。

**Q2：层 20 是 c1（不压缩），那 20-39 层的压缩 KV 长度就是全量 token 数，会不会太占显存？**
A：会，但这是设计取舍：c1 段（后 20 层）承担最强的长上下文检索，c2 段（前 18 层）承担更早的历史。显存账见 [5_deepseek_v41_kv_cache.md](5_deepseek_v41_kv_cache.md) §6。

**Q3：`uses_candidate_filter` 为什么用 `index_source > candidate_source` 而不是 `layer_idx`？**
A：因为选择策略是**跟着索引源继承**的。层 26 从不自己算候选块，它复用层 24 的 TopK，而层 24 的 TopK 是在层 20 的候选块里选出来的（[model.py:L546-L548](../model.py#L546) 的注释就是这么写的）。

**Q4：`q_norm_without_weight` 是什么？**
A：一个 `has_weight=False` 的 `RMSNorm(512)`（[L407](../model.py#L407)），供不需要可学习缩放的路径使用。它通过 `DSAModules` 传给算子层（[ops/dsa.py:L39-L55](../../../ops/dsa.py#L39)），在 V4.1 的 Q 主路径上不参与计算（V4.1 没有 wq_b 之后的 Q RMSNorm）。

**Q5：A5 上为什么不走 `npu_quant_lightning_indexer_v3` 而走 QSLI？**
A：A5 的索引 K 是 **mxfp4 打包**的（2 值/字节 + E8M0 scale），QLI V3 吃的是 int8 分页布局。A5 走 `_qsli`（[mixed_quant_attention.py:L106-L175](../mixed_quant_attention.py#L106)），它要求源层有**折叠缓存**（`len(source_cache) != 3` 直接报错，L123-L124），因为 QSLI 一次读 8 行（544 B）而不是逐行读。

**Q6：`attn_sink` 为什么是 fp32 且按头切？**
A：sink 参与 softmax 的 logit 加法，必须在 fp32 里累加才稳；它按头切分（`narrow(0, head_start, heads_per_rank)`，[model.py:L1926](../model.py#L1926)），每个 TP rank 只持有自己那 8 个头。

---

*相关文档：[5_deepseek_v41_kv_cache.md](5_deepseek_v41_kv_cache.md)（缓存布局）、[9_deepseek_v41_decoder_replay.md](9_deepseek_v41_decoder_replay.md)（回放）、[0_deepseek_v41_arch.md](0_deepseek_v41_arch.md)（总览）。*
