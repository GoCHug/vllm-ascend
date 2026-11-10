# DeepSeek V4.1 KV 缓存体系与源共享布局详解

> **本文档讲解 DeepSeek V4.1 的 KV 缓存体系：为什么 40 层共享 4 份压缩 KV、SWA 环 / 压缩 KV / 压缩状态 / 索引 K / 折叠索引 K 五类缓存各自的 Spec 与字节布局、以及「层元组 + 全局 block-ID 池」这套 Ascend 专有的分配方式。**

> **代码版本冻结声明**：本文基于本仓库 `vllm_ascend/models/deepseek_v41/` 源码整理。核心文件与关键位置：
> - [model.py](../model.py)：`AscendDeepseekV41SWACache` L564、`DeepseekV41Attention.__init__` 里四类缓存的创建 L707-L743、共享缓冲 L1000-L1022。
> - [cache_config.py](../cache_config.py)：`STATE_RING_ROWS` L35、A5 判定 `uses_a5_packed_cache` L45、`make_swa_cache_spec` L50、`make_mla_cache_spec` L71、`make_index_cache_spec` L95、`make_folded_index_cache_spec` L121、`is_deepseek_v41_cache` L137、`get_layer_tuples` L159、`group_cache_specs` L199、`get_deepseek_v41_kv_cache_config` L271。
> - [compressor.py](../compressor.py)：`DeepseekV41Compressor` L15、状态环 L36-L47、`pool_projected` L55。
> - [indexer.py](../indexer.py)：`k_cache` L92、`k_cache_folded` L101、`update_keys` L116。
> - [attention/dsa_v41.py](../../../attention/dsa_v41.py)：`compressed_slot_mapping` L246、`scatter_cache_sk` L278、`AscendDSAV41MetadataBuilder` L794（`_cache_kind` 判定 L822-L836）。
> - [core/kv_cache_interface.py](../../../core/kv_cache_interface.py)：`AscendMLAAttentionSpec` L208、`AscendSlidingWindowMLASpec` L380、`real_page_size_bytes` L248 / L407、`get_storage_block_size` L50。
> - 调度器/分配补丁：[patch/platform/patch_kv_cache_utils.py](../../../patch/platform/patch_kv_cache_utils.py)（`_ascend_get_kv_cache_config_from_groups` 里对 `is_deepseek_v41_cache` 的分支）、[patch/worker/patch_bind_kv_cache.py](../../../patch/worker/patch_bind_kv_cache.py)（`DeepseekV41CacheLayer` 的绑定）。

> **统一模型配置**（本文相关行）：`head_dim=512`、`qk_rope_head_dim=64`、`sliding_window=128`、`index_head_dim=128`、`index_n_heads=32`、`index_topk=512`、`candidate_topk_blocks=2048`、`candidate_block_size=8`、`kv_source_layer_ids=[2,8,14,20]`、`index_source_layer_ids=[2,8,14,20,24,28,32,36]`、`candidate_source_layer_id=20`。

`compress_ratios`（43 项，前 40 项属于主干）：

| 下标 | 值 | 计数 | 层角色 |
|------|:---:|:---:|--------|
| `[0:2]` | **0** | 2 | 纯 SWA（无压缩） |
| `[2:20]` | **2** | 18 | c2；源层为 2 / 8 / 14 |
| `[20:40]` | **1** | 20 | c1；源层为 20 |
| `[40:43]` | **0** | 3 | DSpark draft 层（独立 `mtp.*` 树） |

---

## 一、总览：从「每层一份」到「4 份共享」

### 1.1 设计理念

> **生活化类比**：读一本 100 万字的书，V4 的做法是**每 61 个读者各自做一份摘要 + 一份索引**（所以房间要 61 份书架）；V4.1 只请 **4 位速记员**（层 2/8/14/20）各负责一段，其余 36 人**直接来借**。房间里因此只需要 4 份书架——但为了让「借」不出错，速记员必须把**页码**（压缩位坐标）和**目录卡**（索引 K）都额外整理好，并且规定好「谁在第几页」的租约（层元组）。

DSA 在 V4.1 里仍然由三级记忆构成，只是**归属**变了：

```
原始 KV（每 token 512 维 latent）
    │
    ├── SWA（滑动窗口，128 token）  ── 40 层 + 3 draft 层，每层一份（不可共享：每层 wkv 不同）
    │
    ├── 压缩 KV（long KV）          ── 只有 4 个 KV 源层拥有；其余 36 层按 kv_source_layer 就近借阅
    │      ├─ c2（层 2/8/14）：2 token → 1 entry，带 32 行 fp32 环形状态
    │      └─ c1（层 20）   ：1 token  → 1 entry，无状态
    │
    └── 索引 K（index K）           ── 只有 4 个 owns_k 层拥有；8 个索引源共享
           └─ A5 上额外一份「折叠索引 K」（8 行/页），供 QSLI 一次读 8 行
```

### 1.2 逐层缓存清单（本配置）

| 层 | SWA | 压缩 KV | 压缩状态 | 索引 K | 折叠索引 K | 备注 |
|:--:|:---:|:------:|:-------:|:-----:|:---------:|------|
| 0 | ✓ | — | — | — | — | dense |
| **1** | ✓ | — | — | — | — | dense + **Engram 槽 0** |
| **2** | ✓ | **✓ c2** | **✓** | **✓** | — | KV 源 + 索引源 |
| 3 – 7 | ✓ | 借层 2 | — | 借层 2 | — | 消费者 |
| **8** | ✓ | **✓ c2** | **✓** | **✓** | — | KV 源 + 索引源 |
| 9 – 13 | ✓ | 借层 8 | — | 借层 8 | — | 消费者 |
| **14** | ✓ | **✓ c2** | **✓** | **✓** | — | KV 源 + 索引源 + **Engram 槽 1** |
| 15 – 19 | ✓ | 借层 14 | — | 借层 14 | — | 消费者 |
| **20** | ✓ | **✓ c1** | — | **✓** | **✓（A5）** | KV 源 + 索引源 + **候选源** |
| 21 – 23 | ✓ | 借层 20 | — | 借层 20 | — | 消费者 |
| **24 / 28 / 32 / 36** | ✓ | 借层 20 | — | 借层 20 | 借层 20 | **二级索引源**（QSLI） |
| 其余 20-39 | ✓ | 借层 20 | — | 借层 20 | — | 消费者 |
| 40 / 41 / 42 | ✓ | — | — | — | — | DSpark draft（`DeepseekV41DSparkSWACache`） |

**合计物理缓存面（plane）**：

| 类别 | 数量 | 名字后缀 |
|------|:---:|---------|
| SWA | 43 | `…self_attn.swa_cache` |
| 压缩 KV | **4** | `…self_attn.long_kv_cache` |
| 压缩状态 | **3** | `…self_attn.compressor.state_cache` |
| 索引 K | **4** | `…self_attn.indexer.k_cache` |
| 折叠索引 K | **1** | `…self_attn.indexer.k_cache_folded`（仅 A5 + 候选源） |

> 注意 `_cache_kind` 的判定顺序（[dsa_v41.py:L822-L836](../../../attention/dsa_v41.py#L822)）：**先认「折叠索引 K」**（`tokens_per_state == 8 and head_size == 544`），再认 `index_k`（`scale_dim` 非 0），最后才是 `long_kv`。因为折叠页的 scale 是**内联**的（`scale_dim=0`），长得和 long KV 一模一样，认错顺序会把索引 K 当成压缩 KV 去寻址。

---

## 二、四类 Spec：字节布局的唯一真相

### 2.1 构造函数一览

`cache_config.py` 提供四个构造函数，全部落在 `model_version="deepseek_v41"` 上（这是调度器识别 V4.1 布局的标记，`is_deepseek_v41_cache`，[L137-L148](../cache_config.py#L137)）：

| 构造函数 | 产出的 Spec | 关键字段 |
|---------|------------|---------|
| `make_swa_cache_spec`（L50） | `AscendSlidingWindowMLASpec` | `sliding_window=128`、`bounded_replay`、`cache_dtype_str` |
| `make_mla_cache_spec`（L71） | `AscendMLAAttentionSpec` | `tokens_per_state=compress_ratio`、`scale_dtype` |
| `make_index_cache_spec`（L95） | `AscendMLAAttentionSpec` | `scale_dim` = 32（A5）或 1（其它）、`dtype` = uint8 / int8 |
| `make_folded_index_cache_spec`（L121） | `AscendMLAAttentionSpec` | `tokens_per_state=8`、`head_size=544`、`scale_dim=0` |

### 2.2 A5 打包布局的字节换算

A5（`DSV41_PACKED_CACHE`）把值和小数打包进**一段连续字节**，用位数换算（[cache_config.py:L50-L134](../cache_config.py#L50)）：

| 缓存 | 换算代码 | 每 token/entry 字节 | 拆解 |
|------|---------|:---:|------|
| **SWA** | `head_size += (head_size // 32) * 2`（L55） | **544** | 512 个 FP8（1 B/值）+ 16 个 BF16 scale（2 B/个）= 512 + 32 |
| **压缩 KV** | `head_size = head_size // 2 + (head_size // 16) * 2`（L76） | **320** | 256 字节装 512 个 FP4（2 值/字节）+ 32 个 BF16 scale（每 16 值 1 个）= 256 + 64 |
| **索引 K** | `scale_dim = head_size // 32; head_size //= 2`（L99-L100） | **68** | 64 字节装 128 个 FP4 + 4 字节装 32 个 E8M0 scale（每 32 值 1 个，uint8） |
| **折叠索引 K** | `8 * (128//2 + 128//32)`（L128） | **544 / 8 行** | 8 行 × (64 + 4) = 544，正好一页 |
| **压缩状态** | `head_size = 2 * width`, fp32（compress.py L43） | **4096** | `kv_state(512) + score_state(512)` × 4 B |

> **为什么是 544 而不是 512+32=544 恰好整除**：`block_size % INDEX_FOLD_ROWS` 必须为 0（[L123-L124](../cache_config.py#L123)），否则折叠页装不下完整的 8 组——逻辑块 32/64/128 都能被 8 整除，所以总是成立。

### 2.3 非 A5 布局

| 缓存 | dtype | 每 token/entry 字节 |
|------|-------|:---:|
| SWA | bfloat16 | 512 × 2 = **1024** |
| 压缩 KV | bfloat16 | 512 × 2 = **1024** |
| 索引 K | int8 + fp16 scale（`scale_dim=1`） | 128 + 2 = **130** |
| 压缩状态 | float32 | **4096** |

### 2.4 `real_page_size_bytes` 的统一公式

所有 MLA 类 Spec 的页宽都由同一条公式给出（[core/kv_cache_interface.py:L247-L253](../../../core/kv_cache_interface.py#L248)）：

```python
real_page_size_bytes = storage_block_size * num_kv_heads
                       * (head_size * itemsize(dtype) + scale_dim * itemsize(scale_dtype))
```

其中（[L50-L60](../../../core/kv_cache_interface.py#L50)）：

```python
storage_block_size = block_size // tokens_per_state
```

含义是「一个**调度块**（逻辑块）在物理上承载多少个 storage 行」。举例（逻辑块 128）：

| 缓存 | `tokens_per_state` | `storage_block_size` | 一页（128 逻辑块）字节数（A5） |
|------|:---:|:---:|:---:|
| SWA | 1 | 128 | 128 × 544 = **69,632** |
| c2 压缩 KV | 2 | 64 | 64 × 320 = **20,480** |
| c1 压缩 KV | 1 | 128 | 128 × 320 = **40,960** |
| 索引 K（c2 源） | 2 | 64 | 64 × 68 = **4,352** |
| 折叠索引 K | 8 | 16 | 16 × 544 = **8,704** |

> **经典混淆点**：调度器按 `block_size` 计块、按 token 计长度；算子按 `storage_block_size` 计物理行。`tokens_per_state=2` 的意思不是「一个块 2 个 token」，而是「一个 storage 行装 2 个原始 token 的压缩结果」。

---

## 三、压缩器：C2 的环形状态与 C1 的直投

### 3.1 C2 的 fp32 状态环

`ratio == 2` 的源层（2/8/14）会额外建一个环形状态缓存（[compressor.py:L36-L47](../compressor.py#L36)）：

```python
state_spec = AscendCircularBufferSpec if uses_a5_packed_cache() else CircularBufferSpec
self.state_cache = DeepseekV41CacheLayer(vllm_config, f"{prefix}.state_cache",
    state_spec(block_size=STATE_RING_ROWS, num_kv_heads=1,
               head_size=2 * self.width,      # 1024 = kv_state 512 + score_state 512
               dtype=torch.float32, head_size_v=0))
```

| 项 | 值 | 说明 |
|----|-----|------|
| `STATE_RING_ROWS` | **32**（[cache_config.py:L35](../cache_config.py#L35)） | 一块 32 行；一个 C2 组只用 2 行 |
| `head_size` | `2 × 512 = 1024` | `kv` 与 `score` 交织存放，算子一次读写 |
| dtype | **fp32** | softmax 池化的分子/分母都要 fp32 |
| `is_circular` | 是 | `is_circular_kv_cache_spec` 判真（[kv_cache_interface.py:L65-L70](../../../core/kv_cache_interface.py#L65)），块地址按环复用 |

**为什么是「环」**：C2 组只关心「当前未满的两行」。环固定 32 行，位置 `pos % 32` 决定写哪行，无需按序列长度增长。memory 上是常数（`max_num_seqs × 32 × 1024 × 4 B ≈ 128 KB/请求` 量级），几乎免费。

### 3.2 C2 的池化：在设备上算，不在 Python 里算

`pool_projected`（[compressor.py:L55-L67](../compressor.py#L55)）把 `kv`（`(T,512)`）与 `score`（`(T,512)`）交给 Triton 算子，由它按 `c2_ring_metadata` 决定「累加进环」还是「满组了，池化输出」：

```python
pooled = compressor_from_projected(
    kv, scores,
    self.state_cache.kv_cache[0].squeeze(-2),   # 32 行 fp32 环
    metadata.c2_ring_metadata,                  # 5 × num_reqs 的 5 行元数据
    self._ring_pooled[: kv.shape[0]],           # 持久输出缓冲
    max_query_len=metadata.max_query_len,
    num_cores=self._ring_num_cores)
return self.norm(pooled)
```

`c2_ring_metadata` 的 5 行（在 [dsa_v41.py:L1470-L1474](../../../attention/dsa_v41.py#L1470) 可见其 CPU 版构造）：

| 行 | 含义 |
|:--:|------|
| 0 | 请求已完成的 token 数（`seq_lens - query_lens`） |
| 1 | 本步该请求的有效 token 数（`skip_ring_update` 时清零） |
| 2 | 本步 query 起点 |
| 3 | 同上（供算子分块） |
| 4 | 该请求的物理块号（无有效 token 时为 0） |

`_ring_pooled` 是**在 memory profiling 之前**就分配好的持久缓冲（[compressor.py:L25-L33](../compressor.py#L25)），注释写得很直白：*Allocate persistent output before memory profiling, so its footprint is included in the cache budget rather than added after allocation.* 否则 profile 会漏算这块而 OOM。

### 3.3 C1：没有状态，但要「组满」判定

C1 的 `compress_ratio=1`，每个 token 就是一个 entry，所以 `compressor.forward` 就是 `norm(wkv(x))`（[compressor.py:L69-L71](../compressor.py#L69)）。但写入仍需 `compressed_slot_mapping` 换算物理槽位（[dsa_v41.py:L246-L253](../../../attention/dsa_v41.py#L246)）：

```python
valid = (slot_mapping >= 0) & ((slot_mapping + 1) % ratio == 0)
return torch.where(valid, slot_mapping // ratio, -1)
```

`ratio=1` 时恒成立（除了负槽位），`ratio=2` 时只在组尾触发一次写。

---

## 四、索引 K 与折叠索引 K

### 4.1 只有 `owns_k` 的层有索引 K

`DeepseekV41Indexer.__init__` 里 `if owns_k:` 才建 `wk / k_norm / k_cache`（[indexer.py:L84-L100](../indexer.py#L84)）。`owns_k` 来自构造参数的 `role.is_kv_source`（[model.py:L731-L743](../model.py#L731)）——也就是说，**索引 K 的产地和压缩 KV 的产地是同一批层**（2/8/14/20）。这与算法是一致的：索引 K 是「压缩 latent 的投影」，只有自己压缩的层才拿得到 latent。

### 4.2 折叠索引 K：为 QSLI 准备的 8 行页

候选源（层 20）在 A5 上额外建一份 `k_cache_folded`（[indexer.py:L101-L109](../indexer.py#L101)）：

```python
self.k_cache_folded = (DeepseekV41CacheLayer(vllm_config, f"{prefix}.k_cache_folded",
        make_folded_index_cache_spec(block_size=...))
    if is_candidate_source and uses_a5_packed_cache() else None)
```

每写一次索引 K，就同步折叠一次（[indexer.py:L126-L127](../indexer.py#L126)）：

```python
if self.k_cache_folded is not None:
    fold_indexer_cache_rows((k_cache, scale_cache), self.k_cache_folded.kv_cache[0], slots)
```

**为什么需要它**：QSLI（`_qsli`，[mixed_quant_attention.py:L106-L175](../mixed_quant_attention.py#L106)）一次读 8 行连续的 K，如果缓存是「一行一 entry」的稀疏分页布局，8 行可能落在 8 个不同页上；折叠缓存把**同一组的 8 行**压进一页（544 B），QSLI 只需一次连续读。所以 `_qsli` 才会断言 `len(source_cache) == 3`（[L123-L124](../mixed_quant_attention.py#L123)）：普通 K + scale + 折叠页。

### 4.3 索引 K 的写入必须早于压缩 KV

顺序约束见 [3_deepseek_v41_attn.md](3_deepseek_v41_attn.md) §4.3：`update_keys` 吃的是 **RoPE 之前**的 latent，而 `long_kv_cache` 写的是 RoPE **之后**的同一个张量（就地旋转）。顺序反了不会报错，只会悄悄让 TopK 全错。

---

## 五、层元组与全局 block-ID 池：Ascend 的分配方式

### 5.1 为什么不能「一层一页」

V4.1 每层带的缓存种类差别很大：普通层只有 SWA（69 KB/页），源层还有压缩 KV + 索引 K（20–41 KB + 4 KB），候选源还有折叠页（8.7 KB）。如果每种 Spec 单独成组，调度器会看到 50 多个异构组，混合组（hybrid group）的前缀缓存命中逻辑会炸。

`cache_config.py` 的做法是：**把「同一个 KV 源槽」上的所有缓存面塞进一个「层元组（layer tuple）」，一个元组共用一页**。

### 5.2 `get_layer_tuples`：元组的构造

```python
full        = 按层号排序的 long_kv_cache 名字        # 4 个，对应层 2/8/14/20
target_swa  = 主干 SWA 名字（排除 .mtp.）             # 40 个
draft_swa   = draft SWA 名字                         # 3 个

for slot_idx, kv_name in enumerate(full):            # slot 0..3
    prefix      = kv_name.rsplit(".", 1)[0]
    index_name  = prefix + ".indexer.k_cache"
    folded_name = index_name + "_folded"
    aliases = ([state[slot_idx]] if slot_idx < len(state) else []) \
            + target_swa[slot_idx :: len(full)] \    # ← 每个元组拿 10 个 SWA 层
            + ([draft_swa[slot_idx]] if slot_idx < len(draft_swa) else [])
    capacity = max(kv_bytes + index_bytes + folded_bytes,
                   *(specs[name].unpadded_page_size_bytes for name in aliases))
```

关键的两个设计：

1. **页宽取「max」而不是「sum」**（[L190-L193](../cache_config.py#L190)）：一个元组的页宽 = max(压缩 KV + 索引 K + 折叠页, 最大的 alias 单页)。因为 SWA 的面（69 KB）比压缩 KV 组合（20+4 KB）大，页宽实际由 SWA 决定。
2. **SWA 层按列切分**（`target_swa[slot_idx :: len(full)]`）：4 个元组各拿 10 个 SWA 层，且是**交错**的（元组 0 拿层 0/4/8/…/36）。这样后面做组转置时，每个调度侧 SWA 组都能「每个元组各拿一层」，四列共享同一套 block ID。

### 5.3 `group_cache_specs`：转置成调度组

`group_cache_specs`（[L199-L248](../cache_config.py#L199)）先按元组填 `page_size_padded`，再产出 4 类组：

```python
groups = [ MLA 组（4 个 long_kv + 4 个 index_k + 1 个 folded）,
           状态组（3 个 state）,
           *[每个 SWA 列一个组（每组 4 层，来自 4 个不同元组）],
            draft SWA 组（3 层） ]
```

其中 SWA 的转置是：

```python
swa_columns = [[name for name in layer_tuple if 是 SWA 且不是 mtp] for layer_tuple in layer_tuples]  # 4 列 × 10 层
for names in zip(*swa_columns):       # 组 0 = (层0,层1,层2,层3)，组 1 = (层4..7)，…
    groups.append(UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in names}))
```

> **为什么要转置**：混合组（hybrid）里每一组必须有**一致的页宽**（`get_storage_block_size` 会断言这一点）。转置后每组恰好由「4 个元组各 1 层」组成，页宽天然一致（都是 `max`），而 block ID 在各列之间保持同义——调度器分配一个 block ID，就能同时满足四列的同一位置。

### 5.4 `get_deepseek_v41_kv_cache_config`：一层一张 tensor

```python
page_sizes, layer_tuples = get_layer_tuples(...)
capacity   = max(available_memory // sum(page_sizes), 0)
num_blocks = may_override_num_blocks(vllm_config, capacity)
for page_size, layer_names in zip(page_sizes, layer_tuples):
    tensors.append(KVCacheTensor(size=num_blocks * page_size,
                                 layers=list(layer_names),
                                 offset=0, layer_stride=0, block_stride=page_size))
```

| 字段 | 值 | 含义 |
|------|-----|------|
| `size` | `num_blocks × page_size` | 该元组的整块显存 |
| `layers` | 元组内全部层名 | 同一次分配服务这些层 |
| `layer_stride` | 0 | 层间不额外步进 |
| `block_stride` | `page_size` | 一个逻辑块占一页 |

配合 `patch_kv_cache_utils.py` 的内存核算（`_ascend_max_memory_usage_bytes_from_groups`）：严格按照

```
Σ_over_groups( num_layer_tuples × group.max_memory_usage_pages() × layer_tuple_bytes )
```

来算准入内存，其中 `layer_tuple_bytes = Σ page_sizes`。这套「元组内共享、组间转置」的账本由 [patch/platform/patch_kv_cache_utils.py](../../../patch/platform/patch_kv_cache_utils.py) 与 [patch/platform/patch_kv_cache_coordinator.py](../../../patch/platform/patch_kv_cache_coordinator.py)（`_is_deepseek_v4_kv_cache_spec` 同时认 `deepseek_v4` 与 **`deepseek_v41`**）共同维护。

### 5.5 绑定：`DeepseekV41CacheLayer`

绑定走 `patch_bind_kv_cache.py`：`isinstance(layer, DeepseekV41CacheLayer)` 时调用 `layer.bind_kv_cache(kv_cache)`（[patch/worker/patch_bind_kv_cache.py](../../../patch/worker/patch_bind_kv_cache.py)）。`DeepseekV41CacheLayer` 本身是个极薄的 `nn.Module + AttentionLayerBase`（[dsa_v41.py:L1602-L1621](../../../attention/dsa_v41.py#L1602)），只负责：

- 把 Spec 交给调度器（`get_kv_cache_spec`）；
- 指定后端（`get_attn_backend` → `DeepseekV41CacheBackend`，L1568）；
- 存下绑定后的 `kv_cache`（**一个 list**，见下）。

> **`kv_cache` 是 list 而不是 tensor**：`AscendDeepseekV41SWACache` 的消费方（Engram 的 slot cache 适配器、`scatter_cache_sk`）都按 `kv_cache[0]` 取值，原因是 V4 系列沿用「每层 (k, v) 元组」的历史约定（见 [engram/hash_state.py:L31-L47](../engram/hash_state.py#L31) 的注释：`kv_cache is a one-element list and is absent until the KV allocator binds it`）。

### 5.6 块大小的三套表

`DSV4_BLOCK_SIZES` 由 `get_dsv4_block_sizes()` 按设备与 KV 布局选择（[models/layer/attention/layer.py:L35-L57](../../layer/attention/layer.py#L35)）：

| 表 | 条件 | 128 档取值 |
|----|------|-----------|
| `_DSV4_BLOCK_SIZES` | 非 A5 | `[128, 128, 8, 32]` |
| `_DSV4_COMPRESSED_BLOCK_SIZES` | A5 + 压缩布局 | `[128, 128, 8, 16]` |
| `_DSV4_BLOCK_SIZES_A5_BF16` | A5 + BF16 KV | `[128, 128, 8, 16]` |

V4.1 只用到其中两个位置：

| 位置 | 用于 | V4.1 是否使用 |
|------|------|--------------|
| `[0][1]` | SWA 的物理块大小（[model.py:L570](../model.py#L570)） | ✓ |
| `[0][2]` / `[0][3]` | V4 的 c4 / c128 状态块大小 | ✗（V4.1 用固定的 `STATE_RING_ROWS=32`） |

---

## 六、容量估算：1M 上下文到底占多少

按单请求、1M token、逻辑块 128、A5 打包布局估算：

| 缓存 | 份数 | 长度 | 每 token 字节 | 合计 |
|------|:---:|------|:---:|------|
| SWA（40 层） | 40 | 128（封顶） | 544 | 40 × 128 × 544 ≈ **2.8 MB** |
| c2 压缩 KV（层 2/8/14） | 3 | 500K | 320 | 3 × 500K × 320 ≈ **480 MB** |
| c1 压缩 KV（层 20） | 1 | 1M | 320 | 1M × 320 ≈ **320 MB** |
| 索引 K（4 层） | 4 | 500K/500K/500K/1M | 68 | (1.5M + 1M) × 68 ≈ **170 MB** |
| 折叠索引 K（层 20） | 1 | 1M/8 页 | 544/8 行 | 125K × 544 ≈ **68 MB** |
| 压缩状态（3 层） | 3 | 32 行/请求 | 4096 | 可忽略 |
| draft SWA（3 层） | 3 | 128 | 544 | 0.2 MB |
| **合计** | | | | **≈ 1.04 GB / 1M token** |

对比 V4-Pro-0813（非 A5，BF16）：30 个 c4 层各 250K × 1024 B ≈ 7.7 GB，31 个 c128 层各 8K × 1024 B ≈ 0.26 GB，SWA 61 层 × 128 × 1024 ≈ 8 MB → **≈ 8 GB / 1M token**。也就是说 V4.1 把 1M 上下文的 KV 从「8 GB 量级」压到「1 GB 量级」，这正是官方 README 标题 *Pushing the Limits of KV Cache Compression* 的含义。

> **注意**：上表是「单请求长上下文」的估算，实际部署还要乘并发数、加上页对齐（`page_size_padded`）与调度器预留。压缩 KV 的长度是 `seq_len // compress_ratio`，c2 的 500K 是 `1M / 2`。

---

## 七、与调度器 / 前缀缓存的交互

### 7.1 前缀缓存的两个开关

`AscendSlidingWindowMLASpec`（[core/kv_cache_interface.py:L380-L408](../../../core/kv_cache_interface.py#L380)）把「有界回放」直接写进了缓存语义：

```python
@property
def prefix_cacheable(self) -> bool:
    return not self.bounded_replay          # L391-L393

@property
def prefix_replay_tokens(self) -> int:
    return self.sliding_window if self.bounded_replay else 0   # L395-L397
```

即：一旦开了有界回放，SWA 组**不能**做普通前缀缓存命中（因为回放会把命中的最后 128 token 重算掉）；命中时改走 `prefix_replay_tokens`（128）的「重放窗口」语义。细节见 [9_deepseek_v41_decoder_replay.md](9_deepseek_v41_decoder_replay.md)。

### 7.2 三类组分别归谁管

| 组 | Spec 类型 | 由谁管 |
|----|----------|--------|
| MLA（long_kv + index_k + folded） | `AscendMLAAttentionSpec` | `FullAttentionManager` 类语义（按 token 计） |
| 压缩状态 | `CircularBufferSpec` | 环管理器（`is_circular_kv_cache_spec` 判真） |
| SWA（4 列 + draft） | `AscendSlidingWindowMLASpec` | 滑窗管理器（`storage_block_size = block_size`） |

Builder 在 [dsa_v41.py:L822-L836](../../../attention/dsa_v41.py#L822) 里用 `_cache_kind` 区分四种（`compressor_state` / `swa` / `index_k_folded` / `index_k` / `long_kv`），分别走不同的元数据构造分支。

### 7.3 逻辑块 vs 物理行：一处最常见的错

| 层 | 逻辑块（调度） | `tokens_per_state` | 物理行/块 | 换算 |
|----|:---:|:---:|:---:|------|
| SWA | 128 | 1 | 128 | 恒等 |
| c2 压缩 KV | 128 | 2 | 64 | `slot // 2` |
| c1 压缩 KV | 128 | 1 | 128 | 恒等 |
| 索引 K（c2） | 128 | 2 | 64 | `slot // 2` |
| 折叠索引 K | 128 | 8 | 16 | `slot // 8` |

写缓存的算子必须用**物理槽位**（`flat_slot_mapping` / `slot // ratio`），读 attention 时用**逻辑块表**（`block_table`）。两者在同一个 `AscendDSAV41Metadata` 里并排存放（`slot_mapping` / `flat_slot_mapping` / `compress_ratio` / `storage_block_size`，[dsa_v41.py:L1526-L1560](../../../attention/dsa_v41.py#L1526)），取错一个就会出现「数据写进去了但读不到」的诡异现象。

---

## 八、一次 decode 的写入清单

| 顺序 | 动作 | 缓存 | 代码 |
|:---:|------|------|------|
| 1 | Q/KV 投影 + 部分 RoPE | — | `_project_q_kv`（[dsa_v41.py:L383-L386](../../../attention/dsa_v41.py#L383)） |
| 2 | 写 SWA（`kind="win"`） | 本层 SWA | `_write_swa_cache`（[L347](../../../attention/dsa_v41.py#L347)） |
| 3 | （源层）写索引 K + 折叠 | 索引源层 | `indexer.update_keys`（[indexer.py:L116](../indexer.py#L116)） |
| 4 | （源层）写压缩 KV（`kind="cmp"`） | 源层 long_kv | `_write_compressed_source`（[L548](../../../attention/dsa_v41.py#L548)） |
| 5 | （c2 源）更新状态环 | 状态缓存 | `compressor_from_projected`（[compressor.py:L58](../compressor.py#L58)） |
| 6 | （候选源）产出候选块 | 共享缓冲 | `_select_sparse_indices`（[L660-L661](../../../attention/dsa_v41.py#L660)） |

> 第 3 步与第 4 步的先后是**正确性要求**，不是性能偏好（原因见 [3_deepseek_v41_attn.md](3_deepseek_v41_attn.md) §4.3）。

---

## 九、FAQ

**Q1：为什么压缩状态只有 3 份而不是 4 份？**
A：`ratio == 2` 才有状态环。层 20 是 c1，一个 token 就是一个 entry，不需要「暂存未满的组」。

**Q2：SWA 为什么不能让 40 层共享一份？**
A：SWA 是**原文**，每层的 `wkv` 投影不同，写进去的 latent 就不同。能共享的只有「压缩后的摘要」（因为摘要本身就是跨层共享的历史视角，这是训练时定下的拓扑）和「索引 K」（它是摘要的投影）。

**Q3：`page_size_padded` 的 `max` 策略会不会浪费？**
A：会有一点——c2 元组的页宽被 SWA 的 69 KB 撑大，而压缩 KV 只用 20 KB。但换来的是「一个 block ID 通吃四列」，混合组的前缀缓存与 DP 协调都简单得多。真要抠，就得给每类缓存单独成组，代价是调度器的 `find_longest_cache_hit` 复杂度。

**Q4：`DSV4_BLOCK_SIZES[cache_config.block_size][0][1]` 里那个「1」是什么？**
A：`[0]` 是 4 元组 `[storage_block, swa_block, c4_state_block, c128_state_block]`。V4.1 的 SWA 用第二个（`swa_block=128`），压缩状态改用固定 32 行，所以后两个位置在 V4.1 不被读取。

**Q5：A5 的 544 B/行和「折叠页 544 B/8 行」是同一个 544 吗？**
A：**不是巧合，也不是同一个意思**。SWA 的 544 = 512 FP8 + 32 B scale；折叠页的 544 = 8 行 × (64 B FP4 + 4 B scale)。两者数值相同纯属设计上的对齐便利（一页正好能塞一组），不要互相推导。

**Q6：能不能只开 3 个 KV 源、把层 20 也并到层 14？**
A：不能。`compress_ratios` 与 `kv_source_layer_ids` 是训练超参，权重与压缩位坐标强绑定；改拓扑等于换模型。

---

*相关文档：[3_deepseek_v41_attn.md](3_deepseek_v41_attn.md)（注意力如何使用这些缓存）、[9_deepseek_v41_decoder_replay.md](9_deepseek_v41_decoder_replay.md)（回放如何依赖源层缓存）、[0_deepseek_v41_arch.md](0_deepseek_v41_arch.md)（层拓扑总表）。*
