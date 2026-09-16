# DeepSeek V4 KV 缓存体系与 Compressor/Indexer 详解

> **本文档详细讲解 DeepSeek V4 的三级 KV 缓存体系，包括 Compressor（KV 压缩）、Indexer（稀疏索引）和 SWA（滑动窗口）的完整实现与交互。**

---

## 一、三级 KV 缓存总览

### 1.1 设计理念

DeepSeek V4 采用 **DSA (Deep Sparse Attention)** 架构，通过三级 KV 缓存在超长上下文下平衡精度与显存：

```
原始 KV (O(N) 显存，N=上下文长度)
    │
    ├── SWA (滑动窗口): 最近 window_size 个 token 的原始 KV
    │                    → 高精度，近期记忆
    │
    ├── Compressor (KV压缩): 历史 KV 按 compress_ratio 压缩存储
    │                       → 减少显存，长程记忆
    │
    └── Indexer (稀疏索引): compress_ratio=4 时，TopK 选择最重要的 token
                            → 进一步减少计算量
```

### 1.2 按层压缩配置

通过 `get_dsv4_compress_ratio(config, layer_idx)` 按层获取压缩比：

| compress_ratio | Compressor | Indexer | SWA | 适用场景 |
|---------------|-----------|---------|-----|---------|
| **1** | ✗ | ✗ | ✓ | 无压缩层，仅滑动窗口 |
| **4** | ✓ (overlap=True) | ✓ (TopK 选择) | ✓ | 中距离层，稀疏索引+压缩 |
| **128** | ✓ (overlap=False) | ✗ | ✓ | 长距离层，深度压缩 |

> 层索引从 0 开始，不同层有不同的压缩策略，形成分层的上下文表示。

### 1.3 模块关系图

```
DeepseekV4Attention
    │
    ├── swa_cache_layer: AscendDeepseekV4SWACache      [所有层]
    │
    ├── compressor: Compressor                           [compress_ratio > 1]
    │   ├── wkv: ReplicatedLinear                        — KV 投影
    │   ├── wgate: ReplicatedLinear                      — 门控投影
    │   ├── norm: RMSNorm                                — 归一化
    │   ├── ape: Parameter                               — 绝对位置编码
    │   └── state_cache: AscendCompressorStateCache     — 压缩状态缓存
    │
    └── indexer: Indexer                                 [compress_ratio == 4]
        ├── wq_b: ReplicatedLinear                       — Query 投影
        ├── weights_proj: ReplicatedLinear               — 权重投影
        ├── k_cache: AscendDeepseekV4IndexerCache       — Indexer KV 缓存
        └── compressor: Compressor                       — 共享 Compressor (可选)
```

---

## 二、Compressor — KV 压缩器

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L596`

### 2.1 作用与原理

Compressor 将历史 KV 按 `compress_ratio` 倍压缩存储，用状态表示法替代原始 KV，大幅减少显存占用。

**核心思想**：将连续的 `compress_ratio` 个 token 的 KV 压缩为一个状态向量（kv_state + score_state），保留关键信息的同时压缩存储。

### 2.2 核心成员变量

| 成员 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `dim` | int | — | 输入 hidden_size |
| `head_dim` | int | — | 头部维度 |
| `rope_head_dim` | int | — | RoPE 部分维度 |
| `nope_head_dim` | int | — | 非 RoPE 部分维度 |
| `compress_ratio` | int | — | 压缩比 (4 或 128) |
| `overlap` | bool | — | 是否重叠压缩 (compress_ratio==4 时为 True) |
| `coff` | int | — | `1 + overlap`，c4=2, c128=1 |
| `norm_eps` | float | — | RMSNorm epsilon |
| `ape` | Parameter | `(compress_ratio, coff * head_dim)` float32 | 绝对位置编码 |
| `wkv` | ReplicatedLinear | `dim → coff * head_dim` | KV 投影 |
| `wgate` | ReplicatedLinear | `dim → coff * head_dim` | 门控投影 |
| `norm` | RMSNorm | `head_dim` | 归一化 (A5 设备用 float32) |
| `state_cache` | AscendCompressorStateCache | — | 压缩状态缓存 |

### 2.3 overlap 重叠机制

当 `compress_ratio == 4` 时启用重叠压缩：

- `coff = 1 + overlap = 2`
- 状态维度 `state_dim = 2 * coff * head_dim = 4 * head_dim`
- 相邻压缩块之间有重叠，保证信息连续性

当 `compress_ratio == 128` 时无重叠：

- `coff = 1`
- 状态维度 `state_dim = 2 * head_dim`
- 非重叠压缩，更高压缩率

**overlap_transform 方法** (`L666`)：实现重叠变换，将当前块的后半部分与下一块的前半部分重叠：

```python
def overlap_transform(self, tensor, value=0):
    # tensor: (b, s, ratio, 2*head_dim)
    # new_tensor: (b, s, 2*ratio, head_dim)
    # 前 ratio 个 = 上一块的前 head_dim（来自前一位置）
    # 后 ratio 个 = 当前块的后 head_dim
```

### 2.4 rope_single 方法

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L683`

对 Compressor 的 KV 应用 RoPE 旋转位置编码：

```python
def rope_single(self, x, cos, sin, inverse=False):
    # 调用 torch_npu.npu_rotary_mul
    # 支持 TND (3维) 和 BTND (4维) 布局
    # rotary_mode="interleave" 交错模式
    # 内部转 float32 计算，输出转回原 dtype
```

### 2.5 AscendCompressorStateCache

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L108`

Compressor 的状态缓存类，继承自 vLLM 的 `CompressorStateCache`。

| 属性 | 类型 | 说明 |
|------|------|------|
| `state_dim` | int | 状态维度 = `2 * coff * head_dim` (c4) 或 `2 * head_dim` (c128) |
| `dtype` | torch.dtype | 固定 float32 |
| `compress_ratio` | int | 压缩比 |
| `block_size` | int | Block size (从 DSV4_BLOCK_SIZES 获取) |

**get_kv_cache_spec** 返回 `AscendSlidingWindowMLASpec`，关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_kv_heads` | 1 | MLA 单 KV 头 |
| `head_size` | state_dim | 状态维度 |
| `dtype` | float32 | 状态用 float32 存储 |
| `sliding_window` | self.sliding_window | 滑动窗口大小 |
| `page_size_padded` | 根据 state_dim 选择 pads[0] 或 pads[1] | 填充后的页大小 |

---

## 三、Indexer — 稀疏索引器

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L529`

### 3.1 作用与原理

Indexer 在 `compress_ratio == 4` 的层上工作，对压缩后的 KV 做 TopK 稀疏选择，只保留与当前 Query 最相关的 token，进一步减少注意力计算量。

**核心思想**：用一个轻量级的注意力计算（低维 head）筛选最重要的历史 token，只将这些 token 送入主注意力计算。

### 3.2 核心成员变量

| 成员 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `n_heads` | int | — | Indexer 注意力头数 (`config.index_n_heads`) |
| `head_dim` | int | — | Indexer 头部维度 (`config.index_head_dim`) |
| `rope_head_dim` | int | — | RoPE 维度 (`config.qk_rope_head_dim`) |
| `index_topk` | int | — | TopK 选择数量 (`config.index_topk`) |
| `q_lora_rank` | int | — | Query 低秩维度 |
| `softmax_scale` | float | — | `head_dim ** -0.5` |
| `compress_ratio` | int | — | 压缩比 (固定 4) |
| `wq_b` | ReplicatedLinear | `q_lora_rank → n_heads * head_dim` | Query 投影 |
| `weights_proj` | ReplicatedLinear | `hidden_size → n_heads` | 权重投影 (无量化) |
| `k_cache` | AscendDeepseekV4IndexerCache | — | Indexer KV 缓存 |
| `compressor` | Compressor \| None | — | Indexer 自己的 Compressor |

> **注意**: Indexer 的 compressor 与 Attention 的 compressor 是两个独立实例，各有自己的 wkv/wgate/norm/state_cache。

### 3.3 AscendDeepseekV4IndexerCache

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L141`

Indexer 的 KV 缓存类，继承自 vLLM 的 `DeepseekV4IndexerCache`。

| 设备类型 | dtype | 说明 |
|---------|-------|------|
| A5 | float8_e4m3fn | FP8 存储，节省显存 |
| 非 A5 | int8 | INT8 存储 |

**get_kv_cache_spec** 返回 `AscendMLAAttentionSpec`，关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `block_size` | DSV4_BLOCK_SIZES[block_size][0][0] | 与 mla block_size 相同 |
| `num_kv_heads` | 1 | MLA 单 KV 头 |
| `head_size` | head_dim | 头部维度 |
| `model_version` | "deepseek_v4" | 模型版本 |
| `compress_ratio` | 4 | 固定 4 |
| `scale_dim` | 1 if head_dim==128 else 0 | 缩放维度 |
| `scale_dtype` | float (A5) / float16 (非A5) | 缩放因子 dtype |

---

## 四、SWA 滑动窗口缓存

### 4.1 AscendDeepseekV4SWACache

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L177`

滑动窗口 KV 缓存，存储最近 `window_size` 个 token 的完整（未压缩）KV。

| 属性 | 类型 | 说明 |
|------|------|------|
| `head_dim` | int | 头部维度 |
| `window_size` | int | 滑动窗口大小 (`config.sliding_window`) |
| `dtype` | torch.dtype | KV 存储 dtype |
| `block_size` | int | Block size |

**设备类型适配**：

| 特性 | A5 设备 | 非 A5 设备 |
|------|--------|-----------|
| dtype | float8_e4m3fn | bfloat16 |
| cached_head_size | head_dim + 128 | head_dim |

**get_kv_cache_spec** 返回 `AscendSlidingWindowMLASpec`，关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `block_size` | DSV4_BLOCK_SIZES[block_size][0][1] | swa block_size |
| `num_kv_heads` | 1 | MLA 单 KV 头 |
| `head_size` | cached_head_size | A5 额外 +128 |
| `dtype` | 按设备 | float8_e4m3fn / bfloat16 |
| `sliding_window` | window_size | 滑动窗口大小 |
| `model_version` | "deepseek_v4" | 模型版本 |

---

## 五、Block Size 配置

**定义位置**: `vllm_ascend/models/layer/attention/layer.py:L32`

### 5.1 非 A5 设备

```python
_DSV4_BLOCK_SIZES = {
    128: [[128, 128, 8, 32], [16640, 131072]],
    64:  [[64,  64,  4, 16], [8320,  65536]],
    32:  [[32,  32,  2, 8],  [4160,  32768]],
}
```

### 5.2 A5 设备

```python
_DSV4_BLOCK_SIZES_A5 = {
    128: [[128, 128, 8, 16], [16896, 81920]],
    64:  [[64,  64,  4, 8],  [8448,  40960]],
    32:  [[32,  32,  2, 4],  [4224,  20480]],
}
```

### 5.3 索引对应关系

`DSV4_BLOCK_SIZES[global_block_size][0]` 的四个元素：

| 索引 | 名称 | 说明 |
|------|------|------|
| `[0][0]` | mla | MLA KV 缓存 block_size |
| `[0][1]` | swa | SWA 缓存 block_size |
| `[0][2]` | c4 state | Compressor c4 状态 block_size |
| `[0][3]` | c128 state | Compressor c128 状态 block_size |

`DSV4_BLOCK_SIZES[global_block_size][1]` 的两个元素：

| 索引 | 名称 | 说明 |
|------|------|------|
| `[1][0]` | page_size_padded_t1 | 类型1填充页大小 (c4 state) |
| `[1][1]` | page_size_padded_t2 | 类型2填充页大小 (c128 state) |

> **A5 差异**: A5 设备的 c128_state block_size 是非 A5 的一半（16 vs 32, 8 vs 16, 4 vs 8），页大小也不同。

---

## 六、IndexCache — 索引缓存复用

### 6.1 设计理念

V3.2+ 特性，部分层复用前一层的 TopK 索引以减少 Indexer 的计算量。

**触发条件**：
- `compress_ratio == 4`（有 Indexer 的层）
- `config.use_index_cache == True`
- 非 MTP 层（`".mtp." not in prefix`）

### 6.2 两种模式

#### 频率模式 (`index_topk_freq`)

每 `freq` 个 c4 层计算一次 TopK 索引，其余层复用：

```python
indexer_seq_idx = sum(1 for r in compress_ratios[:layer_idx] if r == 4)
skip_topk = max(indexer_seq_idx - 1, 0) % freq != 0
```

- `indexer_seq_idx`: 当前是第几个有 indexer 的层（从 0 开始）
- 第 0 层一定计算，之后每 freq 层计算一次

#### 模式匹配模式 (`index_topk_pattern`)

用字符串模式指定每层是否计算：

```python
pattern = "FSFS"  # F=计算, S=跳过
skip_topk = pattern[indexer_seq_idx] == "S"
```

- 第一个字符必须是 `'F'`
- 按 `indexer_seq_idx` 索引模式字符串

### 6.3 topk_indices_buffer

模型级共享的 TopK 索引缓冲区：

- **形状**: `(max_num_batched_tokens, topk_tokens)` int32
- **写入**: `skip_topk=False` 的层计算索引并写入 buffer
- **读取**: `skip_topk=True` 的层从 buffer 复用索引
- **共享**: 主模型与 MTP 之间共享（模型级 buffer）

---

## 七、KV 缓存 Spec 体系

### 7.1 两类 KVCacheSpec

| Spec 类型 | 使用场景 | 对应缓存 |
|-----------|---------|---------|
| `AscendSlidingWindowMLASpec` | 滑动窗口类 | SWA Cache, Compressor State Cache |
| `AscendMLAAttentionSpec` | MLA 注意力类 | Indexer KV Cache, MLA KV Cache |

### 7.2 三层缓存的 spec 获取

| 缓存 | get_kv_cache_spec 方法位置 | 返回类型 |
|------|--------------------------|---------|
| SWA Cache | `AscendDeepseekV4SWACache.get_kv_cache_spec` | AscendSlidingWindowMLASpec |
| Compressor State | `AscendCompressorStateCache.get_kv_cache_spec` | AscendSlidingWindowMLASpec |
| Indexer KV | `AscendDeepseekV4IndexerCache.get_kv_cache_spec` | AscendMLAAttentionSpec |

---

## 八、A5 设备特殊适配

### 8.1 汇总对比

| 特性 | A5 设备 | 非 A5 设备 |
|------|--------|-----------|
| SWA KV dtype | float8_e4m3fn | bfloat16 |
| SWA cached_head_size | head_dim + 128 | head_dim |
| Indexer KV dtype | float8_e4m3fn | int8 |
| Indexer scale_dtype | float32 | float16 |
| Compressor norm dtype | float32 | 默认 |
| c128_state block_size | 16/8/4 | 32/16/8 |
| page_size_padded | 不同配置 | 标准配置 |

### 8.2 设计原因

A5 设备支持 FP8 高精度存储，用 float8_e4m3fn 替代 int8/bfloat16，在节省显存的同时保持更高精度。额外的 128 维 cached_head_size 用于存储 FP8 的缩放因子等元数据。
