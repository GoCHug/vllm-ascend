# DeepSeek V4 Attention 架构详解

## 一、概述

DeepSeek V4 采用 **MLA (Multi-head Latent Attention)** + **DSA (Deep Sparse Attention)** 架构，通过多层压缩和稀疏选择实现超长上下文的高效推理。在 Ascend NPU 上，整个注意力计算完全卸载到 NPU 后端执行，Python 层仅做模块封装和参数传递。

### 1.1 核心设计思想

```
原始 KV (O(N²) 显存)
    │
    ├── SWA (滑动窗口): 保留最近 window_size 个 token 的完整 KV
    │                    → 高精度，近期记忆
    │
    ├── Compressor (KV压缩): 将历史 KV 按 4x / 128x 压缩存储
    │                       → 减少显存，长程记忆
    │
    └── Indexer (稀疏索引): compress_ratio=4 时，TopK 选择最重要的 token
                            → 进一步减少计算量
```

**三大 KV 缓存体系协同工作**：
- **SWA Cache**: 滑动窗口内的原始精度 KV（近期 token）
- **Compress Cache**: 压缩后的历史 KV（长程记忆，4x 或 128x 压缩）
- **Indexer Cache**: Indexer 专用的压缩 KV（仅 c4 模式，用于 TopK 选择）

### 1.2 整体架构分层

```
┌─────────────────────────────────────────────────────────────────┐
│  第1层: DeepseekV4Attention (Python 封装层)                      │
│  文件: vllm_ascend/models/deepseek_v4.py:L709                    │
│  - 初始化所有权重模块 (wq_a, wq_b, wkv, wo_a, wo_b, ...)        │
│  - 组装 DSAModules 数据类（传递模块引用）                         │
│  - 创建 AscendDeepseekSparseAttention (dsa_attn)                │
│  - forward 直接透传给 dsa_attn                                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  第2层: AscendDeepseekSparseAttention (外层封装)                  │
│  文件: vllm_ascend/ops/dsa.py:L61                                │
│  - 继承 MultiHeadLatentAttentionWrapper                         │
│  - 持有 DSAAttention 实例                                        │
│  - 提供 forward 入口，统一封装 prefill/decode 分发                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  第3层: DSAAttention (注意力层基类)                                │
│  文件: vllm_ascend/models/layer/attention/layer.py               │
│  - 管理 KV cache spec                                            │
│  - 持有 AttentionBackend 和 AttentionImpl 实例                    │
│  - 统一 forward 接口                                             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  第4层: AscendDSABackend (NPU 后端注册)                           │
│  文件: vllm_ascend/attention/dsa_v1.py:L184                      │
│  - 实现 AttentionBackend 接口                                     │
│  - 提供 builder、impl、KV cache shape 等静态方法                  │
│  - 注册到 vLLM 注意力后端系统                                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  第5层: AscendDSAImpl (核心计算实现)                               │
│  文件: vllm_ascend/attention/dsa_v1.py:L1353                     │
│  - _forward_prefill(): prefill 阶段完整计算逻辑                   │
│  - _forward_decode(): decode 阶段完整计算逻辑                     │
│  - MLA Prolog (Query/KV 投影 + RoPE)                             │
│  - Compressor / Indexer 处理                                     │
│  - 稀疏注意力算子调用 (npu_sparse_flash_attention)               │
│  - 输出投影 (wo_a + wo_b)                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 文件索引

| 层级 | 文件路径 | 核心职责 |
|------|---------|---------|
| Python 封装层 | `vllm_ascend/models/deepseek_v4.py:L709-908` | `DeepseekV4Attention` 类，初始化所有子模块 |
| 外层封装 | `vllm_ascend/ops/dsa.py:L61` | `AscendDeepseekSparseAttention`，对外接口 |
| 层基类 | `vllm_ascend/models/layer/attention/layer.py` | `DSAAttention`，KV cache 管理 |
| 后端注册 | `vllm_ascend/attention/dsa_v1.py:L184` | `AscendDSABackend`，后端注册 |
| 核心实现 | `vllm_ascend/attention/dsa_v1.py:L1353` | `AscendDSAImpl`，prefill/decode 计算 |
| Compressor | `vllm_ascend/models/deepseek_v4.py:L596` | KV 压缩器 |
| Indexer | `vllm_ascend/models/deepseek_v4.py:L529` | 稀疏索引选择器 |

***

## 二、DeepseekV4Attention — Python 封装层

### 2.1 类定义与初始化

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L709-908`

```python
class DeepseekV4Attention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        max_position_embeddings: int = 0,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
```

### 2.2 基础配置参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `layer_idx` | int | 当前层索引，从 prefix 中解析 |
| `dim` | int | `config.hidden_size`，隐藏层维度 |
| `n_heads` | int | `config.num_attention_heads`，总注意力头数 |
| `n_local_heads` | int | `n_heads // tp_size`，本地头数（张量并行后） |
| `q_lora_rank` | int | Query 低秩分解秩 |
| `o_lora_rank` | int | 输出低秩分解秩 |
| `head_dim` | int | 每个头的维度 |
| `rope_head_dim` | int | `config.qk_rope_head_dim`，RoPE 作用的维度 |
| `nope_head_dim` | int | `head_dim - rope_head_dim`，无 RoPE 的维度 |
| `n_groups` | int | `config.o_groups`，输出分组数 |
| `n_local_groups` | int | `n_groups // tp_size`，本地分组数 |
| `window_size` | int | 滑动窗口大小 |
| `scale` | float | `head_dim ** -0.5`，注意力缩放因子 |
| `enable_dsa_cp` | bool | 是否启用 DSA 上下文并行 |

### 2.3 Query 路径组件 (低秩分解 + MLA)

Query 采用低秩分解结构：`hidden → wq_a → q_norm → wq_b → Q`

| 组件 | 类型 | 权重 Shape | 输入 Shape | 输出 Shape | 说明 |
|------|------|-----------|-----------|-----------|------|
| `wq_a` | ReplicatedLinear | `(dim, q_lora_rank)` | `(num_tokens, dim)` | `(num_tokens, q_lora_rank)` | Query 低秩投影 A，数据并行复制 |
| `q_norm` | RMSNorm | `(q_lora_rank,)` | `(num_tokens, q_lora_rank)` | `(num_tokens, q_lora_rank)` | Query 归一化 |
| `wq_b` | ColumnParallelLinear / ReplicatedLinear | `(q_lora_rank, n_heads*head_dim)` | `(num_tokens, q_lora_rank)` | `(num_tokens, n_local_heads*head_dim)` | Query 低秩投影 B，列并行；`enable_dsa_cp` 时用 ReplicatedLinear |
| `q_norm_without_weight` | RMSNorm | 无权重 | `(num_tokens, n_local_heads, head_dim)` | `(num_tokens, n_local_heads, head_dim)` | Q 投影后的无权重 RMSNorm |

### 2.4 KV 路径组件 (MLA 联合投影)

KV 采用 MLA 架构，只用一个线性投影生成压缩的 KV 表示：

| 组件 | 类型 | 权重 Shape | 输入 Shape | 输出 Shape | 说明 |
|------|------|-----------|-----------|-----------|------|
| `wkv` | ReplicatedLinear | `(dim, head_dim)` | `(num_tokens, dim)` | `(num_tokens, head_dim)` | KV 联合投影 (MLA 压缩表示) |
| `kv_norm` | RMSNorm | `(head_dim,)` | `(num_tokens, head_dim)` | `(num_tokens, head_dim)` | KV 归一化 |

> **MLA 的核心思想**：Key 和 Value 共享同一个低维 latent 表示（head_dim 维），而不是分别存储完整的 K 和 V。这样 KV cache 的显存占用从 `2 * head_dim` 降到 `head_dim`。

### 2.5 输出投影组件 (低秩分解 + 分组)

输出同样采用低秩分解，并引入分组结构：

| 组件 | 类型 | 权重 Shape | 输入 Shape | 输出 Shape | 说明 |
|------|------|-----------|-----------|-----------|------|
| `wo_a` | ColumnParallelLinear | `(n_heads*head_dim // n_groups, n_groups*o_lora_rank)` | `(num_tokens, n_local_groups, n_heads*head_dim//n_groups)` | `(num_tokens, n_local_groups, o_lora_rank)` | 输出投影 A (组内投影)，列并行 |
| `wo_b` | RowParallelLinear | `(n_groups*o_lora_rank, dim)` | `(num_tokens, n_groups*o_lora_rank)` | `(num_tokens, dim)` | 输出投影 B (最终输出)，行并行 |

### 2.6 位置编码与注意力参数

| 组件 | 类型 | Shape / 说明 |
|------|------|-------------|
| `rotary_emb` | ComplexExpRotaryEmbedding | 旋转位置编码，根据 `compress_ratio` 选择 rope_theta；支持多组 rope（"default" 和 "c{compress_ratio}"） |
| `attn_sink` | nn.Parameter | `(n_heads,)` 或 `(n_local_heads,)`，注意力 sink 参数 |
| `scale` | float | `head_dim ** -0.5`，注意力缩放因子 |

### 2.7 压缩与索引组件

| 组件 | 类型 | 存在条件 | 说明 |
|------|------|---------|------|
| `compressor` | Compressor \| None | `compress_ratio > 1` (4 或 128) | KV 压缩器，将历史 KV 压缩存储以节省显存 |
| `indexer` | Indexer \| None | `compress_ratio == 4` | 稀疏索引选择器，TopK 选择最重要的 token |

**compress_ratio 按层配置**：
- 通过 `get_dsv4_compress_ratio(config, layer_idx)` 按层获取压缩比例
- 可选值：1 (无压缩，仅 SWA)、4 (稀疏索引 + 压缩)、128 (深度压缩)

### 2.8 IndexCache (索引缓存复用)

V3.2+ 特性，部分层复用前一层的 TopK 索引，减少计算量。

**触发条件**：
- `compress_ratio == 4`（必须有 Indexer）
- `config.use_index_cache == True`
- 非 MTP 层（`.mtp.` 不在 prefix 中）

**两种模式**：

| 模式 | 配置项 | 说明 |
|------|--------|------|
| 频率模式 | `index_topk_freq` | 每 freq 层计算一次索引，其余层跳过 |
| 模式匹配 | `index_topk_pattern` | 如 "FSFS"，F=计算, S=跳过；必须以 'F' 开头 |

**skip_topk 标志**：
- `True`: 本层跳过 TopK 计算，复用 `topk_indices_buffer` 中的索引
- `False`: 本层计算 TopK 索引，并写入 `topk_indices_buffer`

### 2.9 SWA 缓存与 DSA 组装

```python
swa_cache_layer = AscendDeepseekV4SWACache(
    head_dim=self.head_dim,
    window_size=self.window_size,
    dtype=k_dtype,          # A5: float8_e4m3fn, 非A5: bfloat16
    prefix=f"{prefix}.swa_cache",
    cache_config=cache_config,
)

dsa_modules = DSAModules(
    wq_a=self.wq_a, q_norm=self.q_norm,
    q_norm_without_weight=self.q_norm_without_weight,
    wq_b=self.wq_b, wkv=self.wkv, kv_norm=self.kv_norm,
    wo_a=self.wo_a, wo_b=self.wo_b,
    attn_sink=self.attn_sink,
    indexer=self.indexer, compressor=self.compressor,
    swa_cache_layer=swa_cache_layer,
    topk_indices_buffer=topk_indices_buffer,
    skip_topk=skip_topk,
)

self.dsa_attn = AscendDeepseekSparseAttention(
    dim=self.dim, n_heads=self.n_heads, scale=self.scale,
    n_local_heads=self.n_local_heads,
    q_lora_rank=self.q_lora_rank, o_lora_rank=self.o_lora_rank,
    head_dim=self.head_dim, rope_head_dim=self.rope_head_dim,
    nope_head_dim=self.nope_head_dim, eps=self.eps,
    n_groups=self.n_groups, n_local_groups=self.n_local_groups,
    window_size=self.window_size, compress_ratio=self.compress_ratio,
    dsa_modules=dsa_modules,
    cache_config=cache_config, quant_config=quant_config,
    prefix=f"{prefix}",
)
```

### 2.10 forward 方法

```python
def forward(
    self,
    positions: torch.Tensor,           # 位置编码
    hidden_states: torch.Tensor,       # 输入: (num_tokens, hidden_size)
    llama_4_scaling: torch.Tensor | None,  # 缩放因子
) -> torch.Tensor:
    return self.dsa_attn(positions, hidden_states, llama_4_scaling)
```

> **关键点**：DeepseekV4Attention.forward 本身只是**透传**调用 `dsa_attn`，所有投影、位置编码、缓存管理、稀疏注意力计算都在 NPU 后端 `AscendDSAImpl` 中完成。

***

## 三、AscendDeepseekSparseAttention — 外层封装

**文件位置**: `vllm_ascend/ops/dsa.py:L61`

继承自 `MultiHeadLatentAttentionWrapper`，是 DSA 注意力对外的统一入口。

### 3.1 DSAModules 数据类

```python
@dataclass
class DSAModules:
    wq_a: torch.nn.Module
    q_norm: torch.nn.Module
    q_norm_without_weight: torch.nn.Module
    wq_b: torch.nn.Module
    wkv: torch.nn.Module
    kv_norm: torch.nn.Module
    wo_a: torch.nn.Module
    wo_b: torch.nn.Module
    attn_sink: torch.nn.Module
    indexer: torch.nn.Module | None
    compressor: torch.nn.Module | None
    swa_cache_layer: torch.nn.Module
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None
    skip_topk: bool = False
```

### 3.2 核心成员变量

从 `dsa_modules` 中提取的权重和模块：
- `wq_a`, `wq_b`, `wkv`: Query / KV 投影权重
- `q_norm`, `q_norm_without_weight`, `kv_norm`: 归一化层
- `wo_a`, `wo_b`: 输出投影
- `attn_sink`: 注意力 sink 参数
- `indexer`, `compressor`: 压缩与索引模块
- `swa_cache_layer`: SWA KV 缓存

### 3.3 CV Wrapper (量化 + 矩阵乘分离)

```python
self.cv_wq_a = CVLinearWrapper(self.wq_a)
self.cv_wkv = CVLinearWrapper(self.wkv)
self.cv_wq_b = CVLinearWrapper(self.wq_b)
```

`CVLinearWrapper` 将线性层拆分为两部分：
- **Vector 阶段**: 量化计算（生成缩放因子等）
- **Cube 阶段**: 矩阵乘法（实际的 matmul）

这样可以在多流并行时，把量化操作和其他计算重叠执行。

***

## 四、AscendDSAImpl — 核心计算实现

**文件位置**: `vllm_ascend/attention/dsa_v1.py:L1353`

这是 DSA 注意力的核心计算类，包含 prefill 和 decode 的完整逻辑。

### 4.1 核心成员变量

```python
# 基础配置
self.num_heads = n_heads
self.n_local_heads = n_local_heads
self.scale = scale
self.head_dim = head_dim
self.rope_head_dim = rope_head_dim
self.nope_head_dim = nope_head_dim
self.softmax_scale = head_dim ** -0.5

# 低秩与分组
self.q_lora_rank = q_lora_rank
self.o_lora_rank = o_lora_rank
self.n_group = n_groups
self.n_local_groups = n_local_groups

# 稀疏注意力
self.window_size = window_size
self.compress_ratio = compress_ratio

# MLA 投影
self.wq_a, self.wq_b, self.wkv
self.q_norm, self.q_norm_without_weight, self.kv_norm

# CV Wrapper
self.cv_wq_a, self.cv_wkv, self.cv_wq_b

# 输出投影
self.wo_a, self.wo_b

# 压缩与索引
self.compressor, self.indexer
self.skip_topk, self.topk_indices_buffer

# 多流
self.multistream_dsv4_dsa_overlap
```

### 4.2 Indexer 相关参数 (compress_ratio == 4 时)

```python
self.indexer_heads          # indexer 头数 (config.index_n_heads)
self.inderxer_dim           # indexer head 维度 (config.index_head_dim)
self.inderxer_wq_b          # indexer Query 投影
self.cv_inderxer_wq_b       # indexer Query CV Wrapper
self.weights_proj           # 权重投影
self.indexer_softmax_scale  # indexer 缩放因子
self.indexer_compress       # indexer 内部的 compressor
self.index_topk             # TopK 数量 (config.index_topk)
```

### 4.3 Compressor 相关参数 (compress_ratio > 1 时)

```python
self.compressor_head_dim    # compressor head 维度
self.compressor_overlap     # 是否重叠 (c4: True, c128: False)
self.compressor_rotate      # 是否应用 RoPE
self.compressor_ape         # APE 参数
self.compressor_wkv         # KV 投影
self.compressor_wgate       # 门控投影
self.compressor_norm        # 归一化
```

***

## 五、完整 Forward 数据流

### 5.1 总览流程图

```
                        hidden_states: (num_tokens, dim)
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  阶段 1: MLA Prolog (Query + KV 投影 + RoPE)                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Query 路径                                                          │   │
│  │  hidden → wq_a → q_norm → wq_b → reshape → q_rms → partial_rope     │   │
│  │  (num_tokens, dim)              (num_tokens, n_local_heads, head_dim) │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  KV 路径                                                             │   │
│  │  hidden → wkv → kv_norm → reshape → partial_rope → SWA cache 写入   │   │
│  │  (num_tokens, dim)        (num_tokens, 1, head_dim)                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  > 优化: 多流并行 (Query 流 + KV 流)，通过事件同步                           │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  阶段 2: Compressor (KV 压缩) — compress_ratio > 1 时执行                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  hidden → wkv/wgate → 状态累积 → norm + RoPE → 压缩 KV cache 写入    │   │
│  │  压缩比: 4x (overlap=True, coff=2) 或 128x (overlap=False, coff=1)   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  阶段 3: Indexer (TopK 稀疏选择) — compress_ratio == 4 时执行                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  qr → indexer.wq_b → 与压缩 KV 计算得分 → TopK 选择 → topk_indices    │   │
│  │  skip_topk=True 时: 直接从 topk_indices_buffer 复用                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  阶段 4: 稀疏注意力计算                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  NPU 算子: npu_sparse_flash_attention                               │   │
│  │  输入: Q + SWA KV + 压缩 KV + TopK 索引 + attn_sink                 │   │
│  │  输出: attn_output (num_tokens, n_local_heads, head_dim)            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  阶段 5: 输出投影                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  attn_out → nope_rope逆向 → reshape分组 → wo_a → flatten → wo_b      │   │
│  │  (num_tokens, n_local_heads, head_dim)     (num_tokens, dim)        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │
                                    ▼
                        输出: (num_tokens, dim)
```

### 5.2 阶段 1: MLA Prolog — Query/KV 投影

#### Query 路径详细步骤

| 步骤 | 操作 | 输入 Shape | 输出 Shape | 说明 |
|------|------|-----------|-----------|------|
| 1 | `cv_wq_a.vector(hidden)` | `(T, dim)` | - | 量化阶段（Vector） |
| 2 | `cv_wq_a.cube(hidden)` | `(T, dim)` | `(T, q_lora_rank)` | 矩阵乘阶段（Cube） |
| 3 | `q_norm(q_a)` | `(T, q_lora_rank)` | `(T, q_lora_rank)` | RMSNorm 归一化 → qr |
| 4 | `cv_wq_b.vector(qr)` | - | - | 量化阶段 |
| 5 | `cv_wq_b.cube(qr)` | `(T, q_lora_rank)` | `(T, n_local_heads * head_dim)` | 低秩投影 B |
| 6 | `unflatten` | - | `(T, n_local_heads, head_dim)` | 重塑为多头 |
| 7 | `q_rms` (q_norm_without_weight) | `(T, n_local_heads, head_dim)` | `(T, n_local_heads, head_dim)` | 无权重 RMSNorm |
| 8 | `partial_rotary_mul` | - | `(T, n_local_heads, head_dim)` | 部分 RoPE（仅 rope_head_dim 部分） |

#### KV 路径详细步骤

| 步骤 | 操作 | 输入 Shape | 输出 Shape | 说明 |
|------|------|-----------|-----------|------|
| 1 | `cv_wkv.vector(hidden)` | `(T, dim)` | - | 量化阶段 |
| 2 | `cv_wkv.cube(hidden)` | `(T, dim)` | `(T, head_dim)` | KV 联合投影 (MLA) |
| 3 | `kv_norm(kv)` | `(T, head_dim)` | `(T, head_dim)` | RMSNorm 归一化 |
| 4 | `view` | - | `(T, 1, head_dim)` | 增加 head 维度 (1 个 KV head) |
| 5 | `partial_rotary_mul` | - | `(T, 1, head_dim)` | 部分 RoPE |
| 6 | `scatter` 到 SWA cache | - | - | 写入滑动窗口缓存 |

#### 多流并行优化

```
主线程:                     辅助流 (overlap stream):
┌─────────────────┐        ┌─────────────────┐
│ wq_a.vector     │        │ wkv.vector      │
│ wq_a.cube       │        │ wkv.cube        │
│ q_norm          │        │ kv_norm         │
│ wq_b.vector     │        │ kv_scatter      │
│ wq_b.cube       │        │ (写入SWA缓存)   │
│ q_rms + rope    │        └─────────────────┘
└─────────────────┘            │
       │                       │
       └───────────┬───────────┘
                   │
                   ▼
              事件同步
```

当 `multistream_dsv4_dsa_overlap=True` 时，Query 路径和 KV 路径在两个流中并行执行，通过事件同步。

### 5.3 阶段 2: Compressor — KV 压缩

**存在条件**：`compress_ratio > 1`（4 或 128）

#### Compressor 内部结构

```python
class Compressor(nn.Module):
    self.ape          # (compress_ratio, coff * head_dim) APE 参数
    self.wkv          # ReplicatedLinear(dim, coff * head_dim) KV 投影
    self.wgate        # ReplicatedLinear(dim, coff * head_dim) 门控投影
    self.norm         # RMSNorm(head_dim) 归一化
    self.state_cache  # 状态缓存 (kv_state + score_state)
    
    self.coff = 1 + self.overlap  # c4: coff=2, c128: coff=1
```

| 参数 | compress_ratio=4 | compress_ratio=128 |
|------|-----------------|-------------------|
| `overlap` | True | False |
| `coff` | 2 | 1 |
| `state_dim` | `2 * coff * head_dim = 4 * head_dim` | `2 * head_dim` |
| 状态组成 | kv_state + score_state (各 coff*head_dim) | kv_state + score_state (各 head_dim) |

#### 压缩流程

```
每 compress_ratio 个 token 压缩成 1 个 token:

token 0 ─┐
token 1  │
token 2  ├─→ 状态累积 (state_cache) ─→ norm + RoPE ─→ 压缩 KV (1个)
token 3  │
        ┘
  ←───── compress_ratio = 4 ─────→
```

**NPU 算子**: `torch.ops._C_ascend.compressor()`

### 5.4 阶段 3: Indexer — TopK 稀疏选择

**存在条件**：`compress_ratio == 4`

#### Indexer 内部结构

```python
class Indexer(nn.Module):
    self.n_heads = config.index_n_heads           # 通常 64
    self.head_dim = config.index_head_dim         # 通常 128
    self.index_topk = config.index_topk           # 通常 512
    self.wq_b = ReplicatedLinear(q_lora_rank, n_heads * head_dim)  # Query 投影
    self.weights_proj = ReplicatedLinear(dim, n_heads)              # 权重投影
    self.k_cache = AscendDeepseekV4IndexerCache                    # Indexer KV 缓存
    self.compressor = Compressor(...)                              # Indexer 内部压缩器
```

#### TopK 计算流程

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | `indexer.wq_b(qr)` | Indexer Query 投影 (从 qr 投影) |
| 2 | 与压缩 KV 计算注意力得分 | 计算 Query 与每个压缩 KV token 的相似度 |
| 3 | TopK 选择 | 选择 topk 个最重要的 token |
| 4 | 写入 topk_indices_buffer | 供后续层复用 (IndexCache) |

**skip_topk 模式**：
- 当 `skip_topk=True` 时，直接从 `topk_indices_buffer` 读取索引，跳过计算
- 当 `skip_topk=False` 时，计算索引并写入 `topk_indices_buffer`

**NPU 算子**: `torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata()`

### 5.5 阶段 4: 稀疏注意力计算

调用 NPU 算子执行完整的稀疏注意力：

```python
torch.ops._C_ascend.npu_sparse_flash_attention(
    q,                          # Query: (T, n_local_heads, head_dim)
    ori_kv=swa_kv_cache,        # 原始 KV (SWA 滑动窗口)
    cmp_kv=compress_kv_cache,   # 压缩 KV
    topk_idxs=compress_topk_idxs, # TopK 索引 (c4 时)
    sinks=attn_sink,            # 注意力 sink
    metadata=sas_metadata,      # 稀疏注意力元数据
    softmax_scale=scale,        # 缩放因子
    ...
)
```

**输出 Shape**: `(num_tokens, n_local_heads, head_dim)`

**掩码模式**：
- `ori_mask_mode=4`: SWA (滑动窗口注意力)，用于原始 KV
- `cmp_mask_mode=3`: Causal (因果注意力)，用于压缩 KV

### 5.6 阶段 5: 输出投影

| 步骤 | 操作 | 输入 Shape | 输出 Shape | 说明 |
|------|------|-----------|-----------|------|
| 1 | `partial_rotary_mul` (逆向) | `(T, n_local_heads, head_dim)` | `(T, n_local_heads, head_dim)` | 对 nope 部分应用逆向 RoPE |
| 2 | `view` 为分组 | - | `(T, n_local_groups, n_heads*head_dim//n_groups)` | 重塑为分组格式 |
| 3 | `wo_a(o_proj_input)` | `(T, n_local_groups, ...)` | `(T, n_local_groups, o_lora_rank)` | 组内投影 (列并行) |
| 4 | `view` flatten | - | `(T, n_local_groups * o_lora_rank)` | 展平 |
| 5 | `wo_b(o)` | `(T, n_groups*o_lora_rank)` | `(T, dim)` | 最终输出投影 (行并行) |

***

## 六、Prefill vs Decode 路径差异

| 维度 | Prefill (预填充) | Decode (解码) |
|------|-----------------|-------------|
| **Query 长度** | 多 token (变长) | 单 token (或少量 spec decode) |
| **KV 写入** | 写入新 token 的 KV 到所有缓存 | 仅写入当前 token |
| **索引计算** | 计算完整 TopK 索引 | 增量更新索引 |
| **注意力模式** | 变长注意力 (TND layout) | 定长解码 (BND layout) |
| **元数据** | `AscendDSAPrefillMetadata` | `AscendDSADecodeMetadata` |
| **实现方法** | `_forward_prefill()` | `_forward_decode()` |

***

## 七、KV 缓存体系

DeepSeek V4 使用**三类独立 KV 缓存**，协同工作以平衡精度和显存：

| 缓存类型 | 所属组件 | 压缩比 | 用途 | dtype (A5/非A5) |
|---------|---------|--------|------|----------------|
| **SWA KV Cache** | `swa_cache_layer` | 1x (无压缩) | 滑动窗口内的原始 KV，近期 token | float8_e4m3fn / bfloat16 |
| **Compress KV Cache** | `compressor` | 4x / 128x | 压缩后的历史 KV，长程记忆 | - |
| **Indexer KV Cache** | `indexer.compressor` | 4x | Indexer 用的压缩 KV，用于 TopK 选择 | float8_e4m3fn / int8 |

**A5 设备额外缓存**:
- `indexer_scale_cache`: FP8 缩放因子缓存
- `indexer_full_cache`: 完整精度缓存

**Compressor 状态缓存**:
- `state_cache`: 存储 `kv_state + score_state`，用于累积 `compress_ratio` 个 token 的状态
- dtype: float32

***

## 八、性能优化要点

### 8.1 并行策略

| 优化项 | 说明 |
|-------|------|
| **张量并行 (TP)** | Query 投影 B (wq_b)、输出投影 A/B (wo_a, wo_b) 做张量并行 |
| **上下文并行 (CP)** | `enable_dsa_cp()` 时启用，attn_sink 全头数，wq_b 用 ReplicatedLinear |
| **多流重叠** | Query 路径和 KV 路径在两个流中并行执行 (`multistream_dsv4_dsa_overlap`) |
| **CV 分离** | 线性层拆分为 Vector(量化) + Cube(矩阵乘)，便于流水重叠 |

### 8.2 存储优化

| 优化项 | 说明 |
|-------|------|
| **MLA 架构** | KV 共享 latent 表示，KV cache 从 2*head_dim 降到 head_dim |
| **KV 压缩** | Compressor 将历史 KV 压缩 4x / 128x，大幅节省显存 |
| **稀疏索引** | Indexer 只选择 TopK 个重要 token 参与注意力，减少计算量 |
| **IndexCache** | 多层复用 TopK 索引，进一步减少计算 |
| **FP8 存储** | A5 设备上 KV cache 使用 float8_e4m3fn 存储 |

### 8.3 计算优化

| 优化项 | 说明 |
|-------|------|
| **低秩分解** | Query 和 Output 都用低秩分解 (LoRA)，减少参数量和计算量 |
| **分组输出** | wo_a 按组投影，减少通信量 |
| **NPU 算子卸载** | 整个稀疏注意力完全卸载到 NPU 算子 `npu_sparse_flash_attention` |
| **无权重 RMSNorm** | Q 投影后用无权重 RMSNorm，节省参数 |
