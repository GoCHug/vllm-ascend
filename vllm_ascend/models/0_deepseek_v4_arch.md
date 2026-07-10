# DeepSeek V4 整体架构详解

> **本文档为 DeepSeek V4 在 Ascend NPU 上的整体架构总览。各子模块的详细实现请参考同级目录下的专题文档。**

---

## 一、架构总览

### 1.1 核心设计理念

DeepSeek V4 是一个融合了多项前沿技术的大规模语言模型，在 Ascend NPU 上的实现围绕三大核心创新展开：

| 技术 | 全称 | 核心作用 |
|------|------|---------|
| **HC** | Hyper-Complex 混合协同处理 | 沿 `hc_mult` 维度扩展特征空间，通过门控机制动态混合多通道特征，增强模型表达能力 |
| **DSA** | Deep Sparse Attention 深度稀疏注意力 | 三级 KV 缓存（SWA + Compressor + Indexer）协同，在超长上下文下平衡精度与显存 |
| **MoE** | Mixture of Experts 混合专家 | 稀疏激活的专家网络，在大参数量下保持合理的每次推理计算量 |

**三者在模型中的位置**：

```
每个 DecoderLayer:
┌────────────────────────────────────────────────────────────┐
│  Attention 分支                                            │
│    hc_pre → input_layernorm → DSA 注意力 → hc_post         │
│    ↑  HC 压缩为单通道计算  ↑  稀疏注意力   ↑  恢复 HC 维度  │
├────────────────────────────────────────────────────────────┤
│  FFN 分支                                                  │
│    hc_pre → post_attention_layernorm → MoE → hc_post       │
│    ↑  HC 压缩为单通道计算  ↑  稀疏专家   ↑  恢复 HC 维度    │
└────────────────────────────────────────────────────────────┘
```

### 1.2 文档体系索引

| 编号 | 文档 | 内容 | 详细程度 |
|------|------|------|---------|
| **0** | `0_deepseek_v4_arch.md` | **本文档** — 整体架构总览，模块关系，数据流总览 | ★☆☆ 总览 |
| **1** | `1_npu_hc_pre_npu.md` | `hc_pre` 算子详解：Python 接口 → C++ 绑定 → NPU 内核实现 | ★★★ 详细 |
| **2** | `2_npu_hc_post_npu.md` | `hc_post` 算子详解：Python 接口 → C++ 绑定 → Tiling → NPU 内核 | ★★★ 详细 |
| **3** | `3_deepseek_v4_attn.md` | DSA 注意力架构详解：五层架构 + 完整 Forward 数据流 + 优化要点 | ★★★ 详细 |
| **4** | `4_deepseek_v4_moe.md` | MoE 前馈网络详解：双路由模式 + FusedMoE + EPLB + NPU 算子 | ★★★ 详细 |
| **5** | `5_deepseek_v4_kv_cache.md` | KV 缓存体系详解：Compressor + Indexer + SWA + Block Size + A5 适配 | ★★★ 详细 |
| **6** | `6_deepseek_v4_mtp.md` | MTP 多 Token 预测详解：投影融合 + 轮询调度 + 延迟 HC 压缩 + 权重映射 | ★★★ 详细 |
| **7** | `7_deepseek_v4_postprocess.md` | 后处理与采样详解：LogitsProcessor + ParallelLMHead + 惩罚项 + TopK/TopP + Reduce Sample + 拒绝采样 | ★★★ 详细 |

### 1.3 完整模块树

```
AscendDeepseekV4ForCausalLM (L1210, 模型入口)
├── model: DeepseekV4Model (L1005)
│   ├── embed_tokens: VocabParallelEmbedding        [仅 PP 首阶段]
│   ├── layers[]: DeepseekV2DecoderLayer × N (L911)
│   │   ├── input_layernorm: RMSNorm
│   │   ├── post_attention_layernorm: RMSNorm
│   │   ├── self_attn: DeepseekV4Attention (L709)
│   │   │   └── dsa_attn: AscendDeepseekSparseAttention
│   │   │       └── [详细见 3_deepseek_v4_attn.md]
│   │   ├── mlp: DeepseekV4MoE (L352) / DeepseekV2MLP (L303)
│   │   ├── hc_pre / hc_post: HC 算子方法
│   │   │   └── [详细见 1_npu_hc_pre_npu.md / 2_npu_hc_post_npu.md]
│   │   └── hc_attn / hc_ffn: {fn, base, scale} 参数组
│   ├── norm: RMSNorm                              [仅 PP 末阶段]
│   ├── hc_head: {fn, base, scale}                [模型级 HC 头]
│   ├── topk_indices_buffer: 模型级共享 TopK 索引缓存
│   └── _mtp_hidden_buffer: MTP 隐藏状态缓冲区
├── lm_head: ParallelLMHead                        [仅 PP 末阶段]
└── logits_processor: LogitsProcessor
```

**KV 缓存体系**：

```
├── AscendDeepseekV4SWACache       — 滑动窗口 KV 缓存 (所有层)
├── AscendCompressorStateCache     — Compressor 状态缓存 (c4/c128 层)
└── AscendDeepseekV4IndexerCache   — Indexer KV 缓存 (c4 层)
```

### 1.4 代码文件索引

| 模块 | 文件路径 | 核心类 / 函数 |
|------|---------|-------------|
| 主模型 | `vllm_ascend/models/deepseek_v4.py` | `AscendDeepseekV4ForCausalLM`, `DeepseekV4Model`, `DeepseekV2DecoderLayer`, `DeepseekV4Attention`, `DeepseekV4MoE`, `Compressor`, `Indexer` |
| MTP | `vllm_ascend/models/deepseek_v4_mtp.py` | `DeepSeekV4MTP`, `DeepSeekMultiTokenPredictor` |
| 注意力层基类 | `vllm_ascend/models/layer/attention/layer.py` | `DSAAttention` |
| DSA 注意力后端 | `vllm_ascend/attention/dsa_v1.py` | `AscendDSABackend`, `AscendDSAImpl` |
| DSA 外层封装 | `vllm_ascend/ops/dsa.py` | `AscendDeepseekSparseAttention`, `DSAModules` |
| HC pre 算子 | `csrc/moe/hc_pre/` | Host + Kernel 实现 |
| HC post 算子 | `csrc/moe/hc_post/` | Host + Kernel 实现 |

---

## 二、模型入口与整体数据流

### 2.1 DeepseekV4Model — 模型主体

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L1005`

#### 核心成员变量

| 成员 | 类型 | 说明 |
|------|------|------|
| `embed_tokens` | VocabParallelEmbedding | 词嵌入层，仅 PP 首阶段实例化 |
| `layers` | ModuleList | 解码器层数组，按 PP 阶段切片 |
| `norm` | RMSNorm | 最终归一化，仅 PP 末阶段实例化 |
| `hc_mult` | int | HC 扩展倍数（通常为 4） |
| `hc_eps` | float | HC 数值稳定性 epsilon |
| `norm_eps` | float | RMSNorm epsilon |
| `hc_head_fn` | nn.Parameter | `(hc_mult, hc_mult * hidden_size)` HC 头投影权重 |
| `hc_head_base` | nn.Parameter | `(hc_mult,)` HC 头偏置 |
| `hc_head_scale` | nn.Parameter | `(1,)` HC 头缩放因子 |
| `topk_indices_buffer` | Tensor \| None | 模型级 TopK 索引缓冲区（IndexCache 用） |
| `_mtp_hidden_buffer` | Tensor \| None | MTP 隐藏状态缓冲区，预 hc_head 残差流 |

### 2.2 Forward 整体流程

**调用链**:

```
Scheduler → model_runner.execute_model() → prepare_inputs()
    → self.model(**model_inputs) → DeepseekV4Model.forward()
```

#### 输入参数

| 参数 | 形状 | 说明 |
|------|------|------|
| `input_ids` | `(num_tokens,)` | 展平的 token ID 序列（连续批处理） |
| `positions` | `(num_tokens,)` | 每个 token 在序列中的绝对位置（用于 RoPE） |
| `intermediate_tensors` | `(num_tokens, hc_mult, hidden_size)` \| None | PP 中间张量（非首阶段） |
| `inputs_embeds` | `(num_tokens, hidden_size)` \| None | 预计算嵌入（优先于 input_ids） |

> **注意**: `num_tokens = sum(seq_len for each request in batch)`，不是 `batch_size * seq_len`，vLLM 采用 continuous batching 机制。

#### Forward 阶段总览

| 阶段 | 操作 | 形状变化 | PP 阶段 |
|------|------|---------|---------|
| 1 | **输入嵌入** | `(num_tokens,) → (num_tokens, hidden_size)` | 首阶段 |
| 2 | **HC 扩展** | `(num_tokens, hidden_size) → (num_tokens, hc_mult, hidden_size)` | 首阶段 |
| 3 | **层前向传播** | `(num_tokens, hc_mult, hidden_size) → (num_tokens, hc_mult, hidden_size)` | 所有阶段 |
| 4 | **MTP 缓存** | 保存预 hc_head 状态到 `_mtp_hidden_buffer` | 末阶段 |
| 5 | **HC 头压缩** | `(num_tokens, hc_mult, hidden_size) → (num_tokens, hidden_size)` | 末阶段 |
| 6 | **最终归一化** | `(num_tokens, hidden_size) → (num_tokens, hidden_size)` | 末阶段 |

### 2.3 HC Head — 模型级 HC 压缩

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L1085`

所有层处理完成后，将 HC 扩展维度压缩回原始维度。

```python
def hc_head(self, x, hc_fn, hc_scale, hc_base):
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()                    # (T, hc_mult * hidden_size)
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt           # 线性投影 + RMSNorm
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps  # 门控
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)  # 加权融合
    return y.to(dtype)
```

**流程示意** (hc_mult=4)：

```
输入: (num_tokens, 4, hidden_size)
    │
    ├─ flatten + float → (T, 4*hidden_size)
    ├─ x² → mean → rsqrt  (RMSNorm 因子)
    ├─ linear(x, hc_fn) * rsqrt → mixes (T, 4)
    ├─ sigmoid(mixes*scale + base) + eps → pre (T, 4) 门控权重
    └─ sum(pre * x, dim=1) → (T, hidden_size)
输出: (num_tokens, hidden_size)
```

---

## 三、DecoderLayer 结构与 HC 机制

### 3.1 DeepseekV2DecoderLayer

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L911`

每个解码器层包含注意力和 FFN 两个分支，都使用 HC 机制。

#### 核心成员变量

| 成员 | 类型 | 说明 |
|------|------|------|
| `self_attn` | DeepseekV4Attention | 自注意力层 |
| `mlp` | DeepseekV4MoE \| DeepseekV2MLP | 前馈网络层（MoE 或 dense） |
| `input_layernorm` | RMSNorm | 注意力前置层归一化 |
| `post_attention_layernorm` | RMSNorm | FFN 前置层归一化 |
| `hc_attn_fn / base / scale` | nn.Parameter | 注意力 HC 参数组 |
| `hc_ffn_fn / base / scale` | nn.Parameter | FFN HC 参数组 |
| `hc_mult` | int | HC 扩展倍数 |
| `hc_sinkhorn_iters` | int | Sinkhorn 迭代次数（通常 20） |

**参数形状**：
- `mix_hc = (2 + hc_mult) * hc_mult` — 混合门控数量
- `hc_dim = hc_mult * hidden_size` — HC 扩展维度
- `hc_fn`: `(mix_hc, hc_dim)`，`hc_base`: `(mix_hc,)`，`hc_scale`: `(3,)`

### 3.2 HC 机制概述

HC (Hyper-Complex) 机制的核心思想：
1. **扩展**: 将隐藏状态从 `hidden_size` 扩展到 `hc_mult * hidden_size`
2. **压缩计算**: 注意力/FFN 只在压缩后的单通道（`hidden_size`）上计算，减少计算量
3. **门控融合**: 通过可学习的门控权重动态混合多通道特征，保持表达能力

**层内 HC 包含两个算子**：
- **hc_pre**: HC 前置处理 — 从 HC 维度压缩到主路径
- **hc_post**: HC 后置处理 — 从主路径恢复到 HC 维度

> **详细实现**：
> - hc_pre 算子 → 参考 `1_npu_hc_pre_npu.md`
> - hc_post 算子 → 参考 `2_npu_hc_post_npu.md`

### 3.3 DecoderLayer Forward 完整流程

```
输入: hidden_states (num_tokens, hc_mult, hidden_size)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Attention 分支                                              │
├─────────────────────────────────────────────────────────────┤
│  1. residual = hidden_states.clone()                        │
│     保存完整 HC 维度残差                                      │
│                                                             │
│  2. hidden_states = hc_pre(                                 │
│        hidden_states, hc_attn_fn, hc_attn_scale, hc_attn_base)│
│     压缩到主路径: (T, hc_mult, H) → (T, H)                   │
│     返回: (hidden_states, post, comb)                       │
│                                                             │
│  3. hidden_states = input_layernorm(hidden_states)          │
│     RMSNorm 归一化                                           │
│                                                             │
│  4. hidden_states = self_attn(positions, hidden_states, ...)│
│     自注意力计算（DSA 稀疏注意力）                            │
│                                                             │
│  5. hidden_states = hc_post(                                │
│        hidden_states, residual, post, comb)                 │
│     恢复 HC 维度: (T, H) → (T, hc_mult, H)                   │
└──────────────────────────────┬──────────────────────────────┘
                               │ hidden_states: (T, hc_mult, H)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  FFN 分支                                                    │
├─────────────────────────────────────────────────────────────┤
│  6. residual = hidden_states.clone()                        │
│     保存完整 HC 维度残差                                      │
│                                                             │
│  7. hidden_states = hc_pre(                                 │
│        hidden_states, hc_ffn_fn, hc_ffn_scale, hc_ffn_base) │
│     压缩到主路径                                             │
│                                                             │
│  8. hidden_states = post_attention_layernorm(hidden_states) │
│     RMSNorm 归一化                                           │
│                                                             │
│  9. hidden_states = mlp(hidden_states)                      │
│     MoE / 稠密 MLP 计算                                      │
│                                                             │
│  10. hidden_states = hc_post(                               │
│         hidden_states, residual, post, comb)                │
│      恢复 HC 维度                                            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
                 返回 (hidden_states, residual)
                 hidden_states: (num_tokens, hc_mult, hidden_size)
                 residual: (num_tokens, hc_mult, hidden_size)
```

> **设计要点**:
> 1. **计算效率**: 注意力/FFN 仅在压缩后的主路径（单通道）执行，大幅减少计算量
> 2. **表达能力**: 通过 HC 门控仍保留 `hc_mult` 个特征通道的信息容量
> 3. **残差连接**: 每个分支前保存完整 HC 维度残差，hc_post 中门控融合
> 4. **返回值**: 返回 `(hidden_states, residual)` 元组（与标准 Transformer 不同）

### 3.4 HC 参数分布汇总

| 参数位置 | fn 形状 | base 形状 | scale 形状 |
|---------|--------|----------|-----------|
| `hc_attn` (每层注意力) | `(mix_hc, hc_dim)` | `(mix_hc,)` | `(3,)` |
| `hc_ffn` (每层 FFN) | `(mix_hc, hc_dim)` | `(mix_hc,)` | `(3,)` |
| `hc_head` (模型级) | `(hc_mult, hc_dim)` | `(hc_mult,)` | `(1,)` |

其中：
- `mix_hc = (2 + hc_mult) * hc_mult`
- `hc_dim = hc_mult * hidden_size`

---

## 四、注意力机制总览

### 4.1 DSA — Deep Sparse Attention

DeepSeek V4 采用 **MLA (Multi-head Latent Attention)** + **DSA (Deep Sparse Attention)** 架构，通过三级 KV 缓存体系实现超长上下文推理。

> **完整详解** → 参考 `3_deepseek_v4_attn.md`

### 4.2 三层 KV 缓存架构

```
原始 KV (O(N²) 显存)
    │
    ├── SWA (滑动窗口): 最近 window_size 个 token 的完整 KV
    │                    → 高精度，近期记忆
    │
    ├── Compressor (KV压缩): 历史 KV 按 4x / 128x 压缩存储
    │                       → 减少显存，长程记忆
    │
    └── Indexer (稀疏索引): compress_ratio=4 时，TopK 选择最重要的 token
                            → 进一步减少计算量
```

### 4.3 按层压缩配置

| compress_ratio | Compressor | Indexer | SWA | 适用场景 |
|---------------|-----------|---------|-----|---------|
| **1** | ✗ | ✗ | ✓ | 无压缩层，仅滑动窗口 |
| **4** | ✓ (overlap=True) | ✓ (TopK 选择) | ✓ | 中距离层，稀疏索引+压缩 |
| **128** | ✓ (overlap=False) | ✗ | ✓ | 长距离层，深度压缩 |

- 通过 `get_dsv4_compress_ratio(config, layer_idx)` 按层获取
- 层索引从 0 开始，不同层有不同的压缩策略

### 4.4 五层架构分层

```
第1层: DeepseekV4Attention       — Python 封装层，初始化所有子模块
第2层: AscendDeepseekSparseAttention — 外层封装，DSAModules 组装
第3层: DSAAttention              — 注意力层基类，KV cache 管理
第4层: AscendDSABackend          — NPU 后端注册
第5层: AscendDSAImpl             — 核心计算实现 (prefill / decode)
```

### 4.5 Forward 五大阶段

| 阶段 | 操作 | 关键组件 |
|------|------|---------|
| 1 | **MLA Prolog** | Query: wq_a → q_norm → wq_b → q_rms → RoPE<br>KV: wkv → kv_norm → RoPE → SWA cache |
| 2 | **Compressor** | KV 压缩 (compress_ratio > 1 时) |
| 3 | **Indexer** | TopK 稀疏选择 (compress_ratio == 4 时) |
| 4 | **稀疏注意力** | `npu_sparse_flash_attention` NPU 算子 |
| 5 | **输出投影** | nope_rope 逆向 → wo_a → wo_b |

> **完整数据流与 shape 追踪** → 参考 `3_deepseek_v4_attn.md` 第五章

### 4.6 IndexCache — 索引缓存复用

V3.2+ 特性，部分层复用前一层的 TopK 索引以减少计算。

| 模式 | 配置项 | 说明 |
|------|--------|------|
| 频率模式 | `index_topk_freq` | 每 freq 层计算一次索引 |
| 模式匹配 | `index_topk_pattern` | 如 "FSFS"，F=计算, S=跳过 |

- `skip_topk=True`: 跳过计算，从 `topk_indices_buffer` 复用
- `skip_topk=False`: 计算索引并写入 buffer

---

## 五、MoE 前馈网络

### 5.1 DeepseekV4MoE

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L352`

#### 核心组成

| 类别 | 组件 | 说明 |
|------|------|------|
| **路由门控** | `gate` | ReplicatedLinear，`(hidden_size, n_routed_experts)`<br>包含 `tid2eid` (hash路由) 或 `e_score_correction_bias` (线性路由) |
| **专家网络** | `experts` | FusedMoE 融合专家网络 |
| **共享专家** | `shared_experts` | DeepseekV2MLP 或 None (mix_placement 时融合到 FusedMoE) |
| **配置** | `n_routed_experts` | 路由专家数 |
| | `n_shared_experts` | 共享专家数 |
| | `n_redundant_experts` | 冗余专家数 (EPLB) |
| | `hash` | 是否使用 hash 路由 (前 num_hash_layers 层) |
| | `enable_eplb` | 启用专家并行负载均衡 |
| | `routed_scaling_factor` | 路由缩放因子 (默认 1.5) |

### 5.2 双路由模式

| 模式 | 触发条件 | 实现方式 |
|------|---------|---------|
| **Hash 路由** | `layer_idx < num_hash_layers` 且非 draft 层 | 预定义 `tid2eid` 映射，无学习参数 |
| **线性路由** | 其他层 | gate 线性层 + `e_score_correction_bias` 修正偏置 |

### 5.3 Forward 流程

```
输入: hidden_states (num_tokens, hidden_size)
    │
    ├─ [序列并行] sequence_parallel_chunk 分片
    │
    ├─ 路由计算:
    │   ├─ hash路由 → 直接用 tid2eid 映射
    │   └─ 线性路由 → F.linear(hidden, gate.weight) → router_logits
    │
    ├─ FusedMoE 专家计算:
    │   └─ experts(hidden_states, router_logits)
    │      返回 (shared_output, final_hidden_states) 或单个 tensor
    │
    ├─ 输出融合:
    │   └─ muls_add_triton: shared + routed * routed_scaling_factor
    │
    ├─ [序列并行] all_gather + 裁剪 pad
    │
    └─ [TP>1] maybe_all_reduce_tensor_model_parallel
```

### 5.4 DeepseekV2MLP (稠密 MLP)

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L303`

非 MoE 层使用的稠密前馈网络：

```
hidden → gate_up_proj → SiluAndMul → down_proj → output
```

| 组件 | 类型 | 说明 |
|------|------|------|
| `gate_up_proj` | MergedColumnParallelLinear | 门控 + 上投影合并 |
| `act_fn` | SiluAndMul | SiLU 激活 + 门控乘法 |
| `down_proj` | RowParallelLinear | 下投影 |

---

## 六、KV 缓存体系

### 6.1 三类 KV 缓存对比

| 缓存类型 | 所属 | 压缩比 | dtype (A5 / 非A5) | 用途 |
|---------|------|--------|-------------------|------|
| **SWA KV Cache** | `swa_cache_layer` | 1x | float8_e4m3fn / bfloat16 | 滑动窗口内原始 KV |
| **Compress State Cache** | `compressor.state_cache` | 4x / 128x | float32 | 压缩状态 (kv_state + score_state) |
| **Indexer KV Cache** | `indexer.k_cache` | 4x | float8_e4m3fn / int8 | Indexer 用压缩 KV |

### 6.2 AscendCompressorStateCache

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L108`

- **state_dim**: `2 * coff * head_dim` (c4) 或 `2 * head_dim` (c128)
  - `coff = 1 + overlap`，c4 时 overlap=True → coff=2，c128 时 overlap=False → coff=1
- **state 组成**: `kv_state + score_state`（各占一半）
- **dtype**: float32
- 返回 `AscendSlidingWindowMLASpec`

### 6.3 AscendDeepseekV4IndexerCache

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L141`

- 仅 `compress_ratio == 4` 时创建
- **dtype**: A5 → float8_e4m3fn，其他 → int8
- A5 设备有额外 `scale_cache` 和 `full_cache`
- 返回 `AscendMLAAttentionSpec`

### 6.4 AscendDeepseekV4SWACache

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L177`

- **window_size**: 来自 `config.sliding_window`
- **dtype**: A5 → float8_e4m3fn，其他 → bfloat16
- **cached_head_size**: A5 设备额外 +128
- 返回 `AscendSlidingWindowMLASpec`

### 6.5 Block Size 配置

**定义位置**: `vllm_ascend/models/layer/attention/layer.py`

| 全局 block_size | mla | swa | c4_state | c128_state |
|----------------|-----|-----|----------|-----------|
| 128 | 128 | 128 | 8 | 32 |
| 64 | 64 | 64 | 4 | 16 |
| 32 | 32 | 32 | 2 | 8 |

A5 设备 c128_state 为 16/8/4（与非 A5 不同）。

---

## 七、MTP 多 Token 预测

### 7.1 MTP 模块架构

**文件位置**: `vllm_ascend/models/deepseek_v4_mtp.py`

```
DeepSeekV4MTP (入口类)
└── model: DeepSeekMultiTokenPredictor
    ├── layers: ModuleDict[str, DeepSeekMultiTokenPredictorLayer] (按 step_idx 轮询)
    │   ├── e_proj: ReplicatedLinear (嵌入投影)
    │   ├── h_proj: ReplicatedLinear (隐藏状态投影)
    │   ├── enorm: RMSNorm (嵌入归一化)
    │   ├── hnorm: RMSNorm (隐藏状态归一化)
    │   ├── shared_head: SharedHead
    │   │   ├── norm: RMSNorm
    │   │   └── head: ParallelLMHead
    │   ├── mtp_block: DeepseekV2DecoderLayer (is_draft_layer=True)
    │   └── hc_head {fn, base, scale}: HC 头参数
    ├── embed_tokens: VocabParallelEmbedding
    └── logits_processor: LogitsProcessor
```

### 7.2 核心特性

| 特性 | 说明 |
|------|------|
| **轮询调度** | `num_mtp_layers` 个 MTP 层按 `spec_step_idx % num_mtp_layers` 轮询使用 |
| **投影融合** | `e_proj(inputs_embeds) + h_proj(previous_hidden_states)` 广播相加 |
| **Mask 机制** | `position == 0` 的输入嵌入被 mask 为 0 |
| **延迟 HC 压缩** | Layer forward 保持 `hc_mult` 维度，仅在 `compute_logits` 时做 hc_head 压缩 |
| **共享 LM Head** | `shared_head` 被所有 MTP 步骤复用 |
| **缓冲区共享** | `topk_indices_buffer` 和 `_mtp_hidden_buffer` 在主模型和 MTP 间共享 |

### 7.3 与主模型的交互

- `_mtp_hidden_buffer`: 主模型 forward 中保存预 hc_head 的隐藏状态
- `get_mtp_target_hidden_states()`: MTP 从主模型获取目标隐藏状态用于训练

---

## 八、并行策略与 NPU 优化

### 8.1 并行策略总览

```
├── TP (Tensor Parallel) 张量并行
│   ├── wq_b: enable_dsa_cp 时复制，否则列切分
│   ├── wo_a: 列切分
│   ├── wo_b: 行切分 + allreduce
│   ├── gate_up_proj: 列切分
│   └── down_proj: 行切分
│
├── PP (Pipeline Parallel) 流水线并行
│   ├── embed_tokens: 首阶段
│   ├── layers: 按 start_layer / end_layer 切片
│   ├── norm + lm_head: 末阶段
│   └── IntermediateTensors 传递 hc_mult 维隐藏状态
│
├── EP (Expert Parallel) 专家并行
│   ├── 物理专家按 ep_size 切分
│   └── EPLB 支持专家负载均衡重映射
│
├── CP (Context Parallel) 上下文并行
│   └── enable_dsa_cp() 时启用，attn_sink 全头数，wq_b 用 ReplicatedLinear
│
└── SP (Sequence Parallel) 序列并行
    ├── MLP 输入 sequence_parallel_chunk 分片
    ├── FlashComm1: 每层输出 reduce_scatter
    └── MTP 缓冲区保存前 all_gather 聚合
```

### 8.2 NPU 核心算子

| 功能 | NPU 算子 |
|------|---------|
| 稀疏注意力 | `torch.ops._C_ascend.npu_sparse_flash_attention` |
| HC 前置处理 | `torch.ops._C_ascend.npu_hc_pre` |
| HC 后置处理 | `torch.ops._C_ascend.npu_hc_post` |
| MoE 路由 | `torch.ops._C_ascend.moe_gating_top_k_hash` |
| RoPE 旋转 | `torch_npu.npu_rotary_mul` |
| Compressor | `torch.ops._C_ascend.compressor` |
| Indexer | `npu_lightning_indexer` / `npu_vllm_quant_lightning_indexer_metadata` |
| 缩放融合 | `muls_add_triton` (Triton 算子) |

### 8.3 设备类型适配

| 特性 | A5 设备 | 非 A5 设备 |
|------|--------|-----------|
| SWA KV dtype | float8_e4m3fn | bfloat16 |
| Indexer KV dtype | float8_e4m3fn | int8 |
| Compressor norm dtype | float32 | 常规 |
| Block size 配置 | 独立配置 | 标准配置 |
| FP8 缩放因子缓存 | 有 | 无 |

### 8.4 性能优化要点

| 优化类别 | 优化项 |
|---------|--------|
| **计算优化** | 低秩分解 (wq_a/wq_b, wo_a/wo_b)、分组输出、无权重 RMSNorm |
| **存储优化** | MLA 架构 (KV 共享 latent)、KV 压缩 (4x/128x)、稀疏索引、IndexCache、FP8 存储 |
| **并行优化** | 多流重叠 (Query/KV 流并行)、CV 分离 (Vector + Cube)、序列并行 |
| **算子卸载** | 整个稀疏注意力完全卸载到 NPU 算子执行 |
