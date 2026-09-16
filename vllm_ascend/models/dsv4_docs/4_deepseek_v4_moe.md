# DeepSeek V4 MoE 前馈网络详解

> **本文档详细讲解 DeepSeek V4 的 MoE（Mixture of Experts）前馈网络实现，包括路由机制、专家网络、共享专家、EPLB 负载均衡等核心技术。**

---

## 一、MoE 架构总览

### 1.1 设计理念

DeepSeek V4 的 FFN 层采用 **MoE (Mixture of Experts)** 架构，通过稀疏激活的专家网络在大参数量下保持合理的每次推理计算量。每个 token 只激活 Top-K 个专家进行计算。

```
输入 hidden_states (num_tokens, hidden_size)
    │
    ├─ 路由门控: 计算每个 token 到各专家的权重
    ├─ Top-K 选择: 选取权重最高的 K 个专家
    └─ 专家计算: 被选中的专家并行处理 token
         ├─ 路由专家 (routed_experts): 稀疏激活
         └─ 共享专家 (shared_experts): 全量激活
    │
    └─ 输出融合: 加权融合所有激活专家的输出
```

### 1.2 核心类关系

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L352`

```
DeepseekV4MoE (MoE 主类)
├── gate: ReplicatedLinear                    — 路由门控线性层
│   ├── tid2eid: Parameter (hash路由时)        — token ID → expert ID 映射
│   └── e_score_correction_bias: Parameter    — 专家得分修正偏置 (线性路由时)
├── experts: FusedMoE                         — 融合专家网络执行器
│   └── [内部包含多个专家FFN的融合实现]
└── shared_experts: DeepseekV2MLP | None      — 共享专家 (mix_placement 时为 None)
```

### 1.3 稠密 MLP 对比

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L303`

非 MoE 层使用标准的稠密 MLP（SiluAndMul 激活）：

```
hidden → gate_up_proj → SiluAndMul → down_proj → output
```

| 组件 | 类型 | 说明 |
|------|------|------|
| `gate_up_proj` | MergedColumnParallelLinear | 门控 + 上投影合并，列并行切分 |
| `act_fn` | SiluAndMul \| SiluAndMulWithClamp | SiLU 激活 + 门控乘法 |
| `down_proj` | RowParallelLinear | 下投影，行并行切分 + allreduce |

---

## 二、DeepseekV4MoE 成员详解

### 2.1 核心配置参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `layer_idx` | int | 当前层索引，从权重名 `prefix.split(sep=".")[-2]` 提取 |
| `routed_scaling_factor` | float | 路由输出缩放因子，默认 1.5 |
| `swiglu_limit` | float \| None | SiLU 激活上限（clamp），None 表示不限制 |
| `tp_size` / `tp_rank` | int | 张量并行世界大小 / 当前 rank |
| `ep_size` / `ep_rank` | int | 专家并行世界大小 / 当前 rank |
| `n_routed_experts` | int | 路由专家总数 |
| `n_shared_experts` | int | 共享专家数 |
| `hash` | bool | 是否使用 hash 路由（前 num_hash_layers 层且非 draft 层） |

### 2.2 EPLB 负载均衡配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `enable_eplb` | bool | 是否启用专家并行负载均衡 |
| `n_redundant_experts` | int | 冗余专家数 |
| `n_logical_experts` | int | 逻辑专家数 = n_routed_experts |
| `n_physical_experts` | int | 物理专家数 = n_logical + n_redundant |
| `n_local_physical_experts` | int | 每个 EP rank 的本地物理专家数 |
| `physical_expert_start/end` | int | 本地物理专家索引范围 |

### 2.3 门控层 gate

**类型**: `ReplicatedLinear(hidden_size → n_routed_experts)`

| 属性 | 类型 | 路由模式 | 说明 |
|------|------|---------|------|
| `weight` | Parameter | 全部 | 门控权重，fp32 精度 (`precast_fp32_weight = True`) |
| `tid2eid` | Parameter `(vocab_size, top_k)` int32 | Hash 路由 | 预定义 token ID → expert ID 映射，无梯度 |
| `e_score_correction_bias` | Parameter `(n_routed_experts,)` float32 | 线性路由 | 专家得分修正偏置，用于负载均衡 |

> **设计要点**: 门控权重始终用 fp32 精度计算，确保路由决策的数值稳定性。

### 2.4 共享专家 shared_experts

共享专家是所有 token 都经过的稠密网络：

- **mix_placement 禁用**: `shared_experts = DeepseekV2MLP`
  - `intermediate_size = moe_intermediate_size * n_shared_experts`
  - 相当于把多个共享专家合并成一个大 MLP
- **mix_placement 启用**: `shared_experts = None`
  - 共享专家融合到 FusedMoE 的 experts 中，统一调度

### 2.5 FusedMoE 专家网络

`FusedMoE` 是融合的专家执行器，构造参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_experts` | `config.n_routed_experts` | 路由专家数 |
| `top_k` | `config.num_experts_per_tok` | 每个 token 激活的专家数 |
| `hidden_size` | `config.hidden_size` | 隐藏维度 |
| `intermediate_size` | `config.moe_intermediate_size` | 专家中间维度 |
| `renormalize` | `config.norm_topk_prob` | 是否归一化 TopK 权重 |
| `use_grouped_topk` | True | 使用分组 TopK |
| `num_expert_group` | `config.n_group` | 专家组数 |
| `topk_group` | `config.topk_group` | 每组选 TopK 个 |
| `scoring_func` | `config.scoring_func` | 得分函数 (softmax 等) |
| `routed_scaling_factor` | `self.routed_scaling_factor` | 路由缩放因子 |
| `hash` | bool | 是否 hash 路由 |
| `tid2eid` | Tensor \| None | hash 路由映射表 |
| `enable_eplb` | bool | 是否启用 EPLB |
| `n_shared_experts` | int (mix_placement 时) | 融合到 experts 的共享专家数 |

---

## 三、双路由模式

### 3.1 Hash 路由

**触发条件**: `layer_idx < config.num_hash_layers and not is_draft_layer`

Hash 路由使用预定义的 `tid2eid` 映射表，直接根据 token ID 分配专家，无学习参数。

```
输入: input_ids (num_tokens,)
    │
    └─ 查表 tid2eid[input_ids] → expert_indices (num_tokens, top_k)
```

**特点**：
- 计算量极小，仅查表操作
- 无训练参数，路由策略固定
- 适用于底层网络层，提供稳定的专家分配

### 3.2 线性路由

**触发条件**: 非 hash 路由层（中高层）

线性路由通过可学习的门控权重计算专家得分：

```
输入: hidden_states (num_tokens, hidden_size)
    │
    ├─ F.linear(hidden.float(), gate.weight) → logits (num_tokens, n_experts)
    │     ↑ 始终用 fp32 计算
    │
    ├─ [+ e_score_correction_bias]  得分修正
    │
    └─ 分组 TopK 选择 → 选出 top_k 个专家
```

**特点**：
- 可学习，路由策略随训练优化
- 分组 TopK 确保专家多样性
- `e_score_correction_bias` 用于动态负载均衡调整

---

## 四、Forward 完整流程

### 4.1 流程图

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L460`

```
输入: hidden_states (num_tokens, hidden_size)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  1. 序列并行分片 (可选)                               │
│     if is_sequence_parallel:                         │
│       hidden_states = sequence_parallel_chunk(x)    │
│     按 TP rank 分片，避免重复计算                     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  2. 路由计算                                          │
│     if is_internal_router:                           │
│       FusedMoE 内部计算路由                           │
│     else:                                            │
│       router_logits = F.linear(x.float(), gate.w)    │
│       FusedMoE 外部传入 router_logits                │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  3. FusedMoE 专家计算                                 │
│     fused_moe_out = experts(x, router_logits)        │
│     输出: tuple(shared_output, final_hidden) 或 tensor│
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  4. 输出融合                                          │
│     if tuple 输出:                                    │
│       if shared_experts 存在:                         │
│         muls_add_triton(routed, shared, factor)      │
│       else:                                          │
│         final *= routed_scaling_factor               │
│     注: fp16 和非 fp16 的参数顺序不同                  │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  5. 序列并行聚合 (可选)                               │
│     if is_sequence_parallel:                         │
│       all_gather + 裁剪 pad 到原 num_tokens          │
│     elif tp>1 and tuple输出:                         │
│       maybe_all_reduce_tensor_model_parallel         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
              输出: (num_tokens, hidden_size)
```

### 4.2 序列并行

当 `parallel_config.use_sequence_parallel_moe = True` 时启用：

| 阶段 | 操作 | 说明 |
|------|------|------|
| 输入 | `sequence_parallel_chunk` | 将 token 序列按 TP rank 分片，每个 rank 只处理一部分 token |
| 输出 | `tensor_model_parallel_all_gather` + 裁剪 | 聚合所有 rank 的结果，裁剪掉 padding |

**设计目的**: 避免 MoE 层中各 TP rank 重复计算相同 token，提高算力利用率。

### 4.3 输出融合模式

`muls_add_triton` Triton 算子实现加权融合：

```python
final_hidden_states = muls_add_triton(
    final_hidden_states,   # 路由专家输出
    shared_output,         # 共享专家输出
    self.routed_scaling_factor  # 缩放因子
)
```

**注意 dtype 差异**：
- `dtype != float16`: `routed * factor + shared`
- `dtype == float16`: `shared + routed / factor`（参数顺序和运算不同）

---

## 五、EPLB 专家并行负载均衡

### 5.1 基本概念

**EPLB (Expert Parallel Load Balancing)** 通过冗余专家解决专家并行中的负载不均衡问题。

```
逻辑专家 (n_logical_experts = n_routed_experts)
    │
    │  + 冗余专家 (n_redundant_experts)
    ▼
物理专家 (n_physical_experts = n_logical + n_redundant)
    │
    │  按 EP rank 切分
    ▼
每个 rank 持有 n_local_physical_experts 个物理专家
```

### 5.2 工作原理

当 token 路由到的逻辑专家在当前 EP rank 上没有副本时，可以路由到冗余副本，减少跨 rank 通信开销。

**配置来源**: `parallel_config.eplb_config.num_redundant_experts`

---

## 六、分组 TopK 路由

### 6.1 分组机制

DeepSeek V4 采用 **分组 TopK** 策略，确保专家选择的多样性：

```
n_routed_experts 个专家
    │
    ├─ 分成 n_group 个组，每组 n_experts_per_group 个
    │
    ├─ 每组内选 topk_group 个专家
    │
    └─ 总共选 top_k = topk_group * n_group? 个专家
```

**配置参数**：
- `n_group`: 专家组数
- `topk_group`: 每组选几个
- `num_experts_per_tok`: 每 token 选的总专家数

### 6.2 得分函数

`scoring_func` 支持多种：
- `softmax`: 标准 softmax 归一化
- 其他可配置的得分函数

**TopK 归一化**: `renormalize = config.norm_topk_prob`，选中的 K 个权重重新归一化。

---

## 七、NPU MoE 相关算子

在 `csrc/moe/` 目录下，DeepSeek V4 的 MoE 在 Ascend NPU 上使用多个定制算子：

| 算子目录 | 功能 |
|---------|------|
| `moe_gating_top_k` / `moe_gating_top_k_hash` | TopK 路由选择（普通版 / hash 版） |
| `moe_init_routing_custom` | 路由初始化与排序 |
| `moe_grouped_matmul` | 分组矩阵乘（专家 FFN 计算） |
| `swiglu_group_quant` | SiGLU 激活 + 量化 |
| `scatter_nd_update_v2` | 散射更新（结果写回） |
| `chunk_gated_delta_rule_fwd_h` | 分块门控 delta 规则 |
| `hamming_dist_top_k` | 汉明距离 TopK（hash 路由相关） |

### 7.1 moe_gating_top_k_hash

Hash 路由专用的 TopK 算子，直接从 `tid2eid` 映射表获取专家索引。

- 支持 arch35（A5 设备）优化
- 支持分组模式 `without_group` / `generalized`
- 支持 `regbase` 寄存器基址优化

### 7.2 moe_init_routing_custom

路由初始化算子，负责：
- 专家 token 计数
- 排序（归并排序 `mrgsort`）
- 行索引 gather
- 静态/动态量化支持
- 单核心 / 多核心性能优化版本

### 7.3 moe_grouped_matmul

分组矩阵乘算子，高效执行多个专家的 FFN 计算：
- `weight_nz` 权重非零结构支持
- L0 缓存优化
- CPU fallback 支持
