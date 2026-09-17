# DeepSeek V4 MoE 前馈网络详解

> **本文档详细讲解 DeepSeek V4 的 MoE（Mixture of Experts）前馈网络实现，包括路由机制、专家网络、共享专家、EPLB 负载均衡以及 W4A8 量化在 MoE 中的落地。**

> **代码版本冻结声明**：本文基于 **vllm-ascend v0.26.0rc** 源码整理，模型主文件以 [deepseek_v4.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py)（共 1565 行）为准，文中 `Lxxx` 均指该文件行号；融合 MoE 执行器位于 [ops/fused_moe/fused_moe.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py)，专家选择逻辑位于 [ops/fused_moe/experts_selector.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py)。

> **统一模型配置**（取自 `dsv4_docs/0_DeepSeek-V4-Pro-0813-w4a8_config.json`，DeepSeek-V4-Pro-0813 W4A8 量化版，本文 MoE 相关行）：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `n_routed_experts` | **384** | 路由专家总数 |
| `num_experts_per_tok` | **6** | 每个 token 激活的路由专家数（Top-K=6） |
| `n_shared_experts` | **1** | 共享专家数（所有 token 全量经过） |
| `moe_intermediate_size` | **3072** | 单个专家 FFN 的中间维度 |
| `num_hash_layers` | **3** | 前 3 层（layer 0/1/2）使用 hash 路由 |
| `routed_scaling_factor` | **2.5** | 路由输出缩放因子（代码 `getattr` 默认 1.5，本配置实际为 2.5） |
| `scoring_func` | **`sqrtsoftplus`** | 路由打分函数：`sqrt(softplus(logits))` |
| `topk_method` | **`noaux_tc`** | 无辅助损失的偏置修正 TopK（`e_score_correction_bias`） |
| `norm_topk_prob` | **true** | 选中后对 Top-K 权重重新归一化 |
| `swiglu_limit` | **10.0** | SiLU 门控激活的 clamp 上限 |
| `expert_dtype` | **`fp4`** | 专家 w1/w2/w3 为 W4A8 动态量化（fp4） |

---

## 一、MoE 架构总览

### 1.1 设计理念

DeepSeek V4 的 FFN 层**全部**采用 MoE 架构——61 个主模型层 + 3 个 draft 层（MTP/DSpark 复用同一层结构）没有任何稠密 FFN 层。通过稀疏激活，每个 token 只走 384 个专家中的 6 个，在总参数量巨大的前提下把单次推理的实际计算量压到很低。

> **生活化类比**：MoE 像一家有 384 位专科医生的超大医院。病人（token）进门后先由分诊台（gate 路由门控）根据病情打 384 个匹配分，只挑出最匹配的 6 位医生（Top-K=6 路由专家）会诊；同时不管什么病都要先过一位全科医生（1 个共享专家）。最终诊断 = 全科医生意见 + 6 位专科医生加权意见。

```
输入 hidden_states (num_tokens, 7168)
    │
    ├─ 路由门控 gate: 计算每个 token 对 384 个专家的得分
    │     ├─ 前 3 层: hash 路由（查 tid2eid 表）
    │     └─ 其余层: 线性路由（fp32 F.linear + sqrtsoftplus + 偏置修正）
    ├─ Top-K 选择: 每 token 选 6 个专家并归一化权重
    └─ FusedMoE 专家计算
         ├─ routed_experts: 384 选 6，稀疏激活（W4A8 fp4）
         └─ shared_experts: 1 个稠密 MLP，全量激活
    │
    └─ 输出融合: shared + routed × factor(2.5)（按 dtype 分两条路径）
```

### 1.2 核心类关系

**文件位置**: `DeepseekV4MoE` 定义于 [deepseek_v4.py:L357](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L357)

```
DeepseekV4MoE (MoE 主类, L357)
├── gate: ReplicatedLinear(7168 → 384)        — 路由门控（每个 EP rank 完整复制一份）
│   ├── weight (bf16) + weight_fp32 (fp32 预转换)
│   ├── tid2eid: Parameter (129280, 6) int32   — 仅前 3 层：token ID → expert ID 查表
│   └── e_score_correction_bias: (384,) fp32   — 仅线性路由层：noaux_tc 得分修正偏置
├── experts: AscendFusedMoE                    — 融合专家网络执行器 (fused_moe.py)
│   ├── 384 个路由专家 FFN（w1/w2/w3，W4A8 fp4）
│   ├── 内部门控/选择/分发/GMM/combine 全流程
│   └── is_internal_router 属性：门控是否在 FusedMoE 内部执行
└── shared_experts: DeepseekV2MLP | None      — 共享专家（mix_placement 融合时为 None）
```

### 1.3 共享专家的本体：DeepseekV2MLP

**文件位置**: [deepseek_v4.py:L308](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L308)

> **v0.26 与早期版本的结构差异**：`DeepseekV2MLP` **不是**"非 MoE 层的稠密 FFN"——V4 不存在稠密 FFN 层。它在本版本中唯一的角色是充当 `shared_experts`（共享专家）的网络本体，在 `DeepseekV4MoE.__init__` 的 L410 处被实例化。

```
hidden (N, 7168)
   │ gate_up_proj: MergedColumnParallelLinear 7168 → 2×3072
   ▼
(N, 6144)  ── SiluAndMulWithClamp(limit=10.0) ──▶ (N, 3072)
   │            （silu(gate) * up，逐元素后 clamp 到 [-10, 10]）
   ▼ down_proj: RowParallelLinear 3072 → 7168（默认 reduce_results=False）
(N, 7168)
```

| 组件 | 类型 | 说明 |
|------|------|------|
| `gate_up_proj` | `MergedColumnParallelLinear` | w1(gate)/w3(up) 合并为一次列并行 GEMM，输出两份 `intermediate_size` |
| `act_fn` | `SiluAndMulWithClamp` / `SiluAndMul` | 配置了 `swiglu_limit=10.0` 时用带 clamp 版本（L346），否则普通版（L348）。**限幅是为了配合 W4A8：激活过大时量化到 int8 会溢出** |
| `down_proj` | `RowParallelLinear` | w2，行并行；共享专家场景 `reduce_results=False`（L417），归约交给后续融合逻辑统一处理 |

当 `n_shared_experts > 1` 时，`intermediate_size = moe_intermediate_size * n_shared_experts`（L408），即把多个共享专家横向拼成一个大 MLP。本配置为 1 个，即 3072。

---

## 二、DeepseekV4MoE 成员详解

### 2.1 核心配置参数（L367-L464）

| 参数 | 来源 / 本配置值 | 说明 |
|------|----------------|------|
| `layer_idx` | `prefix.split(".")[-2]` | 当前层号，决定本层是否 hash 层 |
| `routed_scaling_factor` | `getattr(config, ..., 1.5)` → **2.5** | 代码兜底默认 1.5；V4 配置显式给 2.5 |
| `swiglu_limit` | `getattr(config, ..., None)` → **10.0** | 透传给共享专家与 FusedMoE 的 SiLU 限幅 |
| `tp_size` / `tp_rank` | TP 组 | 注意力相关权重按 TP 切；MoE 内部还涉及 EP |
| `ep_group` / `ep_rank` / `ep_size` | `get_ep_group()` | 专家并行组：384 个专家按 EP rank 分布 |
| `n_routed_experts` / `n_shared_experts` | 384 / 1 | 路由专家数 / 共享专家数 |
| `is_sequence_parallel` | `parallel_config.use_sequence_parallel_moe` | 是否对 token 序列做 SP 分片 |
| `hash` | `layer_idx < num_hash_layers(3) and not is_draft_layer` | 本层是否走 hash 路由（L421） |

### 2.2 EPLB 负载均衡配置（L391-L400）

| 参数 | 说明 |
|------|------|
| `enable_eplb` | 是否启用 Expert Parallel Load Balancing |
| `n_redundant_experts` | 冗余专家数（`eplb_config.num_redundant_experts`） |
| `n_logical_experts` | 逻辑专家数 = `n_routed_experts` = 384 |
| `n_physical_experts` | 物理专家数 = 逻辑 + 冗余 |
| `n_local_physical_experts` | 每个 EP rank 持有的物理专家数 = 物理总数 // ep_size |
| `physical_expert_start/end` | 本 rank 负责的物理专家下标区间 |

### 2.3 门控层 gate（L385-L388）

```python
self.gate = ReplicatedLinear(          # Replicated：门控在每个 rank 上各存一份完整权重
    config.hidden_size,                # 7168
    config.n_routed_experts,           # 384
    bias=False, quant_config=None,     # 门控不量化
    prefix=f"{prefix}.gate",
)
self.gate.precast_fp32_weight = True   # L388：加载后把权重预算成 fp32（weight_fp32）
```

| 属性 | 形状 / dtype | 存在条件 | 说明 |
|------|-------------|---------|------|
| `gate.weight` | `(384, 7168)` | 始终 | bf16 原始权重 |
| `gate.weight_fp32` | `(384, 7168)` fp32 | `precast_fp32_weight=True` 时加载后生成 | 供 fp32 路由打分使用，避免运行时重复 cast |
| `gate.tid2eid` | `(vocab_size=129280, top_k=6)` int32 | hash 层（L425） | 静态查表：每个 token ID 预先指定 6 个专家；用 zeros 初始化是为了 dummy 加载模式不踩非法内存（L423-L432） |
| `gate.e_score_correction_bias` | `(384,)` fp32 | 线性路由层（L436） | **noaux_tc 的核心**：每个专家一个可学习偏置，推理时加在打分上，代替辅助损失实现负载均衡 |

二者互斥：hash 层 `tid2eid` 有值、bias 为 None；其余层反之（L421-L436）。

> **为什么门控要 fp32？** 路由打分决定"这个 token 归谁算"，一旦选专家选错，后面的大 GEMM 全白做。softplus/sqrt 这类函数在低精度下数值误差可能改变 Top-K 边界，所以 V4 的路由输入和门控权重都强制走 fp32。

### 2.4 共享专家 shared_experts（L405-L419）

- **普通模式（本配置）**：`shared_experts = DeepseekV2MLP(intermediate_size=3072, reduce_results=False)`，与路由专家在 FusedMoE 内并行执行，最后融合。
- **mix_placement 模式**：`get_ascend_config().mix_placement=True` 时 `shared_experts = None`（L404-L406），共享专家被当作额外 `n_shared_experts` 个专家塞进 FusedMoE 的专家列表统一调度（构造参数 `n_shared_experts` 在 L461 传入），适合把共享专家打散到 EP 各 rank 上做负载混部。

### 2.5 FusedMoE 构造参数（L438-L464）

```python
self.experts = FusedMoE(
    shared_experts=self.shared_experts,
    gate=self.gate,
    num_experts=384,
    top_k=6,                              # config.num_experts_per_tok
    hidden_size=7168,
    intermediate_size=3072,
    renormalize=config.norm_topk_prob,    # true：Top-K 后权重归一化
    use_grouped_topk=True,
    num_expert_group=getattr(config, "n_group", 1),     # 本配置无 n_group → 1
    topk_group=getattr(config, "topk_group", 1),        # 本配置无 topk_group → 1
    scoring_func="sqrtsoftplus",          # getattr 默认 "softmax"
    routed_scaling_factor=2.5,            # 注意：归一化之后、在路由输出侧缩放
    swiglu_limit=10.0,
    e_score_correction_bias=self.gate.e_score_correction_bias,
    enable_eplb=self.enable_eplb,
    num_redundant_experts=self.n_redundant_experts,
    is_sequence_parallel=self.is_sequence_parallel,
    n_shared_experts=...,                 # 仅 mix_placement 时非 0
    hash=self.hash,
    tid2eid=self.gate.tid2eid,
)
```

| 易错点 | 说明 |
|--------|------|
| `num_expert_group` / `topk_group` | V4 配置里**没有** `n_group`、`topk_group` 字段，`getattr` 兜底为 1。即分组 TopK 机制在代码里保留，但 V4 实际退化为"单组"（384 个专家直接选 Top-6） |
| `scoring_func` | 代码 `getattr` 默认 `"softmax"`（兼容 DeepSeekV2/V3），V4 配置为 **`sqrtsoftplus`** |
| `routed_scaling_factor` | 注释（L452-L454）明确：缩放刻意保留在路由归一化**之外**，顺序为"先归一化 Top-K 权重 → 再缩放路由输出"，以匹配 DeepSeek V4 官方行为；ROCm AITER 路径才在算子内部缩放 |
| `gate` 被传入 FusedMoE | 这是"内部路由"能力的前提（见 §3.3） |

---

## 三、双路由模式 + 内部路由

### 3.1 Hash 路由（layer 0/1/2）

**触发条件**：`layer_idx < num_hash_layers(3) and not is_draft_layer`（L421）

最底层 3 个层的专家分配是**训练前就固定好的**：checkpoint 直接给出 `tid2eid` 表 `(129280, 6)`，推理时按 token 的词表 ID 查出 6 个专家下标，不做任何学习型打分。

```
input_ids (num_tokens,)
    │ （ALLGATHER / pad_and_split 先把各 DP/TP 分片的 input_ids 整理对齐）
    ▼
tid2eid[input_ids]  →  topk_ids (num_tokens, 6)
```

特点：零学习参数、零浮点打分、查表即得，底层语义稳定；draft 层即使层号 < 3 也不启用 hash（`is_draft_layer` 排除）。

### 3.2 线性路由 + sqrtsoftplus（layer 3~60）

中高层每个 token 的专家分配由数据决定：

```
hidden_states_fp32 (N, 7168)          ← 来自 rms_norm_cast 的 fp32 输出（见文档 0/3）
    │ F.linear(., gate.weight_fp32)   ← fp32 门控，权重也是 fp32
    ▼
router_logits (N, 384)
    │ score = sqrt(softplus(logits))  ← scoring_func="sqrtsoftplus"
    │ + e_score_correction_bias       ← noaux_tc：逐专家偏置修正
    ▼
Top-K=6 选择 + Top-K 权重归一化(norm_topk_prob)
    │
    ▼
topk_weights (N, 6), topk_ids (N, 6)
```

- `sqrt(softplus(x))` 相比 softmax：不强制 384 个得分归一，保留专家选择的绝对置信度，多个专家可以同时"高权重"。
- `e_score_correction_bias` 是 **noaux-tc（no auxiliary loss, top-choice correction）**策略：训练时通过偏置项把 token 均匀推向冷门专家，推理时只做一次加法，不需要任何辅助损失前向。
- 选专家的融合算子实现见 [experts_selector.py:L230-L304](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py#L230)：`sqrtsoftplus` 统一走 `moe_gating_top_k_hash` 自定义算子（hash 层带 `tid2eid/input_ids`，线性层传 None）；softmax/sigmoid 才走 `moe_gating_top_k`（`norm_type=0/1`）。

### 3.3 内部路由 vs 外部路由（`is_internal_router`）

[fused_moe.py:L592-L595](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py#L592) 定义：

```python
@property
def is_internal_router(self) -> bool:
    gate = self.gate
    return gate is not None and hasattr(gate, "weight_fp32")   # 预转换 fp32 权重存在
```

V4 的 gate 设置了 `precast_fp32_weight=True`，加载后挂有 `weight_fp32`，因此 **V4 恒为内部路由**。两条路径的区别（MoE forward L486-L494）：

```python
if self.experts.is_internal_router:
    # 把"原始 hidden"直接当 router_logits 传进去，门控 F.linear 在 FusedMoE 内部做
    # hidden_states_fp32 是 rms_norm_cast 产出的 fp32 路由输入，没有则用 bf16 hidden
    router_input = hidden_states if hidden_states_fp32 is None else hidden_states_fp32
    fused_moe_out = self.experts(hidden_states=hidden_states, router_logits=router_input)
else:
    # 外部路径：MoE 自己先算好真正的 router_logits 再传入
    router_input = hidden_states.float() if hidden_states_fp32 is None else hidden_states_fp32
    router_logits = F.linear(router_input, self.gate.weight)
    fused_moe_out = self.experts(hidden_states=hidden_states, router_logits=router_logits)
```

内部路径在 [fused_moe.py:L974-L984 / L1033-L1039](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py#L974) 执行 `F.linear(hidden_states_fp32, gate.weight_fp32)`。把打分放进 FusedMoE 的好处是：门控、选专家、dispatch 可以在同一套 stream/event 编排里紧挨着重排，减少 HBM 物化和同步。判断技巧是 `router_logits.shape[-1] == hidden_size(7168)`——说明传进来的还是 hidden 而非 384 维打分。

---

## 四、Forward 完整流程

**文件位置**: `DeepseekV4MoE.forward` [L466-L527](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L466)

```
输入: hidden_states (N, 7168), hidden_states_fp32 (N, 7168) | None, input_ids
   │
   ▼
1. 序列并行分片（is_sequence_parallel 时，L481-L484）
     hidden_states / hidden_states_fp32 各按 TP rank chunk，
     每个 rank 只保留 1/world_size 的 token，避免各 rank 重复算专家
   │
   ▼
2. 路由（L486-L494）
     内部路由: experts(hidden, router_input=hidden|fp32)
     外部路由: F.linear(fp32_input, gate.weight) → experts(hidden, logits)
   │
   ▼
3. FusedMoE 执行（fused_moe.py：选专家→dispatch→w1/w3 GEMM→
   swiglu_group_quant 限幅激活+量化→w2 GEMM→combine）
     返回 tuple(shared_output, routed_output) 或单个 tensor
   │
   ▼
4. 输出融合（L496-L515，按 dtype 分路径，详见 §4.3）
   │
   ▼
5. 聚合（L519-L525）
     SP: all_gather 后裁掉 padding 回 num_tokens
     否则 TP>1 且为 tuple 旧路径: maybe_all_reduce_tensor_model_parallel
   │
   ▼
输出: (num_tokens, 7168)
```

### 4.1 序列并行（Sequence Parallel MoE）

| 阶段 | 操作 | 说明 |
|------|------|------|
| 输入 | `sequence_parallel_chunk` | token 维按 TP rank 分片，`hidden_states_fp32` 同步分片（L482-L484） |
| 输出 | `tensor_model_parallel_all_gather` + `[:num_tokens]`（L520-L521） | 拼回完整序列并裁掉对齐 padding |

目的：注意力层末尾的 all-reduce 若改/已被 reduce-scatter 替代，MoE 输入天然是分片的，直接各算各的 token，省一次重复计算。

### 4.2 FusedMoE 的输出形态

- **tuple 形态**：`(shared_output, routed_output)`——共享专家在 FusedMoE 旁路并行执行时返回，缩放/相加由 `DeepseekV4MoE.forward` 在外层做（V4 普通模式走这里）。
- **单 tensor 形态**：mix_placement 融合或上游 MoERunner 新路径下，内部已完成归约与融合，外层直接使用。

### 4.3 输出融合：两种 dtype 路径并**不等价**

融合算子 `muls_add_triton(x, y, scale)` 的语义是 `out = x * scale + y`。外层代码（L502-L515）：

```python
if hidden_states.dtype != torch.float16:
    if not self.is_rocm_aiter_moe_enabled:           # NPU 路径必进
        if self.shared_experts is not None:
            final_hidden_states = muls_add_triton(
                final_hidden_states,                 # x = routed
                shared_output,                       # y = shared
                self.routed_scaling_factor,          # scale = 2.5
            )                                         # out = routed * 2.5 + shared
        else:
            final_hidden_states *= self.routed_scaling_factor
elif self.shared_experts is not None:                # fp16 路径
    final_hidden_states = muls_add_triton(
        shared_output,                               # x = shared
        final_hidden_states,                         # y = routed
        1.0 / self.routed_scaling_factor,            # scale = 1/2.5 = 0.4
    )                                                 # out = shared * 0.4 + routed
```

| 路径 | 公式 |
|------|------|
| bf16（本配置） | `out = routed × 2.5 + shared` |
| fp16 | `out = shared × 0.4 + routed` |
| 无共享专家 | `out = routed × 2.5` |

> **重要：两条公式在数学上并不等价**（fp16 路径整体相对 bf16 路径差一个 1/2.5 的量级关系，不能写成"等价变形"）。fp16 动态范围小，先把共享输出缩小再相加，是为了避免中间值溢出/精度饱和而选择的不同融合次序；这是框架按 dtype 刻意保留的两条实现。ROCm AITER 路径在非 fp16 分支里会跳过外层缩放（`is_rocm_aiter_moe_enabled`，L503），因为 AITER 在算子内部已经乘过 factor。

---

## 五、EPLB 专家并行负载均衡

```
384 个逻辑专家
   │ + n_redundant_experts 个冗余副本
   ▼
n_physical_experts 个物理专家
   │ 按 EP rank 均分（n_local_physical_experts = 总数 // ep_size）
   ▼
rank 0 持有 [0, k)，rank 1 持有 [k, 2k)，……
```

冗余专家是热门专家的复制份：当某 token 路由到的逻辑专家在"离得近/更空闲"的 rank 上有冗余副本时，优先分发过去，减少跨 rank all-to-all 的量与长尾。EPLB 重排由 `parallel_config.eplb_config` 驱动，`enable_eplb` 与冗余数透传进 FusedMoE（L458-L459）。

---

## 六、路由选择细节：noaux_tc 与分组 TopK

### 6.1 V4 的实际配置：单组 + noaux_tc

代码保留了 DeepSeekV3 时代的**分组 TopK**接口（`use_grouped_topk=True`、`num_expert_group`、`topk_group`，融合算子参数里对应 `group_count`、`k_group`、`group_select_mode=1`），但 V4 W4A8 配置中**不存在** `n_group/topk_group`，两者兜底为 1，所以实际行为是：

```
384 个专家视为 1 个大组 → sqrtsoftplus 打分并加修正偏置 → 直接选全局 Top-6
```

`topk_method="noaux_tc"` 不引入新的选择算子，它体现在 `e_score_correction_bias` 这个可学习逐专家偏置上：训练阶段用它替代显式辅助损失来均衡专家负载，推理阶段零额外成本。

### 6.2 三种打分函数（experts_selector.py）

| `scoring_func` | 公式 | norm_type | 算子路径 |
|----------------|------|-----------|---------|
| `softmax` | `softmax(logits)`（全专家归一） | 0 | `moe_gating_top_k` |
| `sigmoid` | `sigmoid(logits)`（独立多热） | 1 | `moe_gating_top_k` |
| **`sqrtsoftplus`（V4）** | **`sqrt(softplus(logits))`**（参考实现 L350-L351） | — | `moe_gating_top_k_hash`（hash/非hash 共用） |

`renormalize`（V4 `norm_topk_prob=true`）控制选中的 6 个权重是否重新归一化。注意 `sqrtsoftplus` 分支调用融合算子时**固定传 `renorm=0`**——experts_selector.py L280-L285 的注释说明 hash 定制算子当前拒绝 `renorm != 0`；softmax/sigmoid 分支则透传 `renorm`（`norm_type=0/1`）。纯 Python 参考路径（`_native_select_experts`，L385）会显式调用 `_renormalize_topk_weights`。

---

## 七、NPU MoE 相关定制算子

`csrc/moe/` 目录在 v0.26.0rc 中实际编译的算子子目录如下（仅列与 DeepSeek V4 MoE/路由相关者；`hc_pre*`、`hc_post` 见文档 1/2，`chunk_gated_delta_rule*`、`causal_conv1d*` 属于 gated-delta-net 等其他模型）：

| 算子目录 | 在 V4 MoE 中的作用 |
|---------|-------------------|
| `moe_gating_top_k` | 路由打分 + 分组 TopK 选择（softmax/sigmoid 路径），含 `generalized`/`without_group`/`e_k_fullload` 多套 kernel 与 arch35(A5) 优化 |
| `moe_gating_top_k_hash` | V4 `sqrtsoftplus` 路径专用：hash 层带 `tid2eid + input_ids` 查表，线性层两者传 None；支持 `regbase` 寄存器基址版本 |
| `moe_grouped_matmul` | 分组矩阵乘（GroupedMatmul）：一次性完成所有"专家 × token 分片"的 w1/w3、w2 GEMM；含 `weight_nz`（NZ 权值排布）、L0 缓存接口、CPU fallback（`moe_grouped_matmul_cpu.cpp`） |
| `swiglu_group_quant` | **W4A8 关键融合算子**：按专家分组做 SiLU 门控乘 + clamp（`swiglu_limit=10`）+ 输出动态 per-token 量化为 int8，供 w2 的 A8 GEMM 直接消费 |
| `situ_mx_quant` / `dequant_situ_quant` | 原位（in-situ）MX 风格缩放量化/反量化（按场景用于低秩或辅助量化路径） |
| `dequant_swiglu_quant` | swiglu 量化算子的反量化/变体版本（静态/动态 per-token，bf16/int32 bias） |
| `scatter_nd_update_sk` | combine 阶段把各专家分片结果按 token 散射累加回输出张量（arch22 下有 linear_index/no_sort/large_index 多种实现） |
| `rms_norm_cast` / `add_rms_norm_bias` | 产出 fp32 路由输入的融合 RMSNorm（`rms_norm_cast`）及带 bias 的归约融合（配合内部路由，见文档 3） |
| `transpose_kv_cache_by_block` | 按 block 转置 KV cache（属 KV 路径，见文档 5） |

> 早期文档中出现的 `moe_init_routing_custom`、`hamming_dist_top_k`、`scatter_nd_update_v2` 在 v0.26.0rc 的 `csrc/moe/` 下已不存在（散射算子已更名为 `scatter_nd_update_sk`，路由初始化功能并入 `moe_gating_top_k*` 与 fused_moe Python 层）。

---

## 八、W4A8 量化在 MoE 中的落地（速览）

本 checkpoint 为 W4A8 动态量化版，MoE 相关 dtype 分工（完整逐张量清单见同目录 `0_DeepSeek-V4-Pro-0813-w4a8_quant_model_description.json`）：

| 模块 | 量化方案 | 说明 |
|------|---------|------|
| 路由专家 w1(gate)/w3(up)/w2(down) | **W4A8_DYNAMIC，权重 fp4**（`expert_dtype="fp4"`） | 权重 4-bit 存储，激活 per-token 动态量化到 8-bit；fp4 的 E8M0 scale 以 uint8 view 加载 |
| `swiglu_group_quant` 算子 | A8 输出 | w1/w3 GEMM 后在同一算子里完成 SiLU×gate、clamp(10.0) 和 int8 量化，直接喂 w2 GEMM |
| gate 门控 / `e_score_correction_bias` | FLOAT（不量化） | 门控权重另存 `weight_fp32`，打分全程 fp32 |
| 共享专家 | 随量化配置（gate_up/down 走对应量化线性层） | 本体仍是 `DeepseekV2MLP` |
| 权重 scale 后缀 | fp4 → `.weight_scale`；其他 → `.weight_scale_inv` | 权重加载时的命名映射见文档 6/8 与模型 `load_weights` |

量化的加载细节（fp4 张量 reshape、E8M0 处理、scale 重命名）在各模型文件的 `load_weights` 中完成，本文不再展开。

---

## 九、一句话总结

DeepSeek V4 的每一层 FFN 都是"**1 个全量共享 MLP + 384 选 6 稀疏专家**"的 MoE：前 3 层按词表 hash 静态分派，其余层用 **fp32 门控 + sqrtsoftplus + noaux_tc 偏置**选专家；门控在 V4 上收进 FusedMoE 内部执行（`is_internal_router`），专家 GEMM 走 W4A8 fp4 的分组矩阵乘与 `swiglu_group_quant` 融合算子；路由输出先归一化、再在外部按 `routed_scaling_factor=2.5` 与共享输出融合（bf16 路径 `routed×2.5+shared`，fp16 路径为另一条非等价公式）；EPLB 用冗余专家在 EP 维度均衡负载。
