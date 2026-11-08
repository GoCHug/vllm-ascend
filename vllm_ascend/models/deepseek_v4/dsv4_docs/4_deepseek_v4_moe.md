# DeepSeek V4 MoE 前馈网络详解

> **本文档详细讲解 DeepSeek V4 的 MoE（Mixture of Experts）前馈网络实现，包括路由机制、专家网络、共享专家、EPLB 负载均衡以及 W4A8 量化在 MoE 中的落地。**

> **代码版本冻结声明**：本文基于 **vllm-ascend v0.32.0**（分支 `dsv4-v0.32.0`，对齐 vLLM v0.32.0）源码整理。模型主文件以 [deepseek_v4/model.py](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py)（共 1425 行）为准，文中 `Lxxx` 均指该文件行号；融合 MoE 执行器位于 [ops/fused_moe/fused_moe.py](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py)，路由选择逻辑位于 [ops/fused_moe/router/](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/router/router_factory.py)：[router_factory.py](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/router/router_factory.py)、[fused_topk_router.py](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/router/fused_topk_router.py)、[grouped_topk_router.py](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/router/grouped_topk_router.py)。

> **统一模型配置**（取自 `dsv4_docs/0_DeepSeek-V4-Pro-0813-w4a8_config.json`，DeepSeek-V4-Pro-0813 W4A8 量化版，本文 MoE 相关行）：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `n_routed_experts` | **384** | 路由专家总数 |
| `num_experts_per_tok` | **6** | 每个 token 激活的路由专家数（Top-K=6） |
| `n_shared_experts` | **1** | 共享专家数（所有 token 全量经过） |
| `moe_intermediate_size` | **3072** | 单个专家 FFN 的中间维度 |
| `num_hash_layers` | **3** | 前 3 层（layer 0/1/2）使用 hash 路由（v0.32.0 若配置带 `mlp_layer_types` 则以逐层类型为准） |
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
    │     ├─ hash 层（layer 0/1/2）: 查 tid2eid 表（vision 版本对图像行改用 bias_vl 重排）
    │     └─ 其余层: 线性路由（fp32 F.linear + sqrtsoftplus + 偏置修正）
    ├─ Top-K 选择: 每 token 选 6 个专家并归一化权重（router/ 下 AscendFusedTopKRouter）
    └─ FusedMoE 专家计算（AscendMoERunner）
         ├─ routed_experts: 384 选 6，稀疏激活（W4A8 fp4）
         └─ shared_experts: 1 个稠密 MLP，全量激活
    │
    └─ 输出融合: shared + routed（routed 权重已在路由侧乘 factor=2.5）
```

### 1.2 核心类关系

**文件位置**: `DeepseekV4MoE` 定义于 [model.py:L277](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py#L277)，`forward` 在 [model.py:L404](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py#L404)

```
DeepseekV4MoE (MoE 主类, L277)
├── gate: ReplicatedLinear(7168 → 384)        — 路由门控（每个 EP rank 完整复制一份，L305-L308）
│   ├── weight (bf16) + weight_fp32 (fp32 预转换)
│   ├── tid2eid: Parameter (129280, 6) int32   — 仅 hash 层：token ID → expert ID 查表（L362-L372）
│   ├── e_score_correction_bias: (384,) fp32   — 仅线性路由层：noaux_tc 得分修正偏置（L376）
│   └── bias_vl: Parameter (384,) fp32         — 仅 vision checkpoint（vision_n_layers>0，L353-L361）
├── experts = FusedMoEFactory(...)             — 返回 MoERunner（L378-L402）
│   └── AscendMoERunner (ops/fused_moe/fused_moe.py L85)
│       ├── router: AscendFusedTopKRouter      — 路由选择（router/fused_topk_router.py L80）
│       ├── routed_experts: AscendRoutedExperts — 384 个路由专家 FFN（routed_experts.py L390）
│       └── ascend_shared_experts: AscendSharedExperts（shared_experts.py L99）
└── shared_experts: DeepseekV2MLP | None       — 共享专家（mix_placement 融合时为 None，L322-L339）
```

`FusedMoEFactory` 在 v0.32.0 中已改为**工厂函数**（[vllm/model_executor/layers/fused_moe/layer.py:L88](../../../../../vllm/vllm/model_executor/layers/fused_moe/layer.py#L88)），返回 `MoERunner`；NPU 侧由 [patch/platform/patch_fused_moe.py:L110](../../../../../vllm-ascend/vllm_ascend/patch/platform/patch_fused_moe.py#L110) 的 `_ascend_FusedMoE` 接管（替换点在 L208-L209），因此实际执行体是 `AscendMoERunner`。

### 1.3 共享专家的本体：DeepseekV2MLP

**文件位置**: [model.py:L228](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py#L228)

> **与早期版本的结构说明**：`DeepseekV2MLP` **不是**"非 MoE 层的稠密 FFN"——V4 不存在稠密 FFN 层。它在当前版本中唯一的角色是充当 `shared_experts`（共享专家）的网络本体，在 `DeepseekV4MoE.__init__` 的 L330 处被实例化。

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
| `gate_up_proj` | `MergedColumnParallelLinear` | w1(gate)/w3(up) 合并为一次列并行 GEMM，输出两份 `intermediate_size`（L246-L253） |
| `act_fn` | `SiluAndMulWithClamp` / `SiluAndMul` | 配置了 `swiglu_limit=10.0` 时用带 clamp 版本（L265-L266），否则普通版（L267-L268）。**限幅是为了配合 W4A8：激活过大时量化到 int8 会溢出** |
| `down_proj` | `RowParallelLinear` | w2，行并行；共享专家场景 `reduce_results=False`（L337），归约交给后续融合逻辑统一处理 |

当 `n_shared_experts > 1` 时，`intermediate_size = moe_intermediate_size * n_shared_experts`（L328），即把多个共享专家横向拼成一个大 MLP。本配置为 1 个，即 3072。`forward` 见 L270-L274。

---

## 二、DeepseekV4MoE 成员详解

### 2.1 核心配置参数（L286-L402）

| 参数 | 来源 / 本配置值 | 说明 |
|------|----------------|------|
| `layer_idx` | `prefix.split(".")[-2]`（L289） | 当前层号，决定本层是否 hash 层 |
| `routed_scaling_factor` | `getattr(config, ..., 1.5)` → **2.5**（L291） | 代码兜底默认 1.5；V4 配置显式给 2.5 |
| `swiglu_limit` | `getattr(config, ..., None)` → **10.0**（L292） | 透传给共享专家与 FusedMoEFactory 的 SiLU 限幅 |
| `tp_size` / `tp_rank` | TP 组（L287-L288） | 注意力相关权重按 TP 切；MoE 内部还涉及 EP |
| `ep_group` / `ep_rank` / `ep_size` | `get_ep_group()`（L294-L296） | 专家并行组：384 个专家按 EP rank 分布 |
| `n_routed_experts` / `n_shared_experts` | 384 / 1（L297-L298） | 路由专家数 / 共享专家数 |
| `is_sequence_parallel` | `parallel_config.use_sequence_parallel_moe`（L300） | 是否对 token 序列做 SP 分片（分片/聚合在 prepare/finalize 内完成） |
| `hash` | `mlp_layer_types[layer_idx] == "hash_moe"`，无该字段时回退 `layer_idx < num_hash_layers(3)`（L345-L352） | 本层是否走 hash 路由；draft 层恒为 False |

### 2.2 EPLB 负载均衡配置（L310-L320）

| 参数 | 说明 |
|------|------|
| `enable_eplb` | 是否启用 Expert Parallel Load Balancing（`parallel_config.enable_eplb`，L312） |
| `n_redundant_experts` | 冗余专家数（`eplb_config.num_redundant_experts`，L314） |
| `n_logical_experts` | 逻辑专家数 = `n_routed_experts` = 384（L315） |
| `n_physical_experts` | 物理专家数 = 逻辑 + 冗余（L316） |
| `n_local_physical_experts` | 每个 EP rank 持有的物理专家数 = 物理总数 // ep_size（L317） |
| `physical_expert_start/end` | 本 rank 负责的物理专家下标区间（L319-L320） |

### 2.3 门控层 gate（L305-L308）

```python
self.gate = ReplicatedLinear(          # Replicated：门控在每个 rank 上各存一份完整权重
    config.hidden_size,                # 7168
    config.n_routed_experts,           # 384
    bias=False, quant_config=None,     # 门控不量化
    prefix=f"{prefix}.gate",
)
self.gate.precast_fp32_weight = True   # L308：加载后把权重预算成 fp32（weight_fp32）
```

| 属性 | 形状 / dtype | 存在条件 | 说明 |
|------|-------------|---------|------|
| `gate.weight` | `(384, 7168)` | 始终 | bf16 原始权重 |
| `gate.weight_fp32` | `(384, 7168)` fp32 | `precast_fp32_weight=True` 时加载后生成 | 供 fp32 路由打分使用，避免运行时重复 cast |
| `gate.tid2eid` | `(vocab_size=129280, top_k=6)` int32 | hash 层（L362-L372） | 静态查表：每个 token ID 预先指定 6 个专家。v0.32.0 构造时先填 `(token_ids + expert_offsets) % n_routed_experts` 的轮转占位表（保证 dummy profile 时同一 token 不重复命中同一专家，避免 MC2 dispatch/combine 失败），再被 checkpoint 的真表覆盖 |
| `gate.e_score_correction_bias` | `(384,)` fp32 | 线性路由层（L376） | **noaux_tc 的核心**：每个专家一个可学习偏置，推理时加在打分上，代替辅助损失实现负载均衡 |
| `gate.bias_vl` | `(384,)` fp32 | `config.vision_n_layers > 0`（L353-L361） | 视觉路由偏置：图像 sentinel token 用它代替文本偏置参与 TopK |

互斥关系：hash 层 `tid2eid` 有值、`e_score_correction_bias=None`；其余层反之（L362-L376）。加载时 hash 层若 checkpoint 仍带 `.gate.bias` 会被跳过（`load_weights` L1262-L1266）。

> **为什么门控要 fp32？** 路由打分决定"这个 token 归谁算"，一旦选专家选错，后面的大 GEMM 全白做。softplus/sqrt 这类函数在低精度下数值误差可能改变 Top-K 边界，所以 V4 的路由输入和门控权重都强制走 fp32。

### 2.4 共享专家 shared_experts（L322-L339）

- **普通模式（本配置）**：`shared_experts = DeepseekV2MLP(intermediate_size=3072, reduce_results=False, is_sequence_parallel=...)`，与路由专家在 `AscendMoERunner` 内并行执行，最后融合。
- **mix_placement 模式**：`get_ascend_config().mix_placement=True` 时 `shared_experts = None`（L325-L326），共享专家被当作额外 `n_shared_experts` 个专家塞进专家列表统一调度（构造参数 `n_shared_experts` 在 L400 传入），适合把共享专家打散到 EP 各 rank 上做负载混部。

### 2.5 FusedMoEFactory 构造参数（L378-L402）

```python
self.experts = FusedMoEFactory(
    shared_experts=self.shared_experts,
    gate=self.gate,
    num_experts=384,
    top_k=6,                              # config.num_experts_per_tok
    hidden_size=7168,
    intermediate_size=3072,
    renormalize=config.norm_topk_prob,    # true：Top-K 后权重归一化
    quant_config=quant_config,
    prefix=f"{prefix}.experts",
    scoring_func="sqrtsoftplus",          # getattr 默认 "softmax"
    routed_scaling_factor=2.5,            # 注意：保留在路由路径内，缩放 topk_weights
    swiglu_limit=10.0,
    e_score_correction_bias=self.gate.e_score_correction_bias,
    bias_vl=self.gate.bias_vl,
    image_sentinel_lo=129257,
    enable_eplb=self.enable_eplb,
    num_redundant_experts=self.n_redundant_experts,
    is_sequence_parallel=self.is_sequence_parallel,
    n_shared_experts=...,                 # 仅 mix_placement 时非 0
    hash_indices_table=self.gate.tid2eid, # 取代旧的 hash/tid2eid 双参数
)
```

| 易错点 | 说明 |
|--------|------|
| 不再传 `use_grouped_topk` / `n_group` | v0.32.0 的 V4 模型**没有**传 `use_grouped_topk`、`num_expert_group`、`topk_group`，工厂默认 `use_grouped_topk=False`，分组 TopK 在 V4 上退化为"单组、384 直接选 Top-6" |
| `scoring_func` | 代码 `getattr` 默认 `"softmax"`（兼容 DeepSeekV2/V3），V4 配置为 **`sqrtsoftplus`** |
| `routed_scaling_factor` | 注释（L389-L391）明确：缩放刻意保留在路由侧，顺序为"先归一化 Top-K 权重 → 再缩放路由输出"，以匹配 DeepSeek V4 官方行为；ROCm AITER 路径才在算子内部缩放 |
| `hash_indices_table` | v0.32.0 用单一 `hash_indices_table` 取代早期的 `hash`/`tid2eid` 两参数；补丁层再把它同时透传给 router 的 `tid2eid`（`patch_fused_moe.py` L154-L179） |
| `gate` 被传入工厂 | 这是"内部路由"能力的前提（见 §3.3） |

---

## 三、双路由模式 + 内部路由

### 3.1 Hash 路由（layer 0/1/2）

**触发条件**：`is_draft_layer=False` 且 `mlp_layer_types[layer_idx] == "hash_moe"`；v0.32.0 的 V4 配置没有 `mlp_layer_types`，因此回退为 `layer_idx < num_hash_layers(3)`（L345-L352）。

最底层 3 个层的专家分配是**训练前就固定好的**：checkpoint 直接给出 `tid2eid` 表 `(129280, 6)`，推理时按 token 的词表 ID 查出 6 个专家下标，不做任何学习型打分。

```
input_ids (num_tokens,)
    │ （MC2/All2All 路径先 pad_and_split_input_ids；ALLGATHER 路径 all_gather_input_ids；
    │   必要时再做 sequence_parallel_chunk —— router/fused_topk_router.py L165-L176）
    ▼
tid2eid[input_ids]  →  topk_ids (num_tokens, 6)   ← 在 moe_gating_top_k_hash 内部完成
```

特点：零学习参数、零浮点打分、查表即得，底层语义稳定；draft 层即使层号 < 3 也不启用 hash（L345-L346 直接置 False）。`forward` 在 hash 层强制要求 `input_ids`，缺失时抛 `ValueError`（L410-L411）。

新增（v0.32.0）：**vision checkpoint** 中 `layers.*.ffn.gate.bias_vl` 会被保留（`load_weights` L1254-L1258）。图像 sentinel id 段 `[129257, 129262)` 的 token 行不再用 `tid2eid`，而是用 `sqrt(softplus(logits)) + bias_vl` 重新选 Top-K；文本行仍走 `tid2eid`。相关逻辑见 `select_deepseek_v4_vision_experts`（[fused_topk_router.py:L32-L77](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/router/fused_topk_router.py#L32)），硬件支持时下放到 `moe_gating_top_k_hash` 的 `bias_vl/image_sentinel_lo/image_sentinel_count` 属性。

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
- 选专家的融合算子实现见 [fused_topk_router.py:L159-L223](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/router/fused_topk_router.py#L159)：`sqrtsoftplus` 统一走 `moe_gating_top_k_hash` 自定义算子（hash 层带 `tid2eid/input_ids`，线性层两者传 None，`norm_type=2`）；softmax/sigmoid 才走 `moe_gating_top_k`（`norm_type=0/1`，L224-L239）。V4 下等效为 `group_count=1, k_group=1`，`group_select_mode` 固定为 1。
- 纯 Python 兜底路径在 [grouped_topk_router.py:L79-L127](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/router/grouped_topk_router.py#L79)：`sqrtsoftplus` 的打分在 L91-L92，偏置加法在 L117-L125。

### 3.3 内部路由 vs 外部路由（`is_internal_router`）

[ops/fused_moe/fused_moe.py:L244-L249](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py#L244) 定义：

```python
@property
def is_internal_router(self) -> bool:
    # main (cdc4824a21): vllm#51838 removed the gate branch in
    # DeepseekV2MoE.forward, always passing router_logits=hidden_states.
    # The runner must recompute router_logits via the gate.
    return self.gate is not None
```

> **版本变化**：早期版本的判断条件是 `hasattr(gate, "weight_fp32")`；v0.32.0 改为**只要构造时传入了 `gate` 就成立**（对应上游 vllm#51838 移除了 `DeepseekV2MoE.forward` 里的 gate 分支）。V4 始终传入 gate，因此 **V4 恒为内部路由**。

两条路径的区别（MoE forward L418-L434）：

```python
if self.experts.is_internal_router:
    # 把"原始 hidden"直接当 router_logits 传进去，门控 F.linear 在 runner 内部做
    # hidden_states_fp32 是 rms_norm_cast 产出的 fp32 路由输入，没有则用 bf16 hidden
    router_input = hidden_states if hidden_states_fp32 is None else hidden_states_fp32
    fused_moe_out = self.experts(
        hidden_states=hidden_states, router_logits=router_input, input_ids=input_ids,
    )
else:
    # 外部路径：MoE 自己先算好真正的 router_logits 再传入
    router_input = hidden_states.float() if hidden_states_fp32 is None else hidden_states_fp32
    router_logits = F.linear(router_input, self.gate.weight)
    fused_moe_out = self.experts(
        hidden_states=hidden_states, router_logits=router_logits, input_ids=input_ids,
    )
```

内部路径在 runner 侧分三处执行（[fused_moe.py:L385-L429](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py#L385)）：

1. `_prepare_router_input`：有 `weight_fp32` 时优先复用 fp32 的 `router_logits`，否则把 hidden cast 到 fp32；
2. `_apply_router_linear` / `_compute_router_logits`：`F.linear(router_input, gate.weight_fp32)`；没有 `weight_fp32` 的门控则走 `gate(...)` 的注册前向；
3. `_forward_impl`（L493）在"无共享专家"分支先按 `is_internal_router` 算 logits（L502-L509）；有共享专家时由 `_prepare_router_and_milestones`（L459-L470）用 npu event 把路由 gate matmul 与共享专家的 gate_up 排到两条 stream 上重叠。

把打分放进 runner 的好处是：门控、选专家、dispatch 可以在同一套 stream/event 编排里紧挨着重排，减少 HBM 物化和同步。判断技巧是 `router_logits.shape[-1] == hidden_size(7168)`——说明传进来的还是 hidden 而非 384 维打分。

---

## 四、Forward 完整流程

**文件位置**: `DeepseekV4MoE.forward` [L404-L464](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py#L404)

```
输入: hidden_states (N, 7168), input_ids | None, hidden_states_fp32 (N, 7168) | None
   │
   ▼
1. 入参校验与展平（L410-L416）
     hash 层要求 input_ids 非空，否则抛 ValueError；
     hidden/hidden_states_fp32 统一 view(-1, 7168)
   │
   ▼
2. 路由（L418-L434）
     内部路由: experts(hidden, router_logits=hidden|fp32, input_ids)
     外部路由: F.linear(fp32_input, gate.weight) → experts(hidden, logits, input_ids)
   │
   ▼
3. MoERunner 执行（AscendMoERunner）
     prepare(SP 分片/量化) → 选专家(router) → dispatch → w1/w3 GMM(+swiglu)
     → w2 GMM → combine → finalize（含共享专家并行与输出相加）
   │
   ▼
4. 输出处理（L436-L462）
     runner 默认返回单 tensor → 直接使用；
     若返回 tuple(shared_output, final_hidden_states) 走旧路径，见 §4.3
   │
   ▼
5. 返回 view 回 (num_tokens, 7168)（L464）
```

### 4.1 序列并行（Sequence Parallel MoE）

v0.32.0 已把 SP 的分片/聚合**全部收进 runner 的 prepare/finalize**，模型层不再做 chunk 与 all_gather：

| 阶段 | 位置 | 说明 |
|------|------|------|
| 输入分片/对齐 | [prepare_finalize.py:L327-L348](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py#L327) `pad_and_split_input_ids` | MoE-only SP 保留全量 token ID，模型级 SP 已分片时用 `sequence_parallel_chunk` 对齐到本地 hidden；再按 TP `tensor_split` 或 pad |
| 输入聚合 | [prepare_finalize.py:L420-L421](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py#L420) | EP 组 `maybe_all_gather_and_maybe_unpad(hidden_states/router_logits)` |
| 输出回传 | [prepare_finalize.py:L316-L325](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py#L316) `finalize` | `replace_allreduce` 时直接截回本 rank 的 SP 分片，其余情况走上游 reduce-scatter/all-reduce |

hash 路由的 `input_ids` 也必须与本地 router 行对齐：ALLGATHER 走 `all_gather_input_ids`，MC2/All2All 走 `pad_and_split_input_ids`，必要时再补一次 `sequence_parallel_chunk`（[fused_topk_router.py:L165-L176](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/router/fused_topk_router.py#L165)）。

### 4.2 FusedMoE 的输出形态

- **单 tensor（v0.32.0 默认）**：`AscendMoERunner.forward` 在内部已完成 `shared_output + fused_output` 追加与最终归约（[fused_moe.py:L212-L242](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py#L212)），并返回 `(num_tokens, 7168)`。
- **tuple 形态（兼容路径）**：仅当上层仍返回 `(shared_output, routed_output)` 时，模型层才走 §4.3 的外层融合分支；v0.32.0 的 `AscendMoERunner` 不再返回该形态。

### 4.3 输出融合（模型层仅保留兼容分支）

缩放在 v0.32.0 已前移到**路由侧**：`topk_weights * routed_scaling_factor(2.5)`（[grouped_topk_router.py:L125](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/router/grouped_topk_router.py#L125) 与 [fused_topk_router.py:L212](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/router/fused_topk_router.py#L212)），因此 runner 的 `routed_scaling_factor` 在默认路径下是 1.0（`layer.py` L432-L437 只在 `apply_routed_scale_to_output=True` 时才把 2.5 交给 runner）。

模型层保留的旧分支（L436-L457）：

```python
if fused_moe_out_is_tuple:
    shared_output, final_hidden_states = fused_moe_out
    if hidden_states.dtype != torch.float16:
        if not self.is_rocm_aiter_moe_enabled:           # NPU 路径必进
            if self.shared_experts is not None:
                final_hidden_states = muls_add_triton(
                    final_hidden_states, shared_output, self.routed_scaling_factor,
                )                                         # out = routed * 2.5 + shared
            else:
                final_hidden_states *= self.routed_scaling_factor
    elif self.shared_experts is not None:                # fp16 路径
        final_hidden_states = muls_add_triton(
            shared_output, final_hidden_states, 1.0 / self.routed_scaling_factor,
        )                                                 # out = shared * 0.4 + routed
```

| 路径 | 公式 |
|------|------|
| v0.32.0 默认（路由侧缩放 + runner 相加） | `out = routed(含2.5) + shared` |
| 兼容 bf16 | `out = routed × 2.5 + shared` |
| 兼容 fp16 | `out = shared × 0.4 + routed` |
| 无共享专家 | `out = routed × 2.5` |

> **注意**：兼容分支里 bf16 与 fp16 的公式**数学上并不等价**（fp16 路径整体相对 bf16 路径差一个 1/2.5 的量级关系），这是框架按 dtype 刻意保留的两条旧实现；v0.32.0 的默认执行路径已统一为"路由侧乘 2.5，runner 内 `shared + routed`"，不再区分 dtype。ROCm AITER 路径在非 fp16 分支里会跳过外层缩放（`is_rocm_aiter_moe_enabled`，L443），因为 AITER 在算子内部已经乘过 factor。

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

冗余专家是热门专家的复制份：当某 token 路由到的逻辑专家在"离得近/更空闲"的 rank 上有冗余副本时，优先分发过去，减少跨 rank all-to-all 的量与长尾。EPLB 重排由 `parallel_config.eplb_config` 驱动，`enable_eplb` 与冗余数透传进 FusedMoEFactory（L397-L398）。

v0.32.0 的执行细节在 [routed_experts.py:L390](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/routed_experts.py#L390) 的 `AscendRoutedExperts`：

- `init_eplb`（L482 起）构建 `global_expert_map / ascend_expert_map / log2phy`，逻辑专家到本地物理槽的映射表 `log2phy` 在选专家后立即生效（L641-L642：`topk_ids = self.log2phy[topk_ids]`）。
- 冗余槽位的权重视图通过 `get_expert_weights`（L447 起）暴露给 EPLB 管理器；动态 EPLB 的专家热度在 [eplb.py](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/eplb.py) 中 `record_expert_tokens` 累计，强制均衡（“假负载”）的 TopK 造数在 [force_eplb.py](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/force_eplb.py) 的 `build_force_eplb_topk` 中生成。

---

## 六、路由选择细节：noaux_tc 与分组 TopK

### 6.1 V4 的实际配置：单组 + noaux_tc

v0.32.0 的 V4 模型构造 `FusedMoEFactory` 时**没有传入** `use_grouped_topk`、`num_expert_group`、`topk_group`（见 §2.5 代码），工厂侧取默认值 `use_grouped_topk=False`、`num_expert_group=None`、`topk_group=None`，`router_factory` 再把它们兜底成 1。因此实际行为是：

```
384 个专家视为 1 个大组 → sqrtsoftplus 打分并加修正偏置 → 直接选全局 Top-6
```

融合算子仍接收 `group_count=1, k_group=1, group_select_mode=1`，分组 TopK 的 kernel 能力在代码里保留但 V4 上退化为单组。

`topk_method="noaux_tc"` 不引入新的选择算子，它体现在 `e_score_correction_bias` 这个可学习逐专家偏置上：训练阶段用它替代显式辅助损失来均衡专家负载，推理阶段零额外成本。

### 6.2 三种打分函数与 renorm

| `scoring_func` | 公式 | norm_type | 算子路径 |
|----------------|------|-----------|---------|
| `softmax` | `softmax(logits)`（全专家归一） | 0 | `moe_gating_top_k` |
| `sigmoid` | `sigmoid(logits)`（独立多热） | 1 | `moe_gating_top_k` |
| **`sqrtsoftplus`（V4）** | **`sqrt(softplus(logits))`** | 2 | `moe_gating_top_k_hash`（hash/非 hash 共用） |

`renormalize`（V4 `norm_topk_prob=true`）控制选中的 6 个权重是否重新归一化。`sqrtsoftplus` 分支调用融合算子时**固定传 `renorm=0`**——[fused_topk_router.py:L215-L216](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/router/fused_topk_router.py#L215) 的注释说明 hash 定制算子当前只接受 `renorm=0`；softmax/sigmoid 分支则透传 `renorm`（`norm_type=0/1`）。纯 Python 兜底路径显式调用 `_renormalize_topk_weights`（[grouped_topk_router.py:L50-L56](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/router/grouped_topk_router.py#L50)）。

### 6.3 router_factory 的选择顺序

[router_factory.py:L36-L118](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/router/router_factory.py#L36) 的 `create_ascend_fused_moe_router` 依次判断：

| 条件 | 选用 |
|------|------|
| 显式传入 `custom_routing_function` | 上游 `CustomRoutingRouter` |
| 硬件能力 `FUSED_MOE_COMPATIBILITY`（310P 兼容档） | `AscendGroupedTopKRouter310` |
| `check_npu_moe_gating_top_k` 通过（L15-L33） | **`AscendFusedTopKRouter`（V4 走这里）** |
| 其余 | `AscendGroupedTopKRouter`（纯 Python） |

`check_npu_moe_gating_top_k` 的判定条件：`sigmoid` 不允许 `renormalize=False`、不接受自定义路由函数、`scoring_func` 必须是 `softmax/sigmoid/sqrtsoftplus`、`top_k >= 1`、`1 <= topk_group <= num_expert_group`。V4（`sqrtsoftplus` + renorm + 单组）完全满足。

---

## 七、NPU MoE 相关定制算子

`csrc/moe/` 目录在 v0.32.0 中实际编译的算子子目录如下（仅列与 DeepSeek V4 MoE/路由相关者；`hc_pre*`、`hc_post` 见文档 1/2，`chunk_gated_delta_rule*`、`causal_conv1d*`、`copy_and_expand_eagle_inputs` 属于其他模型路径）：

| 算子目录 | 在 V4 MoE 中的作用 |
|---------|-------------------|
| `moe_gating_top_k` | 路由打分 + 分组 TopK 选择（softmax/sigmoid 路径），含 `generalized`/`without_group`/`e_k_fullload` 多套 kernel |
| `moe_gating_top_k_hash` | V4 `sqrtsoftplus` 路径专用：hash 层带 `tid2eid + input_ids` 查表，线性层两者传 None；v0.32.0 新增 `bias_vl` / `image_sentinel_lo` / `image_sentinel_count` 三个视觉路由输入/属性；支持 regbase 寄存器基址内核（arch35 / ascend950 配置） |
| `swiglu_group_quant` | 按专家分组做 SiLU 门控乘 + clamp（`swiglu_limit=10`）+ per-token int8 动态量化；v0.32.0 主要由**共享专家** W8A8 路径调用（`shared_experts.py` L607/L651） |
| `dequant_situ_quant` | Situ 激活 + 动态量化（W4A8 的 Situ 分支在 `w4a8.py` L485 调用） |
| `dequant_swiglu_quant` | swiglu 量化算子的反量化/变体版本（静态/动态 per-token，bf16/int32 bias） |
| `situ_mx_quant` / `grouped_matmul_situ_quant` | MX 风格缩放量化与 Situ 分组矩阵乘融合（低秩/辅助量化路径） |
| `rms_norm_cast` / `add_rms_norm_bias` | 产出 fp32 路由输入的融合 RMSNorm（`rms_norm_cast`）及带 bias 的归约融合（配合内部路由，见文档 3） |
| `transpose_kv_cache_by_block` | 按 block 转置 KV cache（属 KV 路径，见文档 5） |
| `scatter_nd_update_sk` | 按索引散射写入，v0.32.0 只用于 **KV cache / indexer cache** 更新，不再参与 MoE combine |

> **路由专家 GMM 已迁移**：早期的 `csrc/moe/moe_grouped_matmul` 目录在 v0.32.0 已不存在——普通 GMM 直接走 `torch_npu.npu_grouped_matmul`（[routed_experts.py:L234](../../../../../vllm-ascend/vllm_ascend/ops/fused_moe/routed_experts.py#L234)、L297），W4A8 融合的 "GMM1 + SiLU + clamp + int8 量化" 由 `csrc/gmm/` 下的 `grouped_matmul_swiglu_quant` / `grouped_matmul_swiglu_quant_v2` / `grouped_matmul_swiglu_quant_weight_nz_tensor_list` 提供（调用点 `w4a8.py` L504-L514）。

> 早期文档中出现的 `moe_init_routing_custom`、`hamming_dist_top_k`、`scatter_nd_update_v2` 在 v0.32.0 的 `csrc/moe/` 下仍不存在；路由初始化改由 `DeviceOperator.npu_moe_init_routing`（`token_dispatcher.py` L429）承担，两阶段路径用 `torch_npu.npu_moe_init_routing_v2`（同文件 L706），finalize 走 `torch_npu.npu_moe_finalize_routing`。

---

## 八、W4A8 量化在 MoE 中的落地（速览）

本 checkpoint 为 W4A8 动态量化版，MoE 相关 dtype 分工（完整逐张量清单见同目录 `0_DeepSeek-V4-Pro-0813-w4a8_quant_model_description.json`）：

| 模块 | 量化方案 | 说明 |
|------|---------|------|
| 路由专家 w1(gate)/w3(up)/w2(down) | **W4A8_DYNAMIC，权重 fp4**（`expert_dtype="fp4"`） | 权重 4-bit 存储，激活 per-token 动态量化到 8-bit；fp4 的 E8M0 scale 以 uint8 view 加载 |
| w1/w3 GEMM + 激活 | `grouped_matmul_swiglu_quant_v2`（csrc/gmm） | 一次算子完成 GMM1、SiLU×gate、`swiglu_limit=10.0` clamp 与 int8 量化，输出直接喂 w2 GEMM（`w4a8.py` L504-L514） |
| w2 GEMM | `torch_npu.npu_grouped_matmul` | 带 `per_token_scale` 的 A8W4 分组矩阵乘（`w4a8.py` L544-L559） |
| gate 门控 / `e_score_correction_bias` | FLOAT（不量化） | 门控权重另存 `weight_fp32`，打分全程 fp32 |
| 共享专家 | 随量化配置（gate_up/down 走对应量化线性层） | 本体仍是 `DeepseekV2MLP` |
| 权重 scale 后缀 | fp4 → `.weight_scale`；其他 → `.weight_scale_inv` | 权重加载时的命名映射见文档 6/8 与模型 `load_weights` |

量化的加载细节（fp4 张量 reshape、E8M0 处理、scale 重命名）在各模型文件的 `load_weights` 中完成，本文不再展开。

---

## 九、一句话总结

DeepSeek V4 的每一层 FFN 都是"**1 个全量共享 MLP + 384 选 6 稀疏专家**"的 MoE：前 3 层按词表 hash 静态分派（vision checkpoint 对图像 sentinel 行改用 `bias_vl` 重排），其余层用 **fp32 门控 + sqrtsoftplus + noaux_tc 偏置**选专家；门控在 V4 上收进 MoE runner 内部执行（`is_internal_router = gate is not None`），专家 GEMM 走 W4A8 fp4 的分组矩阵乘与 `grouped_matmul_swiglu_quant_v2` 融合算子；路由权重在路由侧归一化后按 `routed_scaling_factor=2.5` 缩放，runner 内与共享输出直接相加；EPLB 用冗余专家加 `log2phy` 映射在 EP 维度均衡负载。
