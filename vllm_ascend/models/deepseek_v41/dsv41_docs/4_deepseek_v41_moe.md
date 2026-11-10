# DeepSeek V4.1 MoE 前馈网络详解

> **本文档详细讲解 DeepSeek V4.1-Flash 的 MoE（Mixture of Experts）前馈网络在 Ascend NPU 上的实现：路由门控、专家计算、共享专家融合、EPLB 物理专家映射与权重加载映射。**

> **代码版本冻结声明**：本文基于本仓库当前 `vllm_ascend/models/deepseek_v41/` 源码与 `DeepSeek-V4.1-Flash-1017` checkpoint 整理。文中 **`Lxxx` 一律指 [model.py](../model.py) 的行号**；其他文件的行号显式标注文件名，如 `dspark.py L433`、[utils.py](../../../utils.py#L168) `L168`。融合 MoE 执行器在 [ops/fused_moe/fused_moe.py](../../../ops/fused_moe/fused_moe.py)，路由选择在 [ops/fused_moe/router/](../../../ops/fused_moe/router/router_factory.py)，训练语义参考实现在 [reference/model.py](reference/model.py)。

> **统一模型配置**（取自 `dsv41_docs/0_DeepSeek-V4.1-Flash-1017_config.json` 的 `text_config`：MoE 行见该文件 L52-L58，draft 行见 L145-L155）：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `num_hidden_layers` | **40** | 主干 40 层，**全部**是 MoE FFN，无稠密 FFN 层 |
| `hidden_size` / `moe_intermediate_size` | **5120 / 2304** | 隐维 / 单专家中间维（与 V4 相比 V4-Pro 为 7168 / 3072） |
| `n_routed_experts` / `num_experts_per_tok` | **384 / 6** | 每层路由专家数 / 每 token 激活数（Top-K=6） |
| `n_shared_experts` | **1** | 共享专家数（所有 token 全量经过） |
| `scoring_func` / `topk_method` | **`sqrtsoftplus` / `noaux_tc`** | 打分 `sqrt(softplus(logits))`；偏置修正 TopK |
| `norm_topk_prob` / `routed_scaling_factor` | **true / 1.5** | Top-K 权重重归一化；路由输出缩放（兜底默认也是 1.5，`L168`） |
| `swiglu_limit` | **10.0** | SwiGLU 激活限幅（配合 fp4/fp8 量化） |
| `num_hash_layers` / `n_group` | **均不存在** | 归一化为 0 / 1 → hash 路由不生效、单组 384 选 6（utils.py `L170`） |
| `expert_dtype` | **`fp4`** | 路由专家 w1/w2/w3 走 W4A8（config L12-L21） |
| `dspark_n_routed_experts` / `..._experts_per_tok` | **128 / 3** | draft（DSpark）层专家数 / Top-K，比主干窄（config L154-L155） |

---

## 一、MoE 架构总览

### 1.1 设计理念

V4.1-Flash 的 FFN **全部**是 MoE：[DeepseekV41DecoderLayer](../model.py#L811) 无条件构造 `self.mlp = DeepseekV41MoE(...)`（`L811-L817`），代码里没有任何 dense / `mlp_layer_types` 分支可退回稠密 FFN。因此 40 层主干（config L27）加 3 个 DSpark draft 层（`num_nextn_predict_layers=3`，config L145；由 [dspark.py](../dspark.py#L151) 的 `DeepseekV41DSparkModel` 以 `mtp.{idx}` 前缀构造，`L173-L183`）共 43 个 MoE 实例。

单专家参数 ≈ 3 × 5120 × 2304 ≈ **35.4M**。单层 384 个路由专家合计约 13.6B 参数，但每 token 只激活 6 个（≈212M）加 1 个共享专家（≈35M），**激活率约 1.8%**——这就是"总容量很大、单 token 计算量很小"的来源。

**与 V4 相比**的差异必须说清，避免把 V4-Pro-0813 的数字当成 V4.1 的事实：

| 维度 | V4.1-Flash-1017（本文） | 与 V4 相比 |
|------|--------------------------|------------|
| 主干层数 | 40 层，全 MoE | V4-Pro-0813 为 61 层 + 3 draft |
| 隐维 / 专家中间维 | 5120 / 2304 | V4-Pro 为 7168 / 3072 |
| hash 层 | **0 层**（config 无 `num_hash_layers`） | V4-Pro-0813 前 3 层为 hash 路由 |
| `routed_scaling_factor` | **1.5** | V4-Pro 配置为 2.5 |
| 视觉路由 bias | 存在（config 带 `vision_config`） | V4 的 vision checkpoint 同样有 |

> **生活化类比**：MoE 像一个有 384 位厨师的中央厨房。每份订单（token）进门先由前台（gate 路由门控）给 384 位厨师各打一个匹配分，只喊最匹配的 6 位下锅（Top-K=6 路由专家）；同时不管什么菜都要过一位常备配菜师（1 个共享专家）先出一份基础配菜。最终出餐 = 配菜师的固定份 + 6 位厨师按匹配分加权的那份，而加权系数（`routed_scaling_factor=1.5`）在派单时就乘好，端菜时直接相加。

### 1.2 数据流

```
输入: hidden_states (N, 5120) bf16, input_ids (N,), hidden_states_fp32 (N, 5120)   ← fp32 来自 rms_norm_cast（L912-L918）
   ▼
[0] 序列并行分片（可选，L280-L283）：is_sequence_parallel 且 !already_sequence_parallel → sp_shard(hidden) + sp_shard(fp32)
   ▼
[1] 路由（内部路由，L285-L292）：router_input = hidden_states_fp32（缺失退回 bf16 hidden，L287）
      F.linear(router_input, gate.weight_fp32) → logits (N, 384)
      → sqrt(softplus(logits)) → +e_score_correction_bias（图像行换 bias_vl）
      → Top-6 选择 + 归一化 → topk_weights (N,6) ×1.5, topk_ids (N,6)
   ▼
[2] 专家计算（FusedMoEFactory → AscendMoERunner，L239-L263）
      dispatch（AllGather / MC2 / All2All）→ w1/w3 GMM + swiglu(clamp=10.0) + 动态量化 → w2 GMM → combine
      → routed_out (N_local, 5120)；并行支路：shared_experts（1 个稠密 MLP）在另一条 stream 上出 shared_out
   ▼
[3] 输出融合（runner 内，fused_moe.py L212、L235-L238）：out = shared_out + routed_out（routed 已含 ×1.5）
   ▼
[4] 序列并行聚合（L323-L325）：sp_all_gather(out)[:num_tokens] → 返回 (N, 5120)（L331）
```

### 1.3 核心类关系

```
DeepseekV41MoE (MoE 主类, model.py L154)
├── gate: ReplicatedLinear(5120 → 384)          — 每个 EP rank 存完整一份（L179-L181）
│   ├── weight (bf16) + weight_fp32             — precast_fp32_weight=True（L182）
│   ├── tid2eid: None                           — 仅 hash 层，本 checkpoint 恒 None（L236）
│   ├── e_score_correction_bias: (384,) fp32    — 非 hash 层（本 checkpoint 全 40 层，L237）
│   └── bias_vl: (384,) fp32 | None             — vision checkpoint（L214-L222）
└── experts = FusedMoEFactory(...)              — 返回 AscendMoERunner（L239-L263）
    └── AscendMoERunner (fused_moe.py L85)：router = AscendFusedTopKRouter（fused_topk_router.py L80）
        ├── routed_experts: AscendRoutedExperts（routed_experts.py L390）— 384 个路由专家 FFN
        └── ascend_shared_experts: AscendSharedExperts（shared_experts.py L99）
shared_experts: DeepseekV41MLP | None           — mix_placement 时为 None（L196-L211）
```

`FusedMoEFactory` 是上游工厂函数，NPU 侧由 [patch_fused_moe.py](../../../patch/platform/patch_fused_moe.py#L110) 的 `_ascend_FusedMoE` 接管（替换点 `L208-L209`，并回填已导入模块的符号 `L212-L216`），实际执行体是继承上游 `MoERunner` 的 `AscendMoERunner`，用 `_forward_impl`（`L493-L548`）重排共享专家与路由专家的调度顺序；EPLB 接口由 `DeepseekV41MixtureOfExperts`（`L340-L377`）提供。

### 1.4 共享专家的本体：DeepseekV41MLP

**文件位置**: [model.py L107-L151](../model.py#L107)。它**不是**"非 MoE 层的稠密 FFN"（V4.1 没有稠密 FFN 层），唯一角色是 `shared_experts` 的网络本体（`L202-L211`）。

| 组件 | 类型 | 说明 |
|------|------|------|
| `gate_up_proj` | `MergedColumnParallelLinear` 5120→2×2304 | w1(gate)+w3(up) 合并成一次列并行 GEMM（`L125-L132`）；`disable_tp=is_sequence_parallel`（`L130`） |
| `act_fn` | `SiluAndMulWithClamp(10.0)` | 有 `swiglu_limit` 用带 clamp 版（`L142-L143`），否则 `SiluAndMul`（`L144-L145`） |
| `down_proj` | `RowParallelLinear` 2304→5120 | w2 行并行；共享专家场景固定 `reduce_results=False`（`L138` 传入、`L209` 取值） |

`n_shared_experts > 1` 时 `intermediate_size = moe_intermediate_size * n_shared_experts`（`L200`）；本配置为 1，即 2304。`forward` 见 `L147-L151`。

---

## 二、路由门控（gate）

### 2.1 gate 构造与 fp32 预转换

`self.gate = ReplicatedLinear(hidden_size, n_routed_experts, bias=False, quant_config=None, prefix=...gate)`，随后 `self.gate.precast_fp32_weight = True`（`L179-L182`）：每个 rank 存完整一份、不切 TP，门控刻意不量化，加载后额外预算一份 fp32 权重。

| 属性 | 形状 / dtype | 存在条件 | 说明 |
|------|--------------|----------|------|
| `gate.weight` / `weight_fp32` | `(384, 5120)` bf16 / fp32 | 始终 | 原始路由权重；加载后预转换的 fp32 副本 |
| `gate.e_score_correction_bias` | `(384,)` fp32 | 非 hash 层（`L237`） | noaux_tc 的逐专家偏置 |
| `gate.bias_vl` | `(384,)` fp32 | `vision_n_layers > 0`（`L214-L222`） | 图像跨度 token 专用路由偏置 |
| `gate.tid2eid` | `(129280, 6)` int32 | hash 层（`L226-L233`） | 本 checkpoint **不存在**（`L236` 置 None） |

runner 复用这份 fp32 权重的地方在 [fused_moe.py](../../../ops/fused_moe/fused_moe.py#L397)：`_apply_router_linear` 里 `F.linear(router_input, gate.weight_fp32)`（`L403-L404`），没有 `weight_fp32` 的门控才退回 `gate(...)`（`L405-L406`）。路由选择走 [router_factory.py](../../../ops/fused_moe/router/router_factory.py#L36) 的第 3 档 `AscendFusedTopKRouter`（`L88-L105`，别名 `AscendFusedMoERouter` 见 `L9-L11`），因为 V4.1 的 `sqrtsoftplus` + `renormalize=True` + 单组（`topk_group=1 <= num_expert_group=1`）满足 `check_npu_moe_gating_top_k`（`L15-L33`）。

> **为什么门控要 fp32？** 路由打分决定"这个 token 归谁算"，一旦 Top-K 边界选错，后面 6 组 GMM 全白做。`softplus`/`sqrt` 在低精度下的误差足以改变 Top-6 边界，所以路由输入（`hidden_states_fp32`）与门控权重（`weight_fp32`）都强制 fp32，两个偏置也是 fp32 参数（`L216-L222`、`L237`）。

### 2.2 sqrtsoftplus 打分

训练语义以 [reference/model.py](reference/model.py#L794) 的 `Gate.forward` 为准：`scores = linear(x.float(), self.weight.float()) / self.gate_temp`（`L813`），非 softmax/sigmoid 时 `scores = F.softplus(scores).sqrt()`（`L819`）。`gate_temp` 默认 1.0（reference `L67`/`L804`），运行时没有温度入参、等价于 1.0，训练若用过其它温度需在权重侧折算；NPU 融合算子对应 `moe_gating_top_k_hash` 的 `norm_type=2`（fused_topk_router.py `L217`），纯 Python 兜底在 grouped_topk_router.py `L91`。

`sqrt(softplus(x))` 与 softmax 的本质区别：softmax 强制 384 个分数归一（互相抢资源），而 `sqrt(softplus)` 是**逐专家的非归一化置信度**，多个专家可以同时"高分"。这也解释了为什么还要单独提供 `norm_topk_prob`——归一化从打分函数里拿掉后，要在选中的 6 个权重上补一次。

### 2.3 noaux_tc：偏置只影响选择，不影响权重

reference 的注释把语义写得最清楚（`L823-L825`）：

```python
# the bias picks experts but does not scale them: weights come from the raw scores
indices = (scores + bias).topk(self.topk, dim=-1)[1]
weights = scores.gather(1, indices)
```

即 **`bias` 只加在用于排序的临时分数上，选出 indices 后权重回头从未加偏置的 `scores` 里 gather**。偏置是训练期把 token 推给冷门专家的"调度杠杆"，不参与推理数值，因此推理零额外成本。NPU 融合算子的入参名就叫 `bias`（fused_topk_router.py `L207`），Python 兜底同样先算 `scores` 再加偏置选 TopK（`L50-L63`）。偏置形状 `(384,)` fp32，只有非 hash 层创建（`L235-L237`，本 checkpoint 40 层全部有），checkpoint 名 `...ffn.gate.bias` 加载时改名为 `...gate.e_score_correction_bias`（`L1912-L1913`）。

### 2.4 Top-6 与 norm_topk_prob

`renormalize=config.norm_topk_prob`（`L246`）= **true**，参考实现（reference `L826-L828`）：

```python
if self.norm_topk_prob and self.topk > 1:
    weights /= weights.sum(dim=-1, keepdim=True) + 1e-20   # not norm_eps, matches training
weights *= self.route_scale
```

分母加的是 `1e-20` 而**不是** `norm_eps`（V4.1 的 `rms_norm_eps=1e-20` 恰好同值但语义不同），reference 特意标注 `matches training`。顺序是"**先归一化、再缩放**"（本 checkpoint 对应 `routed_scaling_factor=1.5`），会影响输出量级，改实现时必须保持。

NPU 融合路径有一个必须知道的细节：`moe_gating_top_k_hash` 调用处把 `renorm` **写死为 0**（fused_topk_router.py `L215-L216`，注释："The hash custom op currently accepts only renorm=0."）；归一化语义在 Python 兜底路径里是显式实现的（`L71-L74`）。因此**Top-K 权重与参考实现有细微数值差时，第一嫌疑点就是这条 renorm 路径**，而不是打分函数。

### 2.5 静态查表（tid2eid）与图像路由（bias_vl）

**tid2eid 的结构**：`(vocab_size=129280, num_experts_per_tok=6)` int32，含义是"第 `input_ids` 个词表 ID 固定分给这 6 个专家"，由 `L226-L233` 创建（用 `torch.zeros` 而非 `empty`，注释 `L224-L225` 说明是为 `--load-format="dummy"` 不产生垃圾值），以 `hash_indices_table=self.gate.tid2eid` 传给工厂（`L262`），补丁层再分发给 router 的 `tid2eid` 与 legacy 量化路径（patch_fused_moe.py `L151-L154`、`L169-L170`）。

**本 checkpoint 不生效**：`num_hash_layers` 在 released config.json 中缺失，[normalize_deepseek_v41_config](../../../utils.py#L168) 把它补成 **0**（`L170`，同一兜底列表里还有 `n_group=1`、`topk_group=1`），于是 `self.hash = layer_idx < config.num_hash_layers and not is_draft_layer`（`L213`）对 40 层主干与 3 个 draft 层**恒为 False**：`tid2eid = None`（`L236`）、`e_score_correction_bias` 是 Parameter（`L237`），走 sqrtsoftplus + 偏置的动态路由；hash 分支形态仍留在代码里（`L226`/`L234`），但对该 checkpoint 是死路径。

> **明确结论**：V4.1-Flash-1017 **不是** hash 路由模型。hash 能力（`tid2eid` 查表、`moe_gating_top_k_hash` 的 `input_ids/tid2eid` 入参）完整保留但不执行。**与 V4 相比**：V4-Pro-0813 的 `num_hash_layers=3`，前 3 层确实是 hash 层；把 V4 的"前 3 层查表"套到 V4.1 上是错的。

**图像路由 bias_vl**：config 顶层带 `vision_config`（32 层 ViT、hidden 1024、patch 14，config `L157-L169`），运行时据此判定 `vision_n_layers > 0`（`L215`；该字段与 reference `ModelArgs` 的 `vision_*` 字段对应，reference `L116-L126`），`gate.bias_vl` 因此被创建（`L216-L222`）并作为 `bias_vl=` 传给工厂（`L256`）。它的训练名是 `noaux_tc_for_vl`——reference 的 `Gate` docstring 明确写 "Image-span tokens use a separate bias (training `noaux_tc_for_vl`)"（`L795-L796`），前向用 `torch.where` 按 `image_mask` 在文本偏置与 `bias_vl` 之间切换（`L820-L822`）。NPU 侧分两级（fused_topk_router.py `L180-L197`）：硬件支持 `MOE_GATING_TOP_K_HASH_VISION` 时单算子搞定（`moe_gating_top_k_hash(..., bias_vl=..., image_sentinel_lo=..., image_sentinel_count=5)`，`L204-L222`）；否则走 Python 兜底 `select_deepseek_v4_vision_experts`（`L32-L77`），文本行用文本偏置、图像行按 `sqrt(softplus)+bias_vl` 重选（`L50-L69`）。

**sentinel 区间**：`image_sentinel_lo` 由 `getattr(config, "image_sentinel_base_id", 129257)` 给出（`L257`），而它是 [normalize_deepseek_v41_config](../../../utils.py#L183) 从 `image_token_id`（V4.1 为 **129264**，config L11）写入的；窗口宽度固定 5（fused_topk_router.py `L29`），即图像判定区间为 `[129264, 129269)`。**与 V4 相比**：V4 的默认常量是 129257（同文件 `L28`），多模态场景按各自 checkpoint 的 `image_token_id` 核对，不要硬编码。

还有一个容易踩的联动：因为 `bias_vl is not None`，sqrtsoftplus 分支会**强制要求 `input_ids`**（fused_topk_router.py `L160-L162`，缺失即 `raise ValueError("DeepSeek V4 vision/hash MoE routing requires input_ids.")`）。也就是说**即使 hash 路由不生效，vision checkpoint 的每层 MoE 也必须拿到原始 token id**——这是 [vl_model.py](../vl_model.py#L53) 打开 `requires_raw_input_tokens = True` 的原因（`L53-L54`），也是纯文本跑批仍要透传 `input_ids`（`L915`）的原因。

---

## 三、专家计算

### 3.1 FusedMoEFactory 构造参数

```python
self.experts = FusedMoEFactory(                 # L239-L263
    shared_experts=self.shared_experts,          # L240  共享专家 MLP（mix_placement 时为 None）
    gate=self.gate,                              # L241  传入 gate ⇒ 内部路由（§3.4）
    num_experts=config.n_routed_experts,         # L242  384（不含共享专家）
    top_k=config.num_experts_per_tok,            # L243  6
    hidden_size=config.hidden_size,              # L244  5120
    intermediate_size=config.moe_intermediate_size,  # L245  2304
    renormalize=config.norm_topk_prob,           # L246  true
    quant_config=quant_config,                   # L247  fp8 + expert_dtype=fp4
    scoring_func=getattr(config, "scoring_func", "softmax"),   # L249  sqrtsoftplus
    routed_scaling_factor=self.routed_scaling_factor,          # L253  1.5
    swiglu_limit=self.swiglu_limit,              # L254  10.0
    e_score_correction_bias=self.gate.e_score_correction_bias, # L255
    bias_vl=self.gate.bias_vl,                   # L256
    image_sentinel_lo=getattr(config, "image_sentinel_base_id", 129257),  # L257
    enable_eplb=self.enable_eplb,                # L258
    num_redundant_experts=self.n_redundant_experts,   # L259
    is_sequence_parallel=self.is_sequence_parallel,   # L260
    n_shared_experts=config.n_shared_experts if self.is_fusion_moe_shared_experts_enabled else 0,  # L261
    hash_indices_table=self.gate.tid2eid,        # L262
)
```

| 参数 | 易错点 |
|------|--------|
| `gate=self.gate` | 只要传了 gate，`is_internal_router` 就为 True（fused_moe.py `L244-L249`），门控 matmul 搬进 runner |
| `scoring_func` | `getattr` 兜底是 `"softmax"`（兼容 V2/V3），V4.1 实际是 **`sqrtsoftplus`** |
| `routed_scaling_factor` | 注释（`L250-L252`）说明缩放**保留在路由路径内**，顺序与官方一致：先归一化 Top-K，再缩放路由输出 |
| 不传 `use_grouped_topk`/`n_group` | 工厂默认 `False`/`None`，router_factory 兜底成 1 → "384 个专家当 1 个组，直接选全局 Top-6" |
| `n_shared_experts` | 普通模式传 **0**（共享专家走独立 MLP）；只有 `mix_placement=True` 才传 1（§4.4） |

### 3.2 专家 FFN 结构与 swiglu_limit 的位置

每个路由专家就是一个 SwiGLU FFN（reference `Expert`，`L832-L853`）：`w1`(gate, 5120→2304) 与 `w3`(up, 5120→2304) 并行算出两路，若 `swiglu_limit > 0` 则 **up 双侧限幅到 ±10**（reference `L848`）、**gate 只限上限**（reference `L849`），再 `silu(gate) * up`（reference `L850`），最后 `w2`(down, 2304→5120)。

NPU 侧不手写这几步，而交给激活层：`swiglu_limit` 非空时用 `SiluAndMulWithClamp(swiglu_limit)`（`L142-L143`，导入自 `vllm.model_executor.layers.activation`，`L29`），平台实现是 [AscendSiluAndMulWithClamp](../../../ops/activation.py#L43)（`L43-L52`），最终下放到 `DeviceOperator.clipped_swiglu(x, swiglu_limit=..., swiglu_alpha=..., swiglu_beta=...)`（[device_op.py](../../../device/device_op.py#L433)）——**真正产生 clamp 的位置在这个融合算子里**。路由专家走更深的融合：W4A8 的 `w1/w3 GMM + silu + clamp + 动态量化` 一次完成（§3.3）。

> **为什么限幅？** 训练时 clamp 是为了把 fp8/fp4 激活压在量化可表示范围内（reference `Expert` docstring，`L832-L835`）。推理侧同理：激活一旦出现离群值，per-token 动态 scale 会被单个 outlier 拉爆，整行精度崩掉。`swiglu_limit=10.0` 是 checkpoint 的既定契约，改小会改变语义，改大等于放开量化溢出风险。

### 3.3 W4A8（fp8 + fp4）在 MoE 中的落地

`quantization_config`（config `L12-L21`）：`quant_method=fp8`、`activation_scheme=dynamic`、`weight_block_size=[32,32]`、`scale_fmt=ue8m0`、`expert_dtype=fp4`。

| 模块 | 量化方案 | 依据 |
|------|----------|------|
| 路由专家 w1/w3/w2 | **W4A8**：权重 fp4（`float4_e2m1fn_x2`），激活 per-token 动态量化到 8-bit | `expert_dtype="fp4"` → [fp8_config.py](../../../quantization/configs/fp8_config.py#L136) 选 `ds_w4a8_moe` scheme（`L137-L142`）；reference 同语义见 `L874-L886` |
| 共享专家 gate_up/down | **W8A8**（fp8 block，`[32,32]`、`ue8m0`） | 共享专家是普通 `LinearBase`，走 `ds_linear` scheme（fp8_config.py `L131-L135`） |
| `gate` 门控 / 两个 fp32 偏置 | **不量化** | `quant_config=None`（`L180`）+ `precast_fp32_weight`（`L182`）；偏置见 `L216-L222`、`L237` |

`[32, 32]` 的含义：fp8 权重按 32×32 分块各存一个 E8M0（`ue8m0`）scale；fp4 权重的 scale 以 uint8 view 形式随权重一起搬运（加载期处理，见 §6.3）。`scale_fmt="ue8m0"` 决定 scale 只有幂次、没有尾数——**权重加载时 dtype/reshape 一步错，scale 就会静默错位**，这是 MoE 量化里最常见的"精度莫名下降"。

### 3.4 内部路由（is_internal_router）

[fused_moe.py](../../../ops/fused_moe/fused_moe.py#L244) 里 `is_internal_router` 的实现就是 `return self.gate is not None`（`L244-L249`，注释说明上游 vllm#51838 移除了 `DeepseekV2MoE.forward` 的 gate 分支，runner 必须自己重算 logits）。V4.1 在 `L241` 恒定传入 gate，因此**恒为内部路由**，`forward` 走 `L285-L292`：把 fp32 hidden 当 `router_logits` 传进去，真正的 `F.linear` 在 runner 内完成。判断技巧：`router_logits.shape[-1] == hidden_size` 说明传进来的还是 hidden（5120），不是 384 维打分。

runner 侧分三步（`L385-L491`）：`_prepare_router_input`（`L385-L395`）有 `weight_fp32` 时优先复用已是 fp32 的输入、否则 `hidden.float()`；`_apply_router_linear`（`L397-L406`）与 `_compute_router_logits`（`L408-L419`）执行 `F.linear(router_input, gate.weight_fp32)` 并把 fp32 激活生命周期限制在 helper 内尽早释放；`_prepare_router_and_milestones`（`L445-L491`）用 npu event 把"路由 gate matmul"与"共享专家 gate_up"排到两条 stream。

---

## 四、共享专家与输出融合

### 4.1 结构与 reduce_results=False 的原因

`self.shared_experts = DeepseekV41MLP(..., reduce_results=False, is_sequence_parallel=...)`（`L202-L211`）。shared 的 `down_proj` 是行并行、输出是 TP 部分和：若它自己 all-reduce 一次、之后再和 routed 相加，就多一次集合通信；设成 False（`L209`）后把部分和交给 runner，在 `_fused_output_is_reduced`（fused_moe.py `L272-L284`）判定的时机统一归约。`is_sequence_parallel`（`L208`）则直接进 `disable_tp`（`L130`、`L139`）：SP 下 token 已按 TP 组分片，权重保持完整即可。

### 4.2 时序：共享专家与路由专家如何重叠

`_forward_impl`（`L493-L548`）不是顺序流程，而是一张依赖事件图：shared 输入就绪先 `prepare_input_before_routed`（shared_experts.py `L323-L334`）并可与路由 gate matmul 并发；选完专家后 `enqueue_gate_up_after_router`（fused_moe.py `L481-L491`）投递 shared 的 gate_up；routed 侧同时走 dispatch → GMM1 → GMM2 → combine；最后按 `defer_output_wait` 取 shared_out 并相加（`L235-L238`）。开关有三个：`multistream_overlap_shared_expert`（shared_experts.py `L131`，关掉退化为单 stream）、`npu_stream_switch`（`L483`、`L743` 是真正的切换点）、`SharedExpertParallelMode`（`L52-L61`，决定谁来归约，`L219-L237`）。

### 4.3 muls_add_triton 与 fp16 的反向缩放（兼容分支）

模型层保留了一段**外部融合**分支（`L303-L321`），当 runner 返回 `tuple(shared_output, final_hidden_states)` 时生效：bf16 走 `muls_add_triton(final, shared, 1.5)`（`L311-L313`），fp16 走 `muls_add_triton(shared, final, 1.0/1.5)`（`L316-L319`），无共享专家时直接 `final *= 1.5`（`L315`），否则原样透传（`L320-L321`）。`muls_add_triton` 的 kernel 语义只有一行（[mul_add.py](../../../ops/triton/mul_add.py#L29)，`L25`）：`output = x * scale + y`，一次 launch 完成"乘 + 加"。

**必须说清楚的事实**：本 release 的 `AscendMoERunner.forward` 在 `_unpack` 之后用 `torch._foreach_add([shared_output], [fused_output])` 把两路加好（fused_moe.py `L212`、`L235-L238`），**返回单 tensor**，因此 `L303-L321` 的 tuple 分支在当前路径下不进入，`muls_add_triton` 属于兼容保留。

| 路径 | 公式 | 本 release 是否生效 |
|------|------|---------------------|
| runner 内 combine（路由侧已乘 1.5） | `out = routed(×1.5) + shared` | ✅ 生效 |
| 兼容 bf16 / fp16 | `out = routed × 1.5 + shared` / `out = shared × (1/1.5) + routed` | 仅 tuple 形态 |
| 无共享专家 | `out = routed × 1.5` | mix_placement 时随 runner 走 |

> **注意**：bf16 与 fp16 两条兼容公式**数学上并不等价**（fp16 路径整体差一个 1/1.5 的量级关系），这是框架按 dtype 刻意保留的两条历史实现；V4.1 的默认路径不区分 dtype，统一在 runner 内相加。缩放也**不会**被乘两次：`patch_fused_moe.py L166` 构造 router 时传 `routed_scaling_factor=routed_scaling_factor if not apply_routed_scale_to_output else 1.0`，V4.1 没传 `apply_routed_scale_to_output`（默认 False，`L126`），所以 1.5 由路由算子内部乘（fused_topk_router.py `L212`；vision 兜底路径 `L75-L76`），runner 侧 `_maybe_apply_routed_scale_to_output`（fused_moe.py `L230`）不再二次缩放。

### 4.4 mix_placement：把共享专家塞进专家列表

`get_ascend_config().mix_placement=True` 时 `self.shared_experts = None`（`L197-L198`），工厂的 `n_shared_experts` 传 1（`L261`）；选专家后把 `[384, 385)` 这些 id 追到 `topk_ids` 后面、权重填 1（routed_experts.py `L654-L669`），权重槽位由加载期合成到 `mlp.experts.{384+j}.*`（`L1995-L2005`）。好处是共享专家被打散进 EP 各 rank、与路由专家统一调度，天然做负载混部，并省掉一条独立 GEMM 路径；代价是开关必须与权重加载路径同步（`L1931` 的 `is_fusion_moe_shared_experts_layer` 决定是否走切片/改名），**不能只改运行时开关而不重载权重**。

---

## 五、EPLB 与物理专家

```
384 个逻辑专家 (n_logical_experts) + n_redundant_experts 个冗余副本
   ▼ n_physical_experts 个物理专家，按 EP rank 均分（n_local_physical_experts = n_physical_experts // ep_size）
   ▼ rank 0 → [0, k)，rank 1 → [k, 2k)，……（physical_expert_start/end）
```

| 属性 | 定义 | 行号 |
|------|------|------|
| `enable_eplb` / `n_redundant_experts` | `parallel_config.enable_eplb` / `eplb_config.num_redundant_experts` | `L186`、`L188` |
| `n_logical_experts` / `n_physical_experts` | `= n_routed_experts` = 384 / 逻辑 + 冗余 | `L189`、`L190` |
| `n_local_physical_experts` | 物理总数 // ep_size | `L191` |
| `physical_expert_start/end` | 本 rank 负责的物理下标区间 | `L193-L194` |

冗余专家是热门专家的复制份：某 token 路由到的逻辑专家若在"更空闲/更近"的 rank 上有副本，就优先分发过去，压低跨 rank all-to-all 的量与长尾。

### 5.1 与上游 `MixtureOfExperts` 接口的对接

`DeepseekV41MixtureOfExperts`（`L340-L377`）只做两件事：`extract_moe_parameters(example_moe)` 把单层元数据提升到模型级（六个计数，`example_moe is None` 时全清 0，`L346-L362`）；`update_physical_experts_metadata(num_physical, num_local)` 在 EPLB 重排后回写——断言 `num_local` 不变（`L369`）、重算 `num_redundant = physical - logical`（`L372`）、逐层写回并令 `moe.experts.update_expert_map()`（`L373-L377`）。

`set_moe_parameters` 由主模型（`L1790-L1808`）与 draft 模型（dspark.py `L433-L449`）各自实现，逻辑一致：遍历 `self.model.layers`、跳过 `PPMissingLayer`（`L1799-L1800`），把每个 `DeepseekV41MoE` 收进 `moe_mlp_layers`、其 `experts` 收进 `moe_layers`（`L1802-L1806`），最后用**最后一个** MoE 当样例（`L1804`、`L1808`）。

> **V4.1 特有细节**：上游注释写的是"Pick last one layer since the first ones may be dense layers"（`L1803`），但 V4.1 的 40 层全是 MoE，"最后一层"与"任意一层"等价，样例取到的是 layer 39。`num_expert_groups` 取 `getattr(config, "n_group", 1)`（`L1793`），本 config 无 `n_group` → 归一化为 1。

### 5.2 运行时映射：log2phy 与 update_expert_map

物理专家的落地在 `AscendRoutedExperts`（[routed_experts.py](../../../ops/fused_moe/routed_experts.py#L390)）：

| 机制 | 说明 | 行号 |
|------|------|------|
| `ascend_expert_map` / `log2phy` | 逻辑专家 → 本地物理槽的映射表，初始 None | `L430-L431` |
| `init_eplb(n_shared_experts)` | 构建映射；仅在非 v2 runner 下调用 | `L482-L489`、`L435-L436` |
| `dynamic_eplb` | 只有 `log2phy` 非空才允许动态重排 | `L535` |
| `get_expert_weights()` | 把各专家权重视图暴露给 EPLB 管理器（要求 weak-contiguous、首维等于 `local_num_experts`），冗余槽搬移靠这些视图 | `L447-L480` |
| `update_expert_map(...)` | 无参走上游 `super().update_expert_map()`；带参同时更新 `ascend_expert_map` 与 `expert_map_manager._expert_map` | `L597-L603` |
| 生效点 / 读数 | 选完专家后立即 `topk_ids = self.log2phy[topk_ids]`；`get_log2phy_map()` 供 adaptor 读取（[vllm_adaptor.py](../../../eplb/adaptor/vllm_adaptor.py#L141) `L141-L143`） | `L641-L642`、`L605-L606` |
| 热度采集 / 强制均衡 | `moe_load` 累计、`clear_moe_load()` 清零（`L429`、`L612-L617`）；`enable_force_eplb` 走 `get_force_eplb_topk`，profile run 用随机 topk（`L670-L678`、`L696-L697`） | — |

> **热采集的代价**：`moe_load` 是每层一个 device tensor，每次 forward 都要累加，等于给每个 MoE 层增加一次小的 device 写。只在需要重排的窗口期开启，长期常开会稳定吃掉几个百分点。

约束：`n_physical_experts % ep_size == 0` 不成立时 `n_local` 会被截断、`physical_expert_start/end` 与实际槽位错位（`L191`）；vLLM 侧与 Ascend 侧冗余数冲突会在补丁层抛 `ValueError: Conflicting EPLB redundant expert counts`（patch_fused_moe.py `L145-L148`）；量化方法未实现 `get_eplb_weight_views` 则抛 `NotImplementedError`（routed_experts.py `L451-L456`）。

---

## 六、权重加载与命名映射

**文件位置**: `_load_model_weights` [model.py L1842-L2070](../model.py#L1842)。checkpoint 用训练侧命名（`ffn.gate_proj`、`w1` …），运行时是 vllm 命名（`mlp.experts.*.gate_up_proj`）。

### 6.1 名字重写流水线（`L1870-L1903`）

| 规则 | 行号 | 说明 |
|------|------|------|
| `w1/w2/w3` → `gate_proj/down_proj/up_proj` | `L1879-L1884` | **专家权重的第一道重映射**，必须在 stacked/expert 映射之前 |
| 补 `model.` 前缀 / `model.head.`→`lm_head.` | `L1876-L1877`、`L1886-L1889` | 张量前缀与输出头 |
| `.ffn.` → `.mlp.` / `.attn.` → `.self_attn.` | `L1896-L1897`、`L1894-L1895` | MoE 模块前缀；attn 只在缺 `self_attn` 时补 |
| `.scale` → `.weight_scale` | `L1902-L1903` | **专家量化 scale 的约定**（fp4 的 E8M0 scale 也走这条） |
| `rotary_emb.inv_freq` 跳过 | `L1905-L1906` | 由 rope 参数现算 |

### 6.2 gate_up_proj 堆叠

非专家线性层（含非 mix_placement 的共享专家）走 `stacked_params_mapping`（`L1844-L1847`）：`("gate_up_proj","gate_proj",0)` 与 `("gate_up_proj","up_proj",1)`，分别以 `shard_id=0/1` 写进同一个 `MergedColumnParallelLinear` 的两段（`L1933-L1966`）。

**顺序上有个必须先处理的坑**：`mlp.experts.` 的名字若先被这里改成 `gate_up_proj`，到专家映射阶段会拼出 `gate_gate_up_proj`，所以代码在改名之前先判 `if ("mlp.experts." in name) and name not in params_dict: continue`（`L1943-L1944`），把专家权重留给下面的 `expert_params_mapping`；mix_placement 时共享专家也不走 stacked（`L1945-L1946`）。

### 6.3 专家映射与 scale

`expert_params_mapping` 由 `fused_moe_make_expert_params_mapping` 生成（`L1851-L1858`），`num_experts` 为 `n_routed_experts + (n_shared_experts if fuse_shared_experts else 0)`（`L1856`），并传 `num_redundant_experts=self.num_redundant_experts`（`L1857`）。四元组 `(param_name, weight_name, expert_id, shard_id)` 里，`expert_id` 决定写哪个专家槽、`shard_id` 决定 gate/up 的哪一半（`L2010-L2038`）。三个数量细节：① mix_placement 时 `num_experts` 要**加 1**（共享专家占 `experts.384` 槽）；② 传 `num_redundant_experts` 让映射覆盖冗余槽，否则开 EPLB 后加载会漏；③ 加载返回 `success=False` **不是异常**，只说明该专家不在本 rank（`return_success=True` 的用途，`L2030-L2050`）。`.scale → weight_scale` 必须在映射前完成，否则 scale 张量会被当成普通权重而找不到目标参数。PP 缺失层在三条路上都先判 `is_pp_missing_parameter → continue`（`L1960-L1961`、`L2023-L2024`、`L2061-L2062`），专家权重的 EP 归属由 `expert_id` 决定、TP 切分由 `shard_id` 交给 `weight_loader`（`L2031-L2038`）。

### 6.4 gate 相关三行特殊处理

| 张量 | 处理 | 行号 |
|------|------|------|
| `...ffn.gate.bias_vl` | **原名保留**，直接进 `gate.bias_vl`（注释：作为 hash router 的 vision-only 修正偏置，文本行继续用 tid2eid） | `L1907-L1911` |
| `...ffn.gate.bias` | 改名为 `gate.e_score_correction_bias` | `L1912-L1913` |
| hash 层仍带 `.gate.e_score_correction_bias` | 目标参数不存在 → `continue` 跳过，避免 KeyError | `L1918-L1919` |

> V4.1 全是非 hash 层，第三条对主干不触发（draft 与某些转换产物仍可能带这类张量，代码保留是必要的防御）。**与 V4 相比**：V4 前 3 层是 hash 层，这条跳过逻辑在 V4 上是热路径。

### 6.5 mix_placement 的槽位合成

`is_fusion_moe_shared_experts_layer = fuse_shared_experts and ("mlp.shared_experts" in name)`（`L1931`）。命中后按 `num_chunks = getattr(config, "n_shared_experts", 1) or 1`（`L1980`）切片并断言可整除（`L1986-L1988`）：gate/up 按 dim 0 切、down 按 dim 1 切（`L1984`），把 `mlp.shared_experts.*` 合成 `mlp.experts.{384+j}.*`（`L2000-L2005`）再交给 `expert_params_mapping`（`L2010-L2044`）。末尾 `if not is_fusion_moe_shared_experts_layer: loaded_params.add(name)`（`L2067-L2068`）说明这条路径的"已加载集合"由内层维护，别在外层重复加。

### 6.6 DSpark（draft）的 MoE 加载差异

draft 层不是"缩小版主干"，它有独立的专家宽度。桥接发生在 [patch_speculative_config.py](../../../patch/platform/patch_speculative_config.py#L186)：

```python
draft_updates = {
    "n_routed_experts": draft_text_config.dspark_n_routed_experts,       # 128  (L193)
    "n_mtp_layers": getattr(draft_text_config, "num_nextn_predict_layers", 3),  # 3  (L194)
}
draft_text_config.update(draft_updates)                                   # L198
```

（`num_experts_per_tok` 也在 `L196-L197` 被改写成 `dspark_num_experts_per_tok`=3。）注释（`L188-L190`）说明这是刻意的：**target 与 draft 的专家宽度不同**，且改的是 `SpeculativeConfig` 私有的 config 副本，不会污染 target；与 reference 的 `get_moe_config`（`L142-L149`：`layer_id >= n_layers` 时返回 `dspark_n_routed_experts`）语义一致。

加载侧（dspark.py `L541-L618`）有三处差异：`.scale → .weight_scale` 同样重写（`L559-L560`）；视觉路由偏置 `e_score_correction_bias_vl` **故意忽略**，因为 draft 只在文本解码时跑，若 alias 成文本偏置"开投机解码就会改变文本路由"（`L568-L575`）；专家权重走自己的 `expert_mapping`（`num_experts=self.config.n_routed_experts`，已被 patch 改成 128，`L379-L387`、`L515`、`L577-L595`），非专家权重才走 stacked 且要求 `model.layers.` 前缀（`L597-L607`）。

---

## 七、性能要点与常见坑

| # | 坑 / 要点 | 现象与对策 | 依据 |
|---|-----------|-----------|------|
| 1 | **vision checkpoint 强迫带 `input_ids`** | `bias_vl is not None` 使 sqrtsoftplus 分支要求 `input_ids`，缺失直接 `ValueError`；纯文本跑批同样要透传 token id | fused_topk_router.py `L160-L162`；vl_model.py `L53-L54` |
| 2 | **`renorm` 被写死为 0** | 融合路径忽略 `renormalize`，与参考实现的 `weights /= sum` 可能有末位差；数值对不齐先查这里 | fused_topk_router.py `L215-L216`；reference `L826-L827` |
| 3 | **序列并行的集合通信次数** | 每层 MoE 输出 1 次 `sp_all_gather`（`L324`）、attention 前 gather 后 reduce_scatter（`L898`、`L901`）；另 `L326-L329` 可能补一次 TP all-reduce | `L280-L283`、`L323-L329` |
| 4 | **重复分片** | DecoderLayer 已传 `already_sequence_parallel=True`（`L917`），MoE 不再 `sp_shard`；自定义调用方忘传该标志会多做两次分片 | `L270`、`L280-L283` |
| 5 | **`hidden_states_fp32` 缺失** | 退回 bf16 hidden 当路由输入（`L287`），精度与性能都变差；该张量由 `rms_norm_cast` 产出（`L912`） | `L274-L275`、`L912-L918` |
| 6 | **40 层全 MoE 的同步点** | 每层至少一组 dispatch/combine 同步；`_fused_output_is_reduced`（fused_moe.py `L272-L284`）决定是否再 all-reduce，配错就是每层多一次集合通信 | fused_moe.py `L272-L284` |
| 7 | **shared expert 融合开关不匹配** | `mix_placement` 改了却不重载权重 → `mlp.experts.384` 无权重、shared 又不在，输出直接错 | `L1931`、`L1979-L2005` |
| 8 | **fp16 与 bf16 融合公式不等价** | fp16 用 `1/1.5` 反向缩放（`L316-L319`），与 bf16 的 `×1.5` 差一个量级；混跑 dtype 实验时别当同一路径 | `L309-L319` |
| 9 | **EPLB 热采集常开成本** | `moe_load` 每层每次 forward 累加；`dynamic_eplb` 还要求 `log2phy` 非空（`L535`），否则开了也不生效 | routed_experts.py `L429`、`L535`、`L612-L617` |

---

## 八、一句话总结

DeepSeek V4.1-Flash 的每一层 FFN 都是"**1 个全量共享 MLP + 384 选 6 稀疏专家**"的 MoE：**40 层全部如此，没有稠密 FFN，也没有 hash 层**（`num_hash_layers` 缺失、归一化为 0，`self.hash` 恒 False，`tid2eid` 恒 None）；路由用 **fp32 门控 + `sqrt(softplus)` + noaux_tc 偏置**（偏置只影响选择、不进权重），归一化后按 `routed_scaling_factor=1.5` 缩放；vision checkpoint 额外带 `bias_vl`，对 `[129264, 129269)` 的图像 sentinel 行改用视觉偏置选专家，因此**每层 MoE 都必须拿到原始 `input_ids`**；专家计算走 W4A8（权重 fp4、激活动态 fp8、`swiglu_limit=10.0` 限幅）的分组矩阵乘，共享专家以 W8A8 在另一条 stream 上与路由专家重叠执行，最终由 runner 内 `shared + routed` 一次相加；EPLB 用"逻辑→物理"的 `log2phy` 映射在 EP 维度均衡负载，draft（DSpark）层则被单独收窄为 **128 选 3**。
