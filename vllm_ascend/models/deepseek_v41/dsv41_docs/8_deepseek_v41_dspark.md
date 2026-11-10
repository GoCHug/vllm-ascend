# DeepSeek V4.1 DSpark 稀疏块草稿模型详解

> **本文讲解 DeepSeek-V4.1-Flash 的 DSpark（DeepSeek Sparse Block Drafter）在 Ascend 上的实现：`mtp.*` 下 3 个串行 draft block、37/38/39 层残差流融合、3 层共享一次 GEMM 的上下文 KV 预计算、块内非因果 SWA 注意力、延迟 mHC 交接、Markov 加性偏置、Confidence 接受率与 fp8 权重重映射。**

> **冻结声明**：基于本仓库当前 checkout 的 vllm-ascend 源码。Ascend 实现 [dspark.py](../dspark.py)（**662 行**），依赖 [model.py](../model.py)；训练语义见 [reference/model.py](reference/model.py)；配置见 [config.json](0_DeepSeek-V4.1-Flash-1017_config.json)；权重命名按 [权重索引](0_DeepSeek-V4.1-Flash-1017_model_weights.index.json)（**96085** 张量）实测。关键位置：`_apply_dsv4_rope` [L61-L75](../dspark.py#L61)、`_get_dspark_num_mtp_layers` [L78-L82](../dspark.py#L78)、`DeepseekV41DSparkSWACache` [L85-L107](../dspark.py#L85)、`DeepseekV41DSparkAttention` [L109-L142](../dspark.py#L109)、`DeepseekV41DSparkModel` [L151-L387](../dspark.py#L151)、入口类 [L390-L662](../dspark.py#L390)、`_remap_dspark_name` [L419-L431](../dspark.py#L419)、`load_weights` [L509-L621](../dspark.py#L509)、`_remap_checkpoint_name` [L623-L662](../dspark.py#L623)。行号已逐条核对。

> **统一模型配置**（`text_config` 中 DSpark 相关行）：

| 参数 | 值 | 含义 |
|------|-----|------|
| `hidden_size` | **5120** | 残差流宽度（V4 为 7168） |
| `hc_mult` / `hc_eps` | 4 / 1e-6 | mHC 流数 / 门控下限 |
| `num_hidden_layers` | **40** | 主模型层数，draft 层号从 40 起 |
| `vocab_size` | 129280 | full-vocab draft，无 d2t |
| `head_dim` / `qk_rope_head_dim` | 512 / 64 | MLA 头宽 / RoPE 分量 |
| `num_attention_heads` / `o_groups` | 64 / 8 | 头数 / o_proj 分组 |
| `sliding_window` | 128 | SWA 窗口；draft 只走 SWA |
| `num_nextn_predict_layers` | **3** | 重写为 `n_mtp_layers=3` |
| `dspark_block_size` | **5** | 一次前向的 draft token 数 |
| `dspark_noise_token_id` | **128799** | 非锚点位置的噪声占位 id |
| `dspark_target_layer_ids` | **[37,38,39]** | 融合这三层入口残差流 |
| `dspark_markov_rank` | **256** | Markov 瓶颈；confidence 输入 5120+256 |
| `dspark_n_routed_experts` / `_per_tok` | **128 / 3** | draft MoE（主模型 384 / 6） |
| 运行时 draft 层号 | **40 / 41 / 42** | `mtp.0/1/2` → `model.layers.40/41/42` |
| 量化 | fp8 `[32,32]`、`ue8m0`、专家 fp4 | 决定 `main_proj` 必须 fp8（§3.3） |

---

## 一、DSpark 是什么 — 入门篇

### 1.1 生活化类比

> **逐字猜 vs 一口气猜一句**：**串行 MTP** 像逐字猜——你说一个字它猜下一个，猜完喂回去再猜（[7_deepseek_v41_mtp.md](7_deepseek_v41_mtp.md)）。**DSpark** 像一口气猜一整句：先把主模型最近三层"思考过程"（37/38/39 入口残差流）揉成一条上下文，**一次前向**写出整段草稿（5 token），块内 5 个位置互见（非因果），再用 Markov 头从左到右补词与词的搭配，最后交主模型批量验证。
>
> 为什么"稀疏"？draft 层只留一块滑窗 KV（SWA），不做长上下文压缩——3 层都是 `compress_ratio=0` 的纯 SWA 层（[L121-L137](../dspark.py#L121) 的合成 role）。

### 1.2 与串行 MTP 的差别

| 维度 | 串行 MTP | DSpark |
|------|----------|--------|
| 粒度/前向 | 1 token/步，每 token 一次 | 1 个 5-token block，整块一次 |
| 块内可见性 | 因果 | **非因果**（§5.2） |
| 上下文 | 上一步残差流 + 下一 token 嵌入 | 37/38/39 入口 `mean(hc)` → `main_proj` |
| 上下文 KV | 无预算 | `context_wkv_proj` 一次 GEMM（§4） |
| 组件数 | 1 个 `mtp` 层 | **3 个串行 block**（40/41/42） |
| 词表 | 可能 draft-vocab + d2t | full-vocab，映射恒等（[L466-L467](../dspark.py#L466)） |
| 输出头 | `hc_norm` + `lm_head` | `hc_collapse`/`hc_head` + `norm` + `lm_head` |
| 偏置/置信度 | 无 | Markov 加性偏置 + Confidence 接受率 |

### 1.3 三个 block 的定位

权重挂在目标 ckpt 的 `mtp.*` 下，`num_nextn_predict_layers=3` 即 3 个 block：`mtp.0`→`layers.40`（额外挂 `main_proj/main_norm`）、`mtp.1`→`layers.41`（纯 block）、`mtp.2`→`layers.42`（额外挂 `norm/markov_head/confidence_head`）。三者**串行**执行（[L173-L183](../dspark.py#L173) 的 `ModuleDict` + [L294-L302](../dspark.py#L294) 插入序遍历），共享同一份 `main_x` 与同一套预计算 KV。

```
aux(37/38/39 mean hc) → main_proj+main_norm → main_x ─┐
context_states → context_wkv_proj → RoPE → SWA cache ─┼→ layers.40 → 41 → 42
input_ids[cur,128799×4] → embed → repeat(hc=4) ───────┘        │
              hc_collapse → norm → lm_head → markov 顺序采样 ×5 → confidence → (B,5)
```

### 1.4 DSpark / DFlash / MTP：三个名字别混

初读这套代码最容易在这里迷路——**同一个东西在不同层叫不同名字**，而另一个真·不同的东西又频繁出现在 DSpark 的代码路径里。先把账算清：

| 层面 | 名字 | 出处 |
|------|------|------|
| checkpoint 里的权重树 | **`mtp.*`** | 历史命名，直接复用 MTP 的目录；V4.1 的 `mtp.0/1/2` 里放的是 DSpark 的 `main_proj` / `markov_head` / `confidence_head`，**不是**经典 MTP |
| 上游方法名（用户配置） | **`speculative_config.method = "dspark"`** | 上游把两者定义成**并列的两个方法**：`DFlashModelTypes = Literal["dflash"]`、`DSparkModelTypes = Literal["dspark"]`（[speculative.py:L68-L69](../../../../../vllm/vllm/config/speculative.py#L68)） |
| 上游 draft 架构 | **`DSparkV41DraftModel`** | 注册表 [registry.py:L646-L649](../../../../../vllm/vllm/model_executor/models/registry.py#L646) → `vllm/models/deepseek_v41/`，类名仍叫 `DSparkDeepseekV4ForCausalLM` |
| Ascend draft 模型 | **`DeepseekV41DSparkModel`** / `DSparkDeepseekV41ForCausalLM` | [模型注册](../../../models/__init__.py#L68)、[dspark.py:L151](../dspark.py#L151)、[dspark.py:L390](../dspark.py#L390) |
| Ascend proposer | **`AscendDSparkProposer`**，**继承 `AscendDflashProposer`** | [dspark_proposer.py:L23](../../../worker/v1/spec_decode/dspark_proposer.py#L23) |

两个必须知道的推论：

**① V4.1 没有经典 MTP，用 `method="mtp"` 会被上游直接拒绝。** 上游在自动判定 draft 类型时写了这条错误信息（[speculative.py:L1366-L1376](../../../../../vllm/vllm/config/speculative.py#L1366)）：

> *"DeepSeek V4.1 has no classic-MTP draft: its checkpoints ship DSpark stages under mtp.* (main_proj/markov_head/confidence_head) and carry no e_proj/h_proj/enorm/hnorm/hc_head weights. Use speculative method 'dspark' instead of 'mtp'."*

也就是说：**`mtp.*` 只是权重目录名，机制是 DSpark**；看到 "mtp" 就以为走的是 [7_deepseek_v41_mtp.md](7_deepseek_v41_mtp.md) 里的串行 MTP，是错的（那篇讲的是**调度与隐态采集**，机制仍是 DSpark 块草稿）。

**② Ascend 的 DSpark proposer 是 DFlash proposer 的子类。** 它自己的类注释把关系讲得最清楚（[dspark_proposer.py:L23-L28](../../../worker/v1/spec_decode/dspark_proposer.py#L23)）：

> *"DSpark uses vLLM's `mtp` method in user config, but its execution shape is closer to DFlash: target hidden states prepopulate draft K/V, then one anchor-first query block emits all speculative tokens."*

所以 DSpark 路径里出现 `dflash_proposer.py`、`_dflash_hidden_states`、`copy_and_expand_dflash_and_dspark_inputs_kernel`（[dspark_proposer.py:L18-L19](../../../worker/v1/spec_decode/dspark_proposer.py#L18)）这些名字，**不是**跑错了方法：块草稿的"锚点在前、K 个块内 query 一次出完、上下文先预填 KV"这套骨架由 DFlash 提供，DSpark 只在其上加 **Markov 头 / Confidence 头 / DSA 稀疏注意力的上下文 KV 预计算**（§4、§6、§8）。

**③ DFlash 是另一个（兄弟）方法**，服务于别的模型：`DFlashDraftModel`（Qwen3）、`DFlash2DraftModel`（Qwen3-DFlash2）、`DFlashLagunaForCausalLM`（Laguna）、Muse Glimmer 的 draft head（[registry.py:L634-L656](../../../../../vllm/vllm/model_executor/models/registry.py#L634)）。它没有 Markov/Confidence 头，也不做 DSA 感知的上下文 KV。**V4.1 用不到它**，只是复用了它的 proposer 骨架。

> 一句话：**方法叫 DSpark，权重目录叫 `mtp.*`，proposer 骨架来自 DFlash。** 三个名字指的不是三件事，但也不能互换。

---

## 二、模块层级总览 — 基础篇

### 2.1 类图

```
DSparkDeepseekV41ForCausalLM (L390-L662; SupportsEagle3, DeepseekV41MixtureOfExperts)
├── lm_head: ParallelLMHead(129280,5120) L407-L411（不切分）  logits_processor L412
└── model: DeepseekV41DSparkModel L151-L387
    ├── embed_tokens L167-L172
    ├── layers: ModuleDict{"40","41","42"} L173-L183
    │   └── DeepseekV41DSparkDecoderLayer L145-L148（只换 attention_cls）
    │       ├── self_attn: DeepseekV41DSparkAttention L109-L142
    │       │   ├── swa_cache_cls = DeepseekV41DSparkSWACache L112 → L85-L107
    │       │   ├── packed_cache_ops（A5 ABI 或 None）L116
    │       │   └── v41_impl + 合成 LayerRole L121-L137；注册 static ctx L138-L140
    │       ├── mlp: DeepseekV41MoE(is_draft_layer=True) model.py:L811-L817
    │       └── hc_{attn,ffn}_{fn,base,scale} fp32 model.py:L826-L831
    ├── context_wkv_proj L200-L208        ├── main_proj L223-L231
    ├── main_norm / norm L232 / L236      ├── _oproj_buffer_pool L184-L190
    ├── markov_head DSparkMarkovHead(V,V,256) L238-L243
    └── confidence_head DSparkConfidenceHead(5120+256) L244-L249
```

四处挂载：`main_proj/main_norm` 同时**赋给第 40 层**（[L233-L234](../dspark.py#L233)）；`norm/markov_head` 同时**赋给第 42 层**（[L250-L252](../dspark.py#L250)）；`confidence_head` **只挂 model 级**（[L244-L249](../dspark.py#L244)，由入口类调用）；每层另有 mHC 参数（`model.py:L826-L831`），故能直接复用主模型层。

### 2.2 `mtp.*` ↔ 运行时 40/41/42

| checkpoint 前缀 | 运行时 |
|-----------------|--------|
| `mtp.0.main_proj.*` / `main_norm.weight` | `model.layers.40.*`（fp8，实测有 `main_proj.scale`） |
| `mtp.0.attn.*`→`.self_attn.*`、`mtp.0.ffn.*`→`.mlp.*` | `layers.40.*`（[L651](../dspark.py#L651)；`w1/w2/w3`→`gate/down/up`） |
| `mtp.1.*` | `layers.41.*`（纯 draft block） |
| `mtp.2.norm.weight` | `layers.42.norm.weight`（[L644-L645](../dspark.py#L644)） |
| `mtp.2.markov_head.embed/head.weight` | `layers.42.markov_head.markov_w1/w2.weight`（[L428-L429](../dspark.py#L428)） |
| `mtp.2.confidence_head.proj.weight` | `model.confidence_head.proj.weight`（[L630-L631](../dspark.py#L630)） |
| 根 `embed.weight` / `head.weight` | `model.embed_tokens.weight` / `lm_head.weight`（[L542-L545](../dspark.py#L542)） |
| `hc_head_fn/base/scale` | `model.*`（V4 旧名兼容，[L546-L547](../dspark.py#L546)） |

### 2.3 参数账本（用索引验算）

实测：stage 0 = **801**、stage 1 = **798**、stage 2 = **802** 张量。基线 `798` = mHC 6 + attention 13 + 两个 norm 2 + shared_experts 6 + **128 专家 × 6 = 768** + gate 3（`model.py:L397-L446`、`L818-L819`、`L202-L211`、`L179-L222`）。stage 0 多 3 个（`main_proj.weight/.scale`+`main_norm.weight`），stage 2 多 4 个（`norm.weight`+markov 两矩阵+`confidence_head.proj.weight`）。`768=128×6` 是"draft MoE 只有 128 专家"最硬的证据；stage 2 只多 4 个张量说明三个输出头**全在最后一层**。索引里**没有** `mtp.*.embed/head.weight`、也没有 `hc_head_*`——这两个分支是给独立 draft ckpt / V4 旧权重留的。

### 2.4 骨架常量与工具类

**`_apply_dsv4_rope`（[L61-L75](../dspark.py#L61)）**：按 `rotary_emb.layername` 取 cos/sin（draft 层 `layername=f"{prefix}.attn"`，`model.py:L611-L613`），`inverse=True` 时把 **sin 取负**。Ascend 只在 `_project_shared_kv` 正向调用（[L324](../dspark.py#L324)），输出侧逆旋转由 DSA 融合算子内部完成（语义见 [reference/model.py:L1070](reference/model.py#L1070)）。

**`_get_dspark_num_mtp_layers`（[L78-L82](../dspark.py#L78)）**：`n_mtp_layers` > `dspark_num_mtp_layers` > 3。config 只有 `num_nextn_predict_layers=3`，由 [patch_speculative_config.py:L191-L198](../../../patch/platform/patch_speculative_config.py#L191) 重写为 `n_mtp_layers=3`（同时把 draft 专家数改成 128/3），故 `num_dspark_layers=3`、`mtp_start_layer_idx=40`。

**`DeepseekV41DSparkSWACache`（[L85-L107](../dspark.py#L85)）**：只做一件事——换 KV cache spec 的类，让 runner 识别 draft 的 SWA cache。`get_kv_cache_spec` 重建 `AscendSlidingWindowMLASpec`（[L92-L103](../dspark.py#L92)）且**不带** `bounded_replay`（目标侧才带，`model.py:L633-L637` 称其为"对目标 group 的严格追加"）；backend 固定 `DeepseekV41CacheBackend`（[L105-L106](../dspark.py#L105)）。

**`DeepseekV41DSparkAttention`（[L109-L142](../dspark.py#L109)）** 四件事：① 换 `swa_cache_cls`（[L112](../dspark.py#L112)）；② 抓设备算子表 `get_dsv41_packed_cache_ops()`（[L116](../dspark.py#L116)；非 A5 为 `None`（`device_op.py:L52-L54`），A5 为 `MixedQuantPackedCacheOps`（`L1068-L1071`））；③ 自建 `v41_impl` 并**合成** `DeepseekV41LayerRole`——`compress_ratio=0`、source 全 `None`、`engram_slot=None`、`topology=None`（[L121-L137](../dspark.py#L121)），即 **draft 层不参与 V4.1 的 kv/index 源共享拓扑**；注意 `layer_idx=int(prefix.split(".")[-2])` 取到的是 stage id（0/1/2）而非 40/41/42；④ 注册进 `static_forward_context`（[L138-L140](../dspark.py#L138)）、`forward` 绑成 `DeepseekV41Attention.forward`（[L142](../dspark.py#L142)），走 `torch.ops.vllm.dsa_v41_forward`（`model.py:L762-L764`）。

---

## 三、主模型隐藏状态融合 — 基础篇

### 3.1 为什么取 37/38/39

- **参考实现**：在**进入**第 i 层前 append `h.mean(dim=2)`（该层注意力输入），最后 `cat` 成 `main_hidden`（[reference/model.py:L1267-L1274](reference/model.py#L1267)）。
- **Ascend**：runner 按 **one-based** 表达 ckpt 层号，`_get_eagle3_aux_layers_from_config` 返回 `tuple(i+1)` = **(38,39,40)**（[model_runner.py:L823-L838](../../../worker/v1/model_runner.py#L823)）；主循环判 `layer.layer_idx + 1 in aux_hidden_state_layers`，命中即在调用该层前取 `hidden_states.mean(dim=1)`（[model.py:L1577-L1583](../model.py#L1577)）。注释原文：*"DSpark consumes the residual stream entering its configured target layers. The runner expresses checkpoint IDs as one-based."*

两者都取"进入 37/38/39 层的残差流"并对 4 条 mHC 流**取均值**，故每层只贡献 `(T,5120)`、拼接为 `(T,15360)`——因为 `main_proj` 输入宽度固定为 `hidden_size × len(ids)`（[L224](../dspark.py#L224)），给 4 条会变 61440。取这三层因为离输出最近且**不改主模型结构**（只加 aux 采集点）。

### 3.2 `main_proj` + `main_norm`

```python
self.main_proj = ColumnParallelLinear(
    config.hidden_size * len(self.target_layer_ids),   # 15360 → 5120
    config.hidden_size, bias=False, return_bias=False,
    quant_config=main_proj_quant_config,               # §3.3
    prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx}.main_proj"),
    gather_output=True)                                # 每卡拿到完整 5120
self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)   # eps=1e-20
first_layer.main_proj, first_layer.main_norm = self.main_proj, self.main_norm
```

（[L223-L234](../dspark.py#L223)）`gather_output=True` 是关键：下游 3 个 block 的 attention 都要完整 `main_x`。融合只算一次（[L312-L313](../dspark.py#L312)），3 个 block 复用；模块挂第 40 层但**不在层内调用**，由入口类暴露给 proposer（[L494-L495](../dspark.py#L494)、[llm_base_proposer.py:L1059-L1065](../../../worker/v1/spec_decode/llm_base_proposer.py#L1059)）。

### 3.3 fp8 `main_proj` 的坑

`main_proj_quant_config` 仅在 **A5（`get_dsv41_packed_cache_ops() is not None`）且 ckpt（组合 VL config）的 `quantization_config.quant_method == "fp8"`** 时取 `vllm_config.quant_config`，否则 `None`（[L210-L222](../dspark.py#L210)）。注释结论：*"Its DSpark main projection is **FP8 in the release checkpoint**, so constructing a BF16 linear **drops the scale parameter** and makes the draft loader fail on `main_proj.scale`."*

| 场景 | 量化配置 | 后果 |
|------|----------|------|
| A5 + ckpt fp8 | fp8 | 参数集含 `main_proj.weight_scale`，`mtp.0.main_proj.scale` 正常落位 |
| 非 A5 | `None`（bf16） | 无 scale 参数；`mtp.0.main_proj.scale` 经 `.scale→.weight_scale`（[L559-L560](../dspark.py#L559)）后查不到，[L615-L617](../dspark.py#L615) 抛 `KeyError` |

判断依据是**组合（VL）config** 上的 `quantization_config`，不是 `hf_text_config`（后者为 `None`）；条件里的 `packed_cache_ops is not None` 是"A5 才支持这套 fp8 组合"的显式门禁。

**shape 表**

| 张量 | 出处 | shape |
|------|------|-------|
| `aux_hidden_states` | `model.py:L1580` → `combine_hidden_states` | `(T,5120)` → `(T,15360)` |
| `main_proj.weight` | `mtp.0.main_proj.weight` | `(5120,15360)` + `weight_scale` |
| `main_x` | `main_norm(main_proj(aux))` | `(T,5120)` |

---

## 四、上下文 KV 预计算 — 进阶篇

### 4.1 `context_wkv_proj`：一次 GEMM 打完 3 层

```python
self.context_wkv_proj = MergedColumnParallelLinear(
    config.hidden_size, [config.head_dim] * self.num_dspark_layers,   # [512,512,512]
    bias=False, return_bias=False, quant_config=vllm_config.quant_config,
    prefix="mtp.0.self_attn.wkv", disable_tp=True)
```

（[L200-L208](../dspark.py#L200)）① **合并**——draft 的 `wkv` 是 `ReplicatedLinear(5120→512)`（`model.py:L417-L424`，不做 TP 切分），三层合并成 `(1536,5120)` 的一次 GEMM，比 3 次 5120×512 少 2 次算子启动；② **为何能共享**——上下文路径喂给每层的 `context_states` 完全相同（[L195-L199](../dspark.py#L195)），三层只是权重不同；③ **`disable_tp=True`**——源权重是复制的，按 TP 切会切错布局；④ **`prefix="mtp.0.self_attn.wkv"`**——故意借第一层真实前缀，让 ModelSlim 为合成模块选中与 ckpt 一致的 WKV 量化方案。

### 4.2 为什么还要保留每层 `wkv`

`_duplicate_context_wkv_weights`（[`../../deepseek_v4/dspark.py:L86-L105`](../../deepseek_v4/dspark.py#L86)，导入 [L47](../dspark.py#L47)、调用 [L539](../dspark.py#L539)）把每层 `mtp.{i}.attn.wkv.*` **额外**产出一份并打 `shard_id=i`：原权重留给 draft token 单层前向，副本按 shard 装进合并投影。于是显存里同时有每层 `wkv.weight` `(512,5120)`×3 与 `context_wkv_proj.weight` `(1536,5120)`。代价是"每 TP rank 多一份拷贝 + 量化元数据"（[L535-L538](../dspark.py#L535) 原话 *trading bounded memory for a single context projection*）。

### 4.3 `precompute_and_store_context_kv` 的 slot 语义

（[L327-L354](../dspark.py#L327)）`all_kv = context_wkv_proj(context_states).view(T,3,512)` → 若 `context_slot_mapping is None` 直接 return → 否则按 3 个 `layername` 一次取全 cos/sin（[L346-L347](../dspark.py#L346)）→ 逐层 `_project_shared_kv` + `_store_standard_swa_kv`。

- `context_states` 是**主模型输入隐藏状态**（不是 `main_x`），宽 5120；
- `context_slot_mapping` 是**长度 3 的列表**，顺序由 proposer 的 `self._layer_group_idx` 决定（[dspark_proposer.py:L314-L316](../../../worker/v1/spec_decode/dspark_proposer.py#L314)），**不是** `ModuleDict` 插入序；每项可 `None`（该层不在本 batch 的 cache group），整体可 `None`（profiling）；
- `_project_shared_kv`（[L315-L325](../dspark.py#L315)）：`kv_norm → split(nope=448,rope=64) → k_pe 上 RoPE → cat → view(-1,1,512)`，避免 `split→rope→cat→contiguous` 的冗余拷贝；**量化不在这步**；
- slot 物理含义（[ops/triton/spec_decode/utils.py:L246-L252](../../../ops/triton/spec_decode/utils.py#L246)）：`slot = block_id*block_size + pos%block_size`，非本 DCP rank 行为 `-1`。

### 4.4 写缓存两条路：A5 packed vs 2-D scatter

（[L254-L277](../dspark.py#L254)）

| 路径 | 触发 | slot 形态 | 落地算子 |
|------|------|-----------|----------|
| A5 packed（[L259-L269](../dspark.py#L259)） | `packed_cache_ops is not None` | **保持 1-D 线形物理 slot** | `kv_compress_epilog_v2`（`mixed_quant_attention.py:L326-L334`），`kind="win"`→`group_size=32`、`mxfp8_bf16`、`round_scale=True`（L320-L323），写入时顺带量化 |
| 通用（[L270-L277](../dspark.py#L270)） | 其它 | 1-D → `(block_idx,block_off)` **2-D int32**，非法行 `(-1,-1)` | `npu_scatter_nd_update_sk`（`dsa_v41.py:L278-L293`），`(-1,-1)` 跳过；`values.to(cache.dtype)`（L292） |

为什么 A5 必须专用 writer（[L260-L262](../dspark.py#L260)）：A5 的 cache **物理上是打包的**，必须由设备端路由的 writer 写入；且"proposer 已提供线形物理 slot"，保持 1-D 交给原生 kernel。

### 4.5 profiling 下"没有 slot 也要跑投影"

[L336-L344](../dspark.py#L336) 注释：*"Profiling omits slot mappings, but must still execute this projection so its workspace is included in the memory profile."* `dummy_run(is_profile=True)` 只传两个参数（[dspark_proposer.py:L416-L417](../../../worker/v1/spec_decode/dspark_proposer.py#L416)），算完 `all_kv` 就 return——让这次大 GEMM 的 **workspace 进入显存 profile**，否则上线易 OOM。

### 4.6 shape 表

| 张量 | 出处 | shape |
|------|------|-------|
| `context_states` / `context_positions` | 主模型输入 | `(T_ctx,5120)` / `(T_ctx,)` |
| `context_slot_mapping[i]` | proposer 每层一个 | `(T_ctx,)` int32，可 `None` |
| `all_kv` / `shared_kv` / `values` | 投影 → RoPE → squeeze | `(T_ctx,3,512)` / `(T_ctx,1,512)` / `(T_ctx,512)` |
| `rope` | `get_cos_and_sin_dsa(layer_names=3)` | dict：3 layername → cos/sin |

---

## 五、三个 Draft 层的前向 — 进阶篇

### 5.1 延迟 mHC 交接（`pre_mix`）

V4.1 的 block 是"延迟 mHC 系数交接"：**每个子层算出的系数给下一个子层用**（参考 `model.py:L970-L996`，Ascend `model.py:L879-L920` 同构）。DSpark 侧：

```python
pre_mix = hidden_states.new_zeros(hidden_states.shape[0], self.hc_mult, dtype=torch.float32)
pre_mix[:, 0] = 1.0                       # one-hot：默认只用第 0 条流
for layer in self.layers.values():        # 40 → 41 → 42
    hidden_states, pre_mix = layer(positions, hidden_states, pre_mix, llama_4_scaling=None, input_ids=input_ids)
hidden_states = last_layer.hc_collapse(hidden_states, pre_mix)   # 用最后一层交出的 pre_mix 收尾
```

（[L291-L304](../dspark.py#L291)；one-hot 与参考 `make_identity_pre_mix`（[reference/model.py:L1162-L1166](reference/model.py#L1162)）一致）`pre_mix` 是**唯一跨层传递的额外状态**（`(T,4)` fp32），把"如何把 4 条流压成 1 条"从层内固定变成"上一层 ffn 决定下一层 attn"：第 `i` 层 `ffn_pre` → 第 `i+1` 层 `hc_pre`，最后一个 block 的 `ffn_pre` 用于最终 `hc_collapse`。mHC 走融合算子 `npu_hc_pre_v3`（`model.py:L858-L869`）与 `npu_hc_post`（`model.py:L871-L877`）。

### 5.2 块内非因果注意力：topk 索引怎么造

**① draft 层恒为纯 SWA**：合成 role 置 `compress_ratio=0`、source 全 `None`（[L121-L137](../dspark.py#L121)）；参考实现直接 `assert self.compress_ratio == 0`（[reference/model.py:L1036](reference/model.py#L1036)）。draft 层不压缩长上下文、不读 `long_kv_cache`/`indexer`（[L134-L136](../dspark.py#L134)）。

**② topk = 整窗 + 整块**（[reference/model.py:L1022-L1031](reference/model.py#L1022)）：

```python
matrix = torch.cat([torch.arange(min(window_size, start_pos + 1)),   # 历史窗口全部行
                    window_size + torch.arange(block_size)])          # 紧跟窗口的整块
return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()
```

因 `kv = cat([window_kv_cache, kv], dim=1)`（[L1068](reference/model.py#L1068)），块 KV 追加在 `[win, win+block)`，故"整块"索引即 `win + arange(block)`。`expand(bsz, block_size, -1)` 让块内**每个 query 看到同一组索引**——这就是"块内非因果"：第 3 个 token 也能看到第 4、5 个。

**③ 运行时关因果 + 放大窗口**（[dspark_proposer.py:L348-L354](../../../worker/v1/spec_decode/dspark_proposer.py#L348)）：无 `get_draft_attn_causal` 时 `cad.causal = False`、`attn_mask = None`、`attn_state = ChunkedPrefill`。窗口放大（[dsa_v1.py:L542-L546](../../../attention/dsa_v1.py#L542)）：left = `sliding_window + num_speculative_tokens - 1` = 128+5-1 = **132**，right = 0；块占 5 个逻辑位置，多留 `block-1` 才能让块首 token 也看到完整 128 行历史。索引宽度按 128 对齐（[L549-L551](../../../attention/dsa_v1.py#L549)，`align128(128+5)=256`）。

真正生成 slot 的是 `build_dspark_swa_indices`（[dsa_v1.py:L554-L584](../../../attention/dsa_v1.py#L554)，docstring：*"Each token in a draft block sees the trailing context window plus the whole current draft block. Invalid/padded rows get lens=0 and -1 slots."*）：`start_pos = clamp(prefix_lens - window_size, 0)`、`visible_lens = seq_lens - start_pos`（尾部窗口 + 本块）、`slot_ids = block_table[block_nums]*block_size + pos%block_size`、`repeat_interleave` 展开成每 token 一行、越界列置 -1（[L613-L637](../../../attention/dsa_v1.py#L613)）。因 proposer 里 `cad.seq_lens = effective_seq_lens + num_query_per_req`（[dspark_proposer.py:L323](../../../worker/v1/spec_decode/dspark_proposer.py#L323)），`visible_lens` 天然含块——与参考的 `window+block` 等价，只是 Ascend 用"每 token 一行 slot + 长度"表达。

| 阶段 | causal | 可见范围 | 出处 |
|------|:------:|---------|------|
| 主模型 prefill/decode | True | 当前位 + 滑窗历史 | 主模型 DSA |
| 上下文 KV 预计算 | 无 query | 只写 KV | [L327-L354](../dspark.py#L327) |
| **draft block 前向** | **False** | 尾部窗口 + **整块互见** | [L348-L354](../../../worker/v1/spec_decode/dspark_proposer.py#L348) |
| **draft 采样** | 顺序（左→右） | 第 i 位用第 i-1 个已采样 token 的 Markov 偏置 | [llm_base_proposer.py:L1580-L1596](../../../worker/v1/spec_decode/llm_base_proposer.py#L1580) |

> 为什么可以"作弊"：draft 只是提案，最终过主模型**因果**验证；双向注意力只提高草稿质量/接受率，不破坏正确性（被拒 token 按目标分布重采样）。

### 5.3 SWA 环的角色

参考实现的窗口缓存是**物理环**：prefill 按 `cutoff = seqlen % win` 卷绕写入（[reference/model.py:L1046-L1054](reference/model.py#L1046)），decode 时 `window_kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)`（[L1067](reference/model.py#L1067)）——写指针取模即"环"；且 **prefill 的 `forward` 只播种窗口 KV 后直接 `return x`**（[L1125-L1129](reference/model.py#L1125)），`forward_spec` 在 `start_pos==0` 时 `return None`（[L1277-L1285](reference/model.py#L1277)）。

Ascend 把"环"交给**分页 SWA 缓存 + slot 映射**：cache 形状由 `AscendDeepseekV41SWACache` 定（spec 带 `window_size`，`model.py:L564-L584`）；写哪里由 `slot_mapping` 决定（§4.4），环回退化为"新 token 覆盖老页"，由 block table 回收保证窗口语义；draft cache 用 `DeepseekV41DSparkSWACache` 单独出 spec（[L92-L103](../dspark.py#L92)），并用 `get_draft_kv_cache_layer_names()` 精确圈层（[L309-L310](../dspark.py#L309)、[dspark_proposer.py:L138-L139](../../../worker/v1/spec_decode/dspark_proposer.py#L138)）。所以 Ascend 侧**没有显式 `% win` 写指针**，等价语义由"132 的可见集合 + 分页 block table"承担。

### 5.4 复用主模型层的代价

收益：mHC、MoE、RMSNorm、DSA 的 TP/量化路径全复用，只换 attention backend。代价：① 每层自带 6 个 mHC 参数（ckpt 里确实有）；② `is_draft_layer=True` 关掉 hash 路由（`model.py:L213`）并禁止 engram（`model.py:L839`），恰符合"draft 不做查表路由"；③ `layer_idx` 由前缀解析得 0/1/2（`model.py:L793`）而非 40/41/42，凡依赖真实层号的逻辑须靠合成 role 关闭；④ SP 下 `wo_b.reduce_results=False`（`model.py:L835-L836`），draft 需自己 all-gather（[L305-L306](../dspark.py#L305)）。

### 5.5 shape 表

| 张量 | 出处 | shape |
|------|------|-------|
| draft `input_ids` | triton kernel（`[cur,noise×4]`） | `(Q,)`，`Q=B×5` |
| `embed_tokens` / `hidden_states` | [L280](../dspark.py#L280) | `(Q,5120)` / `(Q,4,5120)` |
| `pre_mix` / 每层输出 | [L291-L302](../dspark.py#L291) | `(Q,4)` fp32 / `(Q,4,5120)` |
| q / kv（层内） | `wq_b(q_norm(wq_a(x)))` / `kv_norm(wkv(x))` | `(Q,64,512)` / `(Q,512)` |
| `hc_collapse` 输出 | [L304](../dspark.py#L304) | `(Q,5120)` |

---

## 六、输出头：hc_head → norm → lm_head，Markov 加性偏置，Confidence 接受率 — 进阶篇

### 6.1 `hc_head`：本地 sigmoid 门控压流（[L356-L363](../dspark.py#L356)）

```python
x = x.flatten(1).float()                                               # (T, 4*5120)
rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)  # 手写 RMSNorm（全宽统计）
mixes = torch.nn.functional.linear(x, hc_fn) * rsqrt                    # (T, 4)
pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps           # (T, 4) 门控
y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)                 # (T, 5120)
```

语义：把 4 条流 flatten 成 `4×5120` 全宽，在全宽上算手写 RMSNorm（分母是**全宽均值 + `norm_eps`**，不是逐流），用 `hc_fn` 投到 4 个混合分，过 `sigmoid(mixes*hc_scale+hc_base)+hc_eps` 得 4 个门控，加权求和压回 1 条。它是 **DSpark 侧本地实现**：既不同于主模型 `DeepseekV41DecoderLayer.hc_collapse`（`model.py:L854-L856`，纯加权求和、无 sigmoid），也不同于 V4 的逐流归一化。

> ⚠️ **运行时实际路径**：V4.1 `forward` 结尾是 `last_layer.hc_collapse(hidden_states, pre_mix)`（[L304](../dspark.py#L304)），**不是** `self.hc_head(...)`；且 `hc_head` 引用的 `self.norm_eps`/`self.hc_eps` 在本文件 `__init__`（[L154-L252](../dspark.py#L154)）里**从未赋值**——`grep norm_eps` 只命中两处 `RMSNorm(eps=config.rms_norm_eps)` 与 `hc_head` 内部两行。**`hc_head` 是 V4 遗留的等价参考实现，在 V4.1 已是死代码**，直接调用会 `AttributeError`；复现语义请用 `config.hc_eps`(1e-6)/`config.rms_norm_eps`(1e-20)。改用 `hc_collapse` 的原因：延迟 mHC 交接已把"谁压谁"算进 `pre_mix`（§5.1），直接加权求和更省更一致。

### 6.2 `norm` → `lm_head`

`compute_logits`（[L371-L377](../dspark.py#L371)）= `logits_processor(lm_head, self.norm(hidden_states))` → `(Q,129280)`；入口类丢弃 `spec_step_idx`，因为一次出整块（[L469-L479](../dspark.py#L469)）。`norm` 挂**第 42 层**（[L236](../dspark.py#L236)+[L250-L252](../dspark.py#L250)，前缀 `mtp.2.norm.weight`）；`lm_head` 是 `ParallelLMHead(129280,5120)` 且**不按 TP 切分**（[L407-L411](../dspark.py#L407)），权重来自根 `head.weight`（[L544-L545](../dspark.py#L544)）。full-vocab：`compute_draft_logits` 直接返回基础 logits（[L462-L464](../dspark.py#L462)），`map_draft_to_target` 恒等（[L466-L467](../dspark.py#L466)）。

### 6.3 Markov 加性偏置（逐位置递推采样）

`DSparkMarkovHead(vocab=129280, draft_vocab=同, rank=256)`（[L238-L243](../dspark.py#L238)；上游实现导入于 [L33](../dspark.py#L33)）。参考实现（[reference/model.py:L1080-L1089](reference/model.py#L1080)）：`embed=ParallelEmbedding(V,256)`、`head=ParallelHead(V,256)`，即 `bias(next)=W2·W1[prev]` 作为主 logits 的**加性**偏置——rank 仅 256，本质是"低秩 n-gram 转移表"。Ascend 转发：`markov_embed`（[L365-L366](../dspark.py#L365)）、`markov_bias`（[L368-L369](../dspark.py#L368)，需外部传 `logits_processor`）。采样循环（[llm_base_proposer.py:L1580-L1596](../../../worker/v1/spec_decode/llm_base_proposer.py#L1580)）：

```python
logits = raw_logits.view(-1, num_speculative_tokens, vocab)      # 并行前向一次得整块 logits
draft_token_ids[:, 0].copy_(self._dspark_seed_buffer[:num_blk])   # 首列 = 锚点 next_token_ids
for idx in range(num_speculative_tokens):
    markov_emb = self.model.markov_embed(draft_token_ids[:, idx]) # 依赖"前一个已采样 token"
    logits[:, idx].add_(self.model.markov_bias(markov_emb))
    token_ids = 概率采样 or argmax（有 d2t 时贪婪）
    draft_token_ids[:, idx + 1].copy_(token_ids)
```

**backbone 并行、采样串行** = semi-autoregressive drafting。`_dspark_draft_buffer` 形状 `(max_batch_size, 1+K)`（[dspark_proposer.py:L45-L46](../../../worker/v1/spec_decode/dspark_proposer.py#L45)），首列放锚点 `_dspark_seed_buffer`（[L47](../../../worker/v1/spec_decode/dspark_proposer.py#L47)、[L247-L248](../../../worker/v1/spec_decode/dspark_proposer.py#L247)）。概率采样与 vocab 重映射互斥（[L1558-L1566](../../../worker/v1/spec_decode/llm_base_proposer.py#L1558)）；V4.1 full-vocab（`draft_id_to_target_id=None`，[L392](../dspark.py#L392)），概率采样可用。

### 6.4 Confidence 接受率

`DSparkConfidenceHead(input_dim=5120+256=5376)`（[L244-L249](../dspark.py#L244)），入口类套 sigmoid：

```python
def compute_confidence(self, head_hidden, markov_embed):     # L487-L489
    return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))
```

参考实现：`cat([hidden, markov_embed], -1).float()` → `Linear(5376→1, fp32)` → `squeeze(-1)`（[reference/model.py:L1092-L1100](reference/model.py#L1092)，参数存 fp32，注释说明 ckpt 里 proj 是 bf16）。

- **不参与草稿 token 生成**：token 只由 `logits + markov_bias` 决定；
- **用途**：动态验证预算。`DynamicSpecScheduler._compute_dspark_token_probs`（[utils.py:L336-L394](../../../worker/v1/spec_decode/utils.py#L336)）整理成 `[B,D]` 并 `clamp(1e-6,1.0)`，供 `_update_from_token_probs` 决定验证长度；
- **取哪几列**：置信度对应 D 个位置，而 `draft_token_ids` 是 `[B,D+1]`，只用前 D 列（[utils.py:L359-L368](../../../worker/v1/spec_decode/utils.py#L359)；proposer 同款 [dspark_proposer.py:L108-L125](../../../worker/v1/spec_decode/dspark_proposer.py#L108)）；`markov_embs` 需 `.to(flat_hidden.dtype)`。

### 6.5 shape 表

| 张量 | 出处 | shape |
|------|------|-------|
| `head_hidden` / `logits` | `hc_collapse` / `lm_head` | `(N,5120)`（`N=B×K`）/ `(N,129280)` |
| `markov_w1` / `markov_w2` | `mtp.2.markov_head.{embed,head}.weight` | `(129280,256)` ×2 |
| `markov_embed` / `logits_bias` | `markov_embed` / `markov_bias` | `(N,256)` / `(N,129280)` |
| `confidence` | `confidence_head` → `sigmoid` | `(N,)` → `(B,K)` |
| `confidence_head.proj.weight` | `mtp.2.confidence_head.proj.weight` | `(1,5376)` fp32 |

---

## 七、权重加载与改名 — 专家篇

### 7.1 加载主循环（[L509-L621](../dspark.py#L509)）

`expert_mapping`（[L515](../dspark.py#L515)）、`stacked_params_mapping`（4 条 gate/up 规则，[L519-L524](../dspark.py#L519)）、`weights = _duplicate_context_wkv_weights(weights, num_dspark_layers)`（[L539](../dspark.py#L539)）。每个 `(name, loaded_weight)` 依次过 6 道关（顺序不能乱）：

| # | 关卡 | 代码 | 说明 |
|---|------|------|------|
| 1 | embed/head 直连 | [L542-L545](../dspark.py#L542) | 根 `embed/head.weight` → draft 嵌入/输出头，**仅 `not rotation_path`** |
| 2 | `hc_head_*` 直连 | [L546-L547](../dspark.py#L546) | V4 旧名兼容 |
| 3 | `_remap_dspark_name` | [L549-L552](../dspark.py#L549) | 返回 `None` ⇒ 非 mtp 权重，`continue` |
| 4 | eagle 探测 | [L556](../dspark.py#L556) | `process_eagle_weight(self, name)` |
| 5 | `.scale→.weight_scale` | [L559-L560](../dspark.py#L559) | Ascend 量化后缀约定 |
| 6 | 分类分派 | [L562-L618](../dspark.py#L562) | `context_wkv_proj` → `bias_vl` 跳过 → 专家 → 堆叠 → `attn_sink` → 默认 |

分派细节：**`context_wkv_proj.*`** 用合成权重的 `shard_id` 调 `weight_loader(param, w, w.shard_id)`（[L562-L566](../dspark.py#L562)）；**vision router bias** 命中 `endswith(".e_score_correction_bias_vl") and name not in params_dict` 即 `info_once` 跳过（[L573-L575](../dspark.py#L573)），注释（[L568-L572](../dspark.py#L568)）说明：多模态 ckpt 每个 MTP 层都带一份 vision-router bias，但 DSpark 只在文本 decode 跑、draft gate 不暴露 `bias_vl`，**不要 alias 成文本 correction bias，否则开启投机解码会改变文本路由**；**专家**走 `expert_mapping` 的 expert-aware loader（[L577-L595](../dspark.py#L577)）；**堆叠**仅 `model.layers.` 参数，`for...else` 保证未命中才继续（[L597-L607](../dspark.py#L597)）；**`attn_sink`** 手动 narrow（[L609-L614](../dspark.py#L609)）——参数是 `nn.Parameter(n_local_heads)`（`model.py:L397`）而 ckpt 是完整 64 头，`head_start/end` 来自 `n_local_head = num_attention_heads // tp_size`（[L529-L533](../dspark.py#L529)）；**默认** `getattr(param,"weight_loader",default_weight_loader)`（[L615-L618](../dspark.py#L615)）；结束打 `info_once("DSpark draft model loaded: %d params")`（[L620](../dspark.py#L620)）。

### 7.2 `_remap_dspark_name`（[L419-L431](../dspark.py#L419)）

`context_wkv_proj.*` 直接加 `model.` 前缀（名字无 `mtp.`，`_remap_checkpoint_name` 会返回 `None`）；其余先过 `_remap_checkpoint_name`，再做三条替换：

| ckpt 名（实测） | 替换后 |
|------------------|--------|
| `mtp.2.markov_head.embed.weight` | `model.layers.42.markov_head.markov_w1.weight`（[L428](../dspark.py#L428)） |
| `mtp.2.markov_head.head.weight` | `model.layers.42.markov_head.markov_w2.weight`（[L429](../dspark.py#L429)） |
| `mtp.2.confidence_head.proj.weight` | 已带 `.proj`，`confidence_head.weight`→`.proj.weight` 规则（[L430](../dspark.py#L430)）不触发 |

### 7.3 `_remap_checkpoint_name` 的 stage/offset 规则（[L623-L662](../dspark.py#L623)）

`re.match(r"mtp\.(\d+)\.(.*)")` 取出 `stage` 与 `rest`，不匹配返回 `None`；`first_layer_idx = num_hidden_layers = 40`、`last_layer_idx = 42`：

| 规则 | stage | 落到 |
|------|-------|------|
| ① `confidence_head.*` | 仅末 | `model.confidence_head.*`（不挂层，[L630-L631](../dspark.py#L630)） |
| ② `embed.weight` / ③ `head.weight` | 仅 0 / 仅末 | `model.embed_tokens.weight` / `lm_head.weight`（[L633-L636](../dspark.py#L633)） |
| ④ `hc_head_*` | 任意 | `model.*`（[L637-L638](../dspark.py#L637)） |
| ⑤ `main_proj/main_norm.*` | 通常 0 | 第 **40** 层（偏移=0，[L642-L643](../dspark.py#L642)） |
| ⑥ `norm.*`/`markov_head.*` | 通常末 | 第 **42** 层（偏移=末，[L644-L645](../dspark.py#L644)） |
| ⑦ 其余 | `stage` | 第 **40+stage** 层（[L646-L648](../dspark.py#L646)） |

组件名替换（[L650-L661](../dspark.py#L650)）：`.attn.`→`.self_attn.`、`.ffn_norm.`→`.post_attention_layernorm.`、`.attn_norm.`→`.input_layernorm.`、`.ffn.`→`.mlp.`、`.w1./.w2./.w3.`→`.gate_proj./.down_proj./.up_proj.`、`.mlp.gate.bias`→`.mlp.gate.e_score_correction_bias`。顺序陷阱：元组顺序保证先 `.ffn.`→`.mlp.` 再匹配 `.mlp.gate.bias`，故 `mtp.N.ffn.gate.bias` → `...e_score_correction_bias`；而 `...gate.bias_vl` 因包含 `.mlp.gate.bias` 也被改成 `e_score_correction_bias_vl`，正好落进 §7.1 的跳过判断。

### 7.4 四类特殊权重

| 权重 | 特殊在哪 | 处理 | 代码 |
|------|----------|------|------|
| `context_wkv_proj.*` | 名字无 `mtp.`、需 `shard_id` | 复制+打 `shard_id`，按 shard 装入 | [L539](../dspark.py#L539)、[L562-L566](../dspark.py#L562) |
| `attn_sink` | 参数按 TP 头段切分，ckpt 是全 64 头 | 手动 `w[head_start:head_end]`+`copy_` | [L609-L614](../dspark.py#L609) |
| `gate.bias_vl` | ckpt 有、draft 模块没有 | 命中+不在 `params_dict` ⇒ 跳过 | [L573-L575](../dspark.py#L573) |
| `main_proj.scale` | 只有 fp8 线性才有 `weight_scale` | 依赖 §3.3 量化配置，否则 `KeyError` | [L216-L231](../dspark.py#L216) |

---

## 八、与投机解码链路的接口 — 高级篇

### 8.1 调用契约

入口类对外只暴露这些方法（[L462-L507](../dspark.py#L462)），proposer 按鸭子类型调用：

| 方法 | 输入 | 输出 | 调用方 |
|------|------|------|--------|
| `get_draft_kv_cache_layer_names()` | — | 3 个 `mtp.i.self_attn.swa_cache.prefix` | [proposer:L138](../../../worker/v1/spec_decode/dspark_proposer.py#L138) |
| `combine_hidden_states(aux)` | `(T,15360)` | `(T,5120)` | [llm_base:L1064](../../../worker/v1/spec_decode/llm_base_proposer.py#L1064) |
| `precompute_and_store_context_kv(s,p,slots=None)` | `(T,5120)`/`(T,)`/`list[3]` | 无（写 cache） | [dflash:L320](../../../worker/v1/spec_decode/dflash_proposer.py#L320)（继承自 DFlash proposer，见 §1.4）；profile [proposer:L417](../../../worker/v1/spec_decode/dspark_proposer.py#L417) |
| `forward(input_ids, positions)` | `(Q,)`/`(Q,)` | `(Q,5120)` | [llm_base:L1490](../../../worker/v1/spec_decode/llm_base_proposer.py#L1490) |
| `compute_draft_logits(hidden)` | `(N,5120)` | `(N,129280)` | [llm_base:L1571](../../../worker/v1/spec_decode/llm_base_proposer.py#L1571) |
| `markov_embed(ids)` / `markov_bias(e)` | `(N,)` / `(N,256)` | `(N,256)` / `(N,129280)` | [llm_base:L1581-L1582](../../../worker/v1/spec_decode/llm_base_proposer.py#L1581)；[utils:L363](../../../worker/v1/spec_decode/utils.py#L363) |
| `compute_confidence(h,m)` | `(N,5120)`/`(N,256)` | `(N,)` 概率 | [utils:L375](../../../worker/v1/spec_decode/utils.py#L375)；[proposer:L122](../../../worker/v1/spec_decode/dspark_proposer.py#L122) |
| `map_draft_to_target(ids)` | `(N,)` | 原样 | [llm_base:L1595](../../../worker/v1/spec_decode/llm_base_proposer.py#L1595) |

三个细节：① **`forward` 不吃 `inputs_embeds`/`hidden_states`**，只转发 `input_ids`/`positions`（[L451-L460](../dspark.py#L451)），上下文靠预写 KV——与 `mtp` 方法"传 hidden_states 进模型"是两条路；② **`num_query_per_req` 由 `sample_from_anchor` 决定**：`True`（默认）→ `K`（锚点自己也算一个 query），`False` → `1+K`（[proposer:L39-L43](../../../worker/v1/spec_decode/dspark_proposer.py#L39)），块大小 5 即此值；③ **`markov_bias` 必须复用同一个 `logits_processor`**（[L484-L485](../dspark.py#L484)），否则词表切分不一致会导致形状对不上。

### 8.2 一次 propose 的时序

```
(1) combine_hidden_states(aux) → main_x                  [llm_base:L1064]
(2) set_inputs_first_pass: seed=next_token_ids [proposer:L247]、hidden [L264]、
    triton 填 input_ids/positions/query slot/context slot[3] [L275-L316]
(3) build_model_inputs_first_pass → precompute_and_store_context_kv [dflash:L320，继承自 DFlash proposer]
(4) draft forward: [cur,128799×4] → hc repeat → 40/41/42 → hc_collapse [L279-L307]
(5) compute_draft_logits → markov 顺序采样 ×5            [llm_base:L1571-L1596]
(6) dynamic_spec.update → markov_embed + compute_confidence [utils:L336-L394]
```

第 (2) 步 draft 输入 token 是 **`[当前 token, 128799, 128799, 128799, 128799]`**：triton 在 `q_idx==0` 写锚点、其余写 `parallel_drafting_token_id`（[ops/triton/spec_decode/utils.py:L254-L256](../../../ops/triton/spec_decode/utils.py#L254)），该 id 由 patch 从 `dspark_noise_token_id` 提升（[patch_speculative_config.py:L286-L288](../../../patch/platform/patch_speculative_config.py#L286)：`ptd_token_id = dspark_noise_token_id`）。噪声 token 既是占位符，也是 Markov 链第一步输入（第 0 列随后被 seed 覆盖，§6.3）。

### 8.3 全链路形状总表

| 张量 | 边界 | shape |
|------|------|-------|
| `aux_hidden_states` / `main_x` | 主模型 → drafter / 内部 | `(T,15360)` / `(T,5120)` |
| `context_states` / `all_kv` / `shared_kv` | runner → drafter | `(T_ctx,5120)` / `(T_ctx,3,512)` / `(T_ctx,1,512)` |
| draft `input_ids`/`positions` | proposer → drafter | `(Q,)`，`Q=B×5` |
| `hidden_states`/`pre_mix` | 过层中 | `(Q,4,5120)` / `(Q,4)` |
| `head_hidden`/`logits` | drafter → sampler | `(Q,5120)` / `(Q,129280)→(B,5,V)` |
| `markov_embed`/`logits_bias`/`confidence` | 输出头 | `(N,256)` / `(N,129280)` / `(B,5)` |
| `draft_token_ids` | proposer 输出 | `(B,6)`→`[:,1:]`=`(B,5)`（[llm_base:L1636](../../../worker/v1/spec_decode/llm_base_proposer.py#L1636)） |

---

## 九、常见坑 / FAQ — 实战篇

**Q1 `main_proj` 能用 bf16 吗？** 不能。release ckpt 的 `mtp.0.main_proj` 是 fp8（有 `main_proj.scale`），bf16 线性没有 `weight_scale` 参数，加载时 `model.layers.40.main_proj.weight_scale` 查不到，直接 `KeyError`——这正是 [L210-L222](../dspark.py#L210) 三重条件存在的原因（且额外要求 A5 的 packed ops）。

**Q2 `hc_head` 一调就 `AttributeError: no attribute 'norm_eps'`？** 它是 V4 死代码（§6.1），V4.1 走 `last_layer.hc_collapse(hidden, pre_mix)`（[L304](../dspark.py#L304)）。手写等价实现请用 `config.rms_norm_eps`(1e-20)/`config.hc_eps`(1e-6)，且要知道它与 `hc_collapse` **不等价**（一个学 sigmoid 门控，一个用上层交接的 `pre_mix`）。

**Q3 `context_slot_mapping` 为什么是 list？** 3 个 draft 层可能落在不同 KV cache group（block size 可能不同），故每层一个 mapping；顺序由 `self._layer_group_idx`（排序后的层名→gid）决定（[proposer:L204-L210](../../../worker/v1/spec_decode/dspark_proposer.py#L204)、[L314-L316](../../../worker/v1/spec_decode/dspark_proposer.py#L314)），**不是** `ModuleDict` 插入序。写错顺序 = 把 A 层 KV 写进 B 层 cache，表现为"接受率诡异下降但不报错"。

**Q4 profile 阶段 KV 写不进去也不报错？** 故意的：`is_profile=True` 只传两个参数（[proposer:L416-L417](../../../worker/v1/spec_decode/dspark_proposer.py#L416)），算完投影即 return，为让 GEMM workspace 进显存 profile（§4.5）。

**Q5 有了合并投影能删每层 `wkv` 吗？** 不能。合并投影只服务上下文预计算；draft token 前向仍要 `self_attn.wkv`（`(512,5120)`/层）。反之删合并投影则 `precompute_and_store_context_kv` 直接崩。

**Q6 `e_score_correction_bias_vl` 被忽略会影响精度吗？** 不影响文本精度。它是视觉路由偏置，DSpark 只在文本 decode 跑、文本 config 下 draft MoE 不创建 `bias_vl`；代码拒绝 alias 到文本 correction bias，否则开关投机解码会改变文本路由（[L568-L575](../dspark.py#L568)）。

**Q7 draft MoE 为何是 128/3？** ckpt 里 draft 专家更少（§2.3 的 `768=128×6`），`dspark_n_routed_experts/num_experts_per_tok` 写在 config；patch 只改 **draft 的私有 config 副本**（[patch:L191-L198](../../../patch/platform/patch_speculative_config.py#L191)），不污染目标模型。故 draft 与主模型的 `gate_up_proj/down_proj` 形状不同，**绝不能互相加载**。

**Q8 `attn_sink` 为何手动 narrow？** 它是裸 `nn.Parameter(n_local_heads)`（`model.py:L397`），无自定义 `weight_loader`，而 ckpt 存全 64 头；TP=8 时每卡只拿 8 头（[L529-L533](../dspark.py#L529)、[L609-L614](../dspark.py#L609)）。

**Q9 `causal=False` 会带歪主模型吗？** 不会。`cad` 是 proposer 自建的 draft 专用 metadata（[proposer:L231-L362](../../../worker/v1/spec_decode/dspark_proposer.py#L231)），只喂 draft 层（`draft_attn_metadatas`）。

**Q10 5 个 token 为何窗口开 132？** 可见集合须为"尾部 128 + 当前块 5"，块占 5 个逻辑位置，left 取 `128+5-1=132` 才能让块首 token 也看到完整 128 行历史（[dsa_v1:L542-L546](../../../attention/dsa_v1.py#L542)）；索引宽度对齐到 256（[L549-L551](../../../attention/dsa_v1.py#L549)），预分配 buffer 别按 133 算。

**Q11 `num_speculative_tokens` = `dspark_block_size` 吗？** 不是。前者是**调度侧**每步实际要的草稿数（可被动态投机调小甚至为 0），后者是**模型侧**块宽（[L162](../dspark.py#L162)）。`forward` 用 `input_ids` 实际长度 `Q`、不读 `block_size`，故动态缩短不触发形状断言。

**Q12 `compute_logits` 为何 `del spec_step_idx`？** DSpark 一次前向出整块，没有"第 i 个 spec step"（[L469-L479](../dspark.py#L469)），该参数是 EAGLE/MTP 风格接口残留。

**排障速查**

| 现象 | 最可能原因 | 定位 |
|------|-----------|------|
| `KeyError: ...main_proj.weight_scale` | `main_proj` 建成 bf16 | §3.3 |
| draft 接受率≈0 | `context_slot_mapping` 顺序错/未写 cache | §4.3、Q3 |
| `size mismatch` 于 `mlp.*` | draft 与目标专家数混用（128 vs 384） | Q7 |
| 置信度恒≈0.5 | `markov_embed` 取错列 | §6.4 |
| prefill 后 OOM | 漏跑 profile 期的 `context_wkv_proj` | §4.5 |
| `attn_sink` 形状不符 | TP narrow 未生效 | Q8 |

---

## 十、设计要点小结

1. **3 个串行 block、一次上下文**：`mtp.0/1/2` → 40/41/42；37/38/39 的 `mean(hc)` 只融合一次，3 个 block 共享 `main_x` 与预计算 KV。
2. **块级并行 + 顺序补依赖**：`causal=False` + "整窗 + 整块" topk 一次算完整块，Markov 头再从左到右给 5 个位置加 n-gram 偏置。
3. **KV 预计算是性能核心**：`context_wkv_proj`（3×512 合并、`disable_tp`）把 3 层 WKV 压成 1 次 GEMM；写 cache 分 A5 packed（`kind="win"`，mxfp8）与通用 2-D scatter。
4. **mHC 交接决定输出头**：`pre_mix` 逐层下传、`hc_collapse` 收尾；`hc_head` 在 V4.1 已是死代码。
5. **两个输出头分工**：Markov（rank 256，加性偏置）决定草稿 token；Confidence（5376→1，fp32，sigmoid）只估接受率。
6. **加载靠重映射**：`mtp.*`→`layers.{40,41,42}.*`、组件改名、`markov_w1/w2`、`confidence_head.proj`、`context_wkv_proj` 分片、`attn_sink` TP narrow、`bias_vl` 跳过。

## 十一、一句话总结

DSpark 是挂在 `mtp.*` 下、由 **3 个纯 SWA draft block（运行时 40/41/42）+ 三层残差流融合 `main_proj` + 一次 GEMM 的上下文 KV 预计算 + 延迟 mHC 交接 + Markov 加性偏置 + Confidence 接受率** 组成的块草稿器：它把 37/38/39 层残差流压成一条上下文、把 3 层 WKV 合成一次投影写入 SWA cache，用**块内非因果注意力**一次前向算出 5 个位置的 logits，再用**顺序 Markov 采样**生成 5 个候选 token、用 confidence head 估计逐位置接受率，最后交给主模型用因果注意力批量验证——在不损失正确性的前提下换取更高的推测解码吞吐。
