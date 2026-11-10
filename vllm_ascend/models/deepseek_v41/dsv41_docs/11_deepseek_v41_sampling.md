# DeepSeek V4.1 后处理与采样

> **冻结声明**：本文只描述**本 checkout**（工作区 `vllm-npu`；`vllm-ascend` 侧 `eb5c29cb7`，vendored 上游 `vllm` 侧 `5df63a7203`）实际存在的代码。所有 `Lxxx` 均经 read/grep 在本仓库重新核对，**未**从其他版本文档转抄；凡参照文档（`deepseek_v4/dsv4_docs/7_deepseek_v4_postprocess.md`）提到但本树不存在/已改名的路径与类，正文显式标注并给出本树事实。

| 项 | 值 |
| --- | --- |
| 模型 | DeepSeek-V4.1-Flash（`DeepseekV41ForCausalLM`），词表 **129280**、隐层 **5120**、bf16 |
| 推测解码 | DSpark：`mtp.0/1/2` 三个 draft 层，`dspark_block_size=5`，`dspark_markov_rank=256` |
| 采样栈 | **MRv1 与 MRv2 同时存在**，MRv2 默认，MRv1 回退（§1.2） |
| 覆盖范围 | target logits → LogitsProcessor → Sampler/RejectionSampler → DSpark（logits/Markov/confidence）→ 回写 |

**覆盖范围（本 checkout 实际文件与行号）**

| 层 | 文件 | 关键符号 : 行 |
| --- | --- | --- |
| target | [../model.py](../model.py) | `compute_logits` **L1813-1818**、`ParallelLMHead` **L1659-1664**、`LogitsProcessor` **L1667**、`head→lm_head` **L1886-1889** |
| draft | [../dspark.py](../dspark.py) | `compute_draft_logits` **L462-464**、`map_draft_to_target` **L466-467**、`compute_logits` **L469-479**、`markov_embed` **L481-482**、`markov_bias` **L484-485**、`compute_confidence` **L487-489**、`markov_head` **L238-243**、重映射 **L419-431**/**L623-662** |
| MRv1 | [sampler.py](../../../sample/sampler.py) · [penalties.py](../../../sample/penalties.py) · [rejection_sampler.py](../../../sample/rejection_sampler.py) · [model_runner.py](../../../worker/v1/model_runner.py) | `AscendSampler` **L45**、`forward_native` **L92-111**；`apply_all_penalties` **L25-45**；`AscendRejectionSampler` **L40**、`forward` **L150-258**、`apply_sampling_constraints` **L325**、`rejection_sample` **L382**；`execute_model` **L2347**、`sample_tokens` **L2882**、`_sample` **L3073-3094** |
| MRv1 DSpark | [llm_base_proposer.py](../../../worker/v1/spec_decode/llm_base_proposer.py) · [utils.py](../../../worker/v1/spec_decode/utils.py) · [dspark_proposer.py](../../../worker/v1/spec_decode/dspark_proposer.py) | 顺序采样 **L1544-1611**（**L1571/1581/1582/1593**）；`_compute_dspark_token_probs` **L336-394**；`_compute_confidence` **L108-125**（死代码） |
| MRv2 | [model_runner.py](../../../worker/v2/model_runner.py) · [dspark/speculator.py](../../../worker/v2/spec_decode/dspark/speculator.py) · [patch_triton.py](../../../patch/worker/patch_triton.py) | `sample` **L1193-1250**；`AscendDSparkSpeculator` **L58**、lmhead-TP 包装 **L106-136**；重绑 **L130-167**（`gumbel_sample` **L147**） |
| 上游 | [logits_processor.py](../../../../../vllm/vllm/model_executor/layers/logits_processor.py) · [v1/sample/sampler.py](../../../../../vllm/vllm/v1/sample/sampler.py) · [gpu/sample/sampler.py](../../../../../vllm/vllm/v1/worker/gpu/sample/sampler.py) · [dspark/speculator.py](../../../../../vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py) | `forward` **L98**、`_gather_logits` **L122**、`_apply_head` **L136**、裁剪 **L198**；`forward` **L73**、fp32 **L97**、processors **L375**；`apply_sampling_params` **L226-276**；`_sample_sequential` **L156-199** |
| 参考 / 数据 | [reference/model.py](reference/model.py) · [config](0_DeepSeek-V4.1-Flash-1017_config.json) · [权重索引](0_DeepSeek-V4.1-Flash-1017_model_weights.index.json) | `forward_head` **L1140-1159**、`sample()` **L1288-1295**；`head.weight` **L35312**、`embed.weight` **L11953**、`mtp.2.markov_head.*` **L72735-72737** |

---

## 一、全链路总览

### 1.1 两阶段 `execute_model` / `sample_tokens`

```text
Executor.collective_rpc
├─〔阶段① execute_model〕MRv1 worker/v1/model_runner.py:2347
│   ├─ _model_forward → hidden_states        [num_tokens, 5120]
│   ├─ hidden_states[logits_indices]         :2752 → [num_logits, 5120]
│   └─ model.compute_logits(...)             :2753
│        └─ LogitsProcessor.forward          logits_processor.py:98
│             ├─ _apply_head    :136  本地分片 matmul（fp8 head）
│             ├─ _gather_logits :122  tp>1 → all-gather
│             ├─ [..., :org_vocab_size] :198
│             └─ soft_cap :113-116 / scale :118-119 → 本模型均跳过
│        → logits [num_logits, 129280] bf16，驻留 NPU
│   MRv2 worker/v2/model_runner.py:706 与之对应，其后 Executor 调 sample_tokens
│
└─〔阶段② sample_tokens〕
   MRv1 :2882
    ├─ grammar bitmask（D2H→float→H2D）       :2921-2928
    ├─ _sample(logits, spec_decode_metadata)  :3073-3094
    │    ├─〔无 spec〕self.sampler            :3077-3083 → AscendSampler
    │    └─〔有 spec〕self.rejection_sampler  :3085-3093
    ├─ propose_draft_token_ids(...)           :2946-2967 → :3015/:3019
    │    └─ AscendDSparkProposer → llm_base_proposer.py:1544-1611
    ├─ _bookkeeping_sync(...)                 :3110
    └─ ModelRunnerOutput / Async...Output     :3027 / :3057
   MRv2 :408 → super().sample_tokens :441
         └─ GPUModelRunner.sample  gpu/model_runner.py:1592
              ├─ compute_logits :1614-1615 → apply_grammar_bitmask :1623-1633
              ├─ self.sampler(logits, input_batch)                   :1642
              └─ self.rejection_sampler(..., speculator.draft_logits) :1647
```

阶段① 只产出 logits：prefill / decode / 推测解码的差异只在 `logits_indices` 取法（MRv1 **L1763** 常规、**L1791** 推测、**L1889-1963** 构造）与 `num_logits` 大小，采样逻辑不变。

### 1.2 两套采样栈（本 checkout 都在）

| 维度 | MRv1（回退） | MRv2（默认） |
| --- | --- | --- |
| 采样类 | `AscendSampler` + `AscendRejectionSampler` | 上游 `Sampler`（`gpu/sample/sampler.py:42`）+ 上游 `RejectionSampler`（`gpu/spec_decode/rejection_sampler.py:147`） |
| 注入方式 | 子类覆写 + 换 `topk_topp_sampler` | [patch_triton.py](../../../patch/worker/patch_triton.py) **L130-167** 算子重绑 |
| Top-K/P | `forward_native` → [_apply_top_k_top_p_pytorch:10](../../../worker/v2/sample/apply_top_k_top_p.py#L10) | `apply_top_k_top_p_npu`（同算法，**L149**） |
| 随机采样 | `random_sample`：exp+div+argmax | `categorical_sample`（**L144-148**） |
| DSpark draft | `llm_base_proposer.py:1544-1611` 手写循环 | 上游 `_sample_sequential` + Ascend 包装 |

选择逻辑：[mrv2_utils.py](../../../mrv2_utils.py) `use_v2_model_runner` **L99-133**——未设 `VLLM_USE_V2_MODEL_RUNNER` 默认 V2，只有 `_v2_blacklist`（**L66-96**）或显式 `=0` 才回 V1；DSpark 的 `method=="dspark"`（上游 `config/speculative.py:1355`）不在黑名单，故默认 MRv2。另有 V4.1 专项硬约束 [platform.py:1058-1064](../../../platform.py#L1058)：A5（`DSV41_PACKED_CACHE`）上 `not use_v2_model_runner` 直接 `ValueError` —— **A5 上 MRv1 采样代码对本模型不可达**；其他硬件 MRv1 仍是可用回退。

---

## 二、Target 侧：logits 计算与 TP gather

### 2.1 形状主线

| 阶段 | 形状 | 位置 |
| --- | --- | --- |
| hidden | `[num_tokens, 5120]` | forward 输出 |
| 取行 | `[num_logits, 5120]` | MRv1 **L2752**；MRv2 上游 **:1614** |
| 本地分片 matmul | `[num_logits, 129280/tp]` | `logits_processor.py:136-178` |
| all-gather | `[num_logits, 129280]` | `:122-134` |
| 词表裁剪 | `[num_logits, 129280]`（**空操作**） | `:198` |

### 2.2 `LogitsProcessor` 内部顺序

| # | 步骤 | 行 | 本模型行为 |
| --- | --- | --- | --- |
| 1 | `_apply_head` | `:136` | `lm_head.quant_method.apply`；`head_dtype is None` → **不提升 dtype**（`:143-146`） |
| 2 | `_gather_logits` | `:122` | `use_all_gather`（`:91`）= `current_platform.use_all_gather()`；Ascend 未覆写，基类 `platforms/interface.py:1267` 返回 `True` → all-gather；仅 `tp_size>1`（`:193`） |
| 3 | 词表裁剪 | `:198` | `[..., :org_vocab_size]`，默认 = `vocab_size`（`:87`） |
| 4/5 | `soft_cap` / `scale` | `:113-119` | `None` / `1.0` → 均跳过 |

**soft-cap / scale 的结论是「V4.1 没有」**：构造为 `LogitsProcessor(config.vocab_size)`，只传词表（[../model.py:1667](../model.py#L1667)），无 `scale=`/`soft_cap=`/`org_vocab_size=`；`grep final_logit_softcapping` 在 `models/deepseek_v41/` 下**零命中**，[配置](0_DeepSeek-V4.1-Flash-1017_config.json) 亦无该键。故 target logits 后处理只有「matmul + gather + 裁剪」。draft 侧同样只传 vocab（[../dspark.py:412](../dspark.py#L412)）。

### 2.3 lm_head 权重来源

| 环节 | 事实 | 位置 |
| --- | --- | --- |
| 构造 | `ParallelLMHead(129280, 5120, quant_config, prefix="lm_head")`；非末级 PP rank 用 `PPMissingLayer` | [../model.py:1658-1666](../model.py#L1658) |
| ckpt 键 | 顶层 `head.weight`（索引 **L35312**）；无 `model.head.weight` | [权重索引](0_DeepSeek-V4.1-Flash-1017_model_weights.index.json) |
| 改名 | 先补 `model.`（**L1876-1877**），再 `model.head.`→`lm_head.`、`model.lm_head.`→`lm_head.`（**L1886-1889**） | [../model.py:1886](../model.py#L1886) |
| 量化 | FP8（`weight_block_size=[32,32]`），head 继承 `quant_config` → **量化 head** | 配置 + `logits_processor.py:143-146` |

> 本文件**没有** `compute_logits_local`（V4 文档 §2.7 的本地分片入口在此不存在）。`skip_gather` 只存在于上游 `forward:103`/`_get_logits:189-190`，本模型不用；`batch_shard` 的 `compute_logits_local` 调用在**上游 MRv2** `gpu/model_runner.py:1602`。

### 2.4 TP all-gather 通信量（`[n,129280]` bf16）

每 rank 收到 ≈ `(tp-1)/tp × n × 258,560 B`：n=32 时 **tp=8 ≈ 7.24 MB**、tp=16 ≈ 7.76 MB、tp=32 ≈ 8.01 MB（本地列数依次 16160 / 8080 / 4040，单行全量恒为 258,560 B）。

### 2.5 lmhead TP

开关 `finegrained_tp_config.lmhead_tensor_parallel_size > 1`（[utils.py:971](../../../utils.py#L971)）；容量 `lmhead_tp_max_num_logits = max_num_reqs * rows`，注释明确「跨 rank 漂移会让集合通信失步挂死」（[utils.py:975-978](../../../utils.py#L975)）。MRv1 裁剪在 [model_runner.py:3078-3086](../../../worker/v1/model_runner.py#L3078)，MRv2 pad/trim 在 [model_runner.py:1200-1211](../../../worker/v2/model_runner.py#L1200)，DSpark draft 包装在 [dspark/speculator.py:106-136](../../../worker/v2/spec_decode/dspark/speculator.py#L106)（容量 `max_num_reqs * num_speculative_steps`）。

---

## 三、常规分支：采样

### 3.1 MRv1 实际顺序

`AscendSampler` 只覆写三处：`__init__`（换 `AscendTopKTopPSampler`，**L72-80**）、`apply_penalties`（**L47-70**）、`greedy_sample`（**L82-84**，纯 `argmax(-1)`）；其余走上游 `Sampler`。

```text
Sampler.forward                  vllm/v1/sample/sampler.py:73
 ├─ raw_logprobs                 :86-94   ← 一切后处理之前
 ├─ logits.to(torch.float32)     :97      ← 本路径唯一强制 fp32
 ├─ apply_logits_processors      :99 → :375
 │   ├─ allowed_token_ids_mask   :395-396   ├─ apply_bad_words :399-400
 │   ├─ non_argmax_invariant     :403-404   └─ self.apply_penalties :407
 │        └─ apply_all_penalties sample/penalties.py:25
 │             └─ apply_penalties_triton ops/triton/penalty.py:196
 └─ sample                       :247
     ├─ greedy 判定/提前返回      :260-274
     ├─ apply_temperature        :230-239（原地 div_）
     ├─ topk_topp_sampler        :289-294 → forward_native sample/sampler.py:92-111
     │    └─ apply_top_k_top_p :103 → softmax(fp32) :110 → random_sample :111/:18-42
     └─ torch.where(probs/argmax) :299-304
```

**顺序**：raw logprobs →（fp32）→ processors（allowed ids / bad words / 非 argmax 不变式）→ **penalties（processors 的最后一步）** → temperature → top-k/p → 随机采样。

`apply_all_penalties`（[penalties.py:25-45](../../../sample/penalties.py#L25)）用 `make_tensor_with_pad(pad=vocab_size)` 把 `output_token_ids` 打成 int64 上卡（**L13-22**），`-1` `masked_fill_` 到 `vocab_size`（**L36**），一条 `apply_penalties_triton` 算完 presence/frequency/repetition；`HAS_TRITON` 假则回退上游（**L53-58**）。`random_sample`（**L18-42**）用 Gumbel-max `probs.div_(q).argmax(-1)`，`q~Exp(1)` 在 `global_stream()` 生成后 `wait_stream` 切回，带 seed 请求逐条 `q[i].exponential_(generator=...)`（**L38-39**）。

### 3.2 MRv2 实际顺序

| 步骤 | 位置 | 说明 |
| --- | --- | --- |
| 短路 | `apply_sampling_params:238-239` | `needs_logits_processing` 全假 → **返回 bf16，不上浮** |
| fp32 | `:242` | 仅在需要处理器时 `empty_like(dtype=fp32).copy_` |
| processors / thinking budget | `:255-260` | 逐个 `apply`；强制结束标记放最后 |
| temperature / min_p | `:263/:268` → `states.py:81-103` | 全 0/1 或全 0 时跳过 |
| top-k/p | `:301` / `:345` | `skip_top_k_top_p=True` 时延后 |
| 采样 | `:327-355` | `gumbel_sample`；Ascend 换 `categorical_sample` |

`patch_triton.py` 重绑（**L130-167**）：`penalties.apply_penalties`←[v2/sample/penalties.py:125](../../../worker/v2/sample/penalties.py#L125)（packed kernel `:34`、`bincount` `:246`）、`apply_min_p`、`apply_bad_words`、`apply_temperature`←[v2/sample/gumbel.py:52](../../../worker/v2/sample/gumbel.py#L52)、`gumbel_sample`←`categorical_sample`、`apply_top_k_top_p_triton`←[apply_top_k_top_p.py:60](../../../worker/v2/sample/apply_top_k_top_p.py#L60)、`_apply_grammar_bitmask_kernel`。

> **L147**（`dspark_speculator.gumbel_sample = categorical_sample`）是 V4.1 关键：MRv2 的 DSpark 顺序采样用的是上游 `_sample_sequential` 的模块级 `gumbel_sample`，被整体替换成 Ascend 分块分类采样内核——**MRv2 上 DSpark 的 Markov 采样不走 Gumbel-max 公式**。

### 3.3 Top-K/Top-P 只有一套算法

[_apply_top_k_top_p_pytorch:10](../../../worker/v2/sample/apply_top_k_top_p.py#L10) 是唯一实现，`sample/sampler.py:119` 绑为模块级别名，`apply_top_k_top_p_npu`（`:52`/`:60`）只是 MRv2 包装。`sample/sampler.py:114-118` 注释说明：旧 `npu_top_k_top_p` 在 k 大/批内 k 混合时退化到 5-28 ms，sort+mask 稳定 ~1 ms，故删除硬件能力分发；`VLLM_BATCH_INVARIANT` 为真必须回退上游（`:96-101`）。

### 3.4 grammar / structured output（存在，两条路径不同）

- **MRv1**：[model_runner.py:2921-2928](../../../worker/v1/model_runner.py#L2921) 整块 **D2H → float → `apply_grammar_bitmask` → H2D**；注释说明上游用 `torch.compile`、Ascend 暂不支持（`:2923-2924`）。
- **MRv2**：`structured_outputs_worker.apply_grammar_bitmask(...)`（上游 `gpu/model_runner.py:1623-1629`；本树覆写 [model_runner.py:1216-1224](../../../worker/v2/model_runner.py#L1216)），随后 `grammar_invalid_drafts(...)` 标记「用宽松 grammar 行校验过的 draft」（`:1227-1233`），作为 `invalid_drafts` 传给 rejection sampler。

即 bitmask 在 **target logits 之后、采样之前**；推测解码下 MRv2 还多一个 `invalid_drafts`（MRv1 `_sample` 无此概念）。

---

## 四、推测解码分支：DSpark

### 4.1 draft 侧接口（[../dspark.py](../dspark.py)）

| 方法 | 行 | 实现 | 说明 |
| --- | --- | --- | --- |
| `compute_draft_logits` | [L462-464](../dspark.py#L462) | `compute_logits(hidden)` | 注释：*Full-vocab draft: base logits, no d2t scatter* |
| `map_draft_to_target` | [L466-467](../dspark.py#L466) | 恒等 | 全词表 → draft id 即 target id |
| `compute_logits` | [L469-479](../dspark.py#L469) | `self.model.compute_logits(...)` | 模型侧 `logits_processor(lm_head, self.norm(hidden))`（**L371-377**，**先 norm**） |
| `markov_embed` | [L481-482](../dspark.py#L481) | `markov_head.embed` | `[B] → [B,256]` |
| `markov_bias` | [L484-485](../dspark.py#L484) | `markov_head.bias(emb, logits_processor)` | `[B,256] → [B,129280]` |
| `compute_confidence` | [L487-489](../dspark.py#L487) | `sigmoid(confidence_head(hidden, emb))` | 每位置接受概率 |

`markov_head = DSparkMarkovHead(129280, draft_vocab_size or 129280, 256)`（**L238-243**；配置无 `draft_vocab_size` → **129280**），`confidence_head = DSparkConfidenceHead(5120+256, bias=False, with_markov=True)`（**L244-249**），二者挂最后一个 draft 层（**L250-252**），ckpt 也在 `mtp.2.*`。两个头来自上游 [qwen3_dspark.py:48](../../../../../vllm/vllm/model_executor/models/qwen3_dspark.py#L48)：`markov_w1 = nn.Embedding(vocab,256)`（`:71`）、`markov_w2 = ParallelLMHead(draft_vocab,256,disable_tp=True)`（`:72-79`）、`bias = logits_processor(markov_w2, emb)`（`:87-93`）。因 `disable_tp=True`，`tp_size==1` → `logits_processor.py:193` 不 gather，**`markov_bias` 是本地小 GEMM + 零通信**。

> `apply_markov_bias_gathered` 只有上游 `Qwen3DSparkForCausalLM` 定义（`qwen3_dspark.py:290`），**本 drafter 没有**；故上游 `_sample_sequential_topk`（`dspark/speculator.py:201-248`，**L229** 调它）对本模型不可用，且该分支仅当 `hf_config.dspark_draft_topk` 存在才进入（`:78-80`），V4.1 配置无此字段。

### 4.2 MRv1 调用顺序（[llm_base_proposer.py:1544-1611](../../../worker/v1/spec_decode/llm_base_proposer.py#L1544)）

```python
dspark_has_vocab_mapping = getattr(self.model, "draft_id_to_target_id", None) is not None   # L1558
use_probabilistic = self._enable_probabilistic_draft_probs and not dspark_has_vocab_mapping # L1559
raw_logits = self.model.compute_draft_logits(sample_hidden_states)      # L1571  [B*K, 129280]
if lmhead_tp_enable(): raw_logits = raw_logits[:num_indices]            # L1572-1574
logits = raw_logits.view(-1, self.num_speculative_tokens, -1)           # L1575  [B, K, V]
draft_token_ids[:, 0].copy_(self._dspark_seed_buffer[:num_blk])         # L1577-1578  anchor
for idx in range(self.num_speculative_tokens):                          # L1580
    logits[:, idx].add_(self.model.markov_bias(                          # L1581-1583
        self.model.markov_embed(draft_token_ids[:, idx])))
    if use_probabilistic:                                               # L1588
        token_ids, probs = self._sample_draft_from_logits(logits[:, idx], sampling_metadata)
    else:
        token_ids = logits[:, idx].argmax(dim=-1)                       # L1593 默认贪心
        if dspark_has_vocab_mapping:
            token_ids = self.model.map_draft_to_target(token_ids)       # L1595
    draft_token_ids[:, idx + 1].copy_(token_ids)                        # L1589/1596
if self.dynamic_spec is not None: self.dynamic_spec.update(...)          # L1605-1611
```

**顺序**：`compute_draft_logits` →（lmhead TP 截断）→ reshape `[B,K,V]` → K 次 {`markov_embed(prev)` → `markov_bias` → `add_` → `argmax`（或概率采样）→ 写回下一位置} →（可选）`dynamic_spec.update`（内部 `compute_confidence`）。

边界：① seed 是 anchor，由 `next_token_ids` 填入（[dspark_proposer.py:245-248](../../../worker/v1/spec_decode/dspark_proposer.py#L245)），循环产出 `[:,1:]`（`:1636-1637`）；② d2t 与概率采样互斥（**L1558-1566** `warning_once`：draft-vocab probs 不能进 target 空间拒绝采样），V4.1 `draft_id_to_target_id=None`（**L392**）→ **允许**概率采样；③ dspark 下 `num_indices = batch_size * num_speculative_tokens`（**L1506-1511**）。

### 4.3 MRv2 调用顺序

Ascend 不重写循环，复用上游 [dspark/speculator.py:156-199](../../../../../vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L156)：`compute_draft_logits`（**:167**）→ reshape `[B,K,V]`（**:169**）→ `prev = input_ids[anchor]`（**:177**）→ K 次 {`markov_embed`（**:181**）→ `markov_bias`（**:184**）→ `base_logits[:,i] + bias`（**:185**）→ `_sample_logits`（**:186** → :116-154）→ `prev`}（**:179-190**）→ `compute_confidence(sample_hidden, stack(embeds))` 写 `draft_token_confidence_probs`（**:193-199**）。`_sample_logits` 内含 `map_draft_to_target`（**:126** 贪心）、d2t scatter（**:130-134**，仅 reduced-vocab）、`sampler`（**:143-154**，Ascend 换 `categorical_sample`）。

Ascend 包装只有三件事（[dspark/speculator.py](../../../worker/v2/spec_decode/dspark/speculator.py)）：`load_draft_model`（**L90-104**）、`_lmhead_tp_wrap_draft_logits`（**L106-136**，pad→原方法→trim，**不覆写采样循环**）、`_run_model`（**L400-417**，PCP 复制 rank 广播 backbone 输出以保证 Markov 采样一致）。「不覆写 `sample_draft`」在 **L60-65** 写明：DSpark 走 `compute_draft_logits`、从不调 `sample_draft`。

### 4.4 参考实现语义（[reference/model.py](reference/model.py)）

`forward_head`（**L1140-1159**）：`logits = head(norm(x), full_logits=True)`（**L1148**）→ `output_ids[:,0] = input_ids`（**L1150**）→ `for i in range(block_size)`：`logits_bias, markov_embed = markov_head(output_ids[:,i])`（**L1153**）、`logits[:,i].add_(logits_bias)`（**L1154**）、`output_ids[:,i+1] = sample(logits[:,i], temperature)`（**L1156**）→ `confidence = confidence_head(x, stack(embeds))`（**L1158**）。`sample()`（**L1288-1295**）：`temperature==0 → argmax`；否则 `softmax(fp32)` 后 `probs.div_(exponential_(1)).argmax(-1)`。

| 语义点 | 参考 | 运行时 | 一致性 |
| --- | --- | --- | --- |
| 位置 i 用「前一 token」的 Markov 嵌入 | **L1153** | MRv1 **L1581** / MRv2 **:181** | ✅ |
| bias 只加当前位置 | **L1154**（`add_`） | MRv1 `add_`（**L1583**）；MRv2 `+`（**:185**，非原地） | ⚠️ 原地性不同 |
| 贪心 vs Gumbel-max | **L1292/1295** | MRv1 `argmax`（**L1593**）/ 概率模式；MRv2 `categorical_sample` | ⚠️ 公式不同、分布等价 |
| confidence 一次算全部位置 | **L1158** | MRv1 **L336-394**；MRv2 **:193-199** | ✅ |

### 4.5 confidence 的两个消费点

`compute_confidence = sigmoid(proj([hidden; markov_embed]))`（参考实现 `Linear(input_dim,1,dtype=fp32)`，**L1092-1100**）。

| runner | 链路 | 位置 | 用途 |
| --- | --- | --- | --- |
| MRv1 | `markov_embed(draft[:, :K])` → `compute_confidence` → `[B,K]` → `cumprod` → 预算 | [utils.py:336-394](../../../worker/v1/spec_decode/utils.py#L336)、[:419-461](../../../worker/v1/spec_decode/utils.py#L419) | 动态验证长度（偶数向上取奇） |
| MRv2 | `use_confidence_head` → `draft_token_confidence_probs` → `record_confidences` | 上游 `:106-109`/`:193-199`；`gpu/model_runner.py:2258-2262`；`adaptive_verification.py:256-268` | 自适应验证预算 |

MRv1 开关是 `additional_config.dynamic_spec_config.method == "dspark"`（[ascend_config.py:1273](../../../ascend_config.py#L1273)；`dspark_proposer.py:64-71` 据此建 `DynamicSpecScheduler`），**默认关闭**——`dynamic_spec=None` 时 MRv1 路径上 confidence 头不会被调用。

### 4.6 draft checkpoint 归属

只有 3 个阶段 `mtp.0/1/2.*`（索引去重结果）；**无** `mtp.0.embed.weight`、**无** `mtp.2.head.weight`，draft 复用顶层 `embed.weight`（**L11953**）/`head.weight`（**L35312**），加载时改名为 `model.embed_tokens.weight` / `lm_head.weight`（[../dspark.py:542-545](../dspark.py#L542)）。Markov/confidence 在最后阶段（索引 **L72735-72737**），运行时要重映射为 `markov_w1` / `markov_w2` / `confidence_head.proj`（**L426-431**）。兜底路径 `_remap_checkpoint_name`（**L623-662**）：`stage==last and rest=="head.weight" → lm_head.weight`（**L635-636**）、`stage==0 and rest=="embed.weight"`（**L633-634**）、`confidence_head.*`（**L630-631**）、`norm./markov_head.` 归最后一层（**L644-645**）；阶段数取 `n_mtp_layers` → `dspark_num_mtp_layers` → 默认 **3**（**L78-82**）。

### 4.7 接受 / 拒绝

- **MRv1**：`_sample` 有 `spec_decode_metadata` 时调 `self.rejection_sampler(...)`（[model_runner.py:3085-3093](../../../worker/v1/model_runner.py#L3085)）。`forward`（**L150-258**）：`max_spec_len <= MAX_SPEC_LEN(128)` 断言（**L181**，常量来自上游 `rejection_sampler.py:42`）→ bonus logits 走 `self.sampler`（**L196-208**）→ `raw_target_logits` 上浮 fp32（**L213-216**）→ `apply_logits_processors`（**L222**）→ `apply_sampling_constraints`（**L226**）→ `rejection_sample`（**L228-240**）。模块级 `rejection_sample`（**L382**）读 `AscendConfig`：`using_block_verify = max_spec_len>=3 and enable_block_verify`（**L439**）、`using_entropy_verify`（**L440**）；`synthetic_mode` 与 block verify 互斥（**L446-449**）。默认 K=5 满足 `>=3`，故 block verify **可能**启用但默认关闭。
- **MRv2**：上游 `RejectionSampler`（`gpu/spec_decode/rejection_sampler.py:147`，`_verify` **L244**、`_verify_in_chunks` **L289**），Ascend 只换算子：`rejection_sampler_utils.rejection_sample`（[rejection_sampler_utils.py:279](../../../worker/v2/spec_decode/rejection_sampler_utils.py#L279)，`patch_triton.py:141-142` 重绑）与 `rejection_greedy.py`。

---

## 五、V4.1 增量清单

| # | 增量 | 证据 | 影响 |
| --- | --- | --- | --- |
| 1 | **full-vocab draft**，无 d2t scatter | [L462-464](../dspark.py#L462)、**L466-467**、`draft_id_to_target_id=None`（**L392**） | bias 与 base logits 同空间可直加；draft probs 可进 target 拒绝采样 |
| 2 | **无 d2t 的两处后果** | MRv1 `dspark_has_vocab_mapping=False` → 概率采样不被禁用（**L1558-1566**）；MRv2 `_d2t_scatter_index` 保持 `None`（上游 `:93-105`） | 省掉 `index_copy_` 到 `[B,V]` |
| 3 | **Markov 偏置逐位置** | MRv1 **L1580-1583**；MRv2 **:179-190**；参考 **L1152-1156** | 块内依赖必须串行 |
| 4 | **confidence 门控** | **L487-489**；`utils.py:336-394`；上游 `:192-199` | 决定验证几个 draft token |
| 5 | **Markov rank 仅 256** | `dspark_markov_rank=256`，两个 `[129280,256]` | 见 §6.4 |
| 6 | **block 宽 5 ≠ draft 层数 3** | `dspark_block_size=5`、`num_nextn_predict_layers=3`；上游 `config/speculative.py:1483-1490`：`use_dspark()` 且用户未指定时 `num_speculative_tokens = dspark_block_size`（注释 *parallel width is independent of MTP stages*） | 3=draft 层数，5=每步宽度 |
| 7 | **MRv2 换掉 `gumbel_sample`** | `patch_triton.py:147` | Markov 采样走 Ascend 分类采样内核 |
| 8 | **Markov/confidence 名字 ckpt≠运行时** | **L419-431** | 按 `markov_w1` 找 ckpt 或按 `markov_head.embed` 找参数都会失败 |
| 9 | **draft logits 先 norm** | 模型侧 **L371-377** | target 的 norm 在 `DeepseekV41Model.forward` 末尾（**L1624**） |
| 10 | **A5 强制 MRv2** | [platform.py:1058-1064](../../../platform.py#L1058) | A5 上 MRv1 采样对 V4.1 不可达 |

---

## 六、常见坑

### 6.1 logits dtype：不是「LogitsProcessor 出 fp32」

`_apply_head` 返回 matmul 自然 dtype（本模型 bf16），`forward`（`logits_processor.py:98-120`）**全程不 cast**；`head_dtype` 来自 `model_config.head_dtype`（**L96**），V4.1 未设置 → `None`。MRv1 在 `sampler.py:97` 无条件上浮；MRv2 **按需**（`:238-242`）。后果：同链路 logits 可能是 bf16 也可能是 fp32，写 kernel 不能假设 dtype——`worker/v2/sample/gumbel.py:64-68` 的注释假定「上游恢复了 fp32 上浮」，而 fast path 并不成立。另：`rejection_sampler.py:181` 附近「need to do allgather first」的注释在本树已过时——`rejection_sample:514` 实际对**已 gather 的** `target_logits` 做 `argmax`。

### 6.2 词表对齐：129280 本身已对齐，裁剪是空操作

`DEFAULT_VOCAB_PADDING_SIZE=64`（`vocab_parallel_embedding.py:36`，`pad_vocab_size` **L95-97**）；`129280 % 64 == 0` 且 `% 128 == 0` → padded == org == 129280；可整除 TP 8/16/32（16160/8080/4040）；`image_token_id=129264`、`dspark_noise_token_id=128799` 均在词表内。**风险**：若换 ckpt 使 `vocab_size` 非 64 倍数，`ParallelLMHead` 会 pad，`markov_bias`（独立的 `disable_tp` head）与 target logits 列数可能不一致 → `add_` shape 错；`apply_all_penalties` 用 `logits.shape[1]` 当 `pad`（`sample/penalties.py:34-35`）也会混淆 pad 列。

### 6.3 TP gather 通信量

仅 `lm_head.tp_size>1` 才 gather（`logits_processor.py:193`），Ascend 走 **all-gather**（每 rank 拿全量）。单步 ≈ `(tp-1)/tp × n × 129280 × dtype`；bf16、n=32：tp=8 ≈ 7.24 MB、tp=32 ≈ 8.01 MB。推测解码放大 n（K=5、B=32 → ≈192 行），tp=8 时约 **43 MB/step**，是 V4.1 最值得看的通信热点。lmhead TP 的收益是「分片 head + 行数对齐」而非"少 gather"，跨 rank 必须严格同形（`worker/v2/model_runner.py:1194-1200` docstring）。

### 6.4 markov head 的 rank 256 显存

| 张量 | 形状 | dtype | 单份 | 副本 |
| --- | --- | --- | --- | --- |
| `markov_w1` | `[129280,256]` | bf16 | `66,191,360 B ≈ 63.1 MiB` | 每 rank 1 份（`nn.Embedding` 未切） |
| `markov_w2` | `[129280,256]` | bf16 | ≈63.1 MiB | 每 rank 1 份（`disable_tp=True`） |
| 合计 | | | **≈126.2 MiB/rank，不随 tp 缩小** | 32 卡 ≈3.9 GiB |

坑：① 切成 TP 分片会让每个 draft 位置多一次 all-reduce + 全词表 gather，上游 `qwen3_dspark.py:56-58` docstring 明说这是刻意复制；② `markov_bias` 每次产 `[B,129280]` 再相加，K 次循环产生 K 个中间张量（`B=32,K=5,bf16` ≈8.28 MB/次），MRv2 用 `+`（非原地）而参考用 `add_`（**L1154**）；③ `confidence_head` 输入是 `concat([hidden(5120), markov_embed(256)])`，内部 `.float()`，两输入必须同 dtype（`utils.py:370-373`、`dspark_proposer.py:121` 显式 `.to(flat_hidden.dtype)`）。

### 6.5 其它

| 坑 | 说明 | 证据 |
| --- | --- | --- |
| MRv1 `_compute_confidence` 是**死代码** | 无调用点，且引用的 `self._dspark_confidence_logits_buffer` **全仓库从未被赋值**（仅 `:123` 读取）；生效的是 `DynamicSpecScheduler._compute_dspark_token_probs` | [dspark_proposer.py:108-125](../../../worker/v1/spec_decode/dspark_proposer.py#L108) |
| `_sample_sequential_topk` 不可用 | 需 `apply_markov_bias_gathered`（本 drafter 未定义），且配置无 `dspark_draft_topk` | 上游 `:201-248`、`qwen3_dspark.py:290` |
| MRv1 DSpark 不走 ACLGraph | `self.use_cuda_graph = False`（*DSpark runs eager only*） | [dspark_proposer.py:72-73](../../../worker/v1/spec_decode/dspark_proposer.py#L72) |
| MRv2 DSpark 不调 `sample_draft` | 给 drafter 加 `sample_draft` 的 lmhead-TP 逻辑会**静默无效** | [speculator.py:60-65](../../../worker/v2/spec_decode/dspark/speculator.py#L60) |
| draft seed 未清零会被当真 token | `_dspark_seed_buffer[n:].fill_(0)`；batch 变小时不填 0 会用上一轮 seed 算出"合法" draft | [dspark_proposer.py:245-248](../../../worker/v1/spec_decode/dspark_proposer.py#L245) |
| lmhead TP + prompt logprobs | MRv2 直接 `NotImplementedError`；MRv1 的 `compute_logits` 只在 execute_model（**L2753**）与 profile（**L4638**）调用，`_bookkeeping_sync` 不二次算 | [model_runner.py:409-423](../../../worker/v2/model_runner.py#L409) |
| K 默认值来自 ckpt | 未显式设置 → `dspark_block_size=5`，**不是** `num_nextn_predict_layers=3` | 上游 `config/speculative.py:1483-1490` |

---

## 七、与 V4 文档的差异说明

| V4 文档（v0.32.0 `deepseek_v4` 树） | 本 checkout 实际 | 结论 |
| --- | --- | --- |
| `deepseek_v4/model.py:1154` 的 `compute_logits` | V4.1 在 [../model.py:1813-1818](../model.py#L1813)，两行 | **路径不同** |
| `compute_logits_local`（`:1161`）+ `skip_gather` | **本树 `model.py` 无 `compute_logits_local`**；`skip_gather` 仅在上游 `forward:103`/`_get_logits:189-190`；`batch_shard` 调用在**上游 MRv2** `gpu/model_runner.py:1602` | **V4 特有入口不存在** |
| `LogitsProcessor` 输出 fp32 | 全程不改 dtype；fp32 在采样侧（MRv1 无条件、MRv2 按需） | **需修正** |
| `soft_cap`/`scale` 是常规步骤 | 构造只传 `vocab_size`（**L1667**），配置无 `final_logit_softcapping` | **对 V4.1 不成立** |
| MRv1/MRv2 双栈并存、`AscendSampler` 覆写三处 | 一致（`patch_triton.py` **L130-167**；`sample/sampler.py` 共 119 行） | ✅ |
| `enable_reduce_sample`/`prepare_sampling` 已删 | `prepare_sampling` 全仓零命中；`enable_reduce_sample` **仅**出现在 `tests/ut/test_mrv2_utils.py:76-80` 作为「任意 additional_config 键」测试输入，**无实现**；`_sample`（**L3073-3094**）无 prepare | ✅（措辞修正） |
| `npu_top_k_top_p` 已删，只剩 sort+mask | 一致，`sample/sampler.py:114-118` 注释即为此写 | ✅ |
| `apply_sampling_constraints`(:325)/`rejection_sample`(:382) 是**方法** | 本树是**模块级函数**（同号），由 `forward` 在 **L226/L228** 调用 | **形态变化** |
| MRv2 默认、`mrv2_utils.py:99-133`；MRv1 有 entropy verify | 一致；另有 `platform.py:1058-1064`（A5 强制 MRv2）；`rejection_sampler.py:439-440` 读开关 | ✅ + 新增 |
| 「draft logits 走 LogitsProcessor 不进 sampler」 | 一致：`compute_draft_logits`→`LogitsProcessor`（**L371-377**），draft 由 `argmax`/`_sample_draft_from_logits` 产出，只有 target logits 进拒绝采样 | ✅ |
| 文档未覆盖 | **Markov 偏置、confidence 头、full-vocab 无 d2t、`gumbel_sample→categorical_sample`** | **V4.1 增量** |

**无法在本 checkout 验证 / 明确不存在的项**

| 项 | 状态 |
| --- | --- |
| `models/deepseek_v41/model.py::compute_logits_local` | 不存在 |
| `final_logit_softcapping`（配置或代码） | 不存在（JSON 无键，目录 grep 零命中） |
| `apply_sampling_constraints` / `rejection_sample` 作为**方法** | 不存在（模块级函数） |
| `DSparkDeepseekV41ForCausalLM.apply_markov_bias_gathered` | 不存在（仅上游 `qwen3_dspark.py:290`） |
| `draft_vocab_size != vocab_size` | 无法成立：配置无该字段，`getattr(...) or vocab_size` 恒 129280 |
| `_compute_confidence` 实际调用点 | 找不到（死代码 + 未定义 buffer） |
| MRv1+DSpark 实机跑通证据 | 代码路径完整，但 A5 被 `platform.py:1058-1064` 拦为「必须 MRv2」；非 A5 实测本文无法给出 |
| V4 文档其余 MRv1 行号（`:1763`/`:1798`/`:2016`/`:3505`） | 符号存在但**行号已变**：`propose_draft_token_ids` **L2016**、`execute_model` **L2347**，引用以本文 §0 为准 |

---

## 八、一句话总结

V4.1 后处理仍是「阶段① 出 bf16 logits（matmul + TP all-gather，无 soft-cap/scale）、阶段② 采样」；常规分支 MRv1 为「raw logprobs → processors → penalties → temperature → top-k/p → Gumbel-max」，MRv2 为「按需 fp32 → processors → thinking budget → temperature → min_p → top-k/p → `categorical_sample`」。真正属于 V4.1 的只有推测解码侧：**full-vocab draft logits（无 d2t）+ 逐位置 Markov 偏置 + sigmoid confidence 门控**——MRv1 写在 `llm_base_proposer.py:1571-1611`，MRv2 交给上游 `DSparkSpeculator._sample_sequential`（`:156-199`），Ascend 只补一层 lmhead-TP 行对齐与算子重绑。
