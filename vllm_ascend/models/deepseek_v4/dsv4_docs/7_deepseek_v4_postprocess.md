# DeepSeek V4 后处理与采样全流程

> 本文说明 DeepSeek V4 在 vllm-ascend 上的后处理链路：LM Head → LogitsProcessor → Sampler / RejectionSampler 的完整调用路径、张量形状、关键边界条件、v1 与 v2 两套采样实现，以及 v0.32.0 引入的分支语义。

| 项 | 值 |
| --- | --- |
| 文档版本 | **v0.32.0** |
| vllm-ascend | 分支 `dsv4-v0.32.0`（基于 upstream `releases/v0.32.0rc`） |
| 上游 vLLM | tag `v0.32.0` |
| 覆盖范围 | 后处理（LM Head / logits 裁剪与 gather）、常规采样、推测解码拒绝采样、收尾回写 |
| 行号约定 | 文中 `Lxxx` 均为 v0.32.0 源码当前行号；无法确定行号时只写函数/类名 |

**两套采样实现（v0.32.0 起长期并存，先看这里再看后面）：**

- **MRv1**（[vllm_ascend/worker/v1/model_runner.py](../../../../../vllm-ascend/vllm_ascend/worker/v1/model_runner.py)）：[vllm_ascend/sample/sampler.py](../../../../../vllm-ascend/vllm_ascend/sample/sampler.py)、[penalties.py](../../../../../vllm-ascend/vllm_ascend/sample/penalties.py)、[rejection_sampler.py](../../../../../vllm-ascend/vllm_ascend/sample/rejection_sampler.py)，覆写上游 `vllm/vllm/v1/sample`。DeepSeek V4 的 Ascend 专属采样逻辑主要在这里。
- **MRv2**（[vllm_ascend/worker/v2/model_runner.py](../../../../../vllm-ascend/vllm_ascend/worker/v2/model_runner.py)）：不重写 `Sampler` 类，而是通过 [vllm_ascend/patch/worker/patch_triton.py](../../../../../vllm-ascend/vllm_ascend/patch/worker/patch_triton.py) 把上游 `vllm/vllm/v1/worker/gpu/sample/*` 的算子重绑到 `vllm_ascend/worker/v2/sample/*` 与 `vllm_ascend/ops/triton/v2/sample/*`（Triton-Ascend）。

---

## 一、后处理总览

### 1.1 两阶段：execute_model / sample_tokens

后处理被拆成两个 RPC 阶段，由 `Executor.collective_rpc` 驱动（`vllm/vllm/v1/executor/abstract.py`：`execute_model` L250、`sample_tokens` L270、`collective_rpc` L177/188/199）。

#### （1）MRv1 调用栈（V1 回退路径）

```text
Executor.collective_rpc                                   # vllm/vllm/v1/executor/abstract.py:177/188/199
│
├─ 阶段① execute_model(scheduler_output)                  # vllm_ascend/worker/worker.py:874
│  └─ NPUModelRunner.execute_model                        # vllm_ascend/worker/v1/model_runner.py:2347
│     ├─ _model_forward(...)                              # :3505
│     ├─ hidden_states[logits_indices]                    # :2752  → [num_logits, hidden_size]
│     └─ model.compute_logits(sample_hidden_states)       # :2753
│        └─ vllm_ascend/models/deepseek_v4/model.py:1154
│           └─ LogitsProcessor.forward                     # vllm/vllm/model_executor/layers/logits_processor.py:98
│              ├─ _get_logits                              # :180
│              │  ├─ _apply_head                           # :136   (lm_head matmul)
│              │  ├─ _gather_logits                        # :122   (TP all-gather)
│              │  └─ logits[..., :org_vocab_size]          # :198
│              ├─ soft_cap                                 # :113-116
│              └─ logits / scale                           # :118-119
│                 └─ 结果写入 ExecuteModelState             # :2776
│
└─ 阶段② sample_tokens(grammar_output)                    # vllm_ascend/worker/worker.py:945
   └─ NPUModelRunner.sample_tokens                        # vllm_ascend/worker/v1/model_runner.py:2882
      ├─ grammar bitmask                                  # :2921-2928
      ├─ _sample(logits, spec_decode_metadata)            # :3073
      │  ├─〔常规分支〕AscendSampler.forward                 # vllm_ascend/sample/sampler.py:45（继承 Sampler，未覆写 forward）
      │  │  └─ Sampler.forward                            # vllm/vllm/v1/sample/sampler.py:73
      │  │     ├─ raw_logprobs                            # :85-94
      │  │     ├─ logits.to(torch.float32)                # :97
      │  │     ├─ apply_logits_processors                 # :99 → :375
      │  │     │  ├─ allowed_token_ids_mask               # :395-396
      │  │     │  ├─ apply_bad_words                       # :399-400
      │  │     │  ├─ non_argmax_invariant                  # :403-404
      │  │     │  └─ AscendSampler.apply_penalties        # vllm_ascend/sample/sampler.py:47
      │  │     │     └─ apply_all_penalties               # vllm_ascend/sample/penalties.py:25
      │  │     └─ sample                                  # :247
      │  │        ├─ greedy_sample                        # :242 / AscendSampler:82
      │  │        ├─ apply_temperature                    # :230
      │  │        ├─ argmax_invariant                     # :285-286
      │  │        ├─ topk_topp_sampler(...)               # :289-294
      │  │        │  └─ AscendTopKTopPSampler.forward_native   # vllm_ascend/sample/sampler.py:92
      │  │        │     ├─ apply_top_k_top_p              # :103 → :119
      │  │        │     │  └─ _apply_top_k_top_p_pytorch  # vllm_ascend/worker/v2/sample/apply_top_k_top_p.py:10
      │  │        │     ├─ logits.softmax(fp32)           # sampler.py:110
      │  │        │     └─ random_sample                  # sampler.py:18
      │  │        └─ torch.where(probs/argmax)            # :299-304
      │  │
      │  └─〔推测解码分支〕AscendRejectionSampler.__call__   # vllm_ascend/sample/rejection_sampler.py:40
      │     └─ forward                                    # :150
      │        ├─ bonus logits → self.sampler             # :196-208
      │        ├─ apply_logits_processors                 # :98
      │        ├─ apply_sampling_constraints              # :325
      │        └─ rejection_sample                        # :382
      │
      ├─ _bookkeeping_sync                                # :3110
      └─ ModelRunnerOutput / AsyncGPUModelRunnerOutput     # :3027 / :3057
```

#### （2）MRv2 调用栈（默认路径）

```text
NPUModelRunnerV2.sample_tokens                             # vllm_ascend/worker/v2/model_runner.py:408
                                                           #   super: vllm/vllm/v1/worker/gpu/model_runner.py:2088
└─ sample(hidden_states, input_batch)                       # vllm_ascend/worker/v2/model_runner.py:1193
   └─ GPUModelRunner.sample(...)                           # vllm/vllm/v1/worker/gpu/model_runner.py:1592
      ├─ compute_logits(...)                               # :1614-1615（batch_shard 时为 compute_logits_local :1610）
      ├─ grammar bitmask                                   # :1623
      ├─ self.sampler(logits, input_batch)                 # :1642
      │  └─ Sampler.__call__                               # vllm/vllm/v1/worker/gpu/sample/sampler.py:134
      │     ├─ sample()                                    # :154 → :278
      │     │  ├─ apply_sampling_params                    # :290（此处 skip_top_k_top_p=True）
      │     │  │  ├─ logits processors                     # :255-256
      │     │  │  ├─ thinking budget                       # :260
      │     │  │  ├─ apply_temperature                     # :263 → states.py:81
      │     │  │  └─ apply_min_p                           # :268 → states.py:94
      │     │  ├─ get_top_k_top_p                          # :301 → states.py:105
      │     │  └─ _sample_random                           # :327
      │     │     ├─ apply_top_k_top_p(topk_topp_sampler)  # :345
      │     │     └─ gumbel_sample → categorical_sample    # :346 / patch_triton.py:148
      │     └─ trace_replay / logprobs / SamplerOutput      # :169 / :171-188 / :213
      └─ self.rejection_sampler(...)                       # :1647（有 draft token 时）
         └─ RejectionSampler                               # vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:147
```

> MRv2 的 `sample()` 不再有 `prepare_sampling`、`_bookkeeping_sync` 等 MRv1 专属步骤；采样状态由 `vllm/vllm/v1/worker/gpu/sample/states.py` 的 `SamplingStates` 维护。

### 1.2 模块分层（v0.32.0 实际路径）

**Ascend 侧（MRv1）**

| 层级 | 文件 | 关键符号 |
| --- | --- | --- |
| 采样器 | [`vllm_ascend/sample/sampler.py`](../../../../../vllm-ascend/vllm_ascend/sample/sampler.py) | `random_sample` L18 · `AscendSampler` L45 · `AscendTopKTopPSampler` L87 · `forward_native` L92 · `apply_top_k_top_p` 别名 L119 |
| 惩罚项（Python） | [`vllm_ascend/sample/penalties.py`](../../../../../vllm-ascend/vllm_ascend/sample/penalties.py) | `_convert_to_tensors` L13 · `apply_all_penalties` L25 |
| 惩罚项（Triton） | [`vllm_ascend/ops/triton/penalty.py`](../../../../../vllm-ascend/vllm_ascend/ops/triton/penalty.py) | `apply_penalties_triton` L196 |
| 拒绝采样 | [`vllm_ascend/sample/rejection_sampler.py`](../../../../../vllm-ascend/vllm_ascend/sample/rejection_sampler.py) | `AscendRejectionSampler` L40 · `forward` L150 · `apply_sampling_constraints` L325 · `rejection_sample` L382 |
| 拒绝采样（Triton） | [`vllm_ascend/ops/triton/reject_sample.py`](../../../../../vllm-ascend/vllm_ascend/ops/triton/reject_sample.py) | `rejection_greedy_sample_with_triton` L577 · `expand_triton` L694 · `_REJECTION_RANDOM_SAMPLE_KERNEL` L1061 · `_REJECTION_RANDOM_SAMPLE_BLOCK_VERIFY_KERNEL` L1204 |

**Ascend 侧（MRv2，供上游算子重绑）**

| 层级 | 文件 | 关键符号 |
| --- | --- | --- |
| Top-K / Top-P | [`vllm_ascend/worker/v2/sample/apply_top_k_top_p.py`](../../../../../vllm-ascend/vllm_ascend/worker/v2/sample/apply_top_k_top_p.py) | `_apply_top_k_top_p_pytorch` L10 · `apply_top_k_top_p_npu` L52（0.32）/ L60（其他版本） |
| 惩罚项 | [`vllm_ascend/worker/v2/sample/penalties.py`](../../../../../vllm-ascend/vllm_ascend/worker/v2/sample/penalties.py) | `_penalties_kernel` L34 · `apply_penalties` L125 · `bincount` L246 |
| Gumbel | [`vllm_ascend/worker/v2/sample/gumbel.py`](../../../../../vllm-ascend/vllm_ascend/worker/v2/sample/gumbel.py) | `apply_temperature` L52 · `_gumbel_sample` L177 · `gumbel_sample` L274 |
| 分类采样（实际重绑） | [`vllm_ascend/ops/triton/v2/sample/categorical_sample.py`](../../../../../vllm-ascend/vllm_ascend/ops/triton/v2/sample/categorical_sample.py) | `categorical_sample` L414 |
| Min-P | [`vllm_ascend/worker/v2/sample/min_p.py`](../../../../../vllm-ascend/vllm_ascend/worker/v2/sample/min_p.py) | `apply_min_p` L73 |
| Bad Words | [`vllm_ascend/worker/v2/sample/bad_words.py`](../../../../../vllm-ascend/vllm_ascend/worker/v2/sample/bad_words.py) | `apply_bad_words` L132 |
| Logprob | [`vllm_ascend/worker/v2/sample/logprob.py`](../../../../../vllm-ascend/vllm_ascend/worker/v2/sample/logprob.py) | `compute_token_logprobs` L69 · `compute_topk_logprobs` L121 |
| 贪心拒绝采样 | [`vllm_ascend/worker/v2/sample/rejection_greedy.py`](../../../../../vllm-ascend/vllm_ascend/worker/v2/sample/rejection_greedy.py) | `rejection_greedy_sample_with_triton` L15 · `rejection_greedy_sample_pytorch` L42 |
| 随机拒绝采样（v2） | [`vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py`](../../../../../vllm-ascend/vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py) | `rejection_sample` L279 |
| 重绑入口 | [`vllm_ascend/patch/worker/patch_triton.py`](../../../../../vllm-ascend/vllm_ascend/patch/worker/patch_triton.py) | 导入 L118-L127，重绑 L130-L167 |

**上游共同依赖**

| 层级 | 文件 | 关键符号 |
| --- | --- | --- |
| LM Head / logits 处理 | [`vllm/vllm/model_executor/layers/logits_processor.py`](../../../../../vllm/vllm/model_executor/layers/logits_processor.py) | `LogitsProcessor` L58 · `forward` L98 · `_get_logits` L180 · `get_top_tokens` L201 |
| 批次元数据 | [`vllm/vllm/v1/worker/gpu_input_batch.py`](../../../../../vllm/vllm/v1/worker/gpu_input_batch.py) | `_make_sampling_metadata` L851 |
| 采样编排（v1） | [`vllm/vllm/v1/sample/sampler.py`](../../../../../vllm/vllm/v1/sample/sampler.py) | `Sampler` L22 · `forward` L73 · `sample` L247 |
| TopK/TopP 基类 | [`vllm/vllm/v1/sample/ops/topk_topp_sampler.py`](../../../../../vllm/vllm/v1/sample/ops/topk_topp_sampler.py) | `TopKTopPSampler` L202 · `forward_native` L263 · `apply_top_k_top_p` L460 |
| 采样编排（v2） | [`vllm/vllm/v1/worker/gpu/sample/sampler.py`](../../../../../vllm/vllm/v1/worker/gpu/sample/sampler.py) | `Sampler` L42 · `__call__` L134 · `sample` L278 |
| 采样状态（v2） | [`vllm/vllm/v1/worker/gpu/sample/states.py`](../../../../../vllm/vllm/v1/worker/gpu/sample/states.py) | `SamplingStates` L17 · `apply_temperature` L81 · `apply_min_p` L94 · `apply_top_k_top_p` L114 |

### 1.3 v0.32.0 关键变更（相对旧文档）

| # | 旧行为（更早版本） | v0.32.0 现状 |
| --- | --- | --- |
| 1 | `npu_apply_top_k_top_p` 自定义算子，按机型在 torch_npu 与 PyTorch sort 之间二选一 | **只剩一套 sort+mask**：`_apply_top_k_top_p_pytorch`（`worker/v2/sample/apply_top_k_top_p.py:10`）。`AscendTopKTopPSampler` 直接把模块级别名绑到它（`sample/sampler.py:119`），硬件能力分发已删除 |
| 2 | `enable_reduce_sample` + `prepare_sampling` + 候选 all-gather 分布式 argmax | **整体移除**。`AscendSampler.greedy_sample` 回到 `logits.argmax(dim=-1)`（`sample/sampler.py:82-84`）；`AscendSampler` 精简到约 119 行，仅覆写 `apply_penalties`、`greedy_sample` 并替换 `topk_topp_sampler` |
| 3 | 依赖 `do_async_exponential` / `set_q_event` 做异步随机数 | 仍不存在；`random_sample` 在 `global_stream` 上即时 `exponential_()`，再 `wait_stream` 切回（`sample/sampler.py:31-41`） |
| 4 | 单一采样栈 | **两套**：MRv1 走 `vllm_ascend/sample/*`；MRv2 走上游 `Sampler` + Triton 算子重绑（`patch/worker/patch_triton.py:130-167`） |
| 5 | 默认 MRv1 | **MRv2 为默认**，V1 为回退（见 1.4） |

### 1.4 MRv1 与 MRv2 的取舍

| 维度 | MRv1 | MRv2（默认） |
| --- | --- | --- |
| Runner | `vllm_ascend/worker/v1/model_runner.py` | `vllm_ascend/worker/v2/model_runner.py` |
| 采样类 | `AscendSampler(Sampler)` + `AscendRejectionSampler` | 上游 `Sampler` / `RejectionSampler` 原样使用 |
| Ascend 注入方式 | 子类覆写 + 替换 `topk_topp_sampler` | `patch_triton.py` 运行时重绑定算子 |
| TopK/TopP | `AscendTopKTopPSampler.forward_native` → `_apply_top_k_top_p_pytorch` | `topk_topp_sampler.apply_top_k_top_p_triton` → `apply_top_k_top_p_npu` → 同一 pytorch 实现 |
| 随机采样 | `random_sample`（exponential + div + argmax） | `categorical_sample`（`ops/triton/v2/sample/categorical_sample.py`） |
| block verify / entropy verify | 支持，读 `AscendConfig.rejection_sampler_config` | 仅支持 block verify（`spec_config.rejection_sample_method == "block"`），**无 entropy verify** |

Runner 选择逻辑：`vllm_ascend/mrv2_utils.py::use_v2_model_runner` L99-133。未显式设置 `VLLM_USE_V2_MODEL_RUNNER` 时返回 `True`；只有命中 V2 黑名单（VL encoder graph、`draft_window_size`、suffix/ngram、draft model、dflash2 graph、p-eagle 等）或显式 `VLLM_USE_V2_MODEL_RUNNER=0` 时才回退 V1。另见 `vllm_ascend/platform.py::_validate_model_runner_config` L1058-1064：DeepSeek V4.1 + A5（`DSV41_PACKED_CACHE`）要求 v2，否则直接报错。

---

## 二、阶段①：LM Head 与 logits 裁剪

### 2.1 张量形状主线

```text
hidden_states                 [num_tokens, hidden_size]
  ↓ hidden_states[logits_indices]                 # model_runner.py:2752
sample_hidden_states          [num_logits, hidden_size]
  ↓ compute_logits / LogitsProcessor.forward      # logits_processor.py:98
logits                        [num_logits, vocab_size]  (fp32)
```

`num_logits` 在常规 decode 下等于请求数；在 chunked prefill 下只取每个请求最后一个 token（`_prepare_inputs` 内的 `logits_indices` 计算，`model_runner.py:1763`）。

### 2.2 logits_indices 的三种来源

| 场景 | 位置 | 说明 |
| --- | --- | --- |
| 常规 decode / prefill | `model_runner.py:1763` | 每请求取 `num_computed_tokens - 1` 所在行 |
| 推测解码 | `model_runner.py:1791` | 在 draft token 展开后的序列上取每请求末行 |
| padding / 补齐 | `model_runner.py:1804-1806` | `lmhead_tp` 开启时补齐到 `max_num_reqs * uniform_decode_query_len`，保证 DP 间形状一致 |

（索引入口在 `model_runner.py:1798` 处写回 `self.logits_indices`；正式按 chunk 截取发生在 `:2752`。）

### 2.3 LogitsProcessor 内部顺序

| 步骤 | 位置 | 说明 |
| --- | --- | --- |
| 1. `_apply_head` | `logits_processor.py:136` | 调 `lm_head` 得到 `[num_logits, vocab_size]` |
| 2. `_gather_logits` | `:122` | TP > 1 时 all-gather 词表分片（`lmhead_tp` 场景） |
| 3. 词表裁剪 | `:198` | 只保留 `[..., :org_vocab_size]`，去掉 padding 词表 |
| 4. `soft_cap` | `:113-116` | `logits = tanh(logits / soft_cap) * soft_cap` |
| 5. `scale` | `:118-119` | DeepSeek 系列的 logits 缩放 |
| 6. `_get_logits` | `:180` | 汇总以上步骤，返回 fp32 logits |

### 2.4 模型侧入口

- **NPU 实际执行**：`vllm_ascend/models/deepseek_v4/model.py:1154`（`compute_logits`），内部持有 `LogitsProcessor`（L1114）。
- **上游参考实现**：`vllm/vllm/models/deepseek_v4/nvidia/model.py:1857`（`compute_logits`）。

### 2.5 为什么 chunked prefill 与 decode 走同一路径

`execute_model` 只负责产出 `logits`；`sample_tokens` 再按 `spec_decode_metadata` 是否存在分流。因此 prefill 与 decode 的后处理差异只体现在 `logits_indices` 的取法和 `num_logits` 大小，不在采样逻辑本身。

### 2.6 logits 的 dtype 边界

- `LogitsProcessor.forward` 输出 **fp32** logits（`logits_processor.py:98`）。
- `Sampler.forward` 再显式 `logits.to(torch.float32)`（`vllm/vllm/v1/sample/sampler.py:97`），避免 bf16 采样精度损失。
- v2 侧由 `GPUModelRunner.sample` 传入 logits（`gpu/model_runner.py:1614-1615`），LogitsProcessor 已是 fp32 输出；`worker/v2/sample/gumbel.py` 不再额外 cast（`_gumbel_sample` 直接按输入 dtype 计算，L177-205）。

### 2.7 `get_top_tokens`（跳过全量 gather）

`LogitsProcessor.get_top_tokens`（`logits_processor.py:201`）与 `get_top_k_tokens`（`:253`）用于只取 top-k 结果、不做整词表 gather 的场景，入口是投机 proposer：`vllm/vllm/v1/spec_decode/llm_base_proposer.py:441`（MTP）。

DeepSeek V4 另有本地 logits 入口：`vllm_ascend/models/deepseek_v4/model.py::compute_logits_local`（L1161），内部调 `LogitsProcessor.forward(..., skip_gather=True)`（`logits_processor.py:98-111`、`:189-190`）。MRv2 的 `batch_shard` 在 TP 间切分请求时用它只算本地词表分片，再经 `all_to_all_logits` 合并（`vllm/vllm/v1/worker/gpu/model_runner.py:1610-1612`）。

---

## 三、阶段②：sample_tokens 编排

### 3.1 sample_tokens 的步骤（MRv1）

| 步骤 | 位置 | 说明 |
| --- | --- | --- |
| 1. 取出执行态 | `model_runner.py:2903-2919` | 从 `ExecuteModelState` 还原 `logits` / `grammar` / `input_batch` |
| 2. grammar bitmask | `:2921-2928` | 结构化输出掩码；Ascend 需先 D2H 到 CPU（float32）应用 bitmask，再搬回 NPU（`:2925-2928`） |
| 3. `_sample` | `:2931` | 常规分支 / 推测解码分支的实际选择点 |
| 4. 采样完成事件 | `:2933-2938` | `sampling_done_event` 记录，供异步调度使用 |

### 3.2 `_sample` 的分流

`model_runner.py:3073-3094`：

| 分支 | 位置 | 条件 | 动作 |
| --- | --- | --- | --- |
| 常规 | `:3078-3080` | `spec_decode_metadata is None` | `lmhead_tp` 时把 logits 截回 `num_reqs` 行（`:3078-3079`），再 `self.sampler(...)`（`:3080-3083`） |
| 推测解码 | `:3085-3088` | 存在 `spec_decode_metadata` | 按 `logits_indices` 截断（`:3085-3086`），构造 `draft_probs`（`:3087`），再 `self.rejection_sampler(...)`（`:3088-3093`） |

> v0.32.0 的 `_sample` **不再调用** `prepare_sampling`；采样前的元数据准备已全部下沉到 `InputBatch` 与 `Sampler.forward` 内部。

### 3.3 禁用 TopK/TopP 的语义链（MRv1）

采样参数中的"未设置"要经过 5 层才到达算子，任何一层判定为 `None` 都会走无过滤路径：

| 层 | 位置 | 行为 |
| --- | --- | --- |
| 1. `SamplingParams` | `vllm/vllm/sampling_params.py:312/315/318` | `top_p` 默认 1.0、`top_k` 默认 0、`min_p` 默认 0.0 |
| 2. 贪心强制覆盖 | `sampling_params.py:612-617` | `temperature < _SAMPLING_EPS` 时 `top_p = 1.0`、`top_k = 0`、`min_p = 0.0` |
| 3. `InputBatch` 同值合并 | `gpu_input_batch.py:404-412` | `top_p < 1` 才登记 `top_p_reqs`；`0 < top_k < vocab_size` 才登记 `top_k_reqs`，否则把 `top_k` 规整为 `vocab_size` |
| 4. `SamplingMetadata` 落 `None` | `gpu_input_batch.py:938-939`（`_make_sampling_metadata` 在 `:851`） | `no_top_p → top_p = None`；`no_top_k → top_k = None` |
| 5. 算子 | `topk_topp_sampler.py:460` → `apply_top_k_top_p.py:15-16` | `if p is None and k is None: return logits` |

补充两条边界：

- `no_top_p` / `no_top_k` 由请求集合是否为空决定（`gpu_input_batch.py:1115-1121`），不是逐请求比较。
- 单请求内 `top_k = 0` 或 `-1` 会在 `InputBatch` 中被规整为 `vocab_size`（`gpu_input_batch.py:407-412`），该值在算子内退化为全保留。

**v2 对应层**：`SamplingStates.get_top_k_top_p`（`states.py:105-111`）同样以 `vocab_size / 1.0` 填充后判空；`apply_top_k_top_p`（`:114-123`）直接转发上游同名函数。

---

## 四、常规分支：AscendSampler

### 4.1 覆写范围

`vllm_ascend/sample/sampler.py:45-119` 中，`AscendSampler` 只做三件事：

1. `__init__`（`:72-80`）：构造 `AscendTopKTopPSampler` 取代上游默认实现。
2. `apply_penalties`（`:47-70`）：Triton 可用时走 `apply_all_penalties`，否则回退 `Sampler.apply_penalties`。
3. `greedy_sample`（`:82-84`）：纯 `argmax`。

其余全部沿用 `vllm/vllm/v1/sample/sampler.py::Sampler`。

### 4.2 `Sampler.forward` 的前半段（上游）

`vllm/vllm/v1/sample/sampler.py:73-102`：

```python
num_logprobs = sampling_metadata.max_num_logprobs              # L85
raw_logprobs = None                                           # L86
if num_logprobs is not None or sampling_metadata.logprob_token_ids:  # L87
    if logprobs_mode == "raw_logprobs":
        raw_logprobs = self.compute_logprobs(logits)          # L89
    elif logprobs_mode == "raw_logits":
        if logits.dtype == torch.float32:
            raw_logprobs = logits.clone()                     # L91-92
        else:
            raw_logprobs = logits.to(torch.float32)           # L93-94

logits = logits.to(torch.float32)                             # L97
logits = self.apply_logits_processors(logits, sampling_metadata,
                                      predict_bonus_token)    # L99-101
sampled, processed_logprobs = self.sample(logits, sampling_metadata)  # L103
```

### 4.3 `apply_logits_processors`（上游）

`vllm/vllm/v1/sample/sampler.py:375-421`，按顺序：

| 步骤 | 位置 | 说明 |
| --- | --- | --- |
| allowed token ids | `:395-396` | `masked_fill_(-inf)` |
| bad words | `:399-400` | 禁用词惩罚 |
| 非 argmax 不变 processor | `:403-404` | 如 `MinTokensLogitsProcessor` |
| penalties | `:407` | 调 `self.apply_penalties` → Ascend 覆写 |
| thinking budget | `:408-420` | v0.32.0 新增的 `thinking_budget_state_holder` 处理 |

### 4.4 推测解码下仍需重复应用

`AscendRejectionSampler.apply_logits_processors`（`rejection_sampler.py:98-148`）必须对展开后的每个 draft token 重新应用一次：

```text
repeat_indices = expand_batch_to_tokens(original_indices, ...)   # :122
logits = self.apply_penalties(..., repeat_indices, ...)         # :127
allowed_token_ids_mask → masked_fill_(-inf)                     # :130-132
bad_words                                                        # :135-136
non_argmax_invariant                                             # :138-140
thinking_budget                                                  # :141-147
```

### 4.5 `AscendSampler.apply_penalties`

`vllm_ascend/sample/sampler.py:47-70`：

```python
if not HAS_TRITON:                             # L53
    return Sampler.apply_penalties(...)        # L58
if sampling_metadata.no_penalties:             # L60
    return logits                              # L61
assert sampling_metadata.prompt_token_ids is not None   # L62
return apply_all_penalties(                    # L63-70
    logits, prompt_token_ids,
    presence_penalties, frequency_penalties, repetition_penalties,
    output_token_ids,
)
```

### 4.6 `apply_all_penalties` 与 Triton 内核

`vllm_ascend/sample/penalties.py:25-45`：

1. `_convert_to_tensors`（`:13-22`）把 `list[list[int]]` 的 output token 转成 padded int64 张量，`pad = vocab_size`，`non_blocking=True` 搬上 NPU。
2. `masked_fill_(output_tokens_t == -1, vocab_size)`（`:36`）把非法位置压到词表外。
3. 调 `apply_penalties_triton`（`ops/triton/penalty.py:196`），在一条 kernel 内完成存在性惩罚 / 频率惩罚 / 重复惩罚；`_APPLY_ALL_PENALTIES_KERNEL` 实例在 `:193`。

### 4.7 v2 侧惩罚实现差异

`vllm_ascend/worker/v2/sample/penalties.py` 是**批量打包版**（签名以 `expanded_idx_mapping` 为主）：

- `_penalties_kernel`（`:34`）：多请求 packed 计算，避免逐请求 Python 循环。
- `apply_penalties`（`:125`）：入口。
- `bincount`（`:246`）：为频率惩罚做 token 计数。

MRv1 的 `apply_all_penalties` 面向"一批请求 + spec 展开"，MRv2 的 `apply_penalties` 面向"token 级映射表"，两者不可互换。

---

## 五、常规分支：采样核心（TopK/TopP/Temperature/Greedy）

### 5.1 `Sampler.sample` 的判定顺序

`vllm/vllm/v1/sample/sampler.py:247-305`：

```text
if sampling_metadata.all_random:                 # :260
    greedy_sampled = None                        # :261
else:
    greedy_sampled = self.greedy_sample(logits)  # :263
    if all_greedy: return greedy_sampled, ...    # :264-274（提前返回）

logits = self.apply_temperature(...)             # :279-281
for processor in argmax_invariant: ...           # :285-286
random_sampled, processed_logprobs = \
    self.topk_topp_sampler(...)                  # :289-294
sampled = torch.where(temperature < _SAMPLING_EPS,
                      greedy_sampled, random_sampled)   # :299-304
```

> v0.32.0 的混批逻辑是"先算 greedy、再算 random、最后 `torch.where` 按请求挑选"，不再像旧实现那样在 `apply_temperature` 后单独分支。

### 5.2 `greedy_sample`：v0.32.0 已回到 argmax

`vllm_ascend/sample/sampler.py:82-84`：

```python
@staticmethod
def greedy_sample(logits):
    return logits.argmax(dim=-1).view(-1)
```

上游 `Sampler.greedy_sample`（`vllm/vllm/v1/sample/sampler.py:242-245`）仍保留 CPU 分支（用于 `logprobs` 模式），NPU 上由 `AscendSampler` 覆写。

**旧版 `enable_reduce_sample` 的候选 all-gather + 分布式 argmax 路径在 v0.32.0 已删除**：配置项、`prepare_sampling`、相关 Triton kernel 均不再存在。因此：

- 每张卡独立在**本地 logits 分片**上做 argmax；
- 正确性依赖 `LogitsProcessor._gather_logits`（`logits_processor.py:122`）在采样前已完成 gather，或依赖 rank0 采样模式。

### 5.3 `apply_temperature`

- 上游定义：`vllm/vllm/v1/sample/sampler.py:230-239`，混批时先用 `torch.where(temp < _SAMPLING_EPS, 1.0, temp)` 把贪心请求抬成 1.0（`:237-238`），再原地 `logits.div_(temp)`（`:239`）。
- 贪心请求被强制覆盖为 `temperature = 0` 的语义来自 `SamplingParams`（`sampling_params.py:612-617`）。

### 5.4 `AscendTopKTopPSampler.forward_native`

`vllm_ascend/sample/sampler.py:92-111`：

```python
def forward_native(self, logits, generators, k, p):
    if envs.VLLM_BATCH_INVARIANT:            # :96
        return super().forward_native(...)   # :101
    logits = self.apply_top_k_top_p(logits, k, p)        # :103
    ...
    probs = logits.softmax(dim=-1, dtype=torch.float32)  # :110
    return random_sample(probs, generators), logits_to_return   # :111
```

要点：

1. `self.apply_top_k_top_p` 在 `__init__` 中绑定（`:90`），指向模块级别名（`:119`），**不再按设备能力分派**。
2. `BATCH_INVARIANT` 模式必须回退上游实现，否则批不变性失效（`:94-101`）。
3. `logprobs_mode == "processed_logits" / "processed_logprobs"` 时额外返回 `logits_to_return`（`:105-108`）。
4. NPU 上永远走 `forward_native`：`TopKTopPSampler.__init__` 在 `topk_topp_sampler.py:209-258` 中按平台绑定 `self.forward`，非 CUDA / CPU / XPU / ROCm 分支落到 `self.forward_native`（`:257-258`），而 `forward_native` 定义在 `:263`。

### 5.5 Ascend 侧唯一的 TopK/TopP 实现：sort + mask

**为什么只有一套**：`vllm_ascend/sample/sampler.py:114-118` 的注释说明——旧 `npu_top_k_top_p` 在 k 较大或批内 k 值混合时会退化到 5–28 ms，而 sort+mask 稳定在 ~1 ms；两者语义一致，因此硬件能力分发被删除，实现只保留在 `worker/v2/sample/` 下，以便 MRv1 采样包后续移除后仍可复用。

`_apply_top_k_top_p_pytorch`（`vllm_ascend/worker/v2/sample/apply_top_k_top_p.py:10-43`）：

```python
if p is None and k is None:
    return logits                                        # :15-16

probs = logits.softmax(dim=-1)                           # :18
probs_sort, _ = probs.sort(dim=-1, descending=False)     # :19（唯一一次排序）

if k is not None:
    top_k_count = probs_sort.size(1) - k                 # :22
    top_k_cutoff = probs_sort.gather(-1, top_k_count)    # :24
    no_top_k_mask = (k == logits.shape[1])               # :27
    top_k_cutoff.masked_fill_(no_top_k_mask, -inf)       # :28
    logits.masked_fill_(probs < top_k_cutoff, -inf)      # :30-31

if p is not None:
    cumprob = probs_sort.cumsum(dim=-1)                  # :34
    top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)       # :35（低尾反推）
    top_p_mask[:, -1] = False                            # :36（至少保留一个）
    top_p_count = top_p_mask.sum(dim=-1)                 # :38
    top_p_cutoff = probs_sort.gather(-1, top_p_count)    # :39
    logits.masked_fill_(probs < top_p_cutoff, -inf)      # :40-41
```

几个边界：

1. `k == vocab_size` 被显式识别为"不过滤"（`no_top_k_mask` 把 cutoff 置 `-inf`），而不是依赖下界。
2. `p` 的判定走 `cumprob <= 1 - p` 的**低尾**，等价于"保留累计概率 >= p 的最小集合"；`top_p_mask[:, -1] = False` 保证至少保留一个 token。
3. k、p 同时存在时**只有一次 softmax、一次排序**（`:18-19`），随后分别下 mask；旧文档的"两次排序"不成立。
4. `-inf` 掩码后仍参与 softmax，概率为 0，不影响采样。

`apply_top_k_top_p_npu`（`:52-54` / `:60-67`）只是给 MRv2 上游 `topk_topp_sampler.apply_top_k_top_p_triton` 用的同名包装（`vllm_version_is("0.32.0")` 分支决定是否接受 `has_topp_only_rows` 参数），内部仍调用 `_apply_top_k_top_p_pytorch`；不存在第二套算法。

### 5.6 `random_sample`（MRv1）

`vllm_ascend/sample/sampler.py:18-42`：

```text
with npu_stream_switch(global_stream()):      # :31
    q = torch.empty_like(probs)               # :32
    q.exponential_()                          # :34   （无 per-request seed）
    q[i].exponential_(generator=...)          # :38-39（有 seed 的请求逐条覆盖）
torch.npu.current_stream().wait_stream(global_stream())   # :40
return probs.div_(q).argmax(dim=-1).view(-1)              # :42
```

Gumbel-max 恒等式：`argmax(probs / q) ≡ argmax(log probs - log q)`，其中 `q ~ Exp(1)`，避免 `torch.multinomial` 的 CPU-NPU 同步。

**MRv2 差异**：不再使用 Gumbel-max，而是 `ops/triton/v2/sample/categorical_sample.py::categorical_sample`（别名在 `:414`），按块质量做分层分类采样，支持 `expanded_idx_mapping`、`logits_cache`、`use_fp64` 等参数。`worker/v2/sample/gumbel.py` 仅在补丁未覆盖或回退时使用（`gumbel_sample` 别名在 `:274`）。

### 5.7 `forward` 收尾

`vllm/vllm/v1/sample/sampler.py:103-149`：

| 行 | 内容 |
| --- | --- |
| `:103` | `sampled, logprobs = self.sample(...)` |
| `:110` | `sampled_token_ids = sampled.long()` |
| `:121-137` | 按 `logprobs_mode` 组装 `logprobs` |
| `:140` | `sampled_token_ids.to(torch.int32)` |
| `:143-149` | 构造 `SamplerOutput` |

### 5.8 关键行号速查（MRv1 常规分支）

| 主题 | 位置 |
| --- | --- |
| `Sampler` 类 | `vllm/vllm/v1/sample/sampler.py:22` |
| `forward` | `:73` |
| `apply_logits_processors` | `:375` |
| `apply_temperature` | `:230` |
| `greedy_sample` | `:242` |
| `sample` | `:247` |
| `compute_logprobs` | `:308` |
| `topk_topp_sampler` 调用 | `:289-294` |
| `AscendSampler` | `vllm_ascend/sample/sampler.py:45` |
| `AscendSampler.apply_penalties` | `:47` |
| `AscendSampler.greedy_sample` | `:82` |
| `AscendTopKTopPSampler` | `:87` |
| `forward_native` | `:92` |
| `apply_top_k_top_p` 别名 | `:119` |
| `_apply_top_k_top_p_pytorch` | `worker/v2/sample/apply_top_k_top_p.py:10` |

---

## 六、推测解码分支：AscendRejectionSampler

### 6.1 覆写点（MRv1）

| 符号 | 位置 | 作用 |
| --- | --- | --- |
| `AscendRejectionSampler` | `vllm_ascend/sample/rejection_sampler.py:40` | 继承上游 `RejectionSampler` |
| `apply_penalties` | `:51` | 对展开后的 draft 行做惩罚 |
| `__init__` | `:84-96` | 透传 `sampler / spec_config / device`，初始化 synthetic 模式状态 |
| `apply_logits_processors` | `:98-148` | 避免上游 H2D 同步，改用 `expand_batch_to_tokens` |
| `forward` | `:150-257` | 组装 logits、调 `apply_sampling_constraints` 与 `rejection_sample` |
| `apply_sampling_constraints` | `:325-379` | temperature / top_k / top_p 按 draft token 展开 |
| `rejection_sample` | `:382-689` | 贪心 / 随机 / block-verify 的实际分派 |

> 注：`apply_sampling_constraints`（`:325`）内部调用的 `apply_top_k_top_p` 来自上游 `vllm/vllm/v1/sample/ops/topk_topp_sampler.py:460`（同样基于排序掩码），与本文件常规分支的 `_apply_top_k_top_p_pytorch` 算法等价，但入口不同。

### 6.2 `forward` 主流程

`rejection_sampler.py:150-257`：

```text
1. max_spec_len 上限断言                              # :181
2. bonus_logits = logits[bonus_logits_indices]        # :196
   bonus_token_ids = self.sampler(...)                # :197-208
3. raw_target_logits = logits[target_logits_indices].float()   # :213-216
4. apply_logits_processors(...)                       # :222
5. apply_sampling_constraints(...)                    # :226
6. rejection_sample(...)                              # :228-240
7. logprobs（可选）+ 构造 SamplerOutput                # :244-258
```

### 6.3 block verify 与 entropy verify 仍然存在

`rejection_sample`（`:382`）读 `AscendConfig`：

```python
using_block_verify = max_spec_len >= 3 and \
    bool(get_ascend_config().rejection_sampler_config.enable_block_verify)   # :439
using_entropy_verify = \
    bool(get_ascend_config().rejection_sampler_config.enable_entropy_verify) # :440
posterior_threshold / posterior_alpha                                      # :452-453
```

约束：

- block verify 要求 `max_spec_len >= 3`（`:439`），且与 `synthetic_mode` **互斥**（`:446-449` 直接 `raise ValueError`）。
- entropy verify 在 `synthetic_mode` 下由 kernel 的 `SYNTHETIC_MODE` 分支短路，配置本身允许开启。
- entropy verify 的接受率修正：`threshold = min(exp(-entropy * posterior_alpha), posterior_threshold)`，再用 `target/draft >= threshold * uniform` 判定。PyTorch 实现见 `:895-905`，Triton 实现见 `ops/triton/reject_sample.py:291-297`。

### 6.4 分派矩阵

| 场景 | Triton 路径 | PyTorch 回退 |
| --- | --- | --- |
| 全贪心 / 含贪心请求 | `rejection_greedy_sample_with_triton`（`rejection_sampler.py:517`，实现在 `ops/triton/reject_sample.py:577`） | `rejection_greedy_sample_spec_len_1_pytorch`（`:534`）或 `rejection_greedy_sample_pytorch`（`:544`） |
| 随机采样，无 block verify | `_REJECTION_RANDOM_SAMPLE_KERNEL`（`:590`，定义在 `ops/triton/reject_sample.py:1061`） | `rejection_random_sample_pytorch`（`:618`） |
| 随机采样，block verify | `_REJECTION_RANDOM_SAMPLE_BLOCK_VERIFY_KERNEL`（`:643`，定义在 `ops/triton/reject_sample.py:1204`） | `rejection_random_sample_block_verify_pytorch`（`:669`） |

> 注意：`using_block_verify or using_entropy_verify` 只影响日志级别（`:498-510`），实际分派以 `using_block_verify` 为唯一开关；entropy verify 通过 `entropy_verify=` 参数传入两条随机 kernel（`:610`、`:661`）。

### 6.5 recovered token

先 `sample_recovered_tokens`（`rejection_sampler.py:577` 调用，定义在 `:735-800`）：

- Triton 主路径：`_SAMPLE_RECOVERED_TOKENS_KERNEL`（`:766-777`，定义在 `ops/triton/reject_sample.py:911`）；
- PyTorch 回退：block verify 用 `sample_recovered_tokens_blockwise_pytorch`（`:779-788`，定义在 `:1191`），否则用 `sample_recovered_tokens_pytorch`（`:790-799`，定义在 `:1010`）。

### 6.6 MRv2 的拒绝采样

MRv2 使用上游 `vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py::RejectionSampler`（类在 `:147`，`__init__` 在 `:148`，block 模式在 `:174`，`_verify` 在 `:244`，`_verify_in_chunks` 在 `:289`），Ascend 侧只通过：

- `vllm_ascend/worker/v2/sample/rejection_greedy.py`（`rejection_greedy_sample_with_triton` L15、`rejection_greedy_sample_pytorch` L42）；
- `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py::rejection_sample`（L279）；
- `vllm_ascend/ops/triton/reject_sample.py` 的 kernel

替换底层算子。

**能力差异**：

| 能力 | MRv1 | MRv2 |
| --- | --- | --- |
| 贪心拒绝采样 | ✅ | ✅ |
| 随机拒绝采样 | ✅ | ✅ |
| block verify | ✅（`enable_block_verify` + `max_spec_len >= 3`） | ✅（`spec_config.rejection_sample_method == "block"`） |
| entropy verify | ✅（`enable_entropy_verify`） | ❌ |
| synthetic 模式 | ✅ | ✅（上游实现） |

### 6.7 早期 exit：全贪心

`rejection_sampler.py:557-558`：`if sampling_metadata.all_greedy: return output_token_ids`，随机采样内核完全跳过。这是 DeepSeek V4 投机解码（`temperature=0`）最常用的路径。

> 注：`:512` 注释仍写 "need to do allgather first to get global argmax"，但 `:514` 的 `target_logits.argmax(dim=-1)` 作用在 `LogitsProcessor` 已 gather 的完整词表上（`logits_processor.py:122`），代码本身不做额外 all-gather；该注释为历史遗留。

---

## 七、收尾：回写与调度

### 7.1 `_bookkeeping_sync`（MRv1）

`model_runner.py:3110`：

| 步骤 | 位置 | 说明 |
| --- | --- | --- |
| 1. NaN 计数（可选） | `:3130-3136` | `VLLM_COMPUTE_NANS_IN_LOGITS` 开启时统计；异步调度下留在 device |
| 2. 取出采样结果 | `:3150-3153` | `sampled_token_ids` / `logprobs_tensors` |
| 3. 采样器 generator 状态回滚 | `:3139-3143` | 被丢弃的请求把 generator offset 回退 4，避免下次采样状态错位 |
| 4. 组装输出（`sample_tokens` 内） | `:3027-3038` | 用 bookkeeping 返回的 `req_ids` / `sampled_token_ids` / `logprobs` 构造 `ModelRunnerOutput` |
| 5. 异步输出包装 | `:3057-3065` / `:3066-3069` | `AsyncGPUModelRunnerOutput` + `set_async_sampled_token_ids` |

> accepted token 的统计不在 `_bookkeeping_sync`，而在 `_update_states_after_model_execute`（见 7.2）。

### 7.2 阶段①后的状态写回

`model_runner.py:3044-3051`：仅当 `need_accepted_tokens`（speculative + 需要统计）时，在 `global_stream` 上等待 `sampling_done_event` 后调用 `_update_states_after_model_execute(output_token_ids, scheduler_output)`。

- accepted token 统计：`self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)`（`:1052`）；
- 同步调度下直接 `super()._update_states_after_model_execute(...)`（`:1044`）；
- 异步调度下只对 hybrid 模型（Mamba 类）做额外处理（`:1045-1077`），纯 Transformer（如 DeepSeek V4）只完成 accepted token 计数与 event 记录（`:1052`、`:1076-1077`）。

### 7.3 推测解码的 draft 产出

- 草案生成：`model_runner.py:2016`（`propose_draft_token_ids`），在 `sample_tokens` 内被调用（`:2946` 处的嵌套逻辑，实际调用 `:3015` / `:3019`）。
- draft token 的 `logits` 仍走 `LogitsProcessor`，但不经过 `sampler`；只有主模型目标 logits 进入拒绝采样。

### 7.4 调度侧消费

- `vllm/vllm/v1/engine/core.py:657`（`step`）、`:668`（`execute_model`）、`:669`（`get_grammar_bitmask`）、`:676`（`sample_tokens(grammar_output)`）、`:681`（`update_from_output`）。
- `update_from_output` 读取 `sampled_token_ids`，对每个请求判断是否 EOS / 达到 `max_tokens` / 命中 stop 字符串，决定 `FINISHED_STOPPED` 或继续调度。

### 7.5 端到端时序

```text
Scheduler → Executor.execute_model → NPUModelRunner.execute_model
    → LogitsProcessor → ExecuteModelState（logits 驻留 NPU）
    → （可选）propose_draft_token_ids
Scheduler → Executor.sample_tokens → NPUModelRunner.sample_tokens
    → Sampler / RejectionSampler → sampled_token_ids
    → _bookkeeping_sync → ModelRunnerOutput
    → CoreEngine.update_from_output → 下一轮调度
```

---

## 八、性能与设备行为

| 主题 | v0.32.0 行为 |
| --- | --- |
| TopK/TopP | 唯一实现为 sort+mask（`_apply_top_k_top_p_pytorch`），无 torch_npu 自定义算子、无硬件分发；k 大或批内混合 k 时性能稳定（注释见 `sample/sampler.py:114-118`） |
| 随机采样 | MRv1：exp + div + argmax，global stream 上批量生成；MRv2：分块质量分类采样 |
| Reduce Sample | 已移除，greedy 直接 argmax |
| BATCH_INVARIANT | 强制回退上游 `forward_native`（`sample/sampler.py:96-101`），保证批不变性 |
| 惩罚项 | MRv1 一次 kernel 完成三类惩罚；MRv2 packed 多请求一次 kernel |
| 拒绝采样 | block verify 可提升接受率但要求 `max_spec_len >= 3`；entropy verify 仅 MRv1 |
| 异步调度 | 采样完成事件 `sampling_done_event` 让 logits 驻留 NPU，减少 D2H |
| 默认 runner | MRv2；V1 仅在 `VLLM_USE_V2_MODEL_RUNNER=0` 或命中 V2 黑名单时使用 |

---

## 九、一句话总结

v0.32.0 的 DeepSeek V4 后处理仍是"阶段① 算 logits、阶段② 采样"的两段式；但采样栈已按 runner 拆成两套：**MRv2（默认）用上游 Sampler/RejectionSampler + Triton 算子重绑**，**MRv1（回退）用精简后的 `AscendSampler`/`AscendRejectionSampler`**。TopK/TopP 收敛为唯一的 sort+mask 实现，`npu_top_k_top_p`、`enable_reduce_sample`、`prepare_sampling` 全部退场；MRv1 仍保留 block verify 与 entropy verify，MRv2 只保留 block verify。
