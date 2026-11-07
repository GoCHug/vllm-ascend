# DeepSeek V4 后处理与采样全流程详解

> 从模型最后一层隐藏状态到采样出 token 的完整后处理流水线：词表并行 LM Head、logits 缩放 / soft cap、惩罚项、温度、TopK/TopP、Gumbel 随机采样、TP Reduce Sample，以及推测解码的拒绝采样。DeepSeek V4 本身没有自定义采样逻辑，复用 vLLM v1 通用采样栈 + vllm-ascend 的 NPU 加速实现，因此本文同时覆盖上游 vLLM 与 Ascend 两侧。

**代码基线**：vllm-ascend **v0.26.0rc** + 配套上游 vLLM；文中 `Lxxx` 均指对应文件行号。

| 侧别 | 模块 | 文件 | 关键符号（行号） |
|------|------|------|------------------|
| Ascend | 采样器 | [sample/sampler.py](../../sample/sampler.py) | `random_sample` L20 · `AscendSampler` L47 · `AscendTopKTopPSampler` L110 |
| Ascend | 惩罚项 | [sample/penalties.py](../../sample/penalties.py) | `apply_all_penalties` L25 |
| Ascend | 拒绝采样 | [sample/rejection_sampler.py](../../sample/rejection_sampler.py) | `AscendRejectionSampler` L37 · `rejection_sample` L416 |
| 上游 | LM Head | [logits_processor.py](../../../../vllm/vllm/model_executor/layers/logits_processor.py) | `LogitsProcessor` L24 · `_get_logits` L138 |
| 上游 | 批次元数据 | [gpu_input_batch.py](../../../../vllm/vllm/v1/worker/gpu_input_batch.py) | SamplingMetadata 装配 · L834 |
| 上游 | 采样编排 | [v1/sample/sampler.py](../../../../vllm/vllm/v1/sample/sampler.py) | `Sampler.forward` L72 · `sample` L242 |
| 上游 | TopK/TopP 实现 | [topk_topp_sampler.py](../../../../vllm/vllm/v1/sample/ops/topk_topp_sampler.py) | `TopKTopPSampler.forward_native` L123 |

**统一模型配置**（W4A8 版；下文 TP 举例取 8，每卡本地词表 `V_local = vocab_size / 8`）：

| 配置项 | 值 / 说明 |
|--------|-----------|
| `vocab_size` | 129280（含 padding 的实际词表张量可能更大） |
| `hidden_size` | 7168 |
| 后处理投影 | `(num_tokens, 7168) → (num_tokens, vocab_size)` |

**v0.26 两处重要变更**（旧资料常见过时点）：

| # | 旧做法（已失效） | 本版本现状 |
|---|------------------|------------|
| 1 | 自定义 AscendC 算子 `npu_apply_top_k_top_p`（对应 csrc 目录已移除） | A2/A3 走 `torch_npu.npu_top_k_top_p`；其他机型（含 A5）走纯 PyTorch sort+cumsum（见 §5.5） |
| 2 | `do_async_exponential` / `set_q_event` 那套"异步指数采样" | 方法已不存在；随机数改由 `random_sample` 内部在 global stream 上即时生成（见 §5.6） |

---

## 一、后处理总览

### 1.1 框架调用栈：从 hidden_states 到 sampled_token_ids

框架并不是"一次前向顺手采完"：每个调度步，执行器通过两次 RPC **分阶段**驱动 Worker —— 阶段①前向只产出 logits 并暂存，阶段②再独立采样，两个阶段之间还可以插入 grammar bitmask 等调度逻辑。下面这棵树就是框架处理后处理全流程的真实调用栈（行号锚点见文首代码基线表）：

```
Executor.collective_rpc                          # v1/executor/abstract.py:221 / :241
│
├─ 阶段① execute_model(scheduler_output)                         # worker/worker.py:653
│  └─ NPUModelRunner.execute_model                               # worker/model_runner_v1.py:1799
│     ├─ _model_forward(...)                                     # :2155
│     │  └─ hidden_states (num_tokens, 7168)
│     ├─ hidden_states[logits_indices]                           # :2184  每请求只取最后一个待采样位置
│     └─ model.compute_logits(sample_hidden_states)              # :2185  deepseek_v4/<平台>/model.py:1443
│        └─ LogitsProcessor.forward(lm_head, hidden_states)      # model_executor/layers/logits_processor.py:64
│           ├─ _get_logits                                       # :138
│           │  ├─ _apply_head             词表并行：每卡只算 V_local 列（量化 LM Head）  # :99
│           │  ├─ _gather_logits          rank0 gather / all-gather 拼出全词表          # :85
│           │  └─ [..., :org_vocab_size]  去掉 vocab padding                            # :152
│           ├─ soft_cap：tanh(x/cap)*cap   （DeepSeek V4 为 None，no-op）               # :76
│           └─ scale：logits *= scale      （DeepSeek V4 = 1.0，no-op）                 # :81
│              └─ logits (num_reqs, 129280) 存入 ExecuteModelState，阶段①结束           # :2208
│
└─ 阶段② sample_tokens(grammar_output)                           # worker/worker.py:720
   └─ NPUModelRunner.sample_tokens                               # worker/model_runner_v1.py:2231
      # 采样参数取自 input_batch.sampling_metadata（温度 / k / p / 惩罚 / generators，见 §3.3）
      ├─ _sample(logits, spec_decode_metadata)                   # :2464
      │  ├─ prepare_sampling(max_topk)   仅 reduce_sample 且本批有 top_k 时调用         # :2473
      │  │
      │  ├─〔常规请求：spec_decode_metadata=None〕AscendSampler.__call__               # sample/sampler.py:47
      │  │  └─ Sampler.forward                                                        # v1/sample/sampler.py:72
      │  │     ├─ logits.to(float32)                                                  # :96
      │  │     ├─ apply_logits_processors（greedy 请求同样执行，不能跳过）              # :371
      │  │     │  ├─ allowed_token_ids_mask：屏蔽位置填 -inf                           # :391
      │  │     │  ├─ apply_bad_words                                                  # :395
      │  │     │  ├─ non_argmax_invariant 插件（guided decoding / 最小 token 数…）     # :399
      │  │     │  └─ apply_penalties（Ascend 覆写）                                    # sample/sampler.py:48
      │  │     │     └─ apply_all_penalties → Triton-Ascend 批量 kernel               # sample/penalties.py:25
      │  │     └─ sample                                                               # v1/sample/sampler.py:242
      │  │        ├─ greedy_sample：temperature≈0 取 argmax（all_greedy 在此直接返回）  # :259
      │  │        │  └─ reduce_sample：本地 (max, 全局下标) → 两趟 all-gather 选全局最大 # sample/sampler.py:87
      │  │        ├─ apply_temperature：logits.div_(T)                                 # v1/sample/sampler.py:227
      │  │        ├─ argmax_invariant 插件                                             # :281
      │  │        ├─ topk_topp_sampler(logits, generators, k, p)                      # :285
      │  │        │  └─ AscendTopKTopPSampler.forward_native（构造时已挂为 self.forward）# sample/sampler.py:122
      │  │        │     ├─ apply_top_k_top_p：按机型二选一的分发别名                    # sample/sampler.py:268
      │  │        │     │  ├─ A2/A3 + reduce：torch_npu.npu_top_k_top_p               # :237
      │  │        │     │  └─ A5 等 / 非 reduce：PyTorch sort+cumsum，落选位置 -inf     # :162
      │  │        │     │     （reduce：先本地 topk → 下标加 rank*V_local → all-gather 全局候选）
      │  │        │     ├─ softmax(dim=-1, dtype=fp32)
      │  │        │     ├─ random_sample（Gumbel-max，替代 multinomial）               # sample/sampler.py:20
      │  │        │     │  └─ global_stream 生成指数分布 q → probs.div_(q).argmax（无 host 同步）
      │  │        │     └─ reduce：cand_idx.gather 把候选内位置映射回全局 token id      # :148
      │  │        └─ torch.where：按 temperature 合并 greedy / random 两路              # v1/sample/sampler.py:295
      │  │
      │  └─〔推测解码：spec_decode_metadata≠None〕AscendRejectionSampler.__call__       # sample/rejection_sampler.py:37
      │     └─ forward                                                                  # :150
      │        ├─ bonus token 采样（额外补一位草稿）
      │        ├─ apply_logits_processors + apply_sampling_constraints                  # :104 / :347
      │        │  （温度 / k / p 扩展到每个 draft token，分布式 topk-allgather-topp）
      │        └─ rejection_sample：逐 token 比对 draft，接受 / 拒绝 / 按目标分布重采样   # :416
      │           （block verify、entropy verify，Triton / PyTorch 双实现，见 §6.3）
      │
      ├─ _update_states_after_model_execute(sampled_token_ids)                         # :2412
      └─ ModelRunnerOutput(sampled_token_ids, logprobs, …)
            # :2381 → sampled_token_ids 上交 EngineCore，进入下一步调度
```

### 1.2 章节地图

| 树节点 | 展开章节 |
|--------|----------|
| 阶段① `execute_model`（前向 → logits） | 第二章 |
| 阶段② `sample_tokens` / `_sample`、SamplingMetadata | 第三章 |
| 常规分支 `Sampler.forward`（加工 + 采样决策） | 第四、五章 |
| 推测解码分支 `AscendRejectionSampler` | 第六章 |
| 采样收尾（bookkeeping / 输出封装） | 第七章 |
| 设备适配与性能总结 | 第八章 |

### 1.3 模块分层

| 层级 | 模块 | 文件 : 行 | 职责 |
|------|------|-----------|------|
| 模型层 | `LogitsProcessor` | [logits_processor.py:L24](../../../../vllm/vllm/model_executor/layers/logits_processor.py#L24) | LM Head + soft cap + scale + TP gather |
| 采样编排 | `Sampler` | [v1/sample/sampler.py](../../../../vllm/vllm/v1/sample/sampler.py)（`sample` L242） | 串起惩罚 / 温度 / top-k-top-p / 随机采样 |
| NPU 采样器 | `AscendSampler` | [sampler.py:L47](../../sample/sampler.py#L47) | 覆写惩罚、greedy，装配 NPU top-k-top-p |
| NPU top-k-top-p | `AscendTopKTopPSampler` | [sampler.py:L110](../../sample/sampler.py#L110) | `forward_native` 走 NPU 实现 |
| 过滤函数 | `apply_top_k_top_p` | [sampler.py:L268](../../sample/sampler.py#L268) | 按机型二选一（torch_npu / pytorch） |
| 惩罚项 | `apply_all_penalties` | [penalties.py:L25](../../sample/penalties.py#L25) | Triton-Ascend 批量惩罚 |
| 拒绝采样 | `AscendRejectionSampler` | [rejection_sampler.py:L37](../../sample/rejection_sampler.py#L37) | 推测解码 draft 验证 |

---

## 二、阶段①：execute_model —— 前向产出 logits

### 2.1 为什么采样要拆成两次 RPC

v1 的 EngineCore 每个调度步先向执行器发起一次**非阻塞**前向，趁前向在 Worker 上跑的时候在本地算 grammar bitmask，等前向的 future 返回 `None`（表示 logits 已就绪但尚未采样）后，再发起第二次 RPC 采样：

```python
# v1/engine/core.py:586-595
scheduler_output = self.scheduler.schedule(...)
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)   # 与前向并行
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)    # 阶段②
```

执行器侧两次调用都收敛为同一个 `collective_rpc`（[abstract.py:L221](../../../../vllm/vllm/v1/executor/abstract.py#L221) / [L241](../../../../vllm/vllm/v1/executor/abstract.py#L241)），由 Worker 的两个方法承接（[worker.py:L653](../../worker/worker.py#L653) / [L720](../../worker/worker.py#L720)）。

这样拆分有三个直接好处：

- grammar bitmask 等 CPU 侧调度工作与 NPU 前向**并行**，不占用前向时间；
- logits 不跨进程回传，只留在 Worker 上暂存于 `ExecuteModelState`（[model_runner_v1.py:L2208](../../worker/model_runner_v1.py#L2208)），阶段②直接取用；
- 异步调度 / PP 场景下，非末 rank 可以在阶段①直接返回 `IntermediateTensors`（[:L2166-L2174](../../worker/model_runner_v1.py#L2166)），不参与采样。

### 2.2 定位采样位：hidden_states[logits_indices]

一个调度步里每个请求可能喂入多个 token（chunked prefill、spec decode），但只在**每请求最后一个待采样位置**取隐藏状态。索引在 `_prepare_inputs` 阶段算好（[model_runner_v1.py:L1276](../../worker/model_runner_v1.py#L1276)）：

```python
# worker/model_runner_v1.py:1275-1276（常规批）
num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
```

- 常规批：取每个请求 token 区间的末尾下标，即"每请求 1 个采样位"；
- 推测解码批：改用 `spec_decode_metadata.logits_indices`（[:L1304](../../worker/model_runner_v1.py#L1304)），每请求有 `num_draft_tokens + 1` 个采样位（[:L1305](../../worker/model_runner_v1.py#L1305)）；
- chunked prefill 未完成的请求也会被简单采一个 token，后续由调度器丢弃（[:L1268-L1272](../../worker/model_runner_v1.py#L1268)）；
- 开启 `lmhead_tp_enable` 时索引还会 pad 到跨 DP 统一长度（[:L1317-L1319](../../worker/model_runner_v1.py#L1317)）。

随后在后处理段做花式索引（[:L2184](../../worker/model_runner_v1.py#L2184)）：`sample_hidden_states = hidden_states[logits_indices]`，形状收敛为 `(num_reqs, 7168)`。

### 2.3 词表并行（VocabParallel）

LM Head 本质是以词表维做列并行的线性层：

- 每个 TP rank 只持有 `V_local = vocab_size / tp_size` 列权重，独立算出本地分片 logits `(num_reqs, V_local)`；
- 再通过 gather / all-gather 拼成完整 `(num_reqs, vocab_size)`；
- lm_head 权重通常与输入端 embedding 共享（tie-weight），也可独立。

### 2.4 compute_logits 与 _apply_head

DeepSeek V4 模型侧入口很薄（[nvidia/model.py:L1443](../../../../vllm/vllm/models/deepseek_v4/nvidia/model.py#L1443)，amd / xpu 目录下各有等价实现）：

```python
def compute_logits(self, hidden_states):
    logits = self.logits_processor(self.lm_head, hidden_states)
    return logits
```

`LogitsProcessor.forward`（[logits_processor.py:L64](../../../../vllm/vllm/model_executor/layers/logits_processor.py#L64)）先调 `_get_logits`（[L138](../../../../vllm/vllm/model_executor/layers/logits_processor.py#L138)）：

```python
# model_executor/layers/logits_processor.py:138-153
def _get_logits(self, hidden_states, lm_head, embedding_bias):
    logits = self._apply_head(lm_head, hidden_states, embedding_bias)  # 本地分片
    logits = self._gather_logits(logits)                               # TP 聚合
    if logits is not None:
        logits = logits[..., : self.org_vocab_size]                    # 去掉 padding
    return logits
```

`_apply_head`（[L99-L136](../../../../vllm/vllm/model_executor/layers/logits_processor.py#L99)）封装量化 LM Head 的前向；当配置了 `head_dtype` 且在 CUDA/ROCm 上可直接累加 fp32，NPU 走常规的 `lm_head.quant_method.apply(...)` 路径。

### 2.5 _gather_logits 两种策略

[logits_processor.py:L85-L97](../../../../vllm/vllm/model_executor/layers/logits_processor.py#L85)，由 `self.use_all_gather`（`current_platform.use_all_gather()`）决定：

| 策略 | 函数 | 结果 | 适用 |
|------|------|------|------|
| gather | `tensor_model_parallel_gather` | 仅 rank0 拿到完整 logits，其余返回 None | GPU/NPU，只有 rank0 采样 |
| all-gather | `tensor_model_parallel_all_gather` | 每个 rank 都有完整 logits | TPU 等要求严格 SPMD 的设备 |

这也是 §5.2 Reduce Sample 存在的根因：gather 模式下非 rank0 手里没有完整词表，分布式采样必须额外通信。

### 2.6 soft cap 与 scale

在 `forward`（[L75-L82](../../../../vllm/vllm/model_executor/layers/logits_processor.py#L75)）里，gather 完 logits 后依次处理：

```python
# model_executor/layers/logits_processor.py:76-82
if self.soft_cap is not None:                 # Gemma2 等模型使用
    logits = logits / self.soft_cap
    logits = torch.tanh(logits)
    logits = logits * self.soft_cap           # 压到 [-cap, cap]，保持可导
if self.scale != 1.0:
    logits *= self.scale
```

DeepSeek V4 默认 `soft_cap=None`、`scale=1.0`，这两步实际为 no-op。

### 2.7 get_top_tokens：greedy 专用的通信优化

[logits_processor.py:L155](../../../../vllm/vllm/model_executor/layers/logits_processor.py#L155)。greedy 只需要 argmax，不必 gather 全部 logits：

```
每卡本地 argmax → (local_max_val, local_global_idx)   # 先加 vocab_start 转全局 id
   │  all-gather 成对的 (value, index)，通信量 O(batch * 2 * tp)
   ▼
在所有 rank 的最大值里再 argmax → 全局 token id
```

对比全量 gather 的 `O(batch * vocab_size)`，大词表下（V4 约 12.9 万）通信量大幅下降。NPU 侧等价实现在 `AscendSampler.greedy_sample` 的 Reduce Sample 分支（见 §5.2）。

---

## 三、阶段②入口：sample_tokens 与 _sample

### 3.1 sample_tokens 主流程

[NPUModelRunner.sample_tokens](../../worker/model_runner_v1.py#L2231) 承接第二次 RPC，主干分四步：

1. 解包阶段①暂存的 `ExecuteModelState` 并立即清空（[:L2252-L2268](../../worker/model_runner_v1.py#L2252)）；
2. 施加 grammar bitmask —— Ascend 与上游 GPU 版不同：因 NPU 暂不支持其 `torch.compile` 优化，这里要先把 logits 搬到 CPU 改完再搬回（[:L2271-L2277](../../worker/model_runner_v1.py#L2271)）；
3. 调用 `_sample` 得到 `SamplerOutput`（[:L2280](../../worker/model_runner_v1.py#L2280)）；
4. 推测解码提案、bookkeeping、封装 `ModelRunnerOutput`（第七章）。

若开启 accepted-tokens 异步路径，采样一结束就在 global stream 上记录 `sampling_done_event`（[:L2282-L2287](../../worker/model_runner_v1.py#L2282)），供后续状态搬移与下一步调度重叠。

### 3.2 _sample 的分发逻辑

`_sample`（[:L2464-L2495](../../worker/model_runner_v1.py#L2464)）是常规采样与拒绝采样的分叉口，进入分支前有两件准备工作：

- **lmhead_tp 截断**：词表并行 LM Head 场景下 logits 可能带了 padding 行，先切到真实请求数（常规 [:L2469-L2470](../../worker/model_runner_v1.py#L2469)，spec [:L2479-L2480](../../worker/model_runner_v1.py#L2479)）；
- **prepare_sampling**：仅当本批存在 top_k 请求且开启 reduce_sample 时，从 CPU 侧算出本批有效最大 k（剔除等于词表大小的哨兵值），预先下发给采样器（[:L2471-L2473](../../worker/model_runner_v1.py#L2471)），Reduce 分支据此决定每卡本地取多少候选。

随后按 `spec_decode_metadata` 是否为 None 二选一：

| 条件 | 入口 | 展开 |
|------|------|------|
| `spec_decode_metadata is None` | `self.sampler(...)` → `AscendSampler` | 第四、五章 |
| 推测解码（MTP / DSpark / EAGLE…） | `self.rejection_sampler(...)` → `AscendRejectionSampler` | 第六章 |

> 拒绝采样分支的 `draft_probs` 仅在 PP world_size > 1 时才获取（[:L2484-L2488](../../worker/model_runner_v1.py#L2484)），TP-only 场景传 None。

### 3.3 采样参数的载体：SamplingMetadata 与 k/p 六层链路

两个采样器都不直接读 `SamplingParams`，它们的唯一参数源是 `input_batch.sampling_metadata`（温度、top_k/top_p、三种惩罚、generators、bad words 等都在里面）。以最典型的 top_k/top_p 为例，用户请求里的标量 `k/p` 要经过 6 层才到达过滤算子；关键是两级"禁用"语义：**单请求**用哨兵值，**全批次**用 `None`。

#### 3.3.1 第一层：SamplingParams（用户请求）

[sampling_params.py:L240-L246](../../../../vllm/vllm/sampling_params.py#L240)：

```python
# vllm/sampling_params.py:240-246
top_p: float = 1.0     # =1.0 表示禁用 TopP
top_k: int = 0         # 0 或 -1 表示禁用 TopK
min_p: float = 0.0
```

[sampling_params.py:L492](../../../../vllm/vllm/sampling_params.py#L492) 处，greedy（`temperature < _SAMPLING_EPS`）会强制 `top_p=1.0、top_k=0、min_p=0.0`，即贪婪时不做任何核采样过滤。

#### 3.3.2 第二层：写入 InputBatch

请求入批时（[gpu_input_batch.py:L392-L400](../../../../vllm/vllm/v1/worker/gpu_input_batch.py#L392)）：

```python
# v1/worker/gpu_input_batch.py:392-400
self.top_p_cpu[req_index] = sampling_params.top_p
if sampling_params.top_p < 1:
    self.top_p_reqs.add(req_id)           # 记录"需要 TopP"的请求
top_k = sampling_params.top_k
if 0 < top_k < self.vocab_size:
    self.top_k_reqs.add(req_id)
else:
    top_k = self.vocab_size               # 禁用 → 哨兵值=词表大小（等于不过滤）
self.top_k_cpu[req_index] = top_k
```

每类参数在批次对象里有四件套：`top_k_cpu`（锁页 numpy）、`top_k_cpu_tensor`（共享内存，供拷贝）、`top_k`（NPU tensor）、`top_k_reqs`（set）。

属性 `no_top_p / no_top_k`（[L1101-L1107](../../../../vllm/vllm/v1/worker/gpu_input_batch.py#L1101)）：对应集合为空即全批不需要，是后续快速路径的开关。

#### 3.3.3 第三层：构造 SamplingMetadata

`_make_sampling_metadata`（[gpu_input_batch.py:L834](../../../../vllm/vllm/v1/worker/gpu_input_batch.py#L834)）先把 CPU 切片拷到 NPU，再决定传张量还是 None（[L921-L922](../../../../vllm/vllm/v1/worker/gpu_input_batch.py#L921)）：

```python
# v1/worker/gpu_input_batch.py:921-922
top_p=None if self.no_top_p else self.top_p[:num_reqs],   # 全批不需要 → 直接 None
top_k=None if self.no_top_k else self.top_k[:num_reqs],
```

`SamplingMetadata` 字段定义见 [metadata.py:L20-L21](../../../../vllm/vllm/v1/sample/metadata.py#L20)：`top_p` 为 `[batch]` float 张量或 None，`top_k` 为 `[batch]` int 张量或 None。

#### 3.3.4 第四、五层：Sampler.sample → TopKTopPSampler

上游 `Sampler.sample`（[v1/sample/sampler.py:L242](../../../../vllm/vllm/v1/sample/sampler.py#L242)）在做完温度缩放、argmax-invariant 处理器后，于 [L285](../../../../vllm/vllm/v1/sample/sampler.py#L285) 调用：

```python
# v1/sample/sampler.py:285-290
random_sampled, processed_logprobs = self.topk_topp_sampler(
    logits,
    sampling_metadata.generators,
    sampling_metadata.top_k,     # k：Tensor 或 None
    sampling_metadata.top_p,     # p：Tensor 或 None
)
```

注意参数顺序固定为 `(logits, generators, k, p)`。由于 `AscendSampler.__init__` 已把子模块换成 `AscendTopKTopPSampler`，实际进入 NPU 版 `forward_native`（见 §5.4）。

#### 3.3.5 第六层：过滤函数

最终进入模块级别名 `apply_top_k_top_p`（[sampler.py:L268-L272](../../sample/sampler.py#L268)）：

```python
# sample/sampler.py:268-272
apply_top_k_top_p = (
    _apply_top_k_top_p_torch_npu if get_ascend_device_type() in [A2, A3]
    else _apply_top_k_top_p_pytorch
)
```

#### 3.3.6 两级"禁用"语义

| 层级 | 表示 | 含义 | 判断位置 |
|------|------|------|---------|
| 单请求 | `k[i]=vocab_size`、`p[i]=1.0` | 该请求不过滤 | 过滤实现里逐元素处理 |
| 全批次 | `k=None` 或 `p=None` | 整批都不过滤 | 函数入口 `if p is None and k is None: return logits`（[sampler.py:L206-L207](../../sample/sampler.py#L206)） |

这样混合批次（有的请求要 TopK、有的不要）能共用一张张量，而全批都不要时又能 O(1) 直接跳过。

---

## 四、常规分支（上）：Sampler.forward 的 logits 加工

### 4.1 AscendSampler：装配与三处覆写

[AscendSampler](../../sample/sampler.py#L47) 继承上游 `Sampler`，覆写三处：惩罚项（§4.5）、greedy（§5.2），以及构造时把 top-k-top-p 子模块替换成 NPU 版：

```python
# sample/sampler.py:74-85
class AscendSampler(Sampler):
    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler(...)     # 换成 NPU 版

    def prepare_sampling(self, top_k):
        self.topk_topp_sampler.prepare_sampling(top_k)
```

NPUModelRunner 构造时持有的就是它（[model_runner_v1.py:L348](../../worker/model_runner_v1.py#L348)），所以阶段②的 `self.sampler(...)` 直接进入 NPU 覆写链。

### 4.2 进入 forward：fp32 提升与 raw logprobs

上游 `Sampler.forward`（[v1/sample/sampler.py:L72](../../../../vllm/vllm/v1/sample/sampler.py#L72)）开头先处理 logprobs 记账，再统一精度：

- 如果需要返回 logprobs，**在任何加工之前**用原始 logits 算 `raw_logprobs`（[:L84-L93](../../../../vllm/vllm/v1/sample/sampler.py#L84)），这与 v0 采样器用加工后 logits 算 logprobs 不同；
- `logits = logits.to(torch.float32)`（[:L96](../../../../vllm/vllm/v1/sample/sampler.py#L96)），后续惩罚、温度、softmax 全部在 fp32 上进行。

### 4.3 apply_logits_processors 的固定顺序

真正的加工从 `apply_logits_processors`（[:L371-L417](../../../../vllm/vllm/v1/sample/sampler.py#L371)）开始，顺序固定：

1. `allowed_token_ids_mask` 非空时，把屏蔽位置原地填 `-inf`（[:L391-L392](../../../../vllm/vllm/v1/sample/sampler.py#L391)）；
2. 配置了 bad words 时调用 `apply_bad_words`（[:L395-L396](../../../../vllm/vllm/v1/sample/sampler.py#L395)）；
3. 遍历 **non_argmax_invariant** 插件（[:L399-L400](../../../../vllm/vllm/v1/sample/sampler.py#L399)）；
4. 调 `apply_penalties` 施加三种惩罚（[:L403](../../../../vllm/vllm/v1/sample/sampler.py#L403)）。

注意这一步**不区分 greedy / random**：即使整批 greedy 也必须执行，因为 allowed-mask、bad words、惩罚都会改变 argmax。

### 4.4 LogitsProcessor 插件框架

v1 把"对 logits 的额外加工"抽象成可插拔处理器（guided decoding、最小 token 数、语法约束等），在采样前按需执行。每个处理器可声明自己是否"不改变 argmax"：

- **argmax_invariant**：不影响最大值选择（如温度缩放），在 `sample` 内部、温度缩放之后才遍历（[v1/sample/sampler.py:L281](../../../../vllm/vllm/v1/sample/sampler.py#L281)），greedy 时可以跳过；
- **non_argmax_invariant**：会改变 argmax（如惩罚项、bad words、allowed-token mask），在 §4.3 中无条件执行。

拒绝采样器也只遍历 `sampling_metadata.logitsprocs.non_argmax_invariant`（[rejection_sampler.py:L144](../../sample/rejection_sampler.py#L144)）。

### 4.5 apply_penalties：Ascend 覆写

`AscendSampler.apply_penalties`（[sampler.py:L48-L72](../../sample/sampler.py#L48)）有两道快速路径：

```python
# sample/sampler.py:55-72
if not HAS_TRITON:                              # 没装 Triton-Ascend
    return Sampler.apply_penalties(...)         # 回退上游（有性能告警）
if sampling_metadata.no_penalties:             # 本批无任何惩罚
    return logits                               # 快速路径
return apply_all_penalties(logits, prompt_token_ids,
                           presence_penalties, frequency_penalties,
                           repetition_penalties, output_token_ids)
```

### 4.6 惩罚项系统

**三种惩罚的语义**：

| 惩罚 | 规则 | 目的 |
|------|------|------|
| presence | 出现过的 token：`logit -= penalty` | 抑制已出现 token，鼓励多样性 |
| frequency | `logit -= penalty * 出现次数` | 按次数抑制，减少重复 |
| repetition | 出现过的 token：`logit /= penalty` | 除法形式的重复惩罚 |

批量实现 `apply_all_penalties`（[penalties.py:L25-L45](../../sample/penalties.py#L25)）：

```python
# sample/penalties.py:25-45
def apply_all_penalties(logits, prompt_token_ids, presence_penalties,
                        frequency_penalties, repetition_penalties, output_token_ids):
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)  # padding 挪到越界位
    return apply_penalties_triton(logits, prompt_token_ids, output_tokens_t,
                                  presence_penalties, frequency_penalties,
                                  repetition_penalties)
```

- `_convert_to_tensors`（[L13](../../sample/penalties.py#L13)）用 `make_tensor_with_pad` 把不等长的历史 token 列表补成定长 tensor（`pad=vocab_size`，可锁页内存），一次 H2D 搬到 NPU；
- 真正计算在 Triton-Ascend kernel `apply_penalties_triton`（[ops/triton/penalty.py](../../ops/triton/penalty.py)），全 batch 并行，无 Python 循环。

---

## 五、常规分支（下）：sample 的采样决策

### 5.1 sample 的编排顺序

`Sampler.sample`（[v1/sample/sampler.py:L242](../../../../vllm/vllm/v1/sample/sampler.py#L242)）内部顺序与很多人直觉不同 —— **greedy 先算，温度后缩放**：

```python
# v1/sample/sampler.py:256-301（节选）
if sampling_metadata.all_random:
    greedy_sampled = None
else:
    greedy_sampled = self.greedy_sample(logits)          # 先把 argmax 算好
    if sampling_metadata.all_greedy:
        return greedy_sampled, processed_logprobs        # 整批 greedy：直接返回

logits = self.apply_temperature(logits, temperature, all_random)
for processor in sampling_metadata.logitsprocs.argmax_invariant:
    logits = processor.apply(logits)                    # 温度后的不变插件
random_sampled, processed_logprobs = self.topk_topp_sampler(
    logits, generators, top_k, top_p)
sampled = torch.where(temperature < _SAMPLING_EPS,
                      greedy_sampled, random_sampled,
                      out=greedy_sampled)                # 混合批逐请求二选一
```

要点：

- `all_greedy` 时跳过温度 / TopK/TopP / 随机采样，[:L260-L270](../../../../vllm/vllm/v1/sample/sampler.py#L260) 直接返回；
- 混合批（greedy 与随机请求同批）两条路都算，最后用 `torch.where` 按行选择，复用同一张量不另开内存。

### 5.2 greedy_sample 与 Reduce Sample

`AscendSampler.greedy_sample`（[sampler.py:L87-L107](../../sample/sampler.py#L87)）：

```python
# sample/sampler.py:87-107
@staticmethod
def greedy_sample(logits):                      # logits: [B, V_local]
    if get_ascend_config().enable_reduce_sample:
        tp_group = get_tp_group()
        B, V_local = logits.shape
        rank = tp_group.rank_in_group
        local_max_logits, local_max_indices = logits.max(dim=-1)
        local_global_idx = local_max_indices + rank * V_local     # 本地→全局 id
        gathered_logits = tp_group.all_gather(local_max_logits.unsqueeze(-1), dim=-1)
        gathered_global_idx = tp_group.all_gather(local_global_idx.unsqueeze(-1), dim=-1)
        global_max_rank = gathered_logits.argmax(dim=-1)
        return gathered_global_idx.gather(-1, global_max_rank.unsqueeze(-1)).squeeze(-1)
    return logits.argmax(dim=-1).view(-1)       # 默认：logits 已完整 gather
```

> **Reduce Sample 解决什么**：默认 gather 模式只有 rank0 有完整 logits；开了 PP / 某些并行后需要每张卡都能独立得到采样结果。Reduce Sample 让每卡只上交"本地最大值 + 其全局下标"，两趟 all-gather 后再选全局最大，避免搬运整张词表。

### 5.3 温度缩放与混合批合并

`apply_temperature`（[v1/sample/sampler.py:L226-L236](../../../../vllm/vllm/v1/sample/sampler.py#L226)）原地除法，且对混合批做了除零保护：非 `all_random` 时把 `temp < eps` 的位置替换成 1.0（这些行走 greedy 路，温度无意义），再 `logits.div_(temp.unsqueeze(1))`。

最终 `torch.where`（[:L295-L300](../../../../vllm/vllm/v1/sample/sampler.py#L295)）以 `temperature < _SAMPLING_EPS` 为掩码逐请求合并 greedy / random 两路。

### 5.4 AscendTopKTopPSampler.forward_native

`forward_native`（[sampler.py:L122-L159](../../sample/sampler.py#L122)）按开关分两条实现路径：

```python
# sample/sampler.py:122-159（节选）
def forward_native(self, logits, generators, k, p):
    if envs.VLLM_BATCH_INVARIANT:                      # 要求批内确定性 → 回退上游
        return super().forward_native(logits, generators, k, p)

    if get_ascend_config().enable_reduce_sample:       # 分布式：拿候选 + 全局下标
        cand_logits, cand_idx = self.apply_top_k_top_p(logits, k, p, self.top_k)
        probs = cand_logits.softmax(dim=-1, dtype=torch.float32)
        pos = random_sample(probs, generators)         # 采到的是"候选内位置"
        next_token = cand_idx.gather(1, pos.unsqueeze(1)).squeeze(1)  # 映射回全局 id
        return next_token, logits_to_return
    else:                                              # 普通：原地过滤后直接采
        logits = self.apply_top_k_top_p(logits, k, p)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators), logits_to_return
```

`prepare_sampling`（[L116-L120](../../sample/sampler.py#L116)）在 Reduce 模式下预先记录全局 `top_k`（即 §3.2 从 `_sample` 下发的 `max_topk`），供过滤函数决定本地取多少候选。

### 5.5 TopK/TopP 的两套过滤实现

**纯 PyTorch 版 `_apply_top_k_top_p_pytorch`**（[sampler.py:L162](../../sample/sampler.py#L162)），非 Reduce 分支（L205-L234，A5 等机型默认走这里）：

```python
# sample/sampler.py:205-234（节选）
if p is None and k is None:
    return logits                          # 全批快速路径
probs = logits.softmax(dim=-1)
probs_sort, _ = probs.sort(dim=-1, descending=False)   # 概率升序
if k is not None:                          # TopK：找到第 k 大的概率作为阈值
    top_k_cutoff = probs_sort.gather(-1, (size - k).unsqueeze(1))
    top_k_cutoff.masked_fill_((k == V).unsqueeze(1), -inf)  # 哨兵行 no-op
    logits.masked_fill_(probs < top_k_cutoff, -inf)
if p is not None:                          # TopP(nucleus)：累积概率阈值
    cumprob = torch.cumsum(probs_sort, dim=-1)
    top_p_mask = cumprob <= 1 - p.unsqueeze(1)
    top_p_mask[:, -1] = False              # 至少保留 1 个
    top_p_cutoff = probs_sort.gather(-1, top_p_mask.sum(1, keepdim=True))
    logits.masked_fill_(probs < top_p_cutoff, -inf)
return logits
```

算法直觉：升序排序后从概率最小的一端累加，累计概率达到 `1-p` 的那批全部丢弃，剩下的就是概率和 ≥ p 的最小集合；TopK 同理用"第 k 大阈值"裁剪。

**Reduce Sample 分支**（L168-L204）：先 `torch.topk` 取本地候选，把本地下标加 `rank*V_local` 转全局，两趟 all-gather 拼出全局候选，再在全局候选上做上述 sort+mask，返回 `(gathered_vals, gathered_idx)` 而不是原地改 logits。

**torch_npu 版 `_apply_top_k_top_p_torch_npu`**（[sampler.py:L237](../../sample/sampler.py#L237)，仅 A2/A3 选中）：

- **Reduce Sample**：本地 topk + all-gather 出全局候选后，调用 torch_npu 内置算子 `torch_npu.npu_top_k_top_p(gathered_vals, k=k, p=p)`（L259）做联合过滤；
- **非 Reduce**：代码注释明确指出大 k / 混合 k 批次时 `npu_top_k_top_p` 会退化到 5~28ms，而 sort+mask 稳定约 1ms，因此**非 Reduce 直接回退 PyTorch 版**（L262-L265）。

### 5.6 随机采样：random_sample（Gumbel + global stream）

[sampler.py:L20-L44](../../sample/sampler.py#L20)。不用 `torch.multinomial`（会触发 CPU-NPU 同步），改用 Gumbel-max：

```python
# sample/sampler.py:33-44
with npu_stream_switch(global_stream()):          # 随机数放到全局流
    q = torch.empty_like(probs)
    if len(generators) != probs.shape[0]:
        q.exponential_()                           # 常见情形：批量生成
    if generators:                                 # 有自定义种子的请求逐个覆盖
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
torch.npu.current_stream().wait_stream(global_stream())  # 等随机数就绪
q.record_stream(torch.npu.current_stream())
return probs.div_(q).argmax(dim=-1).view(-1)       # Gumbel：argmax(p/q)
```

**数学等价性**：若 `g ~ Gumbel(0,1)`，则 `argmax(log p + g)` 服从按 `p` 的多项分布；而 `g = -log(e)`、`e ~ Exponential(1)`，故等价于 `argmax(p / e)`。全程在 NPU 上完成，无 host 同步。随机数在 `global_stream` 上生成，可与模型计算流并行，末尾用一次 wait 同步即可。

### 5.7 forward 收尾：logprobs 与 SamplerOutput

采样出的 token 还要经过 `Sampler.forward` 的收尾包装（[v1/sample/sampler.py:L109-L149](../../../../vllm/vllm/v1/sample/sampler.py#L109)）：

- `sampled.long()`（[:L109](../../../../vllm/vllm/v1/sample/sampler.py#L109)）统一成 int64 以兼容后续花式索引；
- 需要 logprobs 时，用**原始 logits** 做 `gather_logprobs` / 指定 token 抓取（[:L114-L136](../../../../vllm/vllm/v1/sample/sampler.py#L114)）；
- 采样结果转 int32 并升维成 `[num_reqs, 1]`，连同 logprobs 装入 `SamplerOutput`（[:L139-L148](../../../../vllm/vllm/v1/sample/sampler.py#L139)）。

### 5.8 调用链锚点速查

| 层级 | 文件 : 行 |
|------|-----------|
| 参数定义 | [sampling_params.py:L240](../../../../vllm/vllm/sampling_params.py#L240) |
| 写入批次 | [gpu_input_batch.py:L392-L400](../../../../vllm/vllm/v1/worker/gpu_input_batch.py#L392) |
| None/Tensor 决策 | [gpu_input_batch.py:L921-L922](../../../../vllm/vllm/v1/worker/gpu_input_batch.py#L921) |
| Sampler 调用 | [v1/sample/sampler.py:L285](../../../../vllm/vllm/v1/sample/sampler.py#L285) |
| NPU forward_native | [sampler.py:L122](../../sample/sampler.py#L122) |
| 机型分发别名 | [sampler.py:L268](../../sample/sampler.py#L268) |

---

## 六、推测解码分支：AscendRejectionSampler

[AscendRejectionSampler](../../sample/rejection_sampler.py#L37)。MTP/DSpark 等草稿模型一次给出多个候选 token，目标模型并行验证：与目标分布一致就接受，首个不一致处拒绝并按目标分布重采样。

### 6.1 关键覆写

| 方法 | 行 | 作用 |
|------|----|------|
| `apply_penalties` | [L48](../../sample/rejection_sampler.py#L48) | 用 `repeat_indices` 把每请求的惩罚参数扩展到每个 draft token，再走 Triton 惩罚 |
| `__init__` | [L87](../../sample/rejection_sampler.py#L87) | 透传 spec_config（synthetic 模式）、初始化 `self.top_k` |
| `apply_logits_processors` | [L104](../../sample/rejection_sampler.py#L104) | 合并 spec token 后施加惩罚 / allowed mask / bad words / 最小 token 处理器 |
| `forward` | [L150](../../sample/rejection_sampler.py#L150) | bonus 采样 → target 处理 → 拒绝采样的总入口 |

模块级辅助函数：`greedy_sample`（[L328](../../sample/rejection_sampler.py#L328)，无开关、始终 all-gather 全局 argmax）、`apply_sampling_constraints`（[L347](../../sample/rejection_sampler.py#L347)，把温度/k/p 扩展到 draft token 并做分布式 topk-allgather-topp）、`rejection_sample`（[L416](../../sample/rejection_sampler.py#L416)）。

### 6.2 forward 三步

```
logits [num_tokens + batch, V]
  ├─ 1. bonus token：取每请求最后一个位置单独采样（补一位草稿）
  ├─ 2. target logits：apply_logits_processors（惩罚/bad words）
  │      → apply_sampling_constraints（温度 + 分布式 topk/topp）
  └─ 3. rejection_sample：逐 token 比对 draft，决定接受/拒绝/重采样
```

### 6.3 block verify 与 entropy verify

`rejection_sample` 内部（[L485-L500](../../sample/rejection_sampler.py#L485)）：

```python
# sample/rejection_sampler.py:485-500
using_block_verify = max_spec_len >= 3 and bool(
    get_ascend_config().rejection_sampler_config.enable_block_verify)
using_entropy_verify = bool(
    get_ascend_config().rejection_sampler_config.enable_entropy_verify)
posterior_threshold = rejection_sampler_config.posterior_threshold
posterior_alpha = rejection_sampler_config.posterior_alpha
```

- **Block Verify**：把一段 draft token 当整体验证（需 `max_spec_len>=3` 且开关打开），借助块内双向注意力的上下文提高整段接受率；有 Triton kernel（`rejection_random_sample_block_verify_kernel`）与 PyTorch 双实现。
- **Entropy Verify**：目标分布熵较高（更"随机"）时放宽接受阈值，阈值取 `min(exp(-entropy * alpha), posterior_threshold)`（[L1234-L1243](../../sample/rejection_sampler.py#L1234)），在保证分布接近的前提下少拒绝。

最终按"全 greedy / 全 random""Reduce 是否开启""block/entropy 是否开启"选择 Triton 或 PyTorch 的具体 kernel（L546 起的多路分支）。

---

## 七、采样收尾：bookkeeping、状态回写与输出封装

`_sample` 返回后，`sample_tokens` 还要完成三件收尾工作，产物才上交 EngineCore：

1. **草稿提案**（推测解码时）：用刚采出的 token 跑 EAGLE / MTP proposer 生成下一步 draft，并拷到 CPU（[:L2291-L2306](../../worker/model_runner_v1.py#L2291)）；
2. **bookkeeping**：Ascend 覆写的 `_bookkeeping_sync`（[:L2499](../../worker/model_runner_v1.py#L2499)）把采样结果与 logprobs 做 D2H 同步、拷贝 req_id 映射，并对被丢弃请求回退 generator offset（[:L2517-L2520](../../worker/model_runner_v1.py#L2517)），产出 `valid_sampled_token_ids` 与 `logprobs_lists`；
3. **状态回写**：`need_accepted_tokens` 路径下，先在 global stream 上等 `sampling_done_event`，再调 `_update_states_after_model_execute`（[:L2405-L2412](../../worker/model_runner_v1.py#L2405)）。该方法（[gpu_model_runner.py:L1545](../../../../vllm/vllm/v1/worker/gpu_model_runner.py#L1545)）只在**混合模型（linear attention）+ 推测解码**时真正做事：统计每请求接受数并搬移循环状态，普通模型为 no-op。

最终封装分同步 / 异步两种：

- 同步路径构造 `ModelRunnerOutput`（[:L2381-L2393](../../worker/model_runner_v1.py#L2381)），携带 `req_ids`、`sampled_token_ids=valid_sampled_token_ids`、`spec_token_ids`、logprobs 等，直接返回（[:L2414-L2424](../../worker/model_runner_v1.py#L2414)）；
- 异步路径包一层 `AsyncGPUModelRunnerOutput`（[:L2448-L2456](../../worker/model_runner_v1.py#L2448)），D2H 在独立 copy stream 上滞后进行，相关 tensor 必须**私有 clone** 以防缓冲被下一步清零造成撕裂读；随后通过 `set_async_sampled_token_ids`（[:L2457](../../worker/model_runner_v1.py#L2457)）把 CPU 侧结果回喂 input_batch。

输出经执行器回到 EngineCore 后，由 `scheduler.update_from_output`（[core.py:L600](../../../../vllm/vllm/v1/engine/core.py#L600)）把 token 追加进各请求序列、判定停止条件，进入下一个调度步 —— 至此完成"hidden_states → sampled_token_ids"的闭环。

---

## 八、设备适配与性能总结

| 主题 | A2 / A3 | A5 等其他机型 |
|------|---------|--------------|
| TopK/TopP | Reduce 下 `torch_npu.npu_top_k_top_p`；否则 PyTorch | 始终 PyTorch sort+cumsum |
| 惩罚项 | Triton-Ascend 可用则用，否则回退上游 | 同左 |
| 随机采样 | Gumbel（global stream 指数分布），全 NPU | 同左 |
| greedy 分布式 | `enable_reduce_sample` 时只 all-gather (max,idx) | 同左 |

| 优化 | 收益 |
|------|------|
| 两阶段 RPC（execute_model / sample_tokens） | grammar bitmask 等调度工作与前向并行，logits 不出 Worker |
| Gumbel-max 替代 multinomial | 消除 CPU-NPU 同步，随机数与前向在不同流并行 |
| Reduce Sample（greedy / top-k-top-p） | 通信量从全词表 `O(B·V)` 降到 `O(B·2·tp)` 或 `O(B·K·tp)` |
| `get_top_tokens` | greedy 路径不 gather 全量 logits |
| 全批 None 快速路径 | 无 top-k/p 需求时 O(1) 跳过 |
| Triton 惩罚项 | 历史 token 一次补齐搬运、批量并行，避免 Python 循环 |
| BATCH_INVARIANT 回退 | 需要批内确定性时回到 vLLM 原生实现保证语义 |

---

## 九、一句话总结

DeepSeek V4 的后处理 = **阶段①词表并行 LM Head 投影并 gather 出 logits（暂存于 Worker）** → **阶段②惩罚 / 温度 / TopK-TopP 加工** → **Gumbel 随机采样（或 greedy）并封装上交**；Ascend 侧通过 [AscendSampler](../../sample/sampler.py#L47) 把惩罚换成 Triton kernel、把 TopK/TopP 按机型分发到 `torch_npu.npu_top_k_top_p`（A2/A3）或 PyTorch sort+cumsum（A5 等），并在 `enable_reduce_sample` 下用"只上交局部最大值 / TopK 候选 + all-gather"规避全词表通信；推测解码则由 [AscendRejectionSampler](../../sample/rejection_sampler.py#L37) 配合 block/entropy verify 完成草稿 token 的接受与重采样。
