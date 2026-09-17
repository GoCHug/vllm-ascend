# DeepSeek V4 后处理与采样详解

> **本文档讲解从模型最后一层隐藏状态到采样出 token 的完整后处理流水线：词表并行 LM Head、logits 缩放/soft cap、惩罚项、温度、TopK/TopP、Gumbel 随机采样、TP Reduce Sample，以及推测解码的拒绝采样。** DeepSeek V4 本身没有自定义采样逻辑，它复用 vLLM v1 通用采样栈 + vllm-ascend 的 NPU 加速实现，因此本文同时覆盖上游 vLLM 与 Ascend 两侧。

> **代码版本冻结声明**：本文基于 **vllm-ascend v0.26.0rc** 与其配套的上游 vLLM 源码整理。Ascend 侧：采样器 [sample/sampler.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/sampler.py)（`random_sample` L20、`AscendSampler` L47、`AscendTopKTopPSampler` L110）、惩罚项 [sample/penalties.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/penalties.py)（`apply_all_penalties` L25）、拒绝采样 [sample/rejection_sampler.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/rejection_sampler.py)（`AscendRejectionSampler` L37、`rejection_sample` L416）。上游侧：[logits_processor.py](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/layers/logits_processor.py)、[gpu_input_batch.py](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu_input_batch.py)、[v1/sample/sampler.py](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/sample/sampler.py)。文中 `Lxxx` 均指对应文件行号。

> **统一模型配置**（W4A8 版，本文相关行）：`vocab_size=129280`（含 padding 的实际词表张量可能更大），`hidden_size=7168`，logits 最后一步是 `(num_tokens, 7168) → (num_tokens, vocab_size)`。下文 TP 举例取 8，每卡本地词表 `V_local = vocab_size / 8`。

> **v0.26 两处重要变更（旧资料常见过时点）**：
> 1. TopK/TopP **不再调用** 旧的自定义 AscendC 算子 `npu_apply_top_k_top_p`（对应 csrc 目录已移除）。现在 A2/A3 走 `torch_npu.npu_top_k_top_p`，其他机型（含 A5）走纯 PyTorch 的 sort+cumsum 实现（见第七章）。
> 2. 旧版 `do_async_exponential` / `set_q_event` 那套"异步指数采样"方法在本版本**已不存在**；随机数改由 `random_sample` 内部在 global stream 上即时生成（见第八章）。

---

## 一、后处理总览

### 1.1 完整流水线

```
hidden_states (num_tokens, 7168)
        │
        ▼
┌──────────────────────────────┐
│ LogitsProcessor（LM Head）    │  词表并行：每卡只算 V_local 列
│  _apply_head → _gather_logits │  rank0 gather / 或 all-gather
│  去掉 vocab padding           │
└───────────────┬──────────────┘
                ▼  logits (num_tokens, vocab_size)
┌──────────────────────────────┐
│ soft cap（可选，tanh 裁剪）   │  logits = tanh(logits/cap)*cap
│ scale 缩放（可选）            │  logits *= scale
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ 惩罚项（presence/frequency/   │  apply_all_penalties（Triton）
│ repetition）+ bad words 等    │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ 温度缩放                       │  logits /= temperature
│ temperature≈0 → greedy argmax │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ TopK / TopP 过滤              │  落选位置 -inf
│ A2/A3: npu_top_k_top_p        │
│ 其他 : PyTorch sort+cumsum    │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ softmax → random_sample       │  Gumbel：指数分布 + div + argmax
└───────────────┬──────────────┘
                ▼
        sampled_token_ids (num_tokens,)
```

### 1.2 模块分层

| 层级 | 模块 | 文件 : 行 | 职责 |
|------|------|-----------|------|
| 模型层 | `LogitsProcessor` | [logits_processor.py:L24](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/layers/logits_processor.py#L24) | LM Head + soft cap + scale + TP gather |
| 采样编排 | `Sampler` | [v1/sample/sampler.py](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/sample/sampler.py)（`sample` L243） | 串起惩罚/温度/top-k-top-p/随机采样 |
| NPU 采样器 | `AscendSampler` | [sampler.py:L47](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/sampler.py#L47) | 覆写惩罚、greedy、装配 NPU top-k-top-p |
| NPU top-k-top-p | `AscendTopKTopPSampler` | [sampler.py:L110](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/sampler.py#L110) | `forward_native` 走 NPU 实现 |
| 过滤函数 | `apply_top_k_top_p` | [sampler.py:L268](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/sampler.py#L268) | 按机型二选一（torch_npu / pytorch） |
| 惩罚项 | `apply_all_penalties` | [penalties.py:L25](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/penalties.py#L25) | Triton-Ascend 批量惩罚 |
| 拒绝采样 | `AscendRejectionSampler` | [rejection_sampler.py:L37](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/rejection_sampler.py#L37) | 推测解码 draft 验证 |

---

## 二、Logits 计算：ParallelLMHead + LogitsProcessor

### 2.1 词表并行（VocabParallel）

LM Head 本质是以词表维做列并行的线性层：

- 每个 TP rank 只持有 `V_local = vocab_size / tp_size` 列权重，独立算出本地分片 logits `(num_tokens, V_local)`；
- 再通过 gather / all-gather 拼成完整 `(num_tokens, vocab_size)`；
- lm_head 权重通常与输入端 embedding 共享（tie-weight），也可独立。

### 2.2 `_get_logits` 与 `_apply_head`

[logits_processor.py:L138-L153](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/layers/logits_processor.py#L138)：

```python
def _get_logits(self, hidden_states, lm_head, embedding_bias):
    logits = self._apply_head(lm_head, hidden_states, embedding_bias)  # L145 本地分片
    logits = self._gather_logits(logits)                               # L148 TP 聚合
    if logits is not None:
        logits = logits[..., : self.org_vocab_size]                    # L152 去掉 padding
    return logits
```

`_apply_head`（L99-L136）封装量化 LM Head 的前向；当配置了 `head_dtype` 且在 CUDA/ROCm 上可直接累加 fp32，NPU 走常规的 `lm_head.quant_method.apply(...)` 路径。

### 2.3 `_gather_logits` 两种策略

[logits_processor.py:L85-L97](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/layers/logits_processor.py#L85)，由 `self.use_all_gather`（`current_platform.use_all_gather()`）决定：

| 策略 | 函数 | 结果 | 适用 |
|------|------|------|------|
| gather | `tensor_model_parallel_gather` | 仅 rank0 拿到完整 logits，其余返回 None | GPU/NPU，只有 rank0 采样 |
| all-gather | `tensor_model_parallel_all_gather` | 每个 rank 都有完整 logits | TPU 等要求严格 SPMD 的设备 |

### 2.4 soft cap 与 scale

在 `forward`（[L75-L82](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/layers/logits_processor.py#L75)）里，gather 完 logits 后依次处理：

```python
if self.soft_cap is not None:                 # Gemma2 等模型使用
    logits = logits / self.soft_cap
    logits = torch.tanh(logits)
    logits = logits * self.soft_cap           # 压到 [-cap, cap]，保持可导
if self.scale != 1.0:
    logits *= self.scale
```

DeepSeek V4 默认 `soft_cap=None`、`scale=1.0`，这两步实际为 no-op。

### 2.5 `get_top_tokens`：greedy 专用的通信优化

[logits_processor.py:L155](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/layers/logits_processor.py#L155)。greedy 只需要 argmax，不必 gather 全部 logits：

```
每卡本地 argmax → (local_max_val, local_global_idx)   # 先加 vocab_start 转全局 id
   │  all-gather 成对的 (value, index)，通信量 O(batch * 2 * tp)
   ▼
在所有 rank 的最大值里再 argmax → 全局 token id
```

对比全量 gather 的 `O(batch * vocab_size)`，大词表下（V4 约 12.9 万）通信量大幅下降。NPU 侧等价实现在 `AscendSampler.greedy_sample` 的 Reduce Sample 分支（第四章）。

---

## 三、v1 LogitsProcessor 插件框架

v1 把"对 logits 的额外加工"抽象成可插拔处理器（guided decoding、最小 token 数、语法约束等），在采样前按需执行。每个处理器可声明自己是否"不改变 argmax"：

- **argmax_invariant**：不影响最大值选择（如温度缩放），greedy（temperature=0）时可以跳过；
- **non_argmax_invariant**：会改变 argmax（如惩罚项、bad words、allowed-token mask），任何情况都必须执行。

拒绝采样器正是只遍历 `sampling_metadata.logitsprocs.non_argmax_invariant`（[rejection_sampler.py:L144](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/rejection_sampler.py#L144)）。

---

## 四、AscendSampler — NPU 采样器

[sampler.py:L47](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/sampler.py#L47)，继承上游 `Sampler`，覆写三处：惩罚项、greedy、构造时替换 top-k-top-p 子模块。

```python
class AscendSampler(Sampler):
    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):     # L74
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler(...)     # 换成 NPU 版
    def prepare_sampling(self, top_k):                          # L84
        self.topk_topp_sampler.prepare_sampling(top_k)
```

### 4.1 `apply_penalties`（L48-L72）

```python
if not HAS_TRITON:                              # 没装 Triton-Ascend
    return Sampler.apply_penalties(...)         # 回退上游（有性能告警）
if sampling_metadata.no_penalties:             # 本批无任何惩罚
    return logits                               # 快速路径
return apply_all_penalties(logits, prompt_token_ids,
                           presence_penalties, frequency_penalties,
                           repetition_penalties, output_token_ids)
```

### 4.2 `greedy_sample` 与 Reduce Sample（L87-L107）

```python
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

> **Reduce Sample 解决什么**：默认 gather 模式只有 rank0 有完整 logits；开了 PP/某些并行后需要每张卡都能独立得到采样结果。Reduce Sample 让每卡只上交"本地最大值 + 其全局下标"，all-gather 后再选全局最大，避免搬运整张词表。

---

## 五、惩罚项系统

### 5.1 三种惩罚

| 惩罚 | 规则 | 目的 |
|------|------|------|
| presence | 出现过的 token：`logit -= penalty` | 抑制已出现 token，鼓励多样性 |
| frequency | `logit -= penalty * 出现次数` | 按次数抑制，减少重复 |
| repetition | 出现过的 token：`logit /= penalty` | 除法形式的重复惩罚 |

### 5.2 `apply_all_penalties`（[penalties.py:L25-L45](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/penalties.py#L25)）

```python
def apply_all_penalties(logits, prompt_token_ids, presence_penalties,
                        frequency_penalties, repetition_penalties, output_token_ids):
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)  # padding 挪到越界位
    return apply_penalties_triton(logits, prompt_token_ids, output_tokens_t,
                                  presence_penalties, frequency_penalties,
                                  repetition_penalties)
```

- `_convert_to_tensors`（L13）用 `make_tensor_with_pad` 把不等长的历史 token 列表补成定长 tensor（`pad=vocab_size`，可锁页内存），一次 H2D 搬到 NPU；
- 真正计算在 Triton-Ascend kernel `apply_penalties_triton`（`vllm_ascend/ops/triton/penalty.py`），全 batch 并行，无 Python 循环。

---

## 六、TopK / TopP 参数传参全链路

用户请求里的标量 `k/p` 要经过 6 层才到达过滤算子。关键是两级"禁用"语义：**单请求**用哨兵值，**全批次**用 `None`。

### 6.1 第一层：SamplingParams（用户请求）

[sampling_params.py:L240-L246](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/sampling_params.py#L240)：

```python
top_p: float = 1.0     # =1.0 表示禁用 TopP
top_k: int = 0         # 0 或 -1 表示禁用 TopK
min_p: float = 0.0
```

[sampling_params.py:L492](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/sampling_params.py#L492) 处，greedy（`temperature < _SAMPLING_EPS`）会强制 `top_p=1.0、top_k=0、min_p=0.0`，即贪婪时不做任何核采样过滤。

### 6.2 第二层：写入 GPUInputBatch

请求入批时（[gpu_input_batch.py:L392-L400](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu_input_batch.py#L392)）：

```python
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

属性 `no_top_p / no_top_k`（[L1101-L1107](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu_input_batch.py#L1101)）：对应集合为空即全批不需要，是后续快速路径的开关。

### 6.3 第三层：构造 SamplingMetadata

`_make_sampling_metadata`（[gpu_input_batch.py:L834](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu_input_batch.py#L834)）先把 CPU 切片拷到 NPU，再决定传张量还是 None（[L921-L922](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu_input_batch.py#L921)）：

```python
top_p=None if self.no_top_p else self.top_p[:num_reqs],   # 全批不需要 → 直接 None
top_k=None if self.no_top_k else self.top_k[:num_reqs],
```

`SamplingMetadata` 字段定义见 [metadata.py:L20-L21](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/sample/metadata.py#L20)：`top_p` 为 `[batch]` float 张量或 None，`top_k` 为 `[batch]` int 张量或 None。

### 6.4 第四、五层：Sampler.sample → TopKTopPSampler

上游 `Sampler.sample`（[v1/sample/sampler.py:L243](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/sample/sampler.py#L243)）在做完温度缩放、argmax-invariant 处理器后，于 [L286](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/sample/sampler.py#L286) 调用：

```python
random_sampled, processed_logprobs = self.topk_topp_sampler(
    logits,
    sampling_metadata.generators,
    sampling_metadata.top_k,     # k：Tensor 或 None
    sampling_metadata.top_p,     # p：Tensor 或 None
)
```

注意参数顺序固定为 `(logits, generators, k, p)`。由于 `AscendSampler.__init__` 已把子模块换成 `AscendTopKTopPSampler`，实际进入 NPU 版 `forward_native`（第九章）。

### 6.5 第六层：过滤函数

最终进入模块级别名 `apply_top_k_top_p`（[sampler.py:L268-L272](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/sampler.py#L268)）：

```python
apply_top_k_top_p = (
    _apply_top_k_top_p_torch_npu if get_ascend_device_type() in [A2, A3]
    else _apply_top_k_top_p_pytorch
)
```

### 6.6 两级"禁用"语义

| 层级 | 表示 | 含义 | 判断位置 |
|------|------|------|---------|
| 单请求 | `k[i]=vocab_size`、`p[i]=1.0` | 该请求不过滤 | 过滤实现里逐元素处理 |
| 全批次 | `k=None` 或 `p=None` | 整批都不过滤 | 函数入口 `if p is None and k is None: return logits`（[L206-L207](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/sampler.py#L206)） |

这样混合批次（有的请求要 TopK、有的不要）能共用一张张量，而全批都不要时又能 O(1) 直接跳过。

---

## 七、TopK / TopP 的两套 NPU 实现

### 7.1 纯 PyTorch 版 `_apply_top_k_top_p_pytorch`（[L162](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/sampler.py#L162)）

**非 Reduce 分支**（L205-L234，A5 等机型默认走这里）：

```python
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

### 7.2 torch_npu 版 `_apply_top_k_top_p_torch_npu`（[L237](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/sampler.py#L237)，仅 A2/A3 选中）

- **Reduce Sample**：本地 topk + all-gather 出全局候选后，调用 torch_npu 内置算子 `torch_npu.npu_top_k_top_p(gathered_vals, k=k, p=p)`（L259）做联合过滤；
- **非 Reduce**：代码注释明确指出大 k / 混合 k 批次时 `npu_top_k_top_p` 会退化到 5~28ms，而 sort+mask 稳定约 1ms，因此**非 Reduce 直接回退 PyTorch 版**（L262-L265）。

### 7.3 调用链锚点速查

| 层级 | 文件 : 行 |
|------|-----------|
| 参数定义 | [sampling_params.py:L240](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/sampling_params.py#L240) |
| 写入批次 | [gpu_input_batch.py:L392-L400](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu_input_batch.py#L392) |
| None/Tensor 决策 | [gpu_input_batch.py:L921-L922](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu_input_batch.py#L921) |
| Sampler 调用 | [v1/sample/sampler.py:L286](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/sample/sampler.py#L286) |
| NPU forward_native | [sampler.py:L122](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/sampler.py#L122) |
| 机型分发别名 | [sampler.py:L268](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/sampler.py#L268) |

---

## 八、随机采样：random_sample（Gumbel + global stream）

[sampler.py:L20-L44](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/sampler.py#L20)。不用 `torch.multinomial`（会触发 CPU-NPU 同步），改用 Gumbel-max：

```python
def random_sample(probs, generators):
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

---

## 九、AscendTopKTopPSampler.forward_native

[sampler.py:L122-L159](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/sampler.py#L122)：

```python
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

`prepare_sampling`（L116-L120）在 Reduce 模式下预先记录一个全局 `top_k`，供过滤函数决定本地取多少候选。

---

## 十、AscendRejectionSampler — 推测解码拒绝采样

[rejection_sampler.py:L37](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/rejection_sampler.py#L37)。MTP/DSpark 等草稿模型一次给出多个候选 token，目标模型并行验证：与目标分布一致就接受，首个不一致处拒绝并按目标分布重采样。

### 10.1 关键覆写

| 方法 | 行 | 作用 |
|------|----|------|
| `apply_penalties` | [L48](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/rejection_sampler.py#L48) | 用 `repeat_indices` 把每请求的惩罚参数扩展到每个 draft token，再走 Triton 惩罚 |
| `__init__` | [L87](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/rejection_sampler.py#L87) | 透传 spec_config（synthetic 模式）、初始化 `self.top_k` |
| `apply_logits_processors` | [L104](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/rejection_sampler.py#L104) | 合并 spec token 后施加惩罚 / allowed mask / bad words / 最小 token 处理器 |
| `forward` | [L150](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/rejection_sampler.py#L150) | bonus 采样 → target 处理 → 拒绝采样的总入口 |

模块级辅助函数：`greedy_sample`（[L328](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/rejection_sampler.py#L328)，无开关、始终 all-gather 全局 argmax）、`apply_sampling_constraints`（[L347](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/rejection_sampler.py#L347)，把温度/k/p 扩展到 draft token 并做分布式 topk-allgather-topp）、`rejection_sample`（[L416](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/rejection_sampler.py#L416)）。

### 10.2 forward 三步

```
logits [num_tokens + batch, V]
  ├─ 1. bonus token：取每请求最后一个位置单独采样（补一位草稿）
  ├─ 2. target logits：apply_logits_processors（惩罚/bad words）
  │      → apply_sampling_constraints（温度 + 分布式 topk/topp）
  └─ 3. rejection_sample：逐 token 比对 draft，决定接受/拒绝/重采样
```

### 10.3 block verify 与 entropy verify

`rejection_sample` 内部（[L485-L500](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/rejection_sampler.py#L485)）：

```python
using_block_verify = max_spec_len >= 3 and bool(
    get_ascend_config().rejection_sampler_config.enable_block_verify)
using_entropy_verify = bool(
    get_ascend_config().rejection_sampler_config.enable_entropy_verify)
posterior_threshold = rejection_sampler_config.posterior_threshold
posterior_alpha = rejection_sampler_config.posterior_alpha
```

- **Block Verify**：把一段 draft token 当整体验证（需 `max_spec_len>=3` 且开关打开），借助块内双向注意力的上下文提高整段接受率；有 Triton kernel（`rejection_random_sample_block_verify_kernel`）与 PyTorch 双实现。
- **Entropy Verify**：目标分布熵较高（更"随机"）时放宽接受阈值，阈值取 `min(exp(-entropy * alpha), posterior_threshold)`（[L1234-L1243](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/rejection_sampler.py#L1234)），在保证分布接近的前提下少拒绝。

最终按"全 greedy / 全 random""Reduce 是否开启""block/entropy 是否开启"选择 Triton 或 PyTorch 的具体 kernel（L546 起的多路分支）。

---

## 十一、设备适配与性能总结

| 主题 | A2 / A3 | A5 等其他机型 |
|------|---------|--------------|
| TopK/TopP | Reduce 下 `torch_npu.npu_top_k_top_p`；否则 PyTorch | 始终 PyTorch sort+cumsum |
| 惩罚项 | Triton-Ascend 可用则用，否则回退上游 | 同左 |
| 随机采样 | Gumbel（global stream 指数分布），全 NPU | 同左 |
| greedy 分布式 | `enable_reduce_sample` 时只 all-gather (max,idx) | 同左 |

| 优化 | 收益 |
|------|------|
| Gumbel-max 替代 multinomial | 消除 CPU-NPU 同步，随机数与前向在不同流并行 |
| Reduce Sample（greedy / top-k-top-p） | 通信量从全词表 `O(B·V)` 降到 `O(B·2·tp)` 或 `O(B·K·tp)` |
| `get_top_tokens` | greedy 路径不 gather 全量 logits |
| 全批 None 快速路径 | 无 top-k/p 需求时 O(1) 跳过 |
| Triton 惩罚项 | 历史 token 一次补齐搬运、批量并行，避免 Python 循环 |
| BATCH_INVARIANT 回退 | 需要批内确定性时回到 vLLM 原生实现保证语义 |

---

## 十二、一句话总结

DeepSeek V4 的后处理 = **词表并行 LM Head 投影并 gather 出 logits** → **惩罚/温度/TopK-TopP 加工** → **Gumbel 随机采样（或 greedy）**；Ascend 侧通过 [AscendSampler](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/sampler.py#L47) 把惩罚换成 Triton kernel、把 TopK/TopP 按机型分发到 `torch_npu.npu_top_k_top_p`（A2/A3）或 PyTorch sort+cumsum（A5 等），并在 `enable_reduce_sample` 下用"只上交局部最大值/TopK 候选 + all-gather"规避全词表通信；推测解码则由 [AscendRejectionSampler](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/sample/rejection_sampler.py#L37) 配合 block/entropy verify 完成草稿 token 的接受与重采样。
