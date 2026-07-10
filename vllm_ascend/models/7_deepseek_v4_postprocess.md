# DeepSeek V4 后处理与采样详解

> **本文档详细讲解 DeepSeek V4 的后处理流水线，包括 LogitsProcessor、ParallelLMHead、惩罚项、TopK/TopP 采样、随机采样优化、分布式 Reduce Sample、以及推测解码的拒绝采样。**

---

## 一、后处理总览

### 1.1 后处理流水线

从模型输出隐藏状态到最终采样 token 的完整流程：

```
hidden_states (num_tokens, hidden_size)
        │
        ▼
┌─────────────────────────┐
│  ParallelLMHead         │  ← 词表并行线性层，TP 分片
│  (VocabParallelEmbedding)│     每个 rank 算 vocab_size/tp_size 部分
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  TP Gather / All-Gather │  ← 聚合各 rank 的 logits
│  _gather_logits         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Soft Cap (Gemma2等)    │  ← tanh 软上限裁剪
│  logits = tanh(logits/cap) * cap
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Scale 缩放              │  ← logits *= scale
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  惩罚项 (Penalties)      │  ← presence / frequency / repetition
│  apply_all_penalties    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Temperature 缩放        │  ← logits /= temperature
│                         │     greedy: temperature=0 → argmax
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  TopK / TopP 过滤       │  ← 保留概率最高的 token
│  apply_top_k_top_p      │     过滤掉的置 -inf
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  随机采样 / Greedy       │  ← random_sample / argmax
│  random_sample (Gumbel) │     Gumbel-max trick 避免 CPU-GPU 同步
└───────────┬─────────────┘
            │
            ▼
    sampled_token_ids
```

### 1.2 关键模块分层

| 层级 | 模块 | 文件 | 功能 |
|------|------|------|------|
| **模型层** | `LogitsProcessor` | `vllm/model_executor/layers/logits_processor.py` | LM Head + logits 缩放 + TP gather |
| **接口层** | `LogitsProcessor` (ABC) | `vllm/v1/sample/logits_processor/interface.py` | v1 插件式 logits processor 接口 |
| **采样层** | `AscendSampler` | `vllm_ascend/sample/sampler.py` | NPU 优化的采样器 |
| **惩罚层** | `apply_all_penalties` | `vllm_ascend/sample/penalties.py` | Triton 加速的惩罚项 |
| **拒绝采样** | `AscendRejectionSampler` | `vllm_ascend/sample/rejection_sampler.py` | 推测解码验证 |
| **算子层** | `npu_apply_top_k_top_p` | `csrc/` | AscendC TopK/TopP 算子 |

---

## 二、ParallelLMHead 与 Logits 计算

### 2.1 VocabParallelEmbedding

LM Head 本质是一个词表并行的线性层，权重与 embedding 层共享（或不共享）。

**TP 分片方式**：词表维度按 TP rank 切分
- 每个 rank 持有 `vocab_size / tp_size` 个词的权重
- 前向计算时各 rank 独立计算本地 logits
- 最后通过 gather/all-gather 聚合完整 logits

### 2.2 _get_logits 流程

**文件位置**: `vllm/model_executor/layers/logits_processor.py:L89`

```python
def _get_logits(self, hidden_states, lm_head, embedding_bias):
    # 1. 量化 LM Head 前向
    logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
    # 2. TP 聚合 logits
    logits = self._gather_logits(logits)
    # 3. 移除词表 padding
    if logits is not None:
        logits = logits[..., : self.org_vocab_size]
    return logits
```

### 2.3 _gather_logits 策略

**文件位置**: `vllm/model_executor/layers/logits_processor.py:L75`

| 模式 | 函数 | 返回值 | 适用场景 |
|------|------|--------|---------|
| **all-gather** | `tensor_model_parallel_all_gather` | 所有 rank 都有完整 logits | TPU 等设备，每个 rank 都需要完整 logits |
| **gather** | `tensor_model_parallel_gather` | 仅 rank 0 有完整 logits，其他 rank 返回 None | GPU/NPU，只有 rank 0 需要采样 |

> 由 `current_platform.use_all_gather()` 决定使用哪种策略。

---

## 三、LogitsProcessor 类详解

**文件位置**: `vllm/model_executor/layers/logits_processor.py:L19`

### 3.1 核心成员

| 成员 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `scale` | float | 1.0 | logits 缩放因子 |
| `vocab_size` | int | — | 词表大小（含 padding） |
| `org_vocab_size` | int | vocab_size | 原始词表大小（不含 padding） |
| `logits_as_input` | bool | False | 输入是否已经是 logits（否则为 hidden_states） |
| `soft_cap` | float \| None | None | Soft cap 值（Gemma2 用） |
| `use_all_gather` | bool | — | 是否用 all-gather 替代 gather |

### 3.2 Soft Cap 机制

用于 Gemma 2 等模型的 logits 软上限：

```python
if self.soft_cap is not None:
    logits = logits / self.soft_cap
    logits = torch.tanh(logits)
    logits = logits * self.soft_cap
```

**效果**: 将 logits 限制在 `[-soft_cap, soft_cap]` 范围内，同时保持可导性。

### 3.3 get_top_tokens 优化

**文件位置**: `vllm/model_executor/layers/logits_processor.py:L106`

Vocab-parallel argmax 优化，不做全量 all-gather：

```
每个 TP rank 本地计算 argmax
    │ 只传 (max_value, max_index) 对
    ▼
all-gather (value, index) 对 [batch, 2*tp_size]
    │
    ▼
在所有 rank 中找全局最大值
    │
    ▼
返回全局 argmax token id
```

**通信量对比**:
- 全量 gather: `O(batch * vocab_size)`
- 优化 gather: `O(batch * 2 * tp_size)`

> 适用于 greedy decoding 只需 argmax 的场景，大幅减少通信量。

---

## 四、v1 LogitsProcessor 插件框架

**文件位置**: `vllm/v1/sample/logits_processor/interface.py:L60`

### 4.1 设计理念

v1 版本将 logits 处理器抽象为可插拔的插件，每个处理器维护自己的状态：

```
SamplingMetadata
    └── logitsprocs
        ├── argmax_invariant: list[LogitsProcessor]    ← 不影响 argmax 的处理器
        └── non_argmax_invariant: list[LogitsProcessor] ← 影响 argmax 的处理器
```

### 4.2 核心方法

| 方法 | 说明 |
|------|------|
| `validate_params(sampling_params)` | 验证采样参数 |
| `__init__(vllm_config, device, is_pin_memory)` | 初始化处理器 |
| `apply(logits) -> Tensor` | 应用处理器到 logits（可原地修改） |
| `is_argmax_invariant() -> bool` | 是否不影响 argmax 结果 |
| `update_state(batch_update)` | 更新状态（新 token 到来时调用） |

### 4.3 argmax invariant 分类

- **argmax_invariant**: 不改变最大 logits 的 token 顺序（如 temperature 缩放）
  - greedy decoding 时可以跳过
- **non_argmax_invariant**: 会改变 argmax 结果（如惩罚项、bad words）
  - 任何情况下都必须执行

---

## 五、AscendSampler — NPU 优化采样器

**文件位置**: `vllm_ascend/sample/sampler.py:L45`

### 5.1 继承关系

```
Sampler (vLLM 基类)
    └── AscendSampler (NPU 优化)
        ├── apply_penalties (Triton 加速)
        ├── greedy_sample (Reduce Sample 支持)
        ├── do_async_exponential (异步指数采样)
        └── topk_topp_sampler: AscendTopKTopPSampler
```

### 5.2 apply_penalties

**文件位置**: `vllm_ascend/sample/sampler.py:L47`

- **Triton 可用时**: 使用 `apply_penalties_triton`（NPU 加速）
- **Triton 不可用时**: 回退到 vLLM 默认实现（性能较差）
- **无惩罚项时**: 直接返回 logits（快速路径）

### 5.3 greedy_sample

**文件位置**: `vllm_ascend/sample/sampler.py:L104`

两种模式：

| 模式 | 实现 | 说明 |
|------|------|------|
| **Reduce Sample** | TP all-gather 找全局 argmax | `enable_reduce_sample=True`，分布式采样 |
| **本地 argmax** | `logits.argmax(dim=-1)` | 默认模式，假设 logits 已经完整 gather |

**Reduce Sample 流程**:
```
1. 每个 rank 本地找 max: local_max, local_idx
2. 转全局索引: local_global_idx = local_idx + rank * V_local
3. all-gather 所有 rank 的 (max_val, global_idx)
4. 找全局最大值所在 rank
5. 返回全局 argmax
```

---

## 六、惩罚项系统 (Penalties)

### 6.1 三种惩罚类型

| 惩罚类型 | 公式 | 作用 |
|---------|------|------|
| **Presence Penalty** | `logits[token] -= penalty` (如果 token 出现过) | 惩罚出现过的 token，鼓励多样性 |
| **Frequency Penalty** | `logits[token] -= penalty * count[token]` | 按出现次数惩罚，减少重复 |
| **Repetition Penalty** | `logits[token] /= penalty` (如果 token 出现过) | 除法形式的重复惩罚 |

### 6.2 apply_all_penalties

**文件位置**: `vllm_ascend/sample/penalties.py:L25`

输入参数：
- `logits`: `[batch, vocab_size]`
- `prompt_token_ids`: prompt 中的 token
- `presence_penalties`: `[batch]`
- `frequency_penalties`: `[batch]`
- `repetition_penalties`: `[batch]`
- `output_token_ids`: 已输出 token 列表

**实现步骤**:
1. 将 `output_token_ids`（list of lists）转换为 padded tensor
2. padding 值用 `vocab_size`（超出范围，不影响）
3. 调用 Triton kernel `apply_penalties_triton` 批量处理

### 6.3 Triton 加速优势

- **避免 Python 循环**: 所有惩罚计算在 Triton kernel 中完成
- **批量并行**: 整个 batch 并行处理
- **减少 H2D**: 提前将 output_token_ids 转为 tensor，减少同步

---

## 七、TopK/TopP 采样

**文件位置**: `vllm_ascend/sample/sampler.py:L268`

### 7.1 两种实现

| 实现 | 适用设备 | 调用方式 |
|------|---------|---------|
| **AscendC 算子** | A2, A3 | `torch.ops._C_ascend.npu_apply_top_k_top_p` |
| **PyTorch 实现** | 其他设备（A5 等） | `_apply_top_k_top_p_pytorch` |

### 7.2 算法原理

```
logits → softmax → probs
    │
    ├── TopK: 保留概率最大的 K 个，其余置 -inf
    │
    └── TopP (nucleus sampling):
            1. 按概率升序排列
            2. 计算累积概率 cumprob
            3. 找到 cumprob <= 1-p 的截止点
            4. 低于截止点的置 -inf
            5. 确保至少保留 1 个 token
```

### 7.3 Reduce Sample 模式

当 `enable_reduce_sample=True` 时，TopK/TopP 支持分布式执行：

```
每个 rank 本地 TopK (取 k_for_topk 个)
    │
    ├── local_vals, local_idx
    ├── local_global_idx = local_idx + rank * V_local
    │
    ▼
all-gather → gathered_vals, gathered_idx (全局 TopK 候选)
    │
    ▼
在全局候选上做 TopP 过滤
    │
    ▼
返回 (cand_logits, cand_idx)
```

**优点**: 避免全量 logits 的 all-gather，只传 TopK 候选，减少通信。

### 7.4 AscendC 算子 `npu_apply_top_k_top_p` 详解

**源码位置**: `csrc/moe/apply_top_k_top_p_custom/`

#### 7.4.1 调用栈分层

```
Python (sampler.py)
    │  _apply_top_k_top_p_ascendc()
    ▼
Torch Binding (apply_top_k_top_p_custom_torch_adpt.h)
    │  npu_apply_top_k_top_p()
    │  EXEC_NPU_CMD(aclnnApplyTopKTopPCustom, ...)
    ▼
aclnn 接口层 (aclnn_apply_top_k_top_p_custom.cpp)
    │  aclnnApplyTopKTopPCustomGetWorkspaceSize()
    │  ├── 1. l0op::Sort(logits)        ← 升序排序
    │  ├── 2. l0op::ApplyTopKTopPCustom ← 核心算子
    │  └── 3. ViewCopy 输出
    ▼
Tiling 层 (apply_top_k_top_p_custom_tiling.cpp)
    │  TilingForApplyTopKTopPCustom()
    │  ├── 形状检查
    │  ├── TilingKey 选择（3种模式）
    │  ├── 核数分配
    │  └── UB 分块计算
    ▼
Kernel 层 (AscendC)
    ├── apply_top_k_top_p_custom.h   ← TopK + TopP 联合 / 仅 TopK
    └── apply_top_p_custom.h         ← 仅 TopP 模式
```

#### 7.4.2 输入输出规格

| 参数 | 类型 | Shape | 说明 |
|------|------|-------|------|
| `logits` | float16/bfloat16/float32 | `[batch_size, vocab_size]` | 输入 logits |
| `k` | int32 (可选) | `[batch_size]` | TopK 的 K 值 |
| `p` | float16/bfloat16/float32 (可选) | `[batch_size]` | TopP 的 P 值（累积概率阈值） |
| `out` | 同 logits | `[batch_size, vocab_size]` | 输出 logits，过滤掉的位置为 -inf |

**约束**: `k` 和 `p` 不能同时为 None。

#### 7.4.3 算子整体流程

```
输入 logits [B, V]
    │
    ▼
┌─────────────────────────────┐
│  Sort (升序, dim=-1)        │  ← aclnn Sort 算子
│  sorted_value, sorted_index │     从小到大排列
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  ApplyTopKTopPCustom Kernel │
│                             │
│  1. 初始化输出全为 -inf      │
│  2. 从大到小（从右往左）遍历 │
│  3. TopK 过滤: 保留最大 K 个 │
│  4. TopP 过滤: 累积概率 <= p │
│  5. Scatter 回填到原始索引   │
└─────────────┬───────────────┘
              │
              ▼
输出 out [B, V]  (保留的位置为原值，其余为 -inf)
```

**关键设计**: 输入先排序，TopK/TopP 过滤后通过 scatter 按原始索引写回。由于排序后概率从左到右递增，可以从右往左提前终止遍历。

#### 7.4.4 三种 Tiling 模式

根据 `k` 和 `p` 是否存在，算子选择不同的 kernel 路径：

| TilingKey | 模式 | Kernel 类 | 说明 |
|-----------|------|-----------|------|
| **0** | TopK + TopP | `ApplyTopKTopPCustom::Process()` | 同时做 TopK 和 TopP 过滤 |
| **1** | 仅 TopK | `ApplyTopKTopPCustom::ProcessTopK()` | 只做 TopK 过滤，无需 softmax/cumsum |
| **2** | 仅 TopP | `ApplyTopPCustom::ProcessTopP()` | 只做 TopP 过滤，使用 Kogge-Stone 前缀和 |

**模式选择逻辑** (tiling.cpp):
```cpp
onlyTopK_ = (k 存在 && p 不存在) ? 1 : 0;
onlyTopP_ = (p 存在 && k 不存在) ? 2 : 0;
tilingKey_ = onlyTopK_ + onlyTopP_;  // 0, 1, 2
```

#### 7.4.5 TopK + TopP 联合模式详解 (TilingKey=0)

**文件**: `op_kernel/apply_top_k_top_p_custom.h`

**核心流程** (`Process()` 方法):

```
for each batch:
    1. 读取最大的 dataNumInit_ 个值（最右侧，概率最高）
       位置: [vocabSize - dataNumInit_, vocabSize)
       dataNumInit_ = min(vocabSize, K_MAX=1024)
    
    2. 计算 kthValue = 第 K 大的值
       从 GM 直接取 sorted_value[vocabSize - k]
    
    3. 判断 dataNumInit_ 中最小值是否 < kthValue:
       ├─ 是 → 所有 K 个都在 dataNumInit_ 内 → ProcessKLtKMax()
       └─ 否 → 需要向左继续找 → ProcessRemain()
```

**ProcessKLtKMax (快速路径)**: 当 TopK 的 K 个 token 都在最右 1024 个内时：
1. 先把整个 out 初始化为 -inf
2. 对 dataNumInit_ 个值做 softmax
3. 做 CumSum（累积和）
4. **从右往左**遍历，遇到 cumprob <= (1-p) 时停止
5. 把保留的值通过 scatter 按原始索引写回 GM

**ProcessRemain (慢速路径)**: 需要向左遍历更多数据：
1. `GetFirstKLoop()`: 从左往右找，跳过所有值都 < kthValue 的块
2. `ScatterFromFirstKLoop()`: 从第一个有有效值的块开始，逐步做 softmax + cumsum + scatter
3. 最后处理最右侧的 dataNumInit_ 块

**TopK 剪枝优化**: 在 TopP 计算之前，先用 TopK 阈值过滤——值 < kthValue 的直接置 -inf，不参与 softmax 计算。

#### 7.4.6 仅 TopK 模式 (TilingKey=1)

**文件**: `op_kernel/apply_top_k_top_p_custom.h` 的 `ProcessTopK()`

相比联合模式的简化：
- 无需 softmax
- 无需 cumsum
- 无需 p 参数
- 直接用 kthValue 比较，从右往左 scatter 直到值 < kthValue

**核心逻辑** (`ScatterCumtomImplTopK`):
```cpp
// 反向遍历，遇到 < kthValue 提前退出
for (loopProb = loopProbNum - 1; loopProb >= 0; loopProb--) {
    if (calLocalFp32[loopProb] < kthValue) break;
    // scatter 到原始索引位置
}
```

#### 7.4.7 仅 TopP 模式 (TilingKey=2)

**文件**: `op_kernel/apply_top_p_custom.h`

特点：
- 使用 **Kogge-Stone 算法**做并行前缀和（cumsum）
- 分两阶段：先算完全局 softmax 和 cumsum，再做 scatter
- 需要 workspace 存储中间 softmax 结果

**流程**:
```
阶段一: 每个 batch 独立计算
    ├─ GetSoftmaxSum: 计算 softmax 的归一化分母（reduce sum）
    ├─ GetSoftMaxRes: 计算 softmax 结果，写入 workspace
    └─ CumsumKoggleStone: Kogge-Stone 前缀和，结果写回 workspace

阶段二: Scatter 阶段（多任务并行）
    └─ ScatterSingleTask: 按 vCnt 维度切分任务
        ├─ 读取 cumsum 结果和 sorted indices
        ├─ 从右往左判断 cumsum > (1-p)
        └─ scatter 写回原始位置
```

**Kogge-Stone 前缀和算法**:
```
迭代 log2(V) 次:
  第 i 次迭代: offset = 2^i
  对于每个位置 j:
    result[j + offset] += result[j]
```
时间复杂度 O(log V)，适合大词表场景。

#### 7.4.8 Tiling 与并行策略

**核间并行**:
- batch 维度按核数切分
- `batchPerCore_ = batchSize_ / coreNum_`
- 前 `tailBatch_` 个核多处理 1 个 batch

**UB 分块策略**:
- `ubFactorElement_`: 每次加载到 UB 的元素数 = min(vocabSize, 1024)
- `dataNumInit_`: 最右侧初始加载块大小 = min(vocabSize, 1024)
- 大词表时分块从右往左（或从左往右）迭代处理

**Workspace 大小**:
- 仅 TopP 模式需要额外 workspace: `16MB + batch_size * vocab_size * 4` (float32 softmax 缓存)
- TopK 模式只需 16MB 系统 workspace

#### 7.4.9 性能优化要点

| 优化技术 | 效果 |
|---------|------|
| **排序后从右往左遍历** | 利用排序后概率递增的特性，找到截止点提前终止 |
| **TopK 先剪枝** | TopK+TopP 模式先用 kthValue 过滤，减少 softmax 计算量 |
| **快速路径 ProcessKLtKMax** | K <= 1024 时只处理最右 1024 个 token，跳过大部分词表 |
| **Kogge-Stone 前缀和** | 仅 TopP 大词表时 O(log V) 复杂度，优于 O(V) |
| **Scatter 直接写回** | 按原始索引 scatter，无需再排序回原始顺序 |
| **数据类型支持** | float16/bfloat16/float32，计算时转 float32 保证精度 |
| **双缓冲流水线** | 使用 TPipe 双缓冲，MTE2/V/MTE3 流水重叠 |

---

## 八、随机采样优化 (random_sample)

**文件位置**: `vllm_ascend/sample/sampler.py:L19`

### 8.1 Gumbel-Max Trick

不使用 `torch.multinomial`（会导致 CPU-NPU 同步），而是用 Gumbel-max trick：

```python
def random_sample(probs, generators):
    q = torch.empty_like(probs)
    q.exponential_()           # Gumbel 分布: -log(-log(uniform))
    return probs.div_(q).argmax(dim=-1)
```

**数学原理**:
- 若 `X_i ~ Gumbel(0, 1)`，则 `argmax_i (log(p_i) + X_i)` 服从 Multinomial(p)
- 等价于 `argmax(p_i / q_i)` 其中 `q_i ~ Exponential(1)`
- 全部在 GPU/NPU 上完成，无 CPU 同步

### 8.2 自定义种子处理

- **无自定义种子**: 直接 `q.exponential_()` 批量生成
- **有自定义种子**: 逐请求用各自的 generator 生成
  - `for i, generator in generators.items(): q[i].exponential_(generator=generator)`

### 8.3 流并行优化

随机数生成在 `global_stream()` 中执行，与模型计算流重叠：

```python
with npu_stream_switch(global_stream()):
    q.exponential_()
torch.npu.current_stream().wait_stream(global_stream())
```

---

## 九、Async Exponential — 异步指数采样

**文件位置**: `vllm_ascend/sample/sampler.py:L89`

### 9.1 设计理念

将指数分布随机数的生成与模型前向计算重叠，隐藏采样延迟：

```
模型执行 (计算流)          指数生成 (global stream)
    │                              │
    │    ┌─────────────────────┐   │
    │    │  do_async_exponential│   │  ← 在模型执行前启动
    │    └─────────────────────┘   │
    │                              │
   ...                            ...
    │                              │
    │   模型执行完毕，等待 q 就绪   │
    │                              │
    └─── 采样时使用预生成的 q ──────┘
```

### 9.2 核心方法

| 方法 | 功能 |
|------|------|
| `do_async_exponential(b_s, head_dim, generators)` | 在 global stream 中预生成指数随机数 |
| `set_q_event(q, event)` | 将生成的 q 和 event 传给 TopKTopPSampler |
| `prepare_sampling(top_k)` | 准备 top_k 参数 |

### 9.3 同步机制

```python
# 采样时同步等待异步生成完成
self.async_event.synchronize()
# 直接使用预生成的 q
return probs.div_(self.q).argmax(dim=-1), logits_to_return
```

---

## 十、AscendTopKTopPSampler

**文件位置**: `vllm_ascend/sample/sampler.py:L127`

### 10.1 扩展功能

继承自 `TopKTopPSampler`，增加：

- **自定义 top_k**: 通过 `prepare_sampling(top_k)` 预设 top_k 参数
- **异步指数采样**: 复用 `do_async_exponential` 预生成的随机数
- **Reduce Sample 支持**: 分布式 TopK/TopP 采样

### 10.2 forward_native 流程

```
logits
  │
  ├── apply_top_k_top_p(logits, k, p, top_k)
  │     │
  │     ├─ Reduce Sample 模式: 返回 (cand_logits, cand_idx)
  │     └─ 普通模式: 返回过滤后的 logits
  │
  ├── logprobs 计算（可选）
  │
  ├── softmax → probs
  │
  └── 随机采样
        ├─ 异步指数模式: probs.div_(self.q).argmax()
        └─ 普通模式: random_sample(probs, generators)
```

---

## 十一、AscendRejectionSampler — 拒绝采样

**文件位置**: `vllm_ascend/sample/rejection_sampler.py:L36`

### 11.1 作用

推测解码 (Speculative Decoding) 的核心验证模块：
- Draft 模型生成多个候选 token
- 主模型验证每个候选 token
- 接受一致的，拒绝不一致的
- 在拒绝位置从主模型分布中重采样

### 11.2 核心优化

| 优化 | 说明 |
|------|------|
| **Reduce Sample** | TP 环境下的分布式采样支持 |
| **Block Verify** | 块验证，提高接受率（MagicMTP） |
| **Entropy Verify** | 熵验证，提高接受率 |
| **Triton Kernels** | 全量 Triton 加速的拒绝采样 kernel |
| **Greedy 快速路径** | Greedy decoding 的专用优化 |

### 11.3 完整 Forward 流程

```
输入: draft_probs, logits (含 bonus + target), sampling_metadata
  │
  ├── 1. Bonus Token 采样
  │     bonus_logits = logits[bonus_indices]
  │     bonus_token_ids = sampler(bonus_logits)
  │
  ├── 2. Target Logits 处理
  │     target_logits = logits[target_indices]
  │     target_logits = apply_logits_processors(...)
  │     target_logits = apply_sampling_constraints(...)
  │
  └── 3. 拒绝采样
        output_token_ids = rejection_sample(
            draft_token_ids,
            target_logits,
            bonus_token_ids,
            ...
        )
```

### 11.4 rejection_sample 路径选择

```
rejection_sample
  │
  ├── 全 Greedy → Greedy 快速路径
  │
  ├── Reduce Sample (target_indices 存在)
  │     ├── Block Verify + Triton
  │     ├── Block Verify + PyTorch
  │     ├── Triton Kernel
  │     └── PyTorch
  │
  └── 普通模式 (fallback)
        ├── Block Verify + Triton
        ├── Block Verify + PyTorch
        ├── Triton Kernel
        └── PyTorch
```

### 11.5 熵验证 (Entropy Verify)

**配置**:
- `posterior_threshold`: 后验概率阈值
- `posterior_alpha`: 平滑系数

**原理**: 当 draft 分布与 target 分布的差异在一定范围内时，提高接受率，减少不必要的拒绝。

### 11.6 块验证 (Block Verify)

**条件**: `max_spec_len >= 3` 且 `enable_block_verify=True`

**原理**: 将多个 draft token 作为一个块一起验证，利用上下文信息提高整体接受率。

---

## 十二、Greedy Sample — 贪婪采样

### 12.1 普通模式

```python
logits.argmax(dim=-1).view(-1)
```

### 12.2 Reduce Sample 模式

**文件位置**: `vllm_ascend/sample/rejection_sampler.py:L249`

```python
def greedy_sample(logits):
    # 1. 每个 rank 本地 argmax
    local_max_logits, local_max_indices = logits.max(dim=-1)
    local_global_idx = local_max_indices + rank * V_local

    # 2. all-gather 所有 (val, idx) 对
    gathered_logits = tp_group.all_gather(local_max_logits.unsqueeze(-1), dim=-1)
    gathered_global_idx = tp_group.all_gather(local_global_idx.unsqueeze(-1), dim=-1)

    # 3. 找全局最大值
    global_max_rank = gathered_logits.argmax(dim=-1)
    target_argmax = gathered_global_idx.gather(dim=-1, index=global_max_rank.unsqueeze(-1)).squeeze(-1)
    return target_argmax
```

---

## 十三、A5 设备特殊适配

| 特性 | 说明 |
|------|------|
| **TopK/TopP 实现** | 使用 PyTorch 实现（`_apply_top_k_top_p_pytorch`） |
| **A2/A3 设备** | 使用 AscendC 算子（`npu_apply_top_k_top_p`） |
| **Reduce Sample** | 可选启用，TP 环境下减少通信 |
| **Async Exponential** | 可选启用，计算与采样重叠 |
| **Triton 惩罚项** | Triton 可用时启用，否则回退 |

---

## 十四、性能优化总结

| 优化技术 | 解决的问题 | 效果 |
|---------|-----------|------|
| **Gumbel-Max Trick** | `torch.multinomial` 导致 CPU-GPU 同步 | 全部在 NPU 上执行，无同步 |
| **Async Exponential** | 随机数生成延迟 | 与模型计算重叠，隐藏延迟 |
| **Reduce Sample** | TP 环境下全量 logits gather 通信量大 | 只 gather TopK 候选，通信量 O(batch * K * tp) |
| **get_top_tokens** | Greedy 时不需要全量 logits | 通信量从 O(batch * V) 降到 O(batch * 2 * tp) |
| **Triton Penalties** | Python 循环惩罚项慢 | Triton kernel 并行加速 |
| **AscendC TopK/TopP** | PyTorch 实现慢 | 专用 NPU 算子加速（A2/A3） |
| **Batch Invariant 支持** | 批处理一致性 | 保证相同输入相同输出 |
