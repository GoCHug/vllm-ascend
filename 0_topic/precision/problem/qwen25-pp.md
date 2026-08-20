# PP=1 vs PP=2 精度差异分析报告

> 分析对象：`true/`（PP=1）与 `false/`（PP=2）两次 vLLM-Ascend 推理 dump 的 step1
> 用户最初看到的关键证据（`true/step1/rank0/dump.json:1167-1168`）：
> ```
>  "Mean": 0.00010585784912109375,
>  "Norm": 2.921875,
> ```

## 一、结论速览

step1 的 embed_tokens 输出在 PP=1 与 PP=2 下出现 Mean/Norm 的微小漂移，**根因并不是 bf16 数值精度漂移**，而是 **PP=2 的调度路径把 step0 解码出的第一个 token (315) 通过 `Tensor.scatter_` 融合进了 step1 chunk2 prefill 的 token buffer，改写了进入 embedding 的输入 token 序列**。后续整个 layer 0–27 + sampling 的差异都是这次输入 token 替换的传播结果，并非真正的计算精度问题。

> **代码级根因**（详见第六节）：PP=2 + async scheduling 下，stage0 的 `_pp_receive_prev_sampled_token_ids_to_input_batch`（`gpu_model_runner.py:4405`）**无条件**把 `prev_sampled_token_ids` 设成来自 last rank 的 `recv`；而 `_prepare_input_ids` 的回填逻辑（`1640`）只看"请求是否在上步也存在"（`prev_index >= 0`），**不区分当前步是 decode 还是 chunked prefill**。于是 chunk1→chunk2 的同一请求被误判为 common decode request，`315` 被 scatter 覆盖了 chunk2 的最后一个 prompt token。修复靶点见第八节。

## 二、两份运行配置

对比 `true.sh` 与 `false.sh`，**唯一的实质差异是 `pipeline-parallel-size`**：

| 参数 | TRUE（基线） | FALSE |
|---|---|---|
| `served-model-name` | qwen25 | qwen2.5 |
| `pipeline-parallel-size` | **1** | **2** |
| `port` | 8119 | 8115 |
| `dump_path` | ./true | ./false |
| 其余 | TP=1, seed=0, dtype=bf16, enable_chunked_prefill=True, max_num_batched_tokens=25, enforce_eager=True, repetition_penalty=1.05, temperature=0.7, top_k=20, top_p=0.8 | 同上 |

文件结构差异（PP=2 把模型按 layer 切分到两个 rank）：

```
true/  step0..step10     每步只有 rank0（709 个模块左右）
false/ step0..step12     每步有 rank0 (stage0: layers 0-13, 362 模块) + rank1 (stage1: layers 14-27, 371 模块)
```

step 数差异（11 vs 13）暗示后续采样与生成长度已经发散。

## 三、step0 完全一致的发现

### 3.1 stage0 共有模块逐项统计全等

对 `true/step0/rank0` 与 `false/step0/rank0` 做模块级逐字段对比（dtype / shape / Max / Min / Mean / Norm），共有 362 个 stage0 模块**全部完全一致**：
- 从 `Torch.index_select.0` 起，到 `Module.model.layers.13.Qwen2DecoderLayer.forward.0` 结束
- 包括 embed_tokens 输出、layers 0-13 的所有 qkv_proj / o_proj / mlp / RMSNorm 等

→ 说明 step0（prompt chunk1, 25 tokens, positions 0–24）在 stage0 上完全一致。

### 3.2 stage1（false rank1）layer 14-27 + 后处理也全等

对 `false/step0/rank1` 与 `true/step0/rank0` 在 layers 14-27 + 最终 RMSNorm + apply_all_penalties_kernel + argmax 上做按 `(layer L, rest)` 逻辑键的逐项对比，**154 个逻辑模块全部一致**。

关键证据：`Module.model.layers.14.input_layernorm.AscendRMSNorm.forward.0` 的 input（即 stage0 跨卡传到 stage1 的隐藏态）**所有字段完全相同**：
- shape=[25, 3584]
- Max=11.4375, Min=-52.25, Mean=-0.005035400390625, Norm=142.0

→ **PP 跨卡传输本身没引入精度损失**——这是后续定位根因的关键前提。

### 3.3 argmax 输出相同

两份 step0 在 `Tensor.argmax.1.forward` 处都给出 token **315**：
```
in  (penalized logits): float32 [1, 152064] Max=13.511905670166016 Min=-17.75
out (next token id):    int64   [1] Max=315 Min=315
```

→ step0 通过 argmax 选出的下一 token 是 315，两端一致。

## 四、step1 首个差异点：embed_tokens 输出

### 4.1 用户选中的字段

`dump.json` 显示 step1 的 `Module.model.embed_tokens.AscendVocabParallelEmbedding.forward.0` 输出：

| 字段 | TRUE (PP=1) | FALSE (PP=2, stage0) |
|---|---|---|
| dtype | torch.bfloat16 | torch.bfloat16 |
| shape | [12, 3584] | [12, 3584] |
| Max | 0.484375 | 0.484375 |
| Min | -0.345703125 | -0.345703125 |
| **Mean** | **0.00010585784912109375** | **0.0001659393310546875** |
| **Norm** | **2.921875** | **2.90625** |

Max/Min 完全一致，只有 Mean/Norm 出现 ~6e-5 / 0.0156 的微小差异。

### 4.2 排除权重与输入指纹不一致的可能

`Functional.embedding.0.forward` 的权重矩阵三次都查到完全一致：
```
bf16 [152064, 3584]
Max=1.0078125  Min=-0.345703125
Mean=7.033348083496094e-06  (17 位有效数字完全相同 → byte-identical 级)
Norm=318.0
```

embed_tokens 的 input_id（int32 [12]）在 dump 中只暴露 Max/Min：
- TRUE：Max=151645, Min=30
- FALSE：Max=151645, Min=30

→ **权重完全一致 + 输入指纹相同，但 embedding 输出不一致**——可疑点恰好就是 embedding lookup 这步本身，或者 input_ids 的中间内容有差异。

### 4.3 step1 整个 forward 都已发散

按 `(layer L, rest)` 逻辑键把 stage0 ∪ stage1 合并后与 TRUE step1 做对比：

| 指标 | 数值 |
|---|---|
| 共有逻辑模块 | 311 |
| 完全一致 | 0 |
| 出现差异 | 311 |
| **首个差异点** | `('emb', 'asc')`（embed_tokens） |
| 最后一个"看似一致"的模块 | 没有，从 embed_tokens 起整个 forward 都 drift |

层 0 的 `Qwen2DecoderLayer.forward.0` INPUT 已能直接看到 embed_tokens 漂移传播进来：
- TRUE：in[1] Mean=0.00010585784912109375 Norm=2.921875
- FALSE：in[1] Mean=0.0001659393310546875 Norm=2.90625

layer 14.stage1 的 INPUT（PP 跨卡传输后的张量）也已经偏离：
- TRUE：Mean=-0.002197265625 Norm=73.5
- FALSE：Mean=-0.0026702880859375 Norm=74.0

→ 这与 step0 stage0→stage1 完全保持精度的情形形成对比，进一步说明 step1 的差异根源在 stage0 内部、更确切在 embed_tokens 之前。

## 五、关键证据：多出来的 `Tensor.scatter_.0.forward`

### 5.1 step1 预处理模块清单对比

逐位置列出 step1 的前 12 个预处理 op：

```
   TRUE step1 r0                              FALSE step1 r0
 0 Torch.index_select.0                       0 Torch.index_select.0          ==
 1 Torch.add.0                                1 Torch.add.0                   ==
 2 Tensor.fill_.0                             2 Tensor.fill_.0                ==
 3 Tensor.fill_.1                             3 Tensor.fill_.1                ==
 4 Tensor.fill_.2                             4 Tensor.scatter_.0  <-- 插入！
 5 Tensor.__add__.0                           5 Tensor.fill_.2
 6 Tensor.__setitem__.0                       6 Tensor.__add__.0
 7 Tensor.__add__.1                           7 Tensor.__setitem__.0
 8 Tensor.__setitem__.1                       8 Tensor.__add__.1
 9 Tensor.fill_.3                             9 Tensor.__setitem__.1
10 Triton._compute_slot_mapping_kernel.0     10 Tensor.fill_.3
11 Tensor.__sub__.0                          11 Triton._compute_slot_mapping_kernel.0
```

跨所有 dump 文件的 `Tensor.scatter_.*` 出现计数：

| 文件 | scatter_. count |
|---|---|
| step0/true/rank0 | 0 |
| step0/false/rank0 | 0 |
| step0/false/rank1 | 0 |
| step1/true/rank0 | 0 |
| **step1/false/rank0** | **1** |
| step1/false/rank1 | 0 |

只有 PP=2 的 step1 stage0 多出这一次 scatter，PP=1 的 step1 没有任何 scatter。

### 5.2 scatter op 的关键内容

`false/step1/rank0/dump.json` 的 `Tensor.scatter_.0.forward`：

```json
{
  "input_args": [{
    "dtype": "torch.int32", "shape": [75],
    "Max": 151645, "Min": 0  // 整张 token buffer
  }],
  "input_kwargs": {
    "dim":   {"type": "int", "value": 0},
    "index": {"dtype": "torch.int64", "shape": [1], "Max": 11, "Min": 11},  // 写入位置 11（第 12 个槽位）
    "src":   {"dtype": "torch.int32", "shape": [1], "Max": 315, "Min": 315} // src = token 315！
  },
  "output": [{
    "dtype": "torch.int32", "shape": [75], "Max": 151645, "Min": 0  // scatter_ in-place
  }]
}
```

**关键解读**：
- src 中那个 1-element int32 tensor 的值正是 **315**（与 step0 argmax 选出的 token 完全相同）。
- 它被写入了 75 长度的 token buffer 的 index=11 处（第 12 个 token 位置，对应 rotary 位置 36）。
- 也就是说 **PP=2 在 step1 chunk2 预处理时把 step0 解码出的 token 315 融合到了 chunk2 batch 的最后位置**。

### 5.3 stack.json 中的调用栈确认

`false/step1/rank0/stack.json` 第 "4" 项的堆栈：

```
Tensor.scatter_.0.forward
 └─ /vllm/v1/worker/gpu_model_runner.py:1701  in _prepare_input_ids
       self.input_ids.gpu.scatter_(
 └─ /vllm-ascend/vllm_ascend/worker/model_runner_v1.py:876  in _prepare_inputs
       self._prepare_input_ids(scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens)
 └─ model_runner_v1.py:1795  in execute_model
 └─ torch/utils/_contextlib.py:124  decorate_context
 └─ vllm_ascend/worker/worker.py:435  execute_model
 └─ ... multiprocessing spawn 调用链
```

→ 该 scatter_ 来自 **vLLM 的标准 `_prepare_input_ids`** 函数，由 `model_runner_v1._prepare_inputs` 在 step1 进入 `execute_model` 时调用。它的作用就是把新来的 token（chunk 上一步采样出的）scatter 到 `self.input_ids.gpu` 的对应槽位——这是 vLLM V1 chunked prefill 与 decode 融合处理时的标准动作。

第 "25" 项进一步证实 embed_tokens 的调用链：

```
Functional.embedding.0.forward
 └─ vllm/model_executor/layers/vocab_parallel_embedding.py:78  in embedding
        return F.embedding(input_, layer.weight)
 └─ vllm_ascend/ops/vocab_parallel_embedding.py:200  in _forward_origin
        output_parallel = self.quant_method.embedding(self, masked_input.long())
 └─ vllm_ascend/ops/vocab_parallel_embedding.py:167  in forward
 └─ qwen2.py:420  in embed_input_ids          return self.embed_tokens(input_ids)
 └─ qwen2.py:433  in forward
 └─ qwen2.py:583 / model_runner_v1.py:2545 等
```

→ `embed_tokens(input_ids)` 接收的 `input_ids` 就是 `_prepare_input_ids` 处理过的 buffer。scatter 修改后的 token 顺序会原样进入 `F.embedding`。

## 六、代码级根因：scatter 为何只在 PP=2 时触发

第五节定位到 scatter 调用点在 `gpu_model_runner.py:1701`（`_prepare_input_ids`），但只是"现象"。本节深入代码解释**为什么 PP=2 会走进这条路而 PP=1 不会**，并给出精确的修复靶点。

### 6.1 scatter 是什么：async scheduling 的"上一步采样 token GPU 端回填"

`_prepare_input_ids`（`gpu_model_runner.py:1602`）一开头就是分叉：

```python
if self.input_batch.prev_sampled_token_ids is None:      # 1619 → normal 路径
    self.input_ids.copy_to_gpu(...)                       #   直接上传 CPU 端完整 input_ids
    return
# else: 1627 "Async scheduling case" → 走 sample_flattened_indices / scatter 路径
```

选中行 `1701` 的 `scatter_` 就在 `else` 分支里。它的设计意图是：在 **decode 步骤之间**，上一步采样的 token 要作为本步的输入；async scheduling 为避免 CPU 同步，把回填放 GPU 上用 `scatter_` 做——根据"哪些请求在上一步也存在"把 `prev_sampled_token_ids` 散射到 `input_ids` 的对应行。

因此 scatter 出现的唯一前提是：**走到本步时 `self.input_batch.prev_sampled_token_ids is not None`。**

### 6.2 PP=1 vs PP=2 的 `prev_sampled_token_ids` 生命周期分叉

**PP=1**（`sample_tokens` 收尾，`gpu_model_runner.py:4202`）：
```python
self.input_batch.prev_sampled_token_ids = None   # 无条件清空
```
→ 下一步 `_prepare_input_ids` 走 normal 路径（`1621` copy_to_gpu），**永远没有 scatter**。

**PP=2 + async scheduling**（`sample_tokens` 开头，`gpu_model_runner.py:4143-4148`）：
```python
if self.execute_model_state is None:
    ...
    if self.use_async_scheduling and get_pp_group().world_size > 1:
        self._pp_receive_prev_sampled_token_ids_to_input_batch()
```
stage0（非 last rank）每个 step 开头都会调 `_pp_receive_prev_sampled_token_ids_to_input_batch()`，该方法（`gpu_model_runner.py:4395-4405`）：

```python
recv = torch.empty((num_reqs, 1), dtype=torch.int32, device=self.device)  # 4401
if not self._is_all_reqs_chunked_prefill():                                 # 4403
    torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)
self.input_batch.prev_sampled_token_ids = recv                              # 4405 ← 无条件赋值！
```

注意 `4405` 行的赋值是**无条件**的：即使当前是 chunked prefill（`_is_all_reqs_chunked_prefill()` 为 True，broadcast 被跳过，`recv` 是 `torch.empty` 未初始化显存），它仍把 `prev_sampled_token_ids` 设成 `recv`（非 `None`）。

而 PP=1 时 `4147` 的守卫 `world_size > 1` 为假，receive 根本不调用，`prev_sampled_token_ids` 保持 `4202` 设的 `None`。

**这是分叉的本质**：

| | `prev_sampled_token_ids` 进入下一步时的状态 | `_prepare_input_ids` 走哪条路 | 是否触发 scatter |
|---|---|---|---|
| PP=1 | `None`（`4202` 清空） | normal（`1621` copy_to_gpu） | 否 |
| PP=2 stage0 | `recv`（`4405` 无条件赋值） | async（`1701` scatter） | **是** |
| PP=2 stage1 | `None`（`4202` 清空，last rank） | normal | 否 |

与 dump 一致——scatter 只在 PP=2 step1 rank0（stage0）出现一次。

### 6.3 回填逻辑不区分 decode / chunked prefill

进入 async 路径后（`gpu_model_runner.py:1640-1707`）：

```python
prev_positions = self.prev_positions.np[:num_reqs]   # 当前请求 → 上步行号，-1 表示新请求
for cur_index in range(num_reqs):
    prev_index = prev_positions[cur_index]
    if prev_index < 0:
        continue                                       # 新请求跳过
    prev_indices.append(prev_index)
    ...
    flattened_index = cu_num_tokens[cur_index].item() - 1   # 本步该请求最后一个 token 的扁平索引
    sample_flattened_indices.append(flattened_index)
...
self.input_ids.gpu.scatter_(dim=0,
    index=sampled_tokens_index_tensor,                         # = [flattened_index] = [11]
    src=self.input_batch.prev_sampled_token_ids[prev_..., 0])  # = recv = [315]
```

判定是否回填的唯一条件是 `prev_index >= 0`，即"**该请求在上一步也在 batch 里**"。chunked prefill 的 chunk1→chunk2 是**同一请求**，所以 `prev_index >= 0`，被当作"common decode request"进入回填。`flattened_index = cu_num_tokens[-1] - 1 = 11`（chunk2 有 12 个 token），scatter 把 `315` 写到 `input_ids[11]`，**覆盖掉原本的 prompt token 36**。

这就是 dump 里看到的 `index=11, src=315`。

### 6.4 token 315 的回传链路

`315` 来自 step0 chunk1 prefill 在 stage1（last rank）sampler 的 argmax。回传链路在 `gpu_model_runner.py:4194-4197`：

```python
if self.use_async_scheduling:
    pp = get_pp_group()
    if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
        self._pp_broadcast_prev_sampled_token_ids(sampler_output.sampled_token_ids)
```

`_pp_broadcast_prev_sampled_token_ids`（`gpu_model_runner.py:4390`）：
```python
# Skip for chunked prefill: sampled tokens are dummy and will be discarded
if not self._is_all_reqs_chunked_prefill():
    torch.distributed.broadcast(sampled_token_ids, src=pp.rank, group=pp.device_group)
```

> 这里有一个**设计意图与实现的不一致**：注释明确说 chunked prefill 的采样 token 是 dummy、应丢弃、无需广播——broadcast/receive 两端都据此跳过通信。但 receive 端 `4405` 行却**无条件**把 `prev_sampled_token_ids` 赋成 `recv`：
>
> - 若本步判定 `_is_all_reqs_chunked_prefill() == False`（例如 chunk2 是最终 chunk 不 discard，或 batch 里混了 decode 请求），broadcast 真实执行，stage0 拿到真值 `315`；
> - 若判定为 True，broadcast 跳过，`recv` 是 `torch.empty` 未初始化显存。
>
> dump 里 src 恰好是 `315`，说明当时的 chunk2 步被判定为"非全部 chunked prefill"，broadcast 实际执行了。
>
> 无论哪种，**根因相同**：`4405` 令 `prev_sampled_token_ids != None`，使 chunk2 误入 async 回填路径。

### 6.5 调度路径图

```
                          PP=1                          PP=2
                     (world_size==1)              (world_size==2, async)
                     ─────────────                ──────────────────────
sample_tokens 收尾    4202: prev = None            stage1: 4194 broadcast 315
                                                   stage0: (下一step开头)
                                                     4147 → 4148 _pp_receive
                                                     4405: prev = recv (=315)  ← 不清空
                                                     ────────────────────────
下一步 _prepare_      1619: prev is None           1619: prev is None? False
input_ids              → 1621 copy_to_gpu           → 1640 循环: prev_index>=0
                       (normal 路径)                    chunk1→chunk2 同请求 ✓
                                                        → flattened_index=11
                     ❌ 无 scatter                  1701: scatter_(index=11, src=315) ✅
                                                        覆盖 input_ids[11]
                                                     (= prompt tok_36 → 315)
```

### 6.6 一句话总结

> 这个 `scatter_` 是 async scheduling 给 **decode 步骤**复用上一步采样 token 的 GPU 端回填。PP=2 时 stage0 通过 `_pp_receive_prev_sampled_token_ids_to_input_batch`（`4405`）**无条件**把 `prev_sampled_token_ids` 设成来自 last rank 的 `recv`；而回填逻辑（`1640`）只看"请求是否在上步也存在"（`prev_index >= 0`），**不区分当前步是 decode 还是 chunked prefill**。于是 chunk1→chunk2 的同一请求被误判为 common decode request，stage1 在 chunk1 末端采出的（本应 discard 的）`315` 被 scatter 覆盖了 chunk2 的最后一个 prompt token。PP=1 因 `4147` 的 `world_size > 1` 守卫根本不调用 receive、且 `4202` 把 `prev_sampled_token_ids` 清成 `None`，所以走 normal 路径，不触发这个回填。

## 七、根因总结（现象级）

1. **step0** 完成 chunk1 prefill 后，两端通过 argmax 都采样到第一个解码 token **315**，整个 forward 的统计量完全一致，跨卡 stage0→stage1 也没引入精度损失。
2. **PP=1 走纯 chunked prefill**：step1 只处理 chunk2 prompt 的 12 个 token，input_ids = `[tok_25, tok_26, …, tok_36]`（位置 25–36）。TRUE step1 没有 scatter_ 这一步，因为单卡不融合 decode。
3. **PP=2 在 step1 把 step0 解码的 315 合并进同一 batch**：vLLM V1 的 `_prepare_input_ids` 通过 `Tensor.scatter_(dim=0, index=11, src=315)` 把 315 写入 token buffer 的第 12 个位置（pos 36）。于是 FALSE step1 实际 input_ids 变成 `[tok_25, tok_26, …, tok_35, 315]`。
4. embed_tokens 是纯 gather：相同权重 + 不同 input_ids → 输出 tensor 只有最后一行不同（`weight[315]` 替换了 `weight[tok_36]`）。两端的 token 序列都包含同样的 `151645` 与 `30`，所以 Max/Min 看不到差异；但只有一行被替换，导致 12×3584 = 43008 个元素的 Mean 漂移 ~6e-5、Norm 漂移 0.015625。
5. 这一行差异经 RMSNorm/Linear/Attention 一路放大，stage0 输出本就 drift，跨卡送到 stage1 后 layer 14 RMSNorm 的 input 已经不一致；最终 logits / 下一个采样 token 可能不同，这也是为什么 PP=2 跑到了 step12、PP=1 只跑到 step10。

| 现象 | 实际成因 |
|---|---|
| "看起来像精度漂移"的 Mean/Norm 微差 | 实际是 input_ids 一行被替换 |
| 触发点 | vLLM V1 `_prepare_input_ids` 调用 `self.input_ids.gpu.scatter_` 把 315 写入 chunk2 batch 的最后位置 |
| 触发条件 | 仅在 PP>1 时——PP=1 单卡不融合 decode 与 chunk2 prefill |
| 直接影响算子 | `Functional.embedding.0` / `AscendVocabParallelEmbedding.forward.0` |
| 后果 | 层 0–27、lm_head、采样路径全链路传播放大 → step 数差异 (11 vs 13) |

## 八、修复方向（含精确代码靶点）

### 8.1 根因修复（二选一）

**方向 A — receive 端对称化（推荐改法小、风险低）**

在 `_pp_receive_prev_sampled_token_ids_to_input_batch`（`gpu_model_runner.py:4405`）加守卫，使 chunked prefill 时保持 `prev_sampled_token_ids = None`，与 broadcast 端 `4390` 对称：

```python
# gpu_model_runner.py:4401-4405  修改后
recv = torch.empty((num_reqs, 1), dtype=torch.int32, device=self.device)
if not self._is_all_reqs_chunked_prefill():
    torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)
    self.input_batch.prev_sampled_token_ids = recv          # 仅非 chunked prefill 时赋值
# else: 保持 None（与 4202 行为一致），下一步走 normal 路径
```

- 优点：一行守卫，改动最小；与 broadcast 端注释"sampled tokens are dummy and will be discarded"的意图完全对齐。
- 风险：需确认 chunked prefill 下游没有别处依赖 `prev_sampled_token_ids != None`（目前看只有 `_prepare_input_ids` `1619` 处用到这个判空分支，安全）。

**方向 B — 回填路径过滤（更稳健，堵住根本路径）**

在 `_prepare_input_ids` 的循环（`gpu_model_runner.py:1640`）里，对 chunked prefill 请求（该请求本步 `scheduled_tokens` 不是单 token decode）跳过 `sample_flattened_indices.append`，不把它的 `flattened_index` 纳入 scatter 目标：

```python
for cur_index in range(num_reqs):
    prev_index = prev_positions[cur_index]
    if prev_index < 0:
        continue
    # 新增：仅对"纯 decode"请求回填，跳过 chunked prefill 请求
    req_id = self.input_batch.req_ids[cur_index]
    num_sched = scheduler_output.num_scheduled_tokens.get(req_id, 0)
    if num_sched > 1:
        continue   # chunked prefill / 多 token 请求，不做上步 token 回填
    prev_indices.append(prev_index)
    ...
```

- 优点：堵住"任何非 decode 请求被误回填"的根本路径，不只 chunked prefill 一种情形。
- 风险：需明确"什么算 decode 请求"的判定条件（`num_scheduled_tokens == 1`？还是看 `num_computed_tokens` vs `num_tokens`？），要和 scheduler 语义对齐，改动面比 A 大。

> 综合：**优先方向 A**（最小正确修复），方向 B 作为更彻底的防线补充。两者也可同时实施——A 保证 chunked prefill 不误触发回填，B 保证即便其他非 decode 场景也不会被误回填。

### 8.2 验证 / 规避手段

1. **测试时关闭融合**：若只想做精度对齐验证，可关掉 `enable_chunked_prefill`、把 `max_num_batched_tokens` 调到最大限度让一次 prefill 完成整个 prompt，跳过 chunk2+decode 融合窗口即可。
2. **报告精度时区分"调度差异"与"数值差异"**：dump 中 Mean/Norm 仅一行变化通常是调度造成的；真正 bf16 数值漂移的特征是 Max/Min 不变但每层平均都微小漂移，可在工具里增加按"输入指纹是否完全相同"的报告维度。

## 九、查证清单

| 验证步骤 | 路径 | 结果 |
|---|---|---|
| step0 stage0 模块逐项对比 | `true/step0/rank0` vs `false/step0/rank0` | 362/362 完全一致 |
| step0 stage1 + 后处理对比 | `true/step0/rank0` vs `false/step0/rank1`，按 (layer 14–27, post-layer, argmax) | 154/154 完全一致 |
| step0 argmax 输出 | `Tensor.argmax.1.forward` | 两端都是 token 315 |
| 跨卡传输精度 | `Module.model.layers.14.input_layernorm` INPUT | step0 完全相同，step1 已 drift |
| stage0→stage1 是否引入 drift | step0 各 layer 14 INPUT 字段对比 | 完全一致，排除 HCCL 路径 |
| step1 首个差异模块 | `('emb', 'asc')` = embed_tokens | 311/311 模块全部 drift |
| 多出来的 op | `Tensor.scatter_.0.forward` 出现稀少 | 仅 PP=2 step1 stage0 一次 |
| scatter 的 src | int32 [1] Max=315 Min=315 | 与 step0 argmax token 完全对应 |
| scatter 的 index | int64 [1] Max=11 Min=11 | 写入 chunk2 batch 的最后位置（pos 36） |
| 调用栈确认 | `stack.json` "4" | 由 `gpu_model_runner.py:1701 _prepare_input_ids` 调起 |
