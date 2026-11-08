# DeepSeek V4 MTP详解


> **代码版本冻结声明**：本文基于 **vllm-ascend v0.32.0** 源码整理，MTP 实现位于 [mtp.py](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/mtp.py)。
> **统一模型配置**（W4A8 版，MTP 相关行）：`num_nextn_predict_layers=1`（1 个 MTP 预测层）、`hidden_size=7168`、`hc_mult=4`、`vocab_size=129280`；MTP 层在 `compress_ratios` 数组中对应下标 61（值 **0=dense**，即 draft 注意力不带 Compressor/Indexer），但 FFN 仍是 MoE（384 专家、Top-6，hash 路由关闭）。

---

## 1. vLLM step总览


### 1.1 异步调度（async scheduling）结合源码详解

> **一句话**：默认的同步调度里，EngineCore 每步都要"调度 → 前向 → **阻塞等采样结果** → 回写"，采样期间 CPU 空转、GPU 空泡；**异步调度**把"前向的 Future"和"采样的 Future"塞进一个 `batch_queue` 先排队，CPU 立刻去调度**下一批**，等采样结果真回来了再回写——用**多步在途**填满 GPU。代价是：`Scheduler` 在**草稿 token 还没算出来**的时候就得先排下一步，于是引入 **placeholder（占位）** 机制与草稿 token id 的**延迟回填**。

#### 1.1.1 开关与调度器选择

配置项 `async_scheduling: bool | None = None`（vLLM `config/scheduler.py:208`）——**None 表示自动**。它决定启用哪个 Scheduler 类（`config/scheduler.py:228-236`）：为真则用 `AsyncScheduler`，否则用同步的 `Scheduler`。EngineCore 侧在构造时读入（`core.py:242-243`），并据此决定 `step_fn` 的走法（`core.py:220-226`）：

```python
# core.py:220-226
self.step_fn = self.step if self.batch_queue is None else self.step_with_batch_queue
self.async_scheduling = vllm_config.scheduler_config.async_scheduling
```

> **注意**：`step_fn` 选的是 `step` 还是 `step_with_batch_queue`，**取决于 `batch_queue` 是否存在**——而 `batch_queue` 在 `batch_queue_size = max_concurrent_batches > 1` 时才建（`core.py:220-226`），这正是 **PP（流水线并行）+ 异步** 的场景。也就是说"异步"与"批队列"是**两个正交但常配对**的机制：异步解决"采样等待空泡"，批队列解决"PP 气泡"，二者叠加即 `step_with_batch_queue` 里同时存在多个在途 Future。

#### 1.1.2 主循环骨架：run_busy_loop → _process_engine_step → step_fn

```python
# core.py:1547-1558
def run_busy_loop(self):
    while self._handle_shutdown():
        self._process_input_queue()     # ① 收请求 / abort，直到有活干
        self._process_engine_step()     # ② 走一步（step_fn）

# core.py:1603-1619
def _process_engine_step(self) -> bool:
    outputs, model_executed = self.step_fn()        # ← 同步=step / 异步+批队列=step_with_batch_queue
    for output in outputs.items() if outputs else ():
        self.output_queue.put_nowait(output)        # 结果推给前端
    self.post_step(model_executed)                  # 草稿回写钩子（见 1.1.4）
    if not model_executed and self.scheduler.has_requests():
        time.sleep(0.001)                           # 无执行但有活：让出 GIL
    return model_executed
```

三者关系：**`run_busy_loop` 是外壳，`_process_engine_step` 是"走一步并交付输出"，`step_fn` 是真正的一步**——同步时指向 `step`（`core.py:657`），异步 + 批队列时指向 `step_with_batch_queue`（`core.py:698`）。判活条件是 `has_work()`（`core.py:1534-1540`）：`engines_running or scheduler.has_requests() or bool(self.batch_queue)`——**队列里还有在途批次也算有活**，这是异步能不空转的关键。

#### 1.1.3 同步 vs 异步：两次 RPC 的推进方式不同

**同步路径**（`EngineCore.step`，`core.py:657-687`）——一条直线，采样后**原地阻塞**取结果：

```python
scheduler_output = self.scheduler.schedule(...)                 # :667
future = self.model_executor.execute_model(..., non_block=True) # :668 前向 Future，立刻返回
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)  # :669
model_output = future.result()                                  # :674 ★阻塞：等前向完
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)   # :676 第二次 RPC，再等
engine_core_outputs = self.scheduler.update_from_output(...)    # :681 回写
```

**异步路径**（`step_with_batch_queue`，`core.py:698-812`）——把 Future **入队**，采样也 `non_block=True`，本轮**不阻塞**：

```python
exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)  # :727-729
...
future = self.model_executor.sample_tokens(grammar_output, non_block=True)         # :743-745 ★采样不阻塞
...
batch_queue.appendleft((future, scheduler_output, exec_future))                    # :753 三元组入队
if len(batch_queue) < self.batch_queue_size and (model_executed or ...):
    return None, model_executed        # :759 ★队列没满就立刻返回，CPU 转去排下一批
# ---- 队列满/没活可排时，才回头处理最老的批次 ----
future, scheduler_output, exec_model_fut = batch_queue.pop()                       # :768
model_output = future.result()                                                         # :773 此时才阻塞
engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)  # :783
```

对照可见：同步是"**排一批 → 等等等 → 回写**"，异步是"**排一批入队 → 立刻排下一批 → …（队列满）→ 取出最老的结算**"。`batch_queue` 里每个元素是 `(采样Future, 该批的SchedulerOutput, 前向Future)`，**多步在途**让采样/前向的 Device 时间与 CPU 调度时间重叠。

#### 1.1.4 草稿 token 的"延迟回填"——异步最核心的差异

这是本题（MTP）最相关的一点。**同步**时，`step()` 阻塞到采样完，`post_step` 即可同步把草稿取回来写进 scheduler：

```python
# core.py:689-696
def post_step(self, model_executed: bool) -> None:
    # 异步调度下无法提前拿到草稿 token id，草稿在 worker 进程内更新，
    # 这里不需要（也不能）回写草稿。
    if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
        draft_token_ids = self.model_executor.take_draft_token_ids()   # :694
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)     # :696
```

**异步**时 `not self.async_scheduling` 为假 → **整个块被跳过**。原因写在注释里：此时 scheduler 已经跑去排**下一步**了，而草稿 token 是**上一步**采样 + 草稿前向才产出的，CPU 侧**还没拿到**——所以草稿留在 **worker 侧的输入批缓冲**里，由 worker 自己在下一步使用；v0.32.0 的 `ModelRunnerOutput` 已**不再携带 `spec_token_ids`**（`vllm/v1/outputs.py:267` 的 `ModelRunnerOutput` 定义中无该字段）。

配套地，`AsyncScheduler` 在 `schedule()` 之后用 **placeholder** 预占位置（vLLM `v1/core/sched/async_scheduler.py:25-57`）：

```python
# async_scheduler.py:25-57
def _update_after_schedule(self, scheduler_output):
    super()._update_after_schedule(scheduler_output)
    ...
    self._spec_token_placeholders = [-1] * scheduler_output.num_spec_tokens_to_schedule
    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests[req_id]
        if request.is_prefill_chunk:
            continue
        ...
        cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
        request.num_output_placeholders += (self.num_sampled_tokens_per_step + cur_num_spec_tokens)  # :45-47
        # 先占位 -1，真实草稿 id 稍后在 worker 进程里回填
        request.spec_token_ids = self._spec_token_placeholders        # :50 ★placeholder
```

`_spec_token_placeholders = [-1] * num_spec_tokens`（`async_scheduler.py:18`）就是**用 `-1` 占住 K 个草稿位**，保证 decode 批形状（`1+K`）不塌；真实 id 由 worker 侧维护。这正是 1.2 节里"新 decode 请求用 `[-1]*K` 补齐"的**异步版本解释**——同步版靠 `Scheduler` 自己补（`scheduler.py:1371-1375`），异步版靠 `AsyncScheduler` 的 placeholder。

`num_output_placeholders` 的账在 `_update_request_with_output`（`async_scheduler.py:59-77`）里回冲：每收到一批真 token 就 `num_output_placeholders -= len(new_token_ids)`（`:70`），并对**非抢占**请求补 `cache_blocks`（`:73-77`）。调度时若 `num_output_placeholders > 0` 还要防多排一步（`scheduler.py:664-678`）。

#### 1.1.5 worker 侧草稿的落盘与取回（v1 回退路径）

> 平台默认走 MRv2（v2 speculator 管线，见 1.1.7）；本小节描述 `VLLM_USE_V2_MODEL_RUNNER=0` 回退到 v1 时的行为。

v1 的 `NPUModelRunner`（`worker/v1/model_runner.py`）把草稿 id 保存在 runner/输入批里，不再通过 `ModelRunnerOutput` 出参：

- 草稿前向结束后调用 `_copy_draft_token_ids_to_cpu(scheduler_output)`（`v1/model_runner.py:2281`，调用点 `:2967`）。**异步 + spec** 下若没有结构化输出、没有 `output_token_ids`、且不是 PP，会直接 `return`（`:2293-2298`）——草稿只留在 worker 的 GPU/输入批缓冲中，正好对应 `post_step` 被跳过。
- `take_draft_token_ids`（`v1/model_runner.py:2323-2344`）在父类结果上再做**动态 SD 截断**：按 `dynamic_spec.num_verify_tokens` 把每请求草稿裁到本步实际验证长度（`:2330-2343`）。
- vLLM 侧草稿缓冲区在 `GPUModelRunner` 初始化时创建：`draft_token_ids_event / copy_stream / draft_token_ids_cpu`（`vllm/v1/worker/gpu_model_runner.py:922-936`），异步 spec 的开关为 `use_async_spec_decode`（`:678-680`）。
- 异步 + 结构化输出时，`step_with_batch_queue` 在计算延迟批次的 grammar bitmask 前先取草稿并过滤非法 spec token：`take_draft_token_ids`（`core.py:796`）→ `Scheduler.update_draft_token_ids_in_output`（`core.py:801`，实现见 `scheduler.py:2571`），非法项补 `-1` 后跳过 bitmask 计算。

> **vllm-ascend 侧的 `use_async_scheduling`**：与 vLLM 主仓 `async_scheduling` 语义一致，`v1/model_runner.py` 中的分支主要集中在三件事：① 采样后是否同步 D2H（`_bookkeeping_sync` / `_copy_draft_token_ids_to_cpu`，`:2281`、`:3110`）；② bookkeeping 的同步/异步分支；③ spec decode 的缓冲与事件（`:2293-2298`、`:3027-3070`）。

#### 1.1.6 一句话对照表

| 维度 | 同步（`step`） | 异步（`step_with_batch_queue` / `AsyncScheduler`） |
|------|------|------|
| 调度器类 | `Scheduler`（`scheduler.py`） | `AsyncScheduler`（`async_scheduler.py:14`） |
| 采样 RPC | `sample_tokens(...)` 阻塞（`core.py:676`） | `sample_tokens(..., non_block=True)` 入队（`core.py:743-745`/`:753`） |
| 多步在途 | 无 | `batch_queue` 存 `(sample_fut, sched_out, exec_fut)`（`core.py:768`） |
| 草稿回写 | `post_step → take_draft_token_ids`（`core.py:694-696`） | **跳过**（`core.py:693`），草稿留在 worker 输入批，结构化输出场景由 `core.py:796-803` 显式取回 |
| 草稿占位 | `[-1]*K` 由 `Scheduler` 补齐 | `AsyncScheduler` placeholder `[-1]*K`（`async_scheduler.py:18`/`:50`） |
| 延迟账本 | 无 | `num_output_placeholders` 预加/回冲（`async_scheduler.py:45-47`/`:70`） |
| 产物一致性 | 一步一结 | 靠 placeholder + worker 侧草稿缓冲保证 `1+K` 形状与最终 logits 一致 |

#### 1.1.7 v2（平台默认）与 v1（回退）的 MTP 差异

v0.32.0 的 Ascend 平台**默认启用 MRv2**：`use_v2_model_runner`（[mrv2_utils.py:99-133](../../../../../vllm-ascend/vllm_ascend/mrv2_utils.py#L99)）在 `VLLM_USE_V2_MODEL_RUNNER` 未设置时返回 `True`（`:132-133`）；只有命中黑名单（VL encoder graph、`draft_window_size`、suffix/ngram、`draft_model`、dflash2、p-eagle；`:124-130`）或显式设 `0`（`:115-122`）才回退 v1。平台侧在 `platform.py::check_and_update_config` 调用 `apply_v2_model_runner_config_patch()`（[platform.py:555](../../../../../vllm-ascend/vllm_ascend/platform.py#L555)），把该选路结果挂成 `VllmConfig.use_v2_model_runner`（`mrv2_utils.py:140-149`）。因此 **v2 是默认路径，v1 是回退路径**，两套 runner 与 MTP 的交互**不要混写**：

| 维度 | v2 runner（平台默认） | v1 runner（回退：`VLLM_USE_V2_MODEL_RUNNER=0`） |
|------|-----------|-----------|
| 草稿驱动 | 上游 speculator 管线：`init_speculator`（`worker/v2/spec_decode/__init__.py:29`，MTP 分支 `:86-91`）返回 `AscendMTPSpeculator`（`v2/spec_decode/mtp/speculator.py:25`），由 `sample_tokens` 调 `self.speculator.propose(...)` | `NPUModelRunner.propose_draft_token_ids`（`:2016`）→ `drafter._propose`（`:2216`） |
| hidden 形态 | `DeepSeekMultiTokenPredictorLayer.forward` 在 `use_v2_model_runner` 时返回 `hidden_states.flatten(1)`（`mtp.py:165-167`），即 `(T, 4*7168)` 拼平，供 speculator 直接消费 | 返回 `(T, 4, 7168)` 未压缩残差流 |
| 目标隐态取用 | 上游 `sample_tokens` 用 `get_drafter_hidden_states`（`gpu/spec_decode/utils.py:16-32`）→ `get_mtp_target_hidden_states()` 取 4 流缓冲；replicated PCP draft 另由 `_restore_replicated_draft_target_states`（`v2/model_runner.py:365-405`，调用 `:377-385`）还原全局布局 | `propose_draft_token_ids` 内直接取 `get_mtp_target_hidden_states()`（`:2174-2178`） |
| 采样入口 | `sample_tokens`（`v2/model_runner.py:408`）→ `super().sample_tokens`（`:441` → `vllm/v1/worker/gpu/model_runner.py:2088`） | `sample_tokens`（`:2882`） |

---

### 1.2 EngineCore step 调用栈（通用，不涉及模型内部）

一轮 step ＝「① 调度 → ② 目标前向（非阻塞发起）→ ③ 验证采样（第二次 RPC，内嵌 ④ 草稿前向）→ ⑤ 结算回写＋草稿回写」。目标前向以 `non_block=True` 发 RPC（core.py:668）立刻拿 Future；`future.result()` 返回 `None` 即"前向已就绪，可采样"信号（:674），EngineCore 再发第二次 RPC `sample_tokens`（:676，携带 grammar bitmask 等调度侧产物）。两次 RPC 之间的中间态（logits、hidden_states、spec 元数据）暂存在 worker 侧 `ExecuteModelState`（v2 由父类 `vllm/v1/worker/gpu/model_runner.py:2064` 写入；v1 回退在 `v1/model_runner.py:2776`）。④ 默认由 v2 speculator 管线驱动（V1 回退子树见本节末尾），只展开到 MTP 草稿模型入口，层内结构见第 3～5 章。

```
run_busy_loop → step_fn() ⇒ EngineCore.step()               # core.py:1547/:1606 → :657
│
├─ ① 调度 Scheduler.schedule()                             # core.py:667 → scheduler.py:591
│     · D(t-1) 装车 scheduled_spec_decode_tokens            #   :861-872，随后清空 spec_token_ids :876
│     · 新 decode 请求用 [-1]*K 补齐到 1+K（prefill 混批不补）#   :1371-1375
│
├─ ② 目标前向 Executor.execute_model(non_block=True)        # core.py:668 ⇢ collective_rpc
│     └ Worker.execute_model                                #   worker.py:874
│         └ NPUModelRunner.execute_model（v2，平台默认）      #     v2/model_runner.py:706
│              └ super().execute_model                       #       :746 → gpu/model_runner.py:1719
│                   · _update_states / 组 batch               #       gpu/model_runner.py
│                   · 一遍前向吃下本步全部 token              #       模型 forward
│                   · 暂存 ExecuteModelState → return None   #       gpu/model_runner.py:2064（可采样信号）
├─ future.result() → None                                    # core.py:674
│
├─ ③ 验证采样 Executor.sample_tokens(grammar_output)         # core.py:676 ⇢ collective_rpc
│     └ Worker.sample_tokens                                #   worker.py:945
│         └ NPUModelRunner.sample_tokens（v2，平台默认）      #     v2/model_runner.py:408
│              · _restore_replicated_draft_target_states     #       :365-405（replicated PCP draft 还原）
│              └ super().sample_tokens                       #       :441 → gpu/model_runner.py:2088
│                   · 解包 ExecuteModelState / 组采样输入     #       gpu/model_runner.py
│                   · 采样：接受 / 拒绝 / 重采样 + bonus token #     vllm/v1/sample/rejection_sampler.py:RejectionSampler.forward（L106）
│                   │
│                   ├─ ④ 草稿前向 self.speculator.propose()  #   gpu/model_runner.py:2239
│                   │    └ AscendMTPSpeculator               #     v2/spec_decode/mtp/speculator.py:25
│                   │         └ AscendTargetDependentARSpeculator.propose  # target_dependent_ar/speculator.py:335
│                   │              ├ _prefill                #       :564（prompt 段填 MTP 层 KV）
│                   │              ├ _generate_draft         #       :507（decode 增量逐圈递推）
│                   │              └ _run_model              #       :485（草稿模型 forward）
│                   │                   └ DeepSeekV4MTP.forward
│                   └ 草稿写入 req_states.draft_tokens
│              · ModelRunnerOutput(...)；step() return        # gpu/model_runner.py / core.py:687
│
├─ ⑤ 结算回写 Scheduler.update_from_output()                 # core.py:681 → scheduler.py:2026
│     · num_accepted / num_computed_tokens -= num_rejected   #   :2118-2128
└─ ⑤ 草稿回写 post_step(model_executed)                      # core.py:1611 → :689
       · take_draft_token_ids                                #   :694 → worker.py:1468
       · update_draft_token_ids                              #   :696 → scheduler.py:2550
           chunk 未完成 → 草稿丢弃 :2560-2564；否则写回 :2567-2569（供下一轮 ①）
```

**V1 回退路径（`VLLM_USE_V2_MODEL_RUNNER=0`）**：两步 RPC 全部落在 `worker/v1/model_runner.py`，草稿前向以闭包形式嵌在 `sample_tokens` 内，细节保留如下：

```
├─ ② NPUModelRunner.execute_model                           #   v1/model_runner.py:2347
│     · _update_states / _prepare_inputs                     #       :1001 / :1386
│     · _model_forward：一遍前向吃下本步全部 token             #       :3505
│     · 暂存 ExecuteModelState → return None                 #       :2776-2796（可采样信号）
├─ ③ NPUModelRunner.sample_tokens                            #   v1/model_runner.py:2882
│     · 解包并清空 execute_model_state                        #       :2903-2919
│     · _sample：prefill→Sampler / decode→AscendRejectionSampler #   :3073
│     ├─ ④ 草稿前向 propose_draft_token_ids 闭包              # :2946 · 调用 :3015（PP 特例提前 :2983-2987）
│     │    └ NPUModelRunner.propose_draft_token_ids          #   :2016
│     │         · MTP：改吃 _mtp_hidden_buffer                #     :2174-2178
│     │         └ drafter._propose                           #       :2216 → llm_base_proposer.py:1009
│     │              · set_inputs_first_pass 右移一位          #         llm_base_proposer.py:1776
│     │              · compute_draft_token_ids → argmax       #      :1422
│     └ _copy_draft_token_ids_to_cpu                         #   :2281 · 调用 :2967
│     · _bookkeeping_sync：汇成本轮有效 token                  # :3110 · 调用 :2998
│     · ModelRunnerOutput(...)；step() return                # :3027-3038 / :3053-3070 / core.py:687
```

### 1.2 各阶段职责

表格按 ①→⑤ 展开：**第 1 列**阶段（含锚点）、**第 2 列**职责、**第 3/4 列** prefill、decode 两种批型的**输入→输出 shape**。先约定符号与 DSV4 特有的隐态交接：

- **符号**：`B`＝本步批内请求数；`K`＝num_speculative_tokens（每请求草稿数）；`L_i`＝请求 i 本步输入 token 数、`T=ΣL_i`；`V=129280`（词表）；`H=7168`；`d=4H=28672`（4 条残差流拼平宽度）；`D(t)`＝第 t 轮产出的草稿，每请求 `(K,)`。
- **DSV4 隐态交接**：目标模型 forward 末尾把**未压缩** 4 流残差 `(T, d)` 写入 `_mtp_hidden_buffer`（[model.py:1039-1041](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py#L1039)），并把 hc_head 压缩＋norm 后的 `(T, H)` 作为返回值（`model.py:1043-1048`，供 ③ 采样）；④ 的 MTP 草稿经 `get_mtp_target_hidden_states`（[model.py:1180-1184](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py#L1180)；v2 由 `get_drafter_hidden_states`（`gpu/spec_decode/utils.py:16-32`）取用，v1 回退在 `v1/model_runner.py:2174-2178`）改吃 4 流缓冲——② 的 `logits_indices` 抠位与 ④ 的 hidden 选择因此彼此独立。
- **批型**：prefill 列含 chunked prefill（含最后一块）；decode 列指稳态 decode 批（首个 decode 步的草稿来源见 ⑤ 行）。

| 阶段（锚点） | 职责 | prefill（含 chunk）输入 → 输出 | decode（稳态）输入 → 输出 |
|------|------|------|------|
| **① 调度**<br>`Scheduler.schedule`<br>core.py:667 → scheduler.py:591 | 编批、分配 KV 块；把 ⑤ 上一步写回的 `D(t-1)` 装进 `scheduled_spec_decode_tokens`（:861-872）并**清空** `request.spec_token_ids`（:876）；新 decode 请求用 `[-1]*K` 占位补齐到 `1+K`，保住 decode 统一形状（:1371-1375；prefill 混批时不补齐） | **无草稿**（表为空）<br>入：各请求 `num_computed_tokens` / prompt 状态<br>出：`SchedulerOutput`：`num_scheduled_tokens[rid]`＝本块长度（`T=Σ` 各块），`scheduled_spec_decode_tokens={}` | **有草稿**（首个 decode 步＝最后一块 prefill ④ 的产出，见 ⑤）<br>入：`request.spec_token_ids=D(t-1)`，每请求 `(K,)`<br>出：`SchedulerOutput`：`num_scheduled_tokens[rid]=1+K`、`scheduled_spec_decode_tokens[rid]=(K,)`（新请求记 `[-1]*K`） |
| **② 目标前向**<br>`execute_model(non_block=True)`<br>core.py:668 → worker.py:874 → v2/model_runner.py:706 → gpu/model_runner.py:1719<br>（v1 回退：v1/model_runner.py:2347，`_model_forward` :3505） | 组 batch / 注意力元数据后**一遍**前向吃下本步全部 token；按 `logits_indices` 抠采样位算 logits；暂存 `ExecuteModelState` 返回 `None`（v2 暂存 gpu/model_runner.py:2064；v1 回退 :2776、返回 None :2796＝"可采样"信号）；副作用：4 流残差写 `_mtp_hidden_buffer`（model.py:1039-1041） | 入：`input_ids/positions (T,)`，`T=Σ` 各请求本块 prompt 长度（chunk 取本块）<br>出：模型返回 `(T, H)`＋副作用 `_mtp_hidden_buffer (T, d)`；`logits_indices`＝每请求末位 → `sample_hidden_states (B, H)` → **logits `(B, V)`**（`spec_decode_metadata=None`） | 入：`input_ids/positions (T,)`，`T=Σ(1+K_i)`（统一形状时 `=B(1+K)`；每请求 `[t_now, D(t-1)]`，`-1` 占位请求仍按 `1+K` 计）<br>出：`sample_hidden_states (Σ(1+K_i), H)` → **logits `(Σ(1+K_i), V)`**＝每请求 K 个验证位＋1 个 bonus 位 |
| **③ 验证与采样**<br>`sample_tokens(grammar_output)`<br>core.py:676 → worker.py:945 → v2/model_runner.py:408 → gpu/model_runner.py:2088<br>（v1 回退：:2882 → `_sample` :3073） | 解包暂存 logits（v1 回退 :2903-2919，可叠 grammar bitmask）；prefill 走普通 `Sampler`、decode 走 `AscendRejectionSampler` 逐 token 接受/拒绝/重采样＋bonus；`_bookkeeping_sync`（v1 回退 :3110）汇成有效 token 表 | 入：logits `(B, V)`（`spec_decode_metadata=None`）<br>出：`sampled_token_ids (B,)`＝每请求 1 个新 token，即**首个输出 token**（未完成 chunk 多采的位在 `_bookkeeping_sync` 丢弃） | 入：logits `(B(1+K), V)`＋`SpecDecodeMetadata`（`draft_token_ids (B·K,)`、target/bonus 位索引）<br>出：`sampled_token_ids (B, 1+K)`（接受位＝真 id、拒绝位＝`-1`、末位 bonus）；簿记后每请求留 **`1~(1+K)` 个有效新 token**＝接受草稿＋1 个新锚点 |
| **④ 草稿前向**（嵌在 ③ RPC 内）<br>默认 v2：`speculator.propose` gpu/model_runner.py:2239 → `AscendMTPSpeculator` v2/spec_decode/mtp/speculator.py:25<br>→ `AscendTargetDependentARSpeculator.propose` :335（`_prefill` :564 / `_generate_draft` :507 / `_run_model` :485）<br>（v1 回退：闭包 :2946 → `propose_draft_token_ids` :2016 → `drafter._propose` :2216 → llm_base_proposer.py:1009） | 驱动 **MTP 层**前向并写**草稿层自己的 KV**：decode 用刚采出的 token 逐圈递推 K 次得 `D(t)`；prefill 走 **draft prefill** 填 MTP 层 KV、并在每请求末位 argmax 出首个 decode 步草稿（K>1 时逐圈递推）；MTP 的 hidden 由 `get_drafter_hidden_states`（gpu/spec_decode/utils.py:16-32）换成 4 流缓冲（v1 回退 :2174-2178） | **draft prefill**<br>入：`target_token_ids (T,)`＝**整段 prompt**、`target_hidden_states (T, d)`、`next_token_ids (B,)`<br>`set_inputs_first_pass`（[llm_base_proposer.py:1776](../../../../../vllm-ascend/vllm_ascend/worker/v1/spec_decode/llm_base_proposer.py#L1776)）右移一位 → 草稿输入 `(T,)`（每请求 `[p₂…p_L, s]`，配 `(h_j, e_{j+1})` 对）<br>K 圈递推（首圈 `(T,)`，后续每圈 `(B,)`）＋经 `block_table[kv_cache_gid]` 写 MTP 层 KV<br>出：**`draft_token_ids (B, K)=D(t)`**＝首个 decode 步草稿（去留由 ⑤ 决定） | **增量**（已剔除上一轮拒绝位）<br>入：`target_token_ids (B,)`＝每请求 1 个刚采出的 token、`target_hidden_states (B, d)`、`next_token_ids (B,)`<br>K 圈递推（每圈 forward `(B,)` token＋`(B, d)` hidden → argmax `(B,)`）<br>出：**`draft_token_ids (B, K)=D(t)`**，供下一 decode 步 ② 验证 |
| **⑤ 结算回写＋草稿回写**<br>`update_from_output`<br>core.py:681 → scheduler.py:2026；<br>`post_step` core.py:689-696<br>→ `update_draft_token_ids` :2550 | 按接受数回退进度、token 写回请求并产出 `EngineCoreOutputs`；把 `D(t)` 写回 `request.spec_token_ids`——`is_prefill_chunk` 为真（chunk 未跑完）则丢弃草稿、否则写回（:2560-2564/:2567-2569） | 入：`sampled_token_ids (B,)`<br>**无拒绝结算**：最后一块的采样 token append 为首个输出 token（未完成 chunk 的位丢弃）<br>草稿去留：chunk 未完 → `D(t)` **丢弃**（:2560-2564，MTP KV 照写）；最后一块已跑完 → **`request.spec_token_ids=(K,)`（:2567-2569）＝首个 decode 步输入** | 入：每请求 `1~(1+K)` 个有效新 token＋`(K,)` 草稿<br>`num_accepted=max(len(generated)−1, 0)`（:2118，`num_sampled=1`：scheduler.py:152）→ `num_computed_tokens −= K−num_accepted`（:2126-2128，被拒位下轮重算）<br>出：**`request.spec_token_ids=D(t)`（:2567-2569）**（非 async 经 `take_draft_token_ids` worker.py:1468；async 时草稿留在 worker 输入批，结构化输出场景由 core.py:796-803 取回并过滤非法项）＝下一步 ① 的草稿源 |

**MTP 层的 prompt KV 并不缺**——它正是在 ④ 的 prefill 分支（draft prefill）里填的：draft prefill 消费整段 prompt 的 4 流隐藏状态（② 副作用写入 `_mtp_hidden_buffer`，见上文"DSV4 隐态交接"），经 `input_batch.block_table[kv_cache_gid]` 把 MTP 层自己的 KV 一并写满。

**草稿从哪来、到哪去（错开一步）**：稳态 decode 下，② 消费 ⑤ 上一步写回的 `D(t-1)`，④ 生产下一轮的 `D(t)`，两者永不重合。**首个 decode 步的草稿有明确来源**：最后一块 prefill 步的 ④ 在每请求末位 argmax＋K 圈递推出 `(B, K)`，且该请求的 `is_prefill_chunk` 已在调度簿记中翻 False（`num_computed_tokens` 追平 `num_tokens` 即非 chunk），⑤ 正常写回（:2567-2569）——这批草稿**不丢弃**，直接成为首个 decode 步的输入；**只有还没跑完的 prefill chunk 请求，草稿才被置空丢弃**（:2560-2564）。

---

## 2. DSV4 MTP 总览

### 2.1 设计理念

MTP 是 DeepSeek V4 的推测解码（Speculative Decoding）草稿模型：一次前向额外预测后续 token 的草稿，再交给主模型并行验证，用一次草稿前向 + 一次批量验证替换多轮串行解码。

> **生活化类比**：主模型像一位严谨但语速慢的主审，MTP 像一位语速快的助理先把"接下来可能要说的 1 个词"写在纸条上；主审一次看完纸条，对的直接采用，错的从错处重写。

```
主模型第 t 步：4 条并行残差流的隐藏状态 (T, 4*7168)
    │  （尚未做 hc_head 压缩，直接存入 _mtp_hidden_buffer）
    ▼
MTP 层（mtp.0）
    ├─ 融合：上一步残差流 h_proj  +  下一 token 嵌入 e_proj
    ├─ 过一个完整 draft DecoderLayer（dense 注意力 + MoE）
    ├─ 输出仍是 4 条残差流（延迟压缩）
    └─ compute_logits 时才 hc_head 压回单流 → norm → LM head
```

### 2.2 模块层级

```
DeepSeekV4MTP (入口类, L240-L290; @support_torch_compile)
└── model: DeepSeekMultiTokenPredictor (L179)
    ├── embed_tokens: VocabParallelEmbedding(129280, 7168)     — 词嵌入（L196-199）
    ├── logits_processor: LogitsProcessor                       — logits 处理（L200）
    └── layers: ModuleDict {"0": DeepSeekMultiTokenPredictorLayer (L61), ...}（L187-195）
        ├── e_proj / h_proj: ReplicatedLinear(7168 → 7168)      — 双路投影（走 quant_config，L70-75）
        ├── enorm / hnorm: RMSNorm(7168)                        — 嵌入 / 残差流归一化（L77-78）
        ├── shared_head: SharedHead (L41)                       — 每层各一份（L94）
        │   ├── norm: RMSNorm(7168)
        │   └── head: ParallelLMHead(129280, 7168)
        ├── mtp_block: DeepseekV4DecoderLayer(is_draft_layer=True, L95-101)
        │   └── dense DSA 注意力 + DeepseekV4MoE（无 hash）
        ├── hc_head_fn/base/scale: Parameter (fp32)             — HC 头门控参数（L106-108）
        ├── hc_norm: RMSNorm(4*7168, has_weight=False, fp32)    — 无权重全宽归一化（L111）
        └── topk_indices_buffer: (max_num_batched_tokens, index_topk) int32
            └─ is_v32 时【本层自建】，不与主模型共享（L82-92）
```

### 2.3 与主模型的交互

| 交互点 | 说明 |
|--------|------|
| `_mtp_hidden_buffer` | 主模型 forward 写入的**预 hc_head 残差流缓冲**，形状 `(max_num_batched_tokens, hc_mult*hidden_size)`，每个 target step 后有效（分配 [model.py:931-944](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py#L931)，写入 `model.py:1039-1041`）。MTP 的 `previous_hidden_states` 即来源于此 |
| `topk_indices_buffer` | **主模型与 MTP 不共享**：主模型各层共用一份模型级 buffer；MTP 层在 `__init__` 自建一份传给自己的 `mtp_block`（`mtp.py:82-92`）。原因见文档 5 §6（spec decode 下 impl 层引用会失效） |
| `spec_step_idx` / `spec_step_index` | 当前推测步号；预测器用它对 `num_mtp_layers` 取模轮询（`mtp.py:215`） |
| 词表/embedding | MTP 自建 `embed_tokens`，checkpoint 中通过权重映射加载与主模型同源的词嵌入 |

---

## 3. 入口类与预测器主体

### 3.1 DeepSeekV4MTP — 入口类（L240-L290）

继承 `nn.Module + SupportsPP + DeepseekV2MixtureOfExperts`。构造时创建 `DeepSeekMultiTokenPredictor`（prefix 自动加 `mtp`，L245）并调用 `set_moe_parameters()`（L247）。

#### 3.1.1 set_moe_parameters（L249-L268）

```python
self.expert_weights = []
self.num_expert_groups = getattr(self.config, "n_group", 1)   # V4 配置无 n_group → 1
for layer in self.model.layers.values():
    if isinstance(layer, PPMissingLayer):    # PP 下不在本 rank 的层跳过
        continue
    layer = layer.mtp_block                  # DeepseekV4DecoderLayer
    if isinstance(layer.mlp, DeepseekV4MoE):
        example_moe = layer.mlp              # 注释：取最后的 MoE，前面可能是 dense 层
        self.moe_mlp_layers.append(layer.mlp)
        self.moe_layers.append(layer.mlp.experts)
self.extract_moe_parameters(example_moe)
```

作用与主模型的 MoE mixin 一致：注册 EPLB/EP 通信组、抽取冗余专家等超参，使 draft 层的 384 个专家同样纳入专家并行调度。

#### 3.1.2 forward / compute_logits / embed_input_ids

三个方法都是纯转发（L270-L290）：`forward → model(...)`、`compute_logits → model.compute_logits(...)`、`embed_input_ids → model.embed_input_ids(...)`。

---

### 3.2 DeepSeekMultiTokenPredictor — 预测器主体（L179-L236）

#### 3.2.1 核心成员

| 成员 | 说明 |
|------|------|
| `mtp_start_layer_idx` | `config.num_hidden_layers` = **61**（L183），MTP 配置数组下标从此开始 |
| `num_mtp_layers` | `getattr(config, "num_nextn_predict_layers", 1)` → **1**（L184） |
| `layers` | ModuleDict，key 为 `"0"`…`"N-1"`（L187-L195） |
| `embed_tokens` | `VocabParallelEmbedding(vocab_size, hidden_size)`（L196-L199） |
| `logits_processor` | `LogitsProcessor(vocab_size)`（L200） |

#### 3.2.2 轮询调度（forward，L205-L222）

```python
if inputs_embeds is None:
    inputs_embeds = self.embed_tokens(input_ids)   # 也允许外部传入已算好的嵌入
current_step_idx = spec_step_idx % self.num_mtp_layers
return self.layers[str(current_step_idx)](
    input_ids, positions, previous_hidden_states, inputs_embeds, current_step_idx,
)
```

当推测步号超过层数时循环复用同一批层。本配置只有 1 个 MTP 层，即每步都用 `layers["0"]`。

---

## 4. DeepSeekMultiTokenPredictorLayer — 单层实现（L61-L176）

### 4.1 核心成员

| 成员 | 形状 | 说明 |
|------|------|------|
| `e_proj` | `7168 → 7168` | ReplicatedLinear，**走 quant_config（W4A8 包中量化）**，嵌入投影（L70-72） |
| `h_proj` | `7168 → 7168` | ReplicatedLinear，同上，残差流投影（逐流作用，L73-75） |
| `enorm` / `hnorm` | RMSNorm(7168) | 嵌入归一化 / 上一步残差流归一化（L77-78） |
| `shared_head` | SharedHead | **每个 MTP 层各持一份**（L94，本配置仅 1 份） |
| `mtp_block` | DeepseekV4DecoderLayer | 完整 draft block，传 `is_draft_layer=True` 与自建 buffer（L95-101） |
| `is_v32` | bool | `hasattr(config, "index_topk")`（L82），V4 恒为 True |
| `hc_eps` / `hc_mult` | 1e-6 / 4 | 来自配置（L102-103） |
| `hc_head_fn` | `(4, 4*7168)` fp32 | HC 头门控矩阵（每个目标流一行，L106） |
| `hc_head_base` | `(4,)` fp32 | 门控偏置（L107） |
| `hc_head_scale` | `(1,)` fp32 | 门控温度缩放（L108） |
| `hc_norm` | RMSNorm(`4*7168`, `has_weight=False`, dtype=**fp32**) | 对拼平的全宽残差流做无权重 RMSNorm（L111） |

### 4.2 投影融合（forward，L113-L167）

MTP 输入由两路融合：

1. **下一 token 嵌入** `inputs_embeds`（`(T, 7168)`）：先 mask，再 `enorm`；
2. **上一步 4 条残差流** `previous_hidden_states`：先 view 回 `(T, 4, 7168)`，再 `hnorm`（对最后一维归一化）。

```
inputs_embeds (T, 7168)
    │ where(position==0 → 0)  [L123]
    │ enorm                    [L124]
    │ e_proj                   [L146]
    ▼ (T, 7168) ── unsqueeze(-2) ──▶ (T, 1, 7168)
                                        │ 广播相加
previous_hidden_states (T, 4*7168)     │
    │ view(-1, 4, 7168)        [L125]  ▼
    │ hnorm                     [L126] ▶ (T, 4, 7168)  hidden_states
    │ h_proj（逐流 7168→7168）  [L146]
    ▼ (T, 4, 7168)
```

```python
# L123-L153（顺序：先两路各自归一化，再投影相加）
inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
inputs_embeds = self.enorm(inputs_embeds)
previous_hidden_states = previous_hidden_states.view(-1, self.hc_mult, self.config.hidden_size)
previous_hidden_states = self.hnorm(previous_hidden_states)
hidden_states = self.e_proj(inputs_embeds).unsqueeze(-2) + self.h_proj(previous_hidden_states)
hidden_states, residual = self.mtp_block(positions=positions, hidden_states=hidden_states, residual=None)
```

要点：
- `e_proj` 输出在倒数第 2 维插一个长度 1 的轴，靠广播**同时加到 4 条残差流上**——同一个"下一 token 提示"注入每一条流；
- `h_proj` 对 4 条流共享同一组权重逐流线性变换；
- `enorm/hnorm` 发生在投影**之前**（先归一化再投影）。
- **v0.32.0 新增序列并行（SP）分支**（L128-L160）：`use_sequence_parallel_moe` 时对 `inputs_embeds` / `previous_hidden_states` 做 `sp_shard`，并在 `forward_context.is_padding` 上临时换成 SP padding mask，结束后恢复；`mtp_block` 输出再 `tensor_model_parallel_all_gather` 并截回 `full_num_tokens`。
- **v2 runner 差异**：`use_v2_model_runner=True` 时返回 `hidden_states.flatten(1)`（L165-L166），供 v2 speculator 直接消费拼平残差流；v1 回退路径仍返回 `(T, 4, 7168)`。

### 4.3 Position 0 Mask（L123）

```python
# masking inputs at position 0, as not needed by MTP
inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
```

position 0 是序列首 token，它没有"前一个 token 需要预测"，其嵌入对 MTP 无意义，置零防止污染。

### 4.4 mtp_block — draft DecoderLayer（L95-L101）

`mtp_block` 复用主模型的 `DeepseekV4DecoderLayer`，两个关键开关：

- `is_draft_layer=True`：`DeepseekV4MoE` 中直接 `if is_draft_layer: self.hash = False`（[model.py:345-352](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py#L345)），因此即便 draft 层号 < 3 也**不用 hash 路由**，走线性路由；
- `topk_indices_buffer=` 自建 buffer：draft 注意力在 `compress_ratios[61]=0` 下是 **dense**（无 Compressor/Indexer），该 buffer 仅在 is_v32 时按需分配以保持 DSA 模块接口一致（L82-L92）。
- FFN 仍是完整的 `DeepseekV4MoE`（384 专家/Top-6/共享专家），**不是稠密 MLP**。

输入 `(T, 4, 7168)` 与 `residual=None`，输出 `(hidden_states, residual)`，4 条残差流的结构与主层完全一致（hc_pre/hc_post 见文档 1/2）。

### 4.5 延迟 HC 头压缩与 hc_head 实现

Layer.forward **不调用** `hc_head`（L162-L163 调用处被注释），直接返回 4 流隐藏状态。原因：

1. 下一步 MTP（若有多层）需要完整的多流输入；
2. 只有出 logits 时才需要单流，压缩延迟到 `compute_logits`；
3. 避免"压缩→投影→再展开"的往返开销。

`hc_head`（[L169-L176](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/mtp.py#L169)）使用 `hc_norm` 模块，与主模型 `DeepseekV4Model.hc_head`（`model.py:949` 起）保持一致：

```python
def hc_head(self, x, hc_fn, hc_scale, hc_base):
    shape, dtype = x.size(), x.dtype          # (T, 4, 7168)
    x = x.flatten(1).float()                  # (T, 4*7168)，升 fp32
    x_norm = self.hc_norm(x)                  # ★ 无权重 RMSNorm，对【全宽 4*7168】归一化
    mixes = F.linear(x_norm, hc_fn)           # (T, 4)：每条目标流一个混合门控 logit
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps   # (T, 4)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)         # ★ 加权用的是【原始 x】
    return y.to(dtype)                        # (T, 7168)
```

| 步骤 | 细节 |
|------|------|
| 全宽归一化 | 门控 logit 由 `x_norm`（全宽 RMSNorm 后）算出，而非旧实现里手写 `rsqrt(x.square().mean())` 乘到线性结果上——数学等价但统一收敛为 `hc_norm` 模块（无权重、fp32） |
| 门控 | `sigmoid(温度缩放后的 mixes + base) + hc_eps`，4 个非负权重 |
| 加权 | 权重乘在**未经归一化的原始 x** 上（`x.view(shape)`），沿 `dim=1`（4 条流）求和压成单流 |

---

## 5. 输出头与完整数据流

### 5.1 SharedHead — 输出归一化与 LM Head（L41-L58）

```python
class SharedHead(nn.Module):
    def __init__(self, config, prefix, quant_config=None):
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(config.vocab_size, config.hidden_size,
                                   quant_config=quant_config, prefix=maybe_prefix(prefix, "head"))

    def forward(self, hidden_states):
        return self.norm(hidden_states)   # 只做 norm；head 矩阵在外部 logits_processor 中使用
```

- 结构上**每个 MTP 层各持有一个** `shared_head`（L94）；本配置只有 1 个 MTP 层，故只有一份；
- 命名为 "shared" 体现在 checkpoint 权重与主模型 LM head 同源（推理时通常 tie embeddings）；
- TP 下 `ParallelLMHead` 按词表维切分，与主模型 logits 路径一致。

---

### 5.2 完整 Forward 数据流

#### 5.2.1 单步草稿前向

```
输入: input_ids (T,)、positions (T,)、previous_hidden_states (T, 4*7168)、
      inputs_embeds (T, 7168)|None、spec_step_index
  1. 缺省嵌入: inputs_embeds = embed_tokens(input_ids)（Predictor L213-L214）
  2. 轮询选层: layers[str(spec_step_idx % 1)]
  3. mask position0 → enorm → e_proj → (T,1,7168)
     prev.view(T,4,7168) → hnorm → h_proj → (T,4,7168)
     两者广播相加得 (T,4,7168)                           [Layer L123-L146]
  4. mtp_block（dense DSA 注意力 + MoE，含 hc_pre/hc_post）→ (T,4,7168)
输出: 4 条残差流隐藏状态（不压缩；v2 runner 下拼平）
```

#### 5.2.2 compute_logits（Predictor L224-L236）

```python
mtp_layer = self.layers[str(spec_step_idx % self.num_mtp_layers)]
hidden_states = hidden_states.view(-1, mtp_layer.hc_mult, mtp_layer.config.hidden_size)  # L231
hidden_states = mtp_layer.hc_head(hidden_states,
                                  mtp_layer.hc_head_fn, mtp_layer.hc_head_scale,
                                  mtp_layer.hc_head_base)                                # (T, 7168)
logits = self.logits_processor(mtp_layer.shared_head.head,
                               mtp_layer.shared_head(hidden_states))                    # (T, 129280)
```

即 **view 回 4 流 → hc_head 压单流 → shared_head.norm → 分片 LM head**。注意进入 `hc_head` 前显式 view：若上游传入的是拼平的 `(T, 4*7168)`，这里恢复 `(T, 4, 7168)`。

---

## 6. 权重加载与训练目标

### 6.1 权重加载与映射（load_weights，L292-L521）

#### 6.1.1 总体流程

1. `get_spec_layer_idx_from_weight_name(config, name)`（[model.py:222-225](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py#L222)：`mtp.` 开头返回 0）筛选属于 MTP 的权重，非 MTP 权重直接 `continue`；并断言名字含 `mtp.0.`（L334）；
2. fp8 特殊重命名：`embed.weight → mtp.0.emb.tok_emb.weight`、`head.weight → mtp.0.head.weight`（L323-L328）；
3. 三段式前缀改写（L335-L340）：
   - 含 `.emb.tok_emb.`：`mtp.0.` → `model.`（预测器顶层嵌入）；
   - `no_mtp_block_in_name(name)` 命中（投影/HC 头/norm/head 等）：`mtp.0.` → `model.layers.0.`；
   - 其余：`mtp.0.` → `model.layers.0.mtp_block.`（draft block 内部权重）；
4. 逐片段改名后经 `stacked_params_mapping`（w1/w3 → gate_up_proj 融合，L295-L298）、`expert_params_mapping`（384 专家，EPLB/mix_placement 时含共享专家，L300-L308）或默认 loader 加载。

#### 6.1.2 权重名片段映射表

| checkpoint 片段 | 加载名片段 | 行号 |
|----------------|-----------|------|
| `.w1.` / `.w2.` / `.w3.` | `.gate_proj.` / `.down_proj.` / `.up_proj.` | L342-L347 |
| `.scale` 结尾 | `.weight_scale` | L349-L350 |
| `.head.` | `.shared_head.head.` | L352-L353 |
| `.norm.` | `.shared_head.norm.` | L355-L356 |
| `.emb.tok_emb.` | `.embed_tokens.` | L358-L359 |
| `.attn.`（且不含 `self_attn`） | `.self_attn.` | L361-L362 |
| `.ffn.` / `.ffn_norm.` / `.attn_norm.` | `.mlp.` / `.post_attention_layernorm.` / `.input_layernorm.` | L363-L368 |
| `.gate.bias` | `.gate.e_score_correction_bias` | L370-L371 |

#### 6.1.3 `no_mtp_block_in_name`（L553-L566）

下列名字直接挂在 `model.layers.0.*` 下，**不加** `.mtp_block.`：

```python
[".hc_head_fn", ".hc_head_base", ".hc_head_scale",
 ".e_proj.", ".h_proj.", ".enorm.", ".hnorm.",
 ".norm.", ".head.", ".emb.tok_emb."]
```

另有 `_rewrite_spec_layer_name`（L523-L551）服务通用多层命名：`embed_tokens/enorm/hnorm/eh_proj/shared_head` 视为预测层直属权重，其余加 `.mtp_block.`，嵌入等共享权重提升到 `model.` 顶层。

#### 6.1.4 Attention Sink 的加载（L373-L385）

```python
if "sink" in name:
    param = params_dict[name]
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
    weight_loader(param, narrow_weight)               # 普通 TP：只切本 rank 的头段
    loaded_params.add(name)
    continue
```

v0.32.0 已移除旧的 `enable_dsa_cp()` 分支：sink 统一按 TP 头段 `narrow`，并改走参数的 `weight_loader`（layerwise reload 时 meta device 只回放 loader 写入）。

#### 6.1.5 mix_placement 融合共享专家（L387-L455）

`ascend_config.mix_placement=True` 时，checkpoint 中合并的共享专家权重被均分为 `n_shared_experts` 份，改名到 `mlp.experts.{384+j}.*`，再借 `expert_params_mapping` 以专家身份加载：`down_proj` 沿 dim 1 切，gate/up 沿 dim 0 切（L428-L449）。

---

### 6.2 训练目标：预压缩残差流缓冲

主模型 `get_mtp_target_hidden_states`（[model.py:1180-1184](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py#L1180)）返回 `self.model._mtp_hidden_buffer`：

```
Pre-hc_head residual stream buffer
形状: (max_num_batched_tokens, hc_mult * hidden_size) = (…, 4*7168)
生命周期: 主模型 forward 填充，每个 target step 后有效
```

缓冲区只在 `is_last_rank` 且启用了 spec（`use_eagle()` 或 `uses_draft_model()`）时分配（`model.py:931-944`）——避免无草稿模型时长期占用 `max_num_batched_tokens * hc_dim` 显存；绑定在编译前创建，保证 FULL graph 重放写入的是同一张量。

即 MTP 训练时对齐的目标是**主模型未经 hc_head 压缩的 4 流状态**——与推理期 MTP 消费的 `previous_hidden_states` 完全同构，保证训练/推理路径一致。

---

## 7. 总结

V4 的 MTP 是一个复用主模型 `DeepseekV4DecoderLayer` 的单预测层草稿模型：把"下一 token 嵌入（enorm+e_proj）"广播加到"上一步 4 条并行残差流（view+hnorm+h_proj）"上，送入 **dense 注意力 + 完整 MoE（无 hash）** 的 draft block；全程保持 4 流结构、只在 `compute_logits` 时用 **hc_norm + sigmoid 门控的 hc_head** 压回单流出 logits；DSSA 的 TopK 索引缓冲区由 MTP 层**自建**；权重加载通过 `mtp.0.*` 三段式前缀改写、w1/w2/w3 与量化 scale 重命名、专家映射与 attn_sink 的 TP 切片完成。v0.32.0 还补齐了 **SP（序列并行）分支**与 **v2 runner 的拼平 hidden 返回**，并让异步调度下的草稿完全留在 worker 侧（`ModelRunnerOutput` 不再携带 `spec_token_ids`）。
