# DeepSeek V4 MTP详解


> **代码版本冻结声明**：本文基于 **vllm-ascend v0.26.0rc** 源码整理。
> **统一模型配置**（W4A8 版，MTP 相关行）：`num_nextn_predict_layers=1`（1 个 MTP 预测层）、`hidden_size=7168`、`hc_mult=4`、`vocab_size=129280`；MTP 层在 `compress_ratios` 数组中对应下标 61（值 **0=dense**，即 draft 注意力不带 Compressor/Indexer），但 FFN 仍是 MoE（384 专家、Top-6，hash 路由关闭）。

---

## 1. vLLM step总览


### 1.1 异步调度（async scheduling）结合源码详解

> **一句话**：默认的同步调度里，EngineCore 每步都要"调度 → 前向 → **阻塞等采样结果** → 回写"，采样期间 CPU 空转、GPU 空泡；**异步调度**把"前向的 Future"和"采样的 Future"塞进一个 `batch_queue` 先排队，CPU 立刻去调度**下一批**，等采样结果真回来了再回写——用**多步在途**填满 GPU。代价是：`Scheduler` 在**草稿 token 还没算出来**的时候就得先排下一步，于是引入 **placeholder（占位）** 机制与 `spec_token_ids` 的**延迟回填**。

#### 1.1.1 开关与调度器选择

配置项 `async_scheduling: bool | None = None`（vLLM `config/scheduler.py:158`）——**None 表示自动**。它决定启用哪个 Scheduler 类（`config/scheduler.py:180-188`）：为真则用 `AsyncScheduler`，否则用同步的 `Scheduler`。EngineCore 侧在构造时读入（`core.py:227`），并据此决定 `step_fn` 的走法（`core.py:224-225`）：

```python
# core.py:224-227
self.step_fn = self.step if self.batch_queue is None else self.step_with_batch_queue
self.async_scheduling = vllm_config.scheduler_config.async_scheduling
```

> **注意**：`step_fn` 选的是 `step` 还是 `step_with_batch_queue`，**取决于 `batch_queue` 是否存在**——而 `batch_queue` 在 `batch_queue_size = max_concurrent_batches > 1` 时才建（`core.py:199-205`），这正是 **PP（流水线并行）+ 异步** 的场景。也就是说"异步"与"批队列"是**两个正交但常配对**的机制：异步解决"采样等待空泡"，批队列解决"PP 气泡"，二者叠加即 `step_with_batch_queue` 里同时存在多个在途 Future。

#### 1.1.2 主循环骨架：run_busy_loop → _process_engine_step → step_fn

```python
# core.py:1358-1366
def run_busy_loop(self):
    while self._handle_shutdown():
        self._process_input_queue()     # ① 收请求 / abort，直到有活干
        self._process_engine_step()     # ② 走一步（step_fn）

# core.py:1399-1416
def _process_engine_step(self) -> bool:
    outputs, model_executed = self.step_fn()        # ← 同步=step / 异步+批队列=step_with_batch_queue
    for output in outputs.items() if outputs else ():
        self.output_queue.put_nowait(output)        # 结果推给前端
    self.post_step(model_executed)                  # 草稿回写钩子（见 1.1.4）
    if not model_executed and self.scheduler.has_requests():
        time.sleep(0.001)                           # 无执行但有活：让出 GIL
    return model_executed
```

三者关系：**`run_busy_loop` 是外壳，`_process_engine_step` 是"走一步并交付输出"，`step_fn` 是真正的一步**——同步时指向 `step`（`core.py:576`），异步 + 批队列时指向 `step_with_batch_queue`（`core.py:617`）。判活条件是 `has_work()`（`core.py:1346-1352`）：`engines_running or scheduler.has_requests() or bool(self.batch_queue)`——**队列里还有在途批次也算有活**，这是异步能不空转的关键。

#### 1.1.3 同步 vs 异步：两次 RPC 的推进方式不同

**同步路径**（`EngineCore.step`，`core.py:576-606`）——一条直线，采样后**原地阻塞**取结果：

```python
scheduler_output = self.scheduler.schedule(...)                 # :587
future = self.model_executor.execute_model(..., non_block=True) # :588 前向 Future，立刻返回
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)  # :589
model_output = future.result()                                  # :594 ★阻塞：等前向完
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)   # :596 第二次 RPC，再等
engine_core_outputs = self.scheduler.update_from_output(...)    # :601 回写
```

**异步路径**（`step_with_batch_queue`，`core.py:617-731`）——把 Future **入队**，采样也 `non_block=True`，本轮**不阻塞**：

```python
exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)  # :647
...
future = self.model_executor.sample_tokens(grammar_output, non_block=True)         # :663 ★采样不阻塞
...
batch_queue.appendleft((future, scheduler_output, exec_future))                    # :673 三元组入队
if len(batch_queue) < self.batch_queue_size and (model_executed or ...):
    return None, model_executed        # :679 ★队列没满就立刻返回，CPU 转去排下一批
# ---- 队列满/没活可排时，才回头处理最老的批次 ----
future, scheduler_output, exec_model_fut = batch_queue.pop()                       # :688
model_output = future.result()                                                         # :693 此时才阻塞
engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)  # :703
```

对照可见：同步是"**排一批 → 等等等 → 回写**"，异步是"**排一批入队 → 立刻排下一批 → …（队列满）→ 取出最老的结算**"。`batch_queue` 里每个元素是 `(采样Future, 该批的SchedulerOutput, 前向Future)`，**多步在途**让采样/前向的 Device 时间与 CPU 调度时间重叠。

#### 1.1.4 草稿 token 的"延迟回填"——异步最核心的差异

这是本题（MTP）最相关的一点。**同步**时，`step()` 阻塞到采样完，`post_step` 即可同步把草稿取回来写进 scheduler：

```python
# core.py:608-616
def post_step(self, model_executed: bool) -> None:
    # 异步调度下无法提前拿到草稿 token id，草稿在 worker 进程内更新，
    # 这里不需要（也不能）回写草稿。
    if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
        draft_token_ids = self.model_executor.take_draft_token_ids()   # :613
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)     # :615
```

**异步**时 `not self.async_scheduling` 为假 → **整个块被跳过**。原因写在注释里：此时 scheduler 已经跑去排**下一步**了，而草稿 token 是**上一步**采样 + 草稿前向才产出的，CPU 侧**还没拿到**——所以草稿改由 **worker 随 `ModelRunnerOutput.spec_token_ids` 出参带回**，在 `update_from_output` 里落库。

配套地，`AsyncScheduler` 在 `schedule()` 之后用 **placeholder** 预占位置（vLLM `v1/core/sched/async_scheduler.py:19-49`）：

```python
# async_scheduler.py:19-44
def _update_after_schedule(self, scheduler_output):
    super()._update_after_schedule(scheduler_output)
    ...
    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests[req_id]
        if request.is_prefill_chunk:
            continue
        ...
        cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
        request.num_output_placeholders += (self.num_sampled_tokens_per_step + cur_num_spec_tokens)  # :39-41
        # 先占位 -1，真实草稿 id 稍后在 worker 进程里回填
        request.spec_token_ids = self._spec_token_placeholders        # :44 ★placeholder
```

`_spec_token_placeholders = [-1] * num_spec_tokens`（`async_scheduler.py:16`）就是**用 `-1` 占住 K 个草稿位**，保证 decode 批形状（`1+K`）不塌，等真草稿回来再覆盖。这正是 1.2 节里"新 decode 请求用 `[-1]*K` 补齐"的**异步版本解释**——同步版靠 `Scheduler` 自己补（`scheduler.py:846-859`），异步版靠 `AsyncScheduler` 的 placeholder + 延迟回填。

`num_output_placeholders` 的账在 `_update_request_with_output`（`async_scheduler.py:51-75`）里回冲：每收到一批真 token 就 `num_output_placeholders -= len(new_token_ids)`（`:67`），并对**非抢占**请求补 `cache_blocks`（`:71-74`）。调度时若 `num_output_placeholders > 0` 还要防多排一步（`scheduler.py:474-488`）。

#### 1.1.5 worker 侧出参：spec_token_ids 怎么随 ModelRunnerOutput 回来

vllm-ascend 的 `NPUModelRunner.sample_tokens`（`model_runner_v1.py:2231`）在草稿前向之后，把草稿 id 组进 `ModelRunnerOutput.spec_token_ids`（`:2365-2385`）：

```python
# model_runner_v1.py:2365-2385
draft_token_ids = self._draft_token_ids if use_pp_spec_decode else None
if draft_token_ids is not None:
    ...
    draft_by_req_id = dict(zip(draft_req_ids, draft_ids_list))       # :2375
    output_spec_token_ids = [draft_by_req_id.get(req_id, []) for req_id in req_ids_output_copy]  # :2376-2379
model_runner_output = ModelRunnerOutput(
    ...
    spec_token_ids=output_spec_token_ids,                            # :2385 ★草稿随出参带回
    ...
)
```

异步 + spec 下，D2H 拷贝被**故意跳过**由 `_copy_draft_token_ids_to_cpu` 处理（`model_runner_v1.py:1767-1777`）：当 `use_async_scheduling` 且没有结构化输出 / 无 output_token_ids / 无 PP 时直接 `return`（`:1772-1777`）——**草稿留在 worker 进程内、不落 CPU**，正好对应 `post_step` 被跳过、由 worker 出参回传的链路。相关开关与缓冲区在初始化时就绪：`model_runner_v1.py:413-432`（`use_async_scheduling and num_spec_tokens` 时建 pinned buffer / 拷贝流）、`:1141-1146`（`use_async_spec_decode`）。

> **vllm-ascend 侧的 `use_async_scheduling`**：与 vLLM 主仓 `async_scheduling` 语义一致，`model_runner_v1.py` 里出现约 20 处分支（如 `:764`、`:828`、`:959`、`:1113`、`:2414`、`:2532`、`:2590`），主要集中在三件事：① 采样后是否同步 D2H（`:2414`/`:2532`）；② bookkeeping 的同步/异步分支（`:828`、`:1113`）；③ spec decode 的缓冲与事件（`:413-432`、`:1141-1146`）。

#### 1.1.6 一句话对照表

| 维度 | 同步（`step`） | 异步（`step_with_batch_queue` / `AsyncScheduler`） |
|------|------|------|
| 调度器类 | `Scheduler`（`scheduler.py`） | `AsyncScheduler`（`async_scheduler.py:12`） |
| 采样 RPC | `sample_tokens(...)` 阻塞（`core.py:596`） | `sample_tokens(..., non_block=True)` 入队（`core.py:663`/`:673`） |
| 多步在途 | 无 | `batch_queue` 存 `(sample_fut, sched_out, exec_fut)`（`core.py:688`） |
| 草稿回写 | `post_step → take_draft_token_ids`（`core.py:613-615`） | **跳过**（`core.py:612`），随 `ModelRunnerOutput.spec_token_ids` 带回（`model_runner_v1.py:2385`） |
| 草稿占位 | `[-1]*K` 由 `Scheduler` 补齐 | `AsyncScheduler` placeholder `[-1]*K`（`async_scheduler.py:16`/`:44`） |
| 延迟账本 | 无 | `num_output_placeholders` 预加/回冲（`async_scheduler.py:39`/`:67`） |
| 产物一致性 | 一步一结 | 靠 placeholder + 出参回填保证 `1+K` 形状与最终 logits 一致 |

---

### 1.2 EngineCore step 调用栈（通用，不涉及模型内部）

一轮 step ＝「① 调度 → ② 目标前向（非阻塞发起）→ ③ 验证采样（第二次 RPC，内嵌 ④ 草稿前向）→ ⑤ 结算回写＋草稿回写」。目标前向以 `non_block=True` 发 RPC（core.py:588）立刻拿 Future；`future.result()` 返回 `None` 即"前向已就绪，可采样"信号（:594），EngineCore 再发第二次 RPC `sample_tokens`（:596，携带 grammar bitmask 等调度侧产物）。两次 RPC 之间的中间态（logits、hidden_states、spec 元数据）暂存在 worker 侧 `ExecuteModelState`（model_runner_v1.py:2208）。④ 只展开到 MTP 草稿模型入口，层内结构见第 3～5 章。

```
run_busy_loop → step_fn() ⇒ EngineCore.step()               # core.py:1358/:1403 → :576
│
├─ ① 调度 Scheduler.schedule()                             # core.py:587 → scheduler.py:425
│     · D(t-1) 装车 scheduled_spec_decode_tokens            #   :634，随后清空 spec_token_ids :638
│     · 新 decode 请求用 [-1]*K 补齐到 1+K（prefill 混批不补）#   :846-859 / :1032-1035
│
├─ ② 目标前向 Executor.execute_model(non_block=True)        # core.py:588 ⇢ collective_rpc
│     └ Worker.execute_model                                #   worker.py:653
│         └ NPUModelRunner.execute_model                     #     model_runner_v1.py:1799
│              · _update_states / _prepare_inputs             #       :1888 / :1940（spec 分支 :1277-1310）
│              · _model_forward：一遍前向吃下本步全部 token     #       :2155 → :2698
│              · logits = compute_logits(hidden[logits_indices])  #   :2184-2185
│              · 暂存 ExecuteModelState → return None        #       :2208-2228（可采样信号）
├─ future.result() → None                                    # core.py:594
│
├─ ③ 验证采样 Executor.sample_tokens(grammar_output)         # core.py:596 ⇢ collective_rpc
│     └ Worker.sample_tokens                                #   worker.py:720
│         └ NPUModelRunner.sample_tokens                      #     model_runner_v1.py:2231
│              · 解包并清空 execute_model_state                #       :2253-2268
│              · _sample：prefill→Sampler / decode→RejectionSampler    #   :2280 → :2464 / :2489
│                  接受 / 拒绝 / 重采样 + bonus token          #     rejection_sampler.py:150
│              │
│              ├─ ④ 草稿前向 propose_draft_token_ids 闭包      # :2291 · 调用 :2353（PP 特例提前 :2327）
│              │    └ NPUModelRunner.propose_draft_token_ids  #   :1512
│              │         · MTP：改吃 _mtp_hidden_buffer      #     :1657-1661
│              │         └ drafter._propose                  #       :1696 → llm_base_proposer.py:1015
│              │              · set_inputs_first_pass 右移一位  #         :1653-1660
│              │              · MTP forward（K>1 时逐圈递推）  #         :1368 / 循环 :1508 → 每圈 :1580
│              │              · compute_draft_token_ids → argmax  #      :1320
│              │    └ _copy_draft_token_ids_to_cpu           #   :2306
│              · _bookkeeping_sync：汇成本轮有效 token          # :2336
│              · ModelRunnerOutput(...)；step() return        # :2381 / :2412 / core.py:606
│
├─ ⑤ 结算回写 Scheduler.update_from_output()                 # core.py:601 → scheduler.py:1589
│     · num_accepted / num_computed_tokens -= num_rejected   #   :1687-1695
└─ ⑤ 草稿回写 post_step(model_executed)                      # core.py:1408 → :608
       · take_draft_token_ids                                #   :613 → worker.py:1109
       · update_draft_token_ids                              #   :615 → scheduler.py:2057
           chunk 未完成 → 草稿丢弃 :2067-2071；否则写回 :2077（供下一轮 ①）
```

### 1.2 各阶段职责

表格按 ①→⑤ 展开：**第 1 列**阶段（含锚点）、**第 2 列**职责、**第 3/4 列** prefill、decode 两种批型的**输入→输出 shape**。先约定符号与 DSV4 特有的隐态交接：

- **符号**：`B`＝本步批内请求数；`K`＝num_speculative_tokens（每请求草稿数）；`L_i`＝请求 i 本步输入 token 数、`T=ΣL_i`；`V=129280`（词表）；`H=7168`；`d=4H=28672`（4 条残差流拼平宽度）；`D(t)`＝第 t 轮产出的草稿，每请求 `(K,)`。
- **DSV4 隐态交接**：目标模型 forward 末尾把**未压缩** 4 流残差 `(T, d)` 写入 `_mtp_hidden_buffer`（deepseek_v4.py:1193/:1196），并把 hc_head 压缩＋norm 后的 `(T, H)` 作为返回值（:1205-1210，供 ③ 采样）；④ 的 MTP 草稿经 `get_mtp_target_hidden_states`（model_runner_v1.py:1657-1661）改吃 4 流缓冲——② 的 `logits_indices` 抠位与 ④ 的 hidden 选择因此彼此独立。
- **批型**：prefill 列含 chunked prefill（含最后一块）；decode 列指稳态 decode 批（首个 decode 步的草稿来源见 ⑤ 行）。

| 阶段（锚点） | 职责 | prefill（含 chunk）输入 → 输出 | decode（稳态）输入 → 输出 |
|------|------|------|------|
| **① 调度**<br>`Scheduler.schedule`<br>core.py:587 → scheduler.py:425 | 编批、分配 KV 块；把 ⑤ 上一步写回的 `D(t-1)` 装进 `scheduled_spec_decode_tokens`（:634）并**清空** `request.spec_token_ids`（:638）；新 decode 请求用 `[-1]*K` 占位补齐到 `1+K`，保住 decode 统一形状（:846-859、:1032-1035；prefill 混批时不补齐） | **无草稿**（表为空）<br>入：各请求 `num_computed_tokens` / prompt 状态<br>出：`SchedulerOutput`：`num_scheduled_tokens[rid]`＝本块长度（`T=Σ` 各块），`scheduled_spec_decode_tokens={}` | **有草稿**（首个 decode 步＝最后一块 prefill ④ 的产出，见 ⑤）<br>入：`request.spec_token_ids=D(t-1)`，每请求 `(K,)`<br>出：`SchedulerOutput`：`num_scheduled_tokens[rid]=1+K`、`scheduled_spec_decode_tokens[rid]=(K,)`（新请求记 `[-1]*K`） |
| **② 目标前向**<br>`execute_model(non_block=True)`<br>core.py:588 → worker.py:653 → model_runner_v1.py:1799<br>`_model_forward` :2155→:2698 | 组 batch / 注意力元数据后**一遍**前向吃下本步全部 token；按 `logits_indices` 抠采样位算 logits；暂存 `ExecuteModelState` 返回 `None`（暂存 :2208、返回 None :2228＝"可采样"信号）；副作用：4 流残差写 `_mtp_hidden_buffer`（deepseek_v4.py:1193） | 入：`input_ids/positions (T,)`，`T=Σ` 各请求本块 prompt 长度（chunk 取本块）<br>出：模型返回 `(T, H)`＋副作用 `_mtp_hidden_buffer (T, d)`；`logits_indices`＝每请求末位 → `sample_hidden_states (B, H)` → **logits `(B, V)`**（`spec_decode_metadata=None`） | 入：`input_ids/positions (T,)`，`T=Σ(1+K_i)`（统一形状时 `=B(1+K)`；每请求 `[t_now, D(t-1)]`，`-1` 占位请求仍按 `1+K` 计）<br>出：`sample_hidden_states (Σ(1+K_i), H)` → **logits `(Σ(1+K_i), V)`**＝每请求 K 个验证位＋1 个 bonus 位 |
| **③ 验证与采样**<br>`sample_tokens(grammar_output)`<br>core.py:596 → worker.py:720 → :2231 → `_sample` :2464 | 解包暂存 logits（:2253-2268，可叠 grammar bitmask）；prefill 走普通 `Sampler`（:2468）、decode 走 `AscendRejectionSampler` 逐 token 接受/拒绝/重采样＋bonus（:2489 → rejection_sampler.py:150）；`_bookkeeping_sync`（:2336）汇成有效 token 表 | 入：logits `(B, V)`（`spec_decode_metadata=None`）<br>出：`sampled_token_ids (B,)`＝每请求 1 个新 token，即**首个输出 token**（未完成 chunk 多采的位在 ⑤ 丢弃，:1266-1272） | 入：logits `(B(1+K), V)`＋`SpecDecodeMetadata`（`draft_token_ids (B·K,)`、target/bonus 位索引）<br>出：`sampled_token_ids (B, 1+K)`（接受位＝真 id、拒绝位＝`-1`、末位 bonus）；簿记后每请求留 **`1~(1+K)` 个有效新 token**＝接受草稿＋1 个新锚点 |
| **④ 草稿前向**（嵌在 ③ RPC 内）<br>闭包 :2291 → `propose_draft_token_ids` :1512 → `drafter._propose` :1696<br>→ llm_base_proposer.py:1015 | 驱动 **MTP 层**前向并写**草稿层自己的 KV**：decode 用刚采出的 token 逐圈递推 K 次得 `D(t)`；prefill 走 **draft prefill** 填 MTP 层 KV、并在每请求末位 argmax 出首个 decode 步草稿（K>1 时逐圈递推）；MTP 的 hidden 经 :1657-1661 换成 4 流缓冲 | **draft prefill**（:1663-1672）<br>入：`target_token_ids (T,)`＝**整段 prompt**、`target_hidden_states (T, d)`、`next_token_ids (B,)`<br>`set_inputs_first_pass` 右移一位（:1653-1660）→ 草稿输入 `(T,)`（每请求 `[p₂…p_L, s]`，配 `(h_j, e_{j+1})` 对）<br>K 圈递推（首圈 `(T,)`，后续每圈 `(B,)`）＋经 `block_table[kv_cache_gid]` 写 MTP 层 KV<br>出：**`draft_token_ids (B, K)=D(t)`**＝首个 decode 步草稿（去留由 ⑤ 决定） | **增量**（:1673-1694，已剔除上一轮拒绝位）<br>入：`target_token_ids (B,)`＝每请求 1 个刚采出的 token、`target_hidden_states (B, d)`、`next_token_ids (B,)`<br>K 圈递推（每圈 forward `(B,)` token＋`(B, d)` hidden → argmax `(B,)`）<br>出：**`draft_token_ids (B, K)=D(t)`**，供下一 decode 步 ② 验证 |
| **⑤ 结算回写＋草稿回写**<br>`update_from_output`<br>core.py:601 → scheduler.py:1589；<br>`post_step` core.py:608-616<br>→ `update_draft_token_ids` :2057 | 按接受数回退进度、token 写回请求并产出 `EngineCoreOutputs`；把 `D(t)` 写回 `request.spec_token_ids`——`is_prefill_chunk` 为真（chunk 未跑完）则丢弃草稿、否则写回（:2067/:2077，标志每步重算 :1254-1256） | 入：`sampled_token_ids (B,)`<br>**无拒绝结算**：最后一块的采样 token append 为首个输出 token（未完成 chunk 的位丢弃）<br>草稿去留：chunk 未完 → `D(t)` **丢弃**（:2067-2071，MTP KV 照写）；最后一块已跑完（标志已翻 False）→ **`request.spec_token_ids=(K,)`（:2077）＝首个 decode 步输入** | 入：每请求 `1~(1+K)` 个有效新 token＋`(K,)` 草稿<br>`num_accepted=max(len(generated)−1, 0)`（:1687，`num_sampled=1`：scheduler.py:122-124）→ `num_computed_tokens −= K−num_accepted`（:1694-1695，被拒位下轮重算）<br>出：**`request.spec_token_ids=D(t)`（:2077）**（非 async 经 `take_draft_token_ids` worker.py:1109；async 时随 `ModelRunnerOutput.spec_token_ids` 出参带回 :2376-2385）＝下一步 ① 的草稿源 |

**MTP 层的 prompt KV 并不缺**——它正是在 ④ 的 prefill 分支（draft prefill）里填的：draft prefill 消费整段 prompt 的 4 流隐藏状态（② 副作用写入 `_mtp_hidden_buffer`，见上文"DSV4 隐态交接"），经 `input_batch.block_table[kv_cache_gid]` 把 MTP 层自己的 KV 一并写满。

**草稿从哪来、到哪去（错开一步）**：稳态 decode 下，② 消费 ⑤ 上一步写回的 `D(t-1)`，④ 生产下一轮的 `D(t)`，两者永不重合。**首个 decode 步的草稿有明确来源**：最后一块 prefill 步的 ④ 在每请求末位 argmax＋K 圈递推出 `(B, K)`，且该请求的 `is_prefill_chunk` 已在调度簿记中翻 False（scheduler.py:1254-1256：`num_computed_tokens` 追平 `num_tokens` 即非 chunk），⑤ 正常写回（:2077）——这批草稿**不丢弃**，直接成为首个 decode 步的输入；**只有还没跑完的 prefill chunk 请求，草稿才被置空丢弃**（:2067-2071）。

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
DeepSeekV4MTP (入口类, L202; @support_torch_compile)
└── model: DeepSeekMultiTokenPredictor (L141)
    ├── embed_tokens: VocabParallelEmbedding(129280, 7168)     — 词嵌入
    ├── logits_processor: LogitsProcessor                       — logits 处理
    └── layers: ModuleDict {"0": DeepSeekMultiTokenPredictorLayer (L56), ...}
        ├── e_proj / h_proj: ReplicatedLinear(7168 → 7168)      — 双路投影（走 quant_config）
        ├── enorm / hnorm: RMSNorm(7168)                        — 嵌入 / 残差流归一化
        ├── shared_head: SharedHead (L36)                       — 每层各一份
        │   ├── norm: RMSNorm(7168)
        │   └── head: ParallelLMHead(129280, 7168)
        ├── mtp_block: DeepseekV2DecoderLayer(is_draft_layer=True, L89)
        │   └── dense DSA 注意力 + DeepseekV4MoE（无 hash）
        ├── hc_head_fn/base/scale: Parameter (fp32)             — HC 头门控参数
        ├── hc_norm: RMSNorm(4*7168, has_weight=False, fp32)    — 无权重全宽归一化
        └── topk_indices_buffer: (max_num_batched_tokens, 1024) int32
            └─ is_v32 时【本层自建】，不与主模型共享（L77-L86）
```

### 2.3 与主模型的交互

| 交互点 | 说明 |
|--------|------|
| `_mtp_hidden_buffer` | 主模型 forward 写入的**预 hc_head 残差流缓冲**，形状 `(max_num_batched_tokens, hc_mult*hidden_size)`，每个 target step 后有效（主模型 [L1336-L1340](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L1336)）。MTP 的 `previous_hidden_states` 即来源于此 |
| `topk_indices_buffer` | **v0.26 中主模型与 MTP 不共享**：主模型各层共用一份模型级 buffer；MTP 层在 `__init__` 自建一份传给自己的 `mtp_block`（L79-L93）。原因见文档 5 §6（spec decode 下 impl 层引用会失效） |
| `spec_step_idx` / `spec_step_index` | 当前推测步号；预测器用它对 `num_mtp_layers` 取模轮询（L177） |
| 词表/embedding | MTP 自建 `embed_tokens`，checkpoint 中通过权重映射加载与主模型同源的词嵌入 |

---

## 3. 入口类与预测器主体

### 3.1 DeepSeekV4MTP — 入口类（L202-L251）

继承 `nn.Module + SupportsPP + DeepseekV2MixtureOfExperts`。构造时创建 `DeepSeekMultiTokenPredictor`（prefix 自动加 `mtp`）并调用 `set_moe_parameters()`（L209）。

#### 3.1.1 set_moe_parameters（L211-L229）

```python
self.expert_weights = []
self.num_expert_groups = getattr(self.config, "n_group", 1)   # V4 配置无 n_group → 1
for layer in self.model.layers.values():
    if isinstance(layer, PPMissingLayer):    # PP 下不在本 rank 的层跳过
        continue
    layer = layer.mtp_block                  # DeepseekV2DecoderLayer
    if isinstance(layer.mlp, DeepseekV4MoE):
        example_moe = layer.mlp              # 注释：取最后的 MoE，前面可能是 dense 层
        self.moe_mlp_layers.append(layer.mlp)
        self.moe_layers.append(layer.mlp.experts)
self.extract_moe_parameters(example_moe)
```

作用与主模型的 MoE mixin 一致：注册 EPLB/EP 通信组、抽取冗余专家等超参，使 draft 层的 384 个专家同样纳入专家并行调度。

#### 3.1.2 forward / compute_logits / embed_input_ids

三个方法都是纯转发（L231-L251）：`forward → model(...)`、`compute_logits → model.compute_logits(...)`、`embed_input_ids → model.embed_input_ids(...)`。

---

### 3.2 DeepSeekMultiTokenPredictor — 预测器主体（L141-L198）

#### 3.2.1 核心成员

| 成员 | 说明 |
|------|------|
| `mtp_start_layer_idx` | `config.num_hidden_layers` = **61**（L145），MTP 配置数组下标从此开始 |
| `num_mtp_layers` | `getattr(config, "num_nextn_predict_layers", 1)` → **1**（L146） |
| `layers` | ModuleDict，key 为 `"0"`…`"N-1"`（L149-L157） |
| `embed_tokens` | `VocabParallelEmbedding(vocab_size, hidden_size)`（L158） |
| `logits_processor` | `LogitsProcessor(vocab_size)`（L162） |

#### 3.2.2 轮询调度（forward，L167-L184）

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

## 4. DeepSeekMultiTokenPredictorLayer — 单层实现（L56-L138）

### 4.1 核心成员

| 成员 | 形状 | 说明 |
|------|------|------|
| `e_proj` | `7168 → 7168` | ReplicatedLinear，**走 quant_config（W4A8 包中量化）**，嵌入投影 |
| `h_proj` | `7168 → 7168` | ReplicatedLinear，同上，残差流投影（逐流作用） |
| `enorm` / `hnorm` | RMSNorm(7168) | 嵌入归一化 / 上一步残差流归一化 |
| `shared_head` | SharedHead | **每个 MTP 层各持一份**（L88，本配置仅 1 份） |
| `mtp_block` | DeepseekV2DecoderLayer | 完整 draft block，传 `is_draft_layer=True` 与自建 buffer（L89-L95） |
| `is_v32` | bool | `hasattr(config, "index_topk")`（L76），V4 恒为 True |
| `hc_eps` / `hc_mult` | 1e-6 / 4 | 来自配置 |
| `hc_head_fn` | `(4, 4*7168)` fp32 | HC 头门控矩阵（每个目标流一行） |
| `hc_head_base` | `(4,)` fp32 | 门控偏置 |
| `hc_head_scale` | `(1,)` fp32 | 门控温度缩放 |
| `hc_norm` | RMSNorm(`4*7168`, `has_weight=False`, dtype=**fp32**) | **v0.26 新增**：对拼平的全宽残差流做无权重 RMSNorm（L103） |

### 4.2 投影融合（forward，L107-L129）

MTP 输入由两路融合：

1. **下一 token 嵌入** `inputs_embeds`（`(T, 7168)`）：先 mask，再 `enorm`；
2. **上一步 4 条残差流** `previous_hidden_states`：先 view 回 `(T, 4, 7168)`，再 `hnorm`（对最后一维归一化）。

```
inputs_embeds (T, 7168)
    │ where(position==0 → 0)  [L117]
    │ enorm                    [L118]
    │ e_proj                   [L122]
    ▼ (T, 7168) ── unsqueeze(-2) ──▶ (T, 1, 7168)
                                        │ 广播相加
previous_hidden_states (T, 4*7168)     │
    │ view(-1, 4, 7168)        [L119]  ▼
    │ hnorm                     [L120] ▶ (T, 4, 7168)  hidden_states
    │ h_proj（逐流 7168→7168）  [L122]
    ▼ (T, 4, 7168)
```

```python
# L117-L124（顺序：先两路各自归一化，再投影相加）
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

### 4.3 Position 0 Mask（L117）

```python
# masking inputs at position 0, as not needed by MTP
inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
```

position 0 是序列首 token，它没有"前一个 token 需要预测"，其嵌入对 MTP 无意义，置零防止污染。

### 4.4 mtp_block — draft DecoderLayer（L89-L95）

`mtp_block` 复用主模型的 `DeepseekV2DecoderLayer`，两个关键开关：

- `is_draft_layer=True`：`DeepseekV4MoE` 中 hash 条件含 `and not is_draft_layer`（主模型 L421），因此即便 draft 层号 < 3 也**不用 hash 路由**，走线性路由；
- `topk_indices_buffer=` 自建 buffer：draft 注意力在 `compress_ratios[61]=0` 下是 **dense**（无 Compressor/Indexer），该 buffer 仅在 is_v32 时按需分配以保持 DSA 模块接口一致。
- FFN 仍是完整的 `DeepseekV4MoE`（384 专家/Top-6/共享专家），**不是稠密 MLP**。

输入 `(T, 4, 7168)` 与 `residual=None`，输出 `(hidden_states, residual)`，4 条残差流的结构与主层完全一致（hc_pre/hc_post 见文档 1/2）。

### 4.5 延迟 HC 头压缩与 hc_head 新实现

Layer.forward **不调用** `hc_head`（L126-L127 调用处被注释），直接返回 4 流隐藏状态。原因：

1. 下一步 MTP（若有多层）需要完整的多流输入；
2. 只有出 logits 时才需要单流，压缩延迟到 `compute_logits`；
3. 避免"压缩→投影→再展开"的往返开销。

v0.26 的 `hc_head`（[L131-L138](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4_mtp.py#L131)）改用 `hc_norm` 模块，与主模型 `DeepseekV4Model.hc_head`（deepseek_v4.py L1130 附近）保持一致：

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

### 5.1 SharedHead — 输出归一化与 LM Head（L36-L53）

```python
class SharedHead(nn.Module):
    def __init__(self, config, prefix, quant_config=None):
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(config.vocab_size, config.hidden_size,
                                   quant_config=quant_config, prefix=maybe_prefix(prefix, "head"))

    def forward(self, hidden_states):
        return self.norm(hidden_states)   # 只做 norm；head 矩阵在外部 logits_processor 中使用
```

- 结构上**每个 MTP 层各持有一个** `shared_head`（L88）；本配置只有 1 个 MTP 层，故只有一份；
- 命名为 "shared" 体现在 checkpoint 权重与主模型 LM head 同源（推理时通常 tie embeddings）；
- TP 下 `ParallelLMHead` 按词表维切分，与主模型 logits 路径一致。

---

### 5.2 完整 Forward 数据流

#### 5.2.1 单步草稿前向

```
输入: input_ids (T,)、positions (T,)、previous_hidden_states (T, 4*7168)、
      inputs_embeds (T, 7168)|None、spec_step_index
  1. 缺省嵌入: inputs_embeds = embed_tokens(input_ids)（Predictor L175-L176）
  2. 轮询选层: layers[str(spec_step_idx % 1)]
  3. mask position0 → enorm → e_proj → (T,1,7168)
     prev.view(T,4,7168) → hnorm → h_proj → (T,4,7168)
     两者广播相加得 (T,4,7168)                           [Layer L117-L122]
  4. mtp_block（dense DSA 注意力 + MoE，含 hc_pre/hc_post）→ (T,4,7168)
输出: 4 条残差流隐藏状态（不压缩）
```

#### 5.2.2 compute_logits（Predictor L186-L198）

```python
mtp_layer = self.layers[str(spec_step_idx % self.num_mtp_layers)]
hidden_states = hidden_states.view(-1, mtp_layer.hc_mult, mtp_layer.config.hidden_size)  # L193
hidden_states = mtp_layer.hc_head(hidden_states,
                                  mtp_layer.hc_head_fn, mtp_layer.hc_head_scale,
                                  mtp_layer.hc_head_base)                                # (T, 7168)
logits = self.logits_processor(mtp_layer.shared_head.head,
                               mtp_layer.shared_head(hidden_states))                    # (T, 129280)
```

即 **view 回 4 流 → hc_head 压单流 → shared_head.norm → 分片 LM head**。注意进入 `hc_head` 前显式 view：若上游传入的是拼平的 `(T, 4*7168)`，这里恢复 `(T, 4, 7168)`。

---

## 6. 权重加载与训练目标

### 6.1 权重加载与映射（load_weights，L253-L479）

#### 6.1.1 总体流程

1. `get_spec_layer_idx_from_weight_name(config, name)`（L291，主模型文件 L302-L305：`mtp.` 开头返回 0）筛选属于 MTP 的权重，非 MTP 权重直接 `continue`；并断言名字含 `mtp.0.`（L295）；
2. fp8 特殊重命名：`embed.weight → mtp.0.emb.tok_emb.weight`、`head.weight → mtp.0.head.weight`（L284-L289）；
3. 三段式前缀改写（L296-L301）：
   - 含 `.emb.tok_emb.`：`mtp.0.` → `model.`（预测器顶层嵌入）；
   - `no_mtp_block_in_name(name)` 命中（投影/HC 头/norm/head 等）：`mtp.0.` → `model.layers.0.`；
   - 其余：`mtp.0.` → `model.layers.0.mtp_block.`（draft block 内部权重）；
4. 逐片段改名后经 `stacked_params_mapping`（w1/w3 → gate_up_proj 融合）、`expert_params_mapping`（384 专家，EPLB 时含冗余）或默认 loader 加载。

#### 6.1.2 权重名片段映射表

| checkpoint 片段 | 加载名片段 | 行号 |
|----------------|-----------|------|
| `.w1.` / `.w2.` / `.w3.` | `.gate_proj.` / `.down_proj.` / `.up_proj.` | L303-L308 |
| `.scale` 结尾 | `.weight_scale` | L310-L311 |
| `.head.` | `.shared_head.head.` | L313-L314 |
| `.norm.` | `.shared_head.norm.` | L316-L317 |
| `.emb.tok_emb.` | `.embed_tokens.` | L319-L320 |
| `.attn.`（且不含 `self_attn`） | `.self_attn.` | L322-L323 |
| `.ffn.` / `.ffn_norm.` / `.attn_norm.` | `.mlp.` / `.post_attention_layernorm.` / `.input_layernorm.` | L324-L329 |
| `.gate.bias` | `.gate.e_score_correction_bias` | L331-L332 |
| `fused_qkv_a_proj` | 可选 QKV 融合加载；目标参数不存在时回退普通加载 | L362-L367 |

#### 6.1.3 `no_mtp_block_in_name`（L511-L524）

下列名字直接挂在 `model.layers.0.*` 下，**不加** `.mtp_block.`：

```python
[".hc_head_fn", ".hc_head_base", ".hc_head_scale",
 ".e_proj.", ".h_proj.", ".enorm.", ".hnorm.",
 ".norm.", ".head.", ".emb.tok_emb."]
```

另有 `_rewrite_spec_layer_name`（L481-L509）服务通用多层命名：`embed_tokens/enorm/hnorm/eh_proj/shared_head` 视为预测层直属权重，其余加 `.mtp_block.`，嵌入等共享权重提升到 `model.` 顶层。

#### 6.1.4 Attention Sink 的 CP 分支（L334-L343）

```python
if "sink" in name:
    if enable_dsa_cp():
        param.data.copy_(loaded_weight)              # 上下文并行：整份复制（本 rank 持有全头）
    else:
        narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
        param.data.copy_(narrow_weight)               # 普通 TP：只切本 rank 的头段
```

#### 6.1.5 mix_placement 融合共享专家（L345-L450）

`ascend_config.mix_placement=True` 时，checkpoint 中合并的共享专家权重被均分为 `n_shared_experts` 份，改名到 `mlp.experts.{384+j}.*`，再借 `expert_params_mapping` 以专家身份加载：`down_proj` 沿 dim 1 切，gate/up 沿 dim 0 切（L386-L413）。

---

### 6.2 训练目标：预压缩残差流缓冲

主模型 `get_mtp_target_hidden_states`（deepseek_v4.py [L1336-L1340](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L1336)）返回 `self.model._mtp_hidden_buffer`：

```
Pre-hc_head residual stream buffer
形状: (max_num_batched_tokens, hc_mult * hidden_size) = (…, 4*7168)
生命周期: 主模型 forward 填充，每个 target step 后有效
```

即 MTP 训练时对齐的目标是**主模型未经 hc_head 压缩的 4 流状态**——与推理期 MTP 消费的 `previous_hidden_states` 完全同构，保证训练/推理路径一致。

---

## 7. 总结

V4 的 MTP 是一个复用主模型 `DeepseekV2DecoderLayer` 的单预测层草稿模型：把"下一 token 嵌入（enorm+e_proj）"广播加到"上一步 4 条并行残差流（view+hnorm+h_proj）"上，送入 **dense 注意力 + 完整 MoE（无 hash）** 的 draft block；全程保持 4 流结构、只在 `compute_logits` 时用 **hc_norm + sigmoid 门控的 hc_head** 压回单流出 logits；DSSA 的 TopK 索引缓冲区由 MTP 层**自建**；权重加载通过 `mtp.0.*` 三段式前缀改写、w1/w2/w3 与量化 scale 重命名、专家映射与 attn_sink 的 CP/TP 分支完成。
