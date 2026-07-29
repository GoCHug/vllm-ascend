# Issue #10253 深度案例：PD 分离 SWA KV 传输混入 stale block 产生 NaN hidden states

> 整理时间: 2026-07-29
>
> 案例对象: [vllm-project/vllm-ascend#10253](https://github.com/vllm-project/vllm-ascend/issues/10253)
>
> 标题: [Bug]: PD disaggregated SWA KV transfer can include stale blocks and produce NaN hidden states
>
> 关联全景文档: [0_kvcache.md](./0_kvcache.md) §2.2 / 高危 TOP 10 第 2 名 / 案例 2
>
> 关键词: PD 分离 · SWA 滑动窗口 · stale block · NaN 传播 · 操作顺序错误 · prompt trim · dirty block · 全 NaN 输出

---

## 1. Issue 概览

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| Issue 编号 | [#10253](https://github.com/vllm-project/vllm-ascend/issues/10253) |
| 类型 | Bug |
| 标题 | [Bug]: PD disaggregated SWA KV transfer can include stale blocks and produce NaN hidden states |
| 状态 | **closed**（已修复） |
| 创建时间 | 2026-06-09 |
| 严重度 | 🔴 **极高** |
| 隐蔽性 | 🟡 **高**（需 PD 分离 + SWA 模型 + 足够并发才触发；单卡/非 SWA 不显现） |
| 影响面 | 端到端输出整体腐败为 NaN，生成完全不可用 |
| 触发链 | SWA tail clip ↓ 选到 dirty/未写入/过期尾部 block ↓ 经 Mooncake 传输到 D 端 ↓ attention 消费 NaN ↓ 全 NaN 逐层传播 |
| 修复方向 | 修正操作顺序：先 prompt-trim 再 SWA clip；对 SWA 组执行 prompt-trim；传输前有效性检查 |
| 全景档归档 | [0_kvcache.md](./0_kvcache.md) §2.2 传输 stale/NaN 数据 · 高危 TOP 10 第 2 名 |

> 典型的“顺序写反一步，整条推理链路 NaN”案例：bug 不在任何单一算子里，而藏在两个本应先后的裁剪步骤被调换/漏掉的顺序错误中——这是 KV 传输正确性里最隐蔽、最致命的一类。

---

## 2. 场景背景

### 2.1 PD 分离部署 (Prefill/Decode Disaggregation)

PD 分离架构中，Prefill 节点（P 端）一次性算完长 prompt 的 KV Cache，再通过 **Mooncake Connector**（基于 Mooncake / RDMA）跨节点传输到 Decode 节点（D 端），避免 D 端重算 prefix。`request_finished_all_groups()` 是 P 端 request 完成时的 HMA（Hybrid Memory Allocator）钩子，负责决定哪些 block 需要延迟释放（delay-free）以待异步传输，并把 `remote_block_ids` 打包进 `kv_transfer_params` 发给 D 端。

### 2.2 SWA 滑动窗口注意力 (Sliding Window Attention)

SWA 只保留序列**最近一个窗口长度**的 KV，窗口外的头部 KV 在 prefill 推进过程中会被 KV Cache Manager 释放/回收。对 P 端而言，待传输的 SWA 组 block 列表只应包含“窗口内”的尾部 block，而非整段 prompt 分配的全部 block。

- vLLM-Ascend 用 `num_swa_blocks[i] = cdiv(sliding_window, block_size) + 1` 估算第 i 组的窗口 block 数（+1 保守覆盖边界重叠）。
- SWA 尾部裁剪（tail clip）：`blocks[-num_swa_blocks[i]:]`，只取窗口尾部。

### 2.3 prompt trim（头部裁剪 / P 端截断）vs SWA tail clip（尾部裁剪）

P 端 prefill 存在**截断**（truncation）行为：为迁移最后一个 token 的 hidden state 到 D 端重算，P 端会从 prompt 尾部 pop 掉最后一个 token（`_truncate_request_for_prefill` / `_p_side_truncated`）。由此，KV Cache Manager 按“原始 prompt 长度”分配的 block 数，可能**多于实际被写入的 block 数**——尾部多出来的 block 是**未写入（unwritten / dirty）**的。

```
两种裁剪方向相反，必须配合：

  prompt trim  : 从 block 列表【尾部】砍掉未写入的 (P 端截断留下的脏 block)
                 blocks[:group_block_len]        ← 头部保留
  SWA tail clip: 从 block 列表【尾部】只取窗口内的 (头部窗外已被 Manager 释放)
                 blocks[-num_swa_blocks:]        ← 尾部保留
```

关键点：**两步都作用于同一份 `block_ids` 列表的尾部，但语义不同**——前者是“丢弃未写入”，后者是“只留窗口”。顺序错了，SWA clip 就会从“还没砍掉脏 block”的列表尾部取窗口，可能恰好把那些 dirty / 未写入 / 已过期的尾部 block 当成有效窗口 block 选中。

### 2.4 触发场景 = PD 分离 + SWA 模型 + 并发够高

SWA + PD 分离本身并不必然触发；dirty block 能否落入 SWA 尾部窗口，取决于 P 端截断后“多分配的 block 数”与“SWA 窗口 block 数”的相对位置。并发越高，不同 request 的 block 分配/复用越频繁，过期尾部 block 被复用进窗口的概率越大——这解释了“并发越高触发概率越大”的现象。

---

## 3. 现象描述

来自 issue 报告与 [0_kvcache.md](./0_kvcache.md) §2.2 的归纳：

1. **全 NaN 输出**：D 端 decode 产出的 hidden states 全部为 NaN，生成内容完全腐败、不可用。NaN 一旦进入 attention，softmax 后会逐层放大，整条 forward 链路被污染。
2. **并发越高触发概率越大**：低并发下可能跑通，升高并发后才复现——因为 dirty/stale block 的产生与跨 request 的 block 复用强相关，并发放大了“过期尾部 block 恰好落在 SWA 窗口”的概率。
3. **单卡 / 非 SWA 模型不触发**：
   - 单卡无跨节点传输，不走 `request_finished_all_groups` 的 SWA clip 路径；
   - 非 SWA 模型 `num_swa_blocks[i] == 0`，`get_sw_clipped_blocks` 原样返回，不产生尾部选取，不会选中 dirty block。
4. **无崩溃、无断言**：传输与 attention 算子不会因为读到 NaN/未初始化数据而 assert fail；表现为 silent 的全 NaN 输出。

---

## 4. 根因分析

### 4.1 操作顺序错误：先 SWA clip 再 prompt trim（或对 SWA 组跳过 trim）

修复前的 `request_finished_all_groups()` 路径中，裁剪顺序被写反 / trim 对 SWA 组被跳过：

```
错误顺序 (bug)                      正确顺序 (fix)
─────────────────────────────       ─────────────────────────────
1. get_sw_clipped_blocks(block_ids)  1. _compute_transfer_block_ids(trim)
   ↑ 从【未 trim】列表尾部取窗口        ↑ 先砍掉尾部未写入的 dirty block
2. _compute_transfer_block_ids       2. get_sw_clipped_blocks
   ↑ 再 trim（已经晚了一步）            ↑ 再从干净列表尾部取窗口
→ SWA 窗口可能选到 dirty/stale block  → 窗口只含有效 block
```

[0_kvcache.md](./0_kvcache.md) §2.2 给出的权威一句话根因：

> `request_finished_all_groups()` **先执行 SWA clip 再做 prompt trim**，顺序颠倒；`_compute_transfer_block_ids()` **对 SWA 组跳过了 prompt-trim** → 结果：SWA tail clip 可能选到未被写入或已过期的尾部 block，这些 dirty block 被传输到 decode 端。

### 4.2 dirty block 如何被 SWA clip 选中

逐步推演（bug 路径）：

1. P 端 prefill 为原始 prompt 长度分配了 `N+1` 个 block（`_p_side_truncated` 截断前）；
2. 截断后实际只写入前 `N` 个 block，第 `N+1` 个 block 是**已分配但未写入**（含历史脏数据 / NaN / 未初始化值）；
3. bug 路径先做 `get_sw_clipped_blocks`：在【未 trim】的 `N+1` 长度列表上取 `blocks[-num_swa_blocks:]`，若 `num_swa_blocks` 覆盖到第 `N+1` 个 → **dirty block 进入待传输列表**；
4. 该 dirty block 经 `kv_transfer_params.remote_block_ids` 被 Mooncake 异步传输到 D 端；
5. D 端 attention 把含 NaN / 未初始化的 K/V 送进 softmax → 全 NaN hidden states。

### 4.3 NaN 传播链路图

```
P 端 prefill
    │  原始 prompt 分配 N+1 个 block
    │  _truncate_request_for_prefill 截断 → 实际写入 N 个
    ▼
block_ids = [b0, b1, ..., b_{N-1}, b_N(dirty/未写入)]
    │
    │  ① BUG: 先 SWA clip (在未 trim 列表尾部取窗口)
    │     blocks[-num_swa_blocks:]  →  含 b_N(dirty)
    ▼
remote_block_ids 含 dirty block
    │  ② Mooncake RDMA 传输
    ▼
D 端接收 K/V (含 NaN / 未初始化)
    │  ③ scatter 进 D 端 KV Cache pool
    ▼
D 端 decode attention:
    Q · K^T  →  含 NaN logits
    softmax  →  全 NaN 概率
    · V      →  NaN hidden state
    ▼
逐层 forward:  NaN 进下一层 Q  →  全 NaN 传播至 logits  →  全 NaN 输出
```

### 4.4 `get_sw_clipped_blocks` 的完整语义（修复后代码佐证）

修复后的 `mooncake_hybrid_connector.py:1442-1445` 用注释明确固化了正确顺序：

```python
# P-side truncation can leave block ids allocated for the original
# prompt length. Drop those unwritten blocks before SWA tail clipping.
computed_block_ids = self._compute_transfer_block_ids(block_ids, request.num_prompt_tokens)  # ① 先 trim
computed_block_ids = self.get_sw_clipped_blocks(computed_block_ids)                          # ② 再 clip
```

非 hybrid 的 `mooncake_connector.py:1849-1850` 同样是 trim → SWA clip：

```python
computed_block_ids = self._get_transfer_block_ids(block_ids, len(request.prompt_token_ids))  # ① 先 trim
computed_block_ids = self._get_swa_transfer_block_ids(computed_block_ids)                    # ② 再 clip
```

`get_sw_clipped_blocks`（hybrid）/`_get_swa_transfer_block_ids`（非 hybrid）的核心动作都是尾部切片：

```python
# mooncake_hybrid_connector.py:1278-1283
return tuple([
    blocks[-self.num_swa_blocks[i]:] if self.num_swa_blocks[i] > 0 else blocks
    for i, blocks in enumerate(block_ids)
])

# mooncake_connector.py:1689-1694 (非 hybrid，额外过滤占位 0)
window_blocks = blocks[-group_info.blocks_per_window:]
transfer_block_ids.append([block_id for block_id in window_blocks if block_id != 0])
```

正因为 clip 取的是**尾部**，trim（也是作用于尾部，砍掉未写入）必须先做——否则 clip 取到的尾部可能正是脏 block。两步方向一致、但语义不同、顺序不可交换。

### 4.5 为什么“对 SWA 组跳过 prompt-trim”同样致命

`_compute_transfer_block_ids` 的 SWA 分支（`mooncake_hybrid_connector.py:1318-1330`）按 `prompt_len // group_compress_ratio` 估算应保留的 block 数并 `blocks[:group_block_len]` 截断。若实现里对 SWA 组误判为“窗口已自管理、无需 trim”而原样返回，则 SWA 组的 dirty 尾部 block 同样会进入后续 clip。两种 bug 模式（顺序反 / 对 SWA 跳过 trim）殊途同归：**dirty block 进了 remote_block_ids**。

---

## 5. 影响与表现

| 维度 | 表现 |
|------|------|
| 触发配置 | PD 分离 + SWA（滑动窗口）模型 + 足够并发 |
| 不触发 | 单卡 / 非 SWA 模型 / 低并发（dirty block 难落入窗口） |
| 现象 | D 端全 NaN hidden states，输出完全腐败 |
| 严重度 | 🔴 **极高**（端到端不可用，非“质量下降”而是“彻底崩坏”） |
| 隐蔽性 | 🟡 **高**（需 SWA + PD + 并发三要素；CI 默认低并发用例难复现） |
| 是否报错 | **不报错**，silent 全 NaN；传输算子与 attention 均无 NaN 断言 |
| 并发敏感性 | 高——并发放大 block 复用，stale tail 落入窗口概率上升 |
| 与 #10413 关系 | GPQA 精度异常（[0_kvcache.md](./0_kvcache.md) §1.3 #10413）的同类 PD 分离路径侧面佐证：SWA+PD 组合的高危性不止于 NaN |

---

## 6. 修复方案

### 6.1 核心修复：修正裁剪顺序 + 对 SWA 组执行 trim

修复后的两处实现（已合入 main）均固化 **先 trim 再 clip**，并配 `Drop those unwritten blocks before SWA tail clipping` 注释作防回归：

**Hybrid connector** — `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py:1442-1445`：

```python
# P-side truncation can leave block ids allocated for the original
# prompt length. Drop those unwritten blocks before SWA tail clipping.
computed_block_ids = self._compute_transfer_block_ids(block_ids, request.num_prompt_tokens)
computed_block_ids = self.get_sw_clipped_blocks(computed_block_ids)
```

**非 hybrid connector** — `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:1848-1850`：

```python
num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)
computed_block_ids = self._get_transfer_block_ids(block_ids, len(request.prompt_token_ids))
computed_block_ids = self._get_swa_transfer_block_ids(computed_block_ids)
```

### 6.2 `_compute_transfer_block_ids` / `_get_transfer_block_ids` 对 SWA 组不再跳过 trim

- Hybrid `_compute_transfer_block_ids`（`mooncake_hybrid_connector.py:1318-1330`）：按 `prompt_len`（compress-aware）算 `group_block_len` 并 `blocks[:group_block_len]`，**SWA 组也按 prompt 长度截断**（仅 compress + `num_swa_blocks==0` 才走 ratio 分支），不放过未写入尾部。
- 非 hybrid `_get_transfer_block_ids`（`mooncake_connector.py:1656-1679`）：对非 state group 按 `cdiv(prompt_len, tokens_per_block*cp_size)` 截断；SWA 组的 `blocks_per_window != 0` 不等于“跳过 trim”——trim 由 prompt 长度决定，与是否 SWA 无关。docstring 明确：`SWA tail clipping is handled as a separate step after this.`

### 6.3 传输前有效性检查（防御层）

`_get_swa_transfer_block_ids` 在 clip 后过滤占位 block `0`（`[block_id for block_id in window_blocks if block_id != 0]`，`mooncake_connector.py:1694`）；接收侧 `KVCacheRecvingThread.invalid_block_ids` / `get_block_ids_with_load_errors`（`mooncake_connector.py:591-605, 2441-2444`）追踪加载失败的 block。这些是“dirty/stale block 漏网后”的兜底，但根治仍靠 6.1/6.2 的顺序保证。

### 6.4 集成测试钉死顺序

代码库已加针对性单测，pin 住 trim→clip 顺序与 SWA 组 trim 行为，防止回归：

- `tests/ut/kv_offload/test_mooncake_hybrid_connector.py:254` `test_compute_transfer_block_ids_trims_swa_groups` — SWA 组经 trim 后长度正确。
- `tests/ut/kv_offload/test_mooncake_hybrid_connector.py:262` `test_request_finished_trims_before_swa_clip` — **直接钉死“先 trim 再 SWA clip”**。
- `tests/ut/kv_offload/test_mooncake_connector.py:1742` `test_transfer_block_ids_trims_mtp_before_swa_zero_filter` — 链式 trim→SWA。
- `tests/ut/kv_offload/test_mooncake_connector.py:1816` `test_request_finished_trims_mtp_before_swa_tail_clip` — `prompt_len=64`、`blocks_per_window=3`，断言 `remote_block_ids == ([12,13,14],)`。
- `tests/ut/kv_offload/test_mooncake_connector.py:1703` `test_get_transfer_block_ids_trims_sliding_window_mtp_blocks`。

---

## 7. 复现与验证

### 7.1 触发矩阵

| PD 分离 | SWA 模型 | 足够并发 | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗ | — | — | ✗（无跨节点传输） |
| ✓ | ✗ | — | ✗（`num_swa_blocks==0`，无 tail clip） |
| ✓ | ✓ | ✗（低并发） | △（难复现，dirty tail 难落窗口） |
| ✓ | ✓ | ✓（高并发） | **✓ 全 NaN 输出** |

### 7.2 排查路径

1. **确认三要素**：PD 分离部署 + 模型含 SWA 层（`SlidingWindowSpec`）+ 复现时并发量。
2. **单卡/非 SWA 对照**：单卡或换非 SWA 模型后 NaN 消失 → 锁定 SWA + PD 传输路径。
3. **降并发对照**：低并发不复现、高并发复现 → 高度怀疑 stale block 复用相关。
4. **D 端 NaN 定位**：在 D 端 dump 接收到的 KV block，检查是否含 NaN/未初始化值；比对 P 端对应 block 的实际写入状态。
5. **检查 `remote_block_ids` 与实际写入 block 数**：若传输列表长度 > 实际写入 block 数 → dirty block 混入。
6. **验证修复**：升级到含 6.1 顺序修复的版本后，高并发压测不再出 NaN。

### 7.3 回归测试建议

- 新增 e2e：PD 分离 + SWA 模型 + 高并发（≥100）压测，断言输出无 NaN、accuracy 回归。
- 单测加“P 端截断 + SWA 窗口恰好覆盖截断尾部 block”的构造用例，强制 dirty block 落入窗口，验证 trim 先行能剔除。
- 在 `request_finished_all_groups` 入口加 invariant 断言：trim 后的 block 数 ≤ 实际写入 block 数（防回归断言）。

---

## 8. 经验与启发

### 8.1 操作顺序敏感性：方向相同 ≠ 可交换

prompt-trim 和 SWA clip 都对 block 列表**尾部**操作，直觉上“都在砍尾巴，先后无所谓”——但语义根本不同（trim 丢脏，clip 选窗口）。**凡是两个裁剪都作用于同一数据结构的同一端，必须严格论证顺序**，否则后者会基于前者未净化的数据做选择。这是本 bug 最核心的教训。

### 8.2 dirty block 防御：传输前必须保证“可见 = 已写入”

跨节点 KV 传输的正确性前提是“`remote_block_ids` 里每个 block 都已被有效写入”。任何上游（截断、MTP 溢出、CP 分组、prefix-cache 复用）带来的“已分配未写入”block，都必须在打包 `remote_block_ids` **之前**剔除。传输层不应承担“鉴别脏 block”的责任——它是最后一公里，不是第一道防线。参见 [0_kvcache.md](./0_kvcache.md) §2.1 并发 chain 断裂（#7707）同为“读到不完整 KV”家族。

### 8.3 SWA + PD 组合是高危叠加面

SWA 的“窗口外 block 被 Manager 释放/复用”与 PD 的“跨节点传输需选块”叠加，使得“选到过期/未写入块”的概率被放大。单独 SWA（单卡）或单独 PD（非 SWA）都不会暴露；二者交叉处是 vllm-ascend 特有的精度雷区（参见 [0_kvcache.md](./0_kvcache.md) §2、§8）。任何 SWA 模型上 PD 分离的改动，都应把“SWA tail clip 顺序”当首要审查项。

### 8.4 并发放大隐蔽 bug

低并发跑通 ≠ 正确。block pool 的复用、stale tail 落入窗口都是概率事件，并发越高越接近必然。精度类 CI 若只跑低并发用例，会系统性漏掉这类问题——建议精度回归纳入高并发压测档。

### 8.5 用注释与单测固化顺序契约

修复点用一行注释 `Drop those unwritten blocks before SWA tail clipping` + 命名清晰的 `test_request_finished_trims_before_swa_clip` 单测，把“顺序不可交换”从隐式知识变成显式契约。后续任何重构（如把 trim/clip 合并、改分组逻辑）都会被单测拦下。

### 8.6 NaN 是“数据腐败”的最强信号，但出现即已晚

传输代码本身不产生 NaN——NaN 是 D 端 attention 消费 dirty block 的下游症状。看到全 NaN 输出应第一时间回溯“是否有 dirty/stale block 进了传输链路”，而非先怀疑算子数值稳定性。

---

## 9. 关联问题

| 编号 | 关联点 |
|------|--------|
| [0_kvcache.md](./0_kvcache.md) §2.2 / 案例 2 | 本案例在全景文档的归档（高危 TOP 10 第 2 名） |
| [0_kvcache.md](./0_kvcache.md) §2.1 / #7707 | KV chain 断裂（100 并发读到不完整 KV）—— 同属“dirty/stale block 进传输”家族 |
| [vllm-ascend#10413](https://github.com/vllm-project/vllm-ascend/issues/10413) | 4P1D 128K GPQA 精度异常 —— PD 分离路径精度问题的侧面佐证 |
| [vllm#49052](https://github.com/vllm-project/vllm/pull/49052) | KV Offload SWA 加载断言 `+1` 边界修复 —— vLLM 侧 SWA+offload 对齐问题 |
| [vllm#48911](https://github.com/vllm-project/vllm/pull/48911) | Preserve reachable tails for hybrid SWA groups —— vLLM 侧 SWA 可达尾部裁剪，`is_store_reachable_swa_chunk` |
| [vllm#42959](https://github.com/vllm-project/vllm/pull/42959) | Prevent offloading stale sliding window blocks —— vLLM 侧“stale SWA block”同源概念 |
| [vllm-ascend#8540](./1_pr8540_tp_unequal_mtp_kv.md) | TP 不等 + MTP 层 KV 漏传输 —— 同属“选 block 时漏算/错算”的 PD 传输精度家族 |

> vLLM 主仓对 SWA + offload 的连续修复（#42959 → #48911 → #49052）显示：**SWA 与 KV 迁移/缓存的交叉面在整个生态里都是高频精度雷区**，vllm-ascend 的 #10253 是同一类风险在 PD 分离 + Mooncake 路径上的本土化表现。

---

## 10. 结论

> **核心教训：跨节点 KV 传输中，凡是对同一份 block 列表做多次裁剪/选取（prompt-trim 与 SWA tail clip），必须保证“丢弃未写入 block”先于“按窗口选尾部”执行——顺序一反，dirty/stale block 即混入传输，在 D 端 attention 中放大为全 NaN 输出。**

---

### 附：根因摘要（3 点）

- **操作顺序错误**：`request_finished_all_groups()` 先做 SWA tail clip 再做 prompt trim（或两步顺序未被强制），导致 SWA clip 在“未剔除未写入尾部 block”的列表上选窗口，选中 dirty block。
- **对 SWA 组跳过 prompt-trim**：`_compute_transfer_block_ids` 对 SWA 组未按实际写入长度截断，使 P 端截断留下的未写入/过期尾部 block 残留于待传输列表。
- **dirty block 经传输 → attention → 全 NaN 传播**：含 NaN/未初始化数据的 block 经 Mooncake 传到 D 端，attention softmax 消费 NaN 后逐层放大为全 NaN hidden states，输出彻底腐败。