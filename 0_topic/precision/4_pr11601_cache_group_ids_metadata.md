# PR #11601 深度案例：Mooncake 传输元数据缺失 cache group id 致混合模型 KV 错配

> 整理时间: 2026-07-29
>
> 案例对象: [vllm-project/vllm-ascend#11601](https://github.com/vllm-project/vllm-ascend/pull/11601)
>
> 标题: [BugFix][v0.23.0][KV Transfer] Use cache group ids in Mooncake split metadata
>
> 关联全景文档: [0_kvcache.md](./0_kvcache.md) §2.3（传输连续性 / 布局不匹配）/ §3（Prefix Cache / hybrid 模型正确性）
>
> 关键词: Mooncake · cache group id · split metadata · 混合/hybrid KV cache · 传输元数据错配 · receive 端

---

## 1. PR 概览

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| PR 编号 | [#11601](https://github.com/vllm-project/vllm-ascend/pull/11601) |
| 类型 | BugFix |
| 作者 | [@Yuli-yx](https://github.com/Yuli-yx) |
| Reviewer | @LCAIZJ, @MengqingCao, @wangxiyuan（hovercard 可见）；@linfeng-yuan 参与讨论 |
| 状态 | closed / **MERGED** |
| 创建时间 | 2026-07-08 |
| 合并时间 | 2026-07-08（同日创建并合并） |
| 合并 commit | `ae870bc5e265a340912cde392f23dad3671a0a88` |
| 改动规模 | **+89 / −7**（含 80 行测试） |
| 影响文件 | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`、`tests/ut/kv_offload/test_mooncake_connector.py` |
| 目标分支 | `main` |
| 提交数 | 4 |
| vLLM 版本标签 | v0.23.0 |

> 说明：GitHub REST API (`api.github.com/repos/.../pulls/11601`) 在采集时返回 **403 rate-limit**，故 §1 元数据来自 PR 渲染页 HTML 解析（`state: MERGED`、合并 sha `ae870bc...`、作者 `Yuli-yx`、reviewer 列表、提交数 4）与本地源码作为 ground truth；diff 来自 `github.com/.../pull/11601.diff`（HTTP 200）。下文所有 `file:line` 均以本地 checkout `/Users/wushanglun/Desktop/vllmgch/vllm-ascend/`（已含本 PR 修复）为准。

---

## 2. 场景背景

### 2.1 两层“组”概念：cache group vs transfer group

vLLM 的 hybrid KV cache manager 把模型层划分为若干 **KV cache group**（`kv_cache_groups`），每个 cache group 内所有层共享同一种 `KVCacheSpec`（同一 block_size、同一 kernel 布局）。例如 GLM5 / DeepSeek-V4 这类“attention + mamba/linear/SSM + compress”混合模型，通常会有：

- cache group 0：FullAttention 层（可能含 compress/MQA 变体）
- cache group 1：Mamba/SSM 层（state-space，无传统 block 分片）

而 vLLM-Ascend 的 Mooncake connector 又在 cache group 之上再做一次 **split**：`_build_kv_group2layeridx` 把一个 cache group 按 KV spec 细粒度（`num_kv_heads`、`total_num_kv_heads`）拆成若干 **transfer group**（`kv_group2layeridx` 的 key，记作 `transfer_group_id`）。于是：

```
cache group (kv_cache_group_id)  ──split──▶  transfer group (transfer_group_id)
        0   (FullAttention, 两批 head 数不同) ──▶  transfer 0, transfer 1
        1   (Mamba)                           ──▶  transfer 2
```

一个 cache group 可对应**多个** transfer group；但**反向不独立**——同一 cache group 的所有 transfer group 在 P/D 端共享同一份 block id 列表（`meta.local_block_ids` / `meta.remote_block_ids` 的下标是 cache group id，不是 transfer group id）。这就埋下了“两套 id 必须显式区分”的隐患。

### 2.2 PD 分离 + 无 CP 的 split metadata 路径

在 PD 分离（P 端 prefill / D 端 decode）且无 context parallel（`pcp_size * dcp_size == 1`）场景下，D 端调用 `_get_kv_split_metadata` 构造本次拉取的端口与 block id 元数据，再交给 receive 端 `_transfer_kv_cache_all_groups` 逐 transfer group 执行实际 RDMA 拷贝。元数据里的 `local_block_ids` / `remote_block_ids` 是一个 **按 cache group id 索引的列表**——这正是本 PR 的核心切入点。

### 2.3 触发模型

GLM5、DeepSeek-V4 等“attention + (compress/MQA) + mamba/SSM”混合模型，当一个 cache group 被拆成 ≥2 个 transfer group（典型：FullAttention 层中存在 head 数不同的子组，或 compress 层单独成组）时即触发。纯 attention 同构模型（单一 spec、head 数一致）不会 split，故不触发。

---

## 3. 根因分析

### 3.1 Bug 代码（修复前）

`_get_kv_split_metadata` 的无 CP 分支用 `append` 把每个 transfer group 的 kernel block id 顺序追加，并以 transfer group 的循环下标 `group_idx` 隐式当作 `meta.local_block_ids` 的索引；`_get_kernel_block_ids` 内部同样直接用 `group_idx` 取 `meta.local_block_ids[group_idx]`：

```python
# vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py（修复前，来自 PR diff - 行）
def _get_kernel_block_ids(self, layer_indices, meta, group_idx, group_spec):
    if group_spec["kv_cache_spec_type"] == "MambaSpec":
-       return list(meta.local_block_ids[group_idx]), list(meta.remote_block_ids[group_idx])
    ...
-   kernel_local = self._expand_block_ids(list(meta.local_block_ids[group_idx]), local_scale)
-   kernel_remote = self._expand_block_ids(list(meta.remote_block_ids[group_idx]), remote_scale)

def _get_kv_split_metadata(self, req_id, meta):
    ...
-   local_block_ids: list = []
-   remote_block_ids: list = []
    for group_idx, (group_spec, layer_indices) in self.kv_group2layeridx.items():
        local_kernel_block_ids, remote_kernel_block_ids = self._get_kernel_block_ids(
            layer_indices, meta, group_idx, group_spec
        )
-       local_block_ids.append(local_kernel_block_ids)
-       remote_block_ids.append(remote_kernel_block_ids)
    local_block_ids_list = [tuple(local_block_ids) for _ in remote_handshake_port_list]
    remote_block_ids_list = [tuple(remote_block_ids) for _ in remote_handshake_port_list]
    return (remote_handshake_port_list, local_block_ids_list, remote_block_ids_list)
```

### 3.2 不一致点：transfer group id ≠ cache group id

`meta.local_block_ids` / `meta.remote_block_ids` 来自 scheduler，长度 = **cache group 数**（这里是 2：attention=0、mamba=1），下标语义是 `kv_cache_group_id`。

而 `kv_group2layeridx.items()` 遍历的是 **transfer group**，`group_idx` 取值 0/1/2（split 后 3 个 transfer group）。修复前代码直接用 `group_idx` 去索引 `meta.local_block_ids`，并 `append` 成新列表，造成两重错配：

```
transfer group 遍历:        group_idx=0 (attn, cache_group=0)
                            group_idx=1 (attn, cache_group=0)   ← 与上一个同 cache group
                            group_idx=2 (mamba, cache_group=1)

修复前 local_block_ids（append 顺序）:
    index 0 ← transfer0 的 kernel ids   (应放 cache group 0)
    index 1 ← transfer1 的 kernel ids   (应放 cache group 0 ── 覆盖/错位!)
    index 2 ← transfer2 的 kernel ids   (应放 cache group 1 ── 越界/错位!)

但 receive 端 _transfer_kv_cache_all_groups 正确地按 kv_cache_group_id 取:
    local_block_ids[0]  ← 期望 cache group 0 的 block ids
    local_block_ids[1]  ← 期望 cache group 1 的 block ids
```

具体两种破坏：

1. **append 顺序覆盖**：transfer0 与 transfer1 同属 cache group 0，修复前 `append` 后 `local_block_ids[0]=transfer0`、`local_block_ids[1]=transfer1`。receive 端取 `local_block_ids[1]` 期望的是 cache group 1（mamba）的 block ids，实际拿到 transfer1（attention 第二批）的 ids → **attention 的 KV 数据被写进 mamba cache 槽位**，mamba 的真实 ids 丢失。
2. **Mamba 分支错读**：`_get_kernel_block_ids` 的 Mamba 分支 `meta.local_block_ids[group_idx]`，当 `group_idx=2` 而 `meta.local_block_ids` 只有 2 个元素时，要么 `IndexError` 崩溃，要么（若 append 后列表被拉长到 3）读到错误的一组 ids。

### 3.3 receive 端早已“正确”，send 端元数据却“错配”

值得注意：receive 端 `_transfer_kv_cache_all_groups`（`mooncake_connector.py:798-809`）在修复前/后都用了 `kv_cache_group_id = group_spec.get("kv_cache_group_id", group_idx)` 并以它索引 `local_block_ids[kv_cache_group_id]`——即 **D 端接收侧的 id 语义一直是正确的**。bug 只在 **send 侧构造 split metadata** 时存在：send 侧用 transfer group 下标组装了一个“错位”的列表，receive 侧再按正确的 cache group id 去读这个错位列表，于是 block ids 被解读到错误的 cache group → KV 数据写入错误的 cache（attention ↔ mamba/compress 互换）。

这就是标题“Use cache group ids in **Mooncake split metadata**”的精确含义：修复点在元数据构造侧，而非传输/接收侧。

### 3.4 后果：silent 精度腐败或崩溃

- **不崩溃路径**（append 后列表恰好够长）：attention 的 block ids 被当作 mamba ids 传给 mamba 层 RDMA 拷贝，mamba state（conv/S4）的内存布局与 attention block 完全不同 → 数据落位到错误 cache → D 端 mamba/attention 层读到错配 KV → **silent 精度腐败**，无断言。
- **崩溃路径**：当 `group_idx` 超过 `meta.local_block_ids` 原始长度且 Mamba 分支直接索引时，`IndexError`，或下游 `split_if_not_byte_contiguous` 因 ids 维度不匹配触发段错误（与 [#7792](https://github.com/vllm-project/vllm-ascend/issues/7792) 的 503900 同源路径）。

---

## 4. 影响与表现

| 维度 | 表现 |
|------|------|
| 触发配置 | PD 分离 + 无 CP + hybrid 模型且一个 cache group 被 split 成 ≥2 transfer group（GLM5/DeepSeek-V4 attention+compress+mamba） |
| 不触发 | 纯 attention 同构模型（无 split：transfer group 数 == cache group 数，`group_idx == kv_cache_group_id` 恒成立） |
| 现象 | attention 与 mamba/compress 的 KV block ids 在元数据层错位；D 端 KV 数据写入错误 cache → 精度异常或 IndexError/段错误 |
| 严重度 | 🟡 中（silent 精度腐败）/ 🔴 高（崩溃路径） |
| 隐蔽性 | 🟡 高：纯 attention 模型永不触发；hybrid 但未 split 也不触发；需 cache group 内多 spec 才暴露 |
| 是否报错 | 部分路径 `IndexError`/段错误；部分路径 silent，无断言 |

---

## 5. 修复方案

### 5.1 引入 `_get_kv_cache_group_id` 显式取 id

新增静态方法，从 `group_spec` 字典读取 `kv_cache_group_id`，缺省回退到 `group_idx`（保持非 split 场景兼容）：

```python
# mooncake_connector.py:2528-2530
@staticmethod
def _get_kv_cache_group_id(group_idx: int, group_spec: dict[str, Any]) -> int:
    return group_spec.get("kv_cache_group_id", group_idx)
```

`kv_cache_group_id` 字段在 `_build_kv_group2layeridx` → `_serialize_kv_group_spec` 时已写入每个 transfer group 的 spec（`mooncake_connector.py:2178-2184` / `2077-2078`），所以 send 侧只需“读出来用”。

### 5.2 `_get_kernel_block_ids`：用 cache group id 索引 meta

```diff
  def _get_kernel_block_ids(self, layer_indices, meta, group_idx, group_spec):
+     kv_cache_group_id = self._get_kv_cache_group_id(group_idx, group_spec)
      if group_spec["kv_cache_spec_type"] == "MambaSpec":
-         return list(meta.local_block_ids[group_idx]), list(meta.remote_block_ids[group_idx])
+         return list(meta.local_block_ids[kv_cache_group_id]), list(meta.remote_block_ids[kv_cache_group_id])
      ...
-     kernel_local = self._expand_block_ids(list(meta.local_block_ids[group_idx]), local_scale)
-     kernel_remote = self._expand_block_ids(list(meta.remote_block_ids[group_idx]), remote_scale)
+     kernel_local = self._expand_block_ids(list(meta.local_block_ids[kv_cache_group_id]), local_scale)
+     kernel_remote = self._expand_block_ids(list(meta.remote_block_ids[kv_cache_group_id]), remote_scale)
```

### 5.3 `_get_kv_split_metadata`：列表按 cache group 数预分配 + 按 id 赋值

```diff
-     local_block_ids: list = []
-     remote_block_ids: list = []
+     local_block_ids: list[list[int]] = [[] for _ in meta.local_block_ids]
+     remote_block_ids: list[list[int]] = [[] for _ in meta.remote_block_ids]
      for group_idx, (group_spec, layer_indices) in self.kv_group2layeridx.items():
          local_kernel_block_ids, remote_kernel_block_ids = self._get_kernel_block_ids(
              layer_indices, meta, group_idx, group_spec
          )
-         local_block_ids.append(local_kernel_block_ids)
-         remote_block_ids.append(remote_kernel_block_ids)
+         kv_cache_group_id = self._get_kv_cache_group_id(group_idx, group_spec)
+         local_block_ids[kv_cache_group_id] = local_kernel_block_ids
+         remote_block_ids[kv_cache_group_id] = remote_kernel_block_ids
```

两点关键：

1. **预分配长度 = cache group 数**（`meta.local_block_ids` 长度），而非 transfer group 数——保证下标语义与 receive 端一致。
2. **按 `kv_cache_group_id` 赋值**：同一 cache group 的多个 transfer group 中，后处理的 transfer group 会覆盖前一个的 block ids。

### 5.4 覆盖语义的合理性

同一 cache group 的多个 transfer group 共享同一组 block ids（因为同一 cache group 内 block_size/kernel 布局一致，P/D 端按 cache group 分配 block），所以“后写覆盖前写”是正确的——最终 `local_block_ids[kv_cache_group_id]` 存的就是该 cache group 唯一的那组 kernel block ids。新增测试 `test_hybrid_no_cp_uses_kv_cache_group_ids_for_split_transfer_groups`（`tests/ut/kv_offload/test_mooncake_connector.py:2890`）正是验证这一点：3 个 transfer group（id 0,1,2）、2 个 cache group（id 0,1），期望输出 `local_ids`/`remote_ids` 长度为 1（单端口）、内层按 cache group 分组长度为 2。

---

## 6. 关键代码上下文

文件：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`

| 位置 | 内容 | 说明 |
|------|------|------|
| `:99` | `kv_group2layeridx: dict[int, tuple[dict, list[int]]]` | transfer group id → (spec, layer_indices)，key 是 **transfer group id** |
| `:2077-2078` | `serialized["kv_cache_group_id"] = kv_cache_group_id` | `_serialize_kv_group_spec` 把 cache group id 写入 spec 字典 |
| `:2125-2189` | `_build_kv_group2layeridx` | cache group → transfer group 的 split：`for kv_cache_group_id, group_spec in enumerate(kv_cache_groups)`，内部按 spec_key 再拆，每个 transfer group 携带原 `kv_cache_group_id` |
| `:2178-2188` | `kv_group2layeridx[transfer_group_id] = (..., kv_cache_group_id=kv_cache_group_id)` | transfer group 与 cache group 的映射落表 |
| `:2528-2530` | `_get_kv_cache_group_id` | **本 PR 新增**：`group_spec.get("kv_cache_group_id", group_idx)` |
| `:2532-2563` | `_get_kernel_block_ids` | **本 PR 修复点 1**：Mamba/Attention 分支均改用 `kv_cache_group_id` 索引 `meta.local_block_ids`/`meta.remote_block_ids` |
| `:2609-2665` | `_get_kv_split_metadata`（无 CP 分支） | **本 PR 修复点 2**：列表预分配 + 按 `kv_cache_group_id` 赋值（取代 `append`） |
| `:798-809` | `_transfer_kv_cache_all_groups`（receive 端） | 早已正确使用 `kv_cache_group_id` 索引——证明 bug 仅在 send 侧元数据 |
| `:897` | `split_if_not_byte_contiguous` 调用处 | receive 端按错位 ids 切分时会触发崩溃/段错误（关联 #7792/#12183） |

测试文件：`tests/ut/kv_offload/test_mooncake_connector.py:2890`（`test_hybrid_no_cp_uses_kv_cache_group_ids_for_split_transfer_groups`），构造 `kv_group2layeridx` 含 3 transfer group / 2 cache group，断言 `_get_kv_split_metadata` 输出按 cache group 分组。

---

## 7. 复现与验证

### 7.1 触发矩阵

| hybrid 模型 | cache group 内 split | 无 CP | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗（纯 attention） | — | ✓ | ✗（transfer==cache group，id 恒等） |
| ✓（attention+mamba） | ✗（每 cache group 单一 spec） | ✓ | ✗（无 split，`group_idx==kv_cache_group_id`） |
| **✓** | **✓（cache group 内多 spec）** | **✓** | **✓ 触发错配** |
| ✓ | ✓ | ✗（有 CP） | 走另一条 `get_cp_group_meta` 路径，非本 bug |

### 7.2 排查方法

1. **确认模型是否 hybrid + split**：dump `worker.kv_group2layeridx`，看是否存在 ≥2 个 transfer group 共享同一 `kv_cache_group_id`。
2. **对比 transfer group 数 vs cache group 数**：若 `len(kv_group2layeridx) > len(kv_cache_config.kv_cache_groups)`，即存在 split，落入本 bug 风险面。
3. **校验元数据长度**：在 `_get_kv_split_metadata` 后断言 `len(local_block_ids) == len(meta.local_block_ids)`（修复前会不相等或错位）。
4. **逐 cache group 比对 ids**：D 端 receive 时，检查 `local_block_ids[kv_cache_group_id]` 是否真正属于该 cache group（attention ids 不应出现在 mamba 槽位）。
5. **崩溃定位**：`IndexError` 或 `split_if_not_byte_contiguous` 段错误时，回溯 `_get_kernel_block_ids` 的 Mamba 分支是否越界。

### 7.3 回归测试（本 PR 已加）

- `test_hybrid_no_cp_uses_kv_cache_group_ids_for_split_transfer_groups`：构造 3 transfer group（2 个 FullAttention 归 cache group 0、1 个 Mamba 归 cache group 1），P 端 TP=4 / D 端 TP=2，断言输出 `local_ids == [([70,71,72,73],[80,81,82,83])]`、`remote_ids == [([50,51,52,53],[60,61,62,63])]`——即**按 cache group 分组、长度为 2**，而非按 transfer group 分组、长度为 3。

---

## 8. 经验与启发

### 8.1 两套 id 必须显式区分，不可用循环下标隐式兼任

本 bug 的本质是 **transfer group id 与 cache group id 被同一个循环变量 `group_idx` 隐式兼任**。凡是一个领域存在“逻辑分组”与“传输/调度分组”两层概念（且为 1:N），在跨边界传递索引时必须显式携带并使用真正的 id，不能依赖“循环顺序恰好等于 id”。这与 [#11886](./3_pr11886_total_kv_heads_transfer.md) “传输组需显式携带 total KV heads”同属一类：元数据要自描述，不可依赖隐式对齐。

### 8.2 send 侧元数据与 receive 侧索引必须同源同语义

receive 端 `_transfer_kv_cache_all_groups` 一直正确使用 `kv_cache_group_id`，但 send 端 `_get_kv_split_metadata` 用 `append` + 顺序下标构造了一个“语义错位”的列表。两端对同一数据结构（`local_block_ids`）的索引语义不一致，是跨节点传输正确性问题的高发模式（参见 [0_kvcache.md](./0_kvcache.md) §2.3）。修复后两端统一以 `kv_cache_group_id` 为唯一索引语义。

### 8.3 split 是隐藏 code path

`_build_kv_group2layeridx` 的 split 只在 cache group 内出现多 spec 时触发（如 attention 层 head 数不一致、compress 层单独成组）。纯 attention 同构模型永不 split，CI 默认用例难以覆盖。任何涉及 `kv_group2layeridx` 遍历的改动，都应把 **“未 split（transfer==cache group）”与“split 后（transfer > cache group）”** 当作两条独立路径测试。

### 8.4 列表预分配 + 按 id 赋值 优于 顺序 append

修复用 `[[] for _ in meta.local_block_ids]` 预分配再 `local_block_ids[kv_cache_group_id] = ...`，而非 `append`。前者天然强制“下标 = 语义 id”，后者依赖顺序巧合。当多对一映射存在时，append 会掩盖覆盖语义，预分配能暴露并正确处理“后写覆盖前写”。

### 8.5 meta 字段的索引语义应在类型/注释中显式化

`meta.local_block_ids: BlockIds` 当前是 `tuple[list[int], ...]`，其“按 cache group id 索引”的语义只在注释里。建议在 `ReqMeta` 定义处（`mooncake_connector.py:112-115`）注明“indexed by kv_cache_group_id”，避免下游再次误用 transfer group id 索引。

---

## 9. 关联问题

| 编号 | 关联点 |
|------|--------|
| [vllm-ascend#8540](https://github.com/vllm-project/vllm-ascend/pull/8540) / [1_pr8540](./1_pr8540_tp_unequal_mtp_kv.md) | TP 不等 + MTP 层 KV 漏传输——同属 Mooncake 传输“层数/分组口径”类问题，本 PR 是其在“cache group 维度”的姊妹案例 |
| [vllm-ascend#11886](https://github.com/vllm-project/vllm-ascend/pull/11886) / [3_pr11886](./3_pr11886_total_kv_heads_transfer.md) | 传输组需显式携带 total KV heads——同类“元数据自描述”问题，同文件同路径 |
| [vllm-ascend#12183](https://github.com/vllm-project/vllm-ascend/pull/12183) / [2_pr12183](./2_pr12183_non_contiguous_mooncake_pa.md) | Non-contiguous Mooncake PA cache inputs——receive 端 `split_if_not_byte_contiguous` 在 ids 错配时会崩溃，本 bug 是其上游元数据错配的成因之一 |
| [vllm-ascend#9500](https://github.com/vllm-project/vllm-ascend/pull/9500) / [5_pr9500](./5_pr9500_shared_by_empty.md) | DeepSeek-V4 `shared_by` 可能为空——同 DeepSeek-V4 hybrid 模型族的 KV 传输正确性 |
| [vllm-ascend#7792](https://github.com/vllm-project/vllm-ascend/issues/7792) / [11_issue7792](./11_issue7792_mooncake_glm5_503900.md) | GLM5 mooncake 503900 段错误——本 bug 的错配 ids 喂给传输层会触发同类 slice 失败 |
| [0_kvcache.md](./0_kvcache.md) §2.3 / §3 | 本案例在全景文档中的归档位置（传输连续性/布局不匹配、hybrid 模型 prefix cache 正确性） |

---

## 10. 时间线小结

| 时间 | 事件 |
|------|------|
| 2026-07-08 03:49 UTC | PR 创建（作者 @Yuli-yx） |
| 2026-07-08 03:50 UTC | 首批 review（@LCAIZJ / @MengqingCao / @wangxiyuan / @linfeng-yuan） |
| 2026-07-08 06:31 UTC | 测试补充 commit |
| 2026-07-08（同日） | 合并入 main，合并 sha `ae870bc5e265a340912cde392f23dad3671a0a88` |

---

> 本案例核心教训：**当 hybrid 模型的一个 cache group 被 split 成多个 transfer group 时，Mooncake 传输元数据的 `local/remote_block_ids` 必须按 `kv_cache_group_id` 构造与索引，而不能用 transfer group 的循环下标 `append`——否则 send 侧元数据与 receive 侧索引语义错位，KV 数据会被写入错误的 cache（attention ↔ mamba/compress），引发 silent 精度腐败或段错误。**