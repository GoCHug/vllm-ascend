# Issue #7792 深度案例：mooncake kv_both + GLM5 场景 Transfer slice failed (503900)

> 整理时间: 2026-07-29
>
> 案例对象: [vllm-project/vllm-ascend#7792](https://github.com/vllm-project/vllm-ascend/issues/7792)
>
> 标题: [Bug]: mooncake kv_both+GLM5场景，报错Transfer slice failed with status: 503900
>
> 关联全景文档: [0_kvcache.md](./0_kvcache.md) §2.3（传输连续性 / 布局不匹配，案例 7）
>
> 关键词: mooncake · kv_both · GLM5 · Transfer slice failed · status 503900 · 传输失败 · non-contiguous

---

## 1. Issue 概览

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| Issue 编号 | [#7792](https://github.com/vllm-project/vllm-ascend/issues/7792) |
| 类型 | Bug |
| 问题作者 | （Issue 报告者，未在元数据中显式署名） |
| 状态 | **closed** |
| 严重度 | 🟡 中 |
| 创建时间 | 2026-03-28 |
| vLLM 版本 | v0.17.0 |
| vLLM-Ascend 版本 | 0.17.0rc2.dev0+ge20f0b1a0.d20260320 |
| 硬件 | Ascend 910B3 ×8（aarch64, Kunpeng-920, CANN 8.5.1） |
| Mooncake 协议 | `ascend`（ ascend_direct_transport） |
| 触发配置 | PD-mixed (`kv_role=kv_both`) + GLM-5-W4A8 + Mooncake multi-instance |
| 复现文档 | [pd_colocated_mooncake_multi_instance.md](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_colocated_mooncake_multi_instance.html) |
| 直接修复 PR | 无单一“一对一”修复 PR；本 Issue 是 §9 所列多条 Mooncake 传输修复链的**源头需求**（#12183 / #11601 / #11886） |

> 本 Issue 是 vllm-ascend Mooncake KV 传输正确性问题矩阵的“发令枪”：它最早报告了 hybrid 模型 + kv_both 双向模式下的 slice 传输失败，随后三个月内社区围绕“非连续输入 / cache group 元数据 / KV head 归属”连续产出多个修复，构成一个完整的根因收敛闭环。

---

## 2. 场景背景

### 2.1 Mooncake kv_both：双向 send+recv 模式

vLLM 的 `kv_transfer_config.kv_role` 有三种取值：

- `kv_producer`：纯 P 端，只发送 KV Cache
- `kv_consumer`：纯 D 端，只接收 KV Cache
- **`kv_both`**：同一个节点**既生产又消费**（PD-mixed / colocated 模式）

`kv_both` 用于单机多实例 co-located 部署（[pd_colocated_mooncake_multi_instance.md](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_colocated_mooncake_multi_instance.html)），实例间互为 P/D，通过 Mooncake TransferEngine（`ascend_direct_transport`）在 NPU 显存间直接搬运 KV Cache block。

关键点：`kv_both` 节点同一时刻既会调用 `put`（写入远端）又会调用 `get`（拉取远端），**两个方向的 slice 地址计算都走同一套 `src_list/dst_list/length_list` 构造逻辑**（见 §4.2）。任何一侧的 block 布局/stride/地址错配，都会让 TransferEngine 拿到非法 slice。

### 2.2 GLM5 模型架构：hybrid attention + linear/SSM

GLM5（`longcat_flash` / `longcat_flash_ngram` model_type）是 **hybrid 架构**模型，其层结构混合了：

- **标准 GQA attention 层**（K/V cache）
- **线性注意力 / SSM / Mamba-like 状态层**（state cache，非传统 KV block）

在 vllm-ascend 的 Mooncake connector 中，hybrid 模型有专门的判定与处理分支（`mooncake_connector.py`）：

```python
# mooncake_connector.py:1141
num_attn_module = 2 if model_type in ("longcat_flash", "longcat_flash_ngram") else 1
```

`num_attn_module = 2` 意味着 GLM5 每个“层索引”下实际有 **2 个 attention 子模块**，layer index 提取逻辑与其他模型不同——这是 GLM5 特有的 KV 层布局复杂度来源。

此外，hybrid 模型的 KV cache group 中可能同时包含 `MambaSpec`（状态组）和普通 attention 组：`mooncake_connector.py:807` 用 `is_mamba_group = group_spec["kv_cache_spec_type"] == "MambaSpec"` 区分两条完全不同的 slice 构造路径，`_append_mamba_transfer_meta` 负责计算 conv/ssm 的 `local_conv_addr` / `local_ssm_addr` 等独立地址。

### 2.3 status 503900：Mooncake TransferEngine 错误码

报错来自 Mooncake 的 C++ 传输层，**不在 vllm-ascend Python 代码内**：

```
E0328 09:06:16.412760 112112 ascend_direct_transport.cpp:836] Transfer slice failed with status: 503900
```

`ascend_direct_transport.cpp:836` 是 Mooncake `AscendDirectTransport` 的 slice 传输实现。`503900` 是 Ascend 侧（CANN / HCCL / HCCN）返回给 TransferEngine 的错误码，表示底层一次 RDMA/直接搬运 slice 操作失败。Mooncake 把一次大块传输拆成多个 **slice**（一段连续 `(src_addr, dst_addr, length)` 描述符）下发，slice 本身的地址/长度/对齐若非法，底层驱动即拒绝并返回此类状态码。

Python 侧表现为 `mooncake_backend.py:85` 的 `Failed to put key ..., res:[-800]`（`-800` 是 Mooncake store 层包装后的 `TRANSFER_FAIL`）。

---

## 3. 现象描述

### 3.1 触发条件

- 部署模式：PD-mixed colocated Mooncake multi-instance（`kv_role = "kv_both"`）
- 模型：**GLM-5-W4A8**（`longcat_flash`，W4A8 量化）
- 操作：按官方文档 [pd_colocated_mooncake_multi_instance](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_colocated_mooncake_multi_instance.html) 启动并发起请求

### 3.2 报错信息原文（取自 Issue body）

```
E0328 09:06:16.412760 112112 ascend_direct_transport.cpp:836] Transfer slice failed with status: 503900
I0328 09:06:16.412804 112112 ascend_direct_transport.cpp:843] transfer failed and disconnect to:100.100.135.188:20081
E0328 09:06:16.412825 114360 client_service.cpp:1126] Transfer failed for key 0318-GLM-5-w4a8@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@4c67fd4f...: TRANSFER_FAIL (Transfer 0 failed)
I0328 09:06:16.413093 114360 client_service.cpp:1227] Successfully revoked failed put for key 0318-GLM-5-w4a8@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@4c67fd4f...
E0328 09:06:16.413112 114360 client_service.cpp:1265] Operation for key 0318-GLM-5-w4a8@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@4c67fd4f... failed: TRANSFER_FAIL (TRANSFER_FAIL: Transfer 0 failed; )
(Worker pid=107182) (Worker_TP2 pid=107182) ERROR 03-28 09:06:16 [mooncake_backend.py:85] Failed to put key ['0318-GLM-5-w4a8@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@4c67fd4f...'],res:[-800]
```

同一时刻多个 TP rank（TP1/TP2/TP6…）并发报 `503900`，目标端口分散（`100.100.135.188:20081 / 20081 / 20137`），表明**多 rank 同时写远端时全部 slice 失败**，不是单点网络问题。

报错后引擎未崩溃，日志仍打印吞吐（`Avg prompt throughput: 102.9 tokens/s, Avg generation throughput: 0.9 tokens/s`），`pool_scheduler` 显示 `Delaying free of 9 blocks`——说明 KV block 分配/调度在继续，但传输链路已被 `TRANSFER_FAIL` 中断，KV Cache 实际无法到位。

### 3.3 失败键的结构

键格式：`0318-GLM-5-w4a8@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@<hash>`

- `pcp0@dcp0`：PCP（Prefill Cache Producer）→ DCP（Decode Cache Consumer）实例对，正是 `kv_both` 互传场景
- `head_or_tp_rank:0` / `pp_rank:0`：TP/PP rank 标识
- 末尾 hash：block 内容指纹

键结构本身正常，失败发生在“键 → slice 下发”环节，即**给定正确的键，但底层数据搬运 slice 非法**。

---

## 4. 根因分析

### 4.1 结论与证据等级

> [!NOTE]
> 本 Issue **无单一直接修复 PR**，根因为**推断 + 证据链**：基于 Issue 现象 + 本地源码 + 后续同簇修复 PR 的 converging evidence。下文区分“已确认”与“推断”。

| 结论 | 证据等级 |
|------|----------|
| slice 地址/长度/对齐非法 → 503900 | 🟢 已确认（错误码语义 + 源码路径） |
| GLM5 hybrid 布局 + kv_both 触发非连续 / cache-group 元数据缺口 | 🟡 推断（强证据，见 §4.3–§4.5） |
| 具体是哪一条 slice 计算分支出错 | 🟡 推断（无复现栈，但与 #12183/#11601/#11886 修复方向一致） |

### 4.2 slice 构造路径（核心代码）

`mooncake_connector.py:780–909` 构造 `src_list/dst_list/length_list` 三元组，下发给 Mooncake TransferEngine。attention 组的关键计算：

```python
# mooncake_connector.py:897-909
transfer_remote_block_ids, transfer_local_block_ids = split_if_not_byte_contiguous(
    grouped_remote_block_ids,
    grouped_local_block_ids,
    src_block_stride=remote_block_stride,
    dst_block_stride=block_stride,
    block_len=inner_block_len,
)
for remote_block_id, local_block_id in zip(transfer_remote_block_ids, transfer_local_block_ids):
    src = src_layer_base_addr + local_block_id[0] * block_stride + inner_offset * inner_block_len
    dst = dst_layer_base_addr + remote_block_id[0] * remote_block_stride
    length = inner_block_len * len(local_block_id)
    src_list.append(src); dst_list.append(dst); length_list.append(length)
```

其中 `block_stride` / `remote_block_stride` / `inner_block_len` / `inner_offset` 全部来自 **cache group 的元数据与 KV head 归属**（`block_stride_per_addr` / `block_len_per_addr` / `remote_block_stride_per_addr`）。

### 4.3 根因一：non-contiguous PA cache 输入（→ #12183 修复方向）

`split_if_not_byte_contiguous`（`mooncake_connector.py:3624`）依赖 `group_concurrent_contiguous` 按“src/dst 字节连续性”重新切分 block 组：

```python
# mooncake_connector.py:3612-3616
src_byte_contiguous = np.diff(src_indices) * src_block_stride == block_len
dst_byte_contiguous = np.diff(dst_indices) * dst_block_stride == block_len
brk = np.where(~(src_byte_contiguous & dst_byte_contiguous))[0] + 1
src_groups = np.split(src_indices, brk)
```

GLM5 的 PA（Prefix Attention）cache 在 hybrid 布局下，block 在内存中**不保证按 id 字节连续**——尤其是跨 attention/linear 子层、跨 cache group 时，`src_block_stride != block_len` 且 `dst_block_stride != block_len`。

**推断**：本 Issue 触发时，`split_if_not_byte_contiguous` 未能正确处理 GLM5 的非连续 PA 输入，导致下发 slice 的 `src`/`length` 跨越了非连续内存边界 → TransferEngine 在 `ascend_direct_transport.cpp:836` 检测到非法 slice → 返回 `503900`。

这一推断被 **#12183**（Fix non-contiguous Mooncake PA cache inputs，2026-07-16 merged）直接佐证——该 PR 正是修复“Mooncake 传输假设输入连续、但 PA cache 实际非连续导致读取错位”的问题。

### 4.4 根因二：cache group ids / KV head 元数据缺失（→ #11601 / #11886 修复方向）

slice 计算依赖 `kv_group2layeridx` 中的 `kv_cache_group_id`、`tp_num_need_pulls`、`inner_offset`（远端 TP 偏移）。在 `kv_both` 双向模式下，节点同时扮演 P 与 D，**两端的 cache group 划分、KV head 归属必须严格对齐**，否则 `inner_offset * inner_block_len` 与 `remote_block_stride` 会算出越界地址。

```python
# mooncake_connector.py:800-806
group_idx = group_pull.group_id
group_spec, layer_indices = self.kv_group2layeridx[group_idx]
kv_cache_group_id = group_spec.get("kv_cache_group_id", group_idx)
...
tp_num_need_pulls = group_pull.num_group_pulls
inner_offset = group_pull.remote_tp_offset
```

**推断**：GLM5 的 hybrid 层被打包成多个 cache group（attention 组 + state 组），而当时的 Mooncake split metadata **未携带显式的 cache group id 与 total KV heads**，导致 `kv_both` 互传时两端对同一 group 的 `tp_num_need_pulls` / `inner_offset` 认知不一致 → slice 偏移/长度错误 → 503900。

这一推断被两条后续 PR 佐证：
- **#11601**（Use cache group ids in Mooncake split metadata，2026-07-08）
- **#11886**（Carry explicit total KV heads in Mooncake transfer groups，2026-07-12）

两者共同补齐了“传输元数据不足以正确划分 slice”的缺口，而 GLM5 的多 cache group + `num_attn_module=2` 正是暴露该缺口的最复杂模型。

### 4.5 根因三：hybrid Mamba/SSM state 组的独立地址路径

GLM5 的 Mamba/SSM 组走 `_append_mamba_transfer_meta`（`mooncake_connector.py:1051`），独立计算 conv/ssm 地址：

```python
# mooncake_connector.py:1130-1134
src_list.append(local_ssm_addr + local_block_id * local_ssm_stride + remote_tp_offset * local_ssm_len // tp_num_need_pulls)
dst_list.append(remote_ssm_addr + remote_block_id * remote_ssm_stride)
length_list.append(remote_ssm_len)
```

该路径的 `local_ssm_len` / `remote_ssm_stride` 依赖远端上报的 Mamba 元数据，且要求 `remote_tp_size >= self.tp_size` 且整除（`mooncake_connector.py:1068-1069`）。

**推断**：在 `kv_both` 模式下，若两端 GLM5 的 state 层布局（conv kernel size / ssm state len）因量化（W4A8）或配置差异不完全一致，ssm slice 的 `length` / `stride` 就会错配 → 503900。这条路径在 Issue 发生时（2026-03，v0.17.0rc2）尚未有专门的 hybrid-state 元数据校验，属于最隐蔽的分支。

### 4.6 为什么是“kv_both + GLM5”组合才触发

| 维度 | 普通 PD（producer/consumer 分离） | kv_both + GLM5 |
|------|------|------|
| 模型布局 | 纯 attention（单 cache group，stride 规整） | hybrid（attention + Mamba/SSM，多 cache group） |
| 传输方向 | 单向，元数据生产端→消费端单向同步 | 双向，两端互为 P/D，元数据需双向对齐 |
| block 连续性 | 通常连续（TP 相等、prefill 整块） | PA cache 命中后非连续块多 |
| slice 计算 | stride == block_len，走快路径 | stride != block_len，走 split/non-contiguous 路径 |
| 元数据口径 | 单侧 config 即可 | 需双侧 group id / total heads / TP offset 一致 |

三者叠加（hybrid 布局 × 双向互传 × 非连续 PA）正是 503900 的触发组合，任一条件退化（非 hybrid / 非 kv_both / 块连续）都会走快路径而不报错——这也是 CI 难以提前覆盖的原因。

---

## 5. 影响与表现

| 维度 | 表现 |
|------|------|
| 触发配置 | `kv_role=kv_both` + GLM-5（`longcat_flash`/`longcat_flash_ngram`）+ Mooncake colocated multi-instance |
| 不触发 | 纯 `kv_producer`/`kv_consumer` + 非 hybrid 模型 / 块连续的常规 PD 分离 |
| 现象 | `Transfer slice failed with status: 503900`；Mooncake `put` 返回 `-800` (TRANSFER_FAIL)；多 TP rank 并发失败 |
| 引擎行为 | 不崩溃，吞吐日志仍在打印，但 KV Cache 实际未传输到位 → 后续 decode 读到空/旧 KV → **silent 精度退化或无效输出** |
| 严重度 | 🟡 中（不崩溃，但功能受损； colocated 场景下影响可用性） |
| 隐蔽性 | 🟡 中高（需 hybrid 模型 + kv_both + PA 非连续同时满足；常规 CI 用例不覆盖） |
| 是否报错 | **报错**（503900，非 silent），但报错点在 Mooncake C++ 层，Python 侧仅见 `res:[-800]`，定位链路长 |

---

## 6. 修复方向 / 已知修复

本 Issue 没有单一的“一对一”修复 commit，而是被拆解为同簇多条 PR 共同收敛。下表按根因对应：

| 根因 | 修复 PR | 标题 | 状态 | 合并时间 |
|------|---------|------|------|----------|
| non-contiguous PA cache 输入 | [#12183](https://github.com/vllm-project/vllm-ascend/pull/12183) | [BugFix][KV Transfer] Fix non-contiguous Mooncake PA cache inputs | closed/merged | 2026-07-16 |
| cache group ids 元数据缺失 | [#11601](https://github.com/vllm-project/vllm-ascend/pull/11601) | [BugFix][KV Transfer] Use cache group ids in Mooncake split metadata | closed/merged | 2026-07-08 |
| total KV heads 元数据缺失 | [#11886](https://github.com/vllm-project/vllm-ascend/pull/11886) | [BugFix][PD] Carry explicit total KV heads in Mooncake transfer groups | closed/merged | 2026-07-12 |
| TP 不等 + MTP 层 KV（同簇先导） | [#8540](https://github.com/vllm-project/vllm-ascend/pull/8540) | [BugFix] [P/D] TP 不等时 MTP 层 KV cache 未处理 | closed/merged | 2026-04-23 |

### 6.1 修复思想

1. **#12183**：传输前显式处理非连续输入——对 PA cache 的 block list 在构造 slice 前确保连续性或在 slice 描述里正确反映 stride，而非假设 `src_block_stride == block_len`。
2. **#11601**：Mooncake split metadata 中显式携带 **cache group ids**，使两端对 hybrid 模型的 attention 组 / state 组划分一致。
3. **#11886**：传输组中显式携带 **total KV heads**，使 `tp_num_need_pulls` / `inner_offset` 在 kv_both 双向、TP 不等时正确计算。

三者合力把“slice 地址 = base + block_id × stride + offset × inner_len”这条公式的**每个变量都变成有显式元数据保证、不再依赖隐式假设**，从而消除 503900 的根因空间。

### 6.2 若需自查的临时规避（未合并到 Issue 官方回复，仅作参考）

- 切换到 `kv_producer` + `kv_consumer` 分离部署，规避双向元数据对齐问题
- 关闭 prefix caching，减少非连续 PA block
- 确保 P/D 端 TP 相等，避免 `inner_offset != 0` 的重排路径
- 升级到含 #12183/#11601/#11886 的 vllm-ascend 版本

---

## 7. 复现与验证

### 7.1 触发矩阵

| kv_both | GLM5 (hybrid) | PA cache 非连续 | 是否触发 503900 |
|:---:|:---:|:---:|:---:|
| ✗（producer/consumer 分离） | — | — | ✗ |
| ✓ | ✗（纯 attention 模型） | — | ✗（单 cache group，stride 规整） |
| ✓ | ✓ | ✗（首请求无 prefix 命中） | 可能不触发 |
| **✓** | **✓** | **✓** | **✓ 触发** |

### 7.2 复现步骤（依据 Issue 引用的官方文档）

1. 按 [pd_colocated_mooncake_multi_instance](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_colocated_mooncake_multi_instance.html) 部署 GLM-5-W4A8，`kv_role=kv_both`，Mooncake multi-instance。
2. 发起带一定 prefix 复用的请求（触发 PA cache 非连续 block 命中）。
3. 观察 Mooncake 日志：`ascend_direct_transport.cpp:836` 出现 `Transfer slice failed with status: 503900`，Python 侧 `mooncake_backend.py:85` 出现 `Failed to put key ..., res:[-800]`。

### 7.3 验证修复

升级至含 #12183 / #11601 / #11886 的版本后，相同配置下：
- `503900` 不再出现
- `put` 返回成功，`res:[0]`
- D 端实际收到完整 KV Cache，decode 输出正常

建议回归用例：PD-mixed (kv_both) + GLM5 hybrid + prefix cache 复用 + TP 不等 的 e2e accuracy 与传输成功率回归。

---

## 8. 经验与启发

### 8.1 kv_both 双向模式是隐藏的元数据对齐风险面

`kv_both` 让一个节点同时承担 P 与 D，**两端的 cache group 划分、KV head 归属、TP offset 必须双向一致**。单向 PD（producer/consumer）只需单侧 config 一致，而 kv_both 把“元数据同步”从单向问题升级为双向一致性问题。任何涉及 `kv_both` 的改动，都应把“双侧元数据对齐”当作独立测试维度（参见 [0_kvcache.md](./0_kvcache.md) §2.3、§8）。

### 8.2 hybrid 模型布局是 Mooncake 传输的高危触发器

GLM5（`longcat_flash`，`num_attn_module=2`）的多 cache group + Mamba/SSM 独立地址路径，使 slice 计算从“单条公式”变成“多分支多地址”。任何“默认连续 / 默认单 group”的隐式假设都会在 hybrid 模型上失效。后续类似 hybrid 架构（attention + Mamba/linear）模型接入时，应默认按“非连续 + 多 group + state 组”构造回归用例，而非等 Issue 报出来再补（参见 [0_kvcache.md](./0_kvcache.md) §4 hybrid 布局问题）。

### 8.3 Mooncake 503900 的排查范式

`503900` 来自 C++ 传输层，Python 侧只见 `res:[-800]`。排查此类错误的范式：
1. **确认 slice 三元组**：从 `src_list/dst_list/length_list` dump 出失败 slice 的 `src`/`dst`/`length`。
2. **校验地址合法性**：`src`/`dst` 是否在已 `register_memory` 的段内、`length` 是否越界、是否对齐。
3. **回溯元数据**：`block_stride` / `inner_block_len` / `inner_offset` / `tp_num_need_pulls` 来自哪条 cache group 元数据，两端是否一致。
4. **对照非 hybrid 基线**：换成纯 attention 模型是否复现 → 锁定 hybrid 布局分支。
5. **对照单向基线**：从 `kv_both` 切到 `kv_producer`/`kv_consumer` 是否复现 → 锁定双向元数据对齐问题。

### 8.4 “slice 地址公式”的每个变量都需显式元数据

`src = base + block_id × stride + offset × inner_len` 这条公式中，`base` / `stride` / `inner_len` / `offset` 在初期都靠隐式 config 推导。#11601 / #11886 的修复方向是**把 group id 与 total KV heads 显式塞进传输元数据**，让消费端不再“猜”。这是分布式 KV 传输正确性的通用范式：地址计算的所有变量都应由显式协议字段保证，而非依赖两端 config 标量巧合一致。

### 8.5 一个 Issue 可以驱动一整条修复链

本 Issue 报告于 2026-03-28，随后三个月内社区产出 #8540（04-23，MTP/TP 不等）、#11601（07-08，cache group ids）、#11886（07-12，total KV heads）、#12183（07-16，non-contiguous）四条 PR，覆盖了 slice 公式的几乎所有变量。这种“一个现象、多个根因、一簇修复”的模式，说明 Mooncake KV 传输的早期正确性缺口是**系统性的元数据/连续性假设问题**，而非单点 bug——本 Issue 是该系统性收敛的起点。

---

## 9. 关联问题

| 编号 | 关联点 |
|------|--------|
| [0_kvcache.md](./0_kvcache.md) §2.3 | 本案例在全景文档中的归档位置（传输连续性 / 布局不匹配，案例 7） |
| [#12183](https://github.com/vllm-project/vllm-ascend/pull/12183) | Non-contiguous Mooncake PA cache inputs — 直接修复根因一（非连续输入） |
| [#11601](https://github.com/vllm-project/vllm-ascend/pull/11601) | Use cache group ids in Mooncake split metadata — 修复根因二（group id 元数据） |
| [#11886](https://github.com/vllm-project/vllm-ascend/pull/11886) | Carry explicit total KV heads in Mooncake transfer groups — 修复根因二（KV head 元数据） |
| [#8540](https://github.com/vllm-project/vllm-ascend/pull/8540) | TP 不等 + MTP 层 KV cache 未处理 — 同簇先导修复（layers 口径），[详见本目录 1_pr8540_tp_unequal_mtp_kv.md](./1_pr8540_tp_unequal_mtp_kv.md) |
| [#10569](https://github.com/vllm-project/vllm-ascend/issues/10569) | Mooncake 传输相关 issue（同簇） |
| [pd_colocated_mooncake_multi_instance 官方文档](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_colocated_mooncake_multi_instance.html) | 本 Issue 的复现操作依据 |

---

## 10. 结论

本 Issue 是 vllm-ascend Mooncake KV 传输正确性的**首个系统性信号**：在 `kv_both` 双向模式 + GLM5 hybrid 布局 + PA cache 非连续三者叠加下，slice 地址计算的多个变量（stride / offset / group id / total heads）因缺乏显式元数据保证而错配，底层 TransferEngine 以 `503900` 拒绝非法 slice。它没有单一修复 commit，而是驱动了 #8540 / #11601 / #11886 / #12183 一整条同簇 PR 链，把 slice 公式的每个变量从“隐式 config 推导”升级为“显式元数据保证”。

> **核心教训：跨节点 KV 传输中，凡是喂给底层 RDMA/slice 引擎的地址、长度、stride、offset，都必须由显式的传输元数据字段保证两端一致——任何“默认连续 / 默认单 group / 默认 config 对齐”的隐式假设，都会在 hybrid 模型 + kv_both 双向模式 + 非连续 PA 的组合下崩溃，表现为 C++ 层 503900 而 Python 层无从定位的传输失败。**