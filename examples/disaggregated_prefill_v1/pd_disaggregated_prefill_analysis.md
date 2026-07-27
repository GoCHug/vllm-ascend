# PD 分离场景下首 token 处理机制详细分析

> 本文档分析 `load_balance_proxy_server_example.py` 在 PD（Prefill-Decode）分离部署场景下，
> 首 token 由 D 节点（decoder）吐出的完整机制，以及 D 节点如何利用迁移而来的 KV cache
> 避免重新执行整段 prompt 的 prefill forward。
>
> 涉及代码：
> - 代理服务器：[`load_balance_proxy_server_example.py`](./load_balance_proxy_server_example.py)
> - vLLM V1 调度器：`vllm/vllm/v1/core/sched/scheduler.py`
> - KV Connector 基类：`vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`

---

## 目录

- [1. 架构概览](#1-架构概览)
- [2. 两阶段请求处理](#2-两阶段请求处理)
  - [2.1 Phase 1 — Prefill 阶段（P 节点）](#21-phase-1--prefill-阶段p-节点)
  - [2.2 Phase 2 — Decode 阶段（D 节点）](#22-phase-2--decode-阶段d-节点)
- [3. 首 token 由 D 节点吐出的机制](#3-首-token-由-d-节点吐出的机制)
- [4. D 节点 KV cache 利用机制：为何不需要重做 prefill](#4-d-节点-kv-cache-利用机制为何不需要重做-prefill)
  - [4.1 决定性证据：full prompt hit 只算 1 个 token](#41-决定性证据full-prompt-hit-只算-1-个-token)
  - [4.2 完整机制链](#42-完整机制链)
  - [4.3 边界情况](#43-边界情况)
- [5. 端到端时序图](#5-端到端时序图)
- [6. 异常兜底：recomputed 重试机制](#6-异常兜底recomputed-重试机制)
- [7. 设计意图](#7-设计意图)
- [8. 代码索引](#8-代码索引)

---

## 1. 架构概览

该代理服务器（proxy）将一个客户端请求拆分为两个阶段，分别由两类后端节点处理：

| 阶段 | 执行节点 | 角色 | 通信方式 | 产出的 token |
|------|---------|------|---------|-------------|
| Phase 1 — Prefill | P 节点 (prefiller) | 对 prompt 做 prefill 计算 + 生成 KV cache 并传输 | **非流式** (stream=False) | 1 个 token（**被代理丢弃**） |
| Phase 2 — Decode | D 节点 (decoder) | 接收 KV cache，执行 decode 并流式输出 | **流式** (保持客户端原始 stream 设置) | 客户端可见的所有 token |

关键结论：

- 从客户端视角看，**所有 token（包括第一个）都来自 D 节点的流式输出**，P 节点的产出对客户端不可见。
- D 节点拿到 KV cache 后**不重算整段 prompt 的 prefill forward**，只需对最后 1 个位置做一次前向以采样首个 decode token。

---

## 2. 两阶段请求处理

### 2.1 Phase 1 — Prefill 阶段（P 节点）

#### 2.1.1 请求改造：`build_prefill_request`

> `load_balance_proxy_server_example.py:818-834`

代理在将请求发给 P 节点前，对请求体做了关键改造：

```python
def build_prefill_request(req_data: dict) -> dict:
    payload = req_data.copy()                          # 注意：操作的是副本，不影响原始 req_data
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,                      # 告知 P 节点：decode 将在远端进行
        "do_remote_prefill": False,                    # P 节点自己做 prefill
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    payload["stream"] = False                          # ← 强制非流式
    payload["max_tokens"] = 1                          # ← 只生成 1 个 token
    payload["min_tokens"] = 1
    ...
```

三个关键变换：

1. **`max_tokens = 1`**：P 节点只生成 **1 个 token**。这是 prefill 的副产物 —— prefill 本质是对整个 prompt 做一次前向传播，最后一个位置的 logits 自然预测出第一个 token，所以这 1 个 token 是"免费的"。
2. **`stream = False`**：强制非流式响应。P 节点的 1 个 token 会被完整缓冲在 HTTP 响应体中返回，而非逐块流式推送。
3. **`do_remote_decode = True`**：告知 P 节点 decode 不在本地进行，需要准备 KV cache 传输。

#### 2.1.2 P 节点响应处理：只取 KV 传输参数，丢弃 token 内容

> `load_balance_proxy_server_example.py:941-957`

```python
response = await send_request_to_service(...)   # 非流式，等待完整响应
response_json = response.json()
kv_transfer_params = response_json.get("kv_transfer_params", {})
if kv_transfer_params:
    req_data["kv_transfer_params"] = kv_transfer_params   # ← 把 KV 传输参数注入原始请求
prefiller_cached_tokens = extract_cached_tokens(response_json)  # 只提取命中缓存数
```

P 节点的完整响应（包含 `choices` 中那 1 个 token 的内容）被 `response.json()` 解析后，**只有两样东西被提取**：

- `kv_transfer_params`：包含 P 节点引擎 ID、KV block IDs、远端 host/port —— D 节点凭此拉取 KV cache。
- `cached_tokens`：prefix cache 命中统计（用于后续 usage 上报）。

**那 1 个 token 的文本内容（`choices[0].text` 或 `choices[0].delta.content`）被完全丢弃**，不转发给客户端。

### 2.2 Phase 2 — Decode 阶段（D 节点）

#### 2.2.1 D 节点选分配

> `load_balance_proxy_server_example.py:959-977`

```python
decoder = await runtime.schedule("pick_decoder", decoder_score)
# ...
return InstanceInfo(
    request_id=request_id,
    prefiller_key=prefiller_key,
    prefiller_score=prefiller_score,
    decoder_key=decoder["key"],
    decoder_score=decoder_score,
    decoder_host=decoder["host"],
    decoder_port=decoder["port"],
    prefiller_cached_tokens=prefiller_cached_tokens,
)
```

D 节点被选中后，代理准备将**原始 `req_data`**（保留客户端原始的 `max_tokens`、`stream` 设置）发给 D 节点，其中已经注入了 P 节点返回的 `kv_transfer_params`。

#### 2.2.2 D 节点收到的请求内容

D 节点收到的请求包含：

- **原始 prompt**（未被 `build_prefill_request` 改造，因为那是在副本上操作的）
- **原始 `max_tokens`**（如客户端设 16 就是 16）
- **`kv_transfer_params`**（来自 P 节点响应）：`remote_engine_id`、`remote_block_ids`、`remote_host`、`remote_port`

凭借这些参数，D 节点的 vLLM 后端会：

1. 通过 Nixl/Mooncake connector 从 P 节点**拉取 KV cache**（远程 prefill）。
2. 利用已就位的 KV cache 跳过本地 prefill 计算，直接从 cache 对应的位置开始 decode。
3. 逐 token 流式生成并返回。

#### 2.2.3 流式转发：客户端的首 token 来自 D 节点

> `load_balance_proxy_server_example.py:1031-1100`

```python
async def generate_stream():
    ...
    while retry:
        retry = False
        decoder_client = await runtime.get_client(ServerRole.DECODE, instance_info.decoder_key)
        async for chunk in stream_service_response_with_retry(
            decoder_client,          # ← 请求发往 D 节点
            api,
            req_data,
            request_id=instance_info.request_id,
            ...
        ):
            if not released_kv and chunk:
                await release_prefill_kv_once()    # 收到 D 节点首个 chunk 后，释放 P 节点 KV 占用
            ...
            yield chunk    # ← 将 D 节点的 chunk 直接转发给客户端
```

关键点：

- 代理与 D 节点之间是**流式**连接（`stream_service_response_with_retry` 内部用 `client.stream("POST", ...)`）。
- D 节点每生成一个 token 就推一个 chunk，代理收到后立即 `yield chunk` 转发给客户端。
- **客户端接收到的第一个 chunk → 来自 D 节点 → 这就是首 token**。

因此，尽管 P 节点在 prefill 时也产出了 1 个 token，但该 token 在非流式响应中被缓冲、被代理解析后丢弃。客户端永远不会看到 P 节点产出的 token。**从端到端角度，首 token 由 D 节点吐出。**

---

## 3. 首 token 由 D 节点吐出的机制

| 考量 | 说明 |
|------|------|
| **流式协议一致性** | 客户端只需面对一个流式数据源（D 节点），无需处理"P 节点 1 个 token + D 节点剩余 token"的拼接逻辑。 |
| **P 节点专注 prefill** | P 节点是计算密集型（处理长 prompt），设 `max_tokens=1` 仅为了触发 prefill 前向传播和 KV cache 生成，token 本身是副产物。 |
| **KV 传输时序** | P 节点必须先完成 prefill 才能产出 KV cache，非流式模式下代理拿到完整响应（含 `kv_transfer_params`）后才开始 D 节点分配，时序清晰。 |
| **快速释放 P 资源** | D 节点首个 chunk 到达后，代理立即 `release_prefill_kv`，P 节点的 KV 压力被释放，可服务下一个请求。 |

---

## 4. D 节点 KV cache 利用机制：为何不需要重做 prefill

### 4.1 决定性证据：full prompt hit 只算 1 个 token

> `vllm/vllm/v1/core/sched/scheduler.py:2406-2409`

D 节点 KV 拉取完成后的处理：

```python
# on a full prompt hit, we need to re-compute the last token
# in order to be able to sample the next token
if request.num_computed_tokens == request.num_tokens:
    request.num_computed_tokens = request.num_tokens - 1
```

这段注释说得很直白：当全部 prompt 的 KV 都命中（full prompt hit），为了让 sampler 能采样下一个 token，需要"re-compute the last token"。实现方式是把 `num_computed_tokens` **回退 1**，于是下一步调度：

```
num_new_tokens = num_tokens - num_computed_tokens = 1
```

model runner 只对**最后 1 个位置**做一次前向（query 是这 1 个新 token，key/value 直接复用 cache 里整段 prompt 的 KV），采样出第一个 decode token。

**这就是"decode 的第一步"，不是"重做 prefill"。** prefill forward 要对整段 prompt（N 个位置）做 attention；而这里只算 1 个位置。

### 4.2 完整机制链

D 节点收到代理转发来的请求（带 `kv_transfer_params`，含 P 节点的 `remote_engine_id`/`remote_block_ids`/`remote_host`/`remote_port`）后：

#### 步骤 1 — connector 报告可拉取的 KV 量

> `vllm/vllm/v1/core/sched/scheduler.py:727-751`

```python
ext_tokens, load_kv_async = (
    self.connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)
)
num_external_computed_tokens = ext_tokens   # 从 P 节点能拉到的 KV token 数
num_computed_tokens = (
    num_new_local_computed_tokens + num_external_computed_tokens
)
```

D 节点的 connector 通过 `get_num_new_matched_tokens` 告诉 scheduler："这个请求有 `ext_tokens` 个 token 的 KV 在外部（P 节点）已经备好可以拉过来。"（基类定义见 `vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py:454-486`）

#### 步骤 2 — 异步拉取，进入等待态

> `vllm/vllm/v1/core/sched/scheduler.py:938-958`

```python
if load_kv_async:
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    # Set num_computed_tokens even though KVs are not yet loaded.
    request.num_computed_tokens = num_computed_tokens
```

KV 还没真正拉到本地 buffer，但 scheduler 已经"记账"——把这些 token 算作已计算。请求进入 `WAITING_FOR_REMOTE_KVS`，等 connector 异步把 KV 从 P 节点搬到 D 节点的 paged buffer。

#### 步骤 3 — 拉取完成，缓存 block 并回退 1 token

> `vllm/vllm/v1/core/sched/scheduler.py:2379-2409`

KV 到位后：缓存这些 block；若全命中则 `num_computed_tokens` 回退 1（即 4.1 节的代码）。

#### 步骤 4 — 重新调度，只算 1 个 token

回到 `WAITING` 后调度时：

> `vllm/vllm/v1/core/sched/scheduler.py:799`

```python
num_new_tokens = request.num_tokens - num_computed_tokens   # = 1
```

只对这个 token 跑一次前向 → 产出第一个 decode token → 之后正常 decode。

### 4.3 边界情况

| 情况 | 行为 | 代码位置 |
|------|------|---------|
| **部分命中**（只拉到部分 KV） | D 节点补算未命中的差额部分（partial prefill），`num_new_tokens` = 未命中数 | `scheduler.py:799` |
| **KV 加载失败**（P 节点 KV 已被驱逐/连接失败） | 按 `recompute_kv_load_failures` 策略，释放 block、`num_computed_tokens` 归零，从零重做完整 prefill | `scheduler.py:129,144,2389-2399` |

注意第二种情况——**这才是真正的 "recompute"**，也正是代理侧 `stop_reason == "recomputed"` 重试机制的来源。

### 4.4 精确表述

> "Decode 节点上是又需要重新进行一次 prefill 的 forward 吗，只是说已经有 kvcache 了"

精确表述应是：

- **不需要重算整段 prompt 的 prefill forward。** D 节点靠 connector 把 P 节点算好的 KV 搬进自己的 paged buffer，相当于这些位置的 K/V 已存在 cache 中。
- **唯一要算的是最后 1 个位置**（full prompt hit 时回退 1 token 那步）：用新 token 的 query 对 cache 里整段 prompt 的 K/V 做 attention，采样出第一个 decode token。这是一次"1-token 前向"，开销与一次 decode step 相当，而非 N-token 的 prefill forward。
- **只有当 KV 拉取失败时**，D 节点才会真正退化成对整段 prompt 做 prefill forward（recompute），并触发代理的重试。

所以 PD 分离的收益成立的关键正是：**KV cache 通过 Nixl/Mooncake 在节点间迁移，让 D 节点免去了对长 prompt 的重复 prefill 计算**，只需承担 decode 阶段的逐步前向。

---

## 5. 端到端时序图

```
┌──────────┐      ┌───────────┐      ┌──────────────────┐      ┌───────────────┐
│  客户端   │      │   Proxy   │      │  P节点(Prefill)  │      │ D节点(Decode)  │
└────┬─────┘      └─────┬─────┘      └────────┬─────────┘      └───────┬───────┘
     │ ① POST请求        │                      │                          │
     │──────────────────>│ ② 选P节点             │                          │
     │                   │──────────────────────>│                          │
     │                   │ ③ 改造后请求          │                          │
     │                   │   max_tokens=1        │                          │
     │                   │   stream=False        │                          │
     │                   │──────────────────────>│ ④ prefill + 生成KV       │
     │                   │                      │  (1 token丢弃)           │
     │                   │                      │──────────┐               │
     │                   │                      │          │               │
     │                   │                      │<─────────┘               │
     │                   │ ⑤ kv_transfer_      │                          │
     │                   │    params            │                          │
     │                   │<─────────────────────│                          │
     │                   │                      │                          │
     │                   │ ⑥ 选D节点             │                          │
     │                   │────────────────────────────────────────────────>│
     │                   │ ⑦ 原始请求 +          │                          │
     │                   │    kv_transfer_       │                          │
     │                   │    params             │                          │
     │                   │────────────────────────────────────────────────>│
     │                   │                      │ ⑧ 拉取KV                │
     │                   │                      │<─────────────────────────│
     │                   │                      │                          │ ⑨ 1-token前向
     │                   │                      │                          │  采样首token
     │                   │                      │                          │──────────┐
     │                   │                      │                          │          │
     │                   │                      │                          │<─────────┘
     │                   │ ⑩ 首token chunk      │                          │
     │                   │<───────────────────────────────────────────────│
     │                   │ ⑪ release_prefill_kv │                          │
     │                   │──────────────────────>│  (释放KV)               │
     │ ⑫ 首token         │                      │                          │
     │<──────────────────│                      │                          │
     │                   │                      │                          │  decode继续...
     │                   │  后续token...        │                          │
     │<──────────────────│<───────────────────────────────────────────────│
     │    ...            │    ...               │                          │    ...
     │                   │                      │                          │
┌────┴─────┐      ┌─────┴─────┐      ┌────────┴─────────┐      ┌───────┴───────┐
│  客户端   │      │   Proxy   │      │  P节点(Prefill)  │      │ D节点(Decode)  │
└──────────┘      └───────────┘      └──────────────────┘      └───────────────┘
```

| 阶段 | 步骤 | 核心动作 |
|:----:|:----:|----------|
| **Prefill** | ①~⑤ | 客户端请求 → 选P节点 → prefill生成KV → 返回`kv_transfer_params` |
| **Decode准备** | ⑥~⑨ | 选D节点 → 发原始请求 → 拉取KV → 1-token前向采样首token |
| **流式输出** | ⑩~⑫ | 首token返回 → 释放P节点KV → 转发客户端 → 后续token继续流式 |

---

## 6. 异常兜底：recomputed 重试机制

如果 D 节点拉取 KV cache 失败（P 节点 KV 已被驱逐），D 节点会返回 `stop_reason == "recomputed"`：

> `load_balance_proxy_server_example.py:1082-1093`

```python
if stop_reason == "recomputed":
    retry = True
    retry_count += 1
    # 把已生成 token 拼回 prompt，重新走 prefill → decode 流程
    req_data["prompt"] = origin_prompt + generated_token
    req_data["max_tokens"] = origin_max_tokens - completion_tokens + retry_count
    instance_info = await reassign_instances(...)  # 重新选 P 节点 + D 节点
    released_kv = False
    break
```

此时代理会：

1. 将已生成的 token 拼入 prompt。
2. 重新选一对 P/D 节点重走整个流程。
3. 重试中首 token 依然由新的 D 节点吐出。

`reassign_instances` 会先释放旧实例的负载，再以非初始请求模式（`is_initial_request=False`）重新分配：

> `load_balance_proxy_server_example.py:980-989`

```python
async def reassign_instances(...):
    await runtime.schedule("release_prefill_kv", previous_instance.prefiller_key, previous_instance.prefiller_score)
    await runtime.schedule("release_decoder", previous_instance.decoder_key, previous_instance.decoder_score)
    return await assign_instances(api, req_data, request_length, is_initial_request=False)
```

---

## 7. 设计意图

| 关注点 | 设计选择 | 理由 |
|--------|---------|------|
| P 节点 `max_tokens=1` | prefill 前向天然产出 1 token，无需额外开销 | 该 token 仅触发 prefill 计算 + KV cache 生成，是"免费的"副产物 |
| P 节点 `stream=False` | 非流式缓冲完整响应 | 简化 KV 传输参数提取，时序清晰（拿到完整响应才开始 D 节点分配） |
| P 节点 token 丢弃 | 不转发给客户端 | 保持客户端流式协议一致性，只面对 D 节点单一数据源 |
| D 节点 `do_remote_prefill` + KV 拉取 | connector 异步拉取 P 节点 KV | D 节点免去重复 prefill forward，直接复用 KV cache |
| D 节点 full hit 回退 1 token | `num_computed_tokens = num_tokens - 1` | 为了采样首个 decode token，需对最后 1 个位置做前向 |
| 首 chunk 到达即释放 P 节点 KV | `release_prefill_kv_once` | 快速回收 P 节点资源，提升吞吐 |
| recomputed 重试 | 重新选 P/D 节点重走流程 | 兜底 KV 驱逐/加载失败场景 |

---

## 8. 代码索引

### 代理服务器侧 (`load_balance_proxy_server_example.py`)

| 功能 | 位置 |
|------|------|
| 请求改造（max_tokens=1, stream=False） | `:818-834` (`build_prefill_request`) |
| P 节点请求发送与响应处理 | `:837-860` (`send_request_to_service`), `:941-957` |
| 提取 KV 传输参数 + cached_tokens | `:953-957` |
| D 节点分配 | `:959-977` (`assign_instances` 内) |
| 流式转发给客户端 | `:1031-1100` (`generate_stream`) |
| 首 chunk 到达释放 P 节点 KV | `:1043-1044` |
| recomputed 重试 | `:1082-1093` |
| 重新分配实例 | `:980-989` (`reassign_instances`) |
| 实例信息数据结构 | `:155-164` (`InstanceInfo`) |
| 请求终止释放资源 | `:913-921` (`_finish_instance`) |

### vLLM 调度器侧 (`vllm/vllm/v1/core/sched/scheduler.py`)

| 功能 | 位置 |
|------|------|
| connector 报告可拉取 KV 量 | `:727-751` (`get_num_new_matched_tokens` 调用) |
| 异步拉取，进入 WAITING_FOR_REMOTE_KVS | `:938-958` |
| KV 拉取完成处理 + full hit 回退 1 token | `:2379-2409` (`_update_waiting_for_remote_kv`) |
| 调度时计算 num_new_tokens | `:799` |
| recompute 策略配置 | `:129,144` |
| KV 加载失败处理 | `:2389-2399` |

### KV Connector 基类 (`vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`)

| 功能 | 位置 |
|------|------|
| `get_num_new_matched_tokens` 抽象方法 | `:454-486` |
| `start_load_kv` 抽象方法 | `:293-308` |
| `update_state_after_alloc` | `:488-507` |
| `request_finished` | `:542-561` |