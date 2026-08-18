# 案例 24：PD 分离 Mooncake P2P 单 shard pull 完成即重排 KV，TP 各 rank 数据不一致

> **一句话定位**：Mooncake PD 分离 P2P 传输下，原实现在**单个 TP-offset pull 任务结束**时立即 reformat（KV 块重排），而同 request 其他 shard/rank 的 pull 未必完成——部分 rank 读到半就绪状态统一重排，TP 静默不等价 → decode 输出精度错误。
>
> **对象**：vllm-project/vllm-ascend [PR #12359](https://github.com/vllm-project/vllm-ascend/pull/12359)（[BugFix] Fix precision issues caused by incorrect reordering timing leading to TP inequality，merged 2026-08-16）

---

## 1. 问题描述

### 1.1 现象

- PD 分离 + mooncake kv_p2p connector + TP>1 下，KV cache 跨节点传输后 decode 输出精度错误；
- 根因：**reformat 时机错误**——单个 TP-offset pull 任务结束就对该部分块重排，同 request 其余 shard/rank 的数据尚未到齐；
- TP 各 rank 数据不一致（部分 rank 用半就绪状态重排），无崩溃、静默精度错误。

### 1.2 触发条件（必现矩阵）

| PD 分离 | mooncake P2P | TP>1（CP shard 并存） | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗ | — | — | ✗ |
| ✓ | ✗ | — | ✗（非 P2P 路径） |
| ✓ | ✓ | ✗（TP=1） | ✗（单 rank 无跨 rank 不一致） |
| **✓** | **✓** | **✓** | **✓ TP 不等价、精度错误** |

### 1.3 影响与严重度

- **严重度**：🔴 高（所有 mooncake P2P PD 分离多卡部署受影响）。
- **隐蔽性**：高（取决于各 rank pull 完成时序，竞态类必现条件）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#12359](https://github.com/vllm-project/vllm-ascend/pull/12359)（BugFix） |
| 状态 | merged（2026-08-16 合入 main） |
| vllm-ascend 版本 | main（2026-08-16） |
| 上游 vLLM | v0.27.1，main commit `58d3918e3ea0a544ffedadad2ba84559e9c51d8f` |

---

## 3. 定位过程

未知（正文一句话；diff 注释给出了修复设计——pending 队列 + 全任务完成判定）。

---

## 4. 解决方案

### 4.1 根因

分布式 KV 传输中，跨 rank 的数据重排必须以「该 request **全部** pull 完成」为前置条件；部分完成即重排 = TP 静默不等价。

### 4.2 修复内容（`mooncake_connector.py`，+59/-1）

```python
# 新增 pending_reformat: dict[request_id][shard_idx] -> 待重排块列表 + 线程锁
self.pending_reformat_lock = threading.Lock()

# pull 任务处理: 先暂存，不立即重排
self._stash_pending_reformat(request_id, shard_idx, ready_block_ids)

# _handle_request finally: 所有任务完成后统一重排
all_tasks_done = self._mark_request_task_done(...)
if all_tasks_done:
    self._reformat_pending_kv_caches(request_id)   # 按 shard_idx 顺序

# 失败路径: 清 pending + _mark_failed_recv_request
```

### 4.3 验证

CI（PR 原文 "By ci"）。

---

## 5. 复现方法

### 5.1 最小复现模型

- **Qwen2.5-7B**（任一支持 KV transfer 的 7B 级模型）。

### 5.2 复现命令

```bash
# P 端
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8010 --tensor-parallel-size 2 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_port":"36000","kv_connector_extra_config":{"prefill":{"tp_size":2},"decode":{"tp_size":2}}}'

# D 端
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8020 --tensor-parallel-size 2 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_port":"36100","kv_connector_extra_config":{"prefill":{"tp_size":2},"decode":{"tp_size":2}}}'
# 发请求 prefill 后切 decode，观察输出是否劣化（多 rank pull 完成时序竞态）
```

> 需已编译 Mooncake（`cmake .. -DUSE_ASCEND_DIRECT=ON`）；`kv_port` 取 `>= 36000` 避开 AscendDirect RDMA 随机端口。

---

## 核心教训

分布式 KV 传输中，跨 rank 的重排/消费动作必须以「该 request 全部传输完成」为前置条件——**部分完成即处理 = TP 静默不等价**；修复模式是「pending 队列 + 完成计数 + 统一重排」。
