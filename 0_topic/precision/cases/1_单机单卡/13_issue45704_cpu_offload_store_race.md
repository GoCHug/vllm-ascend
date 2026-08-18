# 案例13：SimpleCPUOffloadConnector GPU→CPU store 跨流同步缺失，静默 KV 受损

> **一句话定位**：V1 启用 CPU KV 卸载 `SimpleCPUOffloadConnector`，store 在独立低优先级流 + 后台线程 + `srcAccessOrder=ANY(3)` 执行，且无 `stream.wait_stream(compute_stream)` 制造设备侧 happens-before；并发负载下 ~5% 补全静默返回乱码。host 侧 `confirmed_tokens` 簿记不能替代设备侧流同步。
>
> **对象**：vllm-project/vllm [#45704](https://github.com/vllm-project/vllm/issues/45704)（Bug，closed）+ 修复 PR [#46278](https://github.com/vllm-project/vllm/pull/46278)（[Bugfix][KVConnector]，merged）

---

## 1. 问题描述

### 1.1 现象

- 混合注意力模型（滑动窗口 + 全注意力层交错 → 多 KV group 不同 block size）启用 CPU KV 卸载后，并发负载下约 **5%** 补全静默损坏（乱码/混文字），无错误无警告无崩溃；
- 卸载关闭即消失；暂停后恢复的 prefix 命中被重新 load 回 GPU → 毒化该请求 KV → garbage tokens。

### 1.2 触发条件（必现矩阵）

| CPU KV 卸载 | 并发负载 | 混合注意力（多 group） | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗ | — | — | ✗ |
| ✓ | ✗（单请求） | — | △（不易触发） |
| **✓** | **✓** | **✓** | **✓ 约 5% 静默乱码** |

### 1.3 影响与严重度

- **严重度**：🟡 中（比例较低但静默、影响线上 KV 缓存复用）。
- **隐蔽性**：🟡 中（需并发+卸载+多 group）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm |
| 对象 | Issue [#45704](https://github.com/vllm-project/vllm/issues/45704)（Bug，closed） |
| 修复 PR | [#46278](https://github.com/vllm-project/vllm/pull/46278)（BugFix，merged） |
| 复现版本 | 约 v0.23–0.25 时间窗（2026-06 起报） |
| 修复说明 | 已含于 main / 0.25.1（见 [#47282](https://github.com/vllm-project/vllm/issues/47282) 所述） |
| vllm-ascend | 「未知」（Stream 语义/生命周期问题在 NPU 上值得对照复现） |
| 后续缺口 | [#47282](https://github.com/vllm-project/vllm/issues/47282)：load 路径（对称弱点）高并发下仍可能残留乱码，尚待修复 |

---

## 3. 定位过程

1. **记录复现**：并发 + 卸载开启触发、单请求不触发、卸载关闭消失 → 锁定卸载路径；
2. **对照旧实现**：旧 `OffloadingConnector.gpu_worker.SingleDirectionOffloadingHandler.transfer_async` 有 `wait_stream(current_stream)` 且 store 方向用 STREAM 序；新版 `SimpleCPUOffloadConnector` 缺失；
3. **确认根因**：`confirmed_tokens` 是 host 侧调度簿记，不构成设备侧 happens-before；store 在独立流 + `srcAccessOrder=ANY` 且无 `wait_stream`，GPU→CPU DMA 可能读到尚未 retired 的写。

> 定位要点：KV offload 正确性依赖真正的跨流 happens-before 边——host 侧「已确认/已计算」簿记不能替代设备侧 stream 同步。

---

## 4. 解决方案

### 4.1 根因

store 方向缺跨流同步 + 无序访问序，GPU→CPU DMA 在 compute 流写 retired 前执行。

### 4.2 修复（PR #46278）

```diff
# store 前 insert 同步，且 store 方向改有序访问序（load 保留 ANY）
+ store_stream.wait_stream(compute_stream)
  ... store with srcAccessOrder=STREAM（而非 ANY）
```

### 4.3 验证

连接器自带 `cuMemcpyBatchAsync` 微测试，150 iters 配置矩阵：ANY/STREAM + 无 wait_stream 均 150/150 损坏；ANY/STREAM + wait_stream 均 0/150 干净——证明「host 顺序 defer 一步」不足、`wait_stream` 必要且充分。真机 n=456 下 ~5% 乱码 → 修复后 0。

---

## 5. 复现方法

正文给出 `race_microtest.py`（在服务镜像内运行），用 `torch.cuda._sleep` 延迟 compute 流把竞态变成 100% 确定；或真实并发混合注意力模型 + CPU offload 压测观察 ~5% 乱码。

> vllm/vllm-ascend 复现版本：上游 vllm 约 **v0.23–0.25**；vllm-ascend 版本「未知」——此竞态为 Stream 生命周期通用问题，NPU 上无论是 Mooncake 还是 offload 的 H2D/D2H 路径都建议做一次 `wait_stream` 对照。

---

## 核心教训

异步 H2D/D2H 卸载必须在 store 前 `wait_stream(compute_stream)` + 有序访问序；host 侧簿记永远不等价于设备侧同步，任何「确认但不 fence」的异步 offload 都会在并发下静默损坏 KV。