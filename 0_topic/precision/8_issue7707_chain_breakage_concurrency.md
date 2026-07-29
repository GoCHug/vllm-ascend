# Issue #7707 深度案例：高并发下 KVCache chain 断裂导致精度异常

> 整理时间: 2026-07-29
>
> 案例对象: [vllm-project/vllm-ascend#7707](https://github.com/vllm-project/vllm-ascend/issues/7707)
>
> 标题: [Bug]:Ds3.2+A3 KVCache chain breakage at 100 concurrency results in precision problems
>
> 关联全景文档: [0_kvcache.md](./0_kvcache.md) §2.1 / §7.1 / 案例 3
>
> 关键词: DeepSeek 3.2 · A3(910B) · KVCache chain · 100 并发 · 竞态 · 数据竞争 · 精度异常

---

## 1. Issue 概览

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| Issue 编号 | [#7707](https://github.com/vllm-project/vllm-ascend/issues/7707) |
| 类型 | Bug |
| 标题 | [Bug]:Ds3.2+A3 KVCache chain breakage at 100 concurrency results in precision problems |
| 报告人 | [@qiudepei](https://github.com/qiudepei) |
| 标签 | bug · deepseek-v3 |
| 状态 | **open** |
| 严重度 | 🔴 高（高并发精度异常 + 服务不可恢复） |
| 隐蔽性 | ⚫ 极高（低并发不触发；现象不一；silent corruption） |
| 是否有修复 PR | 暂无直接对应合并 PR；根因仍在排查，同类并发问题见 [#9168](https://github.com/vllm-project/vllm-ascend/issues/9168) |
| 创建时间 | 2026-03-27 |
| 首次报错时间 | 2026-03-23 12:47:26（日志时间戳） |
| vLLM-Ascend 版本 | v0.17.0rc1 |
| vLLM 版本 | v0.17.0rc1 |
| 部署形态 | PD 分离 2p1d 2+2（2 Prefill + 2 Decode 节点） |
| 硬件 | A3（Ascend 910B） |
| 模型 | DeepSeek 3.2 (w8a8) |

> 典型的“压测才现形”案例：32 并发一切正常，冲到 100 并发后 Mooncake 连接断裂、D 节点输出现乱码，而真正的高并发数据竞争隐藏在 KV Cache 链式管理的竞态之中。

---

## 2. 场景背景

### 2.1 PD 分离部署

DeepSeek 3.2 (w8a8) + A3 采用 **PD 分离 (Prefill/Decode Disaggregation)** 部署：`2p1d 2+2` 表示 2 组 **2 Prefill + 1 Decode** 的拓扑。P 端一次性算完长 prompt 的 KV Cache，D 端只做逐 token 自回归生成——两者之间必须把 P 端的 KV Cache 通过网络搬到 D 端，否则 D 端要重算 prefix，吞吐不可接受。

vLLM-Ascend 在此场景下走 **Mooncake Connector**（基于 Mooncake / RDMA + Ascend `ascend` 协议）做跨节点 KV 传输：`TransferEngine.batch_transfer_sync_read(...)` 把远端（P 端）NPU 显存中的 block 按 `src_list/dst_list/length_list` 批量 RDMA 读到本端（D 端）KV Cache 显存。

### 2.2 KV Pool / AscendStore 的 chain 链式 block 管理

除 P2P 的 Mooncake 路径外，vLLM-Ascend 还提供 **AscendStore / KV Pool** 池化方案（`kv_connector: AscendStoreConnector`，`backend: mooncake`）。无论哪条路径，一个请求的 KV Cache 都由**一串 block** 组成，上下游把它当作一条“链 (chain)”来管理：

- **P2P 路径**：`MooncakeConnectorScheduler._reqs_need_recv[req_id]` 保存元组 `(Request, prompt_block_ids, gen_block_ids, ...)`，block 有先后顺序，组成逻辑链。
- **AscendStore 路径**：`KVPoolScheduler._request_trackers[req_id]` 持有 `RequestTracker(allocated_block_ids: list[int])`，随 decode 推进不断 `append` 新 block——这是物理意义上的“链表”。
- **block-hash 命中链**：`CompressAttentionManager.find_longest_cache_hit` 在 `single_type_kv_cache_manager.py` 中沿 `logical_block_hashes` 顺序 `computed.append(cached)`，遇 miss 即 `break`——“chain 断裂”的直接体现就是命中链提前截断。

代码中“chain”一词最贴切的出处即此处的注释（`vllm_ascend/core/single_type_kv_cache_manager.py:222`）：

```python
for block_hash in itertools.islice(logical_block_hashes, max_num_blocks):
    # block_hashes is a chain of block hashes. If a block hash is not
    # in cached_block_hash_to_id, the following block hashes are
    # not computed yet for sure.
    if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
        for computed, cached in zip(computed_blocks, cached_block):
            computed.append(cached)
    else:
        break
```

### 2.3 高并发：100 路请求同时跑链

100 并发意味着：100 条请求各自维护一条 KV block chain，同时被 P 端追加、D 端读取、调度器分配/释放。`KVCacheRecvingThread` 用 `ThreadPoolExecutor(max_workers=32)` 按 peer 分发处理，每个 peer 最多每轮 5 个请求（`MAX_REQUESTS_PER_PEER_HANDLER = 5`）；而 `AscendStore` 侧还有独立的 `LookupKeyServer` ZMQ 回复线程并发调用 `lookup_scheduler`。chain 的追加、读取、元数据修改分散在 scheduler 线程、recv 线程池、send 线程、lookup 线程之间——锁覆盖并不完整。

---

## 3. 现象描述

### 3.1 报告人原话（节选自 issue 正文）

> 1、Service starts without any errors;
> 2、Runs fine at 32 concurrent requests;
> 3、Once concurrency hits 100, mooncake starts throwing errors and D nodes show garbled characters.

**临时方案**：`turn off TLS`（根因仍在调查）。

**建议**：`If Plog reports an ERROR and cannot recover, stop the service at once.`

### 3.2 关键现象归纳

| 序号 | 现象 | 备注 |
|:---:|------|------|
| 1 | 服务启动无报错，32 并发稳定 | 低并发一切正常 |
| 2 | 100 并发触发后 Mooncake 连接断裂 | “broken mooncake connection” |
| 3 | D 节点输出乱码（garbled characters） | 即“精度问题”的外在表现 |
| 4 | P 端日志 3s 超时；调到 10s 仍超时 | 不是单纯超时阈值问题 |
| 5 | 首错为 `MemCopySync` / `rtMemcpy` 失败 | 见 §3.3 |
| 6 | Plog 报 ERROR 后**不可自恢复** | 需立即停服 |

### 3.3 首条错误日志（原文）

```
[ERROR] RUNTIME(2933,):2026-03-23-12:47:26.619.392 [api_impl.cc:2677]2933
  MemCopySync:The current capture mode:threadCaptureMode=0,
  exchangeCaptureMode=0, contextCaptureMode=0
[ERROR] RUNTIME(2933,):2026-03-23-12:47:26.619.399 [api_error.cc:1571]2933
  MemCopySync:Memory copy sync failed, cnt=416, kind=1.
[ERROR] RUNTIME(2933,):2026-03-23-12:47:26.619.410 [api_c.cc:1097]2933
  rtMemcpy:ErrCode=107030, desc=[the current capture mode does not support
  this operation], InnerCode=0x703001c
[ERROR] ASCENDCL(2933,):2026-03-23-12:47:26.619.578 [memory.cpp:541]2933
  aclrtMemcpyImpl:synchronized memcpy failed, kind = 1, runtime result = 107030
```

`ErrCode=107030` = “the current capture mode does not support this operation”，三种 capture mode 均为 0（`threadCaptureMode=0, exchangeCaptureMode=0, contextCaptureMode=0`）。这条错误在 NPU runtime 层抛出，说明高并发下 Mooncake 的同步 memcpy（`aclrtMemcpy` / `batch_transfer_sync_read`）被调度进了**不支持的 capture/执行上下文**——并非单纯的“内存拷贝失败”，而是**并发上下文错乱**导致运行时态非法。

> 注：本仓库代码中并无 `CaptureMode`/`threadCaptureMode` 标识符，`capture_mode` 字符串亦未出现；这是 CANN runtime 层的错误向上冒泡，源头在 Mooncake/Ascend 传输任务的并发执行模型与 NPU capture 上下文不兼容。

---

## 4. 根因分析

### 4.1 核心结论：多请求并发修改 chain 结构的竞态

100 并发场景下，多条请求同时维护各自的 KV block chain，而对 chain 的**追加 / 读取 / 元数据修改并未建立正确的同步保护**。典型竞态：

```
Thread 1（请求 A 追加 block）            Thread 2（请求 B 读 / 改同一 chain 元数据）
  chain->last->next = new_block            read chain_head
  chain->last        = new_block           traverse chain ...
        ↑
  如果“追加”这两步不是原子/未加锁，
  Thread 2 可能读到半更新的 chain：
  看到 new_block 但 last 未更新，或反之
  → 链指针断裂、节点错位、block_id 失效
```

断裂后，后续 token 只能访问**部分** KV cache → 上下文信息丢失 → attention 计算基于残缺 KV → 输出错乱/乱码。NPU memcpy 在拿到错乱的 block 地址/长度（或失效句柄）后，进一步触发 `capture mode does not support` 这类运行时态错误。

### 4.2 三个竞态热点（代码定位）

**热点 1：AscendStore scheduler 侧字典无锁**
`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:196-205`

```python
def _get_or_create_request_tracker(self, req_id: str) -> RequestTracker:
    tracker = self._request_trackers.get(req_id)
    if tracker is None:
        tracker = RequestTracker(req_id=req_id, token_len=0, allocated_block_ids=[])
        self._request_trackers[req_id] = tracker
    return tracker
```

`self._request_trackers: dict[str, RequestTracker]` 与 `self.sending_blocks: dict[int, list[int]]` 是**普通 dict**，在调度线程里被 `update_state_after_alloc` 修改（追加 block id），`allocated_block_ids` 不断 append——这是每请求的物理 chain。本文件内未见显式锁。

**热点 2：block-hash 命中链的读-校验竞态**
`vllm_ascend/core/single_type_kv_cache_manager.py:222`（见 §2.2 引用）
`block_pool.get_cached_block(block_hash, ...)` 命中链 traversing 时，若另一线程正在 free/realloc 同一 block hash（高并发 eviction/重分配时极易发生），`break` 提前发生 → 请求读到不完整命中链 → chain 断裂表象之一。

**热点 3：LookupKeyServer 并发 lookup 无锁**
`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:283-329`
独立 ZMQ-reply 线程并发调用 `pool_worker.lookup_scheduler(...)`，而 `pool_worker` 中只有 `_invalid_block_ids` 被 `_invalid_block_ids_lock` 保护，`token_database` / `cache_coordinator` 的查找路径**未见锁**——高并发下 lookup 返回的 block 状态可能与实际写入状态不一致，间接制造半更新链。

**热点 4：Mooncake `batch_transfer_sync_read` 与 NPU capture 上下文**
`mooncake_connector.py:699/799/934`：同步 RDMA 读 + `torch.npu.synchronize()`（line 896，带 FIXME：“if we skip synchronization ... the system will crash in GQA scenarios. However, we still haven't identified the root cause.”）。32 线程池 + 每 peer 5 任务的扇出下，多任务并发触发 `aclrtMemcpy` 时进入未支持的 capture mode → `ErrCode=107030`。

### 4.3 为什么 32 并发不触发、100 并发触发

| 因素 | 32 并发 | 100 并发 |
|------|---------|----------|
| 同时在飞的 transfer 任务数 | 少，线程池基本无排队 | 远超 32 worker，任务密集排队、上下文切换频繁 |
| eviction / block 重分配概率 | 低，chain 稳定 | 显著上升，命中链与释放链交错 |
| lookup 与 put 并发交错 | 稀疏，时序错开 | 密集，半更新窗口被命中 |
| NPU capture 上下文压力 | 单一，合法 | 多任务争抢，易进入非法态 |

竞态类 bug 的典型特征：**概率事件**，触发窗口随并发度非线性放大。32 并发时半更新窗口被命中的概率极低；100 并发下窗口被密集打开，几乎必然出问题。这也是它“隐蔽性 ⚫ 极高”的原因——CI 默认并发度低，根本碰不到。

### 4.4 为什么不报错也是它的一种形态

chain **部分**断裂时：
- attention 仍能跑（只是 KV 残缺），不会 NaN/崩溃；
- 输出“看起来在生成”，只是答非所问或语义错乱；
- 只有当断裂进一步恶化、memcpy 拿到完全失效的地址/句柄时，才冒泡成 `ErrCode=107030`。

这就是 issue 标题“precision problems”的本质：**silent corruption** 为主，显式报错为辅。

---

## 5. 影响与表现

| 维度 | 表现 |
|------|------|
| 触发配置 | DeepSeek 3.2 (w8a8) + A3(910B) + PD 分离 2p1d 2+2 + **并发 ≥ 100** |
| 不触发 | 并发 ≤ 32（报告人实测稳定） |
| 直接现象 | Mooncake 连接断裂；D 节点输出乱码（garbled） |
| 运行时现象 | `rtMemcpy ErrCode=107030` capture mode 非法；3s→10s 超时仍失败；Plog ERROR 不可自恢复 |
| 精度表现 | 读到不完整 KV cache → 上下文丢失 → attention 基于残缺 KV → 输出错乱 |
| 严重度 | 🔴 高（端到端精度异常 + 服务不可恢复，必须停服重启） |
| 隐蔽性 | ⚫ 极高（低并发不触发；silent corruption 为主；需 100+ 并发压测才暴露） |
| 是否崩溃 | 不一定崩溃；多为输出质量下降 + 偶发 runtime ERROR |
| 临时缓解 | 关闭 TLS（turn off TLS）；Plog 报错不可恢复时立即停服 |

---

## 6. 修复方向

### 6.1 细粒度锁 / 原子操作 / 无锁结构

针对 §4.2 的热点：

1. **AscendStore scheduler dict 加锁或改并发结构**
   - `_request_trackers` / `sending_blocks` 的读写用 `threading.Lock` 或 `RLock` 包裹；
   - 或改用 `concurrent.dict` / 分片锁降低争用；
   - `allocated_block_ids.append` 必须在锁内完成，保证“追加两步”对读侧原子可见。

2. **block 命中链读-校验原子化**
   - `find_longest_cache_hit` 遍历期间对涉及的 block pool 区段加读锁，或 snapshot 引用计数，避免遍历中被 free/realloc。

3. **`LookupKeyServer.lookup_scheduler` 加锁或改单线程化**
   - 至少对 `token_database` / `cache_coordinator` 查找路径加锁，保证 lookup 看到的 block 状态与 put/get 一致。

4. **Mooncake transfer 任务的执行上下文治理**
   - 复核 `ThreadPoolExecutor(32)` + `batch_transfer_sync_read` 与 NPU capture mode 的兼容性，明确每个 worker 的 device/capture 上下文初始化（参见 `mooncake_connector.py:470-472` 的 thread-local device 注释，capture 上下文同理需要 thread-local 化）。

### 6.2 chain 完整性校验

- 在 chain 遍历/迁移关键路径加入**完整性断言**：校验 `last->next == nullptr`、`head` 到 `last` 长度与 `len(allocated_block_ids)` 一致；
- transfer 前对 `block_ids` 做有效性检查（非空、连续、属于本请求），失效则 fallback 重算而非传坏数据。

### 6.3 高并发稳定性测试

- 新增 **100+ 并发压测** 回归：PD 分离 + Mooncake/AscendStore，持续运行 ≥ 2 小时，监控首错时间、输出乱码率、`ErrCode=107030` 出现频率；
- 明确并发度阶梯（32 / 64 / 100 / 200），把“低并发不触发”的盲区纳入 CI gate（参见 [0_kvcache.md](./0_kvcache.md) §11 总结中的“并发维度测试缺失”）。

### 6.4 关联 #9168

[#9168](https://github.com/vllm-project/vllm-ascend/issues/9168) “A3 单节点 PD 混部下 AscendStoreConnector mooncake 池化，大量报错 Failed to get key” 是**同一类并发问题**的另一外在表现（`mooncake_backend.py:95/229/250` 的 `Failed to get %d keys out of %d`）。两者均指向 Mooncake / AscendStore 在高并发下的查找/传输竞态，修复应协同推进。

---

## 7. 复现与验证

### 7.1 触发矩阵

| PD 分离 | 并发度 | 后端 | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗ | — | — | ✗（无跨节点传输） |
| ✓ | ≤ 32 | Mooncake P2P | ✗（报告人实测稳定） |
| ✓ | ≥ 100 | Mooncake P2P | **✓ 触发：乱码 + 连接断裂 + ErrCode=107030** |
| ✓ | ≥ 100 | AscendStore(mooncake backend) | **✓ 同类：Failed to get key（#9168）** |

### 7.2 复现步骤（基于 issue）

1. 部署 DeepSeek 3.2 (w8a8) + A3，PD 分离 2p1d 2+2，vLLM-Ascend v0.17.0rc1。
2. 以 32 并发做基线压测，确认输出正常、无 Mooncake 报错。
3. 直接拉到 100 并发持续请求，观察：
   - D 节点是否出现乱码/garbled；
   - P 端日志是否出现 `MemCopySync` / `rtMemcpy ErrCode=107030`；
   - ZMQ/mooncake 是否报超时（3s→10s 仍超时）；
   - Plog ERROR 是否不可自恢复。
4. 对照实验：关闭 TLS（`turn off TLS`）对比是否缓解（临时方案，非根治）。

### 7.3 排查方法

1. **确认三要素**：PD 分离 + Mooncake/AscendStore 后端 + 并发 ≥ 100。
2. **并发度二分**：从 100 逐步降到 64 / 32，定位触发阈值。
3. **后端对照**：Mooncake P2P vs AscendStore(mooncake)，看是否都触发（区分是 P2P 传输竞态还是 Pool 查找竞态）。
4. **日志聚焦**：
   - P 端：`rtMemcpy ErrCode=107030`、capture mode 三值、超时时间；
   - ascend_store：`mooncake_backend.py:95 Failed to get key`（关联 #9168）；
   - D 端：block_ids 有效性、KV block chain 长度是否与请求 token 数匹配。
5. **chain 完整性 dump**：在 `_get_or_create_request_tracker` / `find_longest_cache_hit` 关键节点加日志，打印 chain 长度、block_id 连续性，对比并发前后是否断裂。
6. **长时间压测**：竞态概率事件，短测可能漏；建议 ≥ 2 小时持续压测（参考 [#11127](https://github.com/vllm-project/vllm-ascend/issues/11127) 的“压测 1-2 小时后才现形”）。

---

## 8. 经验与启发

### 8.1 并发度本身就是一条“隐藏 code path”

32 并发是快乐路径，100 并发会切进另一条“密集排队 + eviction 交错 + capture 上下文争抢”的隐性路径。任何 KV 传输/池化相关改动，都应把 **低并发 / 高并发** 当作两条独立路径分别测试——这与 TP 相等/不等是同类教训（见 [1_pr8540](./1_pr8540_tp_unequal_mtp_kv.md) §8.3）。

### 8.2 “追加两步”必须原子可见

链表/chain 的 `last->next = new; last = new;` 是经典竞态教材。在 KV block chain 这类被多线程读写的结构上，任何**非原子的多步结构修改**都是定时炸弹。修复时要么加锁包住完整两步，要么用原子指针/无锁链表，绝不能依赖“读侧自行容错”。

### 8.3 Silent corruption 比崩溃更危险

chain 部分断裂时输出“看起来还在生成”，只是答非所问——没有 NaN、没有断言、没有崩溃。这类 silent 精度下降最难发现：CI 的 accuracy 回归若并发度不够、时长不够，根本测不出来。必须用**高并发 + 长时长**的压测兜底（参见 [0_kvcache.md](./0_kvcache.md) §7.1 渐进性退化）。

### 8.4 运行时态错误是竞态的“冒泡出口”

`ErrCode=107030 capture mode does not support` 表面看是 NPU runtime 错误，实则是上层并发任务把 memcpy 调度进了非法上下文。排查这类错误时，不能只盯着 runtime 层，要向上回溯到**任务的并发分发与上下文初始化**（`ThreadPoolExecutor` 的 `initializer`、thread-local device/capture 设置）。

### 8.5 临时方案 ≠ 根因

报告人给出的 `turn off TLS` 只是缓解。竞态根因未除——只要并发度够高、chain 修改无锁，换任何传输配置都会再次断裂。临时方案可作为止血，但必须配合 §6 的结构性修复。

---

## 9. 关联问题

| 编号 | 关联点 |
|------|--------|
| [vllm-ascend#9168](https://github.com/vllm-project/vllm-ascend/issues/9168) | AscendStoreConnector mooncake 池化高并发 `Failed to get key` —— 同类并发竞态，`mooncake_backend.py:95` |
| [vllm-ascend#11127](https://github.com/vllm-project/vllm-ascend/issues/11127) | KV Pool + MTP 压测 1-2h 后 MTP 接受率 <1% —— 同类渐进性精度退化，长时压测才触发 |
| [vllm#44238](https://github.com/vllm-project/vllm/issues/44238) | MooncakeConnector 并发 PD 传输数据损坏（`batch_transfer_sync_write` race）—— 上游 vLLM 同类并发损坏 |
| [vllm-ascend#8992](https://github.com/vllm-project/vllm-ascend/issues/8992) | MooncakeConnectorV1 + FULL_DECODE_ONLY ACL Graph 问题 —— 与 capture mode 上下文相关 |
| [vllm-ascend#12390](https://github.com/vllm-project/vllm-ascend/issues/12390) | AscendStore KV Pool v0.23.0 known issues —— KV Pool 稳定性汇总 |
| [0_kvcache.md](./0_kvcache.md) §2.1 | 并发传输数据竞争（Mooncake / RDMA race / chain breakage） |
| [0_kvcache.md](./0_kvcache.md) §7.1 | KV Pool 并发精度异常（chain breakage / 渐进退化） |
| [0_kvcache.md](./0_kvcache.md) §10 案例 3 | 本案例在全景文档中的归档位置 |

---

## 10. 结论

> 本案例核心教训：**KV Cache 的链式 block 管理在高并发下必须保证“追加—读取—元数据修改”的原子可见性，否则 100 并发会撕裂 chain、让 D 端读到的只是残缺 KV——精度先 silent 崩坏，再冒泡成 runtime 错误。**