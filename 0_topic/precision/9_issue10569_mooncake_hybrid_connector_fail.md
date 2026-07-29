# Issue #10569 深度案例：DeepSeek-V4 Pro PD 分离 Decode 节点 mooncake_hybrid_connector 报错，KV Cache 传输失败

> 整理时间: 2026-07-29
>
> 案例对象: [vllm-project/vllm-ascend#10569](https://github.com/vllm-project/vllm-ascend/issues/10569)
>
> 标题: [Bug]: Deepseek v4 Pro PD分离部署时Decode节点mooncake_hybrid_connector报错，kv cache传输失败
>
> 关联全景文档: [0_kvcache.md](./0_kvcache.md) §2.1 / §2.3 / 案例 9
>
> 关键词: DeepSeek-V4 Pro · PD 分离 · mooncake_hybrid_connector · hybrid connector · KV 传输失败/报错

---

## 1. Issue 概览

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| Issue 编号 | [#10569](https://github.com/vllm-project/vllm-ascend/issues/10569) |
| 类型 | Bug |
| 报告者 | （issue 作者，未在元数据中显式署名） |
| 状态 | **open**（截至 2026-07-29） |
| 严重度 | 🟡 中（功能可用性受损，但非静默精度损坏） |
| 创建时间 | 2026-06-16 |
| 是否有修复 PR | 尚未发现直接关联的已合并修复 PR（issue 仍 open，comments/timeline 暂无公开修复链接） |
| vLLM 版本 | v0.20.2 |
| vLLM-Ascend 版本 | 0.20.2rc1 |
| 硬件 | 4×A3 (Ascend910) aarch64，Kunpeng 920，CANN 9.0.0 |
| 模型 | DeepSeek-V4-Pro-w4a8-mtp |
| 连接器 | `MooncakeHybridConnector`（`--no-disable-hybrid-kv-cache-manager`） |

> 典型的“配置组合触发边界条件”案例：DeepSeek-V4 的 hybrid attention（Compress-4/Compress-128）+ PD 分离 TP 不等（prefill dp4tp8 / decode dp8tp4）+ MTP 投机解码三者叠加，撞上 hybrid connector 分组传输里一个不对称的长度截断守卫。

---

## 2. 场景背景

### 2.1 PD 分离部署 (Prefill/Decode Disaggregation)

在 PD 分离架构中，Prefill 节点（P 端）负责长 prompt 的一次性计算，Decode 节点（D 端）负责逐 token 自回归生成。P 端计算好的 KV Cache 通过网络传输到 D 端，避免 D 端重复计算 prefix。本案例使用 4 台 A3：P 端 `dp4tp8`，D 端 `dp8tp4`，即 **P/D 端 TP 不等**（TP=8 vs TP=4）。

### 2.2 mooncake_hybrid_connector (MooncakeHybridConnector)

vLLM-Ascend 提供三种 P2P KV 传输 connector 变体（见 `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_p2p_connector.md`）：

| 变体 | 文件 | 适用 |
|------|------|------|
| 整体传输 | `mooncake_connector.py` | 大多数场景 |
| 逐层传输 | `mooncake_layerwise_connector.py` | 模型大、内存有限 |
| 混合传输 | `mooncake_hybrid_connector.py` | hybrid KV cache groups |

**关键澄清**：`MooncakeHybridConnector` 中的“hybrid”**不是**指“mooncake + 另一种传输通道”，而是指 **hybrid KV cache 管理（HMA, Hybrid Memory Allocator）**——模型同时存在多种 KV cache group 类型（FullAttention + Compress/SWA/Mamba）。底层传输始终是 Mooncake/RDMA（`engine.batch_transfer_sync_read`），但 connector 需要按 group 分别计算 block id、stride、base address，并处理跨 group 的地址共享。

实现上，`use_hybrid` 由以下条件决定（`mooncake_hybrid_connector.py:1220-1224` / `1517-1521`）：

```python
self.use_hybrid = (
    not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
    and any(not isinstance(g.kv_cache_spec, FullAttentionSpec) for g in kv_cache_config.kv_cache_groups)
    and len(kv_cache_config.kv_cache_groups) > 1
)
```

### 2.3 DeepSeek-V4 Pro 为何强制走 hybrid 路径

DeepSeek-V4 引入 **hybrid attention architecture**：通过 Compress-4-Attention 与 Compress-128-Attention 提升长上下文效率（见 `docs/source/tutorials/models/DeepSeek-V4-Pro.md:5-11`）。这意味着模型存在多个 KV cache group（FullAttention 组 + Compress 组），且至少有一个 group 的 spec 不是 `FullAttentionSpec` → `use_hybrid=True`。同时 V4 Pro 启用 **MTP**（`--speculative-config '{"method":"deepseek_mtp"}'`），进一步增加待传输 KV 层。

用户配置中显式保留 hybrid 管理器（`--no-disable-hybrid-kv-cache-manager`）并选择 `MooncakeHybridConnector`，正是 V4 的 mandatory 路径。

---

## 3. 现象描述

服务启动正常，首个请求能完成（前若干 step 的 SpecDecoding 接受率一度达到 100%）。但随后 **Decode 节点所有 TP rank** 在 KV cache 传输时持续报错（`mooncake_hybrid_connector.py:463`），错误在每个请求的传输阶段重复出现。

### 3.1 核心错误信息（摘自 issue body，逐字）

Decode 侧 4 个 TP rank（TP0~TP3）同步报出同一异常：

```
(Worker_DP0_TP0_EP0 pid=90261) ERROR 06-16 13:09:56 [mooncake_hybrid_connector.py:463] Failed to transfer KV cache for request cmpl-4099b14d-d7da-4865-9355-63cd08ce4bb6-0-88813fd5.
(Worker_DP0_TP0_EP0 pid=90261) ERROR 06-16 13:09:56 [mooncake_hybrid_connector.py:463] Traceback (most recent call last):
(Worker_DP0_TP0_EP0 pid=90261) ERROR 06-16 13:09:56 [mooncake_hybrid_connector.py:463]   File "/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py", line 460, in _handle_request
(Worker_DP0_TP0_EP0 pid=90261) ERROR 06-16 13:09:56 [mooncake_hybrid_connector.py:463]     self._transfer_kv_cache_all_groups(req_meta)
(Worker_DP0_TP0_EP0 pid=90261) ERROR 06-16 13:09:56 [mooncake_hybrid_connector.py:463]   File "/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py", line 531, in _transfer_kv_cache_all_groups
(Worker_DP0_TP0_EP0 pid=90261) ERROR 06-16 13:09:56 [mooncake_hybrid_connector.py:463]     grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
(Worker_DP0_TP0_EP0 pid=90261) ERROR 06-16 13:09:56 [mooncake_hybrid_connector.py:463]   File "/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py", line 1803, in group_concurrent_contiguous
(Worker_DP0_TP0_EP0 pid=90261) ERROR 06-16 13:09:56 [mooncake_hybrid_connector.py:463]     brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
(Worker_DP0_TP0_EP0 pid=90261) ERROR 06-16 13:09:56 [mooncake_hybrid_connector.py:463] ValueError: operands could not be broadcast together with shapes (44,) (179,)
```

异常在三批请求上反复出现，shape 还会变化（`(44,) (179,)` → `(44,) (178,)`），说明两侧 block id 列表长度不稳定地不等。注意：报错被 `except Exception` 捕获并 `logger.exception` 记录（`_handle_request` 第 606-607 行），**不会使进程崩溃**，但该请求的 KV 传输被跳过——服务继续“运行”，但后续请求的 SpecDecoding 接受率明显波动（从 100% 跌到 14.6% 再回升），表明 KV 未完整到达 D 端。

### 3.2 部署配置关键参数

P 端（`kv_producer`）：
- `--data-parallel-size 4 --tensor-parallel-size 8`（Prefill dp4tp8）
- `--no-disable-hybrid-kv-cache-manager`、`--block-size 128`、`--enforce-eager`、`--quantization ascend`
- `kv_connector_extra_config: {prefill: {dp_size:4, tp_size:8}, decode: {dp_size:8, tp_size:4}}`

D 端（`kv_consumer`）：
- `--data-parallel-size 8 --tensor-parallel-size 4`（Decode dp8tp4）
- `--speculative-config '{"num_speculative_tokens":1,"method":"deepseek_mtp"}'`
- `--async-scheduling`、`--no-disable-hybrid-kv-cache-manager`、`--block-size 128`
- 注意：D 端 `kv-transfer-config` JSON 中 **`kv_port` 出现两次**（`'$mooncake_port'` 与 `"30800"`），后者覆盖前者——这本身可能引入端口错配，但非本异常的直接原因。

---

## 4. 根因分析

### 4.1 崩溃点：`group_concurrent_contiguous` 的广播失败

崩溃发生在纯函数 `group_concurrent_contiguous`（当前 main 分支 `mooncake_hybrid_connector.py:1970-1987`，issue 环境 v0.20.2rc1 对应行号 `1803`）：

```python
def group_concurrent_contiguous(src, dst):
    src_indices = np.array(src, dtype=np.int64)   # remote block ids
    dst_indices = np.array(dst, dtype=np.int64)   # local  block ids
    if src_indices.size == 0:
        return [], []
    # ↓ CRASH HERE
    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)
    ...
```

`np.diff(src_indices)` 的长度 = `len(src)-1`，`np.diff(dst_indices)` 的长度 = `len(dst)-1`。`np.where((...) | (...))` 要求两个 diff 数组**形状可广播**，即 `src` 与 `dst` 长度相等。issue 报错 `shapes (44,) (179,)` 表明：

- 一个 block id 列表长度 ≈ 45（diff=44）
- 另一个长度 ≈ 180（diff=179）

两侧长度严重不等，`|` 操作 broadcasting 失败 → `ValueError`。

### 4.2 上游守卫：只截断一个方向

`group_concurrent_contiguous` 的调用方在 `_transfer_kv_cache_all_groups`（当前 `638-717`，issue 行号 `~531`）：

```python
for i in range(self.hma_group_size):
    if not remote_block_ids[i] or not local_block_ids[i]:
        continue
    cur_remote_block_ids = remote_block_ids[i]
    cur_local_block_ids  = local_block_ids[i]
    # ↓ 守卫：只处理 local < remote 的情况
    if not isinstance(self.kv_cache_specs[i], MambaSpec) and \
       len(cur_local_block_ids) < len(cur_remote_block_ids):
        cur_remote_block_ids = cur_remote_block_ids[-len(cur_local_block_ids):]
    grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
        cur_remote_block_ids, cur_local_block_ids   # ← 长度可能仍不等
    )
```

截断守卫**不对称**：只在 `len(local) < len(remote)` 时把 remote 截到 local 的长度（取尾部，对齐 FullAttention+prefix-trim 语义）。但当 **`len(remote) < len(local)`**（或两者相等性因 compress/MTP 而错位）时，守卫不触发，**直接把不等长的两个列表送进 `group_concurrent_contiguous`** → 4.1 的广播崩溃。

### 4.3 为什么 DeepSeek-V4 Pro PD 分离会触发“不等长”

这是 hybrid attention + TP 不等 + MTP 的组合效应：

1. **Compress 组的 block 计数与 FullAttention 组不同**。DeepSeek-V4 的 Compress-4/Compress-128 把若干 token 压进一个 KV block，导致同一请求在 Compress group 上的 block 数远少于 FullAttention group（这正是 45 vs 180 量级差异的来源——~4:1 与 ~128:1 的压缩比会产生量级差）。

2. **TP 不等放大 P/D 端 block 归属差异**。P 端 TP=8、D 端 TP=4，`tp_num_need_pulls = 2`，KV head 在 P/D 端按不同 TP 分片。在 hybrid 路径里，`local_block_ids[i]`（D 端本 rank 实际持有的 block）与 `remote_block_ids[i]`（P 端对应 rank 持有、需拉取的 block）的**计数口径**取决于各自的 group→rank 映射与 compress ratio，两者不一定逐 group 对齐。

3. **MTP 增加 KV 层但未必同步增加 block id**。MTP 层复用主模型 KV cache，传输时 `end_layer_index` +1（见 `_transfer_kv_cache` 772-775），但 block id 列表是按 group 构造的；若 MTP 层的 block 未被均匀纳入每个 group 的 id 列表，会进一步打破两侧长度匹配。

4. **`shared_by` 地址共享下的 group 错配**。Compress 场景 `use_compress=True`，`register_kv_caches` 用 `kv_cache_tensor.shared_by` 跨 group 共享 base address，并构造 `addr_group_idx`。当 `shared_by` 在某些 group 为空或映射不一致时（同类问题见 [0_kvcache.md](./0_kvcache.md) §2.3 案例 4 / PR #9500），block id 列表的 per-group 切分与 P 端不对齐，直接导致 `remote_block_ids[i]` 与 `local_block_ids[i]` 长度分叉。

**触发条件 = DeepSeek-V4 hybrid attention（compress）+ PD 分离 + P/D TP 不等 +（MTP）**。前三者已足够产生不等长；MTP 与重复 `kv_port` 配置会加重不稳定性（故 shape 在 `(44,) (179,)` 与 `(44,) (178,)` 间波动）。

### 4.4 为什么是“报错”而非“静默精度下降”

与 [1_pr8540_tp_unequal_mtp_kv.md](./1_pr8540_tp_unequal_mtp_kv.md) 的 silent 精度 bug 不同，本 case 在纯 NumPy 层面就抛了 `ValueError`——因为 `group_concurrent_contiguous` 是 vectorised numpy 实现，对形状有硬性要求。一旦长度不等，立即崩溃（被外层 `except` 吞掉日志但跳过传输），所以表现为“KV 传输失败报错”而非“数值错乱”。这也使其比 silent bug 更容易被发现，但更难被 CI 覆盖（需要真实多节点 + V4 compress + TP 不等）。

---

## 5. 影响与表现

| 维度 | 表现 |
|------|------|
| 触发配置 | DeepSeek-V4 (compress_ratios) + PD 分离 + P/D TP 不等 + `MooncakeHybridConnector` |
| 不触发 | 非 hybrid 模型（走 `_transfer_kv_cache` 单组路径）/ TP 相等且 compress 组对齐 / 单节点 |
| 现象 | Decode 全 TP rank 同步报 `ValueError: operands could not be broadcast together with shapes (44,) (179,)`；KV 传输被跳过 |
| 服务影响 | 进程不崩溃，服务“假活”：首个请求靠已传 KV 勉强完成，后续请求 KV 缺失 → SpecDecoding 接受率剧烈波动（100% → 14.6% → 回升）|
| 严重度 | 🟡 中（功能可用性受损，非静默精度损坏；但“假活”状态易误导排查方向） |
| 隐蔽性 | 🟡 中高（需 V4 compress + 多节点 TP 不等；CI 难复现；报错被 `except` 吞掉仅留日志） |
| 是否报错 | ✅ 报错（`ValueError`，非 silent） |

SpecDecoding 指标的剧烈波动是关键侧证：`Mean acceptance length` 从 2.0 跌到 1.15、`Avg Draft acceptance rate` 从 100% 跌到 14.6% 再回升到 97.6%——这与“部分请求 KV 未传输、D 端用错位/缺失 KV 做 speculative draft”完全吻合。

---

## 6. 修复方向

截至 2026-07-29，issue 仍 open，未发现直接关联的已合并修复 PR。基于根因分析，修复应朝以下方向：

### 6.1 对称化长度截断守卫（最小修复）

在 `_transfer_kv_cache_all_groups` 调用 `group_concurrent_contiguous` 前，确保两侧 block id 列表**严格等长**，且截断逻辑对称：

```python
# 伪代码：双向对齐
if not isinstance(self.kv_cache_specs[i], MambaSpec):
    n = min(len(cur_local_block_ids), len(cur_remote_block_ids))
    # 对齐到较短者；注意尾部对齐语义（prefix-trim 期望取尾）
    cur_local_block_ids  = cur_local_block_ids[-n:]
    cur_remote_block_ids = cur_remote_block_ids[-n:]
```

要点：必须确认 Compress 组下“取尾部 block”的语义正确性（compress 的 KV 是按压缩比聚合的，截断点需落在压缩边界）。简单 `min` 截断可能引入 compress 对齐错误，更稳妥的做法是按 compress ratio 向下取整对齐。

### 6.2 在 `group_concurrent_contiguous` 内防御

在 `np.where` 前断言或显式处理不等长，给出可定位的诊断信息而非 NumPy broadcasting 报错：

```python
if src_indices.size != dst_indices.size:
    raise ValueError(
        f"group_concurrent_contiguous: src/dst length mismatch "
        f"(src={src_indices.size}, dst={dst_indices.size}, group_spec=...). "
        f"This usually means hybrid compress group block ids are not aligned "
        f"between prefill and decode under unequal TP."
    )
```

这能把“广播失败”转化为可定位的业务错误，避免排查者在 NumPy 报错里迷失。

### 6.3 根治：per-group block id 对齐协议

更彻底的方向是在 **scheduler 侧构造 block id 元数据时**，就保证 P/D 端每个 group 的 `remote_block_ids[i]` 与 `local_block_ids[i]` 一一对应（考虑 compress ratio、TP 分片、MTP 层）。即：不等长不应在 worker 传输层才被发现，而应在元数据生成阶段消除。这与 [0_kvcache.md](./0_kvcache.md) §2.3 中 PR #11601/#11886（Mooncake 传输元数据携带 KV head / cache group ids）的改进方向一致——元数据需显式携带 per-group 的 block 计数与对齐信息。

### 6.4 配置侧规避（临时）

- 去除 D 端 `kv-transfer-config` 中**重复的 `kv_port` 键**（`'$mooncake_port'` 被 `"30800"` 覆盖），避免端口错配引入的额外不稳定。
- 尝试 P/D 端 TP 相等（如均 tp8）作为对照，若不再报错则进一步确认 TP 不等是触发因子。
- 关注 `kv_cache_tensor.shared_by` 在 V4 compress 下的非空保证（PR #9500 已修一类问题，确认本地版本已包含）。

---

## 7. 复现与验证

### 7.1 触发矩阵

| DeepSeek-V4 compress (hybrid) | PD 分离 | P/D TP 不等 | MTP | 是否触发 |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | — | — | — | ✗（非 hybrid，走单组路径） |
| ✓ | ✗ | — | — | ✗（无跨节点传输） |
| ✓ | ✓ | ✗（相等） | ✓ | 大概率 ✗（group block 计数可能仍对齐） |
| **✓** | **✓** | **✓** | — | **✓ 触发广播失败**（compress + TP 不等已足够） |
| ✓ | ✓ | ✓ | ✓ | ✅ 触发，且更不稳定（shape 波动） |

### 7.2 复现步骤（基于 issue 配置）

1. 4 台 A3，部署 DeepSeek-V4-Pro-w4a8-mtp，P 端 dp4tp8、D 端 dp8tp4。
2. P 端 `kv_role=kv_producer`，D 端 `kv_role=kv_consumer`，均用 `MooncakeHybridConnector` + `--no-disable-hybrid-kv-cache-manager`。
3. D 端开 `--speculative-config '{"method":"deepseek_mtp"}'` + `--async-scheduling`。
4. 发送连续 `/v1/completions` 请求。
5. 观察 D 端 `dp${dp_rank}_decoder.log`，搜索 `Failed to transfer KV cache for request` 与 `operands could not be broadcast together`。

### 7.3 验证修复

- 应用 §6.1 对称截断后，连续 100 请求无 `ValueError`；SpecDecoding 接受率稳定（不再 14.6% 跌幅）。
- 在 `group_concurrent_contiguous` 单测中，构造 `src=[0,1,2,...,179]`、`dst=[10,11,12,...,54]`（长度 180 vs 45）用例，断言不再抛 broadcasting 错误且分组正确。
- 补充 e2e：V4 Pro PD + dp4tp8/dp8tp4 + MTP 的长时压测，监控 `Mean acceptance length` 稳定性。

### 7.4 对照排查

1. **TP 相等对照**：D 端改 tp8（与 P 端同），若报错消失 → 锁定 TP 不等路径。
2. **关闭 hybrid 管理器对照**：去掉 `--no-disable-hybrid-kv-cache-manager`（若模型允许），走单组 `_transfer_kv_cache`，观察是否复现。
3. **dump per-group block id**：在 `_transfer_kv_cache_all_groups` 入口打印 `len(remote_block_ids[i])` vs `len(local_block_ids[i])` per group，定位是哪个 compress 组不等长。
4. **确认 `shared_by` 非空**：检查 `kv_cache_tensor.shared_by` 在 V4 compress 各 group 的取值，排除 PR #9500 类回归。

---

## 8. 经验与启发

### 8.1 “hybrid”命名陷阱：先搞清 hybrid 的是什么

`MooncakeHybridConnector` 的“hybrid”指 **KV cache group 类型混合（HMA）**，不是“多传输通道混合”。在排查“hybrid connector 报错”时，第一反应应是“模型是否有 compress/SWA/Mamba 组导致 per-group block 计数分叉”，而非“mooncake 与另一通道冲突”。文档 `mooncake_p2p_connector.md` 把它描述为“结合整体和逐层”是不精确的简化——实际代码做的是整体 `batch_transfer_sync_read` 多 group 传输，负载差异来自 group 类型而非传输方式。

### 8.2 不对称守卫 = 定时炸弹

`len(local) < len(remote)` 的单向截断是经典的“只覆盖快乐路径”bug。任何对两组列表做 zip/diff/广播的操作，守卫必须**双向对称**（`min(len)` 对齐），否则反方向不等长时会原地爆炸。这是 numpy vectorised 路径的高发陷阱——一旦换成逐元素 Python 循环，不等长会变成静默错位；换成 numpy 又会变成显式 broadcasting 错误。两类都不对。

### 8.3 Compress 组的 block 计数是 hybrid 传输的高风险面

DeepSeek-V4 的 Compress-4/Compress-128 使同一请求在不同 group 上的 block 数相差 4×~128×。任何按 group 独立计算 block id 的逻辑，都必须显式处理“group 间计数不对齐”——不能假设 `remote_block_ids[i]` 与 `local_block_ids[i]` 天然等长。这与 [0_kvcache.md](./0_kvcache.md) §2.3 反复出现的“non-contiguous / 5D / shared_by 空 / TP 不等 head 归属”同属一类：hybrid 布局下，**per-group 的维度/计数假设必须显式校验**。

### 8.4 “假活”状态比崩溃更危险

`except Exception: logger.exception(...)` 让进程不崩、服务继续响应，但 KV 实际没传。第一个请求靠运气完成，后续请求接受率剧烈波动。这种“报错但继续”的模式极易让运维误以为是“偶发网络抖动”而非“每次都失败的硬 bug”。对 KV 传输错误，应考虑**快速失败**（连续 N 次失败则熔断）而非无限吞错，避免长期静默退化。

### 8.5 多键 JSON 重复：配置校验缺失

D 端 `kv-transfer-config` 出现两个 `kv_port`，后者静默覆盖前者。这类“JSON 重复键”在手工拼接 shell 脚本时极易发生，且 Python `json.loads` 默认取最后一个不报错。应在 connector 初始化时对 `kv_transfer_config` 做 schema 校验（拒绝重复键 / 必填键缺失），把配置错误挡在启动阶段。

### 8.6 CI 覆盖盲区：多节点 × V4 compress × TP 不等

本 bug 需要 4 节点 + V4 compress + P/D TP 不等才能触发，单节点单测与 TP 相等的 e2e 都无法覆盖。这类“配置组合爆炸”场景是 vllm-ascend PD 分离的固有测试难点（参见 [0_kvcache.md](./0_kvcache.md) §8、§11.3）。建议至少补充“hybrid compress 组 block id 对齐”的 connector 单测，用构造的 group block 列表直接验证 `group_concurrent_contiguous` 的鲁棒性，无需真实多节点。

---

## 9. 关联问题

| 编号 | 关联点 |
|------|--------|
| [vllm-ascend#9500](https://github.com/vllm-project/vllm-ascend/pull/9500) | DeepSeek-V4 PD `kv_cache_tensor.shared_by` 可能为空——同一“V4 compress + PD”下的地址共享错配，是本 case block id 不对齐的潜在上游诱因 |
| [vllm-ascend#8540](https://github.com/vllm-project/vllm-ascend/pull/8540) | TP 不等时 MTP 层 KV 未处理——同属“PD 分离 + TP 不等 + MTP”组合，见 [1_pr8540_tp_unequal_mtp_kv.md](./1_pr8540_tp_unequal_mtp_kv.md) |
| [vllm-ascend#11601](https://github.com/vllm-project/vllm-ascend/pull/11601) / [#11886](https://github.com/vllm-project/vllm-ascend/pull/11886) | Mooncake 传输元数据携带 KV head / cache group ids——本 case 的“per-group block id 对齐”需同类元数据增强 |
| [vllm-ascend#12183](https://github.com/vllm-project/vllm-ascend/pull/12183) | Non-contiguous Mooncake PA cache inputs——同属 Mooncake 传输输入连续性/形状假设类，见 [0_kvcache.md](./0_kvcache.md) §2.3 |
| [vllm-ascend#7792](https://github.com/vllm-project/vllm-ascend/issues/7792) | `mooncake kv_both+GLM5 报错 Transfer slice failed 503900`——不同模型但同类 Mooncake 传输形状/切片失败 |
| [vllm#44238](https://github.com/vllm-project/vllm/issues/44238) | MooncakeConnector 并发 PD 传输数据损坏（`batch_transfer_sync_write` race）——同属 Mooncake 传输正确性族，见 [0_kvcache.md](./0_kvcache.md) §2.1 |
| [0_kvcache.md](./0_kvcache.md) §2.1 / §2.3 / 案例 9 | 本案例在全景文档中的归档位置；§2.1 并发竞争、§2.3 传输连续性/布局不匹配 |

---

## 10. 结论

Issue #10569 是 DeepSeek-V4 Pro 的 hybrid attention（Compress-4/128）在 PD 分离 + P/D TP 不等下，撞上 `MooncakeHybridConnector._transfer_kv_cache_all_groups` 中一个**不对称的 block-id 列表长度截断守卫**：守卫只处理 `len(local) < len(remote)`，反方向不等长时直接把不等列表送入 `group_concurrent_contiguous`，导致 `np.where((np.diff(src)!=1)|(np.diff(dst)!=1))` 因 `src`/`dst` 长度差（45 vs 180，源自 compress 组 block 压缩比）触发 `ValueError: operands could not be broadcast together with shapes (44,) (179,)`，KV 传输被跳过但服务“假活”，SpecDecoding 接受率剧烈波动。

> 本案例核心教训：**hybrid KV 传输中凡是对两个 per-group block id 列表做 zip/diff/广播的操作，长度截断守卫必须双向对称（min 对齐），并在元数据层显式保证 compress 组的 block 计数对齐**——否则 DeepSeek-V4 compress attention 在 PD 分离 + TP 不等下会以“广播失败”硬报错，而非静默精度下降的方式暴露 per-group 计数错位。