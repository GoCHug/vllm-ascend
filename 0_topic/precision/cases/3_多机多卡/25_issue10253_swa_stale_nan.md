# 案例25：PD 分离 SWA KV 传输先 clip 后 trim 混入 stale block 产生全 NaN 输出

> **一句话定位**：P 端 `request_finished_all_groups()` 对同一份 block 列表的两次尾部裁剪（prompt-trim 砍未写入、SWA tail clip 选窗口）顺序写反 / 对 SWA 组跳过 trim，导致 P 端截断留下的未写入 dirty block 被 SWA clip 选中并经 Mooncake 传输到 D 端，attention 消费 NaN 后逐层放大为全 NaN 输出。
>
> **对象**：vllm-project/vllm-ascend [#10253](https://github.com/vllm-project/vllm-ascend/issues/10253)（Bug，closed 已修复）

---

## 1. 问题描述

### 1.1 现象

在 **PD 分离 + SWA（滑动窗口）模型 + 足够并发** 下：

- D 端 decode 产出的 hidden states **全部为 NaN**，端到端输出彻底腐败、不可用；
- 无崩溃、无断言——传输与 attention 算子不会因读到 NaN / 未初始化数据而 assert fail，表现为 **silent 全 NaN**；
- **并发越高触发概率越大**：低并发可跑通，升高并发才复现（dirty/stale block 的产生与跨 request 的 block 复用强相关，并发放大了「过期尾部 block 恰好落入 SWA 窗口」的概率）。

### 1.2 触发条件（必现矩阵）

| PD 分离 | SWA 模型 | 足够并发 | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗ | — | — | ✗（无跨节点传输） |
| ✓ | ✗ | — | ✗（`num_swa_blocks==0`，无 tail clip） |
| ✓ | ✓ | ✗（低并发） | △（难复现，dirty tail 难落窗口） |
| **✓** | **✓** | **✓（高并发）** | **✓ 全 NaN 输出** |

> 单卡 / 非 SWA 模型不触发；SWA + PD 本身并不必然触发，取决于「截断多分配的 block 数」与「SWA 窗口 block 数」的相对位置。

### 1.3 影响与严重度

- **严重度**：🔴 **极高**（端到端不可用，非「质量下降」而是「彻底崩坏」）。
- **隐蔽性**：🟡 高（需 SWA + PD + 并发三要素；CI 默认低并发用例难复现）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | Issue [#10253](https://github.com/vllm-project/vllm-ascend/issues/10253)（Bug） |
| 状态 | closed（已修复） |
| vllm-ascend 版本 | 复现于 `releases/v0.20.2rc`（main base `77ff286`）；修复 PR [#10254](https://github.com/vllm-project/vllm-ascend/pull/10254)（main）+ [#10255](https://github.com/vllm-project/vllm-ascend/pull/10255)（v0.20.2rc） |
| 上游 vllm | v0.20.2（本地校验用 `/vllm-workspace/vllm`，version 0.20.2） |
| 创建时间 | 2026-06-09 |
| 修复落点 | `mooncake_hybrid_connector.py`（~1442）、`mooncake_connector.py`（~1848）——顺序修正已合入 main |
| 修复集 | 顺序修正 + SWA 组 trim + 传输前过滤占位 block（`block_id != 0`） |

> 说明：该对象为 issue，修复落地于 [#10254](https://github.com/vllm-project/vllm-ascend/pull/10254)（main）与 [#10255](https://github.com/vllm-project/vllm-ascend/pull/10255)（v0.20.2rc）；修复以「先 trim 再 clip」的代码顺序修正落地于两处 connector，配 `Drop those unwritten blocks before SWA tail clipping` 注释与专项单测固化。

---

## 3. 定位过程

1. **确认三要素**：是否 PD 分离 + 是否含 SWA 层（`SlidingWindowSpec`）+ 复现时并发量。
2. **单卡 / 非 SWA 对照**：换单卡或换非 SWA 模型后 NaN 消失 → 锁定 SWA + PD 传输路径。
3. **降并发对照**：低并发不复现、高并发复现 → 高度怀疑 stale block 复用相关。
4. **D 端 dump 检查**：dump 接收到的 KV block，比对 P 端对应 block 的实际写入状态，确认是否含 NaN / 未初始化值。
5. **核对 `remote_block_ids`**：若待传输列表长度 > 实际写入 block 数 → dirty block 混入。
6. **定位根因**：发现 `request_finished_all_groups()` 中 SWA tail clip 与 prompt-trim 顺序颠倒，且 `_compute_transfer_block_ids` 对 SWA 组跳过了 prompt-trim。

> 定位要点：NaN 不是传输代码产生的，而是 D 端 attention 消费 dirty block 的**下游症状**——看到全 NaN 输出应第一时间回溯「是否有 dirty/stale block 进了传输链路」，而非先怀疑算子数值稳定性。

---

## 4. 解决方案

### 4.1 根因

「待传输 block」的两次尾部裁剪方向相同但语义不同，顺序不可交换：

- **prompt-trim**：P 端 `_truncate_request_for_prefill` 截断产生「已分配未写入」的 dirty 尾部 block，应从尾部砍掉；
- **SWA tail clip**：`blocks[-num_swa_blocks:]` 从尾部只取窗口内 block。

修复前先 clip 后 trim（或对 SWA 组跳过 trim），使 SWA clip 在「未剔除 dirty block」的列表尾部取窗口，恰好把未写入 / 过期 block 选中 → 经 Mooncake 传到 D 端 → attention 消费 NaN → 逐层全 NaN。

### 4.2 修复内容

**① 修正顺序：先 trim 再 clip**

```python
# mooncake_hybrid_connector.py（修复后）
# P-side truncation can leave block ids allocated for the original
# prompt length. Drop those unwritten blocks before SWA tail clipping.
computed_block_ids = self._compute_transfer_block_ids(block_ids, request.num_prompt_tokens)  # ① 先 trim
computed_block_ids = self.get_sw_clipped_blocks(computed_block_ids)                          # ② 再 clip
```

```python
# mooncake_connector.py（非 hybrid，修复后）
computed_block_ids = self._get_transfer_block_ids(block_ids, len(request.prompt_token_ids))  # ① 先 trim
computed_block_ids = self._get_swa_transfer_block_ids(computed_block_ids)                    # ② 再 clip
```

**② 对 SWA 组不再跳过 trim**：`_compute_transfer_block_ids` / `_get_transfer_block_ids` 按 prompt 长度截断（与是否 SWA 无关），SWA 组的 dirty 尾部同样被剔除。

**③ 传输前过滤占位 block**：`_get_swa_transfer_block_ids` 在 clip 后过滤占位 block `0`（`[id for id in window_blocks if id != 0]`），作为脏块漏网的兜底。

### 4.3 验证

- 专项单测钉死顺序契约，防回归：
  - `test_request_finished_trims_before_swa_clip` —— 直接钉死「先 trim 再 SWA clip」；
  - `test_compute_transfer_block_ids_trims_swa_groups` —— SWA 组经 trim 后长度正确；
  - `test_request_finished_trims_mtp_before_swa_tail_clip` —— `prompt_len=64`、`blocks_per_window=3`，断言 `remote_block_ids == ([12,13,14],)`。
- 修复后高并发压测不再出 NaN。

## 5. 复现方法

### 5.1 最小复现模型

- **Mistral-7B-v0.1**（`mistralai/Mistral-7B-v0.1`）：经典最小 SWA 模型（sliding_window=4096），dense 架构体量小。
- 触发要点：SWA 层 `num_swa_blocks > 0` 触发 tail clip；配合 P 端截断产生 dirty 尾部 block + 高并发放大 block 复用，「stale block 落入 SWA 窗口」概率升高。

### 5.2 最小服务命令（P/D 两端 TP=8，走 SWA clip 路径）

```bash
# P 端（prefill）
export VLLM_USE_V1=1
vllm serve mistralai/Mistral-7B-v0.1 \
  --port 8010 \
  --tensor-parallel-size 8 \
  --enforce-eager --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_port":"36000","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":8}}}'

# D 端（decode）
export VLLM_USE_V1=1
vllm serve mistralai/Mistral-7B-v0.1 \
  --port 8020 \
  --tensor-parallel-size 8 \
  --enforce-eager --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_port":"36100","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":8}}}'
```

> 关键差异项：SWA 模型（Mistral-7B）+ **高并发**（低并发不触发）。服务起好后用高并发压测打满 block pool（如 `--request-rate 200` 持续请求，`--max-num-seqs` 适当调大），放大 dirty/stale tail 落窗概率；复现前先确认模型含 `SlidingWindowSpec` 且 `num_swa_blocks > 0`。

---

> **核心教训**：跨节点 KV 传输中，凡是对同一份 block 列表做多次裁剪/选取（prompt-trim 与 SWA tail clip），必须保证「丢弃未写入 block」先于「按窗口选尾部」执行——顺序一反，dirty/stale block 即混入传输，在 D 端 attention 中放大为全 NaN 输出。