# 案例33：GLM-5.1 PP4 + ascend_direct RDMA 首请求 V cache 跨 PP stage 未 flush 致乱码

> **一句话定位**：PD 分离 + PP4 + ascend_direct RDMA 下，P 侧 `sfa_v1.py` 每层 `npu_kv_rmsnorm_rope_cache` 将 k_pe 异步写入 `kv_cache[1]`，`kv_ag_handle.wait()` 只等 all-gather 通信完成、不 flush NPU 计算流；prefill forward 返回后 D 侧 RDMA pull 读到未落 HBM 的 stale V cache，在三个 PP stage 边界（layer 20/40/60）引入数值突变甚至正负翻转，首请求输出乱码。
>
> **对象**：vllm-project/vllm-ascend Issue [#12957](https://github.com/vllm-project/vllm-ascend/issues/12957)（Bug，open；修复方案在 issue 评论中验证通过，尚无 merged PR）

---

## 1. 问题描述

### 1.1 现象

在 **GLM-5.1 W8A8 · A3 16 卡 · PD 分离 · Prefill PP=4/TP=8/DP=1 · Decode DP=8/TP=4 · ascend_direct RDMA (ADXL)** 的部署下：

- 服务拉起后**首个请求**为长中文 prompt（prompt_tokens=261，3 个 KV blocks）→ 输出乱码：与 prompt 无关的随机英文，如 `"The is a simple command-line tool that prints its input... echo is a command that prints its input back to the terminal..."`（finish=length，content=null）；
- 乱码后发普通短请求 → **正常**；继续发 badcase → **仍乱码**；
- **对照**：若服务起来后先发一个普通短请求（如"1+1"），再发 badcase → **不乱码**（即使后续发 badcase 也不乱）。

### 1.2 触发条件（必现矩阵）

| PD 分离 | PP>1 (PP4) | ascend_direct RDMA | 首请求为长 prompt | 是否触发 |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | — | — | — | ✗ |
| ✓ | ✗ (DP4, 无 PP) | ✓ | ✓ | ✗（无 PP stage 边界） |
| ✓ | ✓ | ✗ | ✓ | ✗（非 RDMA pull 路径） |
| ✓ | ✓ | ✓ | ✗ (短 prompt) | ✗（时序不触发） |
| **✓** | **✓ (PP4)** | **✓** | **✓ (长 prompt)** | **✓ V cache 突变 + 乱码** |

> 作者消融实验结论：关闭 MTP / prefix cache / reasoning parser / enforce_eager / slot_mapping=-1 / drafter slot_mapping=-1 **均仍乱码**；P 节点改 DP=4（无 PP）**乱码消失** → 锁定 **PP4 暴露时序问题**。

### 1.3 影响与严重度

- **严重度**：🔴 高（首请求即乱码，影响线上服务冷启动后首个用户请求）。
- **隐蔽性**：🟡 中高（需 PP4 + ascend_direct RDMA + 特定长 prompt 首请求才触发；短请求或非首个请求均正常，常规 CI 难覆盖）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | Issue [#12957](https://github.com/vllm-project/vllm-ascend/issues/12957)（Bug，open） |
| vllm 版本 | 0.20.2 |
| vllm-ascend 版本 | 0.20.2 |
| 模型 | glm-5.1-w8a8 |
| 硬件 | 4 台 a3 16 卡 pod（同超节点） |
| Prefill 配置 | PP=4, TP=8, DP=1, `PREFILL_PP_LAYER_PARTITION="20,20,20,18"` |
| Decode 配置 | DP=8, TP=4 |
| kv_role | prefill=kv_producer, decode=kv_consumer |
| 修复状态 | 方案在 issue 评论验证通过，尚无 merged PR |

---

## 3. 定位过程

### 3.1 消融实验（排除无关因素）

作者逐一排除以下因素，均未消除乱码：

- 关闭 MTP
- 关闭 prefix cache（P、D 分别）
- 关闭 reasoning parser
- kvcache 清零（dummy run 后强行覆盖赋值）
- 回退 slot_mapping=-1（claudecode 偶现乱码修复 PR）
- drafter slot_mapping=-1（对上一修复 PR 的补充）
- 关闭图模式（enforce_eager）

**关键转折**：P 节点改 DP=4（无 PP）→ 乱码消失 → 缩小范围至 **PP4 暴露时序问题**。

### 3.2 KV cache 数值比对实验（DP4 vs PP4）

@MarinaMiao 对 D 侧收到的 KV cache 逐层求和比对：

```python
def _transfer_kv_cache(self, req_meta: dict[str, Any]):
    _fl = [b for sub in grouped_local_block_ids for b in sub]
    _kv = list(self.kv_caches.values())
    _sums = []
    for _li in range(first_layer_index, end_layer_index):
        _t = _kv[_li]
        _per_block = []
        for _b in _fl:
            _ksum = round(float(_t[0][_b].abs().sum()), 3)
            _vsum = round(float(_t[1][_b].abs().sum()), 3) if len(_t) > 1 else None
            _spsum = round(float(_t[2][_b].abs().sum()), 3) if len(_t) > 2 else None
            _per_block.append((_ksum, _vsum, _spsum))
            _sums.append(_per_block)
        logger.info("[mrl-kvcmp] req=%s pp_rank=%s ... per_block_k_v_sp_abssum=%s",
                    remote_request_id, prefill_pp_rank, ..., _sums)
```

**比对结果**：

| 维度 | badcase (PP4 乱码) | simple case (正常) |
|------|-------------------|-------------------|
| V cache | layer 21/41/61 处**突变**（PP stage 边界 20→21, 40→41, 60→61），出现正负翻转 | DP4 vs PP4 偏差仅 0–0.9% |
| K cache | DP4 vs PP4 差不多 | 完全匹配 |
| sparse cache | DP4 vs PP4 差不多 | 完全匹配 |

**结论**：V cache 在 PP pipeline 的 stage 间通信出现数值损坏，每个 stage 边界引入一次偏差。

### 3.3 根因定位

在 `vllm_ascend/attention/sfa_v1.py` 的 `attn.forward()` 中：

1. `exec_kv()` 调用 `npu_kv_rmsnorm_rope_cache()` 将 k_pe **异步写入** `kv_cache[1]`（V cache）；
2. 随后 `all_gather_async()` 对 fused KV 做 TP 间异步 all-gather；
3. `kv_ag_handle.wait()` **只等 all-gather 通信完成**，不 flush NPU 计算流上的 pending writes；
4. prefill forward 返回后，D 侧 RDMA pull 直接从 HBM 读 KV cache —— NPU 异步写入尚未落盘，读到 stale k_pe → V cache 在 PP stage 边界数值突变 → 乱码。

---

## 4. 解决方案

### 4.1 根因

P 侧 prefill 每层 `npu_kv_rmsnorm_rope_cache` 将 k_pe 写入 `kv_cache[1]` 后，NPU 异步写入不会自动对跨引擎（D 侧 RDMA pull）可见。`kv_ag_handle.wait()` 仅同步通信流，不同步计算流；prefill 返回后 D 侧立即 pull，读到 stale 数据。PP4 下每个 stage 边界（layer 20→21, 40→41, 60→61）引入一次偏差，三层累积导致 V cache 正负翻转，输出乱码。

### 4.2 修复内容（`vllm_ascend/attention/sfa_v1.py`，+3 行）

```python
# sfa_v1.py, attn.forward(), enable_dsa_cp 分支内
  if self.enable_dsa_cp:
      if kv_ag_handle is not None:
          kv_ag_handle.wait()
+         # Synchronize the current stream after the all-gather of k_pe
+         # completes, so the gathered k_pe is visible to the decode-side
+         # RDMA pull that runs once the prefill forward returns. Without
+         # this, NPU async writes are not automatically visible to the
+         # cross-engine pull, causing stale k_pe and garbled output on
+         # the first request (PP4 + content-specific prompts).
+         torch.npu.current_stream().synchronize()
```

**作用点**：P 侧 prefill 每层 `kv_ag_handle.wait()` 后立即 flush NPU 计算流，确保 k_pe 写入落 HBM 再继续下一层。

**频率**：78 层 × 1 = 78 次 synchronize / prefill（只在 P 侧生效；D 侧 `enable_dsa_cp=False` 不进此分支）。

**代码位置**：`vllm_ascend/attention/sfa_v1.py:1819`，`kv_ag_handle.wait()` 之后。

### 4.3 验证

- 已验证：首发 badcase 请求返回正常（不再乱码）。
- 尚无 CI / 单测覆盖（issue open，无 merged PR）。

---

## 5. 复现方法

### 5.1 最小复现模型

- **glm-5.1-w8a8**（GLM-5.1 W8A8 量化模型）。

### 5.2 最小复现配置

作者进一步缩小范围，以下单节点脚本即可复现（无需完整 4 台 PD 分离）：

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH=/path/to/glm-5.1-w8a8
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HCCL_BUFFSIZE=1024
export VLLM_PP_LAYER_PARTITION="20,20,20,18"

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 8100 --served-model-name auto --trust-remote-code \
  --distributed-executor-backend mp \
  --quantization ascend --enforce-eager --enable-expert-parallel \
  --block-size 128 --gpu-memory-utilization 0.92 --max-model-len 4096 \
  --tensor-parallel-size 4 --pipeline-parallel-size 4 --data-parallel-size 1 \
  --no-async-scheduling --no-enable-prefix-caching
```

### 5.3 完整 PD 分离复现

**Prefill 端**（PP=4, TP=8, DP=1）：

```bash
# start_p.sh
export VLLM_PP_LAYER_PARTITION="20,20,20,18"
vllm serve "$MODEL_PATH" \
  --port 8100 --served-model-name auto --trust-remote-code \
  --distributed-executor-backend mp \
  --quantization ascend --enforce-eager --enable-expert-parallel \
  --block-size 128 --gpu-memory-utilization 0.92 --max-model-len 4096 \
  --tensor-parallel-size 8 --pipeline-parallel-size 4 --data-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_port":"36000","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":4}}}'
```

**Decode 端**（DP=8, TP=4）：

```bash
# start_d.sh
vllm serve "$MODEL_PATH" \
  --port 8200 --served-model-name auto --trust-remote-code \
  --distributed-executor-backend mp \
  --quantization ascend --enforce-eager --enable-expert-parallel \
  --block-size 128 --gpu-memory-utilization 0.92 --max-model-len 4096 \
  --tensor-parallel-size 4 --data-parallel-size 8 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_port":"36100","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":4}}}'
```

### 5.4 触发请求

服务拉起后，**第一个请求**发送以下长中文 prompt（prompt_tokens=261，3 个 KV blocks）：

```bash
curl --location "http://<proxy>:7001/v1/chat/completions" \
  --header 'Content-type: application/json' \
  --data '{
    "model": "auto",
    "messages": [
      {
        "role": "user",
        "content": "人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。自20世纪50年代诞生以来，AI经历了多次起伏，如今得益于大数据、强大算力和先进算法，迎来了蓬勃发展。机器学习作为AI的核心，让计算机通过数据学习规律，而不需要显式编程。深度学习更是通过多层神经网络，在图像识别、语音识别和自然语言处理等领域取得了突破性进展。在实际应用中，AI已广泛用于医疗影像诊断、自动驾驶汽车、智能客服、金融风险预测和个性化推荐等场景，极大提升了效率和便利性。然而，AI的快速发展也带来诸多挑战，如数据隐私、算法偏见、就业替代和武器化风险，引发了社会各界的广泛讨论。为了确保AI的安全可控，研究者们提出了可解释AI、联邦学习、对抗训练等技术，政府和国际组织也纷纷制定伦理准则和法律法规。展望未来，AI有望与脑科学、量子计算等前沿领域融合，创造出更强大的通用人工智能，但也必须警惕潜在的超智能风险。我们应当在发展技术的同时，加强人文关怀，确保AI造福全人类。您对以上关于人工智能的论述有何看法？请结合您的知识，给出您对这篇文章的评价、补充或不同意见。"
      }
    ],
    "stream": false,
    "max_tokens": 128,
    "temperature": 0
  }'
```

**预期（复现成功）**：返回乱码——与 prompt 无关的随机英文（如 `"The is a simple command-line tool... echo is a command..."`），finish=length，content=null。

### 5.5 对照实验

| 操作 | 结果 |
|------|------|
| 服务拉起后首发上述长 prompt | ✅ 乱码 |
| 乱码后发短 prompt（如 "1+1"） | ❌ 正常 |
| 继续发上述长 prompt | ✅ 仍乱码 |
| 服务拉起后**先发**短 prompt，再发长 prompt | ❌ 不乱码 |

---

## 核心教训

P 侧 prefill 每层通过 `npu_kv_rmsnorm_rope_cache` 将 k_pe 异步写入 `kv_cache[1]` 后，`kv_ag_handle.wait()` **只同步通信流、不 flush 计算流**；prefill 返回后 D 侧 RDMA pull 直接从 HBM 读，NPU 异步写入尚未落盘 → stale k_pe → V cache 在 PP stage 边界突变 → 乱码。**跨引擎 KV 消费前必须显式 synchronize NPU current stream**，不能假设 `wait()` 覆盖计算流。PP4（有多 stage 边界）比 DP4（无 PP）更容易暴露此时序问题，因为每个 stage 边界都引入一次偏差累积。