# 案例32：Ascend950 Qwen3.5-397B W8A8 MXFP8 FULL_QUANT PD 分离下精度「一次正常一次异常」交替

> **一句话定位**：超大 MoE 模型（Qwen3.5-397B）W8A8 MXFP8 FULL_QUANT + EP + PD 分离下，发起相同请求时输出在「正常答案」与「异常答案」之间交替出现，属于量化 + 大规模并行 + 跨节点传输叠加下的**非稳定精度（non-deterministic accuracy）**问题。
>
> **对象**：vllm-project/vllm-ascend Issue [#12339](https://github.com/vllm-project/vllm-ascend/issues/12339)（Bug，closed「已解决」）

---

## 1. 问题描述

### 1.1 现象

在 **Ascend950 · Qwen3.5-397B-W8A8-MXFP8-FULL_QUANT · EP（Expert Parallel）· PD 分离（MooncakeConnectorV1）· 不开 MTP** 下：

- 对同一道数学题（"James decides to run 3 sprints…"）发起请求，**一次返回正常答案、一次返回异常答案，交替出现**；
- 异常答案表现为 `completion_tokens=1`（几乎不生成，`stop_reason=248044`），与正常答案（165 tokens、完整推理）交替；
- GPQA diamond 压测 accuracy 异常（47.47），反映端到端精度不稳定。

### 1.2 触发条件（必现矩阵）

| PD 分离 | W8A8 MXFP8 FULL_QUANT | EP（expert parallel） | 超大 MoE（397B） | 是否触发 |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | — | — | — | ✗（无跨节点传输） |
| ✓ | ✗ | — | — | ✗（精度问题随量化 + EP + 超大模型叠加出现） |
| **✓** | **✓** | **✓** | **✓** | ✓ 精度交替异常 |

> 该 issue 未给出严格的单变量对照矩阵；现象明确为「请求一次正常一次异常交替」，且依赖 PD 分离 + FULL_QUANT + EP + 超大 MoE 的组合。

### 1.3 影响与严重度

- **严重度**：🔴 高（端到端回答非确定、正确性不可用，且是 397B 部署级问题）。
- **隐蔽性**：🟡 高（「交替出现」意味着单次复现不稳定，需多次请求对照；同 input 不同 output 的 non-determinism 尤其难定位）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | Issue [#12339](https://github.com/vllm-project/vllm-ascend/issues/12339)（Bug） |
| 状态 | closed（作者回复「已解决」，completed） |
| 硬件 | Ascend950（B130） |
| vllm-ascend 版本 | releases/v0.23.0 [`6cee2e1`](https://github.com/vllm-project/vllm-ascend/commit/6cee2e16308ae34aedb7c3b7ed2df0d42284713e) |
| 上游 vllm 版本 | v0.23.0 |
| torch / CANN | torch 2.10.0 · CANN 9.1.0.B103 |
| 模型 | Qwen3.5-397B-W8A8-MXFP8-FULL_QUANT |
| 关键配置 | TP8 + EP（`--enable-expert-parallel`）+ PD 分离（`MooncakeConnectorV1`）· D 端 `VLLM_USE_V1=1` + `cudagraph_mode=FULL_DECODE_ONLY` |

> 说明：该对象为 issue，作者仅在后续回复「已解决」并 closed，**未在 issue 内记录具体修复 PR 号 / 定位结论**，故定位过程填写「未知」，修复落点待查对应版本后续 commit。

---

## 3. 定位过程

**未知**（issue 正文与评论均未披露具体定位步骤、根因与修复 PR）。

- 已知排除点：不开 MTP（排除 MTP）；请求体一致（排除 prompt 差异）；engine 就绪（排除启动失败）。
- 线索方向：问题与「W8A8 MXFP8 FULL_QUANT + EP + PD 分离 + 大并发」强相关，同 input 异 output 的交替行为指向**量化反量化 / EP all-to-all / 跨节点 KV 传输中的非确定路径**。

> 待补：若能定位到关闭 FULL_QUANT、关闭 EP、关闭 PD 分离的对照组结果，可进一步收敛根因。

---

## 4. 解决方案

### 4.1 根因

未披露（issue 未给出根因分析）。据现象推断与 W8A8 MXFP8 FULL_QUANT 量化 + EP + PD 分离叠加下的数值非确定性有关。

### 4.2 修复内容

未披露具体修复 diff / PR。作者在 2026-08-03 回复「已解决」并 closed as completed。

### 4.3 验证

作者 closed completed；未附验证数据。

---

## 5. 复现方法

### 5.1 最小复现模型

- **Qwen3.5-397B-W8A8-MXFP8-FULL_QUANT**（Ascend950 B130）；触发依赖超大 MoE + FULL_QUANT + EP + PD 分离，无更小替代（量化方法 `ascend`）。

### 5.2 最小服务命令（P/D 两端 TP8 + EP + FULL_QUANT）

```bash
# 公共环境
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# P 端（prefill，MooncakeConnectorV1 producer）
export VLLM_USE_V1=0
vllm serve <Qwen3.5-397B-W8A8-MXFP8-FULL_QUANT> \
  --host <P_IP> --port 8010 \
  --tensor-parallel-size 8 --data-parallel-size 1 \
  --max-num-seqs 32 --max-model-len 133120 --max-num-batched-tokens 20480 \
  --trust-remote-code --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching --distributed-executor-backend mp \
  --quantization ascend --enable-expert-parallel --enforce-eager \
  --additional-config '{"recompute_scheduler_enable": true}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"21000","engine_id":"0","kv_buffer_device":"npu","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}'

# D 端（decode，MooncakeConnectorV1 consumer，V1 + FULL_DECODE_ONLY）
export VLLM_USE_V1=1
vllm serve <Qwen3.5-397B-W8A8-MXFP8-FULL_QUANT> \
  --host <D_IP> --port 8020 \
  --tensor-parallel-size 8 --data-parallel-size 1 \
  --max-num-seqs 16 --max-model-len 133120 --max-num-batched-tokens 20480 \
  --trust-remote-code --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching --distributed-executor-backend mp \
  --quantization ascend --enable-expert-parallel \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"recompute_scheduler_enable":true,"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":false}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"27000","engine_id":"1","kv_buffer_device":"npu","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}'
```

> 关键差异项：`MooncakeConnectorV1` + `--quantization ascend`（W8A8 MXFP8 FULL_QUANT）+ `--enable-expert-parallel`；D 端 `VLLM_USE_V1=1`。复现方式：连续多次发送同一道题（`stream:false` + `chat_template_kwargs.thinking:true`），观察 `completion_tokens` 在 1 与 165 之间交替、答案正确/错误交替。

---

> **核心教训**：超大 MoE + FULL_QUANT + EP + PD 分离是多层非确定性的叠加层，「同 input 异 output」的交替精度问题最棘手——务必用**同一请求连续多次**测出交替模式，再逐层做「关量化 / 关 EP / 关 PD / 关 V1」的对照组收敛根因；不要用单次偶现掩盖非稳定精度。