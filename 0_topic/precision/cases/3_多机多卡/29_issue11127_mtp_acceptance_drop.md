# 案例29：GLM-5.1 PD 分离 + MTP 长时间压测后接受率退化致精度异常（seq_lens 边界比较错误）

> **一句话定位**：PD 分离 + decode 开启 MTP 长期压测后，speculative decode 侧把「序列长度恰好等于 `max_model_len`」的合法序列用 `>=` 误判为越界并 `masked_fill_` 成 1，导致该序列 attention 只算 1 个 token，MTP 接受率退化为 <1%、无法恢复，手工请求输出精度异常。
>
> **对象**：vllm-project/vllm-ascend Issue [#11127](https://github.com/vllm-project/vllm-ascend/issues/11127)（Bug，closed），修复落地于 PR [#10117](https://github.com/vllm-project/vllm-ascend/pull/10117)（BugFix，merged）

---

## 1. 问题描述

### 1.1 现象

在 **GLM-5.1 W8A8 · A3 PD 分离（P/D Disaggregation）· decode 开启 MTP 投机解码** 的部署下：

- 压测 1~2 小时后，**decode 节点的 DP（Data Parallel）逐渐出现 MTP 接受率（acceptance rate）不足 1% 且无法自愈**；
- 接受率退化后**手工 curl 请求出现精度问题**（输出内容错误、非正常作答）；
- 无崩溃、无断言——为 **silent 精度退化**：MTP 候选持续不被接受，实际退化为按 draft 路径乱生成，端到端输出质量雪崩。

### 1.2 触发条件（必现矩阵）

| PD 分离 | 开启 MTP | 长时间压测（逼近 max_model_len） | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗ | — | — | ✗（无跨节点 MTP 链路） |
| ✓ | ✗ | — | ✗（关闭 MTP 后无精度问题，作者排除实验） |
| ✓ | ✓ | ✗（短请求） | △（难复现，序列未触及边界） |
| **✓** | **✓** | **✓** | **✓ 接受率 <1% + 精度异常** |

> 作者排除实验结论：**开启 kv_pool 池化与否都有精度问题**（排除池化）；**关闭 MTP 后无精度问题**（锁定 MTP 链路）。因此三要素为 PD 分离 + MTP + 长期压测逼近 `max_model_len`。

### 1.3 影响与严重度

- **严重度**：🔴 高（接受率永久退化 + 输出精度异常，且「一旦退化无法自愈」）。
- **隐蔽性**：🟡 高（需长时间压测、序列推进到 `max_model_len` 边界才触发；短请求 / 常规 CI 难覆盖）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | Issue [#11127](https://github.com/vllm-project/vllm-ascend/issues/11127)（Bug） |
| 状态 | closed（作者确认「合入 PR #10117 后修复」） |
| 报告版本（vllm-ascend） | **v0.20.0**（issue 标题明确「0.20 版本」） |
| 模型 | GLM-5.1 W8A8（A3 PD 分离版） |
| 修复 PR | [#10117](https://github.com/vllm-project/vllm-ascend/pull/10117)（BugFix，merged 到 main） |
| 修复 PR 的 vllm-ascend 版本 | v0.20.2 |
| 修复 PR 对应的上游 vllm | [`vllm-project/vllm@9090368`](https://github.com/vllm-project/vllm/commit/9090368b650896bf5fc990c921df7eb4c20355a5) |
| 改动规模 | 2 个 commit（含一次 revert `#9221` + 掩码修正） |
| 影响文件 | `vllm_ascend/spec_decode/llm_base_proposer.py` 等 |
| 关联上游 | 上游 vllm [#44185](https://github.com/vllm-project/vllm/issues/44185)（multi-DP 场景 MTP hang） |

---

## 3. 定位过程

1. **排除池化**：作者实验「开启 / 关闭 kv_pool 池化都有精度问题」→ 排除 KV Pool 特性。
2. **锁定 MTP**：作者实验「关闭 MTP 时没有精度问题」→ 把范围收敛到 MTP 投机解码链路（`spec_decode`）。
3. **确认场景叠加**：置 triaged 后 maintainer 要求提供压测 plog 与 MTP 配置，用于定位 MTP + PD 分离下 acceptance 退化的根因。
4. **定位根因**：在 `spec_decode/llm_base_proposer.py` 中，对「超过 max model length 的请求」序列长度做了 `masked_fill_(exceeds_mask, 1)` 处理，但判断条件用了 **`>= self.max_model_len`**：序列长度**恰好等于** `max_model_len` 是合法值（1-based，最大合法长度即 `max_model_len`），`>=` 会把这类合法序列误判为越界、截断成 1，导致 attention 只计算 1 个 token、MTP 候选全部失真。
5. **关联上游**：同一 `input_fits_in_drafter` 逻辑在 multi-DP + MoE 场景还会带来 hang（上游 vllm #44185），一并修复。

> 定位要点：接受率退化的表象在「MTP 不接受候选」，但真正根因是 **边界比较 `>=` 应改为 `>`**——「等于上限」是合法状态，不是越界。看到「压测一段时间后精度退化」应先查序列长度是否推进到了某个边界并被误截断。

---

## 4. 解决方案

### 4.1 根因

`spec_decode` 对「触发 max-model-len 边界」的序列做收缩处理时，比较符用错：

- `seq_lens` 语义上 1-based，长度 = `max_model_len` 仍合法；
- 用 `>=` 判断「是否越界」会把**恰好等于上限**的合法序列误判，`masked_fill_` 成 1 → attention 失效；
- 长时间压测下 GPU/decode 队列中序列逐步推进到 `max_model_len`，命中概率上升 → 表现为「压测 1~2 小时后逐渐出现」。

### 4.2 修复内容

PR #10117 主要做了两件事：

**① 边界比较 `>=` → `>`**

```diff
-  exceeds_mask = common_attn_metadata.seq_lens[:batch_size] >= self.max_model_len
+  exceeds_mask = common_attn_metadata.seq_lens[:batch_size] > self.max_model_len
```

`seq_lens` / `seq_lens_cpu` / `_seq_lens_cpu` 三处（及 CPU 侧 mirror）同步改为严格大于。

**② drop 掉有问题的 `input_fits_in_drafter` 分支**

移除 `model_runner_v1.py` 中 `input_fits_in_drafter` 检查及其 fallback 路径（该逻辑在 multi-DP + MoE 场景引入 hang，见上游 vllm #44185），并 revert 掉此前为解决「MTP tokens 越界」而引入的 #9221 改动。

### 4.3 验证

- 由 CI 覆盖；作者在 issue 明确回复「该问题在主干已解决，合入 pr10117 后问题修复」并 closed as completed。

---

## 5. 复现方法

### 5.1 最小复现模型

- **GLM-5.1 W8A8**（issue 中实际模型，A3 PD 分离版）；最小替换可用 **GLM-4.5 / GLM-5** 系列 dense + MTP 模型，核心是「PD 分离 + MTP + 可把序列推进到 `max_model_len` 的长请求 / 压测」。

### 5.2 最小服务命令（P/D 两端，decode 开 MTP）

```bash
# P 端（prefill）
export VLLM_USE_V1=1
vllm serve <GLM_MODEL> \
  --port 8010 \
  --tensor-parallel-size 8 \
  --enforce-eager --trust-remote-code \
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_port":"36000","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":8}}}'

# D 端（decode，开 MTP）
export VLLM_USE_V1=1
vllm serve <GLM_MODEL> \
  --port 8020 \
  --tensor-parallel-size 8 \
  --enforce-eager --trust-remote-code \
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_port":"36100","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":8}}}'
```

> 关键差异项：decode 端开 MTP（GLM 用 `deepseek_mtp`）+ 长时间压测让序列推进到 `max_model_len`（可故意把 `--max-model-len` 设小、发足长请求加速复现）。对照实验：关闭 MTP 应无精度问题；开/关 kv_pool 都应复现（证明与池化无关）。

---

> **核心教训**：凡是「越界」判定的边界比较，务必区分「等于上限」与「超过上限」——序列长度、索引、token 计数等 1-based 上限都属合法值，`>=` 与 `>` 一字之差会在长压测逼近边界时把合法请求截断、放大成 silent 精度退化；这类问题短请求永远测不出来，必须靠长序列 / 边界值单测钉死。