# 案例31：Ascend PD/PCP/DCP 场景 PA/FIA 混合注意力图回放精度错误

> **一句话定位**：Ascend 注意力后端恢复 PA（Paged Attention）作为 FIA（Flash Attention）的 fallback 后，`update_graph_params` 用 `if using_paged_attention(num_tokens):` 假设「同一 num_tokens 的所有捕获 attention 都是 PA」；但同一 num_tokens 下不同层可能混用 PA/FIA（受 sliding_window/head_size 影响），图回放 unpack 时遇 FIA 参数抛 `ValueError`，导致 PD/PCP/DCP 场景精度错误。
>
> **对象**：vllm-project/vllm-ascend [#13195](https://github.com/vllm-project/vllm-ascend/pull/13195)（Cherry-pick，merged，落 `releases/v0.24.0rc`；代码来自 #12027/#12228/#12255）

---

## 1. 问题描述

### 1.1 现象

Ascend 上 PD/PCP/DCP（PD 分离 / prefix cache / decode 分离）场景精度错误。根因是注意力后端选择与图模式元数据分派：恢复 PA fallback + 引入 `pa_shape_list` 后，`attention_v1.py::update_graph_params` 的 `if using_paged_attention(num_tokens):` 分支逻辑错误。

### 1.2 触发条件（必现矩阵）

| PD/PCP/DCP | 混合注意力（PA+FIA 并存） | 图模式（CUDA/NPU graph） | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗ | — | — | ✗ |
| ✓ | ✗（纯 PA 或纯 FIA） | ✓ | ✗（分派一致） |
| **✓** | **✓（同 num_tokens 层间混用）** | **✓** | **✓ 回放错误 / 精度异常** |

### 1.3 影响与严重度

- **严重度**：🔴 高（检证模型 Qwen3.5-397B-A17B-w8a8-mtp-longseq 在 PD 场景精度错误）。
- **隐蔽性**：🟡 中（需 PA/FIA 混合 + 图模式）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#13195](https://github.com/vllm-project/vllm-ascend/pull/13195)（Cherry-pick → `releases/v0.24.0rc`，merged） |
| 组件 PRs | [#12027](https://github.com/vllm-project/vllm-ascend/pull/12027)、[#12228](https://github.com/vllm-project/vllm-ascend/pull/12228)、[#12255](https://github.com/vllm-project/vllm-ascend/pull/12255) |
| vllm-ascend | v0.24.0（release 分支） |
| 上游 vllm | main `85c09e9` |
| 检证模型 | Qwen3.5-397B-A17B-w8a8-mtp-longseq（nightly） |

---

## 3. 定位过程

1. **恢复 PA fallback**：PD/PCP/DCP 下 FIA 行为异常，恢复 PA 作为 FIA fallback 并配 `pa_shape_list`；
2. **review 抓 bug**：评审 bot 在 `update_graph_params` 定位真实 bug——`if using_paged_attention(num_tokens):` 假设某 token 数的所有捕获 attention 都是 PA；
3. **确认混合**：同一 `num_tokens` 下不同层可能混用 PA/FIA（受 sliding_window/head_size 影响），图回放 unpack 时遇 FIA 参数抛 `ValueError`。

> 定位要点：图模式按 num_tokens 缓存 attention 元数据时，不能假设「同 token 数 = 同 attention 算子类型」；混合 PA/FIA 必须按参数实例类型分派。

---

## 4. 解决方案

### 4.1 根因

按 `num_tokens` 判定是否用 PA，而非按实际参数类型，切图时把 PA 的 graph param 断言用到 FIA 参数上。

### 4.2 修复（PR #13195）

```python
# attention_v1.py
-        if using_paged_attention(num_tokens):
-            ... # 错误假设
-        elif _EXTRA_CTX.sinks:
-            ...
# 改为：删除该基于 num_tokens 的分支，落入既有基于
# isinstance(param, PagedAttentionGraphParam) 的分派，使 PA+FIA 混合图正确回放
```

另含 stateful decode 选择逻辑结合 MTP/投机解码契约。

### 4.3 验证

单测覆盖 `tests/ut/attention/a2/test_attention_v1.py`、`tests/ut/_310p/attention/test_attention_v1_310.py`；检证模型 end-to-end 恢复。

---

## 5. 复现方法

- 正文未给最小复现脚本；触发点在 `vllm_ascend/attention/attention_v1.py`；
- 用 Qwen3.5-397B-A17B 长序列（MTP/PD 场景）greedy 对拍修复前后输出。

> vllm/vllm-ascend 复现版本：vllm-ascend **v0.24.0**，上游 vllm **main `85c09e9`**。

---

## 核心教训

NPU 注意力图模式按 num_tokens 缓存元数据时，不能假设同 token 数 = 同 attention 算子类型；混合 PA/FIA 需按参数实例类型（`isinstance(PagedAttentionGraphParam)`）分派，否则图回放静默精度错。