# 案例 01：Dynamic NTK RoPE 缩放公式常数化，超训练长度输入位置编码全错

> **一句话定位**：`dynamic_ntk_scaling_rope.py` 把「当前序列长度」代回公式化简后，base 缩放退化为与序列长度无关的常数 `α²-α+1`——凡「训练长度短、服务长度长」的 dynamic-NTK 模型，超过训练长度后 RoPE 全错，长文本嵌入质量崩坏。
>
> **对象**：vllm-project/vllm [Issue #41236](https://github.com/vllm-project/vllm/issues/41236) + 修复 [PR #41277](https://github.com/vllm-project/vllm/pull/41277)（merged）

---

## 1. 问题描述

### 1.1 现象

- 使用 dynamic NTK RoPE 缩放的模型（如 `nomic-embed-text-v1.5`，训练 2048、服务 8192）在**输入长度超过训练长度**后，位置编码计算全错；
- 嵌入类模型表现为长文本 embedding 与参考实现（sentence-transformers）显著偏离；生成类模型表现为长上下文输出质量崩坏；
- 短于训练长度的输入完全正常——CI 短用例天然掩盖。

### 1.2 触发条件（必现矩阵）

| rope_scaling=dynamic NTK | 输入长度 > 训练长度 | 是否触发 |
|:---:|:---:|:---:|
| ✗ | — | ✗ |
| ✓ | ✗ | ✗（公式分支不生效） |
| **✓** | **✓** | **✓ 位置编码错误** |

### 1.3 影响与严重度

- **严重度**：🔴 高（长文本输出/嵌入彻底错误，静默无报错）。
- **隐蔽性**：高——短输入一切正常，只有长输入才暴露。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm |
| 对象 | Issue [#41236](https://github.com/vllm-project/vllm/issues/41236) + PR [#41277](https://github.com/vllm-project/vllm/pull/41277) |
| 状态 | merged（2026-04-30，+45/-152，改 `dynamic_ntk_scaling_rope.py` 等 4 文件） |
| 复现版本 | 修复合入前的 vllm main（2026-04-30 之前，issue 2026-04 报告） |
| 修复版本 | 含 #41277 的 vllm main（2026-04-30 后） |
| vllm-ascend 版本 | 未知（纯 Python 层 rope 逻辑，任何 vllm-ascend 版本可在 NPU 上对拍验证） |

---

## 3. 定位过程

1. 长文本（8041 token）下 `LLM.embed()` 输出与 sentence-transformers 参考结果显著不一致；
2. 对照 NTK-Aware 原始公式 `(α * seq_len / orig_len) - (α-1)`，对实现做**代数化简**；
3. 发现实现里 `max_len = max_position_embeddings * scaling_factor` 再代回，化简后 base 缩放变成常数 `α²-α+1`，与当前序列长度完全无关 → 锁定根因。

---

## 4. 解决方案

### 4.1 根因

实现将 `s = seq_len / max_len` 中的 `max_len` 用 `max_position_embeddings * scaling_factor` 代入后未正确保留「当前序列长度」项，缩放因子化简为常数——dynamic NTK 的「随长度动态调整」语义完全丢失。

### 4.2 修复内容

- 按「服务长度 / 训练长度」重算 `s`，恢复 `(α*s) - (α-1)` 的动态形式；
- 用 `max_trained_positions` / `n_positions` 正确判定训练长度边界。

### 4.3 验证

vLLM `llm.embed()` 与 sentence-transformers 在 8041 token 文本上的嵌入逐位近似一致。

---

## 5. 复现方法

### 5.1 最小复现模型

- **`nomic-ai/nomic-embed-text-v1.5`（~137M）**：dynamic NTK rope + 训练 2048 / 服务 8192，是触发本 bug 的最小公开模型；单机单卡（甚至 CPU）即可。

### 5.2 复现命令

```python
from vllm import LLM
llm = LLM(model="nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
out = llm.embed(["<长文本，>2048 token>"])
# 与 sentence-transformers 同文本结果对比余弦相似度；修复前显著偏离
```

> NPU 上复现：vllm-ascend 任意版本加载同一模型，同法对拍即可（rope 计算在 Python 层，跨硬件一致）。

---

## 核心教训

位置编码公式必须做「代数化简自检」——**化简出常数结果**是最容易被忽视的静默精度毒丸；dynamic/随输入变化的分支要用「超过训练长度」的用例覆盖，短输入 CI 测不出来。
