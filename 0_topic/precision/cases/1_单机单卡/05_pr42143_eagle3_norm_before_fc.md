# 案例 05：EAGLE3 嵌套 eagle_config 中 norm_before_fc 被静默跳过，草稿接受率劣化

> **一句话定位**：NVIDIA 发布的 EAGLE3 草稿 checkpoint 把 `norm_before_fc` 存在嵌套 `eagle_config` 字典里，vllm 只读顶层属性 → FC 前 RMSNorm 被静默跳过，草稿头输入分布错误、draft 接受率下降。
>
> **对象**：vllm-project/vllm [PR #42143](https://github.com/vllm-project/vllm/pull/42143)（merged，改 `llama_eagle3.py`）

---

## 1. 问题描述

### 1.1 现象

- 使用 NVIDIA 格式 EAGLE3 草稿 checkpoint（GPT-OSS 系列）时，FC 前 RMSNorm 被静默跳过；
- 草稿头（drafter）输入分布错误 → 投机解码**接受率下降**（精度类劣化：加速比缩水，输出质量也可能受影响）；
- 无报错、无警告——字段读不到就静默用默认值。

### 1.2 触发条件（必现矩阵）

| EAGLE3 草稿 checkpoint | norm_before_fc 存储位置 | 是否触发 |
|:---:|:---:|:---:|
| 自制/官方顶层格式 | 顶层属性 | ✗ |
| **NVIDIA 格式** | **嵌套 eagle_config** | **✓ Norm 被跳过、接受率下降** |

### 1.3 影响与严重度

- **严重度**：🟡 中（接受率/加速比劣化，非输出崩坏）。
- **隐蔽性**：高（配置解析静默回退默认值，无任何告警）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm |
| 对象 | PR [#42143](https://github.com/vllm-project/vllm/pull/42143) |
| 状态 | merged（2026-05-09） |
| 复现版本 | 2026-05-09 修复前的 vllm main |
| 修复版本 | 含 #42143 的 main |
| vllm-ascend 版本 | 未知（checkpoint 配置解析为跨硬件逻辑，NPU 上同现） |

---

## 3. 定位过程

1. NVIDIA 格式 EAGLE3 checkpoint 接受率明显低于预期；
2. 排查 checkpoint 配置字段层级差异，发现 `norm_before_fc` 位于嵌套 `eagle_config`；
3. 对照同文件中 `use_aux_hidden_state` 已有的嵌套读取模式确认遗漏。

---

## 4. 解决方案

### 4.1 根因

同一能力字段在不同发布方 checkpoint 里层级不同（顶层 vs 嵌套 `eagle_config`），解析只覆盖顶层；读不到时静默回退默认值，等于「静默跳过 Norm」。

### 4.2 修复内容

```python
# 先读嵌套 eagle_config，再回退顶层属性（与 use_aux_hidden_state 模式统一）
norm_before_fc = getattr(eagle_config, "norm_before_fc",
                          getattr(config, "norm_before_fc", False))
```

### 4.3 验证

greedy 下 baseline 与 eagle3 输出 4/4 完全一致；吞吐 51.3 → 67.1 tok/s（1.31x）；配置解析单测通过。

---

## 5. 复现方法

### 5.1 最小复现模型

- 原始测试：Qwen3-32B + eagle3 草稿（1×H200）；
- **小模型路径**：任一 <10B 目标模型 + 自制「把 `norm_before_fc` 放进嵌套 `eagle_config`」的 eagle3 草稿 checkpoint，即可触发同一解析分支（配置单测甚至无需 GPU）。

### 5.2 复现命令

```bash
# greedy 一致性 + 吞吐对比（修复前接受率低、吞吐 <1.1x）
vllm serve <TARGET> --speculative-config '{"method":"eagle3","model":"<EAGLE3-DRAFT>","num_speculative_tokens":3}'
```

---

## 核心教训

checkpoint 配置解析缺失 = 静默精度劣化——多发布方字段层级必须「嵌套优先、顶层回退」并加告警；spec decode 的精度体检指标是 greedy 一致性与接受率曲线。
