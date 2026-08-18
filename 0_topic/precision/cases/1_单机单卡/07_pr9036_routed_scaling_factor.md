# 案例 07：Ascend 量化路径漏传 MoE routed_scaling_factor，路由权重静默丢 scale

> **一句话定位**：带 `routed_scaling_factor` 的 MoE 模型（DeepSeek-V2/V3 系）走 vllm-ascend 自有量化路径（W8A8/W4A16/W4A8/MXFP8/MXFP4）时，`select_experts` 漏传 scaling factor、topk_weights 未乘 scale；叠加上游把 `routed_scaling_factor` 在 `FusedMoE.__init__` 里置 1.0，原始值彻底丢失。
>
> **对象**：vllm-project/vllm-ascend [PR #9036](https://github.com/vllm-project/vllm-ascend/pull/9036)（[BugFix] Fix quantization accuracy bug，merged 2026-05-10）

---

## 1. 问题描述

### 1.1 现象

- 使用 vllm-ascend 自有量化方法（`--quantization ascend` 的 w8a8_dynamic / w4a16 / w4a8 / w8a8_mxfp8 / w4a4_mxfp4 及 _310p 路径）推理带 `routed_scaling_factor != 1.0` 的 MoE 模型时，**输出精度错误**；
- 根因是两层叠加：① 各量化方法 `apply()` 调 `select_experts` 时漏传 `routed_scaling_factor`，topk_weights 未乘 scale；② 上游 vLLM `FusedMoE.__init__` 在 `apply_routed_scale_to_output=True` 时会把 `self.routed_scaling_factor` 置 1.0（期望 runner 在输出端 scale），而 vllm-ascend 走自己的 forward path，需要**原始值**。

### 1.2 触发条件（必现矩阵）

| 量化=ascend | MoE `routed_scaling_factor≠1` | 是否触发 |
|:---:|:---:|:---:|
| ✗（bf16/上游量化） | — | ✗（上游路径自带 scale 处理） |
| ✓ | ✗（=1.0） | ✗（无差异） |
| **✓** | **✓** | **✓ 路由权重丢 scale、精度错误** |

### 1.3 影响与严重度

- **严重度**：🔴 高（所有 ascend 量化方法文件的 `apply()` 均受波及）。
- **隐蔽性**：高（无报错，纯输出质量劣化，需与 HF 参考对拍才能发现）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#9036](https://github.com/vllm-project/vllm-ascend/pull/9036)（BugFix） |
| 状态 | merged（2026-05-10 合入 main） |
| vllm-ascend 版本 | main（2026-05-10 合入） |
| 上游 vLLM | v0.20.1，main commit `c7aa186d67b6f051680831418e957c67f34ba7a2` |

---

## 3. 定位过程

未知（PR 正文未描述定位步骤；由量化模型精度回归对拍发现）。

---

## 4. 解决方案

### 4.1 根因

上游把 routed scaling 职责从「权重端」移到「runner 输出端」后，走自定义 forward path 的硬件插件必须自行保存原始参数——vllm-ascend 两处都没接住：量化 `apply()` 漏传 + super().__init__ 后原始值被覆盖丢失。

### 4.2 修复内容

```python
# fused_moe.py: __init__ 里先存原始值再调 super()
self._original_routed_scaling_factor = kwargs.get("routed_scaling_factor", 1.0)
super().__init__(*args, **kwargs)
# forward_impl 两处: routed_scaling_factor=self._original_routed_scaling_factor

# experts_selector.py（ops 与 _310p 两处）:
if routed_scaling_factor != 1.0:
    topk_weights = topk_weights * routed_scaling_factor
```

### 4.3 验证

PR 正文未填具体测试项；改动横跨 6 个量化文件传参 + selector 乘法，随量化精度回归覆盖。

---

## 5. 复现方法

### 5.1 最小复现模型

- **DeepSeek-V2-Lite（W8A8，ModelSlim 导出）**：`routed_scaling_factor=2.5` 的最小 MoE 模型，单卡 910B 可跑。

### 5.2 复现命令

```bash
vllm serve <deepseek-v2-lite-w8a8> --quantization ascend
# 与 HF greedy 输出 / gsm8k 分数对拍；回退到合并前 commit 可见明显劣化
```

---

## 核心教训

上游把 scaling 职责从权重移到 runner 输出端时，自定义 forward path 的硬件插件必须自行保存原始参数——否则量化模型的路由权重静默丢失 scale。
