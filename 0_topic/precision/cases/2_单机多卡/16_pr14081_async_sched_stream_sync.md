# 案例 16：异步调度 + piecewise 图模式下 forward 前缺 stream 同步，静默读脏数据

> **一句话定位**：`--async-scheduling` + piecewise compilation 组合下，model forward 之前缺少 stream synchronization，前向消费尚未写完的数据 → 无崩溃的静默数值错误（qwen3-235b 精度不达标）。
>
> **对象**：vllm-project/vllm-ascend [PR #14081](https://github.com/vllm-project/vllm-ascend/pull/14081)（[BugFix] Fix precision of qwen3-235b in piecewise mode，merged 2026-08-12）

---

## 1. 问题描述

### 1.1 现象

- qwen3-235b 在 **piecewise 图模式 + 异步调度（asynchronous scheduling）** 场景下精度不达标（math 数据集）；
- 根因：异步调度下 model forward 前缺少 stream synchronization，前向读到**尚未写完的数据**；
- 完全静默——无 crash、无 assert，只表现为精度劣化。

### 1.2 触发条件（必现矩阵）

| async-scheduling | piecewise 图模式 | 是否触发 |
|:---:|:---:|:---:|
| ✗ | — | ✗ |
| ✓ | ✗（eager/全图） | ✗ |
| **✓** | **✓** | **✓ 静默数值错误** |

### 1.3 影响与严重度

- **严重度**：🔴 高（必现但难归因——只看精度数字完全想不到是同步缺失）。
- **隐蔽性**：极高（典型「必现但难归因」）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#14081](https://github.com/vllm-project/vllm-ascend/pull/14081)（BugFix） |
| 状态 | merged（2026-08-12 合入 main） |
| vllm-ascend 版本 | main（2026-08-12） |
| 上游 vLLM | v0.23.0，main commit `ee0da84ab9e04ac7610e28580af62c365e898389` |

---

## 3. 定位过程

未知（PR 未描述定位步骤；由 math 数据集精度测试发现）。

---

## 4. 解决方案

### 4.1 根因

NPU 多流异步调度下，跨 stream 的数据依赖若不显式建立 happens-before 边，不会报错——只会表现为精度劣化。

### 4.2 修复内容

在 model forward 之前添加 stream synchronization（PR 原文：Add stream synchronization before model forward）。

### 4.3 验证

math 数据集精度测试达标（PR 原文）。

---

## 5. 复现方法

### 5.1 最小复现模型

- **任一支持 piecewise 模式的 MoE 模型（如 Qwen3-30B-A3B 级）**；qwen3-235b 为原始报告配置。

### 5.2 复现命令

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --async-scheduling \
  --compilation-config '{"level":0,"splitting_ops":["..."],"piecewise":true}'  # 以仓库 piecewise 开启方式为准
# 跑 math/gsm8k 对比非异步调度基线；修复前精度不达标
```

---

## 核心教训

多流异步调度下跨 stream 数据依赖不显式同步不会报错，只会精度劣化——「async-scheduling + 图模式」组合是必查项，怀疑此类问题时先开/关 async-scheduling 做二分。
