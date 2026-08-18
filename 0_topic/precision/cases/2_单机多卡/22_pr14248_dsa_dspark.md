# 案例 22：DSA 注意力 dsa_v1 在 dspark 下 decode 被当作 prefill 处理致精度错误

> **一句话定位**：DSA（DeepSeek Sparse Attention）v1 实现在 dspark 场景下 **decode 被当作 prefill 处理**，稀疏注意力的 mask/语义分支差异直接转化为数值错误。
>
> **对象**：vllm-project/vllm-ascend [PR #14248](https://github.com/vllm-project/vllm-ascend/pull/14248)（[Refactor][BugFix][DSA][4/N] fix the precision issue of dsa_v1 under dspark，merged 2026-08-14）

---

## 1. 问题描述

### 1.1 现象

- DSA v1 注意力实现在 **dspark** 调度场景下精度错误；
- 根因（PR 原文）：当前 dspark 场景中 decode 被当作 prefill 处理——两类请求的稀疏注意力 mask/索引语义不同，混淆后数值错误。

### 1.2 触发条件（必现矩阵）

| DSA 模型（V3.2 类） | dspark 调度 | 是否触发 |
|:---:|:---:|:---:|
| ✗ | — | ✗ |
| ✓ | ✗ | ✗ |
| **✓** | **✓** | **✓ dsa_v1 精度错误** |

### 1.3 影响与严重度

- **严重度**：🔴 高（DSA 模型 dspark 部署输出错误）。
- **隐蔽性**：中（需 DSA 硬件路径 + dspark 组合）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#14248](https://github.com/vllm-project/vllm-ascend/pull/14248)（Refactor/BugFix 系列 4/N） |
| 状态 | merged（2026-08-14 合入 main） |
| vllm-ascend 版本 | main（2026-08-14） |
| 上游 vLLM | v0.27.1，main commit `58d3918e3ea0a544ffedadad2ba84559e9c51d8f` |

---

## 3. 定位过程

未知（正文仅一句结论）。

---

## 4. 解决方案

### 4.1 根因

调度层把请求统一化（decode 当 prefill）时，稀疏注意力 kernel 的 mask/语义分支差异直接转化为数值错误——dsa_v1 对 prefill/decode 有不同的索引构建路径。

### 4.2 修复内容

dsa_v1 重构并**回滚**「decode 视作 prefill」的处理部分，恢复两类请求各自的正确语义分支。

### 4.3 验证

正文未填（随 DSA 精度回归覆盖）。

---

## 5. 复现方法

### 5.1 最小复现模型

- **DeepSeek-V3.2**（DSA 结构，多卡）；无更小公开 DSA 替代。

### 5.2 复现命令

```bash
vllm serve <DeepSeek-V3.2> --tensor-parallel-size 8 \
  # 以仓库当前 DSA attention backend + dspark 开启方式为准
# prefill+decode 混合流量下对比精度（与关闭 dspark 对照）
```

---

## 核心教训

调度层把请求「统一化」时，注意力 kernel 的 prefill/decode 语义分支差异会直接变成数值错误——稀疏/自定义注意力 backend 必须显式区分两类请求的元数据与 mask 构建。
