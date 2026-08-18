# 案例17：DP8 场景量化推理精度问题——MoE 未开 EP 时 SP 数值路径异常

> **一句话定位**：A2（910B2C）单机 8 卡 DP8 + TP1 + EP 下跑量化（`--quantization ascend`）Qwen3-30B，oorability 错误/精度退化；本质是该并行组合落入**序列并行（SP）未满足约束**的非法路径——修复方式是**约束护栏**：MoE 模型 SP 必须 `enable_expert_parallel=True`，否则拒绝启动。
>
> **对象**：vllm-project/vllm-ascend [#4273](https://github.com/vllm-project/vllm-ascend/issues/4273)（Bug，closed completed）+ 修复 PR [#4014](https://github.com/vllm-project/vllm-ascend/pull/4014)（[Bugfix]，merged）

---

## 1. 问题描述

### 1.1 现象

Qwen3-30B（msModelslim 量化，FP 权重）在 **DP8（单机 8 卡）+ TP1 + EP + `--quantization ascend`** 下：

- 输出错误 / 精度退化，界面不报错；量化与非量化模型同样复现；
- DP=2 时返回正常。

### 1.2 触发条件（必现矩阵）

| DP 卡数 | EP | MoE 模型 | 是否触发 |
|:---:|:---:|:---:|:---:|
| 2 | — | ✓ | ✗（正常） |
| 「8」 | ✗（未满足约束） | ✓ | ✓ 非法组合 → 精度退化 |
| 8 | ✓（满足约束） | ✓ | ✗ |

> A2（910B2C×8）复现；A3 上不复现。

### 1.3 影响与严重度

- **严重度**：🟡 中（特定 DP×TP×EP 组合下的静默精度退化；非用户显式非法配置则不受影响）。
- **隐蔽性**：🟡 中（需特定并行组合 + 特定 NPU 平台）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | Issue [#4273](https://github.com/vllm-project/vllm-ascend/issues/4273)（Bug，closed completed） |
| 修复 PR | [#4014](https://github.com/vllm-project/vllm-ascend/pull/4014)（Bugfix，merged，commit `22005c6`） |
| vllm-ascend | v0.11.0rc2.dev6+g51e5806d7（基于 vllm 0.11.0） |
| 上游 vllm | v0.11.0 |
| 环境 | torch 2.7.1 / torch_npu 2.7.1 / CANN 8.3.RC1 / A2 910B2C ×8 |

---

## 3. 定位过程

1. **对拍 DP**：DP8 复现、DP2 正常 → 锁定与 DP/SP 并行路径相关；
2. **collaborator 判定**：该场景本质是 **SP**（MoE 未开 EP 时 SP 路径数值异常），A2 复现、A3 正常；
3. **定位根因**：MoE 模型在 TP1+DP 下启动了 sequence parallelism，但未满足 EP 约束，数值路径异常。

> 定位要点：NPU 上 MoE+量化+并行（SP/EP）的参数合法性会直接改变数值路径；精度排查必须先验证并行配置是否落进受支持组合，而不是只在算子层找 diff。

---

## 4. 解决方案

### 4.1 根因

非法并行组合（MoE 模型走 SP 而未开 EP）触发数值异常路径。

### 4.2 修复（PR #4014，约束型：12 增 3 删）

1. 将 `tp_size > 1` 检查移入 `enable_sp`；
2. **强制「MoE 模型 SP 必须 `enable_expert_parallel=True`，否则拒绝启动」**；
3. Flash Comm v1 不可用时显式报错。

> 这是一次**约束修复**（禁用会触发精度退化的不合法并行组合），而非数值层修复。

### 4.3 验证

非法组合在启动时被显式拒绝/约束，不再静默跑出错误结果。

---

## 5. 复现方法

```bash
ASCEND_RT_VISIBLE_DEVICES=0..7 \
HCCL_OP_EXPANSION_MODE=AIV VLLM_ASCEND_ENABLE_FLASHCOMM=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /Qwen3-30B_quant --data-parallel-size 8 --tensor-parallel-size 1 \
  --quantization ascend --enable-expert-parallel
```

DP2 对照正常。

> vllm/vllm-ascend 复现版本：vllm-ascend **v0.11.0rc2.dev6**，上游 vllm **v0.11.0**。

---

## 核心教训

NPU 并行组合同样是「精度正确性」的一等约束：MoE 模型开 SP 必须配 EP，否则静默退化；把非法配置在启动期拒绝，优于让用户在推理期撞见错误输出。