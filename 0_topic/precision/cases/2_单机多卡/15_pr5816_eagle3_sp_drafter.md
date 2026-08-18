# 案例 15：EAGLE3 + SP（FlashComm 序列并行）下 drafter 走错通信路径，接受率崩塌

> **一句话定位**：EAGLE3 + SP（`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`）时，「是否启用 SP」的判断套用了 target 的 MoE/TP 逻辑，而 drafter 多为 TP=1 的 dense 模型 → drafter 数据切分走错路径，第二个及之后的 draft token 几乎全被拒收。
>
> **对象**：vllm-project/vllm-ascend [Issue #5825](https://github.com/vllm-project/vllm-ascend/issues/5825) + 修复 [PR #5816](https://github.com/vllm-project/vllm-ascend/pull/5816)（merged 2026-01-14，另有 0.13.0 cherry-pick commit 7716fc8）

---

## 1. 问题描述

### 1.1 现象

- EAGLE3 投机解码 + SP 开启时 drafter 模型精度错误：
  - mean acceptance length **1.67**（关 SP 时 2.29）；
  - token 1 位置 acceptance **0.08**（正常 0.40）、token 2 位置 **0.00**——第二个及之后的 draft token 几乎全被拒收；
- 关闭 SP 则完全正常。

### 1.2 触发条件（必现矩阵）

| EAGLE3 | SP（FLASHCOMM1=1） | TP/EP | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗ | — | — | ✗ |
| ✓ | ✗ | — | ✗ |
| **✓** | **✓** | **TP=2（+EP）** | **✓ drafter 精度错误、接受率崩塌** |

### 1.3 影响与严重度

- **严重度**：🔴 高（投机解码加速收益近乎归零，长输出质量连带受损）。
- **隐蔽性**：中（acceptance metrics 可直接观测，但需跑够采样量）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | Issue [#5825](https://github.com/vllm-project/vllm-ascend/issues/5825)（Bug）+ PR [#5816](https://github.com/vllm-project/vllm-ascend/pull/5816)（Bugfix） |
| 状态 | issue closed + PR merged（2026-01-14，cherry-pick 到 0.13.0：commit 7716fc8） |
| vllm-ascend 版本 | 复现：0.13.0rc2.dev154+gff4c1a47b（issue 环境）；修复：main 2026-01-14 / v0.13.0 |
| 上游 vLLM | v0.13.0（main @2f4e654） |

---

## 3. 定位过程

1. 对比 SP on/off 的**逐位置 acceptance 曲线**，锁定问题在 drafter（draft 模型）而非 target；
2. 构造性复现：删除 `num_tokens > 1000` 条件使 sp 恒开，问题稳定复现；
3. 定位到 drafter 的 SP 启用判断复用了 target 的 MoE/TP 逻辑。

---

## 4. 解决方案

### 4.1 根因

「是否启用 SP」的判断必须区分 target 与 drafter——drafter 多为 TP=1 的 dense 模型，套用 target 的 MoE/TP 判断会让 drafter 走错通信/切分路径。

### 4.2 修复内容（PR #5816，7 文件）

- `vllm_ascend/spec_decode/eagle_proposer.py` 引入 `split_inputs_tp_to_sp` 处理 SP 数据切分，替换原 `torch.ops.vllm.maybe_pad_and_reduce`；
- 更精确判断 drafter 是否应启用 sp（区分 target 与 drafter 的 MoE 判断逻辑）；
- drafter 的 `eager` 改为前端真正 eager，规避 fx-graph 问题。

### 4.3 验证

按 #5825 脚本回归，结果与 SP 关闭时完全一致（acceptance length 2.29；token0/1/2 = 0.62/0.40/0.27）。

---

## 5. 复现方法

### 5.1 最小复现模型

- **Qwen3-30B-A3B（target）+ Qwen3-30B-A3B-EAGLE3（draft）**：2 卡即可。

### 5.2 复现命令

```bash
VLLM_ASCEND_ENABLE_FLASHCOMM1=1 vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 2 --enable-expert-parallel --enforce-eager \
  --speculative-config '{"method":"eagle3","model":"Qwen/Qwen3-30B-A3B-EAGLE3","num_speculative_tokens":3}' \
  --max-model-len 32768
# temperature=0.6 采样 1000 token，读 vllm:spec_decode_* metrics 的 acceptance
# 修复前：mean 1.67、token2 位置 0.00；修复后：mean 2.29
```

---

## 核心教训

target 与 drafter 的并行策略判断必须分开实现——「套用主模型判断」在投机解码里会把 draft 模型送进错误的通信路径；acceptance 逐位置曲线是 drafter 精度的体检表。
