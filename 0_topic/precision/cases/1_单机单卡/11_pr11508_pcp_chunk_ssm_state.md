# 案例 11：PCP + chunked prefill 下 SSM 状态递推公式错误，第二 chunk 起精度发散

> **一句话定位**：qwen3.5（hybrid Mamba）在 PCP + chunked prefill 叠加时，correct 状态递推把 `Φ_i·s0` 重复计算——首 chunk（s0=0）碰巧正确，后续 chunk（携带上一 chunk 状态 s0≠0）SSM state 回写错误，输出逐渐发散。
>
> **对象**：vllm-project/vllm-ascend [PR #11508](https://github.com/vllm-project/vllm-ascend/pull/11508)（[BugFix] fix qwen3.5+pcp+chunkprefill accuracy error，merged 2026-07-09）

---

## 1. 问题描述

### 1.1 现象

- qwen3.5 hybrid（Mamba/SSM 层）+ PCP + chunked prefill 同时启用时输出精度发散；
- **首 chunk 正常、后续 chunk 逐渐发散**——短 prompt（单 chunk）完全检测不出。

### 1.2 触发条件（必现矩阵）

| hybrid/SSM 模型 | chunked prefill（多 chunk） | PCP | chunk 序号 | 是否触发 |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | — | — | — | ✗ |
| ✓ | ✗（单 chunk） | — | — | ✗（s0=0 碰巧正确） |
| **✓** | **✓** | **✓** | **第 2+ chunk** | **✓ 状态递推错误、发散** |

### 1.3 影响与严重度

- **严重度**：🔴 高（长 prompt 输出发散）。
- **隐蔽性**：高——单 chunk 短用例天然掩盖，必须把 `--max-num-batched-tokens` 调小强制多 chunk。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#11508](https://github.com/vllm-project/vllm-ascend/pull/11508)（BugFix） |
| 状态 | merged（2026-07-09 合入 main） |
| vllm-ascend 版本 | main（2026-07-09） |
| 上游 vLLM | v0.23.0，main commit `1f486d96a17303ce8db8e02be39545b2be338446` |

---

## 3. 定位过程

PR 正文含完整数学推导：

```
原代码: updated_state[i] = all_final_state[i] + matmul(all_final_h_update[i], updated_state[i-1])  # 缺 -s0
展开 all_final_state[i] = Φ_i·s0 + p_i:
原代码 = (Φ_i·s0 + p_i) + Φ_i·correct_{i-1}   # Φ_i·s0 被算两次
正确:   correct_i = p_i + Φ_i·correct_{i-1}
```

即 correct 状态递推中未减去 `s0` 的贡献，`Φ_i·s0` 项重复；s0=0 时（首 chunk / 新 prompt）误差为零，故短用例不暴露。

---

## 4. 解决方案

### 4.1 根因

状态递推型算子（SSM/Mamba）的 chunk 续算公式中，初值 `s0` 的贡献在 `all_final_state` 里已包含，原代码又在累加侧重复计算；只有「初值≠0 的续算」才暴露。

### 4.2 修复内容

修正 correct 状态递推公式：减去 `s0` 贡献，恢复 `correct_i = p_i + Φ_i·correct_{i-1}`（公式级修正）。

### 4.3 验证

PR 正文 "How was this patch tested" 未填具体项（随量化/精度回归覆盖）。

---

## 5. 复现方法

### 5.1 最小复现模型

- **Qwen3.5 系列最小档（hybrid）**，如 `Eco-Tech/Qwen3.5-27B-w8a8-mtp`（w8a8 单卡 A3 可跑）。

### 5.2 复现命令

```bash
vllm serve Eco-Tech/Qwen3.5-27B-w8a8-mtp \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}' \
  --enable-chunked-prefill --max-num-batched-tokens 512
# 发长 prompt（确保切成 ≥2 个 chunk），temperature=0 对比非 chunked（max-num-batched-tokens 调大）输出
# 修复前：首 chunk 正常、第二 chunk 起逐渐发散
```

---

## 核心教训

状态递推型算子（SSM/Mamba）的修正路径必须用「初值≠0 的续算」用例验证——单 chunk 测试天然掩盖边界错误；chunked prefill 的精度回归要强制多 chunk。
