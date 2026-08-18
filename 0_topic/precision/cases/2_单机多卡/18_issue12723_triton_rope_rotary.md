# 案例18：Ascend Triton RoPE 非 2 的幂 rotary_dim 偏移错误，破坏 GLM-5.2 DSpark draft 的 Q/K 范数

> **一句话定位**：Ascend 平台分发的 `rope_forward_triton` 在 rotary_dim 非 2 的幂（GLM-5.2 DSpark draft：head_dim=rotary_dim=192）时，用 padding 后的维数计算 sin cache 偏移，Q/K 不保范数；但只在多 worker 真实 serving 并行下暴露（孤立单测 max error=0）。
>
> **对象**：vllm-project/vllm-ascend [#12723](https://github.com/vllm-project/vllm-ascend/issues/12723)（Bug，closed not planned）+ 修复 PR [#12963](https://github.com/vllm-project/vllm-ascend/pull/12963)（merged）

---

## 1. 问题描述

### 1.1 现象

在 Ascend A3（16 dies）上 serving GLM-5.2 + DSpark draft（draft bf16、NeoX、head_dim=rotary_dim=192）时：

- 在线 post/post-RoPE 范数比 Q≈0.737 / K≈0.732（离线参考 1.0001 / 0.9997）；
- 首个明显分叉出现在 layer-0 query RoPE 之后（rel-L2 从 ~0.003 跳变到 0.68）；
- **界面不报错但精度退化**：投机 accept_len 全局塌缩到 3.056（8 请求）/ 2.910（32 请求 c8），而离线 teacher-forced 约 5.6；
- 用户 token 级接受率：位置 1 0.718→修复后 0.901；位置 7 0.024→0.394。

### 1.2 触发条件（必现矩阵）

| rotary_dim 非 2 的幂 | 真实 serving 并行（grid/并行） | dispatched Triton rope | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗（128 等 2 幂） | — | — | ✗ |
| ✓ | ✗（单 die 孤立测试） | ✓ | ✗（max error=0） |
| **✓（192）** | **✓（多 worker）** | **✓** | **✓ 精度退化** |

> 关键：非 2 的幂 rotary_dim 是根因；但**必须**在真实 serving 的 grid/并行配置下才复现，孤立 kernel 单测无法发现。

### 1.3 影响与严重度

- **严重度**：🔴 高（投机解码收益大幅缩水、生成质量下降；不崩、不报错，属 silent 精度退化）。
- **隐蔽性**：🔴 高（须「非 2 幂 rotary_dim × 多 worker serving」组合，单测难覆盖）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | Issue [#12723](https://github.com/vllm-project/vllm-ascend/issues/12723)（Bug，closed not planned） |
| 修复 PR | [#12963](https://github.com/vllm-project/vllm-ascend/pull/12963)（Ops [BugFix]，merged 到 main） |
| vllm-ascend | 镜像 v0.23.0（dev994+gfabc77a96），DSpark source fabc77a9 |
| 上游 vllm | v0.23.0 |
| 环境细节 | torch 2.10.0 / torch_npu 2.10.0.post2 / triton 3.2.x / CANN 9.0.1/9.1.0 |
| 模型 | GLM-5.2（DSpark draft，bf16×NeoX×head_dim=192） |
| 定位工具 | dispatched vs `forward_native()` 同输入 A/B 数值对比 |

> 修复以「用真实 rotary_dim 而非 padding 后维数计算偏移」落地于 `vllm_ascend/ops/triton/rope.py`，并新增非 2 幂组合测试。

---

## 3. 定位过程

1. **排除 stride/布局**：dispatched rope 与 `.contiguous()` clone bitwise 相同 → 排除张量布局；
2. **A/B 对照**：dispatched vs `forward_native()` 的 rel-L2 Q=0.678912 / K=0.694659、范数比 0.736589 / 0.732088；native 保范数 → 锁定 Triton rope；
3. **排除 cos/sin cache**：cos²+sin²=1（max err 1.19e-7）→ cache 本身正确；
4. **确认必现**：固定 greedy 请求在固定 draft 位置，选中 native 路径后 proposal 翻转为 target top-1，两次独立新进程均复现；
5. **定位根因**：`_triton_rope` / `_triton_rope_siso` 用 `pad_rope_dim`（padding 后）计算 sin 偏移，非 2 幂 rotary_dim 下偏移错位。

> 定位要点：Ascend 自定义 Triton op 对「非 2 幂 head/rotary_dim」的 cache 索引极易算错，且只在多 worker 真实 serving 下暴露——孤立 kernel 单测无法发现，必须做同输入 dispatched vs native 的 A/B 数值对比。

---

## 4. 解决方案

### 4.1 根因

`sin_offsets = tl.arange(pad_rope_dim // 2, pad_rope_dim)` 用 **padding 后**的维数（192→pad 到 256 之类）计算偏移；当 rotary_dim 是 2 的幂时二者等价，非 2 幂时错位，RoPE 旋转半个周期，Q/K 范数被破坏。

### 4.2 修复（PR #12963）

```diff
# vllm_ascend/ops/triton/rope.py（_triton_rope 与 _triton_rope_siso 两处）
- sin_offsets = tl.arange(pad_rope_dim // 2, pad_rope_dim)
+ sin_offsets = cos_offsets + (rope_dim // 2)   # 用真实 rotary_dim 而非 pad 后维数计算偏移
```

新增非 2 的幂组合测试 `(128, 96)`。另有模型侧 workaround PR #12726：仅将 draft query RoPE 切到 `forward_native()`，accept_len 3.056→5.548（8 serial）/ 5.430（32 请求），40/40 HTTP 200。

---

## 5. 复现方法

```bash
vllm serve <GLM-5.2> \
  --tensor-parallel-size 8 --data-parallel-size 2 \
  --enable-expert-parallel --enforce-eager \
  --speculative-config '{"method":"dflash","model":"<DSPARK_DRAFT>","num_speculative_tokens":7,...}'
```

- draft 需 bf16、NeoX、head_dim=192（非 2 幂）；
- GSM8K greedy 测 accept_len；用 `--enable-expert-parallel` + 多 worker 才能触发；
- 对照法：A/B 换 `forward_native()` 观察 accept_len / 范数比恢复。

---

## 核心教训

Ascend Triton 自定义 op 对「非 2 幂」维度的 cache 偏移索引是高频踩坑点，且在真实 serving 的多 worker 并行下才暴露；凡涉及 RoPE/cache 索引的自定义 kernel，须做「孤立单测（可能过）+ 真实 serving 端到端数值对照」双保险。