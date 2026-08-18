# 案例09：KV CPU 卸载 mamba_cache_mode="all" 在精确 chunk 边界静默输出错误

> **一句话定位**：`OffloadingConnector` 的 `resolve_mamba_align_size()` 只对 `mamba_cache_mode == "align"` 开启 hit-window 对齐；"all" 模式在 token/block 位置同样存 recurrent state，在精确 chunk 边界处前缀命中返回 N-1 token 却恢复边界 N 处的 state（已含 token N），重算 token N 被算两次 → 输出静默变化。
>
> **对象**：vllm-project/vllm [#51094](https://github.com/vllm-project/vllm/issues/51094)（Bug，closed completed）+ 修复 PR [#51100](https://github.com/vllm-project/vllm/pull/51100)

---

## 1. 问题描述

### 1.1 现象

CPU KV offload 重新取回后，当 prompt 长度恰为 offload Mamba chunk 大小的整数倍且 `mamba_cache_mode="all"` 时：

- 完成 CPU 缓存填充后清 GPU prefix cache、重放同 prompt，输出**静默变化且无异常**；
- 实测：`cold: prompt=2816 cached=0 first=[18631,43085,17091,47695,1556]`；`refetch: prompt=2816 cached=2815 first=[18631,43085,17091,47695,1270]`——第 5 个 token 从 1556 变为 1270，引擎正常完成不报错。

### 1.2 触发条件（必现矩阵）

| mamba_cache_mode="all" | 精确 chunk 边界长度 | CPU offload | enable_prefix_caching | 是否触发 |
|:---:|:---:|:---:|:---:|:---:|
| ✗（align） | — | — | ✓ | ✗（align 已对齐） |
| ✓ | ✗（非边界） | ✓ | ✓ | ✗（命中点不漂移） |
| **✓** | **✓（如 2816=2×1408）** | **✓** | **✓** | **✓ 静默输出错误** |

> 本配置 chunk=1408，prompt=2816 恰好 2 个 chunk 触发。

### 1.3 影响与严重度

- **严重度**：🟡 中（输出改变但不报错；仅精确边界 + "all" 模式触发，影响范围有限但静默）。
- **隐蔽性**：🟡 中（需特定模型混合 Mamba + 精确 chunk 长度）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm |
| 对象 | Issue [#51094](https://github.com/vllm-project/vllm/issues/51094)（Bug，closed completed） |
| 修复 PR | [#51100](https://github.com/vllm-project/vllm/pull/51100)（BugFix，merged，关闭该 issue） |
| 复现版本 | vLLM `0.25.1` nightly（commit `752a3a504`）；正文称 v0.26.0 与 main（`43c4bdc`）仍存在 |
| 模型 | `nvidia/NVIDIA-Nemotron-Nano-9B-v2`（hybrid full-attention + Mamba） |
| 关联上游 | [#46453](https://github.com/vllm-project/vllm/issues/46453)（hybrid Mamba per-group prefix-hit divergence） |

> 修复为单行：`resolve_mamba_align_size()` 的判断从 `== "align"` 扩到 `in ("align", "all")`。

---

## 3. 定位过程

1. **复现脚本**：deterministic token prompt（长度=2 个 offload chunk），greedy 生成填充缓存 → 清 GPU prefix cache（`reset_connector=False`）→ 重放同 prompt；
2. **对比冷启动/取回**：冷启动正确、取回后第 5 token 变化 → 锁定 refetch 路径；
3. **定位根因**：`resolve_mamba_align_size()` 只对 `"align"` 模式开启 hit-window 对齐，但 `"all"` 模式同样在 token/block 位置存 recurrent state；prefix lookup 上限在 `num_tokens-1`（末 token 需重算 logits），精确边界处返回 N-1 却恢复边界 N 的 state（已含 token N），重算 N 时 token 被算两次。

> 定位要点：silent 精度退化先怀疑「前缀命中点是否落在有对齐的 recurrent-state 边界」——把将被重算的 token 提前重复应用一次，输出改变但完全不报错。

---

## 4. 解决方案

### 4.1 根因

对齐逻辑只服务 `"align"` 模式，遗漏了 `"all"` 模式（"all" 也按 token/block 位置存状态）。

### 4.2 修复（PR #51100，单行）

```diff
- if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode == "align":
+ if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode in ("align", "all"):
```

修复后 refetch 回退到合法 recurrent-state 边界（本场景 `cached=1408`），生成输出与冷启动逐位一致。

### 4.3 回归测试建议

将现有 `test_mamba_align_cpu_offload` 的精确边界用例参数化到 `"all"` 模式。

---

## 5. 复现方法

1. 用上述配置创建 `LLM`（`enable_prefix_caching=True`、`mamba_cache_mode="all"`、`OffloadingConnector`、eager、greedy）；
2. 用长度恰为 2 个 offload chunk（本配置 2816）的 deterministic prompt；
3. greedy 生成一次填充 CPU offload 缓存；
4. 清 GPU prefix cache（保留 connector 缓存 `reset_connector=False`）；
5. 重放同 prompt → 观察第 5 token 从 1556 变为 1270。

> vllm/vllm-ascend 复现版本：上游 vllm **0.25.1 nightly**；vllm-ascend 版本「未知」（逻辑层通用，NPU 走同一 connector 也可对拍验证）。

---

## 核心教训

KVCache offload 的 prefix 命中点若落在无对齐的 recurrent-state 边界，等于把将重算的 token 重复应用一次，且必然静默错误；任何「按 mode 分支的对齐策略」都必须覆盖所有实际存储状态的 mode 组合。