# 案例12：int8_per_token_head 低精度 KV 在混合注意力 Gemma-4 上静默受损

> **一句话定位**：`--kv-cache-dtype int8_per_token_head` 在 Gemma-4（混合滑动/全局注意力，head_dim 256/512）的 Triton 注意力后端下，追加 per-token scale 后两种 KV page 尺寸(520B/1032B)不再被整除，走非连续路径；`_ensure_scale_caches` 从 `shape` 而非真实 `stride()/storage_offset()` 推导 stride，把 padding 后 scale 视图放错块 → 饱和负载后同一 prompt 输出受损。
>
> **对象**：vllm-project/vllm [#49716](https://github.com/vllm-project/vllm/issues/49716)（Bug，closed fixed）+ 修复 PR [#44455](https://github.com/vllm-project/vllm/pull/44455)（KV-Cache Layout Refactor，merged）

---

## 1. 问题描述

### 1.1 现象

在 `--kv-cache-dtype int8_per_token_head`、Triton 注意力后端、Gemma-4（混合 head_dim 256/512）下：

- 引擎冷启动时请求大多正确；并发饱和负载（约 40s，写+复用+逐出 KV 块）后**同一确定性 prompt** 输出受损——空补全（全特殊 token、`finish_reason=length`）或退化重复（" our our…"）；
- `logprobs=true` 请求开始返回 HTTP 400。

### 1.2 触发条件（必现矩阵）

| int8_per_token_head KV | 混合 head_dim | Triton 后端 | 饱和负载后重放 | 是否触发 |
|:---:|:---:|:---:|:---:|:---:|
| ✗（bf16） | ✓ | — | — | ✗（连续路径） |
| ✓ | ✗（uniform，如 Qwen2.5-0.5B） | ✓ | ✓ | ✗（page size 整除） |
| ✓ | ✓ | ✓（Ampere 仅此后端） | ✓（非冷启动） | ✓ 输出受损 |
| ✓ | ✓ | ✓ | ✗（冷启动） | △（偶发） |

> 与 prefix-caching / kv-sharing-fast-prefill 无关，`--no-enable-prefix-caching` 亦复现。

### 1.3 影响与严重度

- **严重度**：🔴 高（KV 量化路径下 2×24GB Gemma-4 用户静默受损）。
- **隐蔽性**：🟡 中（需「混合 head_dim × 低精度 KV」组合 + 负载时序）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm |
| 对象 | Issue [#49716](https://github.com/vllm-project/vllm/issues/49716)（Bug，closed fixed） |
| 修复 PR | [#44455](https://github.com/vllm-project/vllm/pull/44455)（KV-Cache Layout Refactor，merged 2026-07-11） |
| 复现版本 | vLLM v0.25.0 / v0.25.1 |
| 修复落点 | v0.26.0 |
| vllm-ascend | 「未知」（同机制——page size 因内联 scale 不再整除——NPU 上很可能同类触发） |
| 模型 | `cyankiwi/gemma-4-31B-it-AWQ-4bit`（hybrid，head_dim 256/512） |

---

## 3. 定位过程

1. **确认组合**：仅 Triton 后端、仅混合 head_dim 触发；同栈 uniform 模型逐字节干净；
2. **分析 page size**：混合布局下两种 KV page 附加 per-token scale 后尺寸 520B vs 1032B（不可整除）；bf16 时 262144/524288 整除、连续、正常；
3. **定位根因**：`triton_attn.py::_ensure_scale_caches` 从 tensor shape 推导 stride，把 padding 后 scale 视图放错块。

> 定位要点：「正确性依赖几何布局整除」的路径，一旦 layout 因内联 scale 改变整除关系，就应怀疑 stride 推导用的是 `shape` 还是真实 `stride()/storage_offset()`。

---

## 4. 解决方案

### 4.1 根因

per-token-head 低精度 KV 引入内联 scale 改变两种 page 的整除关系；后端把 scale 视图按 C-contiguous 假设放置到错误 block。

### 4.2 修复（PR #44455）

KV-Cache 布局重构（Pack K/V into content dim），`_ensure_scale_caches` 改为用 `kv_cache.stride()` / `storage_offset()` 而非从 shape 推导 stride。落点 v0.26.0。

### 4.3 验证

原 rig 2×RTX3090 上 v0.26.0 60/60 探针干净、`logprobs` 恢复，质量与 bf16 基本一致；修复前饱和后 12/12 garbage + 全量 HTTP 400。

---

## 5. 复现方法

```bash
vllm serve cyankiwi/gemma-4-31B-it-AWQ-4bit \
  --tensor-parallel-size 2 --dtype bfloat16 --gpu-memory-utilization 0.94 \
  --max-model-len 16384 --max-num-seqs 32 --enable-chunked-prefill \
  --trust-remote-code --attention-backend TRITON_ATTN \
  --kv-cache-dtype int8_per_token_head
```

随后 96-worker 饱和洪泛，再按确定性 prompts 探测。

> vllm/vllm-ascend 复现版本：上游 vllm **v0.25.0/v0.25.1**；vllm-ascend 版本「未知」——建议在 NPU 上以 `int8_per_token_head` + 混合 head_dim 模型对拍验证。

---

## 核心教训

凡「正确性依赖几何布局整除，且 stride 只按形状假设 C-contiguous」的低精度 KV 后端，都必须用真实 `stride()/storage_offset()` 而非 `shape` 推导；内联 scale 会悄然改变整除关系，静默受损难复现。