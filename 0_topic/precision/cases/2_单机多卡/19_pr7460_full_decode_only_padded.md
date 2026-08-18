# 案例 19：FULL_DECODE_ONLY 图模式下 num_reqs_padded 错误致 Qwen3-Next 精度退化

> **一句话定位**：`cudagraph_mode=FULL_DECODE_ONLY` 下 padding 逻辑按 FULL 模式执行（`_pad_query_start_loc_for_fia` 只检查了 runtime mode、没检查编译配置），`num_reqs_padded` 被置错 → Qwen3-Next 精度退化。
>
> **对象**：vllm-project/vllm-ascend [PR #7460](https://github.com/vllm-project/vllm-ascend/pull/7460)（fixed graph mode bug，merged 2026-03-22，收录于 v0.18.0rc1 release checklist #7634）

---

## 1. 问题描述

### 1.1 现象

- 编译配置 `cudagraph_mode=FULL_DECODE_ONLY` 时，`num_reqs_padded` 被置为错误值；
- 根因：`_pad_query_start_loc_for_fia` 的 padding 条件只检查了**运行时模式**，未检查**编译配置**——FULL_DECODE_ONLY 配置下运行时进入 decode 分支时仍按 FULL 模式做 query_start_loc padding；
- 表现为 Qwen3-Next（FIA 注意力）精度退化。

### 1.2 触发条件（必现矩阵）

| cudagraph_mode | 运行时 decode 分支 | 是否触发 |
|:---:|:---:|:---:|
| FULL | ✓ | ✗（padding 本应生效） |
| ✗（eager） | — | ✗ |
| **FULL_DECODE_ONLY** | **✓** | **✓ padding 误生效、num_reqs_padded 错** |

### 1.3 影响与严重度

- **严重度**：🔴 高（FIA 类模型图模式精度退化）。
- **隐蔽性**：中（特定编译配置组合才触发）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#7460](https://github.com/vllm-project/vllm-ascend/pull/7460)（bugfix） |
| 状态 | merged（2026-03-22，v0.18.0rc1 周期） |
| vllm-ascend 版本 | main（2026-03-22） |
| 上游 vLLM | v0.17.0（main @8a68046） |

---

## 3. 定位过程

未知（PR 未详述；由 Qwen3-Next 精度对比发现）。

---

## 4. 解决方案

### 4.1 根因

图模式 padding 行为只在**运行时**判断，未校验**编译期配置**——两者不一致时（FULL_DECODE_ONLY 配置、decode 运行时）静默破坏精度。

### 4.2 修复内容

`vllm_ascend/worker/model_runner_v1.py`：

```python
if (... and self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL):
    # query_start_loc 的 padding 仅在编译配置与运行时均为 FULL 时生效
```

`num_reqs_padded` 不再被错误置为 `num_reqs`。

### 4.3 验证

Qwen3-Next 精度对比（PR 附截图，修复后与 eager 一致）。

---

## 5. 复现方法

### 5.1 最小复现模型

- **Qwen3-Next**（hybrid FIA，模型较大需 8 卡；无更小同类替代）。

### 5.2 复现命令

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tensor-parallel-size 8 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
# 与 --enforce-eager 输出对比；修复前精度退化
```

---

## 核心教训

图模式 padding 行为必须**同时校验编译期配置与运行期模式**，二者不一致时会静默破坏精度；cudagraph_mode 的每个枚举值都应进精度回归矩阵。
