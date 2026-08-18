# 案例28：DeepSeek-V3.1 PD 分离 Mooncake PA cache 输入非连续导致段错误 / 数据错位

> **一句话定位**：PA KV cache 算子替换（#11867）后，三个 Mooncake connector 直接调用 `torch_npu.npu_gather/scatter_pa_kv_cache`，绕过了 `DeviceOperator` 的输入布局归一化，让非连续的 seq_lens / key / value / slot mapping 张量直达 NPU 算子，导致 DeepSeek-V3.1 P/D 分离下段错误与数据传输错位。
>
> **对象**：vllm-project/vllm-ascend [#12183](https://github.com/vllm-project/vllm-ascend/pull/12183)（[v0.23.0][BugFix][KV Transfer]，merged 到 `releases/v0.23.0`）

---

## 1. 问题描述

### 1.1 现象

在 **DeepSeek-V3.1 · P/D 分离（Prefill/Decode Disaggregation）+ Mooncake 跨节点 KV 传输** 下：

- 开启 PA（Page Attention）算子后，**standard / hybrid / layerwise 三种 Mooncake connector** 直接调用 `torch_npu.npu_gather_pa_kv_cache` / `torch_npu.npu_scatter_pa_kv_cache`；
- 这些直接调用点**绕过了 `DeviceOperator` 的输入布局归一化**，导致非连续的 `seq_lens`、`key`、`value`、`slot_mapping` 张量被直接送进 NPU 算子；
- 表现为 **DeepSeek-V3.1 P/D 分离下的 segmentation fault（段错误）**，传输数据错位。

### 1.2 触发条件（必现矩阵）

| PD 分离 | Mooncake connector | 使用 PA 算子 | 输入张量非连续 | 是否触发 |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | — | — | — | ✗（无跨节点传输） |
| ✓ | ✓ | ✗ | — | ✗（走 DeviceOperator 归一化路径） |
| ✓ | ✓ | ✓ | ✗（连续） | △（不触发，但缺乏布局保证脆弱） |
| **✓** | **✓** | **✓** | **✓** | **✓ 段错误 / 数据错位** |

> 根因是「绕过了归一化层」而非「某次必然非连续」——非连续张量一旦出现（如 gather/scatter 前上游产生非连续视图）立即触发。

### 1.3 影响与严重度

- **严重度**：🔴 高（进程崩溃，端到端不可用；且是 KV 传输正确性的基础栅栏缺失）。
- **范围**：standard / hybrid / layerwise 三个 connector 共 **5 个调用点**全数受影响；仅 `use_compress` / NZ 特有路径因走各自分支而豁免（本次未改）。

> 说明：本案例现象为「段错误 + 数据错位」，属 KV 缓存传输正确性问题；它是后续多种**静默精度异常 / NaN** 的上游根因之一——布局不归一，非连续视图读写即错位。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#12183](https://github.com/vllm-project/vllm-ascend/pull/12183)（BugFix） |
| 状态 | merged 到 `releases/v0.23.0` |
| 目标分支 | `releases/v0.23.0` |
| vllm-ascend 版本 | v0.23.0 |
| 对应的上游 vllm | [`vllm-project/vllm@ee0da84`](https://github.com/vllm-project/vllm/commit/ee0da84ab9e04ac7610e28580af62c365e898389) |
| 改动规模 | 1 commit（+回归测试） |
| 影响文件 | 3 个 Mooncake connector（standard / hybrid / layerwise）+ `tests/ut/device/test_device_op.py` |
| 关联引入 | PA 算子替换 [#11867](https://github.com/vllm-project/vllm-ascend/pull/11867) |

---

## 3. 定位过程

1. **确认现象**：DeepSeek-V3.1 P/D 分离 + Mooncake 传输下出现 segmentation fault。
2. **回归溯源**：与 PA KV cache 算子替换 #11867 时间点吻合 → 定位到算子替换引入的调用方式变化。
3. **代码审查**：发现 connector 中 `torch_npu.npu_gather_pa_kv_cache` / `torch_npu.npu_scatter_pa_kv_cache` 被**直接调用**，未走 `DeviceOperator`；而 `DeviceOperator` 里才有「先对 seq_lens / key / value / slot mapping 做 `.contiguous()` 归一化」的逻辑。
4. **确认 5 个调用点**：standard、hybrid、layerwise 三个 connector 共 5 处直连 NPU 算子，均缺连续性保证。
5. **对比 DeviceOperator**：`DeviceOperator.kv_cache_load` / `DeviceOperator.reshape_and_cache` 会在调算子前做 contiguous，并保留 Ascend 310P reshape-and-cache 等设备特有行为 → 走 DeviceOperator 即可修复。

> 定位要点：算子替换最容易踩的坑是「新算子对输入布局/连续性有隐含要求，而老路径没把它显式化」。排查这类回归应重点 diff 新旧调用点是否绕过了统一的归一化入口（DeviceOperator）。

---

## 4. 解决方案

### 4.1 根因

PA KV cache 算子（`npu_gather_pa_kv_cache` / `npu_scatter_pa_kv_cache`）**要求输入张量连续**，但 connector 直连调用时没有保证连续性；`DeviceOperator` 封装了这一归一化（先 `.contiguous()` 再调算子），而直连调用点绕过了它。非连续的 seq_lens / key / value / slot mapping 一进 NPU 算子即读到错位地址 → 段错误 / 数据错位。

### 4.2 修复内容

把 5 个 Mooncake 调用点全部路由回 `DeviceOperator`：

```diff
--            torch_npu.npu_gather_pa_kv_cache(
+-            DeviceOperator.kv_cache_load(
+             # kv_cache_load 内部先对 seq_lens 做 contiguous，再调 gather 算子
```

```diff
--            torch_npu.npu_scatter_pa_kv_cache(
+-            DeviceOperator.reshape_and_cache(
+             # reshape_and_cache 内部先对 key / value / slot_mapping 做 contiguous，再调 scatter 算子
```

要点：

- `DeviceOperator.kv_cache_load` → 先归一化 `seq_lens` 连续性，再调 gather；
- `DeviceOperator.reshape_and_cache` → 先归一化 `key` / `value` / `slot_mapping` 连续性，再调 scatter；
- 保留设备特有行为（Ascend 310P 的 reshape-and-cache 实现），NZ 特有路径不变。

### 4.3 验证

- 新增回归单测：5 个 Mooncake PA cache 调用点必须走 `DeviceOperator`，不得回退成直接 `torch_npu` 调用；
- 目标分支 `releases/v0.23.0`，DeepSeek-V3.1 P/D 复现需 NPU 真机（本地 macOS 无 pytest / NPU runtime，未跑 e2e）。

---

## 5. 复现方法

### 5.1 最小复现模型

- **DeepSeek-V3.1**（P/D 分离 + Mooncake）；触发点在意「PA 算子开启 + 输入张量非连续」，与模型具体规模无强绑定，但需走 standard / hybrid / layerwise connector 之一。

### 5.2 最小服务命令（P/D 两端，走 Mooncake + PA 算子）

```bash
# P 端（prefill）
export VLLM_USE_V1=1
vllm serve <DEEPSEEK_V31_MODEL> \
  --port 8010 \
  --tensor-parallel-size 8 \
  --enforce-eager --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"MooncakeLayerwiseConnector","kv_role":"kv_producer","kv_port":"36000","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":8}}}'

# D 端（decode）
export VLLM_USE_V1=1
vllm serve <DEEPSEEK_V31_MODEL> \
  --port 8020 \
  --tensor-parallel-size 8 \
  --enforce-eager --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"MooncakeLayerwiseConnector","kv_role":"kv_consumer","kv_port":"36100","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":8}}}'
```

> 关键差异项：`kv_connector` 用 layerwise（DeepSeek-V3.1）/ hybrid / standard 之一，并确保运行时走到 PA 算子（`pa_shape_list` 命中批次大小）。复现前确认 connector 源码中的调用点是直连 `torch_npu.npu_gather/scatter_pa_kv_cache`（修复前）还是已走 `DeviceOperator`（修复后）——修复后此路径不再段错误。

---

> **核心教训**：NPU 算子对输入的**布局 / 连续性有隐含契约**，任何替换算子的改动都必须让调用点继续走统一的归一化入口（DeviceOperator），禁止直连底层算子绕开它——连续性问题不报编译错、只在特定张量视图下以「崩溃 / 数据错位」形式爆发，且是后续静默精度的上游源。