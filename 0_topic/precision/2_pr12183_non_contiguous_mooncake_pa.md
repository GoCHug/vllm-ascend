# PR #12183 深度案例：Mooncake PA Cache 传输输入非连续导致段错误

> 整理时间: 2026-07-29
>
> 案例对象: [vllm-project/vllm-ascend#12183](https://github.com/vllm-project/vllm-ascend/pull/12183)
>
> 标题: [v0.23.0][BugFix][KV Transfer] Fix non-contiguous Mooncake PA cache inputs
>
> 关联全景文档: [0_kvcache.md](./0_kvcache.md) §2.1 / §2.3 / 案例 1
>
> 关键词: PD 分离 · Mooncake Connector · PA (Prefix-Attention) cache · non-contiguous · npu_gather/scatter_pa_kv_cache · DeviceOperator · 段错误 (segfault)

---

## 1. PR 概览

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| PR 编号 | [#12183](https://github.com/vllm-project/vllm-ascend/pull/12183) |
| 类型 | BugFix |
| 作者 | [@maoxx241](https://github.com/maoxx241) |
| Reviewer | gemini-code-assist[bot] |
| 状态 | closed / **未合并（draft + merge-conflicts）** |
| 目标分支 | `releases/v0.23.0`（基于 v0.23.0 release 分支，非 main） |
| 创建时间 | 2026-07-16 |
| 关闭时间 | 2026-07-18 |
| Head SHA | `d5dee2608464405113f3ada6969afcfcc20eb8ea` |
| 改动规模 | **+86 / −17**，4 个文件 |
| 影响文件 | `mooncake_connector.py` · `mooncake_hybrid_connector.py` · `mooncake_layerwise_connector.py` · `tests/ut/device/test_device_op.py` |
| vLLM 版本 | v0.23.0 |
| vLLM main | `ee0da84ab9e04ac7610e28580af62c365e898389` |
| 前置 PR | [#11867](https://github.com/vllm-project/vllm-ascend/pull/11867)（PA scatter/gather 算子替换，引入本 bug） |
| 关联 PR | [#12172](https://github.com/vllm-project/vllm-ascend/pull/12172)（独立的 revert 路径，本 PR 不包含该 revert） |
| 测试方式 | AST 回归单测 + py_compile + ruff check/format + format.sh ci |

> 典型的“回归型 bug 修复”：前置 PR #11867 把 KV cache 搬运从旧路径切到 `npu_gather/scatter_pa_kv_cache`，却绕过了 `DeviceOperator` 的连续性归一化，导致 DeepSeek-V3.1 PD 分离触发段错误。本 PR 把 5 处直接调用统一收口回 `DeviceOperator`。

> 注：本 PR 在 v0.23.0 release 分支上以 draft 关闭（存在 merge-conflicts），未 merged。但其修复思路（路由回 `DeviceOperator`）是 v0.23.0 线上恢复 PD 分离可用性的关键补丁之一，技术上具完整参考价值。

---

## 2. 场景背景

### 2.1 PD 分离部署 (Prefill/Decode Disaggregation)

PD 分离架构中，Prefill 节点（P 端）一次性算完长 prompt 的 KV Cache，再通过跨节点传输把 KV Cache 搬到 Decode 节点（D 端），D 端逐 token 自回归生成时直接复用，避免重复计算 prefix。vLLM-Ascend 在此场景使用 **Mooncake Connector**（基于 Mooncake / RDMA 传输引擎）作为 KV 跨节点通路。

### 2.2 PA (Prefix-Attention) cache 与搬运算子

Ascend NPU 上，KV Cache 以 **PA (Prefix Attention) 布局** 存储（paged-attention 的 NPU 实现）。搬运 KV Cache 时用到两个底层 `torch_npu` 算子：

- `torch_npu.npu_gather_pa_kv_cache` —— **gather**：按 block_table 从 PA cache 里把指定 block 的 K/V 拉到连续 buffer（P 端“读出”待传输数据）
- `torch_npu.npu_scatter_pa_kv_cache` —— **scatter**：把 buffer 里的 K/V 按 slot_mapping 写回 PA cache（D 端“写入”收到的数据）

这两个算子对接的是 ACL 底层 C++ 实现，**对输入 tensor 的内存布局有严格要求**：关键入参（seq_len / key / value / slot_mapping）必须是 **contiguous（连续内存）**，否则算子按 stride=1 的假设取数会读到错位数据，甚至触发底层越界 / 段错误。

### 2.3 DeviceOperator —— 被绕过的“归一化入口”

vLLM-Ascend 的 `vllm_ascend/device/device_op.py` 提供了设备抽象 `DeviceOperator`（按设备类型分发到 `BaseDeviceAdaptor` / `A5DeviceAdaptor` / `Ascend310PDeviceAdaptor`）。其中两个方法**专门负责在调用底层算子前做连续性归一化**：

```python
# vllm_ascend/device/device_op.py
class BaseDeviceAdaptor:
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu.npu_scatter_pa_kv_cache(
            key=key.contiguous(),            # ← 显式 .contiguous()
            value=value.contiguous(),
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping.contiguous(),
            cache_mode="Norm",
        )

    @staticmethod
    def kv_cache_load(cache_kv_c, cache_k_pe, block_table, context_seq_len_npu, seq_starts, key, value):
        torch_npu.npu_gather_pa_kv_cache(
            cache_kv_c,
            cache_k_pe,
            block_table,
            context_seq_len_npu.contiguous(),  # ← 显式 .contiguous()
            seq_offset=seq_starts,
            key=key,
            value=value,
        )
```

关键点：`reshape_and_cache` 对 `key/value/slot_mapping` 做 `.contiguous()`；`kv_cache_load` 对 `context_seq_len_npu`（即 seq_len tensor）做 `.contiguous()`。`A5DeviceAdaptor` 同样保留这两个连续性保证（见 `device_op.py:1388-1397`）。此外 `DeviceOperator` 还承载 **设备特定行为**（如 Ascend 310P 走 `_npu_reshape_and_cache` 旧实现，见 `device_op.py:1841-1850`）。

### 2.4 non-contiguous 概念

一个 tensor 可以有非连续内存布局（stride 不等于形状暗示的步长），常见来源：

- `tensor.transpose(...)` / `tensor.permute(...)` —— 仅改 stride，不拷数据
- `tensor.view(...)` 配合 `transpose_` —— 产生非连续视图
- slice / narrow / expand —— 非连续子视图

本案例中，Mooncake connector 在 `reformat_kv_cache` / `_cat_kv_cache` / `save_kv_layer` 路径里，会先对 buffer 做 `transpose_`、`view` 等操作（见 §3.3），产出的 `seq_start_tensor`、`k_buffer`、`slot_mapping` 等**可能是非连续的**。这些 tensor 一旦直接喂给 `npu_gather/scatter_pa_kv_cache`，就会触发底层算子的连续性假设违例。

---

## 3. 根因分析

### 3.1 前置 PR #11867 引入的回归

#11867 把 Mooncake connector 里 KV cache 的搬运从旧实现切换到 `torch_npu.npu_gather_pa_kv_cache` / `npu_scatter_pa_kv_cache`，但其调用点是**直接调 `torch_npu.*`**，而非走 `DeviceOperator`。这等于**绕过了 `DeviceOperator` 内置的 `.contiguous()` 归一化**，让 non-contiguous 的 seq_len / key / value / slot_mapping tensor 直达 NPU 底层算子。

PR body 原文：

> The standard, hybrid, and layerwise Mooncake connectors called `torch_npu.npu_gather_pa_kv_cache` and `torch_npu.npu_scatter_pa_kv_cache` directly. Those call sites bypassed the input-layout normalization in `DeviceOperator`, allowing non-contiguous sequence-length, key, value, or slot mapping tensors to reach the NPU operators.

### 3.2 Bug 代码（5 处直接调用）

修复前，三个 connector 共 5 处直接调 `torch_npu` 底层算子（以 `mooncake_connector.py` 为例）：

```python
# vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py

# ① reformat_kv_cache —— gather（读取 PA cache 到 buffer）
def reformat_kv_cache(self, block_ids, tp_num_need_pulls, ...):
    ...
    for _, (k_cache_layer, v_cache_layer) in kv_caches.items():
-       torch_npu.npu_gather_pa_kv_cache(           # ← BUG：直接调，未 .contiguous()
            k_cache_layer,
            v_cache_layer,
            block_table,
            block_len_tensor,
-           seq_offset=seq_start_tensor,             # ← seq_start_tensor 可能非连续
-           key=k_buffer,
-           value=v_buffer,
        )

# ② _cat_kv_cache —— scatter（把 transpose 后的 buffer 写回 PA cache）
def _cat_kv_cache(self, k_cache_layer, v_cache_layer, k_buffer, v_buffer, ...):
    def _transpose_kv_cache_between_head(buffer):
        buffer = buffer.view(num_blocks, tp_num_need_pulls, self.block_size, -1)
        buffer.transpose_(1, 2)                      # ← transpose_ 产生非连续
        return buffer.contiguous().view(num_tokens, num_kv_heads, -1)

    k_buffer = _transpose_kv_cache_between_head(k_buffer)
    v_buffer = _transpose_kv_cache_between_head(v_buffer)
-   torch_npu.npu_scatter_pa_kv_cache(              # ← BUG：直接调
        key=k_buffer,
        value=v_buffer,
        key_cache=k_cache_layer,
        value_cache=v_cache_layer,
        slot_mapping=slot_mapping,
-       cache_mode="Norm",
    )
```

同样的直接调用模式还出现在：

- `mooncake_hybrid_connector.py:901`（`reformat_kv_cache` 的 gather）
- `mooncake_hybrid_connector.py:939`（`_cat_kv_cache` 的 scatter）
- `mooncake_layerwise_connector.py:1754`（`save_kv_layer` 的 gather）

### 3.3 为什么输入会 non-contiguous

以 `_cat_kv_cache` 为例，`_transpose_kv_cache_between_head` 内部做了 `view → transpose_ → contiguous().view(...)`，其中 `transpose_` 后的中间态是非连续的。虽然最终 `.contiguous()` 拷了一次，但在更上游，`slot_mapping`、`seq_start_tensor`、`block_len_tensor` 等由 `arange` + `reshape` + `flatten` / `view` 构造（见 `mooncake_connector.py:1206-1217`），在特定 batch/group 形状下并不保证连续；而 gather 路径的 `block_table`、`block_len_tensor` 同样是从 list 拼出的 tensor。这些 tensor 在 #11867 之前由旧路径自行处理或本就是连续视图，切换到 PA 算子后假设不再成立。

`mooncake_layerwise_connector.save_kv_layer` 里的 `group_block_table` / `group_block_len_tensor` / `group_seq_start_tensor` 来自按 group 切分的子 tensor（slice），**slice 默认非连续**，风险更高。

### 3.4 触发链路

```
DeepSeek-V3.1 + PD 分离
  → Mooncake connector 跨节点传 KV Cache
  → reformat_kv_cache / _cat_kv_cache / save_kv_layer 调用 PA 算子
  → #11867 改为直接调 torch_npu.npu_gather/scatter_pa_kv_cache
  → 绕过 DeviceOperator 的 .contiguous() 归一化
  → non-contiguous 的 seq_len / key / value / slot_mapping 直达 NPU 算子
  → 底层 C++ 算子按 stride=1 假设取数 → 越界 / 数据错位
  → 段错误 (segmentation fault)，进程崩溃
```

**触发条件 = DeepSeek-V3.1 + PD 分离 + #11867 的 PA 算子替换**。其中 non-contiguous 的具体来源取决于 batch 形状 / TP 配比 / group 切分，因此并非每次必现，但 DeepSeek-V3.1 的 PD 分离在任何一组产生 slice/transpose 输入的路径上都会暴露。

---

## 4. 影响与表现

| 维度 | 表现 |
|------|------|
| 触发配置 | DeepSeek-V3.1 + PD 分离 + 已合入 #12172 之前的 PA 算子替换（#11867） |
| 不触发 | 非 PD 分离 / 未应用 #11867 / 走 NZ 专属路径（`_nz_kv_cache` 未改） |
| 现象 | 进程段错误 (segmentation fault) 崩溃，KV 传输中断，PD 分离不可用 |
| 严重度 | 🔴 高（功能性阻断：PD 分离直接崩溃，非 silent 精度问题） |
| 隐蔽性 | 🟡 中（崩溃型，易被察觉；但根因藏在“直接调 vs 走 DeviceOperator”这一看似等价的替换里，定位需熟悉 device 抽象） |
| 是否报错 | **是**：段错误，进程级崩溃（非 Python 异常，无 traceback，需 core/栈定位） |
| 与 #8540 对比 | #8540 是 silent 精度下降（不崩）；本 PR 是 hard crash（崩）。两者都属 Mooncake KV 传输正确性问题，但表现形态相反 |

段错误发生在 NPU 底层算子的 C++ 栈中，Python 层无异常抛出，因此 CI 的 pytest 用例（不跑真实 NPU）无法捕获——这也是本 PR 选择用 **AST 静态检查**而非运行时单测来防回归的原因（见 §5.2）。

---

## 5. 修复方案

### 5.1 核心改动（diff）

把 5 处直接 `torch_npu.*` 调用统一改为 `DeviceOperator.*`，并同步调整参数顺序/去掉 `cache_mode="Norm"`（因为 `DeviceOperator.reshape_and_cache` 内部已固定 `cache_mode="Norm"`）：

```diff
# mooncake_connector.py —— ① gather
-           torch_npu.npu_gather_pa_kv_cache(
+           DeviceOperator.kv_cache_load(
                k_cache_layer,
                v_cache_layer,
                block_table,
                block_len_tensor,
-               seq_offset=seq_start_tensor,
-               key=k_buffer,
-               value=v_buffer,
+               seq_start_tensor,
+               k_buffer,
+               v_buffer,
            )

# mooncake_connector.py —— ② scatter
-       torch_npu.npu_scatter_pa_kv_cache(
+       DeviceOperator.reshape_and_cache(
            key=k_buffer,
            value=v_buffer,
            key_cache=k_cache_layer,
            value_cache=v_cache_layer,
            slot_mapping=slot_mapping,
-           cache_mode="Norm",
        )
```

同样的替换镜像出现在 `mooncake_hybrid_connector.py`（`reformat_kv_cache` + `_cat_kv_cache`）和 `mooncake_layerwise_connector.py`（`save_kv_layer`）。`mooncake_layerwise_connector.py` 还顺带删掉了不再需要的 `import torch_npu`。

### 5.2 回归测试：AST 静态守卫

由于段错误在无 NPU 环境下无法复现，本 PR 在 `tests/ut/device/test_device_op.py` 新增了一个**基于 AST 的静态回归测试**，确保 5 处调用点永远走 `DeviceOperator`、不再回退到直接 `torch_npu` 调用：

```python
# tests/ut/device/test_device_op.py
def _get_function_call_names(relative_path: str, function_name: str) -> set[str]:
    repo_root = Path(__file__).parents[3]
    tree = ast.parse((repo_root / relative_path).read_text())
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    ]
    assert functions
    ...
    return {get_name(node.func) for function in functions for node in ast.walk(function) if isinstance(node, ast.Call)}

@pytest.mark.parametrize(
    ("relative_path", "function_name", "expected_call", "forbidden_call"),
    [
        ("vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py",
         "reformat_kv_cache", "DeviceOperator.kv_cache_load", "torch_npu.npu_gather_pa_kv_cache"),
        ("vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py",
         "_cat_kv_cache", "DeviceOperator.reshape_and_cache", "torch_npu.npu_scatter_pa_kv_cache"),
        (".../mooncake_hybrid_connector.py", "reformat_kv_cache",
         "DeviceOperator.kv_cache_load", "torch_npu.npu_gather_pa_kv_cache"),
        (".../mooncake_hybrid_connector.py", "_cat_kv_cache",
         "DeviceOperator.reshape_and_cache", "torch_npu.npu_scatter_pa_kv_cache"),
        (".../mooncake_layerwise_connector.py", "save_kv_layer",
         "DeviceOperator.kv_cache_load", "torch_npu.npu_gather_pa_kv_cache"),
    ],
)
def test_mooncake_pa_cache_calls_use_device_operator(...):
    call_names = _get_function_call_names(relative_path, function_name)
    assert expected_call in call_names
    assert forbidden_call not in call_names
```

这是一个**“结构性回归测试”**：不验证数值正确性，而是用 AST 断言“调用点必须经过 DeviceOperator、禁止直接 torch_npu”，从源头封死再次绕过归一化的可能。对在无 NPU 的 CI/macOS 上防回归尤其有效。

### 5.3 核心思想

**把对底层 PA 算子的访问全部收口到 `DeviceOperator` 这一层**，由设备抽象统一负责 `.contiguous()` 归一化与设备特定分发（含 Ascend 310P 的 `_npu_reshape_and_cache` 旧路径、310P/`A5` 的差异）。任何 connector / 业务代码都不得直接调 `torch_npu.npu_gather/scatter_pa_kv_cache`。NZ 专属路径（`_nz_kv_cache`）因有自己的布局约定，保持不变。

---

## 6. 关键代码上下文

文件：`vllm_ascend/device/device_op.py`

| 位置 | 内容 | 说明 |
|------|------|------|
| `49-57` | `BaseDeviceAdaptor.reshape_and_cache` | scatter 入口：`key/value/slot_mapping` 显式 `.contiguous()`，固定 `cache_mode="Norm"` |
| `298-307` | `BaseDeviceAdaptor.kv_cache_load` | gather 入口：`context_seq_len_npu.contiguous()` |
| `1388-1397` | `A5DeviceAdaptor.kv_cache_load` | A5 重载，同样保留 `.contiguous()` |
| `1841-1850` | `Ascend310PDeviceAdaptor.reshape_and_cache` | 310P 走 `_npu_reshape_and_cache`（无 `cache_mode` 参数），正是“走 DeviceOperator 才能保留设备行为”的体现 |
| `1878-1887` | `get_device_adaptor` / `DeviceOperator` | 按设备类型分发到对应 Adaptor |

文件：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`

| 位置 | 内容 | 说明 |
|------|------|------|
| `1185-1260` | `reformat_kv_cache` | 调用 gather；本 PR 把 `1227` 处 `torch_npu.npu_gather_pa_kv_cache` 改为 `DeviceOperator.kv_cache_load` |
| `1206-1217` | `slot_mapping` 构造 | `arange + reshape + flatten`，非连续风险来源 |
| `1262-1291` | `_cat_kv_cache` | 内含 `_transpose_kv_cache_between_head`（transpose_ 非连续），调用 scatter；本 PR 改 `1284` |
| `1293-1311` | `_nz_kv_cache` | NZ 专属路径，未改（保持 `npu_scatter_pa_kv_cache`） |

文件：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py`

| 位置 | 内容 | 说明 |
|------|------|------|
| `861-` / `901` | `reformat_kv_cache` gather | 同样改为 `DeviceOperator.kv_cache_load` |
| `926-` / `939` | `_cat_kv_cache` scatter | 同样改为 `DeviceOperator.reshape_and_cache` |
| `948-956` | `_nz_kv_cache` | NZ 路径，未改 |

文件：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`

| 位置 | 内容 | 说明 |
|------|------|------|
| `1691-` / `1754` | `save_kv_layer` gather | `group_block_table` / `group_seq_start_tensor` 为按 group slice 的子 tensor（非连续风险高），本 PR 改为 `DeviceOperator.kv_cache_load` |

文件：`tests/ut/device/test_device_op.py`

| 位置 | 内容 | 说明 |
|------|------|------|
| `1-` 新增 | `_get_function_call_names` + `test_mooncake_pa_cache_calls_use_device_operator` | AST 结构性回归测试，5 个 parametrize 用例 |

---

## 7. 复现与验证

### 7.1 触发矩阵

| DeepSeek-V3.1 | PD 分离 | #11867 PA 算子替换 | NZ 路径 | 是否触发 |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | — | — | — | ✗（非目标模型） |
| ✓ | ✗ | ✓ | — | ✗（无跨节点 PA 搬运） |
| ✓ | ✓ | ✗（旧路径） | — | ✗（未切到 PA 算子） |
| ✓ | ✓ | ✓ | ✗（走 Norm 路径） | **✓ 段错误** |
| ✓ | ✓ | ✓ | ✓（走 `_nz_kv_cache`） | ✗（NZ 路径未改，自有布局约定） |

### 7.2 排查方法

1. **确认版本与前置**：是否在 v0.23.0 线、是否已合入 #11867、是否合入 #12172 的 revert。若 #11867 已在、#12172 未在 → 高度怀疑本路径。
2. **确认模型+部署**：DeepSeek-V3.1 + PD 分离。
3. **崩溃形态**：进程段错误、无 Python traceback、core dump 落在 `npu_gather_pa_kv_cache` / `npu_scatter_pa_kv_cache` 的 C++ 栈。
4. **代码审查定位**：grep `torch_npu.npu_gather_pa_kv_cache` / `torch_npu.npu_scatter_pa_kv_cache`，凡出现在 `mooncake_*connector.py` 业务路径（非 `device_op.py`、非 `_nz_kv_cache`）即为漏网调用点。
5. **本地无 NPU 验证**：运行 `test_mooncake_pa_cache_calls_use_device_operator`，AST 断言会直接指出哪个函数仍含禁止调用。
6. **修复后验证**：确认 5 处调用点均改为 `DeviceOperator.kv_cache_load` / `reshape_and_cache`；在目标 NPU 上跑 DeepSeek-V3.1 PD 分离 e2e，不再段错误。

### 7.3 回归测试建议

- 保留并扩展 AST 测试：凡新增 Mooncake PA cache 调用点，必须加入 `test_mooncake_pa_cache_calls_use_device_operator` 的 parametrize 表。
- 补充运行时单测（需 NPU）：构造 non-contiguous 的 seq_len / key / value / slot_mapping，分别走 `DeviceOperator.reshape_and_cache` / `kv_cache_load`，断言不崩溃且输出与 contiguous 基线一致。
- e2e 回归：DeepSeek-V3.1 + PD 分离 + 多组 TP 配比，覆盖 `reformat_kv_cache` / `_cat_kv_cache` / `save_kv_layer` 三条路径。

---

## 8. 经验与启发

### 8.1 算子替换必须保留“入口归一化层”

#11867 把搬运切到 PA 算子本身没错，错在“直接调 `torch_npu`”而非“走 `DeviceOperator`”。底层算子对 stride/contiguity 的假设由设备抽象统一兜底，任何绕过入口层的直接调用都会把归一化责任散落到每个调用点——迟早有人漏。**规则：业务代码永不直接调 `torch_npu.*_pa_kv_cache`，一律走 `DeviceOperator`。**（参见 [0_kvcache.md](./0_kvcache.md) §2.3 第 1 条“non-contiguous 输入问题”。）

### 8.2 “等价替换”是最危险的回归来源

`torch_npu.npu_gather_pa_kv_cache(...)` 与 `DeviceOperator.kv_cache_load(...)` 在 happy path（输入恰好连续）下行为等价，CI 的无 NPU 单测也看不出差异。但 device 抽象层做的 `.contiguous()` 是对**所有入参**的防御性归一化，一旦省略，non-contiguous 输入在特定形状下才崩——典型的“等价但非等价”回归。做算子/接口替换时，必须核对被替换接口的全部副作用（含连续性、设备分发、cache_mode 默认值）。

### 8.3 崩溃型 bug 要用结构性测试防回归

段错误在 CI（无 NPU）上无法复现。本 PR 用 AST 静态检查断言“调用结构”（expected_call in / forbidden_call not in），把“必须走 DeviceOperator”这一架构约束固化成测试。对“无法在 CI 环境复现、但代码结构可静态判定”的 bug，**结构性 AST 测试是性价比最高的回归守卫**。

### 8.4 slice / transpose / view 是 non-contiguous 高发源

`mooncake_layerwise_connector.save_kv_layer` 的 `group_*` tensor 来自按 group 切分的 slice，默认非连续；`_cat_kv_cache` 的 buffer 经 `transpose_` 后非连续。审查任何要喂给底层算子的 tensor 时，应沿着它的构造链回溯：是否有 `transpose/permute/slice/narrow/expand`？若有，且下游算子假设连续，就必须在入口 `.contiguous()`。

### 8.5 设备抽象的“单一收口”价值

`DeviceOperator` 不只是 syntactic sugar——它同时承载：①连续性归一化、②设备特定分发（310P 走旧 `_npu_reshape_and_cache`、A5 与 Base 的差异）、③`cache_mode` 等默认值统一。把调用收口到这一层，等于把这三类一致性保证“免费”继承。本 PR 的修复本质是**恢复被 #11867 破坏的单点收口**。

### 8.6 release 分支的修复策略：补而非退

v0.23.0 线上同时存在两条修复路径：#12172（revert #11867）与本 PR #12183（保留 #11867，补齐归一化）。本 PR 选择“保留新算子、补上漏掉的 DeviceOperator 路由”，而非简单 revert，表明 PA 算子替换是性能/功能方向上的既定演进——修 bug 不应回退方向，而应补齐工程化收口。

---

## 9. 关联问题

| 编号 | 关联点 |
|------|--------|
| [vllm-ascend#11867](https://github.com/vllm-project/vllm-ascend/pull/11867) | PA scatter/gather 算子替换 —— 引入本 bug 的前置 PR |
| [vllm-ascend#12172](https://github.com/vllm-project/vllm-ascend/pull/12172) | 独立的 revert 路径（v0.23.0 线另一条修复） |
| [vllm-ascend#8540](https://github.com/vllm-project/vllm-ascend/pull/8540) | TP 不等 + MTP 层 KV 漏算 —— 同属 Mooncake KV 传输正确性，见 [1_pr8540](./1_pr8540_tp_unequal_mtp_kv.md) |
| [vllm-ascend#11886](https://github.com/vllm-project/vllm-ascend/pull/11886) / [#11601](https://github.com/vllm-project/vllm-ascend/pull/11601) | Mooncake 传输元数据携带 KV head / cache group ids —— 同属 Mooncake KV 传输布局/归属正确性 |
| [vllm#47791](https://github.com/vllm-project/vllm/pull/47791) | 5D KV cache 布局处理 —— 同属“布局/连续性假设”类 bug（[0_kvcache.md](./0_kvcache.md) §2.3 第 3 条） |
| [0_kvcache.md](./0_kvcache.md) §2.1 / §2.3 / 案例 1 | 本案例在全景文档中的归档位置（§2.3 表格 #1） |

---

## 10. 时间线小结

| 时间 | 事件 |
|------|------|
| 2026-07-16 09:49 | PR #12183 创建（draft，base: `releases/v0.23.0`），提交修复 + AST 回归测试 |
| 2026-07-16 09:49 | gemini-code-assist 发 summary，09:51 给 2 条 review（PR 标题前缀顺序、`Path(__file__).parents[3]` 应 `.resolve()`） |
| 2026-07-16 14:55 | CI 标记 merge-conflicts，PR 进入 dirty 状态 |
| 2026-07-18 09:50 | PR 关闭（draft + merge-conflicts，未合并） |
| 后续 | v0.23.0 线通过 #12172（revert）或后续补丁恢复 PD 分离可用性；本 PR 的“收口 DeviceOperator”思路作为工程参考 |

---

> 本案例核心教训：**底层 PA 算子的所有调用必须收口到 `DeviceOperator` 这一层——由设备抽象统一兜底 `.contiguous()` 归一化与设备分发；任何在 connector 业务代码里直接调 `torch_npu.npu_gather/scatter_pa_kv_cache` 的“等价替换”都会在 non-contiguous 输入下触发段错误。**