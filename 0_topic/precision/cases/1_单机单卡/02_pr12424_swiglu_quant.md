# 案例02：npu_dequant_swiglu_quant 算子小形状 tiling bug 导致量化精度错误（x.shape=[2,192]）

> **一句话定位**：`npu_dequant_swiglu_quant` 算子在输出维 `outDimy` 非 64 对齐时，DynamicQuant 的 scale 计算与 tiling 的 UB（统一缓冲区）容量预留漏算了新 scale buffer 的空间，导致小形状（如 `x.shape=[2,192]`）下量化尺度算错、算子精度错误。
>
> **对象**：vllm-project/vllm-ascend [#12424](https://github.com/vllm-project/vllm-ascend/pull/12424)（BugFix，merged 到 `releases/v0.22.1rc`）

---

## 1. 问题描述

### 1.1 现象

在 **INT8 动态量化（dynamic quantization）路径** 使用 `npu_dequant_swiglu_quant` 算子的场景下：

- 输入形状未 64 对齐（典型 `x.shape=[2,192]`，`outDimy` 非 64 对齐）时，算子输出**精度错误**（与黄金值对不齐）；
- 根因有两处叠加：DynamicQuant 在 `outDimy` 非 64 对齐时返回**错误 scale**；且修复 scale 需要引入临时 `scaleBuf_` 缓冲区，但 tiling 的 UB 容量预留**没算上这块新增内存**（under-allocation），进一步踩踏数据。

### 1.2 触发条件（必现矩阵）

| 使用该算子 | outDimy 为 64 的倍数 | 是否触发 |
|:---:|:---:|:---:|
| ✗ | — | ✗ |
| ✓ | ✓ | ✗（64 对齐路径正确） |
| **✓** | **✗（非 64 对齐）** | **✓ 精度错误** |

> 小形状 tiling（small shape tiling）是触发面：`x.shape=[2,192]` 这类非 64 对齐的小形状必现。

### 1.3 影响与严重度

- **严重度**：🟡 中（影响 INT8 动态量化反量化 + SwiGLU 融合路径的数值正确性；非 64 对齐形状必现，但 64 对齐形状不受影响）。
- **隐蔽性**：🟡 高（是「数值精度错误」而非崩溃，需对拍黄金值才能在单算子层发现）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#12424](https://github.com/vllm-project/vllm-ascend/pull/12424)（BugFix） |
| 状态 | merged 到 `releases/v0.22.1rc`（merged commit `368111f`） |
| vllm-ascend 版本 | v0.22.1（修复同时 backport 到 v0.23.0） |
| 对应的上游 vllm | [`vllm-project/vllm@967c5c3`](https://github.com/vllm-project/vllm/commit/967c5c3bc38891f4465d3f4e99917ed837bb3833) |
| 改动规模 | +49 / −46（含新增单算子测试用例） |
| 影响文件 | `dequant_swiglu_quant_tiling.cpp`、kernel 实现、`tests/e2e/nightly/single_node/ops/singlecard_ops/test_dequant_swiglu_quant.py` |
| 最小复现形状 | `x.shape=[2,192]`（outDimy 非 64 对齐） |

---

## 3. 定位过程

1. **确认现象**：`npu_dequant_swiglu_quant` 在 `x.shape=[2,192]` 下输出与黄金值（golden）不符，精度错误。
2. **锁定 tiling bug**：问题只在「非 64 对齐的小形状」出现，64 对齐形状正常 → 定位到 small shape tiling 路径。
3. **review 反馈聚焦**：`DynamicQuant` 在 `outDimy` 非 64 对齐时返回错误 scale；为修正它引入临时广播结果缓存 `scaleBuf_`，但 tiling 计算漏掉了这块 UB 空间。
4. **确认 UB 容量漏算**：`dequant_swiglu_quant_tiling.cpp` 的 `numerator` 仅扣除了 `UB_RESERVE + BLOCK_SIZE + db * BLOCK_SIZE + sizeof(float)`，未扣除 `scaleBuf_` 所需的 `BLOCK_ELEM * BLOCK_ELEM * sizeof(float)` 与 broadcast 中间量 → 修正。

> 定位要点：「算子输出精度错」且只在非对齐形状出现时，优先怀疑 **tiling（分块）的 UB 容量计算**是否在某条非对齐分支漏算了新增缓冲区/中间量——under-allocation 踩踏相邻数据，表现就是静默精度错误而非崩溃。

---

## 4. 解决方案

### 4.1 根因

`npu_dequant_swiglu_quant` 的小形状 tiling：

1. DynamicQuant 在 `outDimy` 非 64 对齐时算错 scale；
2. 为修 scale 需在 kernel 里引入临时 `scaleBuf_` buffer，但 tiling 的 UB 容量预算没有为它预留空间，导致 under-allocation、数据踩踏。

### 4.2 修复内容

**① tiling 补上 scale buffer 的 UB 空间**

```diff
-  int64_t numerator = static_cast<int64_t>(ubSize_) - UB_RESERVE - BLOCK_SIZE - db * BLOCK_SIZE - static_cast<int64_t>(sizeof(float));
+  int64_t numerator = static_cast<int64_t>(ubSize_) - UB_RESERVE - BLOCK_SIZE - db * BLOCK_SIZE - static_cast<int64_t>(sizeof(float)) -
+                       BLOCK_ELEM * BLOCK_ELEM * static_cast<int64_t>(sizeof(float));
```

**② kernel 引入 scale buffer + 修正 DynamicQuant scale 计算**

- 增加专用 `scaleBuf_` 缓存临时 broadcast 结果；
- DynamicQuant 对非 64 对齐的 `outDimy` 正确计算 scale。

### 4.3 验证

- 新增参数化单算子测试 `test_dequant_swiglu_quant.py`，覆盖 `[2,192]`（非对齐）、`[1,256]`（单行 64 对齐）等形状；
- 修复后将原本注释掉的正确性断言重新启用（`torch.testing.assert_close`），并放宽 int8 量化输出容差到 `atol=1` 以覆盖取整误差；
- `x.shape=[2,192]` 输出与黄金值对齐。

---

## 5. 复现方法

### 5.1 最小复现形状

- **`x.shape=[2,192]`**（`outDimy=192` 非 64 对齐）——单算子级即可复现，无需整模型。

### 5.2 最小复现命令（单算子测试）

```bash
# 在 vllm-ascend 仓库内运行单算子 e2e 测试
pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_dequant_swiglu_quant.py -v

# 或直接构造输入对拍（示意）
# x = torch.randn(2, 192, dtype=torch.float16, device="npu")
# 调 npu_dequant_swiglu_quant，与 CPU golden 实现 assert_close
```

> 关键差异项：输入形状的 `outDimy` **非 64 对齐**（如 192）。修复前输出与 golden 偏差超容差、scale 错误；修复后 `atol=1` 内对齐。对照：`[1,256]`（64 对齐）修复前后皆正确。

---

> **核心教训**：tiling / 分块类算子凡新增任何临时 buffer（scale、reduction、broadcast 中间量），都必须同步更新 **UB 容量预算公式**——under-allocation 不会 crash，而是踩踏相邻数据，表现为「只在非对齐形状下才出的精度错误」，隐蔽性极强；单算子级对拍（golden compare）+ 非对齐形状参数化用例是最有效的回归护栏。