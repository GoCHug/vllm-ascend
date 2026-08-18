# 案例10：W4A4 MXFP4 权重量化尺度 K 维不匹配（floor 分配丢弃尾部块）

> **一句话定位**：`w4a4_mxfp4.py` 中 `get_pergroup_param` 用 `input_size // group_size`（floor）分配 `weight_scale`，在 K 非 64 对齐时丢掉尾部 block（67 块），而激活 scale 用 ceil（68 块），68 vs 67 尺度长度不一致被算子拒绝 → W4A4 MXFP4 模型 K 非 64 倍数时推理报错。
>
> **对象**：vllm-project/vllm-ascend [#12201](https://github.com/vllm-project/vllm-ascend/pull/12201)（BugFix，merged 到 main）

---

## 1. 问题描述

### 1.1 现象

在 **W4A4 MXFP4 量化模型**（K 维非 64 的倍数，如 Qwen3.5-27B vit mxfp4，K=68 需分 68 块）**推理（inference）** 时：

- `aclnnQuantMatmulV5` 报错 **`Invalid_Argument (EZ0027)`**：
  ```
  x1Scale K 68, x2Scale K 67 ... scale K must equal K ceildivided by 64.
  ```
- 激活 scale 的 K=68（ceil），权重量化尺度 `weight_scale` 的 K=67（floor）——两者长度不同，算子拒绝执行，推理直接失败。

### 1.2 触发条件（必现矩阵）

| W4A4 MXFP4 量化 | K 为 64 的倍数 | 是否触发 |
|:---:|:---:|:---:|
| ✗ | — | ✗（非该量化方法） |
| ✓ | ✓ | ✗（`// group_size` == ceil，无丢弃） |
| **✓** | **✗（K 非对齐）** | **✓ 推理报错** |

> 根因是「权重 scale 用 floor、激活 scale 用 ceil」的口径不统一，只有当 K 非 64 对齐（产生一个不满 64 的尾部 block）时才会暴露。

### 1.3 影响与严重度

- **严重度**：🟡 中（K 非对齐的 W4A4 MXFP4 模型**完全不可服务**，但 K 常见的 64 对齐模型不受影响）。
- **隐蔽性**：🟡 中（与具体模型 hidden size 相关；对齐的 K 永不触发）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#12201](https://github.com/vllm-project/vllm-ascend/pull/12201)（BugFix） |
| 状态 | merged 到 main（merged commit `dc47bc4`） |
| vllm-ascend 版本 | v0.25.0 |
| 对应的上游 vllm | [`vllm-project/vllm@85c09e9`](https://github.com/vllm-project/vllm/commit/85c09e9885e346ea1612da30ebff5a75f67d2350) |
| 改动规模 | 1 commit（单文件，`w4a4_mxfp4.py`） |
| 影响文件 | `vllm_ascend/quantization/methods/w4a4_mxfp4.py` |
| 最小复现模型 | Qwen3.5-27B vit mxfp4（K 非 64 对齐） |

---

## 3. 定位过程

1. **确认现象**：serving W4A4 MXFP4 模型（Qwen3.5-27B vit mxfp4）在 K 非 64 倍数时推理报 `aclnnQuantMatmulV5 ... x1Scale K 68, x2Scale K 67`。
2. **解读报错**：`x1Scale K 68`（激活 scale，ceil 分块）vs `x2Scale K 67`（权重 scale，floor 分块）——两尺度 K 维长度不一致。
3. **定位分配逻辑**：`w4a4_mxfp4.py` 的 `get_pergroup_param` 分配 `weight_scale` 时用 `input_size // group_size`（floor 除法），非对齐 K 丢了尾部 block（67 blocks）；而激活侧 scale 用 ceil（68 blocks）。
4. **对比 MXFP8**：MXFP8 linear 方案已用 `cdiv` + 奇数长度 padding 规避此问题，MXFP4 缺失同样处理。
5. **确认修复点**：`get_pergroup_param` 改 `cdiv(input_size, group_size)`；`process_weights_after_loading` 在 K 为奇数时对 `weight_scale` 补一列后再 reshape。

> 定位要点：量化算子报「scale K 长度不一致」时，优先检查**分块数是否对 K 用错了 floor/ceil**——同一个「分块数」概念在权重侧与激活侧各算一遍、口径不同，是最典型的量化精度 / 报错雷区。

---

## 4. 解决方案

### 4.1 根因

`weight_scale` 用 floor 除法 `input_size // group_size` 分配，K 非 64 对齐时尾部不足一组的 block 被丢弃（67），而激活 scale 用 ceil（68），两者长度不一致，被 `aclnnQuantMatmulV5` 以 `Invalid_Argument` 拒绝。MXFP8 已有 `cdiv` + odd padding 的修法，MXFP4 未同步。

### 4.2 修复内容

对齐 MXFP8 linear 方案：

```diff
  # get_pergroup_param
- params_dict["weight_scale"] = torch.empty(output_size, input_size // self.group_size, dtype=torch.uint8)
+ params_dict["weight_scale"] = torch.empty(output_size, cdiv(input_size, self.group_size), dtype=torch.uint8)
```

```diff
  # process_weights_after_loading
+ # K 为奇数时，weight_scale 补一列，保证 reshape 后 K 维与激活 scale 对齐
```

核心：`weight_scale` 的 K 维必须用 **ceil**（`cdiv`）而非 floor；K 为奇数时补列再 reshape。

### 4.3 验证

- vllm-ascend v0.25.0 + 上游 vllm `85c09e9`；单文件改动，Qwen3.5-27B vit mxfp4 推理不再报错。
- (review 建议 MoE 量化方案 `AscendW4A4MXFP4DynamicFusedMoEMethod` 也应用同样 ceil + padding，防止同类 crash。)

---

## 5. 复现方法

### 5.1 最小复现模型

- **Qwen3.5-27B vit mxfp4**（W4A4 MXFP4，K 非 64 对齐）。任意 W4A4 MXFP4 且 `hidden_size` 非 64 倍数的模型均可触发。

### 5.2 最小服务命令（单机，不需要 PD 分离）

```bash
# 单节点推理（无需 PD 分离，W4A4 MXFP4 K 非对齐即可触发）
vllm serve Qwen3.5-27B-vit-mxfp4 \
  --port 8010 \
  --tensor-parallel-size 8 \
  --enforce-eager --trust-remote-code \
  --quantization ascend
```

> 关键差异项：模型为 W4A4 MXFP4 且 K（hidden size / 分组后）**非 64 的倍数**。修复前推理直接报 `x1Scale K 68, x2Scale K 67`；修复后正常出结果。若手头模型 K 恰好对齐，可构造一个 K=68 的 W4A4 MXFP4 张量直测 `get_pergroup_param` 的分配长度。

---

> **核心教训**：同一「分块数」绝不能在权重侧与激活侧各算一遍还口径不一——量化尺度（scale）的每个维度必须用与算子约定一致的**统一 ceil 规则**（`cdiv`），并对非对齐尾部显式补块/padding；floor 丢块会在非对齐 K 上静默埋雷，直到算子报「scale K 不匹配」才暴露。