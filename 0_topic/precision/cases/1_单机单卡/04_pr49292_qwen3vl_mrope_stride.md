# 案例 04：Qwen3-VL M-RoPE 位置张量 stride 视图错误致位置索引越界

> **一句话定位**：transformers 建模后端下 Qwen3-VL 的 M-RoPE 位置张量，`[3,N]→[3,1,N]` 的 unsqueeze 视图携带旧 stride，torch.compile 动态 shape 推导把 seq 长度解析成 `8192+1`，位置索引溢出边界 → 位置编码错误/编译失败。
>
> **对象**：vllm-project/vllm [PR #49292](https://github.com/vllm-project/vllm/pull/49292)（merged，改 `transformers/multimodal.py`）

---

## 1. 问题描述

### 1.1 现象

transformers backend 加载 Qwen3-VL 时两条路径分别出错：

- **eager 模式**：把空 `video_grid_thw` 张量传给 transformers（其检查 None 语义），提示即失败；
- **compile 模式**：M-RoPE buffer 的非连续 stride 与动态 shape 推导交互，把 seq 长度解析为 `8192+1`，位置索引越过真实边界 → 位置编码错误或编译失败。

### 1.2 触发条件（必现矩阵）

| 模型 | backend=transformers | 是否触发 |
|:---:|:---:|:---:|
| dense LLM | ✓ | ✗ |
| Qwen3-VL（M-RoPE） | ✗（vllm 原生） | ✗ |
| **Qwen3-VL（M-RoPE）** | **✓** | **✓ eager 报错 / compile 位置越界** |

### 1.3 影响与严重度

- **严重度**：🟡 中（限定 transformers backend + 多模态 M-RoPE 组合）。
- **隐蔽性**：中（eager 快速失败；compile 是静默位置错误）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm |
| 对象 | PR [#49292](https://github.com/vllm-project/vllm/pull/49292) |
| 状态 | merged（2026-07-21） |
| 复现版本 | 2026-07-21 修复前的 vllm main |
| 修复版本 | 含 #49292 的 main |
| vllm-ascend 版本 | 未知（transformers backend 可在 NPU 上使用，eager 路径跨硬件一致） |

---

## 3. 定位过程

1. eager/compile 两条路径分别二分；
2. eager 定位到「空张量 vs None」的语义差（transformers 期望 None）；
3. compile 定位到 mrope buffer 非连续 stride 与动态 shape 推导的交互——`[3,N]→[3,1,N]` unsqueeze 视图的 stride 元信息泄入 shape 推导。

---

## 4. 解决方案

### 4.1 根因

- 视频/图像 `grid_thw` 为空时传了空张量而非 None；
- unsqueeze 产生的视图「逻辑正确」但携带旧 stride，torch.compile 动态 shape 推导据 stride 解析出越界 seq 长度。

### 4.2 修复内容

- 空 thw 输入改传 `None`；
- M-RoPE 位置张量通过调整 stride 信息（后续提交以改 stride 替代 contiguous 物化），确保动态 shape 推导不超过真实 seq len。

### 4.3 验证

Qwen3-VL transformers backend eager/compile 两模式下多模态请求正常、位置索引不越界。

---

## 5. 复现方法

### 5.1 最小复现模型

- **`Qwen/Qwen3-VL-4B-Instruct-FP8`**（<10B，单卡）。

### 5.2 复现命令

```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct-FP8 --model-impl transformers
# 修复前：eager 直接报错；去掉 --enforce-eager（compile）则位置索引越界
# 发任意图文/视频请求观察
```

---

## 核心教训

视图操作的 stride 元信息会泄入 torch.compile 的动态 shape 推导——「逻辑正确」的张量视图也可能产生位置越界；空输入要遵守下游库的 None 语义而非空张量。
