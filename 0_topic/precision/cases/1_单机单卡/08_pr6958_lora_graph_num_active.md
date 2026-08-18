# 案例 08：LoRA × 图模式下 dummy run 与图捕获的激活 LoRA 数不一致致精度错误

> **一句话定位**：上游 vLLM 改变 LoRA 处理（#32005）后，vllm-ascend `model_runner_v1.py` 的 `has_lora` 判断与图捕获时实际激活的 LoRA 数不一致，capture 的图与真实执行状态不匹配 → 输出错误。
>
> **对象**：vllm-project/vllm-ascend [PR #6958](https://github.com/vllm-project/vllm-ascend/pull/6958)（[bugfix][LoRA] Fix the lora accuracy issue introduced by the upstream vLLM changed，merged 2026-03）

---

## 1. 问题描述

### 1.1 现象

- `--enable-lora` + 图模式（AclGraph capture/dummy run）下 LoRA e2e 精度测试失败；
- 根因在 `_determine_batch_execution_and_padding`：`has_lora` 独立计算，与 dummy run / graph capture 时实际传入的 `num_active_loras` 不一致 → 捕获的图与真实执行的 LoRA 激活状态不匹配，输出错。

### 1.2 触发条件（必现矩阵）

| enable-lora | 图模式（AclGraph） | 是否触发 |
|:---:|:---:|:---:|
| ✗ | — | ✗ |
| ✓ | ✗（enforce-eager） | ✗ |
| **✓** | **✓** | **✓ 图与执行状态不匹配** |

### 1.3 影响与严重度

- **严重度**：🟡 中（LoRA 服务输出错误）。
- **隐蔽性**：中（LoRA e2e 测试可捕获；上游无变更时长期潜伏）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#6958](https://github.com/vllm-project/vllm-ascend/pull/6958) |
| 状态 | merged（2026-03，收录于 v0.16.0rc1 release checklist #6970） |
| vllm-ascend 版本 | main（v0.16.0rc1 周期） |
| 上游 vLLM | v0.16.0（main @15d76f7）；引入变更为上游 PR #32005 |

---

## 3. 定位过程

LoRA 单卡 e2e 测试精度失败 → 定位到上游 vllm#32005 引入的接口/语义变更波及 vllm-ascend 适配层的 `has_lora` 判定。

---

## 4. 解决方案

### 4.1 根因

图捕获的 dummy run 用一套 LoRA 激活数、真实执行用另一套（`has_lora` 独立推导），两套口径在上游语义变更后失配。

### 4.2 修复内容

`vllm_ascend/worker/model_runner_v1.py`（2 文件）：

- `_determine_batch_execution_and_padding` 新增 `force_num_active_loras` 参数；
- `has_lora` 改由 `num_active_loras` 推导（不再独立计算）；
- `_dummy_run` 去掉 `activate_lora` 参数，统一把 `num_active_loras` 传给 batch descriptor 与 `LoRAContext`，消除 dummy run 与 graph capture 的 LoRA 数不一致。

### 4.3 验证

`pytest -sv tests/e2e/singlecard/test_llama32_lora.py` 通过。

---

## 5. 复现方法

### 5.1 最小复现模型

- 仓库自带用例：**Llama-3.2 小模型 + LoRA adapter**（单卡）。

### 5.2 复现命令

```bash
pytest -sv tests/e2e/singlecard/test_llama32_lora.py
# 修复前（对应 v0.16.0rc1 周期 base）该测试精度失败，修复后通过
```

---

## 核心教训

平台适配层对上游语义变更不同步时，图捕获的 dummy run 与真实执行不一致会**静默产生精度错误**——图模式相关字段（LoRA 数、batch 形状）必须单一口径推导。
