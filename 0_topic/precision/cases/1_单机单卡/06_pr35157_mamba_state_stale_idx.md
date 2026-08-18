# 案例 06：prefix cache 重置后 linear/mamba 模型残留 stale mamba_state_idx 致越界

> **一句话定位**：`reset_prefix_cache` / KV flush 触发强制抢占时，请求进入 `resumed_req_ids` 但没有对应 `preempted_req_ids` 条目，旧的 `mamba_state_idx` 残留并可能指向重置后更小的块分配之外 → 状态读越界 / 读到错误状态。
>
> **对象**：vllm-project/vllm [PR #35157](https://github.com/vllm-project/vllm/pull/35157)（merged，2026-02-25，含新增单测）

---

## 1. 问题描述

### 1.1 现象

- linear attention（mamba 类）模型开 prefix caching，运行中调用 reset prefix cache / KV flush；
- 在飞请求被强制抢占后恢复时，携带**重置前的旧 `mamba_state_idx`**；
- 旧索引指向重置后（更小的）块分配之外 → 状态读越界或读到错误 mamba 状态，输出错误。

### 1.2 触发条件（必现矩阵）

| mamba/linear 模型 | prefix caching | 运行中 reset/flush | 有在飞请求被抢占 | 是否触发 |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | — | — | — | ✗（无 mamba 状态索引） |
| ✓ | ✓ | ✗ | — | ✗ |
| ✓ | ✓ | ✓ | ✗ | ✗ |
| **✓** | **✓** | **✓** | **✓** | **✓ 状态索引残留** |

### 1.3 影响与严重度

- **严重度**：🔴 高（越界读/错误状态，输出静默错误）。
- **隐蔽性**：高（需要「重置时机 × 在飞请求」组合，常规 CI 不覆盖）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm |
| 对象 | PR [#35157](https://github.com/vllm-project/vllm/pull/35157) |
| 状态 | merged（2026-02-25） |
| 复现版本 | 2026-02-25 修复前的 vllm main |
| 修复版本 | 含 #35157 的 main |
| vllm-ascend 版本 | 未知（调度层逻辑，NPU 上同样可触发；vllm-ascend hybrid/mamba 模型适用） |

---

## 3. 定位过程

1. 外部 bug 报告：reset prefix cache 后 mamba 模型输出异常/越界；
2. 分析 `resumed_req_ids` 与 `preempted_req_ids` 的**集合差集**：恢复请求缺少对应抢占条目；
3. 确认恢复路径未清理 `mamba_state_idx`，旧索引在重置后的新（更小）分配下越界。

---

## 4. 解决方案

### 4.1 根因

「缓存重置」没有同步失效与每请求绑定的状态索引：reset/flush 强制抢占产生的 resumed 请求绕过了正常抢占路径的状态清理。

### 4.2 修复内容

为被恢复（resumed）的请求清理 `mamba_state_idx`，防止陈旧索引指向重置后更小的分配之外；新增单测覆盖「prefix caching + 运行中 reset + 在飞请求」场景。

### 4.3 验证

新增单测通过（构造该组合场景，断言状态索引被清理、无越界）。

---

## 5. 复现方法

### 5.1 最小复现模型

- **`LFM2-1.2B` / `Falcon-H1-1.5B`**（linear/mamba 混合架构最小公开模型，单卡）；dense 模型无替代（触发条件本身要求 mamba 态）。

### 5.2 复现命令

```python
from vllm import LLM, SamplingParams
llm = LLM(model="liquid/LFM2-1.2B", enable_prefix_caching=True)
# 1) 先发若干请求制造在飞/缓存状态 2) 压测中调用 reset_prefix_cache 3) 观察被抢占恢复请求的输出/索引
```

> PR 以单测形式覆盖该场景，可直接引用其测试思路。

---

## 核心教训

任何「缓存重置」操作都必须同步失效与之绑定的每请求状态索引；`resumed/preempted` 集合差集是这类残留 bug 的定位钥匙。
