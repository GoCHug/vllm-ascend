# 案例 14：PCP 持久化 buffer 跨调用未重置，第二个请求起读到脏 slot mapping

> **一句话定位**：PCP（parallel chunked prefill）模式下 `pcp_padded_slot_mapping` buffer 跨多次调用复用但未初始化，保留上一次运行的残留值 → 第二次以后的 prefill 计算结果错误（state leakage）。
>
> **对象**：vllm-project/vllm-ascend [PR #5647](https://github.com/vllm-project/vllm-ascend/pull/5647)（[bugfix (pcp)] fix chunked prefill accurancy issue，merged 2026-01-07）

---

## 1. 问题描述

### 1.1 现象

- PCP 模式（`pcp_world_size>1` 多卡）下，同一服务内连续发送多条长 prompt 分块预填充请求；
- `pcp_padded_slot_mapping` 是跨调用复用的持久化 buffer，未在入口重置 → **第二次以后的调用读到上一次的脏 slot mapping**，KV 写错位置，计算结果错误；
- 第一条请求正常——单请求 CI 不触发。

### 1.2 触发条件（必现矩阵）

| PCP（多卡） | 同服务连续多次 prefill | 是否触发 |
|:---:|:---:|:---:|
| ✗ | — | ✗ |
| ✓ | 1 次 | ✗（buffer 首用无残留） |
| **✓** | **≥2 次** | **✓ 脏 slot mapping、输出错乱** |

### 1.3 影响与严重度

- **严重度**：🔴 高（第二个请求起输出错乱）。
- **隐蔽性**：高（单请求冒烟测试完全掩盖）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#5647](https://github.com/vllm-project/vllm-ascend/pull/5647)（bugfix） |
| 状态 | merged（2026-01-07） |
| vllm-ascend 版本 | main（2026-01-07，v0.13.0 周期） |
| 上游 vLLM | v0.13.0（main @2f4e654） |

---

## 3. 定位过程

无关联 issue；由代码评审发现 state leakage——持久化 buffer 无重置逻辑。评审同时指出 `update_tokens_for_pcp` 中 `pcp_unpad_mask_cpu_tensor` 存在同类未重置问题。

---

## 4. 解决方案

### 4.1 根因

跨调用复用的持久化 buffer（padded slot mapping / unpad mask）没有「每次入口显式重置」的约定，上一轮残留值被本轮当作有效数据消费。

### 4.2 修复内容

```python
def get_padded_slot_mapping(...):
    pcp_padded_slot_mapping.fill_(-1)   # 每次调用先清空
    ...
# 评审建议 update_tokens_for_pcp 的 pcp_unpad_mask_cpu_tensor 入口处 zero_()，同类处理
```

### 4.3 验证

多卡 PCP 连续请求场景输出正确（随 e2e 回归覆盖）。

---

## 5. 复现方法

### 5.1 最小复现模型

- 模型不限（可小模型），需 PCP 多卡（`pcp_world_size>1`）。

### 5.2 复现命令

```bash
vllm serve <SMALL-MODEL> --tensor-parallel-size 2 --enable-chunked-prefill \
  --compilation-config '{"pcp_world_size":2}'  # 以仓库当前 PCP 开启方式为准
# 同一服务连续发送 ≥2 条长 prompt（分块预填充），对比单条请求时的输出
# 未修复版本：第 2 条起输出错乱
```

---

## 核心教训

跨调用复用的持久化 buffer 必须每次显式重置（`fill_(-1)` / `zero_()`）——「第二个请求开始」静默读到脏数据是 state leakage 类 bug 的统一画像。
