# 案例27：Mooncake 传输元数据缺失 cache group id 致混合模型 KV 错配

> **一句话定位**：hybrid 模型（attention + mamba/compress）的一个 cache group 被 split 成多个 transfer group 时，send 侧用 transfer group 的循环下标 `append` 构造 block id 元数据，而 receive 侧按 `kv_cache_group_id` 索引，两端语义错位 → KV 数据写入错误 cache。
>
> **对象**：vllm-project/vllm-ascend [#11601](https://github.com/vllm-project/vllm-ascend/pull/11601)（BugFix，merged）

---

## 1. 问题描述

### 1.1 现象

在 **PD 分离 + 无 CP + hybrid 模型**（GLM5 / DeepSeek-V4 等 attention+compress+mamba），且 **一个 cache group 被 split 成 ≥2 个 transfer group** 时：

- `_get_kv_split_metadata` 用 `append` + 循环下标 `group_idx` 组装 `local/remote_block_ids`，把 transfer group 顺序当成了 cache group 索引；
- receive 端 `_transfer_kv_cache_all_groups` 按正确的 `kv_cache_group_id` 去读这个「错位」列表；
- 结果 **attention 的 KV 数据被写进 mamba/compress 的 cache 槽位**（attention ↔ mamba 互换），mamba 真实 ids 丢失。

两个破坏路径：
1. **silent 精度腐败**（列表恰好够长时）：错位 block 被喂给 mamba 层 RDMA 拷贝，数据落位错误 → 无断言、输出腐败。
2. **崩溃**（`group_idx` 越界时）：`IndexError`，或下游 `split_if_not_byte_contiguous` 段错误（与 #7792 的 503900 同源）。

### 1.2 触发条件（必现矩阵）

| hybrid 模型 | cache group 内 split | 无 CP | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗（纯 attention） | — | ✓ | ✗（transfer==cache group，id 恒等） |
| ✓ | ✗（每 cache group 单一 spec） | ✓ | ✗（无 split） |
| **✓** | **✓（cache group 内多 spec）** | **✓** | **✓ 触发错配** |
| ✓ | ✓ | ✗（有 CP） | 走 `get_cp_group_meta` 另一路径，非本 bug |

### 1.3 影响与严重度

- **严重度**：🟡 中（silent 精度腐败）/ 🔴 高（崩溃路径）。
- **隐蔽性**：纯 attention 模型永不触发；hybrid 但未 split 也不触发。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#11601](https://github.com/vllm-project/vllm-ascend/pull/11601)（BugFix） |
| 状态 | closed / merged |
| vllm-ascend 版本 | v0.23.0（backport 自 main [#11439](https://github.com/vllm-project/vllm-ascend/pull/11439)，落到 `releases/v0.23.0`） |
| 上游 vllm | [`vllm-project/vllm@ee0da84`](https://github.com/vllm-project/vllm/commit/ee0da84ab9e04ac7610e28580af62c365e898389) |
| 创建 / 合并 | 2026-07-08（同日创建并合并） |
| 合并 commit | `ae870bc5e265a340912cde392f23dad3671a0a88` |
| 改动规模 | +89 / −7（含 80 行测试） |
| 影响文件 | `mooncake_connector.py`、`tests/ut/kv_offload/test_mooncake_connector.py` |

---

## 3. 定位过程

1. **确认模型是否 hybrid + split**：dump `worker.kv_group2layeridx`，看是否存在 ≥2 个 transfer group 共享同一 `kv_cache_group_id`。
2. **对比两类 group 数量**：若 `len(kv_group2layeridx) > len(kv_cache_config.kv_cache_groups)`，即存在 split，落入风险面。
3. **校验元数据长度**：在 `_get_kv_split_metadata` 后断言 `len(local_block_ids) == len(meta.local_block_ids)`（修复前会不相等或错位）。
4. **发现根因**：`meta.local_block_ids` 下标语义是 cache group id，但 `_get_kernel_block_ids` / `_get_kv_split_metadata` 直接用 transfer group 循环下标 `group_idx` 索引并 `append`，两套 id 被同一个循环变量隐式兼任。

> 定位要点：receive 端 `_transfer_kv_cache_all_groups` 一直正确用 `kv_cache_group_id` 索引——bug 只在 send 侧元数据构造，两端对同一数据结构的索引语义不一致。

---

## 4. 解决方案

### 4.1 根因

**transfer group id 与 cache group id 被同一个循环变量 `group_idx` 隐式兼任**。一个 cache group 可对应多个 transfer group，但 block id 列表的下标语义是 cache group id，不可用 transfer group 顺序顶替。

### 4.2 修复内容

1. 新增 `_get_kv_cache_group_id(group_idx, group_spec) = group_spec.get("kv_cache_group_id", group_idx)` 显式取 cache group id（缺省回退保兼容）。
2. `_get_kernel_block_ids` 的 Mamba/Attention 分支改用 `kv_cache_group_id` 索引 `meta.local_block_ids`。
3. `_get_kv_split_metadata` 改为**按 cache group 数预分配 + 按 id 赋值**（取代 `append`）：

```diff
- local_block_ids: list = []
- remote_block_ids: list = []
+ local_block_ids: list[list[int]] = [[] for _ in meta.local_block_ids]
+ remote_block_ids: list[list[int]] = [[] for _ in meta.remote_block_ids]
   for group_idx, (group_spec, layer_indices) in self.kv_group2layeridx.items():
       ...
-      local_block_ids.append(local_kernel_block_ids)
-      remote_block_ids.append(remote_kernel_block_ids)
+      kv_cache_group_id = self._get_kv_cache_group_id(group_idx, group_spec)
+      local_block_ids[kv_cache_group_id] = local_kernel_block_ids
+      remote_block_ids[kv_cache_group_id] = remote_kernel_block_ids
```

> 覆盖语义说明：同一 cache group 的多个 transfer group 共享同一组 block ids，故「后写覆盖前写」是正确的。

### 4.3 验证

- 新增单测 `test_hybrid_no_cp_uses_kv_cache_group_ids_for_split_transfer_groups`：3 个 transfer group / 2 个 cache group，断言输出按 cache group 分组（长度 2 而非 3）。

## 5. 复现方法

### 5.1 最小复现模型

- **Qwen3-Next-80B-A3B**（`Qwen/Qwen3-Next-80B-A3B-Instruct`，以官方权重为准）：hybrid（Gated DeltaNet + Attention 混合）MoE，是「attention + linear」混合架构中**相对较小**的可复现选择；真机参考 GLM5 / DeepSeek-V4。
- 触发要点：hybrid 模型一个 cache group 内含多个 spec（attention + mamba/compress），无 CP 时 `_get_kv_split_metadata` 把 transfer group 顺序错当 cache group id → 触发错配。

> 说明：hybrid 架构目前**不存在小模型**，80B 已是最小边界；本案例的「cache group 内 split」依赖目标模型的 group/spec 划分，复现前先用 `worker.kv_group2layeridx` 确认存在「>1 个 transfer group 共享同一 `kv_cache_group_id`」。

### 5.2 最小服务命令（P/D 两端同 TP=8，无 CP，触发 cache group split）

```bash
# P 端（prefill）
export VLLM_USE_V1=1
vllm serve <HYBRID_MODEL> \
  --port 8010 \
  --tensor-parallel-size 8 \
  --enforce-eager --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_port":"36000","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":8}}}'

# D 端（decode）
export VLLM_USE_V1=1
vllm serve <HYBRID_MODEL> \
  --port 8020 \
  --tensor-parallel-size 8 \
  --enforce-eager --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_port":"36100","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":8}}}'
```

> 关键差异项：hybrid 模型 + **无 CP**（不传 CP 相关 config）+ 一个 cache group 内多 spec（split）。两端 TP 相等即可，本案例与 TP 无关。

---

> **核心教训**：凡存在「逻辑分组」与「传输/调度分组」两层 1:N 概念时，跨边界传递索引必须显式携带真正的 id，不能依赖「循环顺序恰好等于 id」——元数据要自描述，不可依赖隐式对齐。