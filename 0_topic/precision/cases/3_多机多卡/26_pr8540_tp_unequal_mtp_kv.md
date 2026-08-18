# 案例26：PD 分离 + TP 不等时 MTP 层 KV Cache 未处理（silent 精度下降）

> **一句话定位**：P(D) 分离 + P/D 端 TP 不等 + 开启 MTP 三者同时满足时，reformat 融合算子的「层数」参数少算了 MTP 层，导致 MTP 层 KV Cache 未被按 TP 重排，D 端读到错位 KV，draft 质量下降、投机接受率降低。
>
> **对象**：vllm-project/vllm-ascend [#8540](https://github.com/vllm-project/vllm-ascend/pull/8540)（BugFix，merged）

---

## 1. 问题描述

### 1.1 现象

在**预填充/解码分离（PD 分离）**且 **P 端与 D 端 Tensor Parallel 度数不等**、同时**开启 MTP 投机解码**的部署下：

- P 端计算好的 KV Cache 通过 Mooncake Connector 跨节点传输到 D 端时，**MTP 层的 KV Cache 未被按 TP 重排**；
- D 端 MTP 层读到**头归属错位**的 KV，draft token 质量下降、投机接受率降低；
- 表现是 **silent 精度下降**——不报错、无断言，但端到端输出质量退化。

### 1.2 触发条件（必现矩阵）

| PD 分离 | TP(P)≠TP(D) | 开启 MTP | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗ | — | — | ✗（无跨节点传输） |
| ✓ | ✗（相等） | ✓ | ✗（走逐层 hybrid_linear 路径，MTP 层被自然覆盖） |
| ✓ | ✓ | ✗ | ✗（`kv_caches` 长度 = `num_hidden_layers`，层数不少算） |
| **✓** | **✓** | **✓** | **✓ 触发精度异常** |

> 三个条件缺一不可——这正是它隐蔽性高、CI 默认用例难以覆盖的原因。

### 1.3 影响与严重度

- **严重度**：🔴 高（端到端精度异常，silent）。
- **不是单纯的吞吐问题**：draft 候选质量变差 → 接受率下降 → 吞吐退化，且累积扰动影响最终输出。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#8540](https://github.com/vllm-project/vllm-ascend/pull/8540)（BugFix） |
| 状态 | closed / merged |
| vllm-ascend 版本 | v0.19.0 |
| 上游 vllm | `vllm-project/vllm@6f786f2` |
| 创建 / 合并 | 2026-04-21 / 2026-04-23 |
| 合并 commit | `5dba75bade4a0d5b36c26bb4611a9f545ceb821c` |
| 改动规模 | +1 / −1（单行修改） |
| 影响文件 | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` |

---

## 3. 定位过程

1. **复现三要素**：确认部署是否 PD 分离、P/D 端 TP 是否不等、是否开启 MTP。
2. **TP 相等对照**：把 D 端 TP 设成与 P 端相同，若精度恢复正常 → 高度怀疑 reformat/重排路径。
3. **关闭 MTP 对照**：关 MTP 后精度正常 → 锁定 MTP 层的 KV 处理。
4. **定位代码路径**：`_transfer_kv_cache_all_groups` 只在 `tp_ratio != 1`（TP 不等）时走 `reformat_kv_cache_with_fused_op`，进一步确认 bug 只在该路径激活。
5. **发现根因**：`reformat_kv_cache_with_fused_op` 内用 `layers = num_hidden_layers`（= N），但实际收集的 `k_caches/v_caches` 列表来自 `kv_caches.items()`，**包含 MTP 层（N+1）**。融合算子以 `layers=N` 为准只处理前 N 层，**最后一层（MTP 层）被静默跳过**。

> 定位要点：上层 `_get_group_kv_caches` 已正确把 MTP 层放进待传输字典，但下层 fuse-op 又用 config 标量重算层数——「同一概念在两处分别计算、口径不同」。

---

## 4. 解决方案

### 4.1 根因

「待传输层数」上层按真实字典筛选、下层却按模型 config 的 `num_hidden_layers` 重算，两处口径分叉。MTP 追加层使 `kv_caches` 长度（N+1）大于 `num_hidden_layers`（N）。

### 4.2 修复内容

```diff
- layers = self.model_config.hf_text_config.num_hidden_layers
+ layers = len(self.kv_caches)
```

核心思想：`layers` 应取**融合算子实际要处理的 KV Cache 层数**（即容器长度），而非 config 标量。

> 演进记录：初版硬编码 `if speculative_config: layers += 1` 被 review 否定（①条件过宽，非 MTP 投机如 EAGLE 不追加主模型 KV 层；②不兼容 PP 分片），改为一劳永逸的 `len(self.kv_caches)`。

### 4.3 验证

- 由 nightly 覆盖；触发矩阵中「✓✓✓」场景精度恢复正常。
- 建议补充回归：PD 分离 + TP=8/TP=4 + MTP 的 e2e accuracy 用例，单测断言 `layers == len(kv_caches)`。

## 5. 复现方法

### 5.1 最小复现模型

- **`Eco-Tech/Qwen3.5-27B-w8a8-mtp`**：仓库 CI 中最小的 MTP 模型（w8a8 量化，单节点 A3 可跑），speculative method 为 `qwen3_5_mtp`。
- 触发要点：MTP 使 `kv_caches` 长度 = `num_hidden_layers + 1`，配合 TP(P)≠TP(D) 走 `reformat_kv_cache_with_fused_op` 路径，暴露「层数少算」bug。

### 5.2 最小服务命令（P 端 TP=8 / D 端 TP=4，均开 MTP）

```bash
# P 端（prefill）
export VLLM_USE_V1=1
vllm serve Eco-Tech/Qwen3.5-27B-w8a8-mtp \
  --port 8010 \
  --tensor-parallel-size 8 \
  --enforce-eager --trust-remote-code \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_port":"36000","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":4}}}'

# D 端（decode）
export VLLM_USE_V1=1
vllm serve Eco-Tech/Qwen3.5-27B-w8a8-mtp \
  --port 8020 \
  --tensor-parallel-size 4 \
  --enforce-eager --trust-remote-code \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_port":"36100","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":4}}}'
```

> 关键差异项：`--tensor-parallel-size` 8 vs 4（TP 不等）+ `--speculative-config`（MTP）。去掉任一即不触发。CI 原场景见 `tests/e2e/weekly/multi_node/external_dp/config/DeepSeek_V3.1T_MTP1_PD.yaml`（P TP8 / D TP1 + `deepseek_mtp`）。

---

> **核心教训**：传给底层算子的 `layers/num_heads/dim` 等标量，应取自**实际要操作的数据结构长度**，而非从 config 重新推导——MTP、PP、量化、hybrid 场景下两者口径频繁不一致。