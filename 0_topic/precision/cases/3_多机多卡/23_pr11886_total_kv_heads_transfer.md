# 案例23：PD 分离 + TP 不等时 Mooncake 传输组缺失全局 KV head 元数据

> **一句话定位**：序列化的 KV cache spec 只携带 rank-local 的 `num_kv_heads`，未携带模型级 `total_num_kv_heads`；当 TP 超过全局 head 数（head 被复制）或投机 draft 与 target head 数不同时，无法还原全局 head 归属，导致 target/draft 传输组误合并、rank pull 映射错位。
>
> **对象**：vllm-project/vllm-ascend [#11886](https://github.com/vllm-project/vllm-ascend/pull/11886)（BugFix，merged）

---

## 1. 问题描述

### 1.1 现象

在 **PD 分离 + P/D TP 不等**、且存在 **head 复制**（TP 度数 > 全局 KV head 数）或 **投机 draft 与 target 的 KV head 数不同**时：

- 序列化 spec 用 local `num_kv_heads` 无法反推全局 head（`local_heads * TP` 在 head 复制 / draft 差异下出错）；
- `_get_kv_transfer_spec_key` 只按 `(spec 类型, local head)` 分组 → **target 层与 draft 层被误合并进同一个 transfer group**；
- D 端 rank 的 pull 数计算错误 → **周期性传输失败 + KV head 归属错位 → 精度异常**。

真机失败模式：6 请求 gate 在 round 3、6 失败；20 轮全量回归在 round 3/6/9/12/15/18 周期性失败。

### 1.2 触发条件（必现矩阵）

| PD 分离 | TP(P)≠TP(D) | head 复制 / draft head 不同 | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗ | — | — | ✗ |
| ✓ | ✗（相等） | — | ✗（pulls=1，head 一一对齐，不需反推全局） |
| ✓ | ✓ | ✗（head 整除切分，local 能反推 total） | ✗（`local*TP == total`，无歧义） |
| **✓** | **✓** | **✓（head 复制 或 draft head ≠ target）** | **✓ 触发传输组误合并 + pull 错位** |

### 1.3 影响与严重度

- **严重度**：🟡 中（真机为周期性请求失败 + 精度异常）。
- **与 #8540 的区别**：本案例真机下会以传输失败暴露（可被 gate 抓到），而非 #8540 的纯 silent 精度下降。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#11886](https://github.com/vllm-project/vllm-ascend/pull/11886)（BugFix） |
| 状态 | closed / merged |
| vllm-ascend 版本 | v0.23.0 |
| 上游 vllm | `vllm-project/vllm@1f486d9` |
| 创建 / 合并 | 2026-07-12 / 2026-07-13 |
| 合并 commit | `3bfb521d6c1935681b875e6168acbe16c8901585` |
| 改动规模 | +188 / −55（2 个文件） |
| 影响文件 | `mooncake_connector.py`、`tests/ut/kv_offload/test_mooncake_connector.py` |
| 关联 issue | Fixes [#11885](https://github.com/vllm-project/vllm-ascend/issues/11885) |

真机验证配置：四节点 Ascend A2，Kimi K2.7 Code W4A8 + Kimi-K2.5-DFlash，P 端 DP2×TP8 / D 端 DP8×TP2，65542 token 串行请求。

---

## 3. 定位过程

1. **周期性失败模式**：真机请求在固定 round（3/6/9/12/15/18）失败 → 高度怀疑 group 拆分 / pull 映射错位。
2. **查日志 spec**：搜索 `FullAttentionSpec(local=..., total=...)`，发现 P/D 两端 local 相同（如都塌缩成 1）却无法区分 target/draft。
3. **对照 TP 相等基线**：把 D 端 TP 设成与 P 端相同，失败消失 → 锁定 TP 不等路径。
4. **定位根因**：D 端算 `num_need_pulls = num_d_block_heads // num_p_block_heads` 依赖全局 head，但 `_get_kv_transfer_spec_key` 只用 local head 分组，`_get_attention_group_num_key_value_heads` 也只读得到 local head → `local * TP` 在 head 复制 / draft 差异下无法还原全局。

> 定位要点：rank-local 标量无法反推模型级全局标量——head 在 TP 超过全局数时被复制，draft 的全局 head 数又可能与 target 不同，两者 local spec 看上去完全一样。

---

## 4. 解决方案

### 4.1 根因

序列化元数据缺「模型级全局 head 数」字段，接收端被迫用 `local_heads * TP` 反推，在 head 复制 / draft 差异场景下必然出错。

### 4.2 修复内容（三件事）

1. **序列化写入**：`_serialize_kv_group_spec` 新增 `total_num_kv_heads` 字段。
2. **按 owning model 取值**：新增 `_get_spec_total_num_kv_heads`，target 层取 `model_config.get_total_num_kv_heads()`，draft 层（`layer_idx >= total_layers` 且有 `draft_model_config`）取 draft 的；MLA 保持 local 语义。
3. **分组键升级**：`_get_kv_transfer_spec_key` 从 `(spec_type, local_heads)` 升级为 `(spec_type, local_heads, total_heads)`，target/draft 即使 local 都是 1 也因 total 16 vs 8 分到不同组。
4. **读取优先级**：`_get_attention_group_num_key_value_heads` 的查找顺序改为 `total_num_kv_heads` 优先，`num_kv_heads`/`num_key_value_heads` 兜底（向后兼容旧 metadata）。

关键 diff 片段：

```diff
- def _get_kv_transfer_spec_key(cls, spec: Any) -> tuple[str, int | None]:
-     return (type(spec).__name__, cls._get_spec_num_key_value_heads(spec))
+ def _get_kv_transfer_spec_key(cls, spec: Any, total_num_kv_heads: int | None,
+ ) -> tuple[str, int | None, int | None]:
+     return (type(spec).__name__, cls._get_spec_num_key_value_heads(spec), total_num_kv_heads)
```

### 4.3 验证

- 单测 95 tests 全过，覆盖 P local=1/D local=4 + total=8 的不等 TP、target total=16/draft total=8 的 group 拆分、group-aware rank pulls。
- 真机：修复后 6 请求 gate `6/6`、20 轮全量 `20/20` 通过，四节点日志零 Mooncake/HCCL 错误匹配。

## 5. 复现方法

### 5.1 最小复现模型

- **Qwen2.5-1.5B-Instruct**（`Qwen/Qwen2.5-1.5B-Instruct`）：GQA，`num_kv_heads=2`，体量最小。
- 触发要点：`num_kv_heads=2` 且 TP=8（> 2）→ head 被复制到多 rank，`local_heads × TP` 无法还原全局 head，落入「序列化 spec 缺全局 head」风险面；再配合 TP(P)≠TP(D) 触发 pull 错位。

### 5.2 最小服务命令（P 端 TP=8 / D 端 TP=2，触发 head 复制 + TP 不等）

```bash
# P 端（prefill，TP=8 > kv_heads=2 → head 复制）
export VLLM_USE_V1=1
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --port 8010 \
  --tensor-parallel-size 8 \
  --enforce-eager --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_port":"36000","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":2}}}'

# D 端（decode）
export VLLM_USE_V1=1
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --port 8020 \
  --tensor-parallel-size 2 \
  --enforce-eager --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_port":"36100","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":2}}}'
```

> 关键差异项：`--tensor-parallel-size` 8 vs 2（TP 不等）+ `num_kv_heads=2 < TP=8`（head 复制）。另一种触发方式（draft head ≠ target）需另配 draft 模型，head 复制这一路即可最小复现。

---

> **核心教训**：跨节点传输的序列化元数据必须**显式携带全局值**（total heads），不能让接收端用 `local × TP` 反推——这是「local ↔ global 口径不一致」的典型精度雷区。