# PR #11886 深度案例：PD 分离 TP 不等时 Mooncake 传输组缺失全局 KV head 元数据

> 整理时间: 2026-07-29
>
> 案例对象: [vllm-project/vllm-ascend#11886](https://github.com/vllm-project/vllm-ascend/pull/11886)
>
> 标题: [BugFix][PD] Carry explicit total KV heads in Mooncake transfer groups
>
> 关联全景文档: [0_kvcache.md](./0_kvcache.md) §2.3 / 案例 6
>
> 关键词: PD 分离 · TP 不等 · Mooncake · KV head 归属 · 传输元数据

---

## 1. PR 概览

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| PR 编号 | [#11886](https://github.com/vllm-project/vllm-ascend/pull/11886) |
| 类型 | BugFix |
| 作者 | [@QwertyJack](https://github.com/QwertyJack) |
| Reviewer | @wangxiyuan, @LCAIZJ, @MengqingCao（requested） |
| 状态 | closed / **merged** |
| 创建时间 | 2026-07-12 |
| 合并时间 | 2026-07-13 |
| 合并者 | [@linfeng-yuan](https://github.com/linfeng-yuan) |
| 合并 commit | `3bfb521d6c1935681b875e6168acbe16c8901585` |
| 改动规模 | **+188 / −55**（2 个文件） |
| 影响文件 | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`、`tests/ut/kv_offload/test_mooncake_connector.py` |
| vLLM 版本 | v0.23.0 |
| vLLM main | `1f486d96a17303ce8db8e02be39545b2be338446` |
| Fixes | [#11885](https://github.com/vllm-project/vllm-ascend/issues/11885) |
| Related | #10991, #11877 |
| 测试方式 | 单元测试（95 tests）+ 四节点 A2 真机回归（Kimi K2.7 Code W4A8 + Kimi-K2.5-DFlash） |

> 典型的“元数据缺字段”类 bug：序列化的 KV cache spec 只携带了 rank-local 的 `num_kv_heads`，却没有携带模型级 `total_num_kv_heads`，导致 P/D 端 TP 不等时无法还原全局 head 归属，传输组错误合并、rank pull 映射错位。

---

## 2. 场景背景

### 2.1 PD 分离部署 (Prefill/Decode Disaggregation)

在 PD 分离架构中，Prefill 节点（P 端）负责长 prompt 一次性计算，Decode 节点（D 端）逐 token 自回归生成。P 端计算好的 **KV Cache** 通过 **Mooncake Connector**（基于 RDMA）跨节点传输到 D 端，避免 D 端重复计算 prefix。

### 2.2 TP 不等 (TP unbalanced)

为最大化吞吐与资源利用，P 端和 D 端常采用**不同的 Tensor Parallel 度数**。本 PR 真机验证用的配置即为极端比例：

- P 端 `DP2 × TP8`（prefill 计算密集）
- D 端 `DP8 × TP2`（decode 计算稀疏）

TP 不等意味着 P 端每个 rank 持有的 KV head 数量与 D 端不同，传输 KV Cache 时**不能直接字节拷贝**，必须按 head 重新分片/重排。参考 [1_pr8540](./1_pr8540_tp_unequal_mtp_kv.md) §2.2 与 [0_kvcache.md](./0_kvcache.md) 案例 6 的 TP 分片示例：

```
全局 8 个 KV head
TP=4 (P 端) 时：rank0: head 0,1  rank1: head 2,3  rank2: head 4,5  rank3: head 6,7
TP=8 (D 端) 时：rank0: head 0  rank1: head 1  ... rank7: head 7

若直接把 P 端 rank0 的数据（head 0,1）传给 D 端 rank0（只应有 head 0）
  → head 归属错误 → 精度异常
```

### 2.3 投机解码下 target / draft 的 KV head 差异

本 PR 触发场景比 #8540 更进一步：当模型启用**投机解码**（EAGLE / MTP 等）时，主模型（target）与 draft 模型可能有**不同的全局 KV head 数量**。例如：

- target 模型：`total_num_kv_heads = 16`
- draft 模型：`total_num_kv_heads = 8`

而在高 TP 下，两者**每个 rank 的 local head 数都可能塌缩成 1**（head 被复制而非切分）。此时仅凭 local `num_kv_heads` 已无法区分 target 层与 draft 层——它们的 local spec 看起来完全一样。

三者组合（PD 分离 × TP 不等 × target/draft KV head 不同）即为本 PR 修复的触发场景。

---

## 3. 根因分析

### 3.1 Bug 代码：序列化 spec 只有 local head

修复前，`_serialize_kv_group_spec` 序列化出的 `kv_cache_spec` 字典只携带 rank-local 的 `num_kv_heads` / `num_key_value_heads`，**没有全局 head 数**：

```python
# vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py（修复前）
@classmethod
def _get_kv_transfer_spec_key(cls, spec: Any) -> tuple[str, int | None]:
    # 传输组的分过键只看 (spec 类型名, local head 数)
    return (type(spec).__name__, cls._get_spec_num_key_value_heads(spec))
```

`_get_spec_num_key_value_heads` 取自 `FullAttentionSpec.num_kv_heads`——这是 **TP 分片后该 rank 实际持有的物理 head 数**，不是模型级全局数。

### 3.2 不一致点：local head 无法还原全局 head

PR 描述点明了根因（原文）：

> `num_kv_heads` in a serialized KV cache spec describes the rank-local cache layout. It cannot reliably recover the model-level head count with `local_heads * TP`: heads may be replicated when TP exceeds the global KV-head count, and a speculative group may belong to a draft model whose KV-head count differs from the target model.

也就是说，D 端想从收到的 spec 反推全局 head 数时，`local_heads * TP` 在两种情况下会错：

1. **TP 超过全局 KV head 数时，head 被复制（replicated）**。例如全局 8 head、TP=16，则每个 rank 持有 `max(1, 8//16)=1` 个 head（复制），`1*16=16 ≠ 8`。本 PR 真机配置（P 端 `local=1, total=8`）正是此情形。
2. **投机组归属 draft 模型时，其全局 head 数与 target 不同**。target 16 head / draft 8 head 在高 TP 下 local 都可能是 1，两者 spec 看上去一样。

### 3.3 后果：传输组错误合并 + rank pull 映射错位

因为 `_get_kv_transfer_spec_key` 只按 `(spec 类型, local head)` 分组：

- **target 层和 draft 层被合并进同一个 Mooncake transfer group**（两者都是 `FullAttentionSpec` + `local=1`），而它们本应有不同的 pull 映射。
- `_get_attention_group_num_key_value_heads` 在 D 端读 spec 时只拿到 local head，用它去算 `num_need_pulls = num_d_block_heads // num_p_block_heads`（见 `mooncake_connector.py:3210-3214`），得到**错误的 pull 数**。
- 于是 D 端某些 rank 向错误的 P 端 rank 拉 KV，或拉的数量不对 → **KV head 归属错位** → attention 用错 K/V → 精度异常/转输失败。

这正是 #11885 报告的问题：在 P 端 `local=1/total=8`、D 端 `local=4/total=8` 的不等 TP 组合下，rounds 3/6/9/12/15/18 等周期性失败。

### 3.4 为什么 TP 相等时不显现

TP 相等时，`tp_num_need_pulls = 1`，P 端 rank 与 D 端 rank 一一对应、head 分片完全对齐，不需要“反推全局 head”来计算 pull 映射——所以 local head 够用，bug 不显现。只有 TP 不等（需按 head 比例重排、计算多 pull）时，缺失的全局 head 才成为必需。这与 #8540 的“TP 不等才触发 reformat 路径”同构（见 [1_pr8540](./1_pr8540_tp_unequal_mtp_kv.md) §3.4）。

**触发条件 = PD 分离 + TP(P) ≠ TP(D) +（head 复制 或 target/draft head 不同）**。

---

## 4. 影响与表现

| 维度 | 表现 |
|------|------|
| 触发配置 | PD 分离 + P/D 端 TP 不等 +（TP 超过全局 KV head 致复制，或投机 draft 与 target head 数不同） |
| 不触发 | TP 相等 / 无投机 / head 未被复制 |
| 现象 | Mooncake 传输组错误合并 target/draft 层；D 端 rank pull 数计算错误；周期性传输失败（真机见 rounds 3/6/9/12/15/18 失败）；KV head 归属错位致精度异常 |
| 严重度 | 🟡 中（真机表现为周期性请求失败 + 精度异常，但可被 gate 抓到） |
| 隐蔽性 | 🟡 中高（需 TP 不等 + head 复制/draft 差异同时满足；TP 相等时不显现，CI 单测易漏） |
| 是否报错 | 真机下会以传输失败/请求失败暴露（不同于 #8540 的纯 silent 精度下降）；单测层面无断言失败 |
| 真机复现 | 四节点 A2，Kimi K2.7 Code W4A8 + Kimi-K2.5-DFlash，P=DP2×TP8 / D=DP8×TP2，65542 token 串行请求 |

真机失败模式：6 请求 gate 在 round 3、6 失败（`6/6` 修复后才通过）；20 轮全量回归在 round 3/6/9/12/15/18 失败（修复后 `20/20`）。日志中可观测到 Mooncake 传输/HCCL batch-get 相关错误（修复后四节点日志零匹配）。

---

## 5. 修复方案

### 5.1 核心思想：在传输元数据显式携带 total KV heads

给每个序列化的 attention KV spec 新增 `total_num_kv_heads` 字段，并把它纳入 transfer group 的分组键。具体三件事：

1. **序列化时写入**：`_serialize_kv_group_spec` 新增 `total_num_kv_heads` 参数，写入 `serialized_kv_cache_spec["total_num_kv_heads"]`。
2. **按 owning model config 取值**：新增 `_get_spec_total_num_kv_heads`，target 层取 `model_config.get_total_num_kv_heads()`，draft 层（`layer_idx >= total_layers` 且有 `draft_model_config`）取 `draft_model_config.get_total_num_kv_heads()`；MLA 保持 local 语义。
3. **分组键升级**：`_get_kv_transfer_spec_key` 从 `(spec_type, local_heads)` 升级为 `(spec_type, local_heads, total_heads)`，target/draft 即使 local 都是 1 也因 total 16 vs 8 分到不同 transfer group。
4. **读取优先级**：`_get_attention_group_num_key_value_heads` 的 key 查找顺序改为 `("total_num_kv_heads", "num_kv_heads", "num_key_value_heads")`，优先用全局数算 pull；无新字段时回退（向后兼容旧 metadata）。

### 5.2 关键 diff（mooncake_connector.py）

```diff
@@ _serialize_kv_group_spec
         kv_cache_group_id: int | None = None,
+        total_num_kv_heads: int | None = None,
     ) -> dict[str, Any]:
         ...
         if num_key_value_heads is not None:
             serialized_kv_cache_spec["num_kv_heads"] = num_key_value_heads
             serialized_kv_cache_spec["num_key_value_heads"] = num_key_value_heads
+        if total_num_kv_heads is not None:
+            serialized_kv_cache_spec["total_num_kv_heads"] = total_num_kv_heads

@@ _get_kv_transfer_spec_key
-    def _get_kv_transfer_spec_key(cls, spec: Any) -> tuple[str, int | None]:
-        return (type(spec).__name__, cls._get_spec_num_key_value_heads(spec))
+    def _get_kv_transfer_spec_key(
+        cls, spec: Any, total_num_kv_heads: int | None,
+    ) -> tuple[str, int | None, int | None]:
+        return (
+            type(spec).__name__,
+            cls._get_spec_num_key_value_heads(spec),
+            total_num_kv_heads,
+        )
+
+    def _get_spec_total_num_kv_heads(self, spec: Any, layer_idx: int) -> int | None:
+        local_num_kv_heads = self._get_spec_num_key_value_heads(spec)
+        if local_num_kv_heads is None or isinstance(spec, MLAAttentionSpec):
+            return local_num_kv_heads
+        model_config = self.vllm_config.model_config
+        speculative_config = self.vllm_config.speculative_config
+        if (
+            layer_idx >= self.total_layers
+            and speculative_config is not None
+            and speculative_config.draft_model_config is not None
+        ):
+            model_config = speculative_config.draft_model_config
+        return model_config.get_total_num_kv_heads()

@@ _build_kv_group2layeridx （统一 UniformTypeKVCacheSpecs 与非 uniform 两条路径）
-            if isinstance(group_spec.kv_cache_spec, UniformTypeKVCacheSpecs):
-                spec_groups ... = OrderedDict()
-                for layer_name, layer_idx in layer_entries:
-                    layer_spec = group_spec.kv_cache_spec.kv_cache_specs[layer_name]
-                    spec_key = self._get_kv_transfer_spec_key(layer_spec)
-                    ...
-                for entries in spec_groups.values():
-                    ...
-                continue
-
-            layer_names = [layer_name for layer_name, _ in layer_entries]
-            ...（非 uniform 直接整组一个 transfer group，不细分）
+            spec_groups: OrderedDict[
+                tuple[str, int | None, int | None],
+                list[tuple[str, int, Any, int | None]],
+            ] = OrderedDict()
+            for layer_name, layer_idx in layer_entries:
+                kv_cache_spec = group_spec.kv_cache_spec
+                if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
+                    kv_cache_spec = kv_cache_spec.kv_cache_specs[layer_name]
+                total_num_kv_heads = self._get_spec_total_num_kv_heads(kv_cache_spec, layer_idx)
+                spec_key = self._get_kv_transfer_spec_key(kv_cache_spec, total_num_kv_heads)
+                spec_groups.setdefault(spec_key, []).append((layer_name, layer_idx, kv_cache_spec, total_num_kv_heads))
+            ...
+            for entries in spec_groups.values():
+                ...
+                total_num_kv_heads = entries[0][3]
+                kv_group2layeridx[transfer_group_id] = (
+                    self._serialize_kv_group_spec(
+                        group_spec, layer_names=layer_names,
+                        kv_cache_spec=kv_cache_spec,
+                        kv_cache_group_id=kv_cache_group_id,
+                        total_num_kv_heads=total_num_kv_heads,
+                    ),
+                    layer_indices,
+                )

@@ _get_attention_group_num_key_value_heads （读取优先级）
-            for key in ("num_kv_heads", "num_key_value_heads"):
+            for key in ("total_num_kv_heads", "num_kv_heads", "num_key_value_heads"):
```

### 5.3 附带修复：统一分组路径 + HMA 触发

diff 还顺带做了两件正确性收尾：

- **统一分组路径**：修复前只有 `UniformTypeKVCacheSpecs` 才按 spec 细分 transfer group，非 uniform 组直接整组一个 transfer group（不看 head）。修复后两条路径统一走同一个 `spec_groups` 逻辑，非 uniform 组也会按 `(spec_type, local, total)` 正确细分。
- **`_requires_group_aware_attention_transfer`**：新增判定——当存在多个不同的 `total_num_kv_heads`（即 target/draft head 不同）时，强制启用 group-aware 传输（`self._is_hma_required = True`，见 `mooncake_connector.py:2291`），走 `_get_hybrid_remote_rank_group_pulls` 而非朴素 `_get_remote_rank`，确保各 transfer group 独立算 pull。

### 5.4 Review 价值：fail-fast vs 防御性回退

Gemini Code Assist 在 review 中建议对 `self.tp_size` / `self.num_key_value_heads` 为 `None` 的情况加防御性回退（`tp_size = self.tp_size or 1`）。作者明确拒绝（见 review comment）：

> `tp_size` and `num_key_value_heads` are required runtime state ... Falling back to TP=1 or an uncapped value would hide an invalid worker initialization and could silently produce an incorrect transfer mapping. ... I am keeping the current fail-fast behavior.

这正是一个重要的设计取向：**传输映射类代码宁可显式崩，也不可静默错**（与 #8540 的 silent 精度下降形成对照——前者 fail-fast，后者 silent，后者更危险）。

---

## 6. 关键代码上下文

文件：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`（行号为当前 main 分支）

| 位置 | 内容 | 说明 |
|------|------|------|
| `1955` | `self.num_key_value_heads = hf_text_config.num_key_value_heads` | target 模型全局 KV head 数（rank 间共享标量） |
| `2004-2006` | `num_d_block_heads // num_p_block_heads` → `tp_num_need_pulls` | TP 不等时每 rank 的 pull 数；依赖全局 head 数 |
| `2033-2070` | `_serialize_kv_group_spec` | **本 PR 修复点**：新增 `total_num_kv_heads` 参数并写入 spec |
| `2088-2093` | `_get_spec_num_key_value_heads` | 取 spec 的 local head（TP 分片后） |
| `2096-2108` | `_get_kv_transfer_spec_key` | **本 PR 修复点**：分组键升为 3 元组 `(spec_type, local, total)` |
| `2110-2123` | `_get_spec_total_num_kv_heads` | **本 PR 新增**：按 layer_idx 判定 target/draft，取 owning model config 的 `get_total_num_kv_heads()`；MLA 直接返回 local |
| `2125-2189` | `_build_kv_group2layeridx` | **本 PR 重构**：统一 uniform/非 uniform 两条路径，按 3 元组键细分 transfer group |
| `2194-2200` | `_requires_group_aware_attention_transfer` | **本 PR 新增**：多个不同 total head 时强制 HMA |
| `2290-2291` | `register_kv_caches` 内 | `self._is_hma_required = self._is_hma_required or self._requires_group_aware_attention_transfer()` |
| `3154-3184` | `_get_hybrid_remote_rank_group_pulls` | HMA 路径：按 group 独立算 `num_group_pulls` 与 remote rank |
| `3210-3214` | `_get_attention_group_num_need_pulls` | `num_d_block_heads // num_p_block_heads`，依赖 `_get_attention_group_num_key_value_heads` 返回的**全局** head |
| `3216-3230` | `_get_attention_group_num_key_value_heads` | **本 PR 修复点**：key 查找顺序改为 `total_num_kv_heads` 优先，回退 `num_kv_heads`/`num_key_value_heads` |

测试文件：`tests/ut/kv_offload/test_mooncake_connector.py`

| 测试 | 覆盖点 |
|------|--------|
| `test_attention_group_uses_explicit_total_heads_for_unequal_pd_tp` | MLA local=1→返回 1；FullAttention local=4/total=8→返回 8；local=1/total=8→返回 8；pulls=4 |
| `test_build_kv_group2layeridx_splits_uniform_group_by_kv_heads` | MLA(local=1,total=128) vs FullAttention(local=1,total=8) 按 total 拆 2 组 |
| `test_build_kv_group2layeridx_splits_equal_local_heads_by_total_heads` | target total=16 / draft total=8，local 均=1 → 拆 2 组；target pulls=2、draft pulls=1 |
| `test_hybrid_rank_pulls_use_transfer_group_kv_heads` | metadata 携带 `total_num_kv_heads` 字段的回退兼容 |

---

## 7. 复现与验证

### 7.1 触发矩阵

| PD 分离 | TP(P)≠TP(D) | head 复制 / draft head 不同 | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗ | — | — | ✗（无跨节点传输） |
| ✓ | ✗（相等） | — | ✗（pulls=1，head 一一对齐，不需反推全局） |
| ✓ | ✓ | ✗（head 整除切分，local 能反推 total） | ✗（local*TP = total，无歧义） |
| **✓** | **✓** | **✓（head 复制 或 draft head ≠ target）** | **✓ 触发传输组误合并 + pull 错位** |

### 7.2 真机验证（PR 描述）

四节点 Ascend A2，Kimi K2.7 Code W4A8 + Kimi-K2.5-DFlash，P 端 DP2×TP8 / D 端 DP8×TP2，65542 token 串行请求：

- 修复前：6 请求 gate 在 round 3、6 失败；20 轮回归在 round 3/6/9/12/15/18 失败。
- 修复后：
  - 10 个 P/D `/v1/models` 端点均 HTTP 200；
  - 短请求 HTTP 200 且 JSON 合法；
  - Prefill 日志 `FullAttentionSpec(local=1, total=8)`，Decode 日志 `FullAttentionSpec(local=4, total=8)`；
  - 6 请求 gate `6/6` 通过（含原失败的 round 3、6）；
  - 20 轮全量 `20/20` 通过（含原失败的 round 3/6/9/12/15/18）；
  - 四节点日志零 HCCL batch-get / Mooncake transfer / KV load / traceback / error 匹配。

### 7.3 单元测试

```text
PYTHONPATH=$PWD python -m unittest tests.ut.kv_offload.test_mooncake_connector
Ran 95 tests in 2.474s
OK
```

回归用例覆盖：P local=1/D local=4 + total=8 的不等 TP；target total=16 / draft total=8 且 local 均=1 的 group 拆分；group-aware rank pulls（target pulls=2、draft pulls=1）。

### 7.4 排查方法

1. **确认三要素**：PD 分离、P/D TP 不等、是否存在 head 复制（TP > 全局 KV head）或 draft/target head 不同。
2. **查日志 spec**：搜索 `FullAttentionSpec(local=..., total=...)`，确认 P/D 两端 total 是否一致、是否按 total 拆 group。
3. **周期性失败模式**：若请求在固定 round（如 3/6/9...）失败，高度怀疑 group 拆分/pull 映射错位。
4. **对照 TP 相等基线**：把 D 端 TP 设成与 P 端相同，若失败消失 → 锁定本路径。
5. **关闭投机对照**：关 draft 后正常 → 锁定 target/draft head 差异类。

---

## 8. 经验与启发

### 8.1 local 维度不可反推 global 维度

凡是 TP/PP 分片后产生的 rank-local 标量（`num_kv_heads`、层数、head_dim 等），**都不可用 `local * TP` 反推全局**——head 会在 TP 超过全局数时被复制，draft 模型的全局数与 target 不同。跨节点传输的元数据必须**显式携带全局值**，而不是让接收端去“算回来”。这是 #8540（`layers` 取 local 列表长度而非 config 标量）的同源教训的另一面：一个教训“别用 config 标量代替 local 实际”，本教训“别用 local 反推 global 标量”——方向相反，同源：**口径不一致**。

### 8.2 分组键要包含所有“影响行为的维度”

`_get_kv_transfer_spec_key` 修复前只按 `(spec_type, local_heads)` 分组，漏了 `total_heads`——而 `total_heads` 直接决定 pull 映射。分组键/哈希键的设计原则：凡是会影响下游行为（pull 数、remote rank、reformat 比例）的字段，都必须进键，否则会出现“键相同但行为应不同”的误合并。这是传输/路由类代码的高发 bug 模式。

### 8.3 target/draft 层必须按 owning model 区分

`_get_spec_total_num_kv_heads` 的关键设计：用 `layer_idx >= self.total_layers` 判定 draft 层，进而取 `draft_model_config` 而非 `model_config`。这与 #8540 中 `_get_group_kv_caches` 用 `layer_idx >= num_layers` 判定 MTP 层如出一辙（见 [1_pr8540](./1_pr8540_tp_unequal_mtp_kv.md) §3.3）——**“超过主模型层数的层归属 draft”** 是处理投机解码的通用判定模式，应在所有涉及 target/draft 差异的代码路径统一使用。

### 8.4 fail-fast 优于 silent 回退

review 中作者坚持不为 `tp_size`/`num_key_value_heads` 为 `None` 加回退：回退会隐藏初始化错误并静默产出错误映射。传输映射类代码的正确取向是“宁可显式崩，不可静默错”——这与 #8540 的 silent 精度下降（更危险、更难查）形成鲜明对照。两类 bug 都高危，但 silent 类更应在前置防御。

### 8.5 向后兼容：新字段优先读、旧 metadata 回退

`_get_attention_group_num_key_value_heads` 的 key 查找顺序改为 `total_num_kv_heads` 在前，但保留 `num_kv_heads`/`num_key_value_heads` 作回退——保证旧 P 端（未带新字段）的 metadata 仍可被新 D 端读取。跨节点协议演进必须做这种前向/后向兼容，避免升级即断。

### 8.6 统一 code path 优于特判分支

修复前 `_build_kv_group2layeridx` 对 `UniformTypeKVCacheSpecs` 与非 uniform 用两套不同逻辑（前者细分、后者不细分）。修复后统一为一条 `spec_groups` 路径。两套逻辑分叉是 #8540 同款 bug 温床（见 [1_pr8540](./1_pr8540_tp_unequal_mtp_kv.md) §8.2）——统一路径能消除“口径分叉”。

---

## 9. 关联问题

| 编号 | 关联点 |
|------|--------|
| [vllm-ascend#8540](https://github.com/vllm-project/vllm-ascend/pull/8540) / [1_pr8540](./1_pr8540_tp_unequal_mtp_kv.md) | 同族“PD 分离 + TP 不等 + 投机”精度 bug：#8540 修“层数口径”（MTP 层被丢），#11886 修“head 数口径”（全局 head 被漏）。一个 local→config 标量，一个 local→global 标量，同源不同向 |
| [vllm-ascend#11601](https://github.com/vllm-project/vllm-ascend/pull/11601) | Mooncake split metadata 携带 cache group ids——同属“传输元数据缺字段”类，本 PR 在其基础上再加 `total_num_kv_heads` |
| [vllm-ascend#9500](https://github.com/vllm-project/vllm-ascend/pull/9500) | Deepseek-V4 PD 分离 `kv_cache_tensor.shared_by` 可能为空——同属 Mooncake 传输元数据/分组正确性 |
| [vllm-ascend#12183](https://github.com/vllm-project/vllm-ascend/pull/12183) | Non-contiguous Mooncake PA cache inputs——同属 Mooncake KV 传输正确性（见 [0_kvcache.md](./0_kvcache.md) §2.3 第 1 条） |
| [vllm-ascend#11885](https://github.com/vllm-project/vllm-ascend/issues/11885) | 本 PR Fixes 的 issue |
| [vllm-ascend#10991](https://github.com/vllm-project/vllm-ascend/pull/10991) / #11877 | PR 描述中标注的 Related |
| [0_kvcache.md](./0_kvcache.md) §2.3 / 案例 6 | 本案例在全景文档中的归档位置（§2.3 第 2 条，案例 6 为 #8540） |

---

## 10. 时间线小结

| 时间 | 事件 |
|------|------|
| 2026-07-12 18:18 | PR 创建，首 commit `584063d0`（含 `to_global_num_heads` 内嵌函数） |
| 2026-07-12 18:20 | Gemini Code Assist review，建议对 `tp_size`/`num_key_value_heads` 为 None 加防御回退（high） |
| 2026-07-12 19:05 | 作者回复拒绝回退，坚持 fail-fast；后改为当前 `total_num_kv_heads` 显式字段方案（commit `3c2a9969`） |
| 2026-07-13 13:49 | 合并入 main（merge commit `3bfb521d`） |
| 真机回归 | 四节点 A2，Kimi K2.7 + K2.5-DFlash，gate `6/6` + 全量 `20/20` 通过 |

---

> 本案例核心教训：**跨节点 KV 传输的序列化元数据必须显式携带“模型级全局 head 数”，不可让接收端从 rank-local 的 `num_kv_heads` 反推——否则在 PD 分离 + TP 不等 + head 复制/draft head 不同于 target 的组合下，target 与 draft 传输组会被误合并、rank pull 映射错位，引发周期性传输失败与精度异常。**