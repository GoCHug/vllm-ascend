# PR #8540 深度案例：PD 分离 TP 不等时 MTP 层 KV Cache 未处理

> 整理时间: 2026-07-29
>
> 案例对象: [vllm-project/vllm-ascend#8540](https://github.com/vllm-project/vllm-ascend/pull/8540)
>
> 标题: [BugFix] [P/D] In scenarios where TP is not equal, the KV cache at the MTP layer is not handled.
>
> 关联全景文档: [0_kvcache.md](./0_kvcache.md) §2.3 / §6.2 / 案例 6
>
> 关键词: PD 分离 · TP 不等 (TP unbalanced) · MTP 投机解码 · Mooncake Connector · KV Cache 传输 · 精度异常

---

## 1. PR 概览

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| PR 编号 | [#8540](https://github.com/vllm-project/vllm-ascend/pull/8540) |
| 类型 | BugFix |
| 作者 | [@liziyu179](https://github.com/liziyu179) |
| Reviewer | @LCAIZJ, @MengqingCao |
| 状态 | closed / **merged** |
| 创建时间 | 2026-04-21 |
| 合并时间 | 2026-04-23 |
| 合并 commit | `5dba75bade4a0d5b36c26bb4611a9f545ceb821c` |
| 改动规模 | **+1 / −1**（单行修改） |
| 影响文件 | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` |
| vLLM 版本 | v0.19.0 |
| vLLM main | `6f786f2c506cb07f4566771fdc62e640e2c4a176` |
| 测试方式 | nightly |

> 典型的“一行修复，背后一个大坑”案例：改动仅 1 行，但触及 PD 分离 + TP 不等 + MTP 三者交叉的精度盲区。

---

## 2. 场景背景

### 2.1 PD 分离部署 (Prefill/Decode Disaggregation)

在 PD 分离架构中，Prefill 节点（P 端）负责长 prompt 的一次性计算，Decode 节点（D 端）负责逐 token 自回归生成。两者之间需要把 P 端计算好的 **KV Cache** 通过网络传输到 D 端，避免 D 端重复计算 prefix。

vLLM-Ascend 在 PD 分离场景下使用 **Mooncake Connector** 作为 KV Cache 跨节点传输通路（基于 Mooncake / RDMA）。

### 2.2 TP 不等 (TP unbalanced)

为最大化吞吐与资源利用，P 端和 D 端常采用**不同的 Tensor Parallel 度数**，例如：

- P 端 `TP=8`（prefill 计算密集，用更多卡并行）
- D 端 `TP=4`（decode 计算稀疏，少卡即可）

TP 不等意味着：P 端每个 rank 持有的 KV head 数量与 D 端不同。传输 KV Cache 时**不能直接字节拷贝**，必须按 head 重新分片/重排（reformat / transpose）。

### 2.3 MTP 投机解码 (Multi-Token Prediction)

MTP 在主模型最后一层之后追加一个（或多个）**MTP 层（draft layer）**，用于推测多个候选 token。关键点：

- MTP 层**自己有独立的 KV Cache**，参与 KV 传输
- vLLM-Ascend 中，MTP 各步复用同一份 KV cache 层，故只需传输一次（`self.num_draft_layers = 1`）

三者组合（PD 分离 × TP 不等 × MTP）即为本 PR 修复的触发场景。

---

## 3. 根因分析

### 3.1 Bug 代码

修复前，`reformat_kv_cache_with_fused_op` 中用模型配置的 `num_hidden_layers` 作为“传输层数”：

```python
# vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
def reformat_kv_cache_with_fused_op(self, block_ids, tp_num_need_pulls, kv_caches=None):
    ...
    num_kv_head = max(self.model_config.hf_text_config.num_key_value_heads // self.tp_size, 1)
-   layers = self.model_config.hf_text_config.num_hidden_layers   # ← BUG：只算 attention 层
    flat_block_ids = [item for sublist in block_ids for item in sublist]
    block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int64, device=device)

    k_caches = []
    v_caches = []
    for _, (k_cache_layer, v_cache_layer) in kv_caches.items():
        k_caches.append(k_cache_layer)        # ← 实际把所有层（含 MTP）都收集进来
        v_caches.append(v_cache_layer)

    torch.ops._C_ascend.transpose_kv_cache_by_block(
        k_caches, v_caches, block_ids_tensor, block_size,
        num_kv_head, head_dim, tp_num_need_pulls, layers,   # ← layers 传给融合算子
    )
```

### 3.2 不一致点

- `k_caches` / `v_caches` 列表来自 `kv_caches.items()`，**包含 MTP 层**（见 §3.3），列表长度 = `num_hidden_layers + num_draft_layers`（MTP 场景为 `N + 1`）。
- 而 `layers = num_hidden_layers`（= `N`），**比实际列表少 1**。

融合算子 `transpose_kv_cache_by_block` 以 `layers` 为准执行 transpose/reformat，于是：

```
k_caches 实际长度 = N + 1（含 MTP 层）
layers 传入值    = N     （num_hidden_layers）
→ 融合算子只处理前 N 层，最后一层（MTP 层）KV Cache 被跳过
→ MTP 层 KV Cache 未被 transpose / 未按 TP 重新分片
→ D 端 MTP 层拿到错位/未重排 的 KV 数据 → 精度异常
```

### 3.3 MTP 层为何会进入 kv_caches

`_get_group_kv_caches` 在筛选本组要传输的层时，显式把 MTP 层纳入：

```python
def _get_group_kv_caches(self, group_idx, layer_indices=None):
    ...
    model_type = self.vllm_config.model_config.hf_text_config.model_type
    num_attn_module = 2 if model_type in ("longcat_flash", "longcat_flash_ngram") else 1

    def layer_in_group(layer_name: str) -> bool:
        if "mtp" in layer_name:
            # MTP 层的 index ≥ num_layers，单独判定纳入
            return any(layer_idx >= self.num_layers for layer_idx in layer_index_set)
        return extract_layer_index(layer_name, num_attn_module) in layer_index_set

    return {
        layer_name: layer_cache
        for layer_name, layer_cache in self.kv_caches.items()
        if layer_in_group(layer_name)
    }
```

也就是说，上层逻辑**已经正确地把 MTP 层放进了待传输的 `kv_caches` 字典**，但 `reformat_kv_cache_with_fused_op` 内部又用 `num_hidden_layers` 重新算了一遍层数，导致两边对“层数”的认知不一致——典型的**同一概念在两处分别计算、且口径不同**的 bug。

### 3.4 为什么“TP 不等”才触发

`reformat_kv_cache_with_fused_op` / `reformat_kv_cache` 这条 **reformat 路径**只在 P、D 端 TP 度数不同（`tp_num_need_pulls != 1`，即 `tp_ratio != 1`）时被调用，用于按 TP 比例对 KV head 做重排搬运：

```python
# _transfer_kv_cache_all_groups 内
if tp_ratio == 1:
    # TP 相等：直接走 conv/ssm 跨步拷贝，逐层遍历真实 kv_caches，不使用 layers 参数
    self.reformat_kv_cache_hybrid_linear_torch(grouped_local_block_ids, num_group_pulls, group_kv_caches)
else:
    # TP 不等：走融合算子 reformat，依赖 layers 参数 ← BUG 触发点
    if need_fused_op:
        self.reformat_kv_cache_with_fused_op(reformat_block_ids, num_group_pulls, group_kv_caches)
    else:
        self.reformat_kv_cache(reformat_block_ids, num_group_pulls, False, need_nz_cache, group_kv_caches)
```

- **TP 相等**：走 `reformat_kv_cache_hybrid_linear_torch`，按 `kv_caches.items()` 逐层处理，MTP 层被自然覆盖 → **不出错**。
- **TP 不等 + MTP**：走融合算子路径，`layers` 少算 1 → MTP 层 KV Cache 被丢弃/错位 → **精度异常**。

这正是 PR 标题里“TP is not equal”的由来：bug 潜伏在 reformat 路径里，而该路径只在 TP 不等时激活；同时 MTP 必须开启，否则 `kv_caches` 长度本就等于 `num_hidden_layers`，`layers` 不会少算。

**触发条件 = PD 分离 + TP(P) ≠ TP(D) + 开启 MTP**，三者缺一不可——这也是它隐蔽性高、CI 难以覆盖的原因。

---

## 4. 影响与表现

| 维度 | 表现 |
|------|------|
| 触发配置 | PD 分离 + P/D 端 TP 不等 + MTP 投机解码 |
| 不触发 | TP 相等 / 不开 MTP / 单卡 |
| 现象 | MTP（draft）层 KV Cache 未被按 TP 重排；D 端 MTP 层读到错位 KV，draft token 质量下降，投机接受率低 |
| 严重度 | 🔴 高（端到端精度异常） |
| 隐蔽性 | 🟡 高（需 TP 不等 + MTP 同时满足；TP 相等时不显现，CI 易漏） |
| 是否报错 | 通常**不报错**，silent 精度下降；融合算子只是少处理一层，不会触发断言 |

由于 D 端 MTP 层 KV Cache 头归属错位，draft model 的注意力计算 K/V 不匹配，导致：
1. draft 候选 token 质量变差 → 接受率下降 → 吞吐退化
2. 接受的错误候选被回滚，但累积扰动可能影响最终输出质量
3. 长上下文/高并发下误差更显著

---

## 5. 修复方案

### 5.1 最终改动（merged）

```diff
  layers = self.model_config.hf_text_config.num_hidden_layers
+ layers = len(self.kv_caches)
```

（当前 main 分支在后续重构中进一步改为 `layers = len(kv_caches)`，即使用传入的局部 `kv_caches` 字典长度，语义更精确——见 `mooncake_connector.py:1171`。）

**核心思想**：`layers` 应反映**融合算子实际要处理的、本 rank 真实持有的 KV Cache 层数**（即 `k_caches`/`v_caches` 列表长度），而不是模型配置里的总 attention 层数。两者在“MTP 追加层”和“PP 分片”下都不相等。

### 5.2 修复方案演进（来自 review 历史）

PR 初版用的是**硬编码 +1**：

```python
layers = self.model_config.hf_text_config.num_hidden_layers
if self.vllm_config.speculative_config is not None:
    layers = layers + 1
```

Gemini Code Assist 在 review 中指出两个问题（high priority），值得作为通用教训：

1. **`speculative_config is not None` 条件过宽**
   - 不是所有投机解码方法都给主模型追加 KV 层——这是 **MTP 特有**的（EAGLE 等 draft-model 方案的 KV 层在 draft model 内，不进主模型 `kv_caches`）。
   - 笼统 `+1` 会在非 MTP 的投机场景下多算一层。

2. **硬编码 `+ 1` 不考虑 Pipeline Parallelism (PP)**
   - `num_hidden_layers` 是**整个模型**的总层数；
   - 而 `self.kv_caches` 在 PP 下只包含**当前 rank 分到的层**。
   - 用 `num_hidden_layers (+1)` 在 PP 场景下与实际列表长度严重不符。
   - 补充：MTP 层只在**最后一个 PP stage** 才出现在 `self.kv_caches` 中，进一步加剧“层数口径”的复杂性。

review 给出的推荐写法正是最终被采纳的：

```python
layers = len(self.kv_caches)
```

> 用“实际数据结构的长度”替代“配置里的标量”，一劳永逸地兼容 MTP 追加层、PP 分片、不同投机方法等所有口径差异。这是处理“层数/维度”类参数的稳健范式。

---

## 6. 关键代码上下文

文件：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`

| 位置 | 内容 | 说明 |
|------|------|------|
| `~516` | `self.num_layers = hf_text_config.num_hidden_layers` | 模型 attention 总层数（不含 MTP） |
| `~530-541` | `self.num_draft_layers` 计算 | MTP 时为 1；其他 draft model 取 draft 的 `num_hidden_layers` |
| `~1136-1152` | `_get_group_kv_caches` | 按组筛选待传输层，**显式纳入 MTP 层**（`layer_idx >= self.num_layers`） |
| `~1159-1183` | `reformat_kv_cache_with_fused_op` | **本 PR 修复点**：`layers = len(kv_caches)`，调用 `transpose_kv_cache_by_block` |
| `~1185+` | `reformat_kv_cache`（torch 路径） | 同样依赖 `layers`，同类风险 |
| 调用处 | `_transfer_kv_cache_all_groups` | 仅 `tp_ratio != 1` 时走 reformat 路径 |

修复后，`layers` 与 `k_caches`/`v_caches` 列表长度严格一致，融合算子能覆盖全部待传输层（含 MTP）。

---

## 7. 复现与验证

### 7.1 触发矩阵

| PD 分离 | TP(P)≠TP(D) | MTP | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗ | — | — | ✗（无跨节点传输） |
| ✓ | ✗（相等） | ✓ | ✗（走 hybrid_linear_torch 路径，逐层处理） |
| ✓ | ✓ | ✗ | ✗（`kv_caches` 长度 = num_hidden_layers，layers 不少算） |
| **✓** | **✓** | **✓** | **✓ 触发精度异常** |

### 7.2 排查方法

1. **确认三要素**：部署是否 PD 分离、P/D 端 TP 是否不等、是否开启 MTP。
2. **对比 TP 相等基线**：把 D 端 TP 设成与 P 端相同，若精度恢复正常 → 高度怀疑本路径。
3. **关闭 MTP 对照**：关 MTP 后精度正常 → 锁定 MTP 层 KV 处理。
4. **逐层 KV 校验**：在 D 端 dump MTP 层 KV Cache，与 P 端对应层做数值对比（TP 重排后的期望值），检查 head 归属是否错位。
5. **eager / 非 fused-op 对照**：走 `reformat_kv_cache`（torch 路径）或关闭 fused op，看是否复现，定位是 `layers` 参数问题还是算子本身。
6. **Nightly / 长时间压测**：本 PR 修复由 nightly 覆盖验证；精度类问题建议补充 TP 不等 + MTP 的 e2e accuracy 回归。

### 7.3 回归测试建议

- 新增 e2e 用例：PD 分离 + P 端 TP=8 / D 端 TP=4（及互换）+ MTP 开启，对比 TP 相等基线的端到端 accuracy / draft 接受率。
- 在 mooncake_connector 单测中，构造 `kv_caches` 含 MTP 层的字典，断言传给 `transpose_kv_cache_by_block` 的 `layers == len(kv_caches)`。

---

## 8. 经验与启发

### 8.1 “层数/维度”一律以实际数据结构为准

凡是要传给底层算子的 `layers`/`num_heads`/`dim` 等标量，应直接取自**实际要操作的 tensor / list / dict 的形状**，而非从模型 config 重新推导。配置标量与实际数据在 MTP、PP、量化、hybrid 等场景下口径频繁不一致——这是 KV Cache 精度问题的高发根源（参见 [0_kvcache.md](./0_kvcache.md) §1.1 MLA reshape、§4 布局/reshape 问题）。

### 8.2 同一概念不要在多处分别计算

本 bug 中，“待传输层数”在上层（`_get_group_kv_caches`）按真实字典筛选，在下层（`reformat_*`）又按 config 重算——两处口径分叉。应**单一来源**：上层筛出多少层，下层就用多少层。

### 8.3 TP 不等 = 隐藏的 code path

TP 相等是“快乐路径”，TP 不等会切到另一条 reformat/重排路径，里面常潜藏只有在跨 TP 分片时才暴露的 bug。任何涉及 KV 传输/重排的改动，都应把 **TP 相等 / TP 不等** 当作两条独立路径分别测试。这是 vllm-ascend PD 分离的特有风险面（参见 [0_kvcache.md](./0_kvcache.md) §2.3、§8、§11.3）。

### 8.4 条件判断要精确到“到底是哪种投机”

`speculative_config is not None` 过宽——MTP 追加 KV 层，EAGLE/draft-model 不追加主模型 KV 层。处理投机解码分支时，应按 `method`（`mtp` vs 其他）精确分支，避免把 MTP 专属逻辑套到所有投机方法上（参见 [0_kvcache.md](./0_kvcache.md) §6.1 draft dtype 继承）。

### 8.5 Review 价值：把“能跑的 hack”逼成“正确的抽象”

初版 `layers + 1` 能让 MTP 场景跑通，但在非 MTP 投机、PP 分片下会错。Review 把它升级为 `len(self.kv_caches)`，从“打补丁”变成“根治”。这类“用实际长度替代配置标量”的 review 建议，是 KV Cache 代码质量提升的典型范式。

### 8.6 一行修复 ≠ 一行风险

改动虽仅 1 行，但：
- 触发条件苛刻（三要素组合），CI 默认用例不易覆盖；
- 表现为 silent 精度下降，无崩溃、无断言；
- 涉及跨节点传输 + 融合算子，排查链路长。

这正是它入选 [0_kvcache.md](./0_kvcache.md) 高危 TOP 10（案例 6）的原因。

---

## 9. 关联问题

| 编号 | 关联点 |
|------|--------|
| [vllm-ascend#11829](https://github.com/vllm-project/vllm-ascend/pull/11829) | Layerwise KV Pool + MTP IndexError — 同属 “MTP 带来额外 KV 层、索引/层数漏算” 类问题 |
| [vllm-ascend#11929](https://github.com/vllm-project/vllm-ascend/pull/11929) | PP 非 last stage MTP finalize — MTP 层只在 last PP stage 出现的同类口径问题 |
| [vllm-ascend#11470](https://github.com/vllm-project/vllm-ascend/pull/11470) | Qwen3.x 多层 KV cache binding — 多层 KV 归属问题 |
| [vllm-ascend#11601](https://github.com/vllm-project/vllm-ascend/pull/11601) / [#11886](https://github.com/vllm-project/vllm-ascend/pull/11886) | Mooncake 传输元数据携带 KV head / cache group ids — TP 不等场景 head 归属同类 |
| [vllm-ascend#12183](https://github.com/vllm-project/vllm-ascend/pull/12183) | Non-contiguous Mooncake PA cache inputs — 同属 Mooncake KV 传输正确性 |
| [0_kvcache.md](./0_kvcache.md) §2.3 / 案例 6 | 本案例在全景文档中的归档位置 |

---

## 10. 时间线小结

| 时间 | 事件 |
|------|------|
| 2026-04-21 | PR 创建，初版用 `if speculative_config: layers += 1` |
| 2026-04-21~22 | Gemini Code Assist review，指出条件过宽 + 不兼容 PP，建议 `len(self.kv_caches)` |
| 2026-04-23 | 采纳建议改为 `layers = len(self.kv_caches)`，合并入 main |
| 后续重构 | 进一步精确为 `layers = len(kv_caches)`（使用局部参数） |

---

> 本案例核心教训：**跨节点 KV 传输中，凡涉及 TP 重排的融合算子，其“层数”参数必须取自实际 `kv_caches` 容器长度，而非模型 config 的 `num_hidden_layers`**——否则在 PD 分离 + TP 不等 + MTP 组合下，MTP 层 KV Cache 会被静默丢弃，引发难以复现的精度异常。