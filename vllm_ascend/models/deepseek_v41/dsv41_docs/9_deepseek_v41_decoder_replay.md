# DeepSeek V4.1 Decoder 有界回放（bounded replay）详解

> **代码版本冻结声明**：本文基于当前 checkout 内 `vllm-ascend` 源码整理。实现在 [decoder_replay_layers.py](../decoder_replay_layers.py)（141 行，planner + 层适配器）与 [decoder_replay_state.py](../decoder_replay_state.py)（442 行，状态机）；模型入口 [model.py](../model.py) L1095-1096 / L1568-1575 / L1629-1639；注意力侧可见长度与 cache 重定基 [dsa_v41.py](../../../attention/dsa_v41.py) L95-136、L1158-1177、L1239-1253、L664-742；行集合的 device 算子 [decoder_replay.py](../../../ops/triton/decoder_replay.py)。未特别注明的 `Lxxx` 指该小节标题标注的文件。上游开关来自 vLLM #56227（`cache_config.swa_bounded_replay`），Ascend 在其上加了**第二级裁剪**："只有最后一个 KV 源之后的层做紧凑重放"。

> **一句话结论**：V4.1 的 21–39 层不写任何压缩 KV / Index K，只读第 20 层的两份源缓存加自己 128 宽的 SWA；于是 32K prefill 在这 19 层上必须逐 token 算的只有**最后 128 行**。`DecoderReplayState` 压缩行集合、跨 DP/PCP 协调、重建 `AscendDSAV41Metadata` 并替换 forward context；`AscendDecoderReplayLayers` 聚成紧凑批次跑完 21–39 层再散回全尺寸。收益量级：19 × (32768−128) = **620,160 个 token-层**。

> **为什么 V4.1 才有这个优化**：V4 每层自持压缩 KV 与 Indexer 状态，删掉任何一层的 token 都会让后续层读到"不存在的历史"。V4.1 把源层收敛到 4 个（`kv_source_layer_ids=[2,8,14,20]`），20 之后再无 `is_kv_source` 的层：21–39 层只经 `long_kv_source_prefix`/`index_k_source_prefix` 指向第 20 层，`is_kv_source` 为假时 `_write_compressed_source` 不执行（[dsa_v41.py](../../../attention/dsa_v41.py) L395-396、L428-435）。层内唯一的 token 间依赖是窗长 128 的 SWA，而"最后 128 行"对 SWA 是**闭包**：窗口内的行只依赖窗口内的行，窗口外的行既不影响它们、也不再被下游读取（除非该请求要回 logits）。V4 无此性质，所以这优化只属于 V4.1。

---

## 一、问题：长 prefill 在源共享拓扑下浪费了什么

`text_config`：`num_hidden_layers=40`、`sliding_window=128`、`kv_source_layer_ids=[2,8,14,20]`、`index_source_layer_ids=[2,8,14,20,24,28,32,36]`、`candidate_source_layer_id=20`、`engram_layer_ids=[1,14]`、`compress_ratios` 为 `[0,0, 2×18, 1×20, 0,0,0]`。按 `build_layer_plan`（[model.py](../model.py) L506-561）的 `_latest_source` 展开：

| 层段 | `ratio` | 本层写什么 | 源读谁 |
|------|:-------:|-----------|--------|
| 0–1 | 0（dense） | 只有 SWA | 无长上下文 |
| 2 / 8 / 14 | 2（c2） | SWA + c2 状态 + 长 KV + Index K | 自己 |
| 其余 2–19 | 2 | 只有 SWA | 2 / 8 / 14 |
| **20** | **1** | **SWA + 长 KV + Index K** | 自己 |
| **21–39** | **1** | **只有 SWA** | **20** |

要点不在层数，而在 **21–39 层没有任何自己的压缩状态**：SWA 无条件写（L394、L424），`_write_compressed_source` 却被 `self.role.is_kv_source` 挡住（L395、L428）。这 19 层的输出只取决于本 token 的 query、第 20 层已写好的完整源缓存、本层 SWA 里最近 128 个 token 的 KV。

32K prefill 的账：0–20 层必须逐 token 算完（第 20 层要为每个 token 产出 c1 长 KV + Index K，否则 21 层没有历史可读）；21–39 层每层只写 SWA，行 t 的 SWA KV 只可能被 t..t+127 读到，故**窗口外的 32640 行只通过 SWA 影响后来者，而 SWA 只回看 128**。浪费量 **19 × (32768 − 128) = 620,160 个 token-层**。

但"只算 128 行"不是切个 tensor：全 batch 视角的 `seq_lens`/`slot_mapping`/`is_padding`/`mc2_mask` 会错位（§5、§6）；行到"原第几行 / 哪个位置 / 哪个 slot"的映射会丢（§4）；SWA 会读到自己没重建的块（§7.2）；要回 logits 的请求不能被裁（§7.3）。

---

## 二、何时启用 / 何时自动关闭

### 2.1 开关本身会不会被留下

| 条件 | 结果 | 说明 | 位置 |
|------|------|------|------|
| `architectures` 不是 V4.1 的两个注册名 | `resolve_bounded_replay` 返回 False | 能力写死在 V4.1 树，V4 树不读开关；DSpark 草稿也排除（它的 SWA cache 无此字段） | [kv_cache_interface.py](../../../core/kv_cache_interface.py) L108-119、L142-147 |
| 4 种 Ascend 调度模式（recompute / dyntra / profiling_chunk / balance） | 开关被置 False | 它们用旧版 `schedule()` 的逐字拷贝替换了上游方法，既不回退前缀命中、也认不出"这一步是重放" | [platform.py](../../../platform.py) L1527、L1530-1555、L1558-1589 |
| 通过 | 写进 SWA 的 cache spec | `bounded_replay=True` ⇒ `prefix_cacheable=False`、`prefix_replay_tokens=128` | [cache_config.py](../cache_config.py) L50-68、[kv_cache_interface.py](../../../core/kv_cache_interface.py) L389-397、[model.py](../model.py) L637 |

判定分两级：`resolve_bounded_replay` 决定"SWA 组要不要重放"（cache 平面），`make_decoder_replay` 决定"21–39 层要不要裁行"（执行平面）。前者开、后者关就退化成"只有调度回退、没有 decoder 裁剪"——正确，但省不到那 620K token-层。

### 2.2 planner 开关表（`make_decoder_replay`，decoder_replay_layers.py L84-141）

| 条件 | 结果 | 原因（代码/注释原话） | 位置 |
|------|------|----------------------|------|
| `swa_bounded_replay` 为假 | None | 开关没开 | L85-86 |
| `cut = max(kv_source_layer_ids)`；`cut >= end_layer - 1` | None | 最后一个源层之后没有层（本模型 `cut=20`，不触发） | L93-95 |
| SP / PP>1 / DCP>1 / `use_ubatching` | None | "SP, PP, DCP and microbatching need a separate replay layout" | L96-102 |
| `cudagraph_mode ∉ {NONE, FULL_DECODE_ONLY}` | None | "replay prefill requires eager execution" | L103-104 |
| `engram_layer_ids` 中有 `> cut` 的层 | None | "an Engram layer follows the last KV source" | L105-106 |
| 有投机且非 DSpark | None | "the replay adapter currently supports DSpark drafting only" | L107-108 |
| 草稿无 window / window > 目标 / 草稿层非 sliding | None | "the drafter ... reads hidden states outside the target's 128-token window" | L109-117 |
| `lora_config is not None` | None | "LoRA token mappings need to follow the replay batch" | L118-119 |
| 全部通过 | `decoder_replay_start = cut + 1 = 21` | `logger.info_once` | L121-139 |

关闭统一走 `logger.warning_once("Ascend decoder bounded replay is disabled: %s.", reason)`（L140），**排障先看这行的 reason 字符串**。

### 2.3 运行时旁路（`DecoderReplayState.prepare`，decoder_replay_state.py L213-256）

`prepare` 开头先把 `layers.rows / context_factory / num_padded_rows` 置 None（L214-220）：**任何一步都不带上一步的重放状态**。

| 条件 | 行为 | 位置 |
|------|------|------|
| 首次调用（`slots is None`） | 建 builder + `_gather(warmup=True)` 编译 kernel | L222-224 |
| `for_capture` / 运行时图模式 ≠ NONE / profiling run | 返回 | L225-226 |
| DP>1 且 DP 协调被跳过 | 返回（压缩长度无法协商） | L229-233 |
| 本步没有任何副本需要压缩 | 返回 | L244-246 |
| 对端压缩、本 rank 不压缩 | 只挂 `context_factory`，不动行集合 | L251-253 |
| dummy batch | `num_actual_tokens = 0`，不逐请求裁剪 | L239-241、L277-279 |

---

## 三、三个角色：planner / layer adapter / state machine

```
AscendDeepseekV41Model                                (model.py)
├── decoder_replay_start : int                        # L1095，默认 = end_layer
├── decoder_replay_layers : AscendDecoderReplayLayers | None   # L1096 ← make_decoder_replay(self, cfg)
│     ├── window       = config.sliding_window        # 128
│     ├── run_layers   = model._run_replay_layers     # L130
│     ├── row_buffers  = [topk_indices_buffer, candidate_indices_buffer,
│     │                   candidate_lengths_buffer, topk_lengths_buffer]   # L131-136
│     └── rows / num_actual_rows / num_padded_rows / context_factory / sequence_parallel
└── _run_replay_layers(hidden, pre_mix, positions, input_ids)  # L1629-1639，只跑 21..end_layer-1

EngramModelState(AscendModelState)                    (engram/model_state.py)
├── decoder_replay = None                             # L84-87：覆盖父类，V4.1 用自己的状态机
└── decoder_replay_state : DecoderReplayState | None  # L100-101
      ├── keeps_rows : np.bool_[max_num_seqs]         # 每 request slot：是否必须逐行精确
      ├── rows / positions / padding / offsets / dropped_before / replay_start
      ├── builders : [(group_id, layer_names, AscendDSAV41MetadataBuilder)]
      ├── metadata / num_tokens / num_actual_tokens / dp_tokens / replay_pcp_context
      └── prepare() / _build_replay_plan() / _coordinate_replay_across_dp()
          / _prepare_compact_replay() / _build_metadata() / forward_context()
```

- **planner**（`make_decoder_replay`，建模型时一次）：判定能力、写 `decoder_replay_start`、构造适配器并塞入 4 个行缓冲；失败返回 None ⇒ 21 层继续走普通循环。
- **layer adapter**（`__call__`，每次 forward）：紧凑化行集合 → 跑层 → 散回全尺寸；`rows is None` 时原样跑。
- **state machine**（`DecoderReplayState`，每步 `prepare_attn`）：算行集合、协调、重建元数据、换 context；早退 ⇒ 本步不重放。

一步之内（调度词汇沿用 MTP 文档 §1.2 的 ①→⑤，回放全在 ② 内部）：

```
① 调度：前缀命中时 chunk 回退到命中点前的 128 窗口（上游 rewind），
        被打断的 SWA slot 打 PAD_SLOT_ID                  (engram/model_state.py L204-216)
② 目标前向 execute_model（一遍前向吃下本步全部 token）
   ├─ prepare_attn → DecoderReplayState.prepare(...)       (model_state.py L235-243)
   │     _build_replay_plan → _coordinate_replay_across_dp → _prepare_compact_replay
   │     然后把 rows / context_factory 挂到 layers
   ├─ 0–20 层全行（第 20 层写完整源缓存）
   │   └─ 到 layer_idx == 21：AscendDecoderReplayLayers.__call__   (model.py L1569-1575)
   │         row_buffers 原地紧凑 + padding 清零 → index_select → _run_replay_layers
   │         → index_copy_ 散回；break，last_layer = layers[39]
   └─ hc_collapse + norm 在全尺寸缓冲上                   (model.py L1621-1624)
③ 采样 logits（只取每请求最后一行；被裁行是零值，绝不能进 logits）
```

挂点细节：循环里一旦 `layer.layer_idx == self.decoder_replay_start`（未启用时该值等于 `end_layer`，永不相等），剩余层栈交给适配器，`last_layer` 设为 `layers[end_layer-1]` 后 `break`——**hc_collapse/norm 与 MTP 缓冲写入都在全尺寸缓冲上**（L1612-1627），"裁行"对上游不可见。

---

## 四、行集合（rows）的构造

### 4.1 先算长度（`_build_replay_plan`，L258-295）

```python
trim_requests = batch.is_prefilling_np[:num_reqs] & ~self.keeps_rows[batch.idx_mapping_np[:num_reqs]]  # L275-276
if batch.is_dummy or inputs.pcp_context is not None:
    trim_requests = np.zeros(num_reqs, dtype=np.bool_)                                     # L277-279
kept_lens = np.where(trim_requests, np.minimum(query_lens, self.layers.window), query_lens)  # L280
requires_compaction = bool((kept_lens < query_lens).any())                                  # L281
num_tokens = batch.num_tokens_after_padding if not requires_compaction else int(kept_lens.sum())  # L283-287
```

三个约束在此定死：**只裁 prefill**（decode/verify 必须逐行）；**`keeps_rows` 优先**；**dummy 与"有 PCP 上下文但无 PCP 重放计划"一律不裁**（跨 rank 通信宽度不能由单 rank 私改）。`requires_compaction=False` 时 `num_tokens` 取 `num_tokens_after_padding` 而非真实 token 数，正是为保住宽度，含空 PCP rank 与空闲 DP 副本的 padding 行（L283-285 注释）。

### 4.2 DP 协调：两行 int32 的 all_reduce（L297-311）

`replica_info` 是 `(2, dp_size)` int32，每列只由该列 rank 写：row 0 = 是否压缩，row 1 = 执行长度（含 padding）；`all_reduce(dp.cpu_group)` 后每列即"某副本的真相"。`dp_tokens = replica_info[1]`（L310）作为 `num_tokens_across_dp`（L421）；返回 `replica_info[0].any()`（L311）⇒ **任一副本压缩，本 rank 就必须有动作**（L251-253）。必需性：DP 内 MoE all-to-all 是同一集合通信，一副本 4096 行、另一副本 128 行会打架；`dp_size == 1` 直接返回 `plan.requires_compaction`（L299-300）。

### 4.3 PCP 协调：全局紧凑批次 + `hidden_restore_idx`

PCP 下每请求 chunk 被切到多 rank，"最后 128 行"在全局是一条尾巴，可能落在不同 rank，故单独走 `_pcp_kept_lengths`（L83-117）：

| 步骤 | 做了什么 | 位置 |
|------|---------|------|
| 1 | 用**全局 batch** 算 `global_lens`、查全局 `keeps_rows`；`global_kept = min(global_lens, window)`，无请求被裁则返回 None | L88-97 |
| 2 | `num_padded = global_kept.sum()`；若 `>= num_tokens_after_padding` 返回 None（压了不如不压） | L98-100 |
| 3 | 全局起点 `seq_lens − window` 映到本 rank：`kept_lens = clip(seq_lens − local_start, 0, local_len)` | L102-110 |
| 4 | `replay_start = max(encoder_replay_start, pcp_start)` | L111-117 |

`_build_replay_pcp_context`（L119-171）把它落成**紧凑全局 batch**（`replace(global_batch, ...)`，L149-163），`global_slot_mappings` 按同批行 select（L167）。最要紧的是恢复索引：

```python
gathered_rows = get_pcp_group().all_gather(padded_rows.contiguous(), dim=0)   # L144
order         = torch.argsort(gathered_rows)                                   # L145
restore_idx   = order[torch.searchsorted(gathered_rows[order], global_rows)]   # L146
```

即"全局紧凑行第 i 行 → 它在 gather 后序列里的实际位置"；子批跑完后用 `torch.index_select(gathered, 0, metadata.hidden_restore_idx)` 摆回紧凑全局顺序（[context_parallel/dsa_v41_cp.py](../../../attention/context_parallel/dsa_v41_cp.py) L341；非重放路径为 `arange`，见 [pcp_manager.py](../../../worker/v2/pcp_manager.py) L690）。

### 4.4 device 端生成：`gather_replay_batch`

`_prepare_compact_replay`（L313-344）先算每请求边界累计丢掉多少行，再调 kernel：

```python
offsets = np.zeros(num_reqs + 1, dtype=np.int32); np.cumsum(plan.kept_lens, out=offsets[1:])  # L317-318
self.dropped_before.np[: num_reqs + 1] = batch.query_start_loc_np[: num_reqs + 1] - offsets     # L320-322
self._gather(plan.replay_inputs)                                                               # L326
```

`gather_replay_batch`（[decoder_replay.py](../../../ops/triton/decoder_replay.py) L69-115）一个 program 处理一个请求：

| 行为 | 规则 | 位置 |
|------|------|------|
| 输出行边界 | `kept_begin/kept_end` = 原边界减 `dropped_before` | L33-38 |
| `replay_start` | `max(seq_len − kept_len − history, 0)` 再与 `encoder_replay_start` 取大；`history = 0 if 被裁 else window−1` | L39-43 |
| 取哪几行 | **尾部**：`first_row = end − kept_len` | L45-55 |
| padding 行 | 最后一个 program 填 `rows=0`、`positions=positions[0]`、**`slot=−1`** | L57-66 |

两处防泄漏：padding 行 `slot=−1`（L66）——会被算（占住 collective 宽度）但**绝不写进任何 cache**；且复用真实第 0 行当输入（L63），让垃圾值有确定来源。

### 4.5 适配器侧：紧凑化、散回、补零（decoder_replay_layers.py L35-81）

```python
positions = states[2]                                    # L44：位置始终不分片
with self.context_factory():                             # L46：换上下文（§6）
    for buffer in self.row_buffers:                      # L48-51
        buffer[: rows.numel()].copy_(buffer.index_select(0, rows))    # 原地紧凑
        if self.num_padded_rows is not None: buffer[rows.numel(): self.num_padded_rows].zero_()
    state = state.index_select(0, rows)                  # L60
    if self.num_padded_rows is not None:                 # L63-65：PCP 补零
        state = torch.cat((state, state.new_zeros((padding, *state.shape[1:]))))
    row_outputs = self.run_layers(*replay_states)        # L67
    output = row_output.new_zeros((positions.shape[0], *row_output.shape[1:]))      # L75
    output.index_copy_(0, rows[: self.num_actual_rows], row_output[: self.num_actual_rows])  # L76
```

| 细节 | 为什么 |
|------|--------|
| `row_buffers` **原地**紧凑 | TopK/候选缓冲按行号寻址：21–23 层读第 20 层的 `topk_indices`，24/28/32/36 层写自己的 TopK，行号必须先与紧凑子批对齐（`index_select` 先分配临时张量，原地覆盖安全） |
| 输出 `new_zeros` + `index_copy_` | 只有 `num_actual_rows` 行是真值，其余为 0，配合 §6 的 `is_padding` 让 MoE/量化算子跳过 |
| `positions` 不分片（`index != 2`） | 位置是每 token 一个标量，必须与全局行布局一致 |
| `num_padded_rows` 只有 PCP 设 | 非 PCP 的 `rows` 已含 padding 行（L341）；PCP 只取真实行，故下游补齐 |

`sequence_parallel` 分支（L57-62、L73-79）当前不可达：SP 为真时 planner 直接关闭功能（L96-97），`replay.sequence_parallel` 恒为 False（L138）；它们是为"将来 SP 也支持重放"预留的布局。

---

## 五、元数据重建：为什么要重算 `AscendDSAV41Metadata`

### 5.1 不能复用原 step 的元数据

重放子批与原批次在四处不同：行数（`num_tokens_after_padding` → `num_tokens`）、真实行数（→ `num_actual_tokens`）、`query_start_loc`（每请求只剩尾巴）、`slot_mapping`（尾部真实 slot，padding 行 −1）。而 `AscendDSAV41Metadata` 是"调度坐标 + cache 平面坐标"的契约（[dsa_v41.py](../../../attention/dsa_v41.py) L139-148：`seq_lens` 保持原始 token 坐标，`cache_*` 描述 cache 平面可见行），且每 group 的 `block_table`/`slot_mapping` 不同，只能**每 group 各重建一份**。

### 5.2 `_build_metadata` 做的事（L346-390）

| 步骤 | 实现 | 位置 |
|------|------|------|
| 取 positions/slots | `self.positions[:num_tokens]`；PCP 时 `slots` 需 all_gather | L350-353 |
| 一次算 RoPE | `get_cos_and_sin_dsa(positions, use_cache=False)`（重放位置不在全局表里） | L354 |
| 每 group 组 common metadata | `query_start_loc=offsets`、`seq_lens=原 seq_lens`、`num_input_tokens=num_tokens`、`replay_start` | L359-377 |
| 交给 builder | `builder.build(0, common, ..., rope_views=rope)` | L378-385 |
| 跑设备端任务 + 扇出 | `take_device_metadata_tasks()` 逐条 run；同 group 层名共享同一对象 | L386-389 |

易忽略点：**`max_query_len` 不能变小**（L367 取 `max(np.diff(offsets).max(), batch.max_query_len or 0)`）——band metadata 按它分配行；**builder 跨步缓存**（`_initialize` L173-187 建一次并 `prepare_source_rope()`）。

### 5.3 `replay_start` 如何进入 SWA 的可见长度

```
① kernel 逐请求算 replay_start（decoder_replay.py L39-43）→ self.replay_start（device int32）
② 状态机塞进 common metadata：replay_start=self.replay_start[:num_reqs]（L374；
   字段语义见 attention/utils.py L426-432）
③ builder 只对 SWA group 生成 per-token 下界（dsa_v41.py L1239-1253）
④ 算子以 band 模式读 ori_topk_length：每 query token 的"可见 ori KV 条数"（L1326、L735）
```

`build_replay_swa_visible_lens`（L95-136）只有三行，但坐标必须与算子自己的推导同源（否则两套坐标系会漂移）：

```python
positions       = seq_lens[req_ids] - query_lens[req_ids] + token_offsets   # 与算子同源（L133）
lower           = replay_start[req_ids]                                    # L134
start_positions = max(positions - window + 1, lower).clamp_min(0)          # L135
return (positions - start_positions + 1).to(int32).unsqueeze(1)            # 可见条数
```

### 5.4 另一条通道：SWA cache 视图的重定基

builder 里还有一条**独立**的重定基路径（L1158-1177），由 `build()` 关键字参数（`replay_start` / `replay_seq_lens` / `replay_max_seq_len`）触发：`replay_seq_lens = (seq_lens − replay_start).clamp_min(0)`（L1162），再按 `replay_start + arange(capacity)` 取块表与块内偏移，得到 `replay_cache_blocks` / `replay_cache_offsets` / `replay_block_table`（L1169-1177）。消费端在 `_forward_attention`（L709-715）：重定基 SWA 缓存视图、替换 `ori_block_table` 与 `seq_lens`；注释写明"只重定基 SWA：A3 CSA 不能用稀疏 ori 索引掩盖缺失前缀"。

> **两者分工**：重定基解决"把窗口平移到 `replay_start` 起的物理页"，band 下界解决"逐 token 不许越过 `replay_start`"。`DecoderReplayState._build_metadata` 只经 `common.replay_start` 传下界（L374），**不**传 `replay_start=` 关键字（因此不生成 `replay_block_table`）：走"原始块表 + 逐 token 下界"，`seq_lens` 保持原始长度。带 `replay_max_seq_len` 的重定基通道由其它重放实现使用（[worker/v1/model_runner.py](../../../worker/v1/model_runner.py) L3474-3476、[worker/v2/model_states/default.py](../../../worker/v2/model_states/default.py) L309-311），改一处别假设另一处会跟着变。

两条路都受一个硬断言保护（dsa_v41.py L1136-1148）：掩码下界用的窗（text config）与回放填充用的窗（cache spec，经 `prefix_replay_tokens`）必须是同一个数，否则"填一个窗口、界另一个窗口"是静默精度损失而非报错。

---

## 六、forward context 替换

`forward_context`（decoder_replay_state.py L392-442）是 `@contextmanager`：进入时把父上下文改写成紧凑子批的上下文，退出时恢复它动过的共享缓冲。

| 字段 | 处置 | 位置 |
|------|------|------|
| `attn_metadata` | 紧凑时换 `self.metadata`，否则沿用 outer | L418 |
| `num_tokens` | `self.num_tokens`（含 padding） | L420 |
| `num_tokens_across_dp` | `self.dp_tokens`（§4.2 row 1） | L421 |
| `cudagraph_runtime_mode` | **强制 `NONE`** | L422 |
| `slot_mapping` | 紧凑时传 `None` | L423 |
| `is_padding` | 前 `num_actual_tokens` 行按 `rows` 从 outer 取，其余填 True | L396-413 |
| `mc2_mask` | 尾部 `fill_(False)`，再与 `~full_padding` 与；**退出时拷回** | L414-415、L430-438、L440-442 |
| `eplb_heat_collection_status` | 从 outer 透传 | L427-429 |

- **`is_padding`**：outer 的掩码按全尺寸 batch 打（MoE 跳过、量化空行）。子批只有 `num_actual_tokens` 行是真 token，其余必须标 padding，否则 MoE 会对零值隐藏态做真实路由；SP 下要先 all_gather 才能按全局行号索引（L399-400），再切回本 rank（L411-413）。
- **`mc2_mask`**：MC2 参与掩码是**共享持久缓冲**而非新张量，必须"进来改、出去还原"（L414-415 `clone()`、L440-442 `copy_`）。尾部一律不参与（L432）；`num_tokens` 之内再与 `~full_padding` 与一次（L438），把"外层就标 padding 的行"和"新产生的后缀 padding 行"一起排除（L433-435 注释）。
- **`slot_mapping=None`**：只关掉**通用路径**按全尺寸写缓存的入口。DSA 真正写 SWA 用 metadata 的 `slot_mapping`（dsa_v41.py L348-359、L394、L424），它来自 §4.4 的 kernel（padding 行 −1）。**通用入口关掉，DSA 入口按紧凑坐标写**。
- **`cudagraph_runtime_mode=NONE`**：即使外层是 FULL_DECODE_ONLY（decode 走图），prefill 重放也必须 eager；子批行数与元数据形状都是动态的。

---

## 七、正确性边界

### 7.1 三条依赖，缺一不可

1. **源完整性**：21–39 层读的第 20 层源缓存必须覆盖**全部** token——由"裁剪只在 21 层及以后"保证（decoder_replay_layers.py L121）；
2. **SWA 闭包**：窗口内任一行只依赖窗口内的行——由"保留**尾部** `window` 行"保证（decoder_replay.py L45）；
3. **可见性不越界**：窗口内任一行不许读到 `replay_start` 之下——由 per-token 下界保证（§7.2）。

### 7.2 为什么必须给 SWA 加 per-token 下界（null block 论证）

`build_replay_swa_visible_lens` 的 docstring（dsa_v41.py L102-125）把理由写得很直白：

> 上游 SWA 算子会对每个请求做一次夹取（`start_pos = max(start_pos, replay_start)`），这个夹取**不是优化而是正确性要求**：被重放的请求只重算命中点的最后窗口，所以滑动窗口组在 `replay_start` 之下**不持有自己的 KV**。那些块已被退休、替换成 **null block**，而上游明确记载 null block 的 KV 全为零。于是窗口读到的不是别的请求的数据，而是**幻影 key**：它们打分为 0，却在被重放 token 的 softmax 里占权重。

设 `window=128`、某请求被裁成尾部 128 行、`replay_start=3968`、`seq_len=4096`：

| 保留的行 | 位置 | 无下界时窗口 | 命中幻影位置数 | 有下界后可见条数 |
|---------|------|-------------|---------------|----------------|
| 第 0 行 | 3968 | 3841 … 3968 | 127 | **1** |
| 第 1 行 | 3969 | 3842 … 3969 | 126 | **2** |
| … | … | … | … | … |
| 第 127 行 | 4095 | 3968 … 4095 | 0 | **128** |

没有下界时，前 127 行的注意力分布塞满"零分但非零权重"的位置：`exp(0)=1` 与真实 key 的 `exp(score)` 同台竞争，输出**被系统性稀释**——不报错、不崩溃，精度悄悄掉。所以下界必须 per-token：第 0 行只能看 1 条、第 127 行要能看满 128 条，**没有单一标量能同时表达**；而算子的窗口宽度 `ori_win_left` 恰是**整个调用一个标量**（L1255）。下界因此改走另一条通道：`ori_topk_length` 在 band 模式下的语义正是"本 query token 能看多少条 ori KV，从自己的右边界往回数"（L1149-1154 注释），于是 `position − floor + 1` 正好是"保留以本行结尾的窗口、但不越过 floor"。

三个性质保证只变严不变松：只作用于 SWA group（`cache_kind == "swa"`，L1241；压缩 KV 仍用因果坐标，L1235-1238）；只可能收窄（窗口宽度仍是标量上界，L1231-1233）；只对重放请求生效（非重放 `replay_start=0`，下界 0 等价整段可见，L135 与 kernel L41-42）。

### 7.3 为什么只对"不返回 logits 的 prefill"裁剪

被裁行在全尺寸缓冲里隐藏态是**零**（适配器 L75 `new_zeros`），`pre_mix` 同样。它们能安全存在的唯一前提是**没有下游读者**，而读者只有三类：

| 读者 | 需要哪些行 | 保护机制 |
|------|-----------|---------|
| 采样 / logits | 每请求**最后一个** prompt token | 尾部窗口必含最后一行 |
| `prompt_logprobs` | 该请求**每一行** | `keeps_rows` |
| MTP / DSpark | 通常 target 最后一两行 | `_mtp_hidden_buffer` 取尾部（model.py L1614-1619）；草稿窗 ⊆ 目标窗（§2.2） |

`keeps_rows` 是"要不要逐行精确"的**按 request slot 布尔掩码**（decoder_replay_state.py L67），由 runner 在请求进入时填：

| 路径 | 写法 | 位置 |
|------|------|------|
| V4.1（EngramModelState，v2 默认） | `keeps_rows[req_index] = params.prompt_logprobs is not None` | [engram/model_state.py](../engram/model_state.py) L117-122 |
| v2 通用 model state | `keep_rows = worker.uses_prompt_logprobs[idx_mapping_np]` | [default.py](../../../worker/v2/model_states/default.py) L210-213 |
| v1 回退 | `keeps_rows = np.array([req in self.num_prompt_logprobs ...])` | [model_runner.py](../../../worker/v1/model_runner.py) L3376 |

三条判据一致：**只有 `prompt_logprobs` 是"必须逐行"的理由**。写进 slot 而非"请求 id 集合"，是因为 DP/PCP 下每个 rank 只知道自己那部分请求（L275 用 `idx_mapping_np` 映回 slot）。

### 7.4 与投机解码的相互约束

| 约束 | 方向 | 位置 |
|------|------|------|
| 只支持 DSpark 草稿 | 回放 → 草稿 | decoder_replay_layers.py L107-108 |
| 草稿必须有 `sliding_window`、`<= target_window`、草稿层全为 `sliding_attention` | 回放 → 草稿 | L109-117 |
| DSpark 的 SWA 索引复用 `ori_topk_length` 通道 | 草稿 → 回放（共享） | dsa_v41.py L1179-1189、L1222-1223 |

`ori_topk_length` 在 band 模式是"可见条数"、稀疏模式是"索引条数"，两个生产者（`build_dspark_swa_indices` 与 `build_replay_swa_visible_lens`）写同一字段，而 `ori_mask_mode` 恒为 band（L1249-1253 注释）。所以"重放 prefill"与"草稿 verify"不能指望同一份 metadata 表达两种语义——门禁要求草稿窗落在目标窗内，正是为了让两者下界共存。

### 7.5 缓存写入的边界

保留的真实行写到真实 slot（metadata 的 `slot_mapping` 来自 kernel 的 `slot_mappings_out`）；padding 行**不写**（slot = −1，decoder_replay.py L66；`scatter_cache_sk` 把 `[-1,-1]` 当跳过，dsa_v41.py L278-293）；被裁行**不写**（不在子批里，调度侧已先标 `PAD_SLOT_ID`，engram/model_state.py L204-216）。

---

## 八、性能模型

设 prompt 长度 `L`、窗口 `W=128`、重放层数 `D = end_layer − 1 − cut = 19`：`省掉 token-层 = D × (L − W)`，`重放段剩下 = D × W`，`整模型 ≈ 40L`（近似：每层每 token 代价等价）。

| 项（`L=32768`） | 数值 |
|----|------|
| 21–39 层不裁 | 19 × 32768 = **622,592** token-层 |
| 21–39 层裁剪后 | 19 × 128 = **2,432** token-层 |
| 省掉 | 19 × 32640 = **620,160** token-层（占重放段 **99.61%**） |
| 占整模型（40 层） | 620,160 / 1,310,720 = **47.31%** |

| prompt `L` | 省掉 token-层 | 占重放段 | 占整模型 |
|-----------:|-------------:|---------:|---------:|
| 8,192 | 153,216 | 98.44% | 46.76% |
| 32,768 | 620,160 | 99.61% | 47.31% |
| 131,072 | 2,487,936 | 99.90% | 47.45% |
| 1,048,576 | 19,920,512 | 99.99% | 47.49% |

- **"占重放段 99.6%"** 才是这层机制的实际上限：21–39 层的 token-层计算几乎被整体消掉。
- **"占整模型 47%"** 是上界中的上界，不是端到端加速比：① 0–20 层必须算全量，且 c2 压缩与 Indexer 比 21–39 层更贵，后者时间占比本就低于 19/40；② 剩下 128 行**延迟受限**，MoE 在 128 行上的时间不等于线性缩放；③ 每步固定成本——numpy diff/cumsum、2×dp int32 的 CPU `all_reduce`（L305-311）、每 group 一次 metadata 重建（L346-390）、一次 kernel。
- **什么时候白干**：`query_len <= W` 时 `kept_lens == query_lens`（L280）→ `requires_compaction=False`（L281）→ 早退（L244-246）；chunked prefill 的 chunk < 128 同理。

---

## 九、常见坑 / 排障

### 9.1 为什么 FULL graph 下不生效

| 图模式 | 结果 | 原因 |
|--------|------|------|
| `NONE` | 生效 | 全 eager，prefill 可自由换行集合 |
| `FULL_DECODE_ONLY` | 生效 | 只有 decode 走整图；prefill 的 runtime mode 是 `NONE` |
| `FULL` / `PIECEWISE` | **关闭**（planner 返回 None） | 重放要整体换掉 prefill 的 attention 元数据并在子批里强制 `cudagraph_runtime_mode=NONE`（L422），与"prefill 也被捕获进图"不兼容 |

即使 planner 通过，运行时还有第二道闸：`for_capture or mode != CUDAGraphMode.NONE or get_mrv2_in_profile_run()` 直接返回（decoder_replay_state.py L225-226）——**捕获步、profile 步、以及任何以图模式运行的步都不重放**。这是行为不是 bug：捕获形状必须与运行期一致，而重放行数是动态的。

### 9.2 为什么 LoRA / PP 直接关闭

- **LoRA**：`lora_config is not None` → 关，reason "LoRA token mappings need to follow the replay batch"（L118-119）。LoRA 的 token→adapter 映射按全 batch 行号注册，重放会整体重排行号，映射得先跟着重排，当前未实现。
- **PP > 1**：关（L98、L102）。21–39 层在 PP 下可能分属不同 stage，而"最后一个源层之后"是模型级判定。
- **SP / DCP / ubatching**：共用一条 reason（L96-102）。SP 切 token 轴（适配器的 gather/scatter 分支正是为它预留）；DCP 让"谁持有哪段 KV"随 rank 变；microbatch 需要每份 microbatch 各自的行集合。

### 9.3 排查清单（按日志 → 代码）

| 现象 | 先看哪里 | 结论 |
|------|---------|------|
| 没有 `Ascend MRV2 decoder bounded replay: layers ...` | `warning_once(... is disabled: %s.)`（L140） | reason 对应 §2.2 某一行 |
| warning 的 reason 是调度模式 | `SWA bounded replay is disabled: <mode> schedules with a pinned copy of the scheduler...`（[platform.py](../../../platform.py) L1585-1589） | **开关级**降级：连调度回退都没了，SWA 组重新可前缀缓存 |
| 启用了但没省时间 | 打点看 `kept_lens` 是否真小于 `query_lens` | `prompt_logprobs` 常开让 `keeps_rows` 全真（§7.3）；或 chunk ≤ 128 |
| DP 下部分 rank 不省 | `_coordinate_replay_across_dp` 的 `replica_info`（L305-311） | 只有 `requires_compaction` 的 rank 才裁；对端裁了本 rank 没裁是正常分支（L251-253） |
| 精度掉但不报错 | 查 `spec_window == window_size` 断言与 SWA 是否真拿到 `replay_start` | §5.3/§5.4 两条通道任一处漏传，都退化成"读幻影 key" |
| 想确认裁了多少 | v1 的 `logger.debug("Decoder bounded replay DP%d: %d -> %d tokens (%d padded).")`（worker/v1/model_runner.py L3405-3411）；DP 不同步时还有 `Decoder bounded replay requires synchronized Ascend DP execution modes`（L3348-3352） | V4.1 的 `DecoderReplayState` 无对应日志，需自己在 `_prepare_compact_replay` 后打 `num_tokens / num_actual_tokens` |

### 9.4 改这块代码时的四个"别动"

1. **`decoder_replay_start` 必须是 `cut + 1`**：早于最后源层会让源不完整，晚于它则少裁；`cut >= end_layer - 1`（L94-95）是"没有层可裁"的守卫。
2. **`keeps_rows` 的语义是"逐行精确"，不是"重要"**：只应由 `prompt_logprobs` 及同类逐 token 输出需求置真，置真太多等于关掉功能。
3. **per-token 下界只能收窄**：任何让 `ori_topk_length` 变大的改动都会把幻影 key 放回 softmax（§7.2）。
4. **`mc2_mask` 必须还原**：它是共享持久缓冲，忘记在 `finally` 里 `copy_` 回去（L440-442）会把"被裁剪的 padding 布局"泄漏到下一步与 decode 步。
