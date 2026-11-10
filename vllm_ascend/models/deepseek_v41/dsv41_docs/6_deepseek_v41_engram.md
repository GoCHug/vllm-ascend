# DeepSeek V4.1 Engram 详解：n-gram 哈希记忆与非参数化残差注入

> **本文档讲解 DeepSeek V4.1 新增的 Engram 模块：16M 词表基数的 n-gram 哈希表、每层 3.84 亿行 × 256 维的大表如何分片与查表、门控如何只在该短语「与当前语境匹配」时写回残差流，以及 Ascend 上的异步重叠、DP×PCP 行交换与主机内存卸载。**

> **代码版本冻结声明**：本文基于本仓库 `vllm_ascend/models/deepseek_v41/engram/` 包整理。文件与关键位置：
> - [common.py](../engram/common.py)：`load_engram_rotation_block` L17、`engram_enabled` L31、`valid_engram_token_mask` L37、`engram_gate` L46。
> - [hash_state.py](../engram/hash_state.py)：`AscendEngramSlotCache` L31、`engram_dead_mask` L50、`create_engram_hash_state` L65、`AscendNgramHashState` L72、`forward` L79。
> - [embedding.py](../engram/embedding.py)：`AscendParallelEngramEmbedding` L51、`_allocate_weights` L140、`close_host_offload` L173、`bind_checkpoint` L203、`load_checkpoint` L217、`_load_into_storage` L268、`embed_gathered` L335、`wait_lookup` L360、`forward` L370、`lookup` L381、`engram_storage_dtype` L433、`preflight_engram_checkpoint` L449。
> - [layer.py](../engram/layer.py)：`AscendEngram` L15、`forward` L34。
> - [npu.py](../engram/npu.py)：`engram_cpu_offload` L40、`quantize_engram_rows` L51、`gather_dequantize_engram_int8` L72、`HostUvaBuffer` L156、`gather_dequantize_host_uva` L232、`SharedUvaBuffer` L288。
> - [parallel.py](../engram/parallel.py)：`resolve_dp_shared_memory` L27、`engram_gathered_num_tokens` L32、`gather_engram_hashes` L61、`exchange_engram_rows` L79。
> - [model_state.py](../engram/model_state.py)：`EngramModelState` L69（MRv2 路径）。
> - [model.py](../model.py)：`prepare_engram` L1110、`_can_overlap_engram_preparation` L1254、`can_capture_engram_producer` L1266、`captured_engram_inputs` L1282、`retire_engram_lookups` L1331、`prepare_engram_inputs` L1404、`prepare_engram_graph_inputs` L1502、层循环里的消费者等待 L1584。
>
> **训练语义的唯一可信参考**：[reference/engram.py](reference/engram.py)（184 行，官方最小实现）。本文的哈希与门控公式全部以它为准，Ascend 侧只是把它搬到设备上并做了分片/异步/卸载。文中 `Lxxx` 未带文件名时指 `engram/embedding.py`。

> **统一模型配置**（Engram 相关行）：

| 配置项 | 值 | 含义 |
|--------|-----|------|
| `engram_layer_ids` | `[1, 14]` | 只有层 1 与层 14 挂 Engram |
| `engram_num_embeddings` | `[384006168, 384016682]` | 两张表的行数（不相等！） |
| `engram_max_ngram_size` | 4 | 最大 4-gram（实际用 2/3/4-gram） |
| `engram_n_heads` | 8 | 每种 n-gram 大小分 8 个头 |
| `engram_head_dim` | 256 | 每个哈希列向量的宽度 |
| `engram_vocab_size` | 16000000 | 素数桶的下界基数（不是真实词表） |
| `engram_compressed_vocab_size` | 99092 | 压缩词表大小（哈希乘子由它决定） |
| `engram_pad_token_id` | 2 | 无效/越界位置的填充 id |
| `hc_mult` / `hidden_size` | 4 / 5120 | 残差流条数与宽度 |

---

## 一、Engram 是什么

### 1.1 一句话版本

> **Engram 是一份「按 n-gram 索引的短语记忆库」：看到当前 token 及其前 1~3 个 token 组成的短语，就去 3.84 亿行的表里取出 24 个 256 维向量，经一次投影得到「4 条 key + 1 条 value」，再用一个**归一化点积门控**决定要不要把 value 加到残差流上。**

它和 MoE 的类比最贴切：MoE 是「按语义路由到专家」，Engram 是「按字面串路由到记忆条目」。区别在于：

| | MoE | Engram |
|---|-----|--------|
| 索引方式 | 学习的权重投影 + Top-K | **哈希**（无参数、不可训练） |
| 参数 | 384 个专家 | **一张 384M × 256 的嵌入表**（层 1/14 各一张） |
| 写入位置 | 替换 FFN 输出 | **叠加到残差流**（4 条流各一份门控） |
| 是否影响历史 | 否 | 否，只影响当前 token |

> **生活化类比**：读到一个词「画蛇」，Engram 就是墙上的**成语词典**——它立刻翻出「添足」这一条。但词典不能乱念：`gate` 会衡量「这条成语和当前的上下文像不像」，像才念（门控接近 1），不像就几乎不念（门控接近 0）。而「画蛇」这种四字串是用**哈希**查的，不是用语义查的，所以它极快、极省参数，但对「同义不同形」不敏感——这正是为什么哈希之前要先做一次**正字法归一化**（见 §2.2）。

### 1.2 在模型里的位置

```
每个 DecoderLayer（残差流 = 4 条）:
┌──────────────────────────────────────────────────────────────┐
│  [仅层 1 / 层 14] Engram:                                     │
│     hidden_states (T, 4, 5120)                                │
│         + gate ⊙ value(1 条 5120)  ──► 写回 4 条流             │
├──────────────────────────────────────────────────────────────┤
│  hc_pre → input_layernorm → DSA 注意力 → hc_post              │
├──────────────────────────────────────────────────────────────┤
│  hc_pre → post_attention_layernorm → MoE → hc_post            │
└──────────────────────────────────────────────────────────────┘
```

模型 forward 里的挂载点（[model.py:L1584-L1610](../model.py#L1584)）：

```python
if layer.engram is not None and token_mask.numel():
    ...
    hidden_states[:n] = layer.engram(hidden_states[:n], lookup, active_mask,
                                     self.engram_rotation if self.engram_rotated else None)
hidden_states, pre_mix = layer(positions, hidden_states, pre_mix, None, input_ids=input_ids)
```

注意两点：Engram **在 hc_pre 之前**注入（它直接改 4 条流，而不是改单流），并且它**不参与** `pre_mix` 的延迟交接。

---

## 二、哈希：一个位置如何变成一个行号

### 2.1 三个层次

```
token id ──① 压缩词表映射──► compressed id (0..99091)
                                   │
                                   ▼
                    ② 逐层、逐 n-gram 大小、逐头的
                       乘子 + 滚动 XOR
                                   │
                                   ▼
                    ③ mod 素数（每头一个专属素数） + 偏移
                                   │
                                   ▼
                        表内行号（该头的桶区间内）
```

### 2.2 ① 压缩词表：让「同形不同写」落到同一行

`build_compressed_token_map`（[reference/engram.py:L17-L61](reference/engram.py#L17)）把 tokenizer 的每个 token 解码成字符串，再做一串归一化：

```python
normalizer = normalizers.Sequence([
    NFKC(), NFD(), StripAccents(), Lowercase(),
    Replace(r"[ \t\r\n]+", " "),          # 多空白压成一个
    Replace(r"^ $", sentinel),             # 单个空格用私用区字符占位
    Strip(),
    Replace(sentinel, " "),
])
```

归一化后**文本相同**的 token 合并成同一个 compressed id。于是 `" The"`、`"the"`、`"THE"`、`"the "` 全部落到同一行——这正是「按字面串查表」能泛化的原因。实测压缩后词表是 **99,092**（配置里的 `engram_compressed_vocab_size`，[reference/engram.py:L146](reference/engram.py#L146) 处有断言，对不上会直接报错，因为所有哈希乘子都派生自这个数）。

> **为什么是 99092 而不是 129280**：129280 是 token 总数，归一化后大量重复（大小写、前导空格、变体）。这个数字变了，整张表就「重新洗牌」了，所以官方把它写进 config 并在加载时断言。

### 2.3 ② 滚动 XOR：把 n-gram 压成一个整数

对每个位置，取它自己与前 `shift` 个 token（`shift = 0..max_ngram-1`），先做「死 token」掩码，再乘上该层的乘子并滚动异或（[reference/engram.py:L167-L184](reference/engram.py#L167)）：

```python
compressed = token_map[input_ids]                 # 压缩 id
blocked = (positions < shift) | (source == DEAD)  # 越过序列首 / 命中图像
tokens[shift] = where(blocked, pad_id, source)

products = tokens.unsqueeze(2) * multipliers      # [T, layers, max_ngram]
rolling  = products[..., 0]
for i in range(1, max_ngram):
    rolling = rolling ^ products[..., i]          # ← 第 i 步的结果就是 (i+1)-gram 的哈希
    hashes.append(rolling % primes[:, i-1])       # 每一步落进自己那组素数桶
```

要点：

| 项 | 值 | 说明 |
|----|-----|------|
| `multipliers` | 每层 `max_ngram=4` 个 | 由 `np.random.default_rng(10007 * layer_id)` 生成，**乘 2 加 1 保证奇数**（[reference/engram.py:L64-L83](reference/engram.py#L64)），并保证 `token_id * multiplier` 不溢出 int64 |
| 为何用 XOR | — | 2-gram 的结果是 3-gram 滚动值的中间态，天然「一个位置出 3 个 n-gram」 |
| 为何不同层不同乘子 | — | 让两张表（层 1 / 层 14）的哈希分布独立，避免同一短语在两层命中「同构」的桶 |

### 2.4 ③ 素数桶：让 24 个头互不越界

`EngramLayout`（[reference/engram.py:L86-L126](reference/engram.py#L86)）给每个 (层, n-gram 大小, 头) 分配一个**专属素数**作为模数，素数从 `engram_vocab_size - 1 = 15,999,999` 开始向上找，**全局不重复**：

```
层 1 ：3 个 n-gram 大小 × 8 头 = 24 个素数
层 14：3 个 n-gram 大小 × 8 头 = 24 个素数
                                        共 48 个互不相同的素数
```

每个头的桶区间是 `[offset_head, offset_head + prime_head)`，`offset` 是该层内前面所有头的素数之和（[reference/engram.py:L148-L153](reference/engram.py#L148) 用 `np.cumsum` 算 offsets）。这样 24 个头在**同一张表**里各占一段连续行，互不覆盖，也不需要额外索引结构。

于是每张表的行数 = 48 个素数里属于本层的 24 个之和 + 一点余量：

- 层 1：**384,006,168** 行
- 层 14：**384,016,682** 行

两者不相等，正是因为「后 24 个素数比前 24 个素数大一点」。

### 2.5 「历史」从哪来：`NgramHashState` + 槽位缓存

一个 4-gram 需要前 3 个 token 的 id。prefill 与 decode 是分批的，所以必须维护历史。上游的 `NgramHashState` 读两个东西（[hash_state.py:L5-L17](../engram/hash_state.py#L5)）：

1. `swa_cache_module.block_size`（构造时读一次）；
2. `swa_cache_module.kv_cache`（每次 `ensure_cache()` 读，用于按槽位维护历史）。

Ascend 的 `AscendEngramSlotCache`（[L31-L47](../engram/hash_state.py#L31)）就是把这个「SWA 缓存视图」适配给上游：块大小来自 `DSV4_BLOCK_SIZES`（**不是**调度器的逻辑块大小），`kv_cache` 在 profiling 阶段可能还没绑定，所以要容忍空/单元素 list。

两条路径（由 `use_slot_cache` 决定）：

| 路径 | 历史来源 | 需要 `block_table` | 运行器 |
|------|---------|:---:|------|
| **V1（MRv1）** | 写进 SWA 缓存的槽位历史（`_write_hash_cache_kernel`） | ✓ | `slot_mapping` + `block_table` |
| **V2（MRv2）** | 运行器提供的 `lookback_token_ids` 窗口（无槽位） | ✗ | `token_lookback_depth` |

`token_lookback_depth` 由 `deepseek_v41` 模型暴露给运行器（[model.py:L1771-L1776](../model.py#L1771)），运行器为每个请求维护**块起点之前的 `depth` 个 token**（[engram/model_state.py:L102-L108](../engram/model_state.py#L102) 分配 `lookback_token_ids`，初值 -1）。V2 路径下 `lookback_token_ids` 是必需的，缺失会直接报错（[model.py:L1190-L1193](../model.py#L1190)）：

```python
if lookback_token_ids is None:
    if not hash_state.use_slot_cache:
        raise RuntimeError("MRV2 Engram requires device lookback_token_ids")
```

### 2.6 死 token：图像区间不进 n-gram

`engram_dead_mask`（[L50-L62](../engram/hash_state.py#L50)）把两种图像哨兵都标成死：

```python
return (token_ids == image_token_id) | (token_ids == image_pad_token_id)
```

死 token 的处理是「**截断历史**」而不是「跳过该位置」：任何跨越死 token 的 n-gram 都退化成「只用更短的 n-gram」或者被 `pad_id` 填满（[reference/engram.py:L171-L175](reference/engram.py#L171)）。这样图像内容不会污染文本的短语记忆，也不会让模型把「图片里出现过的字」当成固定搭配。

---

## 三、大表：384M 行怎么放进 NPU

### 3.1 规模与分片

每张表 3.84 亿行 × 256 维：

| 存储 dtype | 每行字节 | 整表大小 | 每 TP rank（TP=8） |
|-----------|:---:|:---:|:---:|
| bfloat16 | 512 | ≈ 196 GB | ≈ 24.5 GB |
| int8 + fp32 group-32 scale | 256 + 32 | ≈ 110 GB | ≈ 13.9 GB |
| **mxfp8**（E4M3 + E8M0 scale） | 256 + 8 | ≈ 101 GB | ≈ 12.7 GB |

> 代码注释里说的「per-rank allocation below is 24-51 GiB」（[model.py:L1039-L1042](../model.py#L1039)）就是按 BF16 与量化版两种情形估算的，并且强调：**在分配之前**就要确认 checkpoint 可读（`preflight_engram_checkpoint`），否则「先花 24-51 GiB 分配、再发现读不了」是不可接受的失败模式。

分片规则（[embedding.py:L99-L111](../engram/embedding.py#L99)）：

```python
num_shards, head_rank = self._get_shard_info()      # dp_size==1 → (tp_size, tp_rank)；否则 (tp_size*dp_size, engram_head_shard_rank())
if self.n_hash_cols % num_shards:                    # 24 个头必须能均分
    raise ValueError("Engram requires uniform head shards: ...")
self.part_n_hash_cols = self.n_hash_cols // num_shards
self.head_start  = head_rank * self.part_n_hash_cols
self.vocab_start_idx = sum(head_sizes[: self.head_start])          # 行区间 = 该 rank 拥有的头的桶区间之和
self.vocab_end_idx   = sum(head_sizes[: self.head_start + self.part_n_hash_cols])
```

关键点：**分片是按「头」而不是按「行」均分**。因为每个头独占一段行区间，所以「哪个 rank 拥有哪些头」直接决定了「哪个 rank 需要装载哪些行」——行区间是连续的一段，不需要重排表。TP=8 时每个 rank 装 3 个头（24/8）。

`head_sizes` 的构造在 [model.py:L1048-L1049](../model.py#L1048)：

```python
head_sizes = tuple(size for order in self.engram_layout.primes[slot] for size in order)
```

即把 `[n-gram 大小][头]` 展平成一维的素数列表——与 `primes`/`offsets` 缓冲的排布顺序一致。

### 3.2 三种存储 dtype 与加载

`engram_storage_dtype`（[L433-L446](../engram/embedding.py#L433)）**只看表头**，不看 dense 层的量化配置：

```python
dtype = file.get_slice(key).get_dtype()
if dtype == "BF16":  return torch.bfloat16
if dtype in ("I8","INT8"): return torch.int8
if dtype in ("F8_E4M3","F8_E4M3FN"): return torch.float8_e4m3fn
```

加载时按源 dtype 走四条分支（`_load_into_storage`，[L268-L333](../engram/embedding.py#L268)）：

| 目标存储 | 源 dtype | 处理 |
|---------|---------|------|
| bf16 | BF16 | 直接按 64K 行一块流式拷 |
| mxfp8 | F8_E4M3(FN) | **位拷贝**（`.view(torch.uint8)`），scale 也位拷贝（E8M0 是指数，不能当数值 cast） |
| int8 | I8/INT8 | 值与 fp32 scale 一起拷 |
| int8 | F8_E4M3(FN) | 先按 group-32 反量化成 fp32，再 `quantize_engram_rows` 重新量化成 int8 |
| int8 | 其它 | 直接 fp32 量化 |

> **位拷贝的必要性**（[L294-L299](../engram/embedding.py#L294) 的注释）：mxfp8 的 scale 是 E8M0 格式，`torch.uint8` 存的**就是指数本身**，用数值 cast 会把它变成错的数。这也是为什么 `scale_dtype` 在 mxfp8 下是 `torch.uint8`（[L68](../engram/embedding.py#L68)）。

### 3.3 主机内存卸载：24 GB 放到 CPU 侧

当 `engram_cpu_offload(vllm_config)` 为真（[npu.py:L40-L48](../engram/npu.py#L40)），表不再放 NPU 显存，而是放**CANN 注册过的主机内存**，用「分块设备指针表」读：

| 类 | 用途 | 关键点 |
|----|------|--------|
| `HostUvaBuffer`（[npu.py:L156-L230](../engram/npu.py#L156)） | 每 rank 一份主机表 | `aclrtMallocHost` 分配 → 注册成 UVA → 设 `dummy_weight_value` 让 dummy 加载器跳过真实读 |
| `SharedUvaBuffer`（[npu.py:L288-L384](../engram/npu.py#L288)） | DP×PCP 共享同一份物理页 | 用 POSIX 共享内存 + `_madvise_hugepage`，只有一个 rank 真正写 |

`dp_shared_memory` 模式（[parallel.py:L27-L29](../engram/parallel.py#L27)）下 `self.dp_size = 1`（[embedding.py:L85](../engram/embedding.py#L85)）——因为本节点所有 DP/PCP 副本映射的是**同一份** TP 分片，各自查自己的 token，不再需要每步的 DP 查表集合通信。

两个工程细节值得记：

1. **`torch.zeros` 而不是 `torch.empty`**（[L145-L153](../engram/embedding.py#L145) 的注释）：引擎会在读 checkpoint **之前**跑一次 profiling forward，这次 forward 会真的执行 Engram 查表。若表是未初始化的垃圾值，会污染整条流，进而让同一次 forward 后面的 MoE 路由算出非法值。
2. **释放必须成对**（`close_host_offload`，[L173-L201](../engram/embedding.py#L173)）：参数是主机内存的**别名**，释放后若不把参数换成空张量，后续读者会顺着活张量读到已解除映射的地址。

### 3.4 加载前预检

`preflight_engram_checkpoint`（[L449-L490](../engram/embedding.py#L449)）在**任何表分配之前**检查：

1. 索引文件（`quant_model_weights.safetensors.index.json` / `model.safetensors.index.json`，[L239-L242](../engram/embedding.py#L239)）里有没有 `layers.{id}.engram.embed.weight`；
2. 索引指向的分片文件是否存在；
3. 若是 int8/mxfp8，`....embed.scale` 是否在索引里、文件里、且 mxfp8 的 scale 是不是 E8M0。

失败信息都写明「哪个层、哪个 key、期望什么、实际什么」，因为这类错误在 8 卡上排查代价极高。

---

## 四、前向：从 24 列到 1 条 value

### 4.1 形状链路

| 阶段 | 张量 | 形状 | dtype |
|------|------|------|-------|
| 哈希 | `hashes` | `(T, 2 层, 24)` | int32 |
| 聚合（DP×PCP） | `gathered` | `(slot × EDP, 24)` | int32 |
| 本地查表 | `out` | `(slot × EDP, 3 头, 256)` | bf16 |
| 行交换（AllToAll） | `out` | `(T, EDP × 3, 256)` | bf16 |
| TP 聚合（all_gather 头维） | `lookup` | `(T, 24, 256)` | bf16 |
| 投影 | `kv = wkv(lookup.flatten(1))` | `(T, 25600)` | bf16 |
| 拆分 | `key` / `value` | `(T, 4×5120)` / `(T, 5120)` | bf16 |
| 门控输出 | `hidden + gate ⊙ value` | `(T, 4, 5120)` | bf16 |

`wkv` 是一个 `ReplicatedLinear(6144, 25600)`（[layer.py:L23-L30](../engram/layer.py#L23)）：

```python
self.wkv = ReplicatedLinear((max_ngram_size - 1) * n_heads * head_dim,   # 3*8*256 = 6144
                            (hc_mult + 1) * dim,                          # 5*5120 = 25600
                            bias=False, quant_config=quant_config, prefix=f"{prefix}.wkv")
```

一次矩阵乘同时产出 **4 条 key + 1 条 value**——这是 Engram 唯一的可训练计算。

### 4.2 门控公式

`engram_gate`（[common.py:L46-L72](../engram/common.py#L46)）与 reference 实现（[reference/model.py:L350-L365](reference/model.py#L350)）逐行对应：

```python
original = hidden.float()                     # (T, 4, 5120)
rstd  = rsqrt(original.square().mean(-1) + eps)        # 每个 (token, 流) 一个统计量
rstd *= rsqrt(key.square().mean(-1) + eps)
dot   = (original * channel_weight * key).sum(-1) * rstd * dim**-0.5
magnitude = dot.abs().clamp_min(1e-6).sqrt()
gate  = sigmoid(where(signbit(dot), -magnitude, magnitude))   # 带符号的平方根再 sigmoid
gate  = gate.masked_fill(~token_mask.unsqueeze(-1), 0)        # 死 token 门控归零
return (hidden.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)
```

四个必须理解的细节：

1. **归一化是「每个 (token, 流) 各自」的**，不是 4 条流合起来算一个 `rstd`（reference 实现的注释特意强调 *NOT jointly over the copies*，[reference/model.py:L358](reference/model.py#L358)）。因为 4 条流是 4 个独立的表示，共用统计量会让门控互相干扰。
2. **`channel_weight = q_weight * k_weight`**（[layer.py:L53](../engram/layer.py#L53)）：两个 `(4, 5120)` 的 bf16 参数只以**乘积**形式出现，训练时是分开学的（类似「查询向量 × 键向量」的对角近似）。
3. **带符号平方根**（`copysign(sqrt(|dot|), dot)`）：这与训练 kernel 一致（reference 注释 *signed sqrt before the sigmoid, matching the training kernel*）。少了这一步，门控的饱和行为就和训练对不上。
4. **死 token 门控置 0**：图像区间的 token 既不贡献记忆，也不接收记忆。

### 4.3 融合 Triton 快路径

当「没有旋转基」且硬件支持 `ENGRAM_UNROTATED_GATE` 时，门控走单个 Triton 核（[layer.py:L42-L47](../engram/layer.py#L42)）：

```python
return fused_engram_gate(hidden_states, kv, self.q_weight, self.k_weight, token_mask, self.eps)
```

省掉 `split → float() → mean → rstd → einsum` 的一串小算子。否则走 `engram_gate` 的通用实现。

### 4.4 旋转基（Quarot）：一个必须精确还原的等价变换

部分 Ascend checkpoint 的权重是**旋转基**下的（Quarot 全局旋转）。Engram 的门控要求「原始基」下的 hidden 与 key，所以要先**反向旋转** hidden：

```python
original = (original.unflatten(-1, (-1, 32)) @ rotation_block.float().T).flatten(-2)   # common.py:L63-L64
```

`rotation_block` 是 32×32 的分块对角矩阵，整张大旋转是它的重复（[common.py:L17-L28](../engram/common.py#L17)）：

```python
block = rotation[:32, :32]
expected = torch.block_diag(*[block] * (hidden_size // 32))
if not torch.equal(rotation, expected):
    raise ValueError("Engram gate requires repeated block32 global rotation")
```

**只用 32×32 的块**这一事实让「还原」变成 O(dim) 的小矩阵乘，而不是 O(dim²)——这是 V4.1 能把门控做得足够便宜的原因之一。A5 的原生身份基下不需要旋转，`engram_rotated` 为 False（[model.py:L1082-L1089](../model.py#L1082)），直接走融合核。

---

## 五、查表：DP×PCP 的哈希聚合与行交换

### 5.1 为什么需要交换

表是按**头**分片的（TP×EDP）。一个 rank 只拥有 3 个头，但要为**所有** token 生成 24 个头的查表结果。于是：

```
① 每个 rank 有本 rank 的 token 的 24 个哈希 id
        ↓  gather（DP×PCP 组内 all_gather，pad 到统一 token slot）
② 每个 rank 拿到 DP×PCP 所有 rank 的哈希 id（[slot×EDP, 24]）
        ↓  本地查表（只查自己那 3 个头）
③ 每个 rank 产出 [slot×EDP, 3, 256]
        ↓  AllToAll（每个目标 rank 只收自己那段 token 的行）
④ 每个 rank 拿到 [T, EDP×3, 256]
        ↓  TP 组内 all_gather（头维）
⑤ 拿到 [T, 24, 256]，还原成 checkpoint 的头顺序
```

代码：`gather_engram_hashes`（[parallel.py:L61-L76](../engram/parallel.py#L61)）→ `embed_gathered`（[L335-L358](../engram/embedding.py#L335)）→ `exchange_engram_rows`（[parallel.py:L79-L99](../engram/parallel.py#L79)）。

### 5.2 两个「必须用固定 slot」的理由

`engram_gathered_num_tokens`（[parallel.py:L32-L58](../engram/parallel.py#L32)）决定聚合的 token slot：

```python
if 是 PD decode-recompute 调度:
    # 该调度可以跳过 DP 元数据 all_reduce，本地 token 向量只含本地计数，
    # 空闲 rank 或不同 graph bucket 的 rank 数值都不同 → 必须用配置推算
    return max(get_potential_max_tokens(),
               min(max_num_batched_tokens, max_num_seqs * query_len))
return dp_metadata.num_tokens_across_dp_cpu[...].max()
```

即：**多数情况下**用 DP 元数据里的真实最大值；但在「重计算解码」调度下，DP 元数据可能没同步，只能用配置推算一个**固定上界**。`gather_engram_hashes` 再用 `DEAD_ID` 把不足的行补齐——这是为了让集合通信的形状在各 rank 之间**恒等**，否则 ACL Graph 无法回放。

### 5.3 行交换为什么用 AllToAll 而不是 AllGather

`exchange_engram_rows` 的注释（[L79-L86](../engram/parallel.py#L79)）：

> *Send each padded DP × PCP token block directly to its owning rank … instead of materializing every destination's block as all-gather does.*

AllGather 会让每个 rank 都拿到全部 `EDP × slot` 行，显存与带宽都是 `O(EDP)`；AllToAll 只把「该目标 rank 的那段 token」发过去，带宽降到 `O(1)`。对 `[slot×EDP, 3, 256]` 这种每行 768 B 的张量，在 EDP=8/16 时差别很大。

### 5.4 本 rank 拥有的头之外的 id 怎么办

`lookup`（[L381-L404](../engram/embedding.py#L381)）只写自己的列；`_torch_lookup` 的 CPU 参考实现（[L407-L430](../engram/embedding.py#L407)）把「不属于本 rank 行区间」的 id 映射到 0 并**输出 0**：

```python
owned = (columns >= vocab_start_idx) & (columns < vocab_end_idx)
local = where(owned, columns - vocab_start_idx, 0)
...
out[:, :heads].copy_(where(owned.unsqueeze(-1), decoded, 0.0))
```

于是 AllToAll 之后，每个头只由「拥有它的那个 rank」贡献非零行，其余 rank 贡献 0——**用加法语义拼出完整的 24 头**，不需要额外的掩码。

---

## 六、异步与图捕获：把查表藏进主图

### 6.1 为什么要重叠

查表是**纯访存**（随机读 384M 行表），与主干的矩阵乘在资源上几乎正交。V4.1 让它跑在**辅助流**上，与主干的早期层并行。

### 6.2 三种运行模式

| 模式 | 触发 | 机制 | 代码 |
|------|------|------|------|
| **同步** | 默认 / eager / 不支持的并行布局 | `prepare_engram` 直接返回，`forward` 里同步用 | [L1470-L1472](../model.py#L1470) |
| **辅助流 + 普通 Event** | `multistream_engram_overlap` 开 + NONE 图模式 | `prepare_engram_inputs` 在辅助流上跑，消费者 `stream.wait_event` | [L1449-L1500](../model.py#L1449) |
| **图内捕获 + ExternalEvent** | FULL 图 + V2 生产者可捕获 | 哈希与查表**被捕获进图**，用 `ExternalEvent` 做跨图握手 | [L1266-L1330](../model.py#L1266)、[L1502-L1515](../model.py#L1502) |

`can_capture_engram_producer`（[L1266-L1280](../model.py#L1266)）的四个条件：MRv2 slotless 哈希、不开 SP、PP=1、所有表的 `dp_size==1 && tp_size==1`（即无需图内集合通信）。

### 6.3 消费者的「延迟等待」

`forward` 里等到**第一个真正的消费者**才 `wait_lookup`（[L1584-L1595](../model.py#L1584)），而不是在图入口就等：

```python
if engram_mask_ready_event is not None:
    AscendParallelEngramEmbedding.wait_lookup(engram_mask_ready_event, external=engram_graph_events)
    engram_mask_ready_event = None
if engram_pending is not None and layer.layer_idx in engram_pending:
    AscendParallelEngramEmbedding.wait_lookup(engram_pending[layer.layer_idx], ...)
```

层 1 之前还有一层 layer 0 可以算（layer 0 没有 Engram），这段时间正好被用来掩盖查表延迟。`wait_lookup`（[L360-L368](../engram/embedding.py#L360)）区分两种事件：

```python
if external:          # ExternalEvent：图内捕获的握手
    done.wait(stream); done.reset(stream)
else:                 # 普通 npu.Event
    stream.wait_event(done)
```

`ExternalEvent` 多一次 `reset` 是必须的：它由 CANN 维护、跨图生效，不 reset 会在下一次捕获时拿到陈旧状态。

### 6.4 退役与清理

`retire_engram_lookups`（[L1331-L1342](../model.py#L1331)）在输入复用之前 join 辅助流；`reset_events=True` 时把所有图事件的记录清掉——**失败的 forward 可能跳过某个捕获的 wait/reset**，不清理的话下一次生产者会复用脏事件。

---

## 七、显存与性能账本

### 7.1 显存

| 项 | 量级 | 说明 |
|----|------|------|
| 两张表（TP=8，mxfp8） | ≈ 2 × 12.7 GB ≈ **25 GB** | 主要开销 |
| 两张表（TP=8，bf16） | ≈ 2 × 24.5 GB ≈ **49 GB** | 不开量化时 |
| `q_weight`/`k_weight` | 2 × 4 × 5120 bf16 ≈ 80 KB | 可忽略 |
| `wkv` | 6144 × 25600 ≈ 157M 参数 | 与量化配置有关 |
| 查表输出缓冲 | `2 × max_tokens × 24 × 256 bf16` | `max_tokens=8192` 时 ≈ 200 MB |
| 掩码缓冲 | `max_tokens` bool | 可忽略 |

> `engram_cpu_offload` 能把表搬到主机内存（代价是查表带宽下降），这是 8 卡 64 GB 机器跑得起来的前提。

### 7.2 时间

| 阶段 | 特点 | 优化 |
|------|------|------|
| 哈希 | 纯整数运算，`num_warps=4`，一个 kernel 覆盖所有层（[hash_state.py:L134-L172](../engram/hash_state.py#L134)） | 请求索引搜索被**单独拆成一个 kernel**（`_engram_req_index_kernel`，L105-L114），因为 Triton-Ascend 编译不了「搜索 + 哈希」合并体 |
| 查表 | 随机访存，行宽 256 B（mxfp8） | 主机 UVA + 大页（`_madvise_hugepage`）；DP 共享内存避免重复装载 |
| 投影 | 一次 6144×25600 GEMM | 与主干可量化融合 |
| 门控 | 逐元素 + 4 次规约 | `fused_engram_gate` 单核 |

> **Triton-Ascend 的编译器限制**是本模块里唯一「为编译器折腰」的地方（[hash_state.py:L72-L77](../engram/hash_state.py#L72) 的注释）：上游是一个 kernel 里先搜请求再哈希，Ascend 侧必须拆成两个 launch，但**仍然一次覆盖所有层**。

---

## 八、常见坑

| # | 坑 | 症状 | 位置 |
|---|----|------|------|
| 1 | 压缩词表大小与 config 不符 | 加载即断言失败（`vocab_size == engram_compressed_vocab_size`） | [reference/engram.py:L146](reference/engram.py#L146) |
| 2 | mxfp8 的 scale 被数值 cast 而不是位拷贝 | 查出来的值差几个数量级 | [embedding.py:L294-L299](../engram/embedding.py#L294) |
| 3 | DP 空闲 rank 跳过哈希 | 集合通信挂死 | `participates` 分支（[model.py:L1162-L1164](../model.py#L1162)） |
| 4 | 用 `torch.empty` 建表 | profiling forward 后 MoE 路由异常 | [embedding.py:L145-L171](../engram/embedding.py#L145) |
| 5 | 忘记还原旋转基 | 门控整体偏移、长文本退化 | [common.py:L63-L64](../engram/common.py#L63) |
| 6 | 图像 token 未排除 | 图文混排时文本记忆被污染 | [hash_state.py:L50-L62](../engram/hash_state.py#L50) |
| 7 | 释放主机表后参数未置空 | 偶发段错误 | [embedding.py:L173-L201](../engram/embedding.py#L173) |
| 8 | 头数不能被 TP×EDP 整除 | 构造即报错（要求 uniform head shards） | [embedding.py:L101-L106](../engram/embedding.py#L101) |
| 9 | FULL 图捕获时未 prime ExternalEvent | 回放时首次 wait 超时/挂死 | [model.py:L1380-L1385](../model.py#L1380) |
| 10 | `dp_shared_memory` 与 `use_slot_cache` 混用 | 查表拿到别人的 token | `gather_engram_hashes` 的 `dp_shared_memory` 分支（[parallel.py:L68-L69](../engram/parallel.py#L68)） |

---

## 九、FAQ

**Q1：为什么两张表的行数不一样？**
A：素数桶。每层的 24 个头各占一段连续行，行数是这 24 个素数之和。层 14 抽到的素数比层 1 大一点，所以多 10,514 行。

**Q2：Engram 会不会让「生僻短语」过拟合？**
A：它的门控是归一化点积，只有当 hidden 的方向与 key 方向接近时才放大。生僻短语的 key 与当前上下文不匹配时，`dot` 很小 → `sqrt` 后被压缩 → gate 接近 0.5 以下，贡献很小。真正的风险是**记忆条目被错误召回**，这由训练时的门控分布控制。

**Q3：`engram_layer_ids` 能不能改？**
A：不能。表是按层训练的（层 1 与层 14 的乘子/桶完全不同），且 `engram_num_embeddings` 与层 ID 一一对应。

**Q4：为什么门控要做「带符号平方根」？**
A：这是训练 kernel 的形式。`sigmoid` 的输入如果直接用 `dot`，大 `|dot|` 会立刻饱和到 0/1，梯度消失；先取 `sqrt(|dot|)` 再保留符号，相当于把门控的**动态范围压缩**，让中间区域更宽。改了它，数值就和训练对不上。

**Q5：主机卸载后查表慢多少？**
A：取决于 PCIe 与主机内存带宽。查表是随机读、每次 256 B，UVA 下每行一次 DMA 明显更慢；因此 Ascend 侧把「DP×PCP 共享同一份表」做成默认选项（`dp_shared_memory`），至少保证同一份表在节点内只装一次。

**Q6：`preflight_engram_checkpoint` 为什么不直接读表？**
A：读表要先把 24-51 GiB 分配出来。预检只读**索引 + safetensors 头**，代价是几个 KB，却能在「分配之后才发现读不了」之前失败。

---

*相关文档：[0_deepseek_v41_arch.md](0_deepseek_v41_arch.md)（Engram 在整体架构中的位置）、[3_deepseek_v41_attn.md](3_deepseek_v41_attn.md)（SWA 槽位缓存的构造，`AscendEngramSlotCache` 复用它）、[9_deepseek_v41_decoder_replay.md](9_deepseek_v41_decoder_replay.md)（Engram 层必须位于最后一个 KV 源之前的约束）。*
