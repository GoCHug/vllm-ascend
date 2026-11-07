# DeepSeek V4 Attention 架构详解（MLA + DSA）

> **本文档讲解 DeepSeek V4 的注意力子系统：低秩 MLA 投影、SWA 滑动窗口、Compressor 历史压缩、Indexer 稀疏选块，以及它们在 Ascend NPU 上如何被打包进一个融合稀疏注意力算子。**

> **代码版本冻结声明**：本文基于 **vllm-ascend v0.26.0rc** 源码整理。模型侧封装见 [deepseek_v4.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py)（共 1565 行，`DeepseekV4Attention` 在 L729、`Indexer` 在 L546、`Compressor` 在 L613）；外层封装与模块袋见 [ops/dsa.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/ops/dsa.py)（`DSAModules` L40、`AscendDeepseekSparseAttention` L61）；NPU 后端与核心实现见 [attention/dsa_v1.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py)（`AscendDSABackend` L287、`AscendDSAImpl` L1725、`_forward_prefill` L2239、`_forward_decode` L2562）。文中 `Lxxx` 均指对应文件行号。

> **统一模型配置**（W4A8 版，本文相关行，TP 举例取 8）：

| 参数 | 值 | 含义 |
|------|-----|------|
| `hidden_size` | 7168 | 残差流宽度（`dim`） |
| `num_attention_heads` | 128 | 总 Query 头数；TP=8 时 `n_local_heads=16` |
| `head_dim` | 512 | 每头维度（MLA latent 宽度也是 512） |
| `qk_rope_head_dim` | 64 | 走 RoPE 的维度；剩余 `nope_head_dim=448` |
| `q_lora_rank` | 1536 | Query 低秩瓶颈 |
| `o_lora_rank` | 1024 | 输出低秩瓶颈 |
| `o_groups` | 16 | 输出分组数；TP=8 时 `n_local_groups=2` |
| `sliding_window` | 128 | SWA 窗口大小 |
| `index_n_heads` / `index_head_dim` | 64 / 128 | Indexer 的头数与头维度 |
| `index_topk` | 1024 | 每个 Query 选出的历史块数 |
| `rope_theta` / `compress_rope_theta` | 10000 / 160000 | dense 层 / 压缩层的 RoPE 基座 |

---

## 一、DSA 是什么：一套"眼前窗口 + 历史摘要 + 索引选块"的注意力

### 1.1 一句话定位

DeepSeek V4 的注意力叫 **DSA（Deep Sparse Attention，深度稀疏注意力）**，它在经典 **MLA（Multi-head Latent Attention，多头潜在注意力，低秩压缩 KV）** 的基础上，再叠加两级"记忆系统"：

> **生活化类比**：可以把注意力想象成"开卷考试答题"。
> - **SWA** = 摊在桌面上的**最近 128 个字**的原文，随时能看，精度最高；
> - **Compressor** = 你给更早的内容做的**章节摘要卡片**（每 4 个字压一张 / 每 128 个字压一张）；
> - **Indexer** = 书末的**索引**——答题前先在摘要卡片上快速打分，只把最相关的 **1024 张**卡片调出来细读，其余历史一律不看。
>
> 这样每个 token 真正做注意力的对象从"全部历史"变成"128 个原文 + 1024 张摘要"，在 1M 上下文下把计算量和显存都压了下来。

三种层按 `compress_ratio` 区分（配置数组 64 项的实测分布，详见 [5_deepseek_v4_kv_cache.md](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/dsv4_docs/5_deepseek_v4_kv_cache.md)）：

| `compress_ratio` | SWA | Compressor | Indexer | 层 |
|------------------|:---:|:----------:|:-------:|----|
| **0 = dense** | ✓ | ✗ | ✗ | 3 个 draft 层（数组下标 61~63） |
| **4 = c4** | ✓ | ✓（`overlap=True`） | ✓ TopK=1024 | 30 个主模型层 |
| **128 = c128** | ✓ | ✓（`overlap=False`） | ✗（全部摘要都看） | 31 个主模型层 |

> **注意（v0.26 的口径）**：dense 的取值是 **0 不是 1**，由 `get_dsv4_compress_ratio`（[utils.py:L103-L108](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/utils.py#L103)）在缺省/越界时返回 0。Compressor 构造器只接受 4/128，其他值在 L682-L684 抛 `ValueError`。

### 1.2 五层调用结构

整个注意力在代码里被拆成 5 层，职责从"装参数"一路下沉到"调 NPU 算子"：

```
┌────────────────────────────────────────────────────────────────────┐
│ 第1层  DeepseekV4Attention        deepseek_v4.py:L729              │
│   建权重(wq_a/wq_b/wkv/wo_a/wo_b/norm/attn_sink)、建压缩/索引模块、 │
│   把它们装进 DSAModules 数据袋，再 new 一个 dsa_attn；forward 透传  │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 第2层  AscendDeepseekSparseAttention  ops/dsa.py:L61               │
│   继承 MultiHeadLatentAttentionWrapper；从数据袋取出模块，          │
│   内部持有一个 DSAAttention，并把自己注册进 static_forward_context  │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 第3层  DSAAttention(基类)  models/layer/attention/layer.py:L53     │
│   AttentionLayerBase：管理 KV cache spec、持有后端 backend 与 impl  │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 第4层  AscendDSABackend        attention/dsa_v1.py:L287            │
│   AttentionBackend 实现：给出 impl_cls、metadata builder、KV 形状  │
│   配套 Metadata：Prefill L331 / Decode L369 / 合并 L408 / Builder L547│
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 第5层  AscendDSAImpl            attention/dsa_v1.py:L1725          │
│   真正干活：MLA 投影+RoPE、写 SWA、Compressor、Indexer TopK、       │
│   融合稀疏注意力、输出投影；prefill L2239 / decode L2562 两条路径   │
└────────────────────────────────────────────────────────────────────┘
```

### 1.3 文件索引

| 角色 | 文件 : 行 | 说明 |
|------|-----------|------|
| 模型封装 | [deepseek_v4.py:L729](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L729) | `DeepseekV4Attention`，建权重 + 组装 |
| Indexer | [deepseek_v4.py:L546](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L546) | 稀疏选块模块（Python `forward` 是空壳 L609） |
| Compressor | [deepseek_v4.py:L613](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L613) | KV 压缩模块（Python `forward` 是 `pass` L694） |
| 模块袋/外壳 | [ops/dsa.py:L40](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/ops/dsa.py#L40) | `DSAModules`、`AscendDeepseekSparseAttention` |
| 注意力层基类 | [layer.py:L53](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/layer/attention/layer.py#L53) | `DSAAttention` |
| Impl 基类 | [abstract.py:L18](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/abstract.py#L18) | `DSAAttentionImpl` |
| NPU 后端 | [dsa_v1.py:L287](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L287) | `AscendDSABackend` |
| 核心实现 | [dsa_v1.py:L1725](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L1725) | `AscendDSAImpl` |
| 算子分发表 | [device_op.py:L693](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/device/device_op.py#L693) | 按机型返回不同 NPU 算子 |
| CV 拆分 | [ops/cv_linear.py:L10](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/ops/cv_linear.py#L10) | `CVLinearWrapper`：量化/矩阵乘分离 |

---

## 二、第 1 层：DeepseekV4Attention（建权重 + 组装）

文件：[deepseek_v4.py:L729-L934](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L729)。它是一个普通 `nn.Module`，构造函数把这一层注意力需要的**全部权重和子模块建好**，`forward`（L928）只有一句话——透传给 `self.dsa_attn`。

### 2.1 层号与基础形状（L741-L759）

```python
layer_idx = int(prefix.split(sep=".")[-2])          # 从 prefix 解析运行时层号
self.layer_idx = layer_idx
config_layer_idx = extract_dsv4_layer_index(config, prefix)  # MTP 层映射成 61/62/63
tp_size = get_tensor_model_parallel_world_size()
self.dim = config.hidden_size                       # 7168
self.n_heads = config.num_attention_heads           # 128
self.n_local_heads = self.n_heads // tp_size        # TP=8 → 16
self.q_lora_rank = config.q_lora_rank               # 1536
self.o_lora_rank = config.o_lora_rank               # 1024
self.head_dim = config.head_dim                     # 512
self.rope_head_dim = config.qk_rope_head_dim        # 64
self.nope_head_dim = self.head_dim - self.rope_head_dim  # 448
self.n_groups = config.o_groups                     # 16
self.n_local_groups = self.n_groups // tp_size      # TP=8 → 2
self.window_size = config.sliding_window            # 128
self.scale = self.head_dim ** -0.5                  # 注意力 softmax 缩放
self.enable_dsa_cp = enable_dsa_cp()                # 是否开上下文并行
```

> **为什么要两个层号**：`layer_idx` 是运行时名字里的编号；`config_layer_idx` 是查 `compress_ratios` 数组用的编号。MTP 层运行时叫 `mtp.0`，经 `extract_dsv4_layer_index`（[utils.py:L80](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/utils.py#L80)）映射成 `61/62/63`，正好落在数组末尾 3 个 0（dense）上。

### 2.2 权重清单（shape 用统一配置 + TP=8）

**Query 路径（低秩压缩）**：

| 组件 | 类型 | 权重 shape | 作用 |
|------|------|-----------|------|
| `wq_a` | `ReplicatedLinear` | `(7168, 1536)` | 先压到低秩瓶颈，各 TP 卡各存一份 |
| `q_norm` | `RMSNorm(1536)` | `(1536,)` | 瓶颈上的归一化 |
| `wq_b` | `ColumnParallelLinear`（CP 时改 `ReplicatedLinear`，L773） | `(1536, 128*512)` | 升回多头；列并行，本地出 `16*512` |
| `q_norm_without_weight` | `RMSNorm(512, has_weight=False)` | 无权重 | Q 升维后按头做无参 RMSNorm |

**KV 路径（MLA 联合投影）**：

| 组件 | 类型 | 权重 shape | 作用 |
|------|------|-----------|------|
| `wkv` | `ReplicatedLinear` | `(7168, 512)` | 一次性投出 MLA latent（K/V 共享这 512 维） |
| `kv_norm` | `RMSNorm(512)` | `(512,)` | latent 归一化 |

> **MLA 省显存的关键**：不像传统注意力分别缓存完整的 K 和 V，MLA 只缓存一个 512 维的潜在向量，运行时再由后端展开成 K/V，KV cache 宽度近似减半。

**输出路径（低秩 + 分组）**：

| 组件 | 类型 | 权重 shape | 作用 |
|------|------|-----------|------|
| `wo_a` | `ColumnParallelLinear` | 输入 `128*512/16=4096`，输出 `16*1024` | 组内先降维；非 A5 且走特定 TP 路径时保留 ND 权重（L803） |
| `wo_b` | `RowParallelLinear` | `(16*1024=16384, 7168)` | 跨头/卡求和后投回残差流 |

**注意力 sink**（L761-L762）：

```python
attn_sink_heads = self.n_heads if self.enable_dsa_cp else self.n_local_heads
self.attn_sink = nn.Parameter(torch.empty(attn_sink_heads, dtype=torch.float32))
# CP 时每个 token 都要看到全部头 → (128,)；否则只存本地头 → (16,)；恒为 fp32
```

### 2.3 RoPE：按压缩比选基座与 rope 组（L814-L834）

```python
self.compress_ratio = get_dsv4_compress_ratio(config, config_layer_idx)  # 0/4/128
if self.compress_ratio > 1:                      # c4 / c128
    config.rope_parameters["rope_theta"] = config.compress_rope_theta    # 160000
    rope_groups = ["default", f"c{self.compress_ratio}"]  # 两段位置编码
else:                                            # dense
    config.rope_parameters["rope_theta"] = config.rope_theta             # 10000
    rope_groups = ["default"]
self.rotary_emb = ComplexExpRotaryEmbedding(..., rope_groups=rope_groups)
```

压缩层的历史摘要是"另一条时间轴"，所以要多一组 RoPE（`c4` 或 `c128`），用更大的 `compress_rope_theta=160000`。

### 2.4 挂载 Compressor / Indexer（L836-L858）

```python
self.compressor, self.indexer = None, None
if self.compress_ratio > 1:                      # 4 或 128 才建压缩器
    self.compressor = Compressor(..., head_dim=self.head_dim, ...)
    if self.compress_ratio == 4:                 # 只有 c4 才建索引器
        self.indexer = Indexer(...)
```

### 2.5 IndexCache：层间复用 TopK 索引（L860-L877）

V4 支持相邻 c4 层**复用同一套 TopK 索引**（省一次 Indexer 计算），通过 hf-overrides 开启：

```python
skip_topk = False
if self.compress_ratio == 4 and getattr(config, "use_index_cache", False) and ".mtp." not in prefix:
    compress_ratios = getattr(config, "compress_ratios", None) or []
    indexer_seq_idx = sum(1 for r in compress_ratios[:config_layer_idx] if r == 4)  # 前面第几个 c4 层
    pattern = getattr(config, "index_topk_pattern", None)   # 如 "FSFS"，F=算/S=复用
    freq = getattr(config, "index_topk_freq", 1)            # 每隔几层算一次
    if pattern is None:
        skip_topk = max(indexer_seq_idx - 1, 0) % freq != 0
    else:
        assert pattern[0] == "F"                            # 第一层必须自己算
        if 0 <= indexer_seq_idx < len(pattern):
            skip_topk = pattern[indexer_seq_idx] == "S"
```

- `skip_topk=False`：本层算 TopK 并写进共享的 `topk_indices_buffer`；
- `skip_topk=True`：本层不算，直接读 buffer 里前一个 c4 层留下的索引。
- MTP 层被显式排除（`".mtp." not in prefix`），因为 spec decode 在模型级共享 buffer，impl 内的引用会失效。

### 2.6 组装数据袋并创建外壳（L879-L926）

```python
k_dtype = torch.float8_e4m3fn if device == A5 else torch.bfloat16   # SWA KV 的存储 dtype
swa_cache_layer = AscendDeepseekV4SWACache(head_dim=512, window_size=128, dtype=k_dtype, ...)

dsa_modules = DSAModules(            # 把所有模块"打包成一个袋子"交给后端
    wq_a=..., q_norm=..., q_norm_without_weight=..., wq_b=...,
    wkv=..., kv_norm=..., wo_a=..., wo_b=..., attn_sink=...,
    indexer=self.indexer, compressor=self.compressor,
    swa_cache_layer=swa_cache_layer,
    topk_indices_buffer=topk_indices_buffer, skip_topk=skip_topk,
)
self.dsa_attn = AscendDeepseekSparseAttention(..., compress_ratio=self.compress_ratio,
                                              dsa_modules=dsa_modules, ...)
```

---

## 三、第 2 层：DSAModules 数据袋 + AscendDeepseekSparseAttention

### 3.1 DSAModules（[ops/dsa.py:L40-L58](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/ops/dsa.py#L40)）

就是一个 `@dataclass`，把一层注意力要用到的所有权重/子模块/缓存一次性打包，避免构造函数参数过长：

```python
@dataclass
class DSAModules:
    wq_a, q_norm, q_norm_without_weight, wq_b: Module   # Query 路径
    wkv, kv_norm: Module                                 # KV(MLA) 路径
    wo_a, wo_b: Module                                   # 输出路径
    attn_sink: Module                                    # 注意力 sink（fp32 参数）
    indexer: Module | None                               # 仅 c4
    compressor: Module | None                            # c4 / c128
    swa_cache_layer: Module                              # SWA KV 缓存
    topk_indices_buffer: torch.Tensor | None             # 层间复用的 TopK 缓冲
    indexer_rotary_emb: Module | None = None
    skip_topk: bool = False
```

### 3.2 外壳类（[ops/dsa.py:L61](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/ops/dsa.py#L61)）

`AscendDeepseekSparseAttention` 继承上游的 `MultiHeadLatentAttentionWrapper`，做三件事：

1. 从数据袋把模块平铺到自身属性（L99-L115）；
2. 内部 `self.dsa_attn = DSAAttention(...)`（L117-L150），把模块以 `kwargs` 传给基类，由基类去挑后端、建 impl；
3. 把自己登记进 `compilation_config.static_forward_context[prefix]`（L152-L155），供图编译期按层名取到该层实例（重名会抛错）。

---

## 四、第 3、4 层：基类、后端与 Metadata

### 4.1 DSAAttention 基类

[layer.py:L53](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/layer/attention/layer.py#L53) 的 `DSAAttention(nn.Module, AttentionLayerBase)` 负责与 vLLM 调度侧对接：声明本层需要哪几类 KV cache（SWA / c4 state / c128 state）、它们的 spec 和 block 尺寸，并持有后端返回的 `AscendDSAImpl`。block 尺寸表 `get_dsv4_block_sizes`（同文件 L32）按机型区分，细节见 KV 缓存文档。

### 4.2 AscendDSABackend（[dsa_v1.py:L287](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L287)）

`AttentionBackend` 的 Ascend 实现，告诉 vLLM：用哪个 impl（`AscendDSAImpl`）、用哪个 metadata builder、各类 KV cache 的形状/块大小。它本身不算数据，是"后端工厂 + 形状描述"。

### 4.3 四类 Metadata

| 类 | 行 | 作用 |
|----|----|------|
| `AscendDSAPrefillMetadata` | [L331](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L331) | 一次 prefill 批次里 SWA/compressor/indexer 各自的 block_table、cu_seqlens、sas_metadata |
| `AscendDSADecodeMetadata` | [L369](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L369) | decode 步对应的同类信息（单/少量 token） |
| `AscendDSAMetadata` | [L408](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L408) | 上面两者的统一容器，`forward` 里按阶段取 |
| `AscendDSAMetadataBuilder` | [L547](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L547) | 调度后构建 Metadata，并异步发起设备端 metadata 算子 |

> **metadata 是什么**：融合稀疏注意力算子需要知道"每个请求的 Q 有多长、KV 块在显存里怎么排、TopK 索引怎么对齐"。这些不是权重、也不是激活，而是随批次变化的"索引说明书"，由 builder 在 Python/设备端预先算好。

---

## 五、第 5 层：AscendDSAImpl 核心实现

文件：[dsa_v1.py:L1725](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L1725)。入口 `forward`（L2047）按 metadata 类型分流到 `_forward_prefill`（L2239）或 `_forward_decode`（L2562）。

### 5.1 构造时取出的关键成员（L1760-L1829）

```python
# MLA 权重
self.wq_a, self.wq_b, self.wkv = kwargs["wq_a"], kwargs["wq_b"], kwargs["wkv"]
self.q_norm, self.q_norm_without_weight, self.kv_norm = ...
# CV 包装：把一层 Linear 拆成 Vector(量化) + Cube(矩阵乘)，便于多流重叠
self.cv_wq_a = CVLinearWrapper(self.wq_a)
self.cv_wkv  = CVLinearWrapper(self.wkv)
self.cv_wq_b = CVLinearWrapper(self.wq_b)
self.wo_a, self.wo_b = kwargs["wo_a"], kwargs["wo_b"]
self.attn_sink = kwargs["attn_sink"]
self.multistream_dsv4_dsa_overlap = ascend_config.multistream_dsv4_dsa_overlap  # 默认 True
```

### 5.2 Indexer 参数（仅 c4，L1788-L1806）

> 注意：源码里 `inderxer_*` 是**原版拼写**（少一个 x），不是本文笔误。

```python
if self.indexer is not None:
    self.indexer_heads = self.indexer.n_heads          # 64
    self.inderxer_dim  = self.indexer.head_dim         # 128（源码拼写 inderxer）
    self.inderxer_wq_b = self.indexer.wq_b             # Indexer 自己的 Query 投影
    self.cv_inderxer_wq_b = CVLinearWrapper(self.inderxer_wq_b)
    self.weights_proj = self.indexer.weights_proj      # 每头内容偏置投影
    self.indexer_softmax_scale = self.inderxer_dim ** -0.5
    self.indexer_compress = self.indexer.compressor    # Indexer 内部那个小 Compressor
    self.indexcom_ape / _wkv / _wgate / _norm = ...    # 小压缩器的参数
    self.indexcom_head_dim, self.indexcom_rotate = ...
    self.index_topk = self.indexer.index_topk          # 1024
```

### 5.3 Compressor 参数（c4/c128，L1809-L1818）

```python
if self.compressor is not None:
    self.compressor_head_dim = self.compressor.head_dim   # 512
    self.compressor_overlap = self.compressor.overlap    # c4=True, c128=False
    self.compressor_rotate  = self.compressor.rotate     # 主压缩器 False
    self.compressor_ape / _wkv / _wgate / _norm / _norm_eps = ...
```

IndexCache 侧还有 `skip_topk`、`topk_indices_buffer`、`use_index_cache`（L1823-L1829），配套两个辅助方法：读 `_get_indexcache_topk_indices`（L1843）、写 `_update_indexcache_topk_indices`（L1851）。

### 5.4 CV 分离与多流重叠

`CVLinearWrapper`（[ops/cv_linear.py:L10](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/ops/cv_linear.py#L10)）把一次线性层拆成：

- **Vector 阶段**：量化（求缩放因子、整数量化），跑在 Vector 核；
- **Cube 阶段**：真正的矩阵乘，跑在 Cube 核。

当 `multistream_dsv4_dsa_overlap=True`（[ascend_config.py 默认开](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/ascend_config.py)），Query 路径与 KV 路径放进两条流，量化/写缓存可以和另一条流的矩阵乘重叠，用 event 做同步。

---

## 六、Compressor 与 Indexer 模块本身

这两个类定义在 deepseek_v4.py，但它们的 Python `forward` 一个是 `pass`、一个是 `return`（L609/L694）——**真正的计算都在 AscendDSAImpl 里直接吃参数调 NPU 算子**，模块类主要负责"持有权重 + 建状态缓存"。

### 6.1 Compressor（[L613-L726](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L613)）

```python
self.overlap = compress_ratio == 4          # c4=True, c128=False
self.coff = 1 + self.overlap                # c4 → 2，c128 → 1
self.ape = nn.Parameter(torch.empty(compress_ratio, coff * head_dim, dtype=fp32))  # 压缩位置编码
self.wkv   = ReplicatedLinear(dim, coff * head_dim, ...)   # 内容/门控两路 KV
self.wgate = ReplicatedLinear(dim, coff * head_dim, ...)
self.norm  = RMSNorm(head_dim, dtype=torch.float32)         # 压缩核只收 fp32 norm
```

| 项 | c4（ratio=4） | c128（ratio=128） |
|----|---------------|-------------------|
| `overlap` / `coff` | True / 2 | False / 1 |
| 投影输出宽度 | `2*512=1024` | `1*512=512` |
| state cache 宽度 `state_dim` | `2*coff*head_dim=2048`（kv_state+score_state） | `2*head_dim=1024` |
| state block size 取值 | block 表 `[0][2]`（L671） | block 表 `[0][3]`（L679） |

`overlap_transform`（L686）在 c4 下把当前块与前一块的状态错开拼到一起，实现"跨块边界不丢信息"；`rope_single`（L703）封装 `torch_npu.npu_rotary_mul`，对压缩状态做 RoPE/逆 RoPE。

### 6.2 Indexer（[L546-L610](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L546)）

```python
self.n_heads = config.index_n_heads          # 64
self.head_dim = config.index_head_dim        # 128
self.index_topk = config.index_topk          # 1024
self.wq_b = ReplicatedLinear(q_lora_rank, n_heads * head_dim, ...)   # (1536, 64*128)
self.weights_proj = ReplicatedLinear(hidden_size, n_heads, ...)      # (7168, 64)
# c4 才有自己的 k_cache：A5 用 fp8，非 A5 用 int8（L585）
self.k_cache = AscendDeepseekV4IndexerCache(head_dim=128, dtype=k_dtype, compress_ratio=4, ...)
self.compressor = Compressor(..., head_dim=128, rotate=True, ...)    # Indexer 内部小压缩器
```

Indexer 的工作：把 Query 低秩向量 `qr` 投成 64 个索引头，与"压缩后的历史摘要"打分，每头/每 token 选出 Top-1024 个历史块。

---

## 七、完整 Forward 数据流

### 7.1 总览

```
 hidden_states (num_tokens, 7168)
        │
        ▼
 阶段1 MLA Prolog：Query 流 ‖ KV 流（多流重叠）
   Q: wq_a→q_norm→wq_b→reshape(16头)→q_rms(无参)→partial RoPE
   KV: wkv→kv_norm→reshape(1头)→partial RoPE→写 SWA cache
        │
        ▼
 阶段2 Compressor（c4/c128）：wkv/wgate → 状态累积 → fp32 norm+RoPE → 写压缩 KV
        │
        ▼
 阶段3 Indexer（仅 c4）：算 Top-1024（skip_topk 时直接读 buffer）
        │
        ▼
 阶段4 融合稀疏注意力：Q 对「SWA 原文 + 选中的历史摘要」+ sink 一次算完
        │  attn_output (num_tokens, 16, 512)
        ▼
 阶段5 输出投影：逆 RoPE → 分组 reshape → wo_a → flatten → wo_b
        │
        ▼
 (num_tokens, 7168)
```

### 7.2 阶段 1：MLA Prolog

**Query 流**（shape，`T=num_tokens`，TP=8）：

| 步骤 | 操作 | 输出 shape |
|------|------|-----------|
| 1 | `cv_wq_a.vector/cube(hidden)` | `(T, 1536)` |
| 2 | `q_norm` | `(T, 1536)`（得 `qr`，Indexer 也复用它） |
| 3 | `cv_wq_b.vector/cube(qr)` | `(T, 16*512)` |
| 4 | `unflatten` | `(T, 16, 512)` |
| 5 | `q_norm_without_weight`（无参 RMSNorm） | `(T, 16, 512)` |
| 6 | partial RoPE（只转后 64 维） | `(T, 16, 512)` |

**KV 流**：`wkv`→`kv_norm` 得 `(T, 512)`，`view` 成 `(T, 1, 512)`，partial RoPE 后 scatter 进 SWA cache。两条流用 event 同步。

### 7.3 阶段 2：Compressor

实现在 `_forward_compressor`（[L1877](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L1877)），核心算子：

```python
compressed_kv = torch.ops._C_ascend.compressor(...)   # L1890
```

它把连续 `compress_ratio` 个 token 的 `wkv/wgate` 投影累积进 fp32 `state_cache`，到边界做 fp32 norm + RoPE，吐出 1 个压缩 KV 写入压缩 cache。配套 metadata 算子 `compressor_metadata_out`（L89）/ `compressor_metadata`（L142）负责算槽位映射。

### 7.4 阶段 3：Indexer TopK

仅 c4。两类算子配合（metadata 与计算分离）：

| 用途 | 算子 | prefill / decode 行 |
|------|------|--------------------|
| 生成选块 metadata | `torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata` | L1066 / L1326 |
| 真正选 TopK | `torch.ops._C_ascend.npu_quant_lightning_indexer_v2` | L2484 / L2792 |

```python
compress_topk_idxs, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(...)  # L2484/L2792
```

- `skip_topk=False`：算完后 `_update_indexcache_topk_indices` 写共享 buffer；
- `skip_topk=True`：`_get_indexcache_topk_indices` 直接取，不调选块算子。

> **版本提示**：旧资料里的 `npu_vllm_quant_lightning_indexer_metadata` 在 v0.26 已更名为 `npu_quant_lightning_indexer_v2_metadata`，并拆出独立的计算算子 `..._v2`。

### 7.5 阶段 4：融合稀疏注意力

算子不是写死的，而是按机型从 `DeviceOperator` 取（[device_op.py:L693/L1502](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/device/device_op.py#L693)）：

| 机型 | 稀疏注意力算子 |
|------|----------------|
| 非 A5（默认） | `torch.ops._C_ascend.npu_sparse_attn_sharedkv`（L695） |
| A5（KV 量化） | `torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv`（L1503，带 `kv_quant_mode/tile_size/rope_head_dim`） |

c4 prefill 的调用形态（[L2515-L2535](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L2515)）：

```python
attn_output = attn_op(
    q,                                  # (T, 16, 512)
    ori_kv=swa_kv_cache,                # SWA 原文 KV（块表寻址）
    cmp_kv=compress_kv_cache,           # 压缩历史 KV
    cmp_sparse_indices=compress_topk_idxs,  # 仅 c4：Top-1024 索引
    ori_block_table=...,                # SWA 块表
    cmp_block_table=...,                # 压缩 KV 块表
    cu_seqlens_q=..., seqused_kv=...,   # 变长边界
    sinks=self.attn_sink,               # 每头一个可学习 sink
    metadata=sas_metadata,
    softmax_scale=self.softmax_scale,
    cmp_ratio=4,
    ori_mask_mode=4,                    # 原文 KV：SWA 滑窗掩码
    cmp_mask_mode=3,                    # 压缩 KV：causal 因果掩码
    ori_win_left=..., ori_win_right=...,
    layout_q="TND", layout_kv="PA_ND",
)[0]                                     # → (T, 16, 512)
```

c128 没有 `cmp_sparse_indices`（L2536 起的 else 分支），即"所有历史摘要都参与"。`ori_mask_mode=4` / `cmp_mask_mode=3` 分别对应滑窗掩码与因果掩码。

### 7.6 阶段 5：输出投影

| 步骤 | 操作 | 输出 shape |
|------|------|-----------|
| 1 | 对 nope 部分做逆 RoPE，把 Q 转回与输出对齐的基 | `(T, 16, 512)` |
| 2 | reshape 成 `n_local_groups=2` 组 | `(T, 2, 4096)` |
| 3 | `wo_a`（列并行） | `(T, 2, 1024)` |
| 4 | flatten | `(T, 2*1024)`（全组拼回 `(T, 16*1024)` 口径） |
| 5 | `wo_b`（行并行，跨卡求和） | `(T, 7168)` |

---

## 八、Prefill 与 Decode 的差异

| 维度 | Prefill（`_forward_prefill` L2239） | Decode（`_forward_decode` L2562） |
|------|-------------------------------------|-----------------------------------|
| Query 长度 | 一次多个 token（变长） | 单 token（或少量 spec token） |
| Q layout | `TND` | `BND` |
| KV 写入 | 整段新 token 写入三类缓存 | 只写当前 token |
| Metadata | `AscendDSAPrefillMetadata`（L331） | `AscendDSADecodeMetadata`（L369） |
| Indexer | 成批选 TopK（L2484） | 增量选 TopK（L2792） |
| 注意力算子 | 同一 `attn_op`，传 cu_seqlens/变长掩码 | 同一 `attn_op`，传 decode 块表 |

两条路径共用同一套权重与 Compressor/Indexer 参数，区别只在"批量形状与索引方式"。

---

## 九、NPU 算子清单（v0.26 实测命名）

| 阶段 | 算子 | 位置 |
|------|------|------|
| 压缩 metadata | `_C_ascend.compressor_metadata_out` / `compressor_metadata` | dsa_v1.py L89 / L142 |
| 压缩计算 | `_C_ascend.compressor` | dsa_v1.py L1890 |
| 选块 metadata | `_C_ascend.npu_quant_lightning_indexer_v2_metadata` | L1066 / L1326 |
| 选块计算 | `_C_ascend.npu_quant_lightning_indexer_v2` | L2484 / L2792 |
| 稀疏注意力（非 A5） | `_C_ascend.npu_sparse_attn_sharedkv` | device_op.py L695 |
| 稀疏注意力（A5） | `_C_ascend.npu_kv_quant_sparse_attn_sharedkv` | device_op.py L1503 |
| SWA/压缩 KV scatter | `_C_ascend.npu_scatter_nd_update_sk` 等 | device_op.py L710 起 |
| RoPE | `torch_npu.npu_rotary_mul` | deepseek_v4.py L719 |

---

## 十、性能优化要点小结

| 手段 | 做了什么 |
|------|---------|
| **MLA 低秩** | Q 走 `7168→1536→多头`、O 走分组低秩；K/V 只存 512 维 latent，KV 显存近似减半 |
| **三级稀疏** | SWA 保 128 个原文 + Compressor 压历史 + Indexer 只选 1024 块，把长上下文注意力压成近常数规模 |
| **IndexCache** | 相邻 c4 层共享一套 TopK 索引，跳过整次选块计算 |
| **CV 分离** | 量化（Vector）与矩阵乘（Cube）拆开，分别占用不同运算单元 |
| **多流重叠** | Query 流与 KV 流并行，event 同步（`multistream_dsv4_dsa_overlap` 默认开） |
| **算子融合** | SWA+压缩 KV+sink 的注意力一次进 `npu_sparse_attn_sharedkv`，不落中间结果 |
| **低精度存储** | A5 上 SWA KV 用 `float8_e4m3fn`、Indexer K 用 fp8；非 A5 SWA 用 bf16、Indexer K 用 int8；attn_sink/norm/state 保持 fp32 |
| **TP / CP** | wq_b、wo_a、wo_b 张量并行；CP 时 wq_b 改复制、attn_sink 存全头 |

---

## 十一、一句话总结

DeepSeek V4 的注意力 = **MLA 低秩骨架** + **"SWA 眼前 128 字 + Compressor 历史摘要 + Indexer 只挑 1024 块"的三级稀疏记忆**；Python 侧（L729 的 `DeepseekV4Attention`）只负责建权重和打包 `DSAModules`，真正的投影、压缩、选块与融合注意力全部下沉到 [AscendDSAImpl](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L1725)，通过 `compressor` / `npu_quant_lightning_indexer_v2` / `npu_sparse_attn_sharedkv` 三组 NPU 算子在一次稀疏注意力里完成。
