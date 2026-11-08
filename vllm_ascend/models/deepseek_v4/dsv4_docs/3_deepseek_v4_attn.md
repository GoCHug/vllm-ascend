# DeepSeek V4 Attention 架构详解（MLA + DSA）

> **本文档讲解 DeepSeek V4 的注意力子系统：低秩 MLA 投影、SWA 滑动窗口、Compressor 历史压缩、Indexer 稀疏选块，以及它们在 Ascend NPU 上如何被打包进一个融合稀疏注意力算子。**

> **代码版本冻结声明**：本文基于 **vllm-ascend v0.32.0**（分支 `dsv4-v0.32.0`）与 **vLLM v0.32.0** 源码整理。模型侧封装已拆分为多个文件：[model.py](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py)（`DeepseekV4Attention` 在 L483）、[compressor.py](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/compressor.py)（`Compressor` 在 L105）、[indexer.py](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/indexer.py)（`DeepseekV4Indexer` 在 L297）。外层封装与模块袋见 [ops/dsa.py](../../../../../vllm-ascend/vllm_ascend/ops/dsa.py)（`DSAModules` L39、`AscendDeepseekSparseAttention` L57）；NPU 后端与核心实现见 [attention/dsa_v1.py](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py)（`AscendDSABackend` L272、`AscendDSAImpl` L2270、`forward` L2573、`_forward_attention` L2968）。文中 `Lxxx` 均指对应文件行号。

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

三种层按 `compress_ratio` 区分（配置数组 64 项的实测分布，详见 [5_deepseek_v4_kv_cache.md](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/dsv4_docs/5_deepseek_v4_kv_cache.md)）：

| `compress_ratio` | SWA | Compressor | Indexer | 层 |
|------------------|:---:|:----------:|:-------:|----|
| **0 = dense** | ✓ | ✗ | ✗ | 3 个 draft 层（数组下标 61~63） |
| **4 = c4** | ✓ | ✓（`overlap=True`） | ✓ TopK=1024 | 30 个主模型层 |
| **128 = c128** | ✓ | ✓（`overlap=False`） | ✗（全部摘要都看） | 31 个主模型层 |

> **注意（v0.32.0 的口径）**：dense 的取值是 **0 不是 1**，由 `get_dsv4_compress_ratio`（[utils.py:L122-L127](../../../../../vllm-ascend/vllm_ascend/utils.py#L122)）在缺省/越界时返回 0。Compressor 构造器只接受 4/128，其他值在 [compressor.py:L181-L183](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/compressor.py#L181) 抛 `ValueError`。

### 1.2 五层调用结构

整个注意力在代码里被拆成 5 层，职责从"装参数"一路下沉到"调 NPU 算子"：

```
┌────────────────────────────────────────────────────────────────────┐
│ 第1层  DeepseekV4Attention        models/deepseek_v4/model.py:L483 │
│   建权重(wq_a/wq_b/wkv/wo_a/wo_b/norm/attn_sink)、建压缩/索引模块、 │
│   把它们装进 DSAModules 数据袋，再 new 一个 dsa_attn；forward 透传  │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 第2层  AscendDeepseekSparseAttention  ops/dsa.py:L57               │
│   继承 MultiHeadLatentAttentionWrapper；从数据袋取出模块，          │
│   持有 DSAAttention，并把自己注册进 static_forward_context；        │
│   forward 走 torch.ops.vllm.dsa_forward（L163）                    │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 第3层  DSAAttention(基类)  models/layer/attention/layer.py:L68     │
│   AttentionLayerBase：按 compress_ratio 选后端、管理 KV cache spec  │
│   持有后端 backend 与 impl（AscendDSAImpl）                         │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 第4层  AscendDSABackend        attention/dsa_v1.py:L272            │
│   AttentionBackend 实现 + 6 个角色子类：C4 L354 / C128 L370 /       │
│   SWA L386 / C4State L400 / C128State L414 / Indexer L430           │
│   配套 Metadata：Req L465 / Batch L502 / Layer L530                 │
│   配套 Builder：通用 L755 + State L1760 / Attention L1780 /        │
│   SWA L2119 / Indexer L2164                                         │
└───────────────────────────────┬────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ 第5层  AscendDSAImpl            attention/dsa_v1.py:L2270          │
│   真正干活：MLA 投影+RoPE、写 SWA、Compressor、Indexer TopK、       │
│   融合稀疏注意力、输出投影；forward L2573 → _forward_attention L2968│
│   MLA prolog 单流 L2660 / 多流 L2746；压缩+选块 L2887               │
└────────────────────────────────────────────────────────────────────┘
```

### 1.3 文件索引

| 角色 | 文件 : 行 | 说明 |
|------|-----------|------|
| 模型封装 | [model.py:L483](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py#L483) | `DeepseekV4Attention`，建权重 + 组装 |
| Indexer | [indexer.py:L297](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/indexer.py#L297) | 稀疏选块模块，`forward` L437 真正执行 |
| Indexer 算子面 | [indexer.py:L198](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/indexer.py#L198) | `AscendIndexerOps`：量化/写缓存/选 TopK |
| Indexer cache | [indexer.py:L136](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/indexer.py#L136) | `AscendDeepseekV4IndexerCache` |
| Compressor | [compressor.py:L105](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/compressor.py#L105) | KV 压缩模块，`forward` L201 调 NPU 压缩算子 |
| 模块袋/外壳 | [ops/dsa.py:L39](../../../../../vllm-ascend/vllm_ascend/ops/dsa.py#L39) | `DSAModules`、`AscendDeepseekSparseAttention` |
| 注意力层基类 | [layer.py:L68](../../../../../vllm-ascend/vllm_ascend/models/layer/attention/layer.py#L68) | `DSAAttention` |
| Impl 基类 | `vllm.v1.attention.backend` | `AttentionImplBase`（上游类，非本仓库文件） |
| NPU 后端 | [dsa_v1.py:L272](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L272) | `AscendDSABackend` + 角色子类 |
| 核心实现 | [dsa_v1.py:L2270](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L2270) | `AscendDSAImpl` |
| KV 执行计划 | [dsa_attn_kv_plan.py:L139](../../../../../vllm-ascend/vllm_ascend/attention/dsa_attn_kv_plan.py#L139) | `DsaAttnKvPlan`：按机型/dtype 选算子与布局 |
| 算子选择入口 | [dsa_attn_kv_plan.py:L221](../../../../../vllm-ascend/vllm_ascend/attention/dsa_attn_kv_plan.py#L221) | `get_dsa_attn_kv_plan` |
| RoPE | [rope_dsv4.py:L233](../../../../../vllm-ascend/vllm_ascend/ops/rope_dsv4.py#L233) | `ComplexExpRotaryEmbedding`（default + c4/c128 组） |
| 算子分发表 | [device_op.py:L658](../../../../../vllm-ascend/vllm_ascend/device/device_op.py#L658) | 按机型返回不同 NPU 算子 |
| CV 拆分 | [cv_linear.py:L13](../../../../../vllm-ascend/vllm_ascend/ops/cv_linear.py#L13) | `CVLinearWrapper`：量化/矩阵乘分离 |

---

## 二、第 1 层：DeepseekV4Attention（建权重 + 组装）

文件：[model.py:L483-L688](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py#L483)。它是一个普通 `nn.Module`，构造函数把这一层注意力需要的**全部权重和子模块建好**，`forward`（L682）只有一句话——透传给 `self.dsa_attn`。

### 2.1 层号与基础形状（L496-L513）

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
self.eps = self.norm_eps = config.rms_norm_eps
self.scale = self.head_dim ** -0.5                  # 注意力 softmax 缩放
```

> **为什么要两个层号**：`layer_idx` 是运行时名字里的编号；`config_layer_idx` 是查 `compress_ratios` 数组用的编号。MTP 层运行时叫 `mtp.0`，经 `extract_dsv4_layer_index`（[utils.py:L99](../../../../../vllm-ascend/vllm_ascend/utils.py#L99)）映射成 `61/62/63`，正好落在数组末尾 3 个 0（dense）上。

### 2.2 权重清单（shape 用统一配置 + TP=8）

**Query 路径（低秩压缩）**：

| 组件 | 类型 | 权重 shape | 作用 |
|------|------|-----------|------|
| `wq_a` | `ReplicatedLinear` | `(7168, 1536)` | 先压到低秩瓶颈，各 TP 卡各存一份 |
| `q_norm` | `RMSNorm(1536)` | `(1536,)` | 瓶颈上的归一化 |
| `wq_b` | `ColumnParallelLinear` | `(1536, 128*512)` | 升回多头；列并行，本地出 `16*512` |
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
| `wo_a` | `ColumnParallelLinear` | 输入 `128*512/16=4096`，输出 `16*1024` | 组内先降维；权重强制保持 ND（L555） |
| `wo_b` | `RowParallelLinear` | `(16*1024=16384, 7168)` | 跨头/卡求和后投回残差流 |

> **ND 权重约束**：所有 DSA o_proj 路径都用 `npu_transpose_batchmatmul` / `npu_transpose_quant_batchmatmul` 直接读 `wo_a.weight`，所以 `self.wo_a.skip_weight_nz_conversion = True`（L555），不再有"特定 TP 才保留 ND"的分支。

**注意力 sink**（L515）：

```python
self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
# 每个本地头一个可学习 sink，恒为 fp32
```

### 2.3 RoPE：按压缩比选基座与 rope 组（L565-L591）

```python
self.compress_ratio = get_dsv4_compress_ratio(config, config_layer_idx)  # 0/4/128
if self.compress_ratio > 1:                      # c4 / c128
    rope_theta = config.compress_rope_theta      # 160000
    rope_groups = ["default", f"c{self.compress_ratio}"]  # 两段位置编码
else:                                            # dense
    rope_theta = config.rope_theta               # 10000
    rope_groups = ["default"]

yarn_rope_parameters = config.rope_parameters
compress_rope_parameters = yarn_rope_parameters.get("compress")
if isinstance(compress_rope_parameters, dict):   # 嵌套 compress 组优先
    yarn_rope_parameters = compress_rope_parameters

self.rotary_emb = ComplexExpRotaryEmbedding(
    ..., head_size=self.rope_head_dim, rotary_dim=self.rope_head_dim,
    is_neox_style=False,
    scaling_factor=yarn_rope_parameters.get("factor", 1.0),
    base=rope_theta,
    beta_fast=yarn_rope_parameters.get("beta_fast", 32),
    beta_slow=yarn_rope_parameters.get("beta_slow", 1),
    rope_groups=rope_groups,
)
```

压缩层的历史摘要是"另一条时间轴"，所以要多一组 RoPE（`c4` 或 `c128`），用更大的 `compress_rope_theta=160000`。`ComplexExpRotaryEmbedding`（[rope_dsv4.py:L233](../../../../../vllm-ascend/vllm_ascend/ops/rope_dsv4.py#L233)）按 `config_key`（rotary_dim/base/YaRN 参数）全局缓存 cos/sin 表，每个 `rope_groups` 条目各自预计算一套频率；层内只注册非持久 buffer，sleep 模式可原地备份恢复。

### 2.4 挂载 Compressor / Indexer（L593-L632）

```python
self.compressor: Compressor | None = None
self.indexer: DeepseekV4Indexer | None = None

if self.compress_ratio > 1:                      # 4 或 128 才建压缩器
    self.compressor = Compressor(..., head_dim=self.head_dim, ...)
    if self.compress_ratio == 4:                 # 只有 c4 才建索引器
        self.indexer = DeepseekV4Indexer(
            ..., skip_topk=skip_topk, use_index_cache=use_index_cache,
            topk_indices_buffer=topk_indices_buffer, ...)
```

### 2.5 IndexCache：层间复用 TopK 索引（L596-L608）

V4 支持相邻 c4 层**复用同一套 TopK 索引**（省一次 Indexer 计算），通过 hf-overrides 开启。判定逻辑已抽到 `dsv4_skips_indexer_topk`（[utils.py:L130](../../../../../vllm-ascend/vllm_ascend/utils.py#L130)）：

```python
use_index_cache = getattr(config, "use_index_cache", False)
skip_topk = False
if self.compress_ratio == 4 and use_index_cache and ".mtp." not in prefix:
    pp_start_layer, _ = get_pp_indices(config.num_hidden_layers, pp_group.rank_in_group, pp_group.world_size)
    skip_topk = dsv4_skips_indexer_topk(config, config_layer_idx, pp_start_layer)
```

- `skip_topk=False`：本层算 TopK 并写进共享的 `topk_indices_buffer`；
- `skip_topk=True`：本层不算，直接读 buffer 里前一个 c4 层留下的索引；
- 判定仍支持 `index_topk_pattern`（如 `"FSFS"`，F=算/S=复用）与 `index_topk_freq`；
- PP 场景下每个 stage 的**第一个本地 c4 层强制自己算**（`pp_start_layer` 锚点），避免跨 stage 读到未初始化 buffer；MTP 层被显式排除（`".mtp." not in prefix`），因为 spec decode 在模型级共享 buffer。

### 2.6 组装数据袋并创建外壳（L634-L680）

```python
kv_cache_dtype = kv_cache_dtype_str_to_dtype(vllm_config.cache_config.cache_dtype, vllm_config.model_config)
if is_turboquant(vllm_config):
    kv_cache_dtype = torch.bfloat16   # TurboQuant 的 SWA 侧保持 bf16
swa_cache_layer = AscendDeepseekV4SWACache(
    head_dim=512, window_size=128, dtype=kv_cache_dtype, prefix=f"{prefix}.swa_cache", ...)

dsa_modules = DSAModules(            # 把所有模块"打包成一个袋子"交给后端
    wq_a=..., q_norm=..., q_norm_without_weight=..., wq_b=...,
    wkv=..., kv_norm=..., wo_a=..., wo_b=..., attn_sink=...,
    indexer=self.indexer, compressor=self.compressor,
    swa_cache_layer=swa_cache_layer,
)
self.dsa_attn = AscendDeepseekSparseAttention(..., compress_ratio=self.compress_ratio,
                                              dsa_modules=dsa_modules, ...)
```

> `topk_indices_buffer` 与 `skip_topk` 不再放进 `DSAModules`，而是由 `DeepseekV4Indexer` 自己持有（L333-L336）。

---

## 三、第 2 层：DSAModules 数据袋 + AscendDeepseekSparseAttention

### 3.1 DSAModules（[ops/dsa.py:L39-L54](../../../../../vllm-ascend/vllm_ascend/ops/dsa.py#L39)）

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
    indexer_rotary_emb: Module | None = None             # V4 不传，供其它 DSA 模型复用
```

### 3.2 外壳类（[ops/dsa.py:L57](../../../../../vllm-ascend/vllm_ascend/ops/dsa.py#L57)）

`AscendDeepseekSparseAttention` 继承上游的 `MultiHeadLatentAttentionWrapper`，做四件事：

1. 从数据袋把模块平铺到自身属性（L95-L109）；
2. 内部 `self.dsa_attn = DSAAttention(...)`（L111-L142），把模块以 `kwargs` 传给基类，由基类去挑后端、建 impl；
3. 把自己登记进 `compilation_config.static_forward_context[prefix]`（L144-L147），供图编译期按层名取到该层实例（重名会抛错）；
4. `forward`（L149）不再直接跑 Python 逻辑，而是统一进自定义算子 `torch.ops.vllm.dsa_forward`（L163），这是 ACL Graph 捕获所必需的。`dsa_forward`（L170）从 `forward_context.no_compile_layers[prefix]` 取回外壳，`_build_kv_cache`（L208）拼出 KV cache 元组，再调 `self.dsa_attn.impl.forward(...)`（L187）。

> `_build_kv_cache` 在不支持 DSV4 压缩缓存的机型（如 A3）返回 **6 元组**（压缩 KV / SWA / state / indexer state / indexer K / indexer scale），在支持该能力的 A5 机型返回 **7 元组**（多一个 `indexer_full_cache`）。

---

## 四、第 3、4 层：基类、后端与 Metadata

### 4.1 DSAAttention 基类

[layer.py:L68](../../../../../vllm-ascend/vllm_ascend/models/layer/attention/layer.py#L68) 的 `DSAAttention(nn.Module, AttentionLayerBase)` 负责与 vLLM 调度侧对接：

- 按 `compress_ratio` 选**主注意力缓存**的后端：c4 → `AscendDSAC4Backend`（L127-L132），c128 → `AscendDSAC128Backend`，dense → `AscendDSASWABackend`；
- 校验 `kv_cache_dtype` 是否被该后端支持（L137-L149）；
- 建 impl（L159-L176），并把 `swa_cache_layer` 等 `extra_impl_args` 一起传下去；
- `get_kv_cache_spec`（L208）按缓存类型返回 spec；SWA 单独由 `AscendDeepseekV4SWACache` 管理；
- `forward`（L192）是一个返回 `q` 的空壳 —— 真正的计算全部在 impl 里。

block 尺寸表 `get_dsv4_block_sizes`（同文件 L33）按机型区分：A3 用 `DSV4_BLOCK_SIZES`（L57），A5 的压缩缓存用 `_DSV4_COMPRESSED_BLOCK_SIZES`，A5 且显式 bf16 KV 时用新增的 `DSV4_BLOCK_SIZES_A5_BF16`（L58）。调度期通过 `dsv4_block_sizes(vllm_config)`（L61）在两张 A5 表之间选择，细节见 KV 缓存文档。

### 4.2 AscendDSABackend 与角色子类（[dsa_v1.py:L272](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L272)）

基类 `AscendDSABackend` 提供 dtype 白名单（`supports_kv_cache_dtype` L286）、impl/builder 工厂、KV cache 形状。v0.32.0 把原先"一个大后端 + 按 ratio 分支"的结构拆成**一个基类 + 6 个角色子类**，每类缓存有独立后端和自己的 metadata builder：

| 后端 | 行 | 用途 | builder |
|------|----|------|---------|
| `AscendDSABackend` | [L272](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L272) | 公共基类（dtype/形状/impl） | 通用 `AscendDSAMetadataBuilder` |
| `AscendDSAC4Backend` | [L354](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L354) | c4 压缩注意力缓存 | `AscendDSAAttentionMetadataBuilder` |
| `AscendDSAC128Backend` | [L370](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L370) | c128 压缩注意力缓存 | `AscendDSAAttentionMetadataBuilder` |
| `AscendDSASWABackend` | [L386](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L386) | SWA 原文缓存 | `AscendDSASWAMetadataBuilder` |
| `AscendDSAC4StateBackend` | [L400](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L400) | c4 压缩状态（`state_cache`） | `AscendDSAStateMetadataBuilder` |
| `AscendDSAC128StateBackend` | [L414](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L414) | c128 压缩状态 | `AscendDSAStateMetadataBuilder` |
| `AscendDSAIndexerBackend` | [L430](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L430) | Indexer 独立缓存 | `AscendDSAIndexerMetadataBuilder` |

角色 builder 的映射由 `_get_dsa_role_builder`（L442）完成；开启 PCP 时改由 [context_parallel/dsa_cp.py](../../../../../vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py) 的 `AscendDSAPCP*MetadataBuilder` / `AscendDSAPCPImpl` 接管。

### 4.3 三类 Metadata

| 类 | 行 | 作用 |
|----|----|------|
| `AscendDSAReqMetadata` | [L465](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L465) | 一次注意力调用的**统一请求级 metadata**：block_table / seq_lens / slot_mapping / query_start_loc / start_pos / cos/sin / compressor 元数据 / sas、qli metadata / DSpark、vision 索引等 |
| `AscendDSAMetadata` | [L502](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L502) | **单缓存组/单层**的容器：token 数、decode/prefill 计数、`req_metadata`、`hadamard`、`reshape_cache_event` |
| `AscendDSALayerMetadata` | [L530](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L530) | **一层实际消费的四个视图**：`attention` / `swa` / `compressor` / `indexer`，由 `AscendDSAImpl._get_layer_metadata`（L2375）按前缀组装 |

> **重要变化**：v0.32.0 已取消 `AscendDSAPrefillMetadata` / `AscendDSADecodeMetadata` 两个类，prefill/decode 共用一套 `AscendDSAReqMetadata`；区别只体现在长度上限、`start_pos` 与 `has_prefill` 分支里。

> **metadata 是什么**：融合稀疏注意力算子需要知道"每个请求的 Q 有多长、KV 块在显存里怎么排、TopK 索引怎么对齐"。这些不是权重、也不是激活，而是随批次变化的"索引说明书"，由 builder 在 Python/设备端预先算好。

### 4.4 Metadata Builder 与角色拆分

| Builder | 行 | 职责 |
|---------|----|------|
| `AscendDSAMetadataBuilder` | [L755](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L755) | 通用实现：`build` L1025、`build_req_metadata` L1296、`_prepare_rope` L1354、`_new_req_metadata` L1329 |
| `AscendDSAStateMetadataBuilder` | [L1760](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L1760) | state cache：只产 block_table + 请求布局 |
| `AscendDSAAttentionMetadataBuilder` | [L1780](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L1780) | 压缩注意力：SAS query `_build_sas_metadata` L1787、压缩缓存 `_build_compressed_cache` L2082 |
| `AscendDSASWAMetadataBuilder` | [L2119](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L2119) | SWA：SAS query + 原文 KV 写 `_build_swa_cache` L2147 |
| `AscendDSAIndexerMetadataBuilder` | [L2164](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L2164) | Indexer：QLI metadata `_build_qli_metadata` L2167 + indexer query L2239 |

> 每个角色 builder 只组装自己缓存组消费的字段；通用路径（legacy / deferred device metadata / vision）仍走 `AscendDSAMetadataBuilder.build_req_metadata`，由它在内部复用各角色 builder 的表构建函数（L1320-L1326）。

---

## 五、第 5 层：AscendDSAImpl 核心实现

文件：[dsa_v1.py:L2270](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L2270)。入口 `forward`（L2573）不再按 metadata 类型分 prefill/decode 两个方法，而是统一走到 `_forward_attention`（L2968），用 `has_prefill`（L2997）只切换少量形状/参数。

### 5.1 构造时取出的关键成员（L2291-L2353）

```python
# MLA 权重
self.wq_a, self.wq_b, self.wkv = kwargs["wq_a"], kwargs["wq_b"], kwargs["wkv"]
self.q_norm, self.q_norm_without_weight, self.kv_norm = ...
# CV 包装：把一层 Linear 拆成 Vector(量化) + Cube(矩阵乘)，便于多流重叠
self.cv_wq_a = CVLinearWrapper(self.wq_a)
self.cv_wkv  = CVLinearWrapper(self.wkv)
self.cv_wq_b = CVLinearWrapper(self.wq_b)
# 压缩 / 索引模块整体持有（参数不再逐项拷贝进 impl）
self.indexer   = kwargs.get("indexer")
self.compressor = kwargs.get("compressor")
self.swa_cache_layer = kwargs.get("swa_cache_layer")
self.wo_a, self.wo_b = kwargs["wo_a"], kwargs["wo_b"]
self.attn_sink = kwargs["attn_sink"]
self.multistream_dsv4_dsa_overlap = ascend_config.multistream_dsv4_dsa_overlap  # 默认 True
self._oproj_buffer_pool = OProjBufferPool()   # 模型级注入前的本地兜底
```

### 5.2 Indexer / Compressor 不再逐项拷贝参数

旧实现把 `indexer` 的 `n_heads`、`head_dim`、`wq_b`、`weights_proj`、内部 `compressor` 参数全部摊到 impl 上（以及 `inderxer_*` 的历史拼写）。v0.32.0 改为**模块自治**：

- Indexer 自己持有 `wq_b` / `cv_wq_b` / `weights_proj` / `k_cache` / 内部 `Compressor` / `AscendIndexerOps`（[indexer.py:L323-L367](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/indexer.py#L323)）；
- impl 只把 `qr`、`hidden_states`、`kv_cache`、`layer_metadata.indexer` 和一个 `IndexerOverlapPlan` 交给 `self.indexer(...)`（[dsa_v1.py:L2932-L2946](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L2932)）；
- Compressor 自己持有 `wkv/wgate/ape/norm/state_cache`，impl 只调 `compressor(hidden_states, state_cache, metadata)`（[compressor.py:L201-L232](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/compressor.py#L201)）。

IndexCache 侧仍由 Indexer 持有 `skip_topk`、`topk_indices_buffer`、`use_index_cache`，读写方法 `_get_cached_topk_indices`（L416）/ `_update_cached_topk_indices`（L424）。

### 5.3 CV 分离与多流重叠

`CVLinearWrapper`（[ops/cv_linear.py:L13](../../../../../vllm-ascend/vllm_ascend/ops/cv_linear.py#L13)）把一次线性层拆成：

- **Vector 阶段**：量化（求缩放因子、整数量化），跑在 Vector 核；
- **Cube 阶段**：真正的矩阵乘，跑在 Cube 核。

当 `multistream_dsv4_dsa_overlap=True`（[ascend_config.py:L570](../../../../../vllm-ascend/vllm_ascend/ascend_config.py#L570)，默认开）时，`_mla_prolog_multistream`（L2746）把 prolog 排成 **3 段流水**：

| 段 | 主流 | 辅流（`dsv4_dsa_overlap_stream()`） |
|----|------|-----------------------------------|
| Part1 | `wq_a.quantize`（V）→ `wq_a.matmul`（C） | `wkv.quantize`（V） |
| Part2 | `q_norm` + `wq_b.quantize`（V） | `wkv.matmul`（C） |
| Part3 | `wq_b.matmul`（C） | `kv_norm` + RoPE + scatter（V/AIV） |
| Tail | `q_rms` + RoPE（V） | Compressor（可选，与 q 尾部重叠） |

`dsv4_dsa_overlap_stream`（L227）是一个**进程级复用的单例 aux stream**，所有层的重叠任务共用，避免每层建流。

### 5.4 OProjBufferPool 与细粒度 O-Proj TP

`OProjBufferPool`（L98）是**模型级**的地址稳定 scratch 池：`DeepseekV4Model`（model.py L833）在构造时创建一个（model.py L881），注入每个 `AscendDSAImpl`（L887），并在 model 侧调用 `impl.reserve_oproj_buffers`（方法定义见 L2456）预留 all_to_all / reduce_scatter 缓冲。这样 ACL Graph 重放时集合通信的地址不变，避免每步重新分配导致 HCCL 算子失配。`_forward_o_proj`（L2482）在 `oproj_tp_enable()` 时走 `all_to_all_single` → `_wo_a_bmm` → `wo_b` → `reduce_scatter_tensor`，否则直接 `_wo_a_bmm → wo_b`。

### 5.5 Compressor metadata 记忆化

`get_or_compute_compressor_metadata`（L140）把同一 `cache_group_key` 的设备端 `compressor_metadata` 结果缓存在当前 forward context 的 `additional_kwargs` 里，保证一个 cache group 只跑一次 metadata 算子；`reset_compressor_metadata_cache`（L130）由 spec decode 的 proposer 在进入下一个 substep 前清空（[llm_base_proposer.py:L1728-L1730](../../../../../vllm-ascend/vllm_ascend/worker/v1/spec_decode/llm_base_proposer.py#L1728)）。异步设备任务路径则由 `build_compressor_metadata_out`（L195）写预分配 buffer。

---

## 六、Compressor 与 Indexer 模块本身

这两个类已从 `model.py` 拆到独立文件，并且 **Python `forward` 不再是空壳**——它们直接持有参数并调用 NPU 算子。

### 6.1 Compressor（[compressor.py:L105-L232](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/compressor.py#L105)）

```python
self.overlap = compress_ratio == 4          # c4=True, c128=False
self.coff = 1 + self.overlap                # c4 → 2，c128 → 1
self.ape = nn.Parameter(torch.empty(compress_ratio, coff * head_dim, dtype=fp32))  # 压缩位置编码
self.wkv   = ReplicatedLinear(dim, coff * head_dim, ...)   # 内容/门控两路 KV
self.wgate = ReplicatedLinear(dim, coff * head_dim, ...)
self.norm  = RMSNorm(head_dim, dtype=torch.float32)         # 压缩核只收 fp32 norm
self.state_cache = AscendCompressorStateCache(...)          # kv_state + score_state
```

| 项 | c4（ratio=4） | c128（ratio=128） |
|----|---------------|-------------------|
| `overlap` / `coff` | True / 2 | False / 1 |
| 投影输出宽度 | `2*512=1024` | `1*512=512` |
| state cache 宽度 `state_dim` | `2*coff*head_dim=2048` | `2*head_dim=1024` |
| state block size 取值 | 表 `[0][2]`（L170） | 表 `[0][3]`（L178） |

> 支持 DSV4 压缩缓存的机型上，`wkv/wgate` 使用 `quant_config=None`（L138-L150），因为压缩算子直接吃 ND 原始权重（`skip_weight_nz_conversion=True`，L156-L157）。

`forward`（L201）先 `_compute_metadata`（L185）拿到 `compress_cos/sin` 与 `slot_mapping`，再调：

```python
compressed_kv = torch.ops._C_ascend.compressor(
    hidden_states, self.wkv.weight, self.wgate.weight,
    state_cache.squeeze(-2), self.ape, self.norm.weight,
    compress_sin, compress_cos,
    state_block_table=state_metadata.block_table,
    cu_seqlens=compressor_metadata.query_start_loc,
    start_pos=..., rope_head_dim=self.rope_head_dim,
    cmp_ratio=self.compress_ratio, coff=2 if self.overlap else 1,
    norm_eps=self.norm_eps, rotary_mode=2, cache_mode=1,
)
```

V4 的 `Compressor` 已不再使用 `overlap_transform`（Python 侧块间拼接）与 `rope_single`（这两个工具目前只保留在 `sfa_v1.py` / `mla_v1.py` 等其它实现中）：重叠拼接由算子按 `coff=2` 完成，RoPE 由传入的 `compress_cos/sin` 完成。

### 6.2 Indexer（[indexer.py:L297-L768](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/indexer.py#L297)）

```python
self.n_heads = config.index_n_heads          # 64
self.head_dim = config.index_head_dim        # 128
self.index_topk = config.index_topk          # 1024
self.wq_b = ReplicatedLinear(q_lora_rank, n_heads * head_dim, ...)   # (1536, 64*128)
self.cv_wq_b = CVLinearWrapper(self.wq_b)
self.weights_proj = ReplicatedLinear(hidden_size, n_heads, ...)      # (7168, 64)
self.ops = AscendIndexerOps(index_topk=self.index_topk)
# c4 才有自己的 k_cache：A5 用 fp8，A3 用 int8（get_indexer_k_dtype, L113）
self.k_cache = AscendDeepseekV4IndexerCache(head_dim=128, dtype=k_dtype, compress_ratio=4, ...)
self.compressor = Compressor(..., head_dim=128, rotate=True, ...)    # Indexer 内部小压缩器
self.topk_indices_buffer = topk_indices_buffer                        # IndexCache 共享缓冲
```

`AscendIndexerOps`（L198）是算子门面：`quantize_query`（L208）、`quantize_key_and_update_cache`（L211）、`update_scale_cache`（L225）、`select_topk`（L237，调 `npu_quant_lightning_indexer_v2`）、`quantize_update_cache_and_select_topk`（L268）。

Indexer 的工作：把 Query 低秩向量 `qr` 投成 64 个索引头，与"压缩后的历史摘要"打分，每头/每 token 选出 Top-1024 个历史块。`forward`（L437）三条路径：`skip_topk` 直接读 buffer；有多流时走 `_cv_compute_query_and_update_cache_multistream`（L499）+ `_select_topk_multistream`（L621）；否则走 `_select_topk_serial`（L724）。

---

## 七、完整 Forward 数据流

### 7.1 总览

```
 hidden_states (num_tokens, 7168)
        │
        ▼
 阶段1 MLA Prolog：Query 流 ‖ KV 流（单流或 3 段多流重叠）
   Q: wq_a→q_norm→wq_b→reshape(16头)→q_rms(无参)→partial RoPE
   KV: wkv→kv_norm→reshape(1头)→partial RoPE→写 SWA cache
        │
        ▼
 阶段2 Compressor（c4/c128）：wkv/wgate → 状态累积 → fp32 norm+RoPE → 写压缩 KV
        │   （metadata 按 cache group 记忆化；多流时与阶段3/Tail 重叠）
        ▼
 阶段3 Indexer（仅 c4）：算 Top-1024（skip_topk 时直接读 buffer）
        │
        ▼
 阶段4 融合稀疏注意力：Q 对「SWA 原文 + 选中的历史摘要」+ sink 一次算完
        │  attn_output (num_tokens, 16, 512)
        ▼
 阶段5 输出投影：逆 RoPE → 分组 reshape → wo_a → flatten → wo_b（可选 OProj TP）
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

**KV 流**：`wkv`→`kv_norm` 得 `(T, 512)`，`view` 成 `(T, 1, 512)`，partial RoPE 后 scatter 进 SWA cache。单流路径 `_mla_prolog_single_stream`（L2660）用 `inplace_partial_rotary_mul`（L2703/L2728）就地旋转；多流路径见 5.3。

### 7.3 阶段 2：Compressor

实现在 `Compressor.forward`（[compressor.py:L201](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/compressor.py#L201)），核心算子：

```python
compressed_kv = torch.ops._C_ascend.compressor(...)   # L212
```

它把连续 `compress_ratio` 个 token 的 `wkv/wgate` 投影累积进 fp32 `state_cache`，到边界做 fp32 norm + RoPE，吐出 1 个压缩 KV 写入压缩 cache。metadata 计算由 `compressor_metadata`（[L179](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L179)）/ `compressor_metadata_out`（[L213](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L213)）完成，并按 cache group 记忆化（见 5.5）。

### 7.4 阶段 3：Indexer TopK

仅 c4。两类算子配合（metadata 与计算分离）：

| 用途 | 算子 | 位置 |
|------|------|------|
| 生成选块 metadata | `torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata` | [dsa_v1.py:L2190](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L2190) |
| 真正选 TopK | `torch.ops._C_ascend.npu_quant_lightning_indexer_v2` | [indexer.py:L247](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/indexer.py#L247) |

```python
topk_idxs, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
    query=query, key=key_cache, weights=..., query_dequant_scale=...,
    key_dequant_scale=..., topk=self.index_topk, quant_mode=...,
    cu_seqlens_q=metadata.qli_cu_seqlens_q, seqused_k=metadata.qli_seqused_k,
    cmp_residual_k=metadata.qli_cmp_residual_k, block_table=metadata.block_table,
    metadata=metadata.qli_metadata, layout_q="TND", layout_k="PA_BBND",
    mask_mode=3, cmp_ratio=4, return_value=0,
)
```

- `skip_topk=False`：算完后 `_update_cached_topk_indices` 写共享 buffer；
- `skip_topk=True`：`_get_cached_topk_indices` 直接取，不调选块算子；
- 多流时，Indexer 的 Q 量化/写缓存与主注意力 Compressor 通过 `IndexerOverlapPlan`（L190）在 aux stream 上重叠。

> **版本提示**：旧资料里的 `npu_vllm_quant_lightning_indexer_metadata` 在 v0.32.0 中继续沿用 `npu_quant_lightning_indexer_v2_metadata`，并与独立的计算算子 `..._v2` 分离；metadata 与计算都不再按 prefill/decode 分成两份实现。

### 7.5 阶段 4：融合稀疏注意力

算子不再写死在 `DeviceOperator` 里，而是由 `DsaAttnKvPlan`（[dsa_attn_kv_plan.py:L139](../../../../../vllm-ascend/vllm_ascend/attention/dsa_attn_kv_plan.py#L139)）按机型 + cache dtype + 压缩比选出：

| 条件 | 稀疏注意力算子 | KV 布局 | 附加动作 |
|------|----------------|---------|---------|
| 非 A5（A3，无压缩缓存能力） | `torch.ops._C_ascend.npu_sparse_attn_sharedkv`（L244） | `PA_ND` | 用 `npu_sparse_attn_sharedkv_metadata`（L245） |
| A5 + bf16 KV | `cann_ops_transformer.sparse_flash_mla`（L18） | `PA_BBND` | metadata 走 `sparse_flash_mla_metadata` |
| A5 + fp8 KV | `cann_ops_transformer.mixed_quant_sparse_flash_mla`（fp8，L34） | `PA_ND` | 写缓存走 `kv_compress_epilog`（L210） |
| A5 + TurboQuant c4 | `mixed_quant_sparse_flash_mla`（quant_mode=3） | `PA_BBND` | 压缩 KV 走 TurboQuant 打包 |

c4 的调用形态（[dsa_v1.py:L3070-L3114](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L3070)）：

```python
result = attn_op(
    q,                                  # (T, 16, 512)
    ori_kv=swa_kv_cache,                # SWA 原文 KV（块表寻址）
    ori_block_table=swa_req_metadata.block_table,
    cu_seqlens_q=...,
    seqused_kv=...,
    sinks=self.attn_sink,               # 每头一个可学习 sink
    metadata=common_metadata.sas_metadata,
    softmax_scale=self.softmax_scale,
    cmp_ratio=...,
    ori_mask_mode=4,                    # 原文 KV：SWA 滑窗掩码
    ori_win_left=..., ori_win_right=...,
    layout_q="TND", layout_kv=...,
    # c4 / c128 追加：
    cmp_kv=compress_kv_cache,
    cmp_block_table=common_metadata.block_table,
    cmp_mask_mode=3,                    # 压缩 KV：causal 因果掩码
    cmp_sparse_indices=compress_topk_idxs,  # 仅 c4
)[0]                                     # → (T, 16, 512)
```

c128 没有 `cmp_sparse_indices`，即"所有历史摘要都参与"。`ori_mask_mode=4` / `cmp_mask_mode=3` 分别对应滑窗掩码与因果掩码。

### 7.6 阶段 5：输出投影

| 步骤 | 操作 | 输出 shape |
|------|------|-----------|
| 1 | 对 RoPE 部分（后 64 维）做逆旋转（`inplace_partial_rotary_mul` + `negate_sin`，L2643-L2652） | `(T, 16, 512)` |
| 2 | reshape 成 `n_local_groups=2` 组 | `(T, 2, 4096)` |
| 3 | `wo_a`（列并行，`_wo_a_bmm` L2429） | `(T, 2, 1024)`（本卡） |
| 4 | flatten | `(T, 2048)`（对应全局 `(T, 16*1024)` 的本卡切片） |
| 5 | `wo_b`（行并行，跨卡求和；可选 OProj TP） | `(T, 7168)` |

---

## 八、Prefill 与 Decode 的差异

v0.32.0 已合并原先的 `_forward_prefill` / `_forward_decode` 两条实现，统一由 `forward`（L2573）→ `_forward_attention`（L2968）处理，用 `has_prefill = common_attn_metadata.num_prefills > 0`（L2997）切换差异。

| 维度 | Prefill | Decode |
|------|---------|--------|
| Query 长度 | 一次多个 token（变长） | 单 token（或少量 spec token） |
| Q layout | `TND` | `TND`（同一算子约定） |
| KV 写入 | 整段新 token 写入三类缓存 | 只写当前 token |
| Metadata | 同一 `AscendDSAReqMetadata`（角色 builder 产出） | 同一 `AscendDSAReqMetadata` |
| 长度上限 | `max_seqlen_q` 取真实 query 长度 | `decode_threshold` 含 spec token 数（L800-L846） |
| Indexer | 同一 `select_topk`（L247） | 同一 `select_topk`（L247），增量写入 |
| 注意力算子 | 同一 `attn_op`，传 `cu_seqlens_q` | 同一 `attn_op`，传 `seqused_kv`/块表 |

两条路径共用同一套权重与 Compressor/Indexer 参数，区别只在"批量形状与索引方式"。

---

## 九、NPU 算子清单（v0.32.0 实测命名）

| 阶段 | 算子 | 位置 |
|------|------|------|
| 压缩 metadata | `_C_ascend.compressor_metadata` / `compressor_metadata_out` | dsa_v1.py L179 / L213 |
| 压缩计算 | `_C_ascend.compressor` | compressor.py L212 |
| 选块 metadata | `_C_ascend.npu_quant_lightning_indexer_v2_metadata` | dsa_v1.py L2190 |
| 选块计算 | `_C_ascend.npu_quant_lightning_indexer_v2` | indexer.py L247 |
| 稀疏注意力（A3） | `_C_ascend.npu_sparse_attn_sharedkv`（+ `_metadata`） | dsa_attn_kv_plan.py L244 / L245 |
| 稀疏注意力（A5 bf16） | `cann_ops_transformer.sparse_flash_mla`（+ `_metadata`） | sparse_flash_mla.py L18 |
| 稀疏注意力（A5 fp8 / TQ） | `cann_ops_transformer.mixed_quant_sparse_flash_mla`（+ `_metadata`） | mixed_quant_sparse_flash_mla.py L34 |
| 压缩 KV 写缓存（A5 fp8） | `cann_ops_transformer.kv_compress_epilog` | dsa_attn_kv_plan.py L210 |
| SWA/Indexer KV scatter | `_C_ascend.npu_scatter_nd_update_sk` | device_op.py L660 / L686-L710 |
| RoPE（partial/in-place） | `_C_ascend.inplace_partial_rotary_mul` | dsa_v1.py L2703 / L2728 |
| RoPE（通用） | `torch_npu.npu_rotary_mul` | rope_dsv4.py L400 |
| Q RMS（三路） | `_C_ascend.npu_rms_norm_dynamic_quant` / `apply_dsa_q_rms` | dsa_v1.py L2686 / device_op.py L753 |

---

## 十、性能优化要点小结

| 手段 | 做了什么 |
|------|---------|
| **MLA 低秩** | Q 走 `7168→1536→多头`、O 走分组低秩；K/V 只存 512 维 latent，KV 显存近似减半 |
| **三级稀疏** | SWA 保 128 个原文 + Compressor 压历史 + Indexer 只选 1024 块，把长上下文注意力压成近常数规模 |
| **IndexCache** | 相邻 c4 层共享一套 TopK 索引，跳过整次选块计算；PP 下按 stage 首层强制计算 |
| **CV 分离** | 量化（Vector）与矩阵乘（Cube）拆开，分别占用不同运算单元 |
| **多流重叠** | prolog 3 段流水 + Compressor Tail / Indexer 与辅助流重叠（`multistream_dsv4_dsa_overlap` 默认开，`dsv4_dsa_overlap_stream` 全局复用） |
| **metadata 记忆化** | 同一 cache group 的 compressor metadata 只算一次（`get_or_compute_compressor_metadata`），跨 substep 由 `reset_compressor_metadata_cache` 清空 |
| **算子融合** | SWA+压缩 KV+sink 的注意力一次进融合算子，不落中间结果；具体算子由 `DsaAttnKvPlan` 按机型选择 |
| **OProj TP + BufferPool** | 细粒度 O-Proj 张量并行，模型级 `OProjBufferPool` 提供 ACL Graph 安全的集合通信 scratch |
| **低精度存储** | A5：bf16 走 SparseFlashMla、fp8 走 kv_compress_epilog + MixedQuantSparseFlashMla，Indexer K 用 fp8；A3：SWA bf16、Indexer K 用 int8；attn_sink/norm/state 保持 fp32 |
| **TP / CP** | wq_b、wo_a、wo_b 张量并行；PCP 时由 `dsa_cp.py` 的 `AscendDSAPCPImpl` / 角色 builder 接管 |

---

## 十一、一句话总结

DeepSeek V4 的注意力 = **MLA 低秩骨架** + **"SWA 眼前 128 字 + Compressor 历史摘要 + Indexer 只挑 1024 块"的三级稀疏记忆**；Python 侧（[model.py:L483](../../../../../vllm-ascend/vllm_ascend/models/deepseek_v4/model.py#L483) 的 `DeepseekV4Attention`）只负责建权重和打包 `DSAModules`，真正的投影、压缩、选块与融合注意力全部下沉到 [AscendDSAImpl](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py#L2270)，通过 `compressor` / `npu_quant_lightning_indexer_v2` / 融合稀疏注意力算子三组 NPU 算子在一次稀疏注意力里完成。
