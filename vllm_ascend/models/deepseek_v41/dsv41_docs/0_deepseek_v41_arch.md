# DeepSeek V4.1 整体架构详解

> **本文档为 DeepSeek V4.1（`DeepSeek-V4.1-Flash-1017`）在 Ascend NPU 上的整体架构总览。各子模块的详细实现请参考同级目录下的专题文档。**

> **代码版本冻结声明**：本文基于本仓库 `vllm_ascend/models/deepseek_v41/` 包的当前源码（`model.py` 共 2076 行）与 `DeepSeek-V4.1-Flash-1017` checkpoint 配置整理。文中 `Lxxx` 未带文件名时均指该主文件行号，跨文件引用会显式标注。V4.1 的模型代码被拆成如下包（与上游 `vllm/models/deepseek_v41/` 的包名对齐）：

| 文件 | 行数 | 职责 |
|------|------|------|
| [model.py](../model.py) | 2076 | 文本主干：MoE/MLP、层拓扑（`build_layer_plan`）、注意力封装、Decoder 层、主干 forward、权重加载 |
| [compressor.py](../compressor.py) | 71 | C1/C2 压缩器：C1 直投、C2 fp32 环形状态 + softmax 池化 |
| [indexer.py](../indexer.py) | 308 | QLI/QSLI 两级索引投影与选块 |
| [cache_config.py](../cache_config.py) | 297 | 混合缓存 Spec 构造、层元组（layer tuple）布局、全局 block-ID 池分配 |
| [decoder_replay_layers.py](../decoder_replay_layers.py) | 141 | 有界回放（bounded replay）的启用判定与层适配器 |
| [decoder_replay_state.py](../decoder_replay_state.py) | 442 | 有界回放的批plan、DP/PCP 协调、元数据重建、forward context 替换 |
| [dspark.py](../dspark.py) | 662 | DSpark 3 层块草稿模型（`mtp.*` 权重树） |
| [mixed_quant_attention.py](../mixed_quant_attention.py) | 356 | A5 打包混合量化缓存 ABI（QLI/QSLI/Q-SMLA 算子绑定） |
| [vision.py](../vision.py) | 127 | ViT 塔 + Aligner（图像） |
| [vl_model.py](../vl_model.py) | 336 | 多模态封装：图像 span 拼装、嵌入合并、权重分流 |
| [engram/\_\_init\_\_.py](../engram/__init__.py) | 21 | Engram 导出面 |
| [engram/common.py](../engram/common.py) | 72 | 旋转基、门控公式、有效 token 掩码 |
| [engram/hash_state.py](../engram/hash_state.py) | 173 | n-gram 哈希（上游 `NgramHashState` 的 Ascend 子类）+ SWA 槽位适配 |
| [engram/embedding.py](../engram/embedding.py) | 490 | 头分片大表：BF16/INT8/MXFP8 存储、按行流式加载、UVA 主机内存卸载 |
| [engram/layer.py](../engram/layer.py) | 57 | `AscendEngram`：`wkv` 投影 + 门控写回残差流 |
| [engram/model_state.py](../engram/model_state.py) | 333 | MRv2 的 Engram 模型状态（lookback、图事件） |
| [engram/npu.py](../engram/npu.py) | 401 | 设备侧查表/反量化/UVA 缓冲 |
| [engram/parallel.py](../engram/parallel.py) | 99 | DP×PCP 哈希聚合与行交换（AllToAll） |

> **统一模型配置**（取自本仓库 `dsv41_docs/0_DeepSeek-V4.1-Flash-1017_config.json`，即 **DeepSeek-V4.1-Flash-1017** 发布版 checkpoint，`dtype=bfloat16`，FP8 权重 + FP4 专家）。文中所有 shape 标注均以此配置为例：

| 分组 | 配置项 | 值 | 与 V4-Pro-0813 的差异 |
|------|--------|-----|----------------------|
| **基础** | `vocab_size` / `hidden_size` / `num_hidden_layers` | 129280 / **5120** / **40** | V4 为 7168 / 61 层 |
| | `max_position_embeddings` | 1048576（1M；YaRN `factor=16`，`original_max_position_embeddings=65536`） | 相同 |
| | `rms_norm_eps` | 1e-20 | V4 为 1e-6 |
| **mHC** | `hc_mult` / `hc_eps` / `hc_sinkhorn_iters` | 4 / 1e-6 / 20 | 相同 |
| **注意力 MLA** | `head_dim` / `qk_rope_head_dim` | 512 / 64 | 相同 |
| | `num_attention_heads` / `num_key_value_heads` | **64** / 1 | V4 为 128 |
| | `q_lora_rank` / `o_lora_rank` / `o_groups` | **1280** / 1024 / **8** | V4 为 1536 / 16 |
| | `sliding_window` / `rope_theta` / `compress_rope_theta` | 128 / 10000 / 160000 | 相同 |
| **压缩比** | `compress_ratios`（43 项） | `[0,0]` + **18×`2`** + **20×`1`** + `[0,0,0]` | V4 为 `0/4/128` 三档 |
| **源共享** | `kv_source_layer_ids` | **`[2, 8, 14, 20]`**（4 个） | V4 无此机制（每层自带） |
| | `index_source_layer_ids` | **`[2, 8, 14, 20, 24, 28, 32, 36]`**（8 个） | V4 无跨层复用 |
| | `candidate_source_layer_id` | **20** | 无 |
| **两级索引** | `index_n_heads` / `index_head_dim` / `index_topk` | **32** / 128 / **512** | V4 为 64 头 / TopK 1024 |
| | `candidate_topk_blocks` / `candidate_block_size` | **2048** / **8** | 无（覆盖 16384 个压缩位） |
| **MoE** | `n_routed_experts` / `num_experts_per_tok` / `n_shared_experts` | 384 / 6 / 1 | 相同 |
| | `moe_intermediate_size` | **2304** | V4 为 3072 |
| | `routed_scaling_factor` / `scoring_func` / `topk_method` | **1.5** / `sqrtsoftplus` / `noaux_tc`（`norm_topk_prob=true`） | V4 为 2.5 |
| | `swiglu_limit` | 10.0 | 相同 |
| | `num_hash_layers` | **未提供 → 归一化为 0（不启用 hash 路由）** | V4 为 3 |
| **Engram** | `engram_layer_ids` | **`[1, 14]`** | V4 无此模块 |
| | `engram_num_embeddings` | `[384006168, 384016682]` | — |
| | `engram_max_ngram_size` / `engram_n_heads` / `engram_head_dim` | 4 / 8 / 256 | — |
| | `engram_vocab_size` / `engram_compressed_vocab_size` | 16000000 / 99092 | — |
| | `engram_pad_token_id` | 2 | — |
| **推测解码** | `num_nextn_predict_layers`（DSpark 层数） | **3** | V4 为 1 |
| | `dspark_block_size` / `dspark_markov_rank` / `dspark_noise_token_id` | 5 / **256** / 128799 | V4 为 512 rank |
| | `dspark_target_layer_ids` | **`[37, 38, 39]`** | V4 为 `[58,59,60]` |
| | `dspark_n_routed_experts` / `dspark_num_experts_per_tok` | **128 / 3** | V4 未降配 |
| **量化** | `quantization_config` | `quant_method=fp8`、`activation_scheme=dynamic`、`weight_block_size=[32,32]`、`scale_fmt=ue8m0`、`expert_dtype=fp4` | V4-Pro 为 W4A8 包 |
| **视觉** | `vision_config` | 32 层 / 1024 / 16 头 / 2816 / patch 14 / downsample 3 | V4 配置中无图像塔 |
| | `image_token_id` | 129264 | — |

> 配置里的 `compress_ratios` 比主干层数多 3 项，对应 3 个 DSpark draft 层（数组下标 40/41/42 均为 `0`）。`build_layer_plan` 会先 `ratios[:num_hidden_layers]` 截断，只有主干前 40 项描述主干（[model.py:L518-L528](../model.py#L518)）。

---

## 一、架构总览

### 1.1 一句话定位

DeepSeek V4.1 是 V4 的**续作（minor update）**：主干仍然是「mHC 超连接 + DSA 稀疏注意力 + 全 MoE 前馈」的三件套，但把 **KV 缓存压缩**推到了新的极限——官方 README 的引用标题就是 *“Pushing the Limits of KV Cache Compression”*。为此 V4.1 在主干上另加了三块新东西：

| 技术 | 全称 / 定位 | 核心作用 | 代码入口 |
|------|------------|---------|---------|
| **mHC** | Manifold-Constrained Hyper-Connections（流形约束超连接） | 把一条残差流拓宽为 `hc_mult=4` 条并行流；V4.1 额外做**延迟系数交接**（下一层的 `pre` 由上一层 FFN 产出） | [model.py:L768-L920](../model.py#L768) |
| **源共享混合缓存**<br>（Source-Shared Hybrid KV） | 只有 4 个「KV 源层」真正做压缩 KV，其余 36 层**直接读源层缓存** | 40 层的压缩 KV 从 40 份降为 4 份 | [model.py:L449-L561](../model.py#L449)、[cache_config.py](../cache_config.py) |
| **两级索引**<br>（QLI + QSLI） | 先用 QLI 在层 20 选出 2048 个候选块（×8 token），后续层只在候选块内用 QSLI 选 Top-512 | 稀疏选择的成本与 1M 上下文解耦 | [indexer.py:L137-L308](../indexer.py#L137) |
| **Engram** | 16M 行级 n-gram 哈希记忆表，挂在层 1 与层 14 | 给模型一份**非参数化的短语记忆**，以门控方式写回残差流 | [engram/](../engram/) |
| **DSpark** | 3 层块草稿器（一次出 5 个 token）+ **Decoder 有界回放** | 推测解码吞吐；回放让源层之后的 19 层只算最后 128 个 token | [dspark.py](../dspark.py)、[decoder_replay_state.py](../decoder_replay_state.py) |

> **生活化类比**：把 V4.1 的主干想象成一家**共享档案室的公司**。
> - **SWA** 是每个人桌上摊开的**最近 128 页文件**（每层都有一份，最准）；
> - **压缩 KV** 是档案室里**只有 4 位档案管理员**（层 2/8/14/20）整理的**摘要档案**——其余 36 人不再各自整理一份，而是直接来借阅对应管理员的那一份（这就是「源共享」，档案室面积因此缩到原来的 1/10）；
> - **QLI** 是门口的**粗筛目录**（2048 个柜子，每柜 8 份），**QSLI** 才是进柜子后的**细挑**（512 份）；
> - **Engram** 是墙上那本**成语大词典**：看到「画蛇」就条件反射翻到「添足」，但只在它和当前句子的语境足够匹配（门控值大）时才真的念出来；
> - **DSpark** 是负责打草稿的实习生，一次写 5 个 token，`markov_head` 给它补上下文偏置，`confidence_head` 自评「这 5 个我有多确信」。

### 1.2 与 V4 的关键差异（读 V4 文档前必看）

| 维度 | DeepSeek V4(-Pro-0813) | DeepSeek V4.1-Flash-1017 |
|------|------------------------|--------------------------|
| 主干规模 | 61 层 / 7168 隐维 / 128 注意力头 / `o_groups=16` | **40 层 / 5120 隐维 / 64 头 / `o_groups=8`** |
| 压缩档位 | `compress_ratio ∈ {0, 4, 128}` | **`{0, 1, 2}`**（c1 直投、c2 二合一池化） |
| 压缩缓存归属 | **每层**自带 Compressor/Indexer | **只有源层**（4 个 KV 源 / 8 个索引源），消费者读源层缓存 |
| 索引层数 | 每 c4 层一个 Indexer | **8 个索引源**，其中 4 个（24/28/32/36）走**候选块过滤** |
| 选块粒度 | Top-1024 个压缩位置 | **两级**：2048 块 × 8 token → Top-512 位置 |
| 额外记忆 | 无 | **Engram**（2 层，384M 行 / 层） |
| 草稿层 | 1 层串行 MTP | **3 层 DSpark 块草稿**（block_size=5，draft MoE 128 专家 Top-3） |
| mHC 系数 | 每子层各自算 `pre` | **延迟交接**：本层 FFN 算出的 `pre` 给下一层注意力用 |
| 预填充优化 | 无 | **Decoder 有界回放**（层 21-39 只回放最后 128 token） |
| 算子入口 | `npu_hc_pre_v2`（3 输出） | **`npu_hc_pre_v3`**（带可选 `pre_mix`，4 输出） |

> **注意**：不要把 V4 文档里的 `7168 / 61 / c4 / c128 / Top-1024 / 单层 MTP` 当作 V4.1 的事实。本文档所有专题分册都按 V4.1 的配置重写过。

### 1.3 文档体系索引

| 编号 | 文档 | 内容 | 详细程度 |
|------|------|------|---------|
| **0** | `0_deepseek_v41_arch.md` | **本文档** — 整体架构、层拓扑、数据流总览 | ★☆☆ 总览 |
| **1** | `1_npu_hc_pre_npu.md` | `hc_pre` 算子逐行精读（Ascend C），附 V4.1 `v3` 入口差异 | ★★★ 详细 |
| **2** | `2_npu_hc_post_npu.md` | `hc_post` 算子逐行精读（Ascend C） | ★★★ 详细 |
| **3** | `3_deepseek_v41_attn.md` | DSA 注意力：源共享 + 两级索引 + 融合稀疏 MLA 算子 | ★★★ 详细 |
| **4** | `4_deepseek_v41_moe.md` | MoE：路由、专家、共享专家、EPLB、量化落地 | ★★★ 详细 |
| **5** | `5_deepseek_v41_kv_cache.md` | 混合缓存：SWA 环 + C1/C2 压缩 + 索引 K + 折叠缓存 + 层元组分配 | ★★★ 详细 |
| **6** | `6_deepseek_v41_engram.md` | Engram：n-gram 哈希、头分片大表、门控写回、异步查表、DP×PCP 交换 | ★★★ 详细 |
| **7** | `7_deepseek_v41_mtp.md` | MTP/DSpark 调度：延迟 mHC 交接、hidden 采集、权重映射 | ★★★ 详细 |
| **8** | `8_deepseek_v41_dspark.md` | DSpark 块草稿器：上下文 KV 预计算、Markov/Confidence 头、权重改名 | ★★★ 详细 |
| **9** | `9_deepseek_v41_decoder_replay.md` | Decoder 有界回放：行集合、DP/PCP 协调、可见长度下界 | ★★★ 详细 |
| **10** | `10_deepseek_v41_multimodal.md` | 多模态：ViT、Aligner、图像 span、VL 路由偏置 | ★★☆ 中等 |
| **11** | `11_deepseek_v41_sampling.md` | 后处理与采样：logits、采样、DSpark 接受链路 | ★★☆ 中等 |

**原始资料（同目录）**

| 文件 | 说明 |
|------|------|
| `0_DeepSeek-V4.1-Flash-1017_config.json` | 官方 `config.json`（本文所有数值的来源） |
| `0_DeepSeek-V4.1-Flash-1017_README.md` | 官方模型卡（含 benchmark 与推荐采样参数） |
| `0_DeepSeek-V4.1-Flash-1017_tokenizer_config.json` | tokenizer 配置 |
| `0_DeepSeek-V4.1-Flash-1017_model_weights.index.json` | 48 个 safetensors 分片的 `weight_map`（查权重名用） |
| `0_DeepSeek-V4.1-Flash-1017_configuration.json` | ModelScope 的 `configuration.json` |
| [reference/](reference/README.md) | **官方最小推理实现**（`model.py` 61 KB，含 ViT/DSA/Engram/MoE/mHC/DSpark 的可读版本），以及 prompt `encoding/`。对照阅读是理解训练期语义的最快路径 |
| `reference/config.json` | 官方推理端的 `ModelArgs`（字段名与 HF config 不同，便于对照） |

---

## 二、层拓扑：谁生产缓存、谁消费缓存

### 2.1 `build_layer_plan` 的三条规则

整个 V4.1 的「省 KV」逻辑可以用三条规则概括（[model.py:L510-L561](../model.py#L510)）：

1. **压缩比来自 `compress_ratios[layer_idx]`**；`ratio == 0` 表示该层是纯滑窗（dense），不参与压缩 KV。
2. **KV 源 = 最近的、编号不超过自己的源层**：`kv_source = _latest_source(layer_idx, kv_source_layer_ids)`（[L533](../model.py#L533)，`_latest_source` 在 [L506-L507](../model.py#L506)）。只有源层自己 `is_kv_source=True`，才会创建 `long_kv_cache` / `compressor`。
3. **索引源同理**，且「消费者继承其索引源的选择策略」：`uses_candidate_filter = index_source is not None and index_source > candidate_source`（[L548](../model.py#L548)）——因为层 26 复用的是层 24 的 TopK，而层 24 的 TopK 本来就只在层 20 的候选块内选出来的。

### 2.2 40 层主干 + 3 层 draft 的角色表

| 层区间 | `compress_ratio` | KV 源 | 索引源 | 特殊角色 | 说明 |
|--------|:---:|:---:|:---:|------|------|
| **0 – 1** | 0 | — | — | **层 1：Engram** | 纯滑窗；无压缩缓存 |
| **2 – 7** | 2 | **2** | **2** | **层 2：KV 源 + 索引源** | c2 池化源；后 5 层读层 2 的压缩 KV 与索引 K |
| **8 – 13** | 2 | **8** | **8** | **层 8：KV 源 + 索引源** | 同上 |
| **14 – 19** | 2 | **14** | **14** | **层 14：KV 源 + 索引源 + Engram** | 同上 |
| **20 – 23** | 1 | **20** | **20** | **层 20：KV 源 + 索引源 + 候选源** | c1 直投源；同时产出 2048 个候选块 |
| **24 – 27** | 1 | 20 | **24** | 层 24：索引源（**二级**） | 在层 20 的候选块内用 QSLI 细选 Top-512 |
| **28 – 31** | 1 | 20 | **28** | 层 28：索引源（二级） | 同上 |
| **32 – 35** | 1 | 20 | **32** | 层 32：索引源（二级） | 同上 |
| **36 – 39** | 1 | 20 | **36** | 层 36：索引源（二级） | 同上 |
| **40 – 42** | 0 | — | — | DSpark 3 个 draft 层 | 独立 `mtp.*` 权重树，只做块内 SWA |

由这张表可以读出 V4.1 的缓存账本：

- **压缩 KV（long KV）只有 4 份**：层 2/8/14 各一份 c2（长度 = token/2），层 20 一份 c1（长度 = token）。
- **索引 K 只有 4 份**（层 2/8/14/20 各自 `owns_k`），却服务 **8 个索引源**；其中层 20 的索引 K 在 A5 上还额外有一份**折叠缓存**（`k_cache_folded`，8 行/页）供 QSLI 使用。
- **压缩状态（compressor state）只有 3 份**：只有 `ratio == 2` 的源（层 2/8/14）需要 32 行 fp32 环形状态；层 20 的 c1 是「一 token 一 entry」，不需要暂存组。
- **SWA 环 40 份**（每层一份，128 token）+ 3 份 draft SWA。

### 2.3 数据流总览（一次 decode 前向）

```
input_ids ──► embed_tokens ──► unsqueeze+repeat(hc_mult=4) ──► x: (T, 4, 5120)
                                                                  │
      pre_mix: (T, 4) fp32, 初值 one-hot [1,0,0,0]  ───────────────┘
                                                                  │
   ┌───────────────────────── 每层（40 次）────────────────────────────┐
   │  [Engram?] 层 1 / 层 14：gather 哈希 → 24 列查表 → gate → 写回 x   │
   │      ↓                                                            │
   │  hc_pre(x, hc_attn_fn, …, pre_mix)  ← pre_mix 来自上一层 FFN      │
   │      ↓  (y, post, comb, pre)                                      │
   │  input_layernorm → DSA 注意力（SWA + 源层压缩 KV + 索引 TopK）     │
   │      ↓                                                            │
   │  hc_post(attn_out, 旧 x, post, comb) ──► 新的 4 条流              │
   │      ↓                                                            │
   │  hc_pre(x, hc_ffn_fn, …, attn_pre) ← 复用本次注意力算出的 pre      │
   │      ↓  rms_norm_cast（归一化 + fp32 路由输入一次算完）            │
   │  MoE（384 选 6 + 1 共享专家）                                     │
   │      ↓                                                            │
   │  hc_post(moe_out, 旧 x, ffn_post, ffn_comb) ──► 新的 4 条流       │
   │      ↓  返回 (x, ffn_pre) —— ffn_pre 交给下一层的注意力            │
   └───────────────────────────────────────────────────────────────────┘
                                    │
              hc_collapse(x, 最后一层的 ffn_pre) ──► (T, 5120)
                                    │
                            RMSNorm → lm_head → logits
```

对应代码：入口 repeat 与 `pre_mix` 初始化在 [L1563-L1565](../model.py#L1563)，层循环在 [L1568-L1611](../model.py#L1568)，出口 `hc_collapse` 在 [L1621](../model.py#L1621)，单层的两段 `hc_pre`/`hc_post` 在 [L879-L920](../model.py#L879)。

> **延迟交接的意义**：`hc_pre` 需要在每层算出「怎么把 4 条流读成 1 条」。如果本层注意力要等本层的 `hc_attn_fn` 投影算完才能开工，Cube（矩阵乘）就会空转。V4.1 让**上一层 FFN 顺手算出下一层注意力要用的 `pre`**：`forward` 最后 `return hidden_states, ffn_pre`（[L920](../model.py#L920)），模型循环把 `pre_mix` 一路传下去。这样 `hc_pre` 的输入矩阵乘与上一层的 FFN 完全并行，关键路径上少了一次依赖。细节见 [7_deepseek_v41_mtp.md](7_deepseek_v41_mtp.md) 与 [1_npu_hc_pre_npu.md](1_npu_hc_pre_npu.md)。

### 2.4 三类「共享」把缓存压下来

| 共享对象 | 生产 | 消费 | 机制 |
|---------|------|------|------|
| **压缩 KV**（`long_kv_cache`） | 4 个 KV 源层 | 全部 40 层（按 `kv_source_layer` 就近取） | `long_kv_source_prefix` 指向源层缓存（[L744-L747](../model.py#L744)），算子直接吃源层页 |
| **索引 K**（`indexer.k_cache`） | 4 个 `owns_k` 层 | 8 个索引源 | `index_k_source_prefix`；`update_keys` 在 latent 被 RoPE 覆盖**之前**写（[indexer.py:L116-L135](../indexer.py#L116)） |
| **候选块**（`candidates`） | 层 20（`candidate_source_layer_id`） | 24/28/32/36 及其消费者 | `DeepseekV41SharedAttentionState`（[L490-L503](../model.py#L490)）在一次 forward 内传递 `topk_indices` / `candidates` / 两个长度张量 |

三个共享状态的缓冲区在模型构造时一次性分配（[L1000-L1022](../model.py#L1000)）：

| 缓冲区 | 形状 | dtype | 用途 |
|--------|------|-------|------|
| `topk_indices_buffer` | `(max_num_batched_tokens, index_topk=512)` | int32 | 当前层可看的压缩位置 TopK（-1 表示无效） |
| `candidate_indices_buffer` | `(max_tokens, 1, 2048)` | int32 | 层 20 产出的候选**块号**（不是位置） |
| `candidate_lengths_buffer` | `(max_tokens, 1)` | int32 | 每 token 实际产出的候选块数 |
| `topk_lengths_buffer` | `(max_tokens, 1)` | int32 | 每 token 实际的 TopK 命中数（A5 Q-SMLA 需要） |

`DeepseekV41SharedAttentionState.reset()` 是**空实现**（[L499-L503](../model.py#L499)）：源层总在任何消费者读之前覆写有效行，保持地址不变可以让 **ACL Graph** 拿到固定地址，回放不依赖 Python 侧的状态变更。

---

## 三、一次请求的生命周期

### 3.1 预填充（prefill）

1. **哈希与查表（Engram，可与主干重叠）**：`prepare_engram_inputs`（[L1404-L1500](../model.py#L1404)）在**辅助流**上完成 n-gram 哈希 + 大表查表，用 `ExternalEvent` 与主图握手；`forward` 里延迟到**第一个真实消费者**（层 1 之前）才 `wait_lookup`（[L1584-L1595](../model.py#L1584)）。细节见 [6_deepseek_v41_engram.md](6_deepseek_v41_engram.md)。
2. **逐层前向**：每层按「hc_pre → LN → DSA → hc_post → hc_pre → LN → MoE → hc_post」走；源层在写到自己的位置时**顺手**产出压缩 KV / 索引 K；索引源产出 TopK。
3. **有界回放（可选）**：当 `cache_config.swa_bounded_replay` 打开且满足 [decoder_replay_layers.py:L84-L141](../decoder_replay_layers.py#L84) 的全部前置条件时，层 21-39 **只对每个请求最后 128 个 token 计算**，其余行的 hidden 由源层缓存补出来。细节见 [9_deepseek_v41_decoder_replay.md](9_deepseek_v41_decoder_replay.md)。
4. **出口**：`hc_collapse` → `norm` → `lm_head`。若开了投机解码，还要在 `hc_collapse` **之前**把完整 4 流 hidden 拷进 `_mtp_hidden_buffer`（[L1614-L1619](../model.py#L1614)），供 DSpark 使用。

### 3.2 解码（decode）

单 token 前向，三条缓存各写一次：

| 缓存 | 写入时机 | 写入算子 |
|------|---------|---------|
| SWA 环（每层） | 每层投影 KV 后 | `write_attention_cache(kind="win")`（A5）或 `npu_scatter_nd_update_sk`（其它） |
| 压缩 KV + 索引 K（仅源层） | 该 token 使压缩组**满**时（c2：奇偶配对完成；c1：每 token） | `kind="cmp"` / `write_index_cache` |
| 压缩状态（仅 c2 源） | 每 token（组未满也写环形状态） | `compressor_from_projected` Triton |

### 3.3 硬件分档

| 能力 | 开关 | 影响 |
|------|------|------|
| `DSV41_PACKED_CACHE`（A5） | `uses_a5_packed_cache()`（[cache_config.py:L45-L47](../cache_config.py#L45)） | 缓存变成「FP8/FP4 值 + BF16 scale」打包字节流；注意力走 `qsmla`；索引走 `run_a5_indexer`（QLI/QSLI）；模型层要求 `requires_uncompiled_fallback`（[model.py:L1655](../model.py#L1655)） |
| 其它设备 | 同上返回 False | 缓存 BF16/INT8；注意力走 `npu_sparse_flash_mla`；索引走 QLI V3（int8 量化） |
| `ENGRAM_MXFP8` / `ENGRAM_UNROTATED_GATE` | `get_current_hardware_profile().supports(...)` | Engram 表可用原生 MXFP8 存储 / 门控可用融合 Triton 核 |

---

## 四、怎么读这套文档

| 你的问题 | 推荐路线 |
|---------|---------|
| 我完全没接触过这个模型 | 本文 → [3 注意力](3_deepseek_v41_attn.md) 第 1 章 → [5 缓存](5_deepseek_v41_kv_cache.md) 第 1 章 |
| 我想搞清楚为什么 KV 能省这么多 | [5_deepseek_v41_kv_cache.md](5_deepseek_v41_kv_cache.md) → [9 有界回放](9_deepseek_v41_decoder_replay.md) |
| 我在调精度/对齐数值 | [3 注意力](3_deepseek_v41_attn.md)（RoPE 基座、压缩位坐标、候选块） → [6 Engram](6_deepseek_v41_engram.md)（门控与旋转基） |
| 我在调性能 | [3](3_deepseek_v41_attn.md) 的多流重叠、[5](5_deepseek_v41_kv_cache.md) 的页宽、[9](9_deepseek_v41_decoder_replay.md)、[4](4_deepseek_v41_moe.md) 的 EPLB |
| 我要写/改算子 | [1](1_npu_hc_pre_npu.md)、[2](2_npu_hc_post_npu.md)（Ascend C 从零到逐行） |
| 我在接投机解码 | [7](7_deepseek_v41_mtp.md) → [8](8_deepseek_v41_dspark.md) → [11](11_deepseek_v41_sampling.md) |
| 我想先看 mHC 的来龙去脉（ResNet → HC → mHC） | 概念篇见 V4 文档树的 [mhc.md](../../deepseek_v4/dsv4_docs/mhc.md)（与模型版本无关的背景阅读）；V4.1 的**延迟系数交接**见 [7](7_deepseek_v41_mtp.md) §2 |
| 我要接多模态 | [10](10_deepseek_v41_multimodal.md) |

---

## 五、术语速查

| 缩写 | 全称 | 一句话解释 |
|------|------|-----------|
| **mHC** | Manifold-Constrained Hyper-Connections | 4 条并行残差流，混合矩阵经 Sinkhorn 约束成双随机矩阵 |
| **DSA** | Deep Sparse Attention | SWA + 压缩 KV + 索引选块的稀疏注意力 |
| **SWA** | Sliding Window Attention | 最近 128 token 的完整 KV（环形缓冲） |
| **C1 / C2** | Compress ratio 1 / 2 | c1 = 每 token 一个压缩 entry（直投）；c2 = 2 个 token 用 softmax 权重池化成 1 个 |
| **QLI** | Quantized Lightning Indexer | 一级索引：在压缩位置上打分（可选产出候选块） |
| **QSLI** | Quantized Sparse Lightning Indexer | 二级索引：只在候选块内细选 Top-512 |
| **kv_source_layer** | — | 真正写压缩 KV 的层（`[2,8,14,20]`） |
| **index_source_layer** | — | 真正跑索引的层（`[2,8,14,20,24,28,32,36]`） |
| **candidates** | — | 层 20 产出的 2048 个候选**块号**，块宽 8 个压缩位 |
| **Engram** | — | n-gram 哈希记忆表（16M 词表基数、384M 行、256 维） |
| **DSpark** | DeepSeek Sparse Block Drafter | 块草稿器，一次前向出 5 个 draft token；**V4.1 的草稿方法就是它**（`method="dspark"`） |
| **DFlash** | — | **另一个**块草稿方法（Qwen3 / Laguna / Glimmer 用），与 DSpark 并列；Ascend 的 DSpark proposer 继承它的骨架，所以 DSpark 路径里会出现 `dflash_*` 名字 |
| **`mtp.*`** | — | V4.1 checkpoint 里 DSpark 权重所在的**目录名**（历史沿用），不代表机制是经典 MTP |
| **bounded replay** | — | 有界回放：源层之后的层只算最后 128 token |
| **MRv1 / MRv2** | Model Runner v1 / v2 | 两套 worker 运行时；Engram 与回放都有 v1/v2 两条路径 |
| **A5** | Ascend 950 系列代号 | 支持打包混合量化缓存（`DSV41_PACKED_CACHE`）的设备档 |

---

*文档版本：与 `vllm_ascend/models/deepseek_v41/` 源码同步；配置数据来自 `DeepSeek-V4.1-Flash-1017`（ModelScope `deepseek-ai/DeepSeek-V4.1-Flash-1017`）。*
