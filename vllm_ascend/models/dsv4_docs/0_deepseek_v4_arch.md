# DeepSeek V4 整体架构详解

> **本文档为 DeepSeek V4 在 Ascend NPU 上的整体架构总览。各子模块的详细实现请参考同级目录下的专题文档。**

> **代码版本冻结声明**：本文基于 **vllm-ascend v0.26.0rc** 源码整理，主文件以 [deepseek_v4.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py)（共 1565 行）为准，文中 `Lxxx` 均指该文件行号。三类 KV 缓存类在本版本中**继承上游 vLLM 基类**（`vllm.models.deepseek_v4.*` / `vllm.v1.attention.backends.mla.sparse_swa`），阅读时请勿与早期版本中"全部在 Ascend 侧自定义"的旧结构混淆。

> **统一模型配置**（取自本仓库 `dsv4_docs/0_DeepSeek-V4-Pro-0813-w4a8_config.json`，即 **DeepSeek-V4-Pro-0813 W4A8 量化版** checkpoint，`torch_dtype=bfloat16`）。文中所有 shape 标注均以此配置为例：

| 分组 | 配置项 | 值 |
|------|--------|-----|
| **基础** | `vocab_size` / `hidden_size` / `num_hidden_layers` | 129280 / 7168 / 61 |
| | `max_position_embeddings` | 1048576（1M 上下文；YaRN 外推 `factor=16`，`original_max_position_embeddings=65536`） |
| **mHC** | `hc_mult` / `hc_eps` / `hc_sinkhorn_iters` | 4 / 1e-6 / 20 |
| **注意力 MLA** | `head_dim` / `qk_rope_head_dim` | 512 / 64 |
| | `q_lora_rank` / `o_lora_rank` / `o_groups` | 1536 / 1024 / 16 |
| | `num_attention_heads` / `num_key_value_heads` | 128 / 1 |
| | `sliding_window` / `rope_theta` / `compress_rope_theta` | 128 / 10000 / 160000 |
| **DSA 稀疏** | `index_topk` / `index_head_dim` / `index_n_heads` | 1024 / 128 / 64 |
| **MoE** | `n_routed_experts` / `num_experts_per_tok` / `n_shared_experts` | 384 / 6 / 1 |
| | `moe_intermediate_size` / `num_hash_layers` | 3072 / 3 |
| | `routed_scaling_factor` / `scoring_func` / `topk_method` | 2.5 / `sqrtsoftplus` / `noaux_tc`（`norm_topk_prob=true`） |
| | `swiglu_limit` | 10.0 |
| **推测解码** | `num_nextn_predict_layers`（MTP） | 1 |
| | `dspark_block_size` / `dspark_markov_rank` / `dspark_noise_token_id` | 5 / 512 / 128799 |
| | `dspark_target_layer_ids` | `[58, 59, 60]`（主模型倒数 3 层） |
| **量化（W4A8 包）** | 专家权重 / 注意力高秩投影 | 专家 `W4A8_DYNAMIC`（`expert_dtype="fp4"`，w1/w2/w3）；`wq_b`、`wo_b` 为 `W8A8_DYNAMIC`；LoRA 低秩侧、RMSNorm、embedding、hc_head 保持 FLOAT（逐张量 dtype 清单见同目录 `0_DeepSeek-V4-Pro-0813-w4a8_quant_model_description.json`） |

---

## 一、架构总览

### 1.1 核心设计理念

DeepSeek V4 是一个融合了多项前沿技术的大规模语言模型，在 Ascend NPU 上的实现围绕三大核心创新展开：

| 技术 | 全称 | 核心作用 |
|------|------|---------|
| **HC / mHC** | **Hyper-Connections（超连接）** 及其流形约束版本 **Manifold-Constrained Hyper-Connections（mHC，流形约束超连接）** | 把一条残差流拓宽为 `hc_mult`（=4）条并行残差流：层前 Pre Mapping 把多流"读出"为一条普通 hidden 送注意力/FFN，层后 Post Mapping 再"写回"；残差混合矩阵经 Sinkhorn 迭代（20 次）约束为双随机矩阵，使超连接在深层网络中仍保持信号稳定 |
| **DSA** | Deep Sparse Attention 深度稀疏注意力 | 三级 KV 缓存（SWA + Compressor + Indexer）协同，在超长上下文下平衡精度与显存 |
| **MoE** | Mixture of Experts 混合专家 | 稀疏激活的专家网络，在大参数量下保持合理的每次推理计算量 |

**三者在模型中的位置**：

```
每个 DecoderLayer（残差流宽度 = hc_mult = 4）:
┌────────────────────────────────────────────────────────────┐
│  Attention 分支                                            │
│    hc_pre → input_layernorm → DSA 注意力 → hc_post         │
│    ↑ Pre Mapping 读出单流  ↑  稀疏注意力  ↑ Post Mapping 写回│
├────────────────────────────────────────────────────────────┤
│  FFN 分支                                                  │
│    hc_pre → post_attention_layernorm → MoE → hc_post       │
│    ↑ Pre Mapping 读出单流  ↑  稀疏专家    ↑ Post Mapping 写回│
└────────────────────────────────────────────────────────────┘
```

### 1.2 文档体系索引

| 编号 | 文档 | 内容 | 详细程度 |
|------|------|------|---------|
| **0** | `0_deepseek_v4_arch.md` | **本文档** — 整体架构总览，模块关系，数据流总览 | ★☆☆ 总览 |
| **1** | `1_npu_hc_pre_npu.md` | `hc_pre` 算子详解：Python 接口 → C++ 绑定 → NPU 内核实现 | ★★★ 详细 |
| **2** | `2_npu_hc_post_npu.md` | `hc_post` 算子详解：Python 接口 → C++ 绑定 → Tiling → NPU 内核 | ★★★ 详细 |
| **3** | `3_deepseek_v4_attn.md` | DSA 注意力架构详解：五层架构 + 完整 Forward 数据流 + 优化要点 | ★★★ 详细 |
| **4** | `4_deepseek_v4_moe.md` | MoE 前馈网络详解：双路由模式 + FusedMoE + EPLB + NPU 算子 | ★★★ 详细 |
| **5** | `5_deepseek_v4_kv_cache.md` | KV 缓存体系详解：Compressor + Indexer + SWA + Block Size + A5 适配 | ★★★ 详细 |
| **6** | `6_deepseek_v4_mtp.md` | MTP 多 Token 预测详解：投影融合 + 轮询调度 + 延迟 HC 压缩 + 权重映射 | ★★★ 详细 |
| **7** | `7_deepseek_v4_postprocess.md` | 后处理与采样详解：LogitsProcessor + ParallelLMHead + 惩罚项 + TopK/TopP + Reduce Sample + 拒绝采样 | ★★★ 详细 |
| **8** | `8_deepseek_v4_dspark.md` | DSpark 稀疏块草稿详解：Markov 头 + 上下文 KV 预计算 + aux 多层隐藏状态融合 + HC 头压缩 + 权重映射 | ★★★ 详细 |

### 1.3 完整模块树

```
AscendDeepseekV4ForCausalLM (L1253, 模型入口)
│   继承: nn.Module + SupportsPP + DeepseekV2MixtureOfExperts
│         + SupportsLoRA + SupportsEagle3
├── model: DeepseekV4Model (L1042, 同时继承 EagleModelMixin)
│   ├── embed_tokens: VocabParallelEmbedding        [仅 PP 首阶段]
│   ├── layers[]: DeepseekV2DecoderLayer × N (L937)
│   │   ├── input_layernorm: RMSNorm
│   │   ├── post_attention_layernorm: RMSNorm
│   │   ├── self_attn: DeepseekV4Attention (L729)
│   │   │   ├── wq_a / q_norm / q_norm_without_weight / wq_b
│   │   │   ├── wkv / kv_norm / wo_a / wo_b / attn_sink
│   │   │   ├── rotary_emb: ComplexExpRotaryEmbedding (支持 c4/c128 分组 RoPE)
│   │   │   ├── compressor: Compressor (L613, compress_ratio>1 时)
│   │   │   ├── indexer: Indexer (L546, 仅 c4 层)
│   │   │   └── dsa_attn: AscendDeepseekSparseAttention (内含 DSAModules)
│   │   │       └── [详细见 3_deepseek_v4_attn.md]
│   │   ├── mlp: DeepseekV4MoE (L357)                [本版本所有层统一 MoE]
│   │   │   └── shared_experts: DeepseekV2MLP (L308) [mix_placement 时为 None]
│   │   ├── hc_pre / hc_post / rms_norm_cast: HC 与归一化算子方法
│   │   │   └── [详细见 1_npu_hc_pre_npu.md / 2_npu_hc_post_npu.md]
│   │   └── hc_attn / hc_ffn: {fn, base, scale} 参数组
│   ├── norm: RMSNorm                              [仅 PP 末阶段]
│   ├── hc_norm: 无权重 RMSNorm(hc_dim, fp32)      [hc_head 专用归一化]
│   ├── hc_head: {fn, base, scale}                [模型级 HC 头]
│   ├── aux_hidden_state_layers: tuple            [Eagle3/DSpark 取数层]
│   ├── topk_indices_buffer: 模型级共享 TopK 索引缓存 (is_v32 时分配)
│   └── _mtp_hidden_buffer: MTP 隐藏状态缓冲区 (max_num_batched_tokens, hc_dim)
├── lm_head: ParallelLMHead                        [仅 PP 末阶段]
└── logits_processor: LogitsProcessor
```

**注意力子模块补充（c4 / c128 层才有）**：

```
Compressor (L613, attention 内 head_dim=512)
├── ape / wkv / wgate: ReplicatedLinear × 3
├── norm: RMSNorm(head_dim, fp32 权重)
└── state_cache: AscendCompressorStateCache (L113)

Indexer (L546, 仅 c4 层, head_dim=index_head_dim=128)
├── wq_b / weights_proj: ReplicatedLinear × 2
├── k_cache: AscendDeepseekV4IndexerCache (L146)
└── compressor: Compressor(rotate=True, head_dim=128)
```

**KV 缓存体系**：

```
├── AscendDeepseekV4SWACache       (L182) — 滑动窗口 KV 缓存 (所有层)
│                                   继承 vllm ...sparse_swa.DeepseekV4SWACache
├── AscendCompressorStateCache     (L113) — Compressor 状态缓存 (c4/c128 层)
│                                   继承 vllm ...deepseek_v4.compressor.CompressorStateCache
└── AscendDeepseekV4IndexerCache   (L146) — Indexer KV 缓存 (c4 层)
                                    继承 vllm ...deepseek_v4.attention.DeepseekV4IndexerCache
```

### 1.4 代码文件索引

| 模块 | 文件路径 | 核心类 / 函数 |
|------|---------|-------------|
| 主模型 | `vllm_ascend/models/deepseek_v4.py` | `AscendDeepseekV4ForCausalLM`, `DeepseekV4Model`, `DeepseekV2DecoderLayer`, `DeepseekV4Attention`, `DeepseekV4MoE`, `DeepseekV2MLP`, `Indexer`, `Compressor` |
| MTP | `vllm_ascend/models/deepseek_v4_mtp.py` | `DeepSeekV4MTP`, `DeepSeekMultiTokenPredictor` |
| 注意力层基类 | `vllm_ascend/models/layer/attention/layer.py` | `DSAAttention`, `DSV4_BLOCK_SIZES`（全局 block size 与 padded page 字典） |
| DSA 注意力后端 | `vllm_ascend/attention/dsa_v1.py` | `AscendDSABackend`, `AscendDSAImpl` |
| DSA 外层封装 | `vllm_ascend/ops/dsa.py` | `AscendDeepseekSparseAttention`, `DSAModules` |
| RoPE | `vllm_ascend/ops/rope_dsv4.py` | `ComplexExpRotaryEmbedding`（default / c4 / c128 分组频率） |
| MoE 路由选择 | `vllm_ascend/ops/fused_moe/experts_selector.py` | `moe_gating_top_k_hash` 等 TopK 选择算子（FusedMoE 内部调用） |
| 缩放融合 | `vllm_ascend/ops/triton/mul_add.py` | `muls_add_triton`（shared + routed 输出融合） |
| 上游缓存基类 | `vllm/models/deepseek_v4/attention.py`、`vllm/models/deepseek_v4/compressor.py`、`vllm/v1/attention/backends/mla/sparse_swa.py` | `DeepseekV4IndexerCache`, `CompressorStateCache`, `DeepseekV4SWACache` |
| HC pre 算子 | `csrc/moe/hc_pre/` | Host + Kernel 实现（注册名 `npu_hc_pre_v2`，旧版 `npu_hc_pre` 仍保留） |
| HC post 算子 | `csrc/moe/hc_post/` | Host + Kernel 实现（注册名 `npu_hc_post`） |

---

## 二、模型入口与整体数据流

### 2.1 DeepseekV4Model — 模型主体

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L1042`（`@support_torch_compile` 装饰，继承 `nn.Module` + `EagleModelMixin`）

#### 核心成员变量

| 成员 | 类型 | 说明 |
|------|------|------|
| `embed_tokens` | VocabParallelEmbedding | 词嵌入层，仅 PP 首阶段实例化 |
| `layers` | ModuleList | 解码器层数组，按 PP 阶段切片（`start_layer` / `end_layer`） |
| `norm` | RMSNorm | 最终归一化，仅 PP 末阶段实例化 |
| `hc_mult` | int | HC 扩展倍数（配置值为 4） |
| `hc_eps` | float | HC 门控 sigmoid 后加的 epsilon（`1e-6`） |
| `norm_eps` | float | RMSNorm epsilon（`1e-6`） |
| `is_v32` | bool | `hasattr(config, "index_topk")`，决定是否分配 TopK 共享缓冲区 |
| `hc_head_fn` | nn.Parameter | `(hc_mult, hc_mult * hidden_size)` HC 头投影权重，fp32 |
| `hc_head_base` | nn.Parameter | `(hc_mult,)` HC 头偏置，fp32 |
| `hc_head_scale` | nn.Parameter | `(1,)` HC 头缩放因子，fp32 |
| `hc_norm` | RMSNorm | **v0.26 新增**：`RMSNorm(hc_dim, has_weight=False, dtype=fp32)`，hc_head 专用的无权重 fp32 归一化 |
| `aux_hidden_state_layers` | tuple[int, ...] | 来自 `EagleModelMixin`（类级默认空元组），需要导出辅助隐藏状态的层号；Eagle3 / DSpark 推测解码时由 `set_aux_hidden_state_layers` 写入 |
| `topk_indices_buffer` | Tensor \| None | 模型级 TopK 索引缓冲区，形状 `(max_num_batched_tokens, index_topk)`，int32 |
| `_mtp_hidden_buffer` | Tensor \| None | MTP 隐藏状态缓冲区，形状 `(max_num_batched_tokens, hc_dim)`，预 hc_head 残差流 |

### 2.2 Forward 整体流程

**调用链**:

```
Scheduler → model_runner.execute_model() → prepare_inputs()
    → self.model(**model_inputs) → DeepseekV4Model.forward()
```

#### 输入参数

| 参数 | 形状 | 说明 |
|------|------|------|
| `input_ids` | `(num_tokens,)` | 展平的 token ID 序列（连续批处理） |
| `positions` | `(num_tokens,)` | 每个 token 在序列中的绝对位置（用于 RoPE） |
| `intermediate_tensors` | `(num_tokens, hc_mult, hidden_size)` \| None | PP 中间张量（非首阶段） |
| `inputs_embeds` | `(num_tokens, hidden_size)` \| None | 预计算嵌入（优先于 input_ids） |

> **注意**: `num_tokens = sum(seq_len for each request in batch)`，不是 `batch_size * seq_len`，vLLM 采用 continuous batching 机制。

#### Forward 阶段总览

| 阶段 | 操作 | 形状变化 | PP 阶段 |
|------|------|---------|---------|
| 1 | **输入嵌入** | `(num_tokens,) → (num_tokens, hidden_size)` | 首阶段 |
| 2 | **拓宽残差流（mHC）** | `unsqueeze(1).repeat` → `(num_tokens, hc_mult, hidden_size)`，1 条流复制为 4 条并行残差流 | 首阶段 |
| 2.5 | **llama4 scaling** | 按位置计算缩放系数（当前版本脚手架，`llama_4_scaling_config` 恒为 `None`，结果为 `None`） | 所有阶段 |
| 3 | **层前向传播** | `(num_tokens, hc_mult, hidden_size) → (num_tokens, hc_mult, hidden_size)`，层内 Pre/Post Mapping 负责读出/写回 | 所有阶段 |
| 3.5 | **aux 隐藏状态收集** | 命中 `aux_hidden_state_layers` 的层，取 `hidden_states.mean(dim=1)` 存入列表 | 末阶段 |
| 4 | **MTP 缓存** | flatten 后 copy 到 `_mtp_hidden_buffer`；FlashComm1 下先 all-gather 再裁 pad | 末阶段 |
| 5 | **mHC head 合流** | `(num_tokens, hc_mult, hidden_size) → (num_tokens, hidden_size)`，4 条残差流加权合并为 1 条 | 末阶段 |
| 6 | **最终归一化** | `(num_tokens, hidden_size) → (num_tokens, hidden_size)` | 末阶段 |
| 7 | **返回** | 普通请求返回 `hidden_states`；aux 列表非空时返回 `(hidden_states, aux_hidden_states)` 元组 | 末阶段 |

> **v0.26 关键点 — aux_hidden_states（Eagle3 / DSpark 取数通道）**：
> 每层 forward 后判断 `layer.layer_idx + 1 in self.aux_hidden_state_layers`（层号按 1 起始计数），命中则把该层 4 条残差流平均后的单流状态 `(num_tokens, hidden_size)` 追加进 `aux_hidden_states`。这是 DSpark 块草稿"同时取主模型多层隐藏状态"的数据来源，详见 `8_deepseek_v4_dspark.md`。
>
> **v0.26 关键点 — MTP buffer 与 FlashComm1**：阶段 4 通过 `get_forward_context().flash_comm_v1_enabled` 判断序列并行是否开启。开启时 token 已被 reduce_scatter 切到各 TP rank，必须先 `tensor_model_parallel_all_gather(..., dim=0)` 聚合并切掉 `pad_size` 个填充 token，再写入 `_mtp_hidden_buffer`，否则非 0 号 rank 的缓冲是陈旧数据，会导致 MTP 出现 NaN、接受率下降。

### 2.3 mHC Head — 模型级末端合流（hc_head）

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L1130`

所有层处理完成后，mHC head 把 4 条并行残差流加权合并回普通 `hidden_size`。v0.26 版本把归一化抽成了独立的无权重 fp32 RMSNorm 模块 `self.hc_norm`（`L1115`），不再在函数内手写 `rsqrt`：

```python
def hc_head(self, x, hc_fn, hc_scale, hc_base):
    shape, dtype = x.size(), x.dtype          # (T, hc_mult, H)
    x = x.flatten(1).float()                  # (T, hc_mult * H)，升 fp32
    x_norm = self.hc_norm(x)                  # 无权重 RMSNorm（fp32），只归一化不缩放
    mixes = F.linear(x_norm, hc_fn)           # (T, hc_mult) 每通道门控打分
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps  # (T, hc_mult) 门控权重
    # 注意：加权用的是未归一化的原始 x，不是 x_norm
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)  # (T, H)
    return y.to(dtype)
```

**流程示意** (hc_mult=4)：

```
输入: (num_tokens, 4, hidden_size)
    │
    ├─ flatten + float → (T, 4*hidden_size)
    ├─ hc_norm(x) → x_norm (T, 4*hidden_size)   [无权重 fp32 RMSNorm]
    ├─ linear(x_norm, hc_fn) → mixes (T, 4)
    ├─ sigmoid(mixes*scale + base) + eps → pre (T, 4) 门控权重
    └─ sum(pre * 原始x.view(shape), dim=1) → (T, hidden_size)
输出: (num_tokens, hidden_size)
```

> **与旧版本的差异**：旧实现为 `mixes = F.linear(x, hc_fn) * rsqrt`（先线性投影再乘 RMSNorm 因子）；v0.26 改为先 `hc_norm` 归一化、再投影，语义更清晰，且归一化模块可被编译器/算子单独融合。加权求和始终使用**归一化前**的 `x`，保留原始幅度信息。

---

## 三、DecoderLayer 结构与 HC 机制

### 3.1 DeepseekV2DecoderLayer

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L937`

每个解码器层包含注意力和 FFN 两个分支，都使用 HC 机制。

#### 核心成员变量

| 成员 | 类型 | 说明 |
|------|------|------|
| `self_attn` | DeepseekV4Attention | 自注意力层 |
| `mlp` | DeepseekV4MoE | 前馈网络层（**v0.26 所有解码器层统一为 MoE**，不再按层切换 dense MLP；`DeepseekV2MLP` 仅作为共享专家存在） |
| `input_layernorm` | RMSNorm | 注意力前置层归一化 |
| `post_attention_layernorm` | RMSNorm | FFN 前置层归一化 |
| `hc_attn_fn / base / scale` | nn.Parameter | 注意力 HC 参数组（fp32） |
| `hc_ffn_fn / base / scale` | nn.Parameter | FFN HC 参数组（fp32） |
| `hc_mult` | int | HC 扩展倍数 |
| `hc_sinkhorn_iters` | int | Sinkhorn 迭代次数（配置值为 20） |
| `hc_eps` | float | HC 门控 epsilon |
| `routed_scaling_factor` | float | 路由输出缩放因子（层内备用，实际缩放发生在 MoE 内，`getattr` 默认 1.0） |

**参数形状**：
- `mix_hc = (2 + hc_mult) * hc_mult` — Pre/Post/Residual 三类映射合计的门控权重数
- `hc_dim = hc_mult * hidden_size` — 拓宽后并行残差流的总维度
- `hc_fn`: `(mix_hc, hc_dim)`，`hc_base`: `(mix_hc,)`，`hc_scale`: `(3,)`（三个标量依次为 pre / post / comb 门控缩放）

### 3.2 mHC 机制概述（Hyper-Connections）

**术语澄清**：代码里的 HC 是 **Hyper-Connections（超连接）**，**不是** "Hyper-Complex"。DeepSeek V4 实际使用的是在 HC 基础上加了稳定性约束的 **mHC（Manifold-Constrained Hyper-Connections，流形约束超连接）**：

- 前身：Hyper-Connections（Zhu et al., 2024，arXiv:2409.19606）
- V4 采用：mHC（arXiv:2512.24880；DeepSeek-V4 技术报告 §2.2），并行残差流数 `hc_mult = 4`

**生活化类比**：普通残差连接 `x + F(x)` 像一条单车道快速路，所有信息挤一条道；HC 在旁边并行修了 4 条车道（4 条并行残差流），层与层之间可以学习"跨车道换道"。但无约束地随便换道会在深层网络中引发信号爆炸/消失，mHC 就用"流形约束"给换道规则立规矩——跨流混合必须是一个**非负、且每行每列之和都为 1 的双随机矩阵**（等价于做加权平均/置换混合，谱范数 ≤ 1），信号无论穿过多少层都不会被放大。

**mHC 的三类映射与代码对应**：

| mHC 概念 | 代码位置 | 作用 |
|---------|---------|------|
| **Pre Mapping（前置映射）** | `hc_pre()` → `npu_hc_pre_v2` | 从 4 条并行残差流中"读出"一条普通 `hidden_size` 向量（后续过 RMSNorm 送注意力/FFN）；算子内部同时算出 Post Mapping 要用的 `post`、`comb` 中间量 |
| **Residual Mapping（残差混合矩阵）** | `npu_hc_pre_v2` 内部，参数 `hc_sinkhorn_iters=20` | 控制残差流之间如何混合；经 **Sinkhorn-Knopp 迭代 20 次**投影到**双随机矩阵流形（Birkhoff 多胞形）**，保证非扩张性与连乘封闭性 |
| **Post Mapping（后置映射）** | `hc_post()` → `npu_hc_post` | 把子层（注意力/FFN）输出按**非负有界门控**（`sigmoid + hc_eps`）"写回" 4 条残差流 |
| **mHC Head（末端合流）** | `DeepseekV4Model.hc_head()`（§2.3） | 所有层结束后，把 4 条残差流加权合并回 1 条 `hidden_size` 向量，再做最终 RMSNorm |

**核心思想三步走**：
1. **拓宽残差流**：输入端把 `(num_tokens, hidden_size)` 沿 `hc_mult` 维复制为 `(num_tokens, 4, hidden_size)`；注意力/FFN 子层始终只在普通 `hidden_size` 宽度上算，不增加子层 FLOPs，额外开销只在流间小规模混合
2. **带约束的跨流混合**：残差混合矩阵走 Sinkhorn 双随机约束（`hc_sinkhorn_iters=20`）；Pre/Post 门控走 sigmoid 非负有界约束。每层的 `hc_scale` 是 `(3,)` 三个标量，分别作用于 **pre 门控 / post 门控 / comb（残差混合）门控**
3. **末端合流**：模型级 mHC head（`hc_head_fn/base/scale` + 无权重 `hc_norm`）把 4 条流压回 1 条

**层内相关算子方法**（`L996` / `L1007` / `L1013`）：
- **hc_pre**: Pre Mapping + Residual Mapping — v0.26 调用的算子注册名是 **`npu_hc_pre_v2`**（旧版 `npu_hc_pre` 仍在 C++ 侧保留注册，详见 `1_npu_hc_pre_npu.md`），返回 `(hidden_states, post, comb)` 三元组
- **hc_post**: Post Mapping — 调用 `npu_hc_post`（入参会先 `unsqueeze(0)` 补 batch 维，返回后 `squeeze(0)`）
- **rms_norm_cast**（v0.26 新增）: FFN 分支专用，一次归一化同时产出 bf16/fp16 的 `hidden_states`（送专家计算）和 fp32 的 `hidden_states_fp32`（送路由器打分）；开启 `enable_custom_op()` 时走融合算子 `npu_rms_norm_cast`，否则退回 `post_attention_layernorm` + `.float()`

> **详细实现**：
> - hc_pre 算子 → 参考 `1_npu_hc_pre_npu.md`
> - hc_post 算子 → 参考 `2_npu_hc_post_npu.md`

### 3.3 DecoderLayer Forward 完整流程

```
输入: hidden_states (num_tokens, hc_mult, hidden_size)
      （forward 签名里的 residual 入参会被立即覆盖，以 clone 为准）
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Attention 分支                                              │
├─────────────────────────────────────────────────────────────┤
│  1. residual = hidden_states.clone()                        │
│     保存完整 4 条并行残差流                                   │
│                                                             │
│  2. hidden_states, post, comb = hc_pre(                     │
│        hidden_states, hc_attn_fn, hc_attn_scale, hc_attn_base)│
│     npu_hc_pre_v2 / Pre Mapping 读出单流: (T, 4, H) → (T, H) │
│     （内部同时做 Sinkhorn 双随机残差混合，产出 post/comb）    │
│                                                             │
│  3. hidden_states = input_layernorm(hidden_states)          │
│     RMSNorm 归一化                                           │
│                                                             │
│  4. hidden_states = self_attn(                              │
│        positions, hidden_states, llama_4_scaling)           │
│     DSA 稀疏注意力（llama_4_scaling 当前恒为 None）           │
│                                                             │
│  5. hidden_states = hc_post(                                │
│        hidden_states, residual, post, comb)                 │
│     Post Mapping 写回 4 条残差流: (T, H) → (T, 4, H)         │
└──────────────────────────────┬──────────────────────────────┘
                               │ hidden_states: (T, hc_mult, H)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  FFN 分支                                                    │
├─────────────────────────────────────────────────────────────┤
│  6. residual = hidden_states.clone()                        │
│     保存完整 4 条并行残差流                                   │
│                                                             │
│  7. hidden_states, post, comb = hc_pre(                     │
│        hidden_states, hc_ffn_fn, hc_ffn_scale, hc_ffn_base) │
│     npu_hc_pre_v2 / Pre Mapping 读出单流                     │
│                                                             │
│  8. hidden_states, hidden_states_fp32 = rms_norm_cast(...)  │
│     一次归一化产出两份:                                       │
│       hidden_states      (T, H) bf16/fp16 → 送专家计算       │
│       hidden_states_fp32 (T, H) fp32      → 送路由器打分     │
│                                                             │
│  9. hidden_states = mlp(                                    │
│        hidden_states, hidden_states_fp32=hidden_states_fp32)│
│     DeepseekV4MoE 计算（统一 MoE，无 dense 分支）            │
│                                                             │
│  10. hidden_states = hc_post(                               │
│         hidden_states, residual, post, comb)                │
│      Post Mapping 写回 4 条残差流                            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
                 返回 (hidden_states, residual)
                 hidden_states: (num_tokens, hc_mult, hidden_size)
                 residual:      (num_tokens, hc_mult, hidden_size)
```

> **为什么路由器要单独吃一份 fp32 输入？** 路由 logits 决定每个 token 选哪些专家，对数值精度敏感；而专家 GEMM 走低精度（bf16/fp16 甚至 fp8）即可。`npu_rms_norm_cast` 用一次算子同时给出两种 dtype，避免"归一化一次 + 再 cast 一次"的额外开销。

> **设计要点**:
> 1. **计算效率**: 注意力/FFN 子层始终只在 Pre Mapping 读出的普通 `hidden_size` 宽度上执行，拓宽残差流几乎不增加子层 FLOPs
> 2. **表达能力 + 稳定性**: 4 条并行残差流 + 学习型跨流混合保留多流信息容量；残差混合矩阵经 Sinkhorn 约束为双随机矩阵，深层堆叠也不会信号爆炸/消失（这是 mHC 相对原始 HC 的关键改进）
> 3. **残差连接**: 每个分支前保存完整 4 流残差（`clone()`），hc_post 中按门控写回
> 4. **返回值**: 返回 `(hidden_states, residual)` 元组（与标准 Transformer 不同）

### 3.4 mHC 参数分布汇总

| 参数位置 | fn 形状 | base 形状 | scale 形状 |
|---------|--------|----------|-----------|
| `hc_attn` (每层注意力) | `(mix_hc, hc_dim)` | `(mix_hc,)` | `(3,)` |
| `hc_ffn` (每层 FFN) | `(mix_hc, hc_dim)` | `(mix_hc,)` | `(3,)` |
| `hc_head` (模型级) | `(hc_mult, hc_dim)` | `(hc_mult,)` | `(1,)` |

其中：
- `mix_hc = (2 + hc_mult) * hc_mult`
- `hc_dim = hc_mult * hidden_size`

---

## 四、注意力机制总览

### 4.1 DSA — Deep Sparse Attention

DeepSeek V4 采用 **MLA (Multi-head Latent Attention)** + **DSA (Deep Sparse Attention)** 架构，通过三级 KV 缓存体系实现超长上下文推理。

> **完整详解** → 参考 `3_deepseek_v4_attn.md`

### 4.2 三层 KV 缓存架构

```
原始 KV (O(N²) 显存)
    │
    ├── SWA (滑动窗口): 最近 window_size 个 token 的完整 KV
    │                    → 高精度，近期记忆
    │
    ├── Compressor (KV压缩): 历史 KV 按 4x / 128x 压缩存储
    │                       → 减少显存，长程记忆
    │
    └── Indexer (稀疏索引): compress_ratio=4 时，TopK 选择最重要的 token
                            → 进一步减少计算量
```

### 4.3 按层压缩配置

| compress_ratio（config 原值） | Compressor | Indexer | SWA | 适用场景 |
|---------------|-----------|---------|-----|---------|
| **0** | ✗ | ✗ | ✓ | dense 无压缩：不建 Compressor/Indexer，仅滑动窗口。本配置中对应 3 个 draft 层（MTP/DSpark） |
| **4** | ✓ (overlap=True) | ✓ (TopK 选择) | ✓ | 中距离层，稀疏索引+压缩 |
| **128** | ✓ (overlap=False) | ✗ | ✓ | 长距离层，深度压缩 |

- 通过 `get_dsv4_compress_ratio(config, config_layer_idx)` 获取（`L814`），代码统一以 **`compress_ratio > 1`** 判断是否建 Compressor、以 **`== 4`** 判断是否建 Indexer；数组越界时按 0（dense）兜底
- `config_layer_idx` 由 `extract_dsv4_layer_index(config, prefix)` 得到：主层用物理层号，MTP/DSpark 层（prefix 含 `.mtp.`）重映射为 `num_hidden_layers + layer_idx`，以便索引 `compress_ratios` 等按层配置数组
- 本配置 `compress_ratios` 共 **64 项 = 61 个主层 + 3 个 draft 层**：主层为 `128, 128, 4, 128, 4, …, 128, 4`（**31 个 c128 层 + 30 个 c4 层**，前两层均为 128，之后交替，末层为 c4）；末尾追加 3 个 `0` 对应索引 61~63 的 3 个 draft 层（dense）

### 4.4 五层架构分层

```
第1层: DeepseekV4Attention       — Python 封装层，初始化所有子模块
第2层: AscendDeepseekSparseAttention — 外层封装，DSAModules 组装
第3层: DSAAttention              — 注意力层基类，KV cache 管理
第4层: AscendDSABackend          — NPU 后端注册
第5层: AscendDSAImpl             — 核心计算实现 (prefill / decode)
```

### 4.5 Forward 五大阶段

| 阶段 | 操作 | 关键组件 |
|------|------|---------|
| 1 | **MLA Prolog** | Query: wq_a → q_norm → wq_b → q_rms → RoPE<br>KV: wkv → kv_norm → RoPE → SWA cache |
| 2 | **Compressor** | KV 压缩 (compress_ratio > 1 时) |
| 3 | **Indexer** | TopK 稀疏选择 (compress_ratio == 4 时) |
| 4 | **稀疏注意力** | `npu_sparse_flash_attention` NPU 算子 |
| 5 | **输出投影** | nope_rope 逆向 → wo_a → wo_b |

> **完整数据流与 shape 追踪** → 参考 `3_deepseek_v4_attn.md` 第五章

### 4.6 IndexCache — 索引缓存复用

通过 hf-overrides 开启 `use_index_cache` 后，部分 c4 层可以复用更早层算出的 TopK 索引以减少计算（参考论文链接见源码 `L861` 注释）。

**生效前提（三个条件同时满足，`L866-867`）**：
1. 本层 `compress_ratio == 4`（真正拥有 Indexer）
2. 配置中 `use_index_cache=True`
3. 层名 prefix **不含** `.mtp.`（MTP 草稿层在模型级共享 `topk_indices_buffer`，impl 层引用会过期，故排除）

**"第几号 indexer 层"的计数方式**：不看物理层号，而是统计 `compress_ratios[:config_layer_idx]` 中 `4` 出现的次数，得到 `indexer_seq_idx`（从 0 开始）。

| 模式 | 配置项 | skip_topk 判定 |
|------|--------|---------------|
| 频率模式 | `index_topk_freq`（默认 1） | `max(indexer_seq_idx - 1, 0) % freq != 0` 即跳过（首个 indexer 层永远自己算） |
| 模式匹配 | `index_topk_pattern`，如 `"FSFS"` | 取 `pattern[indexer_seq_idx]`，`"S"` 跳过；断言必须以 `"F"` 开头 |

- `skip_topk=True`: 跳过本层 TopK 计算，从模型级 `topk_indices_buffer` 复用前序结果
- `skip_topk=False`: 计算索引并把结果写入 `topk_indices_buffer`

### 4.7 v0.26 注意力相关补充

| 要点 | 说明 |
|------|------|
| **分组 RoPE** | c4/c128 压缩层把 `rope_theta` 改为 `config.compress_rope_theta`（配置值 160000），并向 `ComplexExpRotaryEmbedding` 注册 `rope_groups=["default", f"c{compress_ratio}"]`；普通层用 `rope_theta=10000`、`["default"]` |
| **q_norm_without_weight** | 除带权重的 `q_norm`（作用于 q_lora_rank）外，新增 `RMSNorm(head_dim, has_weight=False)` 无权重归一化，随 `DSAModules` 一并下发给 DSA 后端 |
| **attn_sink 形状** | 开启 `enable_dsa_cp()`（CP 并行）时为全头数 `n_heads`，否则按 TP 切分为 `n_local_heads`；权重加载时 CP 整份 copy、非 CP 按 `narrow` 切头 |
| **forward 第三参数** | `DeepseekV4Attention.forward(positions, hidden_states, llama_4_scaling)`，逐层透传模型级 llama4 缩放系数（当前版本为 `None`） |

---

## 五、MoE 前馈网络

### 5.1 DeepseekV4MoE

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L357`

#### 核心组成

| 类别 | 组件 | 说明 |
|------|------|------|
| **路由门控** | `gate` | ReplicatedLinear，`(hidden_size, n_routed_experts)`，`precast_fp32_weight=True`；hash 层挂 `tid2eid`，线性层挂 `e_score_correction_bias` |
| **专家网络** | `experts` | FusedMoE 融合专家网络；v0.26 可由 `experts.is_internal_router` 决定 gate 是否在 FusedMoE **内部**执行 |
| **共享专家** | `shared_experts` | DeepseekV2MLP 或 `None`：`n_shared_experts` 为空、或 Ascend 配置 `mix_placement=True` 时为 `None`（共享专家被融进 FusedMoE 专家槽位） |
| **配置** | `n_routed_experts` / `n_shared_experts` | 路由专家数 384 / 共享专家数 1 |
| | `n_redundant_experts` | 冗余专家数 (EPLB)，并据此算出物理/本卡专家数 |
| | `hash` | 是否使用 hash 路由：`layer_idx < num_hash_layers and not is_draft_layer`（前 3 层） |
| | `enable_eplb` | 启用专家并行负载均衡 |
| | `routed_scaling_factor` | 路由输出缩放因子：代码 `getattr` 默认 1.5，本模型配置实际为 **2.5** |
| | `scoring_func` | 路由打分函数，配置为 `sqrtsoftplus`（默认 `softmax`），透传给 FusedMoE |
| | `swiglu_limit` | **v0.26 新增**，配置值 10.0，激活函数做 clamp（见 5.4），透传给 FusedMoE 与共享专家 |
| | `is_sequence_parallel` | 来自 `parallel_config.use_sequence_parallel_moe`，开启后输入按 SP 分片、权重复制 |

> **缩放顺序的版本约定（`L452-455` 注释）**：DeepSeek V4 的顺序是"先归一化 top-k 权重 → 再对 routed 输出整体乘 `routed_scaling_factor`"，所以缩放保留在 router 路径之外；仅 ROCm AITER 路径会在内部自行应用该因子。

### 5.2 双路由模式

| 模式 | 触发条件 | 实现方式 |
|------|---------|---------|
| **Hash 路由** | `layer_idx < num_hash_layers` 且非 draft 层（前 3 层） | 预定义 `tid2eid` 表 `(vocab_size, num_experts_per_tok)`，int32 不可学习，`e_score_correction_bias=None`；实际 TopK 选择算子 `moe_gating_top_k_hash` 在 FusedMoE 内部（`vllm_ascend/ops/fused_moe/experts_selector.py`）被调用 |
| **线性路由** | 其他层 | gate 线性层（或 FusedMoE 内部路由器）+ `e_score_correction_bias` 修正偏置，`sqrtsoftplus` 打分 + 分组 topk |

forward 中按 `self.experts.is_internal_router` 二选一（`L486-494`）：

- **内部路由**：直接把（fp32 的）`router_input` 作为 `router_logits` 传给 `self.experts(...)`，gate 在 FusedMoE 类里跑；
- **外部路由**：`router_logits = F.linear(router_input.float(), self.gate.weight)`，shape 为 `(num_tokens, n_routed_experts)`，再交给 FusedMoE。

两种情况下路由输入都优先使用 DecoderLayer 传入的 **fp32** `hidden_states_fp32`，没有时才对 `hidden_states` 现场 `.float()`。

### 5.3 Forward 流程

```
输入: hidden_states      (num_tokens, hidden_size) bf16/fp16
      hidden_states_fp32 (num_tokens, hidden_size) fp32（来自 rms_norm_cast）
    │
    ├─ [序列并行] sequence_parallel_chunk 对两份输入同时分片
    │
    ├─ 路由计算:
    │   ├─ is_internal_router=True  → router_input 直接传 FusedMoE 内部跑 gate
    │   └─ is_internal_router=False → F.linear(router_input, gate.weight) → router_logits
    │
    ├─ FusedMoE 专家计算:
    │   └─ experts(hidden_states, router_logits=...)
    │      返回 (shared_output, final_hidden_states) 元组 或 单个 tensor
    │
    ├─ 输出融合（仅元组路径，按 dtype 分支；算子语义 out = x*scale + y）:
    │   ├─ 非 fp16 且非 ROCm AITER:
    │   │     有共享专家 → muls_add_triton(final, shared, 2.5)
    │   │                 = routed * 2.5 + shared
    │   │     无共享专家 → final *= 2.5
    │   └─ fp16 且有共享专家 → muls_add_triton(shared, final, 1/2.5)
    │            = shared / 2.5 + routed
    │            （fp16 动态范围小，用"先除因子再相加"避免 routed×2.5 溢出；
    │             与非 fp16 路径整体相差一个 factor 尺度，是独立的 fp16 约定，勿混用）
    │
    ├─ [序列并行] all_gather(dim=0) + 裁掉 num_tokens 之后的 pad
    ├─ [TP>1 且为旧元组输出] maybe_all_reduce_tensor_model_parallel
    │       （新版单 tensor 输出已由上游 MoERunner 做过最终归约，不再重复）
    └─ 输出: (num_tokens, hidden_size)
```

### 5.4 DeepseekV2MLP（共享专家 / 稠密 MLP）

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L308`

v0.26 中它**不再作为解码器层的 dense 分支**（解码器层统一是 `DeepseekV4MoE`），主要用途是 MoE 的 `shared_experts`（`intermediate_size = moe_intermediate_size * n_shared_experts`，且 `reduce_results=False`）：

```
hidden → gate_up_proj → act_fn(SiluAndMul / SiluAndMulWithClamp) → down_proj → output
```

| 组件 | 类型 | 说明 |
|------|------|------|
| `gate_up_proj` | MergedColumnParallelLinear | 门控 + 上投影合并；SP 模式下 `disable_tp=True` 权重复制 |
| `act_fn` | SiluAndMul / **SiluAndMulWithClamp** | 配置了 `swiglu_limit`（10.0）时使用带 clamp 的版本，防止激活值过大 |
| `down_proj` | RowParallelLinear | 下投影；SP 模式下 `disable_tp=True`，共享专家路径 `reduce_results=False` |

---

## 六、KV 缓存体系

### 6.1 三类 KV 缓存对比

| 缓存类型 | 所属 | 压缩比 | dtype (A5 / 非A5) | 上游基类 |
|---------|------|--------|-------------------|---------|
| **SWA KV Cache** | `swa_cache_layer` | 1x | float8_e4m3fn / bfloat16 | `vllm.v1.attention.backends.mla.sparse_swa.DeepseekV4SWACache` |
| **Compress State Cache** | `compressor.state_cache` | 4x / 128x | float32（两种设备一致） | `vllm.models.deepseek_v4.compressor.CompressorStateCache` |
| **Indexer KV Cache** | `indexer.k_cache` | 4x | float8_e4m3fn / int8 | `vllm.models.deepseek_v4.attention.DeepseekV4IndexerCache` |

> v0.26 起三个 Ascend 缓存类都改为**继承上游 vLLM 基类、只重写 `get_kv_cache_spec` / `get_attn_backend`**，构造与通用逻辑复用上游。

### 6.2 AscendCompressorStateCache

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L113`

- **state_dim**: `2 * coff * head_dim`（c4）或 `2 * head_dim`（c128）
  - `coff = 1 + overlap`，c4 时 `overlap=True → coff=2`；c128 时 `overlap=False → coff=1`
  - 注意 attention 内 Compressor 的 `head_dim=512`，而 Indexer 内部那个 Compressor 的 `head_dim=index_head_dim=128`
  - 前半维放 `kv_state`、后半维放 `score_state`，故整体乘 2
- **dtype**: float32
- **block_size**: 取自 `DSV4_BLOCK_SIZES[全局block_size][0][2]`（c4）/ `[0][3]`（c128）
- **page_size_padded**: 取自字典第二组 `pads`；当 `state_dim == 2 * 256 且 compress_ratio == 4`（即 Indexer 的 c4 state）用 `pads[0]`，其余用 `pads[1]`
- 返回 `AscendSlidingWindowMLASpec`

### 6.3 AscendDeepseekV4IndexerCache

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L146`

- 仅 `compress_ratio == 4` 时由 Indexer 创建
- **dtype**: A5 → float8_e4m3fn（同时回写 `cache_config.cache_dtype`），其他设备 → int8
- **block_size**: `DSV4_BLOCK_SIZES[全局block_size][0][0]`
- 返回 `AscendMLAAttentionSpec`，v0.26 携带的新字段：
  - `model_version="deepseek_v4"`、`compress_ratio`、`cache_dtype_str`
  - `scale_dim=1 if head_dim == 128 else 0`（是否带 per-token 缩放维）
  - `scale_dtype`: A5 → float32，其他设备 → float16

### 6.4 AscendDeepseekV4SWACache

**文件位置**: `vllm_ascend/models/deepseek_v4.py:L182`

- **window_size**: 来自 `config.sliding_window`（128）
- 构造时先以 `torch.uint8` 调上游基类，再用 `self.dtype` 覆盖为 A5 → float8_e4m3fn / 其他 → bfloat16
- **block_size**: `DSV4_BLOCK_SIZES[全局block_size][0][1]`
- **cached_head_size**: A5 设备额外 `+128`（放置 FP8 缩放等附加数据）
- spec 中同样带 `cache_dtype_str` 与 `model_version="deepseek_v4"`
- 返回 `AscendSlidingWindowMLASpec`

### 6.5 Block Size 配置

**定义位置**: `vllm_ascend/models/layer/attention/layer.py` 中的 `DSV4_BLOCK_SIZES`（模块导入时按设备类型一次性选定）。

字典结构为 `{全局block_size: [[indexer/mla, swa, c4_state, c128_state], [padded_a, padded_b]]}`：

| 全局 block_size | indexer/mla | swa | c4_state | c128_state | padded (a / b) |
|----------------|-----|-----|----------|-----------|----------------|
| **非 A5** ||||||
| 128 | 128 | 128 | 8 | 32 | 16640 / 131072 |
| 64 | 64 | 64 | 4 | 16 | 8320 / 65536 |
| 32 | 32 | 32 | 2 | 8 | 4160 / 32768 |
| **A5** ||||||
| 128 | 128 | 128 | 8 | **16** | 16896 / 81920 |
| 64 | 64 | 64 | 4 | **8** | 8448 / 40960 |
| 32 | 32 | 32 | 2 | **4** | 4224 / 20480 |

- `c4/c128_state` 是 Compressor 状态块的逻辑 block size（压缩后 token 密度不同，故远小于 128）
- A5 与非 A5 的主要差异在 **c128_state（减半）** 和 **padded page 大小**
- padded 组用于 `AscendCompressorStateCache` 的 `page_size_padded`：Indexer 的 c4 state 用 `padded_a`，attention Compressor 用 `padded_b`

---

## 七、MTP 多 Token 预测

### 7.1 MTP 模块架构

**文件位置**: `vllm_ascend/models/deepseek_v4_mtp.py`

```
DeepSeekV4MTP (入口类)
└── model: DeepSeekMultiTokenPredictor
    ├── layers: ModuleDict[str, DeepSeekMultiTokenPredictorLayer] (按 step_idx 轮询)
    │   ├── e_proj: ReplicatedLinear (嵌入投影)
    │   ├── h_proj: ReplicatedLinear (隐藏状态投影)
    │   ├── enorm: RMSNorm (嵌入归一化)
    │   ├── hnorm: RMSNorm (隐藏状态归一化)
    │   ├── shared_head: SharedHead
    │   │   ├── norm: RMSNorm
    │   │   └── head: ParallelLMHead
    │   ├── mtp_block: DeepseekV2DecoderLayer (is_draft_layer=True)
    │   └── hc_head {fn, base, scale}: HC 头参数
    ├── embed_tokens: VocabParallelEmbedding
    └── logits_processor: LogitsProcessor
```

### 7.2 核心特性

| 特性 | 说明 |
|------|------|
| **轮询调度** | `num_mtp_layers` 个 MTP 层按 `spec_step_idx % num_mtp_layers` 轮询使用 |
| **投影融合** | `e_proj(inputs_embeds) + h_proj(previous_hidden_states)` 广播相加 |
| **Mask 机制** | `position == 0` 的输入嵌入被 mask 为 0 |
| **延迟 HC 压缩** | Layer forward 保持 `hc_mult` 维度，仅在 `compute_logits` 时做 hc_head 压缩 |
| **共享 LM Head** | `shared_head` 被所有 MTP 步骤复用 |
| **缓冲区共享** | `topk_indices_buffer` 和 `_mtp_hidden_buffer` 在主模型和 MTP 间共享 |

### 7.3 与主模型的交互

- `_mtp_hidden_buffer`: 主模型每层跑完后、hc_head 之前的残差流，形状 `(max_num_batched_tokens, hc_mult * hidden_size)`；在 cudagraph 池外分配稳定地址，forward 中以 `copy_` 刷新，保证不同 capture shape 下都有效
- **FlashComm1（序列并行）注意**：此时缓冲写入前要先做 `tensor_model_parallel_all_gather(dim=0)` 并裁掉 `pad_size`，否则 MTP 拿到的是分片+陈旧数据（详见 §2.2）
- `get_mtp_target_hidden_states()`（`L1336`）: 返回该缓冲供 MTP **draft 模型**使用（每个 target step 之后有效）

### 7.4 DSpark — 另一条推测解码路径

除经典串行 MTP 外，v0.26 主模型还为 **DSpark（DeepSeek Sparse Block Drafter）稀疏块草稿**预留了取数通道：

| 对比项 | MTP（文档 6） | DSpark（文档 8） |
|--------|--------------|------------------|
| 草稿方式 | 逐 token 串行多步 | 一次性生成整个 draft block |
| 主模型取数 | `_mtp_hidden_buffer`（最终层前单份状态） | `aux_hidden_states`（`aux_hidden_state_layers` 指定的多层状态） |
| 入口协议 | MTP 模块 | Eagle3（`SupportsEagle3` + `EagleModelMixin`，`set_aux_hidden_state_layers`） |
| 权重命名空间 | `mtp.*` | 同样存放在 checkpoint 的 `mtp.*` 下，加载时按语义区分 |

> DSpark 的 Markov 头、上下文 KV 预计算、多层隐藏状态融合等完整机制 → 参考 `8_deepseek_v4_dspark.md`

---

## 八、并行策略与 NPU 优化

### 8.1 并行策略总览

```
├── TP (Tensor Parallel) 张量并行
│   ├── wq_b: enable_dsa_cp 时复制，否则列切分
│   ├── wo_a: 列切分
│   ├── wo_b: 行切分 + allreduce
│   ├── gate_up_proj: 列切分
│   └── down_proj: 行切分
│
├── PP (Pipeline Parallel) 流水线并行
│   ├── embed_tokens: 首阶段
│   ├── layers: 按 start_layer / end_layer 切片
│   ├── norm + lm_head: 末阶段
│   └── IntermediateTensors 传递 hc_mult 维隐藏状态
│
├── EP (Expert Parallel) 专家并行
│   ├── 物理专家按 ep_size 切分
│   └── EPLB 支持专家负载均衡重映射
│
├── CP (Context Parallel) 上下文并行
│   └── enable_dsa_cp() 时启用，attn_sink 全头数，wq_b 用 ReplicatedLinear
│
└── SP (Sequence Parallel) 序列并行
    ├── MLP 输入 sequence_parallel_chunk 分片
    ├── FlashComm1: 每层输出 reduce_scatter
    └── MTP 缓冲区保存前 all_gather 聚合
```

### 8.2 NPU 核心算子

| 功能 | NPU 算子 | v0.26 备注 |
|------|---------|-----------|
| 稀疏注意力 | `torch.ops._C_ascend.npu_sparse_flash_attention` | — |
| HC 前置处理 | `torch.ops._C_ascend.npu_hc_pre_v2` | **本版本改用 v2**；旧 `npu_hc_pre` 仍注册保留 |
| HC 后置处理 | `torch.ops._C_ascend.npu_hc_post` | — |
| 归一化 + dtype 转换 | `torch.ops._C_ascend.npu_rms_norm_cast` | **新增**：一次产出 bf16/fp16 隐藏状态 + fp32 路由输入，受 `enable_custom_op()` 开关控制 |
| MoE hash 路由 | `torch.ops._C_ascend.moe_gating_top_k_hash` | 不在主模型文件直接调用，而在 FusedMoE 内部的 `ops/fused_moe/experts_selector.py` 中使用 |
| RoPE 旋转 | `torch_npu.npu_rotary_mul` | 主 RoPE（`ComplexExpRotaryEmbedding`）与 Compressor 的 `rope_single` 都用它，`rotary_mode="interleave"` |
| Compressor | `torch.ops._C_ascend.compressor` | — |
| Indexer | `npu_lightning_indexer` / `npu_vllm_quant_lightning_indexer_metadata` | — |
| 缩放融合 | `muls_add_triton`（Triton 算子，`ops/triton/mul_add.py`） | shared + routed × factor 的融合乘加 |

### 8.3 设备类型适配

| 特性 | A5 设备 | 非 A5 设备 |
|------|--------|-----------|
| SWA KV dtype | float8_e4m3fn | bfloat16 |
| Indexer KV dtype | float8_e4m3fn | int8 |
| Compressor state / norm dtype | float32（两种设备一致，norm 权重固定 fp32） | float32 |
| Indexer 缩放因子 | `scale_dim=1`（head_dim=128 时），`scale_dtype=float32` | `scale_dim=1`，`scale_dtype=float16` |
| SWA cached_head_size | `head_dim + 128`（附加缩放数据） | `head_dim` |
| Block size 配置 | `_DSV4_BLOCK_SIZES_A5`（c128_state 与 padded 值不同） | `_DSV4_BLOCK_SIZES` 标准表 |

### 8.4 性能优化要点

| 优化类别 | 优化项 |
|---------|--------|
| **计算优化** | 低秩分解 (wq_a/wq_b, wo_a/wo_b)、分组输出、无权重 RMSNorm |
| **存储优化** | MLA 架构 (KV 共享 latent)、KV 压缩 (4x/128x)、稀疏索引、IndexCache、FP8 存储 |
| **并行优化** | 多流重叠 (Query/KV 流并行)、CV 分离 (Vector + Cube)、序列并行 |
| **算子卸载** | 整个稀疏注意力完全卸载到 NPU 算子执行 |
