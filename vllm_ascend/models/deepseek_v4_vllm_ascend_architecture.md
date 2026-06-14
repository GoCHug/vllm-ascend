# DeepSeekV4 架构文档

***

## 一、完整架构概览

```
AscendDeepseekV4ForCausalLM (模型入口类)
├── model: DeepseekV4Model
│   ├── embed_tokens (VocabParallelEmbedding)
│   ├── layers[]: DeepseekV2DecoderLayer × N
│   │   ├── input_layernorm (RMSNorm)
│   │   ├── self_attn: DeepseekV4Attention
│   │   ├── post_attention_layernorm (RMSNorm)
│   │   ├── mlp: DeepseekV4MoE / DeepseekV2MLP
│   │   ├── hc_pre/hc_post (混合协同处理)
│   │   └── hc_head_fn/hc_head_scale/hc_head_base
│   ├── norm (RMSNorm)
│   ├── hc_head_fn/hc_head_scale/hc_head_base
│   └── _mtp_hidden_buffer (MTP隐藏状态缓冲区)
├── lm_head (ParallelLMHead)
└── logits_processor (LogitsProcessor)
```

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L1212`

***

## 二、核心组件详解

### 2.1 DeepseekV4Model

**初始化组件**:

- `embed_tokens`: 词嵌入层 (VocabParallelEmbedding)
- `layers[]`: 解码器层堆叠 (DeepseekV2DecoderLayer × N)
- `norm`: 最终归一化 (RMSNorm)
- `hc_mult`: HC倍数
- `hc_eps`: HC epsilon
- `hc_head_fn/base/scale`: HC头参数
- `topk_indices_buffer`: TopK索引缓冲区
- `_mtp_hidden_buffer`: MTP隐藏状态缓冲区

**forward 输入参数说明**:

`forward` 方法接收来自 vLLM model runner 阶段的输入参数。以下是从 vLLM Scheduler 到模型 forward 的完整调用链路：

```
Scheduler → model_runner.execute_model() → prepare_inputs() → self.model(**model_inputs) → DeepseekV4Model.forward()
```

| 参数                     | 类型                            | 形状                         | 来源与说明                                                                                                                                                                                                                |
| ---------------------- | ----------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_ids`            | `torch.Tensor`                | `(num_tokens,)`            | 展平的 token ID 序列，由 vLLM model runner 从多个请求聚合而来**vLLM 源码**: `vllm/v1/worker/gpu/model_runner.py:1142`**聚合函数**: `prepare_inputs()` 第871行**准备输入**: `input_ids = self.input_buffers.input_ids[:num_tokens_after_padding]` |
| `positions`            | `torch.Tensor`                | `(num_tokens,)`            | 每个 token 在序列中的绝对位置索引，用于 RoPE 旋转位置编码**vLLM 源码**: `vllm/v1/worker/gpu/input_batch.py:262`**准备函数**: `prepare_pos_seq_lens()` 内核计算**计算逻辑**: `positions[i] = num_computed_tokens[req_idx] + offset_in_req`                |
| `intermediate_tensors` | `IntermediateTensors \| None` | `(num_tokens, hc_mult, H)` | PP（流水线并行）时传递的中间隐藏状态**来源**: 前一个 PP stage 的输出**文件位置**: `deepseek_v4.py:1118`                                                                                                                                           |
| `inputs_embeds`        | `torch.Tensor \| None`        | `(num_tokens, H)`          | 可选的预计算嵌入，优先于 `input_ids`**场景**: 多模态模型的 encoder 输出或多模态嵌入**优先级**: `inputs_embeds > input_ids`                                                                                                                          |

**vLLM 输入准备流程详解** (`model_runner.py:756-922`):

| 步骤 | 函数/操作                                   | 输出                                              |
| -- | --------------------------------------- | ----------------------------------------------- |
| 1  | `scheduler_output.num_scheduled_tokens` | 获取每个请求的 token 数量 `{req_id: num_tokens}`         |
| 2  | `prepare_prefill_inputs()`              | 准备 prefill 阶段的 token IDs                        |
| 3  | `prepare_pos_seq_lens()`                | 计算每个 token 的 position（基于 `num_computed_tokens`） |
| 4  | `combine_sampled_and_draft_tokens()`    | 合并 decode 和 draft tokens                        |
| 5  | `InputBatch` 构建                         | 包含 `input_ids`, `positions`, `seq_lens` 等       |

**关键说明**:

- `input_ids` 的形状为 `(num_tokens,)` 而非 `(batch_size, seq_len)`，这是因为 vLLM 采用 **continuous batching**（连续批处理）机制，将多个请求的 token 平铺成一维序列
- `num_tokens = sum(seq_len for each request in batch)`，即所有请求的 token 总数
- `positions` 用于旋转位置编码（RoPE），表示每个 token 在其原始序列中的位置（从 0 开始）

**forward 流程**:

| 阶段 | 操作           | 组件/函数                                                             | 形状变化                                                    | 作用                   |
| -- | ------------ | ----------------------------------------------------------------- | ------------------------------------------------------- | -------------------- |
| 1  | 输入嵌入         | `embed_input_ids(input_ids)` / `inputs_embeds`                    | `(num_tokens,)` → `(num_tokens, H)`                     | 将 token IDs 转换为词嵌入向量 |
| 2  | Llama-4 缩放计算 | `_get_llama_4_scaling()` (当前为 None)                               | 标量                                                      | 计算注意力缩放因子            |
| 3  | HC 扩展        | `unsqueeze(1).repeat(1,      , 1)`                                | `(num_tokens, H)` → `(num_tokens, hc_mult, H)`          | 沿 hc\_mult 维度扩展      |
| 4  | 层前向传播        | `layers[]` (DeepseekV2DecoderLayer)                               | `(num_tokens, hc_mult, H)` → `(num_tokens, hc_mult, H)` | 遍历所有解码器层             |
| 5  | MTP 状态保存     | `_mtp_hidden_buffer[:num_tokens].copy_()`                         | `(num_tokens, hc_mult, H)` → `(num_tokens, hc_mult*H)`  | 展平保存用于多 Token 预测     |
| 6  | HC 头处理       | `hc_head(hidden_states, hc_head_fn, hc_head_scale, hc_head_base)` | `(num_tokens, hc_mult, H)` → `(num_tokens, H)`          | 门控加权融合，压缩回原始维度       |
| 7  | 最终归一化        | `norm(hidden_states)`                                             | `(num_tokens, H)` → `(num_tokens, H)`                   | 输出归一化                |
| 8  | 返回输出         | `IntermediateTensors` / `hidden_states`                           | `(num_tokens, H)`                                       | 返回隐藏状态或中间张量          |

**形状符号说明**:

- `num_tokens`: 所有请求的 token 总数（`sum(seq_len for each request)`），不是 batch\_size × seq\_len
- `H`: hidden\_size（隐藏层维度）
- `hc_mult`: HC扩展倍数（通常为 2）
- `hc_mult*H`: 表示 hc\_mult \* H 的乘积

**层前向传播内部流程** (第4阶段):

```
┌─────────────────────────────────────────────────────────────┐
│  注意力子流程: hc_pre → input_layernorm → self_attn → hc_post  │
│  FFN子流程: hc_pre → post_attention_layernorm → mlp → hc_post  │
└─────────────────────────────────────────────────────────────┘
```

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L1007`

#### **HC 头处理详解** (第6阶段)

**输入输出形状**:

- **输入**: `(num_tokens, hc_mult, H)` - 经过 HC 扩展和层前向传播后的隐藏状态
- **输出**: `(num_tokens, H)` - 压缩回原始隐藏层维度

```python
def hc_head(self, x, hc_fn, hc_scale, hc_base):
    shape, dtype = x.size(), x.dtype           # shape = (num_tokens, hc_mult, H)
    x = x.flatten(1).float()                    # (num_tokens, hc_mult*H)
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = torch.nn.functional.linear(x, hc_fn) * rsqrt       # 通过线性层投影并RMS归一化 (num_tokens, hc_mult*H)@(hc_mult, hc_mult*H)T=(num_tokens, hc_mult)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps  # 门控权重 (num_tokens, hc_mult)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)    # 按门控权重加权融合 hc_mult 个通道  (num_tokens, hc_mult, 1) * (num_tokens, hc_mult, H) = (num_tokens, hc_mult, H) -> (num_tokens, H)
    return y.to(dtype)
```

**参数说明**:
- `hc_fn`: 线性投影权重矩阵，形状为 `(hc_mult, hc_mult*H)`
- `hc_scale`: 门控缩放因子（标量）
- `hc_base`: 门控偏置，形状为 `(hc_mult,)`
- `hc_eps`: 数值稳定性 epsilon（防止除零）

**核心机制**: 通过学习 `hc_mult` 个门控权重，动态混合 HC 扩展后的多个特征通道，将高维特征压缩回原始维度，增强模型表达能力。

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L1094`

***

### 2.2 DeepseekV2DecoderLayer

**初始化组件**:

- `self_attn`: 自注意力层 (DeepseekV4Attention)
- `mlp`: MoE/MLP层 (DeepseekV4MoE / DeepseekV2MLP)
- `input_layernorm`: 输入层归一化 (RMSNorm)
- `post_attention_layernorm`: 注意力后归一化 (RMSNorm)
- `hc_mult/hc_sinkhorn_iters/hc_eps`: HC参数
- `hc_attn_fn/base/scale`: 注意力HC参数
- `hc_ffn_fn/base/scale`: FFN HC参数

**forward 流程**:

```
┌─────────────────────────────────────────────────────────────┐
│  注意力前处理: hc_pre → input_layernorm                     │
│  注意力计算: self_attn(positions, hidden_states, scaling)   │
│  注意力后处理: hc_post                                      │
├─────────────────────────────────────────────────────────────┤
│  FFN前处理: hc_pre → post_attention_layernorm               │
│  FFN计算: mlp(hidden_states)                                │
│  FFN后处理: hc_post                                         │
└─────────────────────────────────────────────────────────────┘
```

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L913`

***

### 2.3 DeepseekV4Attention

**初始化组件**:

| 组件                      | 类型                                    | 说明                         |
| ----------------------- | ------------------------------------- | -------------------------- |
| `wq_a`                  | ReplicatedLinear                      | Query低秩投影A                 |
| `q_norm`                | RMSNorm                               | Query归一化                   |
| `q_norm_without_weight` | RMSNorm                               | Query无权重归一化                |
| `wq_b`                  | ColumnParallelLinear/ReplicatedLinear | Query低秩投影B                 |
| `wkv`                   | ReplicatedLinear                      | KV投影                       |
| `kv_norm`               | RMSNorm                               | KV归一化                      |
| `wo_a`                  | ColumnParallelLinear                  | 输出投影A                      |
| `wo_b`                  | RowParallelLinear                     | 输出投影B                      |
| `rotary_emb`            | ComplexExpRotaryEmbedding             | 旋转位置编码                     |
| `attn_sink`             | nn.Parameter                          | 注意力sink参数                  |
| `compressor`            | Compressor \| None                    | 压缩器 (compress\_ratio > 1)  |
| `indexer`               | Indexer \| None                       | 索引器 (compress\_ratio == 4) |
| `dsa_attn`              | AscendDeepseekSparseAttention         | NPU稀疏注意力                   |

**forward 流程**:

```
hidden_states → dsa_attn(positions, hidden_states, llama_4_scaling)
    ├── Query投影: wq_a → q_norm → wq_b
    ├── KV投影: wkv → kv_norm
    ├── 旋转位置编码: rotary_emb
    ├── 压缩器处理 (如果启用)
    ├── 索引器处理 (如果启用)
    ├── 稀疏注意力计算: AscendDSABackend
    └── 输出投影: wo_a → wo_b
```

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L711`

***

### 2.4 AscendDSABackend (NPU稀疏注意力后端)

**forward 流程详解**:

| 阶段             | 处理内容                                                                                                                        |
| -------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **1. 输入预处理**   | Query路径: `hidden_states → wq_a → q_norm → wq_b`KV路径: `hidden_states → wkv → kv_norm`旋转位置编码: `rotary_emb.apply_rotary_emb()` |
| **2. 压缩器处理**   | `compressor.forward()` → KV压缩 + 压缩KV缓存更新                                                                                    |
| **3. 索引器处理**   | `indexer.forward()` → Query投影 + TopK稀疏索引                                                                                    |
| **4. 稀疏注意力计算** | `torch.ops._C_ascend.npu_sparse_flash_attention()`                                                                          |
| **5. 输出投影**    | `wo_a` (组间投影) → `wo_b` (输出投影)                                                                                               |
| **6. KV缓存管理**  | 压缩KV缓存 / 索引器KV缓存 / SWA缓存更新                                                                                                  |

***

### 2.5 Compressor (压缩器)

**初始化组件**:

- `ape`: 压缩参数 (nn.Parameter)
- `wkv`: KV投影 (ReplicatedLinear)
- `compress_ratio`: 压缩比例 (2/4)
- `overlap`: 重叠压缩 (bool)
- `rotate`: 是否旋转 (bool)
- `k_cache`: 压缩KV缓存 (DeepseekV4IndexerCache)

**forward 流程**:

```
KV投影 → 压缩变换(ape参数) → 旋转操作(可选) → KV缓存更新 → 返回压缩后的KV
```

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L598`

***

### 2.6 Indexer (索引器)

**初始化组件**:

- `wq_b`: Query低秩投影B (ReplicatedLinear)
- `weights_proj`: 权重投影 (ReplicatedLinear)
- `n_heads`: 索引头数
- `head_dim`: 头维度
- `index_topk`: TopK索引数
- `q_lora_rank`: Query低秩维度
- `k_cache`: 索引KV缓存 (DeepseekV4IndexerCache)

**forward 流程**:

```
Query投影 → 旋转位置编码 → 索引选择(npu_lightning_indexer) → TopK稀疏索引 → KV缓存更新 → 返回稀疏注意力索引
```

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L531`

***

### 2.7 DeepseekV2MLP

**初始化组件**:

- `gate_up_proj`: 门控和上投影合并 (MergedColumnParallelLinear)
- `act_fn`: SiLU激活函数 (SiluAndMul)
- `down_proj`: 下投影 (RowParallelLinear)

**forward 流程**:

```python
gate_up = gate_up_proj(x)
x = act_fn(gate_up)  # SiLU激活
x = down_proj(x)
return x
```

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L308`

***

### 2.8 DeepseekV4MoE

**初始化组件**:

| 类别       | 组件                         | 说明                                                                                                                                            |
| -------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **路由门控** | `gate`                     | 路由门控 (ReplicatedLinear)├── `tid2eid`: Token到专家映射└── `e_score_correction_bias`: 专家得分修正偏置                                                       |
| **专家网络** | `experts`                  | 融合专家网络 (FusedMoE)├── `runner`: NPU MoE运行器├── `quant_method`: NPU量化方法├── `global_expert_map/_expert_map/log2phy`: EPLB配置└── `moe_config`: 并行配置 |
| **共享专家** | `shared_experts`           | 共享专家 (DeepseekV2MLP \| None)                                                                                                                  |
| **配置参数** | `n_routed_experts`         | 路由专家数                                                                                                                                         |
| <br />   | `n_shared_experts`         | 共享专家数                                                                                                                                         |
| <br />   | `n_redundant_experts`      | 冗余专家数 (EPLB)                                                                                                                                  |
| <br />   | `n_logical_experts`        | 逻辑专家数                                                                                                                                         |
| <br />   | `n_physical_experts`       | 物理专家数                                                                                                                                         |
| <br />   | `n_local_physical_experts` | 本地物理专家数                                                                                                                                       |
| <br />   | `enable_eplb`              | 启用EPLB                                                                                                                                        |
| <br />   | `is_sequence_parallel`     | 序列并行                                                                                                                                          |
| <br />   | `routed_scaling_factor`    | 路由缩放因子                                                                                                                                        |

**forward 流程**:

```
序列并行分块 → 路由计算(hash+线性) → 专家计算 → 共享专家计算(可选)
    → 结果融合 → 序列并行聚合 → TP聚合
```

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L354`

***

### 2.9 DeepSeekV4MTP (多Token预测器)

**初始化组件**:

- `model`: MTP模型 (DeepSeekMultiTokenPredictor)
  - `layers[]`: MTP层堆叠 (DeepSeekMultiTokenPredictorLayer × N)
    - `e_proj`: E投影 (ReplicatedLinear)
    - `h_proj`: H投影 (ReplicatedLinear)
    - `enorm`: E归一化 (RMSNorm)
    - `hnorm`: H归一化 (RMSNorm)
    - `shared_head`: 共享头 (SharedHead)
    - `mtp_block`: MTP块 (DeepseekV2DecoderLayer)
    - `hc_head_fn/base/scale`: HC头参数
    - `topk_indices_buffer`: TopK索引缓冲区
  - `embed_tokens`: 词嵌入层 (VocabParallelEmbedding)
- `logits_processor`: Logits处理器

**forward 流程**:

```
输入嵌入 → MTP层选择 → MTP层前向传播 → 返回 hidden_states
```

**compute\_logits 流程**:

```
MTP层选择 → HC头处理 → 共享头处理 → Logits计算
```

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4_mtp.py#L201`

***

### 2.10 AscendFusedMoE (NPU MoE融合)

**初始化组件**:

- 继承父类 `FusedMoE` 的所有属性
- `quant_method`: NPU量化方法 (AscendUnquantizedFusedMoEMethod)
- `runner`: NPU Runner (AscendMoERunner)
- `global_expert_map/_expert_map/log2phy`: EPLB配置
- `moe_config`: 并行配置 (tp\_group, dp\_group, ep\_group, mc2\_group)

**forward\_impl 流程**:

```
[可选] shared_expert计算 → select_experts() → moe_comm_method.prepare()
    → quant_method.apply() → 返回 FusedMoEResult
```

**文件位置**: `vllm_ascend/ops/fused_moe/fused_moe.py#L335`

***

## 三、关键特性说明

### 3.1 混合协同处理 (HC - Hybrid Collaboration)

```
hc_pre: torch.ops._C_ascend.npu_hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
hc_post: torch.ops._C_ascend.npu_hc_post(x, residual, post, comb)
hc_head:
    归一化: rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    混合: linear(x, hc_fn) * rsqrt
    激活: sigmoid(mixes * hc_scale + hc_base) + hc_eps
    聚合: sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
```

***

### 3.2 稀疏注意力 (DSA - DeepSeek Sparse Attention)

- **压缩器**: KV压缩 (compress\_ratio: 2/4)
- **索引器**: TopK稀疏索引 (compress\_ratio == 4)
- **注意力Sink**: attn\_sink参数
- **NPU算子**: `npu_sparse_flash_attention`

***

### 3.3 混合专家 (MoE - Mixture of Experts)

- **路由机制**: gate线性层 + hash路由 (tid2eid)
- **EPLB**: 专家并行负载均衡
- **共享专家**: shared\_experts (可选)
- **NPU融合算子**: `moe_gating_top_k_hash`

***

### 3.4 多Token预测 (MTP - Multi-Token Prediction)

- **MTP层**: DeepSeekMultiTokenPredictorLayer
- **MTP块**: DeepseekV2DecoderLayer
- **共享头**: SharedHead
- **HC头**: hc\_head处理

***

### 3.5 NPU优化

| 优化类型  | NPU算子                                            |
| ----- | ------------------------------------------------ |
| 稀疏注意力 | `torch.ops._C_ascend.npu_sparse_flash_attention` |
| MoE路由 | `torch.ops._C_ascend.moe_gating_top_k_hash`      |
| HC处理  | `torch.ops._C_ascend.npu_hc_pre/npu_hc_post`     |
| 索引器   | `npu_lightning_indexer`                          |

***

### 3.6 并行策略

```
├── TP (Tensor Parallel): 张量并行
├── PP (Pipeline Parallel): 流水线并行
├── EP (Expert Parallel): 专家并行
├── DP (Data Parallel): 数据并行
├── MC2 (MoE Communication): MoE通信
└── SP (Sequence Parallel): 序列并行
```

***

## 四、文件位置参考

| 文件/模块  | 路径                                                        |
| ------ | --------------------------------------------------------- |
| 主模型文件  | `vllm-ascend/vllm_ascend/models/deepseek_v4.py`           |
| MTP文件  | `vllm-ascend/vllm_ascend/models/deepseek_v4_mtp.py`       |
| 注意力层   | `vllm-ascend/vllm_ascend/models/layer/attention/layer.py` |
| MoE融合  | `vllm_ascend/ops/fused_moe/fused_moe.py`                  |
| DSA注意力 | `vllm_ascend/ops/dsa/`                                    |
| 旋转编码   | `vllm_ascend/ops/rope_dsv4.py`                            |
| NPU算子  | `torch.ops._C_ascend.*`                                   |

***

## 五、架构关系图

```
                    ┌─────────────────────────────────────────┐
                    │    AscendDeepseekV4ForCausalLM         │
                    └─────────────────────┬─────────────────┘
                                          │
                    ┌─────────────────────▼─────────────────┐
                    │          DeepseekV4Model               │
                    └─────────────────────┬─────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │                                 │                                 │
┌───────▼───────┐               ┌────────▼────────┐               ┌─────────▼─────────┐
│ embed_tokens  │               │    layers[]     │               │      norm         │
│(词嵌入层)     │               │(解码器层堆叠)   │               │   (RMSNorm)       │
└───────────────┘               └────────┬────────┘               └───────────────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │       DeepseekV2DecoderLayer          │
                    └───────────────────┬───────────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           │                            │                            │
    ┌──────▼──────┐             ┌───────▼───────┐             ┌──────▼──────┐
    │self_attn    │             │input_layernorm│             │   mlp       │
    │(DSA注意力)  │             │  post_attn_   │             │(MoE/MLP)    │
    └──────┬──────┘             │  layernorm    │             └─────────────┘
           │                    └───────────────┘
           │
    ┌──────▼───────────────────────────────────────────────┐
    │              DeepseekV4Attention                     │
    └──────┬──────┬───────────────┬───────────────────────┘
           │      │               │
    ┌──────▼──────┼──────┐  ┌─────▼─────────┐  ┌───────────▼───────────┐
    │  compressor │indexer│  │dsa_attn      │  │rotary_emb|attn_sink  │
    │  (压缩器)   │(索引器)│  │(NPU稀疏注意力)│  │(位置编码)            │
    └─────────────┴──────┘  └───────────────┘  └──────────────────────┘
```

