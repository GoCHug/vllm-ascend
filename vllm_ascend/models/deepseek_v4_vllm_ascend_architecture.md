# DeepSeekV4 架构文档

***

## 一、完整架构概览

```
AscendDeepseekV4ForCausalLM (模型入口类, L1203)
├── model: DeepseekV4Model
│   ├── embed_tokens (VocabParallelEmbedding) [PP首阶段]
│   ├── layers[]: DeepseekV2DecoderLayer × N (L904)
│   │   ├── input_layernorm (RMSNorm)
│   │   ├── self_attn: DeepseekV4Attention (L702)
│   │   │   └── dsa_attn: AscendDeepseekSparseAttention
│   │   │       └── dsa_modules: DSAModules (封装所有注意力子模块)
│   │   │           ├── wq_a, q_norm, q_norm_without_weight, wq_b
│   │   │           ├── wkv, kv_norm, wo_a, wo_b, attn_sink
│   │   │           ├── compressor: Compressor | None
│   │   │           ├── indexer: Indexer | None
│   │   │           ├── swa_cache_layer: AscendDeepseekV4SWACache
│   │   │           └── topk_indices_buffer, skip_topk
│   │   ├── post_attention_layernorm (RMSNorm)
│   │   ├── mlp: DeepseekV4MoE (L345) / DeepseekV2MLP (L299)
│   │   ├── hc_pre/hc_post: HC处理方法 (NPU算子封装)
│   │   └── hc_attn/hc_ffn: {fn, base, scale} HC参数组
│   ├── norm (RMSNorm) [PP末阶段]
│   ├── hc_head {fn, base, scale}: HC头参数
│   ├── topk_indices_buffer: TopK索引缓冲区 (模型级共享)
│   └── _mtp_hidden_buffer: MTP隐藏状态缓冲区 (预hc_head残差流)
├── lm_head (ParallelLMHead) [PP末阶段]
└── logits_processor (LogitsProcessor)
```

**缓存组件层次**:
```
├── AscendCompressorStateCache: 压缩器KV缓存 (compress_ratio=4/128)
├── AscendDeepseekV4IndexerCache: 索引器KV缓存 (compress_ratio=4)
└── AscendDeepseekV4SWACache: 滑动窗口注意力缓存 (所有层)
```

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L1203`

***

## 二、核心组件详解

### 2.1 DeepseekV4Model

**初始化组件**:

- `embed_tokens`: 词嵌入层 (VocabParallelEmbedding)，仅PP首阶段实例化
- `layers[]`: 解码器层堆叠 (DeepseekV2DecoderLayer × N)，支持PP分层
- `norm`: 最终归一化 (RMSNorm)，仅PP末阶段实例化
- `hc_mult`: HC扩展倍数
- `hc_eps`: HC epsilon
- `norm_eps`: RMSNorm epsilon
- `hc_head_fn/base/scale`: HC头参数
  - `hc_head_fn`: `(hc_mult, hc_mult*hidden_size)`
  - `hc_head_base`: `(hc_mult,)`
  - `hc_head_scale`: `(1,)`
- `topk_indices_buffer`: TopK索引缓冲区 (模型级共享，V3.2+配置)
- `_mtp_hidden_buffer`: MTP隐藏状态缓冲区，形状 `(max_num_batched_tokens, hc_mult*hidden_size)`
- `make_empty_intermediate_tensors`: PP中间张量工厂，创建 `(batch_size, hc_mult, hidden_size)` 张量

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

**forward 流程** (L1094-1160):

| 阶段 | 操作           | 组件/函数                                                             | 形状变化                                                    | 作用                   |
| -- | ------------ | ----------------------------------------------------------------- | ------------------------------------------------------- | -------------------- |
| 1  | PP阶段判断       | `get_pp_group().is_first_rank/last_rank`                          | -                                                       | 流水线并行阶段路由            |
| 2  | 输入嵌入 (PP首阶段) | `embed_input_ids(input_ids)` / `inputs_embeds`                    | `(num_tokens,)` → `(num_tokens, H)`                     | 将 token IDs 转换为词嵌入向量 |
| 3  | 中间张量加载 (非首阶段) | `intermediate_tensors["hidden_states"]`                           | `(num_tokens, hc_mult, H)`                              | 加载前一PP阶段输出           |
| 4  | Llama-4 缩放   | `_get_llama_4_scaling()` (当前配置为 None, 不启用)                    | 标量或 None                                                | 计算注意力缩放因子(当前未启用)     |
| 5  | HC 扩展 (PP首阶段) | `unsqueeze(1).repeat(1, hc_mult, 1)`                              | `(num_tokens, H)` → `(num_tokens, hc_mult, H)`          | 沿 hc_mult 维度扩展      |
| 6  | 层前向传播        | `layers[start_layer:end_layer]` (DeepseekV2DecoderLayer)          | `(num_tokens, hc_mult, H)` → `(num_tokens, hc_mult, H)` | 遍历当前PP阶段的解码器层        |
| 7  | MTP 状态保存     | `_mtp_hidden_buffer[:num_tokens].copy_(flatten)`                  | `(num_tokens, hc_mult, H)` → `(num_tokens, hc_mult*H)`  | 展平保存预-hc_head状态用于MTP |
| 8  | PP中间输出 (非末阶段) | `IntermediateTensors` 返回                                         | -                                                       | 传递给下一PP阶段            |
| 9  | HC 头处理 (PP末阶段) | `hc_head(hidden_states, hc_head_fn, hc_head_scale, hc_head_base)` | `(num_tokens, hc_mult, H)` → `(num_tokens, H)`          | 门控加权融合，压缩回原始维度       |
| 10 | 最终归一化 (PP末阶段) | `norm(hidden_states)`                                             | `(num_tokens, H)` → `(num_tokens, H)`                   | 输出归一化                |
| 11 | 返回输出         | `hidden_states`                                                   | `(num_tokens, H)`                                       | 返回最终隐藏状态             |

**FlashComm1 序列并行特殊处理** (L1136-1148):
- 启用FlashComm时，每层输出经过reduce_scatter分片到TP rank
- MTP缓冲区保存前需要执行`tensor_model_parallel_all_gather`聚合完整token集
- 处理填充token裁剪，避免MTP接收到无效数据

**形状符号说明**:

- `num_tokens`: 所有请求的 token 总数（`sum(seq_len for each request)`），不是 batch_size × seq_len
- `H`: hidden_size（隐藏层维度）
- `hc_mult`: HC扩展倍数（通常为 2）
- `hc_mult*H`: 表示 hc_mult * H 的乘积

**层前向传播内部流程** (第6阶段):

```
┌─────────────────────────────────────────────────────────────────────┐
│  注意力子流程: residual.clone() → hc_pre → input_layernorm → self_attn → hc_post  │
│  FFN子流程: residual.clone() → hc_pre → post_attention_layernorm → mlp → hc_post  │
│  返回: (hidden_states, residual)                                                              │
└─────────────────────────────────────────────────────────────────────┘
```

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L997`

#### **HC 头处理详解** (第9阶段)

**输入输出形状**:

- **输入**: `(num_tokens, hc_mult, H)` - 经过 HC 扩展和层前向传播后的隐藏状态
- **输出**: `(num_tokens, H)` - 压缩回原始隐藏层维度

```python
def hc_head(self, x, hc_fn, hc_scale, hc_base):
    shape, dtype = x.size(), x.dtype           # shape = (num_tokens, hc_mult, H)
    x = x.flatten(1).float()                    # (num_tokens, hc_mult*H)
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = torch.nn.functional.linear(x, hc_fn) * rsqrt       # 线性投影 + RMS归一化 (num_tokens, hc_mult)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps  # 门控权重 (num_tokens, hc_mult)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)    # 加权融合 hc_mult 个通道
    return y.to(dtype)
```

**流程示意图** (hc_mult=4为例):

```
输入: (num_tokens, 4, H)
   │
   ├─────────────────────────────────────────────────┐
   │ 展开 flatten(1) + float()                        │
   ▼                                                 │
x: (num_tokens, 4*H)                                 │
   │                                                 │
   ├──────────┐                                      │
   │ x² → mean → sqrt → rsqrt                        │
   ▼          │                                      │
rsqrt: (num_tokens, 1)                               │
   │          │                                      │
   ├──┐       │                                      │
   │  ▼       ▼                                      │
   │  x @ hc_fn  *  rsqrt   →  mixes                │
   │  │        线性投影+RMSNorm                       │
   │  ▼                                              │
   │ mixes: (num_tokens, 4)  →  sigmoid(mixes*scale+base) + eps
   │  │                                              │
   │  ▼                                              │
   │ pre: (num_tokens, 4)  →  门控权重 (0~1之间)      │
   │  │                                              │
   │  │ pre.unsqueeze(-1): (num_tokens, 4, 1)         │
   └──┼──────────────────────────────────────────────┘
      │
      ▼  x.view(shape)还原为 (num_tokens, 4, H)
pre * x: (num_tokens, 4, H)   # 每个通道乘独立门控权重
      │
      ▼ sum(dim=1)
输出: (num_tokens, H)
```

**核心步骤**:
1. **flatten**: 将hc_mult个通道展平为单一大维度
2. **RMSNorm归一化**: 计算均方根归一化因子rsqrt，保证数值稳定
3. **线性投影**: 将4*H维度映射回4个混合系数
4. **Sigmoid门控**: 通过scale和base可学习参数控制门控温度和偏置
5. **加权融合**: 4个通道分别乘对应权重后求和，压缩回H维度

**参数说明**:
- `hc_fn`: 线性投影权重矩阵，形状为 `(hc_mult, hc_mult*H)`
- `hc_scale`: 门控缩放因子，形状 `(1,)`
- `hc_base`: 门控偏置，形状为 `(hc_mult,)`
- `hc_eps`: 数值稳定性 epsilon（防止除零）

**核心机制**: 通过学习 `hc_mult` 个门控权重，动态混合 HC 扩展后的多个特征通道，将高维特征压缩回原始维度，增强模型表达能力。

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L1085`

***

### 2.2 DeepseekV2DecoderLayer

**初始化组件** (L904-961):

- `self_attn`: 自注意力层 (DeepseekV4Attention)
- `mlp`: MoE/MLP层 (DeepseekV4MoE / DeepseekV2MLP)
- `input_layernorm`: 输入层归一化 (RMSNorm)
- `post_attention_layernorm`: 注意力后归一化 (RMSNorm)
- `hc_mult/hc_sinkhorn_iters/hc_eps`: HC参数
- `norm_eps`: RMSNorm epsilon
- `hc_attn_fn/base/scale`: 注意力HC参数组
  - `hc_attn_fn`: `(mix_hc, hc_dim)` where `mix_hc = (2 + hc_mult) * hc_mult`, `hc_dim = hc_mult * hidden_size`
  - `hc_attn_base`: `(mix_hc,)`
  - `hc_attn_scale`: `(3,)`
- `hc_ffn_fn/base/scale`: FFN HC参数组
  - 形状同hc_attn参数组

**HC处理方法**:
- `hc_pre(x, hc_fn, hc_scale, hc_base)`: 调用NPU算子 `torch.ops._C_ascend.npu_hc_pre`
- `hc_post(x, residual, post, comb)`: 调用NPU算子 `torch.ops._C_ascend.npu_hc_post`，内部处理unsqueeze/squeeze维度

**forward 流程** (L975-994):

```
输入: (positions, hidden_states, residual, llama_4_scaling)
输出: (hidden_states, residual)

┌─────────────────────────────────────────────────────────────┐
│  注意力分支:
│  1. residual = hidden_states.clone()                         │
│  2. hidden_states, post, comb = hc_pre(hidden_states, hc_attn_*)
│  3. hidden_states = input_layernorm(hidden_states)          │
│  4. hidden_states = self_attn(positions, hidden_states, llama_4_scaling)
│  5. hidden_states = hc_post(hidden_states, residual, post, comb)
├─────────────────────────────────────────────────────────────┤
│  FFN分支:
│  6. residual = hidden_states.clone()                         │
│  7. hidden_states, post, comb = hc_pre(hidden_states, hc_ffn_*)
│  8. hidden_states = post_attention_layernorm(hidden_states) │
│  9. hidden_states = mlp(hidden_states)                      │
│ 10. hidden_states = hc_post(hidden_states, residual, post, comb)
└─────────────────────────────────────────────────────────────┘
```

**注意**: forward 返回元组 `(hidden_states, residual)` 而非仅 hidden_states。

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L904`

***

### 2.3 DeepseekV4Attention

**初始化组件** (L702-893):

| 组件                      | 类型                                    | 说明                         |
| ----------------------- | ------------------------------------- | -------------------------- |
| `wq_a`                  | ReplicatedLinear                      | Query低秩投影A: `(dim, q_lora_rank)`                 |
| `q_norm`                | RMSNorm                               | Query归一化: `(q_lora_rank,)`                   |
| `q_norm_without_weight` | RMSNorm                               | Query无权重归一化: `(head_dim,)`                |
| `wq_b`                  | ColumnParallelLinear/ReplicatedLinear | Query低秩投影B: `(q_lora_rank, n_heads*head_dim)`，enable_dsa_cp时用ReplicatedLinear                 |
| `wkv`                   | ReplicatedLinear                      | KV投影: `(dim, head_dim)`                       |
| `kv_norm`               | RMSNorm                               | KV归一化: `(head_dim,)`                      |
| `wo_a`                  | ColumnParallelLinear                  | 输出投影A: `(n_heads*head_dim//n_groups, n_groups*o_lora_rank)`                      |
| `wo_b`                  | RowParallelLinear                     | 输出投影B: `(n_groups*o_lora_rank, dim)`                     |
| `rotary_emb`            | ComplexExpRotaryEmbedding             | 旋转位置编码，根据compress_ratio选择rope_theta                     |
| `attn_sink`             | nn.Parameter                          | 注意力sink参数，enable_dsa_cp时形状为`(n_heads,)`否则为`(n_local_heads,)`                  |
| `compressor`            | Compressor \| None                    | 压缩器 (compress_ratio > 1，支持4/128)  |
| `indexer`               | Indexer \| None                       | 索引器 (compress_ratio == 4) |
| `swa_cache_layer`       | AscendDeepseekV4SWACache              | 滑动窗口注意力缓存，始终创建 |
| `dsa_modules`           | DSAModules                            | 封装所有子模块供NPU后端使用 |
| `dsa_attn`              | AscendDeepseekSparseAttention         | NPU稀疏注意力入口，统一封装所有DSA逻辑                   |

**关键配置逻辑**:
- `compress_ratio = get_dsv4_compress_ratio(config, layer_idx)`: 按层获取压缩比例
- `skip_topk`: 索引缓存复用逻辑 (V3.2+配置)，基于`use_index_cache`、`index_topk_freq`、`index_topk_pattern`配置
- A5设备上IndexerCache使用float8_e4m3fn，SWA Cache head_size额外+128字节
- DSV4_BLOCK_SIZES按设备类型(A5/非A5)和block_size(128/64/32)配置块大小

**forward 流程** (L895-901):

```
hidden_states → dsa_attn(positions, hidden_states, llama_4_scaling)
    └── 内部由AscendDSABackend实现完整注意力逻辑
```

DeepseekV4Attention.forward本身只是透传给dsa_attn，所有投影、位置编码、缓存、稀疏注意力计算都在AscendDSABackend中完成。

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L702`

***

### 2.4 AscendDSABackend (NPU稀疏注意力后端)

实现在 `vllm_ascend/attention/dsa_v1.py`，通过 `dsa_attn` 统一入口调用。

**核心处理流程**:

| 阶段             | 处理内容                                                                                                                        |
| -------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **1. 输入预处理**   | Query路径: `hidden_states → wq_a → q_norm → wq_b`KV路径: `hidden_states → wkv → kv_norm`旋转位置编码: `rotary_emb` |
| **2. 压缩器处理**   | `compressor` → KV压缩 + 压缩KV缓存更新 (compress_ratio=4/128)                                                                                    |
| **3. 索引器处理**   | `indexer` → Query投影 + TopK稀疏索引选择 (compress_ratio=4)，支持skip_topk复用历史索引                                                                                    |
| **4. 稀疏注意力计算** | NPU算子 `torch.ops._C_ascend.npu_sparse_flash_attention()`                                                                          |
| **5. 滑动窗口缓存** | `swa_cache_layer` SWA缓存更新                                                                                               |
| **6. 输出投影**    | `wo_a` (组间投影) → `wo_b` (输出投影)                                                                                               |
| **7. KV缓存管理**  | 压缩KV缓存 / 索引器KV缓存 / SWA缓存 三类缓存独立管理                                                                                                  |

***

### 2.5 KV缓存类

#### AscendCompressorStateCache (L104-134)
- **state_dim**: `2 * coff * head_dim` (c4) 或 `2 * head_dim` (c128)，coff=2当overlap=True
- **compress_ratio**: 4 或 128
- **block_size**: 按DSV4_BLOCK_SIZES配置 (c4:8/4/2, c128:32/16/8 对应block_size=128/64/32)
- **dtype**: float32
- 返回 `AscendSlidingWindowMLASpec` 作为KV缓存规格

#### AscendDeepseekV4IndexerCache (L137-170)
- **compress_ratio**: 4
- **dtype**: A5设备用float8_e4m3fn，其他用int8
- **block_size**: DSV4_BLOCK_SIZES中mla块大小 (128/64/32)
- **scale_dim**: 1当head_dim=128，否则0
- 返回 `AscendMLAAttentionSpec`

#### AscendDeepseekV4SWACache (L173-207)
- **window_size**: 滑动窗口大小来自config.sliding_window
- **dtype**: A5设备用float8_e4m3fn，其他用bfloat16
- **block_size**: DSV4_BLOCK_SIZES中swa块大小 (128/64/32)
- **cached_head_size**: A5设备额外+128
- 返回 `AscendSlidingWindowMLASpec`

**DSV4_BLOCK_SIZES配置** (`vllm_ascend/models/layer/attention/layer.py:32-50`):
```
非A5设备:
  block_size=128: [[mla=128, swa=128, c4_state=8, c128_state=32], [pad1=16640, pad2=131072]]
  block_size=64:  [[mla=64, swa=64, c4_state=4, c128_state=16], [pad1=8320, pad2=65536]]
  block_size=32:  [[mla=32, swa=32, c4_state=2, c128_state=8], [pad1=4160, pad2=32768]]
A5设备:
  block_size=128: [[128, 128, 8, 16], [16896, 81920]]
  block_size=64:  [[64, 64, 4, 8], [8448, 40960]]
  block_size=32:  [[32, 32, 2, 4], [4224, 20480]]
```

***

### 2.6 Compressor (压缩器)

**初始化组件** (L589-657):

| 组件 | 类型 | 形状/说明 |
| ---- | ---- | ------- |
| `ape` | nn.Parameter | `(compress_ratio, coff*head_dim)` 压缩变换参数 |
| `wkv` | ReplicatedLinear | `(dim, coff*head_dim)` KV投影 |
| `wgate` | ReplicatedLinear | `(dim, coff*head_dim)` 门控投影 (新增) |
| `norm` | RMSNorm | `(head_dim,)` 归一化 (A5用float32) |
| `compress_ratio` | int | 压缩比例，支持 **4 或 128** |
| `overlap` | bool | 重叠压缩，compress_ratio==4时为True |
| `coff` | int | 1 + overlap = 2 (c4) 或 1 (c128) |
| `rotate` | bool | 是否旋转 (Indexer内置compressor为True) |
| `state_cache` | AscendCompressorStateCache | 压缩状态缓存 |

**关键方法**:
- `overlap_transform(tensor, value)`: 重叠变换处理 (L659-665)
- `rope_single(x, cos, sin, inverse)`: 单张量RoPE旋转，调用NPU算子`npu_rotary_mul` (L676-699)
- `forward(x, start_pos, cos, sin)`: 具体实现在AscendDSABackend中

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L589`

***

### 2.7 Indexer (索引器)

**初始化组件** (L522-583):

| 组件 | 类型 | 说明 |
| ---- | ---- | ---- |
| `wq_b` | ReplicatedLinear | Query低秩投影B: `(q_lora_rank, n_heads*head_dim)` |
| `weights_proj` | ReplicatedLinear | 权重投影: `(hidden_size, n_heads)` |
| `n_heads` | int | 索引头数 (config.index_n_heads) |
| `head_dim` | int | 头维度 (config.index_head_dim) |
| `index_topk` | int | TopK索引数 (config.index_topk) |
| `q_lora_rank` | int | Query低秩维度 |
| `softmax_scale` | float | `head_dim**-0.5` |
| `compress_ratio` | int | 压缩比例 (4) |
| `k_cache` | AscendDeepseekV4IndexerCache | 索引KV缓存，仅当compress_ratio==4时创建 |
| `compressor` | Compressor | 内置压缩器，compress_ratio>1时创建，rotate=True |

**注意**: `forward` 方法当前为空占位，实际逻辑在AscendDSABackend中实现。

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L522`

***

### 2.8 DeepseekV2MLP

**初始化组件** (L299-336):

- `gate_up_proj`: 门控和上投影合并 (MergedColumnParallelLinear) `(hidden_size, [intermediate_size]*2)`
- `act_fn`: SiLU激活函数 (SiluAndMul)
- `down_proj`: 下投影 (RowParallelLinear) `(intermediate_size, hidden_size)`
- `is_sequence_parallel`: 序列并行时禁用TP，权重复制
- `reduce_results`: down_proj是否allreduce结果

**forward 流程** (L338-342):

```python
gate_up, _ = self.gate_up_proj(x)  # 忽略bias
x = self.act_fn(gate_up)            # SiLU激活 + 门控乘法
x, _ = self.down_proj(x)            # 下投影，忽略bias
return x
```

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L299`

***

### 2.9 DeepseekV4MoE

**初始化组件** (L345-451):

| 类别       | 组件                         | 说明                                                                                                                                            |
| -------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **路由门控** | `gate`                     | 路由门控 (ReplicatedLinear) `(hidden_size, n_routed_experts)`，`precast_fp32_weight=True`├── `tid2eid`: Token到专家映射，hash路由时创建，形状`(vocab_size, num_experts_per_tok)`└── `e_score_correction_bias`: 专家得分修正偏置，非hash路由时创建，形状`(n_routed_experts,)`                                                       |
| **专家网络** | `experts`                  | 融合专家网络 (FusedMoE)，参数包含hash、tid2eid、e_score_correction_bias、enable_eplb、n_shared_experts等                                                                 |
| **共享专家** | `shared_experts`           | 共享专家 (DeepseekV2MLP \| None)，`mix_placement=True`(AITER融合共享专家)时为None，由FusedMoE内部融合处理                                                                                                                  |
| **配置参数** | `hash`                     | bool: `layer_idx < num_hash_layers and not is_draft_layer`，是否使用hash路由                                                                                                  |
| <br />   | `n_routed_experts`         | 路由专家数                                                                                                                                         |
| <br />   | `n_shared_experts`         | 共享专家数                                                                                                                                         |
| <br />   | `n_redundant_experts`      | 冗余专家数 (EPLB)                                                                                                                                  |
| <br />   | `n_logical_experts`        | 逻辑专家数 = n_routed_experts                                                                                                                         |
| <br />   | `n_physical_experts`       | 物理专家数 = n_logical + n_redundant                                                                                                                  |
| <br />   | `n_local_physical_experts` | 本地物理专家数 = n_physical // ep_size                                                                                                              |
| <br />   | `enable_eplb`              | 启用EPLB (parallel_config.enable_eplb)                                                                                                                      |
| <br />   | `is_sequence_parallel`     | 序列并行 (use_sequence_parallel_moe)                                                                                                          |
| <br />   | `routed_scaling_factor`    | 路由缩放因子 (默认1.5)                                                                                                                                |
| <br />   | `ep_group/ep_rank/ep_size` | 专家并行组信息                                                                                                                                       |

**关键逻辑**:
- `is_fusion_moe_shared_experts_enabled = get_ascend_config().mix_placement`
- mix_placement启用时，n_shared_experts传入FusedMoE内部处理，不单独创建shared_experts

**forward 流程** (L453-503):

```
输入: hidden_states(num_tokens, hidden_dim), input_ids(可选)
输出: final_hidden_states

1. 形状调整: hidden_states = hidden_states.view(-1, hidden_dim)
2. [序列并行] hidden_states = sequence_parallel_chunk(hidden_states)
3. 路由判断:
   ├─ is_internal_router=True → experts(hidden_states, router_logits=hidden_states)
   └─ is_internal_router=False → router_logits=F.linear(hidden_states.float(), gate.weight) → experts(hidden_states, router_logits)
4. 输出处理:
   ├─ tuple输出(shared_output, final_hidden_states):
   │  ├─ dtype != float16 且非AITER:
   │  │  ├─ 有shared_experts: muls_add_triton(final, shared, routed_scaling_factor)
   │  │  └─ 无shared_experts: final *= routed_scaling_factor
   │  └─ dtype == float16 且有shared_experts:
   │     └─ muls_add_triton(shared, final, 1.0/routed_scaling_factor)
   └─ tensor输出: final_hidden_states = fused_moe_out
5. [序列并行] all_gather → 裁剪pad到num_tokens
6. [TP>1且tuple输出] maybe_all_reduce_tensor_model_parallel
7. 返回view回(num_tokens, hidden_dim)
```

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L345`

***

### 2.10 DeepseekV2MixtureOfExperts 抽象类

(L1163-1200) 提供MoE参数提取和专家负载均衡更新接口：
- `extract_moe_parameters(example_moe)`: 从示例MoE层提取超参数
- `update_physical_experts_metadata()`: EPLB专家重映射时更新物理专家数量
- `moe_mlp_layers` / `moe_layers`: 收集所有MoE层用于EPLB操作

***

### 2.11 DeepSeekV4MTP (多Token预测器)

**类层次**:
```
DeepSeekV4MTP (L201, 入口类)
├── model: DeepSeekMultiTokenPredictor (L140)
│   ├── layers: ModuleDict[str, DeepSeekMultiTokenPredictorLayer] (按step_idx轮询)
│   │   ├── e_proj: ReplicatedLinear (hidden_size, hidden_size)
│   │   ├── h_proj: ReplicatedLinear (hidden_size, hidden_size)
│   │   ├── enorm: RMSNorm (embedding归一化)
│   │   ├── hnorm: RMSNorm (hidden_states归一化)
│   │   ├── shared_head: SharedHead
│   │   │   ├── norm: RMSNorm
│   │   │   └── head: ParallelLMHead
│   │   ├── mtp_block: DeepseekV2DecoderLayer (is_draft_layer=True)
│   │   └── hc_head {fn, base, scale}: HC头参数 (仅compute_logits时使用)
│   ├── embed_tokens: VocabParallelEmbedding
│   └── logits_processor: LogitsProcessor
```

**SharedHead** (L36-53):
- forward只做`norm(hidden_states)`，不做投影
- LM Head投影在compute_logits中显式调用

**DeepSeekMultiTokenPredictorLayer forward 流程** (L106-128):
```
输入: input_ids, positions, previous_hidden_states, inputs_embeds, spec_step_index
输出: hidden_states (形状 (num_tokens, hc_mult, hidden_size)，未经过hc_head压缩)

1. Mask position==0的输入: inputs_embeds = where(positions==0, 0, inputs_embeds)
2. 嵌入归一化: inputs_embeds = enorm(inputs_embeds)
3. 前序隐藏状态reshape+归一化: previous_hidden_states = hnorm(previous_hidden_states.view(-1, hc_mult, H))
4. 投影融合: hidden_states = e_proj(inputs_embeds).unsqueeze(-2) + h_proj(previous_hidden_states)
   形状: (num_tokens, 1, H) + (num_tokens, hc_mult, H) → (num_tokens, hc_mult, H) 广播
5. MTP解码器块: hidden_states, residual = mtp_block(positions, hidden_states, residual=None)
6. 返回hidden_states (不执行hc_head，保持hc_mult维度)
```

**注意**: Layer forward中hc_head调用被注释掉，保持hc_mult维度，在compute_logits阶段才做hc_head压缩。

**DeepSeekMultiTokenPredictor forward** (L166-183):
- `current_step_idx = spec_step_idx % num_mtp_layers` 轮询选择MTP层
- 若inputs_embeds为None则用embed_tokens嵌入
- 调用对应layer的forward

**compute_logits 流程** (L185-197):
```
输入: hidden_states, spec_step_idx
输出: logits

1. current_step_idx = spec_step_idx % num_mtp_layers
2. mtp_layer = layers[str(current_step_idx)]
3. reshape: hidden_states = hidden_states.view(-1, hc_mult, hidden_size)
4. HC头压缩: hidden_states = mtp_layer.hc_head(hidden_states, hc_head_fn, hc_head_scale, hc_head_base)
5. SharedHead归一化: hidden_states = mtp_layer.shared_head(hidden_states)
6. Logits计算: logits = logits_processor(mtp_layer.shared_head.head, hidden_states)
7. 返回logits
```

**get_mtp_target_hidden_states**:
- AscendDeepseekV4ForCausalLM.get_mtp_target_hidden_states() (L1286-1290)
- 返回model._mtp_hidden_buffer，即主模型forward中保存的预-hc_head隐藏状态

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4_mtp.py#L201`

***

## 三、关键特性说明

### 3.1 混合协同处理 (HC - Hybrid Collaboration)

HC机制在两个层级应用：
1. **层内HC** (DecoderLayer中hc_pre/hc_post): 注意力和FFN分支各自独立的HC处理，使用独立的fn/base/scale参数组
2. **模型级HC头** (Model.hc_head): 所有层处理完成后的最终融合压缩

```
层内 hc_pre:  torch.ops._C_ascend.npu_hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
层内 hc_post: torch.ops._C_ascend.npu_hc_post(x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)).squeeze(0)

模型级 hc_head:
    归一化: rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    混合:   linear(x, hc_fn) * rsqrt
    激活:   sigmoid(mixes * hc_scale + hc_base) + hc_eps
    聚合:   sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
```

**HC参数形状总结**:
| 参数位置 | fn形状 | base形状 | scale形状 |
| ------- | ----- | ------- | -------- |
| hc_attn (每层) | (mix_hc, hc_dim) | (mix_hc,) | (3,) |
| hc_ffn (每层) | (mix_hc, hc_dim) | (mix_hc,) | (3,) |
| hc_head (模型/每个MTP层) | (hc_mult, hc_dim) | (hc_mult,) | (1,) |

其中:
- `mix_hc = (2 + hc_mult) * hc_mult`
- `hc_dim = hc_mult * hidden_size`

***

### 3.2 稀疏注意力 (DSA - DeepSeek Sparse Attention)

- **三级缓存架构**: Compressor状态缓存 + Indexer KV缓存 + SWA滑动窗口缓存
- **压缩比例**: 支持compress_ratio=4和compress_ratio=128两种配置
  - c4层: Compressor + Indexer (TopK稀疏选择)
  - c128层: 仅Compressor，无Indexer
  - 非压缩层(c1): 仅SWA缓存
- **索引缓存复用(skip_topk)**: V3.2配置支持按模式(`index_topk_pattern`)或频率(`index_topk_freq`)复用历史TopK索引，减少计算
- **注意力Sink**: attn_sink参数，enable_dsa_cp时完整复制否则按TP rank分片
- **RoPE配置**: 压缩层使用`compress_rope_theta`，非压缩层使用`rope_theta`，通过rope_groups区分
- **NPU算子**: `torch.ops._C_ascend.npu_sparse_flash_attention`

***

### 3.3 混合专家 (MoE - Mixture of Experts)

- **双路由模式**:
  - Hash路由: 前`num_hash_layers`层使用预定义的tid2eid映射，无学习偏置
  - 线性路由: 后续层使用gate线性层 + e_score_correction_bias
- **分组TopK**: `use_grouped_topk=True`，支持n_group和topk_group组级选择
- **EPLB**: 专家并行负载均衡，支持n_redundant_experts冗余专家热备
- **共享专家融合**: mix_placement配置启用时，共享专家融合到FusedMoE内部统一计算，不单独创建shared_experts
- **序列并行**: use_sequence_parallel_moe启用时，输入chunk分片到各TP rank，输出all_gather聚合
- **NPU相关**: muls_add_triton用于缩放融合，AITER融合算子可选
- **MTP层MoE**: MTP的mtp_block标记is_draft_layer=True，不使用hash路由

***

### 3.4 多Token预测 (MTP - Multi-Token Prediction)

- **轮询调度**: num_mtp_layers个MTP层按spec_step_idx轮询使用
- **投影融合**: e_proj(嵌入) + h_proj(前序隐藏状态)相加融合，广播到hc_mult维度
- **Mask机制**: position==0的输入嵌入被mask为0
- **延迟HC压缩**: Layer forward保持hc_mult维度输出，仅在compute_logits时做hc_head压缩
- **共享嵌入**: MTP不共享主模型embed_tokens，拥有独立embed_tokens
- **共享LM Head**: shared_head同时拥有norm和head，所有MTP步骤复用
- **缓冲区共享**: topk_indices_buffer和_mtp_hidden_buffer在主模型和MTP之间共享

***

### 3.5 流水线并行 (PP) 支持

- **条件实例化**: embed_tokens和norm分别仅在PP首/末阶段实例化，其他阶段用PPMissingLayer占位
- **层切片**: `make_layers`配合start_layer/end_layer切分层范围到各PP stage
- **中间张量**: make_empty_intermediate_tensors工厂创建形状为`(batch_size, hc_mult, hidden_size)`的零张量
- **HC扩展只做一次**: 仅在PP首阶段做unsqueeze+repeat扩展，后续PP阶段直接传递hc_mult维度张量
- **HC头只做一次**: 仅在PP末阶段执行hc_head压缩和最终norm

***

### 3.6 NPU优化

| 优化类型  | NPU算子/组件                                         |
| ----- | -------------------------------------------------- |
| 稀疏注意力 | `torch.ops._C_ascend.npu_sparse_flash_attention`   |
| HC预处理 | `torch.ops._C_ascend.npu_hc_pre`                   |
| HC后处理 | `torch.ops._C_ascend.npu_hc_post`                  |
| MoE路由 | `torch.ops._C_ascend.moe_gating_top_k_hash`        |
| RoPE旋转 | `torch_npu.npu_rotary_mul`                         |
| 缩放融合 | `muls_add_triton` (vllm_ascend/ops/triton/mul_add) |
| 索引器   | `npu_lightning_indexer`                            |

**设备类型适配**:
- A5设备: float8_e4m3fn kv-cache dtype，特殊block_size配置，compressor norm用float32
- 其他设备: int8(indexer)/bfloat16(swa) kv-cache dtype

***

### 3.7 并行策略

```
├── TP (Tensor Parallel): 张量并行
│   ├── wq_b: enable_dsa_cp时复制，否则列切分
│   ├── wo_a: 列切分
│   ├── wo_b: 行切分，结果allreduce
│   └── 序列并行时MLP禁用TP，输出all_gather
├── PP (Pipeline Parallel): 流水线并行
│   ├── embed_tokens: 首阶段
│   ├── layers: 按start_layer/end_layer切片
│   ├── norm + lm_head: 末阶段
│   └── IntermediateTensors传递hc_mult维隐藏状态
├── EP (Expert Parallel): 专家并行
│   ├── 物理专家按ep_size切分
│   ├── n_local_physical_experts = n_physical // ep_size
│   └── EPLB支持专家负载均衡重映射
├── DP (Data Parallel): 数据并行
├── MC2 (MoE Communication): MoE通信组
└── SP (Sequence Parallel): 序列并行
    ├── MLP输入sequence_parallel_chunk分片
    ├── FlashComm1: 每层输出reduce_scatter
    └── MTP缓冲区保存前all_gather聚合
```

***

## 四、文件位置参考

| 文件/模块  | 路径                                                        |
| ------ | --------------------------------------------------------- |
| 主模型入口  | `vllm-ascend/vllm_ascend/models/deepseek_v4.py`           |
| MTP模块  | `vllm-ascend/vllm_ascend/models/deepseek_v4_mtp.py`       |
| 注意力层封装 | `vllm-ascend/vllm_ascend/models/layer/attention/layer.py` |
| DSA后端  | `vllm-ascend/vllm_ascend/attention/dsa_v1.py`             |
| MoE融合  | `vllm-ascend/vllm_ascend/ops/fused_moe/`                  |
| DSA算子 | `vllm-ascend/vllm_ascend/ops/dsa/`                        |
| 旋转编码   | `vllm-ascend/vllm_ascend/ops/rope_dsv4.py`                |
| Triton算子 | `vllm-ascend/vllm_ascend/ops/triton/`                     |
| KV缓存接口 | `vllm-ascend/vllm_ascend/core/kv_cache_interface.py`      |
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
                    │  [PP首阶段: embed_tokens]              │
                    │  [层循环: start_layer→end_layer]       │
                    │  [MTP缓冲区: _mtp_hidden_buffer]       │
                    │  [PP末阶段: hc_head → norm]            │
                    └─────────────────────┬─────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │                                 │                                 │
┌───────▼───────┐               ┌────────▼────────┐               ┌─────────▼─────────┐
│ embed_tokens  │               │    layers[]     │               │      norm         │
│(VocabParallel)│               │DeepseekV2Decoder│               │   (RMSNorm)       │
│[PP首阶段]     │               │     Layer       │               │ [PP末阶段]        │
└───────────────┘               └────────┬────────┘               └───────────────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │       DeepseekV2DecoderLayer          │
                    │  residual.clone() → hc_pre → norm     │
                    │  → attn/mlp → hc_post → (h, residual) │
                    └───────────────────┬───────────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           │                            │                            │
    ┌──────▼──────┐             ┌───────▼───────┐             ┌──────▼──────┐
    │self_attn    │             │input/post_attn│             │   mlp       │
    │DeepseekV4   │             │  _layernorm   │             │DeepseekV4MoE│
    │Attention    │             │  (RMSNorm)    │             │/DeepseekV2  │
    └──────┬──────┘             └───────────────┘             │   MLP       │
           │                                                  └─────────────┘
           │
    ┌──────▼──────────────────────────────────────────────────────────┐
    │              dsa_attn: AscendDeepseekSparseAttention            │
    │                     (DSAModules封装)                             │
    └──────┬──────────┬───────────────┬────────────────┬───────────────┘
           │          │               │                │
    ┌──────▼──────┬───▼──────┐  ┌─────▼─────────┐  ┌───▼──────────────┐
    │  compressor │indexer   │  │swa_cache_layer│  │rotary_emb|attn_sink│
    │(c4/c128压缩) │(c4 TopK) │  │(滑动窗口缓存)  │  │(位置编码/sink)   │
    └─────────────┴──────────┘  └───────────────┘  └──────────────────┘
           │
    ┌──────▼──────────────────────────────────────────────────────────┐
    │                    AscendDSABackend (NPU实现)                    │
    │  wq_a→q_norm→wq_b → wkv→kv_norm → rotary → compressor/indexer   │
    │  → npu_sparse_flash_attention → wo_a → wo_b                     │
    └─────────────────────────────────────────────────────────────────┘
```
