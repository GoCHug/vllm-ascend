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

**类定义**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L904`

DeepSeekV4解码器层采用**HC（Hyper-Complex）机制**增强模型表达能力。每层包含注意力和FFN两个分支，每个分支都有独立的HC参数组，通过 `hc_pre` 和 `hc_post` 算子实现高维特征空间的门控混合。

---

#### 2.2.1 初始化组件 (L904-961)

| 组件 | 类型 | 形状/说明 |
| --- | --- | --- |
| `self_attn` | DeepseekV4Attention | 自注意力层 |
| `mlp` | DeepseekV4MoE | MoE前馈网络层 |
| `input_layernorm` | RMSNorm | 注意力前置层归一化 |
| `post_attention_layernorm` | RMSNorm | FFN前置层归一化 |
| `hc_mult` | int | HC扩展倍数（通常为4） |
| `hc_sinkhorn_iters` | int | Sinkhorn迭代次数（通常为20） |
| `hc_eps` | float | HC数值稳定性epsilon |
| `norm_eps` | float | RMSNorm epsilon |
| `hc_attn_fn` | nn.Parameter | 注意力HC门控权重 `(mix_hc, hc_dim)`，float32 |
| `hc_attn_base` | nn.Parameter | 注意力HC偏置 `(mix_hc,)`，float32 |
| `hc_attn_scale` | nn.Parameter | 注意力HC缩放因子 `(3,)`，float32 |
| `hc_ffn_fn` | nn.Parameter | FFN HC门控权重 `(mix_hc, hc_dim)`，float32 |
| `hc_ffn_base` | nn.Parameter | FFN HC偏置 `(mix_hc,)`，float32 |
| `hc_ffn_scale` | nn.Parameter | FFN HC缩放因子 `(3,)`，float32 |

**参数形状推导**:
- `mix_hc = (2 + hc_mult) * hc_mult` — 混合门控数量
- `hc_dim = hc_mult * hidden_size` — HC扩展后的特征维度

---

#### 2.2.2 HC核心机制详解

**HC机制概述**: 将隐藏状态从 `hidden_size` 维度扩展到 `hc_mult * hidden_size` 的高维空间，通过可学习的门控权重动态混合多个特征通道，再压缩回原始维度，增强模型的表达能力。

---

##### 2.2.2.1 `hc_pre` 算子 — 前置处理

**函数签名** (L963-967):
```python
def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, 
           hc_scale: torch.Tensor, hc_base: torch.Tensor):
    y = torch.ops._C_ascend.npu_hc_pre(
        x, hc_fn, hc_scale, hc_base, 
        self.hc_mult, self.hc_sinkhorn_iters, 
        self.norm_eps, self.hc_eps
    )
    return y
```

**输入输出形状**:
| 张量 | 形状 | dtype | 说明 |
| --- | --- | --- | --- |
| `x` (输入) | `(batch, hc_mult, hidden_size)` | bfloat16 | HC扩展后的隐藏状态 |
| `hc_fn` | `(mix_hc, hc_mult * hidden_size)` | float32 | 门控线性投影权重 |
| `hc_scale` | `(3,)` | float32 | 门控缩放因子 |
| `hc_base` | `(mix_hc,)` | float32 | 门控偏置 |
| `y` (输出0) | `(batch, hidden_size)` | bfloat16 | 压缩后的主路径特征，送入后续Norm + Attention/FFN |
| `post` (输出1) | `(batch, hc_mult)` | float32 | 后处理门控权重，传递给 `hc_post` |
| `comb_frag` (输出2) | `(batch, hc_mult, hc_mult)` | float32 | 组合矩阵片段，传递给 `hc_post` |

**算子内部实现流程** (`run_hc_pre_composite` in `csrc/torch_binding.cpp#L1438`):

```
1. InvRMS 归一化
   └── rsqrt = 1 / sqrt(mean(x²) + norm_eps)    # 形状: (batch, 1)

2. 线性投影计算 mixes
   ├── x_flat = x.float().flatten(1)            # (batch, hc_mult*H)
   └── mixes = x_flat @ hc_fn.T                 # (batch, mix_hc)

3. Sinkhorn 归一化 + 门控生成
   ├── 输入: mixes, rsqrt, hc_scale, hc_base
   ├── 通过 Sinkhorn 迭代 (hc_sinkhorn_iters次) 生成双随机矩阵
   ├── 计算 post 门控权重: sigmoid(mixes * scale + base) + hc_eps
   ├── 计算 comb_frag 组合矩阵: (batch, hc_mult, hc_mult)
   └── 输出: y (压缩后特征), post, comb_frag

4. 输出类型转换
   └── y = y.to(original_dtype)                 # 转回 bfloat16
```

---

###### `hc_pre` 数据流公式与 Shape 推导

**符号约定**:
- `B`: batch size (即 num_tokens，一次处理的token数量)
- `C`: hc_mult (HC通道数，固定为4)
- `H`: hidden_size (隐藏层维度，4096 或 7168)
- `M`: mix_hc (混合门控数，M = (2+C)*C = 24)
- `T`: hc_sinkhorn_iters (Sinkhorn迭代次数，默认20)

**输入张量详解**:

| 张量 | Shape | dtype | 是什么 | 怎么理解 |
|------|-------|-------|--------|---------|
| `x` | `(B, C, H)` | bfloat16 | HC扩展后的隐藏状态 | 每个token有C=4个通道，每个通道H维特征 |
| `hc_fn` | `(M, C*H)` | float32 | 门控线性投影权重 | 把 C*H 维 映射到 M=24 维的门控参数 |
| `hc_scale` | `(3,)` | float32 | 门控缩放因子 | 3个标量，分别控制3种门控的"温度" |
| `hc_base` | `(M,)` | float32 | 门控偏置 | 24个标量，每个门控参数一个偏置 |

> **hc_scale 的3个值分别对应什么？**
> - `hc_scale[0]`: pre门控缩放（用于特征压缩前的门控）
> - `hc_scale[1]`: post门控缩放（用于hc_post的门控）
> - `hc_scale[2]`: comb门控缩放（用于组合矩阵Sinkhorn的门控）
>
> 它们是3个独立的标量，不是向量！每个用在不同的地方。

**输出张量详解**:

| 张量 | Shape | dtype | 是什么 | 怎么理解 |
|------|-------|-------|--------|---------|
| `y` | `(B, H)` | bfloat16 | 压缩后的主路径特征 | C=4个通道压缩成1个，送给注意力/FFN计算 |
| `post` | `(B, C)` | float32 | 后处理门控权重 | 每个通道一个权重，用于hc_post时的门控 |
| `comb_frag` | `(B, C, C)` | float32 | 组合矩阵片段 | C×C的双随机矩阵，描述通道间如何混合 |

---

**步骤1: InvRMS 归一化**

调用算子: `aclnnHcPreInvRms`

**这一步做什么？**
计算每个样本的"均方根倒数"，用于后续的归一化。
简单说：先算所有元素平方的平均值，开根号，再取倒数。

**Shape 变化**:
```
输入 x:  (B, C, H) bfloat16
         │
         │  对每个 batch b，计算 C*H 个元素的均方根倒数
         │  即：把 (B, C, H) 的后两维全部"压缩"成一个标量
         ▼
输出 rsqrt: (B, 1) float32
```

**计算公式**:
```
对于第 b 个样本:
  rsqrt[b] = 1 / sqrt( mean(x[b,:,:]^2) + norm_eps )

展开写:
  mean_val = (1/(C*H)) * sum(x[b,0,0]^2 + x[b,0,1]^2 + ... + x[b,C-1,H-1]^2)
  rsqrt[b] = 1 / sqrt(mean_val + norm_eps)

shape 说明:
  x.shape      = (B, C, H)  # 每个样本有 C*H 个值
  rsqrt.shape  = (B, 1)     # 每个样本只有 1 个值
```

> **为什么 rsqrt 是 (B, 1) 而不是 (B,)?**
> 因为 (B, 1) 可以方便地和其他二维/三维张量做广播乘法。
> 比如 mixes 是 (B, M)，rsqrt 是 (B, 1)，相乘时会自动广播成 (B, M)。

---

**步骤2: 线性投影计算 mixes**

调用算子: `at::linear` (PyTorch 原生矩阵乘)

**这一步做什么？**
把展平的特征通过一个线性层（矩阵乘法），映射到 M=24 维的门控参数空间。

**Shape 变化**:
```
x: (B, C, H) bfloat16
    │
    │  步骤2.1: 转 float32（计算精度要求）
    ▼
x_float: (B, C, H) float32
    │
    │  步骤2.2: 展平 C 和 H 维度
    │  把每个样本的 C*H 个值排成一个长向量
    ▼
x_flat: (B, C*H) float32
    │
    │  步骤2.3: 矩阵乘法 (线性投影)
    │  x_flat @ hc_fn.T
    │  (B, C*H) × (C*H, M) = (B, M)
    ▼
mixes: (B, M) float32
```

**计算公式**:
```
x_float = x.to(torch.float32)     # shape: (B, C, H)   转float32
x_flat = x_float.flatten(1)       # shape: (B, C*H)   第1维开始展平
mixes = x_flat @ hc_fn.T          # shape: (B, M)     矩阵乘法

shape 验证:
  x_flat.shape = (B, C*H)         # 输入特征
  hc_fn.shape  = (M, C*H)         # 权重矩阵，M=24
  hc_fn.T.shape = (C*H, M)        # 转置后
  mixes.shape  = (B, M)           # 输出，每个样本24个门控值
```

> **mixes 的 M=24 维分别是什么？**
> - 前 C=4 维: post 门控（用于 hc_post）
> - 接下来 C*C=16 维: comb 组合矩阵（用于通道混合）
> - 最后 C=4 维: pre 门控（用于特征压缩）
> 总共: 4 + 16 + 4 = 24 = M

---

**步骤3: Sinkhorn 归一化 + 门控生成 + 特征压缩**

调用算子: `aclnnHcPreSinkhorn`
源码位置: `csrc/moe/hc_pre_sinkhorn/op_kernel/`

**这一步做什么？**
这是最复杂的一步，主要做四件事：
1. 用 mixes 的前 C 维 + scale[0] + base[0:C] → 计算 pre 门控
2. 用 mixes 的中间 C 维 + scale[1] + base[C:2C] → 计算 post 门控
3. 用 mixes 的后 C*C 维 + scale[2] + base[2C:] → 做 Sinkhorn 迭代，生成组合矩阵 comb_frag
4. 用 pre 门控作为权重，把 x 从 C 个通道加权求和压缩成 1 个通道

> **重要说明**：本步骤的计算流程是直接从 NPU 内核源码（`hc_pre_sinkhorn_perf.h` 和 `hc_pre_sinkhorn_base.h`）反推出来的，不是推测！

**输入输出 Shape**:
```
输入:
  mixes:    (B, M)     float32   # 步骤2的输出，M = C + C + C*C = 4+4+16 = 24维
  rsqrt:    (B, 1)     float32   # 步骤1的输出，归一化因子
  hc_scale: (3,)       float32   # 3个标量：[pre_scale, post_scale, comb_scale]
  hc_base:  (M,)       float32   # 24个偏置：[pre_base(C个), post_base(C个), comb_base(C*C个)]
  x:        (B, C, H)  bfloat16  # 原始特征，用于压缩
    │
    ▼
输出:
  y:         (B, H)       bfloat16  # 压缩后的主路径特征（pre门控加权求和）
  post:      (B, C)       float32   # post门控权重，给hc_post用
  comb_frag: (B, C, C)    float32   # 双随机组合矩阵，给hc_post用
```

> **重要：hc_scale 的用法（解答你的疑问）**
>
> hc_scale 是 (3,)，不是向量！它是**3个独立的标量数**，分别用在3个不同的地方：
> - `hc_scale[0]` → pre 门控的缩放因子（标量，1个数）
> - `hc_scale[1]` → post 门控的缩放因子（标量，1个数）
> - `hc_scale[2]` → comb 门控的缩放因子（标量，1个数）
>
> **不是 (B, M) × (3,) 矩阵乘法！** 而是"标量 × 矩阵"的广播乘法。
> 标量乘以矩阵，就是矩阵里每个元素都乘这个标量，shape 不变。
>
> 比如:
> ```
> mixes.shape = (B, M)       # 比如 (8, 24)
> hc_scale[0] = 0.5          # 一个标量数
> mixes * hc_scale[0]        # shape 还是 (B, M) = (8, 24)
>                            # 每个元素都乘 0.5
> ```

---

**mixes 和 hc_base 的分段结构（源码级精确）**

mixes 总共有 M = C + C + C*C = 24 维，分成三段：

```
mixes (B, 24)
  ├── 第 0~3 维 (C=4维)     → pre 门控的原始参数
  ├── 第 4~7 维 (C=4维)     → post 门控的原始参数
  └── 第 8~23 维 (C*C=16维) → comb 组合矩阵的原始参数

hc_base (24,)
  ├── 第 0~3 维 (C=4个)     → pre 门控的偏置 (hcBase0Local)
  ├── 第 4~7 维 (C=4个)     → post 门控的偏置 (hcBase1Local)
  └── 第 8~23 维 (C*C=16个) → comb 门控的偏置 (hcBase2Local)
```

> **源码依据**: 
> - `hc_pre_sinkhorn_perf.h:95-97` 中 hcBase 的三次 CopyIn 分别取前 C、中间 C、后 C*C 个
> - `hc_pre_sinkhorn_perf.h:106-161` 中 mixes 的拷贝也是同样的分段

---

**内部子步骤（按源码实际执行顺序）**:

---

**3.1 pre 门控计算（对应源码 `ProcessPre` 函数）**

源码位置: `hc_pre_sinkhorn_base.h:387-400`

**输入**:
- pre_mixes: mixes 的前 C 维 → shape `(B, C)`
- rsqrt: 归一化因子 → shape `(B, 1)`
- hc_scale[0]: pre 缩放因子 → 标量
- hc_base[0:C]: pre 偏置 → shape `(C,)`
- hc_eps: 很小的数，防止除0 → 标量

**计算顺序（严格按源码）**:

```
# 第1步: 乘 rsqrt（行归一化）
# (B, C) * (B, 1) → (B, C) 广播乘法
# 效果：每一行的 C 个值都乘上这一行的 rsqrt
pre_tmp1 = pre_mixes * rsqrt
pre_tmp1.shape = (B, C)

# 第2步: 乘 scale（标量缩放）
# (B, C) * scalar → (B, C)
# 效果：每个元素都乘 scale[0]
pre_tmp2 = pre_tmp1 * hc_scale[0]
pre_tmp2.shape = (B, C)

# 第3步: 加 base（加偏置）
# (B, C) + (C,) → (B, C) 广播加法
# 效果：每一行都加上同一个 base 向量
pre_logits = pre_tmp2 + hc_base[0:C]
pre_logits.shape = (B, C)

# 第4步: sigmoid 激活
# 把值压缩到 (0, 1) 区间
pre_sigmoid = sigmoid(pre_logits)
pre_sigmoid.shape = (B, C)

# 第5步: 加 eps（防止为0）
# 每个元素加一个很小的数 hc_eps
pre_gate = pre_sigmoid + hc_eps
pre_gate.shape = (B, C)
```

> **广播机制小科普**
> - (B, C) 和 (B, 1) 相乘：PyTorch 把 (B, 1) 的最后一维"复制"C份，变成 (B, C)，然后对应位置相乘。效果是每一行的 C 个元素都乘上这一行的 rsqrt 值。
> - (B, C) 和 (C,) 相加：PyTorch 把 (C,) 的第0维"复制"B份，变成 (B, C)，然后对应位置相加。效果是每一行都加上同一个 base 向量。
> - 标量乘矩阵：标量自动扩展成和矩阵一样的 shape，每个元素都乘这个标量。

---

**3.2 特征压缩 x → y（对应源码 `ProcessY` 函数）**

源码位置: `hc_pre_sinkhorn_base.h:463-472`

> **注意**：从源码执行顺序看，特征压缩在 post 门控和 Sinkhorn 之前计算！
> 因为 pre 门控算完之后，马上就可以算 y 了。

**输入**:
- x: 原始特征 → shape `(B, C, H)`, dtype bfloat16
- pre_gate: pre 门控 → shape `(B, C)`, dtype float32

**计算顺序（严格按源码）**:

```
# 第1步: x 从 bfloat16 cast 成 float32
x_float = x.to(torch.float32)
x_float.shape = (B, C, H)

# 第2步: x 乘以 pre_gate（广播乘法）
# x_float: (B, C, H)
# pre_gate: (B, C) → 广播成 (B, C, 1) → 广播成 (B, C, H)
# 效果：每个通道的所有 H 维值都乘上该通道的门控权重
x_gated = x_float * pre_gate.unsqueeze(-1)
x_gated.shape = (B, C, H)

# 第3步: 对 C 维（通道维）reduce sum，压缩成 1 个通道
# 把 4 个通道加权求和，变成 1 个通道
y_float = x_gated.sum(dim=1)   # 或者用 einsum: torch.einsum("bch,bc->bh", x_float, pre_gate)
y_float.shape = (B, H)

# 第4步: cast 回 bfloat16
y = y_float.to(torch.bfloat16)
y.shape = (B, H)
```

> **直观理解**：
> 原来每个 token 有 C=4 个通道的特征，每个通道 H 维。
> pre_gate 的 4 个值就是 4 个权重（0~1之间），
> 我们用这 4 个权重把 4 个通道的特征加权求和，
> 得到 1 个 H 维的特征，送给后面的注意力/FFN层计算。
> 这样计算量就变成了原来的 1/4。

> **源码依据**：`ProcessY` 函数里依次调用了 `CastTwoDim` → `MulABLastDimBrcInline` → `ReduceSumARAPerf` → `CastTwoDim`，完全对应上面的四步。

---

**3.3 post 门控计算（对应源码 `ProcessPost` 函数）**

源码位置: `hc_pre_sinkhorn_base.h:475-488`

**输入**:
- post_mixes: mixes 的中间 C 维 → shape `(B, C)`
- rsqrt: 归一化因子 → shape `(B, 1)`
- hc_scale[1]: post 缩放因子 → 标量
- hc_base[C:2C]: post 偏置 → shape `(C,)`

**计算顺序（严格按源码）**:

```
# 第1步: 乘 rsqrt
post_tmp1 = post_mixes * rsqrt       # (B, C) * (B, 1) -> (B, C)
post_tmp1.shape = (B, C)

# 第2步: 乘 scale
post_tmp2 = post_tmp1 * hc_scale[1]  # (B, C) * scalar -> (B, C)
post_tmp2.shape = (B, C)

# 第3步: 加 base
post_logits = post_tmp2 + hc_base[C:2C]  # (B, C) + (C,) -> (B, C)
post_logits.shape = (B, C)

# 第4步: sigmoid 激活
post_sigmoid = sigmoid(post_logits)  # 压缩到 (0, 1)
post_sigmoid.shape = (B, C)

# 第5步: 乘 2（注意：没有加 eps！）
post = post_sigmoid * 2.0            # 放大到 (0, 2) 区间
post.shape = (B, C)
```

> **pre 和 post 的区别**：
> - pre: sigmoid + eps → 范围 (eps, 1+eps)，接近 (0, 1)
> - post: sigmoid × 2 → 范围 (0, 2)，没有 eps
> - pre 用于压缩时的加权求和（权重需要大于0）
> - post 用于扩展时的门控（可以大于1，相当于放大）

---

**3.4 comb 组合矩阵 + Sinkhorn 迭代**

源码位置: `hc_pre_sinkhorn_perf.h:167-192` + `hc_pre_sinkhorn_base.h:509-523`

**输入**:
- comb_mixes: mixes 的后 C*C 维 → shape `(B, C*C)`
- rsqrt: 归一化因子 → shape `(B, 1)`
- hc_scale[2]: comb 缩放因子 → 标量
- hc_base[2C:2C+C*C]: comb 偏置 → shape `(C*C,)`
- hc_sinkhorn_iters: Sinkhorn 迭代次数 T → 整数
- hc_eps: 防止除0的小数 → 标量

**计算顺序（严格按源码）**:

```
# ============================================
# 第1阶段: logits 计算 + reshape
# ============================================

# 1.1 乘 rsqrt
comb_tmp1 = comb_mixes * rsqrt          # (B, C*C) * (B, 1) -> (B, C*C)
comb_tmp1.shape = (B, C*C)

# 1.2 乘 scale
comb_tmp2 = comb_tmp1 * hc_scale[2]     # (B, C*C) * scalar -> (B, C*C)
comb_tmp2.shape = (B, C*C)

# 1.3 加 base
comb_logits = comb_tmp2 + hc_base[2C:2C+C*C]  # (B, C*C) + (C*C,) -> (B, C*C)
comb_logits.shape = (B, C*C)

# 1.4 reshape 成矩阵形式
P0 = comb_logits.reshape(B, C, C)        # (B, 16) -> (B, 4, 4)
P0.shape = (B, C, C)
# 每个样本对应一个 C×C 的矩阵

# ============================================
# 第2阶段: 初始 Softmax（行归一化）
# ============================================
# 源码: SoftmaxFP32Perf (hc_pre_sinkhorn_base.h:509-523)
# 数值稳定的 Softmax：先减 max，再 exp，再除 sum

# 2.1 减 max（每行减该行的最大值，防止 exp 溢出）
row_max = P0.max(dim=-1, keepdim=True).values   # (B, C, 1)
P_minus_max = P0 - row_max                      # (B, C, C)

# 2.2 exp 指数
P_exp = P_minus_max.exp()                       # (B, C, C)

# 2.3 除行和（行归一化：每行的和为 1）
row_sum = P_exp.sum(dim=-1, keepdim=True)       # (B, C, 1)
P = P_exp / (row_sum + hc_eps)                  # (B, C, C)
# 此时 P 的每行和为 1（近似，因为加了 eps）

# ============================================
# 第3阶段: 初始列归一化
# ============================================

# 3.1 求列和
col_sum = P.sum(dim=-2, keepdim=True)           # (B, 1, C)

# 3.2 列归一化：每列除以列和
P = P / (col_sum + hc_eps)                      # (B, C, C)
# 此时 P 的每列和为 1（近似）

# ============================================
# 第4阶段: Sinkhorn 迭代 (T-1 次)
# ============================================
# 源码: for (iter = 0; iter < iterTimes - 1; iter++)
# 每次迭代 = 行归一化 + 列归一化

for t = 1 to T-1:
    # 4.1 行归一化
    row_sum = P.sum(dim=-1, keepdim=True)       # (B, C, 1)
    P = P / (row_sum + hc_eps)                  # (B, C, C)
    
    # 4.2 列归一化
    col_sum = P.sum(dim=-2, keepdim=True)       # (B, 1, C)
    P = P / (col_sum + hc_eps)                  # (B, C, C)

# ============================================
# 第5阶段: 最终结果
# ============================================
comb_frag = P
comb_frag.shape = (B, C, C)
```

> **什么是双随机矩阵？**
> 每行加起来等于1，每列加起来也等于1的矩阵。
> 比如 4x4 的矩阵：
> - 第1行 4个数加起来 ≈ 1
> - 第2行 4个数加起来 ≈ 1
> - ...
> - 第1列 4个数加起来 ≈ 1
> - 第2列 4个数加起来 ≈ 1
> - ...
> 这种矩阵可以看作是"通道间的转移概率"——描述了 4 个输入通道如何混合成 4 个输出通道。

> **Sinkhorn 迭代的次数**：
> 源码中循环 `iterTimes - 1` 次，每次循环做 1 次行归一化 + 1 次列归一化。
> 加上循环前的 1 次 softmax（行归一化） + 1 次列归一化，
> 总共做了 T 次行归一化和 T 次列归一化（T = hc_sinkhorn_iters）。
> 迭代次数越多，矩阵越接近严格的双随机矩阵。

---

**步骤4: 输出总结**

```
最终输出三个张量:
  y:         (B, H)     bfloat16  # pre门控加权压缩后的主路径特征
  post:      (B, C)     float32   # post门控 (0~2之间)，给hc_post用
  comb_frag: (B, C, C)  float32   # 双随机组合矩阵，给hc_post用
```

> **注意**: 上述所有子步骤（3.1 ~ 3.4）都在 `aclnnHcPreSinkhorn` 单个 CANN 算子内完成，
> 上面的公式是直接从 NPU 内核源码反推的，计算顺序和逻辑与源码一致。

**两种实现模式**:
- **融合模式 (`run_hc_pre_fusion`)**: 单算子 `aclnnHcPre` 完成全部计算（910B默认 / 950满足条件时使用）
- **组合模式 (`run_hc_pre_composite`)**: 拆分为 `aclnnHcPreInvRms` + `at::linear` + `aclnnHcPreSinkhorn` 三步（950大batch时使用）

---

###### `npu_hc_pre` 完整调用链

从 Python 调用到 NPU 硬件执行的完整路径如下：

```
┌─────────────────────────────────────────────────────────────────────┐
│  第1层: Python 调用层                                               │
│  文件: deepseek_v4.py:L963-967                                     │
│  函数: hc_pre()                                                    │
│  调用: torch.ops._C_ascend.npu_hc_pre(x, hc_fn, ...)               │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  第2层: PyTorch Dispatcher 调度层                                   │
│  算子注册: torch_binding.cpp:L2583-2590                            │
│  ops.def("npu_hc_pre(...) -> (Tensor out0, Tensor out1, Tensor out2)")
│  ops.impl("npu_hc_pre", kPrivateUse1, &npu_hc_pre_npu)             │
│  根据输入张量的设备类型 (NPU) 调度到对应后端实现                     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  第3层: C++ NPU 入口函数                                            │
│  文件: torch_binding.cpp:L1479-1485                                │
│  函数: npu_hc_pre_npu()                                            │
│  ├── check_hc_pre_shape_and_dtype()  # 参数形状/类型校验 (L1376)   │
│  │   ├── 校验维度: x是3D/4D, hc_fn是2D, hc_scale是(3,)             │
│  │   ├── 校验大小: hc_mult=4, hidden_size=4096/7168, mix_hc=24     │
│  │   └── 校验类型: x=bfloat16, hc_*=float32                        │
│  └── return run_hc_pre_composite()   # 调用组合模式实现             │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  第4层: 组合模式调度 (Composite Mode)                               │
│  文件: torch_binding.cpp:L1438-1463                                │
│  函数: run_hc_pre_composite()                                      │
│                                                                     │
│  子步骤1: InvRMS 归一化                                             │
│  ├── construct_hc_pre_rsqrt_output_tensor()  # 构造输出张量 (L1359)│
│  └── EXEC_NPU_CMD(aclnnHcPreInvRms, x, norm_eps, rsqrt)            │
│      调用 CANN 算子库的 aclnnHcPreInvRms                           │
│                                                                     │
│  子步骤2: 线性投影计算 mixes                                        │
│  ├── x_float = x.to(at::kFloat)        # 转float32                 │
│  ├── x_flattened = x_float.flatten(1,-1) # 展平最后两维            │
│  └── mixes = at::linear(x_flattened, hc_fn)  # PyTorch 原生linear  │
│                                                                     │
│  子步骤3: Sinkhorn 归一化 + 门控生成                                │
│  ├── construct_hc_pre_output_tensor()  # 构造y/post/comb输出张量   │
│  │   ├── y: (batch, H), bfloat16                                   │
│  │   ├── post: (batch, hc_mult), float32                           │
│  │   └── comb_frag: (batch, hc_mult, hc_mult), float32             │
│  └── EXEC_NPU_CMD(aclnnHcPreSinkhorn, mixes, rsqrt, ...)           │
│      调用 CANN 算子库的 aclnnHcPreSinkhorn                         │
│                                                                     │
│  后处理: y = y.to(original_dtype)   # 转回bfloat16                 │
│  返回: std::tuple(y, post, comb_frag)                              │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  第5层: CANN 算子库 + NPU 硬件层                                    │
│  aclnnHcPreInvRms    → NPU 硬件执行 InvRMS 计算                    │
│  at::linear          → NPU 硬件执行矩阵乘 (由 PyTorch NPU 后端调度) │
│  aclnnHcPreSinkhorn  → NPU 硬件执行 Sinkhorn 迭代 + 门控生成       │
└─────────────────────────────────────────────────────────────────────┘
```

**v2 版本的差异** (`npu_hc_pre_v2_npu`, L1487-1496):
- v1 固定走组合模式 `run_hc_pre_composite`
- v2 调用 `should_use_hc_pre_fusion(x)` 判断（L1429-1436）：
  - 非 Ascend950 → 用融合模式 `run_hc_pre_fusion`
  - Ascend950 且 batch ≤ 512 → 用融合模式
  - Ascend950 且 batch 是 8192 倍数 → 用融合模式
  - 其他情况 → 用组合模式
- 当前 Python 代码使用的是 v1 版本

---

##### 2.2.2.2 `hc_post` 算子 — 后置处理

**函数签名** (L969-973):
```python
def hc_post(self, x: torch.Tensor, residual: torch.Tensor, 
            post: torch.Tensor, comb: torch.Tensor):
    y = torch.ops._C_ascend.npu_hc_post(
        x.unsqueeze(dim=0), 
        residual.unsqueeze(dim=0), 
        post.unsqueeze(dim=0), 
        comb.unsqueeze(dim=0)
    )
    return y.squeeze(dim=0)
```

**输入输出形状**:
| 张量 | 形状 | dtype | 说明 |
| --- | --- | --- | --- |
| `x` | `(batch, hidden_size)` | bfloat16/float16/float32 | Attention/FFN输出特征 |
| `residual` | `(batch, hc_mult, hidden_size)` | 同x | hc_pre前的残差（克隆自原始hidden_states） |
| `post` | `(batch, hc_mult)` | float32/bfloat16/float16 | hc_pre输出的后处理门控 |
| `comb` | `(batch, hc_mult, hc_mult)` | float32/bfloat16/float16 | hc_pre输出的组合矩阵 |
| `y` (输出) | `(batch, hc_mult, hidden_size)` | 同x | 恢复为HC扩展维度的隐藏状态 |

**算子内部功能** (源码级精确):
- 用 `comb` 组合矩阵对残差 `residual` 的 C 个通道做线性混合（通道间特征重组）
- 用 `post` 门控权重对主路径输出 `x` 做通道级缩放，扩展成 C 个通道
- 两者相加，得到最终的 HC 扩展维度隐藏状态
- 输出形状为 `(batch, hc_mult, hidden_size)`，供下一层使用

> **核心公式** (从 NPU 内核源码 `hc_post_bfloat16.h:205-309` 反推):
> ```
> ┌──────────────────────────────────────────────────────────────────┐
> │  y[b, j, h] = Σ residual[b, i, h] · comb[b, i, j] + x[b, h] · post[b, j]
> │               i=0..C-1
> │
> │  b  : batch 索引
> │  i  : 输入通道索引  (0 ~ C-1)
> │  j  : 输出通道索引  (0 ~ C-1)
> │  h  : hidden 维度索引 (0 ~ H-1)
> └──────────────────────────────────────────────────────────────────┘
> ```

**注意**: Python层做了 `unsqueeze(0)` 和 `squeeze(0)` 处理，用于匹配NPU算子预期的4D输入格式 `(1, batch, hc_mult, hidden_size)`。

---

###### `hc_post` 数据流公式与 Shape 推导

**符号约定**:
- `B`: batch size (即 num_tokens，一次处理的token数量)
- `C`: hc_mult (HC通道数，固定为4)
- `H`: hidden_size (隐藏层维度，4096 或 7168)

**输入张量详解** (Python 层 → 经 unsqueeze(0) 后送入 C++):

| 张量 | Python 层 Shape | NPU算子输入 Shape | dtype | 是什么 | 怎么理解 |
|------|----------------|-----------------|-------|--------|---------|
| `x` | `(B, H)` | `(1, B, H)` | bfloat16 | Attention/FFN 输出特征 | 主路径计算结果，1个通道，H维 |
| `residual` | `(B, C, H)` | `(1, B, C, H)` | bfloat16 | hc_pre 前的残差 | 残差连接，C=4个通道 |
| `post` | `(B, C)` | `(1, B, C)` | float32 | hc_pre 输出的后处理门控 | 每个通道一个权重，0~2之间 |
| `comb` | `(B, C, C)` | `(1, B, C, C)` | float32 | hc_pre 输出的组合矩阵 | 双随机矩阵，描述通道混合方式 |

> **为什么要 unsqueeze(0)?**
> NPU 算子要求输入是 4D 格式（多了一个第0维），所以 Python 层在调用前加了一维，
> 调用完再 squeeze 掉。这是 NPU 算子的常见格式要求，不影响计算逻辑。

**输出张量详解**:

| 张量 | NPU算子输出 Shape | Python 层 Shape | dtype | 是什么 | 怎么理解 |
|------|-----------------|----------------|-------|--------|---------|
| `y` | `(1, B, C, H)` | `(B, C, H)` | bfloat16 | 恢复HC维度的隐藏状态 | 4个通道的特征，供下一层使用 |

---

**计算流程** (调用算子: `aclnnHcPost`):

**这一步做什么？（源码级精确）**
hc_pre 是把 C=4 个通道压缩成 1 个，送给注意力/FFN计算；
hc_post 做两件事：
1. 用 `comb` 组合矩阵把残差 `residual` 的 C 个通道重新混合（通道间特征重组）
2. 用 `post` 门控把主路径输出 `x` 扩展成 C 个通道（每个通道乘一个门控权重）
3. 两者相加，得到最终的 HC 扩展维度隐藏状态

> **重要说明**：本步骤的计算流程是直接从 NPU 内核源码（`hc_post_bfloat16.h` 的 `DoMulAndAdd` 函数）反推出来的，不是推测！

**Shape 变化**:
```
输入:
  x:        (1, B, H)       bfloat16   [主路径输出，1个通道]
  residual: (1, B, C, H)    bfloat16   [残差连接，4个通道]
  post:     (1, B, C)       float32    [门控权重，4个通道各一个]
  comb:     (1, B, C, C)    float32    [组合矩阵，4x4，双随机矩阵]
    │
    │  1. residual 通道混合: residual @ comb  (按通道维矩阵乘)
    │  2. x 扩展 + 门控: x 广播到 C 个通道，每个通道乘 post[j]
    │  3. 两者相加
    ▼
输出:
  y: (1, B, C, H)  bfloat16
  -> Python 层 squeeze(0) -> (B, C, H)
```

---

**源码中的核心计算（`DoMulAndAdd` 函数，L205-309）**

源码里是对每个输出通道 j（hcIndex = j）分别计算：

```cpp
// 输入张量 shape:
//   residual: (C, H)    ← C个输入通道，每个通道H维
//   x:        (H,)      ← 主路径输出，1个通道，H维
//   comb:     (C, C)    ← C×C 组合矩阵
//   post:     (C,)      ← C个门控权重
//   y:        (C, H)    ← C个输出通道，每个通道H维

for each output channel j:   // j = 0, 1, 2, 3
    // 加载 comb 矩阵的第 j 列  → shape: (C,)
    comb[0,j], comb[1,j], comb[2,j], comb[3,j]

    // 加载 post 的第 j 个值  → 标量
    post[j]

    // 计算第 j 个输出通道  → shape: (H,)
    y[j] = residual[0] * comb[0,j]   // 第0个输入通道 × comb[0,j]
         + residual[1] * comb[1,j]   // 第1个输入通道 × comb[1,j]
         + residual[2] * comb[2,j]   // 第2个输入通道 × comb[2,j]
         + residual[3] * comb[3,j]   // 第3个输入通道 × comb[3,j]
         + x * post[j]               // 主路径输出 × post[j]
```

> **源码实际计算顺序**：
> 内核中为了性能优化，加法顺序略有调整：
> ```
> y[j] = residual[0] * comb[0,j]
>      + residual[3] * comb[3,j]
>      + residual[1] * comb[1,j]
>      + residual[2] * comb[2,j]
>      + x * post[j]
> ```
> （先算通道0，再加通道3、1、2，最后加主路径）
> 数学上和正常顺序完全等价。

---

**步骤1：残差通道混合（核心！）**

用 `comb` 组合矩阵对 `residual` 的 C 个输入通道做线性混合，得到 C 个输出通道。

**这一步在做什么？**
把残差的 4 个输入通道，通过一个 C×C 的混合矩阵，重新组合成 4 个输出通道。
这是 HC 机制的核心——通道间的动态特征重组。

**Shape 变化**:
```
residual:  (B, C, H)    bfloat16   # 输入：C个通道，每个通道H维
comb:      (B, C, C)    float32    # 混合矩阵：C输入通道 × C输出通道
    │
    │  对每个 batch b，每个 hidden 位置 h：
    │  residual[b, :, h] 是 C 维行向量
    │  comb[b, :, :] 是 C×C 矩阵
    │  结果 = residual[b, :, h] @ comb[b, :, :]  →  C 维行向量
    ▼
residual_mixed: (B, C, H)  float32
```

**计算公式**:
```
╔══════════════════════════════════════════════════════════════════╗
║  逐元素公式:                                                     ║
║                                                                  ║
║  residual_mixed[b, j, h] = Σ residual[b, i, h] · comb[b, i, j]  ║
║                            i=0..C-1                              ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  einsum 表示:                                                    ║
║                                                                  ║
║  residual_mixed = einsum('b i h, b i j → b j h', residual, comb)║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Shape 验证:                                                     ║
║                                                                  ║
║  residual.shape  = (B, C, H)    ← b=batch, i=输入通道, h=hidden ║
║  comb.shape      = (B, C, C)    ← b=batch, i=输入通道, j=输出通道║
║  结果.shape      = (B, C, H)    ← b=batch, j=输出通道, h=hidden ║
╚══════════════════════════════════════════════════════════════════╝
```

**大白话理解**:
```
╭──────────────────────────────────────────────────────────────────────╮
│                                                                    │
│  要算"第 j 个输出通道的第 h 个值"，                                   │
│  就把"所有输入通道的第 h 个值"                                        │
│  分别乘以"该输入通道对第 j 个输出通道的权重"，                         │
│  然后加起来。                                                       │
│                                                                    │
│  类比：                                                             │
│    输出通道 j = 输入通道0 × 权重0→j                                 │
│                + 输入通道1 × 权重1→j                                │
│                + 输入通道2 × 权重2→j                                │
│                + 输入通道3 × 权重3→j                                │
│                                                                    │
╰──────────────────────────────────────────────────────────────────────╯
```

> **comb 矩阵的理解**：
> - comb[b, i, j] 表示：第 i 个输入通道 对 第 j 个输出通道 的贡献权重
> - 因为 comb 是双随机矩阵（每行和为1，每列和为1），所以这是一个"保能量"的混合
> - 输出通道的总能量 ≈ 输入通道的总能量

---

**步骤2：主路径扩展 + 门控加权**

用 `post` 门控把主路径输出 `x` 从 1 个通道扩展成 C 个通道。

**这一步在做什么？**
x 只有 1 个通道（H维），我们把它"复制"成 C 个通道，
每个通道乘上一个对应的 post 门控权重，实现通道级的自适应缩放。

**Shape 变化**:
```
x:     (B, H)     bfloat16   # 主路径输出，1个通道
post:  (B, C)     float32    # 门控权重，C个通道各一个
    │
    │  广播机制:
    │  x.unsqueeze(1)     → (B, 1, H)
    │  post.unsqueeze(-1) → (B, C, 1)
    │  两者相乘 → 广播成 (B, C, H)
    ▼
x_gated: (B, C, H)  float32
```

**计算公式**:
```
╔══════════════════════════════════════════════════════════════════╗
║  逐元素公式:                                                     ║
║                                                                  ║
║  x_gated[b, j, h] = x[b, h] · post[b, j]                        ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  广播机制示意图:                                                  ║
║                                                                  ║
║  x.shape     = (B,    H)   → unsqueeze(1) → (B, 1, H)           ║
║                                                  ↓ 广播          ║
║  post.shape  = (B, C   )   → unsqueeze(-1) → (B, C, 1)          ║
║                                                  ↓ 广播          ║
║  结果.shape  = (B, C, H)                                         ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Shape 验证:                                                     ║
║                                                                  ║
║  x.shape     = (B, H)       → 广播至 (B, C, H)                   ║
║  post.shape  = (B, C)       → 广播至 (B, C, H)                   ║
║  结果.shape  = (B, C, H)                                         ║
╚══════════════════════════════════════════════════════════════════╝
```

> **post 门控的作用**：
> - post 值范围是 (0, 2)（从 hc_pre 的 ProcessPost 可知：sigmoid × 2）
> - post[j] > 1：第 j 个通道被放大
> - post[j] < 1：第 j 个通道被衰减
> - 这是一种动态的通道注意力机制，每个 token 可以自己决定每个通道的强度

---

**步骤3：两者相加 + 类型转换**

把通道混合后的残差 和 门控扩展后的主路径 加在一起，然后转回 bfloat16。

**计算公式**:
```
╔══════════════════════════════════════════════════════════════════╗
║  最终输出:                                                       ║
║                                                                  ║
║  y_float = residual_mixed + x_gated                              ║
║  y       = y_float.to(bfloat16)                                  ║
║                                                                  ║
║  y.shape = (B, C, H)                                             ║
╚══════════════════════════════════════════════════════════════════╝
```

---

**完整公式（源码级精确）**:

```
╭──────────────────────────────────────────────────────────────────────╮
│                                                                    │
│   y[b, j, h] = Σ residual[b, i, h] · comb[b, i, j]                 │
│                i=0..C-1                                            │
│                                                                    │
│              + x[b, h] · post[b, j]                                │
│                                                                    │
╰──────────────────────────────────────────────────────────────────────╯
```

**符号说明**:

| 符号 | 含义 | 范围 |
|------|------|------|
| `b` | batch 索引 | 0 ~ B-1 |
| `i` | 输入通道索引 | 0 ~ C-1 |
| `j` | 输出通道索引 | 0 ~ C-1 |
| `h` | hidden 维度索引 | 0 ~ H-1 |

---

**矩阵形式**（对每个 batch b 和 hidden h）:

```
╭──────────────────────────────────────────────────────────────────────╮
│                                                                    │
│   y[b, :, h] = residual[b, :, h] @ comb[b, :, :]                   │
│                                                                    │
│              + post[b, :] ⊙ x[b, h]                                │
│                                                                    │
│   其中:                                                             │
│     residual[b, :, h]  ∈ R^(1×C)   ← 行向量                        │
│     comb[b, :, :]      ∈ R^(C×C)   ← 矩阵                          │
│     post[b, :]         ∈ R^C       ← 向量                          │
│     x[b, h]            ∈ R         ← 标量                          │
│     y[b, :, h]         ∈ R^C       ← 行向量                        │
│                                                                    │
╰──────────────────────────────────────────────────────────────────────╯
```

---

**大白话版本**:

```
╭──────────────────────────────────────────────────────────────────────╮
│                                                                    │
│   第 j 个输出通道                                                   │
│        = (所有输入通道按 comb 第 j 列加权求和)  ← 残差通道混合       │
│        + (主路径输出 × post 第 j 个门控)         ← 主路径扩展+门控    │
│                                                                    │
╰──────────────────────────────────────────────────────────────────────╯
```

---

**直观理解 hc_pre 和 hc_post 的关系**：

```
hc_pre (压缩):
  输入: (B, C, H) → pre_gate加权求和 → (B, H)
  同时生成: post 和 comb 给 hc_post 用

hc_post (扩展+混合):
  输入: x (B, H) + residual (B, C, H) + post (B, C) + comb (B, C, C)
  输出: residual @ comb + x * post广播 → (B, C, H)
```

关键点：
- hc_pre 用 pre_gate 把 C 个通道压缩成 1 个（做"编码"）
- hc_post 用 comb 把残差的 C 个通道重新混合（做"通道重组"）
- 主路径 x 只通过 post 门控做简单的缩放扩展
- 两者不是简单的"编解码"关系，而是更复杂的双通道结构

---

###### `npu_hc_post` 完整调用链

从 Python 调用到 NPU 硬件执行的完整路径如下：

```
┌─────────────────────────────────────────────────────────────────────┐
│  第1层: Python 调用层                                               │
│  文件: deepseek_v4.py:L969-973                                     │
│  函数: hc_post()                                                   │
│  ├── x.unsqueeze(dim=0)            # 增加batch维度 → (1, ...)      │
│  ├── residual.unsqueeze(dim=0)     # 增加batch维度                 │
│  ├── post.unsqueeze(dim=0)         # 增加batch维度                 │
│  ├── comb.unsqueeze(dim=0)         # 增加batch维度                 │
│  ├── torch.ops._C_ascend.npu_hc_post(x, residual, post, comb)      │
│  └── y.squeeze(dim=0)              # 去掉batch维度                 │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  第2层: PyTorch Dispatcher 调度层                                   │
│  算子注册: torch_binding.cpp:L2573-2581                            │
│  ops.def("npu_hc_post("                                            │
│      "Tensor x, Tensor residual, Tensor post, Tensor comb"         │
│      ") -> (Tensor out)")                                          │
│  ops.impl("npu_hc_post", kPrivateUse1, &npu_hc_post_npu)           │
│  根据输入张量的设备类型 (NPU) 调度到对应后端实现                     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  第3层: C++ NPU 入口函数                                            │
│  文件: torch_binding.cpp:L1310-1321                                │
│  函数: npu_hc_post_npu()                                           │
│  ├── check_hc_post_shape_and_dtype(x, residual, post, comb)        │
│  │   # 参数形状/类型校验 (L1274-1308)                               │
│  │   ├── 校验维度: x是3D [b,s,d], residual是4D [b,s,hc,d],         │
│  │   │           post是3D [b,s,hc], comb是4D [b,s,hc,hc]           │
│  │   ├── 校验大小: batch/sequence/hidden各维度对应一致, hc通道数一致│
│  │   └── 校验类型: x/residual=float16/bfloat16/float32             │
│  │              post/comb=float16/bfloat16/float32                 │
│  ├── construct_hc_post_output_tensor(residual)                     │
│  │   # 构造输出张量，形状同residual (L1261-1271)                   │
│  ├── EXEC_NPU_CMD(aclnnHcPost, x, residual, post, comb, out)       │
│  │   # 调用 CANN 算子库的 aclnnHcPost                              │
│  └── return out                                                    │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  第4层: CANN 算子库 + NPU 硬件层                                    │
│  内核源码: csrc/moe/hc_post/op_kernel/                             │
│  核心函数: DoMulAndAdd (hc_post_bfloat16.h:L205-309)               │
│                                                                     │
│  核心公式:                                                          │
│    y[b,j,h] = Σ residual[b,i,h] · comb[b,i,j] + x[b,h] · post[b,j]  │
│             i=0..C-1                                                │
│                                                                     │
│  计算步骤:                                                          │
│    ① 残差通道混合: residual × comb  →  C个输出通道                  │
│    ② 主路径扩展:   x 广播 × post   →  C个通道                      │
│    ③ 相加 + 类型转换 → 输出结果                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

#### 2.2.3 Forward 流程 (L975-994)

**输入输出**:
- **输入**:
  - `positions`: 位置编码
  - `hidden_states`: 输入隐藏状态，shape 为 `(num_tokens, hc_mult, H)`
  - `residual`: 残差（未使用，函数内部重新 clone）
  - `llama_4_scaling`: 缩放因子（可选）
- **输出**: `(hidden_states, residual)` 元组
  - `hidden_states`: 输出隐藏状态，shape 为 `(num_tokens, hc_mult, H)`
  - `residual`: FFN 分支前的残差，shape 为 `(num_tokens, hc_mult, H)`

---

**完整流程图与 Shape 变化**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         输入 hidden_states                          │
│                    shape: (num_tokens, hc_mult, H)                  │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    注意力分支 (Attention Branch)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  步骤 1: residual = hidden_states.clone()                           │
│          保存完整 HC 维度的残差                                      │
│          ┌───────────────────────────────────────────┐              │
│          │  residual: (num_tokens, hc_mult, H)       │              │
│          │  hidden_states: (num_tokens, hc_mult, H)  │              │
│          └───────────────────────────────────────────┘              │
│                                                                     │
│  步骤 2: hidden_states, post, comb = hc_pre(                        │
│             hidden_states, hc_attn_fn, hc_attn_scale, hc_attn_base) │
│          HC 预门控：高维特征压缩到主路径维度                           │
│          ┌───────────────────────────────────────────┐              │
│          │  输入 hidden_states: (num_tokens, hc_mult, H)            │
│          │  ───────────────────────────────────────  │              │
│          │  输出 hidden_states: (num_tokens, H)      │              │
│          │  输出 post: (num_tokens, hc_mult)         │              │
│          │  输出 comb: (num_tokens, hc_mult, hc_mult)│              │
│          └───────────────────────────────────────────┘              │
│                                                                     │
│  步骤 3: hidden_states = input_layernorm(hidden_states)             │
│          主路径归一化（RMSNorm）                                     │
│          ┌───────────────────────────────────────────┐              │
│          │  输入: (num_tokens, H)                    │              │
│          │  输出: (num_tokens, H)                    │              │
│          └───────────────────────────────────────────┘              │
│                                                                     │
│  步骤 4: hidden_states = self_attn(positions, hidden_states, ...)   │
│          自注意力计算（主路径单通道）                                 │
│          ┌───────────────────────────────────────────┐              │
│          │  输入: (num_tokens, H)                    │              │
│          │  输出: (num_tokens, H)                    │              │
│          └───────────────────────────────────────────┘              │
│                                                                     │
│  步骤 5: hidden_states = hc_post(hidden_states, residual, post, comb)│
│          HC 后门控：门控加权融合残差，恢复 HC 维度                     │
│          ┌───────────────────────────────────────────┐              │
│          │  输入 x: (num_tokens, H)                  │              │
│          │  输入 residual: (num_tokens, hc_mult, H)  │              │
│          │  输入 post: (num_tokens, hc_mult)         │              │
│          │  输入 comb: (num_tokens, hc_mult, hc_mult)│              │
│          │  ───────────────────────────────────────  │              │
│          │  输出: (num_tokens, hc_mult, H)           │              │
│          └───────────────────────────────────────────┘              │
│                                                                     │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼  hidden_states: (num_tokens, hc_mult, H)
                                     │
┌─────────────────────────────────────────────────────────────────────┐
│                       FFN 分支 (FFN Branch)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  步骤 6: residual = hidden_states.clone()                           │
│          保存完整 HC 维度的残差                                      │
│          ┌───────────────────────────────────────────┐              │
│          │  residual: (num_tokens, hc_mult, H)       │              │
│          │  hidden_states: (num_tokens, hc_mult, H)  │              │
│          └───────────────────────────────────────────┘              │
│                                                                     │
│  步骤 7: hidden_states, post, comb = hc_pre(                        │
│             hidden_states, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)    │
│          HC 预门控：高维特征压缩到主路径维度                           │
│          ┌───────────────────────────────────────────┐              │
│          │  输入 hidden_states: (num_tokens, hc_mult, H)            │
│          │  ───────────────────────────────────────  │              │
│          │  输出 hidden_states: (num_tokens, H)      │              │
│          │  输出 post: (num_tokens, hc_mult)         │              │
│          │  输出 comb: (num_tokens, hc_mult, hc_mult)│              │
│          └───────────────────────────────────────────┘              │
│                                                                     │
│  步骤 8: hidden_states = post_attention_layernorm(hidden_states)    │
│          主路径归一化（RMSNorm）                                     │
│          ┌───────────────────────────────────────────┐              │
│          │  输入: (num_tokens, H)                    │              │
│          │  输出: (num_tokens, H)                    │              │
│          └───────────────────────────────────────────┘              │
│                                                                     │
│  步骤 9: hidden_states = mlp(hidden_states)                         │
│          MoE 前馈计算（主路径单通道）                                 │
│          ┌───────────────────────────────────────────┐              │
│          │  输入: (num_tokens, H)                    │              │
│          │  输出: (num_tokens, H)                    │              │
│          └───────────────────────────────────────────┘              │
│                                                                     │
│  步骤 10: hidden_states = hc_post(hidden_states, residual, post, comb)│
│           HC 后门控：门控加权融合残差，恢复 HC 维度                    │
│           ┌──────────────────────────────────────────┐              │
│           │  输入 x: (num_tokens, H)                 │              │
│           │  输入 residual: (num_tokens, hc_mult, H) │              │
│           │  输入 post: (num_tokens, hc_mult)        │              │
│           │  输入 comb: (num_tokens, hc_mult, hc_mult)│             │
│           │  ──────────────────────────────────────  │              │
│           │  输出: (num_tokens, hc_mult, H)          │              │
│           └──────────────────────────────────────────┘              │
│                                                                     │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
                    ┌───────────────────────────────┐
                    │  返回 (hidden_states, residual)│
                    │  hidden_states: (num_tokens,  │
                    │                 hc_mult, H)   │
                    │  residual: (num_tokens,       │
                    │              hc_mult, H)      │
                    └───────────────────────────────┘
```

---

**Shape 变化汇总表**:

| 步骤 | 操作 | 输入 Shape | 输出 Shape | 说明 |
|------|------|-----------|-----------|------|
| 1 | `residual.clone()` | `(num_tokens, hc_mult, H)` | `residual: (num_tokens, hc_mult, H)` | 保存残差 |
| 2 | `hc_pre` (注意力) | `(num_tokens, hc_mult, H)` | `hidden_states: (num_tokens, H)`<br>`post: (num_tokens, hc_mult)`<br>`comb: (num_tokens, hc_mult, hc_mult)` | 压缩到主路径 |
| 3 | `input_layernorm` | `(num_tokens, H)` | `(num_tokens, H)` | RMSNorm 归一化 |
| 4 | `self_attn` | `(num_tokens, H)` | `(num_tokens, H)` | 自注意力计算 |
| 5 | `hc_post` (注意力) | `x: (num_tokens, H)`<br>`residual: (num_tokens, hc_mult, H)`<br>`post: (num_tokens, hc_mult)`<br>`comb: (num_tokens, hc_mult, hc_mult)` | `(num_tokens, hc_mult, H)` | 恢复 HC 维度 |
| 6 | `residual.clone()` | `(num_tokens, hc_mult, H)` | `residual: (num_tokens, hc_mult, H)` | 保存残差 |
| 7 | `hc_pre` (FFN) | `(num_tokens, hc_mult, H)` | `hidden_states: (num_tokens, H)`<br>`post: (num_tokens, hc_mult)`<br>`comb: (num_tokens, hc_mult, hc_mult)` | 压缩到主路径 |
| 8 | `post_attention_layernorm` | `(num_tokens, H)` | `(num_tokens, H)` | RMSNorm 归一化 |
| 9 | `mlp` (MoE) | `(num_tokens, H)` | `(num_tokens, H)` | MoE 前馈网络 |
| 10 | `hc_post` (FFN) | `x: (num_tokens, H)`<br>`residual: (num_tokens, hc_mult, H)`<br>`post: (num_tokens, hc_mult)`<br>`comb: (num_tokens, hc_mult, hc_mult)` | `(num_tokens, hc_mult, H)` | 恢复 HC 维度 |

> **符号说明**: `num_tokens` = token 数量，`hc_mult` = HC 通道倍数，`H` = hidden_size（隐藏层维度）

---

**设计要点**:
1. **计算效率**: 注意力和 FFN 仅在压缩后的主路径（单通道，shape 为 `(num_tokens, H)`）上执行，大幅减少计算量
2. **表达能力**: 通过 HC 门控机制，仍保留 `hc_mult` 个特征通道的信息容量
3. **残差连接**: 每个分支前保存完整 HC 维度的残差（`(num_tokens, hc_mult, H)`），在 `hc_post` 中门控融合
4. **返回值**: forward 返回元组 `(hidden_states, residual)` 而非仅 hidden_states（与标准 Transformer 不同）

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L904`

***

### 2.3 DeepseekV4Attention

DeepseekV4Attention 是注意力层的**外壳封装类**，负责初始化所有子模块并将 forward 调用透传给 NPU 稀疏注意力后端。

---

#### 2.3.1 整体架构与调用链

```
DeepseekV4Attention (Python 封装层)
    │
    ├── 初始化所有子模块 (wq_a, wq_b, wkv, wo_a, wo_b, ...)
    ├── 组装 DSAModules (封装子模块引用)
    ├── 创建 AscendDeepseekSparseAttention (dsa_attn)
    │       │
    │       └── DSAAttention (注意力层基类)
    │               │
    │               └── AscendDSABackend (NPU 后端)
    │                       │
    │                       └── AscendDSAImpl (核心实现)
    │
    └── forward() → 直接调用 dsa_attn → 最终由 AscendDSAImpl 执行
```

> **关键点**：DeepseekV4Attention 本身不执行任何注意力计算，所有逻辑都在 NPU 后端 `AscendDSAImpl` 中完成。

---

#### 2.3.2 初始化组件详解 (L702-893)

**Query 路径组件** (低秩分解):

| 组件 | 类型 | 权重 Shape | 输入 Shape | 输出 Shape | 说明 |
|------|------|-----------|-----------|-----------|------|
| `wq_a` | ReplicatedLinear | `(dim, q_lora_rank)` | `(num_tokens, dim)` | `(num_tokens, q_lora_rank)` | Query 低秩投影 A，数据并行复制 |
| `q_norm` | RMSNorm | `(q_lora_rank,)` | `(num_tokens, q_lora_rank)` | `(num_tokens, q_lora_rank)` | Query 归一化 |
| `q_norm_without_weight` | RMSNorm | 无权重 | `(num_tokens, n_heads, head_dim)` | `(num_tokens, n_heads, head_dim)` | Query 无权重 RMSNorm |
| `wq_b` | ColumnParallelLinear / ReplicatedLinear | `(q_lora_rank, n_heads*head_dim)` | `(num_tokens, q_lora_rank)` | `(num_tokens, n_heads*head_dim)` | Query 低秩投影 B，列并行；`enable_dsa_cp` 时用 ReplicatedLinear |

**KV 路径组件** (MLA 架构):

| 组件 | 类型 | 权重 Shape | 输入 Shape | 输出 Shape | 说明 |
|------|------|-----------|-----------|-----------|------|
| `wkv` | ReplicatedLinear | `(dim, head_dim)` | `(num_tokens, dim)` | `(num_tokens, head_dim)` | KV 联合投影 (MLA 压缩表示) |
| `kv_norm` | RMSNorm | `(head_dim,)` | `(num_tokens, head_dim)` | `(num_tokens, head_dim)` | KV 归一化 |

**输出投影组件** (低秩分解 + 分组):

| 组件 | 类型 | 权重 Shape | 输入 Shape | 输出 Shape | 说明 |
|------|------|-----------|-----------|-----------|------|
| `wo_a` | ColumnParallelLinear | `(n_heads*head_dim//n_groups, n_groups*o_lora_rank)` | `(num_tokens, n_groups, n_heads*head_dim//n_groups)` | `(num_tokens, n_groups, o_lora_rank)` | 输出投影 A (组内投影)，列并行 |
| `wo_b` | RowParallelLinear | `(n_groups*o_lora_rank, dim)` | `(num_tokens, n_groups*o_lora_rank)` | `(num_tokens, dim)` | 输出投影 B (最终输出)，行并行 |

**位置编码与注意力参数**:

| 组件 | 类型 | Shape | 说明 |
|------|------|-------|------|
| `rotary_emb` | ComplexExpRotaryEmbedding | - | 旋转位置编码，根据 `compress_ratio` 选择 rope_theta；支持多组 rope |
| `attn_sink` | nn.Parameter | `(n_heads,)` 或 `(n_local_heads,)` | 注意力 sink 参数，`enable_dsa_cp` 时形状为 `(n_heads,)`，否则为 `(n_local_heads,)` |
| `scale` | float | - | `head_dim ** -0.5`，注意力缩放因子 |

**压缩与索引组件**:

| 组件 | 类型 | 存在条件 | 说明 |
|------|------|---------|------|
| `compressor` | Compressor \| None | `compress_ratio > 1` (支持 4/128) | KV 压缩器，将 KV 压缩存储以节省显存 |
| `indexer` | Indexer \| None | `compress_ratio == 4` | 稀疏索引选择器，TopK 选择重要 token |

**缓存与后端组件**:

| 组件 | 类型 | 说明 |
|------|------|------|
| `swa_cache_layer` | AscendDeepseekV4SWACache | 滑动窗口注意力 (SWA) 缓存，始终创建 |
| `dsa_modules` | DSAModules | 数据类，封装所有子模块引用，传递给 NPU 后端 |
| `dsa_attn` | AscendDeepseekSparseAttention | NPU 稀疏注意力入口，统一封装所有 DSA 逻辑 |

---

#### 2.3.3 关键配置逻辑

**compress_ratio (按层配置)**:
- 通过 `get_dsv4_compress_ratio(config, layer_idx)` 按层获取压缩比例
- 可选值：1 (无压缩，仅 SWA)、4 (稀疏索引 + 压缩)、128 (深度压缩)
- `compress_ratio > 1` 时启用 `compressor`，`== 4` 时额外启用 `indexer`

**skip_topk (索引缓存复用)**:
- V3.2+ 配置，基于 `use_index_cache`、`index_topk_freq`、`index_topk_pattern`
- 部分层复用前一层的 TopK 索引，减少计算量
- 模式：`index_topk_pattern` (如 "FSFS"，F=计算, S=跳过) 或按频率 `index_topk_freq`

**设备相关配置**:
- A5 设备上 IndexerCache 使用 `float8_e4m3fn`，SWA Cache head_size 额外 +128 字节
- 非 A5 设备使用 `bfloat16` (SWA) 或 `int8` (Indexer)
- `DSV4_BLOCK_SIZES` 按设备类型 (A5/非A5) 和 block_size (128/64/32) 配置

---

#### 2.3.4 forward 方法 (L895-901)

**函数签名**:
```python
def forward(
    self,
    positions: torch.Tensor,           # 位置编码
    hidden_states: torch.Tensor,       # 输入: (num_tokens, H)
    llama_4_scaling: torch.Tensor | None,  # 缩放因子
) -> torch.Tensor:
```

**执行流程**:
```
hidden_states: (num_tokens, H)
        │
        ▼
self.dsa_attn(positions, hidden_states, llama_4_scaling)
        │
        ▼
输出: (num_tokens, H)
```

> **说明**：DeepseekV4Attention.forward 本身只是**透传**调用 `dsa_attn`，所有投影、位置编码、缓存管理、稀疏注意力计算都在 NPU 后端 `AscendDSAImpl` 中完成。

**文件位置**: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L702`

***

### 2.4 AscendDSABackend (NPU 稀疏注意力后端)

AscendDSABackend 是 DeepSeek V4 注意力的 **NPU 后端实现核心**，实现在 `vllm_ascend/attention/dsa_v1.py`。

---

#### 2.4.1 整体架构分层

```
┌─────────────────────────────────────────────────────────────┐
│              AscendDeepseekSparseAttention                  │
│              (dsa.py - 外层封装)                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  DSAAttention                         │  │
│  │              (layer/attention/layer.py)               │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │            AscendDSABackend                     │  │  │
│  │  │  (AttentionBackend - 后端注册/元数据构建)       │  │  │
│  │  │  ┌───────────────────────────────────────────┐  │  │  │
│  │  │  │           AscendDSAImpl                   │  │  │  │
│  │  │  │   (DSAAttentionImpl - 核心计算实现)       │  │  │  │
│  │  │  │   - _forward_prefill()                    │  │  │  │
│  │  │  │   - _forward_decode()                     │  │  │  │
│  │  │  │   - indexer_select_qli()                  │  │  │  │
│  │  │  │   - _mla_prolog_multistream()             │  │  │  │
│  │  │  └───────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

#### 2.4.2 核心组件职责

| 组件 | 所在文件 | 核心职责 |
|------|---------|---------|
| **AscendDeepseekSparseAttention** | `ops/dsa.py` | 最外层封装，通过 custom op (`dsa_forward`) 触发实际计算，支持 ACL Graph 捕获 |
| **DSAAttention** | `models/layer/attention/layer.py` | 注意力层基类，管理 KV cache spec，持有 impl 实例 |
| **AscendDSABackend** | `attention/dsa_v1.py` | 后端注册类，提供 builder、impl、KV cache shape 等静态方法 |
| **AscendDSAMetadataBuilder** | `attention/dsa_v1.py` | 元数据构建器，构建 prefill/decode 所需的所有元数据 |
| **AscendDSAImpl** | `attention/dsa_v1.py` | **核心计算实现**，包含 prefill/decode 的完整计算逻辑 |

---

#### 2.4.3 完整数据流 (Forward 全流程)

```
                        hidden_states: (num_tokens, dim)
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  阶段 1: MLA Prolog (Query + KV 投影)                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Query 路径                                                          │   │
│  │  hidden_states → wq_a → q_norm → wq_b → reshape → q_rms → rope     │   │
│  │  (num_tokens, dim)              (num_tokens, n_local_heads, head_dim) │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  KV 路径                                                             │   │
│  │  hidden_states → wkv → kv_norm → rope → SWA KV cache 写入           │   │
│  │  (num_tokens, dim)        (num_tokens, 1, head_dim)                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  阶段 2: 压缩器处理 (compress_ratio > 1 时执行)                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Compressor (KV 压缩)                                                │   │
│  │  hidden_states → wkv/wgate → 状态累积 → 压缩KV → 压缩 KV cache 写入  │   │
│  │  压缩比: 4 或 128，减少 KV 显存占用                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  阶段 3: 索引器处理 (compress_ratio == 4 时执行)                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Indexer (TopK 稀疏选择)                                             │   │
│  │  - Query 投影 (indexer.wq_b)                                        │   │
│  │  - 与压缩 KV 计算注意力得分                                          │   │
│  │  - TopK 选择最重要的 token (index_topk 个)                           │   │
│  │  - skip_topk: 复用历史索引 (IndexCache)                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  阶段 4: 稀疏注意力计算                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  NPU 算子: npu_sparse_flash_attention                               │   │
│  │  输入: Q + SWA KV + 压缩 KV + TopK 索引                             │   │
│  │  输出: attention_output                                              │   │
│  │  shape: (num_tokens, n_local_heads, head_dim)                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  阶段 5: 输出投影                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  输出路径                                                            │   │
│  │  attn_output → nope_rope(逆向) → wo_a → wo_b → 最终输出             │   │
│  │  (num_tokens, n_local_heads, head_dim)     (num_tokens, dim)        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        输出: (num_tokens, dim)
```

---

#### 2.4.4 分阶段详解

**阶段 1: MLA Prolog - Query/KV 投影**

Query 路径 (低秩分解 + MLA):

| 步骤 | 操作 | 输入 Shape | 输出 Shape | 说明 |
|------|------|-----------|-----------|------|
| 1 | `wq_a(hidden_states)` | `(T, dim)` | `(T, q_lora_rank)` | 低秩投影 A |
| 2 | `q_norm(q_a)` | `(T, q_lora_rank)` | `(T, q_lora_rank)` | RMSNorm 归一化 |
| 3 | `wq_b(qr)` | `(T, q_lora_rank)` | `(T, n_local_heads * head_dim)` | 低秩投影 B |
| 4 | `unflatten` | - | `(T, n_local_heads, head_dim)` | 重塑为多头 |
| 5 | `q_rms` (q_norm_without_weight) | `(T, n_local_heads, head_dim)` | `(T, n_local_heads, head_dim)` | 无权重 RMSNorm |
| 6 | `partial_rotary_mul` | - | `(T, n_local_heads, head_dim)` | 部分 RoPE (只对 rope_head_dim 部分) |

KV 路径 (MLA 联合投影):

| 步骤 | 操作 | 输入 Shape | 输出 Shape | 说明 |
|------|------|-----------|-----------|------|
| 1 | `wkv(hidden_states)` | `(T, dim)` | `(T, head_dim)` | KV 联合投影 (MLA) |
| 2 | `kv_norm(kv)` | `(T, head_dim)` | `(T, head_dim)` | RMSNorm 归一化 |
| 3 | `view` | - | `(T, 1, head_dim)` | 增加 head 维度 (1 个 KV head) |
| 4 | `partial_rotary_mul` | - | `(T, 1, head_dim)` | 部分 RoPE |
| 5 | `scatter` 到 SWA cache | - | - | 写入滑动窗口缓存 |

> **优化**：支持多流并行 (`multistream_dsv4_dsa_overlap`)，Query 和 KV 路径在两个流中并行执行，通过事件同步。

---

**阶段 2: Compressor - KV 压缩**

存在条件：`compress_ratio > 1` (4 或 128)

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | `compressor.wkv` + `compressor.wgate` | 两个独立线性投影，生成 KV 表示和门控值 |
| 2 | 状态累积 (state_cache) | 累积 `compress_ratio` 个 token 的状态 |
| 3 | `compressor.norm` + RoPE | 归一化和位置编码 |
| 4 | 写入压缩 KV cache | 压缩后的 KV 存入独立缓存 |

**关键参数**:
- `coff = 2` (c4, overlap=True) 或 `1` (c128): 重叠因子
- `state_dim = 2 * coff * head_dim`: 状态缓存维度 (kv_state + score_state)
- NPU 算子：`torch.ops._C_ascend.compressor()`

---

**阶段 3: Indexer - 稀疏索引选择**

存在条件：`compress_ratio == 4`

| 步骤 | 操作 | 输入 Shape | 输出 Shape | 说明 |
|------|------|-----------|-----------|------|
| 1 | `indexer.wq_b(qr)` | `(T, q_lora_rank)` | `(T, index_n_heads * index_head_dim)` | Indexer Query 投影 |
| 2 | 与压缩 KV 计算得分 | - | - | 计算 Query 与压缩 KV 的注意力得分 |
| 3 | TopK 选择 | - | `(T, index_n_heads, index_topk)` | 选择 topk 个最重要的 token |
| 4 | (可选) skip_topk 复用 | - | - | IndexCache 模式，复用前一层索引 |

**关键参数**:
- `index_n_heads`: Indexer 头数 (通常 64)
- `index_head_dim`: Indexer head 维度 (通常 128)
- `index_topk`: TopK 数量 (通常 512)
- NPU 算子：`torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata()`

---

**阶段 4: 稀疏注意力计算**

调用 NPU 算子执行稀疏注意力：

```
torch.ops._C_ascend.npu_sparse_flash_attention(
    q,                          # Query: (T, n_local_heads, head_dim)
    ori_kv=swa_kv_cache,        # 原始 KV (SWA 滑动窗口)
    cmp_kv=compress_kv_cache,   # 压缩 KV
    topk_idxs=compress_topk_idxs, # TopK 索引 (c4 时)
    sinks=attn_sink,            # 注意力 sink
    metadata=sas_metadata,      # 稀疏注意力元数据
    softmax_scale=scale,        # 缩放因子
    ...
)
```

**输出 Shape**: `(num_tokens, n_local_heads, head_dim)`

**掩码模式**:
- `ori_mask_mode=4`: SWA (滑动窗口注意力)，用于原始 KV
- `cmp_mask_mode=3`: Causal (因果注意力)，用于压缩 KV

---

**阶段 5: 输出投影**

| 步骤 | 操作 | 输入 Shape | 输出 Shape | 说明 |
|------|------|-----------|-----------|------|
| 1 | `partial_rotary_mul` (逆向) | `(T, n_local_heads, head_dim)` | `(T, n_local_heads, head_dim)` | 对 nope 部分应用逆向 RoPE |
| 2 | `view` 为分组 | - | `(T, n_local_groups, n_heads*head_dim//n_groups)` | 重塑为分组格式 |
| 3 | `wo_a(o_proj_input)` | `(T, n_local_groups, ...)` | `(T, n_local_groups, o_lora_rank)` | 组内投影 (列并行) |
| 4 | `view` flatten | - | `(T, n_local_groups * o_lora_rank)` | 展平 |
| 5 | `wo_b(o)` | `(T, n_groups*o_lora_rank)` | `(T, dim)` | 最终输出投影 (行并行) |

---

#### 2.4.5 KV 缓存体系

DeepSeek V4 使用**三类独立 KV 缓存**：

| 缓存类型 | 所属组件 | 压缩比 | 用途 |
|---------|---------|-------|------|
| **SWA KV Cache** | `swa_cache_layer` | 1x (无压缩) | 滑动窗口内的原始 KV，近期 token |
| **Compress KV Cache** | `compressor` | 4x / 128x | 压缩后的历史 KV，长程记忆 |
| **Indexer KV Cache** | `indexer.compressor` | 4x | Indexer 用的压缩 KV，用于 TopK 选择 (仅 c4) |

**A5 设备额外缓存**:
- `indexer_scale_cache`: FP8 缩放因子缓存
- `indexer_full_cache`: 完整精度缓存

---

#### 2.4.6 Prefill vs Decode 路径差异

| 维度 | Prefill (预填充) | Decode (解码) |
|------|-----------------|--------------|
| **Query 长度** | 多 token (变长) | 单 token (或少量 spec decode) |
| **KV 写入** | 写入新 token 的 KV 到所有缓存 | 仅写入当前 token |
| **索引计算** | 计算完整 TopK 索引 | 增量更新索引 |
| **注意力模式** | 变长注意力 (TND layout) | 定长解码 (BND layout) |
| **元数据** | `AscendDSAPrefillMetadata` | `AscendDSADecodeMetadata` |

---

**文件位置**:
- 后端实现: `vllm-ascend/vllm_ascend/attention/dsa_v1.py`
- 外层封装: `vllm-ascend/vllm_ascend/ops/dsa.py`
- 层基类: `vllm-ascend/vllm_ascend/models/layer/attention/layer.py`

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
