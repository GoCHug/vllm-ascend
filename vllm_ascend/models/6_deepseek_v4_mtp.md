# DeepSeek V4 MTP 多 Token 预测详解

> **本文档详细讲解 DeepSeek V4 的 MTP (Multi-Token Prediction) 多 Token 预测机制，包括模块架构、投影融合、轮询调度、延迟 HC 压缩和权重加载映射。**

---

## 一、MTP 总览

### 1.1 设计理念

MTP (Multi-Token Prediction) 是 DeepSeek V4 的推测解码 (Speculative Decoding) 机制，通过一个小型的 draft 模型（MTP 模块）同时预测多个未来 token，再由主模型验证，从而提升解码吞吐量。

```
主模型输出第 t 步 hidden_state
    │
    ├─ 存入 _mtp_hidden_buffer (预 hc_head 状态)
    │
    └─ MTP 模块:
        ├─ Step 1: 预测第 t+1 个 token
        ├─ Step 2: 预测第 t+2 个 token
        └─ ... (最多 num_nextn_predict_layers 步)
```

**核心思想**：用少量的额外计算（小 draft 模型）换取多步解码的并行验证，减少主模型的串行调用次数。

### 1.2 模块层级

**文件位置**: `vllm_ascend/models/deepseek_v4_mtp.py`

```
DeepSeekV4MTP (入口类，L204)
└── model: DeepSeekMultiTokenPredictor (L143)
    ├── embed_tokens: VocabParallelEmbedding          — 词嵌入
    ├── logits_processor: LogitsProcessor              — Logits 处理器
    └── layers: ModuleDict[str, DeepSeekMultiTokenPredictorLayer] (L59)
        │   (key: "0", "1", ..., 按 step_idx 轮询)
        └── 每个 layer:
            ├── e_proj: ReplicatedLinear               — 嵌入投影
            ├── h_proj: ReplicatedLinear               — 隐藏状态投影
            ├── enorm: RMSNorm                          — 嵌入归一化
            ├── hnorm: RMSNorm                          — 隐藏状态归一化
            ├── shared_head: SharedHead (L39)           — 共享 LM Head
            │   ├── norm: RMSNorm
            │   └── head: ParallelLMHead
            ├── mtp_block: DeepseekV2DecoderLayer       — Draft 层 Transformer Block
            │   └── is_draft_layer=True
            └── hc_head: {fn, base, scale}              — HC 头参数
```

### 1.3 与主模型的关系

| 交互点 | 说明 |
|--------|------|
| `_mtp_hidden_buffer` | 主模型 forward 中保存预 hc_head 的 HC 维度隐藏状态，MTP 用此作为输入起点 |
| `topk_indices_buffer` | 模型级共享 TopK 索引缓冲区，主模型和 MTP 共用 |
| `get_mtp_target_hidden_states()` | MTP 从主模型获取目标隐藏状态（训练用） |
| `spec_step_idx` | 当前推测步数，用于选择 MTP 层和 head |

---

## 二、DeepSeekV4MTP — 入口类

**文件位置**: `vllm_ascend/models/deepseek_v4_mtp.py:L204`

### 2.1 核心功能

- 对外提供 MTP 模块的统一接口
- 实现 `SupportsPP` 接口（流水线并行支持）
- 实现 `DeepseekV2MixtureOfExperts` 接口（MoE 管理）
- 管理权重加载与映射

### 2.2 set_moe_parameters

初始化 MoE 相关参数：

1. 遍历所有 MTP 层的 `mtp_block`
2. 收集所有 MoE 层（`DeepseekV4MoE`）
3. 提取 MoE 参数映射
4. 设置 `num_expert_groups` 等

**注意**: 只收集 `DeepseekV4MoE` 类型的层，`PPMissingLayer` 跳过。

### 2.3 forward 接口

```python
def forward(self, input_ids, positions, hidden_states,
            intermediate_tensors=None, inputs_embeds=None,
            spec_step_idx=0):
    return self.model(input_ids, positions, hidden_states,
                      inputs_embeds, spec_step_idx)
```

直接委托给内部 `DeepSeekMultiTokenPredictor`。

### 2.4 compute_logits 接口

```python
def compute_logits(self, hidden_states, spec_step_idx=0):
    return self.model.compute_logits(hidden_states, spec_step_idx)
```

延迟计算 logits，仅在需要时调用。

---

## 三、DeepSeekMultiTokenPredictor — 预测器主体

**文件位置**: `vllm_ascend/models/deepseek_v4_mtp.py:L143`

### 3.1 核心成员

| 成员 | 类型 | 说明 |
|------|------|------|
| `mtp_start_layer_idx` | int | MTP 层起始索引 = `config.num_hidden_layers` |
| `num_mtp_layers` | int | MTP 层数 = `config.num_nextn_predict_layers` (默认 1) |
| `layers` | ModuleDict[str, Layer] | MTP 层字典，key 为字符串索引 "0", "1", ... |
| `embed_tokens` | VocabParallelEmbedding | 词嵌入层 |
| `logits_processor` | LogitsProcessor | Logits 处理器 |

### 3.2 轮询调度机制

MTP 层按 `spec_step_idx % num_mtp_layers` 轮询使用：

```python
current_step_idx = spec_step_idx % self.num_mtp_layers
return self.layers[str(current_step_idx)](...)
```

**设计意图**：
- 当推测步数 > MTP 层数时，循环复用层
- 不同步数共享部分参数，减少参数量
- 每层学习不同步数的特征

### 3.3 embed_input_ids

```python
def embed_input_ids(self, input_ids):
    return self.embed_tokens(input_ids)
```

将 token ID 转换为嵌入向量。

---

## 四、DeepSeekMultiTokenPredictorLayer — 单层实现

**文件位置**: `vllm_ascend/models/deepseek_v4_mtp.py:L59`

### 4.1 核心成员

| 成员 | 类型 | 形状 / 维度 | 说明 |
|------|------|------------|------|
| `e_proj` | ReplicatedLinear | `hidden_size → hidden_size` | 输入嵌入投影 |
| `h_proj` | ReplicatedLinear | `hidden_size → hidden_size` | 隐藏状态投影 |
| `enorm` | RMSNorm | `hidden_size` | 嵌入归一化 |
| `hnorm` | RMSNorm | `hidden_size` | 隐藏状态归一化 |
| `shared_head` | SharedHead | — | 共享 LM Head |
| `mtp_block` | DeepseekV2DecoderLayer | — | Draft 层 Transformer Block |
| `hc_eps` | float | — | HC epsilon |
| `hc_mult` | int | — | HC 扩展倍数 |
| `hc_head_fn` | Parameter | `(hc_mult, hc_mult * hidden_size)` | HC 头投影权重 |
| `hc_head_base` | Parameter | `(hc_mult,)` | HC 头偏置 |
| `hc_head_scale` | Parameter | `(1,)` | HC 头缩放 |
| `norm_eps` | float | — | RMSNorm epsilon |
| `is_v32` | bool | — | 是否 V3.2+ 模型 (有 index_topk) |

### 4.2 投影融合机制

MTP 层的输入由两部分融合而成：
1. **当前 token 嵌入** (`inputs_embeds`): 预测目标 token 的嵌入
2. **上一步隐藏状态** (`previous_hidden_states`): 主模型或上一步 MTP 的输出

```
inputs_embeds (num_tokens, hidden_size)
    │
    ├─ enorm (RMSNorm)
    └─ e_proj (Linear) → e_proj_out (num_tokens, hidden_size)
                                        │
                                        ├─ unsqueeze(-2) → (T, 1, hidden_size)
previous_hidden_states (T, hc_mult, hidden_size)
    │                                   │ 广播相加
    ├─ hnorm (RMSNorm)                  ▼
    └─ h_proj (Linear) → h_proj_out (num_tokens, hc_mult, hidden_size)
                                        │
                                        ▼
                            hidden_states (T, hc_mult, hidden_size)
```

**代码实现** (`L124`)：
```python
hidden_states = self.e_proj(inputs_embeds).unsqueeze(-2) + self.h_proj(previous_hidden_states)
```

> **设计要点**:
> - `e_proj` 输出 shape: `(T, hidden_size)` → unsqueeze 后 `(T, 1, hidden_size)`
> - `h_proj` 输出 shape: `(T, hc_mult, hidden_size)`
> - 利用广播机制相加，每个 HC 通道都加上相同的嵌入投影
> - 相当于用输入嵌入"调制"所有 HC 通道

### 4.3 Position 0 Mask

**文件位置**: `vllm_ascend/models/deepseek_v4_mtp.py:L119`

```python
# masking inputs at position 0, as not needed by MTP
inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
```

将 `position == 0` 的 token 嵌入置零。

**设计原因**: position 0 是序列起始位置，MTP 从第一步开始预测，起始位置的嵌入无意义（没有前驱 token 需要预测），置零避免干扰。

### 4.4 mtp_block — Draft Transformer Block

`mtp_block` 是一个完整的 `DeepseekV2DecoderLayer`，但 `is_draft_layer=True`：

- 结构与主模型层相同（HC + 注意力 + MoE/MLP）
- `is_draft_layer=True` 影响：
  - 不使用 hash 路由（始终用线性路由）
  - 可能有其他 draft 特定的简化

**输入输出**:
- 输入: `positions`, `hidden_states (T, hc_mult, hidden_size)`, `residual=None`
- 输出: `(hidden_states, residual)` — 与主层一致

### 4.5 延迟 HC 压缩

**重要设计**: MTP 层的 forward **不做** hc_head 压缩，返回完整 HC 维度的隐藏状态：

```python
# 代码中被注释掉，forward 不调用 hc_head
# hidden_states = self.hc_head(hidden_states, self.hc_head_fn,
#                              self.hc_head_scale, self.hc_head_base)
return hidden_states  # shape: (T, hc_mult, hidden_size)
```

**原因**：
1. 下一步 MTP 层需要完整的 HC 维度作为输入
2. logits 计算只在最后一步需要，延迟到 `compute_logits` 时才压缩
3. 避免多步预测中反复压缩/扩展的开销

**hc_head 实现**（与主模型一致）：
```python
def hc_head(self, x, hc_fn, hc_scale, hc_base):
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)
```

---

## 五、SharedHead — 共享 LM Head

**文件位置**: `vllm_ascend/models/deepseek_v4_mtp.py:L39`

### 5.1 结构

```
SharedHead
├── norm: RMSNorm (hidden_size)
└── head: ParallelLMHead (vocab_size, hidden_size)
```

### 5.2 共享机制

所有 MTP 步共享同一个 `shared_head`：

- 减少参数量
- 所有步数的输出空间一致
- 符合推测解码的假设（验证 token 分布相同）

### 5.3 forward

```python
def forward(self, hidden_states):
    return self.norm(hidden_states)
```

> 注意: `forward` 只返回 norm 后的 hidden_states，实际 logits 计算由 `logits_processor` + `head.weight` 完成。

---

## 六、完整 Forward 数据流

### 6.1 单步流程

```
输入:
  input_ids: (num_tokens,)
  positions: (num_tokens,)
  previous_hidden_states: (num_tokens, hc_mult, hidden_size)
  inputs_embeds: (num_tokens, hidden_size) [可选，无则用 embed_tokens]
  spec_step_idx: int

步骤:
  1. Mask position 0 的嵌入
     inputs_embeds = where(positions==0, 0, inputs_embeds)

  2. 嵌入归一化 + 投影
     e_proj_out = e_proj(enorm(inputs_embeds))  → (T, H)
     h_proj_out = h_proj(hnorm(prev_hidden))    → (T, hc_mult, H)

  3. 投影融合
     hidden = e_proj_out.unsqueeze(-2) + h_proj_out
                                                → (T, hc_mult, H)

  4. Draft Transformer Block
     hidden, residual = mtp_block(positions, hidden, residual=None)
                                                → (T, hc_mult, H)

输出:
  hidden_states: (num_tokens, hc_mult, hidden_size)
  (不做 hc_head 压缩，保持 HC 维度)
```

### 6.2 compute_logits 流程

```
输入:
  hidden_states: (num_tokens, hc_mult, hidden_size)
  spec_step_idx: int

步骤:
  1. 选择对应 step 的 MTP 层
     layer = layers[spec_step_idx % num_mtp_layers]

  2. HC 头压缩
     hidden = layer.hc_head(hidden, hc_fn, hc_scale, hc_base)
                                              → (T, hidden_size)

  3. Shared Head 归一化
     hidden = layer.shared_head(hidden)      → (T, hidden_size)

  4. Logits 计算
     logits = logits_processor(head, hidden) → (T, vocab_size)

输出:
  logits: (num_tokens, vocab_size)
```

---

## 七、权重加载与映射

**文件位置**: `vllm_ascend/models/deepseek_v4_mtp.py:L255`

### 7.1 权重名映射规则

MTP 的权重名与主模型命名方式不同，需要一系列转换：

| 原始权重名片段 | 映射后 | 说明 |
|-------------|-------|------|
| `mtp.0.` | `model.` 或 `model.layers.0.` | 前缀替换 |
| `.emb.tok_emb.` | `.embed_tokens.` | 嵌入层命名 |
| `.w1.` | `.gate_proj.` | FFN gate 投影 |
| `.w2.` | `.down_proj.` | FFN down 投影 |
| `.w3.` | `.up_proj.` | FFN up 投影 |
| `.head.` | `.shared_head.head.` | LM Head |
| `.norm.` | `.shared_head.norm.` | 输出 norm |
| `.attn.` (非 self_attn) | `.self_attn.` | 注意力命名 |
| `.ffn.` | `.mlp.` | FFN 命名 |
| `.ffn_norm.` | `.post_attention_layernorm.` | FFN 前置 norm |
| `.attn_norm.` | `.input_layernorm.` | 注意力前置 norm |
| `.gate.bias` | `.gate.e_score_correction_bias` | 门控偏置 |
| `.scale` | `.weight_scale` | 量化缩放 |

### 7.2 命名分类

`no_mtp_block_in_name` 判断哪些权重不需要加 `.mtp_block.` 前缀：

```python
names = [
    ".hc_head_fn", ".hc_head_base", ".hc_head_scale",
    ".e_proj.", ".h_proj.", ".enorm.", ".hnorm.",
    ".norm.", ".head.", ".emb.tok_emb.",
]
```

这些是 MTP 层特有的组件（投影层、hc_head、shared_head 等），不在 mtp_block 内部。

### 7.3 Attention Sink 处理

对于 `attn_sink` 参数：

- `enable_dsa_cp()`: 直接复制（上下文并行，全头数）
- 否则: 按 TP rank 切片，只加载本地 rank 对应的头

### 7.4 融合共享专家

当 `mix_placement` 启用时，共享专家权重融合到 FusedMoE 中：

1. 检测 `mlp.shared_experts` 权重
2. 按 `n_shared_experts` 分片
3. 重命名为 `mlp.experts.{n_routed_experts + j}.` 格式
4. 通过 expert_params_mapping 加载到对应位置

**split_dim 规则**:
- `down_proj.weight`: dim 1 切分
- 其他 (gate/up_proj): dim 0 切分

---

## 八、训练目标获取

### 8.1 get_mtp_target_hidden_states

MTP 从主模型获取目标隐藏状态用于训练：

- 主模型每层输出的隐藏状态作为对应 MTP 步的训练目标
- 实现了多步预测的监督学习

> 具体实现依赖主模型的 `_mtp_hidden_buffer` 机制。
