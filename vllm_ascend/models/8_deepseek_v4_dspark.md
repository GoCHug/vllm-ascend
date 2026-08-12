# DeepSeek V4 DSpark 稀疏块草稿模型详解

> **本文档详细讲解 DeepSeek V4 的 DSpark (DeepSeek Sparse Block Drafter) 稀疏块草稿模型机制，包括 Markov 头、上下文 KV 预计算、HC 头压缩、多隐藏状态融合和权重加载映射。**

---

## 一、DSpark 总览 — 入门篇

### 1.1 DSpark 是什么？（生活化类比）

**MTP vs DSpark 的类比**：
- **MTP（多Token预测）** 像一个**逐字猜的助手**：你说一个字，它猜下一个字，你验证对不对，再猜下一个……串行工作。
- **DSpark（稀疏块草稿）** 像一个**一次性猜一整句话的助手**：它先看你之前写的好几段内容（主模型多层隐藏状态），然后一口气写出一整段草稿（一个 block），你再批量验证。

> **核心差异**：DSpark 不是普通的串行 MTP 模块，而是一个**块草稿器 (block drafter)**。它接收主模型多个指定层的隐藏状态，一次性生成一个完整的 draft block（多个 token），实现更激进的推测解码。

### 1.2 设计理念

DSpark 权重存储在目标模型 checkpoint 的 `mtp.*` 命名空间下，但它的实现路径与普通 MTP 完全不同：

```
主模型 forward 过程中
    │
    ├─ 选定 target_layer_ids 层（例如第 N-3, N-2, N-1 层）
    │   收集这些层的 aux_hidden_states
    │
    └─ DSpark 模块:
        ├─ Step 0: combine_hidden_states → 融合多层隐藏状态
        │           (main_proj 投影 + main_norm 归一化)
        ├─ Step 1: precompute_and_store_context_kv → 预计算并存储上下文 KV
        ├─ Step 2: 多层 Draft Transformer Block 处理
        ├─ Step 3: hc_head 压缩 HC 维度
        ├─ Step 4: markov_head 提供 n-gram 偏置
        └─ 输出: 完整 draft block 的 logits
```

### 1.3 模块层级

**文件位置**: `vllm_ascend/models/deepseek_v4_dspark.py`

```
DSparkDeepseekV4ForCausalLM (入口类，L271)
├── lm_head: ParallelLMHead                              — LM 输出头
├── logits_processor: LogitsProcessor                    — Logits 处理器
└── model: DeepseekV4DSparkModel (L83)                   — DSpark 主体
    ├── embed_tokens: VocabParallelEmbedding             — 词嵌入
    ├── main_proj: ReplicatedLinear                      — 多层隐藏状态融合投影
    │   形状: (hidden_size * len(target_layer_ids)) → hidden_size
    ├── main_norm: RMSNorm                               — 融合后归一化
    ├── norm: RMSNorm                                    — 最终输出归一化
    ├── markov_head: DSparkMarkovHead (L62)              — Markov n-gram 头
    │   ├── markov_w1: VocabParallelEmbedding            —   Markov 嵌入
    │   ├── markov_w2: ParallelLMHead                    —   Markov 投影
    │   └── logits_processor: LogitsProcessor            —   Markov logits
    ├── hc_head_fn: Parameter (hc_mult, hc_mult*H)      — HC 头投影权重
    ├── hc_head_base: Parameter (hc_mult,)               — HC 头偏置
    ├── hc_head_scale: Parameter (1,)                    — HC 头缩放
    └── layers: ModuleDict[str, DeepseekV2DecoderLayer]  — Draft Transformer 层
        │   key 从 mtp_start_layer_idx = num_hidden_layers 开始
        │   共 num_dspark_layers 层（默认3层）
        └── 第一层额外挂接: main_proj, main_norm
            最后一层额外挂接: norm, markov_head, hc_head_*
```

### 1.4 与主模型的关系

| 交互点 | 说明 |
|--------|------|
| `aux_hidden_states` | 主模型在指定 `target_layer_ids` 层收集隐藏状态，供 DSpark 融合 |
| `context_states / context_positions` | Prefill 阶段的上下文隐藏状态，用于预计算 KV 缓存 |
| `DeepseekV2DecoderLayer` | DSpark 复用主模型的 Transformer 层结构（注意力+MoE+HC） |
| `topk_indices_buffer` | 与主模型共享 TopK 索引缓冲区 |
| `mtp.*` 命名空间 | DSpark 权重存储在 checkpoint 的 mtp 前缀下，但做了大量重映射 |

---

## 二、工具函数与常量 — 基础篇

**文件位置**: `vllm_ascend/models/deepseek_v4_dspark.py:L1-L82`

### 2.1 _EXPERT_SCALE_RE — 专家缩放正则

```python
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.scale$")
```

**用途**: 匹配 MoE 专家权重中的缩放因子参数名。
- 匹配形如 `layers.X.mlp.experts.N.gate_proj.scale` 的字符串
- 用于区分专家量化缩放和普通权重量化缩放（两者后缀不同）

### 2.2 _apply_dsv4_rope — DSV4 RoPE 位置编码应用

```python
def _apply_dsv4_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,      # (num_tokens,) 位置索引
    x: torch.Tensor,              # (..., rope_head_dim) 待旋转张量
    *,
    inverse: bool = False,        # 是否逆向旋转（用于解旋转）
) -> torch.Tensor:
```

**处理流程**：
1. 调用 `get_cos_and_sin_dsa(positions)` 获取 DSA 专用的 cos/sin 表
2. 根据 `rotary_emb.layername` 选择对应层的 cos/sin
3. 如果 `inverse=True`，将 sin 取负（逆向旋转）
4. 调用 `rotary_emb(x, cos_t, sin_t)` 应用旋转位置编码

> **为什么需要 layername？** DSA（DeepSeek Sparse Attention）中不同压缩率的层使用不同的 RoPE 参数（rope_theta 不同），因此需要按层名获取对应的 cos/sin。

### 2.3 _get_dspark_num_mtp_layers — 获取 DSpark 层数

```python
def _get_dspark_num_mtp_layers(config: PretrainedConfig) -> int:
    num_layers = getattr(config, "n_mtp_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "dspark_num_mtp_layers", 3)
    return int(num_layers or 3)
```

**优先级**：
1. 首先尝试 `config.n_mtp_layers`（兼容通用 MTP 配置）
2. 其次尝试 `config.dspark_num_mtp_layers`（DSpark 专用配置）
3. 默认值为 **3** 层

---

## 三、DSparkMarkovHead — Markov N-gram 偏置头

**文件位置**: `vllm_ascend/models/deepseek_v4_dspark.py:L62-L80`

### 3.1 是什么？干什么用？

**生活化类比**：Markov 头像一个"**语言习惯数据库**"。它知道"我吃饭"后面大概率接"了"，"今天天"后面大概率接"气"。它利用马尔可夫链（n-gram 统计）的思想，给常见的词对组合一个加分偏置，让 draft 预测更符合语言习惯。

### 3.2 结构

| 成员 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `markov_w1` | VocabParallelEmbedding | `(vocab_size, dspark_markov_rank)` | 将前一个 token ID 嵌入为 Markov 向量 |
| `markov_w2` | ParallelLMHead | `(vocab_size, dspark_markov_rank)` | 将 Markov 向量投影回词表空间，作为 bias |
| `logits_processor` | LogitsProcessor | — | Logits 后处理（如温度、采样等） |

### 3.3 embed 方法

```python
def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
    """
    将前一个 token ID 转换为 Markov 嵌入向量。
    
    参数:
        token_ids: (num_tokens,) 前一个 token 的 ID
    返回:
        markov_embed: (num_tokens, dspark_markov_rank) Markov 嵌入
    """
    return self.markov_w1(token_ids)
```

### 3.4 bias 方法

```python
def bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
    """
    将 Markov 嵌入转换为词表偏置（加到 logits 上）。
    
    参数:
        markov_embed: (num_tokens, dspark_markov_rank) Markov 嵌入
    返回:
        markov_bias: (num_tokens, vocab_size) 每个词的 n-gram 偏置值
    """
    return self.logits_processor(self.markov_w2, markov_embed)
```

> **工作原理**：相当于一个双线性 n-gram 模型 `P(next|prev) = softmax(W2 · W1[prev])`，它直接建模相邻 token 的转移概率，作为主 logits 的加性偏置。

---

## 四、DeepseekV4DSparkModel — DSpark 主体模型

**文件位置**: `vllm_ascend/models/deepseek_v4_dspark.py:L83-L269`

### 4.1 核心配置参数

| 参数 | 来源 | 默认/说明 |
|------|------|----------|
| `config` | `speculative_config.draft_model_config.hf_config` | Draft 模型专用配置 |
| `hc_mult` | `config.hc_mult` | HC (Head Channels) 多通道扩展倍数 |
| `hidden_size` | `config.hidden_size` | 隐藏层维度 |
| `block_size` | `config.dspark_block_size` | DSpark 一次生成的 block 大小（token 数） |
| `target_layer_ids` | `config.dspark_target_layer_ids` | 主模型中用于融合的目标层索引列表 |
| `num_dspark_layers` | 由 `_get_dspark_num_mtp_layers` 获取 | DSpark Transformer 层数（默认3） |
| `mtp_start_layer_idx` | `config.num_hidden_layers` | DSpark 层的起始索引（主模型层数之后） |

### 4.2 __init__ 初始化详解

#### 4.2.1 词嵌入层

```python
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,
    quant_config=vllm_config.quant_config,
    prefix=maybe_prefix(prefix, "embed_tokens"),
)
```
**形状**: `(vocab_size, hidden_size)` — 将 token ID 映射为隐藏向量。

#### 4.2.2 Draft Transformer 层

```python
self.layers = nn.ModuleDict({
    str(self.mtp_start_layer_idx + idx): DeepseekV2DecoderLayer(
        vllm_config,
        prefix=f"mtp.{idx}",
        is_draft_layer=True,  # 关键标记：这是 draft 层
    )
    for idx in range(self.num_dspark_layers)
})
```

**关键点**：
- 层的 key 不是从 0 开始，而是从 `mtp_start_layer_idx = num_hidden_layers` 开始（延续主模型的层编号）
- 所有层都设置 `is_draft_layer=True`，这会影响 MoE 路由（不使用 hash 路由）

#### 4.2.3 多层隐藏状态融合模块

```python
first_layer = self.layers[str(self.mtp_start_layer_idx)]
self.main_proj = ReplicatedLinear(
    config.hidden_size * len(self.target_layer_ids),  # 输入: 多层拼接
    config.hidden_size,                                # 输出: 统一维度
    bias=False,
    return_bias=False,
    quant_config=None,
    prefix=...
)
self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
first_layer.main_proj = self.main_proj  # 挂接到第一层
first_layer.main_norm = self.main_norm
```

**生活化类比**：这就像你写论文时，不会只看最后一段来猜下一段，而是会同时看倒数第3、2、1段，综合理解上下文。`main_proj` 就是把这几段的信息"揉在一起"的投影。

**形状说明**：
- 输入: `(num_tokens, hidden_size * num_target_layers)` — 多层隐藏状态拼接
- 输出: `(num_tokens, hidden_size)` — 融合后的统一表示

#### 4.2.4 最终输出模块

```python
self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
last_layer_idx = self.mtp_start_layer_idx + self.num_dspark_layers - 1

# Markov 头
self.markov_head = DSparkMarkovHead(config, prefix=...)

# HC 头参数
hc_dim = self.hc_mult * config.hidden_size
self.hc_head_fn = nn.Parameter(
    torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
    requires_grad=False,
)
self.hc_head_base = nn.Parameter(
    torch.empty(self.hc_mult, dtype=torch.float32),
    requires_grad=False,
)
self.hc_head_scale = nn.Parameter(
    torch.empty(1, dtype=torch.float32),
    requires_grad=False,
)

# 挂接到最后一层
last_layer = self.layers[str(last_layer_idx)]
last_layer.norm = self.norm
last_layer.markov_head = self.markov_head
last_layer.hc_head_fn = self.hc_head_fn
last_layer.hc_head_base = self.hc_head_base
last_layer.hc_head_scale = self.hc_head_scale
```

> **设计细节**: `main_proj/main_norm` 挂接在第一层，`norm/markov_head/hc_head_*` 挂接在最后一层。这是一种参数共享的设计，让 Transformer 层可以直接访问这些模块。

### 4.3 get_draft_kv_cache_layer_names — 获取 KV 缓存层名

```python
def get_draft_kv_cache_layer_names(self) -> list[str]:
    return [layer.self_attn.dsa_attn.swa_cache_layer.prefix 
            for layer in self.layers.values()]
```

**用途**：返回所有 draft 层的 SWA（Sliding Window Attention）KV 缓存前缀名，用于 KV 缓存内存分配。

### 4.4 combine_hidden_states — 多层隐藏状态融合

```python
def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
    """
    融合主模型多层隐藏状态。
    
    参数:
        aux_hidden_states: (num_tokens, hidden_size * num_target_layers)
                          主模型各目标层隐藏状态的拼接
    返回:
        combined: (num_tokens, hidden_size) 融合归一化后的隐藏状态
    """
    return self.main_norm(self.main_proj(aux_hidden_states))
```

**数据流**：
```
aux_hidden_states (T, H*N)     N = len(target_layer_ids)
    │
    └─ main_proj (Linear) → (T, H)
        │
        └─ main_norm (RMSNorm) → (T, H)
```

### 4.5 _project_shared_kv — 投影共享 KV

```python
def _project_shared_kv(
    self,
    hidden_states: torch.Tensor,    # (num_tokens, hidden_size)
    positions: torch.Tensor,        # (num_tokens,)
    attn: nn.Module | None = None,  # 注意力模块
) -> torch.Tensor:
```

**处理流程**：
1. `kv = attn.kv_norm(attn.wkv(hidden_states))` — 通过 wkv 投影并归一化
   - 形状: `(num_tokens, head_dim)`
2. 拆分 nope（非旋转）和 pe（位置编码）部分：
   ```python
   k_nope, k_pe = kv.split([attn.nope_head_dim, attn.rope_head_dim], dim=-1)
   ```
   - `k_nope`: `(num_tokens, nope_head_dim)`
   - `k_pe`: `(num_tokens, rope_head_dim)`
3. 对 `k_pe` 应用 RoPE：
   ```python
   k_pe = _apply_dsv4_rope(attn.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
   ```
4. 拼接并调整形状：
   ```python
   return torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, attn.head_dim).contiguous()
   ```
   - 输出形状: `(num_tokens, 1, head_dim)`

> **为什么叫 shared_kv？** 这部分 KV 是上下文预计算的，可以在多个 draft 步之间共享。

### 4.6 _store_standard_swa_kv — 存储 SWA KV 到缓存

```python
def _store_standard_swa_kv(
    self,
    shared_kv: torch.Tensor,                    # (num_tokens, 1, head_dim)
    slot_mapping: torch.Tensor | None,          # KV 槽位映射
    attn: nn.Module | None = None,
) -> None:
```

**处理流程**：
1. 边界检查：`slot_mapping` 为空则直接返回
2. 获取 SWA 缓存层：`swa_cache_layer = attn.dsa_attn.swa_cache_layer`
3. 获取 KV cache 张量：`swa_kv_cache = swa_cache_layer.kv_cache`
4. 如果是 1D slot_mapping，格式化为 DSA 所需格式：
   ```python
   slot_mapping = DeviceOperator.format_dsa_slot_mapping(
       slot_mapping, swa_cache_layer.block_size
   )
   ```
5. 调用设备算子压缩并散射存储：
   ```python
   DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, shared_kv, slot_mapping)
   ```

### 4.7 precompute_and_store_context_kv — 预计算上下文 KV（关键方法）

```python
def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,                     # 上下文隐藏状态
    context_positions: torch.Tensor,                  # 上下文位置
    context_slot_mapping: list[torch.Tensor|None]|None = None,  # 每层的槽位映射
) -> None:
```

**这是什么？** Prefill 阶段结束后，在生成 draft token 之前，DSpark 需要把上下文（prompt 部分）的 KV 预先计算好并存入缓存，这样后续 draft 生成时就不用重复计算这部分 KV 了。

**生活化类比**：考试前先把课本知识点整理成小抄（预计算 KV），考试时直接看小抄（查缓存）而不用重新翻书（重新计算）。

**处理流程**：
```python
if context_states.numel() == 0 or context_slot_mapping is None:
    return

for layer_idx, layer in enumerate(self.layers.values()):
    # 获取该层对应的 slot_mapping
    layer_slot_mapping = context_slot_mapping[layer_idx] if context_slot_mapping else None
    
    attn = layer.self_attn
    
    # Step 1: 投影出 KV
    shared_kv = self._project_shared_kv(context_states, context_positions, attn)
    
    # Step 2: 存储到 SWA KV 缓存
    self._store_standard_swa_kv(shared_kv, layer_slot_mapping, attn)
```

> **逐层处理**：每个 draft 层都有独立的 KV 缓存，需要逐层计算和存储。

### 4.8 forward — 前向推理

```python
def forward(
    self,
    input_ids: torch.Tensor,     # (num_tokens,) 输入 token ID
    positions: torch.Tensor,     # (num_tokens,) 位置编码
) -> torch.Tensor:
```

**数据流详解**：

```
Step 1: 词嵌入 + HC 维度扩展
    hidden_states = embed_tokens(input_ids)
                   → (num_tokens, hidden_size)
    hidden_states = hidden_states.unsqueeze(-2).repeat(1, hc_mult, 1)
                   → (num_tokens, hc_mult, hidden_size)
    
    注：这与主模型的处理一致，将单通道扩展为 hc_mult 个 HC 通道
    
Step 2: 初始化 residual
    residual = None

Step 3: 逐层通过 Draft Transformer Block
    for layer in self.layers.values():
        hidden_states, residual = layer(
            positions, hidden_states, residual, llama_4_scaling=None
        )
    # 每层内部流程（DeepseekV2DecoderLayer）：
    #   a. hc_pre: HC 预处理（Sinkhorn 迭代等）
    #   b. input_layernorm + self_attn: 自注意力
    #   c. hc_post: HC 后处理
    #   d. hc_pre + post_attention_layernorm + mlp: FFN/MoE
    #   e. hc_post
    # 输出保持 (num_tokens, hc_mult, hidden_size)

Step 4: HC 头压缩（将 hc_mult 通道压缩回单通道）
    head_hidden = self.hc_head(hidden_states, hc_head_fn, hc_head_scale, hc_head_base)
                 → (num_tokens, hidden_size)

返回: head_hidden
```

### 4.9 hc_head — HC 多通道头压缩

```python
def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, 
            hc_scale: torch.Tensor, hc_base: torch.Tensor):
    """
    将 HC 多通道表示压缩回单通道。
    
    参数:
        x: (num_tokens, hc_mult, hidden_size) 多通道隐藏状态
        hc_fn: (hc_mult, hc_mult * hidden_size) 混合权重
        hc_scale: (1,) 缩放因子
        hc_base: (hc_mult,) 偏置
    
    返回:
        y: (num_tokens, hidden_size) 压缩后的单通道隐藏状态
    """
```

**算法详解**：

```python
shape, dtype = x.size(), x.dtype  # (T, c, H), dtype

# Step 1: 展平 + 转 FP32 计算
x = x.flatten(1).float()  # → (T, c*H)

# Step 2: RMSNorm 风格归一化
rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
# rsqrt: (T, 1) = 1/sqrt(mean(x^2) + eps)

# Step 3: 线性变换得到混合系数
mixes = F.linear(x, hc_fn) * rsqrt
# mixes: (T, c) — 每个 HC 通道的混合权重

# Step 4: Sigmoid 门控 + epsilon
pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
# pre: (T, c) — 归一化后的通道权重（sigmoid 确保 0~1，加 eps 避免全零）

# Step 5: 加权求和
y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
# x.view(shape): (T, c, H)
# pre.unsqueeze(-1): (T, c, 1)
# 相乘: (T, c, H)
# sum(dim=1): (T, H)

return y.to(dtype)  # 转回原 dtype
```

**原理理解**：
- HC (Head Channels) 类似于"多视角"机制：模型维护 `hc_mult` 个不同"视角"的隐藏状态
- `hc_head` 学习如何动态加权融合这些视角：
  - `hc_fn` 决定混合模式
  - `hc_scale` 和 `hc_base` 控制 sigmoid 门控的温度和偏移
  - 最终得到一个统一的表示

### 4.10 markov_embed / markov_bias — Markov 接口

```python
def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
    """token_ids → Markov 嵌入"""
    return self.markov_head.embed(token_ids)

def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
    """Markov 嵌入 → 词表偏置"""
    return self.markov_head.bias(markov_embed)
```

### 4.11 compute_logits — 计算输出 logits

```python
def compute_logits(
    self,
    hidden_states: torch.Tensor,    # (num_tokens, hidden_size)
    lm_head: ParallelLMHead,
    logits_processor: LogitsProcessor,
) -> torch.Tensor:
    """
    从隐藏状态计算词表 logits。
    
    返回:
        logits: (num_tokens, vocab_size)
    """
    return logits_processor(lm_head, self.norm(hidden_states))
```

**数据流**：
```
hidden_states (T, H)
    │
    └─ norm (RMSNorm) → (T, H)
        │
        └─ lm_head (ParallelLMHead) → (T, vocab_size)
            │
            └─ logits_processor → (T, vocab_size)
```

### 4.12 get_expert_mapping — MoE 专家参数映射

```python
def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
    return fused_moe_make_expert_params_mapping(
        self,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.n_routed_experts,
        num_redundant_experts=0,
    )
```

**返回格式**: `[(param_name, weight_name, expert_id, shard_id), ...]`

用于权重加载时，将 checkpoint 中的扁平专家权重名映射到模型中 `FusedMoE` 的实际参数位置。

### 4.13 Block 内双向注意力机制（重要！）

**生活化类比**：正常的因果注意力像"写小说"——你写第3个字时只能看到前2个字。DSpark 的 block 内双向注意力像"考试打草稿"——你可以先把整段草稿的所有字都写在草稿纸上，每个字都能看到同一段草稿里其他字的内容，一起"商量"出最优的整段。

#### 4.13.1 核心机制

DSpark 虽然复用了主模型的 `DeepseekV2DecoderLayer`（包括 DSA 稀疏注意力），但在 **draft block 生成阶段**，通过运行时 metadata 控制注意力 mask，实现了**block 内双向注意力**：

| 阶段 | causal 标志 | 注意力类型 | 可见范围 |
|------|------------|-----------|---------|
| 主模型 forward（prefill/decode） | `causal=True` | 单向因果注意力 | 当前位置 + 滑动窗口历史 |
| DSpark 上下文预计算（KV prefill） | `causal=True` | 单向因果注意力 | 正常因果mask |
| **DSpark draft block 生成** | **`causal=False`** | **Block 内双向注意力** | 滑动窗口历史 + **整个draft block全可见** |

#### 4.13.2 源码证据

在 proposer 层明确设置了 `causal=False`：

**文件位置**: `vllm_ascend/spec_decode/dspark_proposer.py:L292`
```python
cad.causal = False
cad.attn_mask = None
cad.attn_state = AscendAttentionState.ChunkedPrefill
```

DSA 后端在非因果模式下构建特殊的可见索引：

**文件位置**: `vllm_ascend/attention/dsa_v1.py:L388-L391`
```python
def build_dspark_swa_indices(...):
    """Build DSpark non-causal visible slot ids for a paged SWA cache.

    Each token in a draft block sees the trailing context window plus the
    whole current draft block. Invalid/padded rows get lens=0 and -1 slots.
    """
```

窗口范围也被扩大以覆盖整个 block：

**文件位置**: `vllm_ascend/attention/dsa_v1.py:L366-L370`
```python
def get_dspark_sparse_sas_window(vllm_config):
    window_size = int(hf_config.sliding_window)
    block_size = vllm_config.speculative_config.num_speculative_tokens
    return window_size + block_size - 1, 0  # 左窗口扩大到覆盖整个block
```

#### 4.13.3 注意力可见性图解

```
上下文（prompt历史）              draft block（本次推测生成的块）
─────────────────────────────┃────────────────────────────────
  ← sliding_window (历史) →  ┃  pos1  pos2  pos3  pos4  pos5
                             ┃
  pos1 能看到: [历史窗口]     ┃  [pos1, pos2, pos3, pos4, pos5]  ← block内全可见！
  pos2 能看到: [历史窗口]     ┃  [pos1, pos2, pos3, pos4, pos5]  ← 能"偷看"后面的字！
  pos3 能看到: [历史窗口]     ┃  [pos1, pos2, pos3, pos4, pos5]  ← 双向！
  ...
```

#### 4.13.4 为什么 Draft 模型可以"作弊"用双向注意力？

这是推测解码（Speculative Decoding）的一个关键设计：

1. **Draft 只是"提案"**：DSpark 生成的 draft token 不是最终输出，必须经过主模型的严格验证
2. **主模型是"裁判"**：验证阶段使用标准的单向因果注意力，如果某 token 在因果条件下概率过低就会被拒绝
3. **双向提升草稿质量**：block 内双向注意力让 draft 模型可以"整段思考"，生成更连贯、更准确的草稿，从而提高接受率
4. **最终正确性由主模型保证**：不管草稿怎么生成，只有通过主模型因果验证的 token 才会被接受，所以不会破坏自回归生成的正确性

> **重要提醒**：`is_draft_layer=True` 只影响 MoE 路由（不使用 hash 路由），不直接控制注意力因果性。双向注意力是通过 proposer 在运行时设置 `causal=False` 并构建特殊的 slot indices 来实现的。

---

## 五、DSparkDeepseekV4ForCausalLM — 入口类

**文件位置**: `vllm_ascend/models/deepseek_v4_dspark.py:L271-L487`

### 5.1 类继承关系

```python
@support_torch_compile
class DSparkDeepseekV4ForCausalLM(nn.Module, DeepseekV2MixtureOfExperts):
```

| 父类 | 作用 |
|------|------|
| `nn.Module` | PyTorch 模块基类 |
| `DeepseekV2MixtureOfExperts` | MoE 管理混入类，提供专家参数提取、物理专家元数据更新等功能 |
| `@support_torch_compile` | 装饰器，支持 torch.compile 编译优化 |

### 5.2 __init__ 初始化

```python
def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
    super().__init__()
    assert vllm_config.speculative_config is not None
    self.config = vllm_config.speculative_config.draft_model_config.hf_config
    
    # 量化场景下需要独立的 embed/head
    self.has_own_embed_tokens = vllm_config.quant_config is not None
    self.has_own_lm_head = vllm_config.quant_config is not None
    
    # 主体模型
    self.model = DeepseekV4DSparkModel(vllm_config=vllm_config, prefix=...)
    
    # LM Head
    self.lm_head = ParallelLMHead(
        self.config.vocab_size,
        self.config.hidden_size,
        prefix=maybe_prefix(prefix, "lm_head"),
    )
    self.logits_processor = LogitsProcessor(self.config.vocab_size)
    
    # 设置 MoE 参数
    self.set_moe_parameters()
```

### 5.3 set_moe_parameters — MoE 参数设置

```python
def set_moe_parameters(self) -> None:
    self.expert_weights: typing.MutableSequence[typing.Sequence[torch.Tensor]] = []
    self.num_expert_groups = getattr(self.config, "n_group", 1)
    self.moe_layers: list[nn.Module] = []
    self.moe_mlp_layers: list[DeepseekV4MoE] = []
    example_moe = None
    
    for layer in self.model.layers.values():
        if isinstance(layer, PPMissingLayer):
            continue
        assert isinstance(layer, DeepseekV2DecoderLayer)
        if isinstance(layer.mlp, DeepseekV4MoE):
            example_moe = layer.mlp
            self.moe_mlp_layers.append(layer.mlp)
            self.moe_layers.append(layer.mlp.experts)
    
    self.extract_moe_parameters(example_moe)
```

**处理流程**：
1. 遍历所有 draft 层
2. 收集所有 `DeepseekV4MoE` 类型的 MLP 层
3. 用最后一个 MoE 层作为示例提取超参数（前面可能是密集层）
4. 调用 `extract_moe_parameters` 设置 `num_logical_experts`、`num_routed_experts` 等参数

### 5.4 forward — 前向推理入口

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    return self.model(input_ids=input_ids, positions=positions)
```

直接委托给内部 `DeepseekV4DSparkModel`。

### 5.5 compute_logits — 计算 logits

```python
def compute_logits(
    self,
    hidden_states: torch.Tensor,
    spec_step_idx: int = 0,
) -> torch.Tensor | None:
    del spec_step_idx  # DSpark 不区分 spec_step_idx，一次性输出 block
    return self.model.compute_logits(
        hidden_states, self.lm_head, self.logits_processor
    )
```

> **与 MTP 的区别**：MTP 根据 `spec_step_idx` 选择不同的层和 head 轮询，而 DSpark 一次性生成整个 block，不需要按步索引。

### 5.6 委托方法

以下方法直接委托给 `self.model`：

| 方法 | 说明 |
|------|------|
| `markov_embed(token_ids)` | 获取 Markov 嵌入 |
| `markov_bias(markov_embed)` | 获取 Markov 偏置 |
| `get_draft_kv_cache_layer_names()` | 获取 KV 缓存层名 |
| `combine_hidden_states(aux_hidden_states)` | 融合多层隐藏状态 |
| `precompute_and_store_context_kv(...)` | 预计算上下文 KV |

---

## 六、权重加载与映射 — 进阶篇

**文件位置**: `vllm_ascend/models/deepseek_v4_dspark.py:L349-L487`

### 6.1 权重加载的挑战

DSpark 权重存储在 checkpoint 的 `mtp.*` 命名空间下，但模型内部参数命名完全不同。`load_weights` 需要完成复杂的名称重映射。

**类比**：这就像你收到一个写着"笔名"的快递，需要通过"笔名→真名"的对照表，把快递送到正确的人手里。

### 6.2 load_weights 主流程

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

**主要步骤**：

1. **准备映射表**
   ```python
   expert_mapping = self.model.get_expert_mapping()
   expert_scale_suffix = ".weight_scale" if fp4 else ".weight_scale_inv"
   
   stacked_params_mapping = [
       ("mlp.gate_up_proj", "mlp.gate_proj", 0),   # gate 融合到 gate_up 的第0片
       ("mlp.gate_up_proj", "mlp.up_proj", 1),     # up 融合到 gate_up 的第1片
       ("shared_experts.gate_up_proj", "shared_experts.gate_proj", 0),
       ("shared_experts.gate_up_proj", "shared_experts.up_proj", 1),
   ]
   ```

2. **TP 相关计算**
   ```python
   tp_size = get_tensor_model_parallel_world_size()
   tp_rank = get_tensor_model_parallel_rank()
   n_local_head = self.config.num_attention_heads // tp_size
   head_start = n_local_head * tp_rank
   head_end = n_local_head * (tp_rank + 1)
   ```

3. **遍历所有权重进行加载**

### 6.3 特殊权重处理

#### 6.3.1 embed/head 权重（量化兼容）

```python
if name == "embed.weight" and not self.has_own_embed_tokens:
    name = "model.embed_tokens.weight"
elif name == "head.weight" and not self.has_own_lm_head:
    name = "lm_head.weight"
```

- 非量化场景：直接复用主模型的 embed/head 权重
- 量化场景：使用 draft 模型自己的 embed/head

#### 6.3.2 DSpark 名称重映射

```python
mapped_name = self._remap_dspark_name(name)
if mapped_name is None:
    continue  # 不属于 DSpark 的权重跳过
name = mapped_name
```

#### 6.3.3 缩放因子后缀处理

```python
if name.endswith(".scale"):
    suffix = expert_scale_suffix if _EXPERT_SCALE_RE.search(name) else ".weight_scale"
    name = name.removesuffix(".scale") + suffix
```

- 专家权重缩放：FP4 用 `.weight_scale`，其他用 `.weight_scale_inv`
- 普通权重缩放：统一用 `.weight_scale`

#### 6.3.4 E8M0 专家缩放特殊处理

```python
if ".experts." in name:
    if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)
    # 通过 expert_mapping 加载...
```

E8M0 格式的缩放因子需要保留原始指数字节，view 为 uint8。

#### 6.3.5 堆叠参数融合（gate_up_proj）

```python
for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
    if not is_layer_param or f".{weight_name}." not in name:
        continue
    name = name.replace(weight_name, param_name)
    param = params_dict[name]
    param.weight_loader(param, loaded_weight, stacked_shard_id)
    loaded_params.add(name)
    break
```

将 checkpoint 中分开存储的 `gate_proj` 和 `up_proj` 加载到模型中融合的 `gate_up_proj`，通过 `shard_id` 区分偏移。

#### 6.3.6 Attention Sink 处理

```python
if "attn_sink" in name:
    narrow = loaded_weight[head_start:head_end]
    with torch.no_grad():
        params_dict[name][:narrow.shape[0]].copy_(narrow)
    loaded_params.add(name)
    continue
```

Attention sink 参数按 TP rank 切片，只加载当前 rank 负责的头。

### 6.4 _remap_dspark_name — DSpark 名称重映射（核心）

```python
def _remap_dspark_name(self, name: str) -> str | None:
```

#### 6.4.1 正则匹配 mtp 前缀

```python
m = re.match(r"mtp\.(\d+)\.(.*)", name)
if m is None:
    return None  # 不是 mtp 开头的权重，跳过
stage = int(m.group(1))   # mtp 阶段索引
rest = m.group(2)         # 剩余部分
```

#### 6.4.2 过滤掉不需要的权重

```python
if rest.startswith("confidence_head."):
    return None  # confidence_head 不加载（DSpark 不用）
```

#### 6.4.3 特殊组件重映射

```python
# stage 0 的 embed 是全局的
if stage == 0 and rest == "embed.weight":
    return "model.embed_tokens.weight"

# 最后一个 stage 的 head 是全局 lm_head
if stage == self.model.num_dspark_layers - 1 and rest == "head.weight":
    return "lm_head.weight"

# hc_head 参数是全局的
if rest.startswith(("hc_head_fn", "hc_head_base", "hc_head_scale")):
    return f"model.{rest}"
```

#### 6.4.4 层索引映射

```python
first_layer_idx = self.config.num_hidden_layers
last_layer_idx = first_layer_idx + self.model.num_dspark_layers - 1

if rest.startswith(("main_proj.", "main_norm.")):
    # main_proj/main_norm 属于第一层
    layer_idx = first_layer_idx
elif rest.startswith(("norm.", "markov_head.")):
    # norm/markov_head 属于最后一层
    layer_idx = last_layer_idx
else:
    # 其他权重按 stage 映射到对应层
    layer_idx = first_layer_idx + stage

name = f"model.layers.{layer_idx}.{rest}"
```

**层索引映射表**（假设 num_hidden_layers=61, num_dspark_layers=3）：

| Checkpoint 名 | 模型内部名 |
|--------------|-----------|
| `mtp.0.main_proj.*` | `model.layers.61.main_proj.*` |
| `mtp.0.main_norm.*` | `model.layers.61.main_norm.*` |
| `mtp.0.attn.*` | `model.layers.61.self_attn.*` |
| `mtp.1.attn.*` | `model.layers.62.self_attn.*` |
| `mtp.2.attn.*` | `model.layers.63.self_attn.*` |
| `mtp.2.norm.*` | `model.layers.63.norm.*` |
| `mtp.2.markov_head.*` | `model.layers.63.markov_head.*` |
| `mtp.2.head.weight` | `lm_head.weight` |

#### 6.4.5 组件名替换

```python
replacements = (
    (".attn.", ".self_attn."),           # attn → self_attn
    (".ffn_norm.", ".post_attention_layernorm."),  # ffn_norm → post_attention_layernorm
    (".attn_norm.", ".input_layernorm."), # attn_norm → input_layernorm
    (".ffn.", ".mlp."),                  # ffn → mlp
    (".w1.", ".gate_proj."),             # w1 → gate_proj
    (".w2.", ".down_proj."),             # w2 → down_proj
    (".w3.", ".up_proj."),               # w3 → up_proj
    (".mlp.gate.bias", ".mlp.gate.e_score_correction_bias"),  # 门控偏置
)
for checkpoint_name, param_name in replacements:
    name = name.replace(checkpoint_name, param_name)
```

---

## 七、完整 DSpark 推理流程 — 高级篇

### 7.1 整体时序

```
┌─────────────────────────────────────────────────────────────────┐
│                        Prefill 阶段                             │
├─────────────────────────────────────────────────────────────────┤
│  1. 主模型 forward:                                            │
│     - 正常计算所有层                                            │
│     - 在 target_layer_ids 指定层收集 aux_hidden_states          │
│     - 保存 _mtp_hidden_buffer                                   │
│                                                                 │
│  2. DSpark 初始化:                                              │
│     - combine_hidden_states(aux_hidden_states)                  │
│       → 得到融合的上下文起点                                     │
│     - precompute_and_store_context_kv(                          │
│           context_states, context_positions, slot_mappings)     │
│       → 预计算并存储所有 draft 层的上下文 KV                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Decode/Draft 阶段                           │
├─────────────────────────────────────────────────────────────────┤
│  3. 生成 draft block:                                           │
│     for each draft token in block:                              │
│       a. embed_tokens(input_ids) → 扩展 HC 维度                 │
│       b. 通过所有 draft Transformer 层                           │
│       c. hc_head 压缩                                           │
│       d. norm + lm_head → logits                                │
│       e. markov_bias → 加到 logits 上（n-gram 偏置）             │
│       f. 采样得到 next_token_id                                  │
│                                                                 │
│  4. 主模型验证:                                                 │
│     - 主模型对 draft block 并行计算                              │
│     - 逐个 token 验证，接受匹配的前缀                            │
│     - 从第一个不匹配处重新生成                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 张量形状汇总表

| 张量 | 形状 | 说明 |
|------|------|------|
| `input_ids` | `(num_tokens,)` | 输入 token ID |
| `positions` | `(num_tokens,)` | 位置索引 |
| `hidden_states (embed后)` | `(num_tokens, hidden_size)` | 词嵌入 |
| `hidden_states (HC扩展后)` | `(num_tokens, hc_mult, hidden_size)` | HC 多通道 |
| `aux_hidden_states` | `(num_tokens, hidden_size * N)` | N个目标层拼接 |
| `combined_hidden` | `(num_tokens, hidden_size)` | main_proj 融合后 |
| `shared_kv` | `(num_tokens, 1, head_dim)` | 投影后的 KV |
| `hc_head_fn` | `(hc_mult, hc_mult * hidden_size)` | HC 混合权重 |
| `hc_head_base` | `(hc_mult,)` | HC 偏置 |
| `hc_head_scale` | `(1,)` | HC 缩放 |
| `head_hidden` | `(num_tokens, hidden_size)` | HC 压缩后 |
| `markov_embed` | `(num_tokens, markov_rank)` | Markov 嵌入 |
| `markov_bias` | `(num_tokens, vocab_size)` | Markov 偏置 |
| `logits` | `(num_tokens, vocab_size)` | 最终输出 logits |

---

## 八、DSpark vs MTP 对比 — 专家篇

### 8.1 架构差异

| 特性 | MTP (6_deepseek_v4_mtp.md) | DSpark (本文档) |
|------|---------------------------|-----------------|
| **预测方式** | 串行单步预测 | 块级并行预测 |
| **层数** | `num_nextn_predict_layers`（通常1层） | `dspark_num_mtp_layers`（默认3层） |
| **层调度** | 轮询 `spec_step_idx % num_layers` | 顺序通过所有层 |
| **输入来源** | 上一步隐藏状态 + 当前 token 嵌入 | 主模型多层隐藏状态融合 |
| **上下文融合** | 无（单步预测） | `main_proj` 融合 target_layer_ids 多层 |
| **KV 预计算** | 无 | `precompute_and_store_context_kv` |
| **Markov 头** | 无 | 有 `DSparkMarkovHead` 提供 n-gram 偏置 |
| **投影融合** | `e_proj + h_proj` 双投影相加 | `main_proj` 单层拼接投影 |
| **HC 头位置** | 每层都有（延迟到 compute_logits 调用） | 仅最后一层后统一调用 |
| **Shared Head** | 所有步共享一个 head | 独立 lm_head + markov_head |
| **Confidence Head** | 无 | checkpoint 中有但 DSpark 跳过不加载 |
| **block_size** | 1（单token） | `dspark_block_size`（多token块） |

### 8.2 适用场景

| 场景 | 推荐方案 |
|------|---------|
| 保守推测，高接受率 | MTP（步数少，验证开销小） |
| 激进推测，高吞吐量 | DSpark（块生成，一次验证多个） |
| 长上下文 | DSpark（有上下文 KV 预计算优化） |
| 资源受限 | MTP（参数量更少，通常只有1层） |

### 8.3 权重命名空间

两者权重都存在 checkpoint 的 `mtp.*` 命名空间，但内部结构不同：

- **MTP**: `mtp.0.emb.tok_emb`, `mtp.0.mtp_block.*`, `mtp.0.shared_head.*`
- **DSpark**: `mtp.0.attn.*`, `mtp.0.ffn.*`, `mtp.0.main_proj.*`, `mtp.2.markov_head.*`, `mtp.2.hc_head_*`

通过配置参数决定加载哪种 draft 模型。

---

## 九、关键设计要点总结

1. **Block Drafter 而非 Serial MTP**：DSpark 不是逐个预测 token，而是一次性生成完整 block
2. **多层上下文融合**：通过 `main_proj` 融合主模型多个中间层的隐藏状态，提供更丰富的上下文信息
3. **上下文 KV 预计算**：Prefill 后预计算并缓存上下文 KV，避免 draft 生成时重复计算
4. **Markov N-gram 偏置**：利用双线性 n-gram 模型提供局部词共现偏置，提升 draft 质量
5. **复用主模型层结构**：直接使用 `DeepseekV2DecoderLayer`（`is_draft_layer=True`），包括 DSA 注意力、MoE、HC 机制
6. **参数挂接设计**：`main_proj/main_norm` 挂第一层，`norm/markov_head/hc_head_*` 挂最后一层，层内可直接访问
7. **量化兼容**：量化场景使用独立的 embed_tokens 和 lm_head，非量化场景可复用主模型权重
