# DeepSeek V4 KV 缓存体系与 Compressor/Indexer 详解

> **本文档详细讲解 DeepSeek V4 的三级 KV 缓存体系，包括 SWA（滑动窗口）、Compressor（KV 压缩）、Indexer（稀疏索引）的缓存布局、Spec、Block Size 与层间索引复用。**

> **代码版本冻结声明**：本文基于 **vllm-ascend v0.26.0rc** 源码整理，模型主文件以 [deepseek_v4.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py)（共 1565 行）为准，文中 `Lxxx` 均指该文件行号；block size 表在 [layer/attention/layer.py](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/layer/attention/layer.py)，按层压缩比查询函数 `get_dsv4_compress_ratio` 在 [utils.py:L103](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/utils.py#L103)。本版本三类缓存类**继承上游 vLLM 基类**（`vllm.models.deepseek_v4.*` / `vllm.v1.attention.backends.mla.sparse_swa`），只在 Ascend 侧覆写 spec 与后端选择。

> **统一模型配置**（W4A8 版，本文相关行）：`head_dim=512`、`qk_rope_head_dim=64`、`index_head_dim=128`、`index_n_heads=64`、`index_topk=1024`、`sliding_window=128`、`compress_rope_theta=160000`（dense 层用 `rope_theta=10000`）。`compress_ratios` 数组共 **64 项**：

| 下标区间 | 内容 | 计数 |
|---------|------|------|
| `[0:61]` 主模型 61 层 | 仅含 **128**（c128）和 **4**（c4）两种值，序列形如 `128,128,4,128,4,…`，末尾三层为 `4,128,4` | 31 个 c128 + 30 个 c4 |
| `[61:64]` 3 个 draft 层（MTP/DSpark） | **0（dense，无压缩）** | 3 个 0 |

---

## 一、三级 KV 缓存总览

### 1.1 设计理念

DeepSeek V4 采用 **DSA (Deep Sparse Attention)** 架构，用三级 KV 在 1M 上下文下平衡显存与精度：

> **生活化类比**：读一本厚书时，你不会把全书摊在桌上。SWA 是你眼前摊开的**最近 128 个字**（原文、随时可见）；Compressor 是你给每 4 页/128 页写的**章节摘要卡片**（压缩状态，记住梗概）；Indexer 是书后的**索引**——提问时先在摘要上打分，只翻出最相关的 1024 个历史块细看。

```
原始 KV (O(N) 显存)
    │
    ├── SWA（滑动窗口）   ：最近 128 个 token 的完整 KV，每层都有
    │
    ├── Compressor（压缩）：c4 层 / c128 层把历史 KV 压成状态向量
    │
    └── Indexer（稀疏索引）：仅 c4 层，在压缩状态上选 Top-1024 历史块
```

### 1.2 按层压缩配置

压缩比由 `get_dsv4_compress_ratio(config, layer_idx)`（[utils.py:L103-L108](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/utils.py#L103)）给出：

```python
def get_dsv4_compress_ratio(config, layer_idx) -> int:
    compress_ratios = getattr(config, "compress_ratios", None)
    if compress_ratios is None or layer_idx >= len(compress_ratios):
        return 0                      # 缺省 / 越界一律视为 dense
    return compress_ratios[layer_idx]
```

| `compress_ratio` | Compressor | Indexer | SWA | RoPE 基座 | 本配置中的层 |
|------------------|:-----------:|:-------:|:---:|-----------|-------------|
| **0（dense）** | ✗ | ✗ | ✓ | `rope_theta=10000` | 3 个 draft 层（数组下标 61~63） |
| **4（c4）** | ✓ `overlap=True` | ✓ TopK=1024 | ✓ | `compress_rope_theta=160000`，rope 组 `["default","c4"]` | 30 个主层 |
| **128（c128）** | ✓ `overlap=False` | ✗ | ✓ | `compress_rope_theta=160000`，rope 组 `["default","c128"]` | 31 个主层 |

> **注意**：v0.26 中 dense 的取值是 **0 不是 1**（早期文档写 1）。Compressor 构造器也只接受 4/128，传入其他值会在 L682-L684 抛 `ValueError`。
>
> **层号映射**：Attention 初始化时不是直接用运行时层号，而是先经 `extract_dsv4_layer_index`（[utils.py:L80-L93](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/utils.py#L80)）转换——MTP 层运行时叫 `mtp.0`，查配置数组时映射为 `num_hidden_layers + 本地层号`（即 61、62、63），正好对应数组末尾的 3 个 0（deepseek_v4.py L743、L814）。

### 1.3 模块挂载关系（DeepseekV4Attention，L729）

```
DeepseekV4Attention (L729)
│
├── swa_cache_layer: AscendDeepseekV4SWACache       [所有层，L881]
│
├── compressor: Compressor        (L840)            [compress_ratio > 1，即 c4/c128]
│   ├── wkv / wgate: ReplicatedLinear               — KV / 门控投影（A5 不量化）
│   ├── norm: RMSNorm(head_dim, dtype=fp32)         — 固定 fp32 权重
│   ├── ape: Parameter (ratio, coff*head_dim) fp32  — 压缩位置编码
│   └── state_cache: AscendCompressorStateCache     — 压缩状态 KV
│
└── indexer: Indexer            (L851)              [仅 compress_ratio == 4]
    ├── wq_b: ReplicatedLinear(1536 → 64*128)       — Indexer Query 投影（走 quant_config）
    ├── weights_proj: ReplicatedLinear(7168 → 64)   — 头间权重投影（不量化）
    ├── k_cache: AscendDeepseekV4IndexerCache       — Indexer 自身 KV
    └── compressor: Compressor(head_dim=128, rotate=True)  — Indexer 专用压缩器
```

三者的投影权重与缓存最终都打包进 `DSAModules`（L889-L904），交给 `AscendDeepseekSparseAttention`（`ops/dsa.py`）在后端 kernel 中执行；Python 侧的 `Indexer.forward`/`Compressor.forward` 只是占位 `pass`（L610、L701）。

---

## 二、Compressor — KV 压缩器

**类定义**: [L613-L726](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L613)

### 2.1 作用与原理

Compressor 把滑窗"吐出来"的历史 KV 按 `compress_ratio` 个 token 一组压成一个状态向量（`kv_state + score_state`），后续注意力读的是状态而非原始 KV。主注意力的 Compressor `head_dim=512`；Indexer 内部的 Compressor 用 `head_dim=index_head_dim=128` 且 `rotate=True`（L597-L607）。

### 2.2 核心成员（L626-L684）

| 成员 | 形状 / dtype | 说明 |
|------|-------------|------|
| `dim` | 7168 | 输入隐藏维 |
| `head_dim` | 512（主）/ 128（Indexer 内） | 压缩头维度，构造参数 |
| `rope_head_dim` / `nope_head_dim` | 64 / head_dim−64=448 | RoPE 与非 RoPE 切分 |
| `compress_ratio` | 4 或 128 | 压缩比 |
| `overlap` | bool | `ratio == 4` 时为 True（L633） |
| `coff` | int | `1 + overlap`：c4=2，c128=1 |
| `norm_eps` | float | RMSNorm epsilon |
| `ape` | `(ratio, coff*head_dim)` **fp32** | 压缩块内位置编码（Absolute Position Embedding） |
| `wkv` / `wgate` | `dim → coff*head_dim` | KV / 门控投影；**A5 上 `quant_config=None`（不量化）**，非 A5 才走量化（L643、L651） |
| — | — | 两者都置 `skip_weight_nz_conversion=True`（L657-L658）：定制压缩算子直接吃 ND 权重，不做 NZ 转换 |
| `norm` | `RMSNorm(head_dim)` | **固定 fp32 权重**（L660-L661，kernel 只接受 fp32 norm_weight，并非仅 A5） |
| `state_cache` | `AscendCompressorStateCache` | 状态缓存，state_dtype 固定 **float32**（L663） |

状态维度与 block：

| 层型 | `state_dim = kv_state + score_state` | state block_size（全局 128 档为例） |
|------|--------------------------------------|------------------------------------|
| c4（主注意力） | `2 × coff × head_dim = 4 × 512 = 2048` | `DSV4_BLOCK_SIZES[bs][0][2]` = 8 |
| c4（Indexer 内，head_dim=128） | `4 × 128 = 512` | 同上（共享 c4 列） |
| c128 | `2 × head_dim = 1024` | `DSV4_BLOCK_SIZES[bs][0][3]` = 32（A5 为 16） |

### 2.3 overlap 重叠机制（c4 专属）

c4 压缩时 `coff=2`，每个压缩块同时产出"与上一块重叠"的半区，保证块边界处信息连续。`overlap_transform`（[L686-L692](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L686)）实现块间拼接：

```python
def overlap_transform(self, tensor, value=0):
    b, s, _, _ = tensor.size()
    ratio, d = self.compress_ratio, self.head_dim
    new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)   # (b, s, 2*ratio, d)
    new_tensor[:, :, ratio:] = tensor[:, :, :, d:]              # 后半 = 当前块后 d 维
    new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]           # 前半 = 前一块前 d 维
    return new_tensor
```

### 2.4 rope_single（L703-L726）

压缩状态单独旋转位置编码，走 `torch_npu.npu_rotary_mul(..., rotary_mode="interleave")`：内部先转 **fp32** 计算、结束再转回原 dtype；TND（3 维，`tnd_layout=1`）与 BTND（4 维）两种布局都支持，`inverse=True` 时把 sin 取负。

### 2.5 AscendCompressorStateCache（L113-L143）

继承上游 `CompressorStateCache`，`state_dim / dtype(fp32) / compress_ratio / block_size` 四个属性。

`get_kv_cache_spec`（L126-L138）返回 `AscendSlidingWindowMLASpec`：

| 参数 | 取值 | 说明 |
|------|------|------|
| `block_size` | 构造时传入（c4→`[0][2]`，c128→`[0][3]`） | 状态块大小 |
| `num_kv_heads` | 1 | MLA 单 KV 头 |
| `head_size` | `state_dim` | 状态宽度 |
| `dtype` | float32 | 压缩状态始终 fp32 |
| `sliding_window` | 基类属性 | 状态窗口 |
| `page_size_padded` | **精确条件**：`state_dim == 2*256 且 ratio==4` 时取 `pads[0]`，否则 `pads[1]`（L128） | 即 Indexer 的 c4 压缩器（state_dim=512）用 t1 填充页，主注意力 c4/c128 用 t2 |

`get_attn_backend()` 统一返回 `AscendDSABackend`（L142-L143）。

---

## 三、Indexer — 稀疏索引器

**类定义**: [L546-L610](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/deepseek_v4.py#L546)

### 3.1 作用

只挂在 c4 层。它用一套独立的低维注意力（64 头 × 128 维）对"压缩后的历史"打分，选出与当前 query 最相关的 `index_topk=1024` 个块，主注意力只对这些块做精细计算。

> 注意：Indexer 的 Query 输入不是原始 hidden，而是 MLA 低秩 Query 压缩向量 `qr`（`q_lora_rank=1536`），因此 `wq_b` 的输入维是 1536。

### 3.2 核心成员

| 成员 | 形状 | 说明 |
|------|------|------|
| `n_heads` / `head_dim` | 64 / 128 | `index_n_heads` / `index_head_dim` |
| `rope_head_dim` | 64 | 复用 `qk_rope_head_dim` |
| `index_topk` | **1024** | `config.index_topk`（早期资料常误写为 512） |
| `q_lora_rank` | 1536 | 与主注意力共享的低秩 query 宽度 |
| `softmax_scale` | `head_dim ** -0.5` | 打分缩放 |
| `wq_b` | `1536 → 64*128=8192` | ReplicatedLinear，**走 quant_config（本包为 W8A8）**，`return_bias=False` |
| `weights_proj` | `7168 → 64` | ReplicatedLinear，**`quant_config=None` 不量化**，把 64 个头的分布汇总成头权重 |
| `k_cache` | `AscendDeepseekV4IndexerCache` | 仅 c4 时创建（L587-L595） |
| `compressor` | `Compressor(head_dim=128, rotate=True)` | 仅 ratio>1 时创建（L597-L607），与主注意力的 Compressor 是**两个独立实例**，wkv/wgate/norm/state_cache 各自独立 |

### 3.3 AscendDeepseekV4IndexerCache（L146-L179）

继承上游 `DeepseekV4IndexerCache`，KV dtype 由 Indexer 构造时的 `k_dtype` 决定（L585：A5→`float8_e4m3fn`，非 A5→**int8**）。

`get_kv_cache_spec`（L157-L174）返回 `AscendMLAAttentionSpec`：

| 参数 | 取值 | 说明 |
|------|------|------|
| `block_size` | `DSV4_BLOCK_SIZES[bs][0][0]`（128 档=128） | 与主 MLA KV 同块大小 |
| `num_kv_heads` | 1 | 单 KV 头 |
| `head_size` | 128 | index_head_dim |
| `dtype` | A5：fp8（同时把 `cache_dtype` 置 fp8）；非 A5：int8 | KV 存储精度 |
| `model_version` | `"deepseek_v4"` | 版本标记 |
| `compress_ratio` | 本缓存对应压缩比（4） | 传给后端选 kernel |
| `scale_dim` | `1 if head_dim==128 else 0` → 1 | 逐 token scale 放置维度 |
| `scale_dtype` | A5：`torch.float`；非 A5：`torch.float16` | 量化 scale 精度 |

---

## 四、SWA 滑动窗口缓存

### 4.1 AscendDeepseekV4SWACache（L182-L215）

存储最近 `sliding_window=128` 个 token 的完整未压缩 KV，所有层（含 dense draft 层）都有。构造时基类先按 `torch.uint8` 建，再用 `self.dtype` 覆盖（L191-L192）；`block_size` 取 `DSV4_BLOCK_SIZES[bs][0][1]`（128 档=128，L194）。

| 特性 | A5 | 非 A5 |
|------|----|-------|
| KV dtype | `float8_e4m3fn`（L198-L199 同步设置 `cache_dtype`） | `bfloat16`（L880 传入） |
| `cached_head_size` | `head_dim + 128 = 640`（L200） | `head_dim = 512` |

`get_kv_cache_spec`（L196-L210）返回 `AscendSlidingWindowMLASpec`，含 `cache_dtype_str`、`model_version="deepseek_v4"`、`sliding_window=128`、`alignment=None`。

> 多出的 128 维用于承载 FP8 KV 的 scale/元数据，使 FP8 的 K/V 与缩放因子能放进同一个 paged 缓存。

---

## 五、Block Size 配置

**定义位置**: [layer.py:L32-L50](file:///c:/Users/89517/Desktop/vllm同步/vllm-ascend/vllm_ascend/models/layer/attention/layer.py#L32)，函数 `get_dsv4_block_sizes()` 按设备返回，模块加载时固化为全局 `DSV4_BLOCK_SIZES`（L50）。模型侧通过 `_dsv4_block_sizes()` 懒加载引用（deepseek_v4.py L105-L110），规避循环导入。

### 5.1 非 A5（L34-L38）

```python
_DSV4_BLOCK_SIZES = {
    128: [[128, 128, 8, 32], [16640, 131072]],
    64:  [[64,  64,  4, 16], [8320,  65536]],
    32:  [[32,  32,  2, 8],  [4160,  32768]],
}
```

### 5.2 A5（L39-L43）

```python
_DSV4_BLOCK_SIZES_A5 = {
    128: [[128, 128, 8, 16], [16896, 81920]],
    64:  [[64,  64,  4, 8],  [8448,  40960]],
    32:  [[32,  32,  2, 4],  [4224,  20480]],
}
```

### 5.3 索引含义（注释见 layer.py L33）

第一组 `[0]` 四个 block size：

| 索引 | 用途 | 128 档（非 A5 / A5） |
|------|------|----------------------|
| `[0][0]` | mla 主 KV / Indexer KV | 128 / 128 |
| `[0][1]` | SWA KV | 128 / 128 |
| `[0][2]` | c4 压缩状态 | 8 / 8 |
| `[0][3]` | c128 压缩状态 | 32 / **16** |

第二组 `[1]` 两个填充页大小：

| 索引 | 用途 | 128 档（非 A5 / A5） |
|------|------|----------------------|
| `[1][0]` | `page_size_padded_t1`（Indexer c4 state，state_dim=512） | 16640 / 16896 |
| `[1][1]` | `page_size_padded_t2`（主 c4/c128 state） | 131072 / 81920 |

> A5 的 c128 状态块大小减半（32→16 等），t2 填充页也显著更小，与其 FP8 KV/状态布局有关。

---

## 六、IndexCache — 层间 TopK 索引复用

代码位于 Attention 初始化 L860-L877。开启后部分 c4 层不自己算 TopK，直接复用前序 c4 层的索引，省掉一次 Indexer 打分（参考论文链接见 L861 注释）。

**触发条件（三者同时）**：
1. 本层 `compress_ratio == 4`（确有 Indexer）；
2. `config.use_index_cache == True`（hf-overrides 显式打开）；
3. 前缀不含 `.mtp.`——**MTP/draft 层被排除**（L867）。

> **v0.26 的重要细节**：MTP 层不参与复用。注释（L863-L865）解释：spec decode 只在**主模型层级**共享 `topk_indices_buffer`，impl 层拿不到对 MTP 有效的引用，强行复用会拿到陈旧索引。MTP 预测层改为**自建** TopK buffer（见文档 6 §MTP 部分）。

### 6.1 频率模式（`index_topk_freq`）

```python
indexer_seq_idx = sum(1 for r in compress_ratios[:config_layer_idx] if r == 4)  # L869
freq = getattr(config, "index_topk_freq", 1)                                    # L871
skip_topk = max(indexer_seq_idx - 1, 0) % freq != 0                             # L873
```

- `indexer_seq_idx`：本层是第几个 c4 层（只数配置数组中本层之前的 `4`，用的是 **config_layer_idx**，因此 draft 层不会干扰计数）；
- 第 0 个 c4 层必算，之后每隔 `freq` 个 c4 层算一次，其余跳过。

### 6.2 模式串模式（`index_topk_pattern`）

```python
pattern = getattr(config, "index_topk_pattern", None)
assert pattern[0] == "F"                                    # 首个 c4 层必须算
if 0 <= indexer_seq_idx < len(pattern):
    skip_topk = pattern[indexer_seq_idx] == "S"             # L877，越界保持 False
```

如 `"FSFS"` 表示第 0、2 个 c4 层计算（F），第 1、3 个复用（S）。

### 6.3 topk_indices_buffer

- 形状 `(max_num_batched_tokens, index_topk_tokens)`、int32，主模型在模型级创建一份并逐层传入 `DeepseekV4Attention(..., topk_indices_buffer=...)`（L738、L902）；
- `skip_topk=False` 的层写入新索引，`skip_topk=True` 的层直接读用；
- `skip_topk` 与 buffer 一起打包进 `DSAModules`（L902-L903），由 DSA 后端 kernel 消费；
- **MTP 预测层不共享此 buffer**，自建自管（见文档 6）。

---

## 七、KV 缓存 Spec 与后端体系

### 7.1 两类 Spec

| Spec 类型 | 使用对象 |
|-----------|---------|
| `AscendSlidingWindowMLASpec`（`vllm_ascend/core/kv_cache_interface.py`） | SWA Cache、Compressor State Cache |
| `AscendMLAAttentionSpec` | Indexer KV Cache（主 MLA KV 同源） |

### 7.2 三个缓存类的落点

| 缓存类 | 行号 | 基类 | spec | 后端 |
|--------|------|------|------|------|
| `AscendCompressorStateCache` | L113 | 上游 `CompressorStateCache` | SlidingWindowMLA(fp32, padded page) | `AscendDSABackend` |
| `AscendDeepseekV4IndexerCache` | L146 | 上游 `DeepseekV4IndexerCache` | MLA(fp8/int8 + scale) | `AscendDSABackend` |
| `AscendDeepseekV4SWACache` | L182 | 上游 `VllmDeepseekV4SWACache` | SlidingWindowMLA(fp8/bf16, window=128) | `AscendDSABackend` |

三个类的 `forward()` 都是空实现（缓存本身不做计算），真正的读写在 `AscendDSABackend` / `AscendDSAImpl`（`vllm_ascend/attention/dsa_v1.py`，详见文档 3）中随注意力一起完成。

---

## 八、A5 设备适配汇总

| 特性 | A5 | 非 A5 |
|------|----|-------|
| SWA KV dtype | float8_e4m3fn | bfloat16 |
| SWA `cached_head_size` | 512 + 128 = 640 | 512 |
| Indexer KV dtype | float8_e4m3fn | int8 |
| Indexer scale dtype | float32 | float16 |
| Compressor wkv/wgate | 不量化（`quant_config=None`） | 走 quant_config |
| Compressor norm 权重 | fp32（**所有设备均如此**） | fp32 |
| 压缩状态 dtype | float32 | float32 |
| c128 state block_size（128 档） | 16 | 32 |
| t1 / t2 填充页（128 档） | 16896 / 81920 | 16640 / 131072 |

---

## 九、一句话总结

V4 的每一层都带一份 **SWA 全量 KV**（窗长 128）；30 个 c4 层额外挂 **Compressor（4:1 重叠压缩、fp32 状态）+ Indexer（64×128 低维注意力选 Top-1024）**，31 个 c128 层只挂 **128:1 非重叠压缩器**，3 个 draft 层为 dense（比值 **0**）；三类缓存继承上游基类、通过 `AscendSlidingWindowMLASpec / AscendMLAAttentionSpec` 描述布局，块大小与填充页由 `DSV4_BLOCK_SIZES` 按设备（A5/非 A5）统一查表；可选的 IndexCache 让 c4 层按频率或模式串复用 TopK 索引，但 MTP 层被刻意排除。
