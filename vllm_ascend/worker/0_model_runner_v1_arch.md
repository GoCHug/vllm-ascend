# NPUModelRunner (model_runner_v1.py) 由浅入深详细梳理

## 第一章：概述与定位

### 1.1 什么是 ModelRunner

ModelRunner 是 vLLM 推理引擎中的**核心执行单元**，负责：
- 加载和管理模型
- 准备输入数据（token ids、positions、attention metadata等）
- 执行模型前向传播
- 采样生成下一个 token
- 管理 KV Cache
- 处理推测解码（Speculative Decoding）
- 协调分布式并行策略

简单来说：**调度器告诉 ModelRunner 跑哪些请求，ModelRunner 负责真正在 NPU 上把模型跑起来**。

### 1.2 文件定位

| 项目 | 路径 |
|------|------|
| 原始 vLLM (GPU版) | `vllm/v1/worker/gpu_model_runner.py` |
| Ascend 适配版 (NPU版) | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py` |
| 核心类 | `NPUModelRunner` |

- 文件行数：约 5270 行
- 核心类：`NPUModelRunner`，继承自 `GPUModelRunner`
- 设计思路：**尽量复用上游 vLLM 逻辑，只重写 NPU 相关的差异化部分**

---

## 第二章：类继承关系与整体架构

### 2.1 继承关系图

```
GPUModelRunner (vllm 上游，GPU 版本)
    ↑
    └── NPUModelRunner (vllm-ascend，NPU 适配版)
```

### 2.2 设计原则

NPUModelRunner 采用**继承 + 重写**的策略：
- **继承**：大部分通用逻辑从 `GPUModelRunner` 继承（如调度交互、状态管理等）
- **重写**：所有涉及 GPU/NPU 硬件差异的方法都被重写

### 2.3 核心数据成员概览

| 类别 | 关键成员 | 作用 |
|------|----------|------|
| **模型相关** | `self.model` | 实际的神经网络模型（可能被 ACLGraphWrapper 包装） |
| **输入批次** | `self.input_batch` | NPUInputBatch，管理当前批次的所有请求状态 |
| **采样器** | `self.sampler` | AscendSampler，NPU 上的 token 采样器 |
| **拒绝采样器** | `self.rejection_sampler` | AscendRejectionSampler，推测解码用 |
| **注意力后端** | `self.attn_backend` | 注意力后端（AscendAttentionBackend / AscendMLABackend 等） |
| **KV Cache** | 多个 kv_caches | 各层的 key/value 缓存 |
| **推测解码** | `self.drafter` | 草稿模型/提议器（Eagle、Medusa、Ngram 等） |
| **并行策略** | `self.pcp_manager` | PCP（上下文并行）管理器 |

### 2.4 核心方法分层

```
┌─────────────────────────────────────────┐
│          对外接口层 (Public API)         │
│  execute_model() / sample_tokens()      │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│        执行流程层 (Execution Flow)       │
│  _prepare_inputs() / _model_forward()   │
│  _sample() / _bookkeeping_sync()        │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│      功能模块层 (Functional Modules)     │
│  _build_attention_metadata()            │
│  _calc_spec_decode_metadata()           │
│  propose_draft_token_ids()              │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│       基础设施层 (Infrastructure)        │
│  load_model() / initialize_kv_cache()   │
│  _sync_device() / _torch_cuda_wrapper() │
└─────────────────────────────────────────┘
```

---

## 第三章：初始化流程详解

### 3.1 `__init__` 方法总览

位置：`model_runner_v1.py:260-600`

初始化是 NPUModelRunner 最复杂的部分之一，涉及大量配置解析和资源预分配。

**执行顺序：**
```
1. PCP token 数预扩容
2. 设置 use_compress 标志（压缩KV缓存）
3. _torch_cuda_wrapper() 包装 → 调用父类 __init__
4. NPU PrefetchOffloader 替换
5. query_start_loc / gdn_query_start_loc 缓冲区
6. 基础配置（max_num_tokens, max_num_reqs, dp_size等）
7. AscendSampler 初始化
8. Ascend 特有配置（ascend_config）
9. 稀疏注意力配置（use_sparse, sparse_head_dim）
10. 注意力后端选择
11. PCP/DCP 并行初始化
12. 推测解码 Drafter 设置
13. cos/sin、MC2 等全局配置
14. ACL Graph 配置
15. EPLB 动态专家负载均衡
16. NPUInputBatch 初始化
```

### 3.2 关键初始化细节

#### 3.2.1 `_torch_cuda_wrapper()` —— CUDA → NPU 兼容层

位置：`model_runner_v1.py:5200-5241`

**为什么需要它？**
上游 vLLM 的 `GPUModelRunner.__init__()` 里大量使用 `torch.cuda.xxx` API。为了不修改上游代码，NPU 版本通过 monkey patch 的方式，在调用父类初始化前临时把 `torch.cuda` 替换成 `torch.npu`。

```python
# 简化示意
with _torch_cuda_wrapper():
    # 在这个上下文里，torch.cuda.Event 实际是 torch.npu.Event
    super().__init__(vllm_config, device)
```

**替换的 API 列表：**
- `torch.cuda.Event` → `torch.npu.Event`
- `torch.cuda.Stream` → `torch.npu.Stream`
- `torch.cuda.default_stream` → `torch.npu.default_stream`
- `torch.cuda.current_stream` → `torch.npu.current_stream`
- `torch.cuda.synchronize` → `torch.npu.synchronize`
- `torch.cuda.mem_get_info` → `torch.npu.mem_get_info`

#### 3.2.2 稀疏注意力配置

位置：`model_runner_v1.py:358-396`

**判断条件：**
```python
self.use_sparse = hasattr(vllm_config.model_config, "hf_text_config") \
    and hasattr(vllm_config.model_config.hf_text_config, "index_topk") \
    and not hasattr(vllm_config.model_config.hf_text_config, "compress_ratios")
```

**三种 head_dim：**
- `kv_lora_rank`：KV LoRA 秩
- `qk_rope_head_dim`：QK RoPE 头维度
- `index_head_dim`：索引头维度（用于稀疏检索）

**A5 芯片的特殊优化（Sparse C8）：**
当 `enable_sparse_c8=True` 且是 A5 芯片时，kv_lora 和 k_rope 合并成一个 CKV FP8 tensor，减少内存访问。

#### 3.2.3 PCP (Prefill Context Parallel) 初始化

位置：`model_runner_v1.py:427-446`

PCP 是 Ascend 特有的**上下文并行**技术，用于把长 prefill 序列切分到多个 NPU 上并行计算。

```python
if self.pcp_size > 1:
    self.pcp_manager = PCPManager(
        self.pcp_size, self.pcp_rank,
        self.dcp_size, self.dcp_rank,
        max_buffer_num_tokens, self.max_num_reqs,
        self.device, self.vllm_config, ...
    )
```

相关概念：
- **PCP**：Prefill Context Parallel，prefill 阶段的上下文并行
- **DCP**：Decode Context Parallel，decode 阶段的上下文并行
- `use_cp`：`pcp_size * dcp_size > 1` 时为 True

#### 3.2.4 推测解码 Drafter 配置

位置：`model_runner_v1.py:611-641`

支持的推测解码方法：
| 方法 | 类名 | 说明 |
|------|------|------|
| ngram | `AscendNgramProposer` | CPU 上的 ngram 匹配 |
| ngram_npu | `AscendNgramProposerNPU` | NPU 上加速的 ngram |
| eagle / eagle3 | `AscendEagleProposer` | EAGLE 推测解码 |
| mtp / step3p5 | `AscendStep3p5MTPProposer` | DeepSeek MTP |
| draft_model | `AscendDraftModelProposer` | 独立草稿模型 |
| dflash | `AscendDflashProposer` | DeepFlash |
| suffix | `AscendSuffixDecodingProposer` | 后缀匹配 |
| medusa | `AscendMedusaProposer` | Medusa 多头 |
| extract_hidden_states | `AscendExtractHiddenStatesProposer` | 提取隐状态 |

### 3.3 `load_model()` —— 模型加载

位置：`model_runner_v1.py:3822-3930`

**流程：**
```
1. 记录开始时间
2. mix_placement 适配（MoE 共享专家）
3. EPLB 相关 mock（跳过 EP 权重过滤）
4. 调用 get_model() 加载主模型
5. 检查是否有 sink 参数
6. 加载 drafter 模型（如果有）
7. 配置 aux_hidden_states（EAGLE3 用）
8. LoRA 模型加载
9. 统计模型内存占用
10. 用 ACLGraphWrapper 包装模型（如果启用图模式）
11. 启动 dump debugger（如果配置）
```

**ACLGraphWrapper 包装：**
```python
if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
    self.model = ACLGraphWrapper(
        self.model,
        self.vllm_config,
        runtime_mode=CUDAGraphMode.FULL,
        use_eagle=self.use_eagle,
        enable_enpu=self.enable_enpu,
    )
```

---

## 第四章：核心执行流程

### 4.1 Prefill 阶段 vs Decode 阶段

推理分为两个核心阶段，它们的行为差异很大：

| 维度 | Prefill 阶段 | Decode 阶段 |
|------|-------------|-------------|
| **做什么** | 把用户输入的整段提示词喂给模型，生成第一个 token | 逐 token 生成后续内容 |
| **输入 token 数** | 多（整个 prompt） | 少（通常 1 个，推测解码时 1+K 个） |
| **计算瓶颈** | 计算密集（矩阵乘大） | 访存密集（KV cache 读写） |
| **注意力模式** | 全注意力（query 看所有 key） | 滑窗/全注意力（1 个 query 看所有 key） |
| **推测解码** | 不参与 | 参与（提议+验证） |
| **PCP 并行** | 支持（按 token 切分） | 不支持 |
| **KV Cache** | 写入 | 读取 + 追加 |

**注意力状态机（AscendAttentionState）：**

```
PrefillNoCache       ← 首次 prefill，没有缓存
PrefillCacheHit      ← prefill 命中缓存
ChunkedPrefill       ← 分块 prefill（长提示词）
DecodeOnly           ← 纯 decode，无推测解码
SpecDecoding         ← decode + 推测解码
```

---

### 4.2 两阶段执行模型

无论是 prefill 还是 decode，每一轮都遵循同样的**两阶段执行模型**：

```
1. execute_model()  → 执行模型前向，返回 None（异步状态存在 execute_model_state）
2. sample_tokens()  → 对 logits 进行采样 + 生成草稿，输出最终结果
```

为什么分两步？为了支持**异步调度**（async scheduling）：
- execute_model 把计算任务发到 NPU 上就可以返回
- 不用等 NPU 算完，CPU 可以继续准备下一批
- sample_tokens 时再同步取结果

---

### 4.3 Prefill 阶段执行流程

Prefill 阶段的特点：**没有推测解码的验证**（调度器还没带草稿 token 进来），输入是整段 prompt，主要任务是计算 prompt 的表示并写入 KV cache。但如果开启了推测解码，prefill 结束时会生成**第一批草稿**，供下一轮 decode 验证。

**代码判断依据：**
```python
# model_runner_v1.py:1345
use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
```
Prefill 时调度器不带草稿 token → `use_spec_decode = False` → `spec_decode_metadata = None` → 走普通采样。

#### 4.3.1 `execute_model()` — Prefill 路径

```
scheduler_output (prefill 请求列表，无草稿 token)
    │
    ▼
_update_states()
    │
    ▼
_prepare_inputs()
    ├─ 构建 input_ids (整个 prompt)
    ├─ 构建 positions (0, 1, 2, ..., seq_len-1)
    ├─ 构建 slot_mapping (分配新的 KV 槽位)
    ├─ attn_state = PrefillNoCache / ChunkedPrefill / PrefillCacheHit
    └─ spec_decode_metadata = None  ← 没有草稿，不做拒绝采样
    │
    ▼
_build_attention_metadata()
    └─ prefill 模式的注意力元数据（大量 query）
    │
    ▼
_model_forward()  ◄───── 主模型前向，计算量很大
    │
    ▼
计算 logits（只需要最后一个位置的 logits 用于采样）
    │
    ▼
保存 execute_model_state
```

**Prefill 特有优化：**
- **PCP (Prefill Context Parallel)**：把长 prompt 按 token 切分到多卡，每张卡算一部分的注意力
- **Chunked Prefill**：特别长的 prompt 分成多个 chunk 逐步处理，避免一次占满显存
- **Prefix Cache Hit**：如果 prompt 前缀之前算过，直接复用 KV cache

#### 4.3.2 `sample_tokens()` — Prefill 路径

```
从 execute_model_state 取出状态
    │
    ▼
apply_grammar_bitmask() (如果有结构化输出)
    │
    ▼
_sample()  ◄───── 普通采样，生成第一个 token（spec_decode_metadata = None）
    └─ AscendSampler
    │
    ▼
_bookkeeping_sync()
    ├─ 保存 prompt logprobs
    └─ 更新 input_batch 状态
    │
    ▼
propose_draft_token_ids()  ◄── 如果开了推测解码，这里生成第一批草稿
    │                           (这批草稿会返回给调度器，下一轮 decode 带进来验证)
    ▼
返回 ModelRunnerOutput
    └─ spec_token_ids = 第一批草稿 token
```

> **关键点：** Prefill 阶段本身不做验证（没有草稿要验证），但如果开启了推测解码，prefill 结束时会**提议第一批草稿**。这批草稿通过 `ModelRunnerOutput.spec_token_ids` 返回给调度器，下一轮 decode 时调度器会把它们带进来。

---

### 4.4 Decode 阶段执行流程

Decode 阶段的特点：**每轮生成 1 个 token**（推测解码时可能接受 K 个），如果开启了推测解码，从第一轮 decode 开始就走"验证+提议"的流水线。

#### 4.4.1 推测解码的完整时序（从 prefill 开始）

从用户发请求到稳定推测解码，一共经历这些阶段：

```
  Prefill 轮           Decode 第1轮           Decode 第2轮           Decode 第3轮...
  (提议第1批草稿)       (验证第1批+提议第2批)   (应用第1批结果+        (应用第2批结果+
                                               验证第2批+提议第3批)     验证第3批+提议第4批)
  ─────────           ────────────           ────────────           ────────────
execute_model()       execute_model()        execute_model()        execute_model()
  │ 跑整个prompt        │ 跑 1+K 个token        │ 读取 num_accepted     │ 读取 num_accepted
  │ (无草稿输入)        │ (真实token+第1批草稿)  │  (第1批的验证结果)     │  (第2批的验证结果)
  │                     │                       │ 修正序列长度           │ 修正序列长度
  │                     │                       │ 跑 1+K 个token        │ 跑 1+K 个token
  │                     │                       │ (真实+第2批草稿)      │ (真实+第3批草稿)
  ▼                     ▼                       ▼                       ▼
sample_tokens()       sample_tokens()        sample_tokens()        sample_tokens()
  │                     │                       │                       │
  ├ 普通采样            ├ 拒绝采样               ├ 拒绝采样               ├ 拒绝采样
  │   (第1个token)      │   ↑ 验证第1批草稿      │   ↑ 验证第2批草稿      │   ↑ 验证第3批草稿
  │                     │   输出 num_accepted    │   输出 num_accepted    │   输出 num_accepted
  └ propose_draft()     └ propose_draft()       └ propose_draft()       └ propose_draft()
    ↓ 第1批草稿          ↓ 第2批草稿              ↓ 第3批草稿              ↓ 第4批草稿
    调度器保存           调度器保存               调度器保存               调度器保存
```

**各轮的核心动作：**

| 轮次 | 验证了谁的草稿？ | 应用了谁的验证结果？ | 提议了给谁的草稿？ |
|------|-----------------|---------------------|------------------|
| **Prefill** | —（无草稿可验） | — | 给 Decode 第 1 轮的 |
| **Decode 第 1 轮** | Prefill 提议的第 1 批 | —（结果还没传到） | 给 Decode 第 2 轮的 |
| **Decode 第 2 轮** | Decode 第 1 轮提议的第 2 批 | Prefill 第 1 批的验证结果 | 给 Decode 第 3 轮的 |
| **Decode 第 3 轮** | Decode 第 2 轮提议的第 3 批 | Decode 第 1 轮第 2 批的验证结果 | 给 Decode 第 4 轮的 |
| **...** | ... | ... | ... |

**一句话总结：提议晚 1 轮生效，验证晚 2 轮生效。** 这就是为什么叫"三轮流水线"——提议、验证、应用结果分布在连续的三轮里。

**代码判断依据：**
```python
# model_runner_v1.py:1345 — 有没有草稿要验证，看调度器带没带进来
use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0

# model_runner_v1.py:1135-1161 — 验证结果从哪里来？从上一轮 sample_tokens 存在 CPU 上
if self.num_accepted_tokens_event is not None:
    self.num_accepted_tokens_event.synchronize()
    # 从 input_batch.num_accepted_tokens_cpu 读取
```

#### 4.4.2 `execute_model()` — Decode 路径

```
scheduler_output (decode 请求列表 + 上一轮的草稿 token)
    │
    ▼
_update_states()
    │
    ▼
_prepare_inputs()
    ├─ 读取 num_accepted_tokens (从 input_batch CPU 上读)  ◄── 关键点：上上轮的验证结果
    │   └─ Decode 第 1 轮：num_accepted_tokens = 1（没有上一轮验证结果，默认 1）
    │   └─ Decode 第 2 轮起：是上上轮提议的草稿的验证结果
    ├─ 修正 num_computed_tokens（丢弃被拒绝的 token 的计算）
    ├─ 构建 input_ids (1 个真实 token + K 个草稿 token)
    ├─ 构建 positions (续上之前的位置)
    ├─ 构建 slot_mapping (追加新的 KV 槽位)
    ├─ attn_state = DecodeOnly / SpecDecoding
    └─ spec_decode_metadata = 非 None（有草稿要验证）
    │
    ▼
_build_attention_metadata()
    └─ decode 模式的注意力元数据（少量 query，大量 key/value 来自缓存）
    │
    ▼
_model_forward()  ◄───── 主模型前向（顺便把草稿 token 也跑了，用于验证）
    │
    ▼
计算 logits（真实 token 位置 + 所有草稿 token 位置都要算）
    │
    ▼
保存 execute_model_state
```

**Decode 特有优化：**
- **ACL Graph / CUDA Graph**：因为 decode 的输入形状固定，可以用图模式加速
- **EPLB (Early Prompt Late Binding)**：预绑定 prompt 部分的 KV cache，减少每轮的准备开销
- **Sparse KV Cache (C8 格式)**：稀疏存储，减少显存占用和访存

#### 4.4.3 `sample_tokens()` — Decode 路径

```
从 execute_model_state 取出状态
    │
    ▼
apply_grammar_bitmask() (如果有结构化输出)
    │
    ▼
_sample()  ◄────────────────── 拒绝采样（验证上一轮的草稿）
    │
    └─ rejection_sampler (AscendRejectionSampler)
          ├─ 输入：主模型 logits + 上一轮草稿的 probs
          ├─ 逐 token 比较，接受匹配的，拒绝不匹配的
          └─ 输出：sampled_token_ids + num_accepted_tokens
    │
    ▼
_bookkeeping_sync()  ◄─────── 台账同步（D2H 拷贝）
    ├─ 生成有效 token 列表
    ├─ 更新 input_batch 中的 token 记录
    ├─ 更新 request_state 输出
    └─ num_accepted_tokens 同步到 CPU（供下下轮 execute_model 读）
    │
    ▼
propose_draft_token_ids()  ◄── 提议下一轮的草稿
    │
    ├─ padded batch 模式（EAGLE/Medusa/ngram_npu 等）
    │     └─ 在 bookkeeping 之前就跑，直接用 GPU tensor，减少 D2H
    │
    └─ non-padded 模式（ngram CPU/Suffix 等）
          └─ 在 bookkeeping 之后跑，用 CPU 上的 token 列表
    │
    ▼
组装 ModelRunnerOutput
    └─ spec_token_ids = 新一批草稿 token（返回给调度器，下一轮带进来）
    │
    ▼
返回结果（异步/同步）
```

#### 4.4.4 草稿提议的两种时机

| 类型 | 执行时机 | 输入来源 | 原因 |
|------|----------|----------|------|
| **EAGLE / Draft Model / Medusa / MTP / ngram_npu** | `_bookkeeping_sync()` **之前** | GPU 上的 `sampled_token_ids` | 不需要等 CPU 同步，直接用 GPU 张量，减少一次 D2H 拷贝 |
| **ngram (CPU) / Suffix** | `_bookkeeping_sync()` **之后** | CPU 上的 `valid_sampled_token_ids` | 这些算法在 CPU 上跑，必须等同步完 |

---

### 4.5 关键状态 `ExecuteModelState`

位置：`model_runner_v1.py:242-258`

无论是 prefill 还是 decode，`execute_model` 执行完后，把中间状态都封装在这里，供 `sample_tokens` 使用：

```python
ExecuteModelState(
    scheduler_output,        # 调度输出
    logits,                  # [num_sampled_tokens, vocab_size]
    spec_decode_metadata,    # 推测解码元数据
    spec_decode_common_attn_metadata,
    hidden_states,           # 完整隐状态 [num_tokens, hidden_size]
    sample_hidden_states,    # 采样位置的隐状态
    aux_hidden_states,       # 辅助隐状态（EAGLE3等）
    attn_metadata,           # 注意力元数据
    positions,               # 位置编码
    ec_connector_output,     # EC连接器输出
    cudagraph_stats,         # 图执行统计
    batch_desc,              # 批次描述符
)
```

---

## 第五章：关键子模块详解

### 5.1 `_prepare_inputs()` —— 输入准备

位置：`model_runner_v1.py:857-1423`

这是最复杂的方法之一，负责把调度器的输出转换成 NPU 可以直接使用的张量。

**输入：**
- `scheduler_output`：调度器输出
- `num_scheduled_tokens`：每个请求调度的 token 数 `[num_reqs]`

**输出：**
- `logits_indices`：需要计算 logits 的位置索引
- `spec_decode_metadata`：推测解码元数据
- `total_num_scheduled_tokens`：总调度 token 数

#### 5.1.1 核心步骤

```
1. 提交 block_table 到 GPU（异步，与 CPU 计算重叠）
2. _build_attn_state() —— 判断注意力状态
3. 计算 positions（每个 token 的位置）
4. PCP 序列切分（如果 pcp_size > 1）
5. 生成 token_indices 并从 input_batch 中 index_select 出 input_ids
6. 处理 prompt_embeds（多模态模型）
7. 设置 query_start_loc
8. 计算 optimistic_seq_lens（乐观序列长度）
9. 计算 prev_positions（上一步位置映射）
10. 拷贝 input_ids 到 NPU
11. M-RoPE / XD-RoPE 位置计算
12. 计算 discard_request_mask（丢弃的请求）
13. 同步 num_accepted_tokens
14. 更新 num_computed_tokens（GPU上）
15. 异步推测解码时重建输入（如果需要）
16. 计算 slot_mapping（KV缓存槽位映射）
17. 计算 spec_decode_metadata（如果有推测解码）
18. LoRA 激活设置
19. LMHead TP 时的 padding
```

#### 5.1.2 `_build_attn_state()` —— 注意力状态判断

位置：`model_runner_v1.py:1565-1594`

根据当前批次的特征，判断属于哪种注意力执行模式：

| 状态 | 条件 | 说明 |
|------|------|------|
| `PrefillNoCache` | 所有请求 computed_tokens == 0 | 纯 prefill，无缓存 |
| `DecodeOnly` | 所有请求都只调度 1 个 token | 纯 decode 阶段 |
| `SpecDecoding` | 所有有效 token 数为 1 且有推测解码 | 推测解码模式 |
| `ChunkedPrefill` | 启用了 chunked prefill | 分块 prefill |
| `PrefillCacheHit` | 其他情况 | prefill 有缓存命中 |

### 5.2 `_build_attention_metadata()` —— 注意力元数据构建

位置：`model_runner_v1.py:3146-3460`

注意力元数据是注意力算子的"参数包"，告诉算子：
- 每个请求多长（seq_lens）
- KV 缓存在哪（block_table）
- 用什么模式（prefill / decode / spec_decode）
- 是否启用各种优化

**支持的注意力后端：**
- `AscendAttentionBackend`：标准 Ascend 注意力
- `AscendMLABackend`：MLA（Multi-head Latent Attention）
- `GDNAttentionMetadataBuilder`：GDN 注意力
- `AscendDSAMetadataBuilder`：DeepSeek 稀疏注意力（DSA）
- `AscendDSACPMetadataBuilder`：DSA + 上下文并行

**注意力分组（AttentionGroup）：**
如果模型有多种不同配置的注意力层（如滑窗+全注意力），会分成多个 group 分别构建 metadata。

**Prefill vs Decode 的注意力差异：**

| 维度 | Prefill | Decode |
|------|---------|--------|
| query 数量 | 多（等于 prompt token 数） | 少（1 个或 1+K 个） |
| key/value 来源 | 实时计算 | 大部分来自 KV cache，只追加少量 |
| 计算模式 | GEMM 密集（矩阵乘大） | 访存密集（读 KV cache） |
| 算子类型 | FlashAttention 类 | PagedAttention 类 |
| 是否支持图模式 | 不支持（形状不固定） | 支持（形状固定） |

### 5.3 `_sample()` —— 采样与拒绝采样

位置：`model_runner_v1.py:2729-2760`

`_sample()` 有两种工作模式：

| 模式 | 使用的采样器 | 场景 |
|------|-------------|------|
| 普通采样 | `self.sampler` (AscendSampler) | 无推测解码 |
| 拒绝采样 | `self.rejection_sampler` (AscendRejectionSampler) | 有推测解码（验证上一轮的草稿） |

**拒绝采样的输入输出：**

| 输入 | 来源 | 作用 |
|------|------|------|
| `spec_decode_metadata` | `_calc_spec_decode_metadata()` | 草稿 token 的位置索引等元信息 |
| `draft_probs` | 上一轮草稿模型输出 | 草稿模型预测的概率分布 |
| `logits` | 当前轮主模型输出 | 主模型对所有位置的 logits（验证的"标准答案"） |
| `sampling_metadata` | `input_batch` | 采样参数（temperature, top_k 等） |

**输出：**
- `sampled_token_ids`：最终被接受的 token 序列
- `num_accepted_tokens`：每个请求接受了多少个 token（供下一轮修正用）

**Ascend 特有点：**
- `lmhead_tp_enable()` 时裁剪 logits（TP 并行）
- `enable_reduce_sample` 时先做 Top-K reduce 再采样（减少计算量）

### 5.4 验证结果的传递与应用

验证结果 `num_accepted_tokens` 从产生到生效需要跨两轮：

```
第 N+1 轮 sample_tokens():
  rejection_sampler() 输出 num_accepted_tokens
    ↓
  _bookkeeping_sync() 同步到 CPU
    ↓
  input_batch.num_accepted_tokens_cpu  ← 暂存

第 N+2 轮 execute_model():
  _prepare_inputs()
    ↓
  从 input_batch.num_accepted_tokens_cpu 读取
    ↓
  copy 到 self.num_accepted_tokens.gpu
    ↓
  update_num_computed_tokens_for_batch_change()  ← GPU kernel 修正
    ↓
  修正 num_computed_tokens，丢弃被拒绝 token 的计算
```

**代码位置：**
- 读取：`model_runner_v1.py:1135-1161`
- GPU 修正：`model_runner_v1.py:1178-1185`

### 5.5 `propose_draft_token_ids()` —— 草稿提议

位置：`model_runner_v1.py:1752-2012`

根据不同的 drafter 类型，调用不同的提议逻辑：

```
propose_draft_token_ids()
  ├─ AscendNgramProposer → CPU ngram
  ├─ AscendSuffixDecodingProposer → 后缀匹配
  ├─ AscendNgramProposerNPU → NPU ngram
  ├─ AscendMedusaProposer → Medusa
  ├─ AscendExtractHiddenStatesProposer → 提取隐状态
  └─ Eagle / DraftModel / MTP → 调用 drafter._propose()
```

**草稿模型的输入：** 主模型的 hidden_states + 刚采样出的 token。这就是为什么草稿模型必须在 sample_tokens 里跑——它依赖主模型的输出。

**代码定位：** `model_runner_v1.py:1985`
```python
draft_token_ids = self.drafter._propose(
    target_token_ids=target_token_ids,
    target_hidden_states=target_hidden_states,  # 主模型输出的 hidden_states
    ...
)
```

### 5.6 `_calc_spec_decode_metadata()` —— 推测解码元数据

位置：`model_runner_v1.py:1613-1699`

计算推测解码所需的各种索引：
- `logits_indices`：所有需要计算 logits 的位置
- `target_logits_indices`：目标 token 的 logits 位置
- `bonus_logits_indices`：bonus token 的位置
- `cu_num_draft_tokens` / `cu_num_sampled_tokens`：累积和

---

## 第六章：Ascend 特有功能详解

### 6.1 ACL Graph —— 图执行模式

#### 6.1.1 什么是 ACL Graph

ACL Graph 类似于 NVIDIA 的 CUDA Graph，把一整套 kernel 执行流程录制下来，之后可以直接 replay，减少 kernel launch 开销。

#### 6.1.2 相关代码

- 包装类：`ACLGraphWrapper`（从 `vllm_ascend.compilation.acl_graph` 导入）
- 启用条件：`_use_aclgraph()` → cudagraph_mode != NONE 且 compilation_mode == VLLM_COMPILE
- 参数更新：`update_full_graph_params()` —— 每次 replay 前更新动态参数

#### 6.1.3 `_model_forward()` 中的图执行

位置：`model_runner_v1.py:2952-2988`

```python
def _model_forward(self, num_tokens_padded, input_ids, positions, ...):
    forward_context = get_forward_context()
    
    if self.enable_enpu:
        # ENPU 模式：先更新参数，再执行（软分段要求）
        self._update_full_graph_params_if_needed(...)
        hidden_states = run_model()
    else:
        # 普通模式：先执行，再更新参数
        hidden_states = run_model()
        self._update_full_graph_params_if_needed(...)
    
    return hidden_states
```

### 6.2 EPLB —— 动态专家负载均衡

位置：`model_runner_v1.py:503-528, 2364-2365, 2665-2666`

#### 6.2.1 什么是 EPLB

EPLB (Expert Parallel Load Balancing) 是 Ascend 针对 MoE 模型的动态专家负载均衡技术。当不同 expert 的 workload 不均衡时，动态调整 expert 到不同 NPU 上的分布。

#### 6.2.2 相关成员

| 成员 | 类型 | 作用 |
|------|------|------|
| `self.dynamic_eplb` | bool | 是否启用动态 EPLB |
| `self.eplb_enable` | bool | EPLB 是否启用（动态或静态） |
| `self.eplb_loader` | D2DExpertWeightLoader | D2D 专家权重加载器 |
| `self.eplb_process` | EplbProcess | EPLB 独立进程 |
| `self.eplb_updator` | EplbUpdator | EPLB 更新器 |
| `self.shared_dict` | Manager.dict | 进程间共享字典 |

#### 6.2.3 执行流程中的 hook

```
execute_model() 中：
  ├─ forward 前：self.eplb_updator.forward_before()
  └─ forward 后：self.eplb_updator.forward_end()
```

### 6.3 PCP / DCP —— 上下文并行

#### 6.3.1 为什么需要上下文并行

对于非常长的序列（如 128k tokens），单个 NPU 的算力不够 prefill。PCP 把长序列按 token 维度切分到多个 NPU 上并行计算，最后再 all-gather 回来。

#### 6.3.2 PCPManager

位置：`vllm_ascend/worker/pcp_utils.py`

PCP 相关的操作都委托给 `self.pcp_manager`：
- `init_batch_info()`：初始化批次信息
- `update_tokens_for_pcp()`：为 PCP 切分 token
- `generate_pcp_mtp_input()`：生成 PCP + MTP 的输入
- `get_logits_indices()`：获取 logits 索引
- `get_restore_hidden_states()`：PCP all-gather 后恢复

### 6.4 稀疏注意力 (Sparse Attention / DSA)

#### 6.4.1 相关配置

位置：`model_runner_v1.py:358-396`

- `self.use_sparse`：是否使用稀疏注意力
- `self.sparse_head_dim`：稀疏注意力各 head 的维度
- `self.use_sparse_c8_indexer`：是否启用稀疏 C8 索引器
- `self.c8_k_cache_dtype` / `self.c8_k_scale_cache_dtype`：C8 格式的 K 缓存数据类型

#### 6.4.2 DSA 位置计算

位置：`model_runner_v1.py:2292-2303`

在 execute_model 中，对 DSA（DeepSeek Sparse Attention）单独计算一套 `dsa_positions_np`。

### 6.5 `_torch_cuda_wrapper()` —— CUDA 兼容层

位置：`model_runner_v1.py:5200-5241`

这是一个非常巧妙的设计，让 NPUModelRunner 可以直接继承 GPUModelRunner 而无需大量修改上游代码。

**工作原理：**
1. 进入上下文前，保存原始的 `torch.cuda.*`
2. 进入上下文后，把 `torch.cuda.*` 替换成 `torch.npu.*`
3. 调用父类 `__init__`，此时父类里所有 `torch.cuda.xxx` 调用实际走 NPU
4. 退出上下文时，恢复原始的 `torch.cuda.*`

### 6.6 MTP (Multi-Token Prediction) 支持

MTP 是 DeepSeek V4 的多 token 预测技术，和推测解码配合使用。

相关代码分布在：
- `_prepare_inputs()` 中的 PCP + MTP 输入生成
- `propose_draft_token_ids()` 中的 MTP 隐藏状态获取
- `AscendStep3p5MTPProposer` 类

---

## 第七章：性能优化要点

### 7.1 异步调度 (Async Scheduling)

通过 `execute_model_state` 实现两步执行，让 CPU 和 NPU 流水线化：
- NPU 正在跑第 N 批的前向
- CPU 同时在准备第 N+1 批的输入

### 7.2 异步 D2H / H2D 拷贝

大量使用 `non_blocking=True` 和独立的 copy stream，让数据传输和计算重叠：
- `valid_sampled_token_count_copy_stream`
- `draft_token_ids_copy_stream`
- `async_output_copy_stream`

### 7.3 CUDA Graph / ACL Graph

把固定模式的计算（如 uniform decode）录制成图，减少 kernel launch 开销。
- FULL 模式：整个 forward 都在图里
- 支持动态形状通过参数更新

### 7.4 Pinned Memory

大量使用 `pin_memory=True` 的 CPU 张量，加速 H2D/D2H 拷贝。

### 7.5 序列并行 (Sequence Parallelism)

`_pad_for_sequence_parallelism()` 把 token 数 pad 到 TP 大小的整数倍，启用序列并行时分散计算压力。

### 7.6 减少采样计算量

`enable_reduce_sample` 配置下，先做 Top-K reduce 再采样，减少 logits 维度。

### 7.7 Block Table 预提交

在 `_prepare_inputs()` 一开始就提交 block_table 的 GPU 拷贝，和后面的 CPU 计算重叠。

---

## 第八章：关键数据结构速查

### 8.1 NPUInputBatch

位置：`vllm_ascend/worker/npu_input_batch.py`

管理当前批次的所有请求状态，包括：
- 请求 ID 列表、映射
- token id 矩阵（CPU + GPU）
- num_computed_tokens
- num_prompt_tokens
- block_table
- sampling_metadata
- 等等

### 8.2 AttentionGroup

位置：`vllm_ascend/worker/utils.py`

把配置相同的注意力层分到一组，共享同一份 attention metadata。

### 8.3 AscendCommonAttentionMetadata

位置：`vllm_ascend/attention/utils.py`

Ascend 注意力后端通用的元数据，包含：
- seq_lens / query_start_loc
- block_table / slot_mapping
- max_seq_len
- 等等

---

## 第九章：调用关系图（高层视角）

```
Worker (worker.py)
  │
  ├─► NPUModelRunner.__init__()
  │     └─► GPUModelRunner.__init__()  (via _torch_cuda_wrapper)
  │
  ├─► NPUModelRunner.load_model()
  │     ├─► get_model()
  │     └─► ACLGraphWrapper()  (if enabled)
  │
  ├─► NPUModelRunner.initialize_kv_cache()
  │     ├─► initialize_attn_backend()
  │     └─► initialize_kv_cache_tensors()
  │
  └─► 每个推理 step:
        ├─► execute_model(scheduler_output)
        │     ├─► _update_states()
        │     ├─► _prepare_inputs()
        │     │     ├─► _build_attn_state()
        │     │     ├─► 应用 num_accepted_tokens (修正上轮验证结果)
        │     │     ├─► 计算 positions / input_ids
        │     │     └─► _calc_spec_decode_metadata()
        │     ├─► _build_attention_metadata()
        │     ├─► _preprocess()
        │     ├─► _model_forward()         ← 主模型前向(顺便跑草稿验证)
        │     │     └─► self.model(...)
        │     └─► 保存 execute_model_state
        │
        └─► sample_tokens(grammar_output)
              ├─► _sample()
              │     ├─► AscendSampler (普通采样)
              │     └─► AscendRejectionSampler (验证上轮草稿)
              ├─► _bookkeeping_sync()
              ├─► propose_draft_token_ids()  ← 提议下轮草稿
              └─► 返回 ModelRunnerOutput
```

---

## 第十章：常见问题与理解要点

### 10.1 为什么需要 `execute_model_state`？

因为 vLLM v1 支持异步调度。execute_model 把任务发到 NPU 上就返回了，此时 logits 还没算完。sample_tokens 时再从状态里取出参数，继续处理。

### 10.2 optimistic_seq_lens 是什么？

"乐观"的序列长度，假设上一步的所有草稿 token 都被接受了。这是推测解码中的一个优化：先按乐观情况构建 KV 缓存索引，如果后面发现某些 token 被拒绝了，再修正。

### 10.3 query_start_loc 为什么要 pad？

因为 FIA (Flash Infer Attention) 的 TND 布局要求 hidden_states 的第一维等于 actual_seq_lengths_q 的最后一个元素。padding 是为了满足这个约束。

### 10.4 PCP 和 TP 有什么区别？

- **TP (Tensor Parallel)**：把矩阵乘法的权重切分到多个卡上，每个卡算一部分
- **PCP (Prefill Context Parallel)**：把输入序列按 token 切分，每个卡算一部分 token 的注意力
- PCP 主要用于 prefill 阶段加速长序列

### 10.5 草稿模型和验证都在什么时候执行？

从 prefill 开始的完整时序：

- **Prefill 轮**：`sample_tokens()` 末尾 `propose_draft_token_ids()` 生成**第一批草稿**，返回给调度器
- **Decode 第 1 轮**：`execute_model()` 把第一批草稿带进来跑 → `sample_tokens()` 里 `rejection_sampler` **验证第一批** → 末尾 `propose_draft_token_ids()` 提议**第二批草稿**
- **Decode 第 2 轮**：`execute_model()` 开头读取第一批的 `num_accepted_tokens` **修正序列长度** → 跑第二批草稿 → `sample_tokens()` 里 **验证第二批** → 提议**第三批草稿**
- **Decode 第 3 轮及以后**：以此类推，每轮都做"应用上上轮结果 + 验证上轮草稿 + 提议下轮草稿"

一句话总结：**提议晚 1 轮验证，验证晚 2 轮应用。**

### 10.6 为什么要有 `_torch_cuda_wrapper`？

历史原因：vLLM 上游是为 GPU 写的，大量使用 `torch.cuda.*`。Ascend 为了减少对上游的修改，用 monkey patch 的方式兼容。

---

## 附录：关键文件索引

| 功能 | 文件路径 |
|------|----------|
| NPUModelRunner 主文件 | `vllm_ascend/worker/model_runner_v1.py` |
| 输入批次管理 | `vllm_ascend/worker/npu_input_batch.py` |
| PCP 管理器 | `vllm_ascend/worker/pcp_utils.py` |
| Worker 主类 | `vllm_ascend/worker/worker.py` |
| Ascend 采样器 | `vllm_ascend/sample/sampler.py` |
| 拒绝采样器 | `vllm_ascend/sample/rejection_sampler.py` |
| ACL Graph | `vllm_ascend/compilation/acl_graph.py` |
| 注意力后端 | `vllm_ascend/attention/attention_v1.py` |
| DSA 注意力 | `vllm_ascend/attention/dsa_v1.py` |
| MLA 注意力 | `vllm_ascend/attention/mla_v1.py` |
| EPLB | `vllm_ascend/eplb/` |
| 推测解码 | `vllm_ascend/spec_decode/` |
