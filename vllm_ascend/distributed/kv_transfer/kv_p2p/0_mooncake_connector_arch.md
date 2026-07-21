# MooncakeConnector 从小白到专家：KV缓存传输完全指南

> 📖 **阅读指南**：本文档按照「入门 → 基础 → 进阶 → 高级 → 专家」五个层次组织。小白请从第1章开始按顺序阅读；有经验的同学可以直接跳转到感兴趣的章节。

---

## 目录

### 🟢 入门篇（小白友好）
- [1. 从零开始：vLLM 与 KV 缓存是什么？](#1-从零开始vllm-与-kv-缓存是什么)
- [2. 为什么需要 KV 缓存传输？（一个故事看懂）](#2-为什么需要-kv-缓存传输一个故事看懂)
- [3. Mooncake 是什么？（月饼传输引擎）](#3-mooncake-是什么月饼传输引擎)

### 🔵 基础篇（架构入门）
- [4. 整体架构全景图](#4-整体架构全景图)
- [5. 核心数据结构详解（带形状注解）](#5-核心数据结构详解带形状注解)
- [6. 核心类一览图](#6-核心类一览图)

### 🟠 进阶篇（流程深入）
- [7. KV 缓存注册流程（内存注册是怎么回事）](#7-kv-缓存注册流程内存注册是怎么回事)
- [8. KV 缓存传输全流程（一次传输的完整旅程）](#8-kv-缓存传输全流程一次传输的完整旅程)
- [9. KV 分块元数据计算（最复杂的部分，图解）](#9-kv-分块元数据计算最复杂的部分图解)

### 🔴 高级篇（Ascend 特有）
- [10. Ascend 特有优化详解](#10-ascend-特有优化详解)
- [11. HMA 混合内存分配器支持](#11-hma-混合内存分配器支持)
- [12. Mamba / MLA 特殊支持](#12-mamba--mla-特殊支持)

### ⚫ 专家篇（对比与调优）
- [13. 与上游 vLLM Mooncake 的详细对比](#13-与上游-vllm-mooncake-的详细对比)
- [14. 附录：代码索引与环境变量](#14-附录代码索引与环境变量)

---

## 🟢 入门篇（小白友好）

---

## 1. 从零开始：vLLM 与 KV 缓存是什么？

### 1.1 什么是 vLLM？

vLLM 是一个**快速的大语言模型（LLM）推理引擎**。你可以把它想象成一个「聊天机器人的发动机」——负责高效地运行 AI 模型，生成文本。

**vLLM 的核心优势**：
- **快**：比传统推理框架快很多倍
- **省显存**：通过 PagedAttention 技术高效管理显存
- **支持分布式**：可以把大模型拆到多张显卡/NPU 上运行

### 1.2 什么是 KV 缓存？（超重要！）

这是理解 MooncakeConnector 的**第一关键概念**。

#### 1.2.1 一个生活化的例子

想象你在读一本小说：
- **Prefill（预填充阶段）**：你把书的前 100 页快速读完，理解了故事背景
- **Decode（解码阶段）**：从第 101 页开始，你逐页往后读

在 LLM 推理中也是一样：
- **Prefill 阶段**：一次性处理用户输入的所有 prompt（提示词），计算出中间结果
- **Decode 阶段**：一个字一个字地生成回答

而 **KV 缓存** 就是 Prefill 阶段算出来的「中间结果笔记」，存在显存里，Decode 阶段直接用，不用再重新算了。

#### 1.2.2 技术解释（更准确一点）

在 Transformer 模型的自注意力机制中：
- **K = Key**：键向量，形状 `(num_tokens, num_heads, head_dim)`
- **V = Value**：值向量，形状 `(num_tokens, num_heads, head_dim)`

每次生成新 token 时，都需要之前所有 token 的 K 和 V 来计算注意力。如果每次都重新计算，速度会非常慢。

所以我们把计算好的 K 和 V **缓存**起来，这就是 **KV 缓存**。

#### 1.2.3 KV 缓存的形状

以一个 7B 模型为例：

```
假设：
- 模型层数 num_layers = 32
- 每层注意力头数 num_heads = 32
- 每个头的维度 head_dim = 128
- block_size（每块 token 数）= 16
- 数据类型：bfloat16（2 字节）

单个 block 的 K 缓存大小：
  形状：(block_size, num_heads, head_dim)
  即：  (16, 32, 128)
  
单个 block 的 K 缓存字节数：
  16 × 32 × 128 × 2 = 131,072 字节 ≈ 128KB

32 层的 K+V 缓存：
  32 层 × (K + V) × 128KB = 32 × 2 × 128KB = 8192KB = 8MB per block
```

> 💡 **小白笔记**：KV 缓存很大！一个 7B 模型，每 16 个 token 就需要 8MB 显存。如果有 4096 个 token，就需要 2GB 显存！这就是为什么 KV 缓存管理如此重要。

### 1.3 vLLM 的 PagedAttention

vLLM 用了一个叫 **PagedAttention** 的技术来管理 KV 缓存，灵感来自操作系统的**虚拟内存分页**。

**核心思想**：
- 把 KV 缓存分成固定大小的 **block（页/块）**
- 每个 block 存固定数量的 token（比如 16 个）
- 逻辑上连续的 block，物理上可以不连续（就像虚拟内存）
- 需要的时候分配，不用了释放回收

```
物理显存（不连续的 block）：
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ blk0 │ │ blk1 │ │ blk2 │ │ blk3 │
└──────┘ └──────┘ └──────┘ └──────┘

某个请求的逻辑 KV 缓存（逻辑上连续）：
blk0 → blk2 → blk1 （通过块表映射）
```

---

## 2. 为什么需要 KV 缓存传输？（一个故事看懂）

### 2.1 场景：推测解码（Speculative Decoding）

想象一个工厂流水线：

**传统方式（一个人干）**：
```
工人A：先 Prefill（读 prompt）→ 再 Decode（逐字生成）
```
问题：Prefill 很慢，Decode 也慢，一个人干要等很久。

**推测解码（两个人分工）**：
```
工人P（Prefill 专家）：专门负责 Prefill，算得快
工人D（Decode 专家）：专门负责 Decode，吞吐高
```

但是有个问题：工人P 算出来的 KV 缓存，怎么交给工人D 用？

**答案就是：KV 缓存传输！**

```
┌─────────────┐    KV缓存传输     ┌─────────────┐
│  Prefill 节点 │ ───────────────→ │  Decode 节点  │
│  (工人P)      │                  │  (工人D)      │
└─────────────┘                  └─────────────┘
```

### 2.2 Producer 和 Consumer

在 KV 缓存传输中，有两个角色：

| 角色 | 别称 | 干什么的 | 类比 |
|------|------|---------|------|
| **Producer**（生产者） | Prefill / kv_producer | 生成 KV 缓存，等着被人来取 | 外卖餐厅，做好饭等骑手取 |
| **Consumer**（消费者） | Decode / kv_consumer | 需要 KV 缓存，从 Producer 那边拉过来 | 点外卖的人，等着饭送过来 |

> 💡 **小白笔记**：注意是 Consumer 主动去 Producer 那里「拉」（pull）数据，不是 Producer 主动「推」（push）。这很重要！

### 2.3 为什么不直接用网络传？

你可能会问：直接用 TCP/HTTP 传不行吗？

答案是：**可以，但太慢了！**

KV 缓存的数据量非常大（几百 MB 甚至几 GB），用普通网络传会成为瓶颈。

**Mooncake 的优势**：
- **RDMA / HCCL 直接内存访问**：绕过 CPU，直接从一张卡的显存传到另一张卡
- **零拷贝**：数据不经过多次内存拷贝
- **延迟极低**：微秒级延迟

> 🎯 **一句话总结**：MooncakeConnector 就是帮你把 Prefill 节点算好的 KV 缓存，用最快的速度搬到 Decode 节点去用的「快递员」。

---

## 3. Mooncake 是什么？（月饼传输引擎）

### 3.1 Mooncake 简介

**Mooncake**（月饼）是一个专门为 KV 缓存传输优化的高性能传输引擎。

官方项目：https://github.com/kvcache-ai/Mooncake

**核心能力**：
1. 高性能的点对点（P2P）数据传输
2. 支持多种底层传输协议（RDMA、TCP、HCCL 等）
3. 内存注册机制（注册后才能 DMA 传输）
4. 批量传输优化

### 3.2 Mooncake 的核心概念

#### 3.2.1 TransferEngine（传输引擎）

Mooncake 的核心类，负责实际的数据传输。

**主要工作**：
- 管理底层连接
- 注册内存区域
- 执行传输操作

#### 3.2.2 内存注册（Memory Registration）

这是理解高性能传输的**第二关键概念**。

**为什么要注册内存？**

就像你寄快递，快递员需要知道你家地址才能上门取件。RDMA/HCCL 这种「直接内存访问」技术也一样，需要先把内存地址告诉传输引擎，它才能直接去读写。

**注册过程**：
1. 告诉传输引擎：「这块内存地址是 xxx，大小是 yyy」
2. 传输引擎记录下来，建立映射
3. 之后就可以直接对这块内存进行 DMA 传输了

> ⚠️ **重要限制**：昇腾 HCCL 每进程最多只能注册 **256 个**内存区域！超过了会报错。这就是为什么后面会有「合并注册区域」的优化。

#### 3.2.3 批量传输

Mooncake 支持一次传很多块数据，比一块块传效率高多了。

```
普通传输（100 次调用）：
  传 blk0 → 传 blk1 → ... → 传 blk99

批量传输（1 次调用）：
  批量传 [blk0, blk1, ..., blk99]
```

### 3.3 控制通道 vs 数据通道

KV 传输用了两条独立的通道：

| 通道 | 用途 | 技术 |
|------|------|------|
| **控制通道** | 握手、交换元数据、通知状态 | ZMQ (ZeroMQ) |
| **数据通道** | 实际的 KV 缓存数据传输 | Mooncake TransferEngine (HCCL/RDMA) |

就像打电话：
- 控制通道 = 先拨号、说「喂听得到吗」、确认对方是谁
- 数据通道 = 然后才开始说正事、传大文件

---

## 🔵 基础篇（架构入门）

---

## 4. 整体架构全景图

### 4.1 代码文件地图

在开始之前，先搞清楚我们要讲的代码都在哪里：

```
vllm-ascend/
└── vllm_ascend/
    └── distributed/
        └── kv_transfer/
            ├── kv_p2p/
            │   └── mooncake_connector.py    ← 【主角】本文重点
            └── utils/
                ├── mooncake_transfer_engine.py  ← 传输引擎单例
                └── utils.py                      ← 工具函数

vllm/ （上游）
└── vllm/
    └── distributed/
        └── kv_transfer/
            └── kv_connector/
                └── v1/
                    ├── base.py                 ← 【基类】抽象接口
                    └── mooncake/
                        └── mooncake_connector.py  ← 上游版本
```

### 4.2 三层架构（从上往下看）

```
┌─────────────────────────────────────────────────────────────┐
│  第一层：MooncakeConnector（主连接器）                        │
│  作用：对外提供统一接口，对内调度 Scheduler 和 Worker         │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────┐               ┌─────────────────────┐
│  第二层：Scheduler   │               │  第二层：Worker     │
│  （调度器进程中）     │               │  （工作进程中）      │
│  管规划、管元数据    │               │  管传输、管内存      │
└─────────────────────┘               └─────────────────────┘
                                              │
                            ┌─────────────────┼─────────────────┐
                            ▼                 ▼                 ▼
                      ┌──────────┐    ┌──────────┐    ┌──────────┐
                      │ 发送线程 │    │ 接收线程 │    │GlobalTE  │
                      │ (Producer)│    │(Consumer)│    │(传输引擎)│
                      └──────────┘    └──────────┘    └──────────┘
```

### 4.3 继承关系图

```
KVConnectorBase_V1 （上游定义的抽象基类）
         │
         ├── 定义了 Scheduler 端该有哪些方法
         └── 定义了 Worker 端该有哪些方法

SupportsHMA （混合内存支持接口）
         │
         └── 定义了 HMA 场景下的额外方法

MooncakeConnector （本文主角）
         │
         ├── 继承 KVConnectorBase_V1
         ├── 继承 SupportsHMA
         │
         ├── MooncakeConnectorScheduler （Scheduler 侧实现）
         │   └── 运行在调度器进程里
         │
         └── MooncakeConnectorWorker （Worker 侧实现）
             └── 运行在每个 worker 进程里
                 ├── KVCacheSendingThread （发送线程）
                 └── KVCacheRecvingThread （接收线程）
```

### 4.4 进程/线程关系

```
┌─────────────────────────────────────────────────────┐
│  Scheduler 进程（1个）                               │
│  └── MooncakeConnectorScheduler                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Worker 进程（每个GPU/NPU一个）                       │
│  ├── MooncakeConnectorWorker                        │
│  │   ├── KVCacheSendingThread  (后台线程)            │
│  │   └── KVCacheRecvingThread  (后台线程)            │
│  │       └── ThreadPoolExecutor (32个工作线程)       │
│  └── GlobalTE（传输引擎单例）                         │
└─────────────────────────────────────────────────────┘
```

---

## 5. 核心数据结构详解（带形状注解）

> 💡 **小白笔记**：数据结构是理解代码的钥匙。搞懂这些结构体，代码就看懂了一半。每个结构我都会标注「形状」——也就是张量/数组的维度含义，用 `(dim1, dim2, ...)` 表示。

### 5.1 MooncakeAgentMetadata —— 握手时交换的信息

**定义位置**：`mooncake_connector.py:95`

**作用**：两个节点握手的时候，互相告诉对方自己的配置信息。

```python
class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str                    # 我的引擎ID（身份证号）
    te_rpc_port: int                   # 我的 TransferEngine RPC 端口
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]]
    block_size: int                    # 我的块大小（每块多少token）
    kv_caches_base_addr: list[list[int]]  # 每层KV缓存的基地址
    block_size_scale: list[list[int]]   # 块大小缩放比例
    num_blocks: int                    # 我有多少个逻辑块
    block_lens: list[list[int]]        # 每块的字节长度
    block_strides: list[list[int]]      # 块之间的步长（字节）
    local_ip: str = ""                 # 我的IP地址
    handshake_port: int = 0              # 我的握手端口
```

#### 形状详解

**`kv_caches_base_addr` 的形状：`(num_layers, num_caches_per_layer)`**

- `num_layers`：模型有多少层（比如 32 层）
- `num_caches_per_layer`：每层有几个缓存（通常 K 和 V 两个，所以是 2）

举个例子：
```python
kv_caches_base_addr[0][0] = 0x7f000000  # 第0层K缓存基地址
kv_caches_base_addr[0][1] = 0x7f100000  # 第0层V缓存基地址
kv_caches_base_addr[1][0] = 0x7f200000  # 第1层K缓存基地址
# ...
```

**`block_lens` 的形状：`(num_layers, num_caches_per_layer)`**

每个 block 的字节长度。因为不同层、不同缓存（K/V）的大小可能不一样。

**`block_strides` 的形状：`(num_layers, num_caches_per_layer)`**

相邻两个 block 之间的字节距离。

```
内存布局：
┌────────┐ ← base_addr + 0*stride = block0
│ block0 │
├────────┤ ← base_addr + 1*stride = block1
│ block1 │
├────────┤ ← base_addr + 2*stride = block2
│ block2 │
└────────┘
  stride = 一个 block 的大小 + 可能的 padding
```

> 💡 **为什么 stride 可能比 block 大？** 因为内存对齐！为了高性能访问，内存地址需要对齐到某个边界，中间会有空隙（padding）。

### 5.2 ReqMeta —— 单次请求的传输元数据

**定义位置**：`mooncake_connector.py:109`

**作用**：每次要传输一个请求的 KV 缓存时，把所有相关信息打包成这个结构体。

```python
@dataclass
class ReqMeta:
    local_block_ids: BlockIds          # 本地的块ID列表
    num_external_tokens: int             # 需要传输的外部token数
    num_computed_tokens: int             # 已经计算了多少token
    remote_block_ids: BlockIds         # 远程的块ID列表
    remote_host: str                     # 远程主机地址
    remote_port: int                     # 远程握手端口
    remote_engine_id: str                 # 远程引擎ID
    remote_request_id: str               # 远程请求ID
    remote_pcp_size: int                  # 远程Prefill CP大小
    remote_dcp_size: int                  # 远程Decode CP大小
    remote_ptp_size: int | None         # 远程Prefill TP大小
    remote_multi_nodes_meta_mapping: dict[str, dict[str, Any]]
    num_prompt_blocks: int                # prompt有多少块
    remote_block_size: int                # 远程的块大小
```

> 🤔 **小白提问**：本地块 ID 和远程块 ID 是什么关系？
> 
> **答**：同一个 KV 缓存数据，在 Producer 那边存在 `remote_block_ids` 指向的位置，在 Consumer 这边要放到 `local_block_ids` 指向的位置。传输就是把数据从远程块搬到本地方块。

### 5.3 GroupPull —— 一组缓存的拉取描述

**定义位置**：`mooncake_connector.py:128`

**作用**：描述「拉取某一组 KV 缓存」需要哪些参数。

```python
@dataclass(frozen=True)
class GroupPull:
    group_id: int                        # KV缓存组ID
    remote_tp_offset: int                 # 远程TP偏移量
    num_group_pulls: int                  # 需要拉取几次
    prefill_pp_rank: int = 0            # Prefill的PP秩
    is_group_transfer_end: bool = False   # 是不是组传输的结束标记
```

#### 什么是 KV 缓存组？

你可能会问：KV 缓存不就是 K 和 V 吗？为什么还有「组」的概念？

**答案**：因为有些模型的注意力机制比较特殊，不止 K 和 V 两个缓存。

| 模型类型 | 缓存组数 | 说明 |
|---------|---------|------|
| 普通 Attention | 2 组 | K 和 V |
| Mamba | 多组 | 有各种状态缓存 |
| DeepSeek MLA | 多组 | 多头注意力分解后的多个分量 |

所以用「组（group）」来统一管理不同类型的缓存。

#### 什么是 TP 偏移？

TP = Tensor Parallelism（张量并行），就是把模型拆到多张卡上，每张卡只存一部分。

比如 TP=4，每张卡负责 1/4 的注意力头：
```
卡0：head 0~7
卡1：head 8~15
卡2：head 16~23
卡3：head 24~31
```

Consumer 的卡 0 要从 Producer 的卡 0 拉取 head 0~7 的数据，这个偏移就是 0。

> 💡 **GQA 场景下的 num_group_pulls**：
> 
> GQA（Grouped Query Attention）中，K/V 的头数比 Q 少。比如 Q 有 32 头，K/V 只有 8 头。
> 
> 这时候 Consumer 的一张卡可能需要从 Producer 的多张卡拉数据，所以要拉多次，这就是 `num_group_pulls`。

### 5.4 KVCacheTaskTracker —— 任务跟踪器

**定义位置**：`mooncake_connector.py:164`

**作用**：跟踪哪些请求传完了，哪些还在传，防止内存泄漏。

```python
class KVCacheTaskTracker:
    finished_requests: set   # 已经传完的请求
    delayed_free_requests: deque  # 延迟释放的请求（防止有人还在用）
    reqs_to_process: set    # 待处理的请求
```

#### 为什么要有延迟释放？

想象这个场景：
1. KV 缓存传输完成了
2. 通知 Scheduler：传完啦！
3. Scheduler 说：好的，我这边释放块了
4. 但是 Worker 这边可能还有地方在引用这块内存

如果立刻释放，可能会导致「释放后还在使用」的 bug。

所以延迟一会儿再释放，等大家都不用了再真的释放。

#### 超时强制释放

还有个保护机制：如果请求超过 `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` 秒还没完成，就强制释放，防止内存泄漏。

---

## 6. 核心类一览图

### 6.1 类图总览

```
┌──────────────────────────────────────────────────────────────┐
│                     MooncakeConnector                         │
│  （主类，对外统一接口，根据角色创建 Scheduler/Worker）          │
├──────────────────────────────────────────────────────────────┤
│  + __init__(vllm_config, role, kv_cache_config)               │
│  + get_num_new_matched_tokens()  ← Scheduler 方法              │
│  + update_state_after_alloc()       ← Scheduler 方法           │
│  + build_connector_meta()         ← Scheduler 方法             │
│  + request_finished()             ← Scheduler/Worker 方法       │
│  + request_finished_all_groups()  ← HMA 方法                    │
│  + register_kv_caches()           ← Worker 方法                 │
│  + start_load_kv()                ← Worker 方法                 │
│  + wait_for_layer_load()          ← Worker 方法（空实现）        │
│  + save_kv_layer()                ← Worker 方法（空实现）        │
│  + wait_for_save()                ← Worker 方法                 │
│  + get_finished()                 ← Worker 方法                 │
│  + get_block_ids_with_load_errors() ← Worker 方法              │
└──────────────────────────────────────────────────────────────┘
           │
    ┌──────┴───────┐
    ▼              ▼
┌───────────┐  ┌───────────────┐
│ Scheduler │  │    Worker     │
│  端实现    │  │    端实现      │
└───────────┘  └───────────────┘
```

### 6.2 MooncakeConnectorScheduler（调度器端）

**定义位置**：`mooncake_connector.py:1523`

**运行位置**：Scheduler 进程（只有一个）

**核心职责**：

1. **握手元数据管理**：记录各个 Worker 的地址、端口等信息
2. **传输规划**：计算哪个块该从哪里传到哪里
3. **请求生命周期**：请求开始、结束、出错时的状态更新

**主要方法**：

| 方法 | 干什么 |
|------|--------|
| `get_num_new_matched_tokens()` | 获取新匹配了多少 token |
| `update_state_after_alloc()` | 分配 KV 块后更新状态 |
| `build_connector_meta()` | 构建连接器元数据（给别人用的） |
| `request_finished()` | 某个请求完成了，做清理 |
| `request_finished_all_groups()` | 所有组的请求都完成了（HMA） |

### 6.3 MooncakeConnectorWorker（工作端）

**定义位置**：`mooncake_connector.py:1860`

**运行位置**：每个 Worker 进程（每张 GPU/NPU 一个）

**核心职责**：

1. **KV 缓存注册**：把本地 KV 缓存注册到 Mooncake 引擎
2. **发送管理**：Producer 角色时，响应别人的拉取请求
3. **接收管理**：Consumer 角色时，主动去拉取别人的 KV
4. **传输后处理**：数据传过来后，做格式转换

**核心属性**：

```python
class MooncakeConnectorWorker:
    engine: TransferEngine                # Mooncake传输引擎
    kv_send_thread: KVCacheSendingThread  # 发送线程（Producer）
    kv_recv_thread: KVCacheRecvingThread  # 接收线程（Consumer）
    kv_caches_base_addr: list[list[int]]  # KV缓存基地址
    block_len_per_addr: list[list[int]]   # 每块字节长度
    block_stride_per_addr: list[list[int]]  # 块间步长
    kv_group2layeridx: dict               # KV组到层索引的映射
```

> 💡 **注意**：MooncakeConnector 是整体传输，不是逐层传。所以 `save_kv_layer()`、`wait_for_layer_load()` 这些逐层的方法都是空的（pass）。

### 6.4 KVCacheSendingThread（发送线程）

**定义位置**：`mooncake_connector.py:243`

**角色**：Producer 侧的后台线程

**工作模式**：
- 用 ZMQ 的 ROUTER socket 监听端口
- 等着 Consumer 来发请求
- 收到请求后响应

**能处理的请求类型**：

| 请求消息 | 含义 | 响应 |
|---------|------|------|
| `GET_META_MSG` | 「你的元数据是什么？」 | 返回 MooncakeAgentMetadata |
| `DONE_RECVING_MSG` | 「我接收完了，你可以释放了」 | 更新任务跟踪器 |

**端口怎么算的？**

每个设备秩都有自己的握手端口：

```python
device_index = (pp_rank * pcp_size + pcp_rank) * tp_size + tp_rank
handshake_port = side_channel_port + device_index
```

> 💡 小白可能看不懂：PP、TP、PCP 这些是什么？
> 
> 别着急，第 9 章会详细讲！你现在只需要知道：每张卡有个唯一的编号，端口号就是基础端口 + 这个编号。

### 6.5 KVCacheRecvingThread（接收线程）

**定义位置**：`mooncake_connector.py:407`

**角色**：Consumer 侧的后台线程

**工作模式**：
- 维护一个请求队列
- 把请求按对端分组
- 用线程池并发处理

**内部架构**：

```
request_queue（请求队列，所有要传的请求都在这里排队）
    │
    ▼
_submit_request()  提交请求
    │
    ▼
peer_request_queues（按对端分组的队列，每个对端一个队列）
    │
    ▼
ThreadPoolExecutor（32个工作线程，并发处理）
    │
    ▼
_handle_peer_requests()  处理某个对端的请求
    │
    ▼
_handle_request()  处理单个请求
    │
    ▼
_transfer_kv_cache_all_groups()  真正传输所有KV组 ← 【核心】
```

**为什么要按对端分组？**

因为如果有 100 个请求，都是从同一个对端拉数据，全堆在一起可能会把对端压垮。分组后：
- 每个对端每次最多处理 `MAX_REQUESTS_PER_PEER_HANDLER = 5` 个请求
- 防止一个对端占满所有资源，饿死其他对端

---

## 🟠 进阶篇（流程深入）

---

## 7. KV 缓存注册流程（内存注册是怎么回事）

### 7.1 为什么要注册？

前面讲过，RDMA/HCCL 要直接访问内存，必须先注册。就像你去健身房，得先办会员卡，然后才能用器材。

**注册 = 办会员卡**：
- 告诉传输引擎：「这块内存归我管，地址是 xxx，大小是 yyy」
- 传输引擎记录下来，建立映射
- 之后传输引擎可以直接读写这块内存

### 7.2 注册流程全景图

**入口函数**：`MooncakeConnectorWorker.register_kv_caches()`  
**位置**：`mooncake_connector.py:2190`

```
┌─────────────────────────────────────────────────────────┐
│             register_kv_caches()                       │
│              （注册KV缓存到传输引擎）                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1️⃣  构建 kv_group2layeridx 映射                       │
│     └─ 给每个KV缓存组分配对应的层索引                   │
│                                                         │
│  2️⃣  遍历每层KV缓存，收集元数据                       │
│     ├─ kv_caches_base_addr[layer][cache] ← 基地址      │
│     ├─ block_len_per_addr[layer][cache] ← 块大小       │
│     ├─ block_stride_per_addr[layer][cache] ← 步长      │
│     └─ block_size_scale[layer][cache] ← 缩放比例       │
│                                                         │
│  3️⃣  计算需要注册的内存区域                           │
│     ┌─────────────────────────────────────────┐        │
│     │ 不同类型的组，计算方式不同：              │        │
│     │  - Mamba 组 → 特殊计算方式                │        │
│     │  - Hybrid 组 → HMA混合内存方式           │        │
│     │  - 普通组 → 按storage合并后注册           │        │
│     └─────────────────────────────────────────┘        │
│                                                         │
│  4️⃣  检查：注册区域数 ≤ 256（HCCL限制）              │
│     └─ 超了就报错，提示用户减少注册区域               │
│                                                         │
│  5️⃣  调用 global_te.register_buffer() 注册内存       │
│     └─ 传输引擎：好的，这些内存我记下了                │
│                                                         │
│  6️⃣  构建 MooncakeAgentMetadata（我的握手信息）      │
│                                                         │
│  7️⃣  启动后台线程                                    │
│     ├─ Producer角色 → 启动 KVCacheSendingThread       │
│     └─ Consumer角色 → 启动 KVCacheRecvingThread       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 7.3 内存注册区域优化：合并相邻区域

**问题**：HCCL 每进程最多只能注册 256 个区域。如果模型有 32 层，每层 K+V 两个缓存，每个缓存又分成很多段... 256 可能不够用！

**解决方案**：把相邻的内存区域合并成一个大区域注册。

```
合并前（5个区域）：
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│ reg0 │   │ reg1 │   │ reg2 │   │ reg3 │   │ reg4 │
└──────┘   └──────┘   └──────┘   └──────┘   └──────┘
  ↑ 每个小区域都要注册，浪费名额

合并后（1个区域）：
┌─────────────────────────────────────────────────────┐
│                  合并后的大区域                        │
└─────────────────────────────────────────────────────┘
  ↑ 只需要注册1次，节省名额
```

**合并规则**：
- 属于同一个 storage（同一块大显存分配出来的）才能合并
- 中间的间隙小于 `REGISTER_MERGE_GAP_BYTES = 4096` 字节才合并
- 合并后整体注册，使用的时候按偏移量访问

> 🤔 **合并了会不会不安全？**
> 
> 不会。注册只是告诉传输引擎「这块内存你可以访问」，具体读写哪些位置还是由传输指令精确控制的。合并只是为了节省注册名额。

### 7.4 注册验证

注册区域计算好后，会调用 `validate_register_region_count()` 检查数量：

- 如果 ≤ 256：没问题，继续
- 如果 > 256：报错，告诉用户需要调整

报错信息会告诉你：
- 当前有多少个注册区域
- 各个组分别有多少个
- 建议怎么优化

---

## 8. KV 缓存传输全流程（一次传输的完整旅程）

### 8.1 传输流程图

**核心函数**：`KVCacheRecvingThread._transfer_kv_cache_all_groups()`  
**位置**：`mooncake_connector.py:730`

这是整个文件最核心的函数之一！

```
┌─────────────────────────────────────────────────────────┐
│       _transfer_kv_cache_all_groups(req_meta)          │
│              （传输所有KV缓存组）                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1️⃣  检查有没有远程元数据缓存                          │
│     ├─ 有 → 直接用                                     │
│     └─ 没有 → 用ZMQ发GET_META_MSG问对方要              │
│                                                         │
│  2️⃣  遍历每个 GroupPull（每组缓存拉一次）              │
│     │                                                   │
│     ├─ a. 获取这个组的规格和层索引                      │
│     │                                                   │
│     ├─ b. 过滤 PP 层范围                                │
│     │   （PP=流水线并行，只传我负责的那些层）            │
│     │                                                   │
│     ├─ c. 扩展/分组块ID                                 │
│     │   （逻辑块 → 内核块）                            │
│     │                                                   │
│     └─ d. 构建三个列表：                               │
│          ├─ src_list    → 源地址列表                   │
│          ├─ dst_list    → 目标地址列表                 │
│          └─ length_list → 长度列表                     │
│                                                         │
│  3️⃣  调用 Mooncake 引擎批量传输                       │
│     └─ engine.batch_transfer_sync_read(               │
│          remote_engine_id,                             │
│          src_list, dst_list, length_list               │
│        )                                               │
│     （这一步是真正的数据传输，用HCCL/RDMA）              │
│                                                         │
│  4️⃣  传输后处理：KV缓存重格式化                       │
│     ├─ GQA 重排：把多分片合并成完整head维度            │
│     ├─ NZ格式转换：适配NPU的NZ存储格式                │
│     └─ Hybrid Linear重排：HMA混合内存布局适配          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 8.2 传输地址怎么算？（重点！）

这是理解传输的关键：**源地址和目标地址是怎么算出来的？**

```python
src = src_layer_base_addr + local_block_id * block_stride + tp_offset * inner_block_len
dst = dst_layer_base_addr + remote_block_id * remote_block_stride
length = inner_block_len * num_blocks
```

#### 图解说明

```
Producer 端的内存（某层K缓存）：
┌─────────────────────────────────────────────┐
│                  base_addr                   │ ← 基地址
├─────────────────────────────────────────────┤
│  block 0  │  block 1  │  block 2  │  ...    │
├───────────┼───────────┼───────────┤         │
│ TP0 │ TP1 │ TP0 │ TP1 │ TP0 │ TP1 │         │ ← TP分片
└─────┴─────┴─────┴─────┴─────┴─────┴─────────┘
  ↑          ↑
  │          └─ block_stride = 一个完整block的字节大小
  └─ inner_block_len = 一个TP分片的字节大小
     tp_offset = 第几个TP分片（0,1,2,...）
```

**举个具体例子**（数字都是编的，方便理解）：

假设：
- 基地址 base_addr = 0x1000
- block_stride = 1024 字节（每个完整 block 占 1024 字节）
- inner_block_len = 512 字节（TP=2，所以每个分片 512 字节）
- 要传 block_id = 2
- tp_offset = 1（取第二个 TP 分片）

计算：
```
src = 0x1000 + 2 * 1024 + 1 * 512
    = 0x1000 + 2048 + 512
    = 0x1000 + 2560
    = 0x1A00
```

> 💡 **小白笔记**：地址计算就是「基地址 + 第几个块 + 第几个TP分片」。就像找座位：几号厅（基地址）+ 第几排（块号×每排长度）+ 第几个座位（TP偏移）。

### 8.3 传输后为什么要重格式化？

数据传过来了，为什么不能直接用？还要重排？

**答案**：因为 Producer 和 Consumer 的 KV 缓存布局可能不一样！

常见的不一致情况：

| 情况 | 为什么不一样 | 怎么处理 |
|------|-------------|---------|
| **GQA 分组** | Consumer 的 TP 分组方式和 Producer 不一样 | 重新排列 head 维度 |
| **NZ 格式** | NPU 上用 NZ 格式存储更高效 | 转换格式 |
| **HMA 布局** | 混合内存分配器的特殊布局 | 重排维度顺序 |

#### 例子：Hybrid Linear 重排

HMA（混合内存分配器）下，KV 缓存的维度顺序是：
```
传输后（Producer的布局）：
[block, split, token, head_per_split, dim]

重排后（Consumer需要的布局）：
[block, token, split, head_per_split, dim]
```

就是把 `token` 维度和 `split` 维度换个位置。

> 🤔 **为什么不直接用一样的布局？**
> 
> 因为 Producer 和 Consumer 可能用了不同的配置（比如 TP 大小不同、内存分配策略不同）。而且传输的时候按连续内存传最快，传完了再调整成本地需要的格式。

---

## 9. KV 分块元数据计算（最复杂的部分，图解）

> ⚠️ **警告**：这是整个文件最复杂的部分！建议先喝口水再看。
> 
> 如果你只需要理解基本原理，可以先跳过这章，等遇到问题再回来。

### 9.1 为什么需要分块元数据计算？

简单场景下（没有上下文并行），事情很简单：
- Consumer 的卡 0 从 Producer 的卡 0 拉数据
- 一一对映

但是一旦引入了 **CP（Context Parallel，上下文并行）**，事情就复杂了：

```
Prefill 端有 PCP=4（4个CP分片）
Decode  端有 DCP=2（2个CP分片）
```

这时候：
- 块怎么对应？
- 从哪个端口拉？
- 哪部分数据从哪拉？

这就是 `_get_kv_split_metadata()` 要解决的问题。

### 9.2 简单场景：无 CP（PCP=1, DCP=1）

先看简单的，建立信心。

**函数**：`_get_kv_split_metadata()`  
**位置**：`mooncake_connector.py:2518`

当 PCP=1 且 DCP=1 时：

```python
chosen_rank_list = _get_remote_rank(req_id, prefill_tp_size)
remote_handshake_port_list = [[x + meta.remote_port for x in chosen_rank_list]]
```

就是：
1. 选一个远程 TP 秩（按请求ID轮询或者哈希）
2. 端口 = 远程基础端口 + TP 秩号

很简单，对吧？

### 9.3 复杂场景：有 CP（PCP/DCP > 1）

好，现在进入深水区。

#### 9.3.1 什么是上下文并行（CP）？

上下文并行就是把「上下文（token 序列）」切成多份，分给不同的卡算。

比如 4096 个 token，CP=4：
- 卡0算 token 0~1023
- 卡1算 token 1024~2047
- 卡2算 token 2048~3071
- 卡3算 token 3072~4095

每张卡只算一部分，算完了拼起来。

#### 9.3.2 PCP 和 DCP 是什么？

- **PCP** = Prefill Context Parallel size（Prefill 端的 CP 大小）
- **DCP** = Decode Context Parallel size（Decode 端的 CP 大小）

**注意**：PCP 和 DCP 可以不一样！
- Prefill 端可能用 4 个 CP
- Decode 端可能用 2 个 CP

这就导致了块的映射关系非常复杂。

#### 9.3.3 Head Group 和 CP Group

KV 缓存还有个「head group」的概念（GQA 里的）。

映射关系：
```
每个 head group → 对应一个 CP group
```

#### 9.3.4 计算流程概览

```
┌─────────────────────────────────────────────────────────┐
│         _get_kv_split_metadata()                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1️⃣  构建 KV head group → CP group 映射               │
│                                                         │
│  2️⃣  遍历每个 head group                               │
│     ├─ 计算本地块分布 [dcp_rank][pcp_rank]             │
│     ├─ 计算远程端口映射                                 │
│     ├─ 处理前缀缓存命中                                 │
│     └─ 处理块大小不匹配（需要切割/合并）               │
│                                                         │
│  3️⃣  组装结果返回                                     │
└─────────────────────────────────────────────────────────┘
```

### 9.4 Group Pull 元数据计算

**函数**：`_get_group_pulls_metadata()`  
**位置**：`mooncake_connector.py:2962`

这个函数给每个 KV 缓存组生成一组 `GroupPull`，描述「这组缓存该怎么拉」。

要算的东西：
- `group_id`：拉哪个组
- `remote_tp_offset`：从远程的哪个 TP 分片开始拉
- `num_group_pulls`：要拉几次（因为 GQA，可能要从多分片拉）
- `prefill_pp_rank`：Prefill 的 PP 秩（流水线并行分层）

### 9.5 一张图总结并行策略

```
并行策略的维度：
┌─────────────────────────────────────────────────────┐
│  PP（流水线并行）：分层，每层在不同的卡上             │
│  比如 PP=2：前16层在卡0，后16层在卡1                  │
├─────────────────────────────────────────────────────┤
│  TP（张量并行）：分头，每个头在不同的卡上             │
│  比如 TP=4：32个头分成4份，每份8个头                  │
├─────────────────────────────────────────────────────┤
│  CP（上下文并行）：分token，每段在不同的卡上          │
│  比如 CP=2：前半段token在卡0，后半段在卡1             │
└─────────────────────────────────────────────────────┘

三个维度可以组合使用：
PP × TP × CP = 总卡数

比如 PP=2, TP=4, CP=2 → 总共需要 2×4×2 = 16 张卡
```

> 💡 **小白笔记**：如果你暂时搞不懂 CP 也没关系。大部分场景下 PCP=DCP=1，这部分逻辑不会触发。你只要知道「有这个东西，很复杂」就够了，真遇到问题再回来研究。

---

## 🔴 高级篇（Ascend 特有）

---

## 10. Ascend 特有优化详解

### 10.1 GlobalTE：传输引擎单例

**文件**：`vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py`

#### 为什么要用单例？

一个进程里只需要一个 TransferEngine 实例就够了。如果创建多个，会：
- 浪费资源
- 可能冲突
- 内存注册重复

所以用**单例模式**，保证全局只有一个实例。

#### 单例是怎么实现的？

```python
class GlobalTE:
    transfer_engine: TransferEngine | None = None
    is_register_buffer: bool = False

    @classmethod
    def get_instance(cls):
        if cls.transfer_engine is None:
            # 双重检查锁定，线程安全
            cls.transfer_engine = TransferEngine(...)
        return cls.transfer_engine
```

**特点**：
1. **延迟初始化**：第一次用的时候才创建，不用不创建
2. **线程安全**：用锁保证多线程下也只创建一个
3. **注册幂等**：内存注册只做一次，重复调用直接返回

### 10.2 HCCL 注册区域限制（256 限制）

**常量**：`MAX_HCCL_REGISTER_REGIONS = 256`

#### 问题背景

昇腾的 HCCL 通信库有个限制：**每个进程最多注册 256 个内存区域**。

如果超过了，注册会失败，进而导致传输失败。

#### 解决方案

1. **合并相邻区域**：第 7 章讲过的，把能合并的合并起来
2. **注册前检查**：`validate_register_region_count()` 提前检查
3. **报错提示**：如果超了，告诉用户怎么优化

### 10.3 NPU 内存对齐

**函数**：`align_memory()`  
**位置**：`utils.py:48`

```python
def align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    data_ptr = tensor.data_ptr()
    aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
    offset = (aligned_addr - data_ptr) // tensor.element_size()
    return tensor[int(offset):]
```

#### 为什么要对齐？

NPU 的 DMA 传输对内存地址有对齐要求。如果地址不对齐，可能：
- 传输变慢
- 甚至直接报错

**对齐方式**：向上取整到最近的对齐边界。

```
不对齐的地址：0x1234（假设对齐到 4096）
对齐后地址：0x2000
浪费的空间：0x2000 - 0x1234 = 0xDCC 字节
```

> 💡 为了性能，浪费一点空间是值得的。

### 10.4 传输超时计算

**函数**：`get_transfer_timeout_value()`  
**位置**：`utils.py:55`

#### 超时时间怎么算的？

```python
timeout = (4.096 * 2^hccl_rdma_timeout) * hccl_rdma_retry_cnt // 1000 + 3000
```

单位：毫秒

#### 环境变量

| 环境变量 | 作用 | 默认值 |
|---------|------|--------|
| `ASCEND_TRANSFER_TIMEOUT` | 直接指定超时时间（毫秒） | 不设置则自动计算 |
| `HCCL_RDMA_TIMEOUT` | HCCL RDMA 超时指数 | 20 |
| `HCCL_RDMA_RETRY_CNT` | HCCL RDMA 重试次数 | 7 |

#### 自动计算的例子

用默认值算一下：
```
timeout = (4.096 * 2^20) * 7 // 1000 + 3000
        = (4.096 * 1048576) * 7 // 1000 + 3000
        = 4294967.296 * 7 // 1000 + 3000
        ≈ 30064771 // 1000 + 3000
        ≈ 30064 + 3000
        ≈ 33064 毫秒 ≈ 33 秒
```

> 💡 **为什么加 3000 毫秒？** 留一点余量，防止刚好卡在超时边界。

### 10.5 torch_npu 操作优化

昇腾版本大量使用 `torch_npu` 库进行 NPU 加速操作，比如：
- NPU 上的张量操作
- 融合算子（fused op）加速
- NZ 格式转换

---

## 11. HMA 混合内存分配器支持

### 11.1 什么是 HMA？

HMA = Hybrid Memory Allocator（混合内存分配器）。

**核心思想**：把 KV 缓存分成多份，存在不同的内存区域（比如一部分在高速内存，一部分在低速内存），以平衡容量和速度。

#### 为什么需要 HMA？

- NPU 上的高速内存（类似显存）很贵，容量有限
- 但是 KV 缓存又特别占空间
- 所以把热数据放高速内存，冷数据放慢速内存，性价比最高

### 11.2 MooncakeConnector 怎么支持 HMA？

MooncakeConnector 继承了 `SupportsHMA` 接口，提供 HMA 场景下的特殊支持。

#### 特殊支持点：

1. **`request_finished_all_groups()` 方法**
   - 普通场景：一个请求完成了就释放
   - HMA 场景：要等所有组的请求都完成了才释放（因为数据分散在多个组）

2. **混合内存布局的 KV 缓存重排**
   - 传输后需要特殊的重格式化
   - 函数：`reformat_kv_cache_hybrid_linear_torch()`

3. **注册区域计算的特殊逻辑**
   - 普通组用 `collect_storage_merged_register_regions()`
   - Hybrid 组用 `_get_registered_kv_tensor_buffers_hybrid()`

#### Hybrid Linear 重排详解

**函数**：`reformat_kv_cache_hybrid_linear_torch()`  
**位置**：`mooncake_connector.py:984`

HMA 下 KV 缓存的形状：

```
Producer 端布局（传输过来的样子）：
[block, split, token, head_per_split, dim]

Consumer 端需要的布局：
[block, token, split, head_per_split, dim]
```

就是把 `split` 和 `token` 两个维度换个位置。

> 💡 **split 是什么？** split 就是 HMA 里的「分片」，不同分片可能存在不同的内存区域。

---

## 12. Mamba / MLA 特殊支持

### 12.1 Mamba 支持

#### 什么是 Mamba？

Mamba 是一种基于结构化状态空间模型（SSM）的架构，不像传统 Transformer 那样用注意力。

**Mamba 的特点**：
- 没有 KV 缓存的概念
- 但是有各种状态（state）需要缓存
- 状态的形状、数量都和普通 Attention 不一样

#### MooncakeConnector 怎么支持 Mamba？

1. **Mamba 组的特殊识别**
   - 检测到是 Mamba 组，走不同的逻辑分支

2. **Mamba 状态的注册方式不同**
   - 普通 Attention：按层、按 K/V 注册
   - Mamba：按状态类型注册
   - 函数：`_get_registered_kv_tensor_buffers()`

3. **传输时的特殊处理**
   - Mamba 状态的形状、步长都不一样
   - 需要单独计算

### 12.2 DeepSeek MLA 支持

#### 什么是 MLA？

MLA = Multi-head Latent Attention（多头潜在注意力），是 DeepSeek 提出的一种注意力机制。

**MLA 的特点**：
- 把 KV 缓存分解成多个分量
- 缓存组数比普通 Attention 多
- 形状更复杂

#### MooncakeConnector 怎么支持 MLA？

1. **MLA 多组缓存支持**
   - `kv_group2layeridx` 映射里有多个组
   - 每个组对应 MLA 的一个分量

2. **组传输的协调**
   - 多个组的传输要协调好
   - 确保所有组都传完了才算完成

3. **`is_group_transfer_end` 标记**
   - GroupPull 里的这个字段用来标记「这是组传输的最后一个」
   - 收到这个标记才知道所有组都传完了

---

## ⚫ 专家篇（对比与调优）

---

## 13. 与上游 vLLM Mooncake 的详细对比

### 13.1 架构对比表

| 特性 | 上游 vLLM Mooncake | vllm-ascend Mooncake |
|------|---------------------|----------------------|
| **基类** | `KVConnectorBase_V1` | `KVConnectorBase_V1` + `SupportsHMA` |
| **传输方向** | 发送/接收分离 | 发送/接收分离 |
| **并行支持** | TP + PP | TP + PP + PCP + DCP |
| **HMA 支持** | 基础 | 完整深度集成 |
| **Mamba 支持** | ❌ 不支持 | ✅ 支持 |
| **MLA 支持** | ❌ 不支持 | ✅ 支持（DeepSeek MLA） |
| **NZ 格式** | ❌ 不支持 | ✅ 支持（NPU特有） |
| **服务发现** | etcd / bootstrap server | ZMQ 直接握手 |
| **指标系统** | Prometheus HTTP 指标 | 基础指标 |
| **拓扑抽象** | TransferTopology | 无（直连） |

### 13.2 上游 vLLM 的优势

1. **更通用**：支持多种后端、多种传输协议
2. **更完善的可观测性**：Prometheus 指标、分层监控
3. **服务发现机制**：etcd 或 bootstrap server，更灵活
4. **拓扑感知**：TransferTopology 感知网络拓扑，优化传输路径

### 13.3 vllm-ascend 的扩展特性

昇腾版本在上游基础上做了大量扩展：

| 扩展特性 | 说明 |
|---------|------|
| **PCP/DCP 上下文并行** | Prefill 和 Decode 端都支持上下文并行，且大小可不同 |
| **HMA 深度集成** | 混合内存分配器的完整支持，包括特殊注册、重排、生命周期管理 |
| **Mamba 状态传输** | 支持 Mamba 模型的状态缓存传输 |
| **DeepSeek MLA** | 支持 DeepSeek MLA 注意力的多组分片传输 |
| **NPU NZ 格式** | 支持 NPU 特有的 NZ 存储格式，提升注意力计算效率 |
| **HCCL 注册优化** | 256 区域限制的检测和合并优化 |
| **多节点元数据映射** | `remote_multi_nodes_meta_mapping` 支持复杂多节点场景 |
| **torch_npu 融合算子** | 用 NPU 融合算子加速重格式化等操作 |

### 13.4 代码风格差异

| 方面 | 上游 vLLM | vllm-ascend |
|------|-----------|-------------|
| **类型注解** | 完善 | 基本完善 |
| **注释量** | 较多 | 较少 |
| **抽象层次** | 更高（更通用） | 更具体（更偏向NPU优化） |
| **配置方式** | 环境变量 + 配置对象 | 环境变量为主 |

---

## 14. 附录：代码索引与环境变量

### 14.1 核心类位置速查表

| 类名 | 文件 | 行号 |
|------|------|------|
| `MooncakeAgentMetadata` | `mooncake_connector.py` | 95 |
| `ReqMeta` | `mooncake_connector.py` | 109 |
| `GroupPull` | `mooncake_connector.py` | 128 |
| `KVCacheTaskTracker` | `mooncake_connector.py` | 164 |
| `KVCacheSendingThread` | `mooncake_connector.py` | 243 |
| `KVCacheRecvingThread` | `mooncake_connector.py` | 407 |
| `MooncakeConnectorMetadata` | `mooncake_connector.py` | 1374 |
| `MooncakeConnector` | `mooncake_connector.py` | 1405 |
| `MooncakeConnectorScheduler` | `mooncake_connector.py` | 1523 |
| `MooncakeConnectorWorker` | `mooncake_connector.py` | 1860 |
| `GlobalTE` | `mooncake_transfer_engine.py` | - |

### 14.2 关键函数位置速查表

| 函数 | 文件 | 行号 |
|------|------|------|
| `register_kv_caches` | `mooncake_connector.py` | 2190 |
| `_get_kv_split_metadata` | `mooncake_connector.py` | 2518 |
| `_get_group_pulls_metadata` | `mooncake_connector.py` | 2962 |
| `_transfer_kv_cache_all_groups` | `mooncake_connector.py` | 730 |
| `reformat_kv_cache` | `mooncake_connector.py` | - |
| `reformat_kv_cache_hybrid_linear_torch` | `mooncake_connector.py` | 984 |
| `collect_storage_merged_register_regions` | `utils.py` | 363 |
| `validate_register_region_count` | `utils.py` | 428 |
| `get_transfer_timeout_value` | `utils.py` | 55 |
| `align_memory` | `utils.py` | 48 |

### 14.3 环境变量大全

| 环境变量 | 作用 | 默认值 |
|---------|------|--------|
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` | 请求超时强制释放时间（秒） | - |
| `ASCEND_TRANSFER_TIMEOUT` | 昇腾传输超时（毫秒），设了就用这个值 | 自动计算 |
| `HCCL_RDMA_TIMEOUT` | HCCL RDMA 超时指数（用于自动计算超时） | 20 |
| `HCCL_RDMA_RETRY_CNT` | HCCL RDMA 重试次数（用于自动计算超时） | 7 |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | 是否用融合算子逐块转置 KV 缓存 | - |

### 14.4 常量大全

| 常量 | 值 | 说明 |
|------|-----|------|
| `MAX_REQUESTS_PER_PEER_HANDLER` | 5 | 每个对端每次处理的请求数，防止饿死 |
| `MAX_HCCL_REGISTER_REGIONS` | 256 | HCCL 最大注册区域数 |
| `REGISTER_MERGE_GAP_BYTES` | 4096 | 内存合并的间隙阈值（字节） |

---

## 🎓 恭喜你！

如果你读到这里，并且大致看懂了，那么你已经从一个 MooncakeConnector 小白，变成了一位 MooncakeConnector 专家！🎉

**接下来你可以**：
- 回到代码里，对照文档再读一遍 `mooncake_connector.py`，你会发现很多之前看不懂的地方现在都清晰了
- 看看 `utils.py` 和 `mooncake_transfer_engine.py`，了解更多细节
- 研究上游 vLLM 的版本，对比差异
- 动手调试，加深理解

> 💡 **学习建议**：好的代码是最好的老师。带着问题去读代码，边读边画流程图，进步最快！
