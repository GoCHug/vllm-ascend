# MooncakeConnector 从小白到专家：KV缓存传输完全指南

> 📖 **阅读指南**
> 
> 本文档按照**六层递进结构**组织，从入门到精通：
> 
> | 层次 | 篇章 | 适合人群 | 核心内容 |
> |------|------|---------|---------|
> | 第1层 | 🟢 入门篇 | 完全小白 | KV缓存是什么、为什么要传输 |
> | 第2层 | 🔵 架构篇 | 想了解全貌 | 文件地图、数据结构、类体系 |
> | 第3层 | 🟠 核心流程篇 | 想搞懂原理 | 注册流程、传输流程、并行策略 |
> | 第4层 | 🔴 Ascend优化篇 | 昇腾开发者 | NPU特有优化、HMA、特殊模型 |
> | 第5层 | 🟣 上游扩展篇 | 想对比学习 | 上游vLLM实现、Store模式 |
> | 第6层 | ⚫ 总结附录 | 需要查资料 | 对比表、代码索引、环境变量 |
> 
> **主线**：以 **vllm-ascend 的 P2P 模式（整体传输版）** 为主线深入讲解
> 
> **建议路径**：
> - 小白：第1章 → 第2章 → 第3章 → ...（按顺序读）
> - 有经验：直接跳转到感兴趣的章节
> - 调试问题：先看第18章代码索引，定位到具体函数，再回来看对应章节

---

## 目录

### 🟢 入门篇（概念铺垫，小白友好）
- [1. 从零开始：vLLM 与 KV 缓存基础](#1-从零开始vllm-与-kv-缓存基础)
- [2. 为什么需要 KV 缓存传输？（场景驱动）](#2-为什么需要-kv-缓存传输场景驱动)
- [3. Mooncake 是什么？（两种模式概览）](#3-mooncake-是什么两种模式概览)

### 🔵 架构篇（整体认识）
- [4. 代码文件地图与模块划分](#4-代码文件地图与模块划分)
- [5. 核心数据结构详解（带形状注解）](#5-核心数据结构详解带形状注解)
- [6. 核心类体系与继承关系](#6-核心类体系与继承关系)

### 🟠 核心流程篇（P2P 模式主线）
- [7. KV 缓存注册流程（内存注册机制）](#7-kv-缓存注册流程内存注册机制)
- [8. KV 缓存传输全流程（一次传输的完整旅程）](#8-kv-缓存传输全流程一次传输的完整旅程)
- [9. 并行策略与分块元数据计算（PP/TP/CP）](#9-并行策略与分块元数据计算pp-tp-cp)

### 🔴 Ascend 优化篇（vllm-ascend 特色）
- [10. GlobalTE 单例与 HCCL 注册优化](#10-globalte-单例与-hccl-注册优化)
- [11. NPU 内存对齐与传输超时](#11-npu-内存对齐与传输超时)
- [12. HMA 混合内存分配器支持](#12-hma-混合内存分配器支持)
- [13. Mamba / MLA 特殊模型支持](#13-mamba--mla-特殊模型支持)

### 🟣 上游扩展篇（上游 vLLM 补充）
- [14. 上游 MooncakeConnector（P2P）对比分析](#14-上游-mooncakeconnectorp2p对比分析)
- [15. 上游 MooncakeStoreConnector（分布式存储）](#15-上游-mooncakestoreconnector分布式存储)
- [16. 上游 Store 子系统深度解析](#16-上游-store-子系统深度解析)

### ⚫ 总结与附录
- [17. vllm-ascend 与上游 vLLM 详细对比](#17-vllm-ascend-与上游-vllm-详细对比)
- [18. 附录：代码索引与环境变量](#18-附录代码索引与环境变量)

---

## 🟢 入门篇（概念铺垫，小白友好）

---

## 1. 从零开始：vLLM 与 KV 缓存基础

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

## 2. 为什么需要 KV 缓存传输？（场景驱动）

### 2.1 核心场景：分解式预填充（Disaggregated Prefilling）

想象一个餐厅厨房：

**传统方式（一个厨师全包）**：
```
厨师A：先备菜（Prefill）→ 再炒菜（Decode）
```
问题：备菜费时间，炒菜也费时间，一个人干效率低，客人等很久。

**分解式预填充（两个厨房分工）**：
```
备菜厨房（Prefill 节点）：专门负责备菜，量大管饱
炒菜厨房（Decode 节点）：专门负责炒菜，随到随炒
```

但是有个问题：备菜厨房切好的菜，怎么送到炒菜厨房去用？

**答案就是：KV 缓存传输！**

```
┌─────────────┐    KV缓存传输     ┌─────────────┐
│  Prefill 节点 │ ───────────────→ │  Decode 节点  │
│  (备菜厨房)    │                  │  (炒菜厨房)    │
└─────────────┘                  └─────────────┘
```

> 💡 **什么是分解式预填充？**
> 
> 就是把 LLM 推理的两个阶段（Prefill 和 Decode）拆到**不同的机器/实例**上运行：
> - **Prefill 节点**：专门处理 prompt 的预填充，计算量大、延迟敏感
> - **Decode 节点**：专门处理逐 token 生成，吞吐优先、需要大量并发
> 
> 这样两边可以独立扩容、独立优化，整体资源利用率更高。
> 
> 上游 vLLM 官方文档明确指出：KV 缓存传输的主要使用场景就是 **disaggregated prefilling**（分解式预填充）。


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

## 3. Mooncake 是什么？（两种模式概览）

### 3.1 Mooncake 简介

**Mooncake**（月饼）是一个专门为 KV 缓存传输优化的高性能传输引擎。

官方项目：https://github.com/kvcache-ai/Mooncake

**核心能力**：
1. 高性能的点对点（P2P）数据传输
2. 支持多种底层传输协议（RDMA、TCP、HCCL 等）
3. 内存注册机制（注册后才能 DMA 传输）
4. 批量传输优化

### 3.2 Mooncake 的两种使用模式

这是很多人容易混淆的地方！Mooncake 在 vLLM 中有**两种完全不同的使用方式**：

| 模式 | 类名 | 用途 | 类比 |
|------|------|------|------|
| **P2P 直传模式** | `MooncakeConnector` | Producer 直接传给 Consumer，一对一 | 快递点对点送货 |
| **分布式存储模式** | `MooncakeStoreConnector` | 大家都存到一个共享存储池里，需要时从池里取 | 共享网盘 / 云存储 |

```
┌─────────────────────────────────────────────────────────────┐
│                    模式一：P2P 直传模式                        │
│                                                             │
│  ┌──────────┐         KV 直传          ┌──────────┐        │
│  │ Producer │ ───────────────────────→ │ Consumer │        │
│  │ (Prefill)│                          │ (Decode) │        │
│  └──────────┘                          └──────────┘        │
│                                                             │
│  特点：一对一、低延迟、需要知道对方在哪                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 模式二：分布式存储模式                          │
│                                                             │
│  ┌──────────┐        存到池子里        ┌──────────┐        │
│  │ Producer │ ───────────────────────→ │          │        │
│  └──────────┘                        │  Mooncake │        │
│                                      │Distributed│        │
│  ┌──────────┐    从池子里取           │  Store    │        │
│  │ Consumer │ ←─────────────────────── │          │        │
│  └──────────┘                        └──────────┘        │
│                                                             │
│  特点：多对多、支持前缀缓存、解耦 Producer 和 Consumer          │
└─────────────────────────────────────────────────────────────┘
```

> 💡 **小白笔记**：
> - **P2P 模式**：就像你给朋友发微信文件，直接发过去，对方直接收。
> - **Store 模式**：就像你把文件传到百度网盘，别人可以从网盘下载。好处是可以多人共享，还能做去重（一样的文件只存一份）。

### 3.3 为什么需要两种模式？各有什么优缺点？

| 对比项 | P2P 直传模式 | 分布式存储模式 |
|--------|-------------|---------------|
| **延迟** | 更低（一跳） | 稍高（两跳：存+取） |
| **前缀缓存** | ❌ 不支持 | ✅ 支持（哈希去重） |
| **Producer/Consumer 解耦** | ❌ 必须知道对方是谁 | ✅ 完全解耦 |
| **多 Consumer 共享** | ❌ 每个 Consumer 都要传一次 | ✅ 存一份，大家都能取 |
| **实现复杂度** | 较低 | 较高 |
| **适用场景** | 分解式预填充（Prefill/Decode 分离） | 前缀缓存、多实例共享 KV |

> 🤔 **什么是前缀缓存？**
> 
> 想象有 100 个用户都问了「请帮我写一份 Python 教程」，开头的 prompt 都一样。
> - **P2P 模式**：每个用户都要重新算一遍前面的 KV 缓存
> - **Store 模式**：第一个用户算完存到池子里，后面 99 个用户直接从池子里取，不用再算了！
> 
> 这就是前缀缓存的威力——相同前缀的 KV 可以复用。

### 3.4 本文档的讲解范围

本文档以 **vllm-ascend 的 P2P 模式（MooncakeConnector）** 为主线，因为这是 vllm-ascend 最核心、优化最多的部分。

上游 vLLM 的 P2P 模式和 Store 模式放在**扩展篇**（第14-16章）讲解，供大家对比学习。

```
本文档结构：
  入门篇 → 架构篇 → 核心流程篇 → Ascend优化篇 → 上游扩展篇 → 总结附录
```

### 3.5 Mooncake 的核心概念

#### 3.5.1 TransferEngine（传输引擎）

Mooncake 的核心类，负责实际的数据传输。

**主要工作**：
- 管理底层连接
- 注册内存区域
- 执行传输操作

#### 3.5.2 内存注册（Memory Registration）

这是理解高性能传输的**第二关键概念**。

**为什么要注册内存？**

就像你寄快递，快递员需要知道你家地址才能上门取件。RDMA/HCCL 这种「直接内存访问」技术也一样，需要先把内存地址告诉传输引擎，它才能直接去读写。

**注册过程**：
1. 告诉传输引擎：「这块内存地址是 xxx，大小是 yyy」
2. 传输引擎记录下来，建立映射
3. 之后就可以直接对这块内存进行 DMA 传输了

> ⚠️ **重要限制**：昇腾 HCCL 每进程最多只能注册 **256 个**内存区域！超过了会报错。这就是为什么后面会有「合并注册区域」的优化。

#### 3.5.3 批量传输

Mooncake 支持一次传很多块数据，比一块块传效率高多了。

```
普通传输（100 次调用）：
  传 blk0 → 传 blk1 → ... → 传 blk99

批量传输（1 次调用）：
  批量传 [blk0, blk1, ..., blk99]
```

### 3.6 控制通道 vs 数据通道

KV 传输用了两条独立的通道：

| 通道 | 用途 | 技术 |
|------|------|------|
| **控制通道** | 握手、交换元数据、通知状态 | ZMQ (ZeroMQ) |
| **数据通道** | 实际的 KV 缓存数据传输 | Mooncake TransferEngine (HCCL/RDMA) |

就像打电话：
- 控制通道 = 先拨号、说「喂听得到吗」、确认对方是谁
- 数据通道 = 然后才开始说正事、传大文件

---

## 🔵 架构篇（整体认识）

---

## 4. 代码文件地图与模块划分

### 4.1 代码文件地图（完整）

在开始之前，先搞清楚我们要讲的代码都在哪里：

```
vllm/ （上游 vLLM）
└── vllm/
    └── distributed/
        └── kv_transfer/
            └── kv_connector/
                └── v1/
                    ├── base.py                    ← 【基类】抽象接口
                    │
                    └── mooncake/
                        ├── mooncake_connector.py  ← 【模式一】上游 P2P 直传
                        ├── mooncake_utils.py      ← Bootstrap 服务等工具
                        ├── rdma_utils.py          ← RDMA 配置工具
                        ├── stats.py               ← 统计指标
                        │
                        └── store/                 ← 【模式二】上游分布式存储
                            ├── connector.py       ← MooncakeStoreConnector 主类
                            ├── scheduler.py       ← 调度器端逻辑
                            ├── worker.py          ← Worker 端逻辑（含传输线程）
                            ├── coordinator.py     ← 缓存命中协调器
                            ├── data.py            ← 数据类（PoolKey等）
                            ├── protocol.py        ← ZMQ 协议
                            └── metrics.py         ← 指标统计

vllm-ascend/
└── vllm_ascend/
    └── distributed/
        └── kv_transfer/
            ├── kv_p2p/
            │   ├── mooncake_connector.py          ← 【主角】Ascend 版 P2P 连接器（整体传输）
            │   ├── mooncake_layerwise_connector.py  ← 【变体】逐层传输版 MooncakeConnector
            │   └── mooncake_hybrid_connector.py    ← 【变体】混合传输版（支持 HMA 混合内存）
            ├── kv_pool/
            │   └── ascend_store/                   ← Ascend 分布式 KV 存储
            │       ├── ascend_store_connector.py   ← Store 连接器主类
            │       ├── pool_worker.py              ← Worker 端逻辑
            │       ├── pool_scheduler.py           ← 调度器端逻辑
            │       ├── coordinator.py              ← 缓存协调器
            │       ├── kv_transfer.py              ← KV 传输逻辑
            │       ├── config_data.py              ← 配置数据结构
            │       └── backend/                    ← 存储后端（mooncake/yuanrong/memcache）
            ├── ascend_multi_connector.py           ← 多连接器组合（P2P + Store + Offload）
            └── utils/
                ├── mooncake_transfer_engine.py     ← 传输引擎单例（GlobalTE）
                └── utils.py                         ← 工具函数（对齐、合并等）
```

> 💡 **小白笔记**：
> - 上游 vLLM 有两个 connector：`MooncakeConnector`（P2P）和 `MooncakeStoreConnector`（Store）
> - vllm-ascend 在 P2P 模式上做了深度优化，有**三种 P2P 变体**，还有自己的 ascend_store
> - 本文档以 `mooncake_connector.py`（整体传输版）为主线深入讲解
> - 想了解上游 Store 模式，请看扩展篇（第15-16章）

### 4.2 两大体系总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Mooncake 生态全景图                              │
│                                                                     │
│  ┌──────────────────────────┐      ┌──────────────────────────┐   │
│  │    P2P 直传模式           │      │   分布式存储模式          │   │
│  │  （MooncakeConnector）   │      │（MooncakeStoreConnector） │   │
│  ├──────────────────────────┤      ├──────────────────────────┤   │
│  │ Producer ↔ Consumer 直连 │      │  通过 Mooncake Store 中转 │   │
│  │ 低延迟、一对一            │      │ 支持前缀缓存、多对多       │   │
│  │ 用于分解式预填充           │      │ 用于前缀缓存、KV共享       │   │
│  └──────────────────────────┘      └──────────────────────────┘   │
│                                                                     │
│  共同基类：KVConnectorBase_V1 + SupportsHMA                        │
│  共同依赖：Mooncake TransferEngine                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 vllm-ascend 的三种 P2P 变体

vllm-ascend 针对不同场景，提供了**三种 P2P connector 变体**，它们都继承自 `KVConnectorBase_V1` 和 `SupportsHMA`：

| 变体 | 文件 | 特点 | 适用场景 |
|------|------|------|---------|
| **整体传输版** | `mooncake_connector.py` | 一次性传输所有层，延迟最低 | 大多数场景，本文档主线 |
| **逐层传输版** | `mooncake_layerwise_connector.py` | 按层逐步传输，支持流水线 | PP（流水线并行）场景 |
| **混合传输版** | `mooncake_hybrid_connector.py` | 支持 HMA 混合内存，高低速内存搭配 | 显存不足，需要 HMA 的场景 |

> 💡 **怎么选？**
> - 刚开始用，选**整体传输版**（最简单、性能也最好）
> - 用了 PP 流水线并行，选**逐层传输版**
> - 显存不够用，开了 HMA，选**混合传输版**

### 4.4 P2P 模式的三层架构（vllm-ascend 版）

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

### 4.5 进程/线程关系（P2P 模式）

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

> 💡 **小白笔记**：数据结构是理解代码的钥匙。搞懂这些结构体，代码就看懂了一半。
> 
> 本章所有例子都基于**同一个模型**，方便对照理解。

### 🎯 统一模型配置：Llama-3-8B

本章所有数据结构的例子都用下面这个模型配置：

| 配置项 | 值 | 说明 |
|-------|-----|------|
| 模型 | Llama-3-8B | 32层 Transformer |
| Q 头数 | 32 | 32个查询头 |
| KV 头数 | 8 | GQA，8个KV头 |
| head_dim | 128 | 每个头的维度 |
| 数据类型 | FP16 | 2字节 |
| block_size | 16 | 每个块16个token |
| num_blocks | 1024 | 总共1024个逻辑块 |
| TP 大小 | 4 | 4张卡做张量并行 |
| 每层 block 大小 | 8 × 128 × 2 = 2048 字节 = 2KB | 每个KV块的字节数 |

---

### 5.1 MooncakeAgentMetadata —— 握手时交换的信息

**定义位置**：`mooncake_connector.py:95`
**作用**：两个节点握手时，互相告诉对方自己的配置信息。

```python
class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str
    te_rpc_port: int
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]]
    block_size: int
    kv_caches_base_addr: list[list[int]]
    block_size_scale: list[list[int]]
    num_blocks: int
    block_lens: list[list[int]]
    block_strides: list[list[int]]
    local_ip: str = ""
    handshake_port: int = 0
```

#### 🔍 为什么用 `msgspec.Struct`？

因为要跨网络传输，需要高性能序列化。`msgspec` 比 JSON 快很多，体积也小。两个关键参数：
- `omit_defaults=True`：默认值的字段不序列化，省流量
- `dict=True`：支持 `.` 和 `[]` 两种访问方式

> 💡 对比：`ReqMeta` 只是进程内部用，所以用普通 `@dataclass` 就够了。

#### 🎯 真实例子
```python
MooncakeAgentMetadata(
    engine_id = "vllm-worker-0-tp-0",
    te_rpc_port = 50051,
    kv_group2layeridx = {
        0: (
            {
                "layer_names": [f"model.layers.{i}" for i in range(32)],
                "kv_cache_spec_type": "FullAttentionSpec",
                "num_kv_heads": 8,
                "kv_cache_group_id": 0,
            },
            list(range(32)),  # [0, 1, ..., 31]
        )
    },
    block_size = 16,
    kv_caches_base_addr = [
        [0x7f0000000000, 0x7f0000800000],  # 第0层：K, V
        [0x7f0001000000, 0x7f0001800000],  # 第1层：K, V
        # ... 共32层 ...
    ],
    block_size_scale = [[1, 1]] * 32,    # 每层都是1倍缩放
    num_blocks = 1024,
    block_lens = [[2048, 2048]] * 32,    # 每个块2KB
    block_strides = [[2048, 2048]] * 32, # 步长=块大小（无padding）
    local_ip = "192.168.1.100",
    handshake_port = 51000,
)
```

#### 字段逐一讲解

| 序号 | 字段 | 形状/类型 | 含义 | 例子中的值 |
|------|------|----------|------|-----------|
| ① | `engine_id` | str | 引擎唯一ID（身份证号） | `"vllm-worker-0-tp-0"` |
| ② | `te_rpc_port` | int | TransferEngine RPC 端口 | `50051` |
| ③ | `kv_group2layeridx` | `{组ID: (组规格, 层索引列表)}` | 按KV规格分组，不同组传输方式不同 | 1组，32层全是Attention |
| ④ | `block_size` | int | 每个块多少token | `16` |
| ⑤ | `kv_caches_base_addr` | `(num_layers, 2)` | 每层K/V缓存的基地址 | 32层×2=64个地址 |
| ⑥ | `block_size_scale` | `(num_layers, 2)` | 块大小缩放比例 | 全是 `1` |
| ⑦ | `num_blocks` | int | 总逻辑块数 | `1024` |
| ⑧ | `block_lens` | `(num_layers, 2)` | 每个块的字节长度 | 全是 `2048`（2KB） |
| ⑨ | `block_strides` | `(num_layers, 2)` | 相邻块的字节距离 | 全是 `2048`（无padding） |
| ⑩ | `local_ip` | str | 本机IP地址 | `"192.168.1.100"` |
| ⑪ | `handshake_port` | int | 握手服务端口 | `51000` |

**补充说明**：

- **kv_group2layeridx 为什么要分组？** 不同层的 KV 缓存结构可能完全不同（比如 Attention 层 vs Mamba 层），不能混在一起传。普通 Transformer 通常只有 1 组。
- **block_strides 为什么可能比 block_lens 大？** 因为内存对齐，为了高性能访问，地址需要对齐到特定边界，中间会有空隙（padding）。

---

### 5.2 ReqMeta —— 单次请求的传输元数据

**定义位置**：`mooncake_connector.py:110`
**作用**：每次传输一个请求的 KV 缓存时，把所有相关信息打包成这个结构体。

```python
@dataclass
class ReqMeta:
    local_block_ids: BlockIds
    num_external_tokens: int
    num_computed_tokens: int
    remote_block_ids: BlockIds
    remote_host: str
    remote_port: int
    remote_engine_id: str
    remote_request_id: str
    remote_pcp_size: int
    remote_dcp_size: int
    remote_ptp_size: int | None
    remote_multi_nodes_meta_mapping: dict[str, dict[str, Any]]
    num_prompt_blocks: int
    remote_block_size: int
```

#### 🎯 真实例子

**场景**：一个 256 token 的 prompt 请求，从 Prefill 节点传到 Decode 节点。
- prompt 长度：256 token
- 需要的块数：256 ÷ 16 = 16 块
- Prefill TP=4，Decode TP=4

```python
ReqMeta(
    local_block_ids = [100, 101, 102, ..., 115],    # 本地16个块
    num_external_tokens = 256,                       # 需要传256个token
    num_computed_tokens = 256,                       # 已经算了256个
    remote_block_ids = [200, 201, 202, ..., 215],   # 远程16个块
    remote_host = "192.168.1.200",                   # 远程主机
    remote_port = 51000,                              # 远程握手端口
    remote_engine_id = "vllm-prefill-0-tp-0",        # 远程引擎ID
    remote_request_id = "req-abc-123",               # 远程请求ID
    remote_pcp_size = 1,                              # 远程Prefill CP大小
    remote_dcp_size = 1,                              # 远程Decode CP大小
    remote_ptp_size = 4,                              # 远程Prefill TP大小
    remote_multi_nodes_meta_mapping = {},             # 多节点元数据映射
    num_prompt_blocks = 16,                           # prompt共16块
    remote_block_size = 16,                           # 远程块大小
)
```

#### 字段逐一讲解

| 字段 | 含义 | 例子中的值 |
|------|------|-----------|
| `local_block_ids` | 本地块ID列表（数据要放到这些块里） | `[100..115]`，共16块 |
| `num_external_tokens` | 需要传输的外部token数 | `256` |
| `num_computed_tokens` | 已经计算了多少token | `256`（Prefill完了） |
| `remote_block_ids` | 远程块ID列表（数据从这些块来） | `[200..215]`，共16块 |
| `remote_host` | 远程主机地址 | `"192.168.1.200"` |
| `remote_port` | 远程握手端口 | `51000` |
| `remote_engine_id` | 远程引擎ID | `"vllm-prefill-0-tp-0"` |
| `remote_request_id` | 远程请求ID | `"req-abc-123"` |
| `remote_pcp_size` | 远程Prefill CP大小 | `1` |
| `remote_dcp_size` | 远程Decode CP大小 | `1` |
| `remote_ptp_size` | 远程Prefill TP大小 | `4` |
| `remote_multi_nodes_meta_mapping` | 多节点元数据映射 | `{}` |
| `num_prompt_blocks` | prompt有多少块 | `16` |
| `remote_block_size` | 远程的块大小 | `16` |

> 💡 **大白话**：`ReqMeta` 就是一张「快递单」——写清楚了「从哪来、到哪去、有多少东西」。
> 
> - `remote_block_ids` = 发货地址（远程的哪几个块）
> - `local_block_ids` = 收货地址（本地的哪几个块）
> - `num_external_tokens` = 包裹里有多少东西

---

### 5.3 GroupPull —— 一组缓存的拉取描述

**定义位置**：`mooncake_connector.py:129`
**作用**：描述「拉取某一组 KV 缓存」需要哪些参数。用于 Worker 内部组织传输任务。

```python
@dataclass(frozen=True)
class GroupPull:
    group_id: int
    remote_tp_offset: int
    num_group_pulls: int
    prefill_pp_rank: int = 0
    is_group_transfer_end: bool = False
```

#### 🎯 真实例子

**场景**：TP=4，GQA（Q=32头，KV=8头）。Consumer 的 TP0 要拉取第0组（K缓存）的数据。

因为 KV 只有 8 头，TP=4 的话每张卡 2 个 KV 头。Consumer 的一张卡只需要从 Producer 的对应卡 1 次就能拉完。

```python
GroupPull(
    group_id = 0,              # 拉第0组（K缓存）
    remote_tp_offset = 0,      # 从远程TP0拉
    num_group_pulls = 1,       # 只需要拉1次
    prefill_pp_rank = 0,       # Prefill PP第0段
    is_group_transfer_end = False,  # 不是结束标记
)
```

#### 字段逐一讲解

| 字段 | 含义 | 例子中的值 |
|------|------|-----------|
| `group_id` | KV缓存组ID（0=K，1=V，Mamba/MLA有更多组） | `0`（K缓存） |
| `remote_tp_offset` | 远程TP偏移量——从远程的哪张卡拉 | `0`（从TP0拉） |
| `num_group_pulls` | 需要拉几次（GQA场景可能要从多张卡拉） | `1`（拉1次就够） |
| `prefill_pp_rank` | Prefill的PP秩（PP分段时用） | `0`（第0段） |
| `is_group_transfer_end` | 是不是组传输的结束标记 | `False` |

**补充说明**：

- **什么是 KV 缓存组？** 普通 Attention 有 K 和 V 两组；Mamba/MLA 等特殊模型有更多组缓存，用 group_id 区分。
- **什么是 TP 偏移？** TP=4 时每张卡只存 1/4 的头，Consumer 的卡 i 要从 Producer 的卡 i 拉，偏移量就是 i。
- **什么时候 num_group_pulls > 1？** GQA 场景下，如果 KV 头数比 Q 少很多，Consumer 的一张卡可能需要从 Producer 的多张卡拉，这时要拉多次。

---

### 5.4 KVCacheTaskTracker —— 任务跟踪器

**定义位置**：`mooncake_connector.py:164`
**作用**：跟踪哪些请求传完了，哪些还在传，防止内存泄漏。

```python
class KVCacheTaskTracker:
    finished_requests: set[str]
    delayed_free_requests: OrderedDict[str, float]
    reqs_to_process: set[str]
```

#### 🎯 真实例子

**场景**：同时有 3 个请求在传输，1 个传完了，1 个不需要传。

```python
tracker = KVCacheTaskTracker()

# 初始状态
tracker.reqs_to_process = {"req-001", "req-002", "req-003"}  # 3个待处理
tracker.finished_requests = set()       # 还没有完成的
tracker.delayed_free_requests = OrderedDict()  # 没有延迟释放的

# req-002 不需要传输（本地已经有了）
tracker.add_not_transfer_request("req-002")
# → finished_requests = {"req-002"}
# → reqs_to_process = {"req-001", "req-003"}

# req-001 传完了
tracker.update_done_task_count("req-001")
# → finished_requests = {"req-001", "req-002"}
# → reqs_to_process = {"req-003"}
```

#### 字段逐一讲解

| 字段 | 类型 | 含义 |
|------|------|------|
| `finished_requests` | `set[str]` | 已经传完的请求ID集合 |
| `delayed_free_requests` | `OrderedDict[str, float]` | 延迟释放的请求（请求ID → 时间戳），防止有人还在用就释放 |
| `reqs_to_process` | `set[str]` | 待处理的请求ID集合 |

**补充说明**：

- **为什么要有延迟释放？** 传输完通知 Scheduler 释放块，但 Worker 可能还在引用这块内存。延迟一会儿再释放，等大家都不用了再真释放，避免 use-after-free bug。
- **超时强制释放**：如果请求超过 `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` 秒还没完成，就强制释放，防止内存泄漏。

---

## 6. 核心类体系与继承关系

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

### 6.2 继承关系图（完整）

```
KVConnectorBase_V1 （上游定义的抽象基类）
         │
         ├── 定义了 Scheduler 端该有哪些方法
         └── 定义了 Worker 端该有哪些方法

SupportsHMA （混合内存支持接口）
         │
         └── 定义了 HMA 场景下的额外方法
         │
         ┌─────────────────────────────────────┐
         ▼                                     ▼
┌─────────────────────┐            ┌─────────────────────────┐
│  MooncakeConnector  │            │ MooncakeStoreConnector  │
│   （P2P 直传）       │            │    （分布式存储）         │
│  上游 vLLM 版 /      │            └─────────────────────────┘
│  vllm-ascend 版     │
└─────────────────────┘
         │
         ├── MooncakeConnectorScheduler （Scheduler 侧实现）
         │   └── 运行在调度器进程里
         │
         └── MooncakeConnectorWorker （Worker 侧实现）
             └── 运行在每个 worker 进程里
                 ├── KVCacheSendingThread （发送线程）
                 └── KVCacheRecvingThread （接收线程）
```

### 6.3 MooncakeConnectorScheduler（调度器端）

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

### 6.4 MooncakeConnectorWorker（工作端）

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

### 6.5 KVCacheSendingThread（发送线程）

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

### 6.6 KVCacheRecvingThread（接收线程）

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

## 🟠 核心流程篇（P2P 模式主线）

---

## 7. KV 缓存注册流程（内存注册机制）

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

## 9. 并行策略与分块元数据计算（PP/TP/CP）

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

## 🔴 Ascend 优化篇（vllm-ascend 特色）

---

## 10. GlobalTE 单例与 HCCL 注册优化

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
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking（双重检查锁定）
                if self.transfer_engine is None:
                    self.transfer_engine = TransferEngine()
                    # ... 初始化 ...
        return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                self.transfer_engine.register_memory(ptr, size)
            self.is_register_buffer = True

global_te = GlobalTE()  # 全局单例实例
```

**特点**：
1. **延迟初始化**：第一次用的时候才创建，不用不创建
2. **线程安全**：用锁（`transfer_engine_lock` / `register_buffer_lock`）保证多线程下也只创建一个
3. **注册幂等**：内存注册只做一次，重复调用直接返回

> 💡 **使用方式**：通过 `global_te` 这个全局变量来访问，比如 `global_te.get_transfer_engine(hostname, device_name)`。

### 10.2 HCCL 注册区域限制（256 限制）

**常量**：`MAX_HCCL_REGISTER_REGIONS = 256`  
**定义位置**：`utils.py:14`

#### 问题背景

昇腾的 HCCL 通信库有个限制：**每个进程最多注册 256 个内存区域**。

如果超过了，注册会失败，进而导致传输失败。

#### 解决方案

1. **合并相邻区域**：第 7 章讲过的，把能合并的合并起来
2. **注册前检查**：`validate_register_region_count()` 提前检查
3. **报错提示**：如果超了，告诉用户怎么优化

### 10.3 合并注册区域的实现

**函数**：`collect_storage_merged_register_regions()`  
**位置**：`utils.py:363`

这个函数负责把同一块 storage 里的相邻注册区域合并起来。

**合并算法**：
1. 按 storage 分组
2. 每个 storage 内按地址排序
3. 检查相邻区域的间隙是否小于 `REGISTER_MERGE_GAP_BYTES`
4. 小于阈值就合并成一个大区域

> 💡 **生活类比**：就像你有好几个相邻的小仓库，把中间的墙打通，变成一个大仓库。虽然中间有一点空间浪费（墙的位置），但整体更方便管理了。

---

## 11. NPU 内存对齐与传输超时

### 11.1 NPU 内存对齐

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

### 11.2 传输超时计算

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
| `ASCEND_TRANSFER_TIMEOUT` | 直接指定超时时间（毫秒），设了就用这个值 | 不设置则自动计算 |
| `HCCL_RDMA_TIMEOUT` | HCCL RDMA 超时指数（用于自动计算超时） | 20 |
| `HCCL_RDMA_RETRY_CNT` | HCCL RDMA 重试次数（用于自动计算超时） | 7 |

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

### 11.3 torch_npu 操作优化

昇腾版本大量使用 `torch_npu` 库进行 NPU 加速操作，比如：
- NPU 上的张量操作
- 融合算子（fused op）加速
- NZ 格式转换

---

## 12. HMA 混合内存分配器支持

### 12.1 什么是 HMA？

HMA = Hybrid Memory Allocator（混合内存分配器）。

**核心思想**：把 KV 缓存分成多份，存在不同的内存区域（比如一部分在高速内存，一部分在低速内存），以平衡容量和速度。

#### 为什么需要 HMA？

- NPU 上的高速内存（类似显存）很贵，容量有限
- 但是 KV 缓存又特别占空间
- 所以把热数据放高速内存，冷数据放慢速内存，性价比最高

### 12.2 MooncakeConnector 怎么支持 HMA？

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

## 13. Mamba / MLA 特殊模型支持

### 13.1 Mamba 支持

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

### 13.2 DeepSeek MLA 支持

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

## 🟣 上游扩展篇（上游 vLLM 补充）

> 💡 **阅读提示**：本篇讲解上游 vLLM 的 Mooncake 实现，包括 P2P 模式和 Store 模式。vllm-ascend 的版本就是在 P2P 模式基础上扩展优化的。对比着学，理解更深刻。

---

## 14. 上游 MooncakeConnector（P2P）对比分析

### 14.1 上游 P2P 模式的整体架构

上游 vLLM 的 `MooncakeConnector` 和 vllm-ascend 的版本**架构相似但细节不同**。

```
┌─────────────────────────────────────────────────────────────┐
│              上游 MooncakeConnector（P2P）                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  MooncakeConnector（主类）                                   │
│    │                                                        │
│    ├── MooncakeConnectorScheduler（调度器端）                 │
│    │   ├── 管理 reqs_need_recv（待接收请求）                  │
│    │   ├── 管理 reqs_need_send（待发送请求）                  │
│    │   └── request_finished() 请求结束处理                   │
│    │                                                        │
│    └── MooncakeConnectorWorker（Worker 端）                  │
│        ├── TransferEngine（传输引擎）                        │
│        ├── ZMQ 控制通道（握手 + 元数据交换）                  │
│        └── 异步发送/接收机制                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 14.2 核心数据结构（上游版）

#### 14.2.1 TransferRegion —— 传输区域描述

```python
@dataclass(frozen=True)
class TransferRegion:
    layer_name: str        # 层名称
    layer_index: int       # 层索引
    base_addr: int         # 基地址
    block_len: int         # 块长度（字节）
    kv_block_len: int      # K/V 单个的块长度
```

对比 vllm-ascend：上游用 `TransferRegion` 统一描述每层的传输区域，更简洁。

#### 14.2.2 MooncakeXferMetadata —— 传输元数据

```python
class MooncakeXferMetadata(msgspec.Struct, omit_defaults=True):
    remote_hostname: str                    # 远程主机名
    remote_port: int                        # 远程端口
    remote_tp_size: int                     # 远程 TP 大小
    remote_tp_rank: int                     # 远程 TP 秩
    req_blocks: dict[ReqId, tuple[TransferId, list[list[int]]]]  # 请求块映射
    kv_caches_base_addr: list[int]          # KV 缓存基地址列表
    block_lens: list[int]                   # 块长度列表
    registered_layer_names: list[str]       # 已注册的层名称
    registered_layer_indices: list[int]     # 已注册的层索引
```

> 💡 上游用 `msgspec` 做序列化（比 JSON 快很多），vllm-ascend 也沿用了这个做法。

#### 14.2.3 PullReqMeta / SendBlockMeta

```python
@dataclass
class PullReqMeta:
    d_req_id: ReqId             # Decode 端请求 ID
    transfer_id: TransferId     # 传输 ID（P/D 共享）
    local_block_ids: list[list[int]]  # 本地块 ID
    remote_engine_id: EngineId  # 远程引擎 ID
    remote_bootstrap_addr: str  # 远程 bootstrap 地址
    expire_time: float          # 过期时间
    pull_tasks_count: int       # 拉取任务数
```

```python
@dataclass
class SendBlockMeta:
    p_req_id: ReqId             # Prefill 端请求 ID
    transfer_id: TransferId     # 传输 ID
    local_block_ids: list[list[int]]  # 本地块 ID
    ready: asyncio.Event        # 就绪事件
    expire_time: float          # 过期时间
    need_send: int              # 需要发送的数量
    sent: int                   # 已发送数量
    sending: int                # 发送中数量
```

### 14.3 上游特有的：异构 TP 支持

这是上游版本的一个**重要特性**：支持 Producer 和 Consumer 的 TP 大小不一样！

比如：
- Producer 端 TP=4（4 张卡做 Prefill）
- Consumer 端 TP=8（8 张卡做 Decode）

这时候怎么传？

#### 14.3.1 什么是异构 TP？

```
同质 TP（两边一样）：
  Producer TP=4  ←→  Consumer TP=4
  卡0 ↔ 卡0
  卡1 ↔ 卡1
  ...

异构 TP（两边不一样）：
  Producer TP=4  ←→  Consumer TP=8
  卡0 → 卡0, 卡1
  卡1 → 卡2, 卡3
  ...
  （Producer 一张卡的数据要分给 Consumer 两张卡）
```

#### 14.3.2 tp_ratio 是什么？

```python
def _get_tp_ratio(local_tp_size: int, remote_tp_size: int) -> int:
    if local_tp_size >= remote_tp_size:
        return local_tp_size // remote_tp_size  # 正数：本地大
    else:
        return -(remote_tp_size // local_tp_size)  # 负数：远程大
```

- `tp_ratio = 1`：两边一样大，一一对应
- `tp_ratio > 0`：本地产出大，一个本地块对应多个远程块
- `tp_ratio < 0`：远程产出大，一个远程块对应多个本地块

#### 14.3.3 传输计划计算

上游用 `_compute_sender_transfer_plan()` 来计算每次传输的：
- 要不要传（`should_transfer`）
- 源偏移（`src_offset`）
- 目标偏移（`dst_offset`）
- 传输长度（`transfer_len`）

> 🤔 **vllm-ascend 支持异构 TP 吗？**
> 
> vllm-ascend 主要针对昇腾场景优化，目前更侧重**同构 TP + PCP/DCP 上下文并行**的场景。异构 TP 的支持相对上游要弱一些。

### 14.4 上游特有的：Bootstrap Server（引导服务器）

上游有个 `MooncakeBootstrapServer`，用来做**服务发现**。

```
┌─────────────────────────────────────────────────────────┐
│                  Bootstrap Server                       │
│              （运行在 rank 0 的 FastAPI 服务）            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  /register  →  Worker 注册自己的地址、端口、rank 信息     │
│  /query     →  查询所有已注册的 Worker 信息              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**工作流程**：
1. 所有 Prefill Worker 启动后，向 Bootstrap Server 注册自己
2. Decode Worker 需要传输时，先去 Bootstrap Server 查询对方地址
3. 拿到地址后再建立连接

> 💡 vllm-ascend 怎么发现对方？
> 
> vllm-ascend 用的是更直接的方式：通过配置直接指定对端地址，或者通过 ZMQ 直接握手。没有用 Bootstrap Server 这种中间服务。

### 14.5 上游 P2P vs vllm-ascend P2P（快速对比）

| 特性 | 上游 vLLM | vllm-ascend |
|------|-----------|-------------|
| **异构 TP 支持** | ✅ 完善 | ⚠️ 较弱 |
| **PCP/DCP 支持** | ❌ 不支持 | ✅ 完善 |
| **Bootstrap Server** | ✅ 有 | ❌ 无（直接握手） |
| **Mamba 支持** | ❌ 不支持 | ✅ 支持 |
| **MLA 支持** | ❌ 不支持 | ✅ 支持 |
| **HCCL 优化** | 通用 | ✅ 深度优化（256限制、合并注册） |
| **NPU NZ 格式** | ❌ 不支持 | ✅ 支持 |

---

## 15. 上游 MooncakeStoreConnector（分布式存储）

> 💡 **有趣的历史**：上游 vLLM 的 `MooncakeStoreConnector` 实际上是**从 vllm-ascend 的 ascend_store 改编过去的**！上游代码注释里明确写着 "Adapted from vllm-project/vllm-ascend"。
> 
> 所以这章讲的内容，vllm-ascend 也有类似的实现（在 `kv_pool/ascend_store/` 目录下）。本文档以 vllm-ascend P2P 为主线，所以 Store 模式放在扩展篇讲解。

### 15.1 什么是 Store 模式？

前面讲过，Store 模式就像「共享网盘」：
- Producer 把 KV 缓存存到 Mooncake Distributed Store 里
- Consumer 需要时从 Store 里取
- 支持前缀缓存（相同的 KV 只存一份）

**核心优势：前缀缓存**

```
用户1的请求：「请帮我写一份 Python 教程，关于...」
用户2的请求：「请帮我写一份 Python 教程，针对...」
                └──────────┬──────────┘
                           │
                      相同的前缀！
                      这部分 KV 可以复用！
```

在 Store 模式下：
1. 用户 1 请求到达 → 计算前缀 KV → 存到 Store
2. 用户 2 请求到达 → 查 Store → 发现前缀已经有了 → 直接取，不用再算了！

**节省的时间 = 前缀长度对应的 Prefill 时间**

如果前缀很长（比如几千 token），节省的时间非常可观！

### 15.2 Store 模式的核心概念

#### 15.2.1 PoolKey —— KV 块的「身份证」

在 Store 模式下，每个 KV 块都有一个唯一的 key，用来在存储池里定位它。

```python
@dataclass(order=True)
class PoolKey:
    key_metadata: KeyMetadata  # 元数据部分
    chunk_hash: str            # 块哈希（内容的指纹）
```

**KeyMetadata 里有什么？**

```python
@dataclass
class KeyMetadata:
    model_name: str      # 模型名称
    tp_rank: int         # TP 秩
    pcp_rank: int        # Prefill CP 秩
    dcp_rank: int        # Decode CP 秩
    pp_rank: int         # PP 秩
    group_id: int = 0    # KV 缓存组 ID
    cache_prefix: str = ""  # 可选的命名空间前缀
```

> 💡 **小白理解**：
> - `model_name` = 哪个模型的
> - `tp_rank` / `pp_rank` / `pcp_rank` / `dcp_rank` = 哪张卡的
> - `group_id` = 哪组缓存的（K/V/其他）
> - `chunk_hash` = 内容是什么（内容哈希）
> - 合在一起就能唯一定位一个 KV 块

#### 15.2.2 两种运行模式

Store 模式有两种部署方式：

| 模式 | 说明 | 类比 |
|------|------|------|
| **embedded（嵌入式）** | 每个 Worker 贡献一部分内存组成共享池 | 每家出一点地方，凑成一个共用仓库 |
| **standalone-store（独立存储）** | 专门的 mooncake_client 进程管存储，Worker 不贡献内存 | 专业的云存储服务商 |

```
embedded 模式：
  Worker0 ─┐
  Worker1 ─┼─ 共同组成 Store 池
  Worker2 ─┘
  （每个 Worker 出 4GB 内存，凑成 12GB 共享池）

standalone-store 模式：
  Worker0 ─┐
  Worker1 ─┼─→ 专门的 Store 服务进程
  Worker2 ─┘    （有自己的内存甚至 SSD）
```

**配置类**：`MooncakeStoreConfig`

```python
@dataclass
class MooncakeStoreConfig:
    metadata_server: str           # 元数据服务器地址
    master_server_address: str     # 主服务器地址
    protocol: str                  # 传输协议（rdma等）
    device_name: str               # 设备名
    mode: MooncakeMode = "embedded"  # 模式
    global_segment_size: int       # 全局段大小
    local_buffer_size: int         # 本地缓冲区大小
    enable_offload: bool           # 是否启用 SSD 卸载
```

> 💡 **SSD 卸载是什么？**
> 
> 当 KV 缓存太多、内存放不下时，可以把冷数据放到 SSD 上。用的时候再加载到内存。
> 类似 CPU 的虚拟内存：内存不够了就换页到磁盘。

### 15.3 Store 模式的工作流程

#### 15.3.1 Producer 端：存进去

```
Producer 计算完 KV
    │
    ▼
  计算每个块的哈希值
    │
    ▼
  构造 PoolKey
    │
    ▼
  调用 Mooncake Store API 存进去
    │
    ▼
  存储成功！后续别人可以通过 key 来取
```

#### 15.3.2 Consumer 端：查缓存 → 取出来

```
Consumer 收到新请求
    │
    ▼
  计算 prompt 的块哈希
    │
    ▼
  用哈希去 Store 里查：有没有现成的 KV？
    │
    ├─ 有命中 → 从 Store 加载这些块到本地显存
    │
    └─ 没命中 → 自己计算 Prefill
```

> 🎯 **前缀缓存的核心**：通过哈希值判断「这块 KV 我之前是不是算过」。如果算过，直接取就行。

### 15.4 Store 模式的架构

```
┌─────────────────────────────────────────────────────────────────┐
│              MooncakeStoreConnector 架构                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Scheduler 进程（1个）                        │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │         MooncakeStoreScheduler                  │    │   │
│  │  │  - get_num_new_matched_tokens() 查缓存命中        │    │   │
│  │  │  - update_state_after_alloc() 分配后更新         │    │   │
│  │  │  - request_finished() 请求完成清理               │    │   │
│  │  └───────────────────┬─────────────────────────────┘    │   │
│  │                      │ LookupKeyClient（ZMQ客户端）      │   │
│  └──────────────────────┼──────────────────────────────────┘   │
│                         │                                       │
│  ┌──────────────────────┼──────────────────────────────────┐   │
│  │       Worker 进程（每张卡一个）                           │   │
│  │  ┌───────────────────▼─────────────────────────────┐    │   │
│  │  │         MooncakeStoreWorker                     │    │   │
│  │  │  ┌──────────────────────────────────────────┐  │    │   │
│  │  │  │ LookupKeyServer（ZMQ服务端，响应查请求）   │  │    │   │
│  │  │  └──────────────────────────────────────────┘  │    │   │
│  │  │  ┌──────────────────────────────────────────┐  │    │   │
│  │  │  │ KVCacheStoreSendingThread（存到Store）    │  │    │   │
│  │  │  └──────────────────────────────────────────┘  │    │   │
│  │  │  ┌──────────────────────────────────────────┐  │    │   │
│  │  │  │ KVCacheStoreRecvingThread（从Store取）    │  │    │   │
│  │  │  └──────────────────────────────────────────┘  │    │   │
│  │  └───────────────────┬─────────────────────────────┘    │   │
│  │                      │                                   │   │
│  │              MooncakeDistributedStore                    │   │
│  │              （共享 KV 缓存池）                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

> 💡 **Store 模式的核心思想**：
> - Scheduler 负责**决策**：哪些块可以从外部缓存加载？加载多少？
> - Worker 负责**执行**：实际的存/取操作，通过 Mooncake Store 中转
> - 用 ZMQ 做 Scheduler ↔ Worker 之间的查询通道

---

## 16. 上游 Store 子系统深度解析

> ⚠️ **进阶内容**：本章深入上游 Store 模式的内部实现。如果你只关心 vllm-ascend 的 P2P 模式，可以先跳过。

### 16.1 Store 子系统的六大组件

上游的 `store/` 目录下有 7 个文件，组成了完整的 Store 模式实现：

```
store/
├── connector.py    ← MooncakeStoreConnector 主类（对外接口）
├── scheduler.py    ← MooncakeStoreScheduler（调度器端逻辑）
├── worker.py       ← MooncakeStoreWorker（Worker 端逻辑）
├── coordinator.py  ← MooncakeStoreCoordinator（缓存命中协调）
├── data.py         ← 数据类（PoolKey、KeyMetadata等）
├── protocol.py     ← ZMQ 协议定义
└── metrics.py      ← 指标统计
```

### 16.2 Scheduler 端：MooncakeStoreScheduler

**运行位置**：Scheduler 进程

**核心职责**：
1. 查询外部缓存有没有命中
2. 计算需要加载多少 token
3. 管理加载状态

#### 16.2.1 核心方法：get_num_new_matched_tokens()

这是 Scheduler 端最核心的方法：查询某个请求在外部缓存里命中了多少 token。

```python
def get_num_new_matched_tokens(
    self,
    request: Request,
    num_computed_tokens: int,
) -> tuple[int | None, bool]:
    # 1. 计算要查多少 token
    token_len = request.num_tokens // self._block_size * self._block_size
    
    # 2. 调用 LookupKeyClient 去查
    num_external_hit_tokens = self.client.lookup(
        request.request_id,
        token_len,
        request.block_hashes,
        non_block=self.lookup_async,
    )
    
    # 3. 返回命中的 token 数
    return need_to_allocate, self.load_async
```

**返回值说明**：
- `int | None`：命中了多少 token。`None` 表示还在查（异步模式），等下一轮再问
- `bool`：是不是异步加载。True 表示后台加载，不阻塞主流程

#### 16.2.2 LookupKeyClient —— 查询客户端

Scheduler 进程里的查询客户端，通过 ZMQ 向 Worker 发查询请求。

```
Scheduler 进程                     Worker 进程
┌─────────────┐   ZMQ 请求    ┌──────────────┐
│ LookupKey   │ ───────────→  │ LookupKey    │
│   Client    │               │   Server     │
│             │ ←───────────  │              │
└─────────────┘   查询结果    └──────────────┘
```

> 💡 **为什么不直接在 Scheduler 里查？**
> 
> 因为 Mooncake Store 的连接在 Worker 进程里（和 GPU/NPU 在一起）。
> Scheduler 进程不直接接触存储，通过 ZMQ 问 Worker 进程。

### 16.3 Worker 端：MooncakeStoreWorker

**运行位置**：每个 Worker 进程

**核心职责**：
1. LookupKeyServer：响应 Scheduler 的查询请求
2. KVCacheStoreSendingThread：把 KV 存到 Store
3. KVCacheStoreRecvingThread：从 Store 取 KV

#### 16.3.1 传输线程架构

和 P2P 模式类似，Store 模式也有后台线程：

```
KVCacheStoreSendingThread（存线程）：
  队列里有待存的块 → 批量存到 Store 里

KVCacheStoreRecvingThread（取线程）：
  队列里有待取的块 → 批量从 Store 里取
```

#### 16.3.2 ZMQ 查询协议（protocol.py）

Scheduler 和 Worker 之间的查询用 ZMQ 通信，协议很简单：

| 消息类型 | 作用 | 请求格式 | 响应格式 |
|---------|------|---------|---------|
| `LOOKUP_MSG` | 查询缓存命中数 | token_len + hash_len + 哈希数据 | hit_count（u32） |
| `RESET_MSG` | 重置状态 | 无 | RESP_OK / RESP_ERR |

```
LOOKUP_MSG 请求：
┌──────────┬───────────┬──────────────────┐
│ "lookup" │ hash_len  │  所有块的哈希拼接  │
└──────────┴───────────┴──────────────────┘
  (bytes)   (u16 BE)      (连续字节)

响应：
┌──────────────┐
│  hit_count   │  （u32 BE，命中的块数）
└──────────────┘
```

> 💡 为什么用 ZMQ 而不是直接函数调用？
> 
> 因为 Scheduler 和 Worker 是**不同的进程**，甚至可能在不同的机器上。
> 进程间通信需要用 IPC/RPC，ZMQ 是一种高性能的消息队列。

### 16.4 Coordinator：缓存命中协调器

`MooncakeStoreCoordinator` 是用来计算「最长缓存命中长度」的。

#### 16.4.1 为什么需要 Coordinator？

想象一下：
- 你有 100 个块的哈希值
- 你想知道：从第 0 块开始，最多有多少块是连续命中的？
- 而且不同的 KV 组（K/V/Mamba状态等）命中情况可能不一样

这就是 Coordinator 要解决的问题。

#### 16.4.2 核心方法：find_longest_cache_hit()

```python
def find_longest_cache_hit(
    self,
    block_hashes: Sequence[BlockHash],
    max_length: int,
    cached_block_pool: ExternalCachedBlockPool,
    *,
    apply_eagle: bool = True,
) -> tuple[tuple[list[bool], ...], int]:
    # 返回：
    # - 每个组的加载掩码（load_mask_per_group）
    # - 命中长度（hit_length）
```

**ExternalCachedBlockPool** 是什么？

它是一个「模拟的块池」，里面记录了「哪些哈希对应的块在外部缓存里存在」。

```python
class ExternalCachedBlockPool:
    _exists: set[tuple[int, bytes]]  # (group_id, hash) 的集合
    
    def get_cached_block(self, block_hash, group_ids):
        # 检查这个哈希 + 这些组是不是都在缓存里
        # 都在 → 返回存在的块
        # 有一个不在 → 返回 None（没命中）
```

> 💡 **生活类比**：
> - `block_hashes` = 你想买的书的清单
> - `ExternalCachedBlockPool` = 图书馆的藏书目录
> - `find_longest_cache_hit()` = 从第一本开始数，最多连续多少本图书馆有？
> - 结果 = 前面 N 本都有，从第 N+1 本开始没有

### 16.5 数据处理：紧凑哈希列表

上游有个巧妙的优化：`_CompactChunkHashList`。

#### 16.5.1 问题背景

vLLM 的块哈希是**链式的**：
- 块 0 的哈希 = hash(块 0 的内容)
- 块 1 的哈希 = hash(块 0 的哈希 + 块 1 的内容)
- 块 2 的哈希 = hash(块 1 的哈希 + 块 2 的内容)
- ...

也就是说，**第 n 块的哈希已经包含了前面所有块的信息**。

#### 16.5.2 优化思路

假设：
- `hash_block_size = 4`（每 4 个 token 算一个哈希）
- `block_size = 256`（每个 KV 块存 256 个 token）
- 那每个 KV 块对应 64 个哈希值

如果把 64 个哈希都存起来当 key，太长了！

但是：
- 第 64 个哈希（最后一个）已经包含了前面 63 个的信息
- 所以只用最后一个哈希当 key 就够了！

```
普通方式（64 个哈希当 key）：
  [hash0, hash1, hash2, ..., hash63]  → 太长了！

紧凑方式（只用最后 1 个哈希当 key）：
  hash63  → 但它已经包含了前面所有的信息！
```

这就是 `chunk_hashes_for_block_size()` 做的事情。

> 🎯 **一句话总结**：用块里最后一个子哈希当整个块的 key，既保证唯一性，又节省空间。

### 16.6 Store 模式支持的模型类型

Store 模式不是所有模型都支持，有一些限制：

| 模型类型 | 支持情况 | 说明 |
|---------|---------|------|
| 普通 Full Attention | ✅ 支持 | 标准场景 |
| Sliding Window Attention | ✅ 支持 | 滑动窗口 |
| Mamba | ⚠️ 有限支持 | 需要 block_size 和 cache_config.block_size 一致 |
| Cross Attention | ❌ 不支持 | 交叉注意力暂不支持 |
| 混合注意力 + PCP/DCP | ❌ 不支持 | 多个组 + 上下文并行的组合暂不支持 |

---

## ⚫ 总结与附录

---

## 17. vllm-ascend 与上游 vLLM 详细对比

### 17.1 架构对比表

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

### 17.2 上游 vLLM 的优势

1. **更通用**：支持多种后端、多种传输协议
2. **更完善的可观测性**：Prometheus 指标、分层监控
3. **服务发现机制**：etcd 或 bootstrap server，更灵活
4. **拓扑感知**：TransferTopology 感知网络拓扑，优化传输路径
5. **异构 TP 支持**：Producer 和 Consumer 的 TP 大小可以不同

### 17.3 vllm-ascend 的扩展特性

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

### 17.4 代码风格差异

| 方面 | 上游 vLLM | vllm-ascend |
|------|-----------|-------------|
| **类型注解** | 完善 | 基本完善 |
| **注释量** | 较多 | 较少 |
| **抽象层次** | 更高（更通用） | 更具体（更偏向NPU优化） |
| **配置方式** | 环境变量 + 配置对象 | 环境变量为主 |

---

## 18. 附录：代码索引与环境变量

### 18.1 核心类位置速查表

| 类名 | 文件 | 行号 |
|------|------|------|
| `MooncakeAgentMetadata` | `mooncake_connector.py` | 95 |
| `ReqMeta` | `mooncake_connector.py` | 110 |
| `GroupPull` | `mooncake_connector.py` | 129 |
| `KVCacheTaskTracker` | `mooncake_connector.py` | 164 |
| `KVCacheSendingThread` | `mooncake_connector.py` | 243 |
| `KVCacheRecvingThread` | `mooncake_connector.py` | 407 |
| `MooncakeConnectorMetadata` | `mooncake_connector.py` | 1374 |
| `MooncakeConnector` | `mooncake_connector.py` | 1405 |
| `MooncakeConnectorScheduler` | `mooncake_connector.py` | 1523 |
| `MooncakeConnectorWorker` | `mooncake_connector.py` | 1860 |
| `GlobalTE` | `mooncake_transfer_engine.py` | 4 |

### 18.2 关键函数位置速查表

| 函数 | 文件 | 行号 |
|------|------|------|
| `register_kv_caches` (Worker) | `mooncake_connector.py` | 2190 |
| `_get_kv_split_metadata` | `mooncake_connector.py` | 2518 |
| `_get_group_pulls_metadata` | `mooncake_connector.py` | 2962 |
| `_transfer_kv_cache_all_groups` | `mooncake_connector.py` | 730 |
| `reformat_kv_cache` | `mooncake_connector.py` | 1145 |
| `reformat_kv_cache_hybrid_linear_torch` | `mooncake_connector.py` | 984 |
| `collect_storage_merged_register_regions` | `utils.py` | 363 |
| `validate_register_region_count` | `utils.py` | 428 |
| `get_transfer_timeout_value` | `utils.py` | 55 |
| `align_memory` | `utils.py` | 48 |

### 18.3 环境变量大全

| 环境变量 | 作用 | 默认值 |
|---------|------|--------|
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` | 请求超时强制释放时间（秒） | - |
| `ASCEND_TRANSFER_TIMEOUT` | 昇腾传输超时（毫秒），设了就用这个值 | 自动计算 |
| `HCCL_RDMA_TIMEOUT` | HCCL RDMA 超时指数（用于自动计算超时） | 20 |
| `HCCL_RDMA_RETRY_CNT` | HCCL RDMA 重试次数（用于自动计算超时） | 7 |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | 是否用融合算子逐块转置 KV 缓存 | - |

### 18.4 常量大全

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
