# Mooncake Store Connector 详解

> 📖 **阅读指南**
>
> 本文档详细讲解 **Store 模式的 Mooncake Connector**（又称 KV Pool / 分布式 KV 存储），从 vLLM 上游基类开始，逐层深入到 vLLM-Ascend 的具体实现。
>
> **前置知识**：建议先阅读 [整体架构指南](../mooncake_connector_arch.md)
>
> **注意**：vLLM-Ascend 中的推荐实现是 **AscendStoreConnector**，它比上游的 MooncakeStoreConnector 功能更丰富。本文档以 AscendStoreConnector 为主，同时对比上游实现。

---

## 目录

### 🟢 入门篇
- [1. Store 模式回顾](#1-store-模式回顾)
- [2. 核心概念：哈希、前缀缓存、KV Pool](#2-核心概念哈希前缀缓存kv-pool)

### 🔵 架构篇
- [3. 完整继承链](#3-完整继承链)
- [4. 核心设计模式](#4-核心设计模式)
- [5. 多后端架构](#5-多后端架构)

### 🟠 上游 vLLM 篇
- [6. 上游 MooncakeStoreConnector 总览](#6-上游-mooncakestoreconnector-总览)
- [7. 上游 Scheduler 侧实现](#7-上游-scheduler-侧实现)
- [8. 上游 Worker 侧实现](#8-上游-worker-侧实现)

### 🔴 vLLM-Ascend 篇
- [9. AscendStoreConnector 总览](#9-ascendstoreconnector-总览)
- [10. KVPoolScheduler 详解](#10-kvpoolscheduler-详解)
- [11. KVPoolWorker 详解](#11-kvpoolworker-详解)
- [12. Coordinator 协调器](#12-coordinator-协调器)
- [13. 多后端抽象：Mooncake / Memcached / 元融](#13-多后端抽象mooncake--memcached--元融)

### 🟣 流程篇
- [14. 缓存命中查询流程](#14-缓存命中查询流程)
- [15. KV 缓存加载（Get）流程](#15-kv-缓存加载get流程)
- [16. KV 缓存存储（Put）流程](#16-kv-缓存存储put流程)

### ⚫ 数据结构篇
- [17. BlockHash 与缓存键](#17-blockhash-与缓存键)
- [18. LoadSpec 与 RequestTracker](#18-loadspec-与-requesttracker)
- [19. KeyMetadata](#19-keymetadata)

---

## 🟢 入门篇

---

## 1. Store 模式回顾

### 1.1 什么是 Store 模式？

Store 模式（又称 KV Pool 模式）是 Mooncake 的另一种工作模式：

```
┌─────────────┐
│  Prefill 1  │ ──┐
└─────────────┘   │  put（写入）
                  ▼
           ┌─────────────────┐
           │  KV Pool        │
           │  (Distributed   │
           │   Store)        │ ◄───────┐
           │                 │         │ get（读取）
           └─────────────────┘         │
                  ▲                    │
┌─────────────┐   │       ┌─────────────┐
│  Prefill 2  │ ──┘       │  Decode 1   │ ──┘
└─────────────┘           └─────────────┘
```

**核心特点**：
- 所有节点通过**共享存储池**交换 KV 缓存
- Producer 把 KV **写入**存储池（put）
- Consumer 从存储池**读取** KV（get）
- 支持**前缀缓存复用**：相同前缀的请求可以共享 KV

### 1.2 生活化类比

想象**公共冰箱**的场景：
- **餐厅（Prefill）**：做好菜放进公共冰箱
- **顾客（Decode）**：去冰箱里拿自己想吃的菜
- **公共冰箱（KV Pool）**：大家都可以存取

相比 P2P 模式（外卖直送）：
- P2P：一份菜只能给一个人
- Store：一份菜可以给多个人吃（前缀复用）

### 1.3 为什么需要前缀缓存复用？

在实际应用中，很多请求有相同的前缀：
- 系统提示词（System Prompt）
- 同一个文档的问答
- 多轮对话的历史

如果每次都重新计算这些前缀的 KV，会浪费大量算力。Store 模式通过**哈希去重**，让相同前缀的 KV 只计算一次。

---

## 2. 核心概念：哈希、前缀缓存、KV Pool

### 2.1 BlockHash：块哈希

KV 缓存的最小传输单位是 **block**（通常 16 个 token）。每个 block 有一个唯一的哈希值：

```
Block 0 (tokens 0-15)   → Hash: 0xabc123
Block 1 (tokens 16-31)  → Hash: 0xdef456
Block 2 (tokens 32-47)  → Hash: 0xghi789
```

哈希是根据 token IDs 计算的，相同内容的 block 哈希相同。

### 2.2 前缀缓存复用原理

假设有两个请求：
- 请求 A："你好，请介绍一下人工智能"
- 请求 B："你好，请介绍一下机器学习"

它们的前几个 token 是相同的：

```
请求 A: [你好, 请, 介绍, 一下, 人工, 智能]
            ↓     ↓    ↓    ↓
请求 B: [你好, 请, 介绍, 一下, 机器, 学习]
```

对应的 KV block：
```
请求 A: Block0, Block1, Block2, Block3, Block4a, Block5a
请求 B: Block0, Block1, Block2, Block3, Block4b, Block5b
         ↑       ↑       ↑       ↑
         相同哈希，可以复用！
```

Store 模式就是利用这一点，把每个 block 按哈希存储，相同哈希的 block 只存一份。

### 2.3 KV Pool 架构

KV Pool 是一个**分布式键值存储系统**：

```
Key: BlockHash (哈希)
Value: KV Cache Data (实际的 KV 缓存数据)
```

特点：
- **内容寻址**：通过哈希查找内容
- **去重存储**：相同内容只存一份
- **共享访问**：所有节点都可以读写

---

## 🔵 架构篇

---

## 3. 完整继承链

### 3.1 继承关系图

```
┌─────────────────────────────────────────────────────────────┐
│                    KVConnectorBase_V1                       │
│              (vllm/distributed/kv_transfer/                 │
│               kv_connector/v1/base.py)                      │
│                     ▲                                       │
│                     │ 继承                                  │
│                     │                                       │
│          ┌──────────┴──────────┐                           │
│          │  SupportsHMA        │                           │
│          │  (混合内存支持)     │                           │
│          └──────────┬──────────┘                           │
│                     │ 继承                                  │
│                     │                                       │
└─────────────────────┼───────────────────────────────────────┘
                      │
          ┌───────────┴─────────────┐
          │                         │
┌─────────▼──────────┐   ┌──────────▼─────────────────┐
│ 上游 vLLM          │   │ vLLM-Ascend 扩展           │
│                    │   │                             │
│ MooncakeStore-     │   │ AscendStoreConnector        │
│ Connector          │   │ (推荐使用)                 │
│ (基础版 Store)     │   │                             │
│                    │   │ - 多后端支持               │
│ - 基础 Store 功能  │   │ - Coordinator 协调         │
│ - 哈希前缀缓存     │   │ - 逐层传输                 │
│ - cross-layer      │   │ - MLA / Mamba / SWA 支持   │
│   blocks 优化      │   │ - 更完善的 HMA             │
└────────────────────┘   └────────────────────────────┘
```

### 3.2 为什么推荐 AscendStoreConnector？

源码中有明确提示：
```python
if connector_name == "MooncakeConnectorStoreV1":
    logger.warning(
        "It is recommended to use the AscendStoreConnector, "
        "as the MoonCakeStoreConnector will be removed in the future."
    )
```

**原因**：
- AscendStoreConnector 功能更完善
- 支持多种后端（不只是 Mooncake）
- 性能更好
- 持续维护中

---

## 4. 核心设计模式

### 4.1 Scheduler + Worker 分离

和 P2P 模式一样，Store 模式也遵循 Scheduler + Worker 分离的设计：

```
┌──────────────────────────────────────────────────────────┐
│                AscendStoreConnector                      │
│                                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │  KVPoolScheduler (调度侧)                         │   │
│  │  - 计算 block 哈希                                │   │
│  │  - 查询缓存命中                                   │   │
│  │  - 构建加载/存储元数据                            │   │
│  └─────────────────────┬─────────────────────────────┘   │
│                        │ 元数据 (Metadata)                │
│                        ▼                                  │
│  ┌───────────────────────────────────────────────────┐   │
│  │  KVPoolWorker (工作侧)                            │   │
│  │  - 管理 KV 传输线程                               │   │
│  │  - 调用后端进行 get/put                           │   │
│  │  - Coordinator 协调                               │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### 4.2 三层架构（AscendStoreConnector 特有）

AscendStoreConnector 有三层：

```
┌─────────────────────────────────────────┐
│   AscendStoreConnector (接口层)        │
│   - 继承 KVConnectorBase_V1            │
│   - 对外提供统一接口                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   KVPoolScheduler / KVPoolWorker       │
│   (逻辑层)                              │
│   - 调度逻辑 / 传输逻辑                 │
│   - 哈希计算 / 块管理                   │
│   - Coordinator 协调                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Backend (存储后端层)                  │
│   - MooncakeBackend                     │
│   - MemcacheBackend                     │
│   - YuanrongBackend                     │
└─────────────────────────────────────────┘
```

### 4.3 P 侧和 D 侧的方法行为差异

> 💡 **前置知识**：Store 模式和 P2P 模式遵循同样的设计思想——**相同方法名，不同角色走不同分支**。
>
> 详细原理可参考 [整体架构指南：P/D 侧方法行为对比](../../mooncake_connector_arch.md#47-关键设计p-侧和-d-侧调用相同方法行为不同)

Store 模式的 P/D 侧差异与 P2P 模式略有不同，因为**两侧都可以读写 KV Pool**，但侧重点不同：

| 方法 | D 侧（Consumer / Decoder） | P 侧（Producer / Prefill） |
|------|--------------------------|--------------------------|
| `get_num_new_matched_tokens()` | ✅ 查询缓存命中数量 | ❌ 一般不查询（或查询自己写的） |
| `update_state_after_alloc()` | ✅ 准备加载缓存（get） | ✅ 准备存储缓存（put） |
| `build_connector_meta()` | ✅ 构建加载元数据 | ✅ 构建存储元数据 |
| `request_finished()` | ✅ 可选：把新生成的 KV 写回存储 | ✅ 把计算好的 KV 写入存储 |

**注意**：Store 模式中，`consumer_is_to_put` 配置可以让 D 侧也把新生成的 KV 写回存储池，实现更多的缓存复用机会。

---

## 5. 多后端架构

### 5.1 后端类型

AscendStoreConnector 支持多种存储后端：

| 后端 | 名称 | 特点 | 适用场景 |
|------|------|------|---------|
| **Mooncake** | MooncakeBackend | 高性能 RDMA 传输 | 同机房、低延迟要求 |
| **Memcached** | MemcacheBackend | 通用内存缓存 | 测试、通用场景 |
| **元融** | YuanrongBackend | 元融存储后端 | 特定硬件环境 |

### 5.2 后端映射

**文件**：`kv_pool/ascend_store/pool_worker.py:54`

```python
backend_map = {
    "mooncake": {
        "name": "MooncakeBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend",
    },
    "memcache": {
        "name": "MemcacheBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend",
    },
    "yuanrong": {
        "name": "YuanrongBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend",
    },
}
```

### 5.3 后端接口规范

所有后端都实现统一的接口：

```python
class BackendBase:
    def get(self, keys: list[bytes]) -> list[bytes | None]:
        """批量读取"""
        pass
    
    def put(self, keys: list[bytes], values: list[bytes]) -> None:
        """批量写入"""
        pass
    
    def delete(self, keys: list[bytes]) -> None:
        """批量删除"""
        pass
    
    def exists(self, keys: list[bytes]) -> list[bool]:
        """批量检查存在性"""
        pass
```

---

## 🟠 上游 vLLM 篇

---

## 6. 上游 MooncakeStoreConnector 总览

### 6.1 文件位置

**主文件**：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py`

### 6.2 核心类一览

```
MooncakeStoreConnector (主类)
│
├── MooncakeStoreScheduler (调度侧)
│   ├── get_num_new_matched_tokens()
│   ├── update_state_after_alloc()
│   └── build_connector_meta()
│
├── MooncakeStoreWorker (工作侧)
│   ├── MooncakeDistributedStore (分布式存储)
│   ├── register_kv_caches()
│   ├── start_load_kv()
│   └── save_kv_layer()
│
└── MooncakeStoreConnectorMetadata (元数据)
```

### 6.3 主要特点

- 基于 Mooncake 的分布式存储
- 使用哈希进行前缀缓存去重
- 支持 cross-layer blocks 优化
- 支持 Mamba、SlidingWindow 等特殊模型

---

## 7. 上游 Scheduler 侧实现

### 7.1 get_num_new_matched_tokens

**作用**：查询远端存储中有多少缓存可以命中。

```python
def get_num_new_matched_tokens(
    self, request: "Request", num_computed_tokens: int
) -> tuple[int, bool]:
    # 1. 计算每个 block 的哈希
    block_hashes = self._compute_block_hashes(request, num_computed_tokens)
    
    # 2. 查询哪些哈希在存储中存在
    cached_hashes = self._query_cache_exists(block_hashes)
    
    # 3. 计算命中的 token 数
    num_cached_tokens = self._count_cached_tokens(
        block_hashes, cached_hashes, num_computed_tokens
    )
    
    return num_cached_tokens, True  # (数量, 异步)
```

### 7.2 update_state_after_alloc

**作用**：分配块后，准备加载缓存。

### 7.3 build_connector_meta

**作用**：构建加载/存储元数据。

---

## 8. 上游 Worker 侧实现

### 8.1 MooncakeDistributedStore

上游使用 Mooncake 自带的分布式存储：

```python
from mooncake.store import MooncakeDistributedStore

self.store = MooncakeDistributedStore(...)
```

### 8.2 主要操作

| 操作 | 方法 | 说明 |
|------|------|------|
| **读取** | `start_load_kv()` | 异步加载 KV 缓存 |
| **写入** | `save_kv_layer()` | 异步保存某层 KV |
| **等待** | `wait_for_layer_load()` | 等待某层加载完成 |

---

## 🔴 vLLM-Ascend 篇

---

## 9. AscendStoreConnector 总览

### 9.1 文件位置

**主文件**：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`

**相关文件**：
- `pool_scheduler.py` - 调度侧实现
- `pool_worker.py` - 工作侧实现
- `coordinator.py` - 协调器
- `kv_transfer.py` - 传输线程
- `config_data.py` - 数据结构
- `backend/` - 多后端实现

### 9.2 主类定义

源码位置：`ascend_store_connector.py:73`

```python
class AscendStoreConnector(KVConnectorBase_V1, SupportsHMA):
    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """逐层模式下需要 PIECEWISE CUDA graph"""
        return extra_config.get("use_layerwise", False)
    
    def __init__(
        self, 
        vllm_config: VllmConfig, 
        role: KVConnectorRole, 
        kv_cache_config: KVCacheConfig | None = None
    ):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        
        # 特性开关
        self.use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "use_layerwise", False
        )
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )
        
        # 根据角色初始化
        if role == KVConnectorRole.SCHEDULER:
            self.pool_scheduler = KVPoolScheduler(...)
            self.pool_worker = None
        elif role == KVConnectorRole.WORKER:
            self.pool_scheduler = None
            self.pool_worker = KVPoolWorker(...)
```

### 9.3 核心特性

| 特性 | 说明 |
|------|------|
| **逐层传输** | `use_layerwise`：一层一层传输，节省内存 |
| **Consumer 也写** | `consumer_is_to_put`：消费端也把 KV 写回存储 |
| **异步加载** | `load_async`：异步加载，隐藏延迟 |
| **HMA 支持** | 混合内存分配器，支持多种 KV 类型 |
| **MLA 支持** | DeepSeek MLA 模型 |
| **Mamba 支持** | 有状态模型 |
| **SWA 支持** | 滑动窗口注意力 |
| **PCP/DCP** | 上下文并行 |

---

## 10. KVPoolScheduler 详解

### 10.1 初始化

源码位置：`pool_scheduler.py:38`

```python
class KVPoolScheduler:
    def __init__(self, vllm_config, use_layerwise, kv_cache_config=None):
        # 基本配置
        self.use_layerwise = use_layerwise
        self.kv_cache_config = kv_cache_config
        
        # 模型相关
        self.use_compress = self.compress_ratios is not None
        self.use_hybrid = self._uses_hybrid_kv_cache(...)
        self.need_truncate = self.use_compress
        self.num_swa_blocks = self._infer_swa_blocks()
        
        # 角色配置
        self.consumer_is_to_load = ...
        self.consumer_is_to_put = ...
        self.load_async = ...
        
        # 查找客户端（查询缓存命中）
        self.client = LookupKeyClient(vllm_config)
        
        # 加载规格（每个请求的缓存命中情况）
        self.load_specs: dict[str, LoadSpec] = {}
        
        # 并行维度
        self.pcp_size = ...
        self.dcp_size = ...
        
        # 块大小
        self.original_block_size = self._infer_group_block_sizes(...)
        self.grouped_block_size = [block_size * cp_scale for ...]
        self.hash_block_size = ...
```

### 10.2 核心方法

| 方法 | 作用 |
|------|------|
| `get_num_new_matched_tokens()` | 查询缓存命中数量 |
| `update_state_after_alloc()` | 分配块后更新状态 |
| `build_connector_meta()` | 构建传输元数据 |
| `request_finished()` | 请求完成回调 |
| `take_events()` | 获取 KV 事件 |

### 10.3 get_num_new_matched_tokens 详解

这是 Store 模式最核心的方法，也是和 P2P 模式最大的区别。

**步骤**：
1. 计算请求每个 block 的哈希
2. 查询这些哈希在 KV Pool 中是否存在
3. 计算连续命中的前缀长度
4. 返回命中的 token 数

```python
def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
    # 1. 计算 block 哈希
    block_hashes = self._compute_block_hashes(request, num_computed_tokens)
    
    # 2. 查询缓存命中
    cached = self.client.lookup(block_hashes)
    
    # 3. 计算连续前缀命中长度
    num_cached_blocks = 0
    for i, h in enumerate(block_hashes):
        if h in cached:
            num_cached_blocks += 1
        else:
            break  # 前缀不连续了，停止
    
    # 4. 转换为 token 数
    num_cached_tokens = num_cached_blocks * self.hash_block_size
    
    # 处理截断（Mamba 等有状态模型）
    if self.need_truncate and num_cached_tokens > 1:
        num_cached_tokens -= 1
    
    return num_cached_tokens, self.load_async
```

---

## 11. KVPoolWorker 详解

### 11.1 初始化

源码位置：`pool_worker.py:70`

```python
class KVPoolWorker:
    def __init__(self, vllm_config, use_layerwize, kv_cache_config=None):
        # 基本配置
        self.use_compress = self.compress_ratios is not None
        self.use_mla = ...
        self.use_sparse = ...
        self.use_layerwise = use_layerwize
        
        # 并行信息
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = ...
        self.pp_rank = ...
        
        # KV 缓存信息
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.block_size = vllm_config.cache_config.block_size
        
        # 存储后端（动态加载）
        self.backend = self._init_backend(vllm_config)
        
        # Coordinator（协调器）
        self.coordinator = AscendStoreCoordinator(...)
        
        # 传输线程
        self.send_thread = KVCacheStoreSendingThread(...)
        self.recv_thread = KVCacheStoreRecvingThread(...)
        
        # 逐层模式专用
        if self.use_layerwise:
            self.layer_send_thread = KVCacheStoreLayerSendingThread(...)
            self.layer_recv_thread = KVCacheStoreLayerRecvingThread(...)
```

### 11.2 核心方法

| 方法 | 作用 |
|------|------|
| `register_kv_caches()` | 注册 KV 缓存内存 |
| `start_load_kv()` | 开始加载 KV（get） |
| `save_kv_layer()` | 保存某层 KV（put） |
| `wait_for_layer_load()` | 等待某层加载完成 |
| `get_finished()` | 获取完成的请求 |

### 11.3 传输线程架构

```
┌─────────────────────────────────────────────────────────┐
│                      KVPoolWorker                       │
│                                                         │
│  ┌─────────────────────┐    ┌──────────────────────┐   │
│  │  SendingThread      │    │  RecvingThread       │   │
│  │  (写入存储池)       │    │  (从存储池读取)      │   │
│  │                     │    │                      │   │
│  │ - 收集要 put 的块   │    │ - 收集要 get 的块    │   │
│  │ - 调用 backend.put  │    │ - 调用 backend.get   │   │
│  │ - 更新 Coordinator  │    │ - 更新 Coordinator   │   │
│  └──────────┬──────────┘    └──────────┬───────────┘   │
│             │                           │               │
│             └───────────┬───────────────┘               │
│                         │                               │
│                ┌────────▼────────┐                      │
│                │   Backend        │                     │
│                │  (Mooncake/      │                     │
│                │   Memcached/      │                     │
│                │   Yuanrong)       │                     │
│                └───────────────────┘                     │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Coordinator (协调器)                            │   │
│  │  - 管理块引用计数                                │   │
│  │  - 管理外部块池                                  │   │
│  │  - 协调多节点                                    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 12. Coordinator 协调器

### 12.1 作用

Coordinator 是 AscendStoreConnector 的"大管家"，负责：
- 管理外部缓存块的引用计数
- 管理外部块池（ExternalCachedBlockPool）
- 协调多节点之间的缓存状态
- 处理缓存淘汰

### 12.2 主要组件

```
AscendStoreCoordinator
├── ExternalCachedBlockPool (外部块池)
│   ├── 已缓存的 block 哈希
│   ├── 引用计数
│   └── LRU 淘汰策略
│
├── ref_cnt: dict[BlockHash, int]
│   └── 每个块的引用计数
│
└── 节点协调
    ├── 本地节点状态
    └── 远端节点状态同步
```

---

## 13. 多后端抽象：Mooncake / Memcached / 元融

### 13.1 后端基类

**文件**：`backend/backend.py`

```python
class BackendBase:
    """存储后端的抽象基类"""
    
    def get(self, keys: list[bytes]) -> list[bytes | None]:
        """批量读取，返回顺序与 keys 对应，不存在的返回 None"""
        raise NotImplementedError
    
    def put(self, keys: list[bytes], values: list[bytes]) -> None:
        """批量写入"""
        raise NotImplementedError
    
    def delete(self, keys: list[bytes]) -> None:
        """批量删除"""
        raise NotImplementedError
    
    def exists(self, keys: list[bytes]) -> list[bool]:
        """批量检查存在性"""
        raise NotImplementedError
```

### 13.2 MooncakeBackend

使用 Mooncake 分布式存储作为后端：
- 高性能 RDMA 传输
- 低延迟
- 适合同机房部署

**文件**：`backend/mooncake_backend.py`

### 13.3 MemcacheBackend

使用 Memcached 作为后端：
- 通用、成熟
- 部署简单
- 适合测试和通用场景

**文件**：`backend/memcache_backend.py`

### 13.4 YuanrongBackend

元融存储后端：
- 特定硬件优化
- 适合特定环境

**文件**：`backend/yuanrong_backend.py`

---

## 🟣 流程篇

---

## 14. 缓存命中查询流程

### 14.1 场景

新请求到来，Scheduler 想知道有多少前缀 KV 已经在 KV Pool 里了。

### 14.2 完整流程

```
Scheduler 进程
┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. 新请求到来                                      │
│     │                                               │
│     ▼                                               │
│  2. get_num_new_matched_tokens(request, 0)          │
│     │                                               │
│     ├─ 2a. 计算 prompt 每个 block 的哈希            │
│     │                                               │
│     ├─ 2b. 调用 LookupKeyClient 查询这些哈希        │
│     │    └── 向 KV Pool 发查询请求                  │
│     │                                               │
│     ├─ 2c. 收到查询结果                             │
│     │    └── 哪些哈希存在，哪些不存在               │
│     │                                               │
│     ├─ 2d. 计算连续前缀命中长度                     │
│     │    └── 从第一个 block 开始数，连续命中几个   │
│     │                                               │
│     └─ 2e. 返回 (num_cached_tokens, is_async)      │
│                                                     │
│  3. 根据命中数决定调度策略                          │
│     - 如果全部命中：直接用缓存，不用重新计算       │
│     - 如果部分命中：从命中位置开始计算             │
│     - 如果没命中：  从头计算                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 15. KV 缓存加载（Get）流程

### 15.1 场景

Decoder 侧（Consumer）从 KV Pool 中加载命中的前缀 KV 缓存。

### 15.2 完整流程

```
Scheduler 进程                     Worker 进程                  KV Pool
┌──────────────┐                   ┌──────────────┐           ┌──────────────┐
│ 1. 分配 KV   │                   │              │           │              │
│    blocks    │                   │              │           │              │
└──────┬───────┘                   │              │           │              │
       │                           │              │           │              │
       ▼                           │              │           │              │
┌──────────────┐                   │              │           │              │
│ 2. update_   │                   │              │           │              │
│    state_    │  准备加载         │              │           │              │
│    after_    │──────────────────►│              │           │              │
│    alloc     │                   │              │           │              │
└──────┬───────┘                   │              │           │              │
       │                           │              │           │              │
       ▼                           │              │           │              │
┌──────────────┐                   │              │           │              │
│ 3. build_    │                   │              │           │              │
│    connector_│  构建加载元数据   │              │           │              │
│    meta      │──────────────────►│              │           │              │
└──────┬───────┘                   │              │           │              │
       │                           │              │           │              │
       └──────────────────────────►│              │           │              │
                                   │              │           │              │
                                   ▼              │           │              │
                          ┌──────────────┐        │           │              │
                          │ 4. start_    │        │           │              │
                          │    load_kv   │        │           │              │
                          └──────┬───────┘        │           │              │
                                 │                │           │              │
                                 ▼                │           │              │
                          ┌──────────────┐        │           │              │
                          │ 5. 收集要     │        │           │              │
                          │    get 的块   │        │           │              │
                          └──────┬───────┘        │           │              │
                                 │                │           │              │
                                 └───────────────────────────►│              │
                                                              │              │
                                                              ▼              │
                                                     ┌──────────────┐        │
                                                     │ 6. 查询并     │        │
                                                     │    返回数据   │        │
                                                     └──────┬───────┘        │
                                                            │                │
                                 ◄──────────────────────────┘                │
                                   │                                        │
                                   ▼                                        │
                          ┌──────────────┐                                   │
                          │ 7. 接收数据   │                                   │
                          │    写入本地   │                                   │
                          │    KV 缓存    │                                   │
                          └──────┬───────┘                                   │
                                 │                                           │
                                 ▼                                           │
                          ┌──────────────┐                                   │
                          │ 8. 通知完成   │                                   │
                          └──────────────┘                                   │
                                                                             │
┌──────────────┐                            ▲                                │
│ 9. get_     │  完成的请求               │                                │
│    finished  │◄───────────────────────────┘                                │
└──────────────┘                                                             │
                                                                             │
                                                                             └──────────────┘
```

---

## 16. KV 缓存存储（Put）流程

### 16.1 场景

Prefill 侧（Producer）把计算好的 KV 缓存写入 KV Pool，供其他节点复用。

### 16.2 完整流程

```
Scheduler 进程                     Worker 进程                  KV Pool
┌──────────────┐                   ┌──────────────┐           ┌──────────────┐
│ 1. 请求计算   │                   │              │           │              │
│    完成       │                   │              │           │              │
└──────┬───────┘                   │              │           │              │
       │                           │              │           │              │
       ▼                           │              │           │              │
┌──────────────┐                   │              │           │              │
│ 2. request_  │  延迟释放块       │              │           │              │
│    finished  │──────────────────►│              │           │              │
└──────┬───────┘                   │              │           │              │
       │                           │              │           │              │
       ▼                           │              │           │              │
┌──────────────┐                   │              │           │              │
│ 3. build_    │                   │              │           │              │
│    connector_│  构建存储元数据   │              │           │              │
│    meta      │──────────────────►│              │           │              │
└──────┬───────┘                   │              │           │              │
       │                           │              │           │              │
       └──────────────────────────►│              │           │              │
                                   │              │           │              │
                                   ▼              │           │              │
                          ┌──────────────┐        │           │              │
                          │ 4. 收集要     │        │           │              │
                          │    put 的块   │        │           │              │
                          └──────┬───────┘        │           │              │
                                 │                │           │              │
                                 └───────────────────────────►│              │
                                                              │              │
                                                              ▼              │
                                                     ┌──────────────┐        │
                                                     │ 5. 写入存储   │        │
                                                     └──────┬───────┘        │
                                                            │                │
                                 ◄──────────────────────────┘                │
                                   │                                        │
                                   ▼                                        │
                          ┌──────────────┐                                   │
                          │ 6. 更新完成   │                                   │
                          │    状态       │                                   │
                          └──────┬───────┘                                   │
                                 │                                           │
                                 ▼                                           │
                          ┌──────────────┐                                   │
                          │ 7. 更新       │                                   │
                          │    Coordinator│                                   │
                          │    引用计数   │                                   │
                          └──────────────┘                                   │
                                                                             │
┌──────────────┐                            ▲                                │
│ 8. get_     │  完成的请求               │                                │
│    finished  │◄───────────────────────────┘                                │
└──────┬───────┘                                                             │
       │                                                                     │
       ▼                                                                     │
┌──────────────┐                                                             │
│ 9. 释放 KV   │                                                             │
│    blocks    │                                                             │
└──────────────┘                                                             │
                                                                             │
                                                                             └──────────────┘
```

---

## ⚫ 数据结构篇

---

## 17. BlockHash 与缓存键

### 17.1 BlockHash

`BlockHash` 是 block 的哈希值，用于在 KV Pool 中唯一标识一个 block 的内容。

```python
# 计算方式：基于 token IDs 计算哈希
BlockHash = int  # 通常是 64 位整数
```

### 17.2 缓存键的构成

在 KV Pool 中，一个完整的缓存键通常包含：
- Block 哈希（内容）
- 层号（哪一层）
- TP 排名（哪个张量并行分片）
- K/V 标识（是 key 还是 value）

```
Key = hash(block_hash, layer, tp_rank, is_key)
Value = KV 缓存数据
```

---

## 18. LoadSpec 与 RequestTracker

### 18.1 LoadSpec

**作用**：记录一个请求的缓存命中情况。

```python
@dataclass
class LoadSpec:
    request_id: str
    num_cached_blocks: int          # 命中的 block 数
    num_cached_tokens: int          # 命中的 token 数
    block_hashes: list[BlockHash]   # 所有 block 的哈希
    cached_hashes: set[BlockHash]   # 命中的哈希集合
    local_block_ids: BlockIds      # 本地 block ID
    load_finished: bool            # 是否加载完成
```

### 18.2 RequestTracker

**作用**：追踪请求的传输状态。

```python
@dataclass
class RequestTracker:
    request_id: str
    total_blocks: int               # 总 block 数
    loaded_blocks: int              # 已加载的 block 数
    saved_blocks: int               # 已存储的 block 数
    is_load_complete: bool         # 加载是否完成
    is_save_complete: bool         # 存储是否完成
```

---

## 19. KeyMetadata

### 19.1 作用

`KeyMetadata` 是 KV Pool 中存储的元数据，描述一个 block 的信息。

### 19.2 包含的信息

```python
@dataclass
class KeyMetadata:
    block_hash: BlockHash          # block 哈希
    layer_idx: int                 # 层索引
    tp_rank: int                   # TP 排名
    block_size: int                # block 大小（token 数）
    data_size: int                 # 数据大小（字节）
    create_time: float             # 创建时间
    ref_count: int                 # 引用计数
    checksum: int                  # 校验和
```

---

## 总结

本文档从 vLLM 上游基类开始，逐层深入讲解了 Store 模式 Mooncake Connector 的实现。关键要点：

1. **核心思想**：基于哈希的前缀缓存复用，相同内容的 KV 只存一份
2. **统一接口**：继承自 `KVConnectorBase_V1`，和 P2P 模式使用相同接口
3. **三层架构**：接口层 → 逻辑层 → 存储后端层
4. **多后端支持**：Mooncake、Memcached、元融，可灵活切换
5. **Coordinator**：协调器管理引用计数和块池
6. **两种操作**：get（读取）和 put（写入）
7. **Ascend 增强**：vLLM-Ascend 的 AscendStoreConnector 比上游功能更丰富

**文档体系回顾**：
- ✅ [整体架构指南](../mooncake_connector_arch.md)
- ✅ [P2P MooncakeConnector 详解](./mooncake_p2p_connector.md)
- ✅ 本文档：Store 模式详解
