# Mooncake P2P Connector 详解

> 📖 **阅读指南**
>
> 本文档详细讲解 **P2P 模式的 MooncakeConnector**，从 vLLM 上游基类开始，逐层深入到 vLLM-Ascend 的具体实现。
>
> **前置知识**：建议先阅读 [整体架构指南](../mooncake_connector_arch.md)
>
> **本文档结构**：
> - 入门篇：基础概念回顾
> - 架构篇：类继承关系与核心组件
> - 上游篇：vLLM 上游 MooncakeConnector 源码解析
> - Ascend 篇：vLLM-Ascend 扩展与优化
> - 流程篇：核心流程详解
> - 数据结构篇：关键数据结构详解

---

## 目录

### 🟢 入门篇
- [1. P2P 模式回顾](#1-p2p-模式回顾)
- [2. 统一模型配置说明](#2-统一模型配置说明)

### 🔵 架构篇
- [3. 完整继承链](#3-完整继承链)
- [4. 核心设计模式](#4-核心设计模式)

### 🟠 上游 vLLM 篇
- [5. 上游 MooncakeConnector 总览](#5-上游-mooncakeconnector-总览)
- [6. 上游 Scheduler 侧实现](#6-上游-scheduler-侧实现)
- [7. 上游 Worker 侧实现](#7-上游-worker-侧实现)

### 🔴 vLLM-Ascend 篇
- [8. Ascend 版 MooncakeConnector 总览](#8-ascend-版-mooncakeconnector-总览)
- [9. GlobalTE 单例优化](#9-globalte-单例优化)
- [10. Ascend Scheduler 侧扩展](#10-ascend-scheduler-侧扩展)
- [11. Ascend Worker 侧扩展](#11-ascend-worker-侧扩展)
- [12. 特殊模型支持：MLA、Mamba、SWA](#12-特殊模型支持mlamambaswa)

### 🟣 流程篇
- [13. KV 缓存注册流程](#13-kv-缓存注册流程)
- [14. Remote Prefill 全流程（Decoder 侧拉取）](#14-remote-prefill-全流程decoder-侧拉取)
- [15. Remote Decode 全流程（Prefill 侧推送）](#15-remote-decode-全流程prefill-侧推送)

### ⚫ 数据结构篇
- [16. 核心数据结构详解](#16-核心数据结构详解)

---

## 🟢 入门篇

---

## 1. P2P 模式回顾

### 1.1 什么是 P2P 模式？

P2P（Peer-to-Peer，点对点）模式是 Mooncake 最基础的传输模式：

```
┌─────────────────┐                         ┌─────────────────┐
│  Prefill 节点    │   KV 缓存直接传输       │   Decode 节点    │
│  (Producer)     │ ──────────────────────► │  (Consumer)     │
│                 │                         │                 │
│  - 计算 prompt  │                         │  - 生成 token   │
│    的 KV 缓存   │                         │  - 使用 KV 缓存 │
└─────────────────┘                         └─────────────────┘
```

**核心特点**：
- Producer 和 Consumer **一一对应**（或一对多）
- KV 缓存**直接传输**，不经过中间存储
- 延迟低，但每个请求的 KV 只能用一次

### 1.2 生活化类比

想象**餐厅外卖**：
- **Prefill 节点 = 餐厅后厨**：负责做菜（计算 KV）
- **Decode 节点 = 顾客家**：负责吃菜（用 KV 生成文本）
- **Mooncake = 外卖骑手**：直接把菜从餐厅送到顾客家

这种模式的好处是**快**（菜刚做好就送），但缺点是**每份菜只能给一个顾客**。

---

## 2. 统一模型配置说明

本文档所有示例都使用以下统一配置，方便对比理解：

| 参数 | 值 | 说明 |
|------|----|------|
| 模型 | Llama-3-8B | 标准 Transformer 模型 |
| 层数 | 32 | num_hidden_layers=32 |
| 头数 | 32 / 8 | num_attention_heads=32, num_key_value_heads=8 |
| Head Dim | 128 | hidden_size=4096, 4096/32=128 |
| Block Size | 16 | 每个 block 包含 16 个 token |
| TP 大小（P） | 8 | Prefill 侧 Tensor Parallel |
| TP 大小（D） | 2 | Decode 侧 Tensor Parallel |
| PP 大小 | 1 | 流水线并行 = 1（简化示例） |

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
│ MooncakeConnector  │   │ MooncakeConnector           │
│ (基础版 P2P)       │   │ (Ascend 优化版 P2P)        │
│                    │   │                             │
│ - 基本 P2P 传输    │   │ - NPU 适配                 │
│ - 异质 TP 支持     │   │ - GlobalTE 单例            │
│ - PP 支持          │   │ - MLA 模型支持             │
│                    │   │ - PCP/DCP 支持             │
└────────────────────┘   │ - HMA 更完善               │
                         │ - 逐层传输 / 混合模式       │
                         └────────────────────────────┘
```

### 3.2 三大变体（vLLM-Ascend）

vLLM-Ascend 提供了三种 P2P 传输变体：

| 变体 | 文件 | 特点 | 适用场景 |
|------|------|------|---------|
| **整体传输** | `mooncake_connector.py` | 一次性传输所有层 | 大多数场景，延迟最低 |
| **逐层传输** | `mooncake_layerwise_connector.py` | 一层一层传，支持流水线 | 模型很大、内存有限 |
| **混合传输** | `mooncake_hybrid_connector.py` | 结合整体和逐层 | 平衡延迟和内存 |

---

## 4. 核心设计模式

### 4.1 Scheduler + Worker 分离模式

所有 Mooncake Connector 都遵循这个模式：

```
┌──────────────────────────────────────────────────────────┐
│                   MooncakeConnector                      │
│                                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │  MooncakeConnectorScheduler (调度侧)              │   │
│  │  - 运行在 Scheduler 进程                          │   │
│  │  - 决定传什么、什么时候传                          │   │
│  │  - 构建元数据                                     │   │
│  └─────────────────────┬─────────────────────────────┘   │
│                        │ 元数据 (Metadata)                │
│                        ▼                                  │
│  ┌───────────────────────────────────────────────────┐   │
│  │  MooncakeConnectorWorker (工作侧)                 │   │
│  │  - 运行在 Worker 进程                             │   │
│  │  - 真正执行传输                                   │   │
│  │  - 管理 TransferEngine                            │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### 4.2 为什么要分离？

因为 vLLM 本身就是**多进程架构**：

| 进程 | 职责 | 能碰 GPU 吗？ |
|------|------|-------------|
| **Scheduler 进程** | 调度请求、管理块 | ❌ 不能 |
| **Worker 进程** | 执行模型、操作 KV 缓存 | ✅ 能 |

KV 传输既需要**调度决策**（Scheduler 侧：决定哪些请求要传），又需要**实际数据搬运**（Worker 侧：操作 GPU 内存、调用 RDMA）。

### 4.3 关键设计：P 侧和 D 侧调用相同方法，行为不同

这是理解 KV Connector 的核心：**Prefill 侧（Producer）和 Decode 侧（Consumer）调用的方法名完全一样，但内部逻辑根据角色走不同分支**。

#### 角色配置

通过 `kv_role` 配置决定角色：
- `kv_producer`：只生产 KV（Prefill 节点）
- `kv_consumer`：只消费 KV（Decode 节点）
- `kv_both`：既生产又消费

```python
self.is_kv_producer = (kv_transfer_config.kv_role == "kv_producer")
self.is_kv_consumer = (kv_transfer_config.kv_role == "kv_consumer")
```

#### Scheduler 侧方法对比

| 方法 | D 侧（Consumer / Decoder） | P 侧（Producer / Prefill） |
|------|--------------------------|--------------------------|
| `get_num_new_matched_tokens()` | ✅ 计算可从远端拉取的 token 数 | ❌ 直接返回 `(0, False)` |
| `update_state_after_alloc()` | ✅ 加入 `_reqs_need_recv`（接收队列） | ✅ 加入 `_reqs_need_send`（发送队列） |
| `build_connector_meta()` | ✅ 填充 `reqs_to_recv` | ✅ 填充 `reqs_to_send` |
| `request_finished()` | ❌ 立即释放块 | ✅ 延迟释放，等 KV 发送完成 |

#### Worker 侧方法对比

| 方法/组件 | D 侧（Consumer / Decoder） | P 侧（Producer / Prefill） |
|----------|--------------------------|--------------------------|
| `start_load_kv()` | ✅ 主动发起拉取请求 | ❌ 不执行 |
| `save_kv_layer()` | ❌ 不执行 | ✅ 被请求时发送 KV |
| `get_finished()` | ✅ 检查接收完成的请求 | ✅ 检查发送完成的请求 |
| 接收线程（RecvThread） | ✅ 启动，主动拉取 | ❌ 不启动 |
| 发送线程（SendThread） | ❌ 不启动 | ✅ 启动，监听请求 |

#### 源码示例：`get_num_new_matched_tokens`

```python
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    params = request.kv_transfer_params
    
    if params.get("do_remote_prefill"):
        # ⚠️ 只有 D 侧（Consumer）才会走这个分支
        assert not self.is_kv_producer
        token_ids = request.prompt_token_ids or []
        count = len(token_ids) - num_computed_tokens
        if count > 0:
            return count, True  # 返回可拉取的 token 数
    
    # P 侧（Producer）直接走到这里，返回 0
    return 0, False
```

#### 生活化类比

想象你有一个手机 App：
- 你用**买家账号**登录：看商品、下单、查物流（拉取 KV）
- 你用**卖家账号**登录：上架商品、发货、等确认（发送 KV）
- App 是**同一个**，但功能完全根据角色变化

KV Connector 就是这样：同一个类、同一套方法名，但 P 侧和 D 侧做的事情完全不同。

---

## 🟠 上游 vLLM 篇

---

## 5. 上游 MooncakeConnector 总览

### 5.1 文件位置

**主文件**：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`

### 5.2 核心类一览

```
MooncakeConnector (主类，继承 KVConnectorBase_V1 + SupportsHMA)
│
├── MooncakeConnectorScheduler
│   ├── get_num_new_matched_tokens()
│   ├── update_state_after_alloc()
│   ├── build_connector_meta()
│   └── request_finished()
│
├── MooncakeConnectorWorker
│   ├── register_kv_caches()
│   ├── start_load_kv()
│   ├── get_finished()
│   ├── TransferEngine (Mooncake 传输引擎)
│   ├── sender_listener (发送监听)
│   └── receiver_loop (接收循环)
│
└── MooncakeConnectorMetadata
    ├── reqs_to_recv (要接收的请求)
    └── reqs_to_send (要发送的请求)
```

### 5.3 主类定义

源码位置：`mooncake_connector.py:412`

```python
class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        
        # 根据角色初始化对应组件
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeConnectorScheduler(...)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(...)
```

**设计要点**：
- 同一个类，通过 `role` 参数决定是 Scheduler 还是 Worker
- Scheduler 模式下只有 `connector_scheduler`
- Worker 模式下只有 `connector_worker`

---

## 6. 上游 Scheduler 侧实现

### 6.1 Scheduler 核心职责

想象你是**餐厅经理**：
1. 🔍 **查单**：看看顾客点的菜厨房有没有做好（`get_num_new_matched_tokens`）
2. 📦 **备餐**：分配餐盒，准备好要送的菜（`update_state_after_alloc`）
3. 📝 **写订单**：把订单信息写好给骑手（`build_connector_meta`）
4. 🧹 **收尾**：顾客吃完了，收拾餐具（`request_finished`）

### 6.2 get_num_new_matched_tokens

源码位置：`mooncake_connector.py:620`

**作用**：查询远端有多少 KV 缓存可以用。

```python
def get_num_new_matched_tokens(
    self, request: "Request", num_computed_tokens: int
) -> tuple[int, bool]:
    params = request.kv_transfer_params
    
    if not params:
        return 0, False
    
    if params.get("do_remote_prefill"):
        # Remote Prefill 模式：从远端拉取所有 prompt 的 KV
        token_ids = request.prompt_token_ids or []
        count = len(token_ids) - num_computed_tokens
        if count > 0:
            return count, True  # (数量, 是否异步)
    
    return 0, False
```

**关键点**：
- 返回值是 `(num_tokens, is_async)`
- `do_remote_prefill` 标志表示要从远端拉取 KV
- 异步加载可以隐藏传输延迟

### 6.3 update_state_after_alloc

源码位置：`mooncake_connector.py:660`

**作用**：分配到 KV 块后，更新连接器状态，准备传输。

```python
def update_state_after_alloc(
    self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
):
    params = request.kv_transfer_params
    
    if params.get("do_remote_prefill"):
        # Decoder 侧：加入接收队列
        local_block_ids = self.get_sw_clipped_blocks(...)
        self._reqs_need_recv[request.request_id] = (request, local_block_ids)
        params["do_remote_prefill"] = False  # 只触发一次
    
    elif params.get("do_remote_decode"):
        # Prefill 侧：加入发送队列
        self._reqs_need_send[request.request_id] = (request, [])
```

### 6.4 build_connector_meta

源码位置：`mooncake_connector.py:709`

**作用**：构建传输元数据，传给 Worker 侧。

```python
def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
    meta = MooncakeConnectorMetadata()
    
    # Decoder 侧：把要接收的请求加入元数据
    if not self.is_kv_producer:
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )
        self._reqs_need_recv.clear()
    
    # Prefill 侧：把要发送的请求加入元数据
    if not self.is_kv_consumer:
        for req_id, (req, block_ids) in self._reqs_need_send.items():
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                load_remote_cache=False,
            )
        self._reqs_need_send.clear()
    
    return meta
```

### 6.5 request_finished

源码位置：`mooncake_connector.py:741`

**作用**：请求完成时的回调，决定是否延迟释放块。

```python
def request_finished(
    self, request: "Request", block_ids: tuple[list[int], ...]
) -> tuple[bool, dict[str, Any] | None]:
    params = request.kv_transfer_params
    
    if not params or not params.get("transfer_id"):
        return False, None  # 立即释放
    
    if params.get("do_remote_decode"):
        # Prefill 侧：需要把 KV 发给 Decoder，延迟释放
        self._reqs_need_send[request.request_id] = (
            request,
            self.get_sw_clipped_blocks(block_ids),
        )
        return True, None  # 延迟释放
    
    return False, None
```

---

## 7. 上游 Worker 侧实现

### 7.1 Worker 核心职责

想象你是**外卖骑手**：
1. 📍 **登记地址**：告诉平台餐厅和顾客的地址（`register_kv_caches`）
2. 🚴 **出发取餐**：开始去餐厅取餐（`start_load_kv`）
3. ✅ **送达确认**：哪些订单送完了（`get_finished`）

### 7.2 TransferEngine

Mooncake 的核心传输引擎，负责 RDMA 数据传输。

```python
from mooncake.engine import TransferEngine

self.engine = TransferEngine()
self.engine.initialize(hostname, "P2PHANDSHAKE", protocol, "")
```

**主要功能**：
- 管理 RDMA 连接
- 内存注册（用于 RDMA 传输）
- 批量数据传输
- 传输完成通知

### 7.3 register_kv_caches

**作用**：注册 KV 缓存的内存地址，让 Mooncake 能直接访问。

类比：告诉骑手**餐厅的具体地址**，这样他才能去取餐。

### 7.4 start_load_kv

**作用**：开始加载 KV 缓存（异步启动传输）。

类比：告诉骑手**可以出发了**，去把菜取回来。

### 7.5 异步传输架构

上游 Mooncake 使用**多线程 + 异步 IO** 的架构：

```
┌─────────────────────────────────────────────────────┐
│                  Worker 进程                        │
│                                                     │
│  ┌──────────────┐      ┌──────────────┐            │
│  │ 主线程       │      │ 发送监听线程 │            │
│  │ (模型执行)   │      │ (ZMQ ROUTER) │            │
│  └──────┬───────┘      └──────┬───────┘            │
│         │                     │                    │
│         ▼                     ▼                    │
│  ┌─────────────────────────────────────┐           │
│  │      TransferEngine (RDMA)          │           │
│  └─────────────────────────────────────┘           │
│                                                     │
│  ┌──────────────┐                                   │
│  │ 接收线程     │                                   │
│  │ (ZMQ 循环)   │                                   │
│  └──────────────┘                                   │
└─────────────────────────────────────────────────────┘
```

---

## 🔴 vLLM-Ascend 篇

---

## 8. Ascend 版 MooncakeConnector 总览

### 8.1 文件位置

**主文件**：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`

### 8.2 核心扩展点

相比上游 vLLM，Ascend 版主要做了以下扩展：

| 扩展点 | 说明 |
|--------|------|
| **GlobalTE 单例** | 全局共享 TransferEngine，避免重复初始化 |
| **NPU 适配** | 使用 Ascend CANN 后端，适配 NPU 内存模型 |
| **MLA 支持** | 支持 DeepSeek MLA 模型的特殊 KV 布局 |
| **PCP/DCP 支持** | 支持 Prefill/Dencode Context Parallel |
| **HMA 增强** | 更完善的混合内存分配器支持 |
| **多节点支持** | master-slave 元数据映射 |
| **传输超时** | 针对 NPU 的传输超时配置 |

### 8.3 主类定义

源码位置：`mooncake_connector.py:1405`

```python
class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self, 
        vllm_config: VllmConfig, 
        role: KVConnectorRole, 
        kv_cache_config: KVCacheConfig | None = None
    ):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeConnectorMetadata()
        
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeConnectorScheduler(...)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(...)
```

和上游结构一样，但内部实现有很大不同。

---

## 9. GlobalTE 单例优化

### 9.1 为什么需要单例？

上游 vLLM 中，每个 Worker 都创建自己的 `TransferEngine`，但在某些场景下（比如 PP 流水线），多个组件可能需要共享同一个传输引擎。

**问题**：
- 重复初始化浪费资源
- 内存注册可能重复
- 端口管理更复杂

### 9.2 GlobalTE 实现

**文件**：`vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py`

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
                if self.transfer_engine is None:  # Double-Checked Locking
                    from mooncake.engine import TransferEngine
                    self.transfer_engine = TransferEngine()
                    self.transfer_engine.initialize(
                        hostname, "P2PHANDSHAKE", "ascend", device_name
                    )
        return self.transfer_engine
    
    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                self.transfer_engine.register_memory(ptr, size)
            self.is_register_buffer = True

global_te = GlobalTE()
```

**设计要点**：
- **双重检查锁定**（Double-Checked Locking）：线程安全的懒汉单例
- **协议固定为 "ascend"**：使用昇腾 CANN 后端
- **内存注册一次性**：只注册一次，避免重复操作

### 9.3 类比

GlobalTE 就像是**公司的总机电话**：
- 上游：每个部门自己装一部电话（每个 Worker 一个 TransferEngine）
- Ascend 版：全公司共用一个总机，转接到各部门（全局共享一个 TransferEngine）

---

## 10. Ascend Scheduler 侧扩展

### 10.1 新增的成员变量

```python
class MooncakeConnectorScheduler:
    def __init__(self, ...):
        # ... 基础初始化 ...
        
        # Ascend 特有
        init_ascend_config(vllm_config)
        self.ascend_config = get_ascend_config()
        self.local_ip = get_ip()
        
        # 更多并行维度
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        
        # 多节点元数据映射
        self.multi_nodes_meta_mapping: dict[str, dict[str, Any]] = {}
        
        # HMA 相关
        self.use_hybrid = ...
        self.use_compress = self._model_uses_compress()
        self.group_transfer_info = [...]
        self.need_truncate = ...
```

### 10.2 get_num_new_matched_tokens 详解

这是 Scheduler 侧最核心的方法，也是 Ascend 版和上游差异最大的地方之一。

**两个分支**：
1. **Decoder 侧（Consumer）**：`do_remote_prefill` 拉取远端 KV
2. **Prefill 侧（Producer）**：`do_remote_decode` 准备推送 KV

详细分析可参考之前的文档，这里重点讲 **need_truncate 逻辑**：

```python
def _state_prefill_token_count(self, num_prompt_tokens: int) -> int:
    """D-side only. 对 Mamba 等有状态模型，最后一个 token 要截掉。"""
    if self.need_truncate and num_prompt_tokens > 1:
        return num_prompt_tokens - 1
    return num_prompt_tokens
```

**为什么要截断？**

对于 Mamba 等有状态模型：
- Prefill 计算了 h(1), h(2), ..., h(N) 共 N 个状态
- Decode 要从 h(N-1) 开始，自己重新算 h(N)
- 所以只传输前 N-1 个 token 的状态

### 10.3 截断的对称性

为了保证 P 侧和 D 侧的 token 数一致，两边都要处理截断：

| 侧 | 方法 | 作用 |
|----|------|------|
| **P 侧** | `_truncate_request_for_prefill` | 提前截掉最后一个 token，少算一个 |
| **D 侧** | `_state_prefill_token_count` | 少算一个 token 的 KV |

这样两边才能对齐。

---

## 11. Ascend Worker 侧扩展

### 11.1 Worker 初始化

源码位置：`mooncake_connector.py:1860`

```python
class MooncakeConnectorWorker:
    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        # 1. 获取 P/D 配置
        self._get_prefill_decode_size(vllm_config)
        
        # 2. 设置传输超时
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())
        
        # 3. 基本信息
        self.tp_rank = get_tensor_model_parallel_rank()
        self.pp_rank = get_pp_group().rank_in_group
        self.pcp_size = get_pcp_group().world_size
        # ... 更多并行维度 ...
        
        # 4. 使用 GlobalTE（关键！）
        device_name = str(torch.npu.current_device()) if self.pp_size > 1 else None
        self.engine = global_te.get_transfer_engine(
            self.side_channel_host,
            device_name=device_name,
        )
        
        # 5. 发送/接收线程
        self.kv_send_thread: KVCacheSendingThread | None = None
        self.kv_recv_thread: KVCacheRecvingThread | None = None
        
        # 6. 握手元数据
        self.xfer_handshake_metadata: MooncakeAgentMetadata | None = None
```

### 11.2 两个核心线程

Ascend 版把发送和接收分成了两个独立的线程类：

| 线程类 | 职责 |
|--------|------|
| `KVCacheSendingThread` | 监听远端请求，发送 KV 缓存 |
| `KVCacheRecvingThread` | 主动发起请求，接收 KV 缓存 |

**架构图**：

```
┌─────────────────────────────────────────────────────────┐
│                      Worker 进程                         │
│                                                         │
│  ┌─────────────────┐          ┌──────────────────────┐  │
│  │ KVCacheSending  │          │ KVCacheRecving       │  │
│  │ Thread          │          │ Thread               │  │
│  │                 │          │                      │  │
│  │ - ZMQ ROUTER    │◄────────►│ - ZMQ DEALER         │  │
│  │ - 监听请求      │  控制信令 │ - 发起请求           │  │
│  │ - 调用 TE 发送  │          │ - 调用 TE 接收       │  │
│  └────────┬────────┘          └──────────┬───────────┘  │
│           │                              │              │
│           └──────────────┬───────────────┘              │
│                          │                              │
│                  ┌───────▼───────┐                      │
│                  │  TransferEngine │                    │
│                  │  (GlobalTE)   │                      │
│                  └───────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

### 11.3 KVCacheSendingThread

源码位置：`mooncake_connector.py:243`

**作用**：Prefill 侧（Producer）运行，监听 Decode 侧的请求，把 KV 发送过去。

**核心循环**：
1. 监听 ZMQ 端口
2. 收到 `GET_META_MSG` → 返回自己的元数据
3. 收到 `DONE_RECVING_MSG` → 确认传输完成，更新任务追踪

### 11.4 KVCacheRecvingThread

源码位置：`mooncake_connector.py:407`

**作用**：Decode 侧（Consumer）运行，主动向 Prefill 侧请求 KV。

**核心功能**：
- 维护远端元数据缓存
- 管理传输任务队列
- 线程池并发处理多个传输
- 任务完成追踪

---

## 12. 特殊模型支持：MLA、Mamba、SWA

### 12.1 MLA（Multi-head Latent Attention）

DeepSeek MLA 模型有特殊的 KV 布局，需要特殊处理：

```python
if self.vllm_config.model_config.is_deepseek_mla:
    self.tp_num_need_pulls = 1  # MLA 只需要拉一次
else:
    # 普通注意力：根据 TP 比例计算需要拉几次
    num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
    num_p_block_heads = max(1, self.num_key_value_heads // self._prefill_tp_size)
    self.tp_num_need_pulls = num_d_block_heads // num_p_block_heads
```

### 12.2 Mamba（有状态模型）

Mamba 是 RNN 类模型，状态和注意力 KV 不同：
- 注意力 KV：可以按 block 传输
- Mamba 状态：每个 token 一个状态，需要对齐

`need_truncate` 标志就是为 Mamba 这类模型准备的。

### 12.3 SWA（Sliding Window Attention）

滑动窗口注意力只需要传输最近窗口内的 KV：

```python
def _get_swa_transfer_block_ids(self, block_ids: BlockIds) -> BlockIds:
    """把 SWA 组裁剪到窗口尾部，去掉占位 block 0。"""
    transfer_block_ids = []
    for blocks, group_info in zip(block_ids, self.group_transfer_info):
        if group_info.is_state_group or group_info.blocks_per_window == 0:
            transfer_block_ids.append(blocks)
        else:
            window_blocks = blocks[-group_info.blocks_per_window:]
            transfer_block_ids.append([
                block_id for block_id in window_blocks if block_id != 0
            ])
    return tuple(transfer_block_ids)
```

---

## 🟣 流程篇

---

## 13. KV 缓存注册流程

### 13.1 为什么要注册？

RDMA 传输需要知道**内存的物理地址**，所以要提前把 KV 缓存的内存地址注册给 Mooncake。

类比：告诉骑手**餐厅的具体地址**，他才能找到地方取餐。

### 13.2 注册流程

```
1. Worker 初始化 KV 缓存
   │
   ▼
2. 调用 connector.register_kv_caches(kv_caches)
   │
   ▼
3. MooncakeConnectorWorker 处理
   ├─ 计算每层 KV 的基地址
   ├─ 计算 block 大小、stride
   └─ 构建 MooncakeAgentMetadata
   │
   ▼
4. 注册内存到 TransferEngine（通过 GlobalTE）
   │
   ▼
5. 启动发送/接收线程
   │
   ▼
6. ✅ 注册完成，可以开始传输了
```

---

## 14. Remote Prefill 全流程（Decoder 侧拉取）

### 14.1 场景说明

Decoder 节点（Consumer）从 Prefill 节点（Producer）拉取 prompt 的 KV 缓存。

### 14.2 完整流程

```
Scheduler 进程                              Worker 进程
┌──────────────┐                           ┌──────────────┐
│ 1. 新请求到来 │                           │              │
└──────┬───────┘                           │              │
       │                                   │              │
       ▼                                   │              │
┌──────────────┐                           │              │
│ 2. get_num_  │                           │              │
│    new_      │  返回 (N, True)           │              │
│    matched_  │──────────────────────────►│              │
│    tokens    │                           │              │
└──────┬───────┘                           │              │
       │                                   │              │
       ▼                                   │              │
┌──────────────┐                           │              │
│ 3. 分配 KV   │                           │              │
│    blocks    │                           │              │
└──────┬───────┘                           │              │
       │                                   │              │
       ▼                                   │              │
┌──────────────┐                           │              │
│ 4. update_   │                           │              │
│    state_    │  加入接收队列             │              │
│    after_    │──────────────────────────►│              │
│    alloc     │                           │              │
└──────┬───────┘                           │              │
       │                                   │              │
       ▼                                   │              │
┌──────────────┐                           │              │
│ 5. build_    │                           │              │
│    connector_│  构建元数据               │              │
│    meta      │──────────────────────────►│              │
└──────┬───────┘                           │              │
       │                                   │              │
       └──────────────────────────────────►│              │
                                           │              │
                                           ▼              │
                                  ┌──────────────┐        │
                                  │ 6. start_    │        │
                                  │    load_kv   │        │
                                  └──────┬───────┘        │
                                         │                │
                                         ▼                │
                                  ┌──────────────┐        │
                                  │ 7. 向 P 侧发  │        │
                                  │    送请求     │        │
                                  └──────┬───────┘        │
                                         │                │
                                         ▼                │
                                  ┌──────────────┐        │
                                  │ 8. 接收 KV    │        │
                                  │    (RDMA)    │        │
                                  └──────┬───────┘        │
                                         │                │
                                         ▼                │
                                  ┌──────────────┐        │
                                  │ 9. 通知完成   │        │
                                  └──────────────┘        │
                                                          │
┌──────────────┐                           ▲              │
│ 10. get_     │  完成的请求               │              │
│     finished  │◄─────────────────────────┘              │
└──────────────┘                                          │
                                                          │
                                                          └──────────────┘
```

---

## 15. Remote Decode 全流程（Prefill 侧推送）

### 15.1 场景说明

Prefill 节点（Producer）在请求完成后，把 KV 缓存推送给 Decode 节点（Consumer）。

### 15.2 完整流程

```
Scheduler 进程                              Worker 进程
┌──────────────┐                           ┌──────────────┐
│ 1. 请求生成   │                           │              │
│    完成了     │                           │              │
└──────┬───────┘                           │              │
       │                                   │              │
       ▼                                   │              │
┌──────────────┐                           │              │
│ 2. request_  │  返回 (True, None)        │              │
│    finished  │──────────────────────────►│ 延迟释放块   │
└──────┬───────┘                           │              │
       │                                   │              │
       ▼                                   │              │
┌──────────────┐                           │              │
│ 3. build_    │                           │              │
│    connector_│  构建发送元数据           │              │
│    meta      │──────────────────────────►│              │
└──────┬───────┘                           │              │
       │                                   │              │
       └──────────────────────────────────►│              │
                                           │              │
                                           ▼              │
                                  ┌──────────────┐        │
                                  │ 4. 发送线程   │        │
                                  │    收到请求   │        │
                                  └──────┬───────┘        │
                                         │                │
                                         ▼                │
                                  ┌──────────────┐        │
                                  │ 5. 调用 TE    │        │
                                  │    发送 KV    │        │
                                  └──────┬───────┘        │
                                         │                │
                                         ▼                │
                                  ┌──────────────┐        │
                                  │ 6. 等待 D 侧  │        │
                                  │    确认       │        │
                                  └──────┬───────┘        │
                                         │                │
                                         ▼                │
                                  ┌──────────────┐        │
                                  │ 7. 更新完成   │        │
                                  │    状态       │        │
                                  └──────────────┘        │
                                                          │
┌──────────────┐                           ▲              │
│ 8. get_      │  完成的请求               │              │
│    finished  │◄─────────────────────────┘              │
└──────┬───────┘                                          │
       │                                                  │
       ▼                                                  │
┌──────────────┐                                          │
│ 9. 释放 KV   │                                          │
│    blocks    │                                          │
└──────────────┘                                          │
                                                          │
                                                          └──────────────┘
```

---

## ⚫ 数据结构篇

---

## 16. 核心数据结构详解

### 16.1 MooncakeAgentMetadata

**作用**：Worker 之间握手时交换的元数据，包含对端的 KV 缓存布局信息。

```python
class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str                    # 引擎 ID
    te_rpc_port: int                  # TransferEngine RPC 端口
    kv_group2layeridx: dict[int, ...] # KV 组到层索引的映射
    block_size: int                   # block 大小
    kv_caches_base_addr: list[list[int]]   # KV 缓存基地址
    block_size_scale: list[list[int]]      # block 大小缩放
    num_blocks: int                   # block 总数
    block_lens: list[list[int]]       # block 长度
    block_strides: list[list[int]]    # block stride
    local_ip: str = ""                # 本地 IP
    handshake_port: int = 0           # 握手端口
```

### 16.2 ReqMeta

**作用**：单个请求的传输元数据。

```python
@dataclass
class ReqMeta:
    local_block_ids: BlockIds        # 本地 block ID
    num_external_tokens: int          # 外部 token 数
    num_computed_tokens: int          # 已计算 token 数
    remote_block_ids: BlockIds       # 远端 block ID
    remote_host: str                  # 远端主机
    remote_port: int                  # 远端端口
    remote_engine_id: str             # 远端引擎 ID
    remote_request_id: str            # 远端请求 ID
    remote_pcp_size: int              # 远端 PCP 大小
    remote_dcp_size: int              # 远端 DCP 大小
    remote_ptp_size: int | None       # 远端 PTP 大小
    remote_multi_nodes_meta_mapping: dict  # 多节点元数据映射
    num_prompt_blocks: int            # prompt block 数
    remote_block_size: int            # 远端 block 大小
```

### 16.3 GroupPull

**作用**：组级别的拉取任务信息。

```python
@dataclass(frozen=True)
class GroupPull:
    group_id: int                     # KV 组 ID
    remote_tp_offset: int             # 远端 TP 偏移
    num_group_pulls: int              # 组拉取次数
    prefill_pp_rank: int = 0          # Prefill PP 排名
    is_group_transfer_end: bool = False # 是否组传输结束
```

### 16.4 KVCacheTaskTracker

**作用**：追踪 KV 缓存传输任务的状态。

```python
class KVCacheTaskTracker:
    def __init__(self):
        self.done_task_lock = threading.Lock()
        self.finished_requests: set[str] = set()        # 已完成的请求
        self.delayed_free_requests: OrderedDict[...]    # 延迟释放的请求
        self.reqs_to_process: set[str] = set()          # 待处理的请求
    
    def add_req_to_process(self, request_id: str)
    def add_not_transfer_request(self, request_id: str)
    def update_done_task_count(self, request_id: str)
    def get_and_clear_finished_requests(self) -> set[str]
    def add_delayed_request(self, request_id: str, delay_start_time: float)
```

---

## 总结

本文档从 vLLM 上游基类开始，逐层深入讲解了 P2P 模式 MooncakeConnector 的实现。关键要点：

1. **统一接口**：所有 Connector 都继承自 `KVConnectorBase_V1`
2. **双角色设计**：Scheduler 侧决策，Worker 侧执行
3. **上游基础**：vLLM 上游提供了基础的 P2P 传输能力
4. **Ascend 扩展**：vLLM-Ascend 做了大量优化，包括 GlobalTE、NPU 适配、特殊模型支持等
5. **两种流程**：Remote Prefill（D 侧拉取）和 Remote Decode（P 侧推送）

下一篇：[Mooncake Store Connector 详解](../kv_pool/mooncake_store_connector.md)
