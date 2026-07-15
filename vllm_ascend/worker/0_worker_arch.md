# NPUWorker 架构详解

> 对应文件：`vllm_ascend/worker/worker.py`
> 上游基类：`vllm.v1.worker.worker_base.WorkerBase`

## 目录

- [第一章：概述](#第一章概述)
- [第二章：初始化全流程](#第二章初始化全流程)
- [第三章：核心执行方法](#第三章核心执行方法)
- [第四章：内存管理与显存探查](#第四章内存管理与显存探查)
- [第五章：模型编译与预热](#第五章模型编译与预热)
- [第六章：分布式与并行](#第六章分布式与并行)
- [第七章：其他功能模块](#第七章其他功能模块)
- [附录：调用关系总览](#附录调用关系总览)

---

## 第一章：概述

### 1.1 NPUWorker 是什么？

`NPUWorker` 是 vLLM Ascend 版的工作节点，负责：
- 管理 NPU 设备生命周期
- 加载和运行模型
- 执行推理计算（prefill + decode）
- 管理 KV cache
- 处理分布式通信（TP/PP/PCP/DP）
- 性能 profiling 和监控

它继承自上游 `WorkerBase`，在 GPU 版本基础上做了 NPU 适配。

### 1.2 核心成员变量

| 成员变量 | 类型 | 作用 |
|---------|------|------|
| `model_runner` | `NPUModelRunner` | 模型执行器，实际跑模型前向的地方 |
| `device` | `torch.device` | 当前 worker 绑定的 NPU 设备 |
| `cache_dtype` | `torch.dtype` | KV cache 的数据类型 |
| `profiler` | `TorchNPUProfilerWrapper` | NPU 性能分析器 |
| `sleep_wakeup_manager` | `SleepWakeupManager` | 睡眠/唤醒模式管理器 |
| `weight_transfer_engine` | `WeightTransferEngine` | 权重在线更新引擎 |
| `init_snapshot` | `MemorySnapshot` | 初始内存快照（用于显存计算） |
| `peak_activation_memory` | `int` | 峰值激活显存占用 |
| `non_torch_memory` | `int` | 非 PyTorch 分配的显存 |
| `npugraph_memory_bytes` | `int` | NPU 图模式占用的显存 |

### 1.3 Worker 生命周期

```
__init__()                ← 构造函数，基础配置
  │
init_device()             ← 设备初始化 + 分布式环境
  │
load_model()              ← 加载模型权重
  │
determine_available_memory()  ← 探查可用显存给 KV cache
  │
compile_or_warm_up_model()    ← 编译 + 预热模型
  │
initialize_from_config()      ← 初始化 KV cache
  │
execute_model() / sample_tokens()  ← 正常推理循环（多轮）
  │
shutdown()                ← 关闭清理
```

---

## 第二章：初始化全流程

### 2.1 `__init__()` —— 构造函数

位置：`worker.py:91`

**做的事情：**

```
__init__(vllm_config, local_rank, rank, ...)
  │
  ├─ 检查 COMPILE_CUSTOM_KERNELS 开关
  │
  ├─ adapt_patch()  ◄────── 注册 vLLM 上游的 monkey patch
  │
  ├─ 注册自定义算子
  │     ├─ ops.register_dummy_fusion_op()
  │     ├─ _register_atb_extensions()  (非 A5 芯片)
  │     └─ register_ascend_customop(vllm_config)
  │
  ├─ init_ascend_config(vllm_config)  ◄─ 初始化 Ascend 配置 + SoC 版本
  │
  ├─ configure_ascend_file_logging()  ◄─ 配置日志
  │
  ├─ check_ascend_device_type()       ◄─ 检查设备类型
  │
  ├─ super().__init__()  ◄──────────── 调用父类 WorkerBase 初始化
  │
  ├─ 确定 cache_dtype（KV cache 数据类型）
  │
  ├─ 初始化 profiler 相关变量（延迟初始化）
  │
  ├─ SleepWakeupManager 初始化（睡眠模式）
  │
  ├─ weight_transfer_engine = None（后面 load_model 时创建）
  │
  ├─ 选择 v1 / v2 model runner
  │     └─ v2 目前还在开发中，默认 v1
  │
  └─ 静态内核卸载信号处理器（enable_static_kernel 时）
        注册 SIGTERM / SIGINT → uninstall_static_kernel()
```

**关键点：**
- `__init__` 阶段**不绑定设备**，也不创建 model_runner
- 主要做：注册 patch、注册算子、读取配置、设置成员变量
- 设备绑定在 `init_device()` 里做

### 2.2 `init_device()` —— 设备初始化

位置：`worker.py:508`

这一步才真正绑定 NPU 设备，创建 model_runner。

```
init_device()
  │
  ├─ self.device = self._init_device()  ◄── 核心：设备初始化
  │     │
  │     ├─ 计算 local_rank（DP 偏移）
  │     ├─ torch.npu.set_device(device)
  │     ├─ gc.collect() + torch.npu.empty_cache()
  │     ├─ A5 芯片：setup_ascend_local_comm_res()
  │     ├─ 拍初始内存快照 init_snapshot
  │     ├─ 检查可用显存是否满足 gpu_memory_utilization
  │     ├─ _init_worker_distributed_environment()
  │     ├─ set_random_seed(seed)
  │     └─ init_device_properties_triton()
  │
  ├─ init_workspace_manager(device, num_ubatches=1)
  │
  └─ 创建 model_runner
        ├─ v2: NPUModelRunnerV2（开发中）
        └─ v1: NPUModelRunner（默认，稳定版）
```

#### 2.2.1 `_init_worker_distributed_environment()`

位置：`worker.py:998`

分布式环境初始化：

```
_init_worker_distributed_environment()
  ├─ init_batch_invariance()  ◄──── 批次不变量初始化
  ├─ init_distributed_environment()  ◄─ 通用分布式环境（HCCL）
  ├─ ensure_model_parallel_initialized()  ◄─ TP / PP / PCP / DCP 组
  ├─ init_ascend_model_parallel()  ◄─── Ascend 特定的并行初始化
  └─ ensure_ec_transfer_initialized()  ◄─ EC（编码器-解码器）传输
```

### 2.3 `load_model()` —— 加载模型

位置：`worker.py:683`

```
load_model()
  │
  ├─ 睡眠模式：用 CaMemAllocator 的 weights 内存池
  │     （否则用默认上下文）
  │
  ├─ self.model_runner.load_model()  ◄── 实际加载权重
  │
  └─ 权重在线更新引擎（如果配置了）
        └─ WeightTransferEngineFactory.create_engine(...)
            需要 model 引用，所以加载完才创建
```

---

## 第三章：核心执行方法

### 3.1 `execute_model()` —— 执行模型前向

位置：`worker.py:613`

Worker 层的 `execute_model` 主要做**调度和通信**，实际计算委托给 `model_runner`。

```
execute_model(scheduler_output)
  │
  ├─ profile_memory()  ◄──── 记录当前显存使用
  │
  ├─ msMonitor 步进（如果开启）
  │
  ├─ 等待上一轮 PP 发送完成（_pp_send_work）
  │
  ├─ PP 非首卡：接收上一阶段的中间张量
  │     └─ irecv_tensor_dict → AsyncIntermediateTensors
  │
  ├─ profiler.step()（如果开启了 profiling）
  │
  ├─ output = self.model_runner.execute_model(...)  ◄── 实际模型前向
  │
  ├─ 如果输出是 ModelRunnerOutput / AsyncModelRunnerOutput / None
  │     直接返回（末卡 / 无 PP 场景）
  │
  └─ 如果输出是 IntermediateTensors（PP 非末卡）
        ├─ isend_tensor_dict 发送给下一阶段
        │   （保存 handle 到 _pp_send_work，下一轮等）
        └─ KV connector 输出透传
```

**两种返回情况：**

| 场景 | 返回类型 | 说明 |
|------|---------|------|
| 无 PP / PP 末卡 | `ModelRunnerOutput` / `AsyncModelRunnerOutput` / `None` | 正常推理输出 |
| PP 非末卡 | `None` / `EMPTY_MODEL_RUNNER_OUTPUT` | 中间张量发给下一张卡，不返回最终结果 |

### 3.2 `sample_tokens()` —— 采样

位置：`worker.py:680`

```python
@torch.inference_mode()
def sample_tokens(self, grammar_output):
    return self.model_runner.sample_tokens(grammar_output)
```

非常薄的一层，直接转发给 `model_runner`。

---

## 第四章：内存管理与显存探查

### 4.1 `determine_available_memory()` —— 可用显存计算

位置：`worker.py:530`

这是初始化阶段的关键一步：跑一遍 dummy forward，看看模型占多少显存，剩下的都给 KV cache。

```
determine_available_memory() -> int (KV cache 可用字节数)
  │
  ├─ 如果用户指定了 kv_cache_memory_bytes
  │     ├─ 跑 profile_run()（只编译，不算显存）
  │     └─ 直接返回用户指定的值
  │
  └─ 否则自动探查：
        │
        ├─ memory_profiling 上下文管理器
        │     （记录前后显存差值）
        │
        ├─ self.model_runner.profile_run()  ◄── 跑 dummy forward
        │
        ├─ 记录 torch 峰值（图捕获之前的值）
        │
        ├─ 计算 non_kv_cache_memory =
        │     non_torch_increase + torch_peak_increase + weights_memory
        │
        ├─ 保存 peak_activation_memory / non_torch_memory
        │
        └─ 返回 available_kv_cache_memory_bytes
```

**为什么要在图捕获之前记录 torch 峰值？**
因为图模式会分配一大块图池内存，那不算激活内存。如果图捕获之后再记，会把图池内存也算进去，导致高估激活内存，KV cache 分配偏少。

### 4.2 `profile_memory()` —— 运行时显存记录

位置：`worker.py:602`

每次 `execute_model` 开头调用，记录当前的 torch 保留/分配显存，用于 debug。

```
profile_memory()
  ├─ self.torch_reserved = torch.npu.memory_reserved()
  ├─ self.torch_allocated = torch.npu.memory_allocated()
  └─ DEBUG 级别日志输出
```

### 4.3 睡眠模式与 CaMemAllocator

**CaMemAllocator**：Ascend 的内存分配器，支持按 tag 分池管理。

睡眠模式相关：

```
enable_sleep_mode
  │
  ├─ load_model()：权重放在 "weights" 内存池
  ├─ initialize_from_config()：KV cache 放在 "kv_cache" 内存池
  └─ SleepWakeupManager：管理睡眠/唤醒时的内存回收与重分配
```

---

## 第五章：模型编译与预热

### 5.1 Warmup 的两层架构

Warmup 不是在某一个地方做的，而是**Worker 层调度 + ModelRunner 层执行**的两层架构：

```
┌─────────────────────────────────────────────────┐
│  NPUWorker（调度/编排层）                         │
│  ─────────────────────────────────────────────  │
│  • 决定 warmup 哪些尺寸                          │
│  • 决定 warmup 顺序（从大到小，避免反复分配）      │
│  • 记录显存占用（profile memory）                 │
│  • 捕获计算图前后的显存差值                        │
│  • 输出建议的 kv-cache-memory 值                  │
│  • ATB 算子预热、CPU 绑核等周边工作                │
└───────────────────┬─────────────────────────────┘
                    │ 调用
                    ▼
┌─────────────────────────────────────────────────┐
│  NPUModelRunner（实际执行层）                      │
│  ─────────────────────────────────────────────  │
│  • 构造假数据（input_ids、positions 等）          │
│  • 跑模型前向（_model_forward）                   │
│  • EPLB 预热（eplb_warmup）                       │
│  • MC2 预热（条件触发）                            │
│  • 捕获 ACL / NPU 计算图（capture_model）          │
│  • 初始化 KV cache 相关结构                        │
└─────────────────────────────────────────────────┘
```

### 5.2 第一次 Warmup：`determine_available_memory()` 中的 profile_run

位置：`worker.py:544 / 564`

在显存探查阶段，会先跑一次 warmup，目的有两个：
1. 触发算子编译，避免第一次真实推理卡顿
2. 测量模型实际占用的显存，计算 KV cache 可用大小

```
determine_available_memory()
  │
  ├─ 如果用户指定了 kv_cache_memory_bytes
  │     └─ self.model_runner.profile_run()  ◄── 只编译，不算显存
  │
  └─ 否则（自动探查模式）
        ├─ memory_profiling 上下文（记录前后显存差）
        ├─ self.model_runner.profile_run()  ◄── 跑 dummy forward
        └─ 计算 KV cache 可用显存
```

> **关于 `profile_run()`**：这是 ModelRunner 层的 warmup 入口，内部会依次执行 EPLB 预热 → MC2 预热 → 主模型 dummy_run。详细实现见 `0_model_runner_v1_arch.md` 第 4.0 节。

### 5.3 第二次 Warmup：`compile_or_warm_up_model()`

位置：`worker.py:715`

这是更全面的编译预热，针对多个 batch size 做 warmup，并捕获计算图。

```
compile_or_warm_up_model() -> CompilationTimes
  │
  ├─ Step 1: 整理 warmup_sizes
  │     ├─ 从 compile_sizes 获取
  │     ├─ 从 cudagraph_capture_sizes 获取（图捕获的 size 单独处理）
  │     └─ 从 compile_ranges 补充遗漏的 size
  │
  ├─ Step 2: 逐个 warmup size 跑 dummy_run
  │     （从大到小跑，避免反复分配）
  │     └─ self.model_runner._dummy_run(size)  ◄── 委托给 ModelRunner
  │
  ├─ Step 3: 捕获 NPU 计算图
  │     └─ npugraph_memory_bytes = self.model_runner.capture_model()  ◄── 委托给 ModelRunner
  │
  ├─ Step 4: 输出建议的 kv-cache-memory 值
  │     （模型权重 + 峰值激活 + non-torch + 图内存 + 150 MiB 缓冲）
  │
  ├─ Step 5: ATB 预热（非 A5 芯片）
  │     └─ _warm_up_atb()  ◄── Worker 自己做的
  │
  ├─ Step 6: CPU 绑核（如果开启）
  │     └─ bind_cpus(local_rank)
  │
  ├─ Step 7: 重置随机种子
  │
  └─ 返回 CompilationTimes
```

**两次 Warmup 的区别：**

| 维度 | 第一次（profile_run） | 第二次（compile_or_warm_up_model） |
|------|----------------------|-----------------------------------|
| 触发时机 | `determine_available_memory` 中 | KV cache 大小确定之后 |
| 调用入口 | `model_runner.profile_run()` | 多个 `model_runner._dummy_run(size)` + `capture_model()` |
| 覆盖尺寸 | 通常 1 个默认尺寸 | 多个尺寸（compile_sizes + cudagraph_sizes） |
| 是否捕获计算图 | 否 | 是 |
| 主要目的 | 算子编译 + 显存测量 | 多尺寸编译 + 图捕获 + 全面预热 |

### 5.4 `_warm_up_atb()` —— ATB 算子预热

位置：`worker.py:807`

Worker 层自己做的预热，非常简短但很重要：跑一次 `_npu_matmul_add_fp32`，否则第一次 ReshapeAndCache 算子会有性能下降。

```python
def _warm_up_atb(self):
    x = torch.rand((2, 4), dtype=torch.float16).npu()
    weight = torch.rand((2, 4), dtype=torch.float16).npu()
    c = torch.rand((4, 4), dtype=torch.float32).npu()
    torch_npu._npu_matmul_add_fp32(x, weight, c)
```

### 5.5 `profile_prefill_latency()` —— Prefill 延迟探查

位置：`worker.py:817`

用于动态 chunk size 调整：给定 token 数，实测 prefill 延迟。这也是 Worker 调度 + ModelRunner 执行的模式。

```
profile_prefill_latency(num_tokens) -> latency_ms
  │
  ├─ clamp 到合法范围 [1, max_num_batched_tokens]
  ├─ torch.npu.synchronize()  同步
  ├─ 记录开始时间
  ├─ self.model_runner._dummy_run(
  │       num_tokens,
  │       force_attention=True,   ← 确保注意力算子执行
  │       profile_cpp=True        ← C++ 端的 profiling
  │   )
  ├─ torch.npu.synchronize()  同步（确保 NPU 跑完）
  └─ 计算并返回延迟（毫秒）
```

**为什么 force_attention=True？** 因为默认 dummy_run 可能不构建注意力元数据，导致注意力算子不执行，测出来的延迟不准。

### 5.6 其他会触发 dummy_run 的地方

| 方法 | 触发场景 | 调用 |
|------|---------|------|
| `execute_dummy_batch()` | 健康检查 / 保活 | `model_runner._dummy_run(decode_token_per_req, uniform_decode=True)` |
| `profile_prefill_latency()` | 动态 chunk size 调整 | `model_runner._dummy_run(num_tokens, force_attention=True)` |

---

## 第六章：分布式与并行

### 6.1 并行维度总览

NPUWorker 支持多种并行方式：

| 并行方式 | 缩写 | 作用 | 初始化位置 |
|---------|------|------|-----------|
| 张量并行 | TP | 权重按行/列切分 | `ensure_model_parallel_initialized` |
| 流水线并行 | PP | 按层切分到多卡 | 同上 |
| Prefill 上下文并行 | PCP | prefill 时按 token 切分注意力 | 同上 |
| Decode 上下文并行 | DCP | decode 时按 token 切分 | 同上 |
| 数据并行 | DP | 不同请求分到不同组 | 父类处理 |
| EC 传输 | EC | 编码器-解码器跨卡传输 | `ensure_ec_transfer_initialized` |
| KV 传输 | KV Transfer | KV cache 跨卡迁移 | `ensure_kv_transfer_initialized` |

### 6.2 PP 流水线通信

在 `execute_model()` 里处理 PP 通信：

```
PP 非首卡：
  execute_model() 开头
    └─ irecv_tensor_dict()  ← 接收上一阶段的中间张量
         （异步，存到 AsyncIntermediateTensors）

PP 非末卡：
  execute_model() 结尾
    └─ isend_tensor_dict()  ← 发给下一阶段
         （handle 存到 _pp_send_work，下一轮开头等）
```

**为什么下一轮才等？** 因为异步调度——这一轮发出去就可以返回，CPU 继续做下一批的准备，等下一轮 execute_model 开头再确认上一轮发完没。

### 6.3 KV Transfer

KV cache 迁移相关：

| 方法 | 作用 |
|------|------|
| `get_kv_connector_handshake_metadata()` | 获取 KV 连接器握手元数据（建连用） |
| `get_kv_cache_spec()` | 获取 KV cache 规格（每层 shape、dtype 等） |
| `initialize_from_config()` | 初始化 KV cache 时同时初始化 KV transfer |

---

## 第七章：其他功能模块

### 7.1 权重在线更新

支持运行时更新模型权重（不重启服务）。

| 方法 | 作用 |
|------|------|
| `init_weight_transfer_engine()` | 初始化权重传输引擎 |
| `start_weight_update()` | 开始权重更新 |
| `update_weights()` | 更新指定层权重 |
| `finish_weight_update()` | 结束权重更新 |

### 7.2 睡眠/唤醒模式

NPU 低功耗模式：

| 方法 | 作用 |
|------|------|
| `sleep(level)` | 让 NPU 进入睡眠模式，释放部分显存 |
| `wake_up(tags)` | 唤醒 NPU，恢复显存 |

管理器：`SleepWakeupManager`

### 7.3 Profiling

性能分析功能，基于 Torch NPU Profiler：

| 方法 | 作用 |
|------|------|
| `profile(is_start, profile_prefix)` | 开始/停止 profiling |
| `profiler.step()` | 每个 step 标记一下（execute_model 里调用） |

### 7.4 LoRA 管理

直接转发给 model_runner：

| 方法 | 作用 |
|------|------|
| `add_lora(lora_request)` | 添加 LoRA 权重 |
| `remove_lora(lora_id)` | 移除 LoRA |
| `list_loras()` | 列出当前加载的 LoRA |
| `pin_lora(lora_id)` | 钉住 LoRA（不被换出） |

### 7.5 健康检查

`check_health()` —— 调用 `npu-smi info -t health` 检查 NPU 卡健康状态，不是 OK 就抛异常。

### 7.6 静态内核卸载

`uninstall_static_kernel()` —— 进程退出时卸载静态内核，释放资源。注册了 SIGTERM / SIGINT 信号自动触发。

---

## 附录：调用关系总览

### A.1 Worker 初始化完整调用链

```
Engine 启动
  │
  └─ NPUWorker.__init__()
        ├─ adapt_patch()
        ├─ 注册自定义算子
        ├─ init_ascend_config()
        └─ super().__init__()
  │
  └─ worker.init_device()
        ├─ _init_device()
        │     ├─ torch.npu.set_device()
        │     ├─ 内存初始快照
        │     └─ _init_worker_distributed_environment()
        ├─ init_workspace_manager()
        └─ 创建 NPUModelRunner
  │
  └─ worker.load_model()
        └─ model_runner.load_model()
  │
  └─ worker.determine_available_memory()
        └─ model_runner.profile_run()  ← warmup
  │
  └─ worker.compile_or_warm_up_model()
        ├─ 多尺寸 dummy_run
        ├─ capture_model()  ← 捕获计算图
        └─ _warm_up_atb()
  │
  └─ worker.initialize_from_config(kv_cache_config)
        └─ model_runner.initialize_kv_cache()
```

### A.2 单轮推理调用链

```
Engine 调度
  │
  └─ worker.execute_model(scheduler_output)
        ├─ profile_memory()
        ├─ PP 非首卡：接收中间张量
        ├─ model_runner.execute_model()  ← 实际跑模型
        └─ PP 非末卡：发送中间张量
  │
  └─ worker.sample_tokens(grammar_output)
        └─ model_runner.sample_tokens()  ← 采样 + 草稿提议
              ├─ 拒绝采样（验证上轮草稿）
              └─ propose_draft_token_ids（提议下轮草稿）
```

### A.3 关键文件索引

| 功能 | 文件路径 |
|------|----------|
| NPUWorker 主文件 | `vllm_ascend/worker/worker.py` |
| 模型执行器 v1 | `vllm_ascend/worker/model_runner_v1.py` |
| 模型执行器 v2 | `vllm_ascend/worker/v2/model_runner.py` |
| NPU 输入批次 | `vllm_ascend/worker/npu_input_batch.py` |
| PCP 管理器 | `vllm_ascend/worker/pcp_utils.py` |
| 睡眠模式管理器 | `vllm_ascend/device_allocator/sleep_mem_optimized.py` |
| CaMem 分配器 | `vllm_ascend/device_allocator/camem.py` |
