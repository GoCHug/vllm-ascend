# Kimi-K2.5 PD 分离 Decode OOM 分析

## 问题背景

**Issue**: https://github.com/vllm-project/vllm-ascend/issues/10784

**现象**: v0.18.0 运行 Kimi-K2.5,PD 分离 2P1D,执行过程中 Decode 节点 OOM。

**测试配置**:
- 图片: 448×448×1
- 并发: 576
- NUM_PROMPTS: 2304
- RANDOM_INPUT_LEN: 4096
- RANDOM_OUTPUT_LEN: 32768

**OOM 现场观察**:
- `_moe_forward_shared` 在最后一次 `segment_map` 操作前扩展到 61GB
- `npu_grouped_matmul_gmm2` (`torch_npu.npu_grouped_matmul`) 申请两段内存: 3.4GB + 112MB
- 112MB 为 workspace,3.4GB 为输出
- OOM 时 `npu_grouped_matmul_gmm2` 输入 shape 从 1024 突增到 **508648**
- 确认有请求触发了 recompute (重新计算)

---

## 508648 的计算推导

### 1. 视觉 token 数计算 (448×448×1 图片)

参考 `vllm/vllm/model_executor/models/kimi_k25_vit.py` 的架构:

```
图片: 448 × 448 × 1 (H × W × T)
  ↓ patch_size = 14
Patch 数: (448/14) × (448/14) = 32 × 32 = 1024 个 patch
  ↓ grid_thw = (1, 32, 32)
  ↓ tpool_patch_merger (merge_kernel_size = (2,2))
  ↓ nh=16, nw=16, 输出 (256, 4, d)
  ↓ mm_projector 展平
视觉 token 数 = 256 个 <|media_pad|> token
```

**每张 448×448 图片插入 256 个视觉占位符 token**。

### 2. MoE hidden_states 的 token 扩展

关键代码在 `vllm_ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py:395`:

```python
sorted_hidden_states, ... = DeviceOperator.npu_moe_init_routing(
    hidden_states,
    topk_ids,
    active_num=num_tokens * self.top_k,  # ← 关键!token 数 × top_k
    ...
)
```

**MoE 的 hidden_states 第一维 = `num_tokens × top_k`**

Kimi-K2.5 基于 DeepseekV3,`top_k = 8` (num_experts_per_tok)。

### 3. 508648 的分解

```
508648 = top_k × num_tokens
       = 8 × 63581
       = 8 × 7 × 9083
       = 8 × 7 × (4096 + 256 + 4731)
              ↑     ↑     ↑
           input  vision  output
```

| 项目 | 值 | 说明 |
|------|-----|------|
| top_k | 8 | DeepseekV3 每个 token 路由到 8 个专家 |
| 重组序列数 | 7 | 同时被 recompute 的序列数 |
| 每序列 input | 4096 | RANDOM_INPUT_LEN=4096 |
| 每序列 vision | 256 | 448×448 图片产生的视觉 token |
| 每序列 output | ~4731 | 已生成的输出 token |
| **总 token 数** | **63581** | 7 × 9083 |
| **MoE hidden_states** | **508648** | 63581 × 8 |

### 4. 为什么 Decode 节点会 OOM

**正常 decode 阶段**:
```
18 sequences × 1 token × 8 (top_k) = 144 token-expert pairs
(约 1024,经 padding)
```

**Recompute 发生时**:
```
7 个被抢占的序列 × (4096 + 256 + 4731) × 8 (top_k) = 508648 token-expert pairs
```

**Token 数从 ~144 暴涨到 508648,增长约 3500 倍!**

---

## OOM 触发链路

```
高并发 (576) → Decode 节点内存压力 → vLLM 抢占请求 (preemption)
  → 被抢占请求重新调度 → recompute (re-prefill 整个序列)
  → 多个长序列同时 recompute → num_tokens 暴涨到 63581
  → MoE 扩展: 63581 × 8 = 508648
  → npu_grouped_matmul_gmm2 分配 3.4GB 显存
  → _moe_forward_shared 已占 61GB
  → 显存不足 → OOM
```

---

## 根本原因

| 因素 | 影响 |
|------|------|
| PD 分离 2P1D | Decode 节点内存有限 |
| 高并发 576 | 内存压力大,触发抢占 |
| 多模态 vision token | 每图 256 token 增加序列长度 |
| Re-prefill 整个序列 | 已生成的 output token 也需重新计算 |
| MoE top_k=8 扩展 | token 数 × 8,放大内存需求 |
| 多序列同时 recompute | token 数叠加,导致雪崩 |

---

## 关键代码位置

| 文件 | 行号 | 说明 |
|------|------|------|
| `vllm_ascend/ops/fused_moe/moe_mlp.py` | L207-224 | `npu_grouped_matmul_gmm2` 调用,hidden_states 突增到 508648 |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py` | L395 | `npu_moe_init_routing` 中 `active_num=num_tokens * self.top_k` |
| `vllm/vllm/model_executor/models/kimi_k25_vit.py` | L524 | `tpool_patch_merger` 时间池化 + 空间下采样 |
| `vllm/vllm/model_executor/models/kimi_k25_vit.py` | L657 | `KimiK25MultiModalProjector` 投影到文本空间 |

---

## 验证建议

在 recompute 分支添加日志验证:

```python
# 在 vllm/v1/core/sched.py 的 preemption 逻辑附近
logger.info(f"Preempted request {req_id}, seq_len={len(seq_data.output_token_ids)}")

# 在 moe_mlp.py 的 npu_grouped_matmul_gmm2 调用前
logger.info(f"gmm2 hidden_states shape: {hidden_states.shape}, num_tokens={hidden_states.shape[0]//top_k}")
```

预期会看到 7 个左右的 preempted request,每个 seq_len ≈ 9083,然后 gmm2 的 hidden_states[0] = 508648。

---

## 缓解方案建议

1. **降低并发**: 减少 MAX_CONCURRENCY,避免触发抢占
2. **增加 Decode 节点内存**: 调整 gpu-memory-utilization 或增加 Decode 节点数
3. **限制 recompute 数量**: 控制 vLLM 同时 recompute 的请求数
4. **优化 recompute 策略**: 对长序列优先 swap out 而非 recompute
5. **减小 vision token 占用**: 调整图片分辨率或 patch 合并策略

---

## Preemption 触发条件分析

### 1. Preemption 触发的根本条件

**位置**: `vllm/vllm/v1/core/sched/scheduler.py:446-489`

Preemption 在调度 RUNNING 请求时触发,核心条件是 **KV cache block 分配失败**:

```python
# scheduler.py:446
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(
        request,
        num_new_tokens,
        num_lookahead_tokens=self.num_lookahead_tokens,
    )

    if new_blocks is not None:
        # The request can be scheduled.
        break

    # The request cannot be scheduled.
    # Preempt the lowest-priority request.
    ...
```

**触发链路**:
```
高并发请求 → KV cache 不足 → allocate_slots() 返回 None
  → 触发 preemption
```

### 2. Preemption 的两种模式

**位置**: `vllm/vllm/v1/core/sched/scheduler.py:453-489`

| 模式 | 触发条件 | 行为 |
|------|----------|------|
| **PRIORITY 模式** | `self.policy == SchedulingPolicy.PRIORITY` | 抢占优先级最低的请求 (`max(running, key=lambda r: (r.priority, r.arrival_time))`) |
| **FCFS 模式** (默认) | 非 PRIORITY 模式 | 抢占队列末尾的请求 (`self.running.pop()`) |

### 3. Preemption 的具体操作

**位置**: `vllm/vllm/v1/core/sched/scheduler.py:935-953`

```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
    assert request.status == RequestStatus.RUNNING
    self.kv_cache_manager.free(request)           # 释放 KV cache
    self.encoder_cache_manager.free(request)       # 释放 encoder cache
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0                # 重置计算进度!
    if request.spec_token_ids:
        request.spec_token_ids = []
    request.num_preemptions += 1
    # 放回 waiting 队列头部
    self.waiting.prepend_request(request)
```

**关键点**: `num_computed_tokens = 0` 意味着整个序列需要重新计算 (re-prefill)。

### 4. Preemption 触发条件汇总

| 条件 | 说明 |
|------|------|
| **KV cache 不足** | `allocate_slots()` 返回 None,无法为新 token 分配 block |
| **token budget 耗尽** | `token_budget <= 0`,无法调度更多 token |
| **running 队列非空** | 必须有可抢占的 RUNNING 请求 |
| **非 PAUSED_ALL 状态** | 调度器未暂停 |

---

## Recompute 调度逻辑分析 (vllm-ascend 特有)

### 1. RecomputeScheduler 概述

**位置**: `vllm_ascend/core/recompute_scheduler.py`

vllm-ascend 提供了 `RecomputeScheduler`,**专为 PD 分离场景设计**。当 Decode 节点 KV cache 不足时,不再在本地 re-prefill,而是将请求**回送给 Prefill 节点**重新计算 KV cache。

### 2. 启用条件

**位置**: `vllm_ascend/platform.py:640-651`

```python
if ascend_config.recompute_scheduler_enable:
    kv_transfer_config = vllm_config.kv_transfer_config
    kv_role = getattr(kv_transfer_config, "kv_role", None)
    if kv_transfer_config is None or kv_role == "kv_both":
        raise ValueError(
            "recompute_scheduler_enable can only be enabled in PD-disaggregated mode "
            "(kv_role='kv_producer' or 'kv_consumer'), and is not supported in PD-mixed mode."
        )
    # 替换 scheduler_config
    recompute_scheduler_config = RecomputeSchedulerConfig.initialize_from_config(vllm_config)
    vllm_config.scheduler_config = recompute_scheduler_config
```

**启用要求**:
- `recompute_scheduler_enable: true` (additional-config)
- **必须为 PD 分离模式** (`kv_role = kv_producer` 或 `kv_consumer`)
- **不能为 PD 混合模式** (`kv_role = kv_both`)

### 3. RecomputeScheduler 的核心差异

**位置**: `vllm_ascend/core/recompute_scheduler.py:335-349`

与原生 vLLM Scheduler 的关键区别在 preemption 逻辑:

```python
# 原生 vLLM: 抢占后放回 waiting 队列,本地 re-prefill
# RecomputeScheduler: 根据 kv_role 分流

transfer_config = self.vllm_config.kv_transfer_config
if transfer_config is not None and not transfer_config.is_kv_producer:
    # ===== Decode 节点 (kv_consumer) =====
    # 不本地 re-prefill,而是回送给 PD proxy / Prefill 节点
    recomputed_req = self.running.pop()
    self.kv_cache_manager.free(recomputed_req)
    recomputed_reqs.append(
        RecomputeReqInfo(
            recomputed_req.request_id,
            recomputed_req.output_token_ids,  # 保留已生成的 output tokens
            recomputed_req.client_index
        )
    )
else:
    # ===== Prefill 节点 (kv_producer) =====
    # 走原生 preemption 逻辑
    preempted_req = self.running.pop()
    self._preempt_request(preempted_req, scheduled_timestamp)
    preempted_reqs.append(preempted_req)
```

### 4. Recompute 的输出处理

**位置**: `vllm_ascend/core/recompute_scheduler.py:832-844`

被 recompute 的请求会以 `EngineCoreOutput` 形式返回给客户端,带 `stop_reason="recomputed"`:

```python
if scheduler_output.recomputed_reqs is not None:
    for req_info in scheduler_output.recomputed_reqs:
        logger.warning("Recompute triggered for request %s.", req_info.request_id)
        outputs[req_info.client_index].append(
            EngineCoreOutput(
                request_id=req_info.request_id,
                finish_reason=FinishReason.STOP,
                new_token_ids=[],
                stop_reason="recomputed",   # ← 客户端可识别
            )
        )
```

### 5. PD 分离场景下的完整 Recompute 流程

```
[Decode 节点 - kv_consumer]
  ↓ KV cache 不足,allocate_slots() 返回 None
  ↓ RecomputeScheduler.schedule() 触发 recompute 分支
  ↓ 从 running 弹出请求,释放 KV cache
  ↓ 加入 recomputed_reqs 列表 (保留 output_token_ids)
  ↓ 返回 EngineCoreOutput (stop_reason="recomputed") 给客户端
  ↓
[PD Proxy / 客户端]
  ↓ 收到 recompute 信号
  ↓ 重新发送请求到 Prefill 节点
  ↓
[Prefill 节点 - kv_producer]
  ↓ 重新 prefill 整个序列 (input + output tokens)
  ↓ 重新计算 KV cache
  ↓ 通过 KV connector 传输给 Decode 节点
  ↓
[Decode 节点 - kv_consumer]
  ↓ 接收 KV cache,继续 decode
```

### 6. RecomputeReqInfo 数据结构

**位置**: `vllm_ascend/core/recompute_scheduler.py:99-102`

```python
@dataclass
class RecomputeReqInfo:
    request_id: str
    output_token_ids: ConstantList   # 已生成的 output tokens
    client_index: int = 0            # 客户端索引,用于路由返回
```

**关键**: `output_token_ids` 被保留,Prefill 节点 recompute 时会重新计算这些 token 的 KV cache。

### 7. AsyncRecomputeScheduler

**位置**: `vllm_ascend/core/recompute_scheduler.py:88-92`

```python
if vllm_scheduler_config.async_scheduling:
    scheduler_config["scheduler_cls"] = "vllm_ascend.core.recompute_scheduler.AsyncRecomputeScheduler"
else:
    scheduler_config["scheduler_cls"] = "vllm_ascend.core.recompute_scheduler.RecomputeScheduler"
```

支持异步调度模式,提升调度吞吐。

---

## OOM 场景的完整调度链路分析

### 1. 为什么会触发 OOM 而非正常 Recompute?

根据 issue 描述,**Decode 节点配置中并未启用 `recompute_scheduler_enable`**:

```bash
# Decode 节点 additional-config
--additional-config {"multistream_overlap_shared_expert":true}
# ↑ 没有 recompute_scheduler_enable: true
```

对比 Kimi-K2.5 官方推荐配置 (`docs/source/tutorials/models/Kimi-K2.5.md:390`):
```bash
--additional-config '{"recompute_scheduler_enable":true}'  # ← 推荐
```

**因此 Decode 节点使用的是原生 vLLM Scheduler,preemption 后走本地 re-prefill 路径**。

### 2. 完整 OOM 触发链路 (修正版)

```
[Decode 节点 - 原生 Scheduler,未启用 recompute_scheduler]
  ↓ 高并发 576 → KV cache 不足
  ↓ allocate_slots() 返回 None
  ↓ 触发 _preempt_request():
    - kv_cache_manager.free(request)
    - num_computed_tokens = 0          # ← 重置!
    - 放回 waiting 队列头部
  ↓
[下一轮调度]
  ↓ waiting 队列中的 preempted 请求被重新调度
  ↓ num_new_tokens = num_tokens_with_spec - num_computed_tokens
  ↓                  = (4096 + 256 + output_len) - 0
  ↓                  = 整个序列长度 (可能 9000+)
  ↓ 多个 preempted 请求同时被调度
  ↓
[MoE forward]
  ↓ num_tokens 累加: 7 × 9083 = 63581
  ↓ npu_moe_init_routing: active_num = 63581 × 8 (top_k) = 508648
  ↓ npu_grouped_matmul_gmm2 分配 3.4GB
  ↓ _moe_forward_shared 已占 61GB
  ↓ OOM!
```

### 3. 关键差异: RecomputeScheduler vs 原生 Scheduler

| 维度 | 原生 Scheduler (当前配置) | RecomputeScheduler (推荐) |
|------|---------------------------|--------------------------|
| Preemption 后行为 | 本地 re-prefill,`num_computed_tokens=0` | 回送 Prefill 节点,本地释放 |
| Decode 节点内存 | 累积多个 re-prefill,内存暴涨 | 仅释放,不累积 |
| MoE hidden_states | `sum(seq_len) × top_k` 暴涨 | 无 re-prefill,保持 decode 规模 |
| OOM 风险 | **高** (多个长序列同时 recompute) | 低 (recompute 卸载到 Prefill) |
| 输出处理 | 请求重新执行 | `stop_reason="recomputed"` 返回客户端 |

---

## 关键代码位置 (补充)

| 文件 | 行号 | 说明 |
|------|------|------|
| `vllm/v1/core/sched/scheduler.py` | L446 | `allocate_slots()` 返回 None 触发 preemption |
| `vllm/v1/core/sched/scheduler.py` | L935-953 | `_preempt_request()`: 重置 `num_computed_tokens=0` |
| `vllm_ascend/core/recompute_scheduler.py` | L335-349 | **RecomputeScheduler 的 recompute 分支** (kv_consumer) |
| `vllm_ascend/core/recompute_scheduler.py` | L832-844 | recompute 请求返回 `stop_reason="recomputed"` |
| `vllm_ascend/platform.py` | L640-651 | `recompute_scheduler_enable` 启用逻辑 |
| `vllm_ascend/ascend_config.py` | L134 | `recompute_scheduler_enable` 配置读取 |

---

## 根因结论 (修正)

### 直接原因
Decode 节点**未启用 `recompute_scheduler_enable`**,导致 KV cache 不足时走原生 vLLM preemption 路径,多个长序列在 Decode 节点本地 re-prefill,MoE hidden_states 暴涨到 508648,触发 OOM。

### 根本原因
1. **配置缺失**: Decode 节点未配置 `recompute_scheduler_enable: true`
2. **高并发压力**: 576 并发导致 KV cache 耗尽,频繁触发 preemption
3. **多模态放大**: 每图 256 vision token 增加序列长度
4. **MoE top_k 扩展**: token 数 × 8 放大内存需求

### 修复建议

**立即修复**: 在 Decode 节点 additional-config 中启用 recompute scheduler:
```bash
--additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert":true}'
```

**参考**: Kimi-K2.5 官方文档 (`docs/source/tutorials/models/Kimi-K2.5.md:390`) 推荐配置。

---

## 验证建议 (补充)

### 1. 确认当前是否走 recompute 分支

在 `vllm_ascend/core/recompute_scheduler.py:340` 添加日志:
```python
if transfer_config is not None and not transfer_config.is_kv_producer:
    recomputed_req = self.running.pop()
    self.kv_cache_manager.free(recomputed_req)
    logger.warning(
        "Recompute triggered for request %s, seq_len=%d, output_len=%d",
        recomputed_req.request_id,
        recomputed_req.num_tokens,
        len(recomputed_req.output_token_ids),
    )
    ...
```

### 2. 确认原生 preemption 路径

在 `vllm/v1/core/sched/scheduler.py:947` 添加日志:
```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
    logger.warning(
        "Preempting request %s, num_tokens=%d, num_computed=%d, output_len=%d",
        request.request_id,
        request.num_tokens,
        request.num_computed_tokens,
        len(request.output_token_ids),
    )
    ...
```

### 3. 确认 MoE hidden_states 来源

在 `vllm_ascend/ops/fused_moe/moe_mlp.py:207` 前添加:
```python
logger.info(
    "gmm2 input: hidden_states.shape=%s, num_tokens=%d, top_k=%d",
    hidden_states.shape,
    hidden_states.shape[0] // 8,  # top_k=8
    8,
)
```

**预期结果**:
- 启用 `recompute_scheduler_enable` 后:不再出现 508648 的大 hidden_states
- 未启用时:会看到多个 preempted request,每个 seq_len ≈ 9083,然后 gmm2 hidden_states[0] = 508648
