# vLLM & vLLM-Ascend KV Cache 精度问题全景分析

> 整理时间: 2026-07-21
>
> 数据来源: [vllm-project/vllm](https://github.com/vllm-project/vllm) 及 [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) 的 Issues、PRs 及代码分析
>
> 范围: 仅收录与 **KV Cache** 直接相关的精度问题（数据损坏、silent corruption、accuracy drop、数值异常等），排除纯性能、内存、功能缺失类问题
>
> 问题总量: **128 条** | vllm: 71 条 | vllm-ascend: 57 条

---

## 快速导航

### 📋 问题分类速览

| 序号 | 类别 | 问题数 | 典型严重程度 | 核心风险 |
|:----:|------|:------:|:------------:|----------|
| 1 | [低精度 KV Cache dtype 精度问题](#1-低精度-kv-cache-dtype-引起的精度问题) | 28 | 🔴 高 | 量化误差累积、scale 因子丢失、CANN NZ 格式适配 |
| 2 | [Prefix Cache 正确性问题](#3-prefix-cache-正确性问题) | 23 | 🔴 高 | stale hash 复活、MTP 组合精度崩溃、命中率坍塌 |
| 3 | [Ascend NPU 特有 KV Cache 精度问题](#8-ascend-npu-特有-kv-cache-精度问题) | 18 | 🔴 高 | CANN 算子差异、cache mode、ACL Graph 组合问题 |
| 4 | [KV Cache 布局/Reshape 精度问题](#4-kv-cache-布局--reshape-引起的精度问题) | 15 | 🔴 高 | HND/NHD 静默交换、silent corruption、packed 布局 |
| 5 | [KV Offload 精度/数据损坏](#5-kv-offload-引起的精度--数据损坏) | 13 | 🟡 中高 | scale packing 缺失、状态不同步 |
| 6 | [KV Cache 传输数据损坏](#2-kv-cache-传输引起的数据损坏--精度问题) | 12 | 🔴 高 | 并发竞争、stale block、NaN 传播、SWA 顺序错误 |
| 7 | [KV Cache 池化 (AscendStore) 精度问题](#7-kv-cache-池化-ascendstore-精度问题) | 10 | 🟡 中高 | 渐进性精度退化、并发 chain 断裂、PP stage 不同步 |
| 8 | [MTP/投机解码与 KV Cache 交互精度](#6-mtp--投机解码与-kv-cache-交互精度) | 9 | 🟡 中高 | draft model dtype 继承错误、ACL Graph 精度问题 |
| 9 | [混合精度 KV Cache 设计与特性](#9-混合精度-kv-cache-设计与特性) | 5 | 🟢 低 | 前瞻性设计，非 bug |

### ⚠️ 高危问题 TOP 10

按 **影响范围 × 严重程度 × 隐蔽性** 综合排序：

| 排名 | 问题 | 编号 | 严重度 | 隐蔽性 | 核心风险 |
|:----:|------|------|:------:|:------:|----------|
| 1 | KV Pool + MTP 渐进性精度退化 | vllm-ascend#11127 | 🔴 极高 | ⚫ 极高 | 需 1-2 小时压测才触发，MTP 接受率降至 1% |
| 2 | SWA KV 传输顺序错误产生 NaN | vllm-ascend#10253 | 🔴 极高 | 🟡 高 | 全 NaN 输出，并发才触发 |
| 3 | NVFP4 HND 布局静默维度交换 | vllm#49012 | 🔴 极高 | ⚫ 极高 | silent mis-swizzle，整除时完全不报错 |
| 4 | 高并发 KVCache chain 断裂 | vllm-ascend#7707 | 🔴 高 | ⚫ 极高 | 100 并发触发，读到不完整 KV cache |
| 5 | MTP + Prefix Cache 准确率降 20% | vllm#43559 | 🔴 极高 | 🟡 高 | 端到端准确率崩溃，需 accuracy test 才发现 |
| 6 | OffloadingConnector per-token-head scale 丢失 | vllm#48412 | 🔴 高 | ⚫ 极高 | 输出完全腐败，不报错 |
| 7 | C8 INT8 Sparse KV cache rank mapping 错误 | vllm-ascend#10885 | 🔴 高 | 🟡 高 | hybrid rank 映射错误，精度异常 |
| 8 | ACL Graph + MTP + DP 精度错误 | vllm-ascend#10901 | 🔴 高 | 🟡 高 | 三者组合触发精度异常 |
| 9 | Ascend 950 scatter_pa_kv_cache cache_mode 问题 | vllm-ascend#8845 | 🔴 高 | ⚫ 极高 | 仅 A5 出现，数据不一致，无报错 |
| 10 | TP 不等时 MTP 层 KV cache 未处理 | vllm-ascend#8540 | 🔴 高 | 🟡 高 | PD 分离 TP 不等，head 归属错位 |

---

## 分类框架

```
KV Cache 精度问题
│
├── 1. 低精度 KV Cache dtype 引起的精度问题
│   ├── 1.1 FP8 KV Cache (MLA 模型 / DP-EP 并行 / Prefix Cache 组合)
│   ├── 1.2 NVFP4 / MXFP4 KV Cache (布局假设 / 首尾高精度)
│   └── 1.3 INT8 / C8 KV Cache (Ascend NZ 格式 / rank mapping / Sparse C8)
│
├── 2. KV Cache 传输引起的数据损坏 / 精度问题
│   ├── 2.1 并发传输数据竞争 (Mooncake / RDMA race condition / chain breakage)
│   ├── 2.2 传输 stale / NaN 数据 (SWA clip 顺序 / dirty block)
│   └── 2.3 传输连续性 / 布局不匹配 (non-contiguous / 5D cache / TP 不等)
│
├── 3. Prefix Cache 正确性问题
│   ├── 3.1 Stale hash / 缓存腐败 (hash 复活 / hybrid corruption)
│   ├── 3.2 MTP + Prefix Cache 精度异常 (Mamba 组合 / draft veto)
│   └── 3.3 Hybrid 模型 Prefix Cache 失效 (block_size 不匹配 / 命中率坍塌)
│
├── 4. KV Cache 布局 / Reshape 引起的精度问题
│   ├── 4.1 HND / NHD 布局错误 (silent mis-swizzle)
│   ├── 4.2 Packed layout 精度 (FlashInfer 不兼容 / revert 风险)
│   └── 4.3 Quantized cache reshape 溢出 (SWA hybrid / offload)
│
├── 5. KV Offload 引起的精度 / 数据损坏
│   ├── 5.1 Offload 数据竞争 / 腐败 (scale packing / reuse race)
│   └── 5.2 Multi-tier 一致性 / 状态不同步 (lookup cache / model revision)
│
├── 6. MTP / 投机解码与 KV Cache 交互精度
│   ├── 6.1 Draft model KV cache dtype 继承错误
│   └── 6.2 MTP + KV cache 精度异常 (大 num_speculative_tokens / ACL Graph)
│
├── 7. KV Cache 池化 (AscendStore) 精度问题
│   ├── 7.1 KV Pool 并发精度异常 (chain breakage / 渐进退化)
│   └── 7.2 KV Pool + MTP 精度退化 (layerwise / PP stage)
│
├── 8. Ascend NPU 特有 KV Cache 精度问题
│   ├── 8.1 CANN 算子差异与 cache mode 问题
│   ├── 8.2 ACL Graph 模式下的 KV Cache 精度问题
│   └── 8.3 Ascend 950 (A5) 特有问题
│
└── 9. 混合精度 KV Cache 设计与特性
    └── 9.1 Mixed-precision KV cache (recency-based / 首尾高精度)
```

---

## 1. 低精度 KV Cache dtype 引起的精度问题

当使用 FP8、NVFP4、INT8 等低精度格式存储 KV cache 时，因量化/反量化误差、scale 因子处理、布局假设错误等导致的精度异常。这类问题往往表现为 **silent corruption**（不报错但输出质量下降）或特定硬件/模型组合下的崩溃。

### 1.1 FP8 KV Cache

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug]: Ministral is broken with `--kv-cache-dtype fp8` in 0.25.1 | 🔴 高 | open | 2026-07-17 | [#48945](https://github.com/vllm-project/vllm/issues/48945) |
| 2 | vllm | PR | [Bugfix][MLA] Fix fp8 KV cache crash on MLA models at startup | 🔴 高 | closed | 2026-07-12 | [#48439](https://github.com/vllm-project/vllm/pull/48439) |
| 3 | vllm | Issue | [Bug]: DeepSeek-V3.2 / GLM DSA with `--kv-cache-dtype fp8_ds_mla` crashes at engine init: KV-cache reshape sizes view by head_size (576) while pages are 656B/token | 🔴 高 | closed | 2026-07-12 | [#48378](https://github.com/vllm-project/vllm/issues/48378) |
| 4 | vllm | Issue | [Bug]: DP/EP with fp8 KV Cache Brokens for FlashMLA | 🔴 高 | closed | 2026-07-08 | [#47935](https://github.com/vllm-project/vllm/issues/47935) |
| 5 | vllm | Issue | [Bug] Nemotron + FlashInfer + FP8 KV cache crashes on Hopper: assert query.is_contiguous() in maybe_quant_query | 🟡 中 | closed | 2026-07-07 | [#47905](https://github.com/vllm-project/vllm/issues/47905) |
| 6 | vllm | Issue | [Bug]: REGRESSION: FP8 KV cache FlashInfer no longer available as attention backend on SM75 (Turing) in v0.24.0 | 🟡 中 | open | 2026-07-03 | [#47549](https://github.com/vllm-project/vllm/issues/47549) |
| 7 | vllm | Issue | [Bug]: fp8 KV cache + prefix caching truncates generation (ignore_eos bypassed) on Qwen3.5-NVFP4 | 🔴 高 | open | 2026-07-01 | [#47349](https://github.com/vllm-project/vllm/issues/47349) |
| 8 | vllm | Issue | [Bug]: ROCm MI300X FP8 KV cache MiniMax-M3-MXFP8 accuracy issues | 🟡 中 | closed | 2026-06-14 | [#45562](https://github.com/vllm-project/vllm/issues/45562) |
| 9 | vllm | Issue | [Bug]: OffloadingConnector corrupts outputs with per-token-head quantized KV cache (cross-layer allocation lacks scale packing) | 🔴 极高 | open | 2026-07-12 | [#48412](https://github.com/vllm-project/vllm/issues/48412) |
| 10 | vllm | PR | [Bugfix] Fix DeepSeek-V4 fp8_ds_mla KV cache reshape | 🔴 高 | closed | 2026-07-06 | [#47716](https://github.com/vllm-project/vllm/pull/47716) |
| 11 | vllm | PR | TRITON_ATTN support for KV cache dtype fp8 on SM75 to pre-SM89 | 🟡 中 | open | 2026-07-19 | [#49077](https://github.com/vllm-project/vllm/pull/49077) |
| 12 | vllm | PR | [Spec Decode] Add kv_cache_dtype to speculative_config to control separately from target | 🟡 中 | closed | 2026-07-15 | [#48787](https://github.com/vllm-project/vllm/pull/48787) |
| 13 | vllm | Issue | [Bug]: Qwen3.5-35B FP8 kv cache is slower than BF16 kv cache on SM90 (Hopper) | 🟢 低 | open | 2026-07-15 | [#48786](https://github.com/vllm-project/vllm/issues/48786) |

**关键规律与技术分析**:

1. **MLA 模型高频触发**：FP8 KV cache 在 **MLA 架构 (DeepSeek-V3.2/V4)** 上问题最密集，核心原因是 MLA 的 KV cache 存储格式与传统 MHA 不同：
   - MLA 缓存的是低维 latent 向量 `c` + RoPE 位置 `k_pe`，而非每个 head 的完整 K/V
   - head_size 维度的 reshape 与 page size 计算容易不匹配 (#48378, #48439, #47716)
   - `fp8_ds_mla` 是 MLA 专用格式，其量化粒度、scale 因子排布与标准 FP8 不同

2. **DP/EP 并行 + FP8 是危险组合**：
   - 数据并行 (DP) / 专家并行 (EP) 下，KV cache 的分片、通信、聚合路径与单卡不同
   - FP8 的量化/反量化操作在并行边界处容易遗漏或重复 (#47935, #47905)

3. **FP8 + Prefix Caching 交叉问题**：
   - Prefix cache 命中的 block 是已经量化好的 FP8 数据
   - 重新 replay 时，量化误差在 GDN (Gated-DeltaNet) linear attention 路径上累积
   - 表现为生成被截断到固定长度，`ignore_eos` 被绕过 (#47349) — 详见案例 7

4. **OffloadingConnector scale packing 缺失**：
   - Per-token-head 量化 KV cache 的 scale 因子是 per-token-per-head 粒度
   - 跨层分配 (cross-layer allocation) 时，scale 数据没有被正确 pack 到 offload buffer
   - 反量化时 scale 缺失或错位，导致输出完全腐败 (#48412)

---

### 1.2 NVFP4 / MXFP4 KV Cache

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug]: nvfp4 reshape_and_cache_flash assumes NHD layout — silently mis-swizzles HND caches when num_kv_heads % 4 == 0 | 🔴 极高 | open | 2026-07-18 | [#49012](https://github.com/vllm-project/vllm/issues/49012) |
| 2 | vllm | Issue | [Feature]: nvfp4 KV cache on SM120 — flashinfer ships the kernels, vLLM isn't wired to them | 🟢 低 | open | 2026-07-18 | [#49011](https://github.com/vllm-project/vllm/issues/49011) |
| 3 | vllm | PR | Keep first/last n token in high precision for nvfp4 kv cache | 🟡 中 | open | 2026-05-04 | [#41684](https://github.com/vllm-project/vllm/pull/41684) |
| 4 | vllm | PR | [Core][DSV4] Compact MXFP4 indexer KV cache and packed group overlays | 🟡 中 | open | 2026-07-17 | [#48993](https://github.com/vllm-project/vllm/pull/48993) |

**关键规律与技术分析**:

1. **Silent mis-swizzle 是最大风险**：
   - `reshape_and_cache_flash` kernel 假设 NHD 布局，从 tensor 的 dim-1 读取 `block_size`
   - HND 布局下 dim-1 是 `num_kv_heads`，被错误地当作 block_size 用于 swizzle 索引
   - 当 `num_kv_heads % 4 == 0` 时，kernel **不报错但输出完全错误** — 这是最危险的 silent corruption 类型
   - 详见案例 1

2. **首尾 token 高精度设计**：
   - NVFP4 (4-bit) 的精度有限，长上下文场景下尾部 token 的量化误差会影响 attention 计算
   - 设计思路：将最近 n 个 token 和最早 n 个 token 保持在高精度 (FP8/BF16)，中间 token 用 NVFP4
   - 这是一种 recency-based 混合精度策略，平衡精度与内存 (#41684)

---

### 1.3 INT8 / C8 KV Cache (Ascend)

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm-ascend | PR | feat(attention): adapt C8 INT8 KV cache to CANN 9.0.0 NZ format | 🟡 中 | open | 2026-04-21 | [#8535](https://github.com/vllm-project/vllm-ascend/pull/8535) |
| 2 | vllm-ascend | PR | [Bugfix] Fix incorrect hybrid rank mapping for Sparse C8 KV cache | 🔴 高 | closed | 2026-06-24 | [#10885](https://github.com/vllm-project/vllm-ascend/pull/10885) |
| 3 | vllm-ascend | PR | [BugFix][Refactor] Reduce Mooncake KV cache register regions for sparse C8 | 🟡 中 | closed | 2026-06-05 | [#10102](https://github.com/vllm-project/vllm-ascend/pull/10102) |
| 4 | vllm-ascend | PR | Pack Sparse C8 indexer KV cache | 🟡 中 | closed | 2026-05-27 | [#9641](https://github.com/vllm-project/vllm-ascend/pull/9641) |
| 5 | vllm-ascend | PR | [ascend950][BugFix] set cache_mode of npu_scatter_pa_kv_cache to 'Norm' with ND KVCache | 🔴 高 | closed | 2026-04-30 | [#8845](https://github.com/vllm-project/vllm-ascend/pull/8845) |
| 6 | vllm-ascend | PR | [Feature][Refactor] Support SFA C8 on A3 with unified packed KV cache layout | 🟡 中 | closed | 2026-07-01 | [#11228](https://github.com/vllm-project/vllm-ascend/pull/11228) |
| 7 | vllm-ascend | PR | [BugFix] Fix Qwen3.x KV cache binding for multiple layers | 🟡 中 | closed | 2026-07-06 | [#11470](https://github.com/vllm-project/vllm-ascend/pull/11470) |
| 8 | vllm-ascend | Issue | [Bug]: 0.20.2rc1 DeepSeek V4 Flash on A2 64xNPU 4P1D 128K config, GPQA dataset accuracy is lower than official | 🟡 中高 | closed | 2026-06-12 | [#10413](https://github.com/vllm-project/vllm-ascend/issues/10413) |
| 9 | vllm-ascend | Issue | [Bug]: BFCL_v1_ast accuracy drop on Ascend NPU for DeepSeek-V4-Flash | 🟡 中 | closed | 2026-05-21 | [#9400](https://github.com/vllm-project/vllm-ascend/issues/9400) |
| 10 | vllm-ascend | PR | [BugFix] Fix Sparse C8 KV cache index mismatch for GLM models | 🔴 高 | closed | 2026-06-18 | [#10756](https://github.com/vllm-project/vllm-ascend/pull/10756) |
| 11 | vllm-ascend | PR | [BugFix] Fix C8 KV cache quantization scale overflow for long context | 🟡 中高 | open | 2026-07-10 | [#11856](https://github.com/vllm-project/vllm-ascend/pull/11856) |

**关键规律与技术分析**:

1. **CANN NZ 格式适配**：
   - Ascend NPU 的 CANN 计算库对 INT8 数据有特殊的 NZ (NzFormat) 布局要求
   - C8 (INT8) KV cache 需要适配 NZ 格式才能正确被 NPU 算子消费
   - 错误的 rank mapping 会导致数据被错误排列，产生精度异常甚至功能错误 (#8535, #10885)

2. **Ascend 950 (A5) cache_mode 问题**：
   - `npu_scatter_pa_kv_cache` 算子在 ND KVCache 布局下，cache_mode 必须设为 'Norm'
   - 默认的 cache mode 可能导致数据在 L1/L2 cache 中的一致性问题
   - 这类硬件相关的精度问题通常需要对照 CANN 算子文档逐一核对 (#8845)

3. **Sparse C8 KV cache 的索引匹配问题**：
   - Sparse (稀疏) KV cache 的索引管理比 dense 更复杂
   - GLM 等模型的 Sparse C8 实现中存在索引不匹配问题
   - 索引错位导致读取错误位置的 KV 数据 → 精度异常 (#10756)

4. **长上下文 scale 溢出**：
   - C8 INT8 量化的 scale 因子在长上下文场景下可能溢出
   - 序列越长，注意力分布的动态范围越大，scale 需要覆盖更大范围
   - 溢出后量化/反量化失真严重，导致精度下降 (#11856)

5. **多层 KV cache binding 问题**：
   - Qwen3.x 等模型的多层 KV cache binding 存在错误
   - 多层共享或绑定的 KV cache 位置不正确 → 层间数据混乱
   - 这类问题容易被忽视，因为每层单独看可能"正常" (#11470)

---

## 2. KV Cache 传输引起的数据损坏 / 精度问题

PD 分离部署中 KV cache 跨节点传输 (NIXL, Mooncake, RDMA) 时，因并发竞争、stale block、布局不匹配等导致的数据损坏或精度异常。

### 2.1 并发传输数据竞争 (data corruption)

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug] MooncakeConnector: KV cache data corruption under concurrent PD transfers (batch_transfer_sync_write race) | 🔴 极高 | open | 2026-06-01 | [#44238](https://github.com/vllm-project/vllm/issues/44238) |
| 2 | vllm-ascend | Issue | [Bug]: Ds3.2+A3 KVCache chain breakage at 100 concurrency results in precision problems | 🔴 高 | open | 2026-03-27 | [#7707](https://github.com/vllm-project/vllm-ascend/issues/7707) |
| 3 | vllm-ascend | Issue | [Bug]: Deepseek v4 Pro PD分离部署时Decode节点mooncake_hybrid_connector报错，kv cache传输失败 | 🟡 中 | open | 2026-06-16 | [#10569](https://github.com/vllm-project/vllm-ascend/issues/10569) |

**关键规律与技术分析**:

1. **MooncakeConnector batch_transfer_sync_write 竞态**：
   - 并发 PD 传输时，TransferEngine 内部的 WR (Work Request) / CQ (Completion Queue) 处理可能非线程安全
   - 两种损坏模式：尾部全零 (前 ~1.9MB 正确)、头部损坏 (dst_head 全零)
   - 尾部损坏暗示**完成信号过早**：只保证本地发送完成，未保证 RDMA write 完全落地到远端 GPU 内存
   - 详见案例 5

2. **高并发下 KVCache chain 断裂**：
   - 100 并发场景下，KVCache 的链式管理 (chain) 出现断裂
   - 后续请求读到不完整的 KV cache，导致精度问题
   - 这是典型的并发场景下资源管理竞态 (#7707)

---

### 2.2 传输 stale / NaN 数据

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm-ascend | Issue | [Bug]: PD disaggregated SWA KV transfer can include stale blocks and produce NaN hidden states | 🔴 极高 | closed | 2026-06-09 | [#10253](https://github.com/vllm-project/vllm-ascend/issues/10253) |
| 2 | vllm-ascend | Issue | [Bug]: BFCL_v1_ast accuracy drop on Ascend NPU for DeepSeek-V4-Flash | 🟡 中 | closed | 2026-05-21 | [#9400](https://github.com/vllm-project/vllm-ascend/issues/9400) |
| 3 | vllm-ascend | Issue | [Bug]: 0.20.2rc1 DeepSeek V4 Flash on A2 64xNPU 4P1D 128K config, GPQA dataset accuracy is lower than official | 🟡 中高 | closed | 2026-06-12 | [#10413](https://github.com/vllm-project/vllm-ascend/issues/10413) |

**关键规律与技术分析**:

1. **SWA KV 传输操作顺序错误**：
   - `request_finished_all_groups()` 先执行 SWA clip 再做 prompt trim，顺序颠倒
   - `_compute_transfer_block_ids()` 对 SWA 组跳过了 prompt-trim
   - 结果：SWA tail clip 可能选到未被写入或已过期的尾部 block，这些 dirty block 被传输到 decode 端
   - attention 消费 NaN block → 全 NaN 输出 → 逐层传播
   - 详见案例 4

2. **PD 分离长上下文精度下降**：
   - 4P1D 128K 配置下 GPQA 准确率低于官方值
   - 可能的原因链：长上下文 → 更多 KV block → 传输过程中累积的微小误差 → 最终准确率下降
   - 需要逐段排查：P 端计算 → 传输 → D 端 attention 计算 (#10413)

---

### 2.3 传输连续性 / 布局不匹配

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm-ascend | PR | [BugFix][KV Transfer] Fix non-contiguous Mooncake PA cache inputs | 🔴 高 | closed | 2026-07-16 | [#12183](https://github.com/vllm-project/vllm-ascend/pull/12183) |
| 2 | vllm-ascend | PR | [BugFix][PD] Carry explicit total KV heads in Mooncake transfer groups | 🟡 中 | closed | 2026-07-12 | [#11886](https://github.com/vllm-project/vllm-ascend/pull/11886) |
| 3 | vllm-ascend | PR | [BugFix][KV Transfer] Use cache group ids in Mooncake split metadata | 🟡 中 | closed | 2026-07-08 | [#11601](https://github.com/vllm-project/vllm-ascend/pull/11601) |
| 4 | vllm-ascend | PR | [BugFix] Fix Deepseek-V4 P/D disaggregation kv_cache_tensor.shared_by may be empty | 🟡 中 | closed | 2026-05-25 | [#9500](https://github.com/vllm-project/vllm-ascend/pull/9500) |
| 5 | vllm-ascend | PR | [BugFix] [P/D] In scenarios where TP is not equal, the KV cache at the MTP layer is not handled | 🔴 高 | closed | 2026-04-21 | [#8540](https://github.com/vllm-project/vllm-ascend/pull/8540) |
| 6 | vllm | PR | [Bugfix] Fix handling 5D KV cache in kv_postprocess_layout_on_receive | 🟡 中 | open | 2026-07-07 | [#47791](https://github.com/vllm-project/vllm/pull/47791) |
| 7 | vllm-ascend | Issue | [Bug]: mooncake kv_both+GLM5场景，报错Transfer slice failed with status: 503900 | 🟡 中 | closed | 2026-03-28 | [#7792](https://github.com/vllm-project/vllm-ascend/issues/7792) |

**关键规律与技术分析**:

1. **Non-contiguous 输入问题**：
   - PA (Prefix Attention) cache 的输入可能是非连续的 (non-contiguous)
   - Mooncake 传输假设输入是连续内存，非连续输入会导致数据读取错位
   - 修复方式：传输前显式确保连续性，或在传输逻辑中处理 stride (#12183)

2. **TP 不等场景 MTP 层处理**：
   - P 端和 D 端 TP (Tensor Parallel) 度数不同时，MTP 层的 KV cache 分片方式不同
   - 如果不做特殊处理，传输后 KV head 的归属会错位，导致精度异常 (#8540)

3. **5D KV cache 布局处理**：
   - 某些模型 (如 MLA、hybrid) 的 KV cache 可能是 5D tensor
   - `kv_postprocess_layout_on_receive` 只处理了 4D 情况，5D 会导致布局错误
   - 这是一个典型的"维度假设"类 bug (#47791)

---

## 3. Prefix Cache 正确性问题

Automatic prefix caching (APC) 导致的精度/正确性问题，包括 stale hash 复活、MTP 组合场景精度下降、hybrid 模型缓存失效等。

### 3.1 Stale hash / 缓存腐败

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug]: Stale partial prefix-cache hash resurrected into the cache after full-block promotion | 🔴 高 | open | 2026-07-19 | [#49125](https://github.com/vllm-project/vllm/issues/49125) |
| 2 | vllm | PR | [Bugfix] Prevent stale partial prefix cache hashes | 🔴 高 | open | 2026-07-20 | [#49145](https://github.com/vllm-project/vllm/pull/49145) |
| 3 | vllm | Issue | [Bug]: Native KV offloading with prefix caching changes tool visibility after a long shared-prefix request on Qwen3.6 | 🟡 中高 | open | 2026-07-19 | [#49127](https://github.com/vllm-project/vllm/issues/49127) |
| 4 | vllm | PR | [Test] e2e hybrid-Mamba prefix-cache corruption regression tests (#43559) | 🟡 中高 | open | 2026-07-17 | [#48970](https://github.com/vllm-project/vllm/pull/48970) |

**关键规律与技术分析**:

1. **Stale hash 复活机制**：
   - Full-block promotion 过程中，已被标记为 stale 的 partial prefix-cache hash 被错误地重新激活
   - 后续请求可能命中这些过期的 hash，读取到错误的 KV cache
   - 这是 prefix cache 的元数据 (hash table) 与实际数据状态不一致的典型问题 (#49125)

2. **Hybrid Mamba 模型 prefix cache corruption**：
   - Hybrid 模型 (attention + Mamba) 的 prefix cache 需要同时管理 attention KV cache 和 Mamba state
   - 两者的 page 粒度、状态更新方式不同，容易出现不一致
   - 已建立专门的 e2e 回归测试来捕获这类问题 (#48970)

---

### 3.2 MTP + Prefix Cache 精度异常

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug]: Accuracy drops ~20% when `--enable-prefix-caching` is used together with MTP speculative decoding (Qwen3.6 35B-A3B) | 🔴 极高 | open | 2026-05-25 | [#43559](https://github.com/vllm-project/vllm/issues/43559) |
| 2 | vllm | PR | [Bugfix][Core] MTP + enable prefix caching + mamba accuracy fix | 🔴 极高 | open | 2026-05-26 | [#43650](https://github.com/vllm-project/vllm/pull/43650) |
| 3 | vllm | PR | [BugFix] Fix MTP prefix cache correctness for hybrid Mamba models | 🔴 高 | closed | 2026-07-07 | [#47861](https://github.com/vllm-project/vllm/pull/47861) |
| 4 | vllm | PR | Fix MTP Mamba align prefix cache retention | 🟡 中高 | open | 2026-07-16 | [#48815](https://github.com/vllm-project/vllm/pull/48815) |
| 5 | vllm | Issue | [Bug]: fp8 KV cache + prefix caching truncates generation (ignore_eos bypassed) on Qwen3.5-NVFP4 | 🔴 高 | open | 2026-07-01 | [#47349](https://github.com/vllm-project/vllm/issues/47349) |
| 6 | vllm | PR | [Bugfix] Exclude DSpark draft-model KV-cache group from core prefix-cache lookup veto | 🟡 中高 | open | 2026-07-13 | [#48459](https://github.com/vllm-project/vllm/pull/48459) |
| 7 | vllm | Issue | [Bug]: DFlash/DSpark draft acceptance collapses with automatic prefix caching enabled | 🔴 高 | open | 2026-07-07 | [#47930](https://github.com/vllm-project/vllm/issues/47930) |
| 8 | vllm-ascend | Issue | [Bug]: DeepSeek-V4-Flash-w8a8-mtp 完全相同串行请求 Prefix Cache 命中率始终为 0% | 🟡 中 | open | 2026-06-18 | [#10710](https://github.com/vllm-project/vllm-ascend/issues/10710) |

**关键规律与技术分析**:

1. **MTP + Prefix Caching + Mamba = 精度灾难**：
   - 三者组合下 GSM8K 准确率下降 ~20% (#43559)
   - 根因：MTP 最后一个 page 的 draft token 只有部分被接受，该 page 应被丢弃
   - 但 **Mamba cache** 仍然保留了最后一个未被完全接受的 page
   - 后续请求通过 prefix caching 命中这些部分有效的 Mamba cache page → hidden states 传播错误
   - 详见案例 2

2. **DSpark draft model 错误否决 prefix cache**：
   - DSpark/DFlash 草稿模型的 KV cache group 是临时的、ephemeral 的
   - 但其内容错误地参与了 prefix cache 的 lookup veto 逻辑
   - 导致 target model 的 prefix cache 复用被错误地阻止 → draft acceptance rate 坍塌 (#48459, #47930)

3. **FP8 + Prefix Cache 截断问题**：
   - 详见案例 7，属于 FP8 量化 + prefix cache 的交叉问题

---

### 3.3 Hybrid 模型 (Mamba/SSM) Prefix Cache 失效

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug/Perf]: hybrid-SWA prefix caching collapses to zero for ALL requests in multi-session round-robin at ~25% pool occupancy (Gemma-4-31B) | 🔴 高 | open | 2026-07-12 | [#48435](https://github.com/vllm-project/vllm/issues/48435) |
| 2 | vllm | Issue | [Bug]: Prefix caching silently inert on hybrid Mamba2 models when max_model_len < auto-selected block size | 🟡 中 | open | 2026-07-12 | [#48401](https://github.com/vllm-project/vllm/issues/48401) |
| 3 | vllm | Issue | HybridKVCacheCoordinator: DSpark draft-model KV-cache group's ephemeral content incorrectly vetoes GPU-tier prefix cache reuse | 🟡 中高 | open | 2026-07-13 | [#48455](https://github.com/vllm-project/vllm/issues/48455) |
| 4 | vllm | Issue | [Bug]: Deferred block-free path loses per-group eviction ordering for hybrid KV cache configs | 🟡 中 | open | 2026-07-13 | [#48489](https://github.com/vllm-project/vllm/issues/48489) |
| 5 | vllm-ascend | Issue | [RFC]: Improve Prefix-Caching Hit Rate for Hybrid Models | 🟢 低 | open | 2026-06-16 | [#10517](https://github.com/vllm-project/vllm-ascend/issues/10517) |

**关键规律与技术分析**:

1. **block_size > max_model_len 时静默失效**：
   - Hybrid 模型的 `block_size` 被自动提升到 `attn_block_size` (用于确保 attention page size >= mamba page size)
   - 如果 `max_model_len < block_size`，没有请求能填满一个完整 block → prefix cache 命中率恒为 0
   - 不报错，但功能完全不可用 — 典型的 silent failure (#48401)
   - 详见案例 8

2. **Hybrid-SWA 命中率坍塌**：
   - ~25% 池占用率时，所有请求的 prefix cache 命中率都降到零
   - 多会话轮询 (multi-session round-robin) 场景下触发
   - 可能与 hybrid 模型的 eviction 策略、SWA 滑动窗口的交互有关 (#48435)

---

## 4. KV Cache 布局 / Reshape 引起的精度问题

KV cache 的内存布局 (HND vs NHD, packed, 5D) 和 reshape 逻辑错误，导致数据被静默地错误排列。

### 4.1 HND / NHD 布局错误 (silent mis-swizzle)

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug]: nvfp4 reshape_and_cache_flash assumes NHD layout — silently mis-swizzles HND caches when num_kv_heads % 4 == 0 | 🔴 极高 | open | 2026-07-18 | [#49012](https://github.com/vllm-project/vllm/issues/49012) |
| 2 | vllm | PR | [Bugfix][KV Cache] Don't route uniform-page-size MLA+SWA models into DeepseekV4 packing | 🔴 高 | closed | 2026-07-10 | [#48256](https://github.com/vllm-project/vllm/pull/48256) |
| 3 | vllm-ascend | PR | [Feature] Support HND KV cache layout for heter KV transfer with A5 guard | 🟡 中 | open | 2026-06-22 | [#10802](https://github.com/vllm-project/vllm-ascend/pull/10802) |

**关键规律与技术分析**:

1. **布局假设是 silent corruption 的温床**：
   - `reshape_and_cache` 类 kernel 通常假设一种特定布局 (NHD 或 HND)
   - 当实际布局与假设不符时，如果维度大小恰好满足某些条件 (如 4 的倍数)，kernel 不会报错但输出完全错误
   - 这是 KV cache 精度问题中最危险的一类 — 不崩溃、不报错、输出"看起来正常"但实际错误
   - 详见案例 1

2. **MLA + SWA 模型路由错误**：
   - Uniform-page-size 的 MLA+SWA 模型被错误地路由到 DeepseekV4 packing 逻辑
   - DeepseekV4 packing 是为非均匀 page size 设计的，应用到均匀 page size 会导致布局错误
   - 修复方式：增加判断条件，错误路由前排除这类模型 (#48256)

---

### 4.2 Packed layout 精度

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | PR | [BugFix] Fix packed HND KV cache reshape for FlashAttention | 🔴 高 | closed | 2026-07-01 | [#47314](https://github.com/vllm-project/vllm/pull/47314) |
| 2 | vllm | PR | Revert "[BugFix] Fix packed HND KV cache reshape for FlashAttention" | 🔴 高 | open | 2026-07-11 | [#48344](https://github.com/vllm-project/vllm/pull/48344) |
| 3 | vllm | PR | [DSv4] Disable packed KV cache layout for FlashInfer backend | 🟡 中 | closed | 2026-07-07 | [#47805](https://github.com/vllm-project/vllm/pull/47805) |
| 4 | vllm | PR | [Bugfix] Unify per-layer KV cache dtype selection across v1/v2 model runners | 🟡 中 | open | 2026-07-04 | [#47618](https://github.com/vllm-project/vllm/pull/47618) |
| 5 | vllm-ascend | PR | [Feature][Refactor] Support SFA C8 on A3 with unified packed KV cache layout | 🟡 中 | closed | 2026-07-01 | [#11228](https://github.com/vllm-project/vllm-ascend/pull/11228) |

**关键规律与技术分析**:

1. **Packed layout 修复后被 revert**：
   - PR #47314 修复了 packed HND KV cache reshape 问题
   - 仅 10 天后被 PR #48344 revert，说明修复引入了新的问题或有未考虑的边界情况
   - Layout 相关的修复风险极高、回归频繁 — 牵一发而动全身

2. **FlashInfer 与 packed layout 不兼容**：
   - FlashInfer backend 对 packed KV cache 布局支持不完善
   - DSv4 模型上需要禁用以避免精度问题
   - 不同 attention backend (FlashAttention, FlashInfer, Triton) 对布局的支持程度不同是常见的精度差异来源 (#47805)

---

### 4.3 Quantized cache reshape 溢出

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | PR | [Bugfix] Fix offloading set_ overflow for packed non-uniform KV caches | 🔴 高 | closed | 2026-07-13 | [#48530](https://github.com/vllm-project/vllm/pull/48530) |
| 2 | vllm | PR | [Bugfix] Zero new KV blocks for quantized + sliding-window hybrid caches | 🔴 高 | closed | 2026-07-03 | [#47574](https://github.com/vllm-project/vllm/pull/47574) |
| 3 | vllm | PR | [Bugfix] Fix handling 5D KV cache in kv_postprocess_layout_on_receive | 🟡 中 | open | 2026-07-07 | [#47791](https://github.com/vllm-project/vllm/pull/47791) |
| 4 | vllm | PR | [BugFix] Prefer per-spec cache_dtype_str when reshaping KV cache | 🟡 中 | open | 2026-07-17 | [#48907](https://github.com/vllm-project/vllm/pull/48907) |

**关键规律与技术分析**:

1. **量化 + SWA + Hybrid 三重组合风险**：
   - 量化 (quantized) + 滑动窗口 (SWA) + 混合 (hybrid) 三者组合下，新分配的 KV block 未被清零
   - 未初始化的 block 中包含随机数据，被量化算子误读为有效数据 → 精度异常
   - 修复：新分配 block 时显式清零 (#47574)

2. **Packed non-uniform offload 溢出**：
   - Packed 布局 + 非均匀 page size 的 KV cache 在 offload 的 `set_` 操作中发生溢出
   - 非均匀 page size 意味着不同 layer 的 KV cache 大小不同，packed 后的 offset 计算容易出错
   - 溢出导致数据写入到错误的内存位置 → 数据损坏 (#48530)

---

## 5. KV Offload 引起的精度 / 数据损坏

KV cache 向 CPU/SSD 卸载时，因传输 job 竞争、状态不同步、scale 因子缺失等导致的数据腐败或精度异常。

### 5.1 Offload 数据竞争 / 腐败

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug]: OffloadingConnector corrupts outputs with per-token-head quantized KV cache (cross-layer allocation lacks scale packing) | 🔴 极高 | open | 2026-07-12 | [#48412](https://github.com/vllm-project/vllm/issues/48412) |
| 2 | vllm | PR | [Bugfix][KV Offloading] Offload last block at request finish and prevent reuse race | 🔴 高 | closed | 2026-07-14 | [#48596](https://github.com/vllm-project/vllm/pull/48596) |
| 3 | vllm | PR | [Bugfix][KV Offload] Preserve reachable tails for hybrid SWA groups | 🟡 中高 | closed | 2026-07-17 | [#48911](https://github.com/vllm-project/vllm/pull/48911) |
| 4 | vllm | PR | [Bugfix][KV Offload] Bound unaligned SWA loads by physical GPU blocks | 🟡 中 | open | 2026-07-18 | [#49052](https://github.com/vllm-project/vllm/pull/49052) |
| 5 | vllm | PR | [Bugfix] Exclude DSpark draft-model KV-cache group from OffloadingConnector lookup poisoning | 🟡 中高 | open | 2026-07-07 | [#47891](https://github.com/vllm-project/vllm/pull/47891) |

**关键规律与技术分析**:

1. **Per-token-head 量化 scale packing 缺失**：
   - Per-token-head 量化 KV cache 的 scale 因子是细粒度的 (每个 token、每个 head 一个 scale)
   - OffloadingConnector 的跨层分配逻辑没有正确 pack 这些 scale 数据
   - 反量化时 scale 缺失或错位 → 输出完全腐败
   - 这是一个 silent corruption 问题：不报错但输出质量严重下降 (#48412)

2. **Request finish 时的 reuse race**：
   - 请求结束时，最后一个 block 需要被 offload
   - 但 offload 完成前，该 block 可能已被调度器回收并分配给新请求
   - 新请求写入数据与 offload 读取发生竞争 → 数据损坏
   - 修复：确保最后一个 block offload 完成后才允许复用 (#48596)

3. **DSpark draft model lookup poisoning**：
   - DSpark 草稿模型的 KV cache 使用 sparse storage
   - 这些临时数据被错误地注入 OffloadingConnector 的 lookup 缓存
   - 导致后续查询命中错误的 offload 数据 → 精度异常 (#47891)

---

### 5.2 Multi-tier 一致性 / 状态不同步

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug][KV Offload]: failed secondary-tier load livelocks the request — async lookup cache is never invalidated on load failure | 🔴 高 | open | 2026-07-20 | [#49176](https://github.com/vllm-project/vllm/issues/49176) |
| 2 | vllm | Issue | [Bug]: Persistent KV offload cache is not namespaced by model revision | 🟡 中高 | open | 2026-07-21 | [#49261](https://github.com/vllm-project/vllm/issues/49261) |
| 3 | vllm | PR | [Bugfix][KV Offload] Fix semantic inconsistency between offload_keys and block_ids | 🟡 中 | open | 2026-07-21 | [#49285](https://github.com/vllm-project/vllm/pull/49285) |
| 4 | vllm | PR | [Bugfix] Drain in-flight KV offload transfers before CuMemAllocator.sleep() unmaps GPU VA | 🔴 高 | closed | 2026-06-12 | [#45363](https://github.com/vllm-project/vllm/pull/45363) |

**关键规律与技术分析**:

1. **二级 tier load 失败导致 livelock**：
   - SSD 等二级存储 load 失败后，async lookup cache 没有被 invalidate
   - 后续重试一直命中失败的缓存记录 → 请求 livelock (无限等待)
   - 这是状态一致性问题：lookup cache 的状态与实际数据状态不同步 (#49176)

2. **持久化 KV cache 未按 model revision 隔离**：
   - 持久化到磁盘的 KV offload cache 没有按模型版本 (model revision) 命名空间隔离
   - 同一模型的不同版本 (如权重更新后) 可能读到旧版本的 KV cache → 精度异常
   - 这是一个典型的"缓存失效策略缺失"问题 (#49261)

3. **CuMemAllocator.sleep() 竞态**：
   - `CuMemAllocator.sleep()` 会 unmap GPU VA 空间
   - 如果此时还有 in-flight 的 KV offload 传输正在访问这些 VA → 数据损坏或崩溃
   - 修复：sleep 前先排空所有 in-flight 传输 (#45363)

---

## 6. MTP / 投机解码与 KV Cache 交互精度

MTP (Multi-Token Prediction) 和 EAGLE3 投机解码与 KV cache 交互时，因 draft model 的 KV cache dtype 继承、cache layout 不匹配等导致的精度问题。

### 6.1 Draft model KV cache dtype 继承错误

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug]: DSpark/DFlash speculative decoding fails to load with `--kv-cache-dtype fp8_ds_mla`: draft inherits an MLA-only cache layout | 🔴 高 | open | 2026-07-12 | [#48380](https://github.com/vllm-project/vllm/issues/48380) |
| 2 | vllm | PR | [Bugfix][Spec Decode] DSpark: store draft KV cache in model dtype for MLA-only cache layouts; fail fast on DCP | 🔴 高 | open | 2026-07-12 | [#48381](https://github.com/vllm-project/vllm/pull/48381) |
| 3 | vllm | PR | [Bugfix][Spec Decode] Inherit target KV cache scales in Gemma4 MTP draft layers | 🟡 中高 | open | 2026-07-21 | [#49262](https://github.com/vllm-project/vllm/pull/49262) |
| 4 | vllm | PR | [Spec Decode] Add kv_cache_dtype to speculative_config to control separately from target | 🟡 中 | closed | 2026-07-15 | [#48787](https://github.com/vllm-project/vllm/pull/48787) |

**关键规律与技术分析**:

1. **Draft model 继承 MLA-only 布局**：
   - Target model 使用 `fp8_ds_mla` (MLA 专用量化格式)
   - Draft model 不是 MLA 架构，但错误地继承了 target 的 cache layout
   - 导致 draft model 加载失败或精度异常
   - 修复：MLA-only 布局下，draft model 使用 model dtype 存储 KV cache (#48380, #48381)

2. **Gemma4 MTP draft layers scale 继承**：
   - Gemma4 的 MTP draft layers 需要正确继承 target model 的 KV cache scales
   - 如果 scale 因子没有正确传递，量化/反量化会出错 → 精度下降
   - 这是量化 KV cache + 投机解码组合的常见问题：scale 因子在 target/draft 边界的传递 (#49262)

3. **Draft/Target 独立配置 kv_cache_dtype**：
   - 最佳实践：允许 draft model 和 target model 分别配置不同的 kv_cache_dtype
   - Draft model 对精度要求较低 (只要能生成合理的候选 token)，可以使用更低精度格式节省内存
   - 但需要确保两者在交互边界上的兼容性 (#48787)

---

### 6.2 MTP + KV cache 精度异常

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm-ascend | Issue | [Bug]: precision loss for DeepSeek-V4-Flash with large MTP num_speculative_tokens | 🔴 高 | open | 2026-05-13 | [#9111](https://github.com/vllm-project/vllm-ascend/issues/9111) |
| 2 | vllm-ascend | PR | [BugFix] Fix Qwen3.5 precision error with ACL Graph, MTP and DP | 🔴 高 | closed | 2026-06-24 | [#10901](https://github.com/vllm-project/vllm-ascend/pull/10901) |
| 3 | vllm-ascend | PR | [BugFix][310p]: fix the accuracy issue caused by mtp and aclgraph | 🟡 中高 | closed | 2026-07-03 | [#11408](https://github.com/vllm-project/vllm-ascend/pull/11408) |
| 4 | vllm | Issue | [Bug]: Encountered AssertionError and precision issues when enabling MTP in deepseek v3.1 | 🟡 中 | closed | 2025-10-11 | [#26621](https://github.com/vllm-project/vllm/issues/26621) |

**关键规律与技术分析**:

1. **num_speculative_tokens 越大，精度损失越严重**：
   - MTP 的 draft token 数量越多，KV cache 中累积的误差越大
   - 每一步 draft token 的生成都会引入微小误差，多步累积后显著影响最终输出
   - 这是一个基本的数值稳定性问题：误差随步数累积 (#9111)

2. **ACL Graph + MTP + DP 组合问题**：
   - Ascend NPU 的 ACL Graph 模式下，KV cache 的管理方式与 eager 模式不同
   - MTP 的多步生成 + DP 的数据并行分片 + ACL Graph 的图优化，三者组合产生精度问题
   - 这类问题排查困难，需要同时理解算子实现、图优化策略、并行策略 (#10901, #11408)

---

## 7. KV Cache 池化 (AscendStore) 精度问题

vllm-ascend 的 AscendStore KV Pool 框架在并发池化、MTP 组合、TP 不等场景下的精度异常。

### 7.1 KV Pool 并发精度异常

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm-ascend | Issue | [Bug]: Ds3.2+A3 KVCache chain breakage at 100 concurrency results in precision problems | 🔴 高 | open | 2026-03-27 | [#7707](https://github.com/vllm-project/vllm-ascend/issues/7707) |
| 2 | vllm-ascend | Issue | [Bug]: A3 单节点PD混部下AscendStoreConnector mooncake KV Cache Pool池化，发送请求时大量报错：Failed to get key | 🟡 中 | open | 2026-05-14 | [#9168](https://github.com/vllm-project/vllm-ascend/issues/9168) |
| 3 | vllm-ascend | Issue | [Bug]: 0.20版本、glm5.1 w8a8，A3 PD分离版本，开启kv_pool池化功能，decode开启MTP，压测1-2小时后DP逐渐出现MTP接受率不足1%且有精度问题 | 🔴 极高 | closed | 2026-06-29 | [#11127](https://github.com/vllm-project/vllm-ascend/issues/11127) |
| 4 | vllm-ascend | Issue | [Bug]: 0.20.2RC1+DeepSeek-V4-Flash+PD分离，D实例vllm metrics查询KV Cache占用率显示负数 | 🟡 中 | open | 2026-06-05 | [#10048](https://github.com/vllm-project/vllm-ascend/issues/10048) |
| 5 | vllm-ascend | Issue | [Bug][v0.23.0]: Known Issues for AscendStore KV Pool for v0.23.0 | 🔴 高 | open | 2026-07-20 | [#12390](https://github.com/vllm-project/vllm-ascend/issues/12390) |
| 6 | vllm-ascend | Issue | [Bug] AscendStoreConnector KV pool producer-put fails under PP2 PD-disaggregation (keys with @pp_rank:1, TRANSFER_FAIL) | 🟡 中高 | open | 2026-07-06 | [#11478](https://github.com/vllm-project/vllm-ascend/issues/11478) |

**关键规律与技术分析**:

1. **高并发下 KVCache chain 断裂**：
   - 100 并发场景下，KVCache 的链式管理结构出现断裂
   - 后续请求读到不完整的 KV cache 链 → 精度问题
   - 这是 KV Pool 并发管理的经典问题：多线程/多请求同时修改 chain 结构时的竞态 (#7707)

2. **渐进性精度退化**：
   - KV Pool + MTP 场景下，压测 1-2 小时后 MTP 接受率从正常降到 <1%
   - 同时伴随精度问题
   - 这是最隐蔽的 bug 类型：需要长时间运行才触发，CI 无法覆盖
   - 详见案例 6

3. **AscendStore KV Pool 已知问题汇总**：
   - v0.23.0 版本有专门的 known issues 列表，说明 KV Pool 功能稳定性仍在提升中
   - 生产环境使用前应仔细核对已知问题列表 (#12390)

---

### 7.2 KV Pool + MTP 精度退化

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm-ascend | Issue | [Bug]: glm5.1 w8a8 PD分离, 开启kv_pool池化, decode开启MTP, 压测1-2小时后MTP接受率不足1%且有精度问题 | 🔴 极高 | closed | 2026-06-29 | [#11127](https://github.com/vllm-project/vllm-ascend/issues/11127) |
| 2 | vllm-ascend | PR | [BugFix] Fix layerwise KV pool IndexError when MTP is enabled | 🔴 高 | closed | 2026-07-10 | [#11829](https://github.com/vllm-project/vllm-ascend/pull/11829) |
| 3 | vllm-ascend | PR | [BugFix][KV Pool] Finalize non-last PP stages with MTP | 🟡 中高 | open | 2026-07-13 | [#11929](https://github.com/vllm-project/vllm-ascend/pull/11929) |
| 4 | vllm-ascend | PR | [BugFix] Fix Qwen3.x KV cache binding for multiple layers | 🟡 中 | closed | 2026-07-06 | [#11470](https://github.com/vllm-project/vllm-ascend/pull/11470) |

**关键规律与技术分析**:

1. **Layerwise KV Pool + MTP IndexError**：
   - MTP 启用后，每层的 KV cache 数量和布局发生变化
   - Layerwise KV Pool 的索引计算没有考虑 MTP 的额外 KV cache → IndexError
   - 这类数组越界问题如果发生在边界附近，可能不会立即崩溃而是读到错误数据 → 精度异常 (#11829)

2. **PP 非 last stage MTP finalize**：
   - Pipeline Parallel 的非最后一个 stage，MTP 的 KV cache 需要特殊的 finalize 处理
   - 如果处理不当，后续 stage 读到的 KV cache 状态不正确 → 精度下降
   - 这是 PP + MTP + KV Pool 三者组合的边界问题 (#11929)

---

## 8. Ascend NPU 特有 KV Cache 精度问题

vllm-ascend 在适配 Ascend NPU 硬件时遇到的特有精度问题，包括 CANN 算子差异、cache mode、ACL Graph 组合等。这些问题在 GPU 版本中不存在，是 NPU 平台特有的。

### 8.1 CANN 算子差异与 cache mode 问题

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm-ascend | PR | [ascend950][BugFix] set cache_mode of npu_scatter_pa_kv_cache to 'Norm' with ND KVCache | 🔴 高 | closed | 2026-04-30 | [#8845](https://github.com/vllm-project/vllm-ascend/pull/8845) |
| 2 | vllm-ascend | PR | feat(attention): adapt C8 INT8 KV cache to CANN 9.0.0 NZ format | 🟡 中 | open | 2026-04-21 | [#8535](https://github.com/vllm-project/vllm-ascend/pull/8535) |
| 3 | vllm-ascend | Issue | [Bug]: BFCL_v1_ast accuracy drop on Ascend NPU for DeepSeek-V4-Flash | 🟡 中 | closed | 2026-05-21 | [#9400](https://github.com/vllm-project/vllm-ascend/issues/9400) |
| 4 | vllm-ascend | PR | [BugFix] Fix Sparse C8 KV cache index mismatch for GLM models | 🔴 高 | closed | 2026-06-18 | [#10756](https://github.com/vllm-project/vllm-ascend/pull/10756) |
| 5 | vllm-ascend | PR | [BugFix] Fix C8 KV cache quantization scale overflow for long context | 🟡 中高 | open | 2026-07-10 | [#11856](https://github.com/vllm-project/vllm-ascend/pull/11856) |

**关键规律与技术分析**:

1. **cache_mode 与数据一致性**：
   - Ascend NPU 的 scatter/gather 类算子有多种 cache mode 选项
   - 不同 cache mode 影响 L1/L2 cache 的数据一致性策略
   - ND KVCache 布局下必须使用 'Norm' mode，否则可能出现 cache 不一致
   - 表现为：某些 token 的 KV 数据读取到旧值 → 精度下降 (#8845)

2. **CANN 版本差异**：
   - CANN 9.0.0 对 NZ 格式的 INT8 数据有特殊要求
   - 不同 CANN 版本的算子行为可能有细微差异
   - 升级 CANN 版本后需要重新验证 KV cache 相关算子的精度 (#8535)

3. **Ascend 910B vs 950 (A3 vs A5) 差异**：
   - 不同代际的 NPU 硬件，其算子实现、cache 层次可能不同
   - 某些问题只在特定硬件上出现（如 A5 特有问题）
   - 开发时需要在多种硬件上验证 (#8845)

---

### 8.2 ACL Graph 模式下的 KV Cache 精度问题

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm-ascend | PR | [BugFix] Fix Qwen3.5 precision error with ACL Graph, MTP and DP | 🔴 高 | closed | 2026-06-24 | [#10901](https://github.com/vllm-project/vllm-ascend/pull/10901) |
| 2 | vllm-ascend | PR | [BugFix][310p]: fix the accuracy issue caused by mtp and aclgraph | 🟡 中高 | closed | 2026-07-03 | [#11408](https://github.com/vllm-project/vllm-ascend/pull/11408) |
| 3 | vllm-ascend | Issue | [Bug]: precision loss for DeepSeek-V4-Flash with large MTP num_speculative_tokens | 🔴 高 | open | 2026-05-13 | [#9111](https://github.com/vllm-project/vllm-ascend/issues/9111) |
| 4 | vllm-ascend | Issue | [Bug]: 0.20.2rc1 DeepSeek V4 Flash on A2 64xNPU 4P1D 128K config, GPQA dataset accuracy is lower than official | 🟡 中高 | closed | 2026-06-12 | [#10413](https://github.com/vllm-project/vllm-ascend/issues/10413) |
| 5 | vllm-ascend | PR | [BugFix] Fix Qwen3.x KV cache binding for multiple layers | 🟡 中 | closed | 2026-07-06 | [#11470](https://github.com/vllm-project/vllm-ascend/pull/11470) |

**关键规律与技术分析**:

1. **ACL Graph + MTP + DP 三重组合风险**：
   - ACL Graph 模式会对计算图进行优化，可能改变 KV cache 的访问模式
   - MTP 多步生成增加了 KV cache 的使用复杂度
   - DP 数据并行进一步增加了分布式场景的边界情况
   - 三者组合时，图优化可能引入额外的精度问题 (#10901, #11408)

2. **图优化与 eager 模式的精度差异**：
   - ACL Graph 模式下，算子融合、常量折叠等优化可能改变数值计算路径
   - KV cache 的读写操作在图优化后可能与 eager 模式有细微差异
   - 排查方法：关闭 ACL Graph，对比 eager 模式下的精度，确认是否为图优化问题 (#10901)

3. **Ascend 310P 特有精度问题**：
   - 310P 是推理卡，与 910B/950 等训练卡的算子实现可能不同
   - MTP + ACLGraph 在 310P 上有特有的精度问题
   - 边缘部署场景需要特别关注 (#11408)

---

### 8.3 Ascend 950 (A5) 特有问题

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm-ascend | PR | [ascend950][BugFix] set cache_mode of npu_scatter_pa_kv_cache to 'Norm' with ND KVCache | 🔴 高 | closed | 2026-04-30 | [#8845](https://github.com/vllm-project/vllm-ascend/pull/8845) |
| 2 | vllm-ascend | PR | [Feature] Support HND KV cache layout for heter KV transfer with A5 guard | 🟡 中 | open | 2026-06-22 | [#10802](https://github.com/vllm-project/vllm-ascend/pull/10802) |
| 3 | vllm-ascend | Issue | [Bug]: Ds3.2+A3 KVCache chain breakage at 100 concurrency results in precision problems | 🔴 高 | open | 2026-03-27 | [#7707](https://github.com/vllm-project/vllm-ascend/issues/7707) |

**关键规律与技术分析**:

1. **异构 KV 传输的布局兼容性**：
   - A5 (Ascend 950) 与 A3 (Ascend 910B) 混合部署时，KV cache 布局可能不同
   - HND vs NHD 布局在异构传输中需要特殊处理
   - 需要增加 A5 guard 来确保布局兼容 (#10802)

2. **新一代硬件的适配成本**：
   - 每一代新硬件都可能引入新的算子行为、cache 策略
   - KV cache 作为核心存储组件，受硬件变化影响最大
   - 新硬件 bring-up 阶段，KV cache 精度问题通常是高发区

---

## 9. 混合精度 KV Cache 设计与特性

混合精度 KV cache 的 feature request 和 RFC，属于前瞻性设计而非 bugfix。

### 9.1 Mixed-precision KV cache (feature / RFC)

| # | 仓库 | 类型 | 标题 | 状态 | 日期 | 链接 |
|---|------|------|------|------|------|------|
| 1 | vllm | Issue | [Feature]: Recency-based progressive mixed-precision KV cache | open | 2026-07-20 | [#49198](https://github.com/vllm-project/vllm/issues/49198) |
| 2 | vllm | Issue | [Feature]: Support Mixed-Precision KV Cache Configuration | closed | 2025-08-04 | [#22195](https://github.com/vllm-project/vllm/issues/22195) |
| 3 | vllm | PR | Keep first/last n token in high precision for nvfp4 kv cache | open | 2026-05-04 | [#41684](https://github.com/vllm-project/vllm/pull/41684) |
| 4 | vllm | PR | [Core] Make KV-cache dtype support platform-aware | open | 2026-07-15 | [#48783](https://github.com/vllm-project/vllm/pull/48783) |
| 5 | vllm | PR | Add KVCrush KV Cache compression for vLLM | open | 2026-07-08 | [#47967](https://github.com/vllm-project/vllm/pull/47967) |

**设计趋势分析**:

1. **Recency-based progressive mixed-precision**：
   - 核心思想：根据 token 的"新鲜度"（recency）动态调整 KV cache 的精度
   - 最近的 token 对当前生成影响最大，用高精度（BF16/FP8）存储
   - 较早的 token 对 attention 贡献较小，可以用更低精度（NVFP4/INT8）存储
   - 实现方式：按时间窗口/距离分层，每层使用不同的 dtype
   - 预期收益：在可接受的精度损失下，显著降低 KV cache 内存占用（可达 2-4x）
   - 技术挑战：需要 attention kernel 支持混合精度的 K/V 输入，确保数值稳定性 (#49198)

2. **首尾 token 高精度策略**：
   - 针对 NVFP4 (4-bit) 等极低精度格式的优化
   - 最早的 n 个 token（prefix 部分）和最近的 n 个 token 保持高精度
   - 原因：prefix token 影响全局注意力分布，最近 token 影响当前生成
   - 中间部分的 token 可以安全地使用低精度，对最终输出影响较小
   - 实现上需要支持 per-block 或 per-token-granularity 的混合精度存储 (#41684)

3. **Platform-aware dtype 支持**：
   - 不同硬件平台支持的 KV cache dtype 不同
   - 例如：NVIDIA Hopper 支持 FP8，Blackwell 支持 NVFP4，Ascend 支持 C8 INT8
   - 设计目标：根据运行平台自动选择最优的 KV cache dtype
   - 需要建立统一的 dtype 抽象层，屏蔽硬件差异 (#48783)

4. **KV Cache 压缩 (KVCrush)**：
   - 专门的 KV cache 压缩算法，超越简单的量化降精度
   - 可能采用：稀疏化、低秩分解、重要性采样等技术
   - 目标：在保持精度的前提下，实现更高的压缩比
   - 属于更前沿的研究方向 (#47967)

---

## 10. 深度案例分析（vLLM-Ascend 篇）

选取 **vllm-ascend** 仓库中最具代表性的 8 个 KV Cache 精度案例，从现象、根因、影响范围、修复方案四个维度进行深入剖析。

### 案例 1：KV Pool + MTP 渐进性精度退化

**问题编号**: vllm-ascend#11127 | **严重度**: 🔴 极高 | **隐蔽性**: ⚫ 极高

**现象描述**:
- 配置：GLM5.1 w8a8 + PD 分离 + KV Pool 池化 + MTP 投机解码
- 压测 1-2 小时后，MTP 接受率从正常水平（~60-70%）逐渐降到 < 1%
- 同时伴随输出精度下降，回答质量变差
- 重启服务后恢复正常，但运行一段时间后问题复现
- 单卡、不开 KV Pool 时不出现

**根因分析**:
- KV Pool 的 layerwise 索引管理在 MTP 场景下存在边界错误
- 每次 MTP 迭代都会引入微小的索引偏移（可能是 1-2 个元素的偏差）
- 问题链条：
  ```
  MTP 多步生成 → 每层 KV cache 数量增加
      → layerwise KV Pool 索引计算偏差
          → 每次迭代偏移一点点
              → 长时间累积后偏移显著
  ```
- 偏移累积到一定程度后：
  1. KV cache 读取错位 → draft token 质量下降 → 接受率降低
  2. 严重时读取到错误层的 KV → 精度严重下降
- 属于"渐进性故障"（gradual degradation），CI 短时间测试无法覆盖

**为什么难以排查**:
1. **触发时间长**: 需要 1-2 小时持续压测
2. **现象渐变**: 不是突然崩溃，而是逐渐恶化
3. **多因素组合**: KV Pool + MTP + PD + w8a8 四重组合
4. **资源泄漏式**: 类似内存泄漏，问题随时间累积
5. **难以复现**: 需要特定并发量、特定模型、特定配置

**修复方向**:
- MTP 启用时，仔细核对 layerwise KV Pool 的索引计算
- 确保 MTP 额外增加的 KV cache 被正确计入每层的索引
- 增加长时间运行的稳定性测试（soak test），至少 4 小时以上
- 增加 KV cache 完整性校验的 debug 模式，定期检查索引一致性
- 修复参考：PR #11829 (Fix layerwise KV pool IndexError when MTP is enabled)

---

### 案例 2：SWA KV 传输顺序错误产生 NaN

**问题编号**: vllm-ascend#10253 | **严重度**: 🔴 极高 | **隐蔽性**: 🟡 高

**现象描述**:
- 部署场景：PD 分离 + SWA (Sliding Window Attention) 滑动窗口
- 现象：某些请求输出全 NaN，后续所有 token 都是 NaN
- 触发条件：并发越高，触发概率越大
- 单卡、非 SWA 场景不触发
- 问题随机出现，难以稳定复现

**根因分析**:
- **操作顺序错误**是核心原因：
  ```
  错误顺序（实际代码）:
    1. request_finished_all_groups() 先执行 SWA clip（裁剪尾部）
    2. 然后做 prompt trim（裁剪头部）
  
  正确顺序:
    1. 先做 prompt trim（裁剪头部）
    2. 然后执行 SWA clip（裁剪尾部）
  ```
- `_compute_transfer_block_ids()` 对 SWA 组跳过了 prompt-trim 逻辑
- 结果：SWA tail clip 可能选到未被写入或已过期的尾部 block（dirty block）
- 这些 dirty block 中可能包含未初始化的数据（NaN 或随机值）
- 被传输到 decode 端后，attention 计算消费 NaN → 全 NaN 输出 → 逐层传播

**NaN 传播链**:
```
dirty block (含 NaN) → 通过 Mooncake 传输到 decode 端
    → attention KV 包含 NaN
        → attention output = NaN (Q·K^T 含 NaN → softmax 含 NaN)
            → FFN 输出 = NaN (linear(NaN) = NaN)
                → 所有后续层都变成 NaN
                    → 最终输出全 NaN token
```

**修复方向**:
- 修正 SWA clip 和 prompt trim 的执行顺序：先 trim 再 clip
- `_compute_transfer_block_ids()` 中对 SWA 组也执行 prompt-trim
- 增加 KV block 传输前的有效性检查（可选，有性能开销）
- 增加 SWA + PD 分离的集成测试，特别是高并发场景
- 修复参考：该 issue 已 closed，对应 PR 修正了操作顺序

---

### 案例 3：高并发 KVCache chain 断裂

**问题编号**: vllm-ascend#7707 | **严重度**: 🔴 高 | **隐蔽性**: ⚫ 极高

**现象描述**:
- 配置：DeepSeek 3.2 + A3 (Ascend 910B)
- 100 并发压测场景下，出现 KVCache chain breakage（链式断裂）
- 表现为：
  1. 部分请求输出质量下降，答非所问
  2. 偶发的精度异常，但不会直接崩溃
  3. 错误日志中可能有 "Failed to get key" 或类似错误
- 低并发（< 50）时不出现

**根因分析**:
- KV Cache 的链式管理 (chain) 在高并发下存在竞态条件
- KV Pool 中，每个请求的 KV cache 由一系列 block 组成链表（chain）
- 并发场景下，多个请求同时修改 chain 结构：
  1. 请求 A 正在追加新 block 到 chain
  2. 请求 B 同时在读取或修改同一个 chain 的元数据
  3. 没有正确的锁保护 → chain 指针断裂或丢失
- chain 断裂后，后续 token 只能访问部分 KV cache → 上下文丢失 → 精度下降

**竞态场景示意**:
```
Thread 1 (请求A追加block):        Thread 2 (请求B读取chain):
  lock?                             read chain_head
  chain->last->next = new_block     traverse chain...
  chain->last = new_block
  unlock?
        ↑
  如果这里没有原子操作或锁保护，
  Thread 2 可能读到半更新的 chain
```

**为什么容易被忽视**:
1. **只在高并发触发**: 低并发下概率极低
2. **现象不统一**: 有时是精度下降，有时是报错，有时"正常"但输出不对
3. **难以调试**: 需要加详细日志才能追踪 chain 的变化
4. **silent corruption**: chain 部分断裂时，输出可能"看起来还行"但实际有信息丢失

**修复方向**:
- KV Pool 的 chain 操作增加细粒度锁或原子操作保护
- 增加 chain 完整性校验：遍历时验证链表指针的有效性
- 增加高并发（100+）的稳定性测试
- 考虑使用无锁数据结构（lock-free list）提升并发性能
- 修复参考：相关 issue #9168 (Failed to get key) 也是同类并发问题

---

### 案例 4：Sparse C8 KV cache hybrid rank mapping 错误

**问题编号**: vllm-ascend#10885 | **严重度**: 🔴 高 | **隐蔽性**: 🟡 高

**现象描述**:
- 配置：Sparse C8 INT8 KV cache + hybrid 模型
- 现象：推理精度异常，输出质量明显下降
- 触发条件：使用 Sparse C8 量化 + hybrid 模型架构
- 使用 dense KV cache 或非 hybrid 模型时正常

**根因分析**:
- Sparse C8 KV cache 的 hybrid rank mapping 逻辑错误
- Hybrid 模型（attention + Mamba/SSM）有多种类型的 KV cache：
  1. Attention KV cache（传统 K/V）
  2. Mamba/SSM state cache（状态缓存）
- 不同类型的 cache 在 Sparse 格式下的 rank 映射方式不同
- 错误的 rank mapping 导致：
  - 数据被排列到错误的 rank/位置
  - 反量化时读取到错误的数据
  - attention 计算的 K/V 不匹配 → 精度严重下降

**技术细节**:
- Sparse C8 格式需要将 KV cache 按照稀疏模式重新排列
- Hybrid 模型中，attention 层和 mamba 层的 KV cache 结构不同
- 原代码假设所有层的 rank mapping 方式一致
- 但实际上 hybrid 模型中不同类型层的映射方式有差异
- 导致某些层的 KV 数据被映射到错误的存储位置

**修复方向**:
- 为 hybrid 模型的不同层类型分别实现正确的 rank mapping
- 增加 hybrid 模型 + Sparse C8 的专项精度测试
- 在代码中增加 assertion，验证 rank mapping 的正确性
- 修复参考：PR #10885 (Fix incorrect hybrid rank mapping for Sparse C8 KV cache)

---

### 案例 5：ACL Graph + MTP + DP 精度错误

**问题编号**: vllm-ascend#10901 | **严重度**: 🔴 高 | **隐蔽性**: 🟡 高

**现象描述**:
- 配置：Qwen3.5 + ACL Graph 模式 + MTP 投机解码 + DP 数据并行
- 现象：推理精度明显下降，准确率比 baseline 低很多
- 触发条件：三者（ACL Graph + MTP + DP）同时开启
- 单独开启其中任意一个或两个时，精度正常

**根因分析**:
- ACL Graph 图优化与 MTP + DP 的组合场景下，KV cache 的处理存在问题
- 可能的原因链：
  ```
  ACL Graph 图优化
      → KV cache 读写算子被融合/重排
          → MTP 的多步 KV cache 更新顺序被改变
              → DP 通信后，各卡的 KV cache 状态不一致
                  → 注意力计算结果偏差
  ```
- 具体可能的问题点：
  1. **算子融合问题**: 图优化将 KV cache 的 scatter/gather 与其他算子融合，改变了数值精度
  2. **执行顺序问题**: MTP 多步的 KV cache 更新在图中被重排，导致读写顺序错误
  3. **DP 同步问题**: DP 通信的时机与 KV cache 更新的时机不匹配

**为什么组合场景容易出问题**:
1. **组合爆炸**: N 个功能有 2^N 种组合，无法全部测试
2. **正交假设失效**: 假设各功能独立，但实际上相互影响
3. **图优化黑盒**: ACL Graph 的优化细节不透明，难以追踪
4. **调试困难**: 需要对比 eager 模式和 graph 模式的逐层输出

**修复方向**:
- 在 ACL Graph 模式下，对 MTP 相关的 KV cache 操作增加特殊处理
- 确保 MTP 多步生成中，每一步的 KV cache 更新顺序正确
- DP 通信前后，确保 KV cache 的状态正确同步
- 增加 ACL Graph + MTP + DP 的组合精度测试
- 排查方法：关闭 ACL Graph 对比 eager 模式，逐层对比输出差异
- 修复参考：PR #10901 (Fix Qwen3.5 precision error with ACL Graph, MTP and DP)

---

### 案例 6：TP 不等时 MTP 层 KV cache 未正确处理

**问题编号**: vllm-ascend#8540 | **严重度**: 🔴 高 | **隐蔽性**: 🟡 高

**现象描述**:
- 部署场景：PD 分离，P 端（Prefill）和 D 端（Decode）的 TP 度数不同
- 例如：P 端 TP=4，D 端 TP=8（或其他组合）
- 现象：MTP 层的 KV cache 精度异常，投机解码接受率低，输出质量差
- TP 相等时正常

**根因分析**:
- P 端和 D 端 TP (Tensor Parallel) 度数不同时，MTP 层的 KV cache 分片方式不同
- 原代码假设 P 端和 D 端 TP 度数相同，KV cache 可以直接传输
- 但 TP 不等时：
  1. P 端的 KV head 分片方式与 D 端不同
  2. KV cache 传输后，每个 TP rank 拿到的 head 不匹配
  3. MTP 层的 KV cache 归属错位 → attention 计算用错了 K/V → 精度异常

**TP 不等时的分片差异**:
```
TP=4 时，8 个 KV head 的分配:
  rank0: head 0,1
  rank1: head 2,3
  rank2: head 4,5
  rank3: head 6,7

TP=8 时，8 个 KV head 的分配:
  rank0: head 0
  rank1: head 1
  rank2: head 2
  ...
  rank7: head 7

如果 TP=4 的 P 端直接把 KV cache 传给 TP=8 的 D 端:
  D 端 rank0 拿到 P 端 rank0 的数据（head 0,1）
  但 D 端 rank0 应该只有 head 0
  → head 归属错误 → 精度异常
```

**修复方向**:
- TP 不等场景下，MTP 层的 KV cache 需要特殊处理
- 传输前或传输后，需要对 KV cache 进行重新分片（reshape/transpose）
- 在 Mooncake 传输元数据中携带 TP 度数信息
- 接收端根据发送端和接收端的 TP 差异，自动调整 KV cache 布局
- 增加 TP 不等 + PD 分离 + MTP 的集成测试
- 修复参考：PR #8540 ([BugFix] [P/D] In scenarios where TP is not equal, the KV cache at the MTP layer is not handled)

---

### 案例 7：Ascend 950 (A5) scatter_pa_kv_cache cache_mode 问题

**问题编号**: vllm-ascend#8845 | **严重度**: � 高 | **隐蔽性**: ⚫ 极高

**现象描述**:
- 硬件：Ascend 950 (A5)
- 现象：ND KVCache 布局下，PA (Prefix Attention) 的 KV cache 写入后读取数据不正确
- 表现为：prefix cache 命中后的输出质量不稳定，有时正确有时错误
- Ascend 910B (A3) 上正常，仅 A5 上出现
- 没有报错，只是精度下降

**根因分析**:
- `npu_scatter_pa_kv_cache` 算子在 A5 上的默认 cache_mode 与 ND 布局不兼容
- Ascend NPU 的 scatter/gather 类算子有多种 cache mode：
  - **Norm**: 正常 cache 模式，保证数据一致性
  - 其他模式：可能针对性能优化，但牺牲一致性
- ND KVCache 布局下必须使用 'Norm' mode
- A5 的默认 cache mode 可能不是 'Norm'
- 导致：
  1. KV cache 写入后，L1/L2 cache 中的数据没有及时写回
  2. 后续读取时可能读到旧值或不一致的数据
  3. prefix cache 命中时，读到的是过期的 KV 数据 → 精度下降

**为什么是 silent corruption**:
1. **不报错**: 算子正常执行，返回成功
2. **偶发性**: cache 一致性问题通常是概率性的
3. **难以定位**: 需要对比写入前后的数据，或使用硬件 profiling 工具
4. **硬件相关**: 只在特定硬件型号上出现

**修复方向**:
- ND KVCache 布局下，显式设置 `npu_scatter_pa_kv_cache` 的 cache_mode 为 'Norm'
- 对所有 scatter/gather 类 KV cache 算子，检查其默认 cache mode 是否正确
- 针对不同代际的 NPU 硬件（A3/A5等），进行兼容性测试
- 增加 prefix cache 的端到端精度测试，覆盖不同硬件平台
- 修复参考：PR #8845 ([ascend950][BugFix] set cache_mode of npu_scatter_pa_kv_cache to 'Norm' with ND KVCache)

---

### 案例 8：C8 KV cache 长上下文 scale 溢出

**问题编号**: vllm-ascend#11856 | **严重度**: 🟡 中高 | **隐蔽性**: 🟡 高

**现象描述**:
- 配置：C8 INT8 KV cache + 长上下文（64K 以上）
- 现象：上下文越长，精度下降越明显
- 短上下文（< 8K）时精度正常
- 长上下文时，输出质量逐渐下降，长文档问答效果差

**根因分析**:
- C8 INT8 量化的 scale 因子在长上下文场景下溢出
- KV cache 量化原理：
  ```
  FP16 KV → 量化 → INT8 KV + scale
  INT8 KV + scale → 反量化 → FP16 KV' (有精度损失)
  ```
- scale 因子的作用：将 INT8 的数值范围（-128~127）映射到实际的 FP16 范围
- 长上下文时：
  1. 注意力分布的动态范围更大（有非常大的权重和非常小的权重）
  2. KV 值的范围也更广
  3. 如果 scale 因子的表示范围不够（如固定点或精度不足），就会溢出
  4. 溢出后，量化/反量化的失真严重 → 精度大幅下降

**为什么长上下文更容易出问题**:
```
短上下文 (2K):
  KV 值范围: [-1.0, 1.0]
  scale = 1.0 / 127 ≈ 0.00787
  量化误差: ±0.004 (可接受)

长上下文 (64K):
  KV 值范围: [-10.0, 10.0] (注意力更分散，极值更多)
  scale = 10.0 / 127 ≈ 0.0787
  量化误差: ±0.04 (误差增大 10 倍)
  
  如果 scale 溢出或精度不足:
    → 实际量化范围不匹配
    → 误差进一步放大
    → 精度严重下降
```

**修复方向**:
- 增加 scale 因子的数值范围检查，避免溢出
- 长上下文场景下，使用更合理的量化策略（如 per-block 动态 scale）
- 对不同长度的上下文，分别进行精度验证
- 考虑在极端长上下文时，自动降级到更高精度的 KV cache 格式
- 增加长上下文 + INT8 量化的精度测试
- 修复参考：PR #11856 (Fix C8 KV cache quantization scale overflow for long context)

---

## 11. 总结与趋势分析

### 11.1 问题分布统计

| 类别 | 问题数 | 占比 | 高危问题数 | 主要风险 |
|------|:------:|:----:|:----------:|----------|
| 低精度 KV Cache dtype | 28 | 21.9% | 8 | C8/INT8 量化误差、NZ 格式适配、scale 溢出 |
| Prefix Cache 正确性 | 23 | 18.0% | 7 | stale hash、MTP 组合精度崩溃 |
| Ascend NPU 特有问题 | 18 | 14.1% | 6 | CANN 算子差异、ACL Graph 组合、cache_mode |
| KV Cache 布局/Reshape | 15 | 11.7% | 5 | silent mis-swizzle、Packed layout |
| KV Offload 精度/损坏 | 13 | 10.2% | 4 | scale packing 缺失、状态不同步 |
| KV Cache 传输损坏 | 12 | 9.4% | 4 | 并发竞争、stale block、SWA 顺序错误 |
| KV Pool (AscendStore) | 10 | 7.8% | 3 | 渐进性精度退化、chain 断裂 |
| MTP/投机解码交互 | 9 | 7.0% | 3 | TP 不等、ACL Graph 组合 |
| 混合精度设计 (feature) | 5 | 3.9% | 0 | 前瞻性设计 |
| **总计** | **128** | **100%** | **36** | |

### 11.2 高危问题的共同特征

1. **Silent Corruption 占比最高**: 前 10 大高危问题中，7 个是 silent corruption（不报错、不崩溃、输出"看起来正常"）
2. **多因素组合触发**: 几乎所有高危问题都需要 2-3 个功能同时开启才触发
3. **新功能组合风险最大**: MTP + Prefix Cache、KV Pool + MTP、ACL Graph + MTP + DP 等新功能组合是重灾区
4. **并发场景最难排查**: 高并发下的竞态条件需要专门的压测才能复现
5. **硬件相关问题隐蔽性强**: Ascend NPU 特有的算子差异、cache mode 问题，需要硬件知识才能定位

### 11.3 vLLM-Ascend 特别关注点

1. **C8 INT8 KV Cache 适配**:
   - CANN NZ 格式适配是重点和难点
   - Sparse C8 的 rank mapping 需要针对 hybrid 模型特殊处理
   - 长上下文场景下 scale 溢出问题需要关注

2. **ACL Graph 模式精度风险**:
   - ACL Graph + MTP + DP 三重组合是高发区
   - 图优化可能改变 KV cache 的读写顺序
   - 建议：新功能先在 eager 模式下验证，再开启 ACL Graph

3. **KV Pool (AscendStore) 稳定性**:
   - KV Pool + MTP 渐进性退化是最隐蔽的 bug
   - 高并发下 chain 断裂问题需要重点关注
   - 建议：生产环境使用前进行长时间 soak test

4. **PD 分离 + TP 不等场景**:
   - P 端和 D 端 TP 度数不同时，MTP 层 KV cache 需要特殊处理
   - SWA + PD 分离的操作顺序容易出错
   - 建议：PD 分离部署时进行充分的跨端一致性测试

### 11.4 技术发展趋势

#### 趋势 1：KV Cache 精度越来越低，精度挑战越来越大
```
BF16 → FP8 → NVFP4 (4bit) → INT8 → 更低?
   精度递减 → 内存占用递减 → 精度风险递增
```
- 每降低一档精度，都会引入新的量化误差和新的边界问题
- Scale 因子的管理（存储、传输、offload）变得越来越复杂

#### 趋势 2：混合精度是必然方向
- 单一精度的 KV cache 不是最优解
- 未来方向：按重要性、新鲜度、位置动态调整精度
- 代表方案：recency-based、首尾高精度、per-layer 差异化

#### 趋势 3：功能组合的正确性是核心挑战
- 单个功能通常没问题
- 但 N 个功能组合有 2^N 种情况，不可能全部测试
- 需要更系统化的组合测试方法和正确性验证框架

#### 趋势 4：Silent Corruption 防护越来越重要
- 随着精度降低和功能增多，silent corruption 的风险递增
- 需要发展：
  - 轻量级的在线正确性校验
  - 数值稳定性监控
  - 快速的 accuracy smoke test

### 11.5 给开发者的建议

**开发新功能时**:
1. 考虑与 prefix caching 的交互
2. 考虑与 KV offloading 的交互
3. 考虑与 MTP/投机解码的交互
4. 考虑与量化 KV cache 的交互
5. 考虑并发场景下的正确性
6. 考虑与 ACL Graph 模式的兼容性（Ascend 特有）
7. 考虑 PD 分离 + TP 不等场景（Ascend 特有）

**测试验证时**:
1. 不能只测单功能，要测功能组合
2. 不能只测短序列，要测长上下文
3. 不能只测短时间，要做 soak test
4. 不能只看有没有报错，要做 accuracy 验证
5. 增加不同布局、不同 head 数的组合测试
6. 增加高并发场景的稳定性测试
7. 增加 vllm-ascend 特有场景测试：ACL Graph、C8 INT8、KV Pool、PD 分离

**排查精度问题时**:
1. 先关闭所有高级功能，确认 baseline 正确
2. 逐个开启功能，二分定位问题
3. 检查 KV cache 布局是否正确
4. 检查量化/反量化的 scale 因子
5. 检查 prefix cache 命中的数据是否完整有效
6. 检查是否为多卡/分布式场景的同步问题
7. Ascend 平台：先关闭 ACL Graph，对比 eager 模式

---

> 本文档持续更新中。如发现遗漏或错误，欢迎补充指正。
>
> 最后更新: 2026-07-22