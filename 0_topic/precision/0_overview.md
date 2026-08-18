# vLLM & vLLM-Ascend 精度问题 · 知识库总览

> 本文档是 `precision/` 目录的**总入口**：把「全景统计 → KV Cache 专题 → 必现案例集 → 候选留档」四层资料串成一张地图，回答三个问题——**这套资料里有什么、各层解决什么问题、应该怎么用**。
>
> 整理时间：2026-08-19（案例集扩展到 32 个并按复现难度重排后刷新）

---

## 0. 文档地图（一层一张表）

```
precision/
├── 0_overview.md   ◀◀◀ 你在这里（体系总览 / 导航）
├── 0_precision.md     ① 全景分析：vllm+vllm-ascend 全部精度问题统计（1915 条）
├── 0_kvcache.md       ② KV Cache 专题：与 KV Cache 直接相关的精度问题（158 条）
└── cases/             ③ 必现案例集：32 个「必现且有解决方案」的深度案例
    ├── README.md         案例索引 + 复现前置 + 共同脉络 + 候选留档
    ├── 1_单机单卡/       案例 01–13（单卡即可复现）
    ├── 2_单机多卡/       案例 14–22（单节点 2~16 卡）
    └── 3_多机多卡/       案例 23–32（多节点 PD 分离）
```

| 层 | 文件 | 数量级 | 回答的问题 | 适合谁 |
|----|------|--------|-----------|--------|
| ① 全景 | `0_precision.md` | 1915 条 | 精度问题整体长什么样、分布与规律 | 想建认知 / 摸底 |
| ② KV 专题 | `0_kvcache.md` | 158 条 | KV Cache（PD/量化/offload）这条线怎么坏 | 排查 KV 类问题 |
| ③ 必现案例 | `cases/`（32 个） | 32 条 | 具体 bug 怎么触发、根因、怎么修、怎么复现 | 遇到具体 bug / 做回归 |
| ④ 候选留档 | `cases/README.md` | 11 条 | 已发现但未确认 / 未合并的问题跟踪 | 跟进趋势 |

> **一句话导航**：抽象规律看 ① ②，动手排查看 ③，前瞻跟踪看 ④。

---

## ① 全景分析（0_precision.md）

### 1.1 问题总量与构成

| 维度 | 数量 |
|------|------|
| 问题总量 | **1915 条** |
| vllm / vllm-ascend | 1252 / 663 |
| 类型 Issue / PR | 838 / 1077 |
| 状态 open / closed | 441 / 1474 |

### 1.2 八大分类速览

| # | 类别 | 问题数 | 核心风险 |
|:--:|------|:------:|----------|
| 1 | 量化与低精度 dtype | 568 | 量化/反量化误差、scale 因子丢失、布局不匹配 |
| 2 | 数值稳定性 | 237 | NaN/Inf 传播、溢出/下溢、除零 |
| 3 | 输出正确性与一致性 | 361 | 与参考不一致、garbage 输出、结果漂移 |
| 4 | 算子与注意力数值精度 | 36 | FlashAttention/PagedAttention/MLA 等算子数值差异 |
| 5 | KV Cache 精度（详见 ②） | 122 | 数据损坏、silent corruption |
| 6 | 精度回归与评测 | 400 | 端到端准确率下降、perplexity 退化 |
| 7 | Ascend NPU 特有精度 | 20 | CANN 算子差异、ACL Graph 图优化差异 |
| 8 | 其他/泛化精度 | 171 | 未归类问题 |

**核心规律**：量化与低精度 dtype（568）是最大单一来源，常表现为 **silent degradation（不报错但质量下降）**；NPU 特有（20 条）问题数少但隐蔽性极高。

> 更细的分类统计与「关键规律与分析」见 [0_precision.md](./0_precision.md) 各类小节约 8 处。

---

## ② KV Cache 专题（0_kvcache.md）

### 2.1 问题总量

| 维度 | 数量 |
|------|------|
| 问题总量 | **158 条**（正文 128 + 2026-07-22→08-18 增量 30） |
| 范围 | 仅 KV Cache 直接相关的精度问题（数据损坏、silent corruption、accuracy drop） |

### 2.2 分类速览

| # | 类别 | 问题数 | 典型严重度 | 核心风险 |
|:--:|------|:------:|:----------:|----------|
| 1 | 低精度 KV Cache dtype | 28 | 🔴 高 | scale 丢失、CANN NZ 格式适配 |
| 2 | Prefix Cache 正确性 | 23 | 🔴 高 | stale hash、MTP 组合崩溃、命中率坍塌 |
| 3 | Ascend NPU 特有 KV | 18 | 🔴 高 | CANN 算子差异、cache mode、ACL Graph |
| 4 | KV Cache 布局/Reshape | 15 | 🔴 高 | HND/NHD 静默交换、packed 布局 |
| 5 | KV Offload | 13 | 🟡 中高 | scale packing 缺失、状态不同步 |
| 6 | KV Cache 传输数据损坏 | 12 | 🔴 高 | 并发竞争、stale block、NaN、SWA 顺序 |
| 7 | KV Cache 池化 (AscendStore) | 10 | 🟡 中高 | 渐进退化、并发 chain 断裂 |
| 8 | MTP/投机解码 × KV | 9 | 🟡 中高 | draft dtype 继承错误、ACL Graph |
| 9 | 混合精度 KV Cache 设计 | 5 | 🟢 低 | 前瞻性设计，非 bug |

### 2.3 高危 TOP 10（影响 × 严重度 × 隐蔽性）

| 排名 | 问题 | 编号 | 核心风险 |
|:--:|------|------|----------|
| 1 | KV Pool + MTP 渐进性精度退化 | vllm-ascend#11127 | 压测 1-2h 触发，MTP 接受率→1% |
| 2 | SWA KV 传输顺序错误产生 NaN | vllm-ascend#10253 | 全 NaN 输出，并发才触发 |
| 3 | NVFP4 HND 布局静默交换 | vllm#49012 | silent mis-swizzle，整除时不报错 |
| 4 | 高并发 KVCache chain 断裂 | vllm-ascend#7707 | 100 并发读到不完整 cache |
| 5 | MTP + Prefix Cache 准确率 -20% | vllm#43559 | 端到端崩溃，accuracy 才暴露 |
| 6 | OffloadingConnector scale 丢失 | vllm#48412 | 输出完全腐败 |
| 7 | C8 INT8 Sparse rank mapping 错 | vllm-ascend#10885 | hybrid rank 映射错 |
| 8 | ACL Graph + MTP + DP 精度错 | vllm-ascend#10901 | 三者组合触发 |
| 9 | scatter_pa_kv_cache cache_mode | vllm-ascend#8845 | 仅 A5，无报错 |
| 10 | TP 不等 MTP 层 KV 未处理 | vllm-ascend#8540 | PD 分离 TP 不等，head 归属错位 |

> 增量更新已在 ① 与 ② 中体现（FP8 KV cache、310P MTP+prefixcache、CUDA Graph × 低精度等新热点），详见 [0_kvcache.md](./0_kvcache.md)。

---

## ③ 必现案例集（cases/，32 个）

**编号即复现难度**：01 → 32 从易到难（单机单卡 → 单节点多卡 → 多节点 PD 分离）。

### 3.1 三个复现梯队

| 梯队 | 文件夹 | 复现前置 | 案例 |
|:----:|:-------|----------|------|
| **1 · 单机单卡** | `cases/1_单机单卡/` | 一张 NPU 卡甚至 CPU 即可 | 01–13 |
| **2 · 单机多卡** | `cases/2_单机多卡/` | 单节点 2~16 卡（个别需 A5 硬件） | 14–22 |
| **3 · 多机多卡 / PD 分离** | `cases/3_多机多卡/` | ≥2 NPU 节点 + 编译 Mooncake（25 需高并发、29 需 1–2h 压测） | 23–32 |

### 3.2 六条共同脉络（根因抽象）

| # | 脉络 | 案例 | 本质 |
|:--:|------|------|------|
| 1 | **元数据/标量口径不一致** | 05 21 23 26 27 30 | 元数据不自描述，依赖隐式对齐 |
| 2 | **操作顺序/语义分支陷阱** | 03 22 25 28 29 | trim/clip 顺序、`>=` 边界、decode≠prefill、哨兵值 |
| 3 | **量化分块/tiling 边界容量** | 02 07 10 20 | `cdiv` 补块、UB 预算、系数显式传递与 dtype 策略 |
| 4 | **几何/stride/公式推导错误** | 01 04 12 18 | 非整除、stride 泄漏、公式常数化 |
| 5 | **状态残留与续算边界** | 06 09 11 13 14 | 「第二次调用/chunk」才暴露的脏状态 |
| 6 | **异步/图模式/并行合法性与跨 rank 时序** | 08 15 16 17 19 24 31 | happens-before、图捕获一致性、SP×EP、传输完成时序 |

### 3.3 32 个案例索引

| # | 对象 | 一句话定位 | 梯队 | 严重度 |
|:--:|------|-----------|:--:|:--:|
| 01 | [vllm PR #41277](https://github.com/vllm-project/vllm/pull/41277) | dynamic NTK RoPE 公式常数化，超训练长度位置编码全错 | 1 | 🔴 高 |
| 02 | [PR #12424](https://github.com/vllm-project/vllm-ascend/pull/12424) | swiglu_quant 小形状 UB 漏算 scale buffer | 1 | 🟡 中 |
| 03 | [vllm PR #46533](https://github.com/vllm-project/vllm/pull/46533) | rejection sampler 接受 -1 占位符越界读 | 1 | 🟡 中 |
| 04 | [vllm PR #49292](https://github.com/vllm-project/vllm/pull/49292) | Qwen3-VL M-RoPE stride 泄入 shape 推导 | 1 | 🟡 中 |
| 05 | [vllm PR #42143](https://github.com/vllm-project/vllm/pull/42143) | EAGLE3 嵌套配置 norm_before_fc 被静默跳过 | 1 | 🟡 中 |
| 06 | [vllm PR #35157](https://github.com/vllm-project/vllm/pull/35157) | reset prefix cache 残留 stale mamba_state_idx | 1 | 🔴 高 |
| 07 | [PR #9036](https://github.com/vllm-project/vllm-ascend/pull/9036) | ascend 量化漏传 routed_scaling_factor | 1 | 🔴 高 |
| 08 | [PR #6958](https://github.com/vllm-project/vllm-ascend/pull/6958) | LoRA × 图模式激活数不一致 | 1 | 🟡 中 |
| 09 | [vllm Issue #51094](https://github.com/vllm-project/vllm/issues/51094) | Mamba offload 精确 chunk 边界静默错 | 1 | 🟡 中 |
| 10 | [PR #12201](https://github.com/vllm-project/vllm-ascend/pull/12201) | MXFP4 weight_scale floor 丢尾部块 | 1 | 🟡 中 |
| 11 | [PR #11508](https://github.com/vllm-project/vllm-ascend/pull/11508) | PCP+chunk SSM 状态递推重复计算 | 1 | 🔴 高 |
| 12 | [vllm Issue #49716](https://github.com/vllm-project/vllm/issues/49716) | int8 KV 混合 head_dim 布局假整除 | 1 | 🔴 高 |
| 13 | [vllm Issue #45704](https://github.com/vllm-project/vllm/issues/45704) | CPU offload store 缺 wait_stream | 1 | 🟡 中 |
| 14 | [PR #5647](https://github.com/vllm-project/vllm-ascend/pull/5647) | PCP buffer 跨调用未重置 | 2 | 🔴 高 |
| 15 | [PR #5816](https://github.com/vllm-project/vllm-ascend/pull/5816) | EAGLE3+SP drafter 走错通信路径 | 2 | 🔴 高 |
| 16 | [PR #14081](https://github.com/vllm-project/vllm-ascend/pull/14081) | async+piecewise 缺 stream 同步 | 2 | 🔴 高 |
| 17 | [Issue #4273](https://github.com/vllm-project/vllm-ascend/issues/4273) | DP8 量化 SP 未开 EP（护栏） | 2 | 🟡 中 |
| 18 | [Issue #12723](https://github.com/vllm-project/vllm-ascend/issues/12723) | Triton RoPE 非 2 幂偏移 | 2 | 🔴 高 |
| 19 | [PR #7460](https://github.com/vllm-project/vllm-ascend/pull/7460) | FULL_DECODE_ONLY 图模式 padded 错误 | 2 | 🔴 高 |
| 20 | [PR #11663](https://github.com/vllm-project/vllm-ascend/pull/11663) | A5 MXFP4 EP topk_weights cast float8 | 2 | 🔴 高 |
| 21 | [PR #7079](https://github.com/vllm-project/vllm-ascend/pull/7079) | EAGLE3+CP 元数据切分用错变量 | 2 | 🟡 中高 |
| 22 | [PR #14248](https://github.com/vllm-project/vllm-ascend/pull/14248) | DSA decode 被当 prefill 处理 | 2 | 🔴 高 |
| 23 | [PR #11886](https://github.com/vllm-project/vllm-ascend/pull/11886) | PD 序列化缺全局 KV head 元数据 |3| 🟡 中 |
| 24 | [PR #12359](https://github.com/vllm-project/vllm-ascend/pull/12359) | PD 单 shard pull 完成即重排致 TP 不一致 |3| 🔴 高 |
| 25 | [Issue #10253](https://github.com/vllm-project/vllm-ascend/issues/10253) | SWA clip 与 trim 顺序颠倒 → 全 NaN |3| 🔴 极高 |
| 26 | [PR #8540](https://github.com/vllm-project/vllm-ascend/pull/8540) | PD TP 不等时 MTP 层 KV 未重排 |3| 🔴 高 |
| 27 | [PR #11601](https://github.com/vllm-project/vllm-ascend/pull/11601) | PD 传输元数据缺 cache group id |3| 🟡 中 |
| 28 | [PR #12183](https://github.com/vllm-project/vllm-ascend/pull/12183) | PD PA 算子直连绕过布局归一化 |3| 🔴 高 |
| 29 | [Issue #11127](https://github.com/vllm-project/vllm-ascend/issues/11127) | PD+MTP `>=` 误判边界，接受率 <1% |3| 🔴 高 |
| 30 | [PR #9500](https://github.com/vllm-project/vllm-ascend/pull/9500) | PD `shared_by` 空未守卫 |3| 🟡 中 |
| 31 | [PR #13195](https://github.com/vllm-project/vllm-ascend/pull/13195) | PD/PCP/DCP 图模式按 num_tokens 判 PA |3| 🔴 高 |
| 32 | [Issue #12339](https://github.com/vllm-project/vllm-ascend/issues/12339) | 超大 MoE FULL_QUANT 输出交替异常 |3| 🔴 高 |

> 每个案例统一五段制：**① 问题描述（现象/触发矩阵/影响）→ ② 版本信息（vllm + vllm-ascend）→ ③ 定位过程 → ④ 解决方案（根因/修复 diff/验证）→ ⑤ 复现方法**，附核心教训。完整清单与复现命令见 [cases/README.md](./cases/README.md)。

---

## ④ 候选留档（尚未「已修复」，跟踪项）

| # | 对象 | 一句话定位 | 状态 |
|:--:|------|-----------|------|
| C1 | [Issue #14339](https://github.com/vllm-project/vllm-ascend/issues/14339) | 310P Qwen3.5 MTP + prefix cache 精度异常 | open，修复 PR 未合并 |
| C2 | [Issue #52413](https://github.com/vllm-project/vllm/issues/52413) | 共享 Triton 缓存目录对齐特化 race → 全 NaN | open，PR #52611 未合并 |
| C3 | [Issue #49449](https://github.com/vllm-project/vllm/issues/49449) | streaming-session rebuild 留 stale hash | open，PR #49619 未合并 |
| C4 | [Issue #44238](https://github.com/vllm-project/vllm/issues/44238) | Mooncake `batch_transfer_sync_write` 并发竞态 | open（每次必现，无合并修复） |
| C5 | [PR #48481](https://github.com/vllm-project/vllm/pull/48481) | PD async scheduling RDMA/zeroing 竞态 | merged（与 C4 同类） |
| C6 | [Issue #47282](https://github.com/vllm-project/vllm/issues/47282) | CPU offload load 路径对称弱点 | open（案例 13 后续缺口） |
| C7 | [Issue #2537](https://github.com/vllm-project/vllm-ascend/issues/2537) | GLM-4.5 DP2+TP8+EP 输出乱码（共享专家 all-reduce 时机冲突） | 上游 [vllm #24849](https://github.com/vllm-project/vllm/pull/24849) 已修复，待 vllm-ascend 同步 |
| C8 | [Issue #7595](https://github.com/vllm-project/vllm-ascend/issues/7595) | Qwen3.5-27B-W8A8 重复输出 | 官方称 0.18.0 已修，无具体 PR |
| C9 | [Issue #6456](https://github.com/vllm-project/vllm-ascend/issues/6456) | FULL 入图精度异常 | closed，未见修复 PR |
| C10 | [Issue #8721](https://github.com/vllm-project/vllm-ascend/issues/8721) | Eagle3 + FULL_DECODE_ONLY 乱码 | 修复落地未确认 |
| C11 | [vllm Issue #37435](https://github.com/vllm-project/vllm/issues/37435) | MTP 草稿丢 `--hf-overrides` 致 YaRN 接受率崩塌 | 修复 PR #37443 open |

> 待对应 PR 合并 / 方案确认后，具备「必现矩阵 + 明确修复」者可提升为正案例。

---

## 5. 怎么用这套资料（使用导航）

- **新手建立整体认知** → 读本文档 + `0_precision.md` 的八大分类；先记住「量化 + silent」是精度问题主线。
- **排查一个具体 KV/PD 精度 bug** → 先对 `cases/README.md` 的触发矩阵（找到 2~3 个条件组合），命中了直接读对应案例的「根因 + 修复 diff + 复现命令」。
- **做特性/回归自测** → 用三梯队复现前置挑最贴近的组合开一条最小复现；尤其优先「并发 / 长压测 / 混合布局 / 第二次调用 / 边界值」这些隐蔽触发面。
- **跟进新热点 / 趋势** → 看 `0_kvcache.md` 增量更新与候选留档 C1–C11（310P MTP、共享缓存竞态、streaming stale hash、GLM-4.5 EP 等）。
- **把某个细节讲给小白** → 每个案例的「核心教训」一句话点透根因，可直接用于讲解。

---

*本总览是纯导航文件，不含原创数据；所有数字与结论均来源于同目录的 `0_precision.md`、`0_kvcache.md` 与 `cases/` 各案例文档。*