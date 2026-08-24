# 精度问题必现案例集（含解决方案）

> 本目录收录 **vllm / vllm-ascend** 仓库中「**必现（有明确触发条件矩阵）** 且 **已有解决方案（merged / closed + 修复落地 / issue 评论验证方案）**」的精度问题案例，共 **33 个**。
>
> 每个案例统一按五个部分整理：**① 问题描述**（现象 / 触发条件矩阵 / 影响）→ **② 版本信息**（vllm + vllm-ascend）→ **③ 定位过程**（无则写「未知」）→ **④ 解决方案**（根因 / 修复 diff / 验证）→ **⑤ 复现方法**（最小复现模型 + 最小服务命令），文末附「核心教训」。
>
> **编号即复现难度**：01 → 32 从易到难排列，并按三个复现梯队分入三个子文件夹：
>
> - [`1_单机单卡/`](./1_单机单卡/) —— 案例 01–13
> - [`2_单机多卡/`](./2_单机多卡/) —— 案例 14–22
> - [`3_多机多卡/`](./3_多机多卡/) —— 案例 23–32（多节点 PD 分离）
>
> 全景梳理见上级目录 [0_precision.md](../0_precision.md) 与 KV Cache 专题 [0_kvcache.md](../0_kvcache.md)，体系总览见 [0_overview.md](../0_overview.md)。

---

## 案例索引

### 梯队 1 · 单机单卡（`1_单机单卡/`，案例 01–13）

| # | 对象 | 一句话定位 | 最小复现模型 | 严重度 |
|---|------|-----------|------------|:---:|
| 01 | [vllm PR #41277](https://github.com/vllm-project/vllm/pull/41277) | dynamic NTK RoPE 公式常数化，超训练长度位置编码全错 | nomic-embed-text-v1.5（137M） | 🔴 高 |
| 02 | [PR #12424](https://github.com/vllm-project/vllm-ascend/pull/12424) | `npu_dequant_swiglu_quant` 小形状 tiling UB 容量漏算 scale buffer | `x.shape=[2,192]`（单算子） | 🟡 中 |
| 03 | [vllm PR #46533](https://github.com/vllm-project/vllm/pull/46533) | rejection sampler 接受占位符 draft token（-1）越界读 | Qwen2.5-1.5B + ngram + grammar | 🟡 中 |
| 04 | [vllm PR #49292](https://github.com/vllm-project/vllm/pull/49292) | Qwen3-VL M-RoPE stride 泄入动态 shape 推导，位置索引越界 | Qwen3-VL-4B-Instruct-FP8 | 🟡 中 |
| 05 | [vllm PR #42143](https://github.com/vllm-project/vllm/pull/42143) | EAGLE3 嵌套 `eagle_config` 的 `norm_before_fc` 被静默跳过 | <10B 目标 + eagle3 草稿 | 🟡 中 |
| 06 | [vllm PR #35157](https://github.com/vllm-project/vllm/pull/35157) | reset prefix cache 后残留 stale `mamba_state_idx` 越界 | LFM2-1.2B（linear/mamba） | 🔴 高 |
| 07 | [PR #9036](https://github.com/vllm-project/vllm-ascend/pull/9036) | ascend 量化路径漏传 MoE `routed_scaling_factor`，路由权重丢 scale | DeepSeek-V2-Lite W8A8 | 🔴 高 |
| 08 | [PR #6958](https://github.com/vllm-project/vllm-ascend/pull/6958) | LoRA × 图模式 dummy run 与真实执行激活数不一致 | Llama-3.2 + LoRA（pytest） | 🟡 中 |
| 09 | [vllm Issue #51094](https://github.com/vllm-project/vllm/issues/51094) | CPU KV 卸载 `mamba_cache_mode="all"` 缺 hit 对齐，chunk 边界静默错 | Nemotron-Nano-9B-v2（Mamba） | 🟡 中 |
| 10 | [PR #12201](https://github.com/vllm-project/vllm-ascend/pull/12201) | W4A4 MXFP4 `weight_scale` floor 分配丢弃尾部块，scale K 68 vs 67 | Qwen3.5-27B vit mxfp4 | 🟡 中 |
| 11 | [PR #11508](https://github.com/vllm-project/vllm-ascend/pull/11508) | PCP + chunked prefill 下 SSM 状态递推 `Φ_i·s0` 重复计算 | Qwen3.5-27B-w8a8-mtp | 🔴 高 |
| 12 | [vllm Issue #49716](https://github.com/vllm-project/vllm/issues/49716) | int8_per_token_head KV 混合 head_dim 下 page 不整除，stride 放错 scale 块 | Gemma-4-31B-IT-AWQ | 🔴 高 |
| 13 | [vllm Issue #45704](https://github.com/vllm-project/vllm/issues/45704) | SimpleCPUOffload store 缺 `wait_stream(compute)`，并发 ~5% 静默乱码 | 混合注意力 + CPU offload | 🟡 中 |

### 梯队 2 · 单机多卡（`2_单机多卡/`，案例 14–22）

| # | 对象 | 一句话定位 | 最小复现模型 | 严重度 |
|---|------|-----------|------------|:---:|
| 14 | [PR #5647](https://github.com/vllm-project/vllm-ascend/pull/5647) | PCP 持久化 buffer 跨调用未重置，第二个请求起读脏 slot mapping | 小模型 + PCP 多卡 | 🔴 高 |
| 15 | [PR #5816](https://github.com/vllm-project/vllm-ascend/pull/5816) | EAGLE3 + SP drafter 走错通信路径，token2 接受率 0.00 | Qwen3-30B-A3B + EAGLE3（2 卡） | 🔴 高 |
| 16 | [PR #14081](https://github.com/vllm-project/vllm-ascend/pull/14081) | async-scheduling + piecewise 图模式 forward 前缺 stream 同步 | Qwen3-30B-A3B 级 MoE | 🔴 高 |
| 17 | [Issue #4273](https://github.com/vllm-project/vllm-ascend/issues/4273) | DP8+TP1 量化启动 SP 但 MoE 未开 EP，数值路径异常（修复为护栏） | Qwen3-30B（DP8，A2 8 卡） | 🟡 中 |
| 18 | [Issue #12723](https://github.com/vllm-project/vllm-ascend/issues/12723) | Triton RoPE 非 2 幂 rotary_dim 用 pad 后维数算 sin 偏移，Q/K 不保范数 | GLM-5.2 DSpark draft（16 dies） | 🔴 高 |
| 19 | [PR #7460](https://github.com/vllm-project/vllm-ascend/pull/7460) | FULL_DECODE_ONLY 图模式 `num_reqs_padded` 错误，Qwen3-Next 精度退化 | Qwen3-Next（8 卡） | 🔴 高 |
| 20 | [PR #11663](https://github.com/vllm-project/vllm-ascend/pull/11663) | A5/950 allgatherEP 下 MXFP4 `topk_weights` 被错误 cast float8 | W4A8-MXFP4 MoE（A5 + EP） | 🔴 高 |
| 21 | [PR #7079](https://github.com/vllm-project/vllm-ascend/pull/7079) | EAGLE3 + CP 切分用错变量（token 总数 vs 请求条数），acceptance 下降 | Qwen3-30B-A3B + EAGLE3 + CP | 🟡 中高 |
| 22 | [PR #14248](https://github.com/vllm-project/vllm-ascend/pull/14248) | DSA dsa_v1 在 dspark 下 decode 被当作 prefill 处理致精度错误 | DeepSeek-V3.2（DSA） | 🔴 高 |

### 梯队 3 · 多机多卡 / PD 分离（`3_多机多卡/`，案例 23–32）

| # | 对象 | 一句话定位 | 最小复现模型 | 严重度 |
|---|------|-----------|------------|:---:|
| 23 | [PR #11886](https://github.com/vllm-project/vllm-ascend/pull/11886) | 序列化 spec 缺全局 KV head 元数据，target/draft 组误合并、pull 映射错位 | Qwen2.5-1.5B-Instruct + draft | 🟡 中 |
| 24 | [PR #12359](https://github.com/vllm-project/vllm-ascend/pull/12359) | Mooncake P2P 单 shard pull 完成即重排 KV，TP 各 rank 不一致 | Qwen2.5-7B（PD + TP2） | 🔴 高 |
| 25 | [Issue #10253](https://github.com/vllm-project/vllm-ascend/issues/10253) | SWA clip 与 prompt-trim 顺序颠倒，stale block 混入传输致全 NaN 输出 | Mistral-7B-v0.1（SWA=4096） | 🔴 极高 |
| 26 | [PR #8540](https://github.com/vllm-project/vllm-ascend/pull/8540) | TP 不等时融合算子「层数」参数少算 MTP 层，MTP 层 KV 未按 TP 重排 | Qwen3.5-27B-w8a8-mtp | 🔴 高 |
| 27 | [PR #11601](https://github.com/vllm-project/vllm-ascend/pull/11601) | 传输元数据缺 cache group id，transfer/cache group 下标错位致混合模型 KV 错配 | Qwen3-Next-80B-A3B（hybrid） | 🟡 中 |
| 28 | [PR #12183](https://github.com/vllm-project/vllm-ascend/pull/12183) | PA 算子直连绕过布局归一化，非连续张量入算子致段错误/数据错位 | DeepSeek-V3.1（PD + Mooncake） | 🔴 高 |
| 29 | [Issue #11127](https://github.com/vllm-project/vllm-ascend/issues/11127) | PD 分离 + MTP 长压测后 `>=` 误判边界、序列被截断为 1，接受率 <1% | GLM-5.1 W8A8（MTP） | 🔴 高 |
| 30 | [PR #9500](https://github.com/vllm-project/vllm-ascend/pull/9500) | `shared_by` 空（hybrid 预留槽）未守卫，注册取首元素崩溃 / 传输错位 | DeepSeek-V4（hybrid） | 🟡 中 |
| 31 | [PR #13195](https://github.com/vllm-project/vllm-ascend/pull/13195) | Ascend PD/PCP/DCP 图模式按 num_tokens 判 PA，PA/FIA 混合层回放出错 | Qwen3.5-397B-A17B | 🔴 高 |
| 32 | [Issue #12339](https://github.com/vllm-project/vllm-ascend/issues/12339) | 超大 MoE FULL_QUANT + EP + PD 下同 input 输出正常/异常交替 | Qwen3.5-397B-W8A8-MXFP8-FULL_QUANT | 🔴 高 |
| 33 | [Issue #12957](https://github.com/vllm-project/vllm-ascend/issues/12957) | PP4 + ascend_direct RDMA 首请求 V cache 跨 stage 未 flush 致乱码 | GLM-5.1 W8A8（PP4 PD 分离） | 🔴 高 |

---

## 案例文件

### `1_单机单卡/`

- [01_issue41277_dynamic_ntk_rope.md](./1_单机单卡/01_issue41277_dynamic_ntk_rope.md)
- [02_pr12424_swiglu_quant.md](./1_单机单卡/02_pr12424_swiglu_quant.md)
- [03_pr46533_rejection_sampler_placeholder.md](./1_单机单卡/03_pr46533_rejection_sampler_placeholder.md)
- [04_pr49292_qwen3vl_mrope_stride.md](./1_单机单卡/04_pr49292_qwen3vl_mrope_stride.md)
- [05_pr42143_eagle3_norm_before_fc.md](./1_单机单卡/05_pr42143_eagle3_norm_before_fc.md)
- [06_pr35157_mamba_state_stale_idx.md](./1_单机单卡/06_pr35157_mamba_state_stale_idx.md)
- [07_pr9036_routed_scaling_factor.md](./1_单机单卡/07_pr9036_routed_scaling_factor.md)
- [08_pr6958_lora_graph_num_active.md](./1_单机单卡/08_pr6958_lora_graph_num_active.md)
- [09_issue51094_mamba_offload_chunk.md](./1_单机单卡/09_issue51094_mamba_offload_chunk.md)
- [10_pr12201_mxfp4_scale.md](./1_单机单卡/10_pr12201_mxfp4_scale.md)
- [11_pr11508_pcp_chunk_ssm_state.md](./1_单机单卡/11_pr11508_pcp_chunk_ssm_state.md)
- [12_issue49716_int8_kv_layout_refactor.md](./1_单机单卡/12_issue49716_int8_kv_layout_refactor.md)
- [13_issue45704_cpu_offload_store_race.md](./1_单机单卡/13_issue45704_cpu_offload_store_race.md)

### `2_单机多卡/`

- [14_pr5647_pcp_slot_mapping_stale.md](./2_单机多卡/14_pr5647_pcp_slot_mapping_stale.md)
- [15_pr5816_eagle3_sp_drafter.md](./2_单机多卡/15_pr5816_eagle3_sp_drafter.md)
- [16_pr14081_async_sched_stream_sync.md](./2_单机多卡/16_pr14081_async_sched_stream_sync.md)
- [17_issue4273_dp8_quant_sp.md](./2_单机多卡/17_issue4273_dp8_quant_sp.md)
- [18_issue12723_triton_rope_rotary.md](./2_单机多卡/18_issue12723_triton_rope_rotary.md)
- [19_pr7460_full_decode_only_padded.md](./2_单机多卡/19_pr7460_full_decode_only_padded.md)
- [20_pr11663_a5_mxfp4_ep.md](./2_单机多卡/20_pr11663_a5_mxfp4_ep.md)
- [21_pr7079_eagle3_cp.md](./2_单机多卡/21_pr7079_eagle3_cp.md)
- [22_pr14248_dsa_dspark.md](./2_单机多卡/22_pr14248_dsa_dspark.md)

### `3_多机多卡/`

- [23_pr11886_total_kv_heads_transfer.md](./3_多机多卡/23_pr11886_total_kv_heads_transfer.md)
- [24_pr12359_pd_reformat_tp_race.md](./3_多机多卡/24_pr12359_pd_reformat_tp_race.md)
- [25_issue10253_swa_stale_nan.md](./3_多机多卡/25_issue10253_swa_stale_nan.md)
- [26_pr8540_tp_unequal_mtp_kv.md](./3_多机多卡/26_pr8540_tp_unequal_mtp_kv.md)
- [27_pr11601_cache_group_ids_metadata.md](./3_多机多卡/27_pr11601_cache_group_ids_metadata.md)
- [28_pr12183_non_contiguous_pa.md](./3_多机多卡/28_pr12183_non_contiguous_pa.md)
- [29_issue11127_mtp_acceptance_drop.md](./3_多机多卡/29_issue11127_mtp_acceptance_drop.md)
- [30_pr9500_shared_by_empty.md](./3_多机多卡/30_pr9500_shared_by_empty.md)
- [31_pr13195_paged_attn_fallback_pd.md](./3_多机多卡/31_pr13195_paged_attn_fallback_pd.md)
- [32_issue12339_qwen_quant.md](./3_多机多卡/32_issue12339_qwen_quant.md)
- [33_issue12957_pp4_rdma_stale_kv.md](./3_多机多卡/33_issue12957_pp4_rdma_stale_kv.md)

---

## 复现前置（按梯队）

编号即难度：**01–13 单机单卡 → 14–22 单节点多卡 → 23–32 多节点 PD 分离**。

### 梯队 1（单机单卡 `1_单机单卡/`，01–13）

- **01**（NTK RoPE）：137M embedding 模型，`LLM.embed()` 与 sentence-transformers 对拍，CPU 亦可。
- **02**（dequant_swiglu_quant 算子）：`pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_dequant_swiglu_quant.py -v`，构造 `x.shape=[2,192]` 与 golden 对拍。
- **03**（rejection sampler）：小模型 + ngram/eagle + grammar + temperature>0；或构造含 -1 的 draft 张量单测。
- **04**（Qwen3-VL M-RoPE）：transformers backend 加载 Qwen3-VL-4B，发多模态请求。
- **05**（EAGLE3 配置）：小目标模型 + 嵌套 `eagle_config` 的自制 eagle3 草稿。
- **06**（mamba 状态索引）：LFM2-1.2B 开 prefix caching，压测中调用 reset。
- **07**（routed_scaling_factor）：DeepSeek-V2-Lite W8A8 单卡，与 HF greedy / gsm8k 对拍。
- **08**（LoRA 图模式）：`pytest -sv tests/e2e/singlecard/test_llama32_lora.py`（仓库自带）。
- **09**（Mamba offload）：上游 `LLM(mamba_cache_mode="all", OffloadingConnector)` + 长度=2 个 offload chunk 的 deterministic prompt，冷启动 vs refetch 对拍。
- **10**（W4A4 MXFP4）：`vllm serve <W4A4-MXFP4-模型> --quantization ascend`，模型 K 必须非 64 的倍数。
- **11**（PCP+chunk SSM）：Qwen3.5 hybrid + MTP + `--enable-chunked-prefill --max-num-batched-tokens 512` 强制多 chunk。
- **12**（int8 KV）：`--kv-cache-dtype int8_per_token_head --attention-backend TRITON_ATTN` + Gemma-4 混合 head_dim，饱和负载后重放确定性 prompt（上游逻辑，NPU 对拍验证）。
- **13**（offload store 竞态）：`race_microtest.py`（`torch.cuda._sleep` 延迟计算流把竞态定化成 100%）。

### 梯队 2（单机多卡 `2_单机多卡/`，14–22）

- **14**（PCP buffer）：多卡开 PCP，同一服务连续发 ≥2 条长 prompt。
- **15**（EAGLE3+SP）：`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`，TP2+EP，读 `vllm:spec_decode_*` acceptance。
- **16**（async+piecewise）：`--async-scheduling` + piecewise 编译配置，跑 math/gsm8k 对比基线。
- **17**（DP8 量化）：A2 单机 8 卡 `--data-parallel-size 8 --tensor-parallel-size 1 --quantization ascend --enable-expert-parallel`。
- **18**（Triton RoPE）：GLM-5.2 + DSpark draft（head_dim=192），`forward_native()` A/B 对照范数比。
- **19**（FULL_DECODE_ONLY）：Qwen3-Next 8 卡 + `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`，与 eager 对比。
- **20**（A5 MXFP4 EP）：需 **A5/950 硬件**，W4A8-MXFP4 MoE + EP，与 TP-only 对比。
- **21**（EAGLE3+CP）：EAGLE3 + CP 长序列高并发压测。
- **22**（DSA dspark）：DeepSeek-V3.2 + dspark，prefill/decode 混合流量。

### 梯队 3（多机多卡 / PD 分离 `3_多机多卡/`，23–32）

1. **环境**：vllm-ascend 镜像 + 已编译 Mooncake（`cmake .. -DUSE_ASCEND_DIRECT=ON`）。AscendDirect RDMA 随机占用端口 `[20000, 20000+npu×1000)`，故 `kv_port` 取 `>= 36000` 避免冲突。
2. **模型越小的策略**：关闭所有与触发无关的特性——量化、`--enable-prefix-caching`、`--enable-chunked-prefill`、cudagraph（用 `--enforce-eager`）、EP、DP>1 等全部关掉；只保留触发必需项（TP 不等 / MTP / SWA / hybrid / P2P）。
3. **最小命令骨架**（P 端 producer + D 端 consumer，差异仅在 `--tensor-parallel-size` / `--kv-transfer-config` / `--speculative-config`）：

```bash
# P 端（prefill，kv_role=kv_producer）
vllm serve <MODEL> \
  --port 8010 \
  --tensor-parallel-size <P_TP> \
  --enforce-eager --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"<CONNECTOR>","kv_role":"kv_producer","kv_port":"36000","kv_connector_extra_config":{"prefill":{"tp_size":<P_TP>},"decode":{"tp_size":<D_TP>}}}'

# D 端（decode，kv_role=kv_consumer）
vllm serve <MODEL> \
  --port 8020 \
  --tensor-parallel-size <D_TP> \
  --enforce-eager --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"<CONNECTOR>","kv_role":"kv_consumer","kv_port":"36100","kv_connector_extra_config":{"prefill":{"tp_size":<P_TP>},"decode":{"tp_size":<D_TP>}}}'
```

> - `kv_connector` 取值：普通 dense 模型用 `MooncakeConnector`；DeepSeek-{V3,R1} layerwise 用 `MooncakeLayerwiseConnector`；DeepSeek-V4 hybrid（MLA+indexer）用 hybrid connector（名以代码注册为准）；32 案例为超大 MoE + EP，使用 `MooncakeConnectorV1`；24 案例 P2P 路径以 `mooncake_connector.py` 注册名为准。
> - `kv_connector_extra_config.prefill/decode.tp_size` 是触发 **TP 不等 / head 复制** 的关键差异项。
> - 环境变量（多节点 RDMA）参考：`HCCL_IF_IP` / `GLOO_SOCKET_IFNAME` / `VLLM_USE_V1=1` / `ASCEND_RT_VISIBLE_DEVICES`，按节点网卡与 NPU 分配设置。
> - 25（SWA）需足够并发才复现；29（MTP 长压测）需 1–2h 压测时长。

---

## 共同脉络（速览）

32 个案例横跨 **上游通用逻辑**、**Ascend 量化/算子**、**特性组合**、**PD 分离 KV 传输** 四大风险面，可归为六条主线：

1. **元数据 / 标量口径不一致**（21、23、26、27、30、05）：`layers`、`total_num_kv_heads`、`cache group id`、`shared_by`、条数 vs token 总数、checkpoint 字段层级——**元数据不自描述、依赖隐式对齐**，是跨层/跨端精度问题的最大雷区。

2. **操作顺序 / 语义分支陷阱**（22、25、28、29、03）：**先 trim 再 clip** 顺序不可交换；「等于上限」≠「超过上限」（`>=` 应改 `>`）；decode 不能当 prefill；采样 kernel 的哨兵值（-1）必须掩蔽——只在特定顺序/边界/分支下爆发，短请求常规 CI 难覆盖。

3. **量化分块 / tiling 的边界与容量**（02、07、10、20）：scale 的 K 维必须统一 ceil（`cdiv`）补块；tiling 新增临时 buffer 要同步更新 **UB 容量预算**；`routed_scaling_factor` 等系数类参数必须显式传递；**系数类张量不能随激活一刀切 cast**。

4. **几何 / stride / 公式推导错误**（01、04、12、18）：非 2 幂 rotary_dim、混合 head_dim 下 page 因内联 scale 不再整除、视图 stride 泄入动态 shape 推导、公式代数化简出常数——凡是「正确性依赖几何整除或推导」的路径，都要用真实 `stride()` 与原始公式自检，孤立单测常 0 error。

5. **状态残留与续算边界**（06、09、11、13、14）：持久化 buffer 跨调用未重置、reset 后状态索引残留、SSM 状态递推初值 s0≠0、offload chunk 边界——**「第二次调用 / 第二个 chunk」是这类 bug 的统一画像**，首轮用例天然掩盖。

6. **异步 / 图模式 / 并行合法性与跨 rank 时序**（08、15、16、17、19、24、31、33）：多流缺 stream 同步只表现精度劣化不报错；图捕获 dummy run 与真实执行必须单一口径；SP 必须配 EP；跨 rank 重排必须等 request 全部传输完成；图模式不能按 num_tokens 假设 attention 类型；跨引擎 KV 消费前必须 flush NPU 计算流——**设备侧 happens-before 边、图分派类型、并行合法组合**是共同要害。

> 触发矩阵是「必现」的关键判据：每个案例都列出了明确的组合条件（多为 2~3 个条件同时成立才触发），据此可做针对性回归与自查。

---

## 候选留档（尚未具备「已修复」条件，暂缓纳入必现集）

以下为检索到的同方向对象，但**当前 open / 修复 PR 未合并 / 方案待完整确认**，暂不满足「已修复」门槛；列为跟踪项：

| # | 对象 | 一句话定位 | 状态 |
|---|------|-----------|------|
| C1 | [Issue #14339](https://github.com/vllm-project/vllm-ascend/issues/14339) | 310P Qwen3.5 MTP + prefix cache 精度异常（async scheduling 压缩重排 batch 与 accepted-token 快照不同步） | open；修复 PR [#14336](https://github.com/vllm-project/vllm-ascend/pull/14336)/[#14342](https://github.com/vllm-project/vllm-ascend/pull/14342) 未合并 |
| C2 | [vllm Issue #52413](https://github.com/vllm-project/vllm/issues/52413) | 多节点共享 Triton 缓存目录对指针做对齐特化，race loser 用错二进制 → 首 prefill token 全 NaN | open；修复 PR [#52611](https://github.com/vllm-project/vllm/pull/52611) 未合并 |
| C3 | [vllm Issue #49449](https://github.com/vllm-project/vllm/issues/49449) | V1 streaming-session rebuild 留 stale prefix-cache hash → 后续请求复用错误 KV | open；修复 PR [#49619](https://github.com/vllm-project/vllm/pull/49619) 未合并 |
| C4 | [vllm Issue #44238](https://github.com/vllm-project/vllm/issues/44238) | MooncakeConnector PD 并发的 `batch_transfer_sync_write` 竞态：dst_hash != 源，大描述符尾全零/小描述符头条零 | open（每次都复现、必现性好，唯有根因无合并修复） |
| C5 | [vllm PR #48481](https://github.com/vllm-project/vllm/pull/48481) | PD async scheduling 竞态：RDMA 完成先于延迟 zeroing kernel，后者覆盖收到的 KV → 精度下降（已修复，主题与 C4 同类） | merged（已修复，可结合 C4 阅读） |
| C6 | [vllm Issue #47282](https://github.com/vllm-project/vllm/issues/47282) | SimpleCPUOffload **load** 路径（#46278 对称弱点）高并发仍可能乱码 | open（案例 13 的后续缺口） |
| C7 | [Issue #2537](https://github.com/vllm-project/vllm-ascend/issues/2537) | GLM-4.5 DP2+TP8+EP 下 allgather `moe_comm_method` 实现错误输出乱码；根因：共享专家 all-reduce 时机静态决定与运行时动态策略冲突 | 正式修复落地**上游** [vllm PR #24849](https://github.com/vllm-project/vllm/pull/24849)（SharedFusedMoE，merged）；vllm-ascend 侧临时 PR #2898 未合并，待随版本同步后可提升 |
| C8 | [Issue #7595](https://github.com/vllm-project/vllm-ascend/issues/7595) | Qwen3.5-27B-W8A8 重复输出 | 官方回复「0.18.0 已修复多个精度问题」，无具体修复 PR 链接 |
| C9 | [Issue #6456](https://github.com/vllm-project/vllm-ascend/issues/6456) | FULL 入图精度异常 | closed，未见明确修复 PR 落地 |
| C10 | [Issue #8721](https://github.com/vllm-project/vllm-ascend/issues/8721) | Eagle3 + FULL_DECODE_ONLY 乱码 | 修复落地未确认 |
| C11 | [vllm Issue #37435](https://github.com/vllm-project/vllm/issues/37435) | MTP 草稿配置丢失 `--hf-overrides` 导致 YaRN 长上下文接受率崩塌 | 修复 PR [#37443](https://github.com/vllm-project/vllm/pull/37443) 仍 open |

> 待上述 PR 合并 / 方案确认后，可将其中具备「必现矩阵 + 明确修复」者提升为正案例。
>
> 另有 [vllm Issue #37591](https://github.com/vllm-project/vllm/issues/37591)（FlashInfer TRTLLM monolithic MoE 全负 router logits 下 GSM8K 0%，修复 [PR #37605](https://github.com/vllm-project/vllm/pull/37605) merged）为 NVIDIA 专属 kernel 路径，**NPU 上不存在该路径**，故不收录；其「融合 MoE 路由 kernel 与 fp32 参考对拍」的教训通用，已体现在案例 07/20。
