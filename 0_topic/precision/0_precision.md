# vLLM & vLLM-Ascend 精度问题全景分析

> 整理时间: 2026-08-18
>
> 数据来源: [vllm-project/vllm](https://github.com/vllm-project/vllm) 及 [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) 的 Issues 与 PRs（基于 GitHub Search API 标题检索）
>
> 范围: 与**数值精度 / 精度下降 / 数据损坏**相关的 issue 与 PR，涵盖量化精度、数值稳定性、输出正确性、算子精度、精度回归等领域。KV Cache 相关的精度问题单独整理于 [0_kvcache.md](0_kvcache.md)
>
> 问题总量: **1915 条** | vllm: 1252 条 | vllm-ascend: 663 条 | Issue: 838 | PR: 1077 | open: 441 | closed: 1474

---

## 快速导航

### 📋 问题分类速览

| 序号 | 类别 | 问题数 | 核心风险 |
|:----:|------|:------:|----------|
| 1 | [量化与低精度 dtype](#1-量化与低精度-dtype) | 568 | 量化/反量化误差、scale 因子丢失、低精度格式布局不匹配 |
| 2 | [数值稳定性](#2-数值稳定性) | 237 | NaN/Inf 传播、溢出/下溢、舍入误差、除零 |
| 3 | [输出正确性与一致性](#3-输出正确性与一致性) | 361 | 输出与参考不一致、静默错误结果、garbage 输出 |
| 4 | [算子与注意力数值精度](#4-算子与注意力数值精度) | 36 | 注意力/Softmax/RoPE/RMSNorm 等算子数值差异 |
| 5 | [KV Cache 精度](#5-KV-Cache-精度) | 122 | 详见 0_kvcache.md（数据损坏、silent corruption） |
| 6 | [精度回归与评测](#6-精度回归与评测) | 400 | 准确率下降、benchmark 退化、perplexity 异常 |
| 7 | [Ascend NPU 特有精度](#7-Ascend-NPU-特有精度) | 20 | CANN 算子差异、ACL Graph 图优化差异、NPU 硬件差异 |
| 8 | [其他 / 泛化精度](#8-其他-/-泛化精度) | 171 | 未归类的精度相关问题 |

---

## 分类框架

```
精度问题
│
├── 1. 量化与低精度 dtype 精度问题
│     ├── 权重量化 (W4A16 / W8A8 / GPTQ / AWQ)
│     ├── 激活量化 (FP8 激活 / per-token / per-tensor)
│     └── 低精度 dtype (FP16/BF16/FP8/FP4/INT8/INT4)
│
├── 2. 数值稳定性问题
│     ├── NaN / Inf 产生与传播
│     ├── 溢出 / 下溢 / 除零
│     └── 舍入误差累积
│
├── 3. 输出正确性与一致性
│     ├── 输出与参考实现不一致 (mismatch)
│     ├── 静默错误输出 (silent corruption / garbage)
│     └── 发散 / 结果漂移 (divergence)
│
├── 4. 算子与注意力数值精度
│     ├── Attention (FlashAttention / PagedAttention / MLA)
│     ├── Softmax / RoPE / RMSNorm / LayerNorm
│     └── 激活函数 / 融合算子
│
├── 5. KV Cache 精度 (详见 0_kvcache.md)
│
├── 6. 精度回归与评测
│     ├── 端到端准确率下降
│     └── benchmark / perplexity 退化
│
├── 7. Ascend NPU 特有精度
│     ├── CANN 算子数值差异
│     └── ACL Graph 图优化差异
│
└── 8. 其他 / 泛化精度问题
```

---

## 1. 量化与低精度 dtype

采用 FP8/FP4/INT8/BF16 等低精度格式存储权重、激活或推理中间结果时，因量化/反量化误差、scale 因子处理、格式布局假设错误导致的精度异常。低精度推理是 vLLM 精度问题的最大单一来源，且常表现为 silent degradation（不报错但质量下降）。

**问题数: 568 条**

### 关键问题

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | PR | [Bugfix][Quantization] Fix OCP MX MoE emulation silently skipping mxfp6 activation QDQ | 🔴 极高 | open | 2026-08-18 | [#52704](https://github.com/vllm-project/vllm/pull/52704) |
| 2 | vllm | Issue | [Bug]: Silent generation stall (Avg generation throughput drops to 0.0, no errors, /health and /v1/chat/completions still return 200) on Qwen3.5-397B-A17B / Qwen3.5-397B-A17B-FP8 / Qwen3.5-122B-A10B across v0.18.0, v0.19.0, v0.25.1 | 🔴 极高 | open | 2026-08-14 | [#52319](https://github.com/vllm-project/vllm/issues/52319) |
| 3 | vllm | Issue | [Bug]: NVFP4 MoE monolithic backend (trtllm_fp4_block_scale_moe) produces semantically corrupt output for non-gated ReLU² MoE (Nemotron-3-Nano) on SM100; flashinfer_cutlass is correct on the same GPU | 🔴 极高 | open | 2026-08-14 | [#52308](https://github.com/vllm-project/vllm/issues/52308) |
| 4 | vllm | Issue | [Bug][ROCm/gfx942]: GLM-5.2-FP8 — first request after GPU idle emits garbage; piecewise CUDA graph cold replay corrupts the request's own prefill (workaround: cudagraph_mode=FULL_DECODE_ONLY) | 🔴 极高 | open | 2026-08-13 | [#52150](https://github.com/vllm-project/vllm/issues/52150) |
| 5 | vllm | Issue | [Bug]: Gemma-4 NVFP4 (fp8 KV) on Blackwell: FLASH_ATTN is selected, silently falls back to FA2, then fails with "FlashAttention only support fp16 and bf16 data type" | 🔴 极高 | open | 2026-08-11 | [#51828](https://github.com/vllm-project/vllm/issues/51828) |
| 6 | vllm | Issue | [Bug]: online FP8 (--quantization fp8) produces corrupted, non-EOS-terminating output on Qwen2.5-1.5B-Instruct | 🔴 极高 | open | 2026-08-07 | [#51456](https://github.com/vllm-project/vllm/issues/51456) |
| 7 | vllm | Issue | [Bug]: int8_per_token_head KV with prefix caching DISABLED corrupts the FIRST generated tokens (Gemma-4 hybrid, Triton, idle KV pool) | 🔴 极高 | open | 2026-08-02 | [#50749](https://github.com/vllm-project/vllm/issues/50749) |
| 8 | vllm | Issue | [Bug]: int8_per_token_head KV + prefix caching corrupts output when the KV pool is pinned at 100% (Gemma-4 hybrid, Triton) | 🔴 极高 | open | 2026-08-01 | [#50702](https://github.com/vllm-project/vllm/issues/50702) |
| 9 | vllm | PR | [ROCm][MoE] Fix Kimi-K3 a8w4 MoE decode garbage by pinning fp8 stage-1 kernel | 🔴 极高 | closed | 2026-07-31 | [#50579](https://github.com/vllm-project/vllm/pull/50579) |
| 10 | vllm | Issue | [Bug]: Nemotron 3 Nano NVFP4 plain TP8 fails on Hopper/Marlin because TP shards split 16-value quantization groups | 🔴 极高 | open | 2026-07-27 | [#49949](https://github.com/vllm-project/vllm/issues/49949) |
| 11 | vllm-ascend | Issue | [Bug]: Ascend-dispatched rotary path (Triton rope) corrupts Q/K norms for GLM-5.2 DSpark draft (bf16, NeoX, head_dim=192); forward_native restores accept_len 3.06 -&gt; 5.55 | 🔴 极高 | open | 2026-07-23 | [#12723](https://github.com/vllm-project/vllm-ascend/issues/12723) |
| 12 | vllm | Issue | [Bug]: VLLM_MARLIN_INPUT_DTYPE=fp8 (Marlin W4A8-FP8) silently corrupts output on GB10/sm_121a — WNA16 INT4 MoE emits repeated &lt;/think&gt; loop at temp 0 (kernel runs ~2.5% faster) | 🔴 极高 | open | 2026-07-23 | [#49546](https://github.com/vllm-project/vllm/issues/49546) |
| 13 | vllm | Issue | [Bug]: MiniMax-M3 NVFP4 produces garbage output + CUDA illegal memory access on Hopper (sm90) via Marlin FP4-MoE | 🔴 极高 | open | 2026-07-19 | [#49070](https://github.com/vllm-project/vllm/issues/49070) |
| 14 | vllm | Issue | [Bug]: XQA decode under FULL cudagraph capture silently corrupts attention output (fp8 and nvfp4 KV) | 🔴 极高 | open | 2026-07-18 | [#49010](https://github.com/vllm-project/vllm/issues/49010) |
| 15 | vllm | Issue | [Bug]: Marlin W4A8 (int8 activations) corrupts on checkpoints with negative group scales | 🔴 极高 | open | 2026-07-17 | [#48905](https://github.com/vllm-project/vllm/issues/48905) |
| 16 | vllm | Issue | [Bug]: VLLM_MARLIN_INPUT_DTYPE=int8 silently ignored for auto-round (INC) checkpoints | 🔴 极高 | open | 2026-07-17 | [#48904](https://github.com/vllm-project/vllm/issues/48904) |
| 17 | vllm | Issue | [Bug]: #47327 dense-MHA split breaks FlashMLA sparse: OOB write in top-k index conversion, corrupted fp8_ds_mla context gather | 🔴 极高 | closed | 2026-07-14 | [#48611](https://github.com/vllm-project/vllm/issues/48611) |
| 18 | vllm | Issue | [Bug]: Missing quant_config in MTP eh_proj causes severe silent precision loss during W8A8 quantization | 🔴 极高 | open | 2026-07-13 | [#48492](https://github.com/vllm-project/vllm/issues/48492) |
| 19 | vllm | Issue | [Bug][XPU] compressed-tensors FP8 W8A8 (dynamic) generates garbage output on Intel Arc Pro B70 (Battlemage) | 🔴 极高 | open | 2026-07-09 | [#48058](https://github.com/vllm-project/vllm/issues/48058) |
| 20 | vllm | PR | [Bugfix][Hardware] DeepSeek-V4 o_proj fp8 einsum NaNs on SM12x: use SM90-style raw f32 block scales | 🔴 极高 | closed | 2026-07-08 | [#48052](https://github.com/vllm-project/vllm/pull/48052) |
| 21 | vllm | Issue | [Bug]: disable_flashinfer_q_quantization + fp8 KV + spec decode silently corrupts sliding-window models on SM100 (trtllm-gen); BF16-Q paths also mis-apply q/k/v scales | 🔴 极高 | open | 2026-07-07 | [#47847](https://github.com/vllm-project/vllm/issues/47847) |
| 22 | vllm | Issue | [Bug]: GLM-5.2 (GlmMoeDsa) on GB300: output corruption when sequence crosses ~4096 tokens (DSA indexer paged-logits, fp8_fp4_paged_mqa_logits) | 🔴 极高 | closed | 2026-07-07 | [#47827](https://github.com/vllm-project/vllm/issues/47827) |
| 23 | vllm | PR | [Bugfix] [EPLB]  Avoid Silent Accuracy Corruption in Quark MXFP4 EPLB | 🔴 极高 | open | 2026-07-04 | [#47588](https://github.com/vllm-project/vllm/pull/47588) |
| 24 | vllm | Issue | [Bug]: Triton block quantized (e.g. MXFP4) MoE kernels producing NaNs due to OOB reads on scale values | 🔴 极高 | closed | 2026-07-01 | [#47303](https://github.com/vllm-project/vllm/issues/47303) |
| 25 | vllm | Issue | [Bug]: INCConfig doesn't support hybrid INT4+FP8 AutoRound checkpoints — weight_scale_inv params not loaded, model produces garbage output | 🔴 极高 | open | 2026-06-21 | [#46311](https://github.com/vllm-project/vllm/issues/46311) |
| 26 | vllm | PR | Revert "[Bugfix] Fix corrupt outputs in MoE FP8 LoRA responses and MoE base model responses when LoRAs are loaded" (#42120) | 🔴 极高 | open | 2026-06-19 | [#46151](https://github.com/vllm-project/vllm/pull/46151) |
| 27 | vllm | Issue | [Bug]:  DeepSeek-V4-Pro (fp4+fp8 mixed) outputs garbage tokens | 🔴 极高 | closed | 2026-05-11 | [#42265](https://github.com/vllm-project/vllm/issues/42265) |
| 28 | vllm | Issue | [Bug]:  FLASHINFER_CUTLASS_MXFP4_MXFP8 produces wrong output under expert parallelism | 🔴 极高 | open | 2026-05-08 | [#42118](https://github.com/vllm-project/vllm/issues/42118) |
| 29 | vllm | PR | [Bugfix][Quark] Fix W8A8 INT8 garbage outputs on Step-3.5-Flash (and other 3-key fused-MoE Quark exports) | 🔴 极高 | closed | 2026-05-07 | [#41892](https://github.com/vllm-project/vllm/pull/41892) |
| 30 | vllm | Issue | [Bug]: Qwen3-Next NVFP4 quants silently produce garbage when linear_attn weights are missing from quantization_config.ignore | 🔴 极高 | closed | 2026-04-18 | [#40252](https://github.com/vllm-project/vllm/issues/40252) |
| 31 | vllm | PR | [Bugfix]Fix issue #38936 NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 offline execution | 🔴 极高 | closed | 2026-04-11 | [#39561](https://github.com/vllm-project/vllm/pull/39561) |
| 32 | vllm | PR | meme (for avoidance of any doubt): how to get promo at AMD according to internal KPI dashboard: ROCm AMD HIP MI200 MI210 MI250 MI300 MI300X MI308 MI325X MI350 MI355X gfx90a gfx940 gfx941 gfx942 gfx1100 gfx1101 gfx1200 gfx1201 hipblaslt hipblas rocblas triton-rocm AITER Meta Facebook AWS Amazon Anthropic SemiAnalysis Microsoft Google Databricks Anyscale Oracle IBM Intel Samsung Hugging Face regression crash broken data corruption hang deadlock OOM segfault SIGABRT urgent critical hotfix failing attention flash_attn paged_attention quantization model_executor model_runner serving scheduler tensor_parallel pipeline_parallel distributed cuda_graph performance benchmark | 🔴 极高 | closed | 2026-04-04 | [#38963](https://github.com/vllm-project/vllm/pull/38963) |
| 33 | vllm | Issue | [Bug]: NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 offline execution fails | 🔴 极高 | closed | 2026-04-03 | [#38936](https://github.com/vllm-project/vllm/issues/38936) |
| 34 | vllm | Issue | [Bug] W4A8-INT compressed_tensors silently runs W4A16 — activations never quantized to int8 | 🔴 极高 | closed | 2026-03-25 | [#38064](https://github.com/vllm-project/vllm/issues/38064) |
| 35 | vllm | Issue | [Bug]: NaNs in vLLM using DeepSeek-R1-0528-NVFP4-v2 | 🔴 极高 | closed | 2026-03-23 | [#37890](https://github.com/vllm-project/vllm/issues/37890) |
| 36 | vllm | PR | [BUGFIX] Fix accuracy regression for NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 with TP&gt;1 | 🔴 极高 | closed | 2026-02-13 | [#34476](https://github.com/vllm-project/vllm/pull/34476) |
| 37 | vllm | Issue | [Bug]: Repeated, wrong results of FP8-Dynamic Llava-OneVision regardless input images | 🔴 极高 | closed | 2025-12-07 | [#30196](https://github.com/vllm-project/vllm/issues/30196) |
| 38 | vllm | Issue | [Bug]: wrong output on L20 using fp8 | 🔴 极高 | closed | 2025-06-17 | [#19779](https://github.com/vllm-project/vllm/issues/19779) |
| 39 | vllm | Issue | [Bug]: CompressedTensorsWNA16MoEMethod rejects grouped int8 MoE checkpoints (regression) | 🔴 高 | open | 2026-08-18 | [#52714](https://github.com/vllm-project/vllm/issues/52714) |
| 40 | vllm | PR | [Bugfix][Quantization] Fix apply_vllm_mapper crash on dict-valued Quark algo_config | 🔴 高 | open | 2026-08-17 | [#52642](https://github.com/vllm-project/vllm/pull/52642) |

### 关键规律与分析

1. **低精度推理是 vLLM 精度问题第一大来源**（568 条），且高频表现为 silent corruption（不报错但输出 garbage）。典型路径：FP8 动态量化 / W8A8 / Marlin W4A8-FP8、NVFP4 MoE（trtllm_fp4_block_scale_moe）、MXFP4 block 量化等。
2. **MoE 与低精度是危险组合**：Nemotron-3-Nano、MiniMax-M3、DeepSeek-V4-Pro 等 MoE 模型在 FP8/FP4 下产生 garbage/NaN 的案例最多，根因多为 expert 路由与量化 group 边界冲突、scale 因子读取越界（OOB read scale）。
3. **scale 因子是核心脆弱点**：per-token/per-head/per-group scale 的丢失、错位、继承错误反复出现（TurboQuant scale 传播、MTP eh_proj 缺 quant_config、负 group scale）。
4. **低精度修复回归风险极高**：多次出现"修复后被 revert"（如 MoE FP8 LoRA #42120→#46151、packed HND reshape #47314→#48344），说明低精度 kernel 修复牵一发动全身。

<details>
<summary>展开全部 568 条</summary>

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | PR | [Bugfix][Quantization] Fix OCP MX MoE emulation silently skipping mxfp6 activation QDQ | 🔴 极高 | open | 2026-08-18 | [#52704](https://github.com/vllm-project/vllm/pull/52704) |
| 2 | vllm | Issue | [Bug]: Silent generation stall (Avg generation throughput drops to 0.0, no errors, /health and /v1/chat/completions still return 200) on Qwen3.5-397B-A17B / Qwen3.5-397B-A17B-FP8 / Qwen3.5-122B-A10B across v0.18.0, v0.19.0, v0.25.1 | 🔴 极高 | open | 2026-08-14 | [#52319](https://github.com/vllm-project/vllm/issues/52319) |
| 3 | vllm | Issue | [Bug]: NVFP4 MoE monolithic backend (trtllm_fp4_block_scale_moe) produces semantically corrupt output for non-gated ReLU² MoE (Nemotron-3-Nano) on SM100; flashinfer_cutlass is correct on the same GPU | 🔴 极高 | open | 2026-08-14 | [#52308](https://github.com/vllm-project/vllm/issues/52308) |
| 4 | vllm | Issue | [Bug][ROCm/gfx942]: GLM-5.2-FP8 — first request after GPU idle emits garbage; piecewise CUDA graph cold replay corrupts the request's own prefill (workaround: cudagraph_mode=FULL_DECODE_ONLY) | 🔴 极高 | open | 2026-08-13 | [#52150](https://github.com/vllm-project/vllm/issues/52150) |
| 5 | vllm | Issue | [Bug]: Gemma-4 NVFP4 (fp8 KV) on Blackwell: FLASH_ATTN is selected, silently falls back to FA2, then fails with "FlashAttention only support fp16 and bf16 data type" | 🔴 极高 | open | 2026-08-11 | [#51828](https://github.com/vllm-project/vllm/issues/51828) |
| 6 | vllm | Issue | [Bug]: online FP8 (--quantization fp8) produces corrupted, non-EOS-terminating output on Qwen2.5-1.5B-Instruct | 🔴 极高 | open | 2026-08-07 | [#51456](https://github.com/vllm-project/vllm/issues/51456) |
| 7 | vllm | Issue | [Bug]: int8_per_token_head KV with prefix caching DISABLED corrupts the FIRST generated tokens (Gemma-4 hybrid, Triton, idle KV pool) | 🔴 极高 | open | 2026-08-02 | [#50749](https://github.com/vllm-project/vllm/issues/50749) |
| 8 | vllm | Issue | [Bug]: int8_per_token_head KV + prefix caching corrupts output when the KV pool is pinned at 100% (Gemma-4 hybrid, Triton) | 🔴 极高 | open | 2026-08-01 | [#50702](https://github.com/vllm-project/vllm/issues/50702) |
| 9 | vllm | PR | [ROCm][MoE] Fix Kimi-K3 a8w4 MoE decode garbage by pinning fp8 stage-1 kernel | 🔴 极高 | closed | 2026-07-31 | [#50579](https://github.com/vllm-project/vllm/pull/50579) |
| 10 | vllm | Issue | [Bug]: Nemotron 3 Nano NVFP4 plain TP8 fails on Hopper/Marlin because TP shards split 16-value quantization groups | 🔴 极高 | open | 2026-07-27 | [#49949](https://github.com/vllm-project/vllm/issues/49949) |
| 11 | vllm-ascend | Issue | [Bug]: Ascend-dispatched rotary path (Triton rope) corrupts Q/K norms for GLM-5.2 DSpark draft (bf16, NeoX, head_dim=192); forward_native restores accept_len 3.06 -&gt; 5.55 | 🔴 极高 | open | 2026-07-23 | [#12723](https://github.com/vllm-project/vllm-ascend/issues/12723) |
| 12 | vllm | Issue | [Bug]: VLLM_MARLIN_INPUT_DTYPE=fp8 (Marlin W4A8-FP8) silently corrupts output on GB10/sm_121a — WNA16 INT4 MoE emits repeated &lt;/think&gt; loop at temp 0 (kernel runs ~2.5% faster) | 🔴 极高 | open | 2026-07-23 | [#49546](https://github.com/vllm-project/vllm/issues/49546) |
| 13 | vllm | Issue | [Bug]: MiniMax-M3 NVFP4 produces garbage output + CUDA illegal memory access on Hopper (sm90) via Marlin FP4-MoE | 🔴 极高 | open | 2026-07-19 | [#49070](https://github.com/vllm-project/vllm/issues/49070) |
| 14 | vllm | Issue | [Bug]: XQA decode under FULL cudagraph capture silently corrupts attention output (fp8 and nvfp4 KV) | 🔴 极高 | open | 2026-07-18 | [#49010](https://github.com/vllm-project/vllm/issues/49010) |
| 15 | vllm | Issue | [Bug]: Marlin W4A8 (int8 activations) corrupts on checkpoints with negative group scales | 🔴 极高 | open | 2026-07-17 | [#48905](https://github.com/vllm-project/vllm/issues/48905) |
| 16 | vllm | Issue | [Bug]: VLLM_MARLIN_INPUT_DTYPE=int8 silently ignored for auto-round (INC) checkpoints | 🔴 极高 | open | 2026-07-17 | [#48904](https://github.com/vllm-project/vllm/issues/48904) |
| 17 | vllm | Issue | [Bug]: #47327 dense-MHA split breaks FlashMLA sparse: OOB write in top-k index conversion, corrupted fp8_ds_mla context gather | 🔴 极高 | closed | 2026-07-14 | [#48611](https://github.com/vllm-project/vllm/issues/48611) |
| 18 | vllm | Issue | [Bug]: Missing quant_config in MTP eh_proj causes severe silent precision loss during W8A8 quantization | 🔴 极高 | open | 2026-07-13 | [#48492](https://github.com/vllm-project/vllm/issues/48492) |
| 19 | vllm | Issue | [Bug][XPU] compressed-tensors FP8 W8A8 (dynamic) generates garbage output on Intel Arc Pro B70 (Battlemage) | 🔴 极高 | open | 2026-07-09 | [#48058](https://github.com/vllm-project/vllm/issues/48058) |
| 20 | vllm | PR | [Bugfix][Hardware] DeepSeek-V4 o_proj fp8 einsum NaNs on SM12x: use SM90-style raw f32 block scales | 🔴 极高 | closed | 2026-07-08 | [#48052](https://github.com/vllm-project/vllm/pull/48052) |
| 21 | vllm | Issue | [Bug]: disable_flashinfer_q_quantization + fp8 KV + spec decode silently corrupts sliding-window models on SM100 (trtllm-gen); BF16-Q paths also mis-apply q/k/v scales | 🔴 极高 | open | 2026-07-07 | [#47847](https://github.com/vllm-project/vllm/issues/47847) |
| 22 | vllm | Issue | [Bug]: GLM-5.2 (GlmMoeDsa) on GB300: output corruption when sequence crosses ~4096 tokens (DSA indexer paged-logits, fp8_fp4_paged_mqa_logits) | 🔴 极高 | closed | 2026-07-07 | [#47827](https://github.com/vllm-project/vllm/issues/47827) |
| 23 | vllm | PR | [Bugfix] [EPLB]  Avoid Silent Accuracy Corruption in Quark MXFP4 EPLB | 🔴 极高 | open | 2026-07-04 | [#47588](https://github.com/vllm-project/vllm/pull/47588) |
| 24 | vllm | Issue | [Bug]: Triton block quantized (e.g. MXFP4) MoE kernels producing NaNs due to OOB reads on scale values | 🔴 极高 | closed | 2026-07-01 | [#47303](https://github.com/vllm-project/vllm/issues/47303) |
| 25 | vllm | Issue | [Bug]: INCConfig doesn't support hybrid INT4+FP8 AutoRound checkpoints — weight_scale_inv params not loaded, model produces garbage output | 🔴 极高 | open | 2026-06-21 | [#46311](https://github.com/vllm-project/vllm/issues/46311) |
| 26 | vllm | PR | Revert "[Bugfix] Fix corrupt outputs in MoE FP8 LoRA responses and MoE base model responses when LoRAs are loaded" (#42120) | 🔴 极高 | open | 2026-06-19 | [#46151](https://github.com/vllm-project/vllm/pull/46151) |
| 27 | vllm | Issue | [Bug]:  DeepSeek-V4-Pro (fp4+fp8 mixed) outputs garbage tokens | 🔴 极高 | closed | 2026-05-11 | [#42265](https://github.com/vllm-project/vllm/issues/42265) |
| 28 | vllm | Issue | [Bug]:  FLASHINFER_CUTLASS_MXFP4_MXFP8 produces wrong output under expert parallelism | 🔴 极高 | open | 2026-05-08 | [#42118](https://github.com/vllm-project/vllm/issues/42118) |
| 29 | vllm | PR | [Bugfix][Quark] Fix W8A8 INT8 garbage outputs on Step-3.5-Flash (and other 3-key fused-MoE Quark exports) | 🔴 极高 | closed | 2026-05-07 | [#41892](https://github.com/vllm-project/vllm/pull/41892) |
| 30 | vllm | Issue | [Bug]: Qwen3-Next NVFP4 quants silently produce garbage when linear_attn weights are missing from quantization_config.ignore | 🔴 极高 | closed | 2026-04-18 | [#40252](https://github.com/vllm-project/vllm/issues/40252) |
| 31 | vllm | PR | [Bugfix]Fix issue #38936 NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 offline execution | 🔴 极高 | closed | 2026-04-11 | [#39561](https://github.com/vllm-project/vllm/pull/39561) |
| 32 | vllm | PR | meme (for avoidance of any doubt): how to get promo at AMD according to internal KPI dashboard: ROCm AMD HIP MI200 MI210 MI250 MI300 MI300X MI308 MI325X MI350 MI355X gfx90a gfx940 gfx941 gfx942 gfx1100 gfx1101 gfx1200 gfx1201 hipblaslt hipblas rocblas triton-rocm AITER Meta Facebook AWS Amazon Anthropic SemiAnalysis Microsoft Google Databricks Anyscale Oracle IBM Intel Samsung Hugging Face regression crash broken data corruption hang deadlock OOM segfault SIGABRT urgent critical hotfix failing attention flash_attn paged_attention quantization model_executor model_runner serving scheduler tensor_parallel pipeline_parallel distributed cuda_graph performance benchmark | 🔴 极高 | closed | 2026-04-04 | [#38963](https://github.com/vllm-project/vllm/pull/38963) |
| 33 | vllm | Issue | [Bug]: NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 offline execution fails | 🔴 极高 | closed | 2026-04-03 | [#38936](https://github.com/vllm-project/vllm/issues/38936) |
| 34 | vllm | Issue | [Bug] W4A8-INT compressed_tensors silently runs W4A16 — activations never quantized to int8 | 🔴 极高 | closed | 2026-03-25 | [#38064](https://github.com/vllm-project/vllm/issues/38064) |
| 35 | vllm | Issue | [Bug]: NaNs in vLLM using DeepSeek-R1-0528-NVFP4-v2 | 🔴 极高 | closed | 2026-03-23 | [#37890](https://github.com/vllm-project/vllm/issues/37890) |
| 36 | vllm | PR | [BUGFIX] Fix accuracy regression for NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 with TP&gt;1 | 🔴 极高 | closed | 2026-02-13 | [#34476](https://github.com/vllm-project/vllm/pull/34476) |
| 37 | vllm | Issue | [Bug]: Repeated, wrong results of FP8-Dynamic Llava-OneVision regardless input images | 🔴 极高 | closed | 2025-12-07 | [#30196](https://github.com/vllm-project/vllm/issues/30196) |
| 38 | vllm | Issue | [Bug]: wrong output on L20 using fp8 | 🔴 极高 | closed | 2025-06-17 | [#19779](https://github.com/vllm-project/vllm/issues/19779) |
| 39 | vllm | Issue | [Bug]: CompressedTensorsWNA16MoEMethod rejects grouped int8 MoE checkpoints (regression) | 🔴 高 | open | 2026-08-18 | [#52714](https://github.com/vllm-project/vllm/issues/52714) |
| 40 | vllm | PR | [Bugfix][Quantization] Fix apply_vllm_mapper crash on dict-valued Quark algo_config | 🔴 高 | open | 2026-08-17 | [#52642](https://github.com/vllm-project/vllm/pull/52642) |
| 41 | vllm | PR | [Bugfix] Fix NVFP4 device mismatch in break_fp4_bytes under VLLM_BATCH_INVARIANT | 🔴 高 | open | 2026-08-15 | [#52432](https://github.com/vllm-project/vllm/pull/52432) |
| 42 | vllm | Issue | [Bug]: VLLM_BATCH_INVARIANT=1 crashes NVFP4 models: emulation break_fp4_bytes device mismatch (lookup table on CPU, indices on GPU) | 🔴 高 | open | 2026-08-15 | [#52407](https://github.com/vllm-project/vllm/issues/52407) |
| 43 | vllm | Issue | [Bug][ROCm]: BF16 MLA with ROCM_AITER_FA broken (gfx950) (fmha opus) | 🔴 高 | open | 2026-08-14 | [#52312](https://github.com/vllm-project/vllm/issues/52312) |
| 44 | vllm | PR | [CI][AMD][Disagg] Fix Kimi K2.5/K2.6 MXFP4 MLA backends on ROCm nightly for accuracy eval (disable aiter OPUS FMHA) | 🔴 高 | open | 2026-08-12 | [#51976](https://github.com/vllm-project/vllm/pull/51976) |
| 45 | vllm-ascend | Issue | [Bug]: 0.23.0 8月10日日构建版本 + glm-5.2 w4a8c8模型，开启enable_sparse_sfa_c8有精度问题（简单curl一条就乱码），注w8a8c8量化权重没问题 | 🔴 高 | open | 2026-08-11 | [#13983](https://github.com/vllm-project/vllm-ascend/issues/13983) |
| 46 | vllm | PR | [Bugfix] vLLM crashes at startup when DeepEP v2 is used with `--enforce-eager` wiht TRTLLM Bf16 | 🔴 高 | open | 2026-08-11 | [#51824](https://github.com/vllm-project/vllm/pull/51824) |
| 47 | vllm | Issue | [Bug]: Massive output-token throughput regression since v0.24.0 with Qwen3.6-35B-A3B-FP8 | 🔴 高 | open | 2026-08-10 | [#51663](https://github.com/vllm-project/vllm/issues/51663) |
| 48 | vllm-ascend | Issue | [Bug]: BF16 model crashes with enable_fused_mc2=1 under CANN 9.1 (regression from #11701) | 🔴 高 | open | 2026-08-10 | [#13924](https://github.com/vllm-project/vllm-ascend/issues/13924) |
| 49 | vllm-ascend | Issue | [Bug]: BF16 model crashes with enable_fused_mc2=1 under CANN 9.1 (regression from #11701) | 🔴 高 | closed | 2026-08-10 | [#13923](https://github.com/vllm-project/vllm-ascend/issues/13923) |
| 50 | vllm | PR | [Bugfix][Quantization] Fix INT8 W8A8 MoE crash in TritonExperts | 🔴 高 | closed | 2026-08-07 | [#51411](https://github.com/vllm-project/vllm/pull/51411) |
| 51 | vllm | Issue | [Bug]: BF16 MoE + LoRA startup crash on non-gated models (TrtLlmBf16LoRAExperts weight-shape assert) | 🔴 高 | closed | 2026-08-04 | [#51001](https://github.com/vllm-project/vllm/issues/51001) |
| 52 | vllm | PR | [Bugfix] Fix NVFP4 shape mismatch in flashinfer_scaled_fp4_mm #50557 | 🔴 高 | open | 2026-07-31 | [#50610](https://github.com/vllm-project/vllm/pull/50610) |
| 53 | vllm | PR | [Bugfix] Fix NVFP4 shape mismatch in flashinfer_scaled_fp4_mm #50557 | 🔴 高 | closed | 2026-07-31 | [#50609](https://github.com/vllm-project/vllm/pull/50609) |
| 54 | vllm | Issue | [Bug]: AxionML/Gemma-4-12B-NVFP4 fails with shape mismatch in flashinfer_scaled_fp4_mm on RTX 5060 Ti (sm_120) | 🔴 高 | open | 2026-07-31 | [#50577](https://github.com/vllm-project/vllm/issues/50577) |
| 55 | vllm | Issue | GSM8K accuracy regression: nightly (0.26.1rc1.dev77) drops ~5% vs latest (0.26.0) on GLM-5.2-FP8 P/D | 🔴 高 | closed | 2026-07-30 | [#50435](https://github.com/vllm-project/vllm/issues/50435) |
| 56 | vllm | Issue | [Bug]: DeepGemm accuracy auto-disable (_DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES) does not apply to the FP8 MoE path | 🔴 高 | open | 2026-07-29 | [#50332](https://github.com/vllm-project/vllm/issues/50332) |
| 57 | vllm | PR | LUT-B quantization accuracy prototype | 🔴 高 | closed | 2026-07-28 | [#50168](https://github.com/vllm-project/vllm/pull/50168) |
| 58 | vllm | Issue | [Bug]: GLM-5.2-NVFP4 produces garbled/incorrect output and hits NotImplementedError in forward_mha on GB10 (SM121a) with FLASHINFER_MLA_SPARSE_SM120 | 🔴 高 | open | 2026-07-26 | [#49886](https://github.com/vllm-project/vllm/issues/49886) |
| 59 | vllm | Issue | KVBlockZeroer crashes with non-uniform page sizes on GLM-5.2 FP8 | 🔴 高 | closed | 2026-07-24 | [#49696](https://github.com/vllm-project/vllm/issues/49696) |
| 60 | vllm | Issue | [Bug]: v0.25.0 regression - Qwen3.5 FP8 on H200 crashes during CUDA graph capture with CUDA illegal memory access | 🔴 高 | open | 2026-07-21 | [#49311](https://github.com/vllm-project/vllm/issues/49311) |
| 61 | vllm | Issue | [Bug]: Kimi-K2.5 (compressed-tensors/int4) crashes with `ValueError: Mismatched mO.strides[0]` in FA4 CuTe MLA prefill context-chunk on Blackwell (long context) | 🔴 高 | closed | 2026-07-20 | [#49200](https://github.com/vllm-project/vllm/issues/49200) |
| 62 | vllm | Issue | [Bug]: Fused_moe dimension mismatch for Qwen mxfp4 model on ROCM | 🔴 高 | closed | 2026-07-20 | [#49141](https://github.com/vllm-project/vllm/issues/49141) |
| 63 | vllm | Issue | [Bug]:  --linear-backend=flashinfer_b12x crashes on Qwen3.6-35B-A3B-NVFP4 — no kernel for FP8-quantized GDN QKVZ layer | 🔴 高 | open | 2026-07-19 | [#49076](https://github.com/vllm-project/vllm/issues/49076) |
| 64 | vllm-ascend | Issue | [Bug]: Ascend950 Qwen3.5-397B-W8A8-MXFP8-FULL_QUANT在PD分离场景下，不开mtp，存在精度问题 | 🔴 高 | closed | 2026-07-18 | [#12339](https://github.com/vllm-project/vllm-ascend/issues/12339) |
| 65 | vllm-ascend | Issue | [Bug]: GLM-5.2 bf16精度 开启MTP接纳率为0 | 🔴 高 | open | 2026-07-17 | [#12275](https://github.com/vllm-project/vllm-ascend/issues/12275) |
| 66 | vllm-ascend | PR | [Bugfix] Fix W4A4 MXFP4 weight scale K-dim mismatch | 🔴 高 | closed | 2026-07-16 | [#12201](https://github.com/vllm-project/vllm-ascend/pull/12201) |
| 67 | vllm | Issue | LoRA Triton kernel crashes with NVFP4 (modelopt_fp4) quantization - CUDA illegal memory access | 🔴 高 | open | 2026-07-16 | [#48862](https://github.com/vllm-project/vllm/issues/48862) |
| 68 | vllm | PR | [ROCm][CI] Fix AITER MLA fp8 decode metadata regression test | 🔴 高 | closed | 2026-07-16 | [#48845](https://github.com/vllm-project/vllm/pull/48845) |
| 69 | vllm-ascend | Issue | [Bug]: qwen3.5-35B-A3B，bf16精度在910B4上四卡部署，开启mtp以后tpot变差一倍左右。抓profiling后发现是host bound造成了快慢卡 | 🔴 高 | open | 2026-07-14 | [#11967](https://github.com/vllm-project/vllm-ascend/issues/11967) |
| 70 | vllm | PR | [Bugfix][NVFP4/FP8 MoE] Fix gated MoE crash on unaligned intermediate | 🔴 高 | open | 2026-07-14 | [#48624](https://github.com/vllm-project/vllm/pull/48624) |
| 71 | vllm | Issue | [Bug]: Qwen3.5-122B-A10B-FP8 serve crashes on nightly/0.25 (CUBLAS_STATUS_EXECUTION_FAILED at profile_run); no version serves FP8 hybrid GDN + DFlash together | 🔴 高 | open | 2026-07-13 | [#48477](https://github.com/vllm-project/vllm/issues/48477) |
| 72 | vllm | PR | [Bugfix] Fix FlashMLA dense fp8 metadata crash (num_sm_parts clamp) | 🔴 高 | closed | 2026-07-08 | [#48045](https://github.com/vllm-project/vllm/pull/48045) |
| 73 | vllm | Issue | [Bug]: Cutlass C3X `dispatch_scaled_mm` crashes on SM120 Blackwell (RTX PRO 6000) with FP8 block-scaled model (DeepSeek-V4-Flash) | 🔴 高 | open | 2026-07-07 | [#47818](https://github.com/vllm-project/vllm/issues/47818) |
| 74 | vllm | PR | [Bugfix][DCP] Cast LSE to fp32 in a2a combine to fix bf16 bitcast crash | 🔴 高 | closed | 2026-07-07 | [#47801](https://github.com/vllm-project/vllm/pull/47801) |
| 75 | vllm | Issue | [Bug]: MLA chunked-context prefill crashes on sm80 with Marlin FP8: kv_c_normed cast to packed-int32 weight dtype (`unsupported \`a\` scalar_type`) | 🔴 高 | open | 2026-07-03 | [#47522](https://github.com/vllm-project/vllm/issues/47522) |
| 76 | vllm | Issue | [Bug]: Block-scaled FP8 (compressed-tensors W8A8) crashes on load on SM120 Blackwell (RTX PRO 6000), v0.24.0 — DeepGEMM "Unknown SF transformation" assertion | 🔴 高 | open | 2026-07-02 | [#47436](https://github.com/vllm-project/vllm/issues/47436) |
| 77 | vllm | Issue | [Usage]: vLLM 0.24.0 crashes on startup with Qwen3.6-27B-FP8 on Blackwell SM120 — DeepGemm warmup ignores auto-disable | 🔴 高 | open | 2026-06-30 | [#47169](https://github.com/vllm-project/vllm/issues/47169) |
| 78 | vllm | Issue | [Bug] v0.24.0: DeepGEMM "Unknown recipe" assertion in FP8 kernel warmup on Blackwell (sm_120) — regression vs 0.23.0 | 🔴 高 | open | 2026-06-30 | [#47130](https://github.com/vllm-project/vllm/issues/47130) |
| 79 | vllm | Issue | [Bug]: AssertionError: Overwriting existing tensor attribute: weight_loader when serving FP8 model on 2x RTX 5090 (Blackwell) | 🔴 高 | open | 2026-06-29 | [#47005](https://github.com/vllm-project/vllm/issues/47005) |
| 80 | vllm | Issue | [Bug]: Llama-4-Scout-FP8 tool calls left in content (not parsed into tool_calls) — pythonic parser vs JSON output mismatch, breaks Claude Code | 🔴 高 | open | 2026-06-26 | [#46863](https://github.com/vllm-project/vllm/issues/46863) |
| 81 | vllm | PR | [Bugfix][Quantization] Fix W8A8 int-quantized scheme selection regression | 🔴 高 | closed | 2026-06-26 | [#46860](https://github.com/vllm-project/vllm/pull/46860) |
| 82 | vllm | PR | fix(quant): resolve unquantized embedding method and key mismatch in inc path for MiniMax-M3 | 🔴 高 | open | 2026-06-24 | [#46630](https://github.com/vllm-project/vllm/pull/46630) |
| 83 | vllm | Issue | [Bug]: test_flashinfer_cutlass_mxfp4_fused_moe accuracy mismatch on H20 (sm90) — 89% mismatch vs 20% threshold | 🔴 高 | closed | 2026-06-24 | [#46585](https://github.com/vllm-project/vllm/issues/46585) |
| 84 | vllm | PR | [ROCm][Test] xfail fused TRITON MXFP4 MoE accuracy on gfx950 | 🔴 高 | open | 2026-06-23 | [#46468](https://github.com/vllm-project/vllm/pull/46468) |
| 85 | vllm | Issue | [ROCm] test_rocm_mxfp4_moe_oracle is stale, and the fused-TRITON case fails accuracy on gfx950 | 🔴 高 | open | 2026-06-21 | [#46261](https://github.com/vllm-project/vllm/issues/46261) |
| 86 | vllm | PR | [Bugfix] Disable FlashInfer autotune for trtllm_bf16_moe on SM100 during warmup to prevent TMA crash | 🔴 高 | open | 2026-06-20 | [#46238](https://github.com/vllm-project/vllm/pull/46238) |
| 87 | vllm | PR | [ROCm] Use vLLM's fp8 quant max in AITER hipBLASLt accuracy test | 🔴 高 | closed | 2026-06-19 | [#46176](https://github.com/vllm-project/vllm/pull/46176) |
| 88 | vllm | PR | [ROCm] Relax AITER hipBLASLt fp8 accuracy check | 🔴 高 | closed | 2026-06-18 | [#46100](https://github.com/vllm-project/vllm/pull/46100) |
| 89 | vllm | Issue | [Bug]: GLM-5.2 (DSA sparse MLA) + fp8_ds_mla — sparse indexer off-by-one crashes concurrent decode at max_model_len &gt;= ~325K | 🔴 高 | closed | 2026-06-18 | [#46074](https://github.com/vllm-project/vllm/issues/46074) |
| 90 | vllm-ascend | Issue | [Usage]: Qwen2.5-7B-w8a8量化配置，没有性能收益（甚至有降低），精度同样降低 | 🔴 高 | closed | 2026-06-17 | [#10604](https://github.com/vllm-project/vllm-ascend/issues/10604) |
| 91 | vllm | Issue | [Perf]: Severe NVFP4 decode throughput regression on Blackwell (B300/GB300): uninitialized swizzled scale buffer in `create_fp4_scale_tensor` | 🔴 高 | closed | 2026-06-15 | [#45741](https://github.com/vllm-project/vllm/issues/45741) |
| 92 | vllm | Issue | [Bug]: `trtllm_bf16_moe` FI autotune crashes with `illegal memory access` when `--enforce-eager` is set | 🔴 高 | open | 2026-06-11 | [#45285](https://github.com/vllm-project/vllm/issues/45285) |
| 93 | vllm-ascend | PR | [Misc] improve error messages for quantization mismatches and netloader config validation   | 🔴 高 | closed | 2026-06-09 | [#10222](https://github.com/vllm-project/vllm-ascend/pull/10222) |
| 94 | vllm | PR | [Bugfix][ROCm] Fix FP8 per-tensor scale rank mismatch causing Inductor assertion failure | 🔴 高 | closed | 2026-06-08 | [#44912](https://github.com/vllm-project/vllm/pull/44912) |
| 95 | vllm | PR | [Bugfix] Fix Qwen3.5-FP8 nightly fail. Guard fused_add_rms_norm input/weight dtype mismatch in RMSNorm + quant fusion | 🔴 高 | closed | 2026-06-05 | [#44694](https://github.com/vllm-project/vllm/pull/44694) |
| 96 | vllm | Issue | [Doc]: INT8 weight-only quantization causes 4x throughput regression at batch=1 on memory-bandwidth-bound GPUs | 🔴 高 | open | 2026-05-26 | [#43700](https://github.com/vllm-project/vllm/issues/43700) |
| 97 | vllm | Issue | DSV4-Pro MTP draft: stacked attn FP8 scale loader gap + MTP forward-path mainline-vs-fork divergence | 🔴 高 | closed | 2026-05-23 | [#43472](https://github.com/vllm-project/vllm/issues/43472) |
| 98 | vllm | Issue | [Bug]: Poor Qwen3.5 NVFP4 disagg GSM8K accuracy with 2p1d (2xTEP8 prefill, 1xDEP8 decode) | 🔴 高 | open | 2026-05-17 | [#42898](https://github.com/vllm-project/vllm/issues/42898) |
| 99 | vllm | Issue | [Bug]: Gemma under NF4 quantization fails to load with AssertionError: Tried to load weights of size torch.Size([3096576, 1])to a parameter of size torch.Size([5376, 1152]) | 🔴 高 | closed | 2026-05-16 | [#42813](https://github.com/vllm-project/vllm/issues/42813) |
| 100 | vllm | Issue | [Bug]: turboquant accuracy and speed are bad than bf16 | 🔴 高 | closed | 2026-05-14 | [#42621](https://github.com/vllm-project/vllm/issues/42621) |
| 101 | vllm | PR | Create `test_kimi_k2_thinking_nvfp4.py` for accuracy check | 🔴 高 | open | 2026-05-11 | [#42351](https://github.com/vllm-project/vllm/pull/42351) |
| 102 | vllm-ascend | PR | [BugFix] Fix quantization accuracy bug | 🔴 高 | closed | 2026-05-10 | [#9036](https://github.com/vllm-project/vllm-ascend/pull/9036) |
| 103 | vllm | Issue | [Bug]: Mixed INT4/INT8 GPTQ MoE models crash on initialization (AssertionError in fused_marlin_moe) | 🔴 高 | open | 2026-05-07 | [#41955](https://github.com/vllm-project/vllm/issues/41955) |
| 104 | vllm | Issue | [Bug]: Engine crashes on startup with 'DeepGEMM backend not available' for standard bf16 models on H100 | 🔴 高 | open | 2026-05-06 | [#41849](https://github.com/vllm-project/vllm/issues/41849) |
| 105 | vllm | Issue | DeepSeek-V4 MTP2 GB200 throughput regression likely tied to FP32-&gt;FP4 cvt path (#41015) | 🔴 高 | closed | 2026-05-04 | [#41603](https://github.com/vllm-project/vllm/issues/41603) |
| 106 | vllm | PR | [Bugfix][ROCm] Fix FP8 per-tensor scale rank mismatch causing Inductor assertion failure | 🔴 高 | closed | 2026-04-29 | [#41293](https://github.com/vllm-project/vllm/pull/41293) |
| 107 | vllm | Issue | [Bug]: CUBLAS_STATUS_EXECUTION_FAILED during CUDA graph compilation of BF16 vision encoder on NVIDIA Jetson AGX Thor (vLLM 0.19.0 regression) | 🔴 高 | open | 2026-04-23 | [#40661](https://github.com/vllm-project/vllm/issues/40661) |
| 108 | vllm | Issue | [CI Failure]: mi355_2: Qwen3-30B-A3B-FP8-block Accuracy (B200-MI355) | 🔴 高 | closed | 2026-04-21 | [#40526](https://github.com/vllm-project/vllm/issues/40526) |
| 109 | vllm | Issue | [CI Failure]: mi300_4: Qwen3-30B-A3B-FP8-block Accuracy (4xH100-4xMI300) | 🔴 高 | closed | 2026-04-21 | [#40514](https://github.com/vllm-project/vllm/issues/40514) |
| 110 | vllm | Issue | [Bug]: R1 NVFP4 flash_infer_one_sided on prefill causes accuracy degradation during P/D | 🔴 高 | closed | 2026-04-20 | [#40373](https://github.com/vllm-project/vllm/issues/40373) |
| 111 | vllm | PR | [Bugfix] turboquant FP8 store crash with bf16 models on Ampere GPUs | 🔴 高 | closed | 2026-04-15 | [#39908](https://github.com/vllm-project/vllm/pull/39908) |
| 112 | vllm | Issue | MiniMaxM2 + NVFP4: shape mismatch in partial rotary embedding kernel (TP=2) | 🔴 高 | closed | 2026-04-12 | [#39625](https://github.com/vllm-project/vllm/issues/39625) |
| 113 | vllm | Issue | [Bug]: Gemma 4 MoE (26B-A4B) — runtime MXFP4 quantization crashes during weight loading in fused MoE layer | 🔴 高 | open | 2026-04-04 | [#39000](https://github.com/vllm-project/vllm/issues/39000) |
| 114 | vllm | PR | Fix Kimi-K2.5 accuracy when Aiter MLA FP8 PS + CUDA graphs are used | 🔴 高 | closed | 2026-04-01 | [#38719](https://github.com/vllm-project/vllm/pull/38719) |
| 115 | vllm | PR | fix(lora): use float32 intermediate buffer in fused MoE LoRA to prevent bf16 precision loss | 🔴 高 | closed | 2026-04-01 | [#38686](https://github.com/vllm-project/vllm/pull/38686) |
| 116 | vllm | PR | [Bugfix] Fix backup token index in async spec decode (fixes Nemotron BF16 accuracy) | 🔴 高 | closed | 2026-03-28 | [#38419](https://github.com/vllm-project/vllm/pull/38419) |
| 117 | vllm | PR | [CI][Eval] Lower Nemotron-3-Super-120B-A12B-BF16 GSM8K accuracy threshold to 0.91 | 🔴 高 | closed | 2026-03-27 | [#38403](https://github.com/vllm-project/vllm/pull/38403) |
| 118 | vllm | PR | Revert "[Bugfix] Fix DeepGemm E8M0 accuracy degradation for Qwen3.5 FP8 on Blackwell" (#38083) | 🔴 高 | closed | 2026-03-27 | [#38357](https://github.com/vllm-project/vllm/pull/38357) |
| 119 | vllm | PR | [Bugfix] Fix DeepGemm E8M0 accuracy degradation for Qwen3.5 FP8 on Blackwell | 🔴 高 | closed | 2026-03-25 | [#38083](https://github.com/vllm-project/vllm/pull/38083) |
| 120 | vllm | PR | [Bugfix] Auto-disable DeepGemm for Qwen3.5 on Blackwell to fix FP8 accuracy degradation | 🔴 高 | closed | 2026-03-22 | [#37806](https://github.com/vllm-project/vllm/pull/37806) |
| 121 | vllm | Issue | [Bug] DeepGemm E8M0 scale format causes accuracy degradation for Qwen3.5 FP8 on Blackwell | 🔴 高 | closed | 2026-03-22 | [#37804](https://github.com/vllm-project/vllm/issues/37804) |
| 122 | vllm | Issue | [Bug]: FlashInfer TRTLLM monolithic MoE produces 0% accuracy for Qwen3.5-35B/122B FP8 | 🔴 高 | closed | 2026-03-19 | [#37591](https://github.com/vllm-project/vllm/issues/37591) |
| 123 | vllm | PR | [Bugfix] Fix EP weight filter breaking EPLB and NVFP4 accuracy | 🔴 高 | closed | 2026-03-17 | [#37322](https://github.com/vllm-project/vllm/pull/37322) |
| 124 | vllm | PR | [Bugfix] Fix incorrect int8 dtype cast for kv_c_normed in MLA prefill | 🔴 高 | open | 2026-03-17 | [#37245](https://github.com/vllm-project/vllm/pull/37245) |
| 125 | vllm | PR | [Bugfix] Fix FP8 block-scale dimension mismatch in fused linear weight loading | 🔴 高 | closed | 2026-03-09 | [#36460](https://github.com/vllm-project/vllm/pull/36460) |
| 126 | vllm | Issue | [Bug]: Qwen3.5-397B-A17B-FP8 crashes with TP=4 + Expert Parallelism - dimension mismatch in fused linear sharding | 🔴 高 | closed | 2026-03-06 | [#36251](https://github.com/vllm-project/vllm/issues/36251) |
| 127 | vllm | Issue | [Bug]: Qwen3-VL-Reranker-8B fails with shape mismatch error when loading with --quantization bitsandbytes | 🔴 高 | closed | 2026-03-05 | [#36148](https://github.com/vllm-project/vllm/issues/36148) |
| 128 | vllm | Issue | [Bug]: Qwen3.5 NVFP4 Checkpoint has poor accuracy | 🔴 高 | closed | 2026-03-05 | [#36094](https://github.com/vllm-project/vllm/issues/36094) |
| 129 | vllm | PR | [Bug Fix] Qwen3.5-nvfp4 MTP Speculative Decoding Weight Shape Mismatch | 🔴 高 | closed | 2026-03-01 | [#35675](https://github.com/vllm-project/vllm/pull/35675) |
| 130 | vllm | PR | [Bugfix][Hardware][AMD] Gate FP4 ops on gfx950 to prevent MI300X crash | 🔴 高 | closed | 2026-02-25 | [#35250](https://github.com/vllm-project/vllm/pull/35250) |
| 131 | vllm | Issue | [Bug]: Qwen/Qwen3.5-397B-A17B-FP8 and Qwen/Qwen3.5-397B-A17B has accuracy issues when running with Flashinfer Attention backend on Blackwell. | 🔴 高 | closed | 2026-02-23 | [#35138](https://github.com/vllm-project/vllm/issues/35138) |
| 132 | vllm | PR | [Bugfix][Hardware][AMD] Gate FP4 BMM on gfx950 to fix MI300X crash | 🔴 高 | closed | 2026-02-23 | [#35103](https://github.com/vllm-project/vllm/pull/35103) |
| 133 | vllm | PR | [Bug Fix] MTP Speculative Decoding with NVFP4: Weight Shape Mismatch | 🔴 高 | closed | 2026-02-22 | [#35041](https://github.com/vllm-project/vllm/pull/35041) |
| 134 | vllm | Issue | [Bug]: MTP Speculative Decoding with NVFP4: Weight Shape Mismatch | 🔴 高 | closed | 2026-02-21 | [#35031](https://github.com/vllm-project/vllm/issues/35031) |
| 135 | vllm | Issue | [Bug]: Qwen3.5 FP8 accuracy degradation with FlashInfer CUTLASS MoE backend | 🔴 高 | closed | 2026-02-19 | [#34892](https://github.com/vllm-project/vllm/issues/34892) |
| 136 | vllm | Issue | [Bug]: AR+rms+fp4 fusion results in total accuracy collapse for DSV3-fp4 | 🔴 高 | closed | 2026-02-12 | [#34395](https://github.com/vllm-project/vllm/issues/34395) |
| 137 | vllm | Issue | [CI Failure]:  mi325_4: Qwen3-30B-A3B-FP8-block Accuracy (H100) | 🔴 高 | closed | 2026-02-02 | [#33598](https://github.com/vllm-project/vllm/issues/33598) |
| 138 | vllm | Issue | [Bug]: lm-eval shows significant accuracy differences on RedHatAI/Qwen3-8B-NVFP4 model (Turing vs. Ampere) | 🔴 高 | closed | 2026-02-02 | [#33560](https://github.com/vllm-project/vllm/issues/33560) |
| 139 | vllm | Issue | [CI Failure]: DeepSeek V2 Lite FP8 0% Accuracy [NIGHTLY] | 🔴 高 | closed | 2026-02-02 | [#33532](https://github.com/vllm-project/vllm/issues/33532) |
| 140 | vllm | PR | [MoE][Fix] Fix PPLX CUTLASS FP8 incorrect output with apply_router_weight_on_input | 🔴 高 | closed | 2026-01-25 | [#33019](https://github.com/vllm-project/vllm/pull/33019) |
| 141 | vllm | PR | [Bugfix] Fix the  fp8_mqa_logits dim mismatch | 🔴 高 | closed | 2026-01-20 | [#32652](https://github.com/vllm-project/vllm/pull/32652) |
| 142 | vllm | PR | [MTP][GLM][Bugfix] Fixed .weight_scale loading logic that dropped MTP prediction accuracy with fp8+mtp | 🔴 高 | closed | 2026-01-11 | [#32101](https://github.com/vllm-project/vllm/pull/32101) |
| 143 | vllm | Issue | [Bug]: nvidia/DeepSeek-R1-NVFP4-v2 accuracy issue with NVFP4 dispatch (CUTEDSL MoE + DeepEP LL) | 🔴 高 | closed | 2026-01-07 | [#31918](https://github.com/vllm-project/vllm/issues/31918) |
| 144 | vllm | Issue | [Bug]: AssertionError in `per_token_quant_int8` for long input | 🔴 高 | closed | 2026-01-01 | [#31599](https://github.com/vllm-project/vllm/issues/31599) |
| 145 | vllm | Issue | [Bug]: accuracy issue with VLLM_USE_FLASHINFER_MOE_FP8=1 for Qwen3-Coder-480B-A35B-Instruct-FP8 | 🔴 高 | closed | 2025-12-26 | [#31394](https://github.com/vllm-project/vllm/issues/31394) |
| 146 | vllm | Issue | [Bug]: Mixtral Fp8 Accuracy is Degraded | 🔴 高 | closed | 2025-12-23 | [#31202](https://github.com/vllm-project/vllm/issues/31202) |
| 147 | vllm | Issue | [Bug]: accuracy issue on MoE online fp8 quantization | 🔴 高 | closed | 2025-12-17 | [#30830](https://github.com/vllm-project/vllm/issues/30830) |
| 148 | vllm | PR | [Bugfix] Fix mismatched nvfp4 gemm output shape | 🔴 高 | closed | 2025-11-30 | [#29742](https://github.com/vllm-project/vllm/pull/29742) |
| 149 | vllm-ascend | Issue | [Bug]: DP8 场景下量化推理存在精度问题 | 🔴 高 | closed | 2025-11-19 | [#4273](https://github.com/vllm-project/vllm-ascend/issues/4273) |
| 150 | vllm | Issue | [Bug]: Vllm loading gptq quantization of int8 accuracy Qwen3-Next-80b-A3B model error KeyError: 'layers.0.mlp.experts.w2_weight' | 🔴 高 | closed | 2025-11-07 | [#28274](https://github.com/vllm-project/vllm/issues/28274) |
| 151 | vllm | Issue | [Bug]: BF16 and INT8 dtype mismatch when running quantized model on vLLM | 🔴 高 | closed | 2025-11-05 | [#28098](https://github.com/vllm-project/vllm/issues/28098) |
| 152 | vllm | Issue | [Bug]: onednn_mm crashes on consecutive bf16, f32 matmuls with same M,K,N | 🔴 高 | closed | 2025-10-24 | [#27465](https://github.com/vllm-project/vllm/issues/27465) |
| 153 | vllm | PR | [Bugfix] Fix accuracy issue of TRTLLM FP8 MOE and improve logging | 🔴 高 | closed | 2025-09-29 | [#25895](https://github.com/vllm-project/vllm/pull/25895) |
| 154 | vllm | Issue | [Bug]: Low Accuracy for MMLU Pro with DeepSeekR1-FP4 | 🔴 高 | closed | 2025-09-18 | [#25209](https://github.com/vllm-project/vllm/issues/25209) |
| 155 | vllm-ascend | Issue | [Bug]: On 910B3, Qwen2.5-VL 72B with w8a8 quantization crashes when concurrency reaches 8 | 🔴 高 | closed | 2025-09-18 | [#2997](https://github.com/vllm-project/vllm-ascend/issues/2997) |
| 156 | vllm | Issue | [Bug]: v0.10.2 Qwen 3 Next 80b FP8 Slow first request and potential format mismatch: seq_len (4) &lt; num_heads (8) | 🔴 高 | closed | 2025-09-15 | [#24865](https://github.com/vllm-project/vllm/issues/24865) |
| 157 | vllm | PR | [Bugfix] Fix accuracy issue for silu_mul + nvfp4 quant fusion kernel | 🔴 高 | closed | 2025-09-14 | [#24833](https://github.com/vllm-project/vllm/pull/24833) |
| 158 | vllm-ascend | Issue | [Bug]: DeepSeek0528量化mtp浮点权重ceval精度不达标 | 🔴 高 | closed | 2025-09-08 | [#2811](https://github.com/vllm-project/vllm-ascend/issues/2811) |
| 159 | vllm-ascend | Issue | [Bug]: function-call accuracy problem with quantized deepseek model | 🔴 高 | closed | 2025-09-01 | [#2673](https://github.com/vllm-project/vllm-ascend/issues/2673) |
| 160 | vllm | Issue | [Bug]: Accuracy under FA3 in FP8 changes drastically with TP size on very long context length | 🔴 高 | closed | 2025-08-28 | [#23813](https://github.com/vllm-project/vllm/issues/23813) |
| 161 | vllm | PR | [Bugfix] fix qwen3 moe fp8 accuracy issue | 🔴 高 | closed | 2025-08-16 | [#23031](https://github.com/vllm-project/vllm/pull/23031) |
| 162 | vllm | PR | [Bugfix] Fix shape mismatch assertion error when loading Gemma3n model with BitsAndBytes quantization | 🔴 高 | closed | 2025-07-29 | [#21808](https://github.com/vllm-project/vllm/pull/21808) |
| 163 | vllm | Issue | [Bug]: Shape mismatch assertion error when loading Gemma3n model with BitsAndBytes quantization | 🔴 高 | closed | 2025-07-28 | [#21743](https://github.com/vllm-project/vllm/issues/21743) |
| 164 | vllm-ascend | PR | [BugFix] Fix accuracy bugs for unquantized deepseekv3 models | 🔴 高 | closed | 2025-05-19 | [#897](https://github.com/vllm-project/vllm-ascend/pull/897) |
| 165 | vllm-ascend | PR | [Bugfix] fix accuracy problem for quantized deepseek models | 🔴 高 | closed | 2025-05-06 | [#768](https://github.com/vllm-project/vllm-ascend/pull/768) |
| 166 | vllm | PR | [BugFix] Accuracy fix for llama4 int4 - improperly casted scales | 🔴 高 | closed | 2025-04-17 | [#16801](https://github.com/vllm-project/vllm/pull/16801) |
| 167 | vllm | Issue | [Feature]: Can you provide a formula for gpu memory usage when deploying a model using VLLM, while providing the number of model parameters, online text length, batchsize, and quantization accuracy | 🔴 高 | closed | 2025-04-09 | [#16297](https://github.com/vllm-project/vllm/issues/16297) |
| 168 | vllm | Issue | [Bug]: FP8 accuracy decreases with long inputs | 🔴 高 | closed | 2025-04-01 | [#15865](https://github.com/vllm-project/vllm/issues/15865) |
| 169 | vllm | PR | [Bugfix] Fix Precision Mismatch in MoE Router of DeepSeek V2/V3 Models and Fused Kernels (BF16 -&gt; FP32) | 🔴 高 | closed | 2025-02-28 | [#14027](https://github.com/vllm-project/vllm/pull/14027) |
| 170 | vllm | Issue | [Bug]:Assertion error(dimension mismatch) when I use HQQ quantization for performance test | 🔴 高 | closed | 2025-02-06 | [#12843](https://github.com/vllm-project/vllm/issues/12843) |
| 171 | vllm | Issue | [Bug]:  Issue running the Granite-7b GGUF quantized model on multiple GPUs with vLLM due to a tensor size mismatch. | 🔴 高 | closed | 2025-01-17 | [#12170](https://github.com/vllm-project/vllm/issues/12170) |
| 172 | vllm | PR | [Bugfix][Core] Pin static FP8 quant pattern to group_shape=None | 🟡 中 | open | 2026-08-18 | [#52753](https://github.com/vllm-project/vllm/pull/52753) |
| 173 | vllm | Issue | [Bug]: Weight-only int8 MoE fails at every group_size under --quantization moe_wna16 | 🟡 中 | open | 2026-08-18 | [#52715](https://github.com/vllm-project/vllm/issues/52715) |
| 174 | vllm-ascend | Issue | [Bug]: Serving native FP8 checkpoints fails with AttributeError: 'Qwen3_5Config' object has no attribute 'o_groups' | 🟡 中 | open | 2026-08-18 | [#14467](https://github.com/vllm-project/vllm-ascend/issues/14467) |
| 175 | vllm | Issue | [Bug]: Qwen3.8-27B-FP8 hangs indefinitely at startup during CUDA-graph capture on Ampere (RTX A5000, TP=4) — fixed by --enforce-eager | 🟡 中 | open | 2026-08-18 | [#52682](https://github.com/vllm-project/vllm/issues/52682) |
| 176 | vllm | Issue | [Bug]: FP8 on RDNA3/gfx1100 exceeds the 600 s engine-ready timeout on first start — no AMD_Radeon_Graphics tuned configs shipped | 🟡 中 | open | 2026-08-17 | [#52663](https://github.com/vllm-project/vllm/issues/52663) |
| 177 | vllm | PR | [Bugfix][Quantization][XPU] Fix GPTQ MoE loading under moe_wna16 | 🟡 中 | open | 2026-08-17 | [#52651](https://github.com/vllm-project/vllm/pull/52651) |
| 178 | vllm | PR | [Bugfix][Quantization] Guard the MXFP8 FlashInfer path on FlashInfer availability | 🟡 中 | closed | 2026-08-17 | [#52648](https://github.com/vllm-project/vllm/pull/52648) |
| 179 | vllm | PR | [Bugfix] Fix K-tile handling in the Triton MoE and block-FP8 GEMMs | 🟡 中 | open | 2026-08-17 | [#52577](https://github.com/vllm-project/vllm/pull/52577) |
| 180 | vllm | Issue | [Bug]: Triton MoE and block-FP8 GEMMs mishandle the K tile — an out-of-bounds weight read, and wrong scales when a tile spans two quantization groups | 🟡 中 | open | 2026-08-17 | [#52576](https://github.com/vllm-project/vllm/issues/52576) |
| 181 | vllm | Issue | [Bug]: qwen3.8-27b-fp8 toolcall auto stop often | 🟡 中 | open | 2026-08-17 | [#52564](https://github.com/vllm-project/vllm/issues/52564) |
| 182 | vllm | Issue | [Bug]: FlashInfer TRT-LLM bf16 MoE weight-conversion OOM recurs post-#45589 at TP2 (120B MoE, GB200) — cumem MemPool still doesn't reclaim under pressure | 🟡 中 | open | 2026-08-16 | [#52511](https://github.com/vllm-project/vllm/issues/52511) |
| 183 | vllm | PR | [Bugfix][Quark] Preserve structured quantization config lists | 🟡 中 | closed | 2026-08-15 | [#52474](https://github.com/vllm-project/vllm/pull/52474) |
| 184 | vllm | PR | [Bugfix] Initialize FP8 partition metadata for ParallelLMHead | 🟡 中 | open | 2026-08-15 | [#52451](https://github.com/vllm-project/vllm/pull/52451) |
| 185 | vllm | PR | [Bugfix][Model] Kimi-K3 MegaMoE: pass situ_beta/situ_linear_beta to fp8_fp4_mega_moe | 🟡 中 | closed | 2026-08-15 | [#52445](https://github.com/vllm-project/vllm/pull/52445) |
| 186 | vllm | PR | [ROCm][gfx942] DSv4 sparse-attn indexer: native fp8 MFMA + corrected LDS occupancy gate | 🟡 中 | open | 2026-08-14 | [#52402](https://github.com/vllm-project/vllm/pull/52402) |
| 187 | vllm | PR | [Bugfix][DSv4] SM12x Triton fallback for fp8_einsum / o_proj (#43743) | 🟡 中 | open | 2026-08-14 | [#52357](https://github.com/vllm-project/vllm/pull/52357) |
| 188 | vllm | PR | [Bugfix][ROCm] Skip FP8 MLA prefill PS-metadata build for chunked-context batches | 🟡 中 | closed | 2026-08-14 | [#52356](https://github.com/vllm-project/vllm/pull/52356) |
| 189 | vllm | PR | [Bugfix][ROCm] Fix a few int4/int8 quantization errors | 🟡 中 | closed | 2026-08-13 | [#52112](https://github.com/vllm-project/vllm/pull/52112) |
| 190 | vllm-ascend | PR | [BugFix] Fix quantization comm | 🟡 中 | open | 2026-08-13 | [#14234](https://github.com/vllm-project/vllm-ascend/pull/14234) |
| 191 | vllm | Issue | [Bug][ROCm]: AITER FP8 BMM segfaults with data parallel attention | 🟡 中 | open | 2026-08-12 | [#51957](https://github.com/vllm-project/vllm/issues/51957) |
| 192 | vllm | Issue | [Bug]: FP8 block-scaled weights fail on sm120 (RTX 5090) — DeepGEMM "Unknown SF transformation" during process_weights_after_loading | 🟡 中 | open | 2026-08-11 | [#51884](https://github.com/vllm-project/vllm/issues/51884) |
| 193 | vllm | PR | [Bugfix][Triton] Make fp8_min/fp8_max constexpr in _quantize_pad_fp8_kernel | 🟡 中 | closed | 2026-08-11 | [#51872](https://github.com/vllm-project/vllm/pull/51872) |
| 194 | vllm | PR | [Bugfix][Spec Decode] Restore DeepSeek-V4 DSpark draft quantization | 🟡 中 | closed | 2026-08-11 | [#51835](https://github.com/vllm-project/vllm/pull/51835) |
| 195 | vllm | Issue | [Bug]: Qwen3.5 GDN CUDA kernels promote BF16 Q/K normalization to FP32 | 🟡 中 | open | 2026-08-11 | [#51779](https://github.com/vllm-project/vllm/issues/51779) |
| 196 | vllm | PR | [Bugfix][CPU] Make the Apple Silicon BF16 probe fall back instead of raising | 🟡 中 | closed | 2026-08-10 | [#51627](https://github.com/vllm-project/vllm/pull/51627) |
| 197 | vllm-ascend | PR | [Refactor] Remove MiniMax-M2 fp8 dequant load_weights patch | 🟡 中 | closed | 2026-08-07 | [#13782](https://github.com/vllm-project/vllm-ascend/pull/13782) |
| 198 | vllm | PR | [Bugfix][Quantization] Fix fp32 weight scale for mxfp4 quantization and per-expert checkpoint mapping | 🟡 中 | closed | 2026-08-07 | [#51419](https://github.com/vllm-project/vllm/pull/51419) |
| 199 | vllm | PR | [ROCm][Bugfix] Use BF16 MLA prefill for short prompts | 🟡 中 | closed | 2026-08-07 | [#51380](https://github.com/vllm-project/vllm/pull/51380) |
| 200 | vllm | PR | [Bugfix][Attention] Forward per-head FP8 descales through FA4 | 🟡 中 | closed | 2026-08-07 | [#51363](https://github.com/vllm-project/vllm/pull/51363) |
| 201 | vllm | PR | [Bugfix] Keep top-level quantization_config when benchmark_moe descends via --model-prefix | 🟡 中 | open | 2026-08-07 | [#51353](https://github.com/vllm-project/vllm/pull/51353) |
| 202 | vllm | Issue | [Bug]: DSL kernel only supports fp16/bf16 inputs when using torch.float32 dtype | 🟡 中 | open | 2026-08-06 | [#51305](https://github.com/vllm-project/vllm/issues/51305) |
| 203 | vllm | PR | [Bugfix][MiniMax-M3] Keep FP8 query allocation stable across CUDA graph replay | 🟡 中 | open | 2026-08-05 | [#51203](https://github.com/vllm-project/vllm/pull/51203) |
| 204 | vllm | Issue | [CI Failure]: test_mha_attn_varlen_forward_aiter_fp8 — "invalid argument for fmha_fwd" on MI300X | 🟡 中 | closed | 2026-08-05 | [#51115](https://github.com/vllm-project/vllm/issues/51115) |
| 205 | vllm | PR | [Bugfix][Humming] Preserve ModelOpt FP8 weight dimensions | 🟡 中 | closed | 2026-08-05 | [#51093](https://github.com/vllm-project/vllm/pull/51093) |
| 206 | vllm | PR | [Bugfix][Quantization] Fix MXFP4 conversion for FlashInfer CUTLASS | 🟡 中 | closed | 2026-08-04 | [#51038](https://github.com/vllm-project/vllm/pull/51038) |
| 207 | vllm | Issue | [Bug][ROCm] Remove `AITER_MXFP4_BF16` and `AITER_MXFP4_FP8` MOE routing hotfix once AITER is bumped | 🟡 中 | open | 2026-08-04 | [#51032](https://github.com/vllm-project/vllm/issues/51032) |
| 208 | vllm | PR | [Bugfix][Build] Fix DeepGEMM CUDA 12.9 FP8 header visibility | 🟡 中 | closed | 2026-08-04 | [#51003](https://github.com/vllm-project/vllm/pull/51003) |
| 209 | vllm | PR | [Bugfix][LoRA] Guard TrtLlm BF16 MoE LoRA gate on activation type | 🟡 中 | closed | 2026-08-04 | [#51002](https://github.com/vllm-project/vllm/pull/51002) |
| 210 | vllm-ascend | PR | [BugFix][FusedMoE] Restore BF16 quant method initialization | 🟡 中 | closed | 2026-08-03 | [#13412](https://github.com/vllm-project/vllm-ascend/pull/13412) |
| 211 | vllm | Issue | [Bug]: Qwen/Qwen3.6-35B-A3B and Qwen/Qwen3.6-35B-A3B-FP8 issue while inferencing on Intel XPU (4xB70) | 🟡 中 | closed | 2026-08-03 | [#50850](https://github.com/vllm-project/vllm/issues/50850) |
| 212 | vllm | PR | [Bugfix][Quantization] Fix dynamic INT8 W8A8 MoE config being built as W8A16 | 🟡 中 | closed | 2026-08-03 | [#50833](https://github.com/vllm-project/vllm/pull/50833) |
| 213 | vllm | PR | [Bugfix] Honor per-model DeepGEMM auto-disable in FP8 MoE backend selection and warmup | 🟡 中 | closed | 2026-08-03 | [#50829](https://github.com/vllm-project/vllm/pull/50829) |
| 214 | vllm | Issue | [Bug]: sm_120 + local CUDA toolkit &lt; 12.9: FlashInfer JIT failures kill engine init in three default paths (sampler, fused-MoE, FP8 KV) instead of falling back | 🟡 中 | open | 2026-08-01 | [#50705](https://github.com/vllm-project/vllm/issues/50705) |
| 215 | vllm | PR | [Bugfix][Model] Kimi-K3 NVIDIA: delegate regular FusedMoE padding to the selected quantization backend | 🟡 中 | open | 2026-07-31 | [#50583](https://github.com/vllm-project/vllm/pull/50583) |
| 216 | vllm | PR | [Bugfix] Handle single-shard FP8 Marlin MoE padding | 🟡 中 | open | 2026-07-31 | [#50568](https://github.com/vllm-project/vllm/pull/50568) |
| 217 | vllm | PR | [ROCm][Bugfix] Keep the MLA query in bf16 for AITER Gluon fp8 KV decode | 🟡 中 | closed | 2026-07-31 | [#50563](https://github.com/vllm-project/vllm/pull/50563) |
| 218 | vllm | PR | [Bugfix] Fix six quantization exception messages split across positional args | 🟡 中 | open | 2026-07-30 | [#50479](https://github.com/vllm-project/vllm/pull/50479) |
| 219 | vllm | PR | [Hardware][AMD][Kernel][CI][Bugfix] Fix ROCm DeepEP FP8 max | 🟡 中 | closed | 2026-07-30 | [#50467](https://github.com/vllm-project/vllm/pull/50467) |
| 220 | vllm | PR | [CI Bugfix] Temp disable Humming wNa8 INT8 H100 CI | 🟡 中 | closed | 2026-07-29 | [#50329](https://github.com/vllm-project/vllm/pull/50329) |
| 221 | vllm | PR | [Bugfix][Quantization] Fix fused AutoGPTQ overrides and mixed-group MoE loading | 🟡 中 | closed | 2026-07-29 | [#50214](https://github.com/vllm-project/vllm/pull/50214) |
| 222 | vllm-ascend | PR | [BugFix]Fix the precision issue of the npu_dequant_swiglu_quant operator when x.shape=[2,192]- #12424 | 🟡 中 | closed | 2026-07-28 | [#12965](https://github.com/vllm-project/vllm-ascend/pull/12965) |
| 223 | vllm-ascend | PR | [BugFix]Fix the precision issue of the npu_dequant_swiglu_quant operator when x.shape=[2,192]- #12424 | 🟡 中 | closed | 2026-07-28 | [#12964](https://github.com/vllm-project/vllm-ascend/pull/12964) |
| 224 | vllm | Issue | [Bug/Feature]: w8a8_block_int8_matmul kernel is dormant | 🟡 中 | open | 2026-07-28 | [#50143](https://github.com/vllm-project/vllm/issues/50143) |
| 225 | vllm | PR | [Bugfix] Don't transpose fused MoE quantization scales in `RoutedExperts.load_weights` | 🟡 中 | closed | 2026-07-28 | [#50137](https://github.com/vllm-project/vllm/pull/50137) |
| 226 | vllm | PR | [Bugfix][Quantization] Reuse online NVFP4 MoE kernel across reloads | 🟡 中 | closed | 2026-07-28 | [#50074](https://github.com/vllm-project/vllm/pull/50074) |
| 227 | vllm-ascend | PR | [Ascend950][Bugfix]Fix allgatherEP MXFPW4A8 quantization (#11663) | 🟡 中 | closed | 2026-07-28 | [#12967](https://github.com/vllm-project/vllm-ascend/pull/12967) |
| 228 | vllm | PR | [Quantization] Preserve precision in online NVFP4 expert packing | 🟡 中 | closed | 2026-07-27 | [#50029](https://github.com/vllm-project/vllm/pull/50029) |
| 229 | vllm-ascend | PR | [BugFix]Fix the precision issue of the npu_dequant_swiglu_quant operator when x.shape=[2,192]、[1,256] | 🟡 中 | closed | 2026-07-27 | [#12913](https://github.com/vllm-project/vllm-ascend/pull/12913) |
| 230 | vllm | PR | bugfix: fp8 MLA with Marlin kernels | 🟡 中 | closed | 2026-07-27 | [#49989](https://github.com/vllm-project/vllm/pull/49989) |
| 231 | vllm | PR | [Bugfix][Kernel] Keep DeepGEMM FP8 quant inside opaque op for TMA scales | 🟡 中 | open | 2026-07-27 | [#49928](https://github.com/vllm-project/vllm/pull/49928) |
| 232 | vllm-ascend | PR | [Cherry-pick][releases/v0.24.0rc][BugFix] Fix commmode setting of mm_reduce_scatter for bf16 in A5 (#12689) | 🟡 中 | closed | 2026-07-27 | [#12899](https://github.com/vllm-project/vllm-ascend/pull/12899) |
| 233 | vllm-ascend | PR | [Cherry-pick][releases/v0.23.0][BugFix] Fix commmode setting of mm_reduce_scatter for bf16 in A5 (#12689) | 🟡 中 | closed | 2026-07-27 | [#12895](https://github.com/vllm-project/vllm-ascend/pull/12895) |
| 234 | vllm | PR | [Bugfix][Quantization] Match draft model quant targets under its root prefix | 🟡 中 | open | 2026-07-26 | [#49900](https://github.com/vllm-project/vllm/pull/49900) |
| 235 | vllm | Issue | [Bug]: v0.26.0 EngineCore fails to start on SM90+ — _warmup_ll_bf16_router_gemm hits AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma' (nvidia-cutlass-dsl 4.6.0) | 🟡 中 | open | 2026-07-26 | [#49859](https://github.com/vllm-project/vllm/issues/49859) |
| 236 | vllm-ascend | PR | [v0.23.0][BugFix] Restrict FP8 quantization detection to A5 devices only | 🟡 中 | closed | 2026-07-25 | [#12826](https://github.com/vllm-project/vllm-ascend/pull/12826) |
| 237 | vllm | PR | [CI][Bugfix] Fix test isolation in block_int8/ptpc_fp8 MoE kernel tests | 🟡 中 | closed | 2026-07-23 | [#49609](https://github.com/vllm-project/vllm/pull/49609) |
| 238 | vllm | Issue | [Bug]: deadlock / hang with GLM-5.2-FP8 EP+DP on GB200 - with ablation table | 🟡 中 | open | 2026-07-23 | [#49594](https://github.com/vllm-project/vllm/issues/49594) |
| 239 | vllm | Issue | [Bug]:  vllm0.25版本支持qwen3-vl fp8算子DeepGemmFp8BlockScaledMMKernel 走原始的deepgemm的后端 | 🟡 中 | open | 2026-07-23 | [#49584](https://github.com/vllm-project/vllm/issues/49584) |
| 240 | vllm | Issue | [Bug]: MTP draft model ignores checkpoint per-layer quantization config (block_name_to_quantize mapped inconsistently vs target) | 🟡 中 | open | 2026-07-23 | [#49552](https://github.com/vllm-project/vllm/issues/49552) |
| 241 | vllm | PR | [Bugfix][Quantization] Keep unquantized routed experts unquantized under AutoGPTQ | 🟡 中 | open | 2026-07-23 | [#49539](https://github.com/vllm-project/vllm/pull/49539) |
| 242 | vllm-ascend | PR | [BugFix] Fix commmode setting of mm_reduce_scatter for bf16 in A5 | 🟡 中 | closed | 2026-07-23 | [#12689](https://github.com/vllm-project/vllm-ascend/pull/12689) |
| 243 | vllm | PR | [Bugfix] Fix MiniMax-M3 ModelOpt FP8 MoE SwiGLU params | 🟡 中 | open | 2026-07-22 | [#49473](https://github.com/vllm-project/vllm/pull/49473) |
| 244 | vllm-ascend | PR | [BugFix]Fix the precision issue of the npu_dequant_swiglu_quant operator when x.shape=[2,192] | 🟡 中 | closed | 2026-07-20 | [#12424](https://github.com/vllm-project/vllm-ascend/pull/12424) |
| 245 | vllm | Issue | [Bug]: [Performance]: Qwen3-VL-32B-FP8 throughput collapses on v0.25.1 (v0.24.0 fine) | 🟡 中 | open | 2026-07-20 | [#49259](https://github.com/vllm-project/vllm/issues/49259) |
| 246 | vllm | Issue | POST /wake_up fails with AttributeError: 'list' object has no attribute 'zero_' in init_fp8_kv_scales, wedging the engine (health stays green, completions hang) | 🟡 中 | open | 2026-07-20 | [#49237](https://github.com/vllm-project/vllm/issues/49237) |
| 247 | vllm | PR | [Bugfix][Quantization] Respect explicit WNA16 MoE backend | 🟡 中 | open | 2026-07-20 | [#49207](https://github.com/vllm-project/vllm/pull/49207) |
| 248 | vllm | PR | [Bugfix][Test] Quantize q to fp8 in test_flashmla_dense_fp8_decode_unified_slot_view | 🟡 中 | closed | 2026-07-20 | [#49191](https://github.com/vllm-project/vllm/pull/49191) |
| 249 | vllm | Issue | [Bug]: inkling bf16 h20*2 start failed with vllm/vllm-openai:inkling | 🟡 中 | closed | 2026-07-19 | [#49105](https://github.com/vllm-project/vllm/issues/49105) |
| 250 | vllm | PR | [Bugfix] Guard scaled_fp4_quant against non-contiguous input | 🟡 中 | open | 2026-07-19 | [#49092](https://github.com/vllm-project/vllm/pull/49092) |
| 251 | vllm | PR | [Bugfix][Quantization] Respect explicit WNA16 MoE backend | 🟡 中 | closed | 2026-07-19 | [#49065](https://github.com/vllm-project/vllm/pull/49065) |
| 252 | vllm | PR | Revert "[Bugfix] Fix activation quantization dispatch for WNA4Int/WNA8Int" (#48785) | 🟡 中 | closed | 2026-07-18 | [#49032](https://github.com/vllm-project/vllm/pull/49032) |
| 253 | vllm-ascend | Issue | [Bug]: 基于2台Atls 800T/800I  A2  使用vllm-ascend 0.21.RC1镜像混部Qwen3-235B-A22B-Instruct-2507（BF16）模型后，curl openai接口后返回的结果乱码，见下图 | 🟡 中 | open | 2026-07-17 | [#12258](https://github.com/vllm-project/vllm-ascend/issues/12258) |
| 254 | vllm | PR | [ROCm] [BugFix] Fix Quark GLM-5.2 Checkpoint inference: indexer wk per-channel FP8 dequant + missing sparse-MLA metadata fields | 🟡 中 | closed | 2026-07-16 | [#48886](https://github.com/vllm-project/vllm/pull/48886) |
| 255 | vllm-ascend | PR | [BugFix]Fix the precision issue of the npu_dequant_swiglu_quant operator when x.shape=[2,192] | 🟡 中 | closed | 2026-07-15 | [#12123](https://github.com/vllm-project/vllm-ascend/pull/12123) |
| 256 | vllm | PR | [Bugfix] Fix activation quantization dispatch for WNA4Int/WNA8Int | 🟡 中 | closed | 2026-07-15 | [#48785](https://github.com/vllm-project/vllm/pull/48785) |
| 257 | vllm | Issue | [Bug]: sample_tokens RPC timeout with GLM-5.2-FP8 + DSpark speculative decoding, TP=8 across 2 nodes (Blackwell GB200) | 🟡 中 | open | 2026-07-15 | [#48752](https://github.com/vllm-project/vllm/issues/48752) |
| 258 | vllm | PR | [Bugfix][ROCm] Only run FP8 AITER MLA prefill when using FP8 KV | 🟡 中 | open | 2026-07-15 | [#48712](https://github.com/vllm-project/vllm/pull/48712) |
| 259 | vllm | PR | [Bugfix] Reject DeepSeek V4 FP4 MoE tensor parallelism | 🟡 中 | closed | 2026-07-15 | [#48697](https://github.com/vllm-project/vllm/pull/48697) |
| 260 | vllm | PR | [Bugfix] Gate FlashInfer FP4 MoE on kernel availability | 🟡 中 | closed | 2026-07-15 | [#48695](https://github.com/vllm-project/vllm/pull/48695) |
| 261 | vllm | Issue | [Bug]: GLM-5.2 nvpf4 quantization not loading with TP | 🟡 中 | open | 2026-07-15 | [#48689](https://github.com/vllm-project/vllm/issues/48689) |
| 262 | vllm | PR | [XPU] Route INC WNA16 MoE to oracle backend instead of bf16 dequant | 🟡 中 | closed | 2026-07-14 | [#48555](https://github.com/vllm-project/vllm/pull/48555) |
| 263 | vllm | PR | [Bugfix] Sparse MLA: enable fp8_ds_mla dense prefill | 🟡 中 | closed | 2026-07-14 | [#48642](https://github.com/vllm-project/vllm/pull/48642) |
| 264 | vllm | PR | [BugFix] Sparse MLA: fix req_id_per_token OOB write and fp8_ds_mla context-prefill routing after the dense-MHA split | 🟡 中 | closed | 2026-07-14 | [#48612](https://github.com/vllm-project/vllm/pull/48612) |
| 265 | vllm | PR | [Bugfix] Pad unaligned N in SM12x CUTLASS blockwise FP8 GEMM | 🟡 中 | open | 2026-07-14 | [#48588](https://github.com/vllm-project/vllm/pull/48588) |
| 266 | vllm | PR | [Bugfix] Reject CUTLASS block-scaled FP8 when N is not a multiple of 128 | 🟡 中 | closed | 2026-07-14 | [#48587](https://github.com/vllm-project/vllm/pull/48587) |
| 267 | vllm | PR | [Bugfix] Reject CUTLASS block-scaled FP8 when N is not a multiple of 128 | 🟡 中 | closed | 2026-07-14 | [#48586](https://github.com/vllm-project/vllm/pull/48586) |
| 268 | vllm | Issue | [Bug]: FlashInfer CUTLASS MoE selected on fp4-less builds (CUDA toolkit &lt; 12.8); gpt-oss dies at engine start | 🟡 中 | open | 2026-07-13 | [#48541](https://github.com/vllm-project/vllm/issues/48541) |
| 269 | vllm | PR | [Bug][Quantization] Fix humming is_layer_skipped for compressed-tensors "re:" ignore entries | 🟡 中 | closed | 2026-07-13 | [#48507](https://github.com/vllm-project/vllm/pull/48507) |
| 270 | vllm-ascend | PR | [v0.23.0][BugFix] Revert allgatherEP mxfp4 quantization | 🟡 中 | closed | 2026-07-13 | [#11905](https://github.com/vllm-project/vllm-ascend/pull/11905) |
| 271 | vllm | PR | [Bugfix][Quantization] Run block kernel post-processing for ModelOpt FP8_PB_WO | 🟡 中 | open | 2026-07-12 | [#48422](https://github.com/vllm-project/vllm/pull/48422) |
| 272 | vllm | PR | [Bugfix] Fix fp8_ds_mla config leak in DSpark speculative decoding | 🟡 中 | closed | 2026-07-12 | [#48415](https://github.com/vllm-project/vllm/pull/48415) |
| 273 | vllm | PR | [Bugfix][Quantization] Avoid buffering absent WNA16 reload tensors | 🟡 中 | closed | 2026-07-11 | [#48356](https://github.com/vllm-project/vllm/pull/48356) |
| 274 | vllm | PR | [Bugfix] Pin FlashInfer bmm_fp8 to cuBLAS on sm_12x to avoid cuDNN hot-path stalls | 🟡 中 | open | 2026-07-10 | [#48210](https://github.com/vllm-project/vllm/pull/48210) |
| 275 | vllm | PR | Fa4 fp8 kv dequant integration clean | 🟡 中 | open | 2026-07-09 | [#48192](https://github.com/vllm-project/vllm/pull/48192) |
| 276 | vllm | PR | [ROCm][Bugfix] Pad block-FP8 MoE intermediate size for TP when not divisible by block_n | 🟡 中 | closed | 2026-07-09 | [#48173](https://github.com/vllm-project/vllm/pull/48173) |
| 277 | vllm-ascend | PR | [Cherry-pick][releases/v0.23.0][Ascend950][Bugfix]Fix allgatherEP MXFPW4A8 quantization (from #11663) | 🟡 中 | closed | 2026-07-09 | [#11718](https://github.com/vllm-project/vllm-ascend/pull/11718) |
| 278 | vllm | PR | [Bug fix] Always pass launch_pdl kwarg in fused_inv_rope_fp8_quant (DeepSeek-V4) | 🟡 中 | open | 2026-07-09 | [#48086](https://github.com/vllm-project/vllm/pull/48086) |
| 279 | vllm-ascend | PR | [Ascend950][Bugfix]Fix allgatherEP MXFPW4A4 quantization | 🟡 中 | open | 2026-07-09 | [#11677](https://github.com/vllm-project/vllm-ascend/pull/11677) |
| 280 | vllm-ascend | PR | [Ascend950][Bugfix]Fix allgatherEP MXFPW4A8 quantization | 🟡 中 | closed | 2026-07-09 | [#11663](https://github.com/vllm-project/vllm-ascend/pull/11663) |
| 281 | vllm | PR | [Bugfix] Use int8 workspace for FlashInfer MLA decode | 🟡 中 | closed | 2026-07-08 | [#48046](https://github.com/vllm-project/vllm/pull/48046) |
| 282 | vllm | PR | [Bugfix] Do not re-apply q/k/v scales on FlashInfer TRTLLM BF16-Q paths | 🟡 中 | open | 2026-07-08 | [#48016](https://github.com/vllm-project/vllm/pull/48016) |
| 283 | vllm-ascend | PR | [BugFix] Revert allgather quantization  | 🟡 中 | closed | 2026-07-08 | [#11643](https://github.com/vllm-project/vllm-ascend/pull/11643) |
| 284 | vllm-ascend | PR | [BugFix] Revert all2all quantization | 🟡 中 | closed | 2026-07-08 | [#11642](https://github.com/vllm-project/vllm-ascend/pull/11642) |
| 285 | vllm-ascend | PR | [BugFix] Revert allgather quantization | 🟡 中 | closed | 2026-07-08 | [#11638](https://github.com/vllm-project/vllm-ascend/pull/11638) |
| 286 | vllm | Issue | [Bug]: SM120 CUTLASS blockwise FP8 GEMM rejects N % 128 != 0 (Invalid status; kv_a_proj N=576 in DeepSeek shapes) | 🟡 中 | open | 2026-07-08 | [#47990](https://github.com/vllm-project/vllm/issues/47990) |
| 287 | vllm | PR | [Bugfix] Handle E8M0 block scales in CUTLASS and Triton FP8 linear kernels | 🟡 中 | open | 2026-07-08 | [#47988](https://github.com/vllm-project/vllm/pull/47988) |
| 288 | vllm | Issue | [Bug]: DSpark launch failed with FP4 target model on GLM 5.2 | 🟡 中 | closed | 2026-07-08 | [#47934](https://github.com/vllm-project/vllm/issues/47934) |
| 289 | vllm-ascend | PR | [BugFix] Revert all2all quantization | 🟡 中 | closed | 2026-07-07 | [#11580](https://github.com/vllm-project/vllm-ascend/pull/11580) |
| 290 | vllm | PR | [Bugfix] [Quantization] Fix loading for CT DSV2 | 🟡 中 | closed | 2026-07-06 | [#47780](https://github.com/vllm-project/vllm/pull/47780) |
| 291 | vllm | Issue | [Bug]: vllm 0.23.0 and 0.24.0 - Qwen3.6-35B-A3B-FP8 - Fails generating code- "400 Unterminated string starting at" | 🟡 中 | open | 2026-07-06 | [#47761](https://github.com/vllm-project/vllm/issues/47761) |
| 292 | vllm | Issue | [Bug]: RTX 5090 / SM120 ModelOpt mixed NVFP4 checkpoint falls back to Marlin W4A16 path and warns no native FP4 support | 🟡 中 | open | 2026-07-06 | [#47749](https://github.com/vllm-project/vllm/issues/47749) |
| 293 | vllm-ascend | PR | [BugFix]Fixing UT case adapted for the w4a16-mxfp4 quantization scheme | 🟡 中 | closed | 2026-07-06 | [#11476](https://github.com/vllm-project/vllm-ascend/pull/11476) |
| 294 | vllm | Issue | [Bug]: RuntimeError: scheduler_metadata must have shape (metadata_size) when using dspark speculative decoding with FP8 quantization on H20-3e | 🟡 中 | open | 2026-07-05 | [#47626](https://github.com/vllm-project/vllm/issues/47626) |
| 295 | vllm | PR | [Bugfix][MoE] Handle older FlashInfer FP8 MoE signatures | 🟡 中 | open | 2026-07-04 | [#47611](https://github.com/vllm-project/vllm/pull/47611) |
| 296 | vllm | PR | [Bugfix][Kernel] Zero TRT-LLM 8x4 FP4 scale padding | 🟡 中 | open | 2026-07-04 | [#47587](https://github.com/vllm-project/vllm/pull/47587) |
| 297 | vllm | Issue | [Bug]: DeepSeek-V4-Pro (DeepseekV4ForCausalLM, scale_fmt=ue8m0 / FP4 MoE) produces garbled / degenerate output under tensor parallelism (TP), while data parallelism + expert parallelism (DP+EP) works correctly | 🟡 中 | open | 2026-07-03 | [#47528](https://github.com/vllm-project/vllm/issues/47528) |
| 298 | vllm | Issue | [Bug]: Gemma4 (probably other models with missing v_proj) breaks with mixed quantization | 🟡 中 | open | 2026-07-02 | [#47473](https://github.com/vllm-project/vllm/issues/47473) |
| 299 | vllm | PR | [BugFix] Fix ModelOpt quantization inference for fused siblings | 🟡 中 | closed | 2026-07-02 | [#47445](https://github.com/vllm-project/vllm/pull/47445) |
| 300 | vllm | PR | [Bugfix] Load per-shard per-tensor FP8 scales for fused GDN in_proj (Qwen3.5/Qwen3-Next) | 🟡 中 | open | 2026-07-02 | [#47396](https://github.com/vllm-project/vllm/pull/47396) |
| 301 | vllm | PR | [BugFix] Fix ModelOpt mixed-precision quantization for sparse `quantized_layers` configs. | 🟡 中 | closed | 2026-07-01 | [#47318](https://github.com/vllm-project/vllm/pull/47318) |
| 302 | vllm | PR | [Bugfix][Kernel] Re-bias exponent in BF16 NVFP4 Marlin scale dequant | 🟡 中 | closed | 2026-07-01 | [#47315](https://github.com/vllm-project/vllm/pull/47315) |
| 303 | vllm | PR | [Bugfix][ROCm] Fix memory access fault in AITER MLA backend for DPA+FP8 KV  | 🟡 中 | closed | 2026-07-01 | [#47276](https://github.com/vllm-project/vllm/pull/47276) |
| 304 | vllm | PR | [Bugifx][INC] Fix INC quantization method selection for non-quantized layers | 🟡 中 | open | 2026-07-01 | [#47237](https://github.com/vllm-project/vllm/pull/47237) |
| 305 | vllm | Issue | [Bug]: nvidia/Qwen3.6-27B-NVFP4 (dense, MIXED_PRECISION W4A16_NVFP4) fails to load — "RuntimeError: start (0) + length (17408) exceeds dimension size (8704)" | 🟡 中 | closed | 2026-06-30 | [#47215](https://github.com/vllm-project/vllm/issues/47215) |
| 306 | vllm | PR | [ROCm][Bugfix] Convert ModelOpt FP8 per-channel weights to e4m3fnuz on MI300/MI325 | 🟡 中 | closed | 2026-06-30 | [#47201](https://github.com/vllm-project/vllm/pull/47201) |
| 307 | vllm | Issue | [Bug]: [ROCm][gfx942] GPU memory access fault with MTP spec-decode + sparse-MLA decode under cuda graph (GLM-5.1-FP8) | 🟡 中 | closed | 2026-06-30 | [#47196](https://github.com/vllm-project/vllm/issues/47196) |
| 308 | vllm | PR | [XPU][Bugfix] Fix FP8 block-scaled GEMM for unaligned N | 🟡 中 | closed | 2026-06-30 | [#47103](https://github.com/vllm-project/vllm/pull/47103) |
| 309 | vllm | Issue | [Bug][ROCm] GLM-5.2-FP8 sparse MLA decode degenerates at long context on gfx942 (MI325X) | 🟡 中 | closed | 2026-06-29 | [#47042](https://github.com/vllm-project/vllm/issues/47042) |
| 310 | vllm | PR | [Bugfix][ROCm][MLA] Pass q/kv dtypes to get_mla_metadata_v1 in FP8 decode | 🟡 中 | closed | 2026-06-29 | [#46997](https://github.com/vllm-project/vllm/pull/46997) |
| 311 | vllm | PR | [Bugfix][ROCm][MLA] Pad FP8 MLA decode head count to a native value on gfx950 (fix AITER 0.1.16-post2) | 🟡 中 | closed | 2026-06-28 | [#46952](https://github.com/vllm-project/vllm/pull/46952) |
| 312 | vllm | Issue | [Bug] TPU: State leak in FP8 MoE shared_experts after AOT compilation fails with prefix caching | 🟡 中 | open | 2026-06-26 | [#46857](https://github.com/vllm-project/vllm/issues/46857) |
| 313 | vllm | PR | [ROCm][Bugfix] window-correct shuffled fp8 decode for SWA layers | 🟡 中 | open | 2026-06-26 | [#46847](https://github.com/vllm-project/vllm/pull/46847) |
| 314 | vllm | PR | [Bugfix] Fix MiniMax-M3 compressed-tensors FP8 MoE SwiGLU params | 🟡 中 | closed | 2026-06-26 | [#46845](https://github.com/vllm-project/vllm/pull/46845) |
| 315 | vllm | Issue | MTP produces invalid output with GLM-5.2-FP8 MoE (1% GSM8K vs 95% without MTP) | 🟡 中 | closed | 2026-06-26 | [#46834](https://github.com/vllm-project/vllm/issues/46834) |
| 316 | vllm-ascend | Issue | [Bug]: Kimi-K2.7-Code 双节点 16 卡部署：TP=16 跨节点输出乱码，DP+EP 显存不足，--quantization ascend 报错 | 🟡 中 | closed | 2026-06-26 | [#11019](https://github.com/vllm-project/vllm-ascend/issues/11019) |
| 317 | vllm | PR | [ROCm][Bugfix][MLA] Fix mla_reduce_v1 num_kv_splits arg for FP8 MLA prefill | 🟡 中 | closed | 2026-06-26 | [#46810](https://github.com/vllm-project/vllm/pull/46810) |
| 318 | vllm | PR | [CPU][BugFix] Multiple fixes to w4a8_int8 CPU MoE path | 🟡 中 | closed | 2026-06-25 | [#46739](https://github.com/vllm-project/vllm/pull/46739) |
| 319 | vllm | PR | [ROCm][Perf][Bugfix] DSv4 indexer: use platform FP8 dtype (fnuz) for Q-quant on gfx942 | 🟡 中 | closed | 2026-06-25 | [#46730](https://github.com/vllm-project/vllm/pull/46730) |
| 320 | vllm | PR | [Bugfix] Use block_k for block-wise FP8 activation group_shape | 🟡 中 | open | 2026-06-24 | [#46593](https://github.com/vllm-project/vllm/pull/46593) |
| 321 | vllm | Issue | [Bug]: MiniMax-M3 INT4 (auto-round / GPTQ-pack) deployment hits 2 quantization-with-ignore-list bugs | 🟡 中 | open | 2026-06-24 | [#46569](https://github.com/vllm-project/vllm/issues/46569) |
| 322 | vllm-ascend | PR | [BugFix][v0.22.1] Fix Error when GLM5 MTP layer has BF16 weights | 🟡 中 | closed | 2026-06-23 | [#10854](https://github.com/vllm-project/vllm-ascend/pull/10854) |
| 323 | vllm-ascend | PR | [BugFix][v0.22.1] Fix Error when GLM5 MTP layer has BF16 weights | 🟡 中 | closed | 2026-06-23 | [#10853](https://github.com/vllm-project/vllm-ascend/pull/10853) |
| 324 | vllm-ascend | PR | [BugFix][v0.22.1] Fix Error when GLM5 MTP layer has BF16 weights | 🟡 中 | closed | 2026-06-23 | [#10852](https://github.com/vllm-project/vllm-ascend/pull/10852) |
| 325 | vllm-ascend | PR | [BugFix] Fix Error when GLM5 MTP layer has BF16 weights | 🟡 中 | closed | 2026-06-23 | [#10842](https://github.com/vllm-project/vllm-ascend/pull/10842) |
| 326 | vllm | Issue | [Bug]: Weird outputs for GLM-5.2-FP8 on 8xB200 | 🟡 中 | closed | 2026-06-22 | [#46367](https://github.com/vllm-project/vllm/issues/46367) |
| 327 | vllm | PR | [Bugfix] Re-enable FP8 MoE on NVIDIA Thor | 🟡 中 | closed | 2026-06-22 | [#46339](https://github.com/vllm-project/vllm/pull/46339) |
| 328 | vllm | Issue | [Bug]: glm-5-fp8 zcode str object has no attribute items | 🟡 中 | open | 2026-06-22 | [#46328](https://github.com/vllm-project/vllm/issues/46328) |
| 329 | vllm | Issue | [Bug]: PP2 tool calling produces garbled output while PP1 works correctly (GLM-5.2-FP8 DSA, chunked-prefill) | 🟡 中 | open | 2026-06-21 | [#46262](https://github.com/vllm-project/vllm/issues/46262) |
| 330 | vllm | PR | [ROCm][Perf] Avoid fp32 round-trip dequant in fp8 KV paged decode | 🟡 中 | open | 2026-06-19 | [#46191](https://github.com/vllm-project/vllm/pull/46191) |
| 331 | vllm | PR | [Bugfix] Preserve FP8 indexer WK pairs across incremental load_weights | 🟡 中 | closed | 2026-06-19 | [#46168](https://github.com/vllm-project/vllm/pull/46168) |
| 332 | vllm | Issue | [Bug]:  GPU coredump during FlashInfer `trtllm_bf16_moe` autotune with Qwen3.5-35B-A3B on B200 (DP=2 + EP) | 🟡 中 | closed | 2026-06-18 | [#46083](https://github.com/vllm-project/vllm/issues/46083) |
| 333 | vllm | Issue | [Bug]: [quantization] The Qwen3 4B model quantized for vLLM inference encounters errors. | 🟡 中 | open | 2026-06-18 | [#46036](https://github.com/vllm-project/vllm/issues/46036) |
| 334 | vllm | Issue | [Bug]: GptOssForCausalLM fails to load BF16 (dequantized) checkpoint — KeyError: 'layers.0.mlp.experts.w2_weight' | 🟡 中 | closed | 2026-06-16 | [#45830](https://github.com/vllm-project/vllm/issues/45830) |
| 335 | vllm | PR | [Quantization] Extend ModelOpt mixed precision and NVFP4 runtime formats | 🟡 中 | open | 2026-06-15 | [#45735](https://github.com/vllm-project/vllm/pull/45735) |
| 336 | vllm-ascend | Issue | [Bug]: Deepseek -V4 is not support on bf16 | 🟡 中 | open | 2026-06-15 | [#10501](https://github.com/vllm-project/vllm-ascend/issues/10501) |
| 337 | vllm-ascend | PR | [Quantization][BugFix] dsv4 quant_model_substr_mapping key fix | 🟡 中 | closed | 2026-06-14 | [#10458](https://github.com/vllm-project/vllm-ascend/pull/10458) |
| 338 | vllm | Issue | [Bug] test_w8a8_block_fp8_fused_moe uses fixed atol that K=7168 quantization noise legitimately exceeds (fails on SM120/RTX PRO 6000) | 🟡 中 | open | 2026-06-11 | [#45332](https://github.com/vllm-project/vllm/issues/45332) |
| 339 | vllm | PR | fix(cohere2_moe): fix weight loader KeyErrors for FP8 and BF16 checkpoints | 🟡 中 | closed | 2026-06-11 | [#45314](https://github.com/vllm-project/vllm/pull/45314) |
| 340 | vllm | PR | [Bugfix] Fix INC XPU MoE quantization routing for RoutedExperts | 🟡 中 | closed | 2026-06-10 | [#45090](https://github.com/vllm-project/vllm/pull/45090) |
| 341 | vllm | PR | [Bugfix] Lazily import the humming quantization backend | 🟡 中 | closed | 2026-06-08 | [#44921](https://github.com/vllm-project/vllm/pull/44921) |
| 342 | vllm-ascend | PR | [BugFix]fix w4a8 mxfp quantization in shared experts | 🟡 中 | closed | 2026-06-08 | [#10153](https://github.com/vllm-project/vllm-ascend/pull/10153) |
| 343 | vllm | Issue | [Bug]:  Prefix caching causes IndexError in Qwen3.5-122B-A10B (BF16) on A2 | 🟡 中 | open | 2026-06-05 | [#44637](https://github.com/vllm-project/vllm/issues/44637) |
| 344 | vllm | PR | [Bugfix][Quantization] Fp8 family: match modules_to_not_convert by substring | 🟡 中 | open | 2026-06-05 | [#44628](https://github.com/vllm-project/vllm/pull/44628) |
| 345 | vllm | PR | [Bugfix] Exclude vision embedder from quantization in Gemma4 Unified | 🟡 中 | closed | 2026-06-04 | [#44571](https://github.com/vllm-project/vllm/pull/44571) |
| 346 | vllm | Issue | [Bug]: mistral w4a16 Quantization slower than FP16 | 🟡 中 | open | 2026-06-04 | [#44529](https://github.com/vllm-project/vllm/issues/44529) |
| 347 | vllm-ascend | Issue | [Bug]: DeepSeek-V4-Flash bf16 pd分离场景，32并发压测 D节点挂死shutdown | 🟡 中 | closed | 2026-06-04 | [#10013](https://github.com/vllm-project/vllm-ascend/issues/10013) |
| 348 | vllm | Issue | [Bug]: Online FP8 (`--quantization fp8`) over-allocates non-gated MoE `w13` (2×intermediate), causing OOM — NemotronH on a single GPU | 🟡 中 | open | 2026-06-04 | [#44489](https://github.com/vllm-project/vllm/issues/44489) |
| 349 | vllm-ascend | PR | [Quantization][BugFix] Fix lm_head prefix mapping for qwen3_5_moe MTP drafter | 🟡 中 | closed | 2026-06-04 | [#9972](https://github.com/vllm-project/vllm-ascend/pull/9972) |
| 350 | vllm-ascend | Issue | [Bug]: GLM5.1-w8a8在commit@a45cd 跑P2D2服务，D节点bench中途挂掉，抛出算子报错 DynamicQuant_bf16_c1_high_performance_10 | 🟡 中 | closed | 2026-06-01 | [#9842](https://github.com/vllm-project/vllm-ascend/issues/9842) |
| 351 | vllm | PR | [AMD][Bugfix][Quantization] Honor fused-name match in is_layer_skipped | 🟡 中 | closed | 2026-05-29 | [#43981](https://github.com/vllm-project/vllm/pull/43981) |
| 352 | vllm | Issue | [Bug]: KeyError: 'layers.0.mlp.gate_up_proj.g_idx'  of  GLM-OCR GPTQ Int8 in v0.21.1rc1 | 🟡 中 | open | 2026-05-29 | [#43967](https://github.com/vllm-project/vllm/issues/43967) |
| 353 | vllm | PR | [Bugfix][DeepSeekV4] Skip MTP quant config for BF16 draft weights | 🟡 中 | closed | 2026-05-28 | [#43893](https://github.com/vllm-project/vllm/pull/43893) |
| 354 | vllm | PR | [Bugfix] Pass `routed_scaling_factor` to FlashInfer TRTLLM BF16 MoE | 🟡 中 | closed | 2026-05-27 | [#43769](https://github.com/vllm-project/vllm/pull/43769) |
| 355 | vllm | PR | [Bugfix][Quantization] Refuse block-FP8 in MarlinFP8.can_implement | 🟡 中 | open | 2026-05-27 | [#43722](https://github.com/vllm-project/vllm/pull/43722) |
| 356 | vllm | PR | [Bugfix] Fix CUTLASS MoE router input weighting for FP8/FP4 | 🟡 中 | open | 2026-05-26 | [#43694](https://github.com/vllm-project/vllm/pull/43694) |
| 357 | vllm-ascend | PR | [BugFix][DeepSeek] Apply SwiGLU clamp before MoE quantization | 🟡 中 | closed | 2026-05-26 | [#9590](https://github.com/vllm-project/vllm-ascend/pull/9590) |
| 358 | vllm-ascend | Issue | [Bug]: Qwen3.5-122B-A3B-W8A8 quantization ascend works wrong, Profiling shows that Full Attention doesn't  have quantization kernel , so this model can't get the W8A8 performance benefits | 🟡 中 | open | 2026-05-22 | [#9443](https://github.com/vllm-project/vllm-ascend/issues/9443) |
| 359 | vllm | PR | [Bugfix][DSV4] MTP draft model: detect BF16 MTP on disk + skip quant_config | 🟡 中 | open | 2026-05-21 | [#43319](https://github.com/vllm-project/vllm/pull/43319) |
| 360 | vllm | Issue | [Bug] DSV4 MTP draft model inherits main quantization scheme; can't load artifacts with BF16 MTP block | 🟡 中 | closed | 2026-05-21 | [#43304](https://github.com/vllm-project/vllm/issues/43304) |
| 361 | vllm-ascend | Issue | [Bug]: Qwen3.6-27B 启动bf16权重推理,第一次推理无法返回结果 | 🟡 中 | open | 2026-05-20 | [#9355](https://github.com/vllm-project/vllm-ascend/issues/9355) |
| 362 | vllm | PR | [BugFix] Kimi-K2.5: skip vision tower dtype conversion when using quantization | 🟡 中 | closed | 2026-05-17 | [#42869](https://github.com/vllm-project/vllm/pull/42869) |
| 363 | vllm | PR | fix(quantization): Fix AWQ dequantize on Intel XPU and refactor AutoAWQ config | 🟡 中 | closed | 2026-05-15 | [#42727](https://github.com/vllm-project/vllm/pull/42727) |
| 364 | vllm | PR | [Bugfix] All pyNCCL copy-only operation to use int8 instead of fp8 | 🟡 中 | open | 2026-05-15 | [#42724](https://github.com/vllm-project/vllm/pull/42724) |
| 365 | vllm | PR | [Bugfix][NVFP4] Expose batch-invariance for FlashInfer + CUTLASS FP4 MoE | 🟡 中 | open | 2026-05-14 | [#42670](https://github.com/vllm-project/vllm/pull/42670) |
| 366 | vllm | PR | [Quantization][ModelOpt] W4A16 NVFP4 fused MoE + mixed-precision dispatch | 🟡 中 | closed | 2026-05-13 | [#42566](https://github.com/vllm-project/vllm/pull/42566) |
| 367 | vllm | Issue | [Bug]: LoRA on AWQ-quantized Llama-3.1-8B / Llama-3.2-3B produces degenerate output (same LoRA infra works on AWQ Mistral and on FP8 / BF16 Llama) | 🟡 中 | open | 2026-05-13 | [#42488](https://github.com/vllm-project/vllm/issues/42488) |
| 368 | vllm | PR | [Bugfix][Quantization] Skip 'bias' tensors in online_process_loader | 🟡 中 | closed | 2026-05-09 | [#42141](https://github.com/vllm-project/vllm/pull/42141) |
| 369 | vllm | PR | [Bugfix][Quantization] Fix INC/AutoRound GPTQ routing for group-misaligned TP shards | 🟡 中 | closed | 2026-05-08 | [#42030](https://github.com/vllm-project/vllm/pull/42030) |
| 370 | vllm | PR | [Bugfix] Route INT8 GPTQ MoE to WNA16 fallback | 🟡 中 | closed | 2026-05-08 | [#42022](https://github.com/vllm-project/vllm/pull/42022) |
| 371 | vllm | Issue | [Bug]: AttributeError when loading Mamba2ForCausalLM with BitsAndBytes quantization | 🟡 中 | open | 2026-05-06 | [#41874](https://github.com/vllm-project/vllm/issues/41874) |
| 372 | vllm | PR | [CI][Bugfix][MoE] Skip trtllm_fp4_block_scale_moe during FlashInfer autotune | 🟡 中 | closed | 2026-04-30 | [#41329](https://github.com/vllm-project/vllm/pull/41329) |
| 373 | vllm | PR | [Bugfix] Use CUDART_VERSION instead of CUDA_VERSION macro guard for fp4 quant kernel | 🟡 中 | closed | 2026-04-30 | [#41310](https://github.com/vllm-project/vllm/pull/41310) |
| 374 | vllm | PR | [Bugfix][Quantization] GGUF: sort merged-column slots by index, not stream order | 🟡 中 | open | 2026-04-27 | [#41057](https://github.com/vllm-project/vllm/pull/41057) |
| 375 | vllm | PR | [Bugfix][Quantization] GGUF: sort merged-column slots by index, not stream order | 🟡 中 | closed | 2026-04-27 | [#41053](https://github.com/vllm-project/vllm/pull/41053) |
| 376 | vllm | Issue | [Bug] Online FP8 quantization ignores logical_widths on MergedColumnParallelLinear | 🟡 中 | open | 2026-04-27 | [#41022](https://github.com/vllm-project/vllm/issues/41022) |
| 377 | vllm-ascend | PR | [Doc][BugFix] Fix Qwen3.5 &lt;--quantization ascend&gt; in Multi-node Deplo… | 🟡 中 | open | 2026-04-27 | [#8751](https://github.com/vllm-project/vllm-ascend/pull/8751) |
| 378 | vllm | Issue | [Bug]: --quantization fp8 fails on Qwen3.5 hybrid (qwen3_next gated delta net) with cutlass_scaled_mm Error Internal on GB10 sm_121 | 🟡 中 | open | 2026-04-26 | [#40934](https://github.com/vllm-project/vllm/issues/40934) |
| 379 | vllm | PR | [Bugfix] Fix `NameError` on `AsyncEngineArgs.quantization_config` forward reference | 🟡 中 | open | 2026-04-24 | [#40847](https://github.com/vllm-project/vllm/pull/40847) |
| 380 | vllm | PR | [Bugfix] Clarify FLASHINFER limitation for per-attention-head KV quantization | 🟡 中 | open | 2026-04-21 | [#40448](https://github.com/vllm-project/vllm/pull/40448) |
| 381 | vllm | Issue | [Bug]: Per-attention-head quantization is currently available only with the Flash Attention backend and requires the calibration pathway provided by llm-compressor. | 🟡 中 | open | 2026-04-21 | [#40444](https://github.com/vllm-project/vllm/issues/40444) |
| 382 | vllm | Issue | [Bug]:  Gemma-4-31B-IT-NVFP4  (modelopt) causing OOM on single RTX 5090, suspect full BF16 weights during init | 🟡 中 | closed | 2026-04-19 | [#40291](https://github.com/vllm-project/vllm/issues/40291) |
| 383 | vllm | Issue | [Bug]: v0.19.1 failed to load AWQ 4bit quantization of Gemma 4 26B-A4B | 🟡 中 | closed | 2026-04-19 | [#40286](https://github.com/vllm-project/vllm/issues/40286) |
| 384 | vllm | PR | [Bugfix] Temporarily disable B200 fp4 MoE layer tests | 🟡 中 | closed | 2026-04-16 | [#40057](https://github.com/vllm-project/vllm/pull/40057) |
| 385 | vllm | PR | [Bugfix] Fix turboquant FP8 cast failure for BF16 models on Ampere GPUs | 🟡 中 | closed | 2026-04-16 | [#39988](https://github.com/vllm-project/vllm/pull/39988) |
| 386 | vllm | PR | [Bugfix] Skip bias tensors in online FP8 quantization pipeline | 🟡 中 | closed | 2026-04-16 | [#39962](https://github.com/vllm-project/vllm/pull/39962) |
| 387 | vllm | PR | [Bugfix] Skip bias tensors in online FP8 quantization pipeline | 🟡 中 | closed | 2026-04-13 | [#39665](https://github.com/vllm-project/vllm/pull/39665) |
| 388 | vllm | Issue | [Bug]: Online FP8 quantization drops bias weights, which breaks Qwen2 and other models with bias=True | 🟡 中 | open | 2026-04-13 | [#39663](https://github.com/vllm-project/vllm/issues/39663) |
| 389 | vllm | PR | [Bug]: Fix FlashInfer CUTLASS BF16 + CUDA graphs IMA | 🟡 中 | closed | 2026-04-12 | [#39593](https://github.com/vllm-project/vllm/pull/39593) |
| 390 | vllm | PR | fix: exclude MTP layers from MoE quantization to fix KeyError with moe_wna16 | 🟡 中 | open | 2026-04-10 | [#39475](https://github.com/vllm-project/vllm/pull/39475) |
| 391 | vllm | PR | [Bugfix][Quantization] Fix Gemma4 AutoRound serving gaps on top of GPTQMarlin row groups | 🟡 中 | closed | 2026-04-09 | [#39460](https://github.com/vllm-project/vllm/pull/39460) |
| 392 | vllm | Issue | [Bug]: FlashInfer CUTLASS MoE backend causes CUDA illegal memory access on H100 during CUDA graph capture (Qwen3-Next-80B BF16) | 🟡 中 | closed | 2026-04-08 | [#39288](https://github.com/vllm-project/vllm/issues/39288) |
| 393 | vllm | Issue | [Bug]: Gemma 4 FP8 dynamic quantization = gibberish output | 🟡 中 | open | 2026-04-05 | [#39049](https://github.com/vllm-project/vllm/issues/39049) |
| 394 | vllm | PR | [Bugfix][cmake] fix FP4 ARCH for CUDA&gt;=13.0 | 🟡 中 | open | 2026-04-03 | [#38957](https://github.com/vllm-project/vllm/pull/38957) |
| 395 | vllm | PR | [Bugfix][Perf] Indexer upcast WK to BF16 for fusion | 🟡 中 | closed | 2026-04-03 | [#38928](https://github.com/vllm-project/vllm/pull/38928) |
| 396 | vllm | PR | [Bugfix][Quantization] Fix PerTensorScale loading with tuple shard_id in MergedColumnParallelLinear | 🟡 中 | closed | 2026-03-30 | [#38517](https://github.com/vllm-project/vllm/pull/38517) |
| 397 | vllm | PR | [Bug fix][Quantization] Fix dummy weight loading | 🟡 中 | closed | 2026-03-29 | [#38478](https://github.com/vllm-project/vllm/pull/38478) |
| 398 | vllm-ascend | PR | [Bugfix]Fix deepseek 3.2 C8 precision by revert quantization layers | 🟡 中 | closed | 2026-03-25 | [#7628](https://github.com/vllm-project/vllm-ascend/pull/7628) |
| 399 | vllm | PR | Revert "[Bug][MoE] Fix TRTLLM NVFP4 Routing Kernel Precision" (#36725) | 🟡 中 | open | 2026-03-24 | [#37945](https://github.com/vllm-project/vllm/pull/37945) |
| 400 | vllm | Issue | [Bug]: NGC vLLM 26.02 rejects Nemotron-3-Super-120B-A12B-NVFP4 — quant_algo MIXED_PRECISION not in whitelist | 🟡 中 | closed | 2026-03-23 | [#37854](https://github.com/vllm-project/vllm/issues/37854) |
| 401 | vllm-ascend | PR | [BugFix][310p]Handle null quantization config in ShardedStateLoader310 | 🟡 中 | closed | 2026-03-23 | [#7546](https://github.com/vllm-project/vllm-ascend/pull/7546) |
| 402 | vllm | Issue | [Bug]: FLASHINFER_CUTLASS and FLASHINFER_TRTLLM do not work for Qwen3.5 Bf16 DP/EP | 🟡 中 | closed | 2026-03-21 | [#37758](https://github.com/vllm-project/vllm/issues/37758) |
| 403 | vllm | PR | Fix SM121 GB10 FP4 quantization sticky CUDA error | 🟡 中 | closed | 2026-03-18 | [#37410](https://github.com/vllm-project/vllm/pull/37410) |
| 404 | vllm | Issue | [Bug]: _C.scaled_fp4_quant produces sticky CUDA error on SM121 (DGX Spark GB10) — contaminates CUDA context | 🟡 中 | closed | 2026-03-18 | [#37402](https://github.com/vllm-project/vllm/issues/37402) |
| 405 | vllm-ascend | PR | [Bugfix][Doc] Fix bf16 type cast error in causal_conv1d Triton kernel | 🟡 中 | closed | 2026-03-14 | [#7281](https://github.com/vllm-project/vllm-ascend/pull/7281) |
| 406 | vllm-ascend | PR | [fix]: fix precision issue in dispatch_ffn_combine_bf16 and remove redundant sync | 🟡 中 | closed | 2026-03-12 | [#7198](https://github.com/vllm-project/vllm-ascend/pull/7198) |
| 407 | vllm | PR | [Bug][MoE] Fix TRTLLM NVFP4 Routing Kernel Precision | 🟡 中 | closed | 2026-03-11 | [#36725](https://github.com/vllm-project/vllm/pull/36725) |
| 408 | vllm-ascend | Issue | [Bug]: MoE weight loading error with Ascend W4A8 quantization | 🟡 中 | open | 2026-03-09 | [#7088](https://github.com/vllm-project/vllm-ascend/issues/7088) |
| 409 | vllm-ascend | PR | [Bugfix] Resolve weight loading error with Ascend W4A8 quantization | 🟡 中 | closed | 2026-03-09 | [#7087](https://github.com/vllm-project/vllm-ascend/pull/7087) |
| 410 | vllm | Issue | [Bug]: Qwen3.5-9B (BF16/AWQ) Illegal Memory Access in vLLM v0.17.0 (WSL2/RTX3090 Ti) | 🟡 中 | closed | 2026-03-08 | [#36408](https://github.com/vllm-project/vllm/issues/36408) |
| 411 | vllm-ascend | PR | [bugfix]Qwen-Omni quantization bugfix | 🟡 中 | closed | 2026-03-06 | [#7042](https://github.com/vllm-project/vllm-ascend/pull/7042) |
| 412 | vllm-ascend | Issue | [Bug]: run failed for qwen2-72b w8a16 quantization | 🟡 中 | closed | 2026-03-06 | [#7039](https://github.com/vllm-project/vllm-ascend/issues/7039) |
| 413 | vllm | PR | docs: fix wrong cc in int8.md | 🟡 中 | closed | 2026-03-06 | [#36209](https://github.com/vllm-project/vllm/pull/36209) |
| 414 | vllm | PR | [Bugfix] Disable FlashInfer TRTLLM BF16 path for non-gated MoE | 🟡 中 | closed | 2026-03-05 | [#36146](https://github.com/vllm-project/vllm/pull/36146) |
| 415 | vllm-ascend | PR | [bugfix]Qwen-Omni quantization model_type bugfix | 🟡 中 | closed | 2026-03-05 | [#7007](https://github.com/vllm-project/vllm-ascend/pull/7007) |
| 416 | vllm | Issue | [Bug]: Segfault at IP=0 during model warmup on AVX512_BF16 host (AMD 7940HS) | 🟡 中 | closed | 2026-03-02 | [#35746](https://github.com/vllm-project/vllm/issues/35746) |
| 417 | vllm-ascend | Issue | [Bug]: A2尝试四机部署GLM5-bf16失败，报错：NPU function error: call aclnnSwiGlu failed, error code is 507014 | 🟡 中 | closed | 2026-02-28 | [#6875](https://github.com/vllm-project/vllm-ascend/issues/6875) |
| 418 | vllm | Issue | [Bug]: MXFP4A16 compressed-tensors quantization produces degenerate output (PPL 22,953 vs 8.74 BF16) | 🟡 中 | closed | 2026-02-27 | [#35562](https://github.com/vllm-project/vllm/issues/35562) |
| 419 | vllm-ascend | PR | [BugFix][Quant] Fix remote model ID handling in quantization config detection | 🟡 中 | closed | 2026-02-26 | [#6836](https://github.com/vllm-project/vllm-ascend/pull/6836) |
| 420 | vllm-ascend | Issue | [Bug]: R1-BF16的大EP。服务拉起，单curl长度超过100token左右，D节点会挂死。 | 🟡 中 | closed | 2026-02-25 | [#6799](https://github.com/vllm-project/vllm-ascend/issues/6799) |
| 421 | vllm | PR | [BugFix] Fix fp4 quant kernel on CUDA 12.8 | 🟡 中 | closed | 2026-02-24 | [#35210](https://github.com/vllm-project/vllm/pull/35210) |
| 422 | vllm | Issue | [Bug]: FlashInfer attn-fp4 fused kernel performs worse than unfused | 🟡 中 | closed | 2026-02-20 | [#34988](https://github.com/vllm-project/vllm/issues/34988) |
| 423 | vllm | Issue | [Bug]: BF16 NVFP4 Marlin produces garbled output on GPUs without native FP4 support | 🟡 中 | open | 2026-02-17 | [#34694](https://github.com/vllm-project/vllm/issues/34694) |
| 424 | vllm | PR | [Bugfix] Enable attn quantization of Llama-4 by correctly permuting scales for rope (int8, fp8) | 🟡 中 | closed | 2026-02-10 | [#34243](https://github.com/vllm-project/vllm/pull/34243) |
| 425 | vllm-ascend | PR | fix bf16 bug | 🟡 中 | closed | 2026-02-08 | [#6616](https://github.com/vllm-project/vllm-ascend/pull/6616) |
| 426 | vllm | Issue | [Bug]: PD report xpxd with deepseekv32 fp4  Assertion error kv.second.dim()==1 | 🟡 中 | closed | 2026-01-27 | [#33144](https://github.com/vllm-project/vllm/issues/33144) |
| 427 | vllm | PR | [Bugfix][MXFP4] Call `trtllm_fp4_block_scale_moe` with kwargs | 🟡 中 | closed | 2026-01-26 | [#33104](https://github.com/vllm-project/vllm/pull/33104) |
| 428 | vllm | Issue | [Bug][cpu][arm]: Failure case for BF16 dispatch on non-bf16 supported arm HW | 🟡 中 | closed | 2026-01-23 | [#32932](https://github.com/vllm-project/vllm/issues/32932) |
| 429 | vllm-ascend | Issue | [Bug]:使用0.11.0rc3和0.13.0版本镜像部署DeepSeek-R1-Distill-Llama-70B（int8）模型，报错  is_skipped = self.quant_description[prefix + '.weight'] == "FLOAT" | 🟡 中 | closed | 2026-01-04 | [#5564](https://github.com/vllm-project/vllm-ascend/issues/5564) |
| 430 | vllm | PR | [Bugfix][Quantization] Ensure input contiguity in per_token_quant_int8 | 🟡 中 | closed | 2026-01-03 | [#31637](https://github.com/vllm-project/vllm/pull/31637) |
| 431 | vllm | Issue | [Bug]: FP8 inference much slower than BF16 on MI300X for GLM-4.7 and MiniMax-M2.1 | 🟡 中 | closed | 2025-12-29 | [#31475](https://github.com/vllm-project/vllm/issues/31475) |
| 432 | vllm | PR | [Bugfix][ROCm] Fix typo: triton_fp4_gemm_dynamic_qaunt -&gt; quant | 🟡 中 | closed | 2025-12-22 | [#31157](https://github.com/vllm-project/vllm/pull/31157) |
| 433 | vllm | PR | [FIX] FP4 quantization kernel padding initialization bug | 🟡 中 | closed | 2025-12-21 | [#31097](https://github.com/vllm-project/vllm/pull/31097) |
| 434 | vllm | Issue | [Bug]: gpt-oss-20b fails to start when using W4A8 Marlin with VLLM_MARLIN_INPUT_DTYPE=int8 | 🟡 中 | closed | 2025-12-19 | [#31033](https://github.com/vllm-project/vllm/issues/31033) |
| 435 | vllm-ascend | Issue | [Bug]: bf16 lora don't work with 0.11.0rc2 | 🟡 中 | closed | 2025-12-15 | [#5021](https://github.com/vllm-project/vllm-ascend/issues/5021) |
| 436 | vllm-ascend | Issue | [Bug]: qwen3-vl-235B-bf16 FULL_DECODE_ONLY + VLLM_ASCEND_ENABLE_NZ=1 压测报错问题 | 🟡 中 | closed | 2025-12-12 | [#4960](https://github.com/vllm-project/vllm-ascend/issues/4960) |
| 437 | vllm-ascend | Issue | [Bug]: 部署满血版DeepSeek R1 0528 BF16失败 | 🟡 中 | closed | 2025-12-12 | [#4948](https://github.com/vllm-project/vllm-ascend/issues/4948) |
| 438 | vllm-ascend | Issue | [Bug][v0.11.0-dev]: The qwen2.5-vl-72b-bf16 model repetition issues when acync-scheduling is enabled. | 🟡 中 | closed | 2025-12-10 | [#4887](https://github.com/vllm-project/vllm-ascend/issues/4887) |
| 439 | vllm-ascend | PR | [bugfix] Fixed the bug in retrieving the quantization method for mlp.… | 🟡 中 | closed | 2025-12-08 | [#4797](https://github.com/vllm-project/vllm-ascend/pull/4797) |
| 440 | vllm | Issue | [Bug]: weights_not_loaded check failed when loading bf16 DeepSeek-V3.2-Exp | 🟡 中 | closed | 2025-12-04 | [#30036](https://github.com/vllm-project/vllm/issues/30036) |
| 441 | vllm | Issue | [Bug]: DeepSeek-V3.1-Terminus-BF16 run error | 🟡 中 | closed | 2025-12-04 | [#30017](https://github.com/vllm-project/vllm/issues/30017) |
| 442 | vllm-ascend | Issue | [Bug]: OpenBMB/MiniCPM-2B-dpo-bf16 start failed by `Duplicate layer name: .down_proj` | 🟡 中 | closed | 2025-12-01 | [#4603](https://github.com/vllm-project/vllm-ascend/issues/4603) |
| 443 | vllm-ascend | PR | [main][bugfix] bugfix for qwen3 moe quantization | 🟡 中 | closed | 2025-12-01 | [#4599](https://github.com/vllm-project/vllm-ascend/pull/4599) |
| 444 | vllm-ascend | PR | [Bugfix] Remove ModelSlim-"M4 Quantization". | 🟡 中 | closed | 2025-12-01 | [#4589](https://github.com/vllm-project/vllm-ascend/pull/4589) |
| 445 | vllm | Issue | [Bug]: DSR1 fp4/fp8 MTP with spec num 3 has perf drop when enable async-scheduling | 🟡 中 | closed | 2025-11-28 | [#29662](https://github.com/vllm-project/vllm/issues/29662) |
| 446 | vllm | Issue | [Bug]: DSR1 fp4 MTP with spec num 3 has perf drop | 🟡 中 | closed | 2025-11-28 | [#29660](https://github.com/vllm-project/vllm/issues/29660) |
| 447 | vllm | Issue | [Bug]: BF16Vec has no fallback options for Arm CPUs with no BF16 support | 🟡 中 | closed | 2025-11-25 | [#29391](https://github.com/vllm-project/vllm/issues/29391) |
| 448 | vllm-ascend | Issue | [Bug]: x86+npu环境，部署DeepSeek-V3.2-Exp-BF16后curl请求程序崩溃 | 🟡 中 | closed | 2025-11-21 | [#4322](https://github.com/vllm-project/vllm-ascend/issues/4322) |
| 449 | vllm-ascend | Issue | [Bug]: KeyError: 'visual.blocks.0.attn.qkv.weight' when Loading FP8 Quantized Qwen2.5-VL Model | 🟡 中 | closed | 2025-11-20 | [#4313](https://github.com/vllm-project/vllm-ascend/issues/4313) |
| 450 | vllm-ascend | Issue | [Bug]: LoRA + Ascend Quantization Fails on Qwen3-32B-W8A8 with AscendRMSNorm AttributeError | 🟡 中 | closed | 2025-11-20 | [#4308](https://github.com/vllm-project/vllm-ascend/issues/4308) |
| 451 | vllm | Issue | [Bug]: RuntimeError: Int8 not supported for this architecture | 🟡 中 | closed | 2025-11-17 | [#28856](https://github.com/vllm-project/vllm/issues/28856) |
| 452 | vllm-ascend | PR | [Bugfix] fix mtp profile run error where main model and mtp model use different quantization | 🟡 中 | closed | 2025-11-10 | [#4102](https://github.com/vllm-project/vllm-ascend/pull/4102) |
| 453 | vllm-ascend | PR | [bugfix] Fixed the bug in retrieving the quantization method for mlp.experts (e.g., DeepSeek_v3.2_exp w8a8) | 🟡 中 | closed | 2025-11-06 | [#4035](https://github.com/vllm-project/vllm-ascend/pull/4035) |
| 454 | vllm | Issue | [Bug]: llama 4 + fp4 is broke | 🟡 中 | closed | 2025-11-05 | [#28107](https://github.com/vllm-project/vllm/issues/28107) |
| 455 | vllm | Issue | [Bug]: Can't run Flashinfer MoE TRTLLM backend FP4 for Qwen3 235B | 🟡 中 | closed | 2025-11-03 | [#28007](https://github.com/vllm-project/vllm/issues/28007) |
| 456 | vllm | Issue | [Bug]: vLLM 0.10.2/0.11.0 bench serve deadlocks when benchmarking DeepSeek-R1-BF16 (sglang 0.4.7), with processes hanging indefinitely during script execution | 🟡 中 | closed | 2025-10-31 | [#27886](https://github.com/vllm-project/vllm/issues/27886) |
| 457 | vllm | Issue | [Bug]: `KeyError: 'layers.47.mlp.experts.w2_weight'` loading a NVFP4 + BF16 mixed-precision `llm-compressor` model | 🟡 中 | closed | 2025-10-27 | [#27607](https://github.com/vllm-project/vllm/issues/27607) |
| 458 | vllm | PR | [Bugfix] Respect ignore list for NVFP4/BF16 mixed MoE checkpoints | 🟡 中 | closed | 2025-10-27 | [#27608](https://github.com/vllm-project/vllm/pull/27608) |
| 459 | vllm | Issue | [Bug]: NVFP4A16 spurious warning that GPU doesn't support Fp4 | 🟡 中 | closed | 2025-10-24 | [#27471](https://github.com/vllm-project/vllm/issues/27471) |
| 460 | vllm | Issue | [Bug]: SM120 int8 unsupport | 🟡 中 | closed | 2025-10-22 | [#27337](https://github.com/vllm-project/vllm/issues/27337) |
| 461 | vllm-ascend | PR | [BugFix][main] Fix quantization related mtp bug with patch | 🟡 中 | closed | 2025-10-22 | [#3620](https://github.com/vllm-project/vllm-ascend/pull/3620) |
| 462 | vllm-ascend | PR | [BugFix][v0.11.0] Fix quantization related mtp bug with patch | 🟡 中 | closed | 2025-10-22 | [#3619](https://github.com/vllm-project/vllm-ascend/pull/3619) |
| 463 | vllm-ascend | Issue | [Bug]: GLM4.6-INT8  KeyError: 'model.layers.0.self_attn.q_proj.weight' | 🟡 中 | closed | 2025-10-17 | [#3520](https://github.com/vllm-project/vllm-ascend/issues/3520) |
| 464 | vllm-ascend | Issue | [Bug]: 无法加载 GLM4.5-Air-FP8 | 🟡 中 | closed | 2025-10-15 | [#3464](https://github.com/vllm-project/vllm-ascend/issues/3464) |
| 465 | vllm | Issue | [Bug]: Memory leak in DeepSeek FP4 on Blackwell | 🟡 中 | closed | 2025-10-02 | [#26142](https://github.com/vllm-project/vllm/issues/26142) |
| 466 | vllm | Issue | [Bug]: DSR1 FP4 + DEP8 on B200 fails with TensorRT-LLM throughput kernels | 🟡 中 | closed | 2025-10-02 | [#26070](https://github.com/vllm-project/vllm/issues/26070) |
| 467 | vllm | PR | [Bugfix] Fix build issue around `cutlass_fp4_group_mm` on platforms without nvfp4 | 🟡 中 | closed | 2025-10-02 | [#26061](https://github.com/vllm-project/vllm/pull/26061) |
| 468 | vllm | PR | Fix INT8 quantization error on Blackwell GPUs (SM100+) | 🟡 中 | closed | 2025-09-30 | [#25935](https://github.com/vllm-project/vllm/pull/25935) |
| 469 | vllm | PR | [Bugfix] Enable padded FP4 quantization | 🟡 中 | closed | 2025-09-30 | [#25947](https://github.com/vllm-project/vllm/pull/25947) |
| 470 | vllm-ascend | PR | [BugFix] Fix ACLgraph bug in Qwen3_32b_int8 case | 🟡 中 | closed | 2025-09-26 | [#3204](https://github.com/vllm-project/vllm-ascend/pull/3204) |
| 471 | vllm-ascend | Issue | [Bug]: Qwen3-4b-instrcut-fp8 can not deploy | 🟡 中 | closed | 2025-09-26 | [#3197](https://github.com/vllm-project/vllm-ascend/issues/3197) |
| 472 | vllm-ascend | Issue | [Bug]: 请问是否支持Qwen2.5-32B-Instruct-GPTQ-Int8 模型？ | 🟡 中 | open | 2025-09-16 | [#2960](https://github.com/vllm-project/vllm-ascend/issues/2960) |
| 473 | vllm-ascend | Issue | [Bug]: vllm==v0.7.3使用昇腾量化版本，vllm serve --quantization ascend没有这个选项 | 🟡 中 | closed | 2025-09-09 | [#2819](https://github.com/vllm-project/vllm-ascend/issues/2819) |
| 474 | vllm | Issue | [Bug]: [FP4 gemm Runner] Failed to initialize cutlass FP4 gemm. | 🟡 中 | closed | 2025-09-06 | [#24377](https://github.com/vllm-project/vllm/issues/24377) |
| 475 | vllm | PR | [bug fix] disable memory pool to release unused `bf16` weights | 🟡 中 | open | 2025-08-29 | [#23875](https://github.com/vllm-project/vllm/pull/23875) |
| 476 | vllm | PR | [Bugfix] fix bf16 multimodal model hash | 🟡 中 | closed | 2025-08-26 | [#23623](https://github.com/vllm-project/vllm/pull/23623) |
| 477 | vllm-ascend | Issue | [Bug]: w8a8 性能无提升比bf16 | 🟡 中 | closed | 2025-08-25 | [#2514](https://github.com/vllm-project/vllm-ascend/issues/2514) |
| 478 | vllm | Issue | [Bug]: FP4 not leverage on RTX 6000 Pro (Blackwell) | 🟡 中 | open | 2025-08-24 | [#23497](https://github.com/vllm-project/vllm/issues/23497) |
| 479 | vllm | Issue | [Bug]: Can't run Qwen3-235B-A22B-Thinking-2507-FP4 NVFP4 model | 🟡 中 | closed | 2025-08-14 | [#22906](https://github.com/vllm-project/vllm/issues/22906) |
| 480 | vllm | Issue | [Bug]: On the V100-SXM2-32GB single-card machine, it is impossible to run Qwen3-30B-A3B-Instruct-2507-AWQ and Qwen3-30B-A3B-Instruct-2507-GPTQ-Int8 using vllm | 🟡 中 | closed | 2025-08-13 | [#22800](https://github.com/vllm-project/vllm/issues/22800) |
| 481 | vllm-ascend | Issue | [Bug]: How to deploy DeepSeek-R1-0528-BF16 on 910B 64G × 32 using DP | 🟡 中 | closed | 2025-08-12 | [#2344](https://github.com/vllm-project/vllm-ascend/issues/2344) |
| 482 | vllm | Issue | [Bug]: 1.7B fp16 + 0.6B draft OOM with gpu_memory_utilization=0.9, while 4B int8 + 0.6B works fine on A800 80 GB | 🟡 中 | closed | 2025-08-11 | [#22624](https://github.com/vllm-project/vllm/issues/22624) |
| 483 | vllm-ascend | PR | [Bugfix] Fix quantization patch bug | 🟡 中 | closed | 2025-08-04 | [#2200](https://github.com/vllm-project/vllm-ascend/pull/2200) |
| 484 | vllm-ascend | Issue | [Bug]: Cannot find bin of op AddRmsNormQuant, integral key 0/1/\|float16/ND/float16/ND/float16/ND/float16/ND/int8/ND/int8/ND/float16/ND/ | 🟡 中 | closed | 2025-08-04 | [#2197](https://github.com/vllm-project/vllm-ascend/issues/2197) |
| 485 | vllm-ascend | PR | [0.9.1][bugfix] fix bf16 multistream | 🟡 中 | closed | 2025-07-29 | [#2075](https://github.com/vllm-project/vllm-ascend/pull/2075) |
| 486 | vllm-ascend | Issue | [Bug]: DeepSeek-R1-bf16-hfd-w8a8 fails to start on 910B + vLLM 0.9.2rc1 with "RuntimeError: Engine core initialization failed. See root cause above" | 🟡 中 | closed | 2025-07-25 | [#2015](https://github.com/vllm-project/vllm-ascend/issues/2015) |
| 487 | vllm | PR | [Bug] Fix Compressed Tensor NVFP4 `cutlass_fp4_group_mm` illegal memory access | 🟡 中 | closed | 2025-07-23 | [#21465](https://github.com/vllm-project/vllm/pull/21465) |
| 488 | vllm | Issue | [Bug]: Compressed Tensor NVFP4 `cutlass_fp4_group_mm` illegal memory access | 🟡 中 | closed | 2025-07-22 | [#21399](https://github.com/vllm-project/vllm/issues/21399) |
| 489 | vllm | Issue | [Bug]: Performance Anomaly: compressed-tensors shows no speedup over BF16 on H100s on vLLM | 🟡 中 | closed | 2025-07-10 | [#20783](https://github.com/vllm-project/vllm/issues/20783) |
| 490 | vllm-ascend | PR | [0.9.1][BugFix] Fix the failure to recognize the actual type of quantization | 🟡 中 | closed | 2025-07-10 | [#1721](https://github.com/vllm-project/vllm-ascend/pull/1721) |
| 491 | vllm-ascend | PR | [Bugfix] Fix quantization patch bug | 🟡 中 | closed | 2025-06-28 | [#1495](https://github.com/vllm-project/vllm-ascend/pull/1495) |
| 492 | vllm-ascend | Issue | [Bug]: w8a8 quantization usage | 🟡 中 | closed | 2025-06-28 | [#1494](https://github.com/vllm-project/vllm-ascend/issues/1494) |
| 493 | vllm-ascend | Issue | [Bug]: vllm-ascend 0.7.3.post1 does not support w8a8 quantization, but 0.9.0rc2 does. | 🟡 中 | closed | 2025-06-20 | [#1329](https://github.com/vllm-project/vllm-ascend/issues/1329) |
| 494 | vllm | Issue | [Bug]: 5090 gemma-3-12b-it using FP8/INT8/FP16 quantization for conncurent requests DOCKER. | 🟡 中 | closed | 2025-06-19 | [#19863](https://github.com/vllm-project/vllm/issues/19863) |
| 495 | vllm | PR | [Bugfix] Enforce contiguous input for dynamic_per_token FP8/INT8 quant | 🟡 中 | closed | 2025-06-11 | [#19452](https://github.com/vllm-project/vllm/pull/19452) |
| 496 | vllm-ascend | Issue | [Bug][main]: vllm (main) serve failed due to quantization choice haven't init | 🟡 中 | closed | 2025-06-01 | [#1042](https://github.com/vllm-project/vllm-ascend/issues/1042) |
| 497 | vllm-ascend | PR | [Bugfix] Fix quantization cli | 🟡 中 | closed | 2025-05-29 | [#1005](https://github.com/vllm-project/vllm-ascend/pull/1005) |
| 498 | vllm-ascend | Issue | [Bug]:  不支持quantization为ascend的量化 | 🟡 中 | closed | 2025-05-20 | [#902](https://github.com/vllm-project/vllm-ascend/issues/902) |
| 499 | vllm | PR | [Bugfix] [ROCm]: Remove assertion logic when using AITER fused moe in unquantizedMethod to reenable LLama4 BF16 | 🟡 中 | closed | 2025-05-15 | [#18205](https://github.com/vllm-project/vllm/pull/18205) |
| 500 | vllm | Issue | [Bug]:Why is the GPU memory usage after quantizing the model to int8 W8A8 with llmcompressor almost the same as before quantization? | 🟡 中 | closed | 2025-04-22 | [#16959](https://github.com/vllm-project/vllm/issues/16959) |
| 501 | vllm | Issue | [Bug]:  An error occurred when deploying DeepSeek-R1-Channel-INT8 on two A100 machines using lws | 🟡 中 | closed | 2025-04-18 | [#16827](https://github.com/vllm-project/vllm/issues/16827) |
| 502 | vllm | PR | [Bugfix] Fix cutlass dispatch for fp8/int8 to properly invoke M&lt;=16 c… | 🟡 中 | closed | 2025-04-17 | [#16751](https://github.com/vllm-project/vllm/pull/16751) |
| 503 | vllm-ascend | Issue | [Bug]: How to enable 128K context length of DeepSeek-R1(BF16) with 32*910B(64GB) ? | 🟡 中 | closed | 2025-04-11 | [#508](https://github.com/vllm-project/vllm-ascend/issues/508) |
| 504 | vllm | Issue | [Bug]: DeepSeek-r1-AWQ (W4A16) can perform normal inference using BF16, but it shows abnormal behavior when using FP16. | 🟡 中 | closed | 2025-03-25 | [#15429](https://github.com/vllm-project/vllm/issues/15429) |
| 505 | vllm | Issue | [Bug]: int8 2:4 sparse time more than fp8 | 🟡 中 | closed | 2025-03-21 | [#15275](https://github.com/vllm-project/vllm/issues/15275) |
| 506 | vllm-ascend | PR | [BugFix] Fix bugs when using ascend quantization | 🟡 中 | closed | 2025-03-08 | [#275](https://github.com/vllm-project/vllm-ascend/pull/275) |
| 507 | vllm | Issue | [Bug][V1]: Loading Llama3.1-8B-INT8 gets OOM when using VLLM_USE_v1=1 but safe using v0 | 🟡 中 | closed | 2025-03-05 | [#14286](https://github.com/vllm-project/vllm/issues/14286) |
| 508 | vllm | Issue | [Bug]: Deepseek R1 671B int8 not working on TPU | 🟡 中 | closed | 2025-03-04 | [#14218](https://github.com/vllm-project/vllm/issues/14218) |
| 509 | vllm-ascend | Issue | Quantization error while running Deepseek-V3-w8a8 | 🟡 中 | closed | 2025-02-20 | [#119](https://github.com/vllm-project/vllm-ascend/issues/119) |
| 510 | vllm | PR | [Bugfix][AMD] Update torch_bindings so that scaled_fp4_quant isn't build on ROCm | 🟡 中 | closed | 2025-02-13 | [#13235](https://github.com/vllm-project/vllm/pull/13235) |
| 511 | vllm | Issue | [Bug]: deepseek-v3-bf16 only generates a null char ""! | 🟡 中 | closed | 2025-01-10 | [#11913](https://github.com/vllm-project/vllm/issues/11913) |
| 512 | vllm | Issue | [Bug]: Cutlass 2:4 Sparsity + FP8/Int8 Quant RuntimeError: Error Internal | 🟡 中 | closed | 2025-01-06 | [#11763](https://github.com/vllm-project/vllm/issues/11763) |
| 513 | vllm | Issue | [Bug]: Cutlass 2:4 Sparsity + FP8/Int8 Quant RuntimeError: Error Internal | 🟡 中 | closed | 2025-01-06 | [#11756](https://github.com/vllm-project/vllm/issues/11756) |
| 514 | vllm | Issue | [Bug]: [RuntimeError: CUDA error: unspecified launch failure  ]int8 w8a8 quantization data set to generate model data, an error occurred when changing the specified data set | 🟡 中 | closed | 2024-12-18 | [#11281](https://github.com/vllm-project/vllm/issues/11281) |
| 515 | vllm | Issue | [Bug]: Nonsensical Sentences Generated When Inferencing INT8 Quantized Qwen2.5-72B Model | 🟡 中 | closed | 2024-12-13 | [#11175](https://github.com/vllm-project/vllm/issues/11175) |
| 516 | vllm | PR | [bugfix] fix the default value of llm_int8_threshold in BitsAndBytesConfig | 🟡 中 | closed | 2024-11-26 | [#10657](https://github.com/vllm-project/vllm/pull/10657) |
| 517 | vllm | Issue | [Bug]: vllm infer for Qwen2-VL-72B-Instruct-GPTQ-Int8  | 🟡 中 | closed | 2024-11-26 | [#10650](https://github.com/vllm-project/vllm/issues/10650) |
| 518 | vllm | PR | [Bugfix] return zero point in static quantization in scaled_int8_quant | 🟡 中 | closed | 2024-11-13 | [#10292](https://github.com/vllm-project/vllm/pull/10292) |
| 519 | vllm | Issue | [Bug]: After 0.6.2 update to 0.6.3, INT8(W8A8) format cannot be loaded at all. No compiled cutlass_scaled_mm for a compute capability less than CUDA device capability: 75 | 🟡 中 | closed | 2024-10-16 | [#9419](https://github.com/vllm-project/vllm/issues/9419) |
| 520 | vllm | PR | [BugFix] [Kernel] Fix GPU SEGV occurring in int8 kernels | 🟡 中 | closed | 2024-10-15 | [#9391](https://github.com/vllm-project/vllm/pull/9391) |
| 521 | vllm | Issue | [Bug]: TPU single-host v5e-8  HBM OOM with Llama 3.1 70B and tpu_int8 quantization | 🟡 中 | closed | 2024-10-14 | [#9331](https://github.com/vllm-project/vllm/issues/9331) |
| 522 | vllm | Issue | [Bug]: MiniCPM-2B-dpo-bf16 output is not same as hugginface transformers | 🟡 中 | closed | 2024-08-01 | [#7029](https://github.com/vllm-project/vllm/issues/7029) |
| 523 | vllm | Issue | [Bug]: First input (bf16) and second input (uint8) must have the same dtype! | 🟡 中 | closed | 2024-07-29 | [#6884](https://github.com/vllm-project/vllm/issues/6884) |
| 524 | vllm | Issue | [Bug]: Unable to run meta-llama/Llama-Guard-3-8B-INT8 | 🟡 中 | closed | 2024-07-24 | [#6756](https://github.com/vllm-project/vllm/issues/6756) |
| 525 | vllm | PR | [Bugfix] Fix w8a8 benchmarks for int8 case | 🟡 中 | closed | 2024-06-18 | [#5643](https://github.com/vllm-project/vllm/pull/5643) |
| 526 | vllm | Issue | [BUG] Compile source code error for ROCM platform when using #include &lt;hip/hip_bf16.h&gt; | 🟡 中 | closed | 2024-02-02 | [#2725](https://github.com/vllm-project/vllm/issues/2725) |
| 527 | vllm | PR | [Bugfix] compressed-tensors: restore int8 grouped WNA16 MoE support | 🟢 低 | open | 2026-08-12 | [#52002](https://github.com/vllm-project/vllm/pull/52002) |
| 528 | vllm | PR | [ROCm][K3] Dequantize the fp8 decode query for MLA backends without quant-query support - TRITON_MLA | 🟢 低 | closed | 2026-08-11 | [#51860](https://github.com/vllm-project/vllm/pull/51860) |
| 529 | vllm | PR | [Test] Add ROCm AITER FP8 MLA prefill accuracy test | 🟢 低 | closed | 2026-08-07 | [#51457](https://github.com/vllm-project/vllm/pull/51457) |
| 530 | vllm | PR | [Bugfix][Model] Validate Kimi-K3 plain FP8 prefill support | 🟢 低 | closed | 2026-08-07 | [#51394](https://github.com/vllm-project/vllm/pull/51394) |
| 531 | vllm | PR | [CI] Add GSM8K accuracy test for amd/DeepSeek-V4-Flash-MXFP4 | 🟢 低 | open | 2026-07-31 | [#50632](https://github.com/vllm-project/vllm/pull/50632) |
| 532 | vllm | PR | [XPU] [BugFix] Add deepseek_v4_fp8 to xpu supported_quantization list | 🟢 低 | closed | 2026-07-30 | [#50434](https://github.com/vllm-project/vllm/pull/50434) |
| 533 | vllm | PR | [Perf][Quantization] Add opt-in NVFP4 load-time dequantization | 🟢 低 | closed | 2026-07-29 | [#50335](https://github.com/vllm-project/vllm/pull/50335) |
| 534 | vllm | PR | [Doc] Expand ModelOpt NVFP4 docs: hardware support, MoE serving, accuracy evaluation | 🟢 低 | open | 2026-07-15 | [#48782](https://github.com/vllm-project/vllm/pull/48782) |
| 535 | vllm | PR | [CI] Add ModelOpt NVFP4 EPLB accuracy test (Qwen3-30B-A3B, 2xB200) | 🟢 低 | open | 2026-07-14 | [#48618](https://github.com/vllm-project/vllm/pull/48618) |
| 536 | vllm | PR | [Bugfix] Support non-gated MoE in online quantization and Marlin MoE tile padding | 🟢 低 | closed | 2026-07-08 | [#48028](https://github.com/vllm-project/vllm/pull/48028) |
| 537 | vllm-ascend | PR | [v0.23.0][BugFix] Revert Add allgatherEP MXFP4 quantization (#11287) to fix w4a8mxfp break | 🟢 低 | closed | 2026-07-08 | [#11653](https://github.com/vllm-project/vllm-ascend/pull/11653) |
| 538 | vllm | PR | [Bugfix] Allow non-contiguous query in FlashInfer FP8 query quantization | 🟢 低 | closed | 2026-07-07 | [#47908](https://github.com/vllm-project/vllm/pull/47908) |
| 539 | vllm | PR | Add GPT-OSS BF16 expert remap regression test | 🟢 低 | open | 2026-07-06 | [#47721](https://github.com/vllm-project/vllm/pull/47721) |
| 540 | vllm | PR | [CI] Add GSM8K accuracy configs for large NVFP4/INT4 MoEs (4xB200) | 🟢 低 | open | 2026-06-30 | [#47171](https://github.com/vllm-project/vllm/pull/47171) |
| 541 | vllm | PR | [CI] Add GSM8K accuracy configs for Blackwell NVFP4/FP8 models | 🟢 低 | open | 2026-06-30 | [#47170](https://github.com/vllm-project/vllm/pull/47170) |
| 542 | vllm | PR | [Bugfix] compressed-tensors: allow int8 grouped WNA16 MoE on Marlin | 🟢 低 | closed | 2026-06-30 | [#47154](https://github.com/vllm-project/vllm/pull/47154) |
| 543 | vllm | PR | [Bugfix][Model] MiMo-V2: support TP &gt; num_kv_heads for the fused FP8 QKV projection | 🟢 低 | open | 2026-06-25 | [#46755](https://github.com/vllm-project/vllm/pull/46755) |
| 544 | vllm | PR | [Bugfix][DeepSeekV4] Add BF16 MTP O-proj fallback for unquantized draft weights | 🟢 低 | open | 2026-06-08 | [#44847](https://github.com/vllm-project/vllm/pull/44847) |
| 545 | vllm | PR | [ROCm][Bugfix] Add quantization compatibility guard for Fused Shared Expert in DeepSeek-V2/V3/Kimi-K2 | 🟢 低 | open | 2026-06-05 | [#44651](https://github.com/vllm-project/vllm/pull/44651) |
| 546 | vllm-ascend | PR | [BugFix][Quantization]: NPU MoE quantization methods support TP only. | 🟢 低 | closed | 2026-06-03 | [#9908](https://github.com/vllm-project/vllm-ascend/pull/9908) |
| 547 | vllm-ascend | PR | [BugFix][v0.20.2rc]: NPU MoE quantization methods support TP only. | 🟢 低 | closed | 2026-06-02 | [#9870](https://github.com/vllm-project/vllm-ascend/pull/9870) |
| 548 | vllm-ascend | PR | [Feature] Fix the precision issue of dsV4 caused by dequant_swiglu_quant | 🟢 低 | closed | 2026-05-26 | [#9600](https://github.com/vllm-project/vllm-ascend/pull/9600) |
| 549 | vllm-ascend | PR | [BugFix][310p]Handle null quantization config in ShardedStateLoader310&[Feature][310P] Support W8A8 dynamic linear method | 🟢 低 | closed | 2026-04-15 | [#8296](https://github.com/vllm-project/vllm-ascend/pull/8296) |
| 550 | vllm | PR | [Bugfix][ROCm]: Allow `gpt_oss_mxfp4` quantization method on rocm | 🟢 低 | closed | 2026-04-14 | [#39754](https://github.com/vllm-project/vllm/pull/39754) |
| 551 | vllm-ascend | PR | [Bugfix][Add]Qwen omni quantization bugfix and add Qwen2.5Omni quantization | 🟢 低 | closed | 2026-03-04 | [#6994](https://github.com/vllm-project/vllm-ascend/pull/6994) |
| 552 | vllm-ascend | PR | [Bugfix] Add quantization and Fix Qwen Omni quantization problem | 🟢 低 | closed | 2026-03-03 | [#6957](https://github.com/vllm-project/vllm-ascend/pull/6957) |
| 553 | vllm | PR | [ROCm] Add hardware detection for FP4 BMM to prevent MI300X crashes | 🟢 低 | closed | 2026-02-16 | [#34647](https://github.com/vllm-project/vllm/pull/34647) |
| 554 | vllm-ascend | PR | [main][Quantization][DFX] Add friendly error checks for quantization and weight dtype mismatch | 🟢 低 | closed | 2026-02-09 | [#6635](https://github.com/vllm-project/vllm-ascend/pull/6635) |
| 555 | vllm-ascend | PR | [EPLB][Bugfix] EPLB support fp/bf16 | 🟢 低 | closed | 2025-12-30 | [#5531](https://github.com/vllm-project/vllm-ascend/pull/5531) |
| 556 | vllm | PR | [Bugfix] Remove spurious NVFP4 'GPU does not support FP4' warning | 🟢 低 | closed | 2025-12-25 | [#31346](https://github.com/vllm-project/vllm/pull/31346) |
| 557 | vllm | PR | [Bugfix][Quantization] Support BF16 tensors on GGUF | 🟢 低 | closed | 2025-12-03 | [#29948](https://github.com/vllm-project/vllm/pull/29948) |
| 558 | vllm-ascend | PR | [Feat][BugFix]Support the Qwen3-Next-80B-A3B-Instruct quantization model&Fix the NZ issue | 🟢 低 | closed | 2025-11-18 | [#4245](https://github.com/vllm-project/vllm-ascend/pull/4245) |
| 559 | vllm | PR | Add the NV-ModelOPT FP8 & FP4 quantization E2E accuracy test case | 🟢 低 | closed | 2025-11-03 | [#27996](https://github.com/vllm-project/vllm/pull/27996) |
| 560 | vllm-ascend | PR | [Bugfix] Add quantization param for multi-node CI | 🟢 低 | closed | 2025-10-11 | [#3383](https://github.com/vllm-project/vllm-ascend/pull/3383) |
| 561 | vllm | PR | [ROCm][Quantization] extend AMD Quark to support mixed-precision quantized model | 🟢 低 | closed | 2025-09-04 | [#24239](https://github.com/vllm-project/vllm/pull/24239) |
| 562 | vllm | PR | [Feature][Quantization] Support Quark for mixed-precision quantized model | 🟢 低 | closed | 2025-09-01 | [#24040](https://github.com/vllm-project/vllm/pull/24040) |
| 563 | vllm-ascend | PR | [0.9.1-dev] [BugFix] add lm_head prefix to resolve Qwen3 DBO quantization issues | 🟢 低 | closed | 2025-08-08 | [#2292](https://github.com/vllm-project/vllm-ascend/pull/2292) |
| 564 | vllm-ascend | PR | [v0.9.1-dev] [BugFix] add lm_head prefix to resolve Qwen3 DBO quantization issues | 🟢 低 | closed | 2025-08-08 | [#2289](https://github.com/vllm-project/vllm-ascend/pull/2289) |
| 565 | vllm | PR | [Quantization]: Support compressed-tensors mixed-precision model loading | 🟢 低 | closed | 2025-08-07 | [#22468](https://github.com/vllm-project/vllm/pull/22468) |
| 566 | vllm | PR | Add accuracy test for SM100 Llama-4-Scout NVFP4 | 🟢 低 | closed | 2025-07-29 | [#21872](https://github.com/vllm-project/vllm/pull/21872) |
| 567 | vllm | PR | [BugFix] Support bf16 in zero-copy tensor serialization | 🟢 低 | closed | 2025-04-18 | [#16860](https://github.com/vllm-project/vllm/pull/16860) |
| 568 | vllm | PR | [AMD][Quantization] Add TritonScaledMMLinearKernel since int8 is broken for AMD | 🟢 低 | closed | 2025-01-21 | [#12282](https://github.com/vllm-project/vllm/pull/12282) |

</details>

---

## 2. 数值稳定性

计算过程中产生 NaN/Inf、发生溢出/下溢或除零、舍入误差累积等，导致数值退化。这类问题往往与长上下文、极端输入、特定算子组合（如 flash attention 的 softmax 缩放）相关，且 NaN 一旦产生会沿网络逐层传播。

**问题数: 237 条**

### 关键问题

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug]: Qwen3.5-9B hybrid-GDN + dynamic LoRA on H20 produces NaN output ("!" tokens) for long sequences — punica Triton kernel bug | 🔴 极高 | open | 2026-08-17 | [#52568](https://github.com/vllm-project/vllm/issues/52568) |
| 2 | vllm | Issue | [Bug]: Mistral-Small-3.1 FP8 (Pixtral) returns NaN on image inputs with compilation enabled; works with --enforce-eager | 🔴 极高 | closed | 2026-08-12 | [#52034](https://github.com/vllm-project/vllm/issues/52034) |
| 3 | vllm-ascend | PR | [Cherry-pick][releases/v0.26.0rc][BugFix][Triton] fix rejection sampler when target_logits are NaN (from #14098) | 🔴 极高 | open | 2026-08-12 | [#14120](https://github.com/vllm-project/vllm-ascend/pull/14120) |
| 4 | vllm-ascend | PR | [v0.23.0][BugFix] fix rejection sampler when target_logits are NaN | 🔴 极高 | closed | 2026-08-12 | [#14116](https://github.com/vllm-project/vllm-ascend/pull/14116) |
| 5 | vllm-ascend | PR | [BugFix][Triton] fix rejection sampler when target_logits are NaN | 🔴 极高 | open | 2026-08-12 | [#14098](https://github.com/vllm-project/vllm-ascend/pull/14098) |
| 6 | vllm-ascend | PR | [BugFix][Triton] Prevent zero-progress speculative decoding when recovery scores are NaN | 🔴 极高 | closed | 2026-08-11 | [#14043](https://github.com/vllm-project/vllm-ascend/pull/14043) |
| 7 | vllm-ascend | PR | [v0.25.1rc][BugFix][Triton] Prevent zero-progress speculative decoding when recovery scores are NaN | 🔴 极高 | closed | 2026-08-11 | [#14042](https://github.com/vllm-project/vllm-ascend/pull/14042) |
| 8 | vllm | PR | [Bugfix][V1] Fix silent -inf logprobs when logprob_token_ids shares a batch | 🔴 极高 | open | 2026-08-11 | [#51789](https://github.com/vllm-project/vllm/pull/51789) |
| 9 | vllm | Issue | [Bug]: [ROCm] v0.26.0 release image: NaN logits from AITER fused-MoE path on gfx942 (Qwen3.5-397B-A17B-FP8) — fixed on main, requesting 0.26.x backport | 🔴 极高 | open | 2026-08-09 | [#51580](https://github.com/vllm-project/vllm/issues/51580) |
| 10 | vllm | PR | [V1] Copy NaN-in-logits counts to host asynchronously | 🔴 极高 | closed | 2026-08-06 | [#51304](https://github.com/vllm-project/vllm/pull/51304) |
| 11 | vllm | Issue | [Model] Kimi-K3: all requests degenerate to a repeated token after long-context prefill (NaN logits; packed KDA prefill suspected) | 🔴 极高 | open | 2026-08-04 | [#51039](https://github.com/vllm-project/vllm/issues/51039) |
| 12 | vllm | PR | [Bugfix][Rust Frontend] Tolerate NaN-corrupted logprobs in engine-core | 🔴 极高 | open | 2026-08-04 | [#51026](https://github.com/vllm-project/vllm/pull/51026) |
| 13 | vllm | PR | Fix/engine core client nan logprobs rank 0 | 🔴 极高 | closed | 2026-08-04 | [#51025](https://github.com/vllm-project/vllm/pull/51025) |
| 14 | vllm | PR | [ROCm][Bugfix] Kimi-K3 Fix KDA NaN on mixed batches and racy autotune config | 🔴 极高 | closed | 2026-08-01 | [#50649](https://github.com/vllm-project/vllm/pull/50649) |
| 15 | vllm | Issue | [Bug]: FlexAttention paged K/V offsets overflow int32 once `num_gpu_blocks ≥ 2**31 / (block_size · 2 · num_kv_heads · head_size)` — crash *or* silent wrong output | 🔴 极高 | open | 2026-07-30 | [#50427](https://github.com/vllm-project/vllm/issues/50427) |
| 16 | vllm | PR | [Core] Add --enable-nan-fault-tolerance for NaN detection and request abort | 🔴 极高 | open | 2026-07-29 | [#50283](https://github.com/vllm-project/vllm/pull/50283) |
| 17 | vllm | PR | [Kernel] Harden top_k_per_row against NaN and under-filled output | 🔴 极高 | open | 2026-07-29 | [#50201](https://github.com/vllm-project/vllm/pull/50201) |
| 18 | vllm | PR | [Bugfix][Spec Decode] Fix NaN handling in rejection sampler tl.argmax | 🔴 极高 | closed | 2026-07-28 | [#50183](https://github.com/vllm-project/vllm/pull/50183) |
| 19 | vllm | PR | [Bugfix] Route DSv4 sparse-indexer prefill top-k around NaN-broken kernel path on SM12x | 🔴 极高 | open | 2026-07-26 | [#49897](https://github.com/vllm-project/vllm/pull/49897) |
| 20 | vllm | Issue | [Bug] DeepSeek-V4 on SM12x: NaN MQA logits drive top_k_per_row_prefill to emit uninitialized smem as indices -&gt; illegal memory access | 🔴 极高 | open | 2026-07-26 | [#49896](https://github.com/vllm-project/vllm/issues/49896) |
| 21 | vllm | Issue | [Bug]: LoRA `lora_expand` Triton kernel outputs NaN on Hopper (sm_90) with `block_n=128`, producing garbled LoRA output | 🔴 极高 | open | 2026-07-14 | [#48590](https://github.com/vllm-project/vllm/issues/48590) |
| 22 | vllm | PR | [Bugfix][Frontend] Handle None/NaN/Inf logprob values when using FP8 quantization | 🔴 极高 | open | 2026-07-14 | [#48585](https://github.com/vllm-project/vllm/pull/48585) |
| 23 | vllm | PR | [Bugfix] Prevent NaN poisoning in xpu_mla_sparse for fully-masked index chunks | 🔴 极高 | closed | 2026-07-11 | [#48366](https://github.com/vllm-project/vllm/pull/48366) |
| 24 | vllm | Issue | [Bug]: xpu_mla_sparse NaN-poisons attention output when a row's leading topk index chunk is fully masked | 🔴 极高 | closed | 2026-07-11 | [#48364](https://github.com/vllm-project/vllm/issues/48364) |
| 25 | vllm | Issue | [Bug]: Gemma4 Unified image requests produce all-NaN logits after BF16-to-FP16 fallback | 🔴 极高 | open | 2026-07-10 | [#48231](https://github.com/vllm-project/vllm/issues/48231) |
| 26 | vllm | PR | [Bugfix] Patch Hopper MXFP4 OOB scales reads leading to NaN | 🔴 极高 | closed | 2026-07-07 | [#47910](https://github.com/vllm-project/vllm/pull/47910) |
| 27 | vllm-ascend | Issue | [Bug]: Rfork第二次实例启动：FusedMoE shared experts split computation does not match the integrated computation.max absolute difference:nan integrated output-sum:nan,norm:nan split output-sum:nan,norm:nan | 🔴 极高 | closed | 2026-07-03 | [#11409](https://github.com/vllm-project/vllm-ascend/issues/11409) |
| 28 | vllm | PR | Fix OOB scale reads in Triton MXFP4 matmul kernel causing NaN outputs | 🔴 极高 | open | 2026-07-01 | [#47323](https://github.com/vllm-project/vllm/pull/47323) |
| 29 | vllm | PR | fix(security): prevent infinite loop in split_audio with NaN audio sa… | 🔴 极高 | closed | 2026-06-23 | [#46463](https://github.com/vllm-project/vllm/pull/46463) |
| 30 | vllm-ascend | PR | [BugFix]Fix moe allgather NaN issue | 🔴 极高 | closed | 2026-06-17 | [#10579](https://github.com/vllm-project/vllm-ascend/pull/10579) |
| 31 | vllm | PR | [BUG] fix hidden states nan for hybrid attention models | 🔴 极高 | closed | 2026-06-16 | [#45849](https://github.com/vllm-project/vllm/pull/45849) |
| 32 | vllm | Issue | [Bug]: ExampleHiddenStatesConnector returns nan for hybrid attention model | 🔴 极高 | closed | 2026-06-15 | [#45734](https://github.com/vllm-project/vllm/issues/45734) |
| 33 | vllm | PR | [Bugfix] Complete one-shot fused all-reduce PDL at end to avoid NaN | 🔴 极高 | closed | 2026-06-12 | [#45448](https://github.com/vllm-project/vllm/pull/45448) |
| 34 | vllm | Issue | [Bug] NVFP4 MoE: missing per-expert input_scale keys load as silent zeros -&gt; 1/0 = inf gscale -&gt; NaN/pad-token output | 🔴 极高 | open | 2026-06-11 | [#45212](https://github.com/vllm-project/vllm/issues/45212) |
| 35 | vllm-ascend | PR | [Attention][BugFix] Fix NaN issue in Lightning Attention NPU kernel caused by unmasked padding | 🔴 极高 | closed | 2026-06-10 | [#10276](https://github.com/vllm-project/vllm-ascend/pull/10276) |
| 36 | vllm | PR | [Bugfix]Fix out-of-vocabulary recovered token on all-NaN logits (root cause of empty spec-decode output) | 🔴 极高 | closed | 2026-06-09 | [#45060](https://github.com/vllm-project/vllm/pull/45060) |
| 37 | vllm-ascend | Issue | [Bug]: PD disaggregated SWA KV transfer can include stale blocks and produce NaN hidden states | 🔴 极高 | closed | 2026-06-09 | [#10253](https://github.com/vllm-project/vllm-ascend/issues/10253) |
| 38 | vllm | Issue | [RFC]: vLLM NaN Reporting | 🔴 极高 | open | 2026-06-01 | [#44211](https://github.com/vllm-project/vllm/issues/44211) |
| 39 | vllm-ascend | PR | [BugFix][MLA][Ascend950] Zero-init o_proj_input padding to prevent NaN propagation in quantization | 🔴 极高 | closed | 2026-05-29 | [#9687](https://github.com/vllm-project/vllm-ascend/pull/9687) |
| 40 | vllm-ascend | PR | [BugFix][Model] Fix NaN hidden states and low MTP acceptance rate when FlashComm1 is enabled | 🔴 极高 | closed | 2026-05-21 | [#9436](https://github.com/vllm-project/vllm-ascend/pull/9436) |

### 关键规律与分析

1. **NaN/Inf 一旦产生会沿网络逐层传播**，最终表现为 all-NaN 输出或 garbage；定位需回溯首个异常层。
2. **高发触发条件**：长上下文跨越切分边界（如 2048/4096 token）、attention softmax 缩放、RoPE 旋转、MoE expert 路由、除零（scale=0）、CUDA Graph 冷路径 replay。
3. **OOB（越界）读取 scale 值**是数值异常的常见根因（如 Triton block 量化 MoE 越界读 scale 产生 NaN）。
4. **"静默"性**使这类问题最危险：很多 case 不崩溃、不报错，只在特定输入/长度下才爆发。

<details>
<summary>展开全部 237 条</summary>

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug]: Qwen3.5-9B hybrid-GDN + dynamic LoRA on H20 produces NaN output ("!" tokens) for long sequences — punica Triton kernel bug | 🔴 极高 | open | 2026-08-17 | [#52568](https://github.com/vllm-project/vllm/issues/52568) |
| 2 | vllm | Issue | [Bug]: Mistral-Small-3.1 FP8 (Pixtral) returns NaN on image inputs with compilation enabled; works with --enforce-eager | 🔴 极高 | closed | 2026-08-12 | [#52034](https://github.com/vllm-project/vllm/issues/52034) |
| 3 | vllm-ascend | PR | [Cherry-pick][releases/v0.26.0rc][BugFix][Triton] fix rejection sampler when target_logits are NaN (from #14098) | 🔴 极高 | open | 2026-08-12 | [#14120](https://github.com/vllm-project/vllm-ascend/pull/14120) |
| 4 | vllm-ascend | PR | [v0.23.0][BugFix] fix rejection sampler when target_logits are NaN | 🔴 极高 | closed | 2026-08-12 | [#14116](https://github.com/vllm-project/vllm-ascend/pull/14116) |
| 5 | vllm-ascend | PR | [BugFix][Triton] fix rejection sampler when target_logits are NaN | 🔴 极高 | open | 2026-08-12 | [#14098](https://github.com/vllm-project/vllm-ascend/pull/14098) |
| 6 | vllm-ascend | PR | [BugFix][Triton] Prevent zero-progress speculative decoding when recovery scores are NaN | 🔴 极高 | closed | 2026-08-11 | [#14043](https://github.com/vllm-project/vllm-ascend/pull/14043) |
| 7 | vllm-ascend | PR | [v0.25.1rc][BugFix][Triton] Prevent zero-progress speculative decoding when recovery scores are NaN | 🔴 极高 | closed | 2026-08-11 | [#14042](https://github.com/vllm-project/vllm-ascend/pull/14042) |
| 8 | vllm | PR | [Bugfix][V1] Fix silent -inf logprobs when logprob_token_ids shares a batch | 🔴 极高 | open | 2026-08-11 | [#51789](https://github.com/vllm-project/vllm/pull/51789) |
| 9 | vllm | Issue | [Bug]: [ROCm] v0.26.0 release image: NaN logits from AITER fused-MoE path on gfx942 (Qwen3.5-397B-A17B-FP8) — fixed on main, requesting 0.26.x backport | 🔴 极高 | open | 2026-08-09 | [#51580](https://github.com/vllm-project/vllm/issues/51580) |
| 10 | vllm | PR | [V1] Copy NaN-in-logits counts to host asynchronously | 🔴 极高 | closed | 2026-08-06 | [#51304](https://github.com/vllm-project/vllm/pull/51304) |
| 11 | vllm | Issue | [Model] Kimi-K3: all requests degenerate to a repeated token after long-context prefill (NaN logits; packed KDA prefill suspected) | 🔴 极高 | open | 2026-08-04 | [#51039](https://github.com/vllm-project/vllm/issues/51039) |
| 12 | vllm | PR | [Bugfix][Rust Frontend] Tolerate NaN-corrupted logprobs in engine-core | 🔴 极高 | open | 2026-08-04 | [#51026](https://github.com/vllm-project/vllm/pull/51026) |
| 13 | vllm | PR | Fix/engine core client nan logprobs rank 0 | 🔴 极高 | closed | 2026-08-04 | [#51025](https://github.com/vllm-project/vllm/pull/51025) |
| 14 | vllm | PR | [ROCm][Bugfix] Kimi-K3 Fix KDA NaN on mixed batches and racy autotune config | 🔴 极高 | closed | 2026-08-01 | [#50649](https://github.com/vllm-project/vllm/pull/50649) |
| 15 | vllm | Issue | [Bug]: FlexAttention paged K/V offsets overflow int32 once `num_gpu_blocks ≥ 2**31 / (block_size · 2 · num_kv_heads · head_size)` — crash *or* silent wrong output | 🔴 极高 | open | 2026-07-30 | [#50427](https://github.com/vllm-project/vllm/issues/50427) |
| 16 | vllm | PR | [Core] Add --enable-nan-fault-tolerance for NaN detection and request abort | 🔴 极高 | open | 2026-07-29 | [#50283](https://github.com/vllm-project/vllm/pull/50283) |
| 17 | vllm | PR | [Kernel] Harden top_k_per_row against NaN and under-filled output | 🔴 极高 | open | 2026-07-29 | [#50201](https://github.com/vllm-project/vllm/pull/50201) |
| 18 | vllm | PR | [Bugfix][Spec Decode] Fix NaN handling in rejection sampler tl.argmax | 🔴 极高 | closed | 2026-07-28 | [#50183](https://github.com/vllm-project/vllm/pull/50183) |
| 19 | vllm | PR | [Bugfix] Route DSv4 sparse-indexer prefill top-k around NaN-broken kernel path on SM12x | 🔴 极高 | open | 2026-07-26 | [#49897](https://github.com/vllm-project/vllm/pull/49897) |
| 20 | vllm | Issue | [Bug] DeepSeek-V4 on SM12x: NaN MQA logits drive top_k_per_row_prefill to emit uninitialized smem as indices -&gt; illegal memory access | 🔴 极高 | open | 2026-07-26 | [#49896](https://github.com/vllm-project/vllm/issues/49896) |
| 21 | vllm | Issue | [Bug]: LoRA `lora_expand` Triton kernel outputs NaN on Hopper (sm_90) with `block_n=128`, producing garbled LoRA output | 🔴 极高 | open | 2026-07-14 | [#48590](https://github.com/vllm-project/vllm/issues/48590) |
| 22 | vllm | PR | [Bugfix][Frontend] Handle None/NaN/Inf logprob values when using FP8 quantization | 🔴 极高 | open | 2026-07-14 | [#48585](https://github.com/vllm-project/vllm/pull/48585) |
| 23 | vllm | PR | [Bugfix] Prevent NaN poisoning in xpu_mla_sparse for fully-masked index chunks | 🔴 极高 | closed | 2026-07-11 | [#48366](https://github.com/vllm-project/vllm/pull/48366) |
| 24 | vllm | Issue | [Bug]: xpu_mla_sparse NaN-poisons attention output when a row's leading topk index chunk is fully masked | 🔴 极高 | closed | 2026-07-11 | [#48364](https://github.com/vllm-project/vllm/issues/48364) |
| 25 | vllm | Issue | [Bug]: Gemma4 Unified image requests produce all-NaN logits after BF16-to-FP16 fallback | 🔴 极高 | open | 2026-07-10 | [#48231](https://github.com/vllm-project/vllm/issues/48231) |
| 26 | vllm | PR | [Bugfix] Patch Hopper MXFP4 OOB scales reads leading to NaN | 🔴 极高 | closed | 2026-07-07 | [#47910](https://github.com/vllm-project/vllm/pull/47910) |
| 27 | vllm-ascend | Issue | [Bug]: Rfork第二次实例启动：FusedMoE shared experts split computation does not match the integrated computation.max absolute difference:nan integrated output-sum:nan,norm:nan split output-sum:nan,norm:nan | 🔴 极高 | closed | 2026-07-03 | [#11409](https://github.com/vllm-project/vllm-ascend/issues/11409) |
| 28 | vllm | PR | Fix OOB scale reads in Triton MXFP4 matmul kernel causing NaN outputs | 🔴 极高 | open | 2026-07-01 | [#47323](https://github.com/vllm-project/vllm/pull/47323) |
| 29 | vllm | PR | fix(security): prevent infinite loop in split_audio with NaN audio sa… | 🔴 极高 | closed | 2026-06-23 | [#46463](https://github.com/vllm-project/vllm/pull/46463) |
| 30 | vllm-ascend | PR | [BugFix]Fix moe allgather NaN issue | 🔴 极高 | closed | 2026-06-17 | [#10579](https://github.com/vllm-project/vllm-ascend/pull/10579) |
| 31 | vllm | PR | [BUG] fix hidden states nan for hybrid attention models | 🔴 极高 | closed | 2026-06-16 | [#45849](https://github.com/vllm-project/vllm/pull/45849) |
| 32 | vllm | Issue | [Bug]: ExampleHiddenStatesConnector returns nan for hybrid attention model | 🔴 极高 | closed | 2026-06-15 | [#45734](https://github.com/vllm-project/vllm/issues/45734) |
| 33 | vllm | PR | [Bugfix] Complete one-shot fused all-reduce PDL at end to avoid NaN | 🔴 极高 | closed | 2026-06-12 | [#45448](https://github.com/vllm-project/vllm/pull/45448) |
| 34 | vllm | Issue | [Bug] NVFP4 MoE: missing per-expert input_scale keys load as silent zeros -&gt; 1/0 = inf gscale -&gt; NaN/pad-token output | 🔴 极高 | open | 2026-06-11 | [#45212](https://github.com/vllm-project/vllm/issues/45212) |
| 35 | vllm-ascend | PR | [Attention][BugFix] Fix NaN issue in Lightning Attention NPU kernel caused by unmasked padding | 🔴 极高 | closed | 2026-06-10 | [#10276](https://github.com/vllm-project/vllm-ascend/pull/10276) |
| 36 | vllm | PR | [Bugfix]Fix out-of-vocabulary recovered token on all-NaN logits (root cause of empty spec-decode output) | 🔴 极高 | closed | 2026-06-09 | [#45060](https://github.com/vllm-project/vllm/pull/45060) |
| 37 | vllm-ascend | Issue | [Bug]: PD disaggregated SWA KV transfer can include stale blocks and produce NaN hidden states | 🔴 极高 | closed | 2026-06-09 | [#10253](https://github.com/vllm-project/vllm-ascend/issues/10253) |
| 38 | vllm | Issue | [RFC]: vLLM NaN Reporting | 🔴 极高 | open | 2026-06-01 | [#44211](https://github.com/vllm-project/vllm/issues/44211) |
| 39 | vllm-ascend | PR | [BugFix][MLA][Ascend950] Zero-init o_proj_input padding to prevent NaN propagation in quantization | 🔴 极高 | closed | 2026-05-29 | [#9687](https://github.com/vllm-project/vllm-ascend/pull/9687) |
| 40 | vllm-ascend | PR | [BugFix][Model] Fix NaN hidden states and low MTP acceptance rate when FlashComm1 is enabled | 🔴 极高 | closed | 2026-05-21 | [#9436](https://github.com/vllm-project/vllm-ascend/pull/9436) |
| 41 | vllm | PR | Nan harness 58c959a | 🔴 极高 | closed | 2026-05-19 | [#43075](https://github.com/vllm-project/vllm/pull/43075) |
| 42 | vllm | PR | Resolve silu mul quant padded NaN corruption correctness | 🔴 极高 | open | 2026-05-18 | [#42984](https://github.com/vllm-project/vllm/pull/42984) |
| 43 | vllm | PR | [Bugfix] Clamp NVFP4 MoE activation scales to prevent NaN from dead experts | 🔴 极高 | open | 2026-05-14 | [#42601](https://github.com/vllm-project/vllm/pull/42601) |
| 44 | vllm | PR | [Bugfix] Handle NaN in QuantFP8 Native Forward | 🔴 极高 | open | 2026-04-30 | [#41427](https://github.com/vllm-project/vllm/pull/41427) |
| 45 | vllm | PR | fix: zero DP padding via torch.where to prevent NaN propagation into MoE | 🔴 极高 | closed | 2026-04-29 | [#41249](https://github.com/vllm-project/vllm/pull/41249) |
| 46 | vllm | PR | test: add nan/inf clamp regression test for fused_topk_bias | 🔴 极高 | closed | 2026-04-21 | [#40553](https://github.com/vllm-project/vllm/pull/40553) |
| 47 | vllm | PR | [Bugfix][CPU][RISC-V] Clamp exp() input to prevent NaN | 🔴 极高 | closed | 2026-04-21 | [#40428](https://github.com/vllm-project/vllm/pull/40428) |
| 48 | vllm | PR | fix: clamp NaN/Inf in topk_softmax to prevent duplicate expert IDs | 🔴 极高 | closed | 2026-04-09 | [#39391](https://github.com/vllm-project/vllm/pull/39391) |
| 49 | vllm | PR | [BugFix][Attention] Fix NaN in Triton merge_attn_states when both LSEs are -inf | 🔴 极高 | closed | 2026-04-07 | [#39148](https://github.com/vllm-project/vllm/pull/39148) |
| 50 | vllm | PR | [Bugfix] Fix NaN corruption from CUDA graph padding in NVFP4 models | 🔴 极高 | closed | 2026-03-28 | [#38436](https://github.com/vllm-project/vllm/pull/38436) |
| 51 | vllm | PR | [Bugfix] Revert "Zero-init MLA attention output buffers to prevent NaN from CUDA graph padding" | 🔴 极高 | closed | 2026-03-27 | [#38359](https://github.com/vllm-project/vllm/pull/38359) |
| 52 | vllm | PR | Fix NaN from stale FP4 scale padding in create_fp4_scale_tensor | 🔴 极高 | closed | 2026-03-25 | [#38148](https://github.com/vllm-project/vllm/pull/38148) |
| 53 | vllm | PR | [Bugfix] Preserve CUDA arch suffix (a/f) for SM12x — fixes NVFP4 NaN on desktop Blackwell | 🔴 极高 | closed | 2026-03-20 | [#37725](https://github.com/vllm-project/vllm/pull/37725) |
| 54 | vllm | PR | [Bugfix] Zero-init NVFP4 padding scales to prevent NaN contamination | 🔴 极高 | closed | 2026-03-19 | [#37564](https://github.com/vllm-project/vllm/pull/37564) |
| 55 | vllm | PR | [Bugfix] Zero-init MLA attention output buffers to prevent NaN from CUDA graph padding | 🔴 极高 | closed | 2026-03-18 | [#37442](https://github.com/vllm-project/vllm/pull/37442) |
| 56 | vllm | PR | FlashInfer NVFP4 NaN propagation plausible fix | 🔴 极高 | closed | 2026-03-17 | [#37356](https://github.com/vllm-project/vllm/pull/37356) |
| 57 | vllm | PR | [ROCm][Bugfix] Fix NaN corruption | 🔴 极高 | closed | 2026-03-10 | [#36709](https://github.com/vllm-project/vllm/pull/36709) |
| 58 | vllm | PR | Fix: Clone NVFP4 MoE weights on SM121 to prevent Marlin kernel NaN | 🔴 极高 | open | 2026-03-05 | [#36183](https://github.com/vllm-project/vllm/pull/36183) |
| 59 | vllm | Issue | [CI] Ultravox audio model HuggingFace reference produces invalid output with NaN logprobs | 🔴 极高 | closed | 2026-02-23 | [#35140](https://github.com/vllm-project/vllm/issues/35140) |
| 60 | vllm | PR | [DO NOT MERGE ]  Evidence for FlashInfer allreduce_fusion one-shot (kARResidualRMSNorm) causes deterministic NaN corruption and GSM8K collapse | 🔴 极高 | closed | 2026-02-12 | [#34412](https://github.com/vllm-project/vllm/pull/34412) |
| 61 | vllm | PR | [Bugfix]fix output Nan/Inf in marlin if dtype=float16 | 🔴 极高 | closed | 2026-02-06 | [#33972](https://github.com/vllm-project/vllm/pull/33972) |
| 62 | vllm-ascend | Issue | [Bug] GME Model: NaN Values & Precision Overflow | 🔴 极高 | closed | 2026-01-28 | [#6332](https://github.com/vllm-project/vllm-ascend/issues/6332) |
| 63 | vllm | Issue | [Bug]: NaN's in MLA with chunked-prefill | 🔴 极高 | closed | 2025-10-24 | [#27491](https://github.com/vllm-project/vllm/issues/27491) |
| 64 | vllm | PR | [Bugfix] fix apply_temperature to avoid nan in probs | 🔴 极高 | closed | 2025-09-12 | [#24734](https://github.com/vllm-project/vllm/pull/24734) |
| 65 | vllm | PR | [Structured Outputs] [Bug] Fix misalignment in apply_grammar_bitmask causing unintended masking and NaN logits | 🔴 极高 | closed | 2025-08-15 | [#22963](https://github.com/vllm-project/vllm/pull/22963) |
| 66 | vllm | Issue | [Bug]: apply_temperature may cause nan in probs | 🔴 极高 | closed | 2025-08-04 | [#22180](https://github.com/vllm-project/vllm/issues/22180) |
| 67 | vllm | Issue | [Bug]: GLM-4-32B-0414-FP8 output !!!!! error (tensor is nan) | 🔴 极高 | closed | 2025-04-25 | [#17154](https://github.com/vllm-project/vllm/issues/17154) |
| 68 | vllm | PR | [V1][Spec Decode] Avoid logging useless nan metrics | 🔴 极高 | closed | 2025-04-03 | [#16023](https://github.com/vllm-project/vllm/pull/16023) |
| 69 | vllm | Issue | [Bug]: External Launcher producing NaN outputs on Large Models when Collocating with Model Training | 🔴 极高 | closed | 2025-03-07 | [#14443](https://github.com/vllm-project/vllm/issues/14443) |
| 70 | vllm | PR | [Misc] Allow for unsigned zero NAN representation in ScalarType | 🔴 极高 | closed | 2024-08-19 | [#7661](https://github.com/vllm-project/vllm/pull/7661) |
| 71 | vllm | Issue | [Bug]: Processed prompts:   5%\|▌         \| 429/8535 [00:27&lt;08:37, 15.68it/s] RuntimeError: probability tensor contains either `inf`, `nan` or element &lt; 0 | 🔴 极高 | closed | 2024-04-17 | [#4151](https://github.com/vllm-project/vllm/issues/4151) |
| 72 | vllm | Issue | NaN in running quantized MoE model (TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ) | 🔴 极高 | closed | 2024-01-05 | [#2359](https://github.com/vllm-project/vllm/issues/2359) |
| 73 | vllm | Issue | There is NAN After "xops.memory_efficient_attention_forward" | 🔴 极高 | closed | 2023-12-13 | [#2078](https://github.com/vllm-project/vllm/issues/2078) |
| 74 | vllm | Issue | RuntimeError: probability tensor contains either inf, nan or element &lt; 0, when using Falcon 7B with vLLM | 🔴 极高 | closed | 2023-12-12 | [#2063](https://github.com/vllm-project/vllm/issues/2063) |
| 75 | vllm | Issue | RuntimeError: probability tensor contains either `inf`, `nan` or element &lt; 0 | 🔴 极高 | closed | 2023-12-12 | [#2053](https://github.com/vllm-project/vllm/issues/2053) |
| 76 | vllm | Issue | RuntimeError: probability tensor contains either `inf`, `nan` or element &lt; 0 | 🔴 极高 | closed | 2023-12-07 | [#1952](https://github.com/vllm-project/vllm/issues/1952) |
| 77 | vllm | Issue | Top p or temperature == 0.0001 RuntimeError: probability tensor contains either `inf`, `nan` or element &lt; 0 | 🔴 极高 | closed | 2023-11-10 | [#1623](https://github.com/vllm-project/vllm/issues/1623) |
| 78 | vllm | Issue | “RuntimeError: probability tensor contains either `inf`, `nan` or element &lt; 0” when use llama2-70B  | 🔴 极高 | closed | 2023-10-23 | [#1448](https://github.com/vllm-project/vllm/issues/1448) |
| 79 | vllm | Issue | [BUG]: NaN issue again after commit e67b4f2 | 🔴 极高 | closed | 2023-09-22 | [#1134](https://github.com/vllm-project/vllm/issues/1134) |
| 80 | vllm | Issue | RuntimeError: probability tensor contains either `inf`, `nan` or element &lt; 0 | 🔴 极高 | closed | 2023-08-02 | [#641](https://github.com/vllm-project/vllm/issues/641) |
| 81 | vllm | Issue | RuntimeError: probability tensor contains either `inf`, `nan` or element &lt; 0 when running mpt-7b | 🔴 极高 | closed | 2023-07-04 | [#363](https://github.com/vllm-project/vllm/issues/363) |
| 82 | vllm | Issue | [Bug] XPU qnorm/rope kernel: int32 overflow in the address computation past 2^31 q elements | 🔴 高 | open | 2026-08-15 | [#52416](https://github.com/vllm-project/vllm/issues/52416) |
| 83 | vllm | PR | [Bugfix] Handle persistent top-k candidate overflow | 🔴 高 | open | 2026-08-13 | [#52149](https://github.com/vllm-project/vllm/pull/52149) |
| 84 | vllm | PR | Revert KV block zeroing generalization due to ROCm launch overflow | 🔴 高 | closed | 2026-08-12 | [#52062](https://github.com/vllm-project/vllm/pull/52062) |
| 85 | vllm | Issue | [Bug]: stack overflow causing system crash: Rust unbounded recursion in the Gemma4 unified parser | 🔴 高 | open | 2026-08-03 | [#50927](https://github.com/vllm-project/vllm/issues/50927) |
| 86 | vllm | Issue | FlashInfer MLA decode workspace buffer overflow with decode-context-parallel | 🔴 高 | open | 2026-08-02 | [#50781](https://github.com/vllm-project/vllm/issues/50781) |
| 87 | vllm | PR | [Bugfix][XPU] Fix Mamba state pointer overflow | 🔴 高 | closed | 2026-07-31 | [#50552](https://github.com/vllm-project/vllm/pull/50552) |
| 88 | vllm | PR | [Bugfix][Kernel] Fix integer overflow in libtorch_stable/activation_kernels.cu | 🔴 高 | closed | 2026-07-24 | [#49660](https://github.com/vllm-project/vllm/pull/49660) |
| 89 | vllm | PR | [Bugfix][Hardware][AMD] Handle AITER unified-attention LDS overflow with Triton fallback | 🔴 高 | open | 2026-07-21 | [#49264](https://github.com/vllm-project/vllm/pull/49264) |
| 90 | vllm-ascend | PR | [CI] Set HCCL_IF_BASE_PORT for 560T cluster to avoid port overflow | 🔴 高 | closed | 2026-07-18 | [#12303](https://github.com/vllm-project/vllm-ascend/pull/12303) |
| 91 | vllm | PR | [Bugfix] Fix int32 address overflow in DeepGEMM ep_gather Triton kernel | 🔴 高 | open | 2026-07-12 | [#48398](https://github.com/vllm-project/vllm/pull/48398) |
| 92 | vllm | PR | [BugFix] Fix `num_output_placeholders` preemption underflow | 🔴 高 | closed | 2026-07-10 | [#48245](https://github.com/vllm-project/vllm/pull/48245) |
| 93 | vllm | PR | [Bugfix][XPU] Fix Mamba state pointer overflow | 🔴 高 | open | 2026-07-09 | [#48109](https://github.com/vllm-project/vllm/pull/48109) |
| 94 | vllm | Issue | [Bug][XPU] Mamba align-mode prefix caching crashes: "Overflow when unpacking long long" storing state.data_ptr() | 🔴 高 | open | 2026-07-09 | [#48059](https://github.com/vllm-project/vllm/issues/48059) |
| 95 | vllm | PR | [Bugfix] Fix int32 offset overflow in rejection sampler kernels | 🔴 高 | open | 2026-07-08 | [#48055](https://github.com/vllm-project/vllm/pull/48055) |
| 96 | vllm | PR | [Bugfix][Core] Discard in-flight async output frames on preemption to fix num_output_placeholders underflow | 🔴 高 | closed | 2026-07-07 | [#47900](https://github.com/vllm-project/vllm/pull/47900) |
| 97 | vllm-ascend | PR | [BugFix] Avoid scatter overflow for large slot mappings | 🔴 高 | closed | 2026-07-06 | [#11480](https://github.com/vllm-project/vllm-ascend/pull/11480) |
| 98 | vllm | PR | [Bugfix] Fix int32 overflow in triton_decode_attention page offsets | 🔴 高 | closed | 2026-07-06 | [#47671](https://github.com/vllm-project/vllm/pull/47671) |
| 99 | vllm | PR | [Bugfix][Model Runner V2][Spec Decode] Fix int32 offset overflow in block verification kernels | 🔴 高 | closed | 2026-07-02 | [#47383](https://github.com/vllm-project/vllm/pull/47383) |
| 100 | vllm | PR | [Bugfix][Model Runner V2][Spec Decode] Fix int32 offset overflow in sampler kernels | 🔴 高 | closed | 2026-06-24 | [#46560](https://github.com/vllm-project/vllm/pull/46560) |
| 101 | vllm | PR | [Bugfix] Resample long audio in blocks to avoid int32 AudioFrame overflow | 🔴 高 | open | 2026-06-22 | [#46377](https://github.com/vllm-project/vllm/pull/46377) |
| 102 | vllm | PR | [Multimodal] fix: resolve memory allocation and buffer overflow in audio resampling | 🔴 高 | open | 2026-06-22 | [#46375](https://github.com/vllm-project/vllm/pull/46375) |
| 103 | vllm | Issue | [Bug]: Transcribing long audio that needs resampling fails — resample_audio_pyav overflows the int32 AudioFrame buffer, masked as "Invalid or unsupported audio file" | 🔴 高 | open | 2026-06-22 | [#46364](https://github.com/vllm-project/vllm/issues/46364) |
| 104 | vllm | PR | [Bugfix][Core] Fix num_output_placeholders underflow with async scheduling + spec decode | 🔴 高 | closed | 2026-06-18 | [#46066](https://github.com/vllm-project/vllm/pull/46066) |
| 105 | vllm | PR | [Bugfix][Kernel][ROCm] Fix Wave32 LDS overflow in top-k merge launch | 🔴 高 | open | 2026-06-18 | [#46012](https://github.com/vllm-project/vllm/pull/46012) |
| 106 | vllm | PR | fix(moe_wna16): prevent int32 overflow in output index for large token counts | 🔴 高 | open | 2026-06-17 | [#45907](https://github.com/vllm-project/vllm/pull/45907) |
| 107 | vllm | Issue | [Bug]: Integer overflow in moe_wna16.cu | 🔴 高 | open | 2026-06-17 | [#45884](https://github.com/vllm-project/vllm/issues/45884) |
| 108 | vllm | PR | [Bugfix][Kernel] Fix int32/uint overflow in merge_attn_states and permute_cols | 🔴 高 | open | 2026-06-13 | [#45527](https://github.com/vllm-project/vllm/pull/45527) |
| 109 | vllm | Issue | [Bug]: Integer overflows in merge_attn_states.cu | 🔴 高 | open | 2026-06-13 | [#45500](https://github.com/vllm-project/vllm/issues/45500) |
| 110 | vllm | Issue | [Bug]: Integer overflow in permute_cols | 🔴 高 | open | 2026-06-12 | [#45469](https://github.com/vllm-project/vllm/issues/45469) |
| 111 | vllm | PR | [Bugfix] Fix int32 overflow in concat_mla_q flat warp index | 🔴 高 | open | 2026-06-12 | [#45384](https://github.com/vllm-project/vllm/pull/45384) |
| 112 | vllm | Issue | [Bug]: Integer overflow bug in concat_mla_q | 🔴 高 | open | 2026-06-12 | [#45373](https://github.com/vllm-project/vllm/issues/45373) |
| 113 | vllm | PR | [Bugfix] Fix gridDim.y overflow for large row counts | 🔴 高 | closed | 2026-06-11 | [#45255](https://github.com/vllm-project/vllm/pull/45255) |
| 114 | vllm-ascend | PR | [Bugfix] Fix bincount Triton kernel grid overflow with repetition_penalty | 🔴 高 | closed | 2026-06-07 | [#10138](https://github.com/vllm-project/vllm-ascend/pull/10138) |
| 115 | vllm | PR | Fix memory pointer overflow in Mamba state buffers | 🔴 高 | closed | 2026-06-05 | [#44665](https://github.com/vllm-project/vllm/pull/44665) |
| 116 | vllm | PR | [Bugfix] Fix integer overflow in libtorch_stable/layernorm_kernels.cu pointer arithmetic | 🔴 高 | open | 2026-05-29 | [#44027](https://github.com/vllm-project/vllm/pull/44027) |
| 117 | vllm | PR | [Bugfix] Fix integer overflow in libtorch_stable/activation_kernels.cu pointer arithmetic | 🔴 高 | closed | 2026-05-29 | [#44026](https://github.com/vllm-project/vllm/pull/44026) |
| 118 | vllm | Issue | [Bug]: integer overflow in fused_add_rms_norm | 🔴 高 | open | 2026-05-22 | [#43390](https://github.com/vllm-project/vllm/issues/43390) |
| 119 | vllm | PR | [Bug] Fix fused_qk_norm_rope 32-bit QKV offset overflow | 🔴 高 | open | 2026-05-20 | [#43166](https://github.com/vllm-project/vllm/pull/43166) |
| 120 | vllm | PR | Fix int32 overflow in csrc/activation_kernels.cu indexing | 🔴 高 | closed | 2026-05-19 | [#43157](https://github.com/vllm-project/vllm/pull/43157) |
| 121 | vllm | PR | Fix int32 overflow in csrc/layernorm_kernels.cu indexing | 🔴 高 | closed | 2026-05-19 | [#43156](https://github.com/vllm-project/vllm/pull/43156) |
| 122 | vllm | PR | [Bug] Fix integer overflow in layernorm_kernels.cu pointer arithmetic | 🔴 高 | closed | 2026-05-17 | [#42863](https://github.com/vllm-project/vllm/pull/42863) |
| 123 | vllm | Issue | [Bug]: integer overflow in layernorm_kernels.cu | 🔴 高 | open | 2026-05-17 | [#42862](https://github.com/vllm-project/vllm/issues/42862) |
| 124 | vllm | PR | [Bug] Fix integer overflow in activation_kernels.cu pointer arithmetic | 🔴 高 | closed | 2026-05-17 | [#42861](https://github.com/vllm-project/vllm/pull/42861) |
| 125 | vllm | Issue | [Bug]: integer overflow in activation_kernels.cu | 🔴 高 | closed | 2026-05-17 | [#42860](https://github.com/vllm-project/vllm/issues/42860) |
| 126 | vllm-ascend | PR | [BugFix]:Fix UB overflow caused by compatibility issues after NPUIR upgrade | 🔴 高 | closed | 2026-05-15 | [#9193](https://github.com/vllm-project/vllm-ascend/pull/9193) |
| 127 | vllm | PR | [v1] Guard resumable streaming updates against max_model_len overflow | 🔴 高 | open | 2026-05-13 | [#42502](https://github.com/vllm-project/vllm/pull/42502) |
| 128 | vllm | Issue | [Bug]: Same-request resumable streaming_update can overflow prompt width in gpu_input_batch.add_request | 🔴 高 | open | 2026-05-13 | [#42489](https://github.com/vllm-project/vllm/issues/42489) |
| 129 | vllm | PR | [Bugfix] Fix int32 overflow in DeepGEMM SiLU/mul FP8 Triton kernel | 🔴 高 | closed | 2026-05-10 | [#42201](https://github.com/vllm-project/vllm/pull/42201) |
| 130 | vllm | Issue | DeepGEMM SiLU/mul FP8 quant Triton kernel overflows int32 addresses for large DPEP warmup shapes | 🔴 高 | closed | 2026-05-09 | [#42173](https://github.com/vllm-project/vllm/issues/42173) |
| 131 | vllm | Issue | [Bug]: agrs Workspace Buffer Sizing Overflow at Large EP | 🔴 高 | open | 2026-05-06 | [#41858](https://github.com/vllm-project/vllm/issues/41858) |
| 132 | vllm | PR | [BugFix]Auto-size FlashInfer workspace buffer to prevent buffer overflows on model  | 🔴 高 | open | 2026-04-20 | [#40383](https://github.com/vllm-project/vllm/pull/40383) |
| 133 | vllm | Issue | [Bug]: Buffer overflow when allocating memory error on Qwen3.5-122B-A10B-GPTQ-Int4 and NVFP4 | 🔴 高 | open | 2026-04-20 | [#40381](https://github.com/vllm-project/vllm/issues/40381) |
| 134 | vllm | PR | [Bugfix][Gemma4] Fix vision fp16 overflow causing &lt;pad&gt; output | 🔴 高 | open | 2026-04-20 | [#40347](https://github.com/vllm-project/vllm/pull/40347) |
| 135 | vllm | Issue | [Bug]: Gemma 4 (31B/26B-A4B) vision outputs only &lt;pad&gt; under fp16 — vision_tower standardize overflows | 🔴 高 | open | 2026-04-19 | [#40290](https://github.com/vllm-project/vllm/issues/40290) |
| 136 | vllm | PR | Fix tp device index overflow | 🔴 高 | closed | 2026-04-19 | [#40265](https://github.com/vllm-project/vllm/pull/40265) |
| 137 | vllm | Issue | [Bug]: FlashInfer workspace buffer overflow during CUDA graph capture | 🔴 高 | closed | 2026-04-16 | [#40023](https://github.com/vllm-project/vllm/issues/40023) |
| 138 | vllm | PR | [ROCm] Fix TurboQuant on ROCm: backend routing, flash-attn compat, int64 overflow | 🔴 高 | closed | 2026-04-15 | [#39953](https://github.com/vllm-project/vllm/pull/39953) |
| 139 | vllm | Issue | [Bug]: Qwen3.5-27B becomes unresponsive after oversized video input (sequence length overflow) | 🔴 高 | closed | 2026-04-15 | [#39876](https://github.com/vllm-project/vllm/issues/39876) |
| 140 | vllm | PR | [Bugfix][Kernel] Fix int32 overflow in LoRA do_expand_kernel and do_shrink_kernel | 🔴 高 | open | 2026-04-11 | [#39585](https://github.com/vllm-project/vllm/pull/39585) |
| 141 | vllm-ascend | Issue | [Bug]: 0.17.0rc1启动qwen3.5-35b出现ub overflow | 🔴 高 | closed | 2026-04-08 | [#8037](https://github.com/vllm-project/vllm-ascend/issues/8037) |
| 142 | vllm | PR | Fix DeepGEMM ep_scatter output address overflow | 🔴 高 | closed | 2026-04-07 | [#39213](https://github.com/vllm-project/vllm/pull/39213) |
| 143 | vllm | PR | Fix async spec decode TOCTOU race and underflow on aborted requests | 🔴 高 | closed | 2026-04-05 | [#39012](https://github.com/vllm-project/vllm/pull/39012) |
| 144 | vllm | PR | fix(v1): Handle max_model_len overflow gracefully instead of crashing | 🔴 高 | closed | 2026-03-29 | [#38483](https://github.com/vllm-project/vllm/pull/38483) |
| 145 | vllm | Issue | [Bug]: V1 Engine: EngineDeadError (AssertionError) on max_model_len overflow during realtime audio streaming | 🔴 高 | closed | 2026-03-28 | [#38428](https://github.com/vllm-project/vllm/issues/38428) |
| 146 | vllm | PR | Fix NVFP4 weight scale underflow in BF16 dequantization | 🔴 高 | closed | 2026-03-18 | [#37464](https://github.com/vllm-project/vllm/pull/37464) |
| 147 | vllm | PR | Revert "[Bugfix] Rescale NVFP4 weight scales to fix BF16 dequant underflow" | 🔴 高 | closed | 2026-03-18 | [#37455](https://github.com/vllm-project/vllm/pull/37455) |
| 148 | vllm | PR | [Bugfix] Fix FP16 overflow in NVFP4 Marlin kernel epilogue and forward input_global_scale on SM75 | 🔴 高 | closed | 2026-03-16 | [#37135](https://github.com/vllm-project/vllm/pull/37135) |
| 149 | vllm | PR | [CI][Bugfix] Fix 500 errors from priority overflow and TemplateError subclasses in schema fuzz tests | 🔴 高 | closed | 2026-03-15 | [#37127](https://github.com/vllm-project/vllm/pull/37127) |
| 150 | vllm | Issue | [Bug]: AsyncScheduler crashes with AssertionError during Realtime ASR streaming (num_output_placeholders underflow) | 🔴 高 | open | 2026-03-02 | [#35755](https://github.com/vllm-project/vllm/issues/35755) |
| 151 | vllm | PR | [Bugfix] Fix uninitialized NVFP4 global scale causing inf overflow | 🔴 高 | closed | 2026-03-02 | [#35693](https://github.com/vllm-project/vllm/pull/35693) |
| 152 | vllm | PR | [Bugfix] Fix uint32 overflow in Mamba selective scan state pointer arithmetic | 🔴 高 | closed | 2026-02-25 | [#35275](https://github.com/vllm-project/vllm/pull/35275) |
| 153 | vllm | PR | [Bugfix][Kernel] Fix integer overflow in layernorm kernel index computations | 🔴 高 | open | 2026-02-18 | [#34842](https://github.com/vllm-project/vllm/pull/34842) |
| 154 | vllm | PR | [Bugfix] Rescale NVFP4 weight scales to fix BF16 dequant underflow | 🔴 高 | closed | 2026-02-15 | [#34577](https://github.com/vllm-project/vllm/pull/34577) |
| 155 | vllm | PR | [Bugfix] Fix fused MoE int32 overflow in stride*offset without perf regression | 🔴 高 | closed | 2026-02-13 | [#34507](https://github.com/vllm-project/vllm/pull/34507) |
| 156 | vllm | PR | [CI][Entrypoints] Validate detokenize token IDs to prevent int64 overflow causing 500 | 🔴 高 | closed | 2026-02-12 | [#34468](https://github.com/vllm-project/vllm/pull/34468) |
| 157 | vllm | Issue | [Bug]: Qwen3 Next with heterogeneous GPU (FP8 overflow?) | 🔴 高 | closed | 2026-02-12 | [#34437](https://github.com/vllm-project/vllm/issues/34437) |
| 158 | vllm-ascend | Issue | [Bug]: bisheng-ub overflow | 🔴 高 | open | 2026-01-19 | [#5983](https://github.com/vllm-project/vllm-ascend/issues/5983) |
| 159 | vllm | Issue | [Bug]: NVFP4 Flashinfer CuteDSL MoE + DeepEP ll + VLLM_MOE_DP_CHUNK_SIZE=1024 + cudagraph numerical accuracy issue on B200 | 🔴 高 | closed | 2026-01-06 | [#31840](https://github.com/vllm-project/vllm/issues/31840) |
| 160 | vllm | PR | [ROCm][CI] Fix ModernBERT token classification test numerical accuracy on ROCm | 🔴 高 | closed | 2026-01-06 | [#31820](https://github.com/vllm-project/vllm/pull/31820) |
| 161 | vllm | PR | [Bugfix] Fix integer overflow in Gemma3n audio processing | 🔴 高 | closed | 2026-01-04 | [#31657](https://github.com/vllm-project/vllm/pull/31657) |
| 162 | vllm | PR | [Bugfix][DSV32] Fix overflow in topk. | 🔴 高 | closed | 2025-12-16 | [#30754](https://github.com/vllm-project/vllm/pull/30754) |
| 163 | vllm-ascend | PR | [BugFix][Triton] Fix ub overflow bug of sample_recover_tokens_kernel | 🔴 高 | closed | 2025-12-03 | [#4673](https://github.com/vllm-project/vllm-ascend/pull/4673) |
| 164 | vllm | PR | [Quantization] fix: overflow with static per-tensor scaling | 🔴 高 | closed | 2025-12-02 | [#29867](https://github.com/vllm-project/vllm/pull/29867) |
| 165 | vllm | Issue | [Bug]: Qwen3-VL-235B-A22B-Instruct Grounding Accuracy Issue in vLLM (&gt;= v0.11.1) | 🔴 高 | closed | 2025-11-27 | [#29595](https://github.com/vllm-project/vllm/issues/29595) |
| 166 | vllm | Issue | [Bug]: Potential Integer Overflow and Out-of-bounds in selective_scan_fwd.cu | 🔴 高 | closed | 2025-11-01 | [#27911](https://github.com/vllm-project/vllm/issues/27911) |
| 167 | vllm | Issue | [Bug]: Potential Integer Overflow and Out-of-bounds in awq_gemm | 🔴 高 | closed | 2025-09-12 | [#24782](https://github.com/vllm-project/vllm/issues/24782) |
| 168 | vllm | Issue | [Bug]: Detokenizer Overflow error occurred on DeepSeek-R1/V3 | 🔴 高 | closed | 2025-09-04 | [#24211](https://github.com/vllm-project/vllm/issues/24211) |
| 169 | vllm | Issue | [Bug]: Potential Integer Overflow and Out-of-bounds in layernorm_kernels | 🔴 高 | closed | 2025-08-07 | [#22419](https://github.com/vllm-project/vllm/issues/22419) |
| 170 | vllm | PR | Fix overflow indexing in causal_conv1d kernel | 🔴 高 | closed | 2025-07-14 | [#20938](https://github.com/vllm-project/vllm/pull/20938) |
| 171 | vllm | PR | [BugFix] fix 3 issues: (1) using metadata for causal-conv1d, (2) indexing overflow in v1 vLLM, and (3) init_states in v0 | 🔴 高 | closed | 2025-07-11 | [#20838](https://github.com/vllm-project/vllm/pull/20838) |
| 172 | vllm-ascend | Issue | [Bug]: longterm test failed due to RuntimeError: value cannot be converted to type at::Half without overflow | 🔴 高 | closed | 2025-06-24 | [#1382](https://github.com/vllm-project/vllm-ascend/issues/1382) |
| 173 | vllm | PR | Workaround for an integer overflow with large CHUNK_SIZE | 🔴 高 | closed | 2025-06-17 | [#19770](https://github.com/vllm-project/vllm/pull/19770) |
| 174 | vllm | Issue | [Bug]: Potential Integer Overflow permute_cols.cu | 🔴 高 | closed | 2025-06-10 | [#19450](https://github.com/vllm-project/vllm/issues/19450) |
| 175 | vllm | Issue | [Usage]: When deploying the GLM-4-32B BF16 model with vLLM 0.8.4, I encountered a GPU memory overflow | 🔴 高 | closed | 2025-04-21 | [#16896](https://github.com/vllm-project/vllm/issues/16896) |
| 176 | vllm | PR | [BugFix][V1] Fix int32 token index overflow when preparing input ids | 🔴 高 | closed | 2025-04-17 | [#16806](https://github.com/vllm-project/vllm/pull/16806) |
| 177 | vllm | Issue | [Bug]: deepseek-r1 awq fp16 overflow | 🔴 高 | closed | 2025-04-14 | [#16579](https://github.com/vllm-project/vllm/issues/16579) |
| 178 | vllm | PR | [Bugfix] Fix qwen2.5-vl overflow issue | 🔴 高 | closed | 2025-02-27 | [#13968](https://github.com/vllm-project/vllm/pull/13968) |
| 179 | vllm | PR | [BugFix] Fix an Overflow Problem for Some Triton Fused MoE Configurations with large BLOCK_SIZE  | 🔴 高 | closed | 2025-02-26 | [#13901](https://github.com/vllm-project/vllm/pull/13901) |
| 180 | vllm | PR | [Bugfix] Fix FP16 overflow for DeepSeek V2 | 🔴 高 | closed | 2025-02-13 | [#13232](https://github.com/vllm-project/vllm/pull/13232) |
| 181 | vllm | Issue | [Bug]: When using the VLLM framework to load visual models, CPU memory overflow occurs while continuously processing data with images. | 🔴 高 | closed | 2025-02-09 | [#12973](https://github.com/vllm-project/vllm/issues/12973) |
| 182 | vllm | PR | Fix integer overflow causing gpu segfault | 🔴 高 | closed | 2024-11-15 | [#10380](https://github.com/vllm-project/vllm/pull/10380) |
| 183 | vllm | PR | [Bugfix][Kernel] Prevent integer overflow in fp8 dynamic per-token quantize kernel | 🔴 高 | closed | 2024-10-16 | [#9425](https://github.com/vllm-project/vllm/pull/9425) |
| 184 | vllm | PR | Compute logits*V in FP32 in attention kernel to improve numerical accuracy | 🔴 高 | closed | 2024-01-08 | [#2368](https://github.com/vllm-project/vllm/pull/2368) |
| 185 | vllm | PR | Optimization to prevent overflow when handling large activation tensors | 🔴 高 | closed | 2023-11-13 | [#1639](https://github.com/vllm-project/vllm/pull/1639) |
| 186 | vllm | Issue | Array Index Overflow for Large #Blocks | 🔴 高 | closed | 2023-10-27 | [#1486](https://github.com/vllm-project/vllm/issues/1486) |
| 187 | vllm | PR | Fix overflow in awq kernel | 🔴 高 | closed | 2023-10-09 | [#1295](https://github.com/vllm-project/vllm/pull/1295) |
| 188 | vllm | Issue | datatype overflow | 🔴 高 | closed | 2023-07-19 | [#521](https://github.com/vllm-project/vllm/issues/521) |
| 189 | vllm | PR | [Kernel] Fix vllm_c fused_add_rms_norm to match native IR rounding semantics | 🟡 中 | open | 2026-08-14 | [#52243](https://github.com/vllm-project/vllm/pull/52243) |
| 190 | vllm | PR | replace batch_norm to numerically identical without cudnn | 🟡 中 | closed | 2026-08-10 | [#51734](https://github.com/vllm-project/vllm/pull/51734) |
| 191 | vllm-ascend | PR | [BugFix][Sampler] Fix inf in Qwen30b RL sampling by recording stream for q releases/v0.13.0 | 🟡 中 | open | 2026-08-03 | [#13406](https://github.com/vllm-project/vllm-ascend/pull/13406) |
| 192 | vllm-ascend | PR | [BugFix][Sampler] Fix inf in Qwen30b RL sampling by recording stream for q releases/v0.26.0rc | 🟡 中 | open | 2026-08-03 | [#13405](https://github.com/vllm-project/vllm-ascend/pull/13405) |
| 193 | vllm-ascend | PR | [BugFix][Sampler] Fix inf in Qwen30b RL sampling by recording stream for q releases/v0.25.1rc | 🟡 中 | open | 2026-08-03 | [#13404](https://github.com/vllm-project/vllm-ascend/pull/13404) |
| 194 | vllm-ascend | PR | [BugFix][Sampler] Fix inf in Qwen30b RL sampling by recording stream for q | 🟡 中 | closed | 2026-08-03 | [#13403](https://github.com/vllm-project/vllm-ascend/pull/13403) |
| 195 | vllm-ascend | PR | [BugFix][Sampler] Fix inf in Qwen30b RL sampling by recording stream for q releases/v0.18.0 | 🟡 中 | open | 2026-08-03 | [#13402](https://github.com/vllm-project/vllm-ascend/pull/13402) |
| 196 | vllm-ascend | PR | [BugFix][Sampler] Fix inf in Qwen30b RL sampling by recording stream for q releases/v0.20.2rc | 🟡 中 | open | 2026-08-03 | [#13400](https://github.com/vllm-project/vllm-ascend/pull/13400) |
| 197 | vllm-ascend | PR | [BugFix][Sampler] Fix inf in Qwen30b RL sampling by recording stream for q releases/v0.21.0rc | 🟡 中 | open | 2026-08-03 | [#13398](https://github.com/vllm-project/vllm-ascend/pull/13398) |
| 198 | vllm-ascend | PR | [BugFix][Sampler] Fix inf in Qwen30b RL sampling by recording stream for q releases/v0.22.1rc | 🟡 中 | open | 2026-08-03 | [#13397](https://github.com/vllm-project/vllm-ascend/pull/13397) |
| 199 | vllm-ascend | PR | [BugFix][Sampler] Fix inf in Qwen30b RL sampling by recording stream for q releases/v0.23.0 | 🟡 中 | open | 2026-08-03 | [#13396](https://github.com/vllm-project/vllm-ascend/pull/13396) |
| 200 | vllm-ascend | PR | [BugFix][Sampler] Fix inf in Qwen30b RL sampling by recording stream for q  releases/v0.24.0rc | 🟡 中 | open | 2026-08-03 | [#13395](https://github.com/vllm-project/vllm-ascend/pull/13395) |
| 201 | vllm-ascend | PR | [BugFix][Sampler] Fix inf in Qwen30b RL sampling by recording stream for q | 🟡 中 | closed | 2026-08-03 | [#13394](https://github.com/vllm-project/vllm-ascend/pull/13394) |
| 202 | vllm | PR | [Bugfix][CPU][RISC-V] Fix FP16 rounding and cross-compilation capability handling | 🟡 中 | closed | 2026-08-02 | [#50743](https://github.com/vllm-project/vllm/pull/50743) |
| 203 | vllm | PR | [rocm] perf: drop redundant -inf prefill of decode paged MQA-logits buffer | 🟡 中 | open | 2026-07-27 | [#50008](https://github.com/vllm-project/vllm/pull/50008) |
| 204 | vllm | PR | [Test][ROCm] Account for gfx950 FP8 RMSNorm rounding | 🟡 中 | closed | 2026-07-25 | [#49839](https://github.com/vllm-project/vllm/pull/49839) |
| 205 | vllm | PR | [Test][ROCm] Account for gfx950 FP8 RMSNorm rounding | 🟡 中 | closed | 2026-07-25 | [#49832](https://github.com/vllm-project/vllm/pull/49832) |
| 206 | vllm | PR | fix(kernel): restore scalar_t RMSNorm intermediate rounding boundary (#49616) | 🟡 中 | open | 2026-07-23 | [#49639](https://github.com/vllm-project/vllm/pull/49639) |
| 207 | vllm-ascend | PR | [bugfix] fix inf in Qwen30b RL | 🟡 中 | closed | 2026-07-20 | [#12401](https://github.com/vllm-project/vllm-ascend/pull/12401) |
| 208 | vllm | PR | [Test] Fix bf16 rounding in selective_state_update reference | 🟡 中 | open | 2026-07-14 | [#48578](https://github.com/vllm-project/vllm/pull/48578) |
| 209 | vllm | PR | [Bugfix][CPU][RISC-V] Use explicit rounding mode for vfcvt float-to-int conversions | 🟡 中 | open | 2026-07-08 | [#47983](https://github.com/vllm-project/vllm/pull/47983) |
| 210 | vllm | PR | [ROCm][Test] Fix test_per_token_group_quant_fp8 tolerance for 1-ULP FP8 rounding on gfx950 | 🟡 中 | closed | 2026-06-28 | [#46944](https://github.com/vllm-project/vllm/pull/46944) |
| 211 | vllm | PR | [ROCm][gpt-oss] Remove vLLM-side MoE padding rounding (AITER handles internally) | 🟡 中 | open | 2026-06-20 | [#46201](https://github.com/vllm-project/vllm/pull/46201) |
| 212 | vllm | PR | [Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len | 🟡 中 | closed | 2026-05-09 | [#42169](https://github.com/vllm-project/vllm/pull/42169) |
| 213 | vllm-ascend | PR | [0.18.0][BugFix] Update capture sizes after rounding operations | 🟡 中 | closed | 2026-04-17 | [#8380](https://github.com/vllm-project/vllm-ascend/pull/8380) |
| 214 | vllm | PR | fix: clamp dA_cumsum differences to prevent Inf in Mamba2 SSD kernels | 🟡 中 | closed | 2026-03-19 | [#37501](https://github.com/vllm-project/vllm/pull/37501) |
| 215 | vllm | Issue | [Bug]: nemotron_h does not work with DeepEP all2all backends due to hidden dim rounding | 🟡 中 | closed | 2026-03-12 | [#36926](https://github.com/vllm-project/vllm/issues/36926) |
| 216 | vllm | PR | [BUGFIX] Fix `test_mla_backends.py`. Scale MLA projection weights to prevent numerical instability  | 🟡 中 | closed | 2026-01-17 | [#32529](https://github.com/vllm-project/vllm/pull/32529) |
| 217 | vllm-ascend | PR | [0.13.0][cherry-pick][bugfix](cp) replace None with zeros/inf tensor to avoid TypeError | 🟡 中 | closed | 2026-01-13 | [#5844](https://github.com/vllm-project/vllm-ascend/pull/5844) |
| 218 | vllm-ascend | PR | [bugfix](cp) replace None with zeros/inf tensor to avoid TypeError | 🟡 中 | closed | 2026-01-13 | [#5837](https://github.com/vllm-project/vllm-ascend/pull/5837) |
| 219 | vllm | Issue | [Feature]: Support `inf` value for burstiness in benchmarks | 🟡 中 | closed | 2025-10-15 | [#26940](https://github.com/vllm-project/vllm/issues/26940) |
| 220 | vllm | PR | [Kernel] Better inf handling for grouped topk cu | 🟡 中 | closed | 2025-09-15 | [#24886](https://github.com/vllm-project/vllm/pull/24886) |
| 221 | vllm-ascend | Issue | [Bug]: vllm-ascend 0.9.1 模型输出logprob包含过多-inf | 🟡 中 | open | 2025-09-15 | [#2934](https://github.com/vllm-project/vllm-ascend/issues/2934) |
| 222 | vllm | PR | Fix GLM-4.5V-FP8 numerical issue | 🟡 中 | closed | 2025-08-15 | [#22949](https://github.com/vllm-project/vllm/pull/22949) |
| 223 | vllm | PR | [Fix] apply_temperature() casuing `inf` logits | 🟡 中 | closed | 2025-08-05 | [#22261](https://github.com/vllm-project/vllm/pull/22261) |
| 224 | vllm | Issue | [RFC]: vLLM vs HuggingFace numerical parity report | 🟡 中 | closed | 2025-07-23 | [#21475](https://github.com/vllm-project/vllm/issues/21475) |
| 225 | vllm | Issue | [Feature]: Automatically detect numerical issues | 🟡 中 | closed | 2025-04-24 | [#17123](https://github.com/vllm-project/vllm/issues/17123) |
| 226 | vllm | Issue | [Bug][V1]: Many decoding batch with only 1 request at the beginning of benchmarking when request-rate is inf | 🟡 中 | closed | 2025-04-14 | [#16615](https://github.com/vllm-project/vllm/issues/16615) |
| 227 | vllm | PR | [Bugfix] FA2 `inf` workaround with minimum overhead for MLA | 🟡 中 | closed | 2025-03-29 | [#15742](https://github.com/vllm-project/vllm/pull/15742) |
| 228 | vllm | Issue | [Bug]: Llama-3.1-Nemotron-70B-Instruct-HF W8A8 has ValueError: Failed to invert hessian due to numerical instability. | 🟡 中 | closed | 2024-12-31 | [#11641](https://github.com/vllm-project/vllm/issues/11641) |
| 229 | vllm | PR | [Bugfix]: Clamp `-inf` logprob values in prompt_logprobs | 🟡 中 | closed | 2024-12-10 | [#11073](https://github.com/vllm-project/vllm/pull/11073) |
| 230 | vllm | Issue | [Bug]: When I use llmcompressor to quantify the llama3 70b model to int8-a8w8,it shows ValueError: Failed to invert hessian due to numerical instability. | 🟡 中 | closed | 2024-12-10 | [#11064](https://github.com/vllm-project/vllm/issues/11064) |
| 231 | vllm-ascend | PR | [Attention][BugFix] Add hardware synchronization and input validation, overflow checks for ngram spec decode. | 🟢 低 | closed | 2026-06-03 | [#9923](https://github.com/vllm-project/vllm-ascend/pull/9923) |
| 232 | vllm | PR | [Mamba] Add stochastic rounding support | 🟢 低 | closed | 2026-03-02 | [#35753](https://github.com/vllm-project/vllm/pull/35753) |
| 233 | vllm | PR | [Bugfix] Allow 64-bit integer values for LoRA IDs to avoid overflow/truncation | 🟢 低 | closed | 2025-10-31 | [#27876](https://github.com/vllm-project/vllm/pull/27876) |
| 234 | vllm | PR | [Feature][Benchmarks] Support `inf` burstiness | 🟢 低 | closed | 2025-10-15 | [#26941](https://github.com/vllm-project/vllm/pull/26941) |
| 235 | vllm | PR | [wip] Add tier and overflow checks | 🟢 低 | closed | 2025-06-05 | [#19220](https://github.com/vllm-project/vllm/pull/19220) |
| 236 | vllm | PR | [VLM] Disallow overflowing `max_model_len` for multimodal models | 🟢 低 | closed | 2024-08-29 | [#7998](https://github.com/vllm-project/vllm/pull/7998) |
| 237 | vllm | PR | fix: add assertion when temperature applied logits are inf | 🟢 低 | closed | 2023-08-28 | [#890](https://github.com/vllm-project/vllm/pull/890) |

</details>

---

## 3. 输出正确性与一致性

模型的最终输出与参考实现（如 HuggingFace Transformers）不一致、产生错误/garbage 结果、或在不同配置（backend、并行度、采样参数）间漂移。这类问题直接反映为端到端正确性下降，是最容易被用户感知的精度问题。

**问题数: 361 条**

### 关键问题

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug]: speculative decoding under pipeline parallelism produces wrong output with --no-async-scheduling | 🔴 极高 | open | 2026-08-12 | [#52071](https://github.com/vllm-project/vllm/issues/52071) |
| 2 | vllm-ascend | Issue | [Bug]: `AscendMRotaryEmbedding.forward_oot` hardcodes Neox-style (chunk-half) rotation for the `torch_npu.npu_mrope` path, producing wrong results for GPT-J style models (e.g. GLM-OCR). | 🔴 极高 | open | 2026-07-24 | [#12765](https://github.com/vllm-project/vllm-ascend/issues/12765) |
| 3 | vllm | Issue | [Bug]: [Parser] kimi_k2 streaming tool-call args skip schema type coercion applied in non-streaming (silent type mismatch) | 🔴 极高 | open | 2026-07-21 | [#49316](https://github.com/vllm-project/vllm/issues/49316) |
| 4 | vllm | Issue | [Bug]: [kernel] MRotaryEmbedding Triton kernel hardcodes Neox-style rotation, producing wrong results for GPT-J style models (like GLM-OCR). | 🔴 极高 | open | 2026-07-21 | [#49290](https://github.com/vllm-project/vllm/issues/49290) |
| 5 | vllm | Issue | [Bug]: kv_load_failure_policy="recompute" gives wrong output when a KV connector rejects a synchronous load (V2 runner) | 🔴 极高 | open | 2026-07-20 | [#49250](https://github.com/vllm-project/vllm/issues/49250) |
| 6 | vllm | PR | nano_nemotron_vl: fix tensor device mismatch exception when video profiling | 🔴 极高 | closed | 2026-04-05 | [#39029](https://github.com/vllm-project/vllm/pull/39029) |
| 7 | vllm-ascend | Issue | [Performance]: vllm-ascend v0.13.0 精度较差 输出错误引入functioncall内定义的参数 | 🔴 极高 | closed | 2026-02-10 | [#6660](https://github.com/vllm-project/vllm-ascend/issues/6660) |
| 8 | vllm | Issue | [Bug]: LoRA adapters with mismatched module name prefixes silently produce base-model output | 🔴 极高 | open | 2026-02-09 | [#34186](https://github.com/vllm-project/vllm/issues/34186) |
| 9 | vllm | PR | [Bugfix]  Fix precision corruption when shared_experts_stream=None | 🔴 极高 | closed | 2025-11-18 | [#28942](https://github.com/vllm-project/vllm/pull/28942) |
| 10 | vllm | Issue | [Bug]: get wrong output in lm_eval test for PP mode. | 🔴 极高 | closed | 2025-11-17 | [#28839](https://github.com/vllm-project/vllm/issues/28839) |
| 11 | vllm-ascend | Issue | [Bug]: deepseek w8a8 dynamic + multi-stream test get wrong output | 🔴 极高 | closed | 2025-08-06 | [#2232](https://github.com/vllm-project/vllm-ascend/issues/2232) |
| 12 | vllm | Issue | [Bug]: reasoning-parser=deepseek_r1 wrong output with enable_thinking=False | 🔴 极高 | closed | 2025-06-05 | [#19222](https://github.com/vllm-project/vllm/issues/19222) |
| 13 | vllm | PR | [doc] wrong output | 🔴 极高 | closed | 2025-06-01 | [#19000](https://github.com/vllm-project/vllm/pull/19000) |
| 14 | vllm-ascend | Issue | [Bug]: Wrong output tensor shape when running DeepSeek model with V1 engine | 🔴 极高 | closed | 2025-05-07 | [#778](https://github.com/vllm-project/vllm-ascend/issues/778) |
| 15 | vllm | Issue | [Bug]: Dynamically load lora got wrong output | 🔴 极高 | closed | 2025-01-20 | [#12199](https://github.com/vllm-project/vllm/issues/12199) |
| 16 | vllm | PR | [BugFix] fix wrong output when using lora and num_scheduler_steps=8 | 🔴 极高 | closed | 2024-12-13 | [#11161](https://github.com/vllm-project/vllm/pull/11161) |
| 17 | vllm | Issue | [Bug]: N-gram speculative decoding got wrong output when some of seeds is None in a batch  | 🔴 极高 | closed | 2024-12-12 | [#11123](https://github.com/vllm-project/vllm/issues/11123) |
| 18 | vllm | Issue | [Bug, V1]: LlaVa outputs wrong results in batch inference with V1 code（V0 code is correct) | 🔴 极高 | closed | 2024-12-04 | [#10891](https://github.com/vllm-project/vllm/issues/10891) |
| 19 | vllm | PR | [BugFix] fix wrong output when using `lora` and `num_scheduler_steps=8` | 🔴 极高 | closed | 2024-10-25 | [#9689](https://github.com/vllm-project/vllm/pull/9689) |
| 20 | vllm | Issue | [Bug]: deepseek_Coder_v2_Instruct give wrong output on vllm==0.5.4, 0.5.5, and 0.6.1.post2 (others not tried) with huggingface standard usage | 🔴 极高 | closed | 2024-09-17 | [#8542](https://github.com/vllm-project/vllm/issues/8542) |
| 21 | vllm | Issue | [Bug]: deepseek_v2 236B  on 8XA100 wrong output   vllm==0.5.4  | 🔴 极高 | closed | 2024-09-09 | [#8283](https://github.com/vllm-project/vllm/issues/8283) |
| 22 | vllm | Issue | wrong output of AsyncLLMEngine | 🔴 极高 | closed | 2024-02-21 | [#2947](https://github.com/vllm-project/vllm/issues/2947) |
| 23 | vllm | Issue | chatglm3 with parallel &gt; 1 gets wrong results | 🔴 极高 | closed | 2024-01-24 | [#2572](https://github.com/vllm-project/vllm/issues/2572) |
| 24 | vllm | Issue | [Bug]: two logger calls have mismatched %-args, so the log record is dropped | 🔴 高 | open | 2026-08-17 | [#52591](https://github.com/vllm-project/vllm/issues/52591) |
| 25 | vllm | Issue | [Bug]: qwen3_5_mtp fails to load at tensor-parallel-size &gt;= 2 (drafter weight shape mismatch) | 🔴 高 | open | 2026-08-16 | [#52480](https://github.com/vllm-project/vllm/issues/52480) |
| 26 | vllm-ascend | PR | [BugFix][Scheduler] Fix Mamba split call argument mismatch | 🔴 高 | open | 2026-08-15 | [#14321](https://github.com/vllm-project/vllm-ascend/pull/14321) |
| 27 | vllm | PR | [ROCm]: Drop pybind11 from Dockerfile.rocm to prevent version mismatch | 🔴 高 | closed | 2026-08-14 | [#52400](https://github.com/vllm-project/vllm/pull/52400) |
| 28 | vllm | PR | [Bugfix] Fix mismatched logger format args and enable ruff PLE1205/PLE1206 | 🔴 高 | open | 2026-08-12 | [#51973](https://github.com/vllm-project/vllm/pull/51973) |
| 29 | vllm | Issue | [Bug]: Qwen3 MoE GPTQ `qzeros` shape mismatch on ROCm gfx1201 | 🔴 高 | open | 2026-08-12 | [#51971](https://github.com/vllm-project/vllm/issues/51971) |
| 30 | vllm-ascend | PR | [BugFix] Fix CopyInKv DMA flipping KV block order when physical address order mismatches logical token order | 🔴 高 | open | 2026-08-11 | [#14040](https://github.com/vllm-project/vllm-ascend/pull/14040) |
| 31 | vllm-ascend | PR | [BugFix] Fix CopyInKv DMA flipping KV block order when physical address order mismatches logical token order | 🔴 高 | open | 2026-08-11 | [#14039](https://github.com/vllm-project/vllm-ascend/pull/14039) |
| 32 | vllm-ascend | PR | [BugFix] Fix CopyInKv DMA flipping KV block order when physical address order mismatches logical token order | 🔴 高 | closed | 2026-08-11 | [#14038](https://github.com/vllm-project/vllm-ascend/pull/14038) |
| 33 | vllm-ascend | Issue | [Bug]: 使用v0.23.0rc1-310部署模型后，执行推理报错： shape mismatch: value tensor of shape [186, 1024] cannot be broadcast to indexing result of shape [196, 1024] | 🔴 高 | open | 2026-08-11 | [#14014](https://github.com/vllm-project/vllm-ascend/issues/14014) |
| 34 | vllm | Issue | [Bug]: DeepSeek-V4-Flash-0731 `response_format` (structured output) crashes the vLLM EngineCore — `apply_grammar_bitmask` tensor size mismatch (4040 vs 4041) | 🔴 高 | closed | 2026-08-08 | [#51467](https://github.com/vllm-project/vllm/issues/51467) |
| 35 | vllm-ascend | Issue | [Bug]:【致命】使用8月5日最新0.23.0日构建版本+glm-5.2 w8a8c8+910c-4机PD分离，64并发测试mmlu-pro精度集，输出长度设置32000，测试2小时候，发现有5条响应存在句子的重复精度问题 | 🔴 高 | closed | 2026-08-06 | [#13719](https://github.com/vllm-project/vllm-ascend/issues/13719) |
| 36 | vllm-ascend | Issue | [bug]:[v0.23][glm5.2] DCP+MTP decode worker crashes with slot_mapping shape mismatch (256 vs 260) under long-prompt high-concurrency | 🔴 高 | open | 2026-08-06 | [#13710](https://github.com/vllm-project/vllm-ascend/issues/13710) |
| 37 | vllm-ascend | PR | [Cherry-pick][releases/v0.24.0rc][BugFix][P/D] Lower Mooncake KV metadata mismatch log level (from #13542) | 🔴 高 | closed | 2026-08-05 | [#13618](https://github.com/vllm-project/vllm-ascend/pull/13618) |
| 38 | vllm-ascend | PR | [Cherry-pick][releases/v0.25.1rc][BugFix][P/D] Lower Mooncake KV metadata mismatch log level (from #13542) | 🔴 高 | open | 2026-08-05 | [#13617](https://github.com/vllm-project/vllm-ascend/pull/13617) |
| 39 | vllm-ascend | PR | [Cherry-pick][releases/v0.26.0rc][BugFix][P/D] Lower Mooncake KV metadata mismatch log level (from #13542) | 🔴 高 | closed | 2026-08-05 | [#13616](https://github.com/vllm-project/vllm-ascend/pull/13616) |
| 40 | vllm-ascend | PR | [Cherry-pick][releases/v0.23.0][BugFix][P/D] Lower Mooncake KV metadata mismatch log level (from #13542) | 🔴 高 | closed | 2026-08-05 | [#13613](https://github.com/vllm-project/vllm-ascend/pull/13613) |

### 关键规律与分析

1. 输出与参考实现（HuggingFace Transformers）不一致，或在不同 backend / 并行度 / 采样参数间漂移。
2. **backend 差异是主要来源**：FlashInfer-CUTLASS vs FlashAttention2、FA2 fallback、XPU/ROCm 后端在同等配置下数值行为不同。
3. **并行边界**高发：TP shard 划分量化 group、expert parallelism 下产生错误输出、PP 各 stage 状态不一致。
4. 这类问题最容易与"性能"混淆——因为常伴随速度提升的优化引入的正确性回归。

<details>
<summary>展开全部 361 条</summary>

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug]: speculative decoding under pipeline parallelism produces wrong output with --no-async-scheduling | 🔴 极高 | open | 2026-08-12 | [#52071](https://github.com/vllm-project/vllm/issues/52071) |
| 2 | vllm-ascend | Issue | [Bug]: `AscendMRotaryEmbedding.forward_oot` hardcodes Neox-style (chunk-half) rotation for the `torch_npu.npu_mrope` path, producing wrong results for GPT-J style models (e.g. GLM-OCR). | 🔴 极高 | open | 2026-07-24 | [#12765](https://github.com/vllm-project/vllm-ascend/issues/12765) |
| 3 | vllm | Issue | [Bug]: [Parser] kimi_k2 streaming tool-call args skip schema type coercion applied in non-streaming (silent type mismatch) | 🔴 极高 | open | 2026-07-21 | [#49316](https://github.com/vllm-project/vllm/issues/49316) |
| 4 | vllm | Issue | [Bug]: [kernel] MRotaryEmbedding Triton kernel hardcodes Neox-style rotation, producing wrong results for GPT-J style models (like GLM-OCR). | 🔴 极高 | open | 2026-07-21 | [#49290](https://github.com/vllm-project/vllm/issues/49290) |
| 5 | vllm | Issue | [Bug]: kv_load_failure_policy="recompute" gives wrong output when a KV connector rejects a synchronous load (V2 runner) | 🔴 极高 | open | 2026-07-20 | [#49250](https://github.com/vllm-project/vllm/issues/49250) |
| 6 | vllm | PR | nano_nemotron_vl: fix tensor device mismatch exception when video profiling | 🔴 极高 | closed | 2026-04-05 | [#39029](https://github.com/vllm-project/vllm/pull/39029) |
| 7 | vllm-ascend | Issue | [Performance]: vllm-ascend v0.13.0 精度较差 输出错误引入functioncall内定义的参数 | 🔴 极高 | closed | 2026-02-10 | [#6660](https://github.com/vllm-project/vllm-ascend/issues/6660) |
| 8 | vllm | Issue | [Bug]: LoRA adapters with mismatched module name prefixes silently produce base-model output | 🔴 极高 | open | 2026-02-09 | [#34186](https://github.com/vllm-project/vllm/issues/34186) |
| 9 | vllm | PR | [Bugfix]  Fix precision corruption when shared_experts_stream=None | 🔴 极高 | closed | 2025-11-18 | [#28942](https://github.com/vllm-project/vllm/pull/28942) |
| 10 | vllm | Issue | [Bug]: get wrong output in lm_eval test for PP mode. | 🔴 极高 | closed | 2025-11-17 | [#28839](https://github.com/vllm-project/vllm/issues/28839) |
| 11 | vllm-ascend | Issue | [Bug]: deepseek w8a8 dynamic + multi-stream test get wrong output | 🔴 极高 | closed | 2025-08-06 | [#2232](https://github.com/vllm-project/vllm-ascend/issues/2232) |
| 12 | vllm | Issue | [Bug]: reasoning-parser=deepseek_r1 wrong output with enable_thinking=False | 🔴 极高 | closed | 2025-06-05 | [#19222](https://github.com/vllm-project/vllm/issues/19222) |
| 13 | vllm | PR | [doc] wrong output | 🔴 极高 | closed | 2025-06-01 | [#19000](https://github.com/vllm-project/vllm/pull/19000) |
| 14 | vllm-ascend | Issue | [Bug]: Wrong output tensor shape when running DeepSeek model with V1 engine | 🔴 极高 | closed | 2025-05-07 | [#778](https://github.com/vllm-project/vllm-ascend/issues/778) |
| 15 | vllm | Issue | [Bug]: Dynamically load lora got wrong output | 🔴 极高 | closed | 2025-01-20 | [#12199](https://github.com/vllm-project/vllm/issues/12199) |
| 16 | vllm | PR | [BugFix] fix wrong output when using lora and num_scheduler_steps=8 | 🔴 极高 | closed | 2024-12-13 | [#11161](https://github.com/vllm-project/vllm/pull/11161) |
| 17 | vllm | Issue | [Bug]: N-gram speculative decoding got wrong output when some of seeds is None in a batch  | 🔴 极高 | closed | 2024-12-12 | [#11123](https://github.com/vllm-project/vllm/issues/11123) |
| 18 | vllm | Issue | [Bug, V1]: LlaVa outputs wrong results in batch inference with V1 code（V0 code is correct) | 🔴 极高 | closed | 2024-12-04 | [#10891](https://github.com/vllm-project/vllm/issues/10891) |
| 19 | vllm | PR | [BugFix] fix wrong output when using `lora` and `num_scheduler_steps=8` | 🔴 极高 | closed | 2024-10-25 | [#9689](https://github.com/vllm-project/vllm/pull/9689) |
| 20 | vllm | Issue | [Bug]: deepseek_Coder_v2_Instruct give wrong output on vllm==0.5.4, 0.5.5, and 0.6.1.post2 (others not tried) with huggingface standard usage | 🔴 极高 | closed | 2024-09-17 | [#8542](https://github.com/vllm-project/vllm/issues/8542) |
| 21 | vllm | Issue | [Bug]: deepseek_v2 236B  on 8XA100 wrong output   vllm==0.5.4  | 🔴 极高 | closed | 2024-09-09 | [#8283](https://github.com/vllm-project/vllm/issues/8283) |
| 22 | vllm | Issue | wrong output of AsyncLLMEngine | 🔴 极高 | closed | 2024-02-21 | [#2947](https://github.com/vllm-project/vllm/issues/2947) |
| 23 | vllm | Issue | chatglm3 with parallel &gt; 1 gets wrong results | 🔴 极高 | closed | 2024-01-24 | [#2572](https://github.com/vllm-project/vllm/issues/2572) |
| 24 | vllm | Issue | [Bug]: two logger calls have mismatched %-args, so the log record is dropped | 🔴 高 | open | 2026-08-17 | [#52591](https://github.com/vllm-project/vllm/issues/52591) |
| 25 | vllm | Issue | [Bug]: qwen3_5_mtp fails to load at tensor-parallel-size &gt;= 2 (drafter weight shape mismatch) | 🔴 高 | open | 2026-08-16 | [#52480](https://github.com/vllm-project/vllm/issues/52480) |
| 26 | vllm-ascend | PR | [BugFix][Scheduler] Fix Mamba split call argument mismatch | 🔴 高 | open | 2026-08-15 | [#14321](https://github.com/vllm-project/vllm-ascend/pull/14321) |
| 27 | vllm | PR | [ROCm]: Drop pybind11 from Dockerfile.rocm to prevent version mismatch | 🔴 高 | closed | 2026-08-14 | [#52400](https://github.com/vllm-project/vllm/pull/52400) |
| 28 | vllm | PR | [Bugfix] Fix mismatched logger format args and enable ruff PLE1205/PLE1206 | 🔴 高 | open | 2026-08-12 | [#51973](https://github.com/vllm-project/vllm/pull/51973) |
| 29 | vllm | Issue | [Bug]: Qwen3 MoE GPTQ `qzeros` shape mismatch on ROCm gfx1201 | 🔴 高 | open | 2026-08-12 | [#51971](https://github.com/vllm-project/vllm/issues/51971) |
| 30 | vllm-ascend | PR | [BugFix] Fix CopyInKv DMA flipping KV block order when physical address order mismatches logical token order | 🔴 高 | open | 2026-08-11 | [#14040](https://github.com/vllm-project/vllm-ascend/pull/14040) |
| 31 | vllm-ascend | PR | [BugFix] Fix CopyInKv DMA flipping KV block order when physical address order mismatches logical token order | 🔴 高 | open | 2026-08-11 | [#14039](https://github.com/vllm-project/vllm-ascend/pull/14039) |
| 32 | vllm-ascend | PR | [BugFix] Fix CopyInKv DMA flipping KV block order when physical address order mismatches logical token order | 🔴 高 | closed | 2026-08-11 | [#14038](https://github.com/vllm-project/vllm-ascend/pull/14038) |
| 33 | vllm-ascend | Issue | [Bug]: 使用v0.23.0rc1-310部署模型后，执行推理报错： shape mismatch: value tensor of shape [186, 1024] cannot be broadcast to indexing result of shape [196, 1024] | 🔴 高 | open | 2026-08-11 | [#14014](https://github.com/vllm-project/vllm-ascend/issues/14014) |
| 34 | vllm | Issue | [Bug]: DeepSeek-V4-Flash-0731 `response_format` (structured output) crashes the vLLM EngineCore — `apply_grammar_bitmask` tensor size mismatch (4040 vs 4041) | 🔴 高 | closed | 2026-08-08 | [#51467](https://github.com/vllm-project/vllm/issues/51467) |
| 35 | vllm-ascend | Issue | [Bug]:【致命】使用8月5日最新0.23.0日构建版本+glm-5.2 w8a8c8+910c-4机PD分离，64并发测试mmlu-pro精度集，输出长度设置32000，测试2小时候，发现有5条响应存在句子的重复精度问题 | 🔴 高 | closed | 2026-08-06 | [#13719](https://github.com/vllm-project/vllm-ascend/issues/13719) |
| 36 | vllm-ascend | Issue | [bug]:[v0.23][glm5.2] DCP+MTP decode worker crashes with slot_mapping shape mismatch (256 vs 260) under long-prompt high-concurrency | 🔴 高 | open | 2026-08-06 | [#13710](https://github.com/vllm-project/vllm-ascend/issues/13710) |
| 37 | vllm-ascend | PR | [Cherry-pick][releases/v0.24.0rc][BugFix][P/D] Lower Mooncake KV metadata mismatch log level (from #13542) | 🔴 高 | closed | 2026-08-05 | [#13618](https://github.com/vllm-project/vllm-ascend/pull/13618) |
| 38 | vllm-ascend | PR | [Cherry-pick][releases/v0.25.1rc][BugFix][P/D] Lower Mooncake KV metadata mismatch log level (from #13542) | 🔴 高 | open | 2026-08-05 | [#13617](https://github.com/vllm-project/vllm-ascend/pull/13617) |
| 39 | vllm-ascend | PR | [Cherry-pick][releases/v0.26.0rc][BugFix][P/D] Lower Mooncake KV metadata mismatch log level (from #13542) | 🔴 高 | closed | 2026-08-05 | [#13616](https://github.com/vllm-project/vllm-ascend/pull/13616) |
| 40 | vllm-ascend | PR | [Cherry-pick][releases/v0.23.0][BugFix][P/D] Lower Mooncake KV metadata mismatch log level (from #13542) | 🔴 高 | closed | 2026-08-05 | [#13613](https://github.com/vllm-project/vllm-ascend/pull/13613) |
| 41 | vllm-ascend | PR | [BugFix][P/D] Lower Mooncake KV metadata mismatch log level | 🔴 高 | closed | 2026-08-05 | [#13542](https://github.com/vllm-project/vllm-ascend/pull/13542) |
| 42 | vllm | PR | [Bugfix][DeepSeek V4] Fix fused MTP RMSNorm dtype mismatch | 🔴 高 | closed | 2026-08-04 | [#50987](https://github.com/vllm-project/vllm/pull/50987) |
| 43 | vllm | Issue | [Bug]: EngineCore dies on first guided-decoding request when dspark speculative decoding is enabled (grammar bitmask width mismatch) | 🔴 高 | closed | 2026-08-03 | [#50924](https://github.com/vllm-project/vllm/issues/50924) |
| 44 | vllm | PR | [Doc] Expand PTX toolchain mismatch troubleshooting | 🔴 高 | open | 2026-08-03 | [#50872](https://github.com/vllm-project/vllm/pull/50872) |
| 45 | vllm | Issue | [Bug]: expanded_block_table_buffer width mismatch in the DSA indexer under DCP | 🔴 高 | closed | 2026-08-03 | [#50825](https://github.com/vllm-project/vllm/issues/50825) |
| 46 | vllm | PR | [Spec Decode] Fix FA3 AOT scheduler head count mismatch for DFlash/DSpark | 🔴 高 | open | 2026-08-01 | [#50694](https://github.com/vllm-project/vllm/pull/50694) |
| 47 | vllm-ascend | PR | [Cherry-pick][releases/v0.25.1rc][Doc][BugFix] Fix port mismatch in Kimi-K2.6 tutorial (from #13062) | 🔴 高 | closed | 2026-07-30 | [#13184](https://github.com/vllm-project/vllm-ascend/pull/13184) |
| 48 | vllm-ascend | PR | [Cherry-pick][releases/v0.24.0rc][Doc][BugFix] Fix port mismatch in Kimi-K2.6 tutorial (from #13062) | 🔴 高 | closed | 2026-07-29 | [#13064](https://github.com/vllm-project/vllm-ascend/pull/13064) |
| 49 | vllm-ascend | PR | [Doc][BugFix] Fix port mismatch in Kimi-K2.6 tutorial | 🔴 高 | closed | 2026-07-29 | [#13062](https://github.com/vllm-project/vllm-ascend/pull/13062) |
| 50 | vllm | PR | [Bugfix] Fix indexer expanded_block_table_buffer size mismatch | 🔴 高 | closed | 2026-07-27 | [#50050](https://github.com/vllm-project/vllm/pull/50050) |
| 51 | vllm | Issue | [Bug]: Qwen3.5/Qwen3-Next GDN attention crashes at warmup with torch.compile stride mismatch (expected size 3==3, stride 4097==512) — recurrence of #29014 on splitting_ops boundary | 🔴 高 | open | 2026-07-27 | [#50046](https://github.com/vllm-project/vllm/issues/50046) |
| 52 | vllm | PR | [XPU] WA topk_sigmoid argument mismatch for XPU platform | 🔴 高 | open | 2026-07-26 | [#49884](https://github.com/vllm-project/vllm/pull/49884) |
| 53 | vllm | Issue | [Bug]: flashinfer cubin version mismatch | 🔴 高 | closed | 2026-07-25 | [#49804](https://github.com/vllm-project/vllm/issues/49804) |
| 54 | vllm | Issue | [Bug]: Suspected RMSNorm precision regression after 225936a causes NGRAM/target token divergence for Qwen3-VL | 🔴 高 | open | 2026-07-23 | [#49616](https://github.com/vllm-project/vllm/issues/49616) |
| 55 | vllm | PR | [XPU] WA of topk_softmax / topk_sigmoid / topk_softplus arg mismatch on XPU | 🔴 高 | closed | 2026-07-22 | [#49428](https://github.com/vllm-project/vllm/pull/49428) |
| 56 | vllm | Issue | [Bug]: Streaming vs non-streaming content whitespace mismatch at tool-call boundaries (ParserEngine: qwen3/nemotron_v3/seed_oss/glm47_moe/gemma4) | 🔴 高 | open | 2026-07-22 | [#49412](https://github.com/vllm-project/vllm/issues/49412) |
| 57 | vllm | PR | [XPU] WA of topk_softplus_sqrt arg mismatch on XPU | 🔴 高 | closed | 2026-07-22 | [#49408](https://github.com/vllm-project/vllm/pull/49408) |
| 58 | vllm | PR | [XPU] WA of topk_softmax arg mismatch on XPU | 🔴 高 | closed | 2026-07-22 | [#49395](https://github.com/vllm-project/vllm/pull/49395) |
| 59 | vllm | Issue | [Bug]: GLM-4.1V fails to start at tensor-parallel size 32 — vision tower head-count mismatch | 🔴 高 | open | 2026-07-21 | [#49368](https://github.com/vllm-project/vllm/issues/49368) |
| 60 | vllm | PR | [Misc][Docs] Fix XPU compute-runtime driver link version mismatch | 🔴 高 | closed | 2026-07-21 | [#49299](https://github.com/vllm-project/vllm/pull/49299) |
| 61 | vllm | PR | [Bug Fix] DiffusionGemma get different outputs compared with transformers version (mismatch sliding window attention) | 🔴 高 | open | 2026-07-20 | [#49202](https://github.com/vllm-project/vllm/pull/49202) |
| 62 | vllm-ascend | PR | [Cherry-pick][releases/v0.24.0rc][BugFix]Fix precision issues caused by incorrect reordering timing leading to TP inequality (from #12359) | 🔴 高 | closed | 2026-07-20 | [#12382](https://github.com/vllm-project/vllm-ascend/pull/12382) |
| 63 | vllm-ascend | PR | [BugFix][v0.23.0] Fix precision issues caused by incorrect reordering timing leading to TP inequality | 🔴 高 | closed | 2026-07-19 | [#12371](https://github.com/vllm-project/vllm-ascend/pull/12371) |
| 64 | vllm-ascend | PR | [BugFix]Fix precision issues caused by incorrect reordering timing leading to TP inequality | 🔴 高 | closed | 2026-07-19 | [#12359](https://github.com/vllm-project/vllm-ascend/pull/12359) |
| 65 | vllm | PR | [BugFix] Handle per-group prefix-hit divergence for hybrid models with KV connector | 🔴 高 | closed | 2026-07-12 | [#48425](https://github.com/vllm-project/vllm/pull/48425) |
| 66 | vllm | PR | [BugFix] Fix per-group prefix-hit divergence for hybrid Mamba + KV connector | 🔴 高 | open | 2026-07-10 | [#48195](https://github.com/vllm-project/vllm/pull/48195) |
| 67 | vllm | PR | [Bugfix][Spec Decode] Fix DFlash draft/target layer-count mismatch | 🔴 高 | closed | 2026-07-09 | [#48113](https://github.com/vllm-project/vllm/pull/48113) |
| 68 | vllm | PR | [bugfix] bge-m3-sparse-plugin mismatch requests | 🔴 高 | closed | 2026-07-09 | [#48112](https://github.com/vllm-project/vllm/pull/48112) |
| 69 | vllm | PR | [XPU] Fix topk_sigmoid arg mismatch on XPU | 🔴 高 | closed | 2026-07-07 | [#47858](https://github.com/vllm-project/vllm/pull/47858) |
| 70 | vllm | PR | [Bugfix] Fix Gemma 4 MTP embedding dimension mismatch | 🔴 高 | closed | 2026-07-07 | [#47819](https://github.com/vllm-project/vllm/pull/47819) |
| 71 | vllm | Issue | [Performance]: Logprob divergence between vLLM and transformers on a fine-tuned Qwen3.5-VL mobile-use model | 🔴 高 | open | 2026-07-02 | [#47425](https://github.com/vllm-project/vllm/issues/47425) |
| 72 | vllm | PR | Detect CUDA toolchain/driver PTX mismatch on GB10, clarify removed env var | 🔴 高 | closed | 2026-07-02 | [#47399](https://github.com/vllm-project/vllm/pull/47399) |
| 73 | vllm | Issue | [Feature]: Detect CUDA toolchain/driver PTX version mismatch at startup and raise an actionable error | 🔴 高 | open | 2026-07-02 | [#47397](https://github.com/vllm-project/vllm/issues/47397) |
| 74 | vllm-ascend | Issue | [Bug]: 高精度推理性能评测后报错 request.num_output_placeholders变负报错AssertionError AsyncLLM output_handler failed | 🔴 高 | open | 2026-07-02 | [#11296](https://github.com/vllm-project/vllm-ascend/issues/11296) |
| 75 | vllm | PR | [Bugfix][Model] Harden DiffusionGemma self-conditioning matmul against torch.compile shape mismatch (#47129) | 🔴 高 | open | 2026-07-01 | [#47278](https://github.com/vllm-project/vllm/pull/47278) |
| 76 | vllm-ascend | PR | [BugFix] Fix precision divergence for EAGLE3 + async scheduling with multi speculative tokens | 🔴 高 | open | 2026-06-30 | [#11215](https://github.com/vllm-project/vllm-ascend/pull/11215) |
| 77 | vllm | Issue | [Bug]: Streaming vs non-streaming tool-parser divergence on a truncated `&lt;tool_call&gt;` opener (ParserEngine / Qwen3) | 🔴 高 | open | 2026-06-30 | [#47137](https://github.com/vllm-project/vllm/issues/47137) |
| 78 | vllm | Issue | [Bug]: DiffusionGemma fails to start with RuntimeError in _compiled_sample_step during torch.compile (matmul shape mismatch) | 🔴 高 | open | 2026-06-30 | [#47129](https://github.com/vllm-project/vllm/issues/47129) |
| 79 | vllm | PR | [Bugfix][MLA] Fix LSE log-base mismatch in DCP + FlashInfer MLA decode | 🔴 高 | closed | 2026-06-29 | [#47079](https://github.com/vllm-project/vllm/pull/47079) |
| 80 | vllm-ascend | Issue | [Bug]: Dynamic EPLB TP+EP E2E tests are broken on main due to expert index mismatch | 🔴 高 | open | 2026-06-29 | [#11150](https://github.com/vllm-project/vllm-ascend/issues/11150) |
| 81 | vllm | Issue | [Bug]: Gemma4 video prompt expansion / timestamps mismatch vs Transformers `Gemma4Processor` | 🔴 高 | closed | 2026-06-29 | [#46988](https://github.com/vllm-project/vllm/issues/46988) |
| 82 | vllm | Issue | [Bug]: Qwen3-VL video prompt expansion drops outer &lt;\|vision_start\|&gt;/&lt;\|vision_end\|&gt; wrapper, mismatching HuggingFace processor | 🔴 高 | open | 2026-06-26 | [#46817](https://github.com/vllm-project/vllm/issues/46817) |
| 83 | vllm | PR | [Bugfix] Fix MLA indexer block_table shape mismatch when max_model_len | 🔴 高 | open | 2026-06-26 | [#46794](https://github.com/vllm-project/vllm/pull/46794) |
| 84 | vllm | Issue | [Bug]: MLA indexer block_table shape mismatch when max_model_len is auto-reduced after InputBatch creation | 🔴 高 | open | 2026-06-26 | [#46787](https://github.com/vllm-project/vllm/issues/46787) |
| 85 | vllm | Issue | [Bug] DeepSeekV4-Flash produces incorrect output with inline system messages after PR #46025 when `preserved in-place` | 🔴 高 | open | 2026-06-25 | [#46710](https://github.com/vllm-project/vllm/issues/46710) |
| 86 | vllm | Issue | [Bug]: Hybrid Mamba + KV connector: per-group prefix-hit divergence and vllm engine crashed | 🔴 高 | open | 2026-06-23 | [#46453](https://github.com/vllm-project/vllm/issues/46453) |
| 87 | vllm-ascend | PR | [0.22.1][BugFix] Fix the number of parameter mismatch for function get_device_tensor() | 🔴 高 | closed | 2026-06-22 | [#10748](https://github.com/vllm-project/vllm-ascend/pull/10748) |
| 88 | vllm | PR | [Bugfix][Quant] Raise actionable error instead of bare assert for group-size/TP mismatch (#46230) | 🔴 高 | closed | 2026-06-20 | [#46236](https://github.com/vllm-project/vllm/pull/46236) |
| 89 | vllm | PR | [Bugfix] DeepseekV4 (nvidia): thread is_sequence_parallel into shared_experts to fix SP-MoE construct/load mismatch | 🔴 高 | closed | 2026-06-19 | [#46174](https://github.com/vllm-project/vllm/pull/46174) |
| 90 | vllm-ascend | PR | [BugFix] Fix the number of parameter mismatch for function get_device_tensor() | 🔴 高 | closed | 2026-06-17 | [#10607](https://github.com/vllm-project/vllm-ascend/pull/10607) |
| 91 | vllm-ascend | PR | [Doc][Model] Fix model name mismatch in DeepSeek-V4-Flash functional verification | 🔴 高 | closed | 2026-06-17 | [#10603](https://github.com/vllm-project/vllm-ascend/pull/10603) |
| 92 | vllm-ascend | PR | [BugFix] Fix _dummy_run warmup mismatch when using --mm-encoder-only | 🔴 高 | open | 2026-06-17 | [#10601](https://github.com/vllm-project/vllm-ascend/pull/10601) |
| 93 | vllm-ascend | Issue | [Bug]: Llama LoRA error about tensor dimensions mismatch when for einsum operator | 🔴 高 | open | 2026-06-17 | [#10577](https://github.com/vllm-project/vllm-ascend/issues/10577) |
| 94 | vllm-ascend | PR | [BugFix][PD] Fix memory leak from key mismatch in proc_not_transfer_request cleanup | 🔴 高 | open | 2026-06-16 | [#10554](https://github.com/vllm-project/vllm-ascend/pull/10554) |
| 95 | vllm-ascend | PR | [DOC] Fix model name mismatch in DeepSeek-V4-Flash Functional Verification | 🔴 高 | closed | 2026-06-16 | [#10527](https://github.com/vllm-project/vllm-ascend/pull/10527) |
| 96 | vllm | Issue | [Bug]: DeepSeek V4 TileLang MHC path produces incorrect output on gfx942 | 🔴 高 | closed | 2026-06-15 | [#45698](https://github.com/vllm-project/vllm/issues/45698) |
| 97 | vllm | PR | [speculative decoding] fix dynamic SD with parallel drafting (PARD/P-Eagle) slot layout mismatch | 🔴 高 | open | 2026-06-15 | [#45695](https://github.com/vllm-project/vllm/pull/45695) |
| 98 | vllm | PR | [Bugfix][Spec Decode] Fix drafter slot mapping type mismatch when DBO is enabled | 🔴 高 | open | 2026-06-12 | [#45378](https://github.com/vllm-project/vllm/pull/45378) |
| 99 | vllm | PR | [xpu][lora]: Align LoRA implementation with Punica GPU: fix _apply_expand rank mismatch, add_inputs hardcode, and MoE EP | 🔴 高 | closed | 2026-06-12 | [#45368](https://github.com/vllm-project/vllm/pull/45368) |
| 100 | vllm | Issue | [Bug][vllm-omni] Qwen3-TTS crashes under concurrent TTS with ref_context_size mismatch | 🔴 高 | open | 2026-06-08 | [#44933](https://github.com/vllm-project/vllm/issues/44933) |
| 101 | vllm | Issue | [Bug]: Inference-time probabilistic error: pre-allocated buffer size mismatch in indexer | 🔴 高 | open | 2026-06-08 | [#44827](https://github.com/vllm-project/vllm/issues/44827) |
| 102 | vllm | Issue | [Bug]: marlin_gemm shape mismatch (size_k doubled) for google/gemma-4-12B-it-qat-w4a16-ct on vLLM v0.22.0 | 🔴 高 | closed | 2026-06-07 | [#44796](https://github.com/vllm-project/vllm/issues/44796) |
| 103 | vllm | PR | [Benchmark] Auto-detect and correct client/server tokenizer mismatch for random dataset | 🔴 高 | closed | 2026-06-06 | [#44708](https://github.com/vllm-project/vllm/pull/44708) |
| 104 | vllm | PR | [Bugfix] MiniCPM-V-4.6 video inference crash: placeholder count mismatches visual embedding count | 🔴 高 | closed | 2026-06-04 | [#44509](https://github.com/vllm-project/vllm/pull/44509) |
| 105 | vllm | PR | [Bugfix][Model] Fix Qwen3 deepstack buffer device mismatch | 🔴 高 | open | 2026-06-03 | [#44384](https://github.com/vllm-project/vllm/pull/44384) |
| 106 | vllm-ascend | Issue | [Bug]: When the deepseek-v4 image is used to request to enable think and "response_format": {"type": "json_object"}, the parameter precision is incorrect. | 🔴 高 | closed | 2026-06-02 | [#9858](https://github.com/vllm-project/vllm-ascend/issues/9858) |
| 107 | vllm | Issue | [Bug]: Qwen3-VL EVS video pruning crashes with CPU/CUDA device mismatch in _create_final_video_embeddings | 🔴 高 | open | 2026-06-01 | [#44200](https://github.com/vllm-project/vllm/issues/44200) |
| 108 | vllm-ascend | PR | [BugFix][KV pool] DSv4 fix lookup/load mismatch | 🔴 高 | closed | 2026-05-30 | [#9745](https://github.com/vllm-project/vllm-ascend/pull/9745) |
| 109 | vllm | PR | [Bugfix] Fix Gemma4 MTP block_table batch_size mismatch under concurrent load | 🔴 高 | closed | 2026-05-29 | [#43982](https://github.com/vllm-project/vllm/pull/43982) |
| 110 | vllm | PR | [Bugfix][MLA] Fix LSE log-base mismatch in DCP + FlashInfer MLA decode | 🔴 高 | closed | 2026-05-28 | [#43919](https://github.com/vllm-project/vllm/pull/43919) |
| 111 | vllm | PR | [Bugfix][MiniCPM-o] Fix cuda/cpu device mismatch in Resampler2_5 pos_embed | 🔴 高 | closed | 2026-05-28 | [#43844](https://github.com/vllm-project/vllm/pull/43844) |
| 112 | vllm-ascend | PR | [BugFix]Fix hidden size dimension mismatch in first-layer attention o… | 🔴 高 | open | 2026-05-26 | [#9553](https://github.com/vllm-project/vllm-ascend/pull/9553) |
| 113 | vllm | PR | [Bugfix] Fix hash topk dtype mismatch | 🔴 高 | closed | 2026-05-22 | [#43425](https://github.com/vllm-project/vllm/pull/43425) |
| 114 | vllm | Issue | [Hybrid SSM] Investigate accuracy divergence between `mamba_chunk_scan` and `selective_state_update` kernels | 🔴 高 | open | 2026-05-21 | [#43301](https://github.com/vllm-project/vllm/issues/43301) |
| 115 | vllm-ascend | PR | [Doc][BugFix] Fix parameter mismatch and improve clarity in DeepSeek-V3.2.md | 🔴 高 | closed | 2026-05-20 | [#9381](https://github.com/vllm-project/vllm-ascend/pull/9381) |
| 116 | vllm-ascend | PR | [Doc][BugFix] Fix parameter mismatch in DeepSeek-V3.2.md | 🔴 高 | closed | 2026-05-20 | [#9369](https://github.com/vllm-project/vllm-ascend/pull/9369) |
| 117 | vllm | PR | [Bugfix] fix device mismatch in MiniCPM-o-4_5 resampler | 🔴 高 | closed | 2026-05-20 | [#43194](https://github.com/vllm-project/vllm/pull/43194) |
| 118 | vllm | Issue | [Bug]: vLLM 0.21: DeepSeek-V4-pro crashes with tensor size mismatch & CUBLAS error during PP+TP inference on 2x8 H800 | 🔴 高 | open | 2026-05-19 | [#43080](https://github.com/vllm-project/vllm/issues/43080) |
| 119 | vllm | PR | [Build] Use no-guess-dev version scheme to fix version mismatch on release branches | 🔴 高 | open | 2026-05-18 | [#42937](https://github.com/vllm-project/vllm/pull/42937) |
| 120 | vllm | Issue | [Bug]: vLLM wheel version mismatch  | 🔴 高 | open | 2026-05-18 | [#42932](https://github.com/vllm-project/vllm/issues/42932) |
| 121 | vllm | Issue | [Bug]: Using the /generative_scoring may cause shape mismatches in the rejection sampler, causing vllm serve to crash | 🔴 高 | open | 2026-05-14 | [#42592](https://github.com/vllm-project/vllm/issues/42592) |
| 122 | vllm | Issue | [Bug]: DFlash speculative decoding crashes with dtype mismatch: float != c10::Half in qwen3_dflash.py | 🔴 高 | closed | 2026-05-14 | [#42588](https://github.com/vllm-project/vllm/issues/42588) |
| 123 | vllm | PR | [Bugfix] Fixes MiniCPM-O resampler device placement to avoid tensor device mismatch | 🔴 高 | closed | 2026-05-11 | [#42332](https://github.com/vllm-project/vllm/pull/42332) |
| 124 | vllm-ascend | Issue | [Bug]: npu_grouped_matmul dimension mismatch with Qwen3-MoE under Expert Parallelism (EP) on Ascend 910B | 🔴 高 | open | 2026-05-09 | [#9007](https://github.com/vllm-project/vllm-ascend/issues/9007) |
| 125 | vllm | PR | [Bugfix] Fix mismatched kernel-per-logical blocks in NIXL HMA transfer | 🔴 高 | closed | 2026-05-08 | [#42097](https://github.com/vllm-project/vllm/pull/42097) |
| 126 | vllm | Issue | [Bug]: vLLM producing incorrect output for GLM-OCR | 🔴 高 | open | 2026-05-08 | [#42016](https://github.com/vllm-project/vllm/issues/42016) |
| 127 | vllm | Issue | [Usage]:  ValueError: mismatch of LoRA layer names for Gemma4 E2B trained with unsloth | 🔴 高 | open | 2026-05-05 | [#41702](https://github.com/vllm-project/vllm/issues/41702) |
| 128 | vllm | PR | Fix Gemma4 TritonAttention buffer mismatch for heterogeneous head_dim (#41656) | 🔴 高 | closed | 2026-05-04 | [#41656](https://github.com/vllm-project/vllm/pull/41656) |
| 129 | vllm | Issue | [Bug]: gpt-oss MoE moe_forward fake-kernel shape mismatch breaks torch.compile + TP &gt; 1 on Blackwell | 🔴 高 | closed | 2026-05-04 | [#41645](https://github.com/vllm-project/vllm/issues/41645) |
| 130 | vllm | PR | [ROCm] Fix TurboQuant shape mismatch on non-power-of-2 head_dim | 🔴 高 | open | 2026-05-04 | [#41597](https://github.com/vllm-project/vllm/pull/41597) |
| 131 | vllm | PR | [Model] Fix Gemma4 MoE activation mismatch | 🔴 高 | closed | 2026-05-03 | [#41574](https://github.com/vllm-project/vllm/pull/41574) |
| 132 | vllm | Issue | [Bug]: ROCM_ATTN produces incorrect output for LiquidAI LFM2 | 🔴 高 | closed | 2026-05-01 | [#41472](https://github.com/vllm-project/vllm/issues/41472) |
| 133 | vllm | Issue | [Bug] Gemma 4 31B crashes on k_eq_v full-attention layers (QKV split shape mismatch) | 🔴 高 | closed | 2026-04-29 | [#41283](https://github.com/vllm-project/vllm/issues/41283) |
| 134 | vllm | Issue | [Bug] _align_hybrid_block_size produces TP-dependent block sizes, currently unsupported when local and remote kernel block size mismatch | 🔴 高 | closed | 2026-04-27 | [#41037](https://github.com/vllm-project/vllm/issues/41037) |
| 135 | vllm | Issue | [Bug]: arange_buffer size mismatch | 🔴 高 | closed | 2026-04-27 | [#40983](https://github.com/vllm-project/vllm/issues/40983) |
| 136 | vllm | Issue | [Bug]: Triton MLA decode kernel shape mismatch for Mistral-Small on ROCm when TP &gt; 1 | 🔴 高 | closed | 2026-04-27 | [#40966](https://github.com/vllm-project/vllm/issues/40966) |
| 137 | vllm | Issue | [Bug]: DeepSeek-V4-Pro H200 DP+EP router dtype mismatch in topk_hash_softplus_sqrt (Long/Int inconsistency) | 🔴 高 | closed | 2026-04-25 | [#40862](https://github.com/vllm-project/vllm/issues/40862) |
| 138 | vllm | PR | [Bugfix] Fix device mismatch triggering in testing | 🔴 高 | closed | 2026-04-23 | [#40739](https://github.com/vllm-project/vllm/pull/40739) |
| 139 | vllm | PR | [fix] mismatch dim during capture graph if with --gpu-memory-utilization | 🔴 高 | open | 2026-04-23 | [#40719](https://github.com/vllm-project/vllm/pull/40719) |
| 140 | vllm-ascend | PR | [BugFix] Fix _dummy_run warmup mismatch when using --language-model-only | 🔴 高 | closed | 2026-04-22 | [#8556](https://github.com/vllm-project/vllm-ascend/pull/8556) |
| 141 | vllm-ascend | PR | [BugFix] Fix _dummy_run warmup mismatch when using --language-model-only | 🔴 高 | closed | 2026-04-22 | [#8549](https://github.com/vllm-project/vllm-ascend/pull/8549) |
| 142 | vllm | PR | [Model] fix(dflash): dtype mismatch in combine_hidden_states | 🔴 高 | closed | 2026-04-20 | [#40334](https://github.com/vllm-project/vllm/pull/40334) |
| 143 | vllm-ascend | PR | Revert "[BugFix] Fix dimension mismatch error when SP padding causes num_tokens_padded != num_tokens_unpadded (#7858)" | 🔴 高 | closed | 2026-04-18 | [#8414](https://github.com/vllm-project/vllm-ascend/pull/8414) |
| 144 | vllm-ascend | PR | Revert "[v0.18.0][BugFix] Fix dimension mismatch error when SP padding causes num_tokens_padded != num_tokens_unpadded" | 🔴 高 | closed | 2026-04-18 | [#8413](https://github.com/vllm-project/vllm-ascend/pull/8413) |
| 145 | vllm | PR | [Bugfix] Fix dtype mismatch in XDRotaryEmbedding for HunyuanOCR | 🔴 高 | open | 2026-04-17 | [#40180](https://github.com/vllm-project/vllm/pull/40180) |
| 146 | vllm | PR | docs: fix doc-code mismatches from audit | 🔴 高 | open | 2026-04-16 | [#40062](https://github.com/vllm-project/vllm/pull/40062) |
| 147 | vllm | Issue | [Bug]: Mismatch between batch queue size and max-num-seqs | 🔴 高 | open | 2026-04-16 | [#39981](https://github.com/vllm-project/vllm/issues/39981) |
| 148 | vllm | Issue | [Bug]: [CI] Nightly Docker image CUDA `libcudart.so` mismatch regression (from `bcc2306`) | 🔴 高 | closed | 2026-04-15 | [#39872](https://github.com/vllm-project/vllm/issues/39872) |
| 149 | vllm | PR | [Bugfix] Fail fast on custom AR graph buffer count mismatch | 🔴 高 | closed | 2026-04-14 | [#39791](https://github.com/vllm-project/vllm/pull/39791) |
| 150 | vllm | PR | [Bugfix] Fix mismatch between global and local attention heads in tensor-parallel mode for param2moe model | 🔴 高 | closed | 2026-04-13 | [#39707](https://github.com/vllm-project/vllm/pull/39707) |
| 151 | vllm | PR | [Bugfix][Kernel][ROCm] Fix triton_w4a16 scales mismatch when BLOCK_K &gt; group_size | 🔴 高 | closed | 2026-04-13 | [#39705](https://github.com/vllm-project/vllm/pull/39705) |
| 152 | vllm | Issue | Gemma 4 31B bnb-4bit: _Gemma4KVSharedSafeProxy AttributeError + weight shape mismatch | 🔴 高 | closed | 2026-04-12 | [#39638](https://github.com/vllm-project/vllm/issues/39638) |
| 153 | vllm | Issue | [Doc]: Docs audit: CLI, plugins, features, env vars, and auth mismatches | 🔴 高 | open | 2026-04-12 | [#39613](https://github.com/vllm-project/vllm/issues/39613) |
| 154 | vllm | PR | [Bugfix] Fix tensor shape mismatch in sparse attention with speculative decoding | 🔴 高 | closed | 2026-04-10 | [#39542](https://github.com/vllm-project/vllm/pull/39542) |
| 155 | vllm-ascend | PR | [v0.18.0][BugFix] Fix dimension mismatch error when SP padding causes num_tokens_padded != num_tokens_unpadded | 🔴 高 | closed | 2026-04-10 | [#8133](https://github.com/vllm-project/vllm-ascend/pull/8133) |
| 156 | vllm | PR | [Bugfix] Fix Gemma4 audio batch shape mismatch for concurrent requests | 🔴 高 | closed | 2026-04-09 | [#39459](https://github.com/vllm-project/vllm/pull/39459) |
| 157 | vllm | Issue | [Bug]: Kimi K2.5 multimodal inference broken — media_placeholder_token_id mismatch with runtime tokenizer | 🔴 高 | closed | 2026-04-08 | [#39261](https://github.com/vllm-project/vllm/issues/39261) |
| 158 | vllm | PR | Log warning for scheduled token mismatch | 🔴 高 | open | 2026-04-07 | [#39184](https://github.com/vllm-project/vllm/pull/39184) |
| 159 | vllm-ascend | Issue | [Bug]: MoE + MTP + FLASHCOMM1 causes tensor dimension mismatch in Qwen3.5-35B | 🔴 高 | closed | 2026-04-06 | [#7996](https://github.com/vllm-project/vllm-ascend/issues/7996) |
| 160 | vllm | Issue | [Bug]: Deepseek R1 produces incorrect output | 🔴 高 | closed | 2026-04-03 | [#38931](https://github.com/vllm-project/vllm/issues/38931) |
| 161 | vllm | PR | [Bugfix] Fix logger.warning format string arg mismatch in Qwen3XML tool parser | 🔴 高 | closed | 2026-04-03 | [#38890](https://github.com/vllm-project/vllm/pull/38890) |
| 162 | vllm | PR | Fix/p2p request id mismatch | 🔴 高 | closed | 2026-04-02 | [#38816](https://github.com/vllm-project/vllm/pull/38816) |
| 163 | vllm | Issue | [Bug]: Qwen3.5 (Qwen3_5ForConditionalGeneration) FLA linear attention tensor format mismatch causes gibberish output | 🔴 高 | open | 2026-03-31 | [#38643](https://github.com/vllm-project/vllm/issues/38643) |
| 164 | vllm-ascend | PR | [BugFix] Fix dimension mismatch error when SP padding causes num_tokens_padded != num_tokens_unpadded | 🔴 高 | closed | 2026-03-31 | [#7858](https://github.com/vllm-project/vllm-ascend/pull/7858) |
| 165 | vllm | PR | [Spec Decode] fix returning size mismatch on extract hidden states proposer | 🔴 高 | closed | 2026-03-31 | [#38610](https://github.com/vllm-project/vllm/pull/38610) |
| 166 | vllm | Issue | [Bug]: LoRA loading fails for Qwen 3.5 MoE (35b-A3b) due to expert module name mismatch | 🔴 高 | open | 2026-03-30 | [#38520](https://github.com/vllm-project/vllm/issues/38520) |
| 167 | vllm-ascend | PR | Debug/sfa cos shape mismatch | 🔴 高 | open | 2026-03-27 | [#7735](https://github.com/vllm-project/vllm-ascend/pull/7735) |
| 168 | vllm | PR | [Doc] fix selector/label mismatch in helm chart | 🔴 高 | closed | 2026-03-27 | [#38340](https://github.com/vllm-project/vllm/pull/38340) |
| 169 | vllm | PR | [Bugfix] Remove false-positive format mismatch warnings in FLA ops | 🔴 高 | closed | 2026-03-26 | [#38255](https://github.com/vllm-project/vllm/pull/38255) |
| 170 | vllm | PR | Fix hidden size mismatch in eagle3 nonparallel draft path (fixes #37966) | 🔴 高 | closed | 2026-03-25 | [#38073](https://github.com/vllm-project/vllm/pull/38073) |
| 171 | vllm-ascend | PR | [Bugfix] Fix hidden_states shape mismatch in AscendDraftModelProposer | 🔴 高 | closed | 2026-03-24 | [#7602](https://github.com/vllm-project/vllm-ascend/pull/7602) |
| 172 | vllm | PR | fix: set device for prepare_inputs_event to avoid device mismatch | 🔴 高 | closed | 2026-03-20 | [#37670](https://github.com/vllm-project/vllm/pull/37670) |
| 173 | vllm | PR | Fix tensor size mismatch in per-channel weight scale loading for MoE … | 🔴 高 | closed | 2026-03-18 | [#37403](https://github.com/vllm-project/vllm/pull/37403) |
| 174 | vllm-ascend | Issue | [Bug]: Tensor mismatch when enabling MTP speculative decoding + PCP | 🔴 高 | open | 2026-03-17 | [#7375](https://github.com/vllm-project/vllm-ascend/issues/7375) |
| 175 | vllm | PR | [Bugfix] dtype mismatch in ngram gpu propose | 🔴 高 | closed | 2026-03-17 | [#37246](https://github.com/vllm-project/vllm/pull/37246) |
| 176 | vllm | PR | [Bugfix] Fix prompt_embeds precision divergence with MTP speculative … | 🔴 高 | closed | 2026-03-16 | [#37170](https://github.com/vllm-project/vllm/pull/37170) |
| 177 | vllm | PR | Fix issue #37103: Remove shape mismatch warnings in FLA operations | 🔴 高 | closed | 2026-03-16 | [#37166](https://github.com/vllm-project/vllm/pull/37166) |
| 178 | vllm | PR | Fix issue #37103: Remove shape mismatch warnings in FLA operations | 🔴 高 | closed | 2026-03-16 | [#37163](https://github.com/vllm-project/vllm/pull/37163) |
| 179 | vllm | PR | Fix issue #37103: Remove shape mismatch warnings in FLA operations | 🔴 高 | closed | 2026-03-16 | [#37161](https://github.com/vllm-project/vllm/pull/37161) |
| 180 | vllm | Issue | [Bug]: UserWarning: Input tensor shape suggests potential format mismatch | 🔴 高 | closed | 2026-03-15 | [#37103](https://github.com/vllm-project/vllm/issues/37103) |
| 181 | vllm-ascend | PR | fix: RotaryEmbedding parameter mismatch in Multi-Node-DP (#3633) | 🔴 高 | closed | 2026-03-14 | [#7258](https://github.com/vllm-project/vllm-ascend/pull/7258) |
| 182 | vllm | PR | [Bugfix] Fix tool call streaming JSON separator mismatch | 🔴 高 | closed | 2026-03-12 | [#36866](https://github.com/vllm-project/vllm/pull/36866) |
| 183 | vllm-ascend | Issue | [Usage]: 运行test_dispatch_ffn_combine.py测试用例时，out输出全为0，是精度有问题还是测试用例有问题？ | 🔴 高 | closed | 2026-03-12 | [#7189](https://github.com/vllm-project/vllm-ascend/issues/7189) |
| 184 | vllm | Issue | [Bug]: DeepSeek-OCR v1 crashes with TensorSchema mismatch when images_crop is empty (small images ≤640px) | 🔴 高 | closed | 2026-03-10 | [#36669](https://github.com/vllm-project/vllm/issues/36669) |
| 185 | vllm | Issue | [Bug]: qwen3.5 Mismatch in `image` token count between text and `input_ids`. Got ids=[4091] | 🔴 高 | closed | 2026-03-10 | [#36653](https://github.com/vllm-project/vllm/issues/36653) |
| 186 | vllm | PR | fix(lora): fix IndexError and GQA tensor size mismatch in QKV LoRA la… | 🔴 高 | closed | 2026-03-10 | [#36603](https://github.com/vllm-project/vllm/pull/36603) |
| 187 | vllm | PR | Guard AWQ-Marlin auto-selection on CUDA driver/toolkit mismatch | 🔴 高 | open | 2026-03-10 | [#36579](https://github.com/vllm-project/vllm/pull/36579) |
| 188 | vllm | Issue | [CPU] AssertionError in CPUModelRunner: device_tensor type mismatch (numpy.ndarray) | 🔴 高 | closed | 2026-03-08 | [#36382](https://github.com/vllm-project/vllm/issues/36382) |
| 189 | vllm-ascend | PR | [Bugfix][Model] Fix Qwen3.5 multimodal placeholder token mismatch | 🔴 高 | closed | 2026-03-07 | [#7055](https://github.com/vllm-project/vllm-ascend/pull/7055) |
| 190 | vllm | PR | [Bugfix] Fix Qwen3-VL timestamp mismatch when using num_frames without fps | 🔴 高 | closed | 2026-03-05 | [#36136](https://github.com/vllm-project/vllm/pull/36136) |
| 191 | vllm | PR | [ROCm][CI] Fix logprob divergence for TitanML/tiny-mixtral under AITER rms_norm | 🔴 高 | closed | 2026-03-05 | [#36101](https://github.com/vllm-project/vllm/pull/36101) |
| 192 | vllm | PR | [ROCm] Fix fused_moe_fake signature mismatch and other AITER bugs | 🔴 高 | closed | 2026-03-05 | [#36100](https://github.com/vllm-project/vllm/pull/36100) |
| 193 | vllm-ascend | PR | [Ops][BugFix] Fix RoPE shape mismatch for mtp models with flashcomm v1 enabled | 🔴 高 | closed | 2026-03-03 | [#6939](https://github.com/vllm-project/vllm-ascend/pull/6939) |
| 194 | vllm | PR | Fix: DBO + DSA shape mismatch | 🔴 高 | open | 2026-03-02 | [#35802](https://github.com/vllm-project/vllm/pull/35802) |
| 195 | vllm | Issue | [Bug]: DSA + Dual batch overlap shape mismatch | 🔴 高 | closed | 2026-03-02 | [#35795](https://github.com/vllm-project/vllm/issues/35795) |
| 196 | vllm-ascend | PR |  Fix RoPE shape mismatch for mtp models with flashcomm v1 enabled | 🔴 高 | closed | 2026-02-28 | [#6870](https://github.com/vllm-project/vllm-ascend/pull/6870) |
| 197 | vllm | Issue | [Bug]: PyTorch version mismatch for uploaded Pypi vLLM wheel v0.16.0 | 🔴 高 | closed | 2026-02-26 | [#35379](https://github.com/vllm-project/vllm/issues/35379) |
| 198 | vllm-ascend | PR | [Ops][BugFix] Fix RoPE shape mismatch for mtp models with shared expert DP | 🔴 高 | closed | 2026-02-26 | [#6821](https://github.com/vllm-project/vllm-ascend/pull/6821) |
| 199 | vllm | Issue | [Feature]: Add ISA-level smoke tests using Intel SDE to catch instruction set mismatches | 🔴 高 | closed | 2026-02-25 | [#35300](https://github.com/vllm-project/vllm/issues/35300) |
| 200 | vllm | PR | [Bugfix] Fix dtype mismatch in RMSNormGated.forward_native() during torch.compile | 🔴 高 | closed | 2026-02-25 | [#35256](https://github.com/vllm-project/vllm/pull/35256) |
| 201 | vllm | Issue | Qwen3.5-27B dtype mismatch in DeltaNet layers during torch.compile (float != c10::Half) | 🔴 高 | closed | 2026-02-24 | [#35238](https://github.com/vllm-project/vllm/issues/35238) |
| 202 | vllm | PR | [CI/Build] Fix gRPC version mismatch | 🔴 高 | closed | 2026-02-21 | [#35013](https://github.com/vllm-project/vllm/pull/35013) |
| 203 | vllm | Issue | [CI] Maverick model QKV weight shape mismatch during load_weights with expert parallelism | 🔴 高 | closed | 2026-02-20 | [#34995](https://github.com/vllm-project/vllm/issues/34995) |
| 204 | vllm | PR | [Bugfix] Fix block_size mismatch for MLA models after #34818 | 🔴 高 | closed | 2026-02-20 | [#34970](https://github.com/vllm-project/vllm/pull/34970) |
| 205 | vllm | PR | [ROCm][Test] Fix beam search determinism failures from batch-size-dependent FP divergence and removed wrong marker | 🔴 高 | closed | 2026-02-19 | [#34878](https://github.com/vllm-project/vllm/pull/34878) |
| 206 | vllm | PR | [WIP][Bugfix] Detect LoRA module name prefix mismatches at load time | 🔴 高 | closed | 2026-02-18 | [#34803](https://github.com/vllm-project/vllm/pull/34803) |
| 207 | vllm | PR | [WIP][Bugfix] Fix type mismatch in causal_conv1d Triton kernels PAD_SLOT_ID checks | 🔴 高 | closed | 2026-02-17 | [#34685](https://github.com/vllm-project/vllm/pull/34685) |
| 208 | vllm | PR | [Bugfix] Fix P2pNcclConnector NCCL send/recv key mismatch in disaggregated prefill XpYd | 🔴 高 | closed | 2026-02-10 | [#34278](https://github.com/vllm-project/vllm/pull/34278) |
| 209 | vllm | Issue | [Bug]: P2pNcclConnector NCCL send/recv key mismatch in disaggregated prefill XpYd architecture due to assign_request_id() random suffix | 🔴 高 | closed | 2026-02-10 | [#34277](https://github.com/vllm-project/vllm/issues/34277) |
| 210 | vllm-ascend | PR | [fix bug] fix tensor mismatch bug in sigmoid operate test case | 🔴 高 | closed | 2026-02-09 | [#6619](https://github.com/vllm-project/vllm-ascend/pull/6619) |
| 211 | vllm | Issue | [Bug]: required numpy version mismatch between modules | 🔴 高 | closed | 2026-02-07 | [#34041](https://github.com/vllm-project/vllm/issues/34041) |
| 212 | vllm-ascend | PR | [BugFix] Fix actual_seq_lengths_q mismatch in eagle proposer first step | 🔴 高 | closed | 2026-02-06 | [#6596](https://github.com/vllm-project/vllm-ascend/pull/6596) |
| 213 | vllm-ascend | PR | [fix bug] fix tensor mismatch bug in sigmoid operate test case | 🔴 高 | closed | 2026-02-06 | [#6595](https://github.com/vllm-project/vllm-ascend/pull/6595) |
| 214 | vllm-ascend | PR | [fix bug] fix tensor mismatch bug in sigmoid operate test case | 🔴 高 | closed | 2026-02-05 | [#6580](https://github.com/vllm-project/vllm-ascend/pull/6580) |
| 215 | vllm-ascend | PR | [BugFix]  Fixed an accuracy issue caused by the fact that the attention calculation result was not saved to the output when the FIA used the sliding window method. | 🔴 高 | closed | 2026-02-05 | [#6558](https://github.com/vllm-project/vllm-ascend/pull/6558) |
| 216 | vllm | PR | [Bugfix] Fix _fused_moe_lora_expand signature mismatch | 🔴 高 | closed | 2026-02-04 | [#33821](https://github.com/vllm-project/vllm/pull/33821) |
| 217 | vllm | Issue | [Bug]: LoRA dtype mismatch in XPU and CPU punica wrappers | 🔴 高 | closed | 2026-02-03 | [#33704](https://github.com/vllm-project/vllm/issues/33704) |
| 218 | vllm | PR | [Bugfix] Fix gpt-oss chat format mismatch with HuggingFace | 🔴 高 | closed | 2026-02-01 | [#33514](https://github.com/vllm-project/vllm/pull/33514) |
| 219 | vllm | Issue | [v0.13.0] Required Transformers version mismatch | 🔴 高 | closed | 2026-01-31 | [#33460](https://github.com/vllm-project/vllm/issues/33460) |
| 220 | vllm | Issue | [Bug]: gpt-oss chat format mismatch with HF apply_chat_template | 🔴 高 | closed | 2026-01-28 | [#33210](https://github.com/vllm-project/vllm/issues/33210) |
| 221 | vllm | PR | [ROCm][Bugfix] Resolve request_id mismatch and prevent crashes in Disaggregated Serving for moriio kv-connector | 🔴 高 | closed | 2026-01-20 | [#32630](https://github.com/vllm-project/vllm/pull/32630) |
| 222 | vllm-ascend | Issue | [Bug]: Transposing during wake_up causing MoE shape mismatch in RL scenarios. | 🔴 高 | open | 2026-01-15 | [#5915](https://github.com/vllm-project/vllm-ascend/issues/5915) |
| 223 | vllm | PR | [Bugfix] Fix xgrammar dtype mismatch on macOS CPU inference | 🔴 高 | closed | 2026-01-15 | [#32384](https://github.com/vllm-project/vllm/pull/32384) |
| 224 | vllm-ascend | Issue | [Bug]:  Incorrect k/v reshaping in `_forward_v1_style` causes TND layout dimension mismatch for `npu_fused_infer_attention_score` | 🔴 高 | closed | 2026-01-13 | [#5838](https://github.com/vllm-project/vllm-ascend/issues/5838) |
| 225 | vllm | PR | [ROCm][Bugfix] Fix Mamba batched decode producing incorrect output | 🔴 高 | closed | 2026-01-10 | [#32099](https://github.com/vllm-project/vllm/pull/32099) |
| 226 | vllm-ascend | Issue | [Bug]: vllm-ascend 0.11.0 does not compatible with vllm 0.11.0 for mismatch of pytorch version | 🔴 高 | closed | 2026-01-08 | [#5710](https://github.com/vllm-project/vllm-ascend/issues/5710) |
| 227 | vllm-ascend | Issue | [BUG]Structured output crash on v0.13.0rc1/main: xgrammar indices type mismatch | 🔴 高 | closed | 2026-01-06 | [#5636](https://github.com/vllm-project/vllm-ascend/issues/5636) |
| 228 | vllm | Issue | [Bug]: TypeError in DeviceCommunicatorBase.dispatch due to method signature mismatch | 🔴 高 | closed | 2026-01-03 | [#31642](https://github.com/vllm-project/vllm/issues/31642) |
| 229 | vllm-ascend | Issue | [Bug][Structured Output]: xgrammar type mismatching error on `apply_token_bitmask_inplace_cpu` | 🔴 高 | closed | 2025-12-30 | [#5524](https://github.com/vllm-project/vllm-ascend/issues/5524) |
| 230 | vllm | PR | [Model] Fix hunyuan-vl shape mismatch | 🔴 高 | closed | 2025-12-27 | [#31403](https://github.com/vllm-project/vllm/pull/31403) |
| 231 | vllm | PR | [Bug] Fix Qwen3-VL 2:4 sparsity shape mismatch during decompression | 🔴 高 | closed | 2025-12-23 | [#31231](https://github.com/vllm-project/vllm/pull/31231) |
| 232 | vllm | PR | [Bugfix] Fix shape mismatch in sparse 2:4 bitmask decompression for vision models | 🔴 高 | closed | 2025-12-22 | [#31130](https://github.com/vllm-project/vllm/pull/31130) |
| 233 | vllm | Issue | [Bug]: GLM4-MoE AssertionError with --all2all-backend pplx (dtype mismatch) | 🔴 高 | closed | 2025-12-20 | [#31056](https://github.com/vllm-project/vllm/issues/31056) |
| 234 | vllm | Issue | [Bug]: Qwen3-VL 2:4 sparsity llm-compressor RuntimeError: shape mismatch (0.12, 0.13rc2) | 🔴 高 | closed | 2025-12-19 | [#31019](https://github.com/vllm-project/vllm/issues/31019) |
| 235 | vllm-ascend | PR | [Fix] Synchronize the host query_start_loc with device values to prevent shape mismatches | 🔴 高 | closed | 2025-12-17 | [#5134](https://github.com/vllm-project/vllm-ascend/pull/5134) |
| 236 | vllm-ascend | PR | Fix FIA query and query_start_loc shape mismatch error | 🔴 高 | closed | 2025-12-10 | [#4851](https://github.com/vllm-project/vllm-ascend/pull/4851) |
| 237 | vllm | Issue | [Bug]: NIXL PD disaggregate with host_buffer has accuracy issue - Prefill scheduled num_block mismatch at update_state_after_alloc and request_finished | 🔴 高 | closed | 2025-12-09 | [#30358](https://github.com/vllm-project/vllm/issues/30358) |
| 238 | vllm-ascend | PR | BugFix: Resolve shape mismatch in eplb update and calculation issues in quant_apply_mlp | 🔴 高 | closed | 2025-12-08 | [#4777](https://github.com/vllm-project/vllm-ascend/pull/4777) |
| 239 | vllm | PR | fix: Force float16 dtype for GGUF models to fix incorrect output | 🔴 高 | closed | 2025-12-04 | [#30090](https://github.com/vllm-project/vllm/pull/30090) |
| 240 | vllm-ascend | PR | [Bugfix] Fix Dcp dimension mismatch when enable Mlapo | 🔴 高 | closed | 2025-12-04 | [#4687](https://github.com/vllm-project/vllm-ascend/pull/4687) |
| 241 | vllm | PR | BUGFIX: Handle layer name mismatches in pipeline parallel training in V1 engine | 🔴 高 | closed | 2025-12-01 | [#29785](https://github.com/vllm-project/vllm/pull/29785) |
| 242 | vllm | Issue | [Installation]: RM has detected an NVML/RM version mismatch | 🔴 高 | closed | 2025-11-29 | [#29737](https://github.com/vllm-project/vllm/issues/29737) |
| 243 | vllm-ascend | PR | [Fix] Fix FIA `query` and `query_start_loc` shape mismatch error | 🔴 高 | closed | 2025-11-28 | [#4518](https://github.com/vllm-project/vllm-ascend/pull/4518) |
| 244 | vllm | PR | [BugFix] Fix `plan` API Mismatch when using latest FlashInfer | 🔴 高 | closed | 2025-11-25 | [#29426](https://github.com/vllm-project/vllm/pull/29426) |
| 245 | vllm | Issue | [Bug]: stride mismatch when using torch compile on graphs with splitting_ops and non-standard tensor dimensions | 🔴 高 | closed | 2025-11-19 | [#29014](https://github.com/vllm-project/vllm/issues/29014) |
| 246 | vllm | PR | [CI/Build] Fix wheel filename&lt;&gt;metadata mismatch for uv compatibility | 🔴 高 | closed | 2025-11-19 | [#29009](https://github.com/vllm-project/vllm/pull/29009) |
| 247 | vllm | Issue | [Bug]: Illegal Memory Access and Incorrect Output with Long Inputs (&gt;5k tokens) on qweN3-next-80b-instruct | 🔴 高 | closed | 2025-11-17 | [#28835](https://github.com/vllm-project/vllm/issues/28835) |
| 248 | vllm | Issue | vLLM 0.11.0 CUDA Library Mismatch on ARM64 with CUDA 13.x | 🔴 高 | closed | 2025-11-13 | [#28669](https://github.com/vllm-project/vllm/issues/28669) |
| 249 | vllm-ascend | Issue | [Bug]: A Server error '500 Internal Server Error' occurred during the DeepSeek V3 accuracy test, resulting in low accuracy | 🔴 高 | closed | 2025-11-04 | [#3966](https://github.com/vllm-project/vllm-ascend/issues/3966) |
| 250 | vllm | PR | [ROCm] Fix DeepSeek R1/V3 incorrect output in eager mode. | 🔴 高 | closed | 2025-10-23 | [#27392](https://github.com/vllm-project/vllm/pull/27392) |
| 251 | vllm | Issue | [Bug]: sglang Qwen3-VL-235B-A22B-Instruct 图片token占位符报错Mismatch: More '&lt;\|vision_start\|&gt;&lt;\|image_pad\|&gt;&lt;\|vision_end\|&gt;' tokens found than corresponding data items provided. | 🔴 高 | closed | 2025-10-22 | [#27349](https://github.com/vllm-project/vllm/issues/27349) |
| 252 | vllm-ascend | PR | [BugFix] Fix the output dimension mismatch issue for full FA | 🔴 高 | closed | 2025-10-19 | [#3539](https://github.com/vllm-project/vllm-ascend/pull/3539) |
| 253 | vllm | Issue | [Bug]: Incorrect outputs with batch size &gt; 1 on AArch64 CPU | 🔴 高 | closed | 2025-10-16 | [#27034](https://github.com/vllm-project/vllm/issues/27034) |
| 254 | vllm | PR | [Bugfix][Multi Modal] Fix incorrect output in Molmo | 🔴 高 | closed | 2025-10-09 | [#26518](https://github.com/vllm-project/vllm/pull/26518) |
| 255 | vllm | Issue | [Bug]: Molmo produces incorrect outputs | 🔴 高 | closed | 2025-10-08 | [#26451](https://github.com/vllm-project/vllm/issues/26451) |
| 256 | vllm | PR | fix[DP][v1]: Prevent hangs from mismatched worker configurations | 🔴 高 | closed | 2025-10-04 | [#26218](https://github.com/vllm-project/vllm/pull/26218) |
| 257 | vllm | Issue | [Bug]:  vLLM merges special XML-like tags into single tokens, causing tokenization mismatch with HuggingFace | 🔴 高 | closed | 2025-10-02 | [#26121](https://github.com/vllm-project/vllm/issues/26121) |
| 258 | vllm | Issue | [Bug]: Incorrect outputs in MLA with chunked prefill | 🔴 高 | closed | 2025-10-01 | [#26042](https://github.com/vllm-project/vllm/issues/26042) |
| 259 | vllm | Issue | [Bug]: Mismatched number of arguments | 🔴 高 | closed | 2025-09-30 | [#25929](https://github.com/vllm-project/vllm/issues/25929) |
| 260 | vllm-ascend | PR | [Bugfix][LoRA] Fix forward error and shape mismatch when using LoRA | 🔴 高 | closed | 2025-09-24 | [#3153](https://github.com/vllm-project/vllm-ascend/pull/3153) |
| 261 | vllm | Issue | [Bug]: Mismatch with prompt logprobs with the same prompt | 🔴 高 | closed | 2025-09-19 | [#25262](https://github.com/vllm-project/vllm/issues/25262) |
| 262 | vllm | Issue | [Bug]: Ray distributed executor backend: CUDA device mismatch when using multiple GPUs on a single node | 🔴 高 | closed | 2025-09-18 | [#25113](https://github.com/vllm-project/vllm/issues/25113) |
| 263 | vllm | PR | Fix random dataset mismatched token length with config. | 🔴 高 | closed | 2025-09-16 | [#24937](https://github.com/vllm-project/vllm/pull/24937) |
| 264 | vllm | PR | Fix implementation divergence for BLOOM models between vLLM and HuggingFace when using prompt embeds | 🔴 高 | closed | 2025-09-11 | [#24686](https://github.com/vllm-project/vllm/pull/24686) |
| 265 | vllm | PR | [Bug] [Spec Decode] Fix model_initialization test and mismatch in aux_hidden_layers | 🔴 高 | closed | 2025-09-10 | [#24613](https://github.com/vllm-project/vllm/pull/24613) |
| 266 | vllm | PR | [BugFix][Multi Modal] Fix TensorSchema shape mismatch in Molmo | 🔴 高 | closed | 2025-09-10 | [#24559](https://github.com/vllm-project/vllm/pull/24559) |
| 267 | vllm-ascend | Issue | [Bug]: doctest failed due to rotary_embedding signatures mismatch | 🔴 高 | closed | 2025-09-04 | [#2742](https://github.com/vllm-project/vllm-ascend/issues/2742) |
| 268 | vllm | Issue | [Bug]: uv installation seems broken for nightly wheels (dependency problem with `outlines` and filename&lt;&gt;metadata mismatch) | 🔴 高 | closed | 2025-09-02 | [#24126](https://github.com/vllm-project/vllm/issues/24126) |
| 269 | vllm | PR | [BugFix][AMD][Deepseek] fix a dtype mismatch error for deepseek running on AMD | 🔴 高 | closed | 2025-08-28 | [#23864](https://github.com/vllm-project/vllm/pull/23864) |
| 270 | vllm | Issue | [Bug]: Incorrect output throughput calculation for concurrent requests in benchmark_serving.py | 🔴 高 | closed | 2025-08-28 | [#23820](https://github.com/vllm-project/vllm/issues/23820) |
| 271 | vllm-ascend | Issue | [Bug]: (0.9.2rc1) Run bge-m3 with 310P3 with TP=2 got shape mismatch in rank1 | 🔴 高 | closed | 2025-08-28 | [#2593](https://github.com/vllm-project/vllm-ascend/issues/2593) |
| 272 | vllm | PR | Adapting Qwen3-32B to Eagle3 mode to resolve head dimension mismatch issues | 🔴 高 | closed | 2025-08-27 | [#23740](https://github.com/vllm-project/vllm/pull/23740) |
| 273 | vllm-ascend | PR | [feat] Adapting Qwen3-32B to Eagle3 mode to resolve head dimension (headdim) mismatch issues | 🔴 高 | closed | 2025-08-27 | [#2578](https://github.com/vllm-project/vllm-ascend/pull/2578) |
| 274 | vllm-ascend | PR | [Bugfix] Fix the bug of incorrect precision | 🔴 高 | closed | 2025-08-21 | [#2479](https://github.com/vllm-project/vllm-ascend/pull/2479) |
| 275 | vllm | PR | fix: qwen3coder stream output tool parameter type mismatch | 🔴 高 | closed | 2025-08-21 | [#23324](https://github.com/vllm-project/vllm/pull/23324) |
| 276 | vllm | PR | [CI/Build] Fix param mismatch in `test_eagle_correctness` | 🔴 高 | closed | 2025-08-13 | [#22847](https://github.com/vllm-project/vllm/pull/22847) |
| 277 | vllm | Issue | [Bug]: Possible mismatch in `truncate_prompt_tokens` value validation for `-1` | 🔴 高 | closed | 2025-08-11 | [#22635](https://github.com/vllm-project/vllm/issues/22635) |
| 278 | vllm | Issue | [CI Failure]: Distributed Tests (2 GPUs) - Mllama TP=2 results divergence and deadlock issue | 🔴 高 | closed | 2025-08-09 | [#22559](https://github.com/vllm-project/vllm/issues/22559) |
| 279 | vllm | Issue | [Bug]: GLM-4.1V lora trained model reports target_module mismatch error | 🔴 高 | closed | 2025-08-01 | [#22077](https://github.com/vllm-project/vllm/issues/22077) |
| 280 | vllm | Issue | [Bug]: Processor mismatch between what is provided by OpenGVLab and VLLM for InternVL leading to outputs of the processor being too large to be decoded for the tokenizer | 🔴 高 | closed | 2025-07-30 | [#21899](https://github.com/vllm-project/vllm/issues/21899) |
| 281 | vllm-ascend | PR | [Refactor] Improve debug message for input address mismatch | 🔴 高 | closed | 2025-07-26 | [#2026](https://github.com/vllm-project/vllm-ascend/pull/2026) |
| 282 | vllm | Issue | [Bug]: Incorrect output when using LoRA modules with tensor parallelism in vLLM | 🔴 高 | closed | 2025-07-23 | [#21471](https://github.com/vllm-project/vllm/issues/21471) |
| 283 | vllm | Issue | [Bug]: [V1 Engine] GLM4-1V video processing fails with token count mismatch: "Attempted to assign X multimodal tokens to Y placeholders" | 🔴 高 | closed | 2025-07-10 | [#20742](https://github.com/vllm-project/vllm/issues/20742) |
| 284 | vllm | Issue | [Bug]: Tensor dimension mismatch when loading Qwen3-Reranker-4B with tensor parallel &gt; 1 | 🔴 高 | closed | 2025-07-09 | [#20670](https://github.com/vllm-project/vllm/issues/20670) |
| 285 | vllm-ascend | PR | [Bugfix] LoRA logits einsum dimension mismatch in add_lora_logits | 🔴 高 | closed | 2025-07-02 | [#1583](https://github.com/vllm-project/vllm-ascend/pull/1583) |
| 286 | vllm | Issue | [Bug]: mismatch between multimodal tokens and placeholders for Qwen_2.5-3B (4 GPUs*24G) | 🔴 高 | closed | 2025-06-15 | [#19666](https://github.com/vllm-project/vllm/issues/19666) |
| 287 | vllm-ascend | Issue | [Bug]: Inference precision mismatch with DeepSeek-V2-Lite when using TP=2 and DP=2, enable expert-parallel | 🔴 高 | closed | 2025-06-11 | [#1171](https://github.com/vllm-project/vllm-ascend/issues/1171) |
| 288 | vllm | PR | Protect vllm engine from crash due to the mismatch of modality and model | 🔴 高 | closed | 2025-06-07 | [#19306](https://github.com/vllm-project/vllm/pull/19306) |
| 289 | vllm | PR | [BugFix][FlashInfer] Fix attention backend interface mismatch with unexpected keyword `use_irope` | 🔴 高 | closed | 2025-06-04 | [#19134](https://github.com/vllm-project/vllm/pull/19134) |
| 290 | vllm | Issue | [Usage]: Implement Method to Obtain Token-Level Log Probabilities from Models with Different Weights for KL Divergence Calculation | 🔴 高 | closed | 2025-06-04 | [#19127](https://github.com/vllm-project/vllm/issues/19127) |
| 291 | vllm | PR | [BugFix][FlashInfer] Fix attention backend interface mismatch with unexpected keyword `use_irope` | 🔴 高 | closed | 2025-06-03 | [#19116](https://github.com/vllm-project/vllm/pull/19116) |
| 292 | vllm | PR | [Bugfix][Core] Prefix caching causes incorrect outputs due to outdated ComputedBlocksTracker | 🔴 高 | closed | 2025-05-30 | [#18957](https://github.com/vllm-project/vllm/pull/18957) |
| 293 | vllm-ascend | PR | [WIP][BugFix]Fix accuracy issues caused by wrong etp_size passed into FusedMoEParallelConfig when using vLLM 0.9.0 | 🔴 高 | closed | 2025-05-26 | [#961](https://github.com/vllm-project/vllm-ascend/pull/961) |
| 294 | vllm | Issue | [Usage]: Dimension mismatch occurs when using vllm to load Eagle3 weights | 🔴 高 | closed | 2025-05-11 | [#17957](https://github.com/vllm-project/vllm/issues/17957) |
| 295 | vllm | PR | [BugFix][Spec Decode] Fix hidden size mismatch between target and eagle head | 🔴 高 | closed | 2025-05-06 | [#17740](https://github.com/vllm-project/vllm/pull/17740) |
| 296 | vllm | Issue | [Bug]: Aria model error due to version mismatch with transformers | 🔴 高 | closed | 2025-04-23 | [#17077](https://github.com/vllm-project/vllm/issues/17077) |
| 297 | vllm | Issue | [Bug]: Shape Mismatch Error with Image Input in vLLM 0.8.4 using Mistral-Small-3.1-24B-Instruct-2503 | 🔴 高 | closed | 2025-04-15 | [#16661](https://github.com/vllm-project/vllm/issues/16661) |
| 298 | vllm | Issue | [Bug]: V0 engines gives incorrect output for Moonlight model | 🔴 高 | closed | 2025-04-15 | [#16658](https://github.com/vllm-project/vllm/issues/16658) |
| 299 | vllm | Issue | [Bug]: value unpack mismatch in TP(0.8.3) and EP(0.8.2) | 🔴 高 | closed | 2025-04-14 | [#16581](https://github.com/vllm-project/vllm/issues/16581) |
| 300 | vllm | Issue | [Bug]: Possible shape mismatch for weights of gemma3 27b bnb 4bit quant from Unsloth | 🔴 高 | closed | 2025-04-02 | [#15959](https://github.com/vllm-project/vllm/issues/15959) |
| 301 | vllm | Issue | [Feature]: Add Warning for Chat Template Mismatches similar to SGLang | 🔴 高 | closed | 2025-03-24 | [#15395](https://github.com/vllm-project/vllm/issues/15395) |
| 302 | vllm | Issue | [Bug]: tests/v1/tpu/test_sampler.py crashes due to ragged_paged_attention arg mismatch | 🔴 高 | closed | 2025-03-21 | [#15257](https://github.com/vllm-project/vllm/issues/15257) |
| 303 | vllm | Issue | [Bug] Mismatch between `get_multimodal_embedding` output and `PlaceholderRange` | 🔴 高 | closed | 2025-03-19 | [#15144](https://github.com/vllm-project/vllm/issues/15144) |
| 304 | vllm | Issue | [Bug]: tensor shape mismatch for `--lora-extra-vocab-size 0` | 🔴 高 | closed | 2025-03-18 | [#15036](https://github.com/vllm-project/vllm/issues/15036) |
| 305 | vllm | PR | [V1][Bugfix][Spec Decode] Fix incorrect outputs in V1 speculative decoding due to batch indexing | 🔴 高 | closed | 2025-03-11 | [#14645](https://github.com/vllm-project/vllm/pull/14645) |
| 306 | vllm | Issue | [Bug]: Docker GPU image is unnecessarily fat due to two (mismatching) copies of CUDA runtime libraries | 🔴 高 | closed | 2025-03-07 | [#14433](https://github.com/vllm-project/vllm/issues/14433) |
| 307 | vllm | Issue | [Bug]: size mismatch when loading MixtralForCausalLM GGUF model | 🔴 高 | closed | 2025-03-07 | [#14423](https://github.com/vllm-project/vllm/issues/14423) |
| 308 | vllm | PR | [Bugfix] Fix DeepSeek MTP crash when using TP1ModelRunner with CUDA graph due to shape mismatch | 🔴 高 | closed | 2025-03-04 | [#14237](https://github.com/vllm-project/vllm/pull/14237) |
| 309 | vllm-ascend | Issue | [Bug]: Qwen2.5-7B-Instruct 模型0.7.1和0.7.3版本vllm-ascend输出不相同，怀疑0.7.3有精度问题 | 🔴 高 | closed | 2025-03-03 | [#221](https://github.com/vllm-project/vllm-ascend/issues/221) |
| 310 | vllm | Issue | [Bug]: Generation mismatch with Model: meta-llama/Llama-3.2-11B-Vision-Instruct | 🔴 高 | closed | 2025-02-24 | [#13763](https://github.com/vllm-project/vllm/issues/13763) |
| 311 | vllm | Issue | [Usage]: Shape mismatch when batch requests with openai chat completion apis and qwen2-vl | 🔴 高 | closed | 2025-01-26 | [#12442](https://github.com/vllm-project/vllm/issues/12442) |
| 312 | vllm | PR | bugfix: Fix signature mismatch in benchmark's `get_tokenizer` function | 🔴 高 | closed | 2025-01-13 | [#11982](https://github.com/vllm-project/vllm/pull/11982) |
| 313 | vllm | Issue | [Usage]: Getting shape mismatch for multimodal input with task="embed" | 🔴 高 | closed | 2025-01-03 | [#11709](https://github.com/vllm-project/vllm/issues/11709) |
| 314 | vllm | Issue | [Bug]: Mismatch multi-modal placeholder of LLava-1.6-Mistral-7B | 🔴 高 | closed | 2025-01-03 | [#11704](https://github.com/vllm-project/vllm/issues/11704) |
| 315 | vllm | Issue | [Bug]: 0.6.6.post1 Qwen/QVQ-72B-Preview crash: shape mismatch | 🔴 高 | closed | 2025-01-02 | [#11678](https://github.com/vllm-project/vllm/issues/11678) |
| 316 | vllm | Issue | [Bug]:  Dimension mismatch error will occur during batch inference when processing image embeddings with minicpmv | 🔴 高 | closed | 2024-12-30 | [#11630](https://github.com/vllm-project/vllm/issues/11630) |
| 317 | vllm | Issue | [Bug]: v0.6.5 breaks AI SDK's `generateObject` with nullable strings in schema (`"type mismatch! call is&lt;type&gt;() before get&lt;type&gt;()" && is&lt;std::string&gt;()`) | 🔴 高 | closed | 2024-12-22 | [#11415](https://github.com/vllm-project/vllm/issues/11415) |
| 318 | vllm | Issue | [Bug]: Mismatch of tqdm when n &gt; 1 | 🔴 高 | closed | 2024-12-06 | [#10949](https://github.com/vllm-project/vllm/issues/10949) |
| 319 | vllm | Issue | [Bug]: v0.6.4.post1 Qwen2-VL-7B-Instruct-AWQ crash：shape mismatch | 🔴 高 | closed | 2024-11-27 | [#10686](https://github.com/vllm-project/vllm/issues/10686) |
| 320 | vllm | Issue | [Bug]: Tokenization Mismatch Between HuggingFace and vLLM | 🔴 高 | closed | 2024-09-27 | [#8904](https://github.com/vllm-project/vllm/issues/8904) |
| 321 | vllm | Issue | [Bug]: OLMoE produces incorrect output with TP&gt;1 | 🔴 高 | closed | 2024-09-23 | [#8747](https://github.com/vllm-project/vllm/issues/8747) |
| 322 | vllm | Issue | [Bug]: mismatch between multimodal tokens and placeholders for Llava-Next (4 GPUs) | 🔴 高 | closed | 2024-09-12 | [#8421](https://github.com/vllm-project/vllm/issues/8421) |
| 323 | vllm | Issue | [Bug]: RuntimeError: shape mismatch: value tensor of shape [3328, 7168] cannot be broadcast to indexing result of shape [3328] for OpenGVLab/InternVL2-40B | 🔴 高 | closed | 2024-09-08 | [#8275](https://github.com/vllm-project/vllm/issues/8275) |
| 324 | vllm | Issue | [Bug]: Mismatch in TTFT count and number of successful requests completed  | 🔴 高 | closed | 2024-09-03 | [#8115](https://github.com/vllm-project/vllm/issues/8115) |
| 325 | vllm | Issue | [Bug]: Mismatch in the number of image tokens and placeholders during batch inference | 🔴 高 | closed | 2024-08-20 | [#7669](https://github.com/vllm-project/vllm/issues/7669) |
| 326 | vllm | Issue | [Bug]: InternVL2 Mismatch in number of image tokens and image embedding size | 🔴 高 | closed | 2024-08-05 | [#7160](https://github.com/vllm-project/vllm/issues/7160) |
| 327 | vllm | PR | [Bugfix] Fix dtype mismatch in PaliGemma | 🔴 高 | closed | 2024-07-12 | [#6367](https://github.com/vllm-project/vllm/pull/6367) |
| 328 | vllm | Issue | [Bug]: Server fails to boot due to a tensor size mismatch when LoRA is enabled for GPTBigCode | 🔴 高 | closed | 2024-07-10 | [#6314](https://github.com/vllm-project/vllm/issues/6314) |
| 329 | vllm | Issue | [Bug]: Command-R incorrect output contains `&lt;EOS_TOKEN&gt;` and seems to do text prediction rather than conversation | 🔴 高 | closed | 2024-05-24 | [#5030](https://github.com/vllm-project/vllm/issues/5030) |
| 330 | vllm | Issue | [Bug]: speculative decoding got `shape mismatch` error with n&gt;1 and random sample | 🔴 高 | closed | 2024-05-21 | [#4934](https://github.com/vllm-project/vllm/issues/4934) |
| 331 | vllm | Issue | [Bug]: Issue Running LLaVA with vLLM Due to Tensor Size Mismatch | 🔴 高 | closed | 2024-04-28 | [#4421](https://github.com/vllm-project/vllm/issues/4421) |
| 332 | vllm | PR | [Bugfix] Fix incorrect output on OLMo models in Tensor Parallelism | 🔴 高 | closed | 2024-04-05 | [#3869](https://github.com/vllm-project/vllm/pull/3869) |
| 333 | vllm | Issue | [Bug]: Incorrect output on OLMo models with `tensor_parallel_size`&gt;1 | 🔴 高 | closed | 2024-04-01 | [#3775](https://github.com/vllm-project/vllm/issues/3775) |
| 334 | vllm | Issue | The accuracy of the inference results of qwen14B accelerated by VLLM has decreased | 🔴 高 | closed | 2024-02-21 | [#2956](https://github.com/vllm-project/vllm/issues/2956) |
| 335 | vllm | Issue | The inference results based on vllm qwen7B also lead to a decrease in accuracy | 🔴 高 | closed | 2024-02-21 | [#2953](https://github.com/vllm-project/vllm/issues/2953) |
| 336 | vllm | Issue | The flow cytometry results of qwen14B show a significant decrease in accuracy | 🔴 高 | closed | 2024-02-21 | [#2951](https://github.com/vllm-project/vllm/issues/2951) |
| 337 | vllm | Issue | Inference based on vllm qwen14B is inconsistent with the original qwen results, and the accuracy will drop significantly | 🔴 高 | closed | 2024-02-21 | [#2950](https://github.com/vllm-project/vllm/issues/2950) |
| 338 | vllm | Issue | The service results based on vllm qwen7B are inconsistent with the original qwen results, and the accuracy will drop significantly | 🔴 高 | closed | 2024-02-21 | [#2949](https://github.com/vllm-project/vllm/issues/2949) |
| 339 | vllm | Issue | llama logits mismatch between TP=1 and TP=2 | 🔴 高 | closed | 2024-01-31 | [#2679](https://github.com/vllm-project/vllm/issues/2679) |
| 340 | vllm | PR | [FIX] Fix shape mismatch for swapped sequences when logprobs &gt; 0 | 🔴 高 | closed | 2023-12-07 | [#1971](https://github.com/vllm-project/vllm/pull/1971) |
| 341 | vllm | Issue | [Error] "IndexError: shape mismatch: indexing tensors could not be broadcast together with shapes"  when logprobs &gt; 0 | 🔴 高 | closed | 2023-11-30 | [#1847](https://github.com/vllm-project/vllm/issues/1847) |
| 342 | vllm | Issue | Docker build fails due to CUDA version mismatch | 🔴 高 | closed | 2023-11-09 | [#1597](https://github.com/vllm-project/vllm/issues/1597) |
| 343 | vllm | Issue |  The detected CUDA version (11.8) mismatches the version that was used to compile       PyTorch (12.1). Please make sure to use the same CUDA versions. | 🔴 高 | closed | 2023-11-02 | [#1548](https://github.com/vllm-project/vllm/issues/1548) |
| 344 | vllm | Issue | The detected CUDA version (11.8) mismatches the version that was used to compile       PyTorch (12.1). Please make sure to use the same CUDA versions | 🔴 高 | closed | 2023-10-24 | [#1453](https://github.com/vllm-project/vllm/issues/1453) |
| 345 | vllm | Issue | Building from source by running `pip install -e .` CUDA version mismatches | 🔴 高 | closed | 2023-09-15 | [#1060](https://github.com/vllm-project/vllm/issues/1060) |
| 346 | vllm | Issue | pip install error - CUDA version mismatch | 🔴 高 | closed | 2023-08-15 | [#763](https://github.com/vllm-project/vllm/issues/763) |
| 347 | vllm | Issue | pip install fails with CUDA version (12.0) mismatch compile  PyTorch (11.7). though I am using torch (2.1.0.dev20230726+cu121) | 🔴 高 | closed | 2023-07-27 | [#602](https://github.com/vllm-project/vllm/issues/602) |
| 348 | vllm | Issue | Incorrect output when using hf-internal-testing/llama-tokenizer | 🔴 高 | closed | 2023-07-25 | [#577](https://github.com/vllm-project/vllm/issues/577) |
| 349 | vllm | Issue | Build failure due to CUDA version mismatch | 🔴 高 | closed | 2023-05-26 | [#129](https://github.com/vllm-project/vllm/issues/129) |
| 350 | vllm-ascend | Issue | [Bug]: Qwen3 Moe precision issue, When SP is enabled, sending a request longer than 1000 tokens results in garbled replies. | 🟡 中 | closed | 2025-10-11 | [#3374](https://github.com/vllm-project/vllm-ascend/issues/3374) |
| 351 | vllm | Issue | [Bug]: since 0.10.1, Pooling output type changed from float32 to bfloat16 (and different numeric results) | 🟡 中 | closed | 2025-08-21 | [#23373](https://github.com/vllm-project/vllm/issues/23373) |
| 352 | vllm | PR | Improve the output precision of embedding models | 🟡 中 | closed | 2025-06-03 | [#19092](https://github.com/vllm-project/vllm/pull/19092) |
| 353 | vllm-ascend | PR | [Test][FLA] Add accuracy tests for layer norm and chunk output kernels | 🟢 低 | open | 2026-08-07 | [#13784](https://github.com/vllm-project/vllm-ascend/pull/13784) |
| 354 | vllm | PR | [Feature] Add --rank-tp-ratio for uneven tensor parallelism on mismatched GPUs | 🟢 低 | closed | 2026-08-02 | [#50735](https://github.com/vllm-project/vllm/pull/50735) |
| 355 | vllm-ascend | PR | [Doc]update qwen_3.5_397b/qwen3_vl_235b/qwen3_vl_30b doc, fill ascend 950DT info;update qwen_3.6_35b doc, change feature guide link and fill accuracy result | 🟢 低 | closed | 2026-07-23 | [#12738](https://github.com/vllm-project/vllm-ascend/pull/12738) |
| 356 | vllm-ascend | PR | [Doc]update qwen_3.5_397b/qwen3_vl_235b/qwen3_vl_30b doc, fill ascend 950DT info;update qwen_3.6_35b doc, change feature guide link and fill accuracy result | 🟢 低 | closed | 2026-07-21 | [#12492](https://github.com/vllm-project/vllm-ascend/pull/12492) |
| 357 | vllm-ascend | PR | [Doc]update qwen_3.5_397b/qwen3_vl_235b/qwen3_vl_30b doc, fill ascend 950DT info;update qwen_3.6_35b doc, change feature guide link and fill accuracy result | 🟢 低 | closed | 2026-07-20 | [#12435](https://github.com/vllm-project/vllm-ascend/pull/12435) |
| 358 | vllm-ascend | PR | [Feature][KV Pool] Support TP-mismatch PD disaggregated KV pooling for GQA | 🟢 低 | closed | 2026-07-07 | [#11582](https://github.com/vllm-project/vllm-ascend/pull/11582) |
| 359 | vllm | PR | [Bugfix] Fix shape mismatch crash and add logprob_token_ids support in RejectionSampler | 🟢 低 | open | 2026-06-06 | [#44727](https://github.com/vllm-project/vllm/pull/44727) |
| 360 | vllm-ascend | PR | [BugFix][SpecDecode] Fix MLA shape mismatch with Eagle3 and add   DeepSeek V2 Eagle3 support | 🟢 低 | closed | 2026-05-29 | [#9703](https://github.com/vllm-project/vllm-ascend/pull/9703) |
| 361 | vllm | PR | llama4_vision_rope: add HIP override to accept (q, k) and avoid (positions, q, k) mismatch | 🟢 低 | closed | 2025-10-14 | [#26790](https://github.com/vllm-project/vllm/pull/26790) |

</details>

---

## 4. 算子与注意力数值精度

Attention（FlashAttention/PagedAttention/MLA）、Softmax、RoPE、RMSNorm、LayerNorm 等核心算子的数值实现差异导致的精度问题。不同 backend（FlashAttention / FlashInfer / Triton / CANN）对同一算子的数值行为可能不同。

**问题数: 36 条**

### 关键问题

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | PR | [Kernel][Bugfix] TRITON_MLA fix silent accuracy drop with dynamic num_kv_splits for SWA | 🔴 极高 | closed | 2026-06-30 | [#47188](https://github.com/vllm-project/vllm/pull/47188) |
| 2 | vllm-ascend | PR | [BugFix][Ops][310p]:fix the accuracy issue caused by MoEGatingTopkSoftmax | 🔴 高 | closed | 2026-07-03 | [#11391](https://github.com/vllm-project/vllm-ascend/pull/11391) |
| 3 | vllm-ascend | PR | [Ascend950] [BugFix] Fix split_qkv_rmsnorm_rope Triton kernel accuracy issue on A5 | 🔴 高 | closed | 2026-06-01 | [#9849](https://github.com/vllm-project/vllm-ascend/pull/9849) |
| 4 | vllm-ascend | Issue | [Bug]: MoE类新模型适配（Mimo-V2-Flash），Attention部分DP4 TP4，MOE部分开启专家并行（--enable-expert-parallel）后精度异常（吐字是乱码），不开启专家并行可以正常吐字 | 🔴 高 | open | 2026-04-14 | [#8223](https://github.com/vllm-project/vllm-ascend/issues/8223) |
| 5 | vllm-ascend | PR | [v0.13.0][bugfix][accuracy] Fix ds indexer accuracy problem caused by k rope | 🔴 高 | open | 2026-03-31 | [#7859](https://github.com/vllm-project/vllm-ascend/pull/7859) |
| 6 | vllm | PR | [Bugfix][Backport] Backport PR #31816 to v0.13.0: Fix ROCM_AITER_TRITON_MLA accuracy for DeepSeek-V3 | 🔴 高 | closed | 2026-03-25 | [#38145](https://github.com/vllm-project/vllm/pull/38145) |
| 7 | vllm-ascend | PR | [bugfix][accuracy] Fix ds indexer accuracy problem caused by k rope | 🔴 高 | closed | 2026-03-16 | [#7341](https://github.com/vllm-project/vllm-ascend/pull/7341) |
| 8 | vllm | PR | [Bugfix] Fix FlashMLA sparse accuracy with topk_length and zero-init padding | 🔴 高 | closed | 2026-03-10 | [#36616](https://github.com/vllm-project/vllm/pull/36616) |
| 9 | vllm | Issue | [Bug]: Accuracy Issue with FlashMLA Sparse on DeepSeek V3.2 | 🔴 高 | closed | 2026-03-09 | [#36524](https://github.com/vllm-project/vllm/issues/36524) |
| 10 | vllm | Issue | [Bug]: qwen3-coder-next inference randomly hangs, accuracy degradation in 0.16.0+ with TP &gt; 1 and  fuse_allreduce_rms=False (H100s on PCIe) | 🔴 高 | closed | 2026-02-27 | [#35504](https://github.com/vllm-project/vllm/issues/35504) |
| 11 | vllm-ascend | PR | [BugFix] [310p] Fix attention accuracy issue | 🔴 高 | closed | 2026-02-25 | [#6803](https://github.com/vllm-project/vllm-ascend/pull/6803) |
| 12 | vllm-ascend | Issue | [Bug]: Potential accuracy && accept rate degradation if rope parameters in eagle3 draft model is different from main model. | 🔴 高 | closed | 2026-02-07 | [#6612](https://github.com/vllm-project/vllm-ascend/issues/6612) |
| 13 | vllm | PR | [ROCm][CI] Fix HuggingFace flash_attention_2 accuracy issue in Isaac vision encoder | 🔴 高 | closed | 2026-01-13 | [#32233](https://github.com/vllm-project/vllm/pull/32233) |
| 14 | vllm | PR | [ROCm][AITER] bugfix accuracy regression in ROCM_AITER_TRITON_MLA backend | 🔴 高 | closed | 2026-01-06 | [#31816](https://github.com/vllm-project/vllm/pull/31816) |
| 15 | vllm | Issue | [Feature][ROCm][AITER]: Speculative Decoding Accuracy Issue with VLLM_ATTENTION_BACKEND=ROCM_AITER_FA | 🔴 高 | closed | 2026-01-02 | [#31625](https://github.com/vllm-project/vllm/issues/31625) |
| 16 | vllm-ascend | Issue | [Bug]: accuracy issue for triton kernel split_qkv_rmsnorm_rope_kernel | 🔴 高 | closed | 2025-12-25 | [#5352](https://github.com/vllm-project/vllm-ascend/issues/5352) |
| 17 | vllm-ascend | PR | Fix the accuracy arange change in normal scene is more than 7 | 🔴 高 | closed | 2025-12-12 | [#4964](https://github.com/vllm-project/vllm-ascend/pull/4964) |
| 18 | vllm | PR | [DCP][Bugfix][CI] Fix accuracy issue of DCP when using FLASH_ATTN_MLA | 🔴 高 | closed | 2025-12-09 | [#30309](https://github.com/vllm-project/vllm/pull/30309) |
| 19 | vllm-ascend | PR | [BugFix][DS 3.2] Fix ds indexer accuracy problem caused by rope. | 🔴 高 | closed | 2025-12-02 | [#4641](https://github.com/vllm-project/vllm-ascend/pull/4641) |
| 20 | vllm | PR | [ROCm][BugFix] Fix accuracy issue for `AiterMLABackend` for newest aiter main branch | 🔴 高 | closed | 2025-11-21 | [#29146](https://github.com/vllm-project/vllm/pull/29146) |
| 21 | vllm-ascend | PR | [BugFix] Fix mlapo accuracy problem related with weight processing. | 🔴 高 | closed | 2025-10-29 | [#3857](https://github.com/vllm-project/vllm-ascend/pull/3857) |
| 22 | vllm-ascend | PR | [BugFix] Fix mlapo accuracy problem related with weight processing. | 🔴 高 | closed | 2025-10-29 | [#3850](https://github.com/vllm-project/vllm-ascend/pull/3850) |
| 23 | vllm-ascend | PR | [Fix] Clears unused slot mappings and fix accuracy issue with MLA models when enabling `FULL_DECODE_ONLY` | 🔴 高 | closed | 2025-10-15 | [#3482](https://github.com/vllm-project/vllm-ascend/pull/3482) |
| 24 | vllm | PR | [BugFix] Fix FI accuracy issue when used for MLA prefill | 🔴 高 | closed | 2025-10-02 | [#26063](https://github.com/vllm-project/vllm/pull/26063) |
| 25 | vllm-ascend | PR | Fix the accuracy issues caused by the mrope operator | 🔴 高 | closed | 2025-08-15 | [#2388](https://github.com/vllm-project/vllm-ascend/pull/2388) |
| 26 | vllm-ascend | PR | Fix the accuracy issues caused by the mrope operator | 🔴 高 | closed | 2025-08-13 | [#2355](https://github.com/vllm-project/vllm-ascend/pull/2355) |
| 27 | vllm | Issue | [Bug]: Cascade Attention Accuracy Issue on A100 | 🔴 高 | closed | 2025-08-01 | [#22103](https://github.com/vllm-project/vllm/issues/22103) |
| 28 | vllm | PR | [BugFix] FA2 MLA Accuracy Issue | 🔴 高 | closed | 2025-05-28 | [#18807](https://github.com/vllm-project/vllm/pull/18807) |
| 29 | vllm | PR | [BugFix] MLA + V1, illegal memory access and accuracy issues | 🔴 高 | closed | 2025-03-05 | [#14253](https://github.com/vllm-project/vllm/pull/14253) |
| 30 | vllm-ascend | Issue | [Bug] Qwen3.5-397B w8a8 hybrid deployment returns gdn_attention_core ERROR in high-concurrency precision test | 🟡 中 | open | 2026-03-31 | [#7863](https://github.com/vllm-project/vllm-ascend/issues/7863) |
| 31 | vllm | PR | Fix MiniMax-M2 rmsnorm precision and remove useless code | 🟡 中 | closed | 2025-10-28 | [#27627](https://github.com/vllm-project/vllm/pull/27627) |
| 32 | vllm | PR | RoPE in float32 precision | 🟡 中 | closed | 2023-08-25 | [#870](https://github.com/vllm-project/vllm/pull/870) |
| 33 | vllm | PR | [ROCm][CI] Add MLA decode accuracy and determinism tests | 🟢 低 | closed | 2026-07-30 | [#50480](https://github.com/vllm-project/vllm/pull/50480) |
| 34 | vllm-ascend | PR | [BugFix] Add value.contiguous in attention to avoid some accuracy problems. | 🟢 低 | closed | 2025-02-19 | [#95](https://github.com/vllm-project/vllm-ascend/pull/95) |
| 35 | vllm | PR | [Model] Update MPT model with GLU and rope and add low precision layer norm | 🟢 低 | closed | 2024-10-18 | [#9500](https://github.com/vllm-project/vllm/pull/9500) |
| 36 | vllm | PR | [Model] Update MPT model with GLU and rope and add low precision layer norm  | 🟢 低 | closed | 2024-04-16 | [#4116](https://github.com/vllm-project/vllm/pull/4116) |

### 关键规律与分析

1. 不同 attention backend（FlashAttention / FlashInfer / Triton / CANN）对同一算子的数值实现存在差异，是跨平台精度不一致的根源。
2. **MLA 与 dense-MHA 的 split 融合**是新架构高频出错点（OOB write、fp8_ds_mla context gather 损坏）。
3. RoPE / RMSNorm / LayerNorm / 激活函数的实现差异、融合算子（silu/gelu）数值路径变化会静默影响输出。

---

## 5. KV Cache 精度

与 KV Cache 直接相关的精度问题（低精度 KV cache dtype、prefix cache 正确性、传输数据损坏、布局 reshape 等），已在 [0_kvcache.md](0_kvcache.md) 单独整理，本节仅列概览。

**问题数: 122 条**

> 本节仅展示概览。完整分类、技术与深度分析见 [0_kvcache.md](0_kvcache.md)。

### 关键问题

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug]: Alignment specialization on causal_conv1d metadata pointers forces inference-time JIT compiles; raced shared-cache writes produced silent all-NaN outputs | 🔴 极高 | closed | 2026-08-15 | [#52413](https://github.com/vllm-project/vllm/issues/52413) |
| 2 | vllm | Issue | [Bug]: Kimi-K3 with --kv-cache-dtype fp8 is unusable on H200/Hopper — assertion demands use_prefill_query_quantization, but that flag is silently ignored on non-Blackwell devices | 🔴 极高 | open | 2026-08-06 | [#51313](https://github.com/vllm-project/vllm/issues/51313) |
| 3 | vllm | Issue | [Bug]: OffloadingConnector can silently return wrong output at exact chunk boundaries with mamba_cache_mode=all | 🔴 极高 | closed | 2026-08-05 | [#51094](https://github.com/vllm-project/vllm/issues/51094) |
| 4 | vllm | PR | [Bugfix] MiniMax-M3 fp8_e5m2 KV cache on SM80 (A100/A800): admit the format Triton can read, fix the CUDA-graph KV corruption | 🔴 极高 | open | 2026-08-03 | [#50882](https://github.com/vllm-project/vllm/pull/50882) |
| 5 | vllm | Issue | [Bug]: MiniMax-M3 fp8 KV cache on SM80 (A100/A800): coherent with --enforce-eager, garbage under CUDA graphs — root cause + working fix | 🔴 极高 | open | 2026-08-03 | [#50881](https://github.com/vllm-project/vllm/issues/50881) |
| 6 | vllm | PR | [Core] Zero KV cache when NaN logits detected | 🔴 极高 | closed | 2026-07-27 | [#50002](https://github.com/vllm-project/vllm/pull/50002) |
| 7 | vllm | Issue | [Bug]: int8_per_token_head KV cache corrupts Gemma-4 (hybrid attention) output under load on Triton | 🔴 极高 | closed | 2026-07-24 | [#49716](https://github.com/vllm-project/vllm/issues/49716) |
| 8 | vllm | Issue | [Bug] Deepseek-V4-Pro corruption: H100 multi-rank startup can select a FlashInfer block-FP8 path with a cold JIT cache race | 🔴 极高 | open | 2026-07-20 | [#49165](https://github.com/vllm-project/vllm/issues/49165) |
| 9 | vllm | Issue | [Bug]: MTP speculative decoding with --kv-cache-dtype auto produces cross-sequence garbage under batching (Gemma-4 W4A16; fp8 KV unaffected) | 🔴 极高 | closed | 2026-06-18 | [#46088](https://github.com/vllm-project/vllm/issues/46088) |
| 10 | vllm | PR | KV Cache MLA NaN Write Reporting | 🔴 极高 | open | 2026-05-28 | [#43880](https://github.com/vllm-project/vllm/pull/43880) |
| 11 | vllm | PR | [DO NOT MERGE] Add kernel-side KV cache NaN/Inf detection for MLA concat_and_cache | 🔴 极高 | closed | 2026-05-21 | [#43318](https://github.com/vllm-project/vllm/pull/43318) |
| 12 | vllm | PR | [Bugfix] Fix V1 dummy run writing NaN to KV cache null block | 🔴 极高 | closed | 2026-04-09 | [#39444](https://github.com/vllm-project/vllm/pull/39444) |
| 13 | vllm | Issue | [Bug]: MLA + FP8 KV cache + CUDA Graph causes random NaN in decode phase | 🔴 极高 | closed | 2026-03-31 | [#38634](https://github.com/vllm-project/vllm/issues/38634) |
| 14 | vllm | Issue | [Bug]: vLLM Serve with LMCache enabled produces wrong output for GPT-OSS-20B | 🔴 极高 | closed | 2025-11-25 | [#29436](https://github.com/vllm-project/vllm/issues/29436) |
| 15 | vllm-ascend | Issue | [Bug]: Qwen3-VL with long text + image causes NaN (!!!!) output and KV Cache Pollution on both float16 and bfloat16 | 🔴 极高 | closed | 2025-10-31 | [#3934](https://github.com/vllm-project/vllm-ascend/issues/3934) |
| 16 | vllm | Issue | [Bug]: [V1] wrong output when using kv cache fp8 | 🔴 极高 | closed | 2025-02-12 | [#13133](https://github.com/vllm-project/vllm/issues/13133) |
| 17 | vllm | PR | [BugFix] Fix NaN errors in paged attention kernel | 🔴 极高 | closed | 2023-09-03 | [#936](https://github.com/vllm-project/vllm/pull/936) |
| 18 | vllm-ascend | Issue | [v0.23.0][Bug]:The accuracy of the MTP and prefix cache of the 310P Qwen3.5 series models is abnormal. | 🔴 高 | open | 2026-08-15 | [#14339](https://github.com/vllm-project/vllm-ascend/issues/14339) |
| 19 | vllm-ascend | Issue | [bug]:[v0.26.0rc][glm5.2] DCP block_table overflows k_cache capacity -&gt; LightningIndexerQuant MTE invalid GM address | 🔴 高 | open | 2026-08-15 | [#14320](https://github.com/vllm-project/vllm-ascend/issues/14320) |
| 20 | vllm | PR | [Bugfix][Model] Fix Inkling NVIDIA sconv cache block-size mismatch | 🔴 高 | open | 2026-08-12 | [#51951](https://github.com/vllm-project/vllm/pull/51951) |
| 21 | vllm | Issue | [Bug]: attention backend probe in cuda.py catches only ImportError; non-ImportError side effects (e.g. cache PermissionError, CUDA runtime   mismatch) crash engine init instead of being recorded as unavailable | 🔴 高 | open | 2026-08-10 | [#51658](https://github.com/vllm-project/vllm/issues/51658) |
| 22 | vllm | PR | [Bugfix][KV Offloading] Fix CPU offload block count mismatch across PP ranks | 🔴 高 | open | 2026-08-01 | [#50653](https://github.com/vllm-project/vllm/pull/50653) |
| 23 | vllm-ascend | PR | [Cherry-pick][v0.24.0rc][BugFix] Restore paged attention fallback, stateful one-token decode & fix PD/PCP/DCP accuracy (from #12027, #12228, #12255) | 🔴 高 | closed | 2026-07-30 | [#13195](https://github.com/vllm-project/vllm-ascend/pull/13195) |
| 24 | vllm | Issue | [Bug]: V1 streaming-session rebuild leaves stale prefix-cache block hashes and can produce incorrect output | 🔴 高 | open | 2026-07-22 | [#49449](https://github.com/vllm-project/vllm/issues/49449) |
| 25 | vllm | Issue | [Bug]: CPU offloading fails with block_size=256 + speculative decoding due to eagle group block_size mismatch | 🔴 高 | open | 2026-07-17 | [#48919](https://github.com/vllm-project/vllm/issues/48919) |
| 26 | vllm | Issue | [Bug]: Ministral is broken with --kv-cache-dtype fp8 in 0.25.1 | 🔴 高 | open | 2026-07-17 | [#48945](https://github.com/vllm-project/vllm/issues/48945) |
| 27 | vllm | PR | [Bugfix] Fix offloading set_ overflow for packed non-uniform KV caches | 🔴 高 | closed | 2026-07-13 | [#48530](https://github.com/vllm-project/vllm/pull/48530) |
| 28 | vllm | PR | [Bugfix][MLA] Fix fp8 KV cache crash on MLA models at startup | 🔴 高 | closed | 2026-07-12 | [#48439](https://github.com/vllm-project/vllm/pull/48439) |
| 29 | vllm | Issue | [Bug]: DeepSeek-V3.2 / GLM DSA with --kv-cache-dtype fp8_ds_mla crashes at engine init: KV-cache reshape sizes view by head_size (576) while pages are 656B/token | 🔴 高 | closed | 2026-07-12 | [#48378](https://github.com/vllm-project/vllm/issues/48378) |
| 30 | vllm | Issue | [Bug][ROCm/gfx942] GPU memory access fault (worker crash) when sequences cross 2048 tokens — DeepSeek V4 flash arch, sparse_attn_indexer + fp8 KV cache, MI325X TP=4 | 🔴 高 | open | 2026-07-10 | [#48266](https://github.com/vllm-project/vllm/issues/48266) |
| 31 | vllm-ascend | PR | [v0.23.0][BugFix] Resolve block table overflow | 🔴 高 | closed | 2026-07-09 | [#11659](https://github.com/vllm-project/vllm-ascend/pull/11659) |
| 32 | vllm | PR | [Bug]: REGRESSION : FP8 KV cache  FlashInfer no longer available as attention backend on SM75 (Turing) in v0.24.0 | 🔴 高 | closed | 2026-07-08 | [#47949](https://github.com/vllm-project/vllm/pull/47949) |
| 33 | vllm | PR | fix: [Bug]: DP/EP with fp8 KV Cache Brokens for FlashMLA | 🔴 高 | closed | 2026-07-08 | [#47940](https://github.com/vllm-project/vllm/pull/47940) |
| 34 | vllm | Issue | [Bug]: DP/EP with fp8 KV Cache Brokens for FlashMLA | 🔴 高 | closed | 2026-07-08 | [#47935](https://github.com/vllm-project/vllm/issues/47935) |
| 35 | vllm | Issue | [Bug] Nemotron + FlashInfer + FP8 KV cache crashes on Hopper: assert query.is_contiguous() in maybe_quant_query | 🔴 高 | closed | 2026-07-07 | [#47905](https://github.com/vllm-project/vllm/issues/47905) |
| 36 | vllm | Issue | RuntimeError: shape mismatch during KV cache init with EP + DP on MoE model | 🔴 高 | open | 2026-07-06 | [#47722](https://github.com/vllm-project/vllm/issues/47722) |
| 37 | vllm-ascend | PR | [BugFix] Resolve block table overflow | 🔴 高 | closed | 2026-07-06 | [#11466](https://github.com/vllm-project/vllm-ascend/pull/11466) |
| 38 | vllm | Issue | [Bug]: DeepSeek-V4-Flash-DSpark fails on H200/SM90 with FlashMLA KV cache shape mismatch | 🔴 高 | closed | 2026-07-05 | [#47648](https://github.com/vllm-project/vllm/issues/47648) |
| 39 | vllm | PR | [Bugfix][DeepSeek V4] Avoid SM100 FlashMLA cache shape mismatch | 🔴 高 | open | 2026-07-04 | [#47610](https://github.com/vllm-project/vllm/pull/47610) |
| 40 | vllm | Issue | [Bug]: REGRESSION : FP8 KV cache  FlashInfer no longer available as attention backend on SM75 (Turing) in v0.24.0 | 🔴 高 | open | 2026-07-03 | [#47549](https://github.com/vllm-project/vllm/issues/47549) |

### 关键规律与分析

KV Cache 相关精度问题（低精度 dtype、prefix cache、传输损坏、布局 reshape、offload 等）是精度问题中的重要子集，已在 [0_kvcache.md](0_kvcache.md) 单独深度分析，本表仅列概览，不在此重复展开。

<details>
<summary>展开全部 122 条</summary>

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | Issue | [Bug]: Alignment specialization on causal_conv1d metadata pointers forces inference-time JIT compiles; raced shared-cache writes produced silent all-NaN outputs | 🔴 极高 | closed | 2026-08-15 | [#52413](https://github.com/vllm-project/vllm/issues/52413) |
| 2 | vllm | Issue | [Bug]: Kimi-K3 with --kv-cache-dtype fp8 is unusable on H200/Hopper — assertion demands use_prefill_query_quantization, but that flag is silently ignored on non-Blackwell devices | 🔴 极高 | open | 2026-08-06 | [#51313](https://github.com/vllm-project/vllm/issues/51313) |
| 3 | vllm | Issue | [Bug]: OffloadingConnector can silently return wrong output at exact chunk boundaries with mamba_cache_mode=all | 🔴 极高 | closed | 2026-08-05 | [#51094](https://github.com/vllm-project/vllm/issues/51094) |
| 4 | vllm | PR | [Bugfix] MiniMax-M3 fp8_e5m2 KV cache on SM80 (A100/A800): admit the format Triton can read, fix the CUDA-graph KV corruption | 🔴 极高 | open | 2026-08-03 | [#50882](https://github.com/vllm-project/vllm/pull/50882) |
| 5 | vllm | Issue | [Bug]: MiniMax-M3 fp8 KV cache on SM80 (A100/A800): coherent with --enforce-eager, garbage under CUDA graphs — root cause + working fix | 🔴 极高 | open | 2026-08-03 | [#50881](https://github.com/vllm-project/vllm/issues/50881) |
| 6 | vllm | PR | [Core] Zero KV cache when NaN logits detected | 🔴 极高 | closed | 2026-07-27 | [#50002](https://github.com/vllm-project/vllm/pull/50002) |
| 7 | vllm | Issue | [Bug]: int8_per_token_head KV cache corrupts Gemma-4 (hybrid attention) output under load on Triton | 🔴 极高 | closed | 2026-07-24 | [#49716](https://github.com/vllm-project/vllm/issues/49716) |
| 8 | vllm | Issue | [Bug] Deepseek-V4-Pro corruption: H100 multi-rank startup can select a FlashInfer block-FP8 path with a cold JIT cache race | 🔴 极高 | open | 2026-07-20 | [#49165](https://github.com/vllm-project/vllm/issues/49165) |
| 9 | vllm | Issue | [Bug]: MTP speculative decoding with --kv-cache-dtype auto produces cross-sequence garbage under batching (Gemma-4 W4A16; fp8 KV unaffected) | 🔴 极高 | closed | 2026-06-18 | [#46088](https://github.com/vllm-project/vllm/issues/46088) |
| 10 | vllm | PR | KV Cache MLA NaN Write Reporting | 🔴 极高 | open | 2026-05-28 | [#43880](https://github.com/vllm-project/vllm/pull/43880) |
| 11 | vllm | PR | [DO NOT MERGE] Add kernel-side KV cache NaN/Inf detection for MLA concat_and_cache | 🔴 极高 | closed | 2026-05-21 | [#43318](https://github.com/vllm-project/vllm/pull/43318) |
| 12 | vllm | PR | [Bugfix] Fix V1 dummy run writing NaN to KV cache null block | 🔴 极高 | closed | 2026-04-09 | [#39444](https://github.com/vllm-project/vllm/pull/39444) |
| 13 | vllm | Issue | [Bug]: MLA + FP8 KV cache + CUDA Graph causes random NaN in decode phase | 🔴 极高 | closed | 2026-03-31 | [#38634](https://github.com/vllm-project/vllm/issues/38634) |
| 14 | vllm | Issue | [Bug]: vLLM Serve with LMCache enabled produces wrong output for GPT-OSS-20B | 🔴 极高 | closed | 2025-11-25 | [#29436](https://github.com/vllm-project/vllm/issues/29436) |
| 15 | vllm-ascend | Issue | [Bug]: Qwen3-VL with long text + image causes NaN (!!!!) output and KV Cache Pollution on both float16 and bfloat16 | 🔴 极高 | closed | 2025-10-31 | [#3934](https://github.com/vllm-project/vllm-ascend/issues/3934) |
| 16 | vllm | Issue | [Bug]: [V1] wrong output when using kv cache fp8 | 🔴 极高 | closed | 2025-02-12 | [#13133](https://github.com/vllm-project/vllm/issues/13133) |
| 17 | vllm | PR | [BugFix] Fix NaN errors in paged attention kernel | 🔴 极高 | closed | 2023-09-03 | [#936](https://github.com/vllm-project/vllm/pull/936) |
| 18 | vllm-ascend | Issue | [v0.23.0][Bug]:The accuracy of the MTP and prefix cache of the 310P Qwen3.5 series models is abnormal. | 🔴 高 | open | 2026-08-15 | [#14339](https://github.com/vllm-project/vllm-ascend/issues/14339) |
| 19 | vllm-ascend | Issue | [bug]:[v0.26.0rc][glm5.2] DCP block_table overflows k_cache capacity -&gt; LightningIndexerQuant MTE invalid GM address | 🔴 高 | open | 2026-08-15 | [#14320](https://github.com/vllm-project/vllm-ascend/issues/14320) |
| 20 | vllm | PR | [Bugfix][Model] Fix Inkling NVIDIA sconv cache block-size mismatch | 🔴 高 | open | 2026-08-12 | [#51951](https://github.com/vllm-project/vllm/pull/51951) |
| 21 | vllm | Issue | [Bug]: attention backend probe in cuda.py catches only ImportError; non-ImportError side effects (e.g. cache PermissionError, CUDA runtime   mismatch) crash engine init instead of being recorded as unavailable | 🔴 高 | open | 2026-08-10 | [#51658](https://github.com/vllm-project/vllm/issues/51658) |
| 22 | vllm | PR | [Bugfix][KV Offloading] Fix CPU offload block count mismatch across PP ranks | 🔴 高 | open | 2026-08-01 | [#50653](https://github.com/vllm-project/vllm/pull/50653) |
| 23 | vllm-ascend | PR | [Cherry-pick][v0.24.0rc][BugFix] Restore paged attention fallback, stateful one-token decode & fix PD/PCP/DCP accuracy (from #12027, #12228, #12255) | 🔴 高 | closed | 2026-07-30 | [#13195](https://github.com/vllm-project/vllm-ascend/pull/13195) |
| 24 | vllm | Issue | [Bug]: V1 streaming-session rebuild leaves stale prefix-cache block hashes and can produce incorrect output | 🔴 高 | open | 2026-07-22 | [#49449](https://github.com/vllm-project/vllm/issues/49449) |
| 25 | vllm | Issue | [Bug]: CPU offloading fails with block_size=256 + speculative decoding due to eagle group block_size mismatch | 🔴 高 | open | 2026-07-17 | [#48919](https://github.com/vllm-project/vllm/issues/48919) |
| 26 | vllm | Issue | [Bug]: Ministral is broken with --kv-cache-dtype fp8 in 0.25.1 | 🔴 高 | open | 2026-07-17 | [#48945](https://github.com/vllm-project/vllm/issues/48945) |
| 27 | vllm | PR | [Bugfix] Fix offloading set_ overflow for packed non-uniform KV caches | 🔴 高 | closed | 2026-07-13 | [#48530](https://github.com/vllm-project/vllm/pull/48530) |
| 28 | vllm | PR | [Bugfix][MLA] Fix fp8 KV cache crash on MLA models at startup | 🔴 高 | closed | 2026-07-12 | [#48439](https://github.com/vllm-project/vllm/pull/48439) |
| 29 | vllm | Issue | [Bug]: DeepSeek-V3.2 / GLM DSA with --kv-cache-dtype fp8_ds_mla crashes at engine init: KV-cache reshape sizes view by head_size (576) while pages are 656B/token | 🔴 高 | closed | 2026-07-12 | [#48378](https://github.com/vllm-project/vllm/issues/48378) |
| 30 | vllm | Issue | [Bug][ROCm/gfx942] GPU memory access fault (worker crash) when sequences cross 2048 tokens — DeepSeek V4 flash arch, sparse_attn_indexer + fp8 KV cache, MI325X TP=4 | 🔴 高 | open | 2026-07-10 | [#48266](https://github.com/vllm-project/vllm/issues/48266) |
| 31 | vllm-ascend | PR | [v0.23.0][BugFix] Resolve block table overflow | 🔴 高 | closed | 2026-07-09 | [#11659](https://github.com/vllm-project/vllm-ascend/pull/11659) |
| 32 | vllm | PR | [Bug]: REGRESSION : FP8 KV cache  FlashInfer no longer available as attention backend on SM75 (Turing) in v0.24.0 | 🔴 高 | closed | 2026-07-08 | [#47949](https://github.com/vllm-project/vllm/pull/47949) |
| 33 | vllm | PR | fix: [Bug]: DP/EP with fp8 KV Cache Brokens for FlashMLA | 🔴 高 | closed | 2026-07-08 | [#47940](https://github.com/vllm-project/vllm/pull/47940) |
| 34 | vllm | Issue | [Bug]: DP/EP with fp8 KV Cache Brokens for FlashMLA | 🔴 高 | closed | 2026-07-08 | [#47935](https://github.com/vllm-project/vllm/issues/47935) |
| 35 | vllm | Issue | [Bug] Nemotron + FlashInfer + FP8 KV cache crashes on Hopper: assert query.is_contiguous() in maybe_quant_query | 🔴 高 | closed | 2026-07-07 | [#47905](https://github.com/vllm-project/vllm/issues/47905) |
| 36 | vllm | Issue | RuntimeError: shape mismatch during KV cache init with EP + DP on MoE model | 🔴 高 | open | 2026-07-06 | [#47722](https://github.com/vllm-project/vllm/issues/47722) |
| 37 | vllm-ascend | PR | [BugFix] Resolve block table overflow | 🔴 高 | closed | 2026-07-06 | [#11466](https://github.com/vllm-project/vllm-ascend/pull/11466) |
| 38 | vllm | Issue | [Bug]: DeepSeek-V4-Flash-DSpark fails on H200/SM90 with FlashMLA KV cache shape mismatch | 🔴 高 | closed | 2026-07-05 | [#47648](https://github.com/vllm-project/vllm/issues/47648) |
| 39 | vllm | PR | [Bugfix][DeepSeek V4] Avoid SM100 FlashMLA cache shape mismatch | 🔴 高 | open | 2026-07-04 | [#47610](https://github.com/vllm-project/vllm/pull/47610) |
| 40 | vllm | Issue | [Bug]: REGRESSION : FP8 KV cache  FlashInfer no longer available as attention backend on SM75 (Turing) in v0.24.0 | 🔴 高 | open | 2026-07-03 | [#47549](https://github.com/vllm-project/vllm/issues/47549) |
| 41 | vllm | Issue | [Bug]: Changing VLLM_CPU_KVCACHE_SPACE drops Qwen 3.5 accuracy on AMD EPYC CPU | 🔴 高 | closed | 2026-06-22 | [#46347](https://github.com/vllm-project/vllm/issues/46347) |
| 42 | vllm-ascend | PR | [CI] to solve cache_csrc mismatch in image building | 🔴 高 | closed | 2026-06-18 | [#10692](https://github.com/vllm-project/vllm-ascend/pull/10692) |
| 43 | vllm | Issue | [Bug]: ROCm MI300X FP8 KV cache MiniMax-M3-MXFP8 accuracy issues | 🔴 高 | closed | 2026-06-14 | [#45562](https://github.com/vllm-project/vllm/issues/45562) |
| 44 | vllm-ascend | PR | [BugFix] fix qwen3.5 accuracy bug when sett cp-kv-cache-interleave-si… | 🔴 高 | closed | 2026-06-13 | [#10442](https://github.com/vllm-project/vllm-ascend/pull/10442) |
| 45 | vllm | Issue | [CI Failure]:  mi300_1: DeepSeek V2-Lite Prefetch Offload Accuracy (H100-MI300) | 🔴 高 | closed | 2026-05-03 | [#41579](https://github.com/vllm-project/vllm/issues/41579) |
| 46 | vllm | Issue | [CI Failure]: mi300_1: DeepSeek V2-Lite Prefetch Offload Accuracy (H100-MI300) | 🔴 高 | closed | 2026-04-21 | [#40485](https://github.com/vllm-project/vllm/issues/40485) |
| 47 | vllm | PR | [Bugfix][LMCache MP Connector] Fix fallback adapter cache_salt signature mismatch | 🔴 高 | open | 2026-04-16 | [#40041](https://github.com/vllm-project/vllm/pull/40041) |
| 48 | vllm | Issue | [Bug]: Turboquant attention crashes on A100 when serving BF16 models with FP8 KV cache | 🔴 高 | closed | 2026-04-16 | [#39992](https://github.com/vllm-project/vllm/issues/39992) |
| 49 | vllm-ascend | Issue | [Bug]: kv-cache-dtype set fp8 and fp8_e5m2, it crash | 🔴 高 | open | 2026-04-09 | [#8102](https://github.com/vllm-project/vllm-ascend/issues/8102) |
| 50 | vllm | Issue | [Bug]: FP8 kv cache on b200 with qwen3.5 has degraded accuracy | 🔴 高 | closed | 2026-03-20 | [#37618](https://github.com/vllm-project/vllm/issues/37618) |
| 51 | vllm | PR | [ROCm][Bugfix] fix cache block size mismatch for aiter unified attention | 🔴 高 | closed | 2026-03-19 | [#37606](https://github.com/vllm-project/vllm/pull/37606) |
| 52 | vllm | PR | [Bugfix] Fix NIXL MLA notification request ID mismatch causing prefill KV cache leak | 🔴 高 | closed | 2026-03-13 | [#36958](https://github.com/vllm-project/vllm/pull/36958) |
| 53 | vllm | Issue | [Bug]: accuracy issue when using multiconnector(Nixl+cpu offloading) | 🔴 高 | closed | 2026-02-13 | [#34526](https://github.com/vllm-project/vllm/issues/34526) |
| 54 | vllm | PR | [ROCm][AITER] KV cache split causes large accuracy regression | 🔴 高 | closed | 2026-01-31 | [#33463](https://github.com/vllm-project/vllm/pull/33463) |
| 55 | vllm-ascend | Issue | [Bug]: Qwen3-VL-235B accuracy degradation in PD-separated scenario when KVcache is full | 🔴 高 | closed | 2025-12-27 | [#5438](https://github.com/vllm-project/vllm-ascend/issues/5438) |
| 56 | vllm | PR | [CI/Build][AMD] Use float16 in test_reset_prefix_cache_e2e to avoid accuracy issues | 🔴 高 | closed | 2025-12-03 | [#29997](https://github.com/vllm-project/vllm/pull/29997) |
| 57 | vllm-ascend | PR | [Bugfix][P/D] TP size larger than KV cache head causes accuracy issues | 🔴 高 | closed | 2025-10-10 | [#3366](https://github.com/vllm-project/vllm-ascend/pull/3366) |
| 58 | vllm | PR | Improve model accuracy by using F32 P*V with v_cache dot product | 🔴 高 | closed | 2025-09-26 | [#25740](https://github.com/vllm-project/vllm/pull/25740) |
| 59 | vllm-ascend | PR | [bugfix][torchair] fix kv_nz accuracy problem and remove redundant reshape_and_cache operation | 🔴 高 | closed | 2025-09-20 | [#3066](https://github.com/vllm-project/vllm-ascend/pull/3066) |
| 60 | vllm | PR | [Bug] [Spec Dec]: Fix kv_cache dtype mismatch for Eagle3 drafter on FP8 target | 🔴 高 | closed | 2025-09-09 | [#24505](https://github.com/vllm-project/vllm/pull/24505) |
| 61 | vllm | PR | Fix kvcache mismatch issue in vllm v0 kv_connector | 🔴 高 | closed | 2025-07-29 | [#21817](https://github.com/vllm-project/vllm/pull/21817) |
| 62 | vllm | PR | [Bugfix] Fix the FP8 kv cache accuracy issue in flashinfer TRT-LLM backend | 🔴 高 | closed | 2025-07-14 | [#20920](https://github.com/vllm-project/vllm/pull/20920) |
| 63 | vllm-ascend | Issue | [Bug]: Accuracy low when set Automatic Prefix Cache Only and Ascend Scheduler  and torchair | 🔴 高 | closed | 2025-07-01 | [#1553](https://github.com/vllm-project/vllm-ascend/issues/1553) |
| 64 | vllm-ascend | PR | [BugFix] Address PrefillCacheHit state to fix prefix cache accuracy bug | 🔴 高 | closed | 2025-06-28 | [#1498](https://github.com/vllm-project/vllm-ascend/pull/1498) |
| 65 | vllm-ascend | PR | [V0.9.1][BugFix] Address PrefillCacheHit state to fix prefix cache accuracy bug | 🔴 高 | closed | 2025-06-28 | [#1492](https://github.com/vllm-project/vllm-ascend/pull/1492) |
| 66 | vllm | Issue | [Bug]: Accuracy degradation in vLLM when prefix‑cache is enabled for recomputation workloads | 🔴 高 | closed | 2025-05-13 | [#18055](https://github.com/vllm-project/vllm/issues/18055) |
| 67 | vllm | PR | Fix integer overflows in attention & cache ops | 🔴 高 | closed | 2023-10-31 | [#1514](https://github.com/vllm-project/vllm/pull/1514) |
| 68 | vllm-ascend | PR | [BugFix][Quantization] Make MLA cache scale mapping idempotent | 🟡 中 | open | 2026-08-18 | [#14470](https://github.com/vllm-project/vllm-ascend/pull/14470) |
| 69 | vllm-ascend | PR | [BugFix][310P] Fix MTP overlay prefixcache precision on 310P | 🟡 中 | open | 2026-08-15 | [#14342](https://github.com/vllm-project/vllm-ascend/pull/14342) |
| 70 | vllm-ascend | PR | [BugFix][310P][v0.23.0] Fix MTP overlay prefixcache precision on 310P | 🟡 中 | open | 2026-08-15 | [#14336](https://github.com/vllm-project/vllm-ascend/pull/14336) |
| 71 | vllm | Issue | [RFC]: fp8_ds_mla KV cache on pre-SM89 (Ampere) GPUs via software-dequant TritonMLA decode | 🟡 中 | open | 2026-08-13 | [#52202](https://github.com/vllm-project/vllm/issues/52202) |
| 72 | vllm | PR | [Bugfix] Reject FLASH_ATTN for fp8 KV cache when local attention forces FA2 fallback | 🟡 中 | open | 2026-08-11 | [#51849](https://github.com/vllm-project/vllm/pull/51849) |
| 73 | vllm | PR | [Bugfix][Kernel] Take a native fp8 KV cache in TRITON_MLA, and fold the non-causal decode | 🟡 中 | open | 2026-08-10 | [#51685](https://github.com/vllm-project/vllm/pull/51685) |
| 74 | vllm | Issue | [Bug]: cache_config_info reports block_size=4 despite --block-size 256, and num_gpu_blocks × block_size ≠ kv_cache_size_tokens (DeepSeek-V4, fp8_ds_mla) | 🟡 中 | open | 2026-08-05 | [#51163](https://github.com/vllm-project/vllm/issues/51163) |
| 75 | vllm | Issue | [Bug]: Kimi-K3 FP8 KV Cache configuration error | 🟡 中 | closed | 2026-07-31 | [#50586](https://github.com/vllm-project/vllm/issues/50586) |
| 76 | vllm | Issue | [Bug]: fp8/bfloat16 KV cache does a full NVML init+shutdown per attention layer per step | 🟡 中 | closed | 2026-07-30 | [#50381](https://github.com/vllm-project/vllm/issues/50381) |
| 77 | vllm | Issue | [Bug]: FlashInfer BatchPrefillWithPagedKVCache fails with "invalid resource handle" on SM121 (GB10) with head_dim 256 + FP8 KV cache | 🟡 中 | open | 2026-07-29 | [#50331](https://github.com/vllm-project/vllm/issues/50331) |
| 78 | vllm | PR | [Bugfix] Fix TurboQuant cache dtype propagation and FP8 store on Ampere | 🟡 中 | open | 2026-07-29 | [#50248](https://github.com/vllm-project/vllm/pull/50248) |
| 79 | vllm | PR | [Bugfix][MLA] Fix fp8 KV cache prefill query quantization selection for Kimi-K3 | 🟡 中 | open | 2026-07-28 | [#50181](https://github.com/vllm-project/vllm/pull/50181) |
| 80 | vllm | Issue | [Bug]: kimi-k3 --kvcache-dtype-fp8 error | 🟡 中 | closed | 2026-07-27 | [#50056](https://github.com/vllm-project/vllm/issues/50056) |
| 81 | vllm-ascend | PR | [BugFix][NetLoader] Fix processed-layout P2P for INT8_CACHE=no | 🟡 中 | closed | 2026-07-26 | [#12885](https://github.com/vllm-project/vllm-ascend/pull/12885) |
| 82 | vllm | PR | [Bugfix] Detect mixed precision in packed KV cache specs | 🟡 中 | closed | 2026-07-23 | [#49623](https://github.com/vllm-project/vllm/pull/49623) |
| 83 | vllm | PR | [Bugfix] Fix SM100 fp8_ds_mla cache scales | 🟡 中 | open | 2026-07-22 | [#49435](https://github.com/vllm-project/vllm/pull/49435) |
| 84 | vllm | PR | [CI][Bugfix] Fix ROCm FP8 KV cache dtype in attention backend test | 🟡 中 | closed | 2026-07-21 | [#49380](https://github.com/vllm-project/vllm/pull/49380) |
| 85 | vllm | Issue | [Feature]: Recency-based progressive mixed-precision KV cache | 🟡 中 | open | 2026-07-20 | [#49198](https://github.com/vllm-project/vllm/issues/49198) |
| 86 | vllm | Issue | [Bug]: Qwen3.5-35B FP8 kv cache is slower than BF16 kv cache on SM90 (Hopper) | 🟡 中 | open | 2026-07-15 | [#48786](https://github.com/vllm-project/vllm/issues/48786) |
| 87 | vllm | PR | [KV Offload] Bypass power-of-2 rounding in KV offload CPU pinned allocation | 🟡 中 | closed | 2026-07-12 | [#48436](https://github.com/vllm-project/vllm/pull/48436) |
| 88 | vllm | Issue | [Bug]: glm-5.2-fp8 with --kvcache-dtype fp8 with h20-3e with dspark error | 🟡 中 | open | 2026-07-12 | [#48406](https://github.com/vllm-project/vllm/issues/48406) |
| 89 | vllm | Issue | [Bug]: glm-5.2-fp8 with --kvcache-dtype fp8 with h20-3e error | 🟡 中 | open | 2026-07-12 | [#48405](https://github.com/vllm-project/vllm/issues/48405) |
| 90 | vllm | Issue | [Bug]: DSpark/DFlash speculative decoding fails to load with --kv-cache-dtype fp8_ds_mla: draft inherits an MLA-only cache layout | 🟡 中 | open | 2026-07-12 | [#48380](https://github.com/vllm-project/vllm/issues/48380) |
| 91 | vllm | Issue | [RFC]: fp8 KV cache for the Ampere sparse-MLA path (TRITON_MLA_SPARSE) via software dequant | 🟡 中 | open | 2026-07-11 | [#48374](https://github.com/vllm-project/vllm/issues/48374) |
| 92 | vllm | Issue | [Bug]: XPU: INC dequantizes int4 MoE experts to bf16 (OOM), which presents as an empty_cache livelock | 🟡 中 | closed | 2026-07-08 | [#47937](https://github.com/vllm-project/vllm/issues/47937) |
| 93 | vllm | PR | [Bugfix]Fix DeepSeek-V4 fp8_ds_mla KV cache reshape | 🟡 中 | closed | 2026-07-06 | [#47716](https://github.com/vllm-project/vllm/pull/47716) |
| 94 | vllm | Issue | [Bug]: fp8 KV cache + prefix caching truncates generation (ignore_eos bypassed) on Qwen3.5-NVFP4 | 🟡 中 | open | 2026-07-01 | [#47349](https://github.com/vllm-project/vllm/issues/47349) |
| 95 | vllm | Issue | [Bug]: vLLM 0.23.0: FlashInfer /  Triton attention + FP8 KV cache doesn't work on H200 (sm_90) | 🟡 中 | closed | 2026-06-29 | [#47037](https://github.com/vllm-project/vllm/issues/47037) |
| 96 | vllm | Issue | [Bug]: Why does kv-cache-dtype=fp8 OOM more easily than bf16/fp16 on long-context GLM-5.1-AWQ runs? | 🟡 中 | open | 2026-06-29 | [#46985](https://github.com/vllm-project/vllm/issues/46985) |
| 97 | vllm | PR | [Bugfix][Test] Skip test_mla_rope_kvcache_cat_fusion for FlashAttnMLA + fp8 KV cache | 🟡 中 | closed | 2026-06-27 | [#46916](https://github.com/vllm-project/vllm/pull/46916) |
| 98 | vllm | Issue | [Bug]: Out of Resource / block size error Kimi K2.7 on SM120 Blackwell with kv_cache_dtype fp8 | 🟡 中 | open | 2026-06-25 | [#46721](https://github.com/vllm-project/vllm/issues/46721) |
| 99 | vllm-ascend | Issue | [Bug]: glm5.1 with llmcache with glm5-fp8 failed | 🟡 中 | closed | 2026-06-25 | [#10936](https://github.com/vllm-project/vllm-ascend/issues/10936) |
| 100 | vllm | Issue | [Bug]: No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization). | 🟡 中 | open | 2026-06-24 | [#46611](https://github.com/vllm-project/vllm/issues/46611) |
| 101 | vllm | Issue | [Bug]: test_mla_rope_kvcache_cat_fusion fails with FlashAttnMLA + fp8 KV cache (NotImplementedError) | 🟡 中 | open | 2026-06-24 | [#46581](https://github.com/vllm-project/vllm/issues/46581) |
| 102 | vllm | PR | [Bugfix] fp8 KV cache: accept checkpoints with only v_scale | 🟡 中 | closed | 2026-06-21 | [#46265](https://github.com/vllm-project/vllm/pull/46265) |
| 103 | vllm | Issue | [Bug]: fp8 KV cache fails to load a checkpoint that provides only v_scale (assert layer.k_scale &gt; 0.0) | 🟡 中 | closed | 2026-06-21 | [#46264](https://github.com/vllm-project/vllm/issues/46264) |
| 104 | vllm | Issue | [Bug]: OOM in sparse_attn_indexer (fp8_fp4_mqa_logits) when KV offloading + DeepSeek-V4 is used with long contexts | 🟡 中 | open | 2026-06-15 | [#45663](https://github.com/vllm-project/vllm/issues/45663) |
| 105 | vllm | PR | [Kernel][Bugfix] Fix INT8 per-token-head KV cache rounding in Triton reshape-and-cache | 🟡 中 | closed | 2026-06-12 | [#45361](https://github.com/vllm-project/vllm/pull/45361) |
| 106 | vllm | PR | [Bugfix][Quantization] Don't reject fp8_e5m2 KV cache for non-fp8 quantized checkpoints | 🟡 中 | closed | 2026-06-09 | [#45040](https://github.com/vllm-project/vllm/pull/45040) |
| 107 | vllm | Issue | [Bug]: GLM-5 FP8 generates nonsense on BF16 KV cache dtype | 🟡 中 | closed | 2026-06-04 | [#44550](https://github.com/vllm-project/vllm/issues/44550) |
| 108 | vllm | Issue | [Bug] DFlash speculative decoding fundamentally incompatible with all KV cache quantization (fp8, turboquant) due to non-causal attention requirement | 🟡 中 | closed | 2026-05-03 | [#41559](https://github.com/vllm-project/vllm/issues/41559) |
| 109 | vllm | Issue | [Bug]:  Gemma 4 fails to initialize with per-token-head KV cache quantization | 🟡 中 | open | 2026-04-20 | [#40388](https://github.com/vllm-project/vllm/issues/40388) |
| 110 | vllm | PR | [Bugfix] Cuda Clean up scales Kvcache fp8/int8_per_token_head | 🟡 中 | closed | 2026-04-07 | [#39224](https://github.com/vllm-project/vllm/pull/39224) |
| 111 | vllm | Issue | [Bug]: Gemma 4 31B INT4 on 2×24GB GPUs (TP=2): GPU KV cache size is 25,200 tokens at max_model_len=131072, gpu_memory_utilization=0.96, BF16 KV | 🟡 中 | closed | 2026-04-07 | [#39133](https://github.com/vllm-project/vllm/issues/39133) |
| 112 | vllm-ascend | Issue | [Bug]:Ds3.2+A3 KVCache chain breakage at 100 concurrency results in precision problems | 🟡 中 | open | 2026-03-27 | [#7707](https://github.com/vllm-project/vllm-ascend/issues/7707) |
| 113 | vllm | Issue | [Feature]: Support Mixed-Precision KV Cache Configuration | 🟡 中 | closed | 2025-08-04 | [#22195](https://github.com/vllm-project/vllm/issues/22195) |
| 114 | vllm-ascend | Issue | [Bug]: qwen2.5 kv-cache quantization, KeyError: 'layers.0.self_attn.qkv_proj.kv_cache_offset' | 🟡 中 | closed | 2025-07-31 | [#2144](https://github.com/vllm-project/vllm-ascend/issues/2144) |
| 115 | vllm | PR | [Model] Fix minimax model cache & lm_head precision | 🟡 中 | closed | 2025-06-13 | [#19592](https://github.com/vllm-project/vllm/pull/19592) |
| 116 | vllm | PR | [Bugfix][Quantization] Reject unsupported compressed tensors KV cache schemes | 🟢 低 | closed | 2026-06-11 | [#45312](https://github.com/vllm-project/vllm/pull/45312) |
| 117 | vllm | PR | Keep first/last n token in high precision for nvfp4 kv cache | 🟢 低 | open | 2026-05-04 | [#41684](https://github.com/vllm-project/vllm/pull/41684) |
| 118 | vllm | PR | [Feature][Scheduler] Add split prefix caching feature to eliminate bf16 GEMM tiling divergence across cache-hit/miss paths | 🟢 低 | open | 2026-02-07 | [#34046](https://github.com/vllm-project/vllm/pull/34046) |
| 119 | vllm | PR | [Bugfix] Add int8 torch dtype for KVCache | 🟢 低 | closed | 2025-03-21 | [#15260](https://github.com/vllm-project/vllm/pull/15260) |
| 120 | vllm-ascend | PR | [BugFix] add int8 cache dtype && modify initialization of attention | 🟢 低 | closed | 2025-02-21 | [#134](https://github.com/vllm-project/vllm-ascend/pull/134) |
| 121 | vllm-ascend | PR | [BugFix]add int8 cache dtype when using attention quantization | 🟢 低 | closed | 2025-02-21 | [#128](https://github.com/vllm-project/vllm-ascend/pull/128) |
| 122 | vllm-ascend | PR | [BugFix]Add int8 cache dtype when using ascend attention quantization | 🟢 低 | closed | 2025-02-21 | [#125](https://github.com/vllm-project/vllm-ascend/pull/125) |

</details>

---

## 6. 精度回归与评测

通过 benchmark、perplexity、准确率评测发现的精度退化/回归问题，通常表现为某一版本或某一配置下评测指标下降。

**问题数: 400 条**

### 关键问题

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | PR | fix(models): pass quant_config to eh_proj in MTP layers to prevent silent precision loss | 🔴 极高 | open | 2026-07-13 | [#48506](https://github.com/vllm-project/vllm/pull/48506) |
| 2 | vllm-ascend | Issue | [Bug]: Accuracy repetitive issue with GLM-5.2-W4A8C8 on BFCL-v3 dataset precision testing | 🔴 高 | open | 2026-08-18 | [#14463](https://github.com/vllm-project/vllm-ascend/issues/14463) |
| 3 | vllm | PR | [Bug][ROCm] DSv4 with MRV2 + FULL_DECODE_ONLY has bad accuracy | 🔴 高 | open | 2026-08-17 | [#52646](https://github.com/vllm-project/vllm/pull/52646) |
| 4 | vllm | Issue | [Bug][ROCm]: DeepSeek V4 accuracy drops with MRV2 on MI350/MI355 when FULL_DECODE_ONLY graph | 🔴 高 | open | 2026-08-17 | [#52644](https://github.com/vllm-project/vllm/issues/52644) |
| 5 | vllm-ascend | PR | [CI] Test deterministic accuracy on A3-560T | 🔴 高 | open | 2026-08-17 | [#14431](https://github.com/vllm-project/vllm-ascend/pull/14431) |
| 6 | vllm-ascend | Issue | [Bug][0.23.0]: Accuracy Fluctuations with GLM-5.2-W4A8C8 on Ascend | 🔴 高 | open | 2026-08-16 | [#14378](https://github.com/vllm-project/vllm-ascend/issues/14378) |
| 7 | vllm-ascend | Issue | [v0.23.0][Bug]: The accuracy of the GSM8K dataset fluctuates and does not meet the requirements for the 310P Qwen3.5-2B-W8A8. | 🔴 高 | open | 2026-08-15 | [#14335](https://github.com/vllm-project/vllm-ascend/issues/14335) |
| 8 | vllm | PR | [CI] Shard Hybrid SSM NixlConnector PD accuracy tests into 4 config groups | 🔴 高 | open | 2026-08-14 | [#52354](https://github.com/vllm-project/vllm/pull/52354) |
| 9 | vllm | PR | [XPU][CI]Adjust source_file_dependencies for NixlConnector PD accuracy (4 GPUs) | 🔴 高 | closed | 2026-07-30 | [#50373](https://github.com/vllm-project/vllm/pull/50373) |
| 10 | vllm | Issue | [CI Failure]: LM Eval Qwen3.5 Models Accuracy issues | 🔴 高 | open | 2026-07-27 | [#50018](https://github.com/vllm-project/vllm/issues/50018) |
| 11 | vllm-ascend | PR | [BugFix] accuracy issue under SP and DP | 🔴 高 | closed | 2026-07-23 | [#12748](https://github.com/vllm-project/vllm-ascend/pull/12748) |
| 12 | vllm-ascend | PR | [Test] Restore custom MoE init routing for A2 accuracy validation | 🔴 高 | open | 2026-07-22 | [#12530](https://github.com/vllm-project/vllm-ascend/pull/12530) |
| 13 | vllm-ascend | Issue | [Bug]:模型internlm2_20b_chat，ceval数据集精度劣化 | 🔴 高 | closed | 2026-07-17 | [#12268](https://github.com/vllm-project/vllm-ascend/issues/12268) |
| 14 | vllm | PR | [XPU] Fix the accuracy issue for DeepSeekV4 on DP scenarios | 🔴 高 | open | 2026-07-16 | [#48859](https://github.com/vllm-project/vllm/pull/48859) |
| 15 | vllm-ascend | PR | [Bugfix] Fixes accuracy issues caused by synchronization problems during transmission. | 🔴 高 | closed | 2026-07-16 | [#12202](https://github.com/vllm-project/vllm-ascend/pull/12202) |
| 16 | vllm-ascend | PR | [BugFix] Fixes accuracy issues caused by synchronization problems during transmission. | 🔴 高 | closed | 2026-07-16 | [#12200](https://github.com/vllm-project/vllm-ascend/pull/12200) |
| 17 | vllm-ascend | PR | [v0.23.0][BugFix][GDN] Fix PD, PCP and DCP accuracy regressions | 🔴 高 | closed | 2026-07-14 | [#12027](https://github.com/vllm-project/vllm-ascend/pull/12027) |
| 18 | vllm-ascend | PR | [Bugfix][310P] Adapt unified spec decoding method and fix spec decoding accuracy issue. | 🔴 高 | closed | 2026-07-13 | [#11920](https://github.com/vllm-project/vllm-ascend/pull/11920) |
| 19 | vllm-ascend | PR | [Bugfix][310P] Adapt unified spec decoding method and fix spec decoding accuracy issue. | 🔴 高 | closed | 2026-07-13 | [#11919](https://github.com/vllm-project/vllm-ascend/pull/11919) |
| 20 | vllm-ascend | PR | [Bugfix][310P] Adapt unified spec decoding method and fix spec decoding accuracy issue. | 🔴 高 | closed | 2026-07-13 | [#11918](https://github.com/vllm-project/vllm-ascend/pull/11918) |
| 21 | vllm-ascend | PR | [Bugfix][310P] Adapt unified spec decoding method and fix spec decoding accuracy issue. | 🔴 高 | closed | 2026-07-13 | [#11913](https://github.com/vllm-project/vllm-ascend/pull/11913) |
| 22 | vllm-ascend | PR | [Bugfix][310P] Adapt unified spec decoding method and fix spec decoding accuracy issue. | 🔴 高 | closed | 2026-07-13 | [#11909](https://github.com/vllm-project/vllm-ascend/pull/11909) |
| 23 | vllm | PR | Re-enable DBO accuracy test on Blackwell | 🔴 高 | open | 2026-07-13 | [#48457](https://github.com/vllm-project/vllm/pull/48457) |
| 24 | vllm-ascend | PR | [CI] Remove aime2025 accuracy benchmark | 🔴 高 | closed | 2026-07-10 | [#11794](https://github.com/vllm-project/vllm-ascend/pull/11794) |
| 25 | vllm-ascend | PR | [CI] Remove aime2025 accuracy benchmark from DeepSeek-V3.2 dua… | 🔴 高 | closed | 2026-07-10 | [#11792](https://github.com/vllm-project/vllm-ascend/pull/11792) |
| 26 | vllm-ascend | PR | [Bugfix][310P] Adapt unified spec decoding method and fix spec decoding accuracy issue. | 🔴 高 | closed | 2026-07-09 | [#11699](https://github.com/vllm-project/vllm-ascend/pull/11699) |
| 27 | vllm | PR | [CI Bug] Fully solve accuracy issue for DSv3.2 + MTP + Sequence Parallel | 🔴 高 | closed | 2026-07-08 | [#48036](https://github.com/vllm-project/vllm/pull/48036) |
| 28 | vllm | PR | [CI Bug Fix] Temp fix for v3.2 accuracy | 🔴 高 | closed | 2026-07-07 | [#47902](https://github.com/vllm-project/vllm/pull/47902) |
| 29 | vllm-ascend | Issue | [Bug]: Qwen3-Next-80B-A3B-Instruct-W8A8  has accuracy issue. | 🔴 高 | closed | 2026-07-06 | [#11509](https://github.com/vllm-project/vllm-ascend/issues/11509) |
| 30 | vllm-ascend | PR | [BugFix]fix qwen3.5+pcp+chunkprefill accuracy error | 🔴 高 | closed | 2026-07-06 | [#11508](https://github.com/vllm-project/vllm-ascend/pull/11508) |
| 31 | vllm-ascend | PR | [BugFix][310p]:fix the accuracy issue caused by mtp and aclgraph | 🔴 高 | closed | 2026-07-03 | [#11408](https://github.com/vllm-project/vllm-ascend/pull/11408) |
| 32 | vllm | PR | [XPU] Fix PP accuracy on XPU device | 🔴 高 | closed | 2026-07-01 | [#47253](https://github.com/vllm-project/vllm/pull/47253) |
| 33 | vllm | Issue | [Bug][Model Runner V2]: [Bug] GLM-5.2 shows low aa_lcr accuracy and large TPOT fluctuations on B300 | 🔴 高 | closed | 2026-07-01 | [#47239](https://github.com/vllm-project/vllm/issues/47239) |
| 34 | vllm | PR | [AMD][Bugfix][EPLB] Fix elastic EP scaling accuracy on ROCm | 🔴 高 | closed | 2026-06-30 | [#47206](https://github.com/vllm-project/vllm/pull/47206) |
| 35 | vllm-ascend | PR | [CI][Misc] Modify accuracy test thresholds for multiple models | 🔴 高 | closed | 2026-06-27 | [#11042](https://github.com/vllm-project/vllm-ascend/pull/11042) |
| 36 | vllm-ascend | PR | [CI][Misc] Modify accuracy test thresholds for multiple models | 🔴 高 | closed | 2026-06-27 | [#11041](https://github.com/vllm-project/vllm-ascend/pull/11041) |
| 37 | vllm-ascend | PR | [Doc][Misc] Correct accuracy evaluation count in Qwen tutorial | 🔴 高 | closed | 2026-06-26 | [#11003](https://github.com/vllm-project/vllm-ascend/pull/11003) |
| 38 | vllm-ascend | PR | [CI] Move accuracy-group-2 to weekly single node configs | 🔴 高 | closed | 2026-06-25 | [#10974](https://github.com/vllm-project/vllm-ascend/pull/10974) |
| 39 | vllm-ascend | PR | [CI] Move accuracy-group-2 to weekly single node configs | 🔴 高 | closed | 2026-06-25 | [#10967](https://github.com/vllm-project/vllm-ascend/pull/10967) |
| 40 | vllm-ascend | PR | [Test] Limit DeepSeek R1 longseq accuracy prompts | 🔴 高 | closed | 2026-06-24 | [#10886](https://github.com/vllm-project/vllm-ascend/pull/10886) |

### 关键规律与分析

1. **CI 精度 failure 是重要信号源**：大量 mi300 / H100 的 ROCm vs NVIDIA 精度回归被 CI 捕获，说明跨硬件数值一致性是持续性挑战。
2. **benchmark / perplexity / 准确率退化**通常定位到具体 PR 或版本，是量化优化、算子替换引入回归的典型暴露方式。
3. 评测类回归容易被性能优化掩盖，需要建立专门的 accuracy test 持续监控。

<details>
<summary>展开全部 400 条</summary>

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm | PR | fix(models): pass quant_config to eh_proj in MTP layers to prevent silent precision loss | 🔴 极高 | open | 2026-07-13 | [#48506](https://github.com/vllm-project/vllm/pull/48506) |
| 2 | vllm-ascend | Issue | [Bug]: Accuracy repetitive issue with GLM-5.2-W4A8C8 on BFCL-v3 dataset precision testing | 🔴 高 | open | 2026-08-18 | [#14463](https://github.com/vllm-project/vllm-ascend/issues/14463) |
| 3 | vllm | PR | [Bug][ROCm] DSv4 with MRV2 + FULL_DECODE_ONLY has bad accuracy | 🔴 高 | open | 2026-08-17 | [#52646](https://github.com/vllm-project/vllm/pull/52646) |
| 4 | vllm | Issue | [Bug][ROCm]: DeepSeek V4 accuracy drops with MRV2 on MI350/MI355 when FULL_DECODE_ONLY graph | 🔴 高 | open | 2026-08-17 | [#52644](https://github.com/vllm-project/vllm/issues/52644) |
| 5 | vllm-ascend | PR | [CI] Test deterministic accuracy on A3-560T | 🔴 高 | open | 2026-08-17 | [#14431](https://github.com/vllm-project/vllm-ascend/pull/14431) |
| 6 | vllm-ascend | Issue | [Bug][0.23.0]: Accuracy Fluctuations with GLM-5.2-W4A8C8 on Ascend | 🔴 高 | open | 2026-08-16 | [#14378](https://github.com/vllm-project/vllm-ascend/issues/14378) |
| 7 | vllm-ascend | Issue | [v0.23.0][Bug]: The accuracy of the GSM8K dataset fluctuates and does not meet the requirements for the 310P Qwen3.5-2B-W8A8. | 🔴 高 | open | 2026-08-15 | [#14335](https://github.com/vllm-project/vllm-ascend/issues/14335) |
| 8 | vllm | PR | [CI] Shard Hybrid SSM NixlConnector PD accuracy tests into 4 config groups | 🔴 高 | open | 2026-08-14 | [#52354](https://github.com/vllm-project/vllm/pull/52354) |
| 9 | vllm | PR | [XPU][CI]Adjust source_file_dependencies for NixlConnector PD accuracy (4 GPUs) | 🔴 高 | closed | 2026-07-30 | [#50373](https://github.com/vllm-project/vllm/pull/50373) |
| 10 | vllm | Issue | [CI Failure]: LM Eval Qwen3.5 Models Accuracy issues | 🔴 高 | open | 2026-07-27 | [#50018](https://github.com/vllm-project/vllm/issues/50018) |
| 11 | vllm-ascend | PR | [BugFix] accuracy issue under SP and DP | 🔴 高 | closed | 2026-07-23 | [#12748](https://github.com/vllm-project/vllm-ascend/pull/12748) |
| 12 | vllm-ascend | PR | [Test] Restore custom MoE init routing for A2 accuracy validation | 🔴 高 | open | 2026-07-22 | [#12530](https://github.com/vllm-project/vllm-ascend/pull/12530) |
| 13 | vllm-ascend | Issue | [Bug]:模型internlm2_20b_chat，ceval数据集精度劣化 | 🔴 高 | closed | 2026-07-17 | [#12268](https://github.com/vllm-project/vllm-ascend/issues/12268) |
| 14 | vllm | PR | [XPU] Fix the accuracy issue for DeepSeekV4 on DP scenarios | 🔴 高 | open | 2026-07-16 | [#48859](https://github.com/vllm-project/vllm/pull/48859) |
| 15 | vllm-ascend | PR | [Bugfix] Fixes accuracy issues caused by synchronization problems during transmission. | 🔴 高 | closed | 2026-07-16 | [#12202](https://github.com/vllm-project/vllm-ascend/pull/12202) |
| 16 | vllm-ascend | PR | [BugFix] Fixes accuracy issues caused by synchronization problems during transmission. | 🔴 高 | closed | 2026-07-16 | [#12200](https://github.com/vllm-project/vllm-ascend/pull/12200) |
| 17 | vllm-ascend | PR | [v0.23.0][BugFix][GDN] Fix PD, PCP and DCP accuracy regressions | 🔴 高 | closed | 2026-07-14 | [#12027](https://github.com/vllm-project/vllm-ascend/pull/12027) |
| 18 | vllm-ascend | PR | [Bugfix][310P] Adapt unified spec decoding method and fix spec decoding accuracy issue. | 🔴 高 | closed | 2026-07-13 | [#11920](https://github.com/vllm-project/vllm-ascend/pull/11920) |
| 19 | vllm-ascend | PR | [Bugfix][310P] Adapt unified spec decoding method and fix spec decoding accuracy issue. | 🔴 高 | closed | 2026-07-13 | [#11919](https://github.com/vllm-project/vllm-ascend/pull/11919) |
| 20 | vllm-ascend | PR | [Bugfix][310P] Adapt unified spec decoding method and fix spec decoding accuracy issue. | 🔴 高 | closed | 2026-07-13 | [#11918](https://github.com/vllm-project/vllm-ascend/pull/11918) |
| 21 | vllm-ascend | PR | [Bugfix][310P] Adapt unified spec decoding method and fix spec decoding accuracy issue. | 🔴 高 | closed | 2026-07-13 | [#11913](https://github.com/vllm-project/vllm-ascend/pull/11913) |
| 22 | vllm-ascend | PR | [Bugfix][310P] Adapt unified spec decoding method and fix spec decoding accuracy issue. | 🔴 高 | closed | 2026-07-13 | [#11909](https://github.com/vllm-project/vllm-ascend/pull/11909) |
| 23 | vllm | PR | Re-enable DBO accuracy test on Blackwell | 🔴 高 | open | 2026-07-13 | [#48457](https://github.com/vllm-project/vllm/pull/48457) |
| 24 | vllm-ascend | PR | [CI] Remove aime2025 accuracy benchmark | 🔴 高 | closed | 2026-07-10 | [#11794](https://github.com/vllm-project/vllm-ascend/pull/11794) |
| 25 | vllm-ascend | PR | [CI] Remove aime2025 accuracy benchmark from DeepSeek-V3.2 dua… | 🔴 高 | closed | 2026-07-10 | [#11792](https://github.com/vllm-project/vllm-ascend/pull/11792) |
| 26 | vllm-ascend | PR | [Bugfix][310P] Adapt unified spec decoding method and fix spec decoding accuracy issue. | 🔴 高 | closed | 2026-07-09 | [#11699](https://github.com/vllm-project/vllm-ascend/pull/11699) |
| 27 | vllm | PR | [CI Bug] Fully solve accuracy issue for DSv3.2 + MTP + Sequence Parallel | 🔴 高 | closed | 2026-07-08 | [#48036](https://github.com/vllm-project/vllm/pull/48036) |
| 28 | vllm | PR | [CI Bug Fix] Temp fix for v3.2 accuracy | 🔴 高 | closed | 2026-07-07 | [#47902](https://github.com/vllm-project/vllm/pull/47902) |
| 29 | vllm-ascend | Issue | [Bug]: Qwen3-Next-80B-A3B-Instruct-W8A8  has accuracy issue. | 🔴 高 | closed | 2026-07-06 | [#11509](https://github.com/vllm-project/vllm-ascend/issues/11509) |
| 30 | vllm-ascend | PR | [BugFix]fix qwen3.5+pcp+chunkprefill accuracy error | 🔴 高 | closed | 2026-07-06 | [#11508](https://github.com/vllm-project/vllm-ascend/pull/11508) |
| 31 | vllm-ascend | PR | [BugFix][310p]:fix the accuracy issue caused by mtp and aclgraph | 🔴 高 | closed | 2026-07-03 | [#11408](https://github.com/vllm-project/vllm-ascend/pull/11408) |
| 32 | vllm | PR | [XPU] Fix PP accuracy on XPU device | 🔴 高 | closed | 2026-07-01 | [#47253](https://github.com/vllm-project/vllm/pull/47253) |
| 33 | vllm | Issue | [Bug][Model Runner V2]: [Bug] GLM-5.2 shows low aa_lcr accuracy and large TPOT fluctuations on B300 | 🔴 高 | closed | 2026-07-01 | [#47239](https://github.com/vllm-project/vllm/issues/47239) |
| 34 | vllm | PR | [AMD][Bugfix][EPLB] Fix elastic EP scaling accuracy on ROCm | 🔴 高 | closed | 2026-06-30 | [#47206](https://github.com/vllm-project/vllm/pull/47206) |
| 35 | vllm-ascend | PR | [CI][Misc] Modify accuracy test thresholds for multiple models | 🔴 高 | closed | 2026-06-27 | [#11042](https://github.com/vllm-project/vllm-ascend/pull/11042) |
| 36 | vllm-ascend | PR | [CI][Misc] Modify accuracy test thresholds for multiple models | 🔴 高 | closed | 2026-06-27 | [#11041](https://github.com/vllm-project/vllm-ascend/pull/11041) |
| 37 | vllm-ascend | PR | [Doc][Misc] Correct accuracy evaluation count in Qwen tutorial | 🔴 高 | closed | 2026-06-26 | [#11003](https://github.com/vllm-project/vllm-ascend/pull/11003) |
| 38 | vllm-ascend | PR | [CI] Move accuracy-group-2 to weekly single node configs | 🔴 高 | closed | 2026-06-25 | [#10974](https://github.com/vllm-project/vllm-ascend/pull/10974) |
| 39 | vllm-ascend | PR | [CI] Move accuracy-group-2 to weekly single node configs | 🔴 高 | closed | 2026-06-25 | [#10967](https://github.com/vllm-project/vllm-ascend/pull/10967) |
| 40 | vllm-ascend | PR | [Test] Limit DeepSeek R1 longseq accuracy prompts | 🔴 高 | closed | 2026-06-24 | [#10886](https://github.com/vllm-project/vllm-ascend/pull/10886) |
| 41 | vllm-ascend | PR | [CI] Modify accuracy threshold for aime2025 dataset to 10 | 🔴 高 | closed | 2026-06-24 | [#10881](https://github.com/vllm-project/vllm-ascend/pull/10881) |
| 42 | vllm-ascend | PR | [0.22.1][CI] Modify the accuracy threshold of the aime2025 dataset to 10 | 🔴 高 | closed | 2026-06-24 | [#10869](https://github.com/vllm-project/vllm-ascend/pull/10869) |
| 43 | vllm-ascend | PR | [CI] Modify the accuracy threshold of glm5 in aime2025 | 🔴 高 | closed | 2026-06-22 | [#10789](https://github.com/vllm-project/vllm-ascend/pull/10789) |
| 44 | vllm-ascend | Issue | [Bug][Upstream]: phi3v accuracy test failed | 🔴 高 | closed | 2026-06-18 | [#10723](https://github.com/vllm-project/vllm-ascend/issues/10723) |
| 45 | vllm | Issue | [Bug]: Qwen3-VL shows inconsistent accuracy between enabled and disabled graph modes on VLLM0.20.2 | 🔴 高 | open | 2026-06-17 | [#45904](https://github.com/vllm-project/vllm/issues/45904) |
| 46 | vllm-ascend | PR | [BugFix] A00282: force ALLGATHER on A2 to isolate MC2 accuracy regression | 🔴 高 | closed | 2026-06-16 | [#10519](https://github.com/vllm-project/vllm-ascend/pull/10519) |
| 47 | vllm-ascend | PR | [Test][Misc] Refactor and consolidate long sequence E2E accuracy tests | 🔴 高 | closed | 2026-06-15 | [#10513](https://github.com/vllm-project/vllm-ascend/pull/10513) |
| 48 | vllm-ascend | PR | [DOC] Fix accuracy evaluation description in Qwen3.5-27B tutorialUpdate Qwen3.5-27B.md | 🔴 高 | closed | 2026-06-12 | [#10416](https://github.com/vllm-project/vllm-ascend/pull/10416) |
| 49 | vllm-ascend | Issue | [Bug]: 0.20.2rc1 DeepSeek V4 Flash on A2 64xNPU 4P1D 128K config, the GPQA dataset accuracy is lower than official accuracy. | 🔴 高 | closed | 2026-06-12 | [#10413](https://github.com/vllm-project/vllm-ascend/issues/10413) |
| 50 | vllm-ascend | Issue | [Bug]: DS-distill-LLaMA-70B模型，ceval数据集精度劣化 | 🔴 高 | open | 2026-06-10 | [#10280](https://github.com/vllm-project/vllm-ascend/issues/10280) |
| 51 | vllm | PR | [Bugfix] Fix nemotron accuracy drop introduced by #41184 | 🔴 高 | closed | 2026-06-09 | [#45037](https://github.com/vllm-project/vllm/pull/45037) |
| 52 | vllm-ascend | PR | [BugFix] [CI] drop accuracy tests | 🔴 高 | closed | 2026-06-09 | [#10243](https://github.com/vllm-project/vllm-ascend/pull/10243) |
| 53 | vllm-ascend | PR | [CI] Remove A2 accuracy group 2 from nightly matrix | 🔴 高 | closed | 2026-06-09 | [#10240](https://github.com/vllm-project/vllm-ascend/pull/10240) |
| 54 | vllm-ascend | PR | [BugFix][CI] Enable chat template for InternVL and LLaVA accuracy configs | 🔴 高 | open | 2026-06-08 | [#10170](https://github.com/vllm-project/vllm-ascend/pull/10170) |
| 55 | vllm-ascend | Issue | [Bug][Upstream]: accuracy test failed for model granite_speech | 🔴 高 | closed | 2026-06-05 | [#10106](https://github.com/vllm-project/vllm-ascend/issues/10106) |
| 56 | vllm-ascend | Issue | [Bug][Upstream]: accuracy test failed for gte | 🔴 高 | closed | 2026-06-05 | [#10105](https://github.com/vllm-project/vllm-ascend/issues/10105) |
| 57 | vllm-ascend | Issue | [Bug]: 使用evalscope+mmlu-pro精度集测试vllm-ascend镜像，得分总是比使用ais-bench低了5-6分 | 🔴 高 | closed | 2026-06-05 | [#10068](https://github.com/vllm-project/vllm-ascend/issues/10068) |
| 58 | vllm-ascend | PR | [BugFix] chunk_scaled_dot_kkt_fwd_kernel accuracy issues | 🔴 高 | closed | 2026-06-04 | [#10033](https://github.com/vllm-project/vllm-ascend/pull/10033) |
| 59 | vllm-ascend | Issue | [Bug]: The minimax model combined with PCP/DCP and Eagle3 has accuracy issues. | 🔴 高 | closed | 2026-06-03 | [#9959](https://github.com/vllm-project/vllm-ascend/issues/9959) |
| 60 | vllm-ascend | PR | [BugFix]fix DeepSeek-V4 PIECEWISE ACLGraph accuracy | 🔴 高 | open | 2026-06-02 | [#9896](https://github.com/vllm-project/vllm-ascend/pull/9896) |
| 61 | vllm-ascend | PR | [BugFix][CI] Fix Qwen3-VL MMMU accuracy test failure by enabling chat template | 🔴 高 | closed | 2026-06-01 | [#9829](https://github.com/vllm-project/vllm-ascend/pull/9829) |
| 62 | vllm-ascend | Issue | [Bug][Upstream]: granite_speech accuracy test failed | 🔴 高 | open | 2026-05-30 | [#9752](https://github.com/vllm-project/vllm-ascend/issues/9752) |
| 63 | vllm-ascend | Issue | [Bug][Upstream]: gte accuracy test failed | 🔴 高 | open | 2026-05-30 | [#9751](https://github.com/vllm-project/vllm-ascend/issues/9751) |
| 64 | vllm | Issue | [Parity with CUDA]: Add top 5-10 popular OSS models into mi355, mi325, mi300 into vLLM nightly performance  & accuracy regression testing & dashboard | 🔴 高 | open | 2026-05-28 | [#43916](https://github.com/vllm-project/vllm/issues/43916) |
| 65 | vllm | PR | [Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950 | 🔴 高 | closed | 2026-05-27 | [#43781](https://github.com/vllm-project/vllm/pull/43781) |
| 66 | vllm | PR | [Bugfix][Core] MTP + enable prefix caching + mamba accuracy fix | 🔴 高 | open | 2026-05-26 | [#43650](https://github.com/vllm-project/vllm/pull/43650) |
| 67 | vllm | PR | Fix Qwen3-VL and Qwen3-omni-thinker accuracy degradation from deepstack inputs under torch.compile | 🔴 高 | closed | 2026-05-25 | [#43617](https://github.com/vllm-project/vllm/pull/43617) |
| 68 | vllm | Issue | [Bug]: Qwen3-VL-2B-Instruct Geo3K accuracy score lower than SGLang with deterministic sampling | 🔴 高 | closed | 2026-05-25 | [#43602](https://github.com/vllm-project/vllm/issues/43602) |
| 69 | vllm | PR | [Misc] Print accuracy value for PD tests even on success  | 🔴 高 | closed | 2026-05-25 | [#43583](https://github.com/vllm-project/vllm/pull/43583) |
| 70 | vllm | PR | [CI] Stabilize Hybrid SSM disagg PD gsm8k accuracy test (#43301) | 🔴 高 | closed | 2026-05-25 | [#43570](https://github.com/vllm-project/vllm/pull/43570) |
| 71 | vllm | Issue | [Bug]: Accuracy drops ~20% when `--enable-prefix-caching` is used together with MTP speculative decoding (Qwen3.6 35B-A3B) | 🔴 高 | closed | 2026-05-25 | [#43559](https://github.com/vllm-project/vllm/issues/43559) |
| 72 | vllm-ascend | PR | [BugFix] Fix Qwen3-Next PCP full graph accuracy | 🔴 高 | closed | 2026-05-22 | [#9479](https://github.com/vllm-project/vllm-ascend/pull/9479) |
| 73 | vllm-ascend | Issue | [Bug]: BFCL_v1_ast accuracy drop on Ascend NPU for DeepSeek-V4-Flash | 🔴 高 | closed | 2026-05-21 | [#9400](https://github.com/vllm-project/vllm-ascend/issues/9400) |
| 74 | vllm | PR | [CI] Lower granite-4.0-h-tiny gsm8k threshold for Hybrid SSM NixlConnector PD accuracy tests (4 GPUs) | 🔴 高 | closed | 2026-05-20 | [#43186](https://github.com/vllm-project/vllm/pull/43186) |
| 75 | vllm-ascend | PR | [BugFix] Fix the accuracy issue in DFlash FULL_DESCODE_ONLY mode caused by attn group sorting | 🔴 高 | closed | 2026-05-19 | [#9322](https://github.com/vllm-project/vllm-ascend/pull/9322) |
| 76 | vllm-ascend | PR | [BugFix] Fix the accuracy issue in DFlash FULL_DESCODE_ONLY mode caused by attn group sorting | 🔴 高 | closed | 2026-05-18 | [#9257](https://github.com/vllm-project/vllm-ascend/pull/9257) |
| 77 | vllm | PR | [Model Runner v2] fix pd accuracy | 🔴 高 | closed | 2026-05-17 | [#42888](https://github.com/vllm-project/vllm/pull/42888) |
| 78 | vllm | PR | [ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy | 🔴 高 | closed | 2026-05-16 | [#42810](https://github.com/vllm-project/vllm/pull/42810) |
| 79 | vllm | Issue | [Bug]: Significant accuracy discrepancies across different vLLM versions. | 🔴 高 | open | 2026-05-16 | [#42801](https://github.com/vllm-project/vllm/issues/42801) |
| 80 | vllm | PR | Fix ITL accuracy when server batches multiple tokens per SSE chunk | 🔴 高 | open | 2026-05-14 | [#42661](https://github.com/vllm-project/vllm/pull/42661) |
| 81 | vllm-ascend | Issue | [Bug]: precision loss for DeepSeek-V4-Flash with large MTP num_speculative_tokens | 🔴 高 | open | 2026-05-13 | [#9111](https://github.com/vllm-project/vllm-ascend/issues/9111) |
| 82 | vllm | Issue | [Bug]: Qwen3.5-27B Disagg accuracy gsm8k collapses with async scheduling when TP==1 | 🔴 高 | closed | 2026-05-09 | [#42182](https://github.com/vllm-project/vllm/issues/42182) |
| 83 | vllm | PR | [Bugfix] Fix GDN KKT precision loss on Hopper GPUs by aligning tl.dot operand layout with WGMMA | 🔴 高 | closed | 2026-05-08 | [#42076](https://github.com/vllm-project/vllm/pull/42076) |
| 84 | vllm-ascend | Issue | [Bug]: FlashComm1 + MTP speculative decoding causes intermittent accuracy degradation on A3 for Qwen3.5-397B-A17B | 🔴 高 | closed | 2026-05-08 | [#8989](https://github.com/vllm-project/vllm-ascend/issues/8989) |
| 85 | vllm-ascend | PR | [BugFix][P/D] Fix Mooncake Connector MTP accuracy bug | 🔴 高 | closed | 2026-05-06 | [#8908](https://github.com/vllm-project/vllm-ascend/pull/8908) |
| 86 | vllm-ascend | PR | [v0.18.0][Bugfix][P/D] Fix Mooncake Connector MTP accuracy bug | 🔴 高 | closed | 2026-05-06 | [#8890](https://github.com/vllm-project/vllm-ascend/pull/8890) |
| 87 | vllm-ascend | Issue | [Bug]:For GLM 5/5.1 under a four-node PD separation setup with TP16 DP2 parallelism, the GPQA accuracy fell short of the standard, achieving only 67.17. | 🔴 高 | closed | 2026-04-30 | [#8844](https://github.com/vllm-project/vllm-ascend/issues/8844) |
| 88 | vllm | Issue | [Bug]: MM accuracy issue caused by transformers upgrade | 🔴 高 | closed | 2026-04-29 | [#41207](https://github.com/vllm-project/vllm/issues/41207) |
| 89 | vllm | PR | [Bugfix] Fix token loss in PP mode which causes degraded accuracy | 🔴 高 | closed | 2026-04-28 | [#41133](https://github.com/vllm-project/vllm/pull/41133) |
| 90 | vllm-ascend | Issue | [Misc]: PD separation accuracy anomaly | 🔴 高 | closed | 2026-04-28 | [#8775](https://github.com/vllm-project/vllm-ascend/issues/8775) |
| 91 | vllm | PR | [Bugfix] Fix DeepSeek V2-Lite Accuracy drop | 🔴 高 | closed | 2026-04-23 | [#40673](https://github.com/vllm-project/vllm/pull/40673) |
| 92 | vllm-ascend | Issue | [Bug]:  Accuracy degradation on DeepSeek-V3.2 when PCP is enabled without Expert Parallel | 🔴 高 | open | 2026-04-22 | [#8579](https://github.com/vllm-project/vllm-ascend/issues/8579) |
| 93 | vllm | Issue | [CI Failure]: mi355_4: Distributed NixlConnector PD accuracy (4 GPUs) | 🔴 高 | closed | 2026-04-21 | [#40527](https://github.com/vllm-project/vllm/issues/40527) |
| 94 | vllm | Issue | [CI Failure]: mi300_4: Hyrbid SSM NixlConnector PD accuracy tests (4 GPUs) | 🔴 高 | closed | 2026-04-21 | [#40513](https://github.com/vllm-project/vllm/issues/40513) |
| 95 | vllm | Issue | [CI Failure]: mi300_4: Distributed NixlConnector PD accuracy (4 GPUs) | 🔴 高 | closed | 2026-04-21 | [#40512](https://github.com/vllm-project/vllm/issues/40512) |
| 96 | vllm | Issue | [CI Failure]: mi300_4: DeepSeek V2-Lite Accuracy (4xH100-4xMI300) | 🔴 高 | closed | 2026-04-21 | [#40510](https://github.com/vllm-project/vllm/issues/40510) |
| 97 | vllm | Issue | [CI Failure]: mi300_4: DP EP Distributed NixlConnector PD accuracy tests (4 GPUs) | 🔴 高 | closed | 2026-04-21 | [#40509](https://github.com/vllm-project/vllm/issues/40509) |
| 98 | vllm | Issue | [CI Failure]: mi300_4: CrossLayer KV layout Distributed NixlConnector PD accuracy tests (4 GPUs) | 🔴 高 | closed | 2026-04-21 | [#40508](https://github.com/vllm-project/vllm/issues/40508) |
| 99 | vllm | Issue | [CI Failure]: mi250_4: Distributed NixlConnector PD accuracy (4 GPUs) | 🔴 高 | closed | 2026-04-21 | [#40481](https://github.com/vllm-project/vllm/issues/40481) |
| 100 | vllm-ascend | Issue | [Bug]: Qwen3.5 experiences accuracy degradation when using PD disaggregation. | 🔴 高 | closed | 2026-04-19 | [#8421](https://github.com/vllm-project/vllm-ascend/issues/8421) |
| 101 | vllm-ascend | Issue | [Bug]: High concurrency or enabling eagle3 and streaming + function call can trigger accuracy issues on kimi k2.5 | 🔴 高 | open | 2026-04-18 | [#8415](https://github.com/vllm-project/vllm-ascend/issues/8415) |
| 102 | vllm | Issue | [Bug]: High concurrency or enabling eagle3 and streaming + function call can trigger accuracy issues on kimi k2.5 | 🔴 高 | open | 2026-04-18 | [#40248](https://github.com/vllm-project/vllm/issues/40248) |
| 103 | vllm | Issue | [CI Failure]: mi355_4: DP EP Distributed NixlConnector PD accuracy tests (4 GPUs) | 🔴 高 | closed | 2026-04-18 | [#40243](https://github.com/vllm-project/vllm/issues/40243) |
| 104 | vllm | Issue | [CI Failure]: mi250_4: Hyrbid SSM NixlConnector PD accuracy tests (4 GPUs) | 🔴 高 | closed | 2026-04-18 | [#40207](https://github.com/vllm-project/vllm/issues/40207) |
| 105 | vllm | Issue | [CI Failure]: mi250_1: Multi-Modal Accuracy Eval (Small Models) | 🔴 高 | closed | 2026-04-18 | [#40201](https://github.com/vllm-project/vllm/issues/40201) |
| 106 | vllm-ascend | Issue | [Bug]: [ v0.11,v0.18.rc1]A general accuracy bug report and solution discussion, theoretically involving all MOE models including the Minimax series and Qwen series. | 🔴 高 | open | 2026-04-17 | [#8359](https://github.com/vllm-project/vllm-ascend/issues/8359) |
| 107 | vllm | PR | [XPU] fix all_reduce all-zero accuracy issue under torch.compile | 🔴 高 | closed | 2026-04-15 | [#39844](https://github.com/vllm-project/vllm/pull/39844) |
| 108 | vllm | PR | [MRv2]fix: model accuracy regression caused by reusing the stale last_sampled_tokens and draft_tokens | 🔴 高 | closed | 2026-04-14 | [#39833](https://github.com/vllm-project/vllm/pull/39833) |
| 109 | vllm-ascend | PR | Update accuracy_groups_a2.json | 🔴 高 | closed | 2026-04-13 | [#8191](https://github.com/vllm-project/vllm-ascend/pull/8191) |
| 110 | vllm-ascend | PR | [CI] Hot fix for nightly accuracy test and doc tests | 🔴 高 | closed | 2026-04-10 | [#8139](https://github.com/vllm-project/vllm-ascend/pull/8139) |
| 111 | vllm | Issue | [Bug][MoE] DeepEP HT hardcodes per_act_token_quant=False, causing crash/accuracy loss | 🔴 高 | closed | 2026-04-05 | [#39022](https://github.com/vllm-project/vllm/issues/39022) |
| 112 | vllm | PR | [PD][HeteroArch]Fix accuracy issue with CPU_ATTN as Decoder and Flash_ATTN as prefiller | 🔴 高 | closed | 2026-04-03 | [#38935](https://github.com/vllm-project/vllm/pull/38935) |
| 113 | vllm | Issue | [Bug]: heterogeneous disaggregated serving XPU (Prefill) + CPU (Decode) accuracy issue | 🔴 高 | closed | 2026-04-01 | [#38710](https://github.com/vllm-project/vllm/issues/38710) |
| 114 | vllm-ascend | PR | [BugFix][P/D] fix padding error on FullGraph mode && fix layerwise connector mamba accuracy | 🔴 高 | closed | 2026-03-20 | [#7506](https://github.com/vllm-project/vllm-ascend/pull/7506) |
| 115 | vllm | PR | [ROCm][CI] Fix accuracy for llama-nemotron-vl pooling tests | 🔴 高 | closed | 2026-03-19 | [#37613](https://github.com/vllm-project/vllm/pull/37613) |
| 116 | vllm-ascend | PR | [Misc] Refactor aclgraph accuracy test to use logprob-based comparison | 🔴 高 | closed | 2026-03-19 | [#7455](https://github.com/vllm-project/vllm-ascend/pull/7455) |
| 117 | vllm | Issue | [Bug]: Accuracy issue running Model Runner V2 with Qwen3.5 | 🔴 高 | closed | 2026-03-18 | [#37471](https://github.com/vllm-project/vllm/issues/37471) |
| 118 | vllm | PR | [Bugfix][Tool Parser] Fix Kimi-K2.5 parser accuracy, buffer limits, and token leaks | 🔴 高 | closed | 2026-03-18 | [#37384](https://github.com/vllm-project/vllm/pull/37384) |
| 119 | vllm-ascend | Issue | [Bug]: test_disaggregated_encoder.py::test_models test case accuracy: open EPLB and close EPLB is different | 🔴 高 | closed | 2026-03-18 | [#7408](https://github.com/vllm-project/vllm-ascend/issues/7408) |
| 120 | vllm | Issue | [Bug]: In_proj_ba of GDN in Qwen3Next use MergeColumnParallelLinear may cause accuracy decrease? | 🔴 高 | closed | 2026-03-17 | [#37271](https://github.com/vllm-project/vllm/issues/37271) |
| 121 | vllm | Issue | [Performance]: vllm and transformer call the same Qwen3-VL-AI4TEST-V1 model, with roughly the same configuration, but the visual label accuracy is 20% lower in testing. | 🔴 高 | closed | 2026-03-17 | [#37257](https://github.com/vllm-project/vllm/issues/37257) |
| 122 | vllm | Issue | [Bug]: Kimi-K2.5 Tool Parser Critical Issues - 87% Accuracy, 8KB Limit, Token Leakage | 🔴 高 | closed | 2026-03-16 | [#37184](https://github.com/vllm-project/vllm/issues/37184) |
| 123 | vllm | PR | [Tool Parser] Kimi K2: guided decoding for tool_choice="auto" — 75% → 100% schema accuracy | 🔴 高 | closed | 2026-03-12 | [#36891](https://github.com/vllm-project/vllm/pull/36891) |
| 124 | vllm | PR | fix for the model not even loading and zero accuracy | 🔴 高 | closed | 2026-03-11 | [#36791](https://github.com/vllm-project/vllm/pull/36791) |
| 125 | vllm-ascend | PR | [310P][Bugfix]: fix ngram graph replay accuracy error | 🔴 高 | closed | 2026-03-10 | [#7134](https://github.com/vllm-project/vllm-ascend/pull/7134) |
| 126 | vllm-ascend | PR | [bugfix][LoRA] Fix the lora accuracy issue introduced by the upstream vLLM changed. | 🔴 高 | closed | 2026-03-03 | [#6958](https://github.com/vllm-project/vllm-ascend/pull/6958) |
| 127 | vllm | Issue | [Bug]: Qwen3-Next accuracy regression on AMD | 🔴 高 | closed | 2026-03-03 | [#35828](https://github.com/vllm-project/vllm/issues/35828) |
| 128 | vllm | Issue | [RFC]: `vllm bench eval` for Unified Accuracy + Performance Evaluation | 🔴 高 | closed | 2026-03-01 | [#35639](https://github.com/vllm-project/vllm/issues/35639) |
| 129 | vllm | Issue | [CI] DBO with DP+EP accuracy regression on GSM8K evaluation | 🔴 高 | closed | 2026-02-26 | [#35407](https://github.com/vllm-project/vllm/issues/35407) |
| 130 | vllm-ascend | PR | [main][bugfix] Fixed an accuracy problem of gdn layer in graph | 🔴 高 | closed | 2026-02-26 | [#6822](https://github.com/vllm-project/vllm-ascend/pull/6822) |
| 131 | vllm-ascend | PR | [Bugfix] Fix DeepseekV3.1 Accuracy issue | 🔴 高 | closed | 2026-02-25 | [#6805](https://github.com/vllm-project/vllm-ascend/pull/6805) |
| 132 | vllm | PR | [CI] Fix flaky spec decode accuracy tests  | 🔴 高 | closed | 2026-02-24 | [#35204](https://github.com/vllm-project/vllm/pull/35204) |
| 133 | vllm | Issue | [CI] Ngram speculative decoding accuracy below 66% threshold | 🔴 高 | closed | 2026-02-24 | [#35168](https://github.com/vllm-project/vllm/issues/35168) |
| 134 | vllm | Issue | [CI] EAGLE speculative decoding accuracy below 60% threshold | 🔴 高 | closed | 2026-02-24 | [#35167](https://github.com/vllm-project/vllm/issues/35167) |
| 135 | vllm | Issue | [CI Failure][ROCm]:  CrossLayer KV layout Distributed NixlConnector PD accuracy tests (4 GPUs) | 🔴 高 | closed | 2026-02-23 | [#35132](https://github.com/vllm-project/vllm/issues/35132) |
| 136 | vllm | PR | [Bugfix] Fix MTP accuracy for GLM-5 | 🔴 高 | closed | 2026-02-11 | [#34385](https://github.com/vllm-project/vllm/pull/34385) |
| 137 | vllm-ascend | Issue | [Misc]: Discussion on accuracy variance | 🔴 高 | closed | 2026-02-09 | [#6623](https://github.com/vllm-project/vllm-ascend/issues/6623) |
| 138 | vllm | PR | [BUGFIX] Fix accuracy bugs in Qwen3-Next MTP | 🔴 高 | closed | 2026-02-08 | [#34077](https://github.com/vllm-project/vllm/pull/34077) |
| 139 | vllm | Issue | [Bug]: The local deployment achieves about 30% higher accuracy compared to the server deployment. | 🔴 高 | closed | 2026-02-05 | [#33871](https://github.com/vllm-project/vllm/issues/33871) |
| 140 | vllm | Issue | [CI Failure]:  mi325_1: Multi-Modal Accuracy Eval (Small Models) | 🔴 高 | closed | 2026-02-02 | [#33596](https://github.com/vllm-project/vllm/issues/33596) |
| 141 | vllm-ascend | PR | [bugfix]Fix accuracy issue in PCP/DCP with speculative decoding | 🔴 高 | closed | 2026-02-02 | [#6491](https://github.com/vllm-project/vllm-ascend/pull/6491) |
| 142 | vllm | PR | [bugfix] Solve the accuracy issue of deepseek ocr2 | 🔴 高 | closed | 2026-01-30 | [#33389](https://github.com/vllm-project/vllm/pull/33389) |
| 143 | vllm-ascend | PR | [0.13.0][cherry-pick][Bugfix] Specify tensorflow version in accuracy test to avoid segmentation fault (#6292) | 🔴 高 | closed | 2026-01-30 | [#6401](https://github.com/vllm-project/vllm-ascend/pull/6401) |
| 144 | vllm-ascend | PR | [Bugfix] Specify tensorflow version in accuracy test to avoid segmentation fault | 🔴 高 | closed | 2026-01-27 | [#6292](https://github.com/vllm-project/vllm-ascend/pull/6292) |
| 145 | vllm | Issue | [Bug]: Whisper large-v3 accuracy degradation in vLLM 0.14.1 (134.56% WER) on L40S - works fine in 0.12.0 | 🔴 高 | closed | 2026-01-26 | [#33107](https://github.com/vllm-project/vllm/issues/33107) |
| 146 | vllm | Issue | [Bug]: Whisper accuracy issue with FA2+CG | 🔴 高 | closed | 2026-01-26 | [#33091](https://github.com/vllm-project/vllm/issues/33091) |
| 147 | vllm | PR | [Bugfix] (grpc): improve GetServerInfo response consistency and accuracy | 🔴 高 | closed | 2026-01-26 | [#33070](https://github.com/vllm-project/vllm/pull/33070) |
| 148 | vllm | PR | [ROCm][CI] Remove DS async eplb accuracy test from AMD CI | 🔴 高 | closed | 2026-01-20 | [#32717](https://github.com/vllm-project/vllm/pull/32717) |
| 149 | vllm | Issue | [RFC]: More robust model accuracy testing with configurable and tiered coverage | 🔴 高 | closed | 2026-01-19 | [#32613](https://github.com/vllm-project/vllm/issues/32613) |
| 150 | vllm | PR | [ROCm][Bugfix] Disable hip sampler to fix deepseek's accuracy issue on ROCm | 🔴 高 | closed | 2026-01-15 | [#32413](https://github.com/vllm-project/vllm/pull/32413) |
| 151 | vllm-ascend | Issue | [Bug]: A3设备qwen3-32B模型进行ceval精度测试运行一段时间报错服务崩溃 | 🔴 高 | closed | 2026-01-15 | [#5925](https://github.com/vllm-project/vllm-ascend/issues/5925) |
| 152 | vllm | PR | [ROCm][CI] Disable Async Scheduling For Qwen3-Next-80B-A3B-Instruct MTP Async EPLB Accuracy Test | 🔴 高 | closed | 2026-01-13 | [#32275](https://github.com/vllm-project/vllm/pull/32275) |
| 153 | vllm | Issue | [CI Failure]:  DP EP NixlConnector PD accuracy tests (Distributed) | 🔴 高 | closed | 2026-01-12 | [#32222](https://github.com/vllm-project/vllm/issues/32222) |
| 154 | vllm | Issue | [CI Failure]:  Qwen3-Next-80B-A3B-Instruct MTP Async EPLB Accuracy | 🔴 高 | closed | 2026-01-12 | [#32221](https://github.com/vllm-project/vllm/issues/32221) |
| 155 | vllm-ascend | Issue | [Bug]: `eagle3` with `sp` will cause an accuracy problem in drafter model | 🔴 高 | closed | 2026-01-12 | [#5825](https://github.com/vllm-project/vllm-ascend/issues/5825) |
| 156 | vllm-ascend | PR | [Bugfix] Fixed an accuracy problem of sp with eagle3 | 🔴 高 | closed | 2026-01-12 | [#5816](https://github.com/vllm-project/vllm-ascend/pull/5816) |
| 157 | vllm-ascend | PR | [0.13.0][cherry-pick][Bugfix] Fixed an accuracy problem of sp with eagle3 | 🔴 高 | closed | 2026-01-12 | [#5814](https://github.com/vllm-project/vllm-ascend/pull/5814) |
| 158 | vllm | PR | [ROCm][Bugfix] Fix AITER speculative decoding accuracy issue | 🔴 高 | closed | 2026-01-10 | [#32084](https://github.com/vllm-project/vllm/pull/32084) |
| 159 | vllm | Issue | [Usage]: Qwen3-VL-Embedding Accuracy | 🔴 高 | closed | 2026-01-10 | [#32069](https://github.com/vllm-project/vllm/issues/32069) |
| 160 | vllm | Issue | [Bug]: [lm_eval crashed] lm eval accuracy test crashed using VLLM MAIN branch but v0.14.0rc0 works | 🔴 高 | closed | 2026-01-09 | [#32017](https://github.com/vllm-project/vllm/issues/32017) |
| 161 | vllm-ascend | PR | [CI] Accuracy issue of qwen3-next-w8a8 nightly test fix. | 🔴 高 | closed | 2026-01-09 | [#5746](https://github.com/vllm-project/vllm-ascend/pull/5746) |
| 162 | vllm | PR | [ROCm][LoRA] Fix MoE accuracy regression by preserving float32 router weight scaling | 🔴 高 | closed | 2026-01-07 | [#31931](https://github.com/vllm-project/vllm/pull/31931) |
| 163 | vllm | Issue | [Bug]: DeepEP LL Accuracy Issue with DeepGEMM E8M0 on B200 | 🔴 高 | closed | 2026-01-06 | [#31844](https://github.com/vllm-project/vllm/issues/31844) |
| 164 | vllm | PR | [Bugfix][CI/Build] Fix failing pooling models test due to Triton kernel accuracy diff | 🔴 高 | closed | 2026-01-06 | [#31776](https://github.com/vllm-project/vllm/pull/31776) |
| 165 | vllm-ascend | Issue | [Bug]: `CI` may has accuracy problem. | 🔴 高 | closed | 2026-01-04 | [#5588](https://github.com/vllm-project/vllm-ascend/issues/5588) |
| 166 | vllm-ascend | Issue | [Bug]: The DeepSeek model may encounter accuracy issues or throw errors when speculative decoding is enabled with extremely short input sequences. | 🔴 高 | closed | 2026-01-04 | [#5583](https://github.com/vllm-project/vllm-ascend/issues/5583) |
| 167 | vllm | Issue | [Bug][ModelOpt]: FlashInfer CUTLASS MoE Accuracy Degraded (Llama4) | 🔴 高 | closed | 2026-01-01 | [#31609](https://github.com/vllm-project/vllm/issues/31609) |
| 168 | vllm | PR | [ROCm][CI] Fix language generation test accuracy by disabling HF flash_sdp and mem_efficient_sdp | 🔴 高 | closed | 2026-01-01 | [#31597](https://github.com/vllm-project/vllm/pull/31597) |
| 169 | vllm | Issue | [Bug]: Qwen3-VL-8B-Instruct has accuracy issue - Multi modal accuracy issue | 🔴 高 | closed | 2025-12-31 | [#31564](https://github.com/vllm-project/vllm/issues/31564) |
| 170 | vllm-ascend | Issue | [Bug]: Verl + vllm-ascend(v0.11.0) dp+ep+tp+server mode inference accuracy has decreased | 🔴 高 | closed | 2025-12-31 | [#5544](https://github.com/vllm-project/vllm-ascend/issues/5544) |
| 171 | vllm | PR | [ROCm][Bugfix] Fix accuracy issue on fmoe when `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` enabled | 🔴 高 | closed | 2025-12-30 | [#31523](https://github.com/vllm-project/vllm/pull/31523) |
| 172 | vllm-ascend | PR | [bugfix] Fix for Pooling Synchronization Accuracy Issue | 🔴 高 | closed | 2025-12-30 | [#5510](https://github.com/vllm-project/vllm-ascend/pull/5510) |
| 173 | vllm | PR | [CI]Test Group 'NixlConnector PD accuracy tests' is fixed | 🔴 高 | closed | 2025-12-28 | [#31460](https://github.com/vllm-project/vllm/pull/31460) |
| 174 | vllm-ascend | Issue | [Bug]: test_qwen3_next_distributed_mp_eager_mtp_similarity_tp4 fail because accuracy drop | 🔴 高 | closed | 2025-12-27 | [#5439](https://github.com/vllm-project/vllm-ascend/issues/5439) |
| 175 | vllm-ascend | PR | [1/N][CI] Refactor accuracy test | 🔴 高 | closed | 2025-12-26 | [#5400](https://github.com/vllm-project/vllm-ascend/pull/5400) |
| 176 | vllm-ascend | PR | [Bugfix] Fix Qwen P/D Disaggregation accuracy issue | 🔴 高 | closed | 2025-12-25 | [#5340](https://github.com/vllm-project/vllm-ascend/pull/5340) |
| 177 | vllm | Issue | [CI Failure]:  mi325_4: DeepSeek V2-Lite Async EPLB Accuracy | 🔴 高 | closed | 2025-12-23 | [#31245](https://github.com/vllm-project/vllm/issues/31245) |
| 178 | vllm-ascend | Issue | [Bug]: Prefill and dcode nodes crash during accuracy testing | 🔴 高 | closed | 2025-12-23 | [#5276](https://github.com/vllm-project/vllm-ascend/issues/5276) |
| 179 | vllm-ascend | Issue | [Bug]: Phi-4-mini-instruct benchmark推理精度异常为0.08% | 🔴 高 | closed | 2025-12-20 | [#5215](https://github.com/vllm-project/vllm-ascend/issues/5215) |
| 180 | vllm-ascend | PR | [test]Corrected the Qwen3-Omni-30B-A3B-Instruct accuracy test configuration in nightly tests. | 🔴 高 | closed | 2025-12-19 | [#5195](https://github.com/vllm-project/vllm-ascend/pull/5195) |
| 181 | vllm | Issue | [CI Failure]:  mi325_4: DeepSeek V2-Lite Async EPLB Accuracy | 🔴 高 | closed | 2025-12-18 | [#30929](https://github.com/vllm-project/vllm/issues/30929) |
| 182 | vllm | Issue | [Bug]: whisper-large-v3-turbo have accuracy problem on nightly build | 🔴 高 | closed | 2025-12-16 | [#30777](https://github.com/vllm-project/vllm/issues/30777) |
| 183 | vllm-ascend | Issue | [Bug]: Launch model with lora, cannot access to the original model, and have accuracy issue | 🔴 高 | closed | 2025-12-15 | [#5031](https://github.com/vllm-project/vllm-ascend/issues/5031) |
| 184 | vllm | PR | [main][BugFix] Fixed an accuracy bug of Qwen3-next-MTP when batched inferring | 🔴 高 | closed | 2025-12-14 | [#30632](https://github.com/vllm-project/vllm/pull/30632) |
| 185 | vllm-ascend | PR | [bugfix] Fix mooncake kvpool accuracy issue | 🔴 高 | closed | 2025-12-12 | [#4976](https://github.com/vllm-project/vllm-ascend/pull/4976) |
| 186 | vllm-ascend | PR | [main][BugFix] Fixed an accuracy bug of Qwen3-next-MTP when batched inferring | 🔴 高 | closed | 2025-12-11 | [#4932](https://github.com/vllm-project/vllm-ascend/pull/4932) |
| 187 | vllm-ascend | PR | [Test]update accuracy test of models | 🔴 高 | closed | 2025-12-11 | [#4911](https://github.com/vllm-project/vllm-ascend/pull/4911) |
| 188 | vllm | PR | [NIXL][BUG FIX] Fix both failing issue and accuracy issue with nixl + host_buffer on CUDA | 🔴 高 | closed | 2025-12-10 | [#30419](https://github.com/vllm-project/vllm/pull/30419) |
| 189 | vllm-ascend | Issue | [Bug]: Qwen2.5-VL-32B-instruct使用lm_eval进行精度测试时报错 | 🔴 高 | closed | 2025-12-10 | [#4880](https://github.com/vllm-project/vllm-ascend/issues/4880) |
| 190 | vllm-ascend | PR | [Fix] fix llava-1.5-7b-hf & Qwen2-Audio-7B-Instruct accuracy test | 🔴 高 | closed | 2025-12-05 | [#4734](https://github.com/vllm-project/vllm-ascend/pull/4734) |
| 191 | vllm-ascend | PR | [CI] Refect accuracy test | 🔴 高 | closed | 2025-12-04 | [#4726](https://github.com/vllm-project/vllm-ascend/pull/4726) |
| 192 | vllm-ascend | Issue | [Bug]: gemma-2-9b-it and gemma-3-4b-it accuracy test failed | 🔴 高 | closed | 2025-12-04 | [#4713](https://github.com/vllm-project/vllm-ascend/issues/4713) |
| 193 | vllm | PR | [ROCm][CI][Bugfix] Disable Flash/MemEfficient SDP on ROCm to avoid HF Transformers accuracy issues | 🔴 高 | closed | 2025-12-02 | [#29909](https://github.com/vllm-project/vllm/pull/29909) |
| 194 | vllm-ascend | Issue | [Bug]: olmOCR-2-7B-1025: Poor Accuracy with vllm-ascend vs A100 vLLM | 🔴 高 | closed | 2025-12-02 | [#4618](https://github.com/vllm-project/vllm-ascend/issues/4618) |
| 195 | vllm-ascend | PR | [bugfix] Repair the problem of moe model accuracy caused by version upgrade. | 🔴 高 | closed | 2025-11-29 | [#4562](https://github.com/vllm-project/vllm-ascend/pull/4562) |
| 196 | vllm | Issue | [Bug]: Rotated samples extraction - accuracy loss | 🔴 高 | closed | 2025-11-28 | [#29645](https://github.com/vllm-project/vllm/issues/29645) |
| 197 | vllm-ascend | PR | [BugFix] Fix eagle3 accuracy problem when enforce_eager=True | 🔴 高 | closed | 2025-11-28 | [#4521](https://github.com/vllm-project/vllm-ascend/pull/4521) |
| 198 | vllm-ascend | Issue | [Bug]:   Accuracy issue on Qwen3-Omni-30B-A3B-Thinking | 🔴 高 | closed | 2025-11-27 | [#4513](https://github.com/vllm-project/vllm-ascend/issues/4513) |
| 199 | vllm-ascend | Issue | [Bug]: Accuracy issue for Prefill-Decode Disaggregation on Qwen2.5VL in 0.11.0rc2 | 🔴 高 | closed | 2025-11-27 | [#4497](https://github.com/vllm-project/vllm-ascend/issues/4497) |
| 200 | vllm-ascend | PR | Bugfix: Fix accuracy degradation caused by EPLB | 🔴 高 | closed | 2025-11-27 | [#4491](https://github.com/vllm-project/vllm-ascend/pull/4491) |
| 201 | vllm-ascend | PR | Bugfix: Fix accuracy degradation caused by EPLB | 🔴 高 | closed | 2025-11-27 | [#4490](https://github.com/vllm-project/vllm-ascend/pull/4490) |
| 202 | vllm | Issue | [CI Failure]: mi325_4: NixlConnector PD accuracy tests (Distributed) | 🔴 高 | closed | 2025-11-26 | [#29530](https://github.com/vllm-project/vllm/issues/29530) |
| 203 | vllm-ascend | Issue | [Bug]: Accuracy issue on Qwen3-30B-A3B | 🔴 高 | closed | 2025-11-26 | [#4461](https://github.com/vllm-project/vllm-ascend/issues/4461) |
| 204 | vllm-ascend | Issue | [Bug]: KV pool accuracy issues | 🔴 高 | closed | 2025-11-24 | [#4412](https://github.com/vllm-project/vllm-ascend/issues/4412) |
| 205 | vllm | PR | [Rocm][CI] Fix DeekSeek V2-Lite Accuracy CI | 🔴 高 | closed | 2025-11-21 | [#29135](https://github.com/vllm-project/vllm/pull/29135) |
| 206 | vllm | PR | [Bugfix] Fix precision loss in LoRA-wrapped RowParallelLinear by fusing bias into GEMM | 🔴 高 | closed | 2025-11-19 | [#28972](https://github.com/vllm-project/vllm/pull/28972) |
| 207 | vllm | Issue | [Bug]: bailing-moe accuracy problem when EP is enables | 🔴 高 | closed | 2025-11-17 | [#28862](https://github.com/vllm-project/vllm/issues/28862) |
| 208 | vllm | Issue | [Bug]: GDN model accuracy is low in DP mode | 🔴 高 | closed | 2025-11-14 | [#28704](https://github.com/vllm-project/vllm/issues/28704) |
| 209 | vllm | PR | [ROCm][Qwen3-32B] Fix AITER MHA accuracy issue cause by #25763 | 🔴 高 | closed | 2025-11-13 | [#28670](https://github.com/vllm-project/vllm/pull/28670) |
| 210 | vllm | Issue | [Bug]: NIXL `run_accuracy_test.sh` is broken for `block_size=128` | 🔴 高 | closed | 2025-11-13 | [#28661](https://github.com/vllm-project/vllm/issues/28661) |
| 211 | vllm | Issue | [Bug] [ROCm] [AITER]: AITER MHA accuracy degrades after PR #25763 | 🔴 高 | closed | 2025-11-12 | [#28598](https://github.com/vllm-project/vllm/issues/28598) |
| 212 | vllm | Issue | [Bug]: gemma3-27b-it shows degraded accuracy in vLLM v0.11.0 | 🔴 高 | closed | 2025-11-12 | [#28539](https://github.com/vllm-project/vllm/issues/28539) |
| 213 | vllm-ascend | PR | [Fixbug] Fix Qwen2-Audio-7B-Instruct accuracy test | 🔴 高 | closed | 2025-11-10 | [#4108](https://github.com/vllm-project/vllm-ascend/pull/4108) |
| 214 | vllm-ascend | Issue | [Bug]: Qwen/Qwen2-Audio-7B-Instruct accuracy test faild | 🔴 高 | closed | 2025-11-10 | [#4107](https://github.com/vllm-project/vllm-ascend/issues/4107) |
| 215 | vllm-ascend | PR | [BugFix] Fixes Qwen3-Next enable nz accuracy problem | 🔴 高 | closed | 2025-11-07 | [#4058](https://github.com/vllm-project/vllm-ascend/pull/4058) |
| 216 | vllm | PR | [Model] Fix bailing_moe accuracy problem | 🔴 高 | closed | 2025-11-07 | [#28277](https://github.com/vllm-project/vllm/pull/28277) |
| 217 | vllm-ascend | PR | [0.11.0] [Cherry-pick #4058] Fixes Qwen3-Next enable nz accuracy problem | 🔴 高 | closed | 2025-11-07 | [#4056](https://github.com/vllm-project/vllm-ascend/pull/4056) |
| 218 | vllm | Issue | [Bug]: Llama4 accuracy issue when torch.compile enabled on MI300x | 🔴 高 | closed | 2025-11-07 | [#28268](https://github.com/vllm-project/vllm/issues/28268) |
| 219 | vllm-ascend | PR | [long_seq] fix A2 accuracy problem | 🔴 高 | closed | 2025-11-06 | [#4030](https://github.com/vllm-project/vllm-ascend/pull/4030) |
| 220 | vllm-ascend | PR | [0.11.0][Fix] Fix Qwen2-Audio-7B-Instruct accuracy test | 🔴 高 | closed | 2025-11-06 | [#4018](https://github.com/vllm-project/vllm-ascend/pull/4018) |
| 221 | vllm-ascend | PR | [Fix]  fix Qwen2-Audio-7B-Instruct accuracy test | 🔴 高 | closed | 2025-11-06 | [#4017](https://github.com/vllm-project/vllm-ascend/pull/4017) |
| 222 | vllm-ascend | PR | [Test] Refactor accuracy test to nightly test | 🔴 高 | closed | 2025-10-28 | [#3814](https://github.com/vllm-project/vllm-ascend/pull/3814) |
| 223 | vllm | PR | [BugFix] Fix failing gemma-3-1b-it test: `test_lm_eval_accuracy_v1_engine[google/gemma-3-1b-it]` | 🔴 高 | closed | 2025-10-17 | [#27111](https://github.com/vllm-project/vllm/pull/27111) |
| 224 | vllm-ascend | PR | [BugFix]GPQA Accuracy Issue Bugfix | 🔴 高 | closed | 2025-10-15 | [#3476](https://github.com/vllm-project/vllm-ascend/pull/3476) |
| 225 | vllm-ascend | PR | GPQA Accuracy Issue Bugfix | 🔴 高 | closed | 2025-10-15 | [#3475](https://github.com/vllm-project/vllm-ascend/pull/3475) |
| 226 | vllm-ascend | Issue | [Bug]: The full graph mode has accuracy issues in dp scenario (qwen3-30b-a3b) | 🔴 高 | closed | 2025-10-14 | [#3444](https://github.com/vllm-project/vllm-ascend/issues/3444) |
| 227 | vllm-ascend | Issue | [Test] Enable accuracy tests for supported models | 🔴 高 | closed | 2025-10-13 | [#3401](https://github.com/vllm-project/vllm-ascend/issues/3401) |
| 228 | vllm-ascend | Issue | [Bug]: Qwen2 VL 7B accuracy test failed | 🔴 高 | closed | 2025-10-12 | [#3395](https://github.com/vllm-project/vllm-ascend/issues/3395) |
| 229 | vllm | Issue | [Bug]: ERNIE-4.5-21B-A3B-Thinking accuracy issue due to PR 23991 | 🔴 高 | closed | 2025-09-28 | [#25833](https://github.com/vllm-project/vllm/issues/25833) |
| 230 | vllm | PR | [XPU] Fix MOE DP accuracy issue on XPU | 🔴 高 | closed | 2025-09-23 | [#25465](https://github.com/vllm-project/vllm/pull/25465) |
| 231 | vllm | PR | [Speculators][Speculative Decoding] Fix gpt-oss eagle3 accuracy issue | 🔴 高 | closed | 2025-09-22 | [#25406](https://github.com/vllm-project/vllm/pull/25406) |
| 232 | vllm-ascend | PR | [Test] Update the format of the accuracy report | 🔴 高 | closed | 2025-09-22 | [#3081](https://github.com/vllm-project/vllm-ascend/pull/3081) |
| 233 | vllm | PR | [Docs] GSM8K Accuracy Evaluation doc update | 🔴 高 | closed | 2025-09-22 | [#25360](https://github.com/vllm-project/vllm/pull/25360) |
| 234 | vllm | Issue | [Bug]: Accuracy Discrepancy in Qwen 4B Embeddings: vLLM vs. Transformers | 🔴 高 | closed | 2025-09-21 | [#25333](https://github.com/vllm-project/vllm/issues/25333) |
| 235 | vllm-ascend | PR | [TEST] Speed up DS V2 accuracy test and turn up accuracy baseline | 🔴 高 | closed | 2025-09-19 | [#3047](https://github.com/vllm-project/vllm-ascend/pull/3047) |
| 236 | vllm-ascend | PR | [Fixbug] Fix accuracy for DeepSeek-V2-Lite | 🔴 高 | closed | 2025-09-18 | [#3016](https://github.com/vllm-project/vllm-ascend/pull/3016) |
| 237 | vllm-ascend | PR | [Bugfix] fix kv nz accuracy bug | 🔴 高 | closed | 2025-09-17 | [#2988](https://github.com/vllm-project/vllm-ascend/pull/2988) |
| 238 | vllm-ascend | Issue | [Bug][PR-2917]: DeepSeek-V2-Lite.yaml accuracy test OOM | 🔴 高 | closed | 2025-09-15 | [#2922](https://github.com/vllm-project/vllm-ascend/issues/2922) |
| 239 | vllm-ascend | PR | [BugFix] Fix glm 4.5 moe accuracy bug. | 🔴 高 | closed | 2025-09-12 | [#2898](https://github.com/vllm-project/vllm-ascend/pull/2898) |
| 240 | vllm-ascend | Issue | [Bug]: accuracy test failed due to `Forward context is not set` | 🔴 高 | closed | 2025-09-12 | [#2876](https://github.com/vllm-project/vllm-ascend/issues/2876) |
| 241 | vllm | Issue | [Bug]: R1 accuracy 0 issue when all 2 all kernel is "naive" | 🔴 高 | closed | 2025-09-09 | [#24530](https://github.com/vllm-project/vllm/issues/24530) |
| 242 | vllm-ascend | Issue | [Bug]: start failed due to _forward_decode_only shape error when test Phi-4-mini-instruct accuracy | 🔴 高 | closed | 2025-09-09 | [#2822](https://github.com/vllm-project/vllm-ascend/issues/2822) |
| 243 | vllm-ascend | Issue | [Bug]: GLM-4.5 Accuracy Issue with DP+EP | 🔴 高 | closed | 2025-09-05 | [#2767](https://github.com/vllm-project/vllm-ascend/issues/2767) |
| 244 | vllm | PR | [CI] Small Accuracy Eval Test for Deepseek Model | 🔴 高 | closed | 2025-09-04 | [#24259](https://github.com/vllm-project/vllm/pull/24259) |
| 245 | vllm-ascend | PR | [Doc] Update accuracy reports for v0.10.1rc1 | 🔴 高 | closed | 2025-09-04 | [#2755](https://github.com/vllm-project/vllm-ascend/pull/2755) |
| 246 | vllm-ascend | PR | [Bugfix][APC] Fix accuracy issue on prefix caching with AscendScheduler | 🔴 高 | closed | 2025-09-03 | [#2714](https://github.com/vllm-project/vllm-ascend/pull/2714) |
| 247 | vllm | PR | [Bug] R1 Accuracy: Fix `routed_scaling_factor` Double Mul Issue | 🔴 高 | closed | 2025-09-02 | [#24119](https://github.com/vllm-project/vllm/pull/24119) |
| 248 | vllm | Issue | [Bug]: R1 Accuracy Issue in Main for `deepep_high_througput` | 🔴 高 | closed | 2025-09-02 | [#24118](https://github.com/vllm-project/vllm/issues/24118) |
| 249 | vllm-ascend | PR | [Bugfix][LoRA][Operator] Fix LoRA custom operators accuracy issue | 🔴 高 | closed | 2025-09-01 | [#2672](https://github.com/vllm-project/vllm-ascend/pull/2672) |
| 250 | vllm | Issue | Accuracy Drop with OpenGVLab/InternVL3-14B when using vLLM | 🔴 高 | closed | 2025-08-30 | [#23988](https://github.com/vllm-project/vllm/issues/23988) |
| 251 | vllm-ascend | Issue | [Bug]: DeepSeek-V2-Lite flaky accuracy test failed | 🔴 高 | closed | 2025-08-29 | [#2634](https://github.com/vllm-project/vllm-ascend/issues/2634) |
| 252 | vllm-ascend | Issue | [Bug]: 双机A3 图模式tochair场景， v1 scheduler场景叠加MTP，使用aisbench测试ceval精度，并发512，出现服务卡死 | 🔴 高 | closed | 2025-08-28 | [#2604](https://github.com/vllm-project/vllm-ascend/issues/2604) |
| 253 | vllm-ascend | PR | [Bugfix] Fix long context seq accuracy problem for `GLM4.5` | 🔴 高 | closed | 2025-08-28 | [#2601](https://github.com/vllm-project/vllm-ascend/pull/2601) |
| 254 | vllm | Issue | [Bug]: Qwen3-Reranker + TP, with a significant loss in accuracy. | 🔴 高 | closed | 2025-08-28 | [#23804](https://github.com/vllm-project/vllm/issues/23804) |
| 255 | vllm-ascend | PR | [CI] Upgrade vllm in accuracy and performance CI | 🔴 高 | closed | 2025-08-25 | [#2527](https://github.com/vllm-project/vllm-ascend/pull/2527) |
| 256 | vllm-ascend | Issue | [Bug]: GLM4.5 accuracy problem on eager mode | 🔴 高 | closed | 2025-08-25 | [#2526](https://github.com/vllm-project/vllm-ascend/issues/2526) |
| 257 | vllm | Issue | [RFC] How could we prevent model like R1 E2E accuracy down to 0? | 🔴 高 | closed | 2025-08-21 | [#23354](https://github.com/vllm-project/vllm/issues/23354) |
| 258 | vllm | PR | [Bug] Fix R1 Accuracy 0 Bug | 🔴 高 | closed | 2025-08-20 | [#23294](https://github.com/vllm-project/vllm/pull/23294) |
| 259 | vllm | Issue | [Bug]: Accuracy issue for R1 DP8 on B200 | 🔴 高 | closed | 2025-08-20 | [#23282](https://github.com/vllm-project/vllm/issues/23282) |
| 260 | vllm | PR | [Bugfix] Fix accuracy issue when using flashinfer cutlass moe, TP=1 and modelopt. | 🔴 高 | closed | 2025-08-18 | [#23125](https://github.com/vllm-project/vllm/pull/23125) |
| 261 | vllm-ascend | PR | [Doc] Update accuracy reports for v0.10.0rc1 | 🔴 高 | closed | 2025-08-08 | [#2285](https://github.com/vllm-project/vllm-ascend/pull/2285) |
| 262 | vllm-ascend | PR | test on tp 4 accuracy | 🔴 高 | closed | 2025-08-08 | [#2282](https://github.com/vllm-project/vllm-ascend/pull/2282) |
| 263 | vllm-ascend | PR | Accuracy report formatting | 🔴 高 | closed | 2025-08-08 | [#2279](https://github.com/vllm-project/vllm-ascend/pull/2279) |
| 264 | vllm-ascend | PR | Fix accuracy test create PR | 🔴 高 | closed | 2025-08-08 | [#2274](https://github.com/vllm-project/vllm-ascend/pull/2274) |
| 265 | vllm-ascend | PR | Fix accuracy test create PR | 🔴 高 | closed | 2025-08-08 | [#2271](https://github.com/vllm-project/vllm-ascend/pull/2271) |
| 266 | vllm-ascend | PR | Accuracy report formatting | 🔴 高 | closed | 2025-08-07 | [#2251](https://github.com/vllm-project/vllm-ascend/pull/2251) |
| 267 | vllm | PR | [Bug] Fix B200 DeepGEMM E8M0 Accuracy Issue | 🔴 高 | closed | 2025-08-06 | [#22399](https://github.com/vllm-project/vllm/pull/22399) |
| 268 | vllm-ascend | PR | Fix accuracy test config --config-list-file | 🔴 高 | closed | 2025-08-01 | [#2163](https://github.com/vllm-project/vllm-ascend/pull/2163) |
| 269 | vllm-ascend | PR | [BugFix] Fix accuracy problem in dp situation. | 🔴 高 | closed | 2025-07-30 | [#2118](https://github.com/vllm-project/vllm-ascend/pull/2118) |
| 270 | vllm-ascend | PR | Auto pr/accuracy report 20250730102903 | 🔴 高 | closed | 2025-07-30 | [#2116](https://github.com/vllm-project/vllm-ascend/pull/2116) |
| 271 | vllm-ascend | PR | Enable pytest and yaml style accuracy test | 🔴 高 | closed | 2025-07-28 | [#2073](https://github.com/vllm-project/vllm-ascend/pull/2073) |
| 272 | vllm-ascend | PR | Refactor accuracy test CI | 🔴 高 | closed | 2025-07-28 | [#2061](https://github.com/vllm-project/vllm-ascend/pull/2061) |
| 273 | vllm-ascend | Issue | [Bug]: Qwen3-Coder-480B-A35B-Instruct poor accuracy | 🔴 高 | closed | 2025-07-28 | [#2053](https://github.com/vllm-project/vllm-ascend/issues/2053) |
| 274 | vllm-ascend | Issue | [Bug]: Llava-1.5-7b-hf bad accuracy with image input via 0.9.2rc1 under offline mod | 🔴 高 | closed | 2025-07-26 | [#2040](https://github.com/vllm-project/vllm-ascend/issues/2040) |
| 275 | vllm-ascend | PR | Refactor accuracy test CI | 🔴 高 | closed | 2025-07-26 | [#2029](https://github.com/vllm-project/vllm-ascend/pull/2029) |
| 276 | vllm-ascend | Issue | [Bug]: Flaky test failed: Qwen/Qwen3-30B-A3B accuracy failed under HDK 23.0.6 | 🔴 高 | closed | 2025-07-25 | [#1999](https://github.com/vllm-project/vllm-ascend/issues/1999) |
| 277 | vllm-ascend | Issue | [RFC]: Refactor accuracy test CI | 🔴 高 | closed | 2025-07-23 | [#1970](https://github.com/vllm-project/vllm-ascend/issues/1970) |
| 278 | vllm-ascend | PR | [MoE][Dist] Fix Qwen MoE accuracy bug in DP scenario | 🔴 高 | closed | 2025-07-17 | [#1856](https://github.com/vllm-project/vllm-ascend/pull/1856) |
| 279 | vllm-ascend | Issue | [Bug]: Qwen/Qwen3-30B-A3B accuracy low when tp=2 dp=2 | 🔴 高 | closed | 2025-07-14 | [#1791](https://github.com/vllm-project/vllm-ascend/issues/1791) |
| 280 | vllm-ascend | PR | [Doc] Update accuracy reports for main | 🔴 高 | closed | 2025-07-14 | [#1777](https://github.com/vllm-project/vllm-ascend/pull/1777) |
| 281 | vllm | PR | Renable google/gemma-3-1b-it accuracy test. | 🔴 高 | closed | 2025-07-13 | [#20866](https://github.com/vllm-project/vllm/pull/20866) |
| 282 | vllm-ascend | PR | [Doc] Update accuracy reports for main | 🔴 高 | closed | 2025-07-11 | [#1752](https://github.com/vllm-project/vllm-ascend/pull/1752) |
| 283 | vllm-ascend | PR | [Test] Remove VLLM_USE_V1 in accuracy test | 🔴 高 | closed | 2025-07-11 | [#1739](https://github.com/vllm-project/vllm-ascend/pull/1739) |
| 284 | vllm-ascend | PR | [Bugfix] Fix accuracy problem caused by mask pollution | 🔴 高 | closed | 2025-07-08 | [#1678](https://github.com/vllm-project/vllm-ascend/pull/1678) |
| 285 | vllm-ascend | Issue | [Bug]: In scenarios involving DP partitioning or combined DP+TP partitioning, executing other MOE models from clients may lead to accuracy issues, manifested as responses keep repeating. | 🔴 高 | closed | 2025-07-02 | [#1597](https://github.com/vllm-project/vllm-ascend/issues/1597) |
| 286 | vllm-ascend | PR | [Test] Remove V0 accuracy test and enable MoE and VL test on V1 | 🔴 高 | closed | 2025-07-02 | [#1574](https://github.com/vllm-project/vllm-ascend/pull/1574) |
| 287 | vllm-ascend | PR | [Doc] Update accuracy reports for v0.9.1-dev | 🔴 高 | closed | 2025-07-01 | [#1569](https://github.com/vllm-project/vllm-ascend/pull/1569) |
| 288 | vllm-ascend | PR | [BugFix] FIX MTP torch npu accuracy | 🔴 高 | closed | 2025-06-30 | [#1538](https://github.com/vllm-project/vllm-ascend/pull/1538) |
| 289 | vllm-ascend | PR | [Doc] Update accuracy reports for main | 🔴 高 | closed | 2025-06-28 | [#1507](https://github.com/vllm-project/vllm-ascend/pull/1507) |
| 290 | vllm-ascend | PR | [Doc] Update accuracy reports for main | 🔴 高 | closed | 2025-06-25 | [#1439](https://github.com/vllm-project/vllm-ascend/pull/1439) |
| 291 | vllm-ascend | PR | [FOLLOWUP] fix name and format in accuracy test (#1288) | 🔴 高 | closed | 2025-06-25 | [#1435](https://github.com/vllm-project/vllm-ascend/pull/1435) |
| 292 | vllm-ascend | PR | [Doc] Update accuracy reports for main | 🔴 高 | closed | 2025-06-25 | [#1429](https://github.com/vllm-project/vllm-ascend/pull/1429) |
| 293 | vllm | Issue | [Bug]: Accuracy Discrepancy in Qwen-7B-Chat Evaluation: vLLM Logprobs Differ from SGLANG, Impacting Performance | 🔴 高 | closed | 2025-06-25 | [#20054](https://github.com/vllm-project/vllm/issues/20054) |
| 294 | vllm-ascend | PR | [CI] Enable merge trigger unit test and accuracy test schedule job | 🔴 高 | closed | 2025-06-21 | [#1345](https://github.com/vllm-project/vllm-ascend/pull/1345) |
| 295 | vllm-ascend | PR | Accuracy test start | 🔴 高 | closed | 2025-06-21 | [#1344](https://github.com/vllm-project/vllm-ascend/pull/1344) |
| 296 | vllm-ascend | PR | Accuracy test  | 🔴 高 | closed | 2025-06-21 | [#1343](https://github.com/vllm-project/vllm-ascend/pull/1343) |
| 297 | vllm-ascend | PR | Accuracy regression test | 🔴 高 | closed | 2025-06-21 | [#1337](https://github.com/vllm-project/vllm-ascend/pull/1337) |
| 298 | vllm-ascend | PR | Accuracy regression test | 🔴 高 | closed | 2025-06-21 | [#1336](https://github.com/vllm-project/vllm-ascend/pull/1336) |
| 299 | vllm-ascend | PR | [v0.9.1][bugfix] fix accuracy prolem for deepseek V3/R1 models with torchair graph in long sequence predictions | 🔴 高 | closed | 2025-06-20 | [#1332](https://github.com/vllm-project/vllm-ascend/pull/1332) |
| 300 | vllm-ascend | PR | [bugfix] fix accuracy prolem for deepseek V3/R1 models with torchair graph in long sequence predictions | 🔴 高 | closed | 2025-06-20 | [#1331](https://github.com/vllm-project/vllm-ascend/pull/1331) |
| 301 | vllm-ascend | PR | [v0.9.1] [bugfix] fix accuracy problems for deepseek V3/R1 model with torchair graph in long sequence scenarios | 🔴 高 | closed | 2025-06-20 | [#1330](https://github.com/vllm-project/vllm-ascend/pull/1330) |
| 302 | vllm-ascend | PR | [0.9.1][BugFix]fix accuracy in dbo after refactor MOE | 🔴 高 | closed | 2025-06-20 | [#1328](https://github.com/vllm-project/vllm-ascend/pull/1328) |
| 303 | vllm-ascend | PR | [Doc] Update accuracy reports for main | 🔴 高 | closed | 2025-06-19 | [#1292](https://github.com/vllm-project/vllm-ascend/pull/1292) |
| 304 | vllm-ascend | PR | [CI]Update accuracy report test | 🔴 高 | closed | 2025-06-18 | [#1288](https://github.com/vllm-project/vllm-ascend/pull/1288) |
| 305 | vllm | Issue | [Bug]: Gemma3 reporting low image accuracy with v1 engine | 🔴 高 | closed | 2025-06-17 | [#19763](https://github.com/vllm-project/vllm/issues/19763) |
| 306 | vllm-ascend | PR | [Doc] Update accuracy reports for main | 🔴 高 | closed | 2025-06-15 | [#1225](https://github.com/vllm-project/vllm-ascend/pull/1225) |
| 307 | vllm | Issue | [Bug]: v1 engine accuracy issue | 🔴 高 | closed | 2025-06-11 | [#19472](https://github.com/vllm-project/vllm/issues/19472) |
| 308 | vllm-ascend | PR | [bugfix] fix deeepseek accuracy | 🔴 高 | closed | 2025-06-07 | [#1118](https://github.com/vllm-project/vllm-ascend/pull/1118) |
| 309 | vllm-ascend | PR | [WIP][bugfix] fix deeepseek accuracy | 🔴 高 | closed | 2025-06-06 | [#1108](https://github.com/vllm-project/vllm-ascend/pull/1108) |
| 310 | vllm | Issue | [Bug]: qwq32b-128k accuracy loss compare with sglang ， with proprietory business benchmark | 🔴 高 | closed | 2025-06-06 | [#19245](https://github.com/vllm-project/vllm/issues/19245) |
| 311 | vllm-ascend | PR | Try pass accuracy test for qwen2.5vl in vllm-ascend v1 | 🔴 高 | closed | 2025-06-05 | [#1082](https://github.com/vllm-project/vllm-ascend/pull/1082) |
| 312 | vllm-ascend | Issue | [Bug]: deepseek-v2-lite tp=8 ep=8 accuracy is not correct | 🔴 高 | closed | 2025-06-05 | [#1077](https://github.com/vllm-project/vllm-ascend/issues/1077) |
| 313 | vllm-ascend | Issue | [Bug][V1]: Failed to start Qwen/Qwen2.5-VL-7B-Instruct accuracy serve | 🔴 高 | closed | 2025-06-03 | [#1044](https://github.com/vllm-project/vllm-ascend/issues/1044) |
| 314 | vllm-ascend | Issue | [Bug][V1]: Qwen/Qwen2.5-7B-Instruct accuracy  ceval-valid failed | 🔴 高 | closed | 2025-06-03 | [#1043](https://github.com/vllm-project/vllm-ascend/issues/1043) |
| 315 | vllm-ascend | PR | Enable accuracy test for PR labeled with "*accuracy-test" | 🔴 高 | closed | 2025-05-31 | [#1040](https://github.com/vllm-project/vllm-ascend/pull/1040) |
| 316 | vllm-ascend | Issue | [Bug]: Qwen2.5-VL-7B-Instruct accuarcy test omm when tp=2 and accuracy decreased by 20% when tp=4 | 🔴 高 | closed | 2025-05-27 | [#968](https://github.com/vllm-project/vllm-ascend/issues/968) |
| 317 | vllm-ascend | PR | [Bugfix]Fix accuracy test | 🔴 高 | closed | 2025-05-26 | [#953](https://github.com/vllm-project/vllm-ascend/pull/953) |
| 318 | vllm | PR | [AMD] [P/D] Compute num gpus for ROCm correctly in run_accuracy_test.sh | 🔴 高 | closed | 2025-05-22 | [#18568](https://github.com/vllm-project/vllm/pull/18568) |
| 319 | vllm-ascend | Issue | [Bug]: tp4 DeepSeek-V2-Lite, accuracy is error，"text":"....................................................................................................." | 🔴 高 | closed | 2025-05-19 | [#894](https://github.com/vllm-project/vllm-ascend/issues/894) |
| 320 | vllm-ascend | PR | [accuracy test] Update cann version and huggingface-hub version for Qwen3 | 🔴 高 | closed | 2025-05-12 | [#823](https://github.com/vllm-project/vllm-ascend/pull/823) |
| 321 | vllm-ascend | PR | [bugfix] DP accuracy issue fix | 🔴 高 | closed | 2025-05-08 | [#791](https://github.com/vllm-project/vllm-ascend/pull/791) |
| 322 | vllm-ascend | Issue | [Accuracy]: vllm-ascend v0.7.3 release accuarcy report | 🔴 高 | closed | 2025-05-08 | [#790](https://github.com/vllm-project/vllm-ascend/issues/790) |
| 323 | vllm | Issue | [Bug]: gemma3 shows degraded accuracy in vLLM v0.8.4 | 🔴 高 | closed | 2025-05-06 | [#17689](https://github.com/vllm-project/vllm/issues/17689) |
| 324 | vllm-ascend | Issue | [Bug]: Accuracy issue after enabling engine v1 mode on v0.7.3 branch | 🔴 高 | closed | 2025-04-22 | [#620](https://github.com/vllm-project/vllm-ascend/issues/620) |
| 325 | vllm-ascend | Issue | [Doc]: Run accuracy test according "Using lm-eval" guide with modelscope source, errors take place: "has no revision: main !" and "Invalid repo_id: dataset, must be of format namespace/name"" | 🔴 高 | closed | 2025-04-22 | [#618](https://github.com/vllm-project/vllm-ascend/issues/618) |
| 326 | vllm | Issue | [Bug]: batch processing affects embedding accuracy | 🔴 高 | closed | 2025-04-18 | [#16815](https://github.com/vllm-project/vllm/issues/16815) |
| 327 | vllm | Issue | [Bug]: TP4 Accuracy Worse Than TP8 for LLaMa4 | 🔴 高 | closed | 2025-04-09 | [#16296](https://github.com/vllm-project/vllm/issues/16296) |
| 328 | vllm-ascend | PR | [Doc][Attn] Upgrade torch-npu to 0320 to fix accuracy issue | 🔴 高 | closed | 2025-03-27 | [#406](https://github.com/vllm-project/vllm-ascend/pull/406) |
| 329 | vllm | Issue | Precision loss occurs when using the MoE sum kernel. | 🔴 高 | closed | 2025-03-18 | [#15045](https://github.com/vllm-project/vllm/issues/15045) |
| 330 | vllm | PR | LLama 3.2 11b lm eval accuracy drop fix | 🔴 高 | closed | 2025-03-08 | [#14477](https://github.com/vllm-project/vllm/pull/14477) |
| 331 | vllm | PR | [Bugfix] DeepSeek Accuracy | 🔴 高 | closed | 2025-03-08 | [#14476](https://github.com/vllm-project/vllm/pull/14476) |
| 332 | vllm | Issue | [Bug]: The accuracy of multiple cards and single card is inconsistent | 🔴 高 | closed | 2025-02-25 | [#13801](https://github.com/vllm-project/vllm/issues/13801) |
| 333 | vllm | Issue | [Bug]: PixtralHF accuracy on MMMU regressed since 0.6.4.post1 | 🔴 高 | closed | 2025-01-07 | [#11816](https://github.com/vllm-project/vllm/issues/11816) |
| 334 | vllm | Issue | [Performance]: test speculative decode accuracy | 🔴 高 | closed | 2024-10-23 | [#9609](https://github.com/vllm-project/vllm/issues/9609) |
| 335 | vllm | Issue | [Bug]: accuracy degradation in llama 3.2 | 🔴 高 | closed | 2024-09-30 | [#8970](https://github.com/vllm-project/vllm/issues/8970) |
| 336 | vllm | Issue | [Usage]: accuracy degradation in llama 3.2 | 🔴 高 | closed | 2024-09-30 | [#8968](https://github.com/vllm-project/vllm/issues/8968) |
| 337 | vllm | Issue | [Bug]: v0.6.2 Shows a Significant Accuracy Drop Serving Qwen2-VL Model | 🔴 高 | closed | 2024-09-29 | [#8936](https://github.com/vllm-project/vllm/issues/8936) |
| 338 | vllm | Issue | [Bug]: The accuracy of vllm-Qwen2-VL-7B-Instruct is low. | 🔴 高 | closed | 2024-09-12 | [#8408](https://github.com/vllm-project/vllm/issues/8408) |
| 339 | vllm | Issue | [Bug]: Broken accuracy on LLaMa 3.1 70B -- worse than even 8B | 🔴 高 | closed | 2024-07-24 | [#6760](https://github.com/vllm-project/vllm/issues/6760) |
| 340 | vllm | PR | compressed-tensors accuracy testing | 🔴 高 | closed | 2024-06-21 | [#5750](https://github.com/vllm-project/vllm/pull/5750) |
| 341 | vllm | Issue | [Bug]: Token accuracy not as expected  ad also got Junk values for starcoderbase-15b model with continuous batching  | 🔴 高 | closed | 2024-04-30 | [#4499](https://github.com/vllm-project/vllm/issues/4499) |
| 342 | vllm | Issue | Qwen streaming inference leads to a significant decrease in accuracy | 🔴 高 | closed | 2024-02-21 | [#2954](https://github.com/vllm-project/vllm/issues/2954) |
| 343 | vllm | Issue | The answer accuracy of the QWen series model is lost | 🔴 高 | closed | 2024-02-21 | [#2952](https://github.com/vllm-project/vllm/issues/2952) |
| 344 | vllm | Issue |  Support vllm to use lm-eval to evaluate model accuracy | 🔴 高 | closed | 2023-08-17 | [#776](https://github.com/vllm-project/vllm/issues/776) |
| 345 | vllm | Issue | [Performance]: benchmark speed & precision of rust preproc vs pytorch | 🟡 中 | open | 2026-07-04 | [#47601](https://github.com/vllm-project/vllm/issues/47601) |
| 346 | vllm-ascend | PR | [CI][E2E] Add Qwen3-32B V1/V2 accuracy migration guards | 🟢 低 | open | 2026-08-13 | [#14200](https://github.com/vllm-project/vllm-ascend/pull/14200) |
| 347 | vllm-ascend | PR | [CI] Add weely performance and accuracy test cases for Qwen3.5-122B-A10B | 🟢 低 | closed | 2026-08-10 | [#13922](https://github.com/vllm-project/vllm-ascend/pull/13922) |
| 348 | vllm | PR | [XPU][Test] Support MultiConnector accuracy testing on XPU | 🟢 低 | closed | 2026-08-05 | [#51160](https://github.com/vllm-project/vllm/pull/51160) |
| 349 | vllm-ascend | PR | [CI] Add weely performance and accuracy test cases for Qwen3.5-122B-A10B | 🟢 低 | closed | 2026-08-03 | [#13424](https://github.com/vllm-project/vllm-ascend/pull/13424) |
| 350 | vllm-ascend | PR | [CI] Add weely performance and accuracy test cases for Kimi-K2.6-W4A8 and Kimi-K2.7-W4A8 | 🟢 低 | closed | 2026-07-31 | [#13285](https://github.com/vllm-project/vllm-ascend/pull/13285) |
| 351 | vllm-ascend | PR | [CI] Add weely performance and accuracy test cases for Qwen3.6-35B-A3B | 🟢 低 | closed | 2026-07-30 | [#13211](https://github.com/vllm-project/vllm-ascend/pull/13211) |
| 352 | vllm | PR | [CI][PD] Add hybrid SSM P_TP&gt;D_TP accuracy sweep entry | 🟢 低 | closed | 2026-07-23 | [#49593](https://github.com/vllm-project/vllm/pull/49593) |
| 353 | vllm | PR | [XPU] [UT] [CI] add xpu config to run gpt-oss accuracy in ut and ci | 🟢 低 | closed | 2026-07-15 | [#48703](https://github.com/vllm-project/vllm/pull/48703) |
| 354 | vllm-ascend | PR | [Doc][Model] Add AI21-Jamba-1.5-Mini tutorial and accuracy config | 🟢 低 | open | 2026-06-13 | [#10429](https://github.com/vllm-project/vllm-ascend/pull/10429) |
| 355 | vllm-ascend | PR | [Feature] Fix ealge accuracy with full graph in mrv2 | 🟢 低 | closed | 2026-06-12 | [#10374](https://github.com/vllm-project/vllm-ascend/pull/10374) |
| 356 | vllm | PR | [CI] Add opt-in statistically-calibrated lm-eval accuracy gate (Wilson lower bound) | 🟢 低 | open | 2026-06-06 | [#44704](https://github.com/vllm-project/vllm/pull/44704) |
| 357 | vllm-ascend | PR | [Test] Add DeepSeek-V4-Flash nightly e2e accuracy test | 🟢 低 | open | 2026-05-20 | [#9366](https://github.com/vllm-project/vllm-ascend/pull/9366) |
| 358 | vllm-ascend | PR | [Test] Add reward model accuracy evaluation test | 🟢 低 | closed | 2026-04-17 | [#8388](https://github.com/vllm-project/vllm-ascend/pull/8388) |
| 359 | vllm-ascend | PR | [Test] Add ASR model accuracy test | 🟢 低 | closed | 2026-04-17 | [#8362](https://github.com/vllm-project/vllm-ascend/pull/8362) |
| 360 | vllm-ascend | PR | [Test] Add ASR model accuracy test | 🟢 低 | closed | 2026-04-16 | [#8331](https://github.com/vllm-project/vllm-ascend/pull/8331) |
| 361 | vllm-ascend | PR | [CI]: add backup of accuracy report to shared volume | 🟢 低 | open | 2026-04-07 | [#8013](https://github.com/vllm-project/vllm-ascend/pull/8013) |
| 362 | vllm-ascend | PR | feat: add backup of accuracy report to shared volume | 🟢 低 | closed | 2026-04-07 | [#8012](https://github.com/vllm-project/vllm-ascend/pull/8012) |
| 363 | vllm-ascend | PR | [CI] Add PR-comment-only accuracy test group for A2 nightly workflow | 🟢 低 | closed | 2026-03-25 | [#7629](https://github.com/vllm-project/vllm-ascend/pull/7629) |
| 364 | vllm | PR | [Test][Nixl] Add YAML-driven test runner for PD accuracy configs | 🟢 低 | closed | 2026-03-14 | [#37069](https://github.com/vllm-project/vllm/pull/37069) |
| 365 | vllm | PR | [Docs] Add GSM8K accuracy benchmark example | 🟢 低 | open | 2026-03-10 | [#36591](https://github.com/vllm-project/vllm/pull/36591) |
| 366 | vllm | PR | [Feat] Add vllm eval CLI subcommand integrating lm_eval accuracy and perf benchmarking | 🟢 低 | closed | 2026-03-05 | [#36172](https://github.com/vllm-project/vllm/pull/36172) |
| 367 | vllm-ascend | PR | [Bugfix] fix dcp_only bug and add e2e accuracy test for dcp only and pcp only | 🟢 低 | closed | 2026-01-04 | [#5565](https://github.com/vllm-project/vllm-ascend/pull/5565) |
| 368 | vllm-ascend | PR | [test] add w4a8 accuracy case | 🟢 低 | closed | 2025-12-17 | [#5110](https://github.com/vllm-project/vllm-ascend/pull/5110) |
| 369 | vllm | PR | [ROCm][CI] Add "Qwen3-Next-80B-A3B-Instruct MTP Async EPLB Accuracy Test" Back Into AMD CI | 🟢 低 | closed | 2025-12-13 | [#30590](https://github.com/vllm-project/vllm/pull/30590) |
| 370 | vllm-ascend | PR | Add gsm8k accuracy test for multi-note Qwen3-235B-A22B | 🟢 低 | closed | 2025-12-08 | [#4802](https://github.com/vllm-project/vllm-ascend/pull/4802) |
| 371 | vllm-ascend | PR | [Test] Add accuracy nightly test for new models | 🟢 低 | closed | 2025-11-18 | [#4262](https://github.com/vllm-project/vllm-ascend/pull/4262) |
| 372 | vllm-ascend | PR | [Test]Add accuracy test for Phi-4-mini-instruct.yaml | 🟢 低 | closed | 2025-11-18 | [#4251](https://github.com/vllm-project/vllm-ascend/pull/4251) |
| 373 | vllm-ascend | PR | [Test][Accuracy] Add accuracy evaluation config for InternVL3_5-8B | 🟢 低 | closed | 2025-11-04 | [#3964](https://github.com/vllm-project/vllm-ascend/pull/3964) |
| 374 | vllm-ascend | PR | [Test][Accuracy] Add accuracy evaluation config for Qwen3-VL-8B-Instruct | 🟢 低 | closed | 2025-11-03 | [#3961](https://github.com/vllm-project/vllm-ascend/pull/3961) |
| 375 | vllm-ascend | PR | [Test]Add accuracy test for multiple models | 🟢 低 | closed | 2025-10-28 | [#3823](https://github.com/vllm-project/vllm-ascend/pull/3823) |
| 376 | vllm-ascend | PR | [Test] Add accuracy test for qwen3-30b-a3b-w8a8 | 🟢 低 | closed | 2025-10-28 | [#3807](https://github.com/vllm-project/vllm-ascend/pull/3807) |
| 377 | vllm-ascend | PR | [Test] Add accuracy test for qwen3-8b-w8a8 | 🟢 低 | closed | 2025-10-27 | [#3799](https://github.com/vllm-project/vllm-ascend/pull/3799) |
| 378 | vllm-ascend | PR | [Text]Add accuracy test for model Mistral-7B-Instruct-v0.1 | 🟢 低 | closed | 2025-10-25 | [#3742](https://github.com/vllm-project/vllm-ascend/pull/3742) |
| 379 | vllm-ascend | PR | [Test]Add accuracy test for model Phi-4-mini-instruct | 🟢 低 | closed | 2025-10-25 | [#3740](https://github.com/vllm-project/vllm-ascend/pull/3740) |
| 380 | vllm-ascend | PR | [Test]Add accuracy test for model MiniCPM3_4B | 🟢 低 | closed | 2025-10-25 | [#3739](https://github.com/vllm-project/vllm-ascend/pull/3739) |
| 381 | vllm-ascend | PR | [Test]Add accuracy test for model ERNIE-4.5-21B-A3B-PT | 🟢 低 | closed | 2025-10-23 | [#3658](https://github.com/vllm-project/vllm-ascend/pull/3658) |
| 382 | vllm-ascend | PR | [MM][CI] Add accuracy CI for `Qwen3-VL-8B-Instruct` | 🟢 低 | closed | 2025-10-21 | [#3580](https://github.com/vllm-project/vllm-ascend/pull/3580) |
| 383 | vllm-ascend | PR | [Test]Add accuracy test for model Meta-Llama-3.1-8B-Instruct | 🟢 低 | closed | 2025-10-21 | [#3575](https://github.com/vllm-project/vllm-ascend/pull/3575) |
| 384 | vllm-ascend | PR | [Test]Add accuracy test for model Qwen2.5-Omni-7B | 🟢 低 | closed | 2025-10-20 | [#3559](https://github.com/vllm-project/vllm-ascend/pull/3559) |
| 385 | vllm-ascend | PR | [Test]add accuracy test for model Qwen3-VL-8B-Instruction | 🟢 低 | closed | 2025-10-16 | [#3503](https://github.com/vllm-project/vllm-ascend/pull/3503) |
| 386 | vllm-ascend | PR | [Test] Add e2e test and accuracy test for Qwen3-Next-80B-A3B-Instruct | 🟢 低 | closed | 2025-10-14 | [#3450](https://github.com/vllm-project/vllm-ascend/pull/3450) |
| 387 | vllm-ascend | PR | add new accuracy test case for aclgraph | 🟢 低 | closed | 2025-10-11 | [#3390](https://github.com/vllm-project/vllm-ascend/pull/3390) |
| 388 | vllm-ascend | PR | [Test] Add accuracy test for Qwen3-VL-30B-A3B-Instruct | 🟢 低 | closed | 2025-10-10 | [#3362](https://github.com/vllm-project/vllm-ascend/pull/3362) |
| 389 | vllm-ascend | PR | [0.9.1][PromptLogprobs][V1] Support prompt logprobs to fix ceval accuracy in V1 | 🟢 低 | closed | 2025-08-30 | [#2654](https://github.com/vllm-project/vllm-ascend/pull/2654) |
| 390 | vllm | PR | [Misc] Support MMMU accuracy benchmark | 🟢 低 | closed | 2025-08-16 | [#23034](https://github.com/vllm-project/vllm/pull/23034) |
| 391 | vllm-ascend | PR | [CI] Add accuracy CI  | 🟢 低 | closed | 2025-08-12 | [#2330](https://github.com/vllm-project/vllm-ascend/pull/2330) |
| 392 | vllm-ascend | PR | Fix accuracy test config and add DeepSeek-V2-Lite test | 🟢 低 | closed | 2025-08-07 | [#2261](https://github.com/vllm-project/vllm-ascend/pull/2261) |
| 393 | vllm | PR | [CI/Build] Add Qwen2.5-VL-7B-Instruct ChartQA Accuracy Tests in CI | 🟢 低 | closed | 2025-07-29 | [#21810](https://github.com/vllm-project/vllm/pull/21810) |
| 394 | vllm-ascend | PR | [PromptLogprobs][V1] Support prompt logprobs to fix ceval accuracy in V1 | 🟢 低 | closed | 2025-06-27 | [#1483](https://github.com/vllm-project/vllm-ascend/pull/1483) |
| 395 | vllm-ascend | PR | [CI] Add accuracy ci for DP and EP and TP and ETP | 🟢 低 | closed | 2025-06-09 | [#1140](https://github.com/vllm-project/vllm-ascend/pull/1140) |
| 396 | vllm-ascend | PR | [v0.7.3][Doc] Add accuracy report | 🟢 低 | closed | 2025-05-08 | [#793](https://github.com/vllm-project/vllm-ascend/pull/793) |
| 397 | vllm-ascend | PR | [CI] Add accuracy test for Qwen2.5-VL-3B-Instruct | 🟢 低 | closed | 2025-05-06 | [#766](https://github.com/vllm-project/vllm-ascend/pull/766) |
| 398 | vllm | PR | [CI] Add mteb testing to test the accuracy of the embedding model | 🟢 低 | closed | 2025-04-25 | [#17175](https://github.com/vllm-project/vllm/pull/17175) |
| 399 | vllm-ascend | PR | [Test] Add accuracy test report workflow | 🟢 低 | closed | 2025-04-16 | [#542](https://github.com/vllm-project/vllm-ascend/pull/542) |
| 400 | vllm-ascend | PR | [CI]Add model basic accuracy test(Qwen2.5-0.5B-Instruct) | 🟢 低 | closed | 2025-04-02 | [#460](https://github.com/vllm-project/vllm-ascend/pull/460) |

</details>

---

## 7. Ascend NPU 特有精度

vLLM-Ascend 在 Ascend NPU 上独有的精度问题，源于 CANN 算子数值差异、ACL Graph 图优化改变计算路径、以及不同代际 NPU（910B / 950 等）硬件差异。

**问题数: 20 条**

### 关键问题

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm-ascend | Issue | [Bug]: Kimi-K2.7-Code-w4a8权重在昇腾A2服务器上配套vLLM Ascend 26.1.0.B092，curl出现乱码，GPQA和AIMEI2025精度测试，掉点严重，查看发现答题中也存在乱码情况 | 🔴 高 | closed | 2026-07-09 | [#11736](https://github.com/vllm-project/vllm-ascend/issues/11736) |
| 2 | vllm-ascend | PR | [BugFix] Fix Qwen3.5 precision error  with ACL Graph, MTP and DP | 🔴 高 | closed | 2026-06-24 | [#10901](https://github.com/vllm-project/vllm-ascend/pull/10901) |
| 3 | vllm-ascend | Issue | [Bug]: CANN9.0 GLM5.1-w4a8 精度问题 | 🔴 高 | closed | 2026-05-21 | [#9395](https://github.com/vllm-project/vllm-ascend/issues/9395) |
| 4 | vllm-ascend | Issue | [Usage]: DeepSeekV3.2模型，在A3机器TP16部署 ，vllm-ascend 0.13.0rc1出现精度问题 | 🔴 高 | open | 2026-03-13 | [#7225](https://github.com/vllm-project/vllm-ascend/issues/7225) |
| 5 | vllm-ascend | Issue | [Usage]: ：vllm ascend 0.14.0RC1 A2机器部署qwen3-rerank-4B精度问题 | 🔴 高 | closed | 2026-03-11 | [#7145](https://github.com/vllm-project/vllm-ascend/issues/7145) |
| 6 | vllm-ascend | Issue | [Bug]: quay.io/ascend/vllm-ascend:glm5-openeuler 部署GLM-OCR 精度劣化严重 | 🔴 高 | open | 2026-03-02 | [#6925](https://github.com/vllm-project/vllm-ascend/issues/6925) |
| 7 | vllm-ascend | Issue | [Bug]: vllm-ascend 0.14.0rc1-a3镜像使用tp1,dp2方式运行qwen3-30b-a3b存在精度问题 | 🔴 高 | open | 2026-02-11 | [#6671](https://github.com/vllm-project/vllm-ascend/issues/6671) |
| 8 | vllm-ascend | Issue | [Bug]: dotsOCR模型早vllm-ascend v0.13.0rc1 openeuler上做单机1P1D分离后精度下降 | 🔴 高 | closed | 2026-01-06 | [#5628](https://github.com/vllm-project/vllm-ascend/issues/5628) |
| 9 | vllm-ascend | Issue | [Bug]: Precision degradation with Qwen3-Next W8A8 on vLLM-Ascend v0.12.0rc1 | 🔴 高 | closed | 2025-12-16 | [#5065](https://github.com/vllm-project/vllm-ascend/issues/5065) |
| 10 | vllm-ascend | Issue | [Performance]: allenai/olmOCR-2-7B-1025模型使用vllm-ascend在910B离线推理，与使用vllm在A100上相比，精度差距过大 | 🔴 高 | closed | 2025-12-01 | [#4586](https://github.com/vllm-project/vllm-ascend/issues/4586) |
| 11 | vllm-ascend | Issue | [Bug]: vllm-ascend 部署 DeepSeek-V3.2-Exp-W8A8 出现精度问题 | 🔴 高 | closed | 2025-11-10 | [#4096](https://github.com/vllm-project/vllm-ascend/issues/4096) |
| 12 | vllm | Issue | [Bug]: SpeculativeConfig method="draft_model" cannot load mixed-precision compressed-tensors checkpoints (config_groups) | 🟡 中 | open | 2026-07-26 | [#49893](https://github.com/vllm-project/vllm/issues/49893) |
| 13 | vllm-ascend | PR | [BugFix]refresh Ascend MoE comm state after L2 wake to fix ep+sleep level2 precision issue | 🟡 中 | open | 2026-07-22 | [#12651](https://github.com/vllm-project/vllm-ascend/pull/12651) |
| 14 | vllm-ascend | Issue | [Bug]: The ascend operators in LoRA has precision problem | 🟡 中 | open | 2026-06-30 | [#11221](https://github.com/vllm-project/vllm-ascend/issues/11221) |
| 15 | vllm-ascend | PR | [BugFix][310P] Repair 310P Qwen3.5 aclgraph precision | 🟡 中 | closed | 2026-05-30 | [#9727](https://github.com/vllm-project/vllm-ascend/pull/9727) |
| 16 | vllm-ascend | PR | [Bugfix] Fix precision issues in moe_mlp (vllm-ascend main) | 🟡 中 | closed | 2025-12-15 | [#5025](https://github.com/vllm-project/vllm-ascend/pull/5025) |
| 17 | vllm-ascend | PR | [Bugfix]Fix precision issues in moe_mlp (vllm-ascend v0.11.0-dev) | 🟡 中 | closed | 2025-12-15 | [#5023](https://github.com/vllm-project/vllm-ascend/pull/5023) |
| 18 | vllm-ascend | Issue | [Bug]: vllm-ascend:v0.11.0rc2 use "async-scheduling flag" Feature encountered precision issues | 🟡 中 | closed | 2025-12-03 | [#4649](https://github.com/vllm-project/vllm-ascend/issues/4649) |
| 19 | vllm | Issue | [Bug]: I cannot able to load the model on TESLA T4 GPU in Full precision  | 🟡 中 | closed | 2024-11-04 | [#9990](https://github.com/vllm-project/vllm/issues/9990) |
| 20 | vllm | Issue | ValueError: The precision of the fractional quantity of resource node:172.16.95.108 cannot go beyond 0.0001 | 🟡 中 | closed | 2023-06-30 | [#317](https://github.com/vllm-project/vllm/issues/317) |

### 关键规律与分析

1. **CANN 算子数值差异**：同一算子（attention、scatter、reshape_and_cache）在 Ascend 与 CUDA 上数值路径不同，可能引入精度差异。
2. **ACL Graph 图优化**：算子融合、常量折叠改变计算路径，导致与 eager 模式精度不一致（常与 MTP/DP 组合放大）。
3. **NZ 格式 / cache mode / 硬件代际（910B vs 950）** 差异是 Ascend 平台特有的精度风险来源。

---

## 8. 其他 / 泛化精度

无法明确归入以上类别的精度相关问题。

**问题数: 171 条**

### 关键问题

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm-ascend | Issue | [Contribution] [Feature][MRV2] Qwen3 系列 ModelRunnerV2 精度与性能对比 v1 | 🔴 高 | open | 2026-08-13 | [#14206](https://github.com/vllm-project/vllm-ascend/issues/14206) |
| 2 | vllm-ascend | Issue | [Contribution] [Feature][MRV2] Step3.7 ModelRunnerV2 适配（精度性能与 MRV1 持平） | 🔴 高 | open | 2026-08-13 | [#14205](https://github.com/vllm-project/vllm-ascend/issues/14205) |
| 3 | vllm-ascend | Issue | [Bug]: Qwen3.5进行推理时，如果blocksize设置2048精度会有问题 | 🔴 高 | open | 2026-08-08 | [#13853](https://github.com/vllm-project/vllm-ascend/issues/13853) |
| 4 | vllm-ascend | Issue | [Test][MRV2] Kimi-K3 DSpark 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13745](https://github.com/vllm-project/vllm-ascend/issues/13745) |
| 5 | vllm-ascend | Issue | [Test][MRV2] qDSV4 DSpark 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13744](https://github.com/vllm-project/vllm-ascend/issues/13744) |
| 6 | vllm-ascend | Issue | [Test][MRV2] qDSV4 MTP 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13743](https://github.com/vllm-project/vllm-ascend/issues/13743) |
| 7 | vllm-ascend | Issue | [Test][MRV2] DeepSeek-3.1 MTP 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13742](https://github.com/vllm-project/vllm-ascend/issues/13742) |
| 8 | vllm-ascend | Issue | [Test][MRV2] Qwen3.5-35B-A3B DSpark 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13741](https://github.com/vllm-project/vllm-ascend/issues/13741) |
| 9 | vllm-ascend | Issue | [Test][MRV2] Qwen3.5-35B-A3B Eagle3 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13740](https://github.com/vllm-project/vllm-ascend/issues/13740) |
| 10 | vllm-ascend | Issue | [Test][MRV2] Qwen3.5-35B-A3B MTP 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13739](https://github.com/vllm-project/vllm-ascend/issues/13739) |
| 11 | vllm-ascend | Issue | [Test][MRV2] Qwen3.5-27B dense DSpark 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13738](https://github.com/vllm-project/vllm-ascend/issues/13738) |
| 12 | vllm-ascend | Issue | [Test][MRV2] Qwen3.5-27B dense Eagle3 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13737](https://github.com/vllm-project/vllm-ascend/issues/13737) |
| 13 | vllm-ascend | Issue | [Test][MRV2] Qwen3.5-27B dense MTP 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13736](https://github.com/vllm-project/vllm-ascend/issues/13736) |
| 14 | vllm-ascend | Issue | [Doc]: GLM-5.2使用精度问题 | 🔴 高 | open | 2026-08-05 | [#13650](https://github.com/vllm-project/vllm-ascend/issues/13650) |
| 15 | vllm-ascend | Issue | [Bug]: DeepSeek-V4-Flash-0731多轮对话频繁出现精度异常问题 | 🔴 高 | closed | 2026-08-04 | [#13441](https://github.com/vllm-project/vllm-ascend/issues/13441) |
| 16 | vllm-ascend | Issue | [Bug]: qwen2.5 7B 0.20.2rc1精度问题, pp+chunked prefill场景下首token不对 | 🔴 高 | closed | 2026-07-15 | [#12060](https://github.com/vllm-project/vllm-ascend/issues/12060) |
| 17 | vllm-ascend | PR | [Bugfix][310p]Fix crash bug on 310p and fix precision bug caused by q.copy() synchronize | 🔴 高 | closed | 2026-07-14 | [#11973](https://github.com/vllm-project/vllm-ascend/pull/11973) |
| 18 | vllm-ascend | PR | [Bugfix][310p]Fix crash bug on 310p and fix precision bug caused by q.copy() synchronize  | 🔴 高 | closed | 2026-07-13 | [#11938](https://github.com/vllm-project/vllm-ascend/pull/11938) |
| 19 | vllm-ascend | Issue | [Bug]: 在310P上使用v0.21.0rc1-310p官方镜像跑GLM-OCR、MinerU2.5-2509-1.2B精度有问题 | 🔴 高 | open | 2026-07-13 | [#11936](https://github.com/vllm-project/vllm-ascend/issues/11936) |
| 20 | vllm-ascend | Issue | [Community Test] Qwen3.5-27B 8卡部署、性能与精度基线验证 | 🔴 高 | open | 2026-07-09 | [#11771](https://github.com/vllm-project/vllm-ascend/issues/11771) |
| 21 | vllm-ascend | Issue | [Community Test] Qwen3.5-27B-W8A8 端到端功能与精度验证 | 🔴 高 | open | 2026-07-09 | [#11743](https://github.com/vllm-project/vllm-ascend/issues/11743) |
| 22 | vllm-ascend | Issue | [Bug]: 0.20版本、glm5.1 w8a8，A3 PD分离版本，开启kv_pool池化功能（经过实验，开不开启池化，都有精度问题），decode开启MTP（排除实验表明跟MTP相关，关闭MTP时没有精度问题），压测1-2小时候，decode节点的DP就逐渐出现MTP接受率不足1% 无法恢复，手工curl出现精度问题 | 🔴 高 | closed | 2026-06-29 | [#11127](https://github.com/vllm-project/vllm-ascend/issues/11127) |
| 23 | vllm-ascend | Issue | [Bug]: deepseekv4-flash在0.21.0.rc1上部分case精度与官网api相差较大 | 🔴 高 | closed | 2026-06-26 | [#11009](https://github.com/vllm-project/vllm-ascend/issues/11009) |
| 24 | vllm-ascend | Issue | [Bug]: v0.22.1rc1 Qwen3.X，PD分离+MTP存在精度问题 | 🔴 高 | open | 2026-06-25 | [#10961](https://github.com/vllm-project/vllm-ascend/issues/10961) |
| 25 | vllm-ascend | Issue | [Bug]: v0.18.0 GLM5.1模型对接Claude Code存在偶现精度问题 | 🔴 高 | closed | 2026-06-16 | [#10524](https://github.com/vllm-project/vllm-ascend/issues/10524) |
| 26 | vllm-ascend | Issue | [Doc]: Qwen3.5-27B模型开箱，精度评估章节“以下是两种精度评估方法”描述有误 | 🔴 高 | closed | 2026-06-12 | [#10389](https://github.com/vllm-project/vllm-ascend/issues/10389) |
| 27 | vllm-ascend | Issue | [Doc]: AISBench工具使用页面执行精度和性能评估缺少gms8K数据集评测的命令行 | 🔴 高 | closed | 2026-06-04 | [#10031](https://github.com/vllm-project/vllm-ascend/issues/10031) |
| 28 | vllm-ascend | Issue | [Bug]: Migstral-Small-2509模型在FULL_DECODE_ONLY模式下存在精度问题，decode首token不对 | 🔴 高 | closed | 2026-05-21 | [#9392](https://github.com/vllm-project/vllm-ascend/issues/9392) |
| 29 | vllm-ascend | Issue | [Bug]: QwQ 32B在v0.19.1RC1版本存在精度下降问题 | 🔴 高 | closed | 2026-05-19 | [#9324](https://github.com/vllm-project/vllm-ascend/issues/9324) |
| 30 | vllm-ascend | Issue | [Bug]: Deepseek-V4-flash 工具调用请求，含有历史对话且末尾角色为assistant，概率触发精度异常 | 🔴 高 | closed | 2026-05-19 | [#9277](https://github.com/vllm-project/vllm-ascend/issues/9277) |
| 31 | vllm-ascend | Issue | [Bug]: EPLB在新部署表有问题的情况下会有精度问题 | 🔴 高 | open | 2026-05-14 | [#9151](https://github.com/vllm-project/vllm-ascend/issues/9151) |
| 32 | vllm-ascend | Issue | [Bug]: DS-V4-PRO-W4A8，集成到cc中使用时，经常出现乱码，导致无法使用，不知道是否和前面提到的vllm服务在处理DeepSeek V4 模型的 Streaming Tool Call（流式工具调用） 时崩溃了相关还是模型精度有问题 | 🔴 高 | closed | 2026-05-13 | [#9122](https://github.com/vllm-project/vllm-ascend/issues/9122) |
| 33 | vllm-ascend | Issue | [Bug]: Mooncake transfer failed后自愈，概率导致后续请求精度异常（答非所问，重复等） | 🔴 高 | closed | 2026-04-20 | [#8427](https://github.com/vllm-project/vllm-ascend/issues/8427) |
| 34 | vllm-ascend | Issue | [Bug]: Qwen3.5-397B-A17B-w4a8权重的精度较原始的全量权重下降超过正常范围 | 🔴 高 | closed | 2026-04-13 | [#8179](https://github.com/vllm-project/vllm-ascend/issues/8179) |
| 35 | vllm-ascend | Issue | [Bug]: Qwen3-235B piece-wise图模式开启后，单server多轮精度评测出现oom | 🔴 高 | closed | 2026-03-30 | [#7824](https://github.com/vllm-project/vllm-ascend/issues/7824) |
| 36 | vllm-ascend | Issue | [Bug]: Qwen2.5 32B 图模型下aisbench测试精度报错 | 🔴 高 | closed | 2026-03-28 | [#7781](https://github.com/vllm-project/vllm-ascend/issues/7781) |
| 37 | vllm-ascend | Issue | [Bug]: 自研模型从0.11升级到0.13精度出现问题 | 🔴 高 | open | 2026-03-27 | [#7712](https://github.com/vllm-project/vllm-ascend/issues/7712) |
| 38 | vllm-ascend | Issue | [Bug]: GLM-5-w8a8 FlashComm1 + PD 分离模式下，P 节点首 token 概率性精度异常 | 🔴 高 | closed | 2026-03-26 | [#7684](https://github.com/vllm-project/vllm-ascend/issues/7684) |
| 39 | vllm-ascend | Issue | [Bug]: qwen3.5在suffix_decoding场景下出现精度问题 | 🔴 高 | closed | 2026-03-10 | [#7101](https://github.com/vllm-project/vllm-ascend/issues/7101) |
| 40 | vllm-ascend | Issue | [Bug]: qwen3.5 0day版本 请求中关闭思考模式存在精度异常 | 🔴 高 | open | 2026-03-10 | [#7094](https://github.com/vllm-project/vllm-ascend/issues/7094) |

### 关键规律与分析

兜底分类，收录无法明确归入以上 7 类的精度相关问题（标题含 precision/accuracy 但具体领域不明确）。

<details>
<summary>展开全部 171 条</summary>

| # | 仓库 | 类型 | 标题 | 严重度 | 状态 | 日期 | 链接 |
|---|------|------|------|:------:|------|------|------|
| 1 | vllm-ascend | Issue | [Contribution] [Feature][MRV2] Qwen3 系列 ModelRunnerV2 精度与性能对比 v1 | 🔴 高 | open | 2026-08-13 | [#14206](https://github.com/vllm-project/vllm-ascend/issues/14206) |
| 2 | vllm-ascend | Issue | [Contribution] [Feature][MRV2] Step3.7 ModelRunnerV2 适配（精度性能与 MRV1 持平） | 🔴 高 | open | 2026-08-13 | [#14205](https://github.com/vllm-project/vllm-ascend/issues/14205) |
| 3 | vllm-ascend | Issue | [Bug]: Qwen3.5进行推理时，如果blocksize设置2048精度会有问题 | 🔴 高 | open | 2026-08-08 | [#13853](https://github.com/vllm-project/vllm-ascend/issues/13853) |
| 4 | vllm-ascend | Issue | [Test][MRV2] Kimi-K3 DSpark 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13745](https://github.com/vllm-project/vllm-ascend/issues/13745) |
| 5 | vllm-ascend | Issue | [Test][MRV2] qDSV4 DSpark 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13744](https://github.com/vllm-project/vllm-ascend/issues/13744) |
| 6 | vllm-ascend | Issue | [Test][MRV2] qDSV4 MTP 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13743](https://github.com/vllm-project/vllm-ascend/issues/13743) |
| 7 | vllm-ascend | Issue | [Test][MRV2] DeepSeek-3.1 MTP 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13742](https://github.com/vllm-project/vllm-ascend/issues/13742) |
| 8 | vllm-ascend | Issue | [Test][MRV2] Qwen3.5-35B-A3B DSpark 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13741](https://github.com/vllm-project/vllm-ascend/issues/13741) |
| 9 | vllm-ascend | Issue | [Test][MRV2] Qwen3.5-35B-A3B Eagle3 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13740](https://github.com/vllm-project/vllm-ascend/issues/13740) |
| 10 | vllm-ascend | Issue | [Test][MRV2] Qwen3.5-35B-A3B MTP 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13739](https://github.com/vllm-project/vllm-ascend/issues/13739) |
| 11 | vllm-ascend | Issue | [Test][MRV2] Qwen3.5-27B dense DSpark 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13738](https://github.com/vllm-project/vllm-ascend/issues/13738) |
| 12 | vllm-ascend | Issue | [Test][MRV2] Qwen3.5-27B dense Eagle3 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13737](https://github.com/vllm-project/vllm-ascend/issues/13737) |
| 13 | vllm-ascend | Issue | [Test][MRV2] Qwen3.5-27B dense MTP 精度和性能对比 v2 vs v1 | 🔴 高 | open | 2026-08-07 | [#13736](https://github.com/vllm-project/vllm-ascend/issues/13736) |
| 14 | vllm-ascend | Issue | [Doc]: GLM-5.2使用精度问题 | 🔴 高 | open | 2026-08-05 | [#13650](https://github.com/vllm-project/vllm-ascend/issues/13650) |
| 15 | vllm-ascend | Issue | [Bug]: DeepSeek-V4-Flash-0731多轮对话频繁出现精度异常问题 | 🔴 高 | closed | 2026-08-04 | [#13441](https://github.com/vllm-project/vllm-ascend/issues/13441) |
| 16 | vllm-ascend | Issue | [Bug]: qwen2.5 7B 0.20.2rc1精度问题, pp+chunked prefill场景下首token不对 | 🔴 高 | closed | 2026-07-15 | [#12060](https://github.com/vllm-project/vllm-ascend/issues/12060) |
| 17 | vllm-ascend | PR | [Bugfix][310p]Fix crash bug on 310p and fix precision bug caused by q.copy() synchronize | 🔴 高 | closed | 2026-07-14 | [#11973](https://github.com/vllm-project/vllm-ascend/pull/11973) |
| 18 | vllm-ascend | PR | [Bugfix][310p]Fix crash bug on 310p and fix precision bug caused by q.copy() synchronize  | 🔴 高 | closed | 2026-07-13 | [#11938](https://github.com/vllm-project/vllm-ascend/pull/11938) |
| 19 | vllm-ascend | Issue | [Bug]: 在310P上使用v0.21.0rc1-310p官方镜像跑GLM-OCR、MinerU2.5-2509-1.2B精度有问题 | 🔴 高 | open | 2026-07-13 | [#11936](https://github.com/vllm-project/vllm-ascend/issues/11936) |
| 20 | vllm-ascend | Issue | [Community Test] Qwen3.5-27B 8卡部署、性能与精度基线验证 | 🔴 高 | open | 2026-07-09 | [#11771](https://github.com/vllm-project/vllm-ascend/issues/11771) |
| 21 | vllm-ascend | Issue | [Community Test] Qwen3.5-27B-W8A8 端到端功能与精度验证 | 🔴 高 | open | 2026-07-09 | [#11743](https://github.com/vllm-project/vllm-ascend/issues/11743) |
| 22 | vllm-ascend | Issue | [Bug]: 0.20版本、glm5.1 w8a8，A3 PD分离版本，开启kv_pool池化功能（经过实验，开不开启池化，都有精度问题），decode开启MTP（排除实验表明跟MTP相关，关闭MTP时没有精度问题），压测1-2小时候，decode节点的DP就逐渐出现MTP接受率不足1% 无法恢复，手工curl出现精度问题 | 🔴 高 | closed | 2026-06-29 | [#11127](https://github.com/vllm-project/vllm-ascend/issues/11127) |
| 23 | vllm-ascend | Issue | [Bug]: deepseekv4-flash在0.21.0.rc1上部分case精度与官网api相差较大 | 🔴 高 | closed | 2026-06-26 | [#11009](https://github.com/vllm-project/vllm-ascend/issues/11009) |
| 24 | vllm-ascend | Issue | [Bug]: v0.22.1rc1 Qwen3.X，PD分离+MTP存在精度问题 | 🔴 高 | open | 2026-06-25 | [#10961](https://github.com/vllm-project/vllm-ascend/issues/10961) |
| 25 | vllm-ascend | Issue | [Bug]: v0.18.0 GLM5.1模型对接Claude Code存在偶现精度问题 | 🔴 高 | closed | 2026-06-16 | [#10524](https://github.com/vllm-project/vllm-ascend/issues/10524) |
| 26 | vllm-ascend | Issue | [Doc]: Qwen3.5-27B模型开箱，精度评估章节“以下是两种精度评估方法”描述有误 | 🔴 高 | closed | 2026-06-12 | [#10389](https://github.com/vllm-project/vllm-ascend/issues/10389) |
| 27 | vllm-ascend | Issue | [Doc]: AISBench工具使用页面执行精度和性能评估缺少gms8K数据集评测的命令行 | 🔴 高 | closed | 2026-06-04 | [#10031](https://github.com/vllm-project/vllm-ascend/issues/10031) |
| 28 | vllm-ascend | Issue | [Bug]: Migstral-Small-2509模型在FULL_DECODE_ONLY模式下存在精度问题，decode首token不对 | 🔴 高 | closed | 2026-05-21 | [#9392](https://github.com/vllm-project/vllm-ascend/issues/9392) |
| 29 | vllm-ascend | Issue | [Bug]: QwQ 32B在v0.19.1RC1版本存在精度下降问题 | 🔴 高 | closed | 2026-05-19 | [#9324](https://github.com/vllm-project/vllm-ascend/issues/9324) |
| 30 | vllm-ascend | Issue | [Bug]: Deepseek-V4-flash 工具调用请求，含有历史对话且末尾角色为assistant，概率触发精度异常 | 🔴 高 | closed | 2026-05-19 | [#9277](https://github.com/vllm-project/vllm-ascend/issues/9277) |
| 31 | vllm-ascend | Issue | [Bug]: EPLB在新部署表有问题的情况下会有精度问题 | 🔴 高 | open | 2026-05-14 | [#9151](https://github.com/vllm-project/vllm-ascend/issues/9151) |
| 32 | vllm-ascend | Issue | [Bug]: DS-V4-PRO-W4A8，集成到cc中使用时，经常出现乱码，导致无法使用，不知道是否和前面提到的vllm服务在处理DeepSeek V4 模型的 Streaming Tool Call（流式工具调用） 时崩溃了相关还是模型精度有问题 | 🔴 高 | closed | 2026-05-13 | [#9122](https://github.com/vllm-project/vllm-ascend/issues/9122) |
| 33 | vllm-ascend | Issue | [Bug]: Mooncake transfer failed后自愈，概率导致后续请求精度异常（答非所问，重复等） | 🔴 高 | closed | 2026-04-20 | [#8427](https://github.com/vllm-project/vllm-ascend/issues/8427) |
| 34 | vllm-ascend | Issue | [Bug]: Qwen3.5-397B-A17B-w4a8权重的精度较原始的全量权重下降超过正常范围 | 🔴 高 | closed | 2026-04-13 | [#8179](https://github.com/vllm-project/vllm-ascend/issues/8179) |
| 35 | vllm-ascend | Issue | [Bug]: Qwen3-235B piece-wise图模式开启后，单server多轮精度评测出现oom | 🔴 高 | closed | 2026-03-30 | [#7824](https://github.com/vllm-project/vllm-ascend/issues/7824) |
| 36 | vllm-ascend | Issue | [Bug]: Qwen2.5 32B 图模型下aisbench测试精度报错 | 🔴 高 | closed | 2026-03-28 | [#7781](https://github.com/vllm-project/vllm-ascend/issues/7781) |
| 37 | vllm-ascend | Issue | [Bug]: 自研模型从0.11升级到0.13精度出现问题 | 🔴 高 | open | 2026-03-27 | [#7712](https://github.com/vllm-project/vllm-ascend/issues/7712) |
| 38 | vllm-ascend | Issue | [Bug]: GLM-5-w8a8 FlashComm1 + PD 分离模式下，P 节点首 token 概率性精度异常 | 🔴 高 | closed | 2026-03-26 | [#7684](https://github.com/vllm-project/vllm-ascend/issues/7684) |
| 39 | vllm-ascend | Issue | [Bug]: qwen3.5在suffix_decoding场景下出现精度问题 | 🔴 高 | closed | 2026-03-10 | [#7101](https://github.com/vllm-project/vllm-ascend/issues/7101) |
| 40 | vllm-ascend | Issue | [Bug]: qwen3.5 0day版本 请求中关闭思考模式存在精度异常 | 🔴 高 | open | 2026-03-10 | [#7094](https://github.com/vllm-project/vllm-ascend/issues/7094) |
| 41 | vllm-ascend | Issue | [Bug]:vllm0.13.0.rc1，打开full算子入图模式后精度异常 | 🔴 高 | closed | 2026-01-31 | [#6456](https://github.com/vllm-project/vllm-ascend/issues/6456) |
| 42 | vllm-ascend | Issue | [Bug]: v0.11.0版本部署Qwen3-235B使用Json Schema特性返回精度有问题 | 🔴 高 | closed | 2026-01-13 | [#5835](https://github.com/vllm-project/vllm-ascend/issues/5835) |
| 43 | vllm-ascend | Issue | [Bug]: [EPLB] 按照训练的强化学习框架的精度标准静态负载均衡可能存在精度问题 | 🔴 高 | open | 2026-01-06 | [#5643](https://github.com/vllm-project/vllm-ascend/issues/5643) |
| 44 | vllm-ascend | Issue | [Bug]: 910B部署后，跑一段时间会崩溃，模型精度也存在问题 | 🔴 高 | closed | 2025-12-25 | [#5339](https://github.com/vllm-project/vllm-ascend/issues/5339) |
| 45 | vllm-ascend | Issue | [Bug]: 昇腾910B部署模型后，测试时出现精度下降问题 | 🔴 高 | closed | 2025-12-17 | [#5125](https://github.com/vllm-project/vllm-ascend/issues/5125) |
| 46 | vllm-ascend | Issue | [Bug]: Qwen2.5 omni7b语音推理精度异常 | 🔴 高 | closed | 2025-12-12 | [#4962](https://github.com/vllm-project/vllm-ascend/issues/4962) |
| 47 | vllm-ascend | Issue | [Bug]: 开dp有精度问题 | 🔴 高 | open | 2025-12-10 | [#4877](https://github.com/vllm-project/vllm-ascend/issues/4877) |
| 48 | vllm-ascend | Issue | [Bug]: PD混布多DP场景下chunksize大于8K会存在精度问题 | 🔴 高 | closed | 2025-12-08 | [#4803](https://github.com/vllm-project/vllm-ascend/issues/4803) |
| 49 | vllm-ascend | Issue | [Bug]: v0.11.0rc2 qwen3-reranker-4b发现有精度问题 | 🔴 高 | closed | 2025-12-02 | [#4616](https://github.com/vllm-project/vllm-ascend/issues/4616) |
| 50 | vllm-ascend | Issue | [Performance]: qwen3-vl-32b qwen3-vl-30b-a3b性能精度不稳定 | 🔴 高 | closed | 2025-12-01 | [#4601](https://github.com/vllm-project/vllm-ascend/issues/4601) |
| 51 | vllm-ascend | Issue | [Bug]: deepseek v32-Exp模型精度相比论文掉了10个点 | 🔴 高 | closed | 2025-11-25 | [#4434](https://github.com/vllm-project/vllm-ascend/issues/4434) |
| 52 | vllm-ascend | Issue | [Bug]: B150镜像，Qwen3-235B模型，开启FULL_DECODE_ONLY，A2四机，2P2D，走mooncake非池化或者llmdatadist，精度异常 | 🔴 高 | closed | 2025-11-20 | [#4290](https://github.com/vllm-project/vllm-ascend/issues/4290) |
| 53 | vllm-ascend | Issue | [Bug]: v0.11.0rc0部署Qwen3-Next-80B-A3B-Instruct在问到数字相关问题是精度较差 | 🔴 高 | closed | 2025-11-07 | [#4057](https://github.com/vllm-project/vllm-ascend/issues/4057) |
| 54 | vllm-ascend | Issue | [Usage]: Qwen3-Next-80B-A3B-Instruct使用aisbench测试精度较低 | 🔴 高 | closed | 2025-10-13 | [#3407](https://github.com/vllm-project/vllm-ascend/issues/3407) |
| 55 | vllm | Issue | [Bug]: Encountered AssertionError and precision issues when enabling MTP in deepseek v3.1 | 🔴 高 | closed | 2025-10-11 | [#26621](https://github.com/vllm-project/vllm/issues/26621) |
| 56 | vllm-ascend | Issue | [Bug]: medgemma-27b-it/gemma3长序列精度问题 | 🔴 高 | open | 2025-09-28 | [#3237](https://github.com/vllm-project/vllm-ascend/issues/3237) |
| 57 | vllm-ascend | Issue | [Bug]: v0.10.0部署GLM-4.5模型精度存在问题 | 🔴 高 | closed | 2025-08-26 | [#2537](https://github.com/vllm-project/vllm-ascend/issues/2537) |
| 58 | vllm-ascend | Issue | [Bug]: Qwen 2.5 VL 多卡ViT qkv部分权重为0导致精度异常 | 🔴 高 | closed | 2025-05-27 | [#976](https://github.com/vllm-project/vllm-ascend/issues/976) |
| 59 | vllm-ascend | Issue | [Bug]:  deepseek-v2-lite-w8a8 精度不对 | 🔴 高 | closed | 2025-05-16 | [#883](https://github.com/vllm-project/vllm-ascend/issues/883) |
| 60 | vllm-ascend | Issue | [Bug]: 单卡推理Deepseek-v2-lite精度异常 | 🔴 高 | closed | 2025-04-29 | [#720](https://github.com/vllm-project/vllm-ascend/issues/720) |
| 61 | vllm-ascend | Issue | [Bug]: v0.7.3rc1 版本Qwen2-Audio-7B-Instruct 精度有问题，出感叹号 | 🔴 高 | closed | 2025-03-14 | [#336](https://github.com/vllm-project/vllm-ascend/issues/336) |
| 62 | vllm-ascend | Issue | Qwen2.5-VL支持吗，似乎能跑起来，但精度有问题 | 🔴 高 | closed | 2025-02-20 | [#118](https://github.com/vllm-project/vllm-ascend/issues/118) |
| 63 | vllm | PR | [Bugfix] Fix Qwen3 XML numeric fallback warnings | 🟡 中 | open | 2026-08-14 | [#52387](https://github.com/vllm-project/vllm/pull/52387) |
| 64 | vllm-ascend | PR | [Refactor][BugFix][DSA][4/N] fix the precision issue of dsa_v1 under dspark | 🟡 中 | closed | 2026-08-13 | [#14248](https://github.com/vllm-project/vllm-ascend/pull/14248) |
| 65 | vllm-ascend | PR | [BugFix] Fix precision of qwen3-235b in piecewise mode | 🟡 中 | closed | 2026-08-12 | [#14081](https://github.com/vllm-project/vllm-ascend/pull/14081) |
| 66 | vllm-ascend | PR | [BugFix] Fix precision of qwen3-235b in piecewise mode | 🟡 中 | open | 2026-08-11 | [#14012](https://github.com/vllm-project/vllm-ascend/pull/14012) |
| 67 | vllm | PR | [ROCm] Use backend-default dot precision for ReplaySSM | 🟡 中 | closed | 2026-07-26 | [#49909](https://github.com/vllm-project/vllm/pull/49909) |
| 68 | vllm | PR | [ROCm][CI] Set "highest" matmul precision for reference hf_runner in `test_bert_for_masked_lm` | 🟡 中 | closed | 2026-07-15 | [#48784](https://github.com/vllm-project/vllm/pull/48784) |
| 69 | vllm | PR | [Bugfix][Gemma4] Fix ModelOpt mixed-precision MoE config mapping | 🟡 中 | closed | 2026-07-14 | [#48563](https://github.com/vllm-project/vllm/pull/48563) |
| 70 | vllm-ascend | PR | [Test] Added test cases for the precision and performance of qwen3 | 🟡 中 | closed | 2026-07-06 | [#11489](https://github.com/vllm-project/vllm-ascend/pull/11489) |
| 71 | vllm-ascend | PR | [Test] Added test cases for the precision and performance of qwen3 | 🟡 中 | closed | 2026-07-06 | [#11483](https://github.com/vllm-project/vllm-ascend/pull/11483) |
| 72 | vllm-ascend | PR | [Test] Added test cases for the precision and performance of qwen3 | 🟡 中 | closed | 2026-07-02 | [#11320](https://github.com/vllm-project/vllm-ascend/pull/11320) |
| 73 | vllm-ascend | PR | [Test] Added test cases for the precision and performance of qwen3 | 🟡 中 | closed | 2026-07-02 | [#11316](https://github.com/vllm-project/vllm-ascend/pull/11316) |
| 74 | vllm-ascend | PR | [CI] Remove the validation of the nightly precision test case limit | 🟡 中 | closed | 2026-06-22 | [#10759](https://github.com/vllm-project/vllm-ascend/pull/10759) |
| 75 | vllm | PR | [Bugfix] Make Kimi's tool parser accept numeric only tool call IDs | 🟡 中 | open | 2026-06-19 | [#46127](https://github.com/vllm-project/vllm/pull/46127) |
| 76 | vllm-ascend | PR | [CI]Precision Testing | 🟡 中 | closed | 2026-06-11 | [#10334](https://github.com/vllm-project/vllm-ascend/pull/10334) |
| 77 | vllm-ascend | PR | [BugFix]fix A3 DSV4 precision | 🟡 中 | closed | 2026-06-09 | [#10221](https://github.com/vllm-project/vllm-ascend/pull/10221) |
| 78 | vllm-ascend | Issue | [Bug]: DeepSeek-V4 service  using the v1/completions API  has a precision issue, the v1/chat/completions API is OK | 🟡 中 | closed | 2026-06-02 | [#9853](https://github.com/vllm-project/vllm-ascend/issues/9853) |
| 79 | vllm-ascend | PR | [BugFix][310P] Fix the precision of the causal_conv1d_v310 operator on 310P | 🟡 中 | closed | 2026-05-29 | [#9720](https://github.com/vllm-project/vllm-ascend/pull/9720) |
| 80 | vllm-ascend | PR | [BugFix] Fix precision anomaly of DeepSeek-V4 on A2 when FlashComm is enabled | 🟡 中 | closed | 2026-05-23 | [#9488](https://github.com/vllm-project/vllm-ascend/pull/9488) |
| 81 | vllm-ascend | PR | [BugFix] Fix precision anomaly of DeepSeek-V4 on A2 when FlashComm is enabled | 🟡 中 | closed | 2026-05-23 | [#9485](https://github.com/vllm-project/vllm-ascend/pull/9485) |
| 82 | vllm-ascend | PR | [BugFix] Fix precision anomaly of DeepSeek-V4 on A2 when FlashComm is enabled | 🟡 中 | closed | 2026-05-23 | [#9483](https://github.com/vllm-project/vllm-ascend/pull/9483) |
| 83 | vllm-ascend | PR | [BugFix] Fix precision anomaly of DeepSeek-V4 on A2 when FlashComm is enabled | 🟡 中 | closed | 2026-05-23 | [#9480](https://github.com/vllm-project/vllm-ascend/pull/9480) |
| 84 | vllm | PR | [BUG] Fix FP64 Gumbel precision coverage | 🟡 中 | closed | 2026-05-19 | [#43150](https://github.com/vllm-project/vllm/pull/43150) |
| 85 | vllm | PR | Fix/prompt embeds mtp precision | 🟡 中 | open | 2026-05-09 | [#42168](https://github.com/vllm-project/vllm/pull/42168) |
| 86 | vllm | PR | Fix EP precision for Qwen3 MoE shared expert under sequence-parallel MoE | 🟡 中 | closed | 2026-05-05 | [#41763](https://github.com/vllm-project/vllm/pull/41763) |
| 87 | vllm-ascend | Issue | [Doc]: The precision of Minimax-2.5 is inconsistent with the paper（2*A2） | 🟡 中 | open | 2026-04-13 | [#8175](https://github.com/vllm-project/vllm-ascend/issues/8175) |
| 88 | vllm | PR | [XPU] Fix all_reduce precision under torch.compile using functional collective | 🟡 中 | closed | 2026-04-10 | [#39507](https://github.com/vllm-project/vllm/pull/39507) |
| 89 | vllm-ascend | Issue | [Bug]: Qwen3.5-397B w8a8 hybrid deployment report OOM in high-concurrency precision test | 🟡 中 | closed | 2026-04-10 | [#8141](https://github.com/vllm-project/vllm-ascend/issues/8141) |
| 90 | vllm-ascend | PR | [BugFix] Fixed eagle's precision problem when num_spec &gt; 1 for model_runner_v2 | 🟡 中 | closed | 2026-04-08 | [#8033](https://github.com/vllm-project/vllm-ascend/pull/8033) |
| 91 | vllm | PR | [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next | 🟡 中 | closed | 2026-04-07 | [#39181](https://github.com/vllm-project/vllm/pull/39181) |
| 92 | vllm | PR | [Bugfix]Fix EP precision for Qwen3.5 | 🟡 中 | closed | 2026-04-02 | [#38795](https://github.com/vllm-project/vllm/pull/38795) |
| 93 | vllm-ascend | PR | [V0.18.0][EPLB][BugFix] Fix moe_load precision in allgather | 🟡 中 | closed | 2026-04-01 | [#7890](https://github.com/vllm-project/vllm-ascend/pull/7890) |
| 94 | vllm-ascend | PR | [EPLB][BugFix] Fix moe_load precision in allgather | 🟡 中 | closed | 2026-04-01 | [#7887](https://github.com/vllm-project/vllm-ascend/pull/7887) |
| 95 | vllm-ascend | Issue | [Bug]: Qwen3.5-397B w8a8 hybrid deployment returns garbled replies in high-concurrency precision test | 🟡 中 | closed | 2026-03-31 | [#7854](https://github.com/vllm-project/vllm-ascend/issues/7854) |
| 96 | vllm-ascend | Issue | [Bug]: Qwen3.5-397B w8a8 hybrid deployment reports errors related to gdn_attn during high-concurrency precision testing | 🟡 中 | closed | 2026-03-31 | [#7848](https://github.com/vllm-project/vllm-ascend/issues/7848) |
| 97 | vllm-ascend | PR | [v0.18.0][BugFix] Fix bug of precision when DSA-CP is enabled on GLM5  | 🟡 中 | closed | 2026-03-31 | [#7843](https://github.com/vllm-project/vllm-ascend/pull/7843) |
| 98 | vllm | PR | [CPU] Added faster exp routine for lower precision data types. | 🟡 中 | closed | 2026-03-25 | [#38112](https://github.com/vllm-project/vllm/pull/38112) |
| 99 | vllm-ascend | PR | [Bugfix]Fix deepseek 3.2 C8  precision by rotary tensor | 🟡 中 | closed | 2026-03-23 | [#7537](https://github.com/vllm-project/vllm-ascend/pull/7537) |
| 100 | vllm-ascend | PR | [MTP][Bugfix] Fix GLM5-W8A8 precision issues caused by rotary quant MTP weights | 🟡 中 | closed | 2026-03-11 | [#7139](https://github.com/vllm-project/vllm-ascend/pull/7139) |
| 101 | vllm | PR | [Bug][MoE] Fix TRTLLM EScoreBias Precision | 🟡 中 | closed | 2026-03-11 | [#36724](https://github.com/vllm-project/vllm/pull/36724) |
| 102 | vllm | PR | [Bugfix] Fixed modelopt mixed precision quant format loading | 🟡 中 | closed | 2026-03-07 | [#36312](https://github.com/vllm-project/vllm/pull/36312) |
| 103 | vllm | Issue | [Bug]: LoRA loading fails for modules with numeric indices (e.g., to_out.0 in Diffusion Transformers)` | 🟡 中 | closed | 2026-03-02 | [#35734](https://github.com/vllm-project/vllm/issues/35734) |
| 104 | vllm | PR | [Bugfix]: Fix LoRA loading failure for modules with numeric indices (e.g., to_out.0 in Diffusion Transformers) | 🟡 中 | closed | 2026-03-02 | [#35732](https://github.com/vllm-project/vllm/pull/35732) |
| 105 | vllm | Issue | [RFC] Change the directory layout from `scaled_mm/` and `mixed_precision/` to backend-first . | 🟡 中 | closed | 2026-02-06 | [#34001](https://github.com/vllm-project/vllm/issues/34001) |
| 106 | vllm-ascend | PR | Remove the default dual-stream implementation of random sampling due to potential precision issues, and enable enable_async_exponential by default. | 🟡 中 | closed | 2026-01-21 | [#6089](https://github.com/vllm-project/vllm-ascend/pull/6089) |
| 107 | vllm-ascend | PR | Remove the default dual-stream implementation of random sampling due to potential precision issues, and enable enable_async_exponential by default. | 🟡 中 | closed | 2026-01-21 | [#6084](https://github.com/vllm-project/vllm-ascend/pull/6084) |
| 108 | vllm-ascend | PR | [0.13.0][Bugfix]Fixed precision issues caused by pooled request pooling | 🟡 中 | closed | 2026-01-20 | [#6057](https://github.com/vllm-project/vllm-ascend/pull/6057) |
| 109 | vllm-ascend | PR | [Bugfix]Fixed precision issues caused by pooled request pooling | 🟡 中 | closed | 2026-01-20 | [#6049](https://github.com/vllm-project/vllm-ascend/pull/6049) |
| 110 | vllm | PR | [ROCm][CI] Fix plugin tests (2 GPUs) failures on ROCm and removing `VLLM_FLOAT32_MATMUL_PRECISION` from all ROCm tests | 🟡 中 | closed | 2026-01-06 | [#31829](https://github.com/vllm-project/vllm/pull/31829) |
| 111 | vllm | Issue | [Bug]: `VLLM_FLOAT32_MATMUL_PRECISION=tf32` does not set cublas tf32 matmul | 🟡 中 | closed | 2025-12-31 | [#31579](https://github.com/vllm-project/vllm/issues/31579) |
| 112 | vllm | Issue | [Bug]: DeepSeek on B300 reports `invalid numeric default value` error | 🟡 中 | closed | 2025-12-31 | [#31557](https://github.com/vllm-project/vllm/issues/31557) |
| 113 | vllm-ascend | PR | [Bugfix] fix the precision issues that may raise from the inter-layer reuse of the workspace in certain scenarios | 🟡 中 | closed | 2025-12-30 | [#5522](https://github.com/vllm-project/vllm-ascend/pull/5522) |
| 114 | vllm-ascend | PR | [Bugfix] fix the precision issues that may raise from the inter-layer reuse of the workspace in certain scenarios. | 🟡 中 | closed | 2025-12-29 | [#5464](https://github.com/vllm-project/vllm-ascend/pull/5464) |
| 115 | vllm | Issue | [Usage]: Question about the dummy run。It seems the dummy run use different precision? | 🟡 中 | closed | 2025-12-25 | [#31361](https://github.com/vllm-project/vllm/issues/31361) |
| 116 | vllm-ascend | Issue | [Bug]: Precision Issue with curl on Ultra-Short Sequences in DeepSeek-V3.2 | 🟡 中 | closed | 2025-12-25 | [#5370](https://github.com/vllm-project/vllm-ascend/issues/5370) |
| 117 | vllm | PR | [ROCm][CI] Set VLLM_FLOAT32_MATMUL_PRECISION="tf32" For terratorch Tests In AMD CI | 🟡 中 | closed | 2025-12-23 | [#31242](https://github.com/vllm-project/vllm/pull/31242) |
| 118 | vllm-ascend | PR | [Bugfix] Fix matmul allreduce precision issue by using original weight | 🟡 中 | closed | 2025-12-12 | [#4939](https://github.com/vllm-project/vllm-ascend/pull/4939) |
| 119 | vllm | PR | [Chore] Fix torch precision warning | 🟡 中 | closed | 2025-12-10 | [#30428](https://github.com/vllm-project/vllm/pull/30428) |
| 120 | vllm-ascend | PR | [Bugfix] Fix kvpool precision synchronization | 🟡 中 | closed | 2025-11-29 | [#4574](https://github.com/vllm-project/vllm-ascend/pull/4574) |
| 121 | vllm-ascend | Issue | [Bug]: Qwen3-235B occasionally occurs precision issues in PD separation scenarios | 🟡 中 | closed | 2025-11-26 | [#4445](https://github.com/vllm-project/vllm-ascend/issues/4445) |
| 122 | vllm-ascend | PR | Precision synchronization issue fix | 🟡 中 | closed | 2025-11-25 | [#4429](https://github.com/vllm-project/vllm-ascend/pull/4429) |
| 123 | vllm-ascend | PR | [0.11.0][Bugfix] Fix ngram precision issue and open e2e ngram test | 🟡 中 | closed | 2025-11-10 | [#4092](https://github.com/vllm-project/vllm-ascend/pull/4092) |
| 124 | vllm-ascend | PR | [main][Bugfix] Fix ngram precision issue and open e2e ngram test | 🟡 中 | closed | 2025-11-10 | [#4090](https://github.com/vllm-project/vllm-ascend/pull/4090) |
| 125 | vllm-ascend | PR | [Bugfix][main] Fix ngram precision issue and open e2e ngram test | 🟡 中 | closed | 2025-11-10 | [#4079](https://github.com/vllm-project/vllm-ascend/pull/4079) |
| 126 | vllm-ascend | PR | [Bugfix][main] Fix ngram precision issue and open e2e ngram test | 🟡 中 | closed | 2025-11-07 | [#4053](https://github.com/vllm-project/vllm-ascend/pull/4053) |
| 127 | vllm-ascend | Issue | [Bug]: precision problem in ngram spec decoding | 🟡 中 | closed | 2025-11-06 | [#4037](https://github.com/vllm-project/vllm-ascend/issues/4037) |
| 128 | vllm-ascend | PR | fix deepseek torchair precision | 🟡 中 | closed | 2025-10-22 | [#3635](https://github.com/vllm-project/vllm-ascend/pull/3635) |
| 129 | vllm-ascend | PR | [BugFix] fix deepseek torchair precision | 🟡 中 | closed | 2025-10-22 | [#3624](https://github.com/vllm-project/vllm-ascend/pull/3624) |
| 130 | vllm-ascend | Issue | [Bug]: DeepSeek R1 precision issue, send 1 token to server, get response containing irrelevant things | 🟡 中 | closed | 2025-08-20 | [#2455](https://github.com/vllm-project/vllm-ascend/issues/2455) |
| 131 | vllm-ascend | Issue | [Bug]: Deepseek-w8a8 has precision issues. | 🟡 中 | closed | 2025-08-18 | [#2413](https://github.com/vllm-project/vllm-ascend/issues/2413) |
| 132 | vllm | Issue | [Bug]: Numerics of Embedding Models | 🟡 中 | closed | 2025-08-13 | [#22862](https://github.com/vllm-project/vllm/issues/22862) |
| 133 | vllm | Issue | [CI Failure]:  Classification test failure for Qwen2.5-1.5B-apeach model in half precision | 🟡 中 | closed | 2025-07-21 | [#21277](https://github.com/vllm-project/vllm/issues/21277) |
| 134 | vllm | Issue | [Bug]: The mixed precision model lacks kernel image in the Blackwell architecture(version:0.9.2 + cu12.8 + RTX5060) | 🟡 中 | closed | 2025-07-08 | [#20605](https://github.com/vllm-project/vllm/issues/20605) |
| 135 | vllm | Issue | [Bug]: Qwen3 Rerank 模型的准确率存在问题 | 🟡 中 | closed | 2025-07-04 | [#20478](https://github.com/vllm-project/vllm/issues/20478) |
| 136 | vllm-ascend | Issue | [Bug]: Qwen3-30B-A3B Shows Precision Issues in DP2+TP2 Parallel Mode | 🟡 中 | closed | 2025-06-18 | [#1289](https://github.com/vllm-project/vllm-ascend/issues/1289) |
| 137 | vllm | Issue | [Bug]: pythonic tool call parsing does not handle negative numeric literals | 🟡 中 | closed | 2025-06-12 | [#19569](https://github.com/vllm-project/vllm/issues/19569) |
| 138 | vllm-ascend | Issue | [Bug]: precision issue: V0 engine + deepseekR1 model + double G8600 + dp2tp16 | 🟡 中 | closed | 2025-05-07 | [#785](https://github.com/vllm-project/vllm-ascend/issues/785) |
| 139 | vllm | Issue | [Bug]: Slight Embedding Precision Difference When Running bge-m3 in vLLM Compared to Original Model | 🟡 中 | closed | 2025-05-06 | [#17713](https://github.com/vllm-project/vllm/issues/17713) |
| 140 | vllm | Issue | [Bug]: [Precision issues] test_flash_attn.py::test_flash_attn_with_paged_kv | 🟡 中 | closed | 2025-05-03 | [#17610](https://github.com/vllm-project/vllm/issues/17610) |
| 141 | vllm | Issue | [Bug]: GLM4V model gets lower precision score on TextVQA since vLLM does not process model's position ids correctly. | 🟡 中 | closed | 2025-03-14 | [#14790](https://github.com/vllm-project/vllm/issues/14790) |
| 142 | vllm | PR | [Jamba] added explicit fp32 precision for scores and weights. | 🟡 中 | closed | 2025-02-17 | [#13432](https://github.com/vllm-project/vllm/pull/13432) |
| 143 | vllm | Issue | [Bug]: InternVL2-40B Inference Precision Problem | 🟡 中 | closed | 2024-12-24 | [#11454](https://github.com/vllm-project/vllm/issues/11454) |
| 144 | vllm | Issue | [Bug]: Triton assertion errors serving Llama-3.1-8b on 4xH100s in FP32 precision | 🟡 中 | closed | 2024-09-18 | [#8579](https://github.com/vllm-project/vllm/issues/8579) |
| 145 | vllm | PR | [Kernel] (1/N) Machete - Hopper Optimized Mixed Precision Linear Kernel  | 🟡 中 | closed | 2024-08-05 | [#7174](https://github.com/vllm-project/vllm/pull/7174) |
| 146 | vllm | PR | [Kernel] Increase precision of GPTQ/AWQ Marlin kernel | 🟡 中 | closed | 2024-07-25 | [#6795](https://github.com/vllm-project/vllm/pull/6795) |
| 147 | vllm | PR | [Bugfix] use float32 precision in samplers/test_logprobs.py for comparing with HF  | 🟡 中 | closed | 2024-07-13 | [#6409](https://github.com/vllm-project/vllm/pull/6409) |
| 148 | vllm | PR | [Bugfix] Fix precisions in Gemma 1 | 🟡 中 | closed | 2024-06-27 | [#5913](https://github.com/vllm-project/vllm/pull/5913) |
| 149 | vllm | Issue | [Bug]: determine_num_available_blocks is not precision | 🟡 中 | closed | 2024-04-26 | [#4382](https://github.com/vllm-project/vllm/issues/4382) |
| 150 | vllm | Issue | ValueError: The precision of the fractional quantity of resource | 🟡 中 | closed | 2023-06-21 | [#173](https://github.com/vllm-project/vllm/issues/173) |
| 151 | vllm-ascend | PR | [Ops][Feature] Enhance batch-invariant Triton ops and add comprehensive precision tests | 🟢 低 | open | 2026-08-15 | [#14331](https://github.com/vllm-project/vllm-ascend/pull/14331) |
| 152 | vllm-ascend | PR | [Ops][Feature] Enhance batch-invariant Triton ops and add comprehensive precision tests | 🟢 低 | closed | 2026-08-15 | [#14330](https://github.com/vllm-project/vllm-ascend/pull/14330) |
| 153 | vllm-ascend | PR | [WIP][Feat][Ops] mamba_ssm_dtype in CGDR support fp32 precision in A2/A3 | 🟢 低 | open | 2026-08-02 | [#13335](https://github.com/vllm-project/vllm-ascend/pull/13335) |
| 154 | vllm-ascend | PR | [Feature][WIP][Kernel] Kimi K3 DSpark three-operator precision investigation | 🟢 低 | closed | 2026-08-01 | [#13315](https://github.com/vllm-project/vllm-ascend/pull/13315) |
| 155 | vllm-ascend | PR | [CI] ADD function : Precision Testing: Error Collection & Analysis | 🟢 低 | closed | 2026-07-06 | [#11460](https://github.com/vllm-project/vllm-ascend/pull/11460) |
| 156 | vllm | PR | [CI] Add TP=4 requirement to `test_mixed_precision_model_accuracies` | 🟢 低 | closed | 2026-06-19 | [#46161](https://github.com/vllm-project/vllm/pull/46161) |
| 157 | vllm-ascend | PR | [BugFix][Ops] Support mixed precision rotary mul | 🟢 低 | closed | 2026-05-20 | [#9363](https://github.com/vllm-project/vllm-ascend/pull/9363) |
| 158 | vllm-ascend | PR | [Feature] Fixed another eagle's precision problem for model_runner_v2 | 🟢 低 | closed | 2026-04-14 | [#8230](https://github.com/vllm-project/vllm-ascend/pull/8230) |
| 159 | vllm | PR | [vLLM IR] Support vLLM IR on XPU Platform and fix precision issue under torch.compile | 🟢 低 | closed | 2026-04-09 | [#39399](https://github.com/vllm-project/vllm/pull/39399) |
| 160 | vllm-ascend | PR | [P/D]Recomputation supports direct use of token IDs to prevent precision anomalies caused by ChatTemplate concatenation. | 🟢 低 | closed | 2026-03-19 | [#7450](https://github.com/vllm-project/vllm-ascend/pull/7450) |
| 161 | vllm | PR | add mixed precision support for modelopt | 🟢 低 | closed | 2026-02-22 | [#35047](https://github.com/vllm-project/vllm/pull/35047) |
| 162 | vllm | PR | [Test] Add env var to disable reduced precision reduction for PyTorch… | 🟢 低 | closed | 2026-02-04 | [#33806](https://github.com/vllm-project/vllm/pull/33806) |
| 163 | vllm | PR | [Compile] Add env `VLLM_FLOAT32_MATMUL_PRECISION` to fix torch warning `TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled` | 🟢 低 | closed | 2025-12-02 | [#29897](https://github.com/vllm-project/vllm/pull/29897) |
| 164 | vllm-ascend | PR | [BugFix]Fix precision issue for LoRA feature | 🟢 低 | closed | 2025-11-12 | [#4141](https://github.com/vllm-project/vllm-ascend/pull/4141) |
| 165 | vllm-ascend | PR | [BugFix]This PR aims to fix the precision issue of the LoRA feature i… | 🟢 低 | closed | 2025-11-07 | [#4046](https://github.com/vllm-project/vllm-ascend/pull/4046) |
| 166 | vllm-ascend | PR | [Test]Add lmhead_tp for MTP multi-card precision testing | 🟢 低 | closed | 2025-11-04 | [#3975](https://github.com/vllm-project/vllm-ascend/pull/3975) |
| 167 | vllm | PR | Fix GPTQ Marlin MoE mixed precision support | 🟢 低 | closed | 2025-10-15 | [#26953](https://github.com/vllm-project/vllm/pull/26953) |
| 168 | vllm | PR | [Kernel] Add Conch backend for mixed-precision linear layer | 🟢 低 | closed | 2025-06-18 | [#19818](https://github.com/vllm-project/vllm/pull/19818) |
| 169 | vllm | PR | [Bugfix] Fix LLaVA-NeXT feature size precision error (for real) | 🟢 低 | closed | 2025-01-06 | [#11772](https://github.com/vllm-project/vllm/pull/11772) |
| 170 | vllm | PR | [Bugfix] Fix precision error in LLaVA-NeXT feature size calculation | 🟢 低 | closed | 2025-01-04 | [#11735](https://github.com/vllm-project/vllm/pull/11735) |
| 171 | vllm | PR | [Kernel] Support Microsoft Runtime Kernel Lib for our Low Precision Computation - BitBLAS | 🟢 低 | closed | 2024-07-01 | [#6036](https://github.com/vllm-project/vllm/pull/6036) |

</details>

---
