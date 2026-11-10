# dsv41_docs — DeepSeek V4.1-Flash 在 Ascend NPU 上的实现文档集

本目录整理了 **DeepSeek-V4.1-Flash-1017** 在 `vllm-ascend` 中的实现细节，配套官方 checkpoint 的原始配置与参考实现，便于离线阅读与对照。

## 从哪里开始

| 你的目的 | 入口 |
|---------|------|
| 先看整体架构与文档索引 | **[0_deepseek_v41_arch.md](0_deepseek_v41_arch.md)** |
| 只看模块设计 | [3 注意力](3_deepseek_v41_attn.md) · [4 MoE](4_deepseek_v41_moe.md) · [5 KV 缓存](5_deepseek_v41_kv_cache.md) · [6 Engram](6_deepseek_v41_engram.md) |
| 只看推测解码 | [7 MTP/调度](7_deepseek_v41_mtp.md) · [8 DSpark](8_deepseek_v41_dspark.md) · [11 采样](11_deepseek_v41_sampling.md) |
| 只看 Ascend C 算子 | [1 `hc_pre`](1_npu_hc_pre_npu.md) · [2 `hc_post`](2_npu_hc_post_npu.md) |
| 查配置数值 | [0_DeepSeek-V4.1-Flash-1017_config.json](0_DeepSeek-V4.1-Flash-1017_config.json) |
| 查权重名 | [0_DeepSeek-V4.1-Flash-1017_model_weights.index.json](0_DeepSeek-V4.1-Flash-1017_model_weights.index.json)（48 个分片的 `weight_map`） |
| 对照训练期语义 | [reference/](reference/README.md)：官方最小推理实现（`model.py` 61 KB，含 ViT / DSA / Engram / MoE / mHC / DSpark 的可读版本） |

## 文档清单

| 文档 | 内容 |
|------|------|
| [0_deepseek_v41_arch.md](0_deepseek_v41_arch.md) | 整体架构、与 V4 的差异、40 层拓扑、数据流、术语表 |
| [1_npu_hc_pre_npu.md](1_npu_hc_pre_npu.md) | `hc_pre` 算子：从 Ascend C 零基础到逐行精读；含 V4.1 `npu_hc_pre_v3` 入口差异 |
| [2_npu_hc_post_npu.md](2_npu_hc_post_npu.md) | `hc_post` 算子：9 种 dtype 组合、Tiling、Kernel 逐行 |
| [3_deepseek_v41_attn.md](3_deepseek_v41_attn.md) | 源共享 DSA：MLA 投影、两档压缩、QLI+QSLI 两级索引、融合稀疏 MLA 算子、多流重叠 |
| [4_deepseek_v41_moe.md](4_deepseek_v41_moe.md) | MoE：sqrtsoftplus 路由、`noaux_tc` 偏置、共享专家融合、EPLB、FP8/FP4 量化落地 |
| [5_deepseek_v41_kv_cache.md](5_deepseek_v41_kv_cache.md) | 混合缓存：SWA 环、C1/C2 压缩、索引 K、折叠缓存、层元组与全局 block-ID 池、1M 容量估算 |
| [6_deepseek_v41_engram.md](6_deepseek_v41_engram.md) | Engram：n-gram 哈希、素数桶布局、头分片大表、门控公式、异步重叠、DP×PCP 交换 |
| [7_deepseek_v41_mtp.md](7_deepseek_v41_mtp.md) | 延迟 mHC 系数交接、草稿隐态捕获、`mtp.*` 权重映射、一次完整投机步 |
| [8_deepseek_v41_dspark.md](8_deepseek_v41_dspark.md) | DSpark 块草稿器：上下文 KV 预计算、块内非因果注意力、Markov/Confidence 头、权重改名 |
| [9_deepseek_v41_decoder_replay.md](9_deepseek_v41_decoder_replay.md) | Decoder 有界回放：启用条件、行集合、DP/PCP 协调、SWA 可见长度下界 |
| [10_deepseek_v41_multimodal.md](10_deepseek_v41_multimodal.md) | 多模态：ViT、3×3 Aligner、图像 span 拼装、VL 路由偏置 |
| [11_deepseek_v41_sampling.md](11_deepseek_v41_sampling.md) | 后处理与采样：logits/TP gather、MRv1/MRv2 采样、DSpark 接受链路 |

## 原始资料

| 文件 | 来源 |
|------|------|
| `0_DeepSeek-V4.1-Flash-1017_config.json` | 官方 `config.json`（本文档集所有数值的唯一来源） |
| `0_DeepSeek-V4.1-Flash-1017_README.md` | 官方模型卡（benchmark、推荐采样参数、prompt encoding 说明） |
| `0_DeepSeek-V4.1-Flash-1017_tokenizer_config.json` / `configuration.json` | 官方 tokenizer / ModelScope 配置 |
| `0_DeepSeek-V4.1-Flash-1017_model_weights.index.json` | 官方权重索引（7.4 MB） |
| `LICENSE` | MIT 许可 |
| `reference/` | 官方 `inference/`（最小推理实现、权重转换脚本、两个示例输入）与 `encoding/`（prompt 编码参考实现与测试） |

> 全部原始资料来自 ModelScope：`deepseek-ai/DeepSeek-V4.1-Flash-1017`。
> 文档中的 `Lxxx` 行号以本仓库 `vllm_ascend/models/deepseek_v41/` 当前源码为准；跨版本阅读时请先确认行号。
