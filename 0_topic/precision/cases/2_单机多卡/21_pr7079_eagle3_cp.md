# 案例 21：EAGLE3 + CP（上下文并行）元数据切分用错变量，接受率下降与高并发出错

> **一句话定位**：EAGLE3 + CP 同时启用时，attention CP 层切分 `query_lens` 用了 `num_decode_tokens`（总 token 数）而非 `num_decodes`（请求条数），decode 元数据被污染；EAGLE 的 `aux_hidden_states` 又被多余切片 → acceptance 下降 + 高并发输出错误。
>
> **对象**：vllm-project/vllm-ascend [PR #7079](https://github.com/vllm-project/vllm-ascend/pull/7079)（[eagle][cp] fix eagle_cp enable bug2，merged 2026-03-10；官方 release notes 列为 "Fix Eagle speculative decoding with Context Parallel enabled. #6981 #7079"）

---

## 1. 问题描述

### 1.1 现象

- EAGLE3 投机解码 + CP（context parallel）长序列场景：
  - 投机解码 acceptance 下降（精度类劣化）；
  - 高并发下输出错误。

### 1.2 触发条件（必现矩阵）

| EAGLE3 | CP（长序列） | 是否触发 |
|:---:|:---:|:---:|
| ✗ | — | ✗ |
| ✓ | ✗ | ✗ |
| **✓** | **✓** | **✓ acceptance 下降 / 高并发出错** |

### 1.3 影响与严重度

- **严重度**：🟡 中高（长序列投机解码加速失效、高并发输出错误）。
- **隐蔽性**：高（需 CP + 投机解码组合 + 高并发）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#7079](https://github.com/vllm-project/vllm-ascend/pull/7079)（系列修复，前作 #6981） |
| 状态 | merged（2026-03-10） |
| vllm-ascend 版本 | main（2026-03-10） |
| 上游 vLLM | v0.16.0（main @4034c3d） |

---

## 3. 定位过程

未知（PR 仅写 "tests and ut"；属 #6981 系列后续修复）。

---

## 4. 解决方案

### 4.1 根因

- CP 与投机解码组合时，「decode **请求条数**」与「decode **token 总数**」两类元数据语义在切分时用错变量（`num_decode_tokens` vs `num_decodes`），污染 attention 元数据；
- EAGLE 需要完整的 `aux_hidden_states`，原实现多余的 `[:num_scheduled_tokens]` 切片把它截断。

### 4.2 修复内容（3 文件）

```python
# attention_cp.py
- query_lens 切分变量由 num_decode_tokens 改为 num_decodes
- 删除 batch_chunk_seq_mask 冗余的 torch.repeat_interleave
- _build_chunked_context_metadata 不再传 batch_chunk_seq_mask

# model_runner_v1.py
- aux_hidden_states 拼接去掉多余的 [:num_scheduled_tokens] 切片
```

### 4.3 验证

仓库 tests + UT。

---

## 5. 复现方法

### 5.1 最小复现模型

- EAGLE3 草稿 + 目标模型组合（如 Qwen3-30B-A3B + EAGLE3 草稿），CP 需多卡。

### 5.2 复现命令

```bash
vllm serve Qwen/Qwen3-30B-A3B --tensor-parallel-size 2 \
  --speculative-config '{"method":"eagle3","model":"<EAGLE3-DRAFT>","num_speculative_tokens":3}' \
  --max-model-len 32768 --enforce-eager
# 以仓库当前 CP 开启方式附加 context parallel；长序列高并发压测观察 acceptance 与输出
```

---

## 核心教训

并行（CP/TP）切分层里「请求条数」与「token 总数」这类同族变量极易混用——投机解码的元数据语义必须逐字段核对；EAGLE 的 aux hidden states 不可截断。
