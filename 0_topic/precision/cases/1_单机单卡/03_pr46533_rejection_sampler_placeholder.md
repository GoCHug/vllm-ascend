# 案例 03：拒绝采样接受占位符 draft token（-1）并输出为真实 token

> **一句话定位**：rejection sampler 按草稿 token id gather 概率，对占位 id=-1（结构化输出的无效 draft / decode 侧 padding）该读越界，非 greedy 路径下可能把 -1 当真实 token 输出。
>
> **对象**：vllm-project/vllm [PR #46533](https://github.com/vllm-project/vllm/pull/46533)（merged，改 V1/V2 两套 rejection sampler kernel）

---

## 1. 问题描述

### 1.1 现象

- 投机解码 + 结构化输出（grammar 约束拒绝部分 draft）或 decode 侧 padding 时，draft token id 中出现 **-1 占位符**；
- 非 greedy（temperature>0）路径下，采样 kernel 以 -1 为索引 gather 概率，读到越界数据，理论上可能把 -1 对应的错误 token 输出为真实 token；
- greedy 路径原本已拒绝 -1，故常规 greedy CI 不触发。

### 1.2 触发条件（必现矩阵）

| 投机解码 | 结构化输出/padding 产生 -1 | temperature>0 | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗ | — | — | ✗ |
| ✓ | ✗ | — | ✗（无占位符） |
| ✓ | ✓ | ✗（greedy） | ✗（greedy 已拒绝） |
| **✓** | **✓** | **✓** | **✓ 越界读/异常输出风险** |

### 1.3 影响与严重度

- **严重度**：🟡 中（低频但真实的正确性漏洞，输出非法 token）。
- **隐蔽性**：高（需要 spec decode × 语法约束/填充 × 非贪婪三要素）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm |
| 对象 | PR [#46533](https://github.com/vllm-project/vllm/pull/46533) |
| 状态 | merged（2026-06-23，改 V1 `rejection_sampler` 与 V2 `rejection_sampler_utils`） |
| 复现版本 | 2026-06-23 修复前的 vllm main |
| 修复版本 | 含 #46533 的 main |
| vllm-ascend 版本 | 未知（rejection sampler 为跨硬件逻辑，可在 NPU 上构造对拍验证） |

---

## 3. 定位过程

代码审查发现：概率 gather 以 draft token id 为索引，-1 哨兵值未被掩蔽（greedy 分支有拒绝逻辑，采样分支没有）。

---

## 4. 解决方案

### 4.1 根因

采样 kernel 中「以外部可控 id 为索引」的 gather 操作没有对哨兵值（-1）做防护：非 greedy 路径按 -1 索引读到越界概率。

### 4.2 修复内容

- 对负 draft id **一律拒绝**（不进入接受判定）；
- 掩蔽概率 gather，使其永不被非法 id 索引（V1/V2 两套实现同步修复）。

### 4.3 验证

构造含 -1 的 draft 张量对拍修复前后行为（确定性单测）。

---

## 5. 复现方法

### 5.1 最小复现模型

- **`Qwen2.5-1.5B-Instruct`** + eagle/ngram 任一投机方式：小模型即可走通「投机解码 + 语法约束 + 采样」链路。

### 5.2 复现命令

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct",
          speculative_config={"method": "ngram", "num_speculative_tokens": 3})
# structured output 让 grammar 拒绝部分 draft + temperature>0
out = llm.generate(["给出一个合法 JSON 对象"], SamplingParams(temperature=0.8,
    guided_decoding={"json": {}}))
# 观察是否出现异常 token / 概率越界读
```

> 确定性路径：直接构造 draft_tokens 含 -1 的张量，调用 rejection sampler 对拍（单测形态，无需大模型）。

---

## 核心教训

采样 kernel 中「以外部可控 id 为索引」必须掩蔽哨兵值——否则哨兵值会穿透成输出；spec decode 的正确性用例必须覆盖「非 greedy + 无效 draft」组合。
