# PR #6469 端到端复现与修复实验（gemma-3-4b-it · vllm-ascend v0.14.0rc1）

> 实验时间：2026-09-16 | 环境：itask pod hw_5（A2 4 卡）
> 结果：**bug 版 curl 乱码 → 应用修复 → curl 正常**，端到端闭环完整达成
> 日志/脚本产物：本目录下 9 个文件（见 §8 清单）

---

## 1. 实验背景与目标

- **对象 PR**：vllm-ascend [#6469](https://github.com/vllm-project/vllm-ascend/pull/6469)（合并 commit `8e66299bf`，2026-02-05 合入 main）——修复 `_forward_fia_slidingwindow` 的 out 参数失效。
- **业务实证**：Ling-1T（原生 SWA 模型）生产 curl 乱码，应用 #6469 修复后乱码消失。
- **本次目标**：换一个主流模型（**google/gemma-3-4b-it**）在干净环境上完整复现「curl 有精度问题 → 修复 → 没问题」闭环，留档全部日志。
- **目标镜像**：v0.14.0rc1 —— **确认未 backport #6469**（该修复只进 main，release 分支拉取点早于 2026-02-05）。

## 2. 实验环境

| 项 | 值 |
|---|---|
| itask pod | hw_5（`itask create --image .../vllm:v0.14.0rc1-openeuler-20260915233247 --4card`，A2 4×hpu910） |
| vllm-ascend | **v0.14.0rc1**（源码方式位于 `/vllm-workspace/vllm-ascend`） |
| vllm | 0.14（`/vllm-workspace/vllm`），CANN 8.5.0，torch 2.9.0 |
| 模型 | **google/gemma-3-4b-it**（ModelScope 下载至 `/data/gemma-3-4b`，8.1G）——原生 5:1 SWA:full 交替（sliding_window=1024），**无任何架构篡改** |
| 服务命令 | `vllm serve /data/gemma-3-4b --enforce-eager --tensor-parallel-size 1 --max-model-len 4096 --port 8000 --gpu-memory-utilization 0.85` |
| 请求 | `POST /v1/chat/completions`，prompt "Which city is the capital of China?"，`temperature=0`，`max_completion_tokens=50` |

服务命令必须有 `--enforce-eager`：图模式 capture 走 `full_graph_fia` 旁路、replay 不执行 Python forward，均不触发 bug 路径；只有 eager 的 SWA decode 走 `_forward_fia_slidingwindow`（`attention_v1.py:766` 分发条件）。

## 3. Bug 代码与触发条件

v0.14.0rc1 的 `vllm_ascend/attention/attention_v1.py:716`（与 PR 修复前完全一致）：

```python
def _forward_fia_slidingwindow(self, query, attn_metadata, output):
    ...
    output, _ = torch_npu.npu_fused_infer_attention_score(...)   # ① 重新绑定局部变量
    output = output.view(batch_size, self.num_heads, self.head_size)  # ② 返回"新对象"
    return output                                                 # 传入的 output buffer 从未被写回
```

触发矩阵：

| 运行模式 | SWA 模型 | 是否触发 |
|:---:|:---:|:---|
| eager（`--enforce-eager`） | gemma-3 / gpt-oss / Ling / Mistral-v0.1 | **✅ 每个 decode step 的 SWA 层都执行 bug 路径** |
| FULL/ACLGraph 捕获 | 任意 | ❌（走 `full_graph_fia` 旁路） |
| 图 replay | 任意 | ❌（Python forward 不执行） |
| eager | 非 SWA（`sliding_window=None`） | ❌（走 PA/通用 FIA） |

## 4. 实验流程（从头到尾可复刻）

### 第一步 · bug 版启动服务

脚本 `serve_bug_v2.sh`（pod `/data/`）：清残留进程 → 启动 gemma-3-4b。服务端日志 → `server_bug.log`（本目录留档）。

```bash
pkill -9 -f 'v[l]m'; pkill -9 -f 'EngineC[o]re'; pkill -9 -f 'multi[p]rocessing'; sleep 5
vllm serve /data/gemma-3-4b --enforce-eager --tensor-parallel-size 1 \
  --max-model-len 4096 --port 8000 --gpu-memory-utilization 0.85
```

服务端日志关键行（`server_bug.log`）：
```
INFO:     Application startup complete.
INFO:     127.0.0.1:51666 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

### 第二步 · curl 请求 → 乱码（复现精度问题）

打屏留档：`curl_bug_screen.txt`：

```json
{"choices":[{"message":{"role":"assistant","content":"Theస్త సెчні𝜎सायिक रસ ikut Shame দিনেocks iti loുం Kazimrest verv drap пере eternalbill sp健 moons বুঝ trọngwx IUnaryologue measure compelling ခု г曇𝘭 agre lái برند itu सबैслу lowering çektiపోкона وهी अल banca ભ captions",...}}],
 "finish_reason":"length","usage":{"completion_tokens":50,...}}
```

**乱码特征**：多语言字符混杂（孟加拉/希腊/缅甸/阿拉伯/汉字），50 token 填满即截断（`finish_reason=length`）；与 Ling-1T 生产现象一致。

### 第三步 · 应用 #6469 修复

打补丁脚本 `fix_pr6469.py`（本目录留档，锚点唯一性断言 + 备份 + py_compile 校验），输出留档 `fix_patch.log`：

```
backup -> .../attention_v1.py.bak_pr6469
anchor count head = 1
anchor count tail = 1
PATCH_APPLIED
SYNTAX_OK
```

修复 diff（与上游 PR #6469 一致）：

```python
attn_output, _ = torch_npu.npu_fused_infer_attention_score(...)
attn_output = attn_output.view(batch_size, self.num_heads, self.head_size)
output[:batch_size] = attn_output[:batch_size]      # ← 关键：out 参数原地写回
return output
```

### 第四步 · 重启服务 → 同款 curl → 正常

服务端日志 → `server_fixed.log`（新进程 pid、startup complete、POST 200）。打屏留档 `curl_fixed_screen.txt`：

```json
{"choices":[{"message":{"content":"The capital of China is **Beijing**. \n\nIt’s a massive, historic city and the political, cultural, and educational center of the country. 😊",...}}],
 "finish_reason":"stop","usage":{"completion_tokens":33,...}}
```

## 5. 实验结果对照

| 阶段 | content 输出 | finish_reason | tokens |
|---|---|---|---|
| **bug 版**（v0.14.0rc1 原始代码） | `Theస్త సెчні𝜎सायिक रસ ikut Shame...`（多语言乱码） | `length`（50 填满） | 这些 |
| **修复版**（+ #6469，其余全同） | `The capital of China is **Beijing**. ...`（正常） | `stop`（自然结束） | 33 |

两次实验配置完全相同（同模型、同 seed 参数、temperature=0），唯一变量是 `attention_v1.py` 的 4 行补丁——**因果链干净，无噪声源**。

## 6. 根因分析

**直接根因（Python 语义层）**：`output, _ = npu_fused_infer_attention_score(...)` 是**局部变量重新绑定**（rebinding），不是 in-place 写回。函数签名的 out 参数契约（把计算结果写入 caller 提供的 buffer）被无声废弃。

**为什么在 v0.14.0rc1 上是端到端灾难（而 v0.13 时代 PR 自述 "No user-facing change"）**：
- vllm 0.14 的 `attention/layer.py:367` 引入了 out 参数执行链路：`output = torch.empty(...)` → `output.view(-1, heads, head_size_v)` → **丢弃 `impl.forward(...)` 的返回值** → `return output.view(-1, hidden_size)`。上层消费的正是传入的 output buffer 本身（vllms-ascend `AscendAttentionBackend.accept_output_buffer = True`，该链路默认激活）。
- bug 不写回 buffer → buffer 保持 `torch.empty` 的未初始化残留值 → 所有 SWA 层（gemma-3 为 5/6 层）decode 每 step 输入上层的是垃圾 → logits 漂移 → temperature=0 采样落到确定性高的随机多语言 token → 乱码。
- v0.13 时代上层用 return 值接力（值碰巧正确），故当时是潜伏 bug；vllm 0.14 out 契约落地后升级为端到端可见——这解释了 PR 作者的"No user-facing change"自述与 Ling-1T/gemma-3 实测乱码的"矛盾"。

**形状契约（复现实验的细节）**：修复写法 `output[:batch_size] = attn_output[:batch_size]` 基于 3D output `(batch_size, heads, head_size)`（layer.py 分配后 view 传入）。复现单测构造 2D 会报 shape mismatch（早期踩坑，反而实证了 3D 契约）。

**乱码为什么是"确定性"的**：未初始化 buffer 内容由 NPU memory allocator 决定性复用而非随机，故乱码文本跨进程逐字符稳定；它与真实乱码报告（Ling-1T）现象一致。

**旁证：单测级三判据**（`repro_pr6469.py`，本目录；不依赖服务，30 行秒级验证）：

| 判据 | bug 版 | 修复版 |
|---|---|---|
| `ret is output`（同一对象） | ❌ False | ✅ True |
| output buffer 脱离 NaN（写回） | ❌ | ✅ |
| 返回值本身有效 | ✅（值接力） | ✅ |

**方法论负结果（同日 Qwen2.5 实验，已记入历史）**：曾试用 Qwen2.5-1.5B + `--hf-overrides` 篡改架构为 Llama 注入 SWA——Qwen2 的 qkv bias 权重被 Llama loader 丢弃，线性投影系统性偏移产生"另一个乱码"，**三层修复均不变**（排除法证明该乱码与 bug 路径无关）。教训：**构造 SWA 复现必须用原生架构模型**，本实验最终选 gemma-3-4b-it 即因此。

## 7. 适用模型判定（哪些模型能被 #6469 修复）

判定标准：**原生 SWA + 走通用 FIA attention 通路 + eager decode**。

- ✅ **已验证**：Ling 系列（生产实证）、gemma-3-4b-it（本实验闭环）
- ✅ 预期可用：gpt-oss-20b/120b（**必须 BF16 版**，走标准 `per_layer_sliding_window` 通路，`gpt_oss.py:116-124` 验证；MXFP4 需 triton，本镜像 Triton disabled）、gemma-2、Mistral-7B-v0.1
- ❌ 不触发：Qwen dense / DeepSeek / GLM / Llama-3（全 full-attention）；Qwen3-Next/3.5（hybrid attention 另路，待验证）

## 8. 实验产物清单（本目录）

| 文件 | 内容 |
|---|---|
| `process.md` | 本文（全过程记录 + 根因分析） |
| `server_bug.log` | bug 版服务端日志（startup / POST 记录，含 ANSI 原色） |
| `curl_bug_request.sh` | bug 版请求脚本（完整 curl 命令存档，pod 原件 `/data/ask_bug_v2.sh`） |
| `curl_bug_screen.txt` | bug 版 curl 请求打屏（请求 + 乱码响应完整 JSON） |
| `fix_patch.log` | #6469 补丁应用输出（anchor=1 / PATCH_APPLIED / SYNTAX_OK） |
| `server_fixed.log` | 修复版服务端日志（新进程 startup / POST 记录） |
| `curl_fixed_request.sh` | 修复版请求脚本（同款 curl 命令存档，pod 原件 `/data/ask_fixed_v2.sh`） |
| `curl_fixed_screen.txt` | 修复版 curl 打屏（请求 + 正常响应完整 JSON） |
| `fix_pr6469.py` | 补丁脚本（锚点断言/备份/语法校验，可复用移植） |
| `repro_pr6469.py` | 单测级三判据复现脚本（3D 契约版，秒级验证） |
| pod `/data/`（hw_5） | 原始运行副本：`serve_bug_v2.sh`、`ask_bug_v2.sh`、`fix_and_serve_v2.sh`、v2_* 日志 |

## 9. 环境坑与执行经验

1. **`VLLM::EngineCore` 子进程改名**：`pkill -f 'vllm serve'` 杀不到（子进程名不含该串），残留进程占满 60G HBM 致下一服务 OOM。清理必须 `pkill -9 -f 'EngineC[o]re'`+`multi[p]rocessing`（bracket trick 防自匹配）。
2. **itask exec 丢 stdout**：python/curl 复合命令输出经常静默丢失。统一模式：脚本落盘到 pod `/data/` + python `-u` + `timeout`，再 `itask exec cat` 拉回本地。复杂命令一律走 base64 推送脚本执行（`itask exec -- bash -c "echo $B64 | base64 -d > x.sh"`）。
3. **`setsid nohup ... &` 后必须 `sleep 10`**：itask exec 会话清理时若子进程未完成分离会被杀（下载/serve 均靠此保活）。
4. **模型来源**：ModelScope 可直连（`google/gemma-3-4b-it`）；pod 内 Azure CDN（openaipublic）不通，影响 gpt-oss（见 §10）。
5. ModelScope 下载 gpt-oss 须走 `unsloth/gpt-oss-20b-BF16`（40G，TP4 可跑）。

## 10. 附：gpt-oss 尝试记录（环境障碍，未完成端到端）

gpt-oss-20b（BF16，已下载 40G）在本镜像上因**两个环境级接口 bug**无法 serve（均与 #6469 无关）：
1. ✗ 已修：`AscendYaRNRotaryEmbedding.__init__` 缺 `truncate` 参数（vllm 0.14 接口不同步）——补丁透传 `truncate: bool = True` 后通过。
2. ✗ 未解决：`openai_harmony`（Rust abi3 二进制包）启动 vocab 需从 `openaipublic.blob.core.windows.net` 下载（Azure CDN 内网不通；受 `TIKTOKEN_RS_CACHE_DIR` 缓存控制 + hash 校验，手动放置需 exact URL/hash 未从 so 挖出）。曾用 `use_harmony=False` 补丁绕过 `serving_chat.py:156` 的调用，但仍有其他启动期调用点（serving_responses/stream_harmony 疑似）触发同一错误。
结论：gpt-oss 的 #6469 复现**留待网络可达（Azure CDN）或离线 vocab 缓存方案就绪后执行**，gemma-3-4b 的本实验闭环为当前权威结果。

---

*复现命令入口（pod 内，一键重跑）：`bash /data/serve_bug_v2.sh` → `bash /data/ask_bug_v2.sh` → `bash /data/fix_and_serve_v2.sh` → `bash /data/ask_fixed_v2.sh`*
