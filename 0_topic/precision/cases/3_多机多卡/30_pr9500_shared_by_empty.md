# 案例30：DeepSeek-V4 P/D 分离中 kv_cache_tensor.shared_by 为空未守卫致启动崩溃 / 传输错位

> **一句话定位**：hybrid 模型分组层数不等时，vLLM core 会合法地产出 `shared_by == []` 的「预留内存槽」；而 vLLM-Ascend 的 Mooncake Hybrid Connector 在 `register_kv_caches` 的 `use_compress` 分支漏掉了空守卫，对空列表取 `min()` / `[0]`，导致 P/D 分离初始化崩溃，或被容错吞成传输错位。
>
> **对象**：vllm-project/vllm-ascend [#9500](https://github.com/vllm-project/vllm-ascend/pull/9500)（BugFix，merged）

---

## 1. 问题描述

### 1.1 现象

在 **DeepSeek-V4 hybrid（MLA + indexer）· PD 分离 · Mooncake Hybrid Connector（use_compress 分支）** 下：

- `register_kv_caches` 遍历 `kv_cache_config.kv_cache_tensors`，对每个 tensor 按 `shared_by` 里的层名收集地址，再对收集结果取 `min(share_tensor_addr)` / `share_tensor_stride[0]`；
- 当某个 tensor 的 `shared_by == []` 时，`for layer_name in shared_by` 循环体**一次不执行**，三个列表保持空 → `min(empty)` 抛 **`ValueError: min() arg is an empty sequence`**、`share_tensor_stride[0]` 抛 **`IndexError: list index out of range`**；
- 表现为 **P 端或 D 端 engine 启动失败**（快失败）；少数被上层 try/except 吞掉的路径则把一个空/错位条目塞进 `ptrs/lengths`，污染 Mooncake 按 block 寻址的元数据 → **传输错位 / 静默精度异常**。

### 1.2 触发条件（必现矩阵）

| hybrid 分组层数不等 | PD 分离 | use_compress 分支 | 是否触发 |
|:---:|:---:|:---:|:---:|
| ✗（单组 uniform） | — | — | ✗（`shared_by` 恒非空） |
| ✓ | ✗ | — | ✗（不调用 `register_kv_caches` 跨节点注册） |
| ✓ | ✓ | ✗（use_mamba） | ✗（分支已有 `if share_tensor_addr:` 守卫） |
| **✓** | **✓** | **✓** | **✓ 崩溃 / 传输错位** |

> hybrid 分组层数不等是 `shared_by` 为空的根因；非 hybrid、非 PD、或走 `use_mamba` 分支均不触发——这解释了 bug 直到 DeepSeek-V4 hybrid 大规模部署才暴露。

### 1.3 影响与严重度

- **严重度**：🟡 中（多数为启动期可见崩溃、快失败；少部分容错路径转化为 silent 精度问题）。
- **隐蔽性**：🟡 中（需 hybrid 分组不等 + PD 分离同时满足）。

---

## 2. 版本信息

| 字段 | 内容 |
|------|------|
| 仓库 | vllm-project/vllm-ascend |
| 对象 | PR [#9500](https://github.com/vllm-project/vllm-ascend/pull/9500)（BugFix） |
| 状态 | closed / merged |
| vllm-ascend 版本 | v0.20.2 |
| 上游 vllm | [`vllm-project/vllm@0d4d334`](https://github.com/vllm-project/vllm/commit/0d4d334eaa583b9c09aa4eb7538c22db99fd84b3) |
| 合并时间 | 2026-05-25 |
| 合并 commit | `261ea0be` |
| 改动规模 | +2 / −0（新增一个 `if not ... : continue` 守卫） |
| 影响文件 | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py` |
| 触发配置 | DeepSeek-V4 hybrid（MLA + indexer）· PD 分离 · Mooncake Hybrid Connector |

> 元数据说明：采集时 GitHub REST API 返回 403 rate-limit，author / reviewer / 精确创建时间未能获取；合并时间据全景文档归档，commit 据 `git log` 确认为 `261ea0be`。

---

## 3. 定位过程

1. **看报错**：启动期 `min() arg is an empty sequence` 或 `IndexError: list index out of range`，栈定位到 `register_kv_caches` 的 `use_compress` 分支 → 高度怀疑空 `shared_by`。
2. **dump kv_cache_tensors**：入口打印 `[t.shared_by for t in self.kv_cache_config.kv_cache_tensors]`，若出现 `[]` 即确认。
3. **核对分组层数**：检查 `kv_cache_groups` 各组 `layer_names` 长度，若不等则 general 分配路径必产生空 `shared_by`。
4. **对照 vLLM core**：确认 `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py` 已用 `if not tensor_layer_names: continue` 跳过同一空 `shared_by`——core 正确、ascend 漏守卫。
5. **回归**：修复后断言 `ptrs` 长度等于非空 `shared_by` 的 tensor 数，不再含空槽条目。

> 定位要点：`shared_by == []` 是 vLLM core 明确定义的「预留内存、无对应模型层」合法状态（`offloading/worker.py:172-175` 注释即契约），下游消费时漏掉空守卫是「上游契约 → 下游镜像守卫」缺失的典型。

---

## 4. 解决方案

### 4.1 根因

hybrid 分组层数不等时，`get_kv_cache_config_from_groups` 的 general 路径用 `group_size = max(len(group.layer_names))` 循环，较短组在尾部下标 `i` 处不贡献层名 → 该 `KVCacheTensor.shared_by` 停留在 `[]`（预留内存、无模型层）。vLLM core 的 offloading worker 已守卫跳过，但 vLLM-Ascend 的 `use_compress` 分支遗漏了对称守卫（同文件 `use_mamba` 分支却有守卫），对空列表取首元素。

### 4.2 修复内容

```diff
--- a/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py
+++ b/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py
 -1456,6 +1456,8  def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
                 for layer_name in group.layer_names:
                     layer_group_idx[layer_name] = i
             for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
+                if not kv_cache_tensor.shared_by:
+                    continue
                 share_tensor_addr = []
                 share_tensor_stride = []
                 cur_tensor_group_idx = []
```

核心思想：遍历 `kv_cache_tensors` 时，对 `shared_by == []` 的「预留内存、无对应模型层」槽直接 `continue` 跳过，与 vLLM core `offloading/worker.py` 的空守卫保持对称。

### 4.3 验证

- 回归：构造含 `shared_by=[]` 的 `KVCacheTensor` 调用 `register_kv_caches`，断言不抛异常且 `ptrs/lengths` 长度 = 非空 tensor 数。
- e2e：DeepSeek-V4 hybrid + PD 分离（4P1D / 2P2D）启动 + 短序列推理，验证不再崩溃且对端拉到无空槽的 block 元数据。

## 5. 复现方法

### 5.1 最小复现模型

- **DeepSeek-V4**（hybrid：MLA + indexer/compress，权重见官方/ModelScope）：本 bug 根因是「MLA 组与 indexer 组层数不等 → 产生空 `shared_by` 预留槽」，依赖 DeepSeek-V4 特定 hybrid 分组，**无更小模型可替代**。

### 5.2 最小服务命令（P/D 两端，走 use_compress 分支）

```bash
# P 端（prefill，hybrid connector 的 use_compress 分支）
export VLLM_USE_V1=1
vllm serve <DEEPSEEK_V4_MODEL> \
  --port 8010 \
  --tensor-parallel-size 8 \
  --enforce-eager --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"<HYBRID_CONNECTOR>","kv_role":"kv_producer","kv_port":"36000","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":8}}}'

# D 端（decode）
export VLLM_USE_V1=1
vllm serve <DEEPSEEK_V4_MODEL> \
  --port 8020 \
  --tensor-parallel-size 8 \
  --enforce-eager --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"<HYBRID_CONNECTOR>","kv_role":"kv_consumer","kv_port":"36100","kv_connector_extra_config":{"prefill":{"tp_size":8},"decode":{"tp_size":8}}}'
```

> 关键差异项：DeepSeek-V4 hybrid + `use_compress` 分支（对应 `mooncake_hybrid_connector.py`，connector 名以代码注册为准）。触发点在 `register_kv_caches` 初始化阶段：空 `shared_by` 未守卫 → `min() arg is an empty sequence` 崩溃。

---

> **核心教训**：消费 `KVCacheTensor.shared_by` 等「上游可能为空集合」的字段前，必须像 vLLM core 一样先做 `if not shared_by: continue` 守卫——hybrid 分组层数不等会合法地产生空 `shared_by` 预留槽，注册/传输路径若不跳过，就会在 P/D 分离初始化时崩溃或把错位元数据喂给对端。