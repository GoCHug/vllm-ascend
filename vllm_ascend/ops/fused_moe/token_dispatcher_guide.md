# MoE Token Dispatcher 详解

## 目录

- [一、整体架构概述](#一整体架构概述)
- [二、TokenDispatcherWithAll2AllV 类](#二tokendispatcherwithall2allv-类)
- [三、token_dispatch 方法详解](#三token_dispatch-方法详解)
- [四、_dispatch_preprocess 方法详解](#四_dispatch_preprocess-方法详解)
- [五、_preprocess 方法详解](#五_preprocess-方法详解)
- [六、token_combine 方法](#六token_combine-方法)
- [七、MoETokenDispatchInput 数据结构](#七moetokendispatchinput-数据结构)
- [八、完整数据流图](#八完整数据流图)

---

## 一、整体架构概述

### MoE（Mixture of Experts）基本流程

在混合专家模型中，每个 Token 会被路由到 Top-K 个专家进行处理。由于专家是分布式部署在不同 GPU/NPU 上的，需要通过 **AllToAll** 通信将 Token 发送到正确的设备。

```
  路由计算        Dispatch 分发        专家计算         Combine 合并
     │                  │                  │                  │
     ▼                  ▼                  ▼                  ▼
  topk_ids    ┌─────────────────┐  expert_mlp   ┌─────────────────┐
  topk_weights │  - 重排token    │  (每个设备    │  - 反向重排      │
               │  - 算splits     │   算本地      │  - 反向AllToAll  │
               │  - AllToAll发送 │   专家)       │  - 恢复顺序      │
               │  - 接收后重排   │               │  - 加权求和      │
               └─────────────────┘               └─────────────────┘
     │                  │                  │                  │
     └──────────────────┴──────────────────┴──────────────────┘
                              ↓
                     完整的 MoE Layer
```

### Dispatch 与 Combine 的对应关系

| 阶段 | 对应步骤 | 说明 |
|------|---------|------|
| **Dispatch（分发）** | 第 3 步：通过 AllToAll 将 Token 发送到对应专家所在的设备 | 把 token **送出去**给各个专家 |
| **Combine（合并）** | 第 4 步：计算完成后再通过 AllToAll 传回来并合并结果 | 把计算结果**收回来**并合并成最终输出 |

---

## 二、TokenDispatcherWithAll2AllV 类

### 类定义

```python
class TokenDispatcherWithAll2AllV(MoETokenDispatcher[MoEAllToAllCombineMetadata]):
```

**核心设计思想**：
- 基于 AllToAll 通信的 Token 分发器
- 专家（Experts）分布式部署在多个 NPU 上（通过专家并行 `ep_group`）
- 每个设备处理一部分专家（本地专家）
- Token 通过 AllToAll 在设备间传输

### 初始化关键参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `num_experts` | int | 全局专家总数 |
| `num_local_experts` | int | 每个设备上的本地专家数量 |
| `ep_group` | ProcessGroup | 专家并行通信组 |
| `ep_rank` | int | 当前设备在 ep_group 中的 rank |
| `ep_size` | int | ep_group 的 world size |
| `local_expert_indices` | list[int] | 本地专家的全局索引 |
| `expert_ids_per_ep_rank` | Tensor | 每个专家对应的本地 ID |

---

## 三、token_dispatch 方法详解

### 方法签名

```python
def token_dispatch(
    self,
    token_dispatch_input: MoETokenDispatchInput,
):
```

### 完整执行流程

#### 阶段 0：初始化与参数提取

```python
with_quant = token_dispatch_input.quant.is_int_quant or token_dispatch_input.quant.is_fp8
hidden_states = token_dispatch_input.hidden_states
topk_weights = token_dispatch_input.topk_weights
topk_ids = token_dispatch_input.topk_ids
```

- 判断是否需要量化（int8 或 fp8）
- 从输入对象中提取核心参数

---

#### 阶段 1：预处理 - `_dispatch_preprocess`

```python
(
    permutated_local_input_tokens,
    reversed_local_input_permutation_mapping,
    tokens_per_expert,
    input_splits,
    output_splits,
    global_input_tokens_local_experts_indices,
    hidden_shape,
    hidden_shape_before_permute,
) = self._dispatch_preprocess(hidden_states, topk_ids)
```

主要完成：
1. 展平 hidden_states：从 3D 变 2D
2. 计算路由元数据：`tokens_per_expert`, `input_splits`, `output_splits` 等
3. Token 重排：通过 `npu_moe_token_permute` 按专家 ID 排序 token
4. 保存形状信息：用于后续恢复

---

#### 阶段 2：量化处理（可选）

```python
dynamic_scale_after_all2all = None
if with_quant:
    dst_type = torch.float8_e4m3fn if token_dispatch_input.quant.is_fp8 else torch.int8
    permutated_local_input_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
        permutated_local_input_tokens, dst_type=dst_type
    )
    _, dynamic_scale_after_all2all, permute2_ep_all_to_all_handle = async_all_to_all(
        dynamic_scale, output_splits, input_splits, self.ep_group
    )
    permute2_ep_all_to_all_handle.wait()
    dynamic_scale.untyped_storage().resize_(0)
```

**量化逻辑详解**：

1. **动态量化**：`torch_npu.npu_dynamic_quant` 将 token 从 bf16/fp16 量化为 int8/fp8
2. **减少通信量**：量化后数据量减半，降低通信带宽压力
3. **Scale 先传**：scale 比较小，先通过 AllToAll 发送
4. **内存释放**：`untyped_storage().resize_(0)` 立即释放不再需要的内存

---

#### 阶段 3：主 AllToAll 通信

```python
_, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
    permutated_local_input_tokens, output_splits, input_splits, self.ep_group
)
permute1_ep_all_to_all_handle.wait()
permutated_local_input_tokens.untyped_storage().resize_(0)
```

**`async_all_to_all` 参数说明**：

| 参数 | 说明 |
|------|------|
| `input_` | 要发送的数据 |
| `output_split_sizes` | 从每个 rank 接收的数据大小 |
| `input_split_sizes` | 发送给每个 rank 的数据大小 |
| `group` | 通信组（这里是 ep_group） |

**AllToAll 可视化**：

```
设备 0 的 token: [给rank0的, 给rank1的, 给rank2的, 给rank3的]
设备 1 的 token: [给rank0的, 给rank1的, 给rank2的, 给rank3的]
设备 2 的 token: [给rank0的, 给rank1的, 给rank2的, 给rank3的]
设备 3 的 token: [给rank0的, 给rank1的, 给rank2的, 给rank3的]

         │
         │ AllToAll 通信
         ▼

设备 0 收到: [rank0发来的, rank1发来的, rank2发来的, rank3发来的]
设备 1 收到: [rank0发来的, rank1发来的, rank2发来的, rank3发来的]
...
```

---

#### 阶段 4：后处理 - `_dispatch_postprocess`

```python
global_input_tokens, dynamic_scale_final, reversed_global_input_permutation_mapping = (
    self._dispatch_postprocess(
        global_input_tokens,
        dynamic_scale_after_all2all,
        global_input_tokens_local_experts_indices,
        with_quant,
    )
)
```

后处理的主要工作：
1. 如果本地只有 1 个专家：直接返回，不需要进一步处理
2. 如果有多个本地专家：
   - 对 token 按本地专家 ID 再次重排
   - 量化情况下同时重排 dynamic_scale
   - 返回反向映射，用于后续恢复顺序

---

#### 阶段 5：封装输出结果

```python
return MoETokenDispatchOutput(
    hidden_states=global_input_tokens,
    dynamic_scale=dynamic_scale_final,
    group_list=tokens_per_expert,
    group_list_type=1,
    combine_metadata=MoEAllToAllCombineMetadata(
        input_splits=input_splits,
        output_splits=output_splits,
        topk_weights=topk_weights,
        reversed_local_input_permutation_mapping=reversed_local_input_permutation_mapping,
        reversed_global_input_permutation_mapping=reversed_global_input_permutation_mapping,
        hidden_shape=hidden_shape,
        hidden_shape_before_permute=hidden_shape_before_permute,
    ),
)
```

**输出结构解析**：

| 字段 | 说明 |
|------|------|
| `hidden_states` | 分发后的 token，已经按专家排好序，可以直接送入专家计算 |
| `dynamic_scale` | 量化时的缩放因子 |
| `group_list` | 每个专家的 token 数量 |
| `group_list_type=1` | 表示 group_list 是 count 模式 |
| `combine_metadata` | 合并阶段需要的元数据 |

**`MoEAllToAllCombineMetadata` 元数据**：

| 字段 | 用途 |
|------|------|
| `input_splits`, `output_splits` | 反向 AllToAll 需要的 split 信息 |
| `topk_weights` | 路由权重，用于加权求和 |
| `reversed_local_input_permutation_mapping` | 第一次 permute 的反向映射 |
| `reversed_global_input_permutation_mapping` | 第二次 permute 的反向映射 |
| `hidden_shape` | 原始形状，用于恢复 |
| `hidden_shape_before_permute` | permute 前的形状 |

---

## 四、_dispatch_preprocess 方法详解

### 方法签名

```python
def _dispatch_preprocess(self, hidden_states, topk_ids):
```

### 执行流程

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | `hidden_shape = hidden_states.shape` | 保存原始形状，用于后续恢复 |
| 2 | `hidden_states = hidden_states.view(-1, hidden_states.size(-1))` | 展平：将 (batch, seq_len, hidden_dim) 变为 (total_tokens, hidden_dim) |
| 3 | 调用 `_preprocess(topk_ids)` | 计算路由所需的各种元数据 |
| 4 | `hidden_shape_before_permute = hidden_states.shape` | 记录 permute 前的形状 |
| 5 | `torch_npu.npu_moe_token_permute(...)` | NPU 专属算子：根据 topk_ids 对 token 进行重排 |
| 6 | 返回所有预处理结果 | 返回给上层用于 AllToAll 和后续计算 |

### 关键参数说明

| 参数 | 类型 | 含义 |
|------|------|------|
| `hidden_states` | Tensor | 输入的隐藏状态 |
| `topk_ids` | Tensor | 每个 token 被路由到的专家 ID |
| `permutated_local_input_tokens` | Tensor | 按专家 ID 重排后的 token |
| `reversed_local_input_permutation_mapping` | Tensor | 反向映射，用于后续恢复原始顺序 |

---

## 五、_preprocess 方法详解

### 方法签名

```python
def _preprocess(self, topk_ids: torch.Tensor):
```

### 执行流程详解

#### 第一步：统计本地每个专家的 token 数量

```python
num_local_tokens_per_expert = torch.histc(topk_ids, bins=self.num_experts, min=0, max=self.num_experts)
```

- `torch.histc`：直方图统计，计算每个 expert ID 出现的次数
- 结果形状：`(num_experts,)`，每个元素表示对应 expert 收到的 token 数量

---

#### 第二步：计算 `input_splits`

```python
input_splits = (
    num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
    .sum(axis=1)
    .to(torch.device("cpu"), non_blocking=True)
    .numpy()
)
```

**逻辑解析**：
1. 将 `num_local_tokens_per_expert` 按 `(ep_size, num_local_experts)` 形状重新排列
2. 沿 axis=1 求和，得到**当前设备需要发送给每个 ep_rank 的 token 数量**
3. 转移到 CPU 并转为 numpy 数组

**示例**：假设 `ep_size=4`, `num_local_experts=2`

```
原始 num_local_tokens_per_expert: [10, 20, 15, 25, 30, 18, 22, 28]  (共8个专家)
reshape 后:
[[10, 20],  # ep_rank 0 的本地专家
 [15, 25],  # ep_rank 1 的本地专家
 [30, 18],  # ep_rank 2 的本地专家
 [22, 28]]  # ep_rank 3 的本地专家

sum(axis=1) -> [30, 40, 48, 50]  (input_splits)
```

`input_splits[i]` 表示当前设备需要发送给 `ep_rank=i` 的 token 数量。

---

#### 第三步：获取全局 token 分布

```python
num_global_tokens_per_expert = gather_from_sequence_parallel_region(
    num_local_tokens_per_expert, group=self.ep_group
).reshape(ep_size, self.num_experts)
num_global_tokens_per_local_expert = num_global_tokens_per_expert[
    :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
]
```

- `gather_from_sequence_parallel_region`：通过 AllGather 收集所有设备的本地统计信息
- 结果 `num_global_tokens_per_expert`：形状 `(ep_size, num_experts)`，表示每个设备上每个专家的 token 数量
- `num_global_tokens_per_local_expert`：提取**当前设备本地专家**在全局的 token 分布

---

#### 第四步：计算 `output_splits`

```python
output_splits = (
    num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True).numpy()
)
num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
```

- `output_splits`：形状 `(ep_size,)`，表示当前设备**从每个 ep_rank 接收的 token 数量**
- `num_tokens_per_local_expert`：形状 `(num_local_experts,)`，表示每个本地专家最终处理的 token 总数

---

#### 第五步：生成全局 token 的本地专家索引

```python
global_input_tokens_local_experts_indices = None
if self.num_local_experts > 1:
    global_input_tokens_local_experts_indices = torch.repeat_interleave(
        self.expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel()
    )
else:
    torch.npu.synchronize()
```

- **当本地专家数量 > 1 时**：需要为每个全局 token 标记其属于哪个本地专家
- `torch.repeat_interleave`：将专家 ID 按 token 数量重复
- **当本地专家数量 = 1 时**：不需要标记，直接同步

**示例**：假设本地有 2 个专家，ID 为 [0, 1]，各自有 [30, 48] 个 token

```python
expert_ids_per_ep_rank = [0, 1]
num_global_tokens_per_local_expert = [30, 48]

torch.repeat_interleave([0, 1], [30, 48]) 
-> [0, 0, ..., 0, 1, 1, ..., 1]  # 30个0, 48个1
```

---

### 返回值汇总

| 返回值 | 类型 | 含义 |
|--------|------|------|
| `num_tokens_per_local_expert` | Tensor | 每个本地专家处理的 token 数 |
| `input_splits` | numpy.ndarray | AllToAll 发送 split 信息 |
| `output_splits` | numpy.ndarray | AllToAll 接收 split 信息 |
| `global_input_tokens_local_experts_indices` | Tensor | 全局 token 到本地专家的映射 |
| `num_out_tokens` | int | 总 token 数量 |

---

## 六、token_combine 方法

### 方法签名

```python
def token_combine(self, hidden_states, combine_metadata, bias=None):
```

### 执行流程

```python
# 1. Preprocess using metadata
hidden_states = self._combine_preprocess(hidden_states, combine_metadata)

# 2. AllToAll
_, permutated_local_input_tokens, handle = async_all_to_all(
    hidden_states,
    combine_metadata.input_splits,
    combine_metadata.output_splits,
    self.ep_group,
)
handle.wait()
hidden_states.untyped_storage().resize_(0)

# 3. Postprocess using metadata
output = self._combine_postprocess(permutated_local_input_tokens, combine_metadata)

return output
```

**与 token_dispatch 的对称关系**：
- Dispatch 用 `input_splits` 发、`output_splits` 收
- Combine 反过来，用 `output_splits` 发、`input_splits` 收（因为方向反过来了）

---

## 七、MoETokenDispatchInput 数据结构

### 结构总览

```python
@dataclass(frozen=True, slots=True)
class MoETokenDispatchInput:
    """Input to token dispatch."""
    hidden_states: torch.Tensor      # 输入隐藏状态
    topk_weights: torch.Tensor       # Top-K 路由权重
    topk_ids: torch.Tensor           # Top-K 专家 ID
    routing: MoERoutingParams        # 路由参数
    quant: MoEQuantParams            # 量化参数
```

**设计说明**：
- `frozen=True`：不可变数据类，创建后不能修改，保证数据一致性
- `slots=True`：使用 `__slots__` 节省内存，访问更快

### 典型场景配置

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 4 | 4 个请求 |
| seq_len | 128 | 每个请求 128 个 token |
| hidden_dim | 4096 | 隐藏层维度 |
| num_experts | 8 | 总共 8 个专家 |
| top_k | 2 | 每个 token 选 2 个专家 |
| ep_size | 4 | 4 个设备做专家并行 |
| num_local_experts | 2 | 每个设备上 2 个本地专家 |

---

### 各字段详解与 Shape 示例

#### 1. `hidden_states` - 输入隐藏状态

**形状**：有两种常见形式

##### 形式 A：3D 张量（训练/prefill 阶段）
```
(batch_size, seq_len, hidden_dim)
```

**例子**：
```python
hidden_states.shape = torch.Size([4, 128, 4096])
# 4 个 batch，每个 128 个 token，每个 token 是 4096 维
```

##### 形式 B：2D 张量（解码阶段 / 已展平）
```
(num_tokens, hidden_dim)
```

**例子**：
```python
hidden_states.shape = torch.Size([512, 4096])
# 4*128 = 512 个 token，每个 4096 维
```

---

#### 2. `topk_ids` - Top-K 专家 ID

**形状**：
```
(num_tokens, top_k)
```

**数据类型**：通常是 `torch.int64` 或 `torch.int32`

**例子**：
```python
topk_ids.shape = torch.Size([512, 2])
# 512 个 token，每个选 2 个专家

# 具体内容示例（前 5 个 token）：
topk_ids[:5] = tensor([
    [3, 1],   # 第 0 个 token 路由到专家 3 和 1
    [0, 2],   # 第 1 个 token 路由到专家 0 和 2
    [5, 7],   # 第 2 个 token 路由到专家 5 和 7
    [1, 4],   # 第 3 个 token 路由到专家 1 和 4
    [6, 2],   # 第 4 个 token 路由到专家 6 和 2
], dtype=torch.int64)
```

**取值范围**：`[0, num_experts - 1]`

---

#### 3. `topk_weights` - Top-K 路由权重

**形状**：
```
(num_tokens, top_k)
```

**数据类型**：通常是 `torch.float32` 或 `torch.bfloat16`

**例子**：
```python
topk_weights.shape = torch.Size([512, 2])
# 512 个 token，每个对应 2 个专家的权重

# 具体内容示例（前 5 个 token）：
topk_weights[:5] = tensor([
    [0.7, 0.3],   # 第 0 个 token: 专家3 占 70%, 专家1 占 30%
    [0.6, 0.4],   # 第 1 个 token: 专家0 占 60%, 专家2 占 40%
    [0.8, 0.2],   # 第 2 个 token: 专家5 占 80%, 专家7 占 20%
    [0.5, 0.5],   # 第 3 个 token: 专家1 占 50%, 专家4 占 50%
    [0.9, 0.1],   # 第 4 个 token: 专家6 占 90%, 专家2 占 10%
], dtype=torch.bfloat16)
```

**特点**：
- 每个 token 的 K 个权重之和通常 = 1（经过 softmax）
- 用于 combine 阶段的**加权求和**

---

#### 4. `routing` - 路由参数（MoERoutingParams）

**结构**：
```python
@dataclass(frozen=True, slots=True)
class MoERoutingParams:
    expert_map: torch.Tensor | None          # 专家映射表
    global_redundant_expert_num: int         # 全局冗余专家数量
    mc2_mask: torch.Tensor | None            # MC2 掩码
    apply_router_weight_on_input: bool       # 是否在输入上应用路由权重
    log2phy: torch.Tensor | None = None      # 逻辑到物理专家的映射
    pertoken_scale: torch.Tensor | None = None  # 逐 token 量化 scale
```

| 字段 | 类型 | Shape 示例 | 说明 |
|------|------|------------|------|
| `expert_map` | `Tensor \| None` | `(num_experts,)` 或 `None` | 专家 ID 映射，用于专家负载均衡 |
| `global_redundant_expert_num` | `int` | `0` 或 `2` | 冗余专家数量 |
| `mc2_mask` | `Tensor \| None` | `(num_tokens, num_experts)` 或 `None` | MC2 掩码 |
| `apply_router_weight_on_input` | `bool` | `False` | 是否在 dispatch 前就把权重乘到 hidden states 上 |
| `log2phy` | `Tensor \| None` | `(num_experts,)` 或 `None` | 逻辑专家 ID 到物理专家 ID 的映射 |
| `pertoken_scale` | `Tensor \| None` | `(num_tokens,)` 或 `None` | 预计算的逐 token 量化缩放因子 |

---

#### 5. `quant` - 量化参数（MoEQuantParams）

**结构**：
```python
@dataclass(frozen=True, slots=True)
class MoEQuantParams:
    quant_type: QuantType = QuantType.NONE      # 量化类型
    comm_quant_mode: int | None = None           # 通信量化模式
    mxfp: MoEMxfpParams | None = None            # MXFP 专用参数
    is_per_channel_weight: bool = False          # 是否逐通道权重量化
```

**便捷属性**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `is_quant` | `bool` | 是否启用了任何量化 |
| `is_int_quant` | `bool` | 是否是整数量化（W8A8, W4A8） |
| `is_fp8` | `bool` | 是否是 FP8 量化 |
| `is_mxfp` | `bool` | 是否是 MXFP 量化 |
| `dispatch_with_quant` | `bool` | dispatch 阶段是否需要量化 |

**常见量化类型**：
- `QuantType.NONE`：不量化（bf16/fp16）
- `QuantType.W8A8`：权重 8bit，激活 8bit
- `QuantType.W4A8`：权重 4bit，激活 8bit
- `QuantType.W8A8FP8`：权重 8bit，激活 fp8
- `QuantType.MXFP8` / `MXFP4`：MX 格式的 FP8/FP4 量化

---

### 完整的数值示例

#### 配置
- batch_size = 1
- seq_len = 3（共 3 个 token）
- hidden_dim = 4
- num_experts = 4
- top_k = 2

#### 1. hidden_states (3, 4)
```python
hidden_states = tensor([
    [0.1, 0.2, 0.3, 0.4],   # token 0
    [1.1, 1.2, 1.3, 1.4],   # token 1
    [2.1, 2.2, 2.3, 2.4],   # token 2
])
```

#### 2. topk_ids (3, 2)
```python
topk_ids = tensor([
    [0, 2],   # token 0 → 专家 0 和 2
    [1, 3],   # token 1 → 专家 1 和 3
    [0, 1],   # token 2 → 专家 0 和 1
])
```

#### 3. topk_weights (3, 2)
```python
topk_weights = tensor([
    [0.7, 0.3],   # token 0: 70% 专家0, 30% 专家2
    [0.6, 0.4],   # token 1: 60% 专家1, 40% 专家3
    [0.8, 0.2],   # token 2: 80% 专家0, 20% 专家1
])
```

#### 路由结果统计

| 专家 | token 列表 | token 数量 |
|------|-----------|-----------|
| 专家 0 | token 0, token 2 | 2 个 |
| 专家 1 | token 1, token 2 | 2 个 |
| 专家 2 | token 0 | 1 个 |
| 专家 3 | token 1 | 1 个 |

---

## 八、完整数据流图

```
输入: hidden_states (B, S, H), topk_ids, topk_weights
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. _dispatch_preprocess                                    │
│    - 展平: (B*S, H)                                         │
│    - histc 统计每个expert的token数                          │
│    - 计算 input_splits, output_splits                      │
│    - npu_moe_token_permute: 按expert重排token              │
└─────────────────────────────────────────────────────────────┘
    │
    ├───────────── 量化分支 (with_quant=True) ─────────────┐
    │                                                      │
    ▼                                                      ▼
┌───────────────────────┐                        ┌───────────────────────┐
│ npu_dynamic_quant     │                        │  跳过量化              │
│  → quant_tokens       │                        │                       │
│  → dynamic_scale      │                        │                       │
└───────────┬───────────┘                        └───────────┬───────────┘
            │                                                │
            ▼                                                │
┌───────────────────────┐                                   │
│ AllToAll (scale)      │                                   │
│  先传小数据 scale      │                                   │
└───────────┬───────────┘                                   │
            │                                                │
            └──────────────────┬─────────────────────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │ AllToAll (tokens)     │
                    │  主通信，传输token     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ _dispatch_postprocess │
                    │  按本地expert再次重排  │
                    └───────────┬───────────┘
                                │
                                ▼
                    输出: MoETokenDispatchOutput
                         - hidden_states (按expert排序)
                         - group_list (每个expert的token数)
                         - combine_metadata (用于后续合并)
```

---

## 九、关键设计亮点

### 1. 内存优化

```python
xxx.untyped_storage().resize_(0)
```

主动释放不再需要的 tensor 内存，这在大模型推理中非常重要。

### 2. 通信优化

- 量化减少通信量
- 异步 AllToAll（支持计算通信重叠）
- scale 和 token 分开传输（小的先传）

### 3. 两次重排设计

- **第一次**（本地）：按全局 expert ID 重排，为 AllToAll 做准备
- **第二次**（全局后）：按本地 expert ID 重排，为专家计算做准备

### 4. 完整的元数据传递

- 所有需要的形状、映射、split 信息都通过 `combine_metadata` 传递
- 确保后续的 `token_combine` 能正确恢复原始顺序和形状

---

## 十、代码位置参考

| 组件 | 文件 | 行号 |
|------|------|------|
| `token_dispatch` | `token_dispatcher.py` | 467 |
| `token_combine` | `token_dispatcher.py` | 531 |
| `_dispatch_preprocess` | `token_dispatcher.py` | 552 |
| `_preprocess` | `token_dispatcher.py` | 581 |
| `_dispatch_postprocess` | `token_dispatcher.py` | 626 |
| `MoETokenDispatchInput` | `moe_stage_contracts.py` | 74 |
| `async_all_to_all` | `comm_utils.py` | 26 |
