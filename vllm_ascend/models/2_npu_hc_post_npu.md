# hc_post 算子 — HC后置处理

## 一、概述

hc_post 是 HC (Hyper-Complex) 机制的后置处理算子，与 hc_pre 配对使用。hc_pre 将 HC 维度压缩后送入注意力/FFN计算，hc_post 则将计算结果恢复回 HC 扩展维度。

**核心功能**:
1. 用 `comb` 组合矩阵对残差 `residual` 的 hc_mult 个通道做线性混合（通道间特征重组）
2. 用 `post` 门控权重对主路径输出 `x` 做通道级缩放，扩展成 hc_mult 个通道
3. 两者相加，得到最终的 HC 扩展维度隐藏状态

**文件位置**:
- Python 层: `vllm-ascend/vllm_ascend/models/deepseek_v4.py#L976`
- C++ 绑定: `vllm-ascend/csrc/torch_binding.cpp#L1425`
- 算子定义: `vllm-ascend/csrc/moe/hc_post/op_host/hc_post_def.cpp`
- 形状推理: `vllm-ascend/csrc/moe/hc_post/op_host/hc_post_proto.cpp`
- Tiling 策略: `vllm-ascend/csrc/moe/hc_post/op_host/hc_post_tiling.cpp`
- NPU 内核: `vllm-ascend/csrc/moe/hc_post/op_kernel/`

***

## 二、Python 层接口

### 2.1 函数签名

```python
def hc_post(self, x: torch.Tensor, residual: torch.Tensor,
            post: torch.Tensor, comb: torch.Tensor):
    y = torch.ops._C_ascend.npu_hc_post(
        x.unsqueeze(dim=0),
        residual.unsqueeze(dim=0),
        post.unsqueeze(dim=0),
        comb.unsqueeze(dim=0)
    )
    return y.squeeze(dim=0)
```

> **注意**: Python 层做了 `unsqueeze(0)` 和 `squeeze(0)` 处理，用于匹配 NPU 算子预期的 4D 输入格式 `(1, num_tokens, ...)`。这是 NPU 算子的常见格式要求，不影响计算逻辑。

### 2.2 形状符号约定

- `num_tokens`: 一次处理的 token 总数（vLLM 的连续批处理方式）
- `hc_mult`: HC 通道数（通常为 4）
- `hidden_size`: 隐藏层维度（4096 或 7168）

### 2.3 输入输出张量

| 张量 | Python 层 Shape | NPU 算子输入 Shape | dtype | 说明 |
|------|----------------|-------------------|-------|------|
| `x` | `(num_tokens, hidden_size)` | `(1, num_tokens, hidden_size)` | float16/bfloat16/float32 | Attention/FFN 输出特征（主路径） |
| `residual` | `(num_tokens, hc_mult, hidden_size)` | `(1, num_tokens, hc_mult, hidden_size)` | 同 x | hc_pre 前的残差（克隆自原始 hidden_states） |
| `post` | `(num_tokens, hc_mult)` | `(1, num_tokens, hc_mult)` | float16/bfloat16/float32 | hc_pre 输出的后处理门控 |
| `comb` | `(num_tokens, hc_mult, hc_mult)` | `(1, num_tokens, hc_mult, hc_mult)` | float16/bfloat16/float32 | hc_pre 输出的组合矩阵（双随机矩阵） |
| `y` (输出) | `(num_tokens, hc_mult, hidden_size)` | `(1, num_tokens, hc_mult, hidden_size)` | 同 x | 恢复为 HC 扩展维度的隐藏状态 |

***

## 三、完整调用链

从 Python 调用到 NPU 硬件执行的完整路径：

```
┌─────────────────────────────────────────────────────────────────────┐
│  第1层: Python 调用层                                               │
│  文件: deepseek_v4.py:L976-980                                      │
│  函数: hc_post()                                                    │
│  ├── x.unsqueeze(dim=0)            # 增加batch维度 → (1, ...)       │
│  ├── residual.unsqueeze(dim=0)     # 增加batch维度                  │
│  ├── post.unsqueeze(dim=0)         # 增加batch维度                  │
│  ├── comb.unsqueeze(dim=0)         # 增加batch维度                  │
│  ├── torch.ops._C_ascend.npu_hc_post(x, residual, post, comb)      │
│  └── y.squeeze(dim=0)              # 去掉batch维度                  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  第2层: PyTorch Dispatcher 调度层                                   │
│  算子注册: torch_binding.cpp:L2680-2687                             │
│  ops.def("npu_hc_post("                                            │
│      "Tensor x, Tensor residual, Tensor post, Tensor comb"         │
│      ") -> (Tensor out)")                                          │
│  ops.impl("npu_hc_post", kPrivateUse1, &npu_hc_post_npu)           │
│  根据输入张量的设备类型 (NPU) 调度到对应后端实现                     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  第3层: C++ NPU 入口函数                                            │
│  文件: torch_binding.cpp:L1425-1436                                 │
│  函数: npu_hc_post_npu()                                            │
│  ├── check_hc_post_shape_and_dtype(x, residual, post, comb)        │
│  │   # 参数形状/类型校验 (L1389-1423)                                │
│  │   ├── 校验维度: x是3D [b,s,d], residual是4D [b,s,hc,d],          │
│  │   │           post是3D [b,s,hc], comb是4D [b,s,hc,hc]            │
│  │   ├── 校验大小: batch/sequence/hidden各维度对应一致, hc通道数一致 │
│  │   └── 校验类型: x/residual=float16/bfloat16/float32              │
│  │              post/comb=float16/bfloat16/float32                  │
│  ├── construct_hc_post_output_tensor(residual)                     │
│  │   # 构造输出张量，形状同residual (L1376-1386)                    │
│  ├── EXEC_NPU_CMD(aclnnHcPost, x, residual, post, comb, out)       │
│  │   # 调用 CANN 算子库的 aclnnHcPost                               │
│  └── return out                                                     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  第4层: Host 端算子框架 (编译期)                                     │
│  文件: csrc/moe/hc_post/op_host/                                    │
│  ├── hc_post_def.cpp         # 算子定义（输入输出、支持格式、平台）  │
│  ├── hc_post_proto.cpp       # InferShape + InferDtype              │
│  └── hc_post_tiling.cpp      # Tiling 策略（切分、核数、分块）       │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  第5层: Device 端 NPU 内核 (运行期)                                  │
│  文件: csrc/moe/hc_post/op_kernel/                                  │
│  ├── hc_post.cpp             # 内核入口函数                         │
│  ├── hc_post_d_split.h       # D维切分主类（数据搬运+计算流水）      │
│  ├── hc_post_bfloat16.h      # bfloat16 计算核心                    │
│  └── hc_post_float32.h       # float32 计算核心                     │
│                                                                     │
│  核心公式:                                                          │
│    y[b,j,h] = Σ residual[b,i,h] · comb[b,i,j] + x[b,h] · post[b,j]  │
│             i=0..hc_mult-1                                          │
└─────────────────────────────────────────────────────────────────────┘
```

***

## 四、核心计算原理

### 4.1 核心公式

```
╭──────────────────────────────────────────────────────────────────────╮
│                                                                    │
│   y[b, j, h] = Σ residual[b, i, h] · comb[b, i, j]                 │
│                i=0..hc_mult-1                                      │
│                                                                    │
│              + x[b, h] · post[b, j]                                │
│                                                                    │
╰──────────────────────────────────────────────────────────────────────╯
```

**符号说明**:

| 符号 | 含义 | 范围 |
|------|------|------|
| `b` | token 索引 | 0 ~ num_tokens-1 |
| `i` | 输入通道索引 | 0 ~ hc_mult-1 |
| `j` | 输出通道索引 | 0 ~ hc_mult-1 |
| `h` | hidden 维度索引 | 0 ~ hidden_size-1 |

### 4.2 矩阵形式（对每个 token b 和 hidden h）

```
╭──────────────────────────────────────────────────────────────────────╮
│                                                                    │
│   y[b, :, h] = residual[b, :, h] @ comb[b, :, :]                   │
│                                                                    │
│              + post[b, :] ⊙ x[b, h]                                │
│                                                                    │
│   其中:                                                             │
│     residual[b, :, h]  ∈ R^(1×hc_mult)   ← 行向量                        │
│     comb[b, :, :]      ∈ R^(hc_mult×hc_mult)   ← 矩阵                          │
│     post[b, :]         ∈ R^hc_mult       ← 向量                          │
│     x[b, h]            ∈ R         ← 标量                          │
│     y[b, :, h]         ∈ R^hc_mult       ← 行向量                        │
│                                                                    │
╰──────────────────────────────────────────────────────────────────────╯
```

### 4.3 计算步骤分解

#### 步骤1：残差通道混合

用 `comb` 组合矩阵对 `residual` 的 hc_mult 个输入通道做线性混合，得到 hc_mult 个输出通道。

**这一步在做什么？**
把残差的 hc_mult 个输入通道，通过一个 hc_mult×hc_mult 的混合矩阵，重新组合成 hc_mult 个输出通道。这是 HC 机制的核心——通道间的动态特征重组。

**Shape 变化**:
```
residual:  (num_tokens, hc_mult, hidden_size)    bfloat16
comb:      (num_tokens, hc_mult, hc_mult)        float32
    │
    │  对每个 token b，每个 hidden 位置 h：
    │  residual[b, :, h] 是 hc_mult 维行向量
    │  comb[b, :, :] 是 hc_mult×hc_mult 矩阵
    │  结果 = residual[b, :, h] @ comb[b, :, :]  →  hc_mult 维行向量
    ▼
residual_mixed: (num_tokens, hc_mult, hidden_size)  float32
```

**逐元素公式**:
```
residual_mixed[b, j, h] = Σ residual[b, i, h] · comb[b, i, j]
                          i=0..hc_mult-1
```

**einsum 表示**:
```
residual_mixed = einsum('b i h, b i j → b j h', residual, comb)
```

> **comb 矩阵的理解**：
> - comb[b, i, j] 表示：第 i 个输入通道 对 第 j 个输出通道 的贡献权重
> - 因为 comb 是双随机矩阵（每行和为1，每列和为1），所以这是一个"保能量"的混合
> - 输出通道的总能量 ≈ 输入通道的总能量

---

#### 步骤2：主路径扩展 + 门控加权

用 `post` 门控把主路径输出 `x` 从 1 个通道扩展成 hc_mult 个通道。

**这一步在做什么？**
x 只有 1 个通道（hidden_size 维），把它"复制"成 hc_mult 个通道，每个通道乘上一个对应的 post 门控权重，实现通道级的自适应缩放。

**Shape 变化**:
```
x:     (num_tokens, hidden_size)     bfloat16
post:  (num_tokens, hc_mult)         float32
    │
    │  广播机制:
    │  x.unsqueeze(1)     → (num_tokens, 1, hidden_size)
    │  post.unsqueeze(-1) → (num_tokens, hc_mult, 1)
    │  两者相乘 → 广播成 (num_tokens, hc_mult, hidden_size)
    ▼
x_gated: (num_tokens, hc_mult, hidden_size)  float32
```

**逐元素公式**:
```
x_gated[b, j, h] = x[b, h] · post[b, j]
```

**广播机制示意图**:
```
x.shape     = (num_tokens,    hidden_size)   → unsqueeze(1) → (num_tokens, 1, hidden_size)
                                                                     ↓ 广播
post.shape  = (num_tokens, hc_mult   )   → unsqueeze(-1) → (num_tokens, hc_mult, 1)
                                                                     ↓ 广播
结果.shape  = (num_tokens, hc_mult, hidden_size)
```

> **post 门控的作用**：
> - post 值范围是 (0, 2)（从 hc_pre 的 ProcessPost 可知：sigmoid × 2）
> - post[j] > 1：第 j 个通道被放大
> - post[j] < 1：第 j 个通道被衰减
> - 这是一种动态的通道注意力机制，每个 token 可以自己决定每个通道的强度

---

#### 步骤3：两者相加 + 类型转换

把通道混合后的残差 和 门控扩展后的主路径 加在一起，然后转回 bfloat16。

```
y_float = residual_mixed + x_gated
y       = y_float.to(bfloat16)

y.shape = (num_tokens, hc_mult, hidden_size)
```

***

## 五、Host 端算子框架详解

Host 端负责算子的注册、形状/类型推理和 Tiling（切分策略），在编译期执行。

### 5.1 算子定义 (hc_post_def.cpp)

```cpp
class HcPost : public OpDef {
public:
    explicit HcPost(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({float16, float16, float16,
                       bfloat16, bfloat16, bfloat16,
                       float32, float32, float32})
            .Format({FORMAT_ND, ...});
        this->Input("residual")
            .ParamType(REQUIRED)
            .DataType({同 x 的9种组合})
        this->Input("post")
            .ParamType(REQUIRED)
            .DataType({float16, bfloat16, float32,
                       float16, bfloat16, float32,
                       float16, bfloat16, float32})
        this->Input("comb")
            .ParamType(REQUIRED)
            .DataType({同 post 的9种组合})
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({同 x})
        
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend950");
    }
};
```

**支持的 dtype 组合 (共9种)**:

| x/residual/y | post/comb |
|-------------|-----------|
| float16 | float16 |
| float16 | bfloat16 |
| float16 | float32 |
| bfloat16 | float16 |
| bfloat16 | bfloat16 |
| bfloat16 | float32 |
| float32 | float16 |
| float32 | bfloat16 |
| float32 | float32 |

**支持的硬件平台**:
- ascend910b
- ascend910_93
- ascend950 (使用 arch35 tiling)

### 5.2 形状与类型推理 (hc_post_proto.cpp)

```cpp
// 输入索引常量
const int32_t INPUT_IDX_X = 0;
const int32_t INPUT_IDX_RESIDUAL = 1;
const int32_t INPUT_IDX_POST = 2;
const int32_t INPUT_IDX_COMB = 3;
const int32_t INDEX_OUTPUT_Y = 0;

// 形状推理：输出 shape = residual 的 shape
static ge::graphStatus InferShape4HcPost(...) {
    *yShape = *residualShape;  // 直接复用 residual 的形状
}

// 类型推理：输出 dtype = x 的 dtype
static ge::graphStatus InferDtype4HcPost(...) {
    context->SetOutputDataType(INDEX_OUTPUT_Y, xDtype);
}
```

**推理逻辑**:
- 输出形状 = residual 形状（因为输出也是 hc_mult × hidden_size 维度）
- 输出类型 = x 的类型（与主路径保持一致，residual 也要求同类型）

### 5.3 Tiling 策略 (hc_post_tiling.cpp)

Tiling 决定了算子如何在多个核之间切分，以及每个核内部如何分块处理。

#### 5.3.1 Tiling 参数结构

```cpp
BEGIN_TILING_DATA_DEF(HcPostTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);      // 实际使用的核数
TILING_DATA_FIELD_DEF(int64_t, bsParam);          // batch size (token 总数)
TILING_DATA_FIELD_DEF(int64_t, hcParam);          // hc_mult
TILING_DATA_FIELD_DEF(int64_t, dParam);           // hidden_size
TILING_DATA_FIELD_DEF(int64_t, batchOneCore);     // 每个核处理的 token 数（完整块）
TILING_DATA_FIELD_DEF(int64_t, batchOneCoreTail); // 每个核处理的 token 数（尾部核）
TILING_DATA_FIELD_DEF(int64_t, frontCore);        // 前面完整处理的核数
TILING_DATA_FIELD_DEF(int64_t, dSplitTime);       // D 维切分次数
TILING_DATA_FIELD_DEF(int64_t, dOnceDealing);     // 每次 D 维处理大小 (= 2048)
TILING_DATA_FIELD_DEF(int64_t, dLastDealing);     // 最后一次 D 维处理大小
END_TILING_DATA_DEF;
```

#### 5.3.2 多核并行策略 (Batch 维度切分)

```
token 维度 (num_tokens) 在多个 AI Core 之间并行

  核0    核1    核2    ...    核N-1
┌─────┐┌─────┐┌─────┐       ┌───────┐
│ t0  ││ tK  ││ t2K │       │ t(N-1)K│
│ t1  ││ tK+1││ t2K+1│      │        │
│ ... ││ ... ││ ... │       │ 尾部   │
│ tK-1││ t2K-1││ t3K-1│     │ 少一点 │
└─────┘└─────┘└─────┘       └───────┘
 ← K → ← K → ← K →
  frontCore 个核，每个处理 batchOneCore 个 token
                    后面的核处理 batchOneCoreTail 个 token
```

**计算方式** (源码 DoOpTiling):
```cpp
useCoreNum     = min(num_tokens, coreNum)
batchOneCore   = ceil(num_tokens / useCoreNum)
batchOneCoreTail = batchOneCore - 1
frontCore      = num_tokens - batchOneCoreTail * useCoreNum
```

**说明**:
- 前 `frontCore` 个核，每个处理 `batchOneCore` 个 token
- 剩下的核，每个处理 `batchOneCoreTail` 个 token
- 这样可以保证负载均衡（尾部核只少 1 个 token）

#### 5.3.3 D 维度切分策略

```
hidden_size 维度 (dParam) 在单核内部分块处理

  dSplitTime 个完整块（每块 2048） + 1 个尾部块
┌──────────┬──────────┬──────────┬───────────┐
│  2048    │  2048    │  ...     │  dLast    │
│  块0     │  块1     │          │  尾部块   │
└──────────┴──────────┴──────────┴───────────┘
 ← dOnceDealing →
```

**常量定义**:
```cpp
constexpr int64_t DEFAULT_DEAL_DPARAM = 2048;  // 每块默认 2048
```

**计算方式**:
```cpp
dSplitTime    = dParam / DEFAULT_DEAL_DPARAM   // 完整块的数量
dOnceDealing  = DEFAULT_DEAL_DPARAM            // 每块大小 = 2048
dLastDealing  = dParam - dSplitTime * DEFAULT_DEAL_DPARAM  // 尾部块大小
```

**为什么要 D 维切分？**
- 片上存储 (UB) 大小有限，无法一次放下整个 hidden_size 的数据
- 切分成 2048 的小块，每块计算完成后写回 GM，再处理下一块
- 配合双缓冲队列，可以实现"计算当前块 + 预取下一块"的流水

#### 5.3.4 平台适配

```cpp
if (socVersion == ASCEND950) {
    HcPostTilingRegbase tiling(context);   // arch35 版本
    return tiling.RunTilingRegbase();
}
HcPostTiling tiling(context);               // 通用版本
return tiling.RunTiling();
```

> ascend950 使用 `HcPostTilingRegbase`（arch35 架构），其他平台使用通用 `HcPostTiling`。

***

## 六、Device 端 NPU 内核详解

Device 端在 NPU 硬件上执行，负责实际的数据搬运和计算。

### 6.1 内核源码结构

```
csrc/moe/hc_post/op_kernel/
├── hc_post.cpp             # 内核入口函数 (Init + Process)
├── hc_post_d_split.h       # D维切分主类（数据搬运+计算流水线）
├── hc_post_bfloat16.h      # bfloat16 计算核心 (DoMulAndAdd)
└── hc_post_float32.h       # float32 计算核心 (DoMulAndAdd)
```

### 6.2 内核入口 (hc_post.cpp)

```cpp
// 内核主函数
extern "C" __global__ __aicore__ void hc_post(
    GM_ADDR x, GM_ADDR residual, GM_ADDR post, GM_ADDR comb, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tilingData)
{
    HcPostKernelDSplit<T1, T2> kernel;
    kernel.Init(x, residual, post, comb, y, workspace, tilingData, pipe);
    kernel.Process();
}
```

执行流程：
1. `Init()`: 初始化全局指针、缓冲区、队列
2. `Process()`: 执行数据搬运 + 计算的完整流水线

### 6.3 D 维切分主类 (HcPostKernelDSplit)

#### 6.3.1 类成员变量

```cpp
class HcPostKernelDSplit {
private:
    TPipe* pipe_;               // 流水线控制
    const HcPostTilingData* tiling_;  // tiling 参数
    
    int32_t blkIdx_ = -1;       // 当前核索引
    int64_t batch_ = 0;         // 当前核处理的 token 数
    int64_t hcParam_ = 0;       // hc_mult
    int64_t dParam_ = 0;        // hidden_size
    int64_t dOnceDealing_ = 0;  // 每块 D 维大小 (=2048)
    int64_t dLastDealing_ = 0;  // 尾部 D 维大小
    int64_t dSplitTime_ = 0;    // D 维切分次数
    int64_t isFrontCore_ = 0;   // 是否是前 frontCore 个核
    
    // 全局内存指针
    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> residualGm_;
    GlobalTensor<T2> postGm_;
    GlobalTensor<T2> combGm_;
    GlobalTensor<T1> yGm_;
    
    // 队列（双缓冲）
    TQue<QuePosition::VECIN, 1> inputQue_;    // 输入队列 (residual/x)
    TQue<QuePosition::VECIN, 1> postQue_;     // post 队列
    TQue<QuePosition::VECIN, 1> combQue_;     // comb 队列
    TQue<QuePosition::VECOUT, 1> outQue_;     // 输出队列
    
    // 计算缓冲区
    TBuf<QuePosition::VECCALC> inputCastBuf_; // 输入类型转换缓冲区
    TBuf<QuePosition::VECCALC> postCastBuf_;  // post 类型转换缓冲区
    TBuf<QuePosition::VECCALC> combCastBuf_;  // comb 类型转换缓冲区
    TBuf<QuePosition::VECCALC> outCastBuf_;   // 输出类型转换缓冲区
    TBuf<QuePosition::VECCALC> tempSumBuf_;   // 临时累加缓冲区
    TBuf<QuePosition::VECCALC> postBrcbBuf_;  // post 广播缓冲区
    TBuf<QuePosition::VECCALC> combBrcbBuf_;  // comb 广播缓冲区
};
```

#### 6.3.2 Init 初始化

```cpp
__aicore__ inline void Init(...) {
    blkIdx_ = GetBlockIdx();  // 获取当前核编号
    if (blkIdx_ >= tilingData->usedCoreNum) return;  // 多余的核直接退出
    
    // 读取 tiling 参数
    hcParam_ = tilingData->hcParam;
    dParam_ = tilingData->dParam;
    dOnceDealing_ = tilingData->dOnceDealing;
    dLastDealing_ = tilingData->dLastDealing;
    dSplitTime_ = tilingData->dSplitTime;
    isFrontCore_ = blkIdx_ < tilingData->frontCore;
    
    // 计算当前核的 GM 偏移（根据是否是 frontCore）
    if (isFrontCore_) {
        xOffset = blkIdx_ * batchOneCore * dParam;
        residualOffset = blkIdx_ * batchOneCore * hcParam * dParam;
        postOffset = blkIdx_ * batchOneCore * hcParam;
        combOffset = blkIdx_ * batchOneCore * hcParam * hcParam;
        yOffset = blkIdx_ * batchOneCore * hcParam * dParam;
    } else {
        xOffset = (blkIdx_ * batchOneCoreTail + frontCore) * dParam;
        residualOffset = (blkIdx_ * batchOneCoreTail + frontCore) * hcParam * dParam;
        postOffset = (blkIdx_ * batchOneCoreTail + frontCore) * hcParam;
        combOffset = (blkIdx_ * batchOneCoreTail + frontCore) * hcParam * hcParam;
        yOffset = (blkIdx_ * batchOneCoreTail + frontCore) * hcParam * dParam;
    }
    
    // 设置全局张量指针
    xGm_.SetGlobalBuffer((__gm__ T1 *)x + xOffset);
    residualGm_.SetGlobalBuffer((__gm__ T1 *)residual + residualOffset);
    postGm_.SetGlobalBuffer((__gm__ T2 *)post + postOffset);
    combGm_.SetGlobalBuffer((__gm__ T2 *)comb + combOffset);
    yGm_.SetGlobalBuffer((__gm__ T1 *)y + yOffset);
    
    // 初始化队列（双缓冲，深度为 2）
    pipe_->InitBuffer(postQue_, 2, hcParam_ * sizeof(T2));
    pipe_->InitBuffer(combQue_, 2, hcParam_ * hcParamAlign_ * sizeof(T2));
    pipe_->InitBuffer(outQue_, 2, hcParam_ * dOnceDealing_ * sizeof(T1));
    pipe_->InitBuffer(inputQue_, 2, hcParam_ * dOnceDealing_ * sizeof(T1));
    
    // 如果是 bfloat16/float16，额外初始化类型转换缓冲区
    if constexpr (sizeof(T1) == 2) {
        pipe_->InitBuffer(outCastBuf_, hcParam_ * dOnceDealing_ * sizeof(float));
        pipe_->InitBuffer(inputCastBuf_, hcParam_ * dOnceDealing_ * sizeof(float));
    }
    // ... post/comb 同理
}
```

#### 6.3.3 Process 主流程 (数据搬运 + 计算流水线)

```
┌──────────────────────────────────────────────────────────────────┐
│  每个 token 的处理流程（外层循环 token）                          │
│                                                                  │
│  for batchIndex in 0..batchSize-1:                               │
│                                                                  │
│      ┌──────────────────────────────────────────────────────┐   │
│      │  step 0: 加载 post 和 comb (只加载一次，因为每个      │   │
│      │          token 的 post/comb 不随 D 维变化)            │   │
│      │  DataCopyInPost(batchIndex)                           │   │
│      │  DataCopyInComb(batchIndex)                           │   │
│      └──────────────────────┬───────────────────────────────┘   │
│                             │                                    │
│      ┌──────────────────────▼───────────────────────────────┐   │
│      │  D 维分块循环（内层循环 D 块）                         │   │
│      │                                                       │   │
│      │  for dLoop in 0..dSplitTime:                          │   │
│      │      ┌──────────┐  ┌──────────┐  ┌──────────┐       │   │
│      │      │ 搬入     │→│  计算    │→│ 搬出     │       │   │
│      │      │ residual │  │DoMulAndAdd│  │ 结果     │       │   │
│      │      │ + x      │  │          │  │          │       │   │
│      │      └──────────┘  └──────────┘  └──────────┘       │   │
│      │      (双缓冲流水：算第i块时，预取第i+1块)               │   │
│      │                                                       │   │
│      │  最后处理尾部块 (dLastDealing)                        │   │
│      └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

**关键优化**：
- **post/comb 重用**: 每个 token 的 post/comb 在 D 维循环外只加载一次，因为它们不随 hidden 维度变化
- **双缓冲队列**: residual/x 的加载使用双缓冲，计算第 i 块时预取第 i+1 块，隐藏访存延迟
- **向量化计算**: 以 256 bit（8 个 float32）为单位做向量运算

### 6.4 核心计算函数 DoMulAndAdd

源码位置: `hc_post_bfloat16.h:L205-309`

**函数签名**:
```cpp
__aicore__ inline void DoMulAndAdd(
    LocalTensor<T1> xUb,           // 主路径输入: (hidden_size,)
    LocalTensor<T2> postUb,        // post 门控: (hc_mult,)
    LocalTensor<T1> residualUb,    // 残差输入: (hc_mult, hidden_size)
    LocalTensor<T2> combUb,        // comb 矩阵: (hc_mult, hc_mult)
    LocalTensor<float> sumTempBuf, // 临时累加缓冲区
    int64_t dOnceDealing           // 当前处理的 D 维大小
)
```

#### 6.4.1 计算逻辑（源码级精确）

内核中是对每个输出通道 j（hcIndex = j）分别计算：

```cpp
// 输入张量 shape (单个 token):
//   residual: (hc_mult, hidden_size)    ← hc_mult个输入通道，每通道hidden_size维
//   x:        (hidden_size,)            ← 主路径输出，1个通道，hidden_size维
//   comb:     (hc_mult, hc_mult)        ← hc_mult×hc_mult 组合矩阵
//   post:     (hc_mult,)                ← hc_mult个门控权重
//   y:        (hc_mult, hidden_size)    ← hc_mult个输出通道，每通道hidden_size维

for each output channel j:   // j = 0, 1, 2, 3
    // 加载 comb 矩阵的第 j 列  → shape: (hc_mult,)
    comb[0,j], comb[1,j], comb[2,j], comb[3,j]

    // 加载 post 的第 j 个值  → 标量
    post[j]

    // 计算第 j 个输出通道  → shape: (hidden_size,)
    y[j] = residual[0] * comb[0,j]   // 第0个输入通道 × comb[0,j]
         + residual[1] * comb[1,j]   // 第1个输入通道 × comb[1,j]
         + residual[2] * comb[2,j]   // 第2个输入通道 × comb[2,j]
         + residual[3] * comb[3,j]   // 第3个输入通道 × comb[3,j]
         + x * post[j]               // 主路径输出 × post[j]
```

> **源码实际计算顺序**：
> 内核中为了性能优化，加法顺序略有调整（先算通道0，再加通道3、1、2，最后加主路径），数学上完全等价。

#### 6.4.2 向量计算细节

- **计算单位**: 以 256 bit（即 8 个 float32）为单位做向量化计算
- **类型转换**: 如果输入是 bfloat16，先转成 float32 计算，最后再转回 bfloat16
- **寄存器使用**: 所有计算都在寄存器中完成，使用 AscendC MicroAPI（Muls、Adds 等指令）

***

## 七、性能优化要点总结

### 7.1 并行层次

```
┌──────────────────────────────────────────────────────┐
│  1. 核间并行: token 维度在多个 AI Core 间切分         │
│     (tiling: usedCoreNum, batchOneCore, frontCore)    │
│                                                      │
│  2. 核内流水: D 维切分 + 双缓冲                       │
│     (tiling: dSplitTime, dOnceDealing=2048)           │
│                                                      │
│  3. 向量级并行: 256 bit 向量指令 (8 float32)          │
│     (AscendC MicroAPI 向量化运算)                     │
└──────────────────────────────────────────────────────┘
```

### 7.2 访存优化

1. **post/comb 重用**: 每个 token 的 post/comb 只加载一次，在 D 维循环内重复使用
2. **双缓冲队列**: residual/x 的加载使用双缓冲，计算与访存重叠
3. **数据局部性**: D 维切分大小 2048 是精心选择的，确保数据能放入片上 UB

### 7.3 计算优化

1. **全寄存器计算**: 核心计算全部在寄存器中完成，无额外访存
2. **类型转换优化**: 仅在输入输出处做类型转换，中间计算全用 float32
3. **指令调度**: 加法顺序经过优化，充分利用寄存器和流水线

***

## 八、hc_pre 与 hc_post 的对应关系

```
hc_pre (压缩):
  输入: (num_tokens, hc_mult, hidden_size)
  过程: pre_gate 加权求和 → (num_tokens, hidden_size)
  输出: y, post, comb
        ↑     ↑     ↑
        │     │     └── 给 hc_post 用的组合矩阵
        │     └──────── 给 hc_post 用的门控权重
        └────────────── 送给注意力/FFN计算的主路径特征

hc_post (扩展+混合):
  输入: x (num_tokens, hidden_size)           ← 注意力/FFN的输出
        residual (num_tokens, hc_mult, hidden_size)  ← hc_pre前的残差
        post (num_tokens, hc_mult)            ← hc_pre生成的门控
        comb (num_tokens, hc_mult, hc_mult)   ← hc_pre生成的组合矩阵
  过程: residual @ comb + x * post(广播)
  输出: (num_tokens, hc_mult, hidden_size)    ← 恢复HC维度，供下一层用
```

**关键点**:
- hc_pre 用 pre_gate 把 hc_mult 个通道压缩成 1 个（做"编码"）
- hc_post 用 comb 把残差的 hc_mult 个通道重新混合（做"通道重组"）
- 主路径 x 只通过 post 门控做简单的缩放扩展
- 两者不是简单的"编解码"关系，而是更复杂的双通道结构

***

## 九、大白话理解

```
第 j 个输出通道
     = (所有输入通道按 comb 第 j 列加权求和)  ← 残差通道混合
     + (主路径输出 × post 第 j 个门控)         ← 主路径扩展+门控
```

类比：
- 残差通道混合 = 把 4 个输入通道的信息，按照 comb 矩阵的权重，重新调配成 4 个输出通道
- 主路径扩展 = 把注意力/FFN算出来的结果，按 post 门控放大/缩小后分配到每个通道
- 两者相加 = 最终的输出 = 重组后的残差 + 缩放后的主路径
