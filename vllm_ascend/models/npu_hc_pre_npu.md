# `hc_pre` 算子

## hc_pre 算子概述

### 函数接口

```python
def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, 
           hc_scale: torch.Tensor, hc_base: torch.Tensor):
    y = torch.ops._C_ascend.npu_hc_pre(
        x, hc_fn, hc_scale, hc_base, 
        self.hc_mult, self.hc_sinkhorn_iters, 
        self.norm_eps, self.hc_eps
    )
    return y
```

### 输入输出形状

| 张量                | 形状                                   | dtype    | 说明                                                      |
| ----------------- | ------------------------------------ | -------- | ------------------------------------------------------- |
| `x` (输入)          | `(num_tokens, hc_mult, hidden_size)` | bfloat16 | HC扩展后的隐藏状态，每个token有hc\_mult个通道，每个通道hidden\_size维特征      |
| `hc_fn`           | `(mix_hc, hc_mult * hidden_size)`    | float32  | 门控线性投影权重，把 hc\_mult \* hidden\_size 维映射到 mix\_hc 维的门控参数 |
| `hc_scale`        | `(3,)`                               | float32  | 门控缩放因子                                                  |
| `hc_base`         | `(mix_hc,)`                          | float32  | 门控偏置                                                    |
| `y` (输出0)         | `(num_tokens, hidden_size)`          | bfloat16 | 压缩后的主路径特征，送入后续Norm + Attention/FFN                      |
| `post` (输出1)      | `(num_tokens, hc_mult)`              | float32  | 后处理门控权重，传递给 `hc_post`                                   |
| `comb_frag` (输出2) | `(num_tokens, hc_mult, hc_mult)`     | float32  | 组合矩阵片段，传递给 `hc_post`                                    |

### 符号约定

- `num_tokens`: token 数量（即 batch size，一次处理的 token 数量）
- `hc_mult`: HC 通道数（通常为 4）
- `hidden_size`: 隐藏层维度（dsv4为 7168）
- `mix_hc`: 混合门控数，`mix_hc = (2 + hc_mult) * hc_mult = 24`
- `hc_sinkhorn_iters`: Sinkhorn 迭代次数（dsv4默认 20）

### 算子注册

TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops) -> TORCH_LIBRARY(_C_ascend, ops)
PyTorch 的核心注册宏，用于向 PyTorch 框架注册一个名为 _C_ascend 的 自定义算子库

- _C_ascend ：算子库的名称，Python 端可以通过 torch.ops._C\ascend.* 访问
- ops ：lambda 参数名，用于在花括号内定义算子（如 ops.def(...) ）
  这个注册块内包含了 vLLM-Ascend 项目的所有 NPU 自定义算子，比如：

```python
    ops.def(
        "npu_hc_pre("
            "Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, "
            "int hc_mult, int hc_sinkhorn_iters, "
            "float norm_eps, float hc_eps"
        ") -> (Tensor out0, Tensor out1, Tensor out2)"
        );
    ops.impl("npu_hc_pre", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_npu);

    ops.def(
        "npu_hc_pre_v2("
            "Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, "
            "int hc_mult, int hc_sinkhorn_iters, "
            "float norm_eps, float hc_eps"
        ") -> (Tensor out0, Tensor out1, Tensor out2)"
        );
    ops.impl("npu_hc_pre_v2", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_v2_npu);


std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_npu(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, hc_mult);
    return run_hc_pre_composite(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_v2_npu(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, hc_mult);
    return run_hc_pre_fusion(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
}
```

### 算子内部实现流程

(`run_hc_pre_composite` in `csrc/torch_binding.cpp#L1438`)

```
1. InvRMS 归一化，调用算子: `aclnnHcPreInvRms`
   └── rsqrt = 1 / sqrt(mean(x²) + norm_eps)    # 形状: (num_tokens, 1)

2. 线性投影计算 mixes
   ├── x_flat = x.float().flatten(1)            # (num_tokens, hc_mult * hidden_size)
   └── mixes = x_flat @ hc_fn.T                 # (num_tokens, mix_hc)

3. Sinkhorn 归一化 + 门控生成
   ├── 输入: mixes, rsqrt, hc_scale, hc_base
   ├── 通过 Sinkhorn 迭代 (hc_sinkhorn_iters次) 生成双随机矩阵
   ├── 计算 post 门控权重: sigmoid(mixes * scale + base) + hc_eps
   ├── 计算 comb_frag 组合矩阵: (num_tokens, hc_mult, hc_mult)
   └── 输出: y (压缩后特征), post, comb_frag

4. 输出类型转换
   └── y = y.to(original_dtype)                 # 转回 bfloat16
```

---

## 步骤1：InvRMS 归一化子算子

### 算子原理

调用算子: `aclnnHcPreInvRms`

#### 功能说明

计算每个 token 的"均方根倒数"，对每个 token，计算 hc_mult * hidden_size 个元素的均方根倒数，即：把 (num_tokens, hc_mult, hidden_size) 的后两维全部"压缩"成一个标量，用于后续的归一化。

简单说：对每个 token，先算所有元素平方的平均值，开根号，再取倒数。

#### 计算公式

```
对于第 b 个 token:
  rsqrt[b] = 1 / sqrt( mean(x[b,:,:]^2) + norm_eps )

shape 说明:
  x.shape      = (num_tokens, hc_mult, hidden_size)  # bfloat16， 每个 token 有 hc_mult * hidden_size 个值
  rsqrt.shape  = (num_tokens, 1)                      # float32， 每个 token 只有 1 个值
```

### AscendC 算子架构

源码位置: `csrc/moe/hc_pre_inv_rms/`

遵循 AscendC 的 Host-Kernel 分离架构

```
┌─────────────────────────────────────────────────────────────┐
│                    AscendC Operator Architecture            │
├─────────────────────────────────────────────────────────────┤
│  Host 侧 (CPU)                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ hc_pre_inv_rms_def.cpp   → 算子定义 (Input/Output/Attr) │   │
│  │ hc_pre_inv_rms_tiling.cpp → Tiling 策略计算            │   │
│  │ hc_pre_inv_rms_proto.cpp  → 算子原型定义               │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓ 编译时生成 TilingData              │
│  Kernel 侧 (AI Core)                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ hc_pre_inv_rms.cpp       → 核函数入口                  │   │
│  │ hc_pre_inv_rms_full_load.h → 核心计算逻辑             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 源码目录结构

```
csrc/moe/hc_pre_inv_rms/
├── op_host/                              # Host 侧（CPU）
│   ├── hc_pre_inv_rms_def.cpp            # 算子定义：输入/输出/属性声明
│   ├── hc_pre_inv_rms_proto.cpp          # 算子原型定义
│   ├── hc_pre_inv_rms_tiling.cpp         # Tiling 策略计算核心逻辑
│   ├── hc_pre_inv_rms_tiling.h           # Tiling 基类定义
│   ├── hc_pre_inv_rms_tiling_arch35.h    # Ascend 950 专用 Tiling
│   └── hc_pre_inv_rms_tiling_large_d.h   # 大 R 值专用 Tiling（R=28672）
└── op_kernel/                            # Kernel 侧（AI Core）
    ├── hc_pre_inv_rms.cpp                # 核函数入口（选择计算策略）
    ├── hc_pre_inv_rms_full_load.h        # 普通模式核心计算逻辑
    ├── hc_pre_inv_rms_full_load_large_d.h # 大 R 值优化计算逻辑
    └── hc_pre_inv_rms_full_load_regbase.h # Ascend 310P 专用计算逻辑
```

### Host 侧实现

#### 算子定义

`hc_pre_inv_rms_def.cpp`：定义算子的输入、输出、属性和支持的平台

```cpp
class HcPreInvRms : public OpDef {
public:
    explicit HcPreInvRms(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        
        this->Attr("epsilon")
            .AttrType(OPTIONAL)
            .Float(1e-6f);
        
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        
        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "hc_pre_inv_rms");
        this->AICore().AddConfig("ascend950", regbaseCfg);
    }
};
OP_ADD(HcPreInvRms);
```

**`OP_ADD` 宏的作用**：
- 将 `HcPreInvRms` 算子注册到 CANN 的算子注册表中
- 在编译时自动生成 `aclnnHcPreInvRms` 和 `aclnnHcPreInvRmsGetWorkspaceSize` 函数
- 这些函数会被打包到 `libvllm_ascend_op.so` 共享库中

#### Tiling 策略

`hc_pre_inv_rms_tiling.cpp`：计算数据切分策略和内存分配方案

##### 核心逻辑

将输入张量从三维 `(num_tokens, hc_mult, hidden_size)` 展平为二维 `(A, R)`，其中：

- `A = num_tokens`（batch 维度，每个 token 作为一个独立的计算单元）
- `R = hc_mult * hidden_size`（特征维度，每个 token 的所有通道特征合并）

> **为什么这样展平？**
> 因为 InvRMS 是对**每个 token 内部的所有元素**求均方根倒数，所以：
>
> - A 轴（第0维）是 token 数量，可以并行切分到不同的 AI Core
> - R 轴（第1、2维合并）是每个 token 的特征总数，在单个 AI Core 内做归约计算
>
> 举个例子：`x.shape = (8, 4, 4096)` → 展平为 `(A=8, R=16384)`

##### 输入维度处理

```cpp
size_t xDimNum = xShape_->GetDimNum();

if (xDimNum == X_INPUT_DIMS) {           // 4维输入场景
    A_ = xShape_->GetDim(DIM_0) * xShape_->GetDim(DIM_1);
    R_ = xShape_->GetDim(DIM_2) * xShape_->GetDim(DIM_3);
} else if (xDimNum == X_INPUT_BS_FUSED_DIMS) {  // 3维输入场景（我们的场景）
    // x.shape = (num_tokens, hc_mult, hidden_size)
    A_ = xShape_->GetDim(DIM_0);             // A = num_tokens
    R_ = xShape_->GetDim(DIM_1) * xShape_->GetDim(DIM_2);  // R = hc_mult * hidden_size
}
```

##### Tiling 关键参数

| 参数             | 含义                                             |
| -------------- | ---------------------------------------------- |
| `A`            | A轴大小 = num_tokens（token 总数）                   |
| `R`            | R轴大小 = hc_mult × hidden_size（每个 token 的特征总数） |
| `blockNumA`    | 使用的 AI Core 核数                                 |
| `blockFactorA` | 每个核处理的 token 数                                 |
| `ubFactorA`    | 每次 UB 循环处理的 token 数                            |

##### A 轴切分策略

按 token 切分到多个 AI Core：

```cpp
int64_t blockFactorA = CeilDiv(A_, static_cast<int64_t>(coreNum_));
int64_t blockNumA = CeilDiv(A_, blockFactorA);
int64_t blockTailFactorA = A_ % blockFactorA == 0 ? blockFactorA : A_ % blockFactorA;
```

**举个例子**：

- A = 100 个 token，coreNum = 32 个 AI Core
- blockFactorA = ceil(100/32) = 4 个 token/核
- blockNumA = ceil(100/4) = 25 个核被使用
- 前 24 个核各处理 4 个 token，第 25 个核处理 100 - 24×4 = 4 个 token

##### UB 循环因子计算

UB（Unified Buffer）是 AI Core 的高速片上内存，`ubFactorA` 决定每次从 GM（Global Memory）加载多少个 token 的数据到 UB。计算公式基于 UB 内存容量和每个 token 需要的内存空间。

```cpp
int64_t rAlignSize = CeilAlign(R_ * inputDtypeSize_, UB_BLOCK_SIZE);
int64_t ubFactorA = 1;
if (inputDtypeSize_ == B16_TYPE_BYTE_SIZE) {
    ubFactorA = ubSize_ / (4 * rAlignSize + 2 * outputDtypeSize_ + R_ / 16);
} else if (inputDtypeSize_ == B32_TYPE_BYTE_SIZE) {
    ubFactorA = ubSize_ / (2 * rAlignSize + 2 * outputDtypeSize_ + R_ / 16);
}
```

##### UB 内存分配分析

从 Kernel 代码（`hc_pre_inv_rms_full_load.h:98-105`）可以看到，每个 token 在 UB 中需要分配以下缓冲区：

| 缓冲区 | bf16/fp16 场景 | fp32 场景 | 说明 |
| ------ | -------------- | --------- | ---- |
| `inQueueX` | `ubFactorA * rAlign * 2` | `ubFactorA * rAlign * 4` | 输入队列（双缓冲 × 2） |
| `outQueueY` | `ubFactorA * 4` | `ubFactorA * 4` | 输出队列（双缓冲 × 2） |
| `reduceBuf` | `ubFactorA * (R/64) * 4` | `ubFactorA * (R/64) * 4` | 归约中间缓冲区 |
| `castBuf` | `ubFactorA * R * 4` | 无 | bf16→fp32 类型转换缓冲区 |

##### 公式推导

将上述缓冲区合并（以字节为单位）：

**bf16/fp16 场景**：
```
总内存 = 2 * (ubFactorA * rAlignSize)    // inQueueX 双缓冲
       + 2 * (ubFactorA * 4)             // outQueueY 双缓冲  
       + ubFactorA * (R / 16) * 4        // reduceBuf (R/64 * 4 ≈ R/16)
       + ubFactorA * rAlignSize          // castBuf (bf16→fp32，大小 = R*4)

简化后每 token 内存 = 4 * rAlignSize + 8 + R/4
```

**fp32 场景**（无 castBuf）：
```
总内存 = 2 * (ubFactorA * rAlignSize)    // inQueueX 双缓冲
       + 2 * (ubFactorA * 4)             // outQueueY 双缓冲
       + ubFactorA * (R / 16) * 4        // reduceBuf

简化后每 token 内存 = 2 * rAlignSize + 8 + R/4
```

##### 具体示例

假设场景：
- `R = 4 * 4096 = 16384`（hc_mult=4, hidden_size=4096）
- `inputDtypeSize_ = 2`（bf16）
- `outputDtypeSize_ = 4`（fp32）
- `ubSize_ = 32 * 1024 * 1024 = 33554432`（Ascend 910B 的 UB 容量约 32MB）
- `UB_BLOCK_SIZE = 32`（内存对齐单位）

**计算步骤**：

1. 计算 `rAlignSize`：
```
rAlignSize = CeilAlign(R * inputDtypeSize_, UB_BLOCK_SIZE)
           = CeilAlign(16384 * 2, 32)
           = CeilAlign(32768, 32)
           = 32768 字节（刚好对齐）
```

2. 代入 bf16 公式：
```
ubFactorA = ubSize_ / (4 * rAlignSize + 2 * outputDtypeSize_ + R_ / 16)
          = 33554432 / (4 * 32768 + 2 * 4 + 16384 / 16)
          = 33554432 / (131072 + 8 + 1024)
          = 33554432 / 132104
          ≈ 254
```

**结果说明**：
- 每次 UB 循环可以处理 **254 个 token**
- 如果某个 AI Core 分配了 `blockFactorA = 4` 个 token，只需 **1 次 UB 循环** 即可完成
- 如果 `blockFactorA = 1000`，则需要 `ceil(1000 / 254) = 4` 次 UB 循环

> **为什么需要 UB 循环？**
> 因为 R = hc_mult × hidden_size 可能很大（如 4×4096 = 16384），一个 token 的数据加上中间缓冲区可能超过 UB 容量，所以需要分多次加载处理。

##### 特殊场景处理

```cpp
auto socVersion = ascendcPlatform.GetSocVersion();
if (socVersion == platform_ascendc::SocVersion::ASCEND950) {
    // Ascend 950 使用动态编译模式
    HcPreInvRmsRegbase::HcPreInvRmsTilingRegbase hcPreInvRmsTilingRegbase(context);
    return hcPreInvRmsTilingRegbase.DoOpTiling();
}

if (R == 28672) {
    // 大 R 值场景使用专用优化
    HcPreInvRmsLargeD::HcPreInvRmsTilingLargeD invRmsTilingLargeD(context);
    return invRmsTilingLargeD.DoOpTiling();
}
```

### Kernel 侧实现

#### 核函数入口

`hc_pre_inv_rms.cpp`：AI Core 核函数入口，根据 TilingKey 选择不同的计算策略

```cpp
extern "C" __global__ __aicore__ void hc_pre_inv_rms(
    GM_ADDR x,      // 输入张量 GM 地址
    GM_ADDR y,      // 输出张量 GM 地址
    GM_ADDR workspace,  // 工作空间（本算子未使用）
    GM_ADDR tiling      // TilingData GM 地址
) {
    TPipe pipe;                        // 创建流水线控制器
    GET_TILING_DATA(tilingData, tiling); // 从 GM 读取 TilingData
    
    // 根据 TilingKey 选择不同的计算策略
    if (TILING_KEY_IS(FULL_LOAD_TILING_KEY)) {
        // 普通模式：数据一次性加载到 UB
        HcPreInvRmsFullLoad<DTYPE_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(FULL_LOAD_LARGE_D_TILING_KEY)) {
        // 大 R 值优化模式（如 R=28672）
        HcPreInvRmsFullLoadLargeD<DTYPE_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    }
    #if defined(__DAV_C310__)
      else if (TILING_KEY_IS(REGBASE_FULL_LOAD_TILING_KEY)) {
        // Ascend 310P 专用优化模式
        HcPreInvRmsFullLoadRegbase<DTYPE_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
      }
    #endif
}
```

**关键关键字**：
- `__aicore__`：标记函数运行在 AI Core 上，不是 CPU
- `extern "C"`：确保函数名不被 C++ 编译器修改，方便动态加载

#### 核心计算逻辑

`hc_pre_inv_rms_full_load.h`：实现完整的计算流程（GM→UB→计算→UB→GM）

##### 常量定义

```cpp
constexpr int32_t BUFFER_NUM = 2;           // 队列缓冲区数量（双缓冲，实现流水线）
constexpr int32_t FLOAT_BTYPE_SIZE = 4;     // float32 占 4 字节
constexpr uint32_t PER_REPEAT_LEN_B32 = 64; // 每次重复计算的 float32 元素数（AI Core 指令粒度）
constexpr uint32_t UB_BLOCK_SIZE = 32;      // UB 内存对齐块大小（32字节）
constexpr int32_t B16_TYPE_BYTE_SIZE = 2;   // bf16/fp16 占 2 字节
constexpr int32_t B32_TYPE_BYTE_SIZE = 4;   // float32 占 4 字节
constexpr int32_t FOUR_FOLD = 4;            // 4-fold 并行，把 R 轴分成 4 份并行归约
```

##### 类结构定义

```cpp
template <typename T>
class HcPreInvRmsFullLoad {
public:
    __aicore__ inline HcPreInvRmsFullLoad() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, 
                                const HcPreInvRmsFullLoadTilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(uint64_t idx, uint64_t curUbFactorA);
    __aicore__ inline void Compute(uint64_t curUbFactorA);
    __aicore__ inline void ComputeB16(uint64_t curUbFactorA);
    __aicore__ inline void ComputeB32(uint64_t curUbFactorA);
    __aicore__ inline void CopyOut(uint64_t idx, uint64_t curUbFactorA);

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<TPosition::VECCALC> castBuf;
    TBuf<TPosition::VECCALC> reduceBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<float> yGm;
    int64_t A, R, blockNumA, blockFactorA, blockTailFactorA, ubFactorA;
    int32_t blockIdx_;
    float epsilon;
    uint32_t curBlockFactorA, rAlign, rAlignB32, reduceBufNum;
};
```

##### Init 方法详解

```cpp
__aicore__ inline void HcPreInvRmsFullLoad<T>::Init(...) {
    A = tiling->A;
    R = tiling->R;
    blockNumA = tiling->blockNumA;
    blockFactorA = tiling->blockFactorA;
    blockTailFactorA = tiling->blockTailFactorA;
    ubFactorA = tiling->ubFactorA;
    epsilon = tiling->epsilon;
    
    rAlign = ((R * sizeof(T) + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE) 
             * (UB_BLOCK_SIZE / sizeof(T));
    rAlignB32 = ((R * FLOAT_BTYPE_SIZE + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE) 
                * (UB_BLOCK_SIZE / FLOAT_BTYPE_SIZE);
    
    blockIdx_ = GetBlockIdx();
    
    if (blockIdx_ < blockNumA - 1) {
        curBlockFactorA = blockFactorA;
    } else if (blockIdx_ == blockNumA - 1) {
        curBlockFactorA = blockTailFactorA;
    } else {
        return;
    }
    
    xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * blockFactorA * R, 
                        curBlockFactorA * R);
    yGm.SetGlobalBuffer((__gm__ float*)y + blockIdx_ * blockFactorA, 
                        curBlockFactorA);
    
    pipe_->InitBuffer(inQueueX, BUFFER_NUM, ubFactorA * rAlign * sizeof(T));
    pipe_->InitBuffer(outQueueY, BUFFER_NUM, ubFactorA * FLOAT_BTYPE_SIZE);
    reduceBufNum = (rAlignB32 + PER_REPEAT_LEN_B32 - 1) / PER_REPEAT_LEN_B32;
    pipe_->InitBuffer(reduceBuf, ubFactorA * reduceBufNum * FLOAT_BTYPE_SIZE);
    if constexpr (sizeof(T) == B16_TYPE_BYTE_SIZE) {
        pipe_->InitBuffer(castBuf, ubFactorA * rAlignB32 * FLOAT_BTYPE_SIZE);
    }
}
```

##### Process 方法详解

```cpp
__aicore__ inline void HcPreInvRmsFullLoad<T>::Process() {
    if (blockIdx_ >= blockNumA) {
        return;
    }
    
    uint64_t aUbLoopCount = (curBlockFactorA + ubFactorA - 1) / ubFactorA;
    uint64_t tailUbFactorA = curBlockFactorA - (aUbLoopCount - 1) * ubFactorA;
    
    uint64_t curUbFactorA = ubFactorA;
    for (uint64_t idx = 0; idx < aUbLoopCount; idx++) {
        if (idx == aUbLoopCount - 1) {
            curUbFactorA = tailUbFactorA;
        }
        CopyIn(idx, curUbFactorA);   // Step 1: GM → UB
        Compute(curUbFactorA);       // Step 2: UB 内计算
        CopyOut(idx, curUbFactorA);  // Step 3: UB → GM
    }
}
```

##### ComputeB16 方法详解（核心计算）

```cpp
__aicore__ inline void HcPreInvRmsFullLoad<T>::ComputeB16(uint64_t curUbFactorA) {
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    LocalTensor<float> castLocal = castBuf.Get<float>();
    LocalTensor<float> reduceLocal = reduceBuf.Get<float>();
    
    int32_t perFoldElems = (rAlignB32 + FOUR_FOLD - 1) / FOUR_FOLD;
    int32_t perFoldRepTime = (perFoldElems + PER_REPEAT_LEN_B32 - 1) / PER_REPEAT_LEN_B32;
    
    // bf16 → fp32 类型转换
    AscendC::Cast(castLocal, xLocal, AscendC::RoundMode::CAST_NONE, R);
    PipeBarrier<PIPE_V>();
    
    // 平方计算 x²
    AscendC::Mul(castLocal, castLocal, castLocal, R);
    
    // 4-fold 并行归约求和
    for (int idx = 0; idx < curUbFactorA; idx++) {
        for (int j = 0; j < FOUR_FOLD; j++) {
            PipeBarrier<PIPE_V>();
            WholeReduceSum(reduceLocal[idx * reduceBufNum + j * perFoldRepTime], 
                           castLocal[idx * rAlignB32 + j * perFoldElems], 
                           PER_REPEAT_LEN_B32, perFoldRepTime,
                           DST_REP_STRIDE, SRC_BLK_STRIDE, SRC_REP_STRIDE);
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(reduceLocal, reduceLocal, PER_REPEAT_LEN_B32, FOUR_FOLD, ...);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(yLocal[idx], reduceLocal, FOUR_FOLD, 1, ...);
    }
    
    // 均值计算
    float meanCof = 1.0f / R;
    PipeBarrier<PIPE_V>();
    AscendC::Muls(yLocal, yLocal, meanCof, curUbFactorA);
    
    // 加 epsilon
    PipeBarrier<PIPE_V>();
    AscendC::Adds(yLocal, yLocal, epsilon, curUbFactorA);
    
    // 开根号
    PipeBarrier<PIPE_V>();
    AscendC::Duplicate(reduceLocal, 1.0f, curUbFactorA);
    PipeBarrier<PIPE_V>();
    AscendC::Sqrt(yLocal, yLocal, curUbFactorA);
    
    // 取倒数
    PipeBarrier<PIPE_V>();
    AscendC::Div(yLocal, reduceLocal, yLocal, curUbFactorA);
    
    outQueueY.EnQue<float>(yLocal);
    inQueueX.FreeTensor(xLocal);
}
```

**4-fold 归约图解**：

```
原始数据 (R=16384 个 float32):
┌─────────────────────────────────────────────────────────┐
│  Fold 0  │  Fold 1  │  Fold 2  │  Fold 3  │           │
│  4096个  │  4096个  │  4096个  │  4096个  │           │
└──────────┴──────────┴──────────┴──────────┘           │
        │          │          │          │              │
        ▼          ▼          ▼          ▼              │
        ┌─────────────────────────────────────┐         │
        │  reduceBuf: 4 × 64 = 256 个部分和    │         │
        │  (每份 4096 个归约成 64 个)            │         │
        └─────────────────────────────────────┘         │
                        │                                │
                        ▼                                │
        ┌─────────────────────────────────────┐         │
        │  reduceBuf: 4 个中间和               │         │
        │  (256 个部分和归约成 4 个)            │         │
        └─────────────────────────────────────┘         │
                        │                                │
                        ▼                                │
        ┌─────────────────────────────────────┐         │
        │  yLocal[idx]: 1 个标量              │         │
        │  (4 个中间和归约成 1 个最终和)        │         │
        └─────────────────────────────────────┘         │
```

### 完整调用链路

#### 第一层：PyTorch C++ 绑定层

```cpp
// torch_binding.cpp:1533
EXEC_NPU_CMD(aclnnHcPreInvRms, x, norm_eps, rsqrt);
```

`EXEC_NPU_CMD` 是一个宏定义，展开后执行以下步骤：

| 步骤 | 代码                                                     | 作用                           |
| -- | ------------------------------------------------------ | ---------------------------- |
| 1  | `GetOpApiFuncAddr("aclnnHcPreInvRms")`                 | 动态查找 `aclnnHcPreInvRms` 函数地址 |
| 2  | `GetOpApiFuncAddr("aclnnHcPreInvRmsGetWorkspaceSize")` | 动态查找获取工作空间大小的函数              |
| 3  | `ConvertTypes(__VA_ARGS__, ...)`                       | 将 PyTorch 张量转换为 ACL 内部数据结构   |
| 4  | 调用 `aclnnHcPreInvRmsGetWorkspaceSize`                  | 获取算子执行所需的工作空间大小              |
| 5  | 分配工作空间张量                                               | 申请 GPU 内存作为算子临时空间            |
| 6  | 通过 `OpCommand` 提交到 NPU 流                               | 将算子调度到 NPU 执行                |

#### 第二层：动态库加载层

`GetOpApiFuncAddr` 函数（`csrc/aclnn_torch_adapter/op_api_common.h:274`）负责动态加载算子 API：

```cpp
inline void *GetOpApiFuncAddr(const char *apiName)
{
    if (!g_custom_lib_path.empty()) {
        for (auto &it : g_custom_lib_path) {
            auto cust_opapi_lib = real_path(it + "/" + GetCustOpApiLibName());
            auto handler = dlopen(cust_opapi_lib.c_str(), RTLD_LAZY);
            auto funcAddr = dlsym(handler, apiName);
            if (funcAddr != nullptr) return funcAddr;
        }
    }
    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    return dlsym(opApiHandler, apiName);
}
```

**关键机制**：
- 使用 `dlopen` + `dlsym` 动态加载共享库
- `GetCustOpApiLibName()` 返回自定义算子库名称（如 `libvllm_ascend_op.so`）
- `GetOpApiLibName()` 返回系统 CANN 算子库名称（如 `libacl_op_api.so`）

#### 第三层：CANN 算子注册层

`aclnnHcPreInvRms` 函数通过 `OP_ADD(HcPreInvRms)` 宏注册到 CANN 框架，编译时自动生成 API 函数。

#### 第四层：Tiling 策略层

`hc_pre_inv_rms_tiling.cpp`：计算 blockFactorA、ubFactorA、核分配等参数。

#### 第五层：AI Core Kernel 层

最终执行 `hc_pre_inv_rms.cpp` → `HcPreInvRmsFullLoad::Process()`。

#### 完整调用链路图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          完整调用链路                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Python 层                                                                 │
│    torch.ops._C_ascend.npu_hc_pre_v2(...)                                  │
│                         │                                                   │
│                         ▼                                                   │
│  PyTorch C++ 绑定层                                                          │
│    torch_binding.cpp: run_hc_pre_composite()                               │
│                         │                                                   │
│                         ▼                                                   │
│    EXEC_NPU_CMD(aclnnHcPreInvRms, x, norm_eps, rsqrt)                      │
│                         │                                                   │
│                         ├── 展开为 GetOpApiFuncAddr("aclnnHcPreInvRms")      │
│                         ├── ConvertTypes() → PyTorch Tensor → ACL Tensor    │
│                         ├── aclnnHcPreInvRmsGetWorkspaceSize()              │
│                         └── OpCommand.Run() → 提交到 NPU 流                  │
│                                                                             │
│                         ▼                                                   │
│  动态库加载层                                                                │
│    GetOpApiFuncAddr() → dlopen(libvllm_ascend_op.so) + dlsym()             │
│                         │                                                   │
│                         ▼                                                   │
│  CANN 算子注册层                                                             │
│    hc_pre_inv_rms_def.cpp: OP_ADD(HcPreInvRms)                             │
│                         │                                                   │
│                         ├── 生成 aclnnHcPreInvRms 函数                      │
│                         └── 生成 aclnnHcPreInvRmsGetWorkspaceSize 函数      │
│                                                                             │
│                         ▼                                                   │
│  Tiling 策略层                                                              │
│    hc_pre_inv_rms_tiling.cpp → 计算 blockFactorA, ubFactorA, 核分配         │
│                         │                                                   │
│                         ▼                                                   │
│  AI Core Kernel 层                                                          │
│    hc_pre_inv_rms.cpp → hc_pre_inv_rms()                                   │
│                         │                                                   │
│                         ▼                                                   │
│    hc_pre_inv_rms_full_load.h → HcPreInvRmsFullLoad::Process()             │
│                         │                                                   │
│                         ├── CopyIn()  → GM → UB                             │
│                         ├── ComputeB16() → 平方 + 归约 + 均值 + 开根号 + 倒数│
│                         └── CopyOut() → UB → GM                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 关键文件汇总

| 文件                                    | 层次          | 作用                             |
| ------------------------------------- | ----------- | ------------------------------ |
| `torch_binding.cpp`                   | PyTorch 绑定层 | 接收 Python 调用，调用 EXEC\_NPU\_CMD |
| `aclnn_torch_adapter/op_api_common.h` | 动态库加载层      | 定义 EXEC\_NPU\_CMD 宏，实现动态函数查找   |
| `hc_pre_inv_rms_def.cpp`              | CANN 算子注册层  | 定义算子输入/输出/属性，注册到 CANN          |
| `hc_pre_inv_rms_tiling.cpp`           | Tiling 策略层  | 计算数据切分策略和内存分配方案                |
| `hc_pre_inv_rms.cpp`                  | Kernel 入口层  | AI Core 核函数入口                  |
| `hc_pre_inv_rms_full_load.h`          | Kernel 计算层  | 核心计算逻辑（UB 循环 + 归约）             |

### 完整计算流程

单个 AI Core 内部，处理 `curBlockFactorA` 个 token：

```
┌──────────────────────────────────────────────────────────────────┐
│                    hc_pre_inv_rms 完整计算流程                    │
├──────────────────────────────────────────────────────────────────┤
│  输入: x (A, R) bf16/fp16/fp32                                   │
│        A = num_tokens 个 token                                    │
│        R = hc_mult * hidden_size  (每个 token 的特征数)            │
│                                                                   │
│  UB 循环：每次加载 ubFactorA 个 token 的数据到 UB                  │
│  共循环 aUbLoopCount = ceil(curBlockFactorA / ubFactorA) 次        │
│                                                                   │
│  ─────────────────────────────────────────────────────────────  │
│  第 idx 次 UB 循环（处理 curUbFactorA 个 token）：                  │
│                                                                   │
│  Step 1: GM → UB  (加载 curUbFactorA 个 token 的数据)              │
│    DataCopyPad(xLocal, xGm[idx * R * ubFactorA], ...)             │
│    xLocal 形状: (curUbFactorA, R)                                │
│                                                                   │
│  Step 2: 类型转换 (仅 bf16/fp16)                                  │
│    Cast(x_local_bf16 → cast_local_fp32)                          │
│                                                                   │
│  Step 3: 平方计算                                                 │
│    Mul(cast_local, cast_local, cast_local)  → x^2                │
│                                                                   │
│  Step 4: 归约求和 (4-fold + Repeat 并行)                           │
│    对每个 token 的 R 个元素求和，得到 curUbFactorA 个标量            │
│    for j in 0..3:                                                │
│      WholeReduceSum(reduce_buf[j], cast_local[j*R/4 : (j+1)*R/4])│
│    WholeReduceSum(final_sum, reduce_buf[0..3], 64, 4)            │
│    WholeReduceSum(y_local[idx], final_sum, 4, 1)                  │
│                                                                   │
│  Step 5: 均值计算                                                  │
│    Muls(y_local, y_local, 1.0/R)  → mean = sum / R               │
│                                                                   │
│  Step 6: 加 epsilon                                               │
│    Adds(y_local, y_local, eps)  → mean + eps                     │
│                                                                   │
│  Step 7: 开根号                                                   │
│    Sqrt(y_local, y_local)  → sqrt(mean + eps)                    │
│                                                                   │
│  Step 8: 取倒数                                                   │
│    Div(y_local, 1.0, y_local)  → 1/sqrt(mean + eps)              │
│                                                                   │
│  Step 9: UB → GM  (写回 curUbFactorA 个结果)                      │
│    DataCopyPad(yGm[idx * ubFactorA], yLocal, ...)                │
│                                                                   │
│  输出: y (A, 1) float32                                          │
│        每个 token 对应一个 rsqrt 值                                │
└──────────────────────────────────────────────────────────────────┘
```

### 关键优化技术

| 优化技术            | 实现方式                                                | 收益                   |
| --------------- | --------------------------------------------------- | -------------------- |
| **多 Core 并行**   | 按 token 切分（A轴），每个 AI Core 处理 `blockFactorA` 个 token | 充分利用 NPU 多核算力        |
| **UB 循环**       | 每次加载 `ubFactorA` 个 token 的数据到 UB，分批处理               | 适配 UB 容量限制，提升内存带宽利用率 |
| **4-fold 并行归约** | 将每个 token 的 R 轴（hc\_mult×hidden\_size）分成 4 份并行归约    | 提高单 token 归约计算的并行度   |
| **bf16 类型转换**   | 在 UB 内完成 bf16→fp32 转换                               | 保证计算精度，同时减少 GM 数据传输  |
| **内存对齐**        | R 轴按 32 字节（UB\_BLOCK\_SIZE）对齐                       | 优化 DMA 传输效率          |

### AI Core 单核心执行流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AI Core 单核心执行流程                          │
├─────────────────────────────────────────────────────────────────────┤
│  GM (显存)                          UB (AI Core 片上内存)          │
│     │                                  │                           │
│     │  ┌───────────────────────────────│                           │
│     │  │ CopyIn()                      │                           │
│     │  │ DataCopyPad                   │                           │
│     │  ▼                               │                           │
│     │  ┌───────────────────────────────▼─────────────────┐         │
│     │  │  xLocal (输入数据)                              │         │
│     │  │         │                                       │         │
│     │  │         ▼                                       │         │
│     │  │  ┌───────────────────────────────────────────┐   │         │
│     │  │  │ ComputeB16()                              │   │         │
│     │  │  │  1. Cast(bf16→fp32)                      │   │         │
│     │  │  │  2. Mul(x²)                              │   │         │
│     │  │  │  3. WholeReduceSum × 4 (4-fold 归约)     │   │         │
│     │  │  │  4. Muls(÷R)                             │   │         │
│     │  │  │  5. Adds(+epsilon)                       │   │         │
│     │  │  │  6. Sqrt()                               │   │         │
│     │  │  │  7. Div(1/x)                             │   │         │
│     │  │  └───────────────────────────────────────────┘   │         │
│     │  │         │                                       │         │
│     │  │         ▼                                       │         │
│     │  │  yLocal (输出数据: rsqrt)                       │         │
│     │  │         │                                       │         │
│     │  │         │ CopyOut()                             │         │
│     │  │         │ DataCopyPad                           │         │
│     │  └─────────▼───────────────────────────────────────┘         │
│     │                                  │                           │
│     ▼                                  ▼                           │
│  yGm[idx*ubFactorA]              释放 yLocal                       │
│  (写入显存)                                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 新手常见疑问

**Q1: 为什么要用模板类 `template <typename T>`？**

> A: 因为算子需要支持多种数据类型（bf16/fp16/fp32），模板可以在编译期生成不同类型的代码，避免重复写三份几乎一样的代码。

**Q2: `PipeBarrier<PIPE_V>()` 是什么？**

> A: AI Core 是流水线架构，计算和数据传输可以重叠。`PipeBarrier` 用于同步流水线，确保前一步操作完成后再执行下一步，防止数据竞争。

**Q3: 为什么要做内存对齐？**

> A: DMA 引擎按固定块大小传输数据（如 32 字节），如果数据不对齐，需要额外处理边界，降低效率。对齐后可以充分利用 DMA 的带宽。

**Q4: `LocalTensor` 和 `GlobalTensor` 的区别？**

> A: `LocalTensor` 指向 UB（片上高速内存），`GlobalTensor` 指向 GM（显存）。UB 访问速度比 GM 快几十倍，但容量小。

**Q5: 多个 AI Core 是怎么并行的？**

> A: 每个 AI Core 执行相同的 `hc_pre_inv_rms` 函数，但 `blockIdx_` 不同，处理的数据范围不同。比如 1000 个 token 分给 10 个核，每个核处理 100 个 token。



## 步骤2：线性投影计算 mixes

调用算子: `at::linear` (PyTorch 原生矩阵乘)

**这一步做什么？**
把展平的特征通过一个线性层（矩阵乘法），映射到 mix\_hc=24 维的门控参数空间。

**Shape 变化**:

```
x: (num_tokens, hc_mult, hidden_size) bfloat16
    │
    │  步骤2.1: 转 float32（计算精度要求）
    ▼
x_float: (num_tokens, hc_mult, hidden_size) float32
    │
    │  步骤2.2: 展平 hc_mult 和 hidden_size 维度
    │  把每个 token 的 hc_mult * hidden_size 个值排成一个长向量
    ▼
x_flat: (num_tokens, hc_mult * hidden_size) float32
    │
    │  步骤2.3: 矩阵乘法 (线性投影)
    │  x_flat @ hc_fn.T
    │  (num_tokens, hc_mult * hidden_size) × (hc_mult * hidden_size, mix_hc) = (num_tokens, mix_hc)
    ▼
mixes: (num_tokens, mix_hc) float32
```

**计算公式**:

```
x_float = x.to(torch.float32)     # shape: (num_tokens, hc_mult, hidden_size)   转float32
x_flat = x_float.flatten(1)       # shape: (num_tokens, hc_mult * hidden_size)   第1维开始展平
mixes = x_flat @ hc_fn.T          # shape: (num_tokens, mix_hc)     矩阵乘法

shape 验证:
  x_flat.shape = (num_tokens, hc_mult * hidden_size)         # 输入特征
  hc_fn.shape  = (mix_hc, hc_mult * hidden_size)             # 权重矩阵，mix_hc=24
  hc_fn.T.shape = (hc_mult * hidden_size, mix_hc)            # 转置后
  mixes.shape  = (num_tokens, mix_hc)                        # 输出，每个token24个门控值
```

> **mixes 的 mix\_hc=24 维分别是什么？**
>
> - 前 hc\_mult=4 维: post 门控（用于 hc\_post）
> - 接下来 hc\_mult\*hc\_mult=16 维: comb 组合矩阵（用于通道混合）
> - 最后 hc\_mult=4 维: pre 门控（用于特征压缩）
>   总共: 4 + 16 + 4 = 24 = mix\_hc

***




## 步骤3：Sinkhorn 归一化子算子

调用算子: `aclnnHcPreSinkhorn`
源码位置: `csrc/moe/hc_pre_sinkhorn/`

### 算子原理

#### 功能说明

这是 hc_pre 算子中最复杂的一步，主要做四件事：

1. 用 mixes 的前 hc_mult 维 + scale[0] + base[0 : hc_mult] → 计算 pre 门控
2. 用 mixes 的中间 hc_mult 维 + scale[1] + base[hc_mult : 2 * hc_mult] → 计算 post 门控
3. 用 mixes 的后 hc_mult * hc_mult 维 + scale[2] + base[2 * hc_mult:] → 做 Sinkhorn 迭代，生成组合矩阵 comb_frag
4. 用 pre 门控作为权重，把 x 从 hc_mult 个通道加权求和压缩成 1 个通道

> **重要说明**：本步骤的计算流程是直接从 NPU 内核源码（`hc_pre_sinkhorn_perf.h` 和 `hc_pre_sinkhorn_base.h`）反推出来的，不是推测！

#### 输入输出形状

```
输入:
  mixes:    (num_tokens, mix_hc)     float32   # mix_hc = hc_mult + hc_mult + hc_mult*hc_mult = 4+4+16 = 24维
  rsqrt:    (num_tokens, 1)          float32   # InvRMS 归一化因子
  hc_scale: (3,)                     float32   # [pre_scale, post_scale, comb_scale]
  hc_base:  (mix_hc,)                float32   # [pre_base(hc_mult个), post_base(hc_mult个), comb_base(hc_mult*hc_mult个)]
  x:        (num_tokens, hc_mult, hidden_size)  bfloat16  # 原始特征
    │
    ▼
输出:
  y:         (num_tokens, hidden_size)       bfloat16  # 压缩后的主路径特征（pre门控加权求和）
  post:      (num_tokens, hc_mult)           float32   # post门控权重，给hc_post用
  comb_frag: (num_tokens, hc_mult, hc_mult)  float32   # 双随机组合矩阵，给hc_post用
```

#### hc_scale 的用法

hc_scale 是 (3,)，不是向量！它是**3个独立的标量数**，分别用在3个不同的地方：

- `hc_scale[0]` → pre 门控的缩放因子（标量，1个数）
- `hc_scale[1]` → post 门控的缩放因子（标量，1个数）
- `hc_scale[2]` → comb 门控的缩放因子（标量，1个数）

#### mixes 和 hc_base 的分段结构

mixes 总共有 mix_hc = 24 维，分成三段：

```
mixes (num_tokens, 24)
  ├── 第 0~3 维 (hc_mult=4维)          → pre 门控的原始参数
  ├── 第 4~7 维 (hc_mult=4维)          → post 门控的原始参数
  └── 第 8~23 维 (hc_mult*hc_mult=16维) → comb 组合矩阵的原始参数

hc_base (24,)
  ├── 第 0~3 维 (hc_mult=4个)          → pre 门控的偏置 (hcBase0Local)
  ├── 第 4~7 维 (hc_mult=4个)          → post 门控的偏置 (hcBase1Local)
  └── 第 8~23 维 (hc_mult*hc_mult=16个) → comb 门控的偏置 (hcBase2Local)
```

> **源码依据**:
>
> - `hc_pre_sinkhorn_perf.h:95-97` 中 hcBase 的三次 CopyIn 分别取前 hc_mult、中间 hc_mult、后 hc_mult*hc_mult 个
> - `hc_pre_sinkhorn_perf.h:106-161` 中 mixes 的拷贝也是同样的分段

### 核心计算流程

#### 3.1 pre 门控计算（ProcessPre）

源码位置: `hc_pre_sinkhorn_base.h:387-400`

**输入**:

- pre\_mixes: mixes 的前 hc\_mult 维 → shape `(num_tokens, hc_mult)`
- rsqrt: 归一化因子 → shape `(num_tokens, 1)`
- hc\_scale\[0]: pre 缩放因子 → 标量
- hc\_base\[0:hc\_mult]: pre 偏置 → shape `(hc_mult,)`
- hc\_eps: 很小的数，防止除0 → 标量

**计算顺序**:

```
# 第1步: 乘 rsqrt（行归一化）
pre_tmp1 = pre_mixes * rsqrt                        # (num_tokens, hc_mult) * (num_tokens, 1) -> (num_tokens, hc_mult)

# 第2步: 乘 scale（标量缩放）
pre_tmp2 = pre_tmp1 * hc_scale[0]                  # (num_tokens, hc_mult) * scalar -> (num_tokens, hc_mult)

# 第3步: 加 base（加偏置）
pre_logits = pre_tmp2 + hc_base[0:hc_mult]          # (num_tokens, hc_mult) + (hc_mult,) -> (num_tokens, hc_mult)

# 第4步: sigmoid 激活
pre_sigmoid = sigmoid(pre_logits)                   # 压缩到 (0, 1)

# 第5步: 加 eps（防止为0）
pre_gate = pre_sigmoid + hc_eps                     # (eps, 1+eps)
```

> **广播机制小科普**
>
> - `(num_tokens, hc_mult)` 和 `(num_tokens, 1)` 相乘：PyTorch 把 `(num_tokens, 1)` 的最后一维"复制"hc\_mult份，变成 `(num_tokens, hc_mult)`，然后对应位置相乘。效果是每一行的 hc\_mult 个元素都乘上这一行的 rsqrt 值。
> - `(num_tokens, hc_mult)` 和 `(hc_mult,)` 相加：PyTorch 把 `(hc_mult,)` 的第0维"复制"num\_tokens份，变成 `(num_tokens, hc_mult)`，然后对应位置相加。效果是每一行都加上同一个 base 向量。
> - 标量乘矩阵：标量自动扩展成和矩阵一样的 shape，每个元素都乘这个标量。

#### 3.2 特征压缩 x → y（ProcessY）

源码位置: `hc_pre_sinkhorn_base.h:463-472`

> **注意**：从源码执行顺序看，特征压缩在 post 门控和 Sinkhorn 之前计算！因为 pre 门控算完之后，马上就可以算 y 了。

**输入**:

- x: 原始特征 → shape `(num_tokens, hc_mult, hidden_size)`, dtype bfloat16
- pre\_gate: pre 门控 → shape `(num_tokens, hc_mult)`, dtype float32

**计算顺序**:

```
# 第1步: x 从 bfloat16 cast 成 float32
x_float = x.to(torch.float32)                       # (num_tokens, hc_mult, hidden_size)

# 第2步: x 乘以 pre_gate（广播乘法）
x_gated = x_float * pre_gate.unsqueeze(-1)          # (num_tokens, hc_mult, hidden_size)

# 第3步: 对 hc_mult 维（通道维）reduce sum，压缩成 1 个通道
y_float = x_gated.sum(dim=1)                        # (num_tokens, hidden_size)

# 第4步: cast 回 bfloat16
y = y_float.to(torch.bfloat16)                      # (num_tokens, hidden_size)
```

> **直观理解**：原来每个 token 有 hc\_mult 个通道的特征，每个通道 hidden\_size 维。pre\_gate 的 hc\_mult 个值就是 hc\_mult 个权重（0\~1之间），我们用这 hc\_mult 个权重把 hc\_mult 个通道的特征加权求和，得到 1 个 hidden\_size 维的特征，送给后面的注意力/FFN层计算。这样计算量就变成了原来的 1/hc\_mult。

> **源码依据**：`ProcessY` 函数里依次调用了 `CastTwoDim` → `MulABLastDimBrcInline` → `ReduceSumARAPerf` → `CastTwoDim`，完全对应上面的四步。

#### 3.3 post 门控计算（ProcessPost）

源码位置: `hc_pre_sinkhorn_base.h:475-488`

**输入**:

- post\_mixes: mixes 的中间 hc\_mult 维 → shape `(num_tokens, hc_mult)`
- rsqrt: 归一化因子 → shape `(num_tokens, 1)`
- hc\_scale\[1]: post 缩放因子 → 标量
- hc\_base\[hc\_mult:2\*hc\_mult]: post 偏置 → shape `(hc_mult,)`

**计算顺序**:

```
# 第1步: 乘 rsqrt
post_tmp1 = post_mixes * rsqrt                      # (num_tokens, hc_mult) * (num_tokens, 1) -> (num_tokens, hc_mult)

# 第2步: 乘 scale
post_tmp2 = post_tmp1 * hc_scale[1]                 # (num_tokens, hc_mult) * scalar -> (num_tokens, hc_mult)

# 第3步: 加 base
post_logits = post_tmp2 + hc_base[hc_mult:2*hc_mult] # (num_tokens, hc_mult) + (hc_mult,) -> (num_tokens, hc_mult)

# 第4步: sigmoid 激活
post_sigmoid = sigmoid(post_logits)                 # 压缩到 (0, 1)

# 第5步: 乘 2（注意：没有加 eps！）
post = post_sigmoid * 2.0                           # 放大到 (0, 2) 区间
```

> **pre 和 post 的区别**：
>
> - pre: sigmoid + eps → 范围 (eps, 1+eps)，接近 (0, 1)
> - post: sigmoid × 2 → 范围 (0, 2)，没有 eps
> - pre 用于压缩时的加权求和（权重需要大于0）
> - post 用于扩展时的门控（可以大于1，相当于放大）

#### 3.4 comb 组合矩阵 + Sinkhorn 迭代

源码位置: `hc_pre_sinkhorn_perf.h:167-192` + `hc_pre_sinkhorn_base.h:509-523`

**输入**:

- comb\_mixes: mixes 的后 hc\_mult\*hc\_mult 维 → shape `(num_tokens, hc_mult*hc_mult)`
- rsqrt: 归一化因子 → shape `(num_tokens, 1)`
- hc\_scale\[2]: comb 缩放因子 → 标量
- hc\_base\[2*hc\_mult:]: comb 偏置 → shape `(hc_mult*hc_mult,)`
- hc\_sinkhorn\_iters: Sinkhorn 迭代次数 → 整数
- hc\_eps: 防止除0的小数 → 标量

**计算顺序**:

```
# ============================================
# 第1阶段: logits 计算 + reshape
# ============================================

# 1.1 乘 rsqrt
comb_tmp1 = comb_mixes * rsqrt                      # (num_tokens, hc_mult*hc_mult) * (num_tokens, 1)

# 1.2 乘 scale
comb_tmp2 = comb_tmp1 * hc_scale[2]                 # (num_tokens, hc_mult*hc_mult) * scalar

# 1.3 加 base
comb_logits = comb_tmp2 + hc_base[2*hc_mult:]       # (num_tokens, hc_mult*hc_mult) + (hc_mult*hc_mult,)

# 1.4 reshape 成矩阵形式
P0 = comb_logits.reshape(num_tokens, hc_mult, hc_mult)  # (num_tokens, 16) -> (num_tokens, 4, 4)

# ============================================
# 第2阶段: 初始 Softmax（行归一化）
# ============================================

# 2.1 减 max（每行减该行的最大值，防止 exp 溢出）
row_max = P0.max(dim=-1, keepdim=True).values      # (num_tokens, hc_mult, 1)
P_minus_max = P0 - row_max                         # (num_tokens, hc_mult, hc_mult)

# 2.2 exp 指数
P_exp = P_minus_max.exp()                          # (num_tokens, hc_mult, hc_mult)

# 2.3 除行和（行归一化：每行的和为 1）
row_sum = P_exp.sum(dim=-1, keepdim=True)          # (num_tokens, hc_mult, 1)
P = P_exp / (row_sum + hc_eps)                     # (num_tokens, hc_mult, hc_mult)

# ============================================
# 第3阶段: 初始列归一化
# ============================================

# 3.1 求列和
col_sum = P.sum(dim=-2, keepdim=True)              # (num_tokens, 1, hc_mult)

# 3.2 列归一化：每列除以列和
P = P / (col_sum + hc_eps)                         # (num_tokens, hc_mult, hc_mult)

# ============================================
# 第4阶段: Sinkhorn 迭代 (hc_sinkhorn_iters - 1 次)
# ============================================

for t = 1 to hc_sinkhorn_iters-1:
    # 4.1 行归一化
    row_sum = P.sum(dim=-1, keepdim=True)          # (num_tokens, hc_mult, 1)
    P = P / (row_sum + hc_eps)                     # (num_tokens, hc_mult, hc_mult)
    
    # 4.2 列归一化
    col_sum = P.sum(dim=-2, keepdim=True)          # (num_tokens, 1, hc_mult)
    P = P / (col_sum + hc_eps)                     # (num_tokens, hc_mult, hc_mult)

# ============================================
# 第5阶段: 最终结果
# ============================================
comb_frag = P                                      # (num_tokens, hc_mult, hc_mult)
```

> **什么是双随机矩阵？**
> 每行加起来等于1，每列加起来也等于1的矩阵。这种矩阵可以看作是"通道间的转移概率"——描述了 hc\_mult 个输入通道如何混合成 hc\_mult 个输出通道。

> **Sinkhorn 迭代的次数**：源码中循环 `iterTimes - 1` 次，每次循环做 1 次行归一化 + 1 次列归一化。加上循环前的 1 次 softmax（行归一化） + 1 次列归一化，总共做了 hc\_sinkhorn\_iters 次行归一化和 hc\_sinkhorn\_iters 次列归一化。

### AscendC 算子架构

遵循 AscendC 的 Host-Kernel 分离架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    AscendC Operator Architecture            │
├─────────────────────────────────────────────────────────────┤
│  Host 侧 (CPU)                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ hc_pre_sinkhorn_def.cpp    → 算子定义                  │   │
│  │ hc_pre_sinkhorn_tiling.cpp → Tiling 策略计算           │   │
│  │ hc_pre_sinkhorn_proto.cpp  → 算子原型定义               │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓ 编译时生成 TilingData              │
│  Kernel 侧 (AI Core)                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ hc_pre_sinkhorn.cpp        → 核函数入口                │   │
│  │ hc_pre_sinkhorn_perf.h     → 核心计算逻辑（性能优化版）│   │
│  │ hc_pre_sinkhorn_base.h     → 基础计算函数（Sigmoid、   │   │
│  │                             Softmax、ReduceSum等）     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Host 侧实现

#### Tiling 策略

将输入张量按 token 维度（bs）切分到多个 AI Core，同时考虑 UB 容量限制进行 d 维度循环。

**关键参数**：

| 参数                 | 含义                         | 计算方式                                              |
| ------------------ | -------------------------- | ------------------------------------------------- |
| `bs`               | 总 token 数 = num\_tokens    | mixes.shape\[0] 或 b×s                             |
| `hcMix`            | mixes 的第二维大小 = 24          | mix\_hc = hc\_mult + hc\_mult + hc\_mult×hc\_mult |
| `hcMult`           | 通道数 = 4                    | 属性参数，默认值为 4                                       |
| `d`                | hidden\_size = 4096/7168   | x 的最后一维                                           |
| `hcMultAlign`      | hcMult 对齐后大小               | RoundUp(hcMult, BLOCK\_SIZE/sizeof(float)) = 8    |
| `rowOfFormerBlock` | 前 N-1 个核处理的 token 数        | CeilDiv(bs, coreNum)                              |
| `rowOfTailBlock`   | 尾核处理的 token 数              | bs - (usedCoreNums-1) × rowOfFormerBlock          |
| `rowFactor`        | 每次 UB 循环处理的 token 数        | 根据 UB 容量动态计算                                      |
| `dFactor`          | 每次 d 循环处理的 hidden\_size 大小 | 根据 UB 容量动态计算                                      |
| `dLoop`            | d 维度的循环次数                  | CeilDiv(d, dFactor)                               |
| `iterTimes`        | Sinkhorn 迭代次数              | 属性参数，默认值为 20                                      |

**Tiling 计算流程**（`CalcRegbaseOpTiling` / `CalcMembaseOpTiling`）：

```cpp
// 1. 计算核分配
rowOfFormerBlock = CeilDiv(bs, coreNum)      // 每个核处理的 token 数
usedCoreNums = min(CeilDiv(bs, rowOfFormerBlock), coreNum)
rowOfTailBlock = bs - (usedCoreNums - 1) * rowOfFormerBlock

// 2. 计算 UB 内存需求（按最小 rowOnceLoop=1 估算）
mix0Size = rowOnceLoop * hcMultAlign * sizeof(float) * DOUBLE_BUFFER
mix1Size = rowOnceLoop * hcMultAlign * sizeof(float) * DOUBLE_BUFFER
mix2Size = rowOnceLoop * hcMult * hcMultAlign * sizeof(float) * DOUBLE_BUFFER
rsqrtSize = RoundUp(rowOnceLoop, BLOCK_SIZE/sizeof(float)) * sizeof(float) * DOUBLE_BUFFER
xSize = rowOnceLoop * hcMult * RoundUp(d, 16) * 2 * DOUBLE_BUFFER  // bf16
ySize = rowOnceLoop * RoundUp(d, 16) * 2 * DOUBLE_BUFFER          // bf16
postSize = rowOnceLoop * hcMultAlign * sizeof(float) * DOUBLE_BUFFER
combFragSize = rowOnceLoop * hcMult * hcMultAlign * sizeof(float) * DOUBLE_BUFFER
base0Size = hcMultAlign * sizeof(float)
base1Size = hcMultAlign * sizeof(float)
base2Size = hcMult * hcMultAlign * sizeof(float)

// 3. 判断是否需要 d 循环
if totalSize <= ubSize:
    dLoop = 1, dFactor = d
else:
    // 计算剩余 UB 空间，逐步增大 d 循环次数
    dFactor = CeilDiv(d, base)  // base 从 2 开始递增
    // 直到 xSize + ySize <= ubRemain

// 4. d 全载时，尝试搬入更多的 bs
if dFactor == d:
    while rowFactor <= rowOfFormerBlock:
        // 重新计算 totalSize
        if totalSize > ubSize:
            rowFactor -= 1
            break
        rowFactor += 1
```

### Kernel 侧实现

#### 完整计算流程

单个 AI Core 内部，处理 `curBlockIdx` 负责的 token：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    hc_pre_sinkhorn 完整计算流程                          │
├──────────────────────────────────────────────────────────────────────────┤
│  输入:                                                                   │
│    mixes: (bs, hcMix) float32           # hcMix = 24                    │
│    rsqrt: (bs, 1) float32               # 归一化因子                    │
│    hcScale: (3,) float32                # [pre_scale, post_scale, comb_scale]│
│    hcBase: (hcMix,) float32             # [pre_base, post_base, comb_base]  │
│    x: (bs, hcMult, d) bf16              # 原始特征                       │
│                                                                          │
│  预处理: 加载 hcBase 的三个分段到 UB                                      │
│    hcBase0Local = hcBase[0:hcMult]         # pre 偏置                     │
│    hcBase1Local = hcBase[hcMult:2*hcMult]  # post 偏置                    │
│    hcBase2Local = hcBase[2*hcMult:]        # comb 偏置                    │
│                                                                          │
│  外层循环：按 token 切分，每次处理 rowFactor 个 token                       │
│  共循环 rowOuterLoop = ceil(curRowNum / rowFactor) 次                      │
│                                                                          │
│  ──────────────────────────────────────────────────────────────────────  │
│  第 rowOuterIdx 次循环（处理 curRowFactor 个 token）：                      │
│                                                                          │
│  Step 1: 加载 mixes 的前 2×hcMult 维 + rsqrt                               │
│    mixes01Local = mixes[rowRange, 0:2*hcMult]  # pre + post 参数         │
│    rsqrtLocal = rsqrt[rowRange]               # 归一化因子               │
│                                                                          │
│  Step 2: 计算 pre 门控 (ProcessPre)                                       │
│    pre_gate = sigmoid(mixes01Local[:,0:hcMult] * rsqrt * scale[0] + base0) + eps │
│    # 存储在 mixes01Local[:,0:hcMult]                                      │
│                                                                          │
│  Step 3: 特征压缩 x → y (ProcessY)                                        │
│    内层循环：按 d 维度切分，每次处理 dFactor 个特征                          │
│    共循环 dLoop = ceil(d / dFactor) 次                                    │
│    for dLoopIdx in 0..dLoop-1:                                           │
│      xLocal = x[rowRange, :, dRange]  # bf16                             │
│      xCast = cast(xLocal, bf16→float32)                                  │
│      xGated = xCast * pre_gate.unsqueeze(-1)  # 广播乘法                  │
│      yCast = reduce_sum(xGated, dim=1)        # 通道压缩                  │
│      yLocal = cast(yCast, float32→bf16)                                   │
│      yGm[rowRange, dRange] = yLocal  # 写回 GM                           │
│                                                                          │
│  Step 4: 计算 post 门控 (ProcessPost)                                     │
│    post = sigmoid(mixes01Local[:,hcMult:2*hcMult] * rsqrt * scale[1] + base1) * 2.0 │
│                                                                          │
│  Step 5: 计算 comb_frag (Sinkhorn 迭代)                                   │
│    mixes2Local = mixes[rowRange, 2*hcMult:]  # comb 参数                 │
│    comb_logits = mixes2Local * rsqrt * scale[2] + base2                   │
│    P = softmax(comb_logits.reshape(-1, hcMult, hcMult), dim=-1)          │
│    P = P / sum(P, dim=-2, keepdim=True)  # 初始列归一化                   │
│    for iter in 0..iterTimes-2:                                           │
│      P = P / sum(P, dim=-1, keepdim=True)  # 行归一化                    │
│      P = P / sum(P, dim=-2, keepdim=True)  # 列归一化                    │
│    comb_frag = P                                                         │
│                                                                          │
│  输出:                                                                   │
│    y: (bs, d) bf16              # 压缩后的主路径特征                      │
│    post: (bs, hcMult) float32   # post 门控                              │
│    comb_frag: (bs, hcMult, hcMult) float32  # 双随机组合矩阵              │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 关键优化技术

| 优化技术          | 实现方式                                                      | 收益                 |
| ------------- | --------------------------------------------------------- | ------------------ |
| **多 Core 并行** | 按 token 切分（bs 轴），每个 AI Core 处理 `rowOfFormerBlock` 个 token | 充分利用 NPU 多核算力      |
| **UB 循环**     | 每次加载 `rowFactor` 个 token 的数据到 UB，分批处理                     | 适配 UB 容量限制         |
| **d 循环**      | 当 hidden\_size 过大时，将 d 轴分成 `dLoop` 份循环处理                  | 适配 UB 容量限制，处理大模型   |
| **双缓冲**       | 输入/输出队列使用双缓冲（DOUBLE\_BUFFER=2）                            | 实现计算和数据传输流水线重叠     |
| **内存对齐**      | hcMult 按 32 字节对齐（BLOCK\_SIZE），d 按 16 对齐                   | 优化 DMA 传输效率        |
| **Brcb 优化**   | 对广播乘法使用 Brcb（Broadcast Replicate）指令优化                     | 减少数据复制，提升广播乘法效率    |
| **Repeat 并行** | 在行/列方向开启 Repeat 并行，一次处理多个元素                               | 提高 AI Core 计算单元利用率 |

### 完整调用链路

#### 第一层：PyTorch C++ 绑定层

```cpp
// torch_binding.cpp:1547-1548
EXEC_NPU_CMD(aclnnHcPreSinkhorn, mixes, rsqrt, hc_scale, hc_base, x, hc_mult, hc_sinkhorn_iters, hc_eps,
                y, post, comb_frag);
```

#### 第二层：动态库加载层

与 `aclnnHcPreInvRms` 相同，通过 `GetOpApiFuncAddr("aclnnHcPreSinkhorn")` 动态加载算子函数地址。

#### 第三层：CANN 算子注册层

`csrc/moe/hc_pre_sinkhorn/op_host/hc_pre_sinkhorn_def.cpp`:

```cpp
class HcPreSinkhorn : public OpDef {
public:
    explicit HcPreSinkhorn(const char* name) : OpDef(name) {
        // 定义输入
        this->Input("mixes").ParamType(REQUIRED).DataType({ge::DT_FLOAT});
        this->Input("rsqrt").ParamType(REQUIRED).DataType({ge::DT_FLOAT});
        this->Input("hc_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT});
        this->Input("hc_base").ParamType(REQUIRED).DataType({ge::DT_FLOAT});
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16});
        
        // 定义输出
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_BF16});
        this->Output("post").ParamType(REQUIRED).DataType({ge::DT_FLOAT});
        this->Output("comb_frag").ParamType(REQUIRED).DataType({ge::DT_FLOAT});
        
        // 定义属性
        this->Attr("hc_mult").AttrType(OPTIONAL).Int(4);
        this->Attr("hc_sinkhorn_iters").AttrType(OPTIONAL).Int(20);
        this->Attr("hc_eps").AttrType(OPTIONAL).Float(1e-6f);
        
        // 指定运行设备
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        // ... Ascend 950 动态编译配置
    }
};

OP_ADD(HcPreSinkhorn);
```

#### 第四层：Tiling 策略层

`csrc/moe/hc_pre_sinkhorn/op_host/hc_pre_sinkhorn_tiling.cpp`:

```cpp
// 1. 获取输入形状信息
// mixes: (b, s, hc_mix) 或 (bs, hc_mix)
size_t mixerDimNum = shapeMixes->GetStorageShape().GetDimNum();
if (mixerDimNum == 2) {
    bs_ = shapeMixes->GetStorageShape().GetDim(0);
    hcMix_ = shapeMixes->GetStorageShape().GetDim(1);
} else if (mixerDimNum == 3) {
    int64_t b = shapeMixes->GetStorageShape().GetDim(0);
    int64_t s = shapeMixes->GetStorageShape().GetDim(1);
    bs_ = b * s;
    hcMix_ = shapeMixes->GetStorageShape().GetDim(2);
}

// 2. 获取属性参数
hcMult_ = hcMultAttr == nullptr ? 4 : *hcMultAttr;
iterTimes_ = iterTimesAttr == nullptr ? 20 : *iterTimesAttr;
eps_ = epsAttr == nullptr ? 1e-5 : *epsAttr;

// 3. 根据平台选择 Tiling 策略
if (socVersion_ == platform_ascendc::SocVersion::ASCEND950) {
    return CalcRegbaseOpTiling();  // Ascend 950 使用寄存器优化版本
}
return CalcMembaseOpTiling();     // Ascend 910B/C 使用内存优化版本
```

#### 第五层：AI Core Kernel 执行层

`csrc/moe/hc_pre_sinkhorn/op_kernel/hc_pre_sinkhorn.cpp`:

```cpp
extern "C" __global__ __aicore__ void hc_pre_sinkhorn(
    GM_ADDR mixes, GM_ADDR rsqrt, GM_ADDR hcScale, GM_ADDR hcBase,
    GM_ADDR x, GM_ADDR y, GM_ADDR post, GM_ADDR combFrag, GM_ADDR workspace,
    GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    
    if (TILING_KEY_IS(0)) {
        HcPreSinkhorn::HcPreSinkhornPerf<DTYPE_X> op;
        op.Init(mixes, rsqrt, hcScale, hcBase, x, y, post, combFrag, userWs, &tilingData, &pipe);
        op.Process();
    }
}
```

### Kernel 源码逐行详解

#### 1. 类结构定义

`hc_pre_sinkhorn_perf.h:25-252`:

```cpp
template <typename T>
class HcPreSinkhornPerf {
public:
    __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR rsqrt, GM_ADDR hcScale, GM_ADDR hcBase, 
                                GM_ADDR x, GM_ADDR y, GM_ADDR post, GM_ADDR combFrag, 
                                GM_ADDR workspace, const HcPreSinkhornTilingData *tilingDataPtr, TPipe *pipePtr);
    __aicore__ inline void Process();

private:
    TPipe *pipe;
    const HcPreSinkhornTilingData *tilingData;
    
    // GM 张量视图
    GlobalTensor<float> mixesGm;
    GlobalTensor<float> rsqrtGm;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;
    
    // UB 队列（双缓冲）
    TQue<QuePosition::VECIN, 1> mixesQue01;   // mixes 前 2×hcMult 维
    TQue<QuePosition::VECIN, 1> mixesQue2;    // mixes 后 hcMult×hcMult 维
    TQue<QuePosition::VECIN, 1> rsqrtQue;     // rsqrt
    TQue<QuePosition::VECIN, 1> xQue;         // x 输入
    TQue<QuePosition::VECOUT, 1> yQue;        // y 输出
    TQue<QuePosition::VECOUT, 1> postQue;     // post 输出
    TQue<QuePosition::VECOUT, 1> combFragQue; // comb_frag 输出
    
    // UB 缓冲区
    TBuf<QuePosition::VECCALC> hcBaseBuf0;    // pre 偏置
    TBuf<QuePosition::VECCALC> hcBaseBuf1;    // post 偏置
    TBuf<QuePosition::VECCALC> hcBaseBuf2;    // comb 偏置
    TBuf<QuePosition::VECCALC> rowBrcbBuf0;   // 行广播缓冲区
    TBuf<QuePosition::VECCALC> hcBrcbBuf1;    // 通道广播缓冲区
    TBuf<QuePosition::VECCALC> reduceBuf;     // 归约缓冲区
    TBuf<QuePosition::VECCALC> xCastBuf;      // x 类型转换缓冲区
    TBuf<QuePosition::VECCALC> yCastBuf;      // y 类型转换缓冲区
    
    // Local 张量（运行时从队列/缓冲区获取）
    LocalTensor<float> mixes01Local;
    LocalTensor<float> mixes2Local;
    LocalTensor<float> rsqrtLocal;
    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> postLocal;
    LocalTensor<float> combFragLocal;
    // ...
};
```

#### 2. Init 方法详解

```cpp
__aicore__ inline void HcPreSinkhornPerf<T>::Init(...) {
    // 第一步：设置 GM 张量视图
    mixesGm.SetGlobalBuffer((__gm__ float *)mixes);
    rsqrtGm.SetGlobalBuffer((__gm__ float *)rsqrt);
    // ... 其他 GM 张量
    
    // 第二步：初始化输入队列（双缓冲，DOUBLE_BUFFER=2）
    int64_t mixesQue01Size = tilingData->rowFactor * tilingData->hcMultAlign * 2 * sizeof(float);
    pipe->InitBuffer(mixesQue01, 2, mixesQue01Size);  // 双缓冲
    pipe->InitBuffer(mixesQue2, 2,
                     tilingData->rowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
    pipe->InitBuffer(rsqrtQue, 2, RoundUp<float>(tilingData->rowFactor) * sizeof(float));
    pipe->InitBuffer(xQue, 2,
                     tilingData->rowFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor) * sizeof(T));
    
    // 第三步：初始化输出队列
    pipe->InitBuffer(yQue, 2,
                     tilingData->rowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
    pipe->InitBuffer(postQue, 2, tilingData->rowFactor * tilingData->hcMultAlign * sizeof(float));
    pipe->InitBuffer(combFragQue, 2,
                     tilingData->rowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
    
    // 第四步：初始化 UB 缓冲区（非队列，单缓冲）
    pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
    pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
    pipe->InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
    pipe->InitBuffer(rowBrcbBuf0, RoundUp<float>(tilingData->rowFactor) * BLOCK_SIZE);
    pipe->InitBuffer(hcBrcbBuf1, RoundUp<float>(tilingData->rowFactor * tilingData->hcMultAlign) * BLOCK_SIZE);
    pipe->InitBuffer(reduceBuf, tilingData->rowFactor * tilingData->hcMultAlign * sizeof(float));
    pipe->InitBuffer(xCastBuf, tilingData->rowFactor * tilingData->hcMult * RoundUp<float>(tilingData->dFactor) * sizeof(float));
    pipe->InitBuffer(yCastBuf, tilingData->rowFactor * RoundUp<float>(tilingData->dFactor) * sizeof(float));
}
```

#### 3. Process 方法详解（核心主流程）

```cpp
__aicore__ inline void HcPreSinkhornPerf<T>::Process() {
    int64_t curBlockIdx = GetBlockIdx();
    int64_t totalBlockNum = GetBlockNum();
    
    // 判断当前核是普通核还是尾核
    int64_t rowOuterLoop =
        (curBlockIdx == totalBlockNum - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
    int64_t tailRowFactor = (curBlockIdx == totalBlockNum - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                 tilingData->tailRowFactorOfFormerBlock;
    
    // ========== 预处理：加载 hcBase 到 UB ==========
    CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
    CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
    CopyIn(hcBaseGm[tilingData->hcMult * 2], hcBase2Local, tilingData->hcMult, tilingData->hcMult);
    // 等待 DMA 完成
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
    
    // 计算当前核的数据起始偏移
    int64_t mixGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMix;
    int64_t xGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult * tilingData->d;
    
    // ========== 外层循环：按 token 分批处理 ==========
    for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
        int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
        
        // 步骤1: 加载 mixes 前 2×hcMult 维 + rsqrt
        mixes01Local = mixesQue01.AllocTensor<float>();
        CopyIn(mixesGm[mixGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->hcMix], 
               mixes01Local, curRowFactor, tilingData->hcMult, tilingData->hcMix - tilingData->hcMult);
        // mixes 的中间 hcMult 维（post 参数）
        CopyIn(mixesGm[mixGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->hcMix + tilingData->hcMult],
               mixes01Local[tilingData->rowFactor * tilingData->hcMultAlign], 
               curRowFactor, tilingData->hcMult, tilingData->hcMix - tilingData->hcMult);
        mixesQue01.EnQue(mixes01Local);
        
        rsqrtLocal = rsqrtQue.AllocTensor<float>();
        CopyIn(rsqrtGm[curBlockIdx * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor],
               rsqrtLocal, 1, curRowFactor);
        rsqrtQue.EnQue(rsqrtLocal);
        
        // 步骤2: 计算 pre 门控
        mixes01Local = mixesQue01.DeQue<float>();
        rsqrtLocal = rsqrtQue.DeQue<float>();
        ProcessPre(mixes01Local, mixes01Local, hcBase0Local, rsqrtLocal, 
                   rowBrcbLocal0, hcBrcbLocal1, hcScaleGm.GetValue(0), 
                   tilingData->eps, curRowFactor, tilingData->hcMult);
        
        // 步骤3: 特征压缩（d 循环）
        for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
            int64_t curDFactor = (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
            xLocal = xQue.AllocTensor<T>();
            CopyIn(xGm[xGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->hcMult * tilingData->d +
                       dLoopIdx * tilingData->dFactor],
                   xLocal, tilingData->rowFactor * tilingData->hcMult, curDFactor, 
                   tilingData->d - curDFactor);
            xQue.EnQue(xLocal);
            xLocal = xQue.DeQue<T>();
            yLocal = yQue.AllocTensor<T>();
            ProcessY(yLocal, xLocal, mixes01Local, hcBrcbLocal1, xCastLocal, yCastLocal, 
                     curRowFactor, tilingData->hcMult, curDFactor);
            xQue.FreeTensor(xLocal);
            yQue.EnQue(yLocal);
            // 写回 y
            yLocal = yQue.DeQue<T>();
            CopyOut(yLocal, yGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->d +
                                rowOuterIdx * tilingData->rowFactor * tilingData->d + dLoopIdx * tilingData->dFactor],
                    curRowFactor, curDFactor, tilingData->d - curDFactor);
            yQue.FreeTensor(yLocal);
        }
        
        // 步骤4: 计算 post 门控
        postLocal = postQue.AllocTensor<float>();
        ProcessPost(postLocal, mixes01Local[tilingData->rowFactor * tilingData->hcMultAlign], 
                    hcBase1Local, rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1, 
                    hcScaleGm.GetValue(1), curRowFactor, tilingData->hcMult);
        mixesQue01.FreeTensor(mixes01Local);
        postQue.EnQue(postLocal);
        // 写回 post
        postLocal = postQue.DeQue<float>();
        CopyOut(postLocal, postGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult +
                                   rowOuterIdx * tilingData->rowFactor * tilingData->hcMult],
                curRowFactor, tilingData->hcMult);
        postQue.FreeTensor(postLocal);
        
        // 步骤5: 计算 comb_frag（Sinkhorn 迭代）
        mixes2Local = mixesQue2.AllocTensor<float>();
        CopyInWithOuterFor(mixesGm[mixGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->hcMix +
                                   tilingData->hcMult * 2],
                           mixes2Local, curRowFactor, tilingData->hcMult, 
                           tilingData->hcMult, tilingData->hcMix);
        mixesQue2.EnQue(mixes2Local);
        mixes2Local = mixesQue2.DeQue<float>();
        
        combFragLocal = combFragQue.AllocTensor<float>();
        // 乘 rsqrt + 乘 scale
        MulABLastDimBrcInline<float, false>(mixes2Local, mixes2Local, rsqrtLocal, rowBrcbLocal0, 
                                            curRowFactor, tilingData->hcMult * tilingData->hcMultAlign);
        Muls(mixes2Local, mixes2Local, hcScaleGm.GetValue(2),
             curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);
        // 加 base
        AddBAFirstDimBrcInline<float>(mixes2Local, mixes2Local, hcBase2Local, 
                                      curRowFactor, tilingData->hcMult * tilingData->hcMultAlign);
        // Softmax（行归一化）
        SoftmaxFP32Perf(mixes2Local, mixes2Local, reduceLocal, hcBrcbLocal1, 
                        curRowFactor * tilingData->hcMult, tilingData->hcMult, tilingData->eps);
        // 初始列归一化
        ReduceSumARAPerf(reduceLocal, mixes2Local, curRowFactor, tilingData->hcMult, tilingData->hcMult);
        Adds(reduceLocal, reduceLocal, tilingData->eps, curRowFactor * tilingData->hcMult);
        DivABABrcInline(combFragLocal, mixes2Local, reduceLocal, curRowFactor, 
                        tilingData->hcMult, tilingData->hcMult);
        // Sinkhorn 迭代
        for (int64_t iter = 0; iter < tilingData->iterTimes - 1; iter++) {
            // 行归一化
            LastDimReduceSumPerf(reduceLocal, combFragLocal, curRowFactor * tilingData->hcMult, tilingData->hcMult);
            Adds(reduceLocal, reduceLocal, tilingData->eps, curRowFactor * tilingData->hcMult);
            DivABLastDimBrcInline<float, true>(combFragLocal, combFragLocal, reduceLocal, 
                                               hcBrcbLocal1, curRowFactor * tilingData->hcMult, tilingData->hcMult);
            // 列归一化
            ReduceSumARAPerf(reduceLocal, combFragLocal, curRowFactor, tilingData->hcMult, tilingData->hcMult);
            Adds(reduceLocal, reduceLocal, tilingData->eps, curRowFactor * tilingData->hcMult);
            DivABABrcInline(combFragLocal, combFragLocal, reduceLocal, curRowFactor, 
                            tilingData->hcMult, tilingData->hcMult);
        }
        mixesQue2.FreeTensor(mixes2Local);
        rsqrtQue.FreeTensor(rsqrtLocal);
        
        // 写回 comb_frag
        combFragQue.EnQue(combFragLocal);
        combFragLocal = combFragQue.DeQue<float>();
        CopyOut(combFragLocal, combFragGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult * tilingData->hcMult +
                                           rowOuterIdx * tilingData->rowFactor * tilingData->hcMult * tilingData->hcMult],
                curRowFactor * tilingData->hcMult, tilingData->hcMult);
        combFragQue.FreeTensor(combFragLocal);
    }
}
```

#### 4. ProcessPre 方法详解（pre 门控计算）

`hc_pre_sinkhorn_base.h:387-400`:

```cpp
__aicore__ inline void ProcessPre(...) {
    // 1. 乘 rsqrt（行归一化）
    MulABLastDimBrcInline<float, true>(mixLocal, mixLocal, rsqrtLocal, tmpBuffer0, curRowNum, curColNum);
    
    // 2. 乘 scale（标量缩放）
    Muls(mixLocal, mixLocal, scale, curRowNum * curColNumAlign);
    
    // 3. 加 base（广播加法）
    AddBAFirstDimBrcInline<float>(mixLocal, mixLocal, hcBaseLocal, curRowNum, curColNum);
    
    // 4. sigmoid 激活
    SigmoidPerf(preLocal, mixLocal, tmpBuffer1, curRowNum * curColNumAlign);
    
    // 5. 加 eps
    Adds(preLocal, preLocal, eps, curRowNum * curColNumAlign);
}
```

#### 5. ProcessY 方法详解（特征压缩）

`hc_pre_sinkhorn_base.h:463-472`:

```cpp
template <typename T>
__aicore__ void inline ProcessY(...) {
    // 1. bf16 → float32 类型转换
    CastTwoDim(xCastLocal, xLocal, dim0 * dim1, dim2);
    
    // 2. 乘 pre_gate（广播乘法）
    MulABLastDimBrcInline<float, true>(xCastLocal, xCastLocal, mix01Local, hcBrcbLocal1, dim0 * dim1, dim2);
    
    // 3. 对通道维 reduce sum
    ReduceSumARAPerf(yCastLocal, xCastLocal, dim0, dim1, dim2);
    
    // 4. float32 → bf16 类型转换
    CastTwoDim(yLocal, yCastLocal, dim0, dim2);
}
```

#### 6. SoftmaxFP32Perf 方法详解（数值稳定的 Softmax）

`hc_pre_sinkhorn_base.h:509-523`:

```cpp
__aicore__ inline void SoftmaxFP32Perf(...) {
    // 1. 每行减最大值（防止 exp 溢出）
    LastDimReduceMaxPerf(tmpReduceBuffer, input, curRowNum, curColNum);
    
    // 2. 减最大值
    SubABLastDimBrcInline<float, true>(output, input, tmpReduceBuffer, tmpBrcbBuffer, curRowNum, curColNum);
    
    // 3. exp 指数
    Exp(output, output, curRowNum * curColNumAlign);
    
    // 4. 每行求和
    LastDimReduceSumPerf(tmpReduceBuffer, output, curRowNum, curColNum);
    
    // 5. 除行和（行归一化）
    DivABLastDimBrcInline<float, true>(output, output, tmpReduceBuffer, tmpBrcbBuffer, curRowNum, curColNum);
    
    // 6. 加 eps
    Adds(output, output, eps, curRowNum * curColNumAlign);
}
```

#### 新手常见疑问

**Q1: 为什么 mixes01 和 mixes2 分开存储？**

> A: 因为 mixes 的前 2×hcMult=8 维和后 hcMult×hcMult=16 维使用方式不同：前 8 维用于 pre/post 门控计算（每行独立），后 16 维需要 reshape 成 4×4 矩阵做 Sinkhorn 迭代。分开存储可以优化内存访问模式。

**Q2: hcBase 为什么只加载一次？**

> A: hcBase 是所有 token 共享的偏置参数，不随 token 变化，所以在循环外一次性加载到 UB 即可，避免重复 DMA 传输。

**Q3: 双缓冲（DOUBLE\_BUFFER=2）是什么？**

> A: 双缓冲是一种流水线优化技术。当处理第 N 批数据时，可以同时将第 N+1 批数据从 GM 加载到 UB 的另一个缓冲区。这样计算和数据传输可以重叠，提高整体吞吐量。

**Q4: d 循环什么时候触发？**

> A: 当 UB 容量不足以同时容纳所有 hidden\_size 数据时触发。比如 d=7168 时，单个 token 的 x 数据量 = hcMult × d × 2 = 4 × 7168 × 2 = 57344 字节，接近 UB 容量（32KB/128KB），需要分多次加载。

**Q5: 为什么 pre 门控加 eps 而 post 门控乘 2？**

> A: pre 门控用于加权求和，需要保证权重 > 0（防止除零）；post 门控用于后续扩展，乘 2 可以让门控范围达到 (0, 2)，允许放大特征。

#### 关键文件汇总

| 文件                           | 层次          | 作用                                         |
| ---------------------------- | ----------- | ------------------------------------------ |
| `torch_binding.cpp`          | PyTorch 绑定层 | 调用 `EXEC_NPU_CMD(aclnnHcPreSinkhorn, ...)` |
| `hc_pre_sinkhorn_def.cpp`    | CANN 算子注册层  | 定义输入/输出/属性，注册算子                            |
| `hc_pre_sinkhorn_tiling.cpp` | Tiling 策略层  | 计算数据切分策略，区分 Ascend 910/950                 |
| `hc_pre_sinkhorn_tiling.h`   | Tiling 头文件  | 定义 TilingData 结构体                          |
| `hc_pre_sinkhorn_proto.cpp`  | 算子原型层       | Shape/Type 推理                              |
| `hc_pre_sinkhorn.cpp`        | Kernel 入口层  | AI Core 核函数入口                              |
| `hc_pre_sinkhorn_perf.h`     | Kernel 计算层  | 主计算流程（Process 方法）                          |
| `hc_pre_sinkhorn_base.h`     | Kernel 工具层  | 基础函数（Sigmoid、Softmax、ReduceSum、Cast等）      |

#### 输出总结

```
最终输出三个张量:
  y:         (num_tokens, hidden_size)           bfloat16  # pre门控加权压缩后的主路径特征
  post:      (num_tokens, hc_mult)               float32   # post门控 (0~2之间)，给hc_post用
  comb_frag: (num_tokens, hc_mult, hc_mult)      float32   # 双随机组合矩阵，给hc_post用
```

> **注意**: 上述所有子步骤（3.1 \~ 3.4）都在 `aclnnHcPreSinkhorn` 单个 CANN 算子内完成，
> 上面的公式是直接从 NPU 内核源码反推的，计算顺序和逻辑与源码一致。

#### 两种实现模式

- **融合模式 (`run_hc_pre_fusion`)**: 单算子 `aclnnHcPre` 完成全部计算（910B默认 / 950满足条件时使用）
- **组合模式 (`run_hc_pre_composite`)**: 拆分为 `aclnnHcPreInvRms` + `at::linear` + `aclnnHcPreSinkhorn` 三步（950大batch时使用）
