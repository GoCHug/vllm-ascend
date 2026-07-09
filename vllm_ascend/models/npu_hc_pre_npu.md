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

## InvRMS 归一化子算子

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
