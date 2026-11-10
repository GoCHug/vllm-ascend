# `hc_post` 算子：从小白到逐行读懂

> 面向读者：完全没写过 Ascend C 算子，但看得懂 Python 和一点 C++。
> 建议先读 [1_npu_hc_pre_npu.md](1_npu_hc_pre_npu.md) 的第 2 章（Ascend C 零基础），那里面把 AIC/AIV、GM/UB、Tiling、TilingKey 从零讲了一遍。本文第 2 章只做**针对 hc_post 的复习和补充**。

> **V4.1 适配说明（先读这一段）**：`hc_post` 在 DeepSeek V4.1 中**实现与 C++ 签名都没有变化**（`csrc/torch_binding.cpp` L1527-L1534 注册、L1527 实现），Ascend C 侧也没改。V4.1 的差异只有三处，都不影响下文对 csrc 的精读与行号：

| 差异 | DeepSeek V4 | DeepSeek V4.1 |
|------|-------------|---------------|
| 调用点 | `models/deepseek_v4/model.py:L789` | **`models/deepseek_v41/model.py:L871-L877`** |
| `hidden_size` | 7168 | **5120**（下文以 7168 举例的形状请等比理解） |
| `post` / `comb` 的来源 | 本层 `hc_pre` 的其余两个输出 | **同样是本层 `hc_pre` 的输出**，但该 `hc_pre` 的 `pre_mix` 来自上一层（延迟系数交接，见 [1_npu_hc_pre_npu.md](1_npu_hc_pre_npu.md) §0.1） |

V4.1 的调用是原样转发：

```python
# models/deepseek_v41/model.py:L871-L877
def hc_post(self, x, residual, post, comb):
    return torch.ops._C_ascend.npu_hc_post(
        x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
    ).squeeze(0)
```

> 也就是说：**`hc_post` 这一侧完全不用改，改了 `hc_pre` 之后 `post`/`comb` 的数值来源仍然一致**——V4.1 改变的是 `pre`（坍缩系数）的**生产者与消费者**，不改变 `post`/`comb` 的定义。下文 §1.2 的「和 hc_pre 的对称关系」表、§4 的 9 种 dtype 组合、§5 的 Kernel 逐行解析全部有效。


## 0. 这份文档怎么读

| 你的问题 | 看哪里 |
| --- | --- |
| hc_post 到底在算什么？ | 第 1 章 |
| hc_post 和 hc_pre 是什么关系？ | 1.2 |
| Ascend C 我只想补必须知道的那几个点 | 第 2 章 |
| 这个算子为什么有 9 种 dtype 组合？ | 4.1 |
| Kernel 里 `Brcb` 和 `src1BlkStride = 0` 是什么？ | 2.5、5.2 |
| arch35 那套 `__VEC_SCOPE__` 在干嘛？ | 2.6、5.3 |
| 从 Python 调一次到结果出来，中间发生了什么？ | 第 6 章 |
| 我卡住了 | 第 8 章 FAQ |

---

## 1. 先搞懂 hc_post 在算什么

### 1.1 一句话版本

> **hc_post 把 4 份 hidden state 按 `comb` 矩阵混合，再加上当前层输出 `x` 的门控贡献，写回成 4 份新的 hidden state。**

如果把 `hc_pre` 比作「从 4 条水管里取水混成 1 条」，那 `hc_post` 就是「把 1 条主水管的水，按配方分回 4 条水管」，并且还要把这 4 条水管里**原有的水**按 `comb` 互相勾兑一下。

### 1.2 和 hc_pre 的对称关系

先看一眼两者在模型里的位置（`DeepseekV4DecoderLayer.forward`）：

```text
hidden_states ──┬──> hc_pre ──> y ──> LayerNorm ──> Attn ──> MoE ──┐
                │                                                  │
                └──> residual (4 份副本) ───────────────────────────┤
                                                                   ▼
                                                        hc_post(x=MoE输出,
                                                                residual=旧副本,
                                                                post, comb) ──> 新的 4 份副本
```

| 维度 | `hc_pre` | `hc_post` |
| --- | --- | --- |
| 方向 | 4 → 1 | 1 → 4（加 4→4 混合） |
| 计算单元 | Cube + Vector（MIX） | **纯 Vector**（无矩阵乘） |
| 有无 Sinkhorn | 有（20 次迭代） | 无（直接用 hc_pre 算好的 `comb`） |
| 输入个数 | 5（含可选 `pre_mix`） | 4 |
| 输出个数 | 4（含可选 `pre`） | 1 |
| dtype 组合 | 1 种 | **9 种** |
| 代码量（Host+Kernel） | ~8500 行 | ~1800 行 |

一句话：**`hc_pre` 负责「算配方」，`hc_post` 负责「按配方施工」**。`post` 和 `comb` 这两个配方，就是 `hc_pre` 的输出、`hc_post` 的输入，两个算子通过它们解耦。

### 1.3 输入输出

以 `bs` 个 token、`d = 7168`、`hc_mult = 4` 为例：

| 名字 | 形状（3D 调用） | 形状（2D 调用） | dtype | 含义 |
| --- | --- | --- | --- | --- |
| `x` | `(b, s, d)` | `(bs, d)` | fp16 / bf16 / fp32 | 当前层的输出（Attn 或 MoE 的结果） |
| `residual` | `(b, s, 4, d)` | `(bs, 4, d)` | 与 `x` 相同 | 4 份旧的 hidden state |
| `post` | `(b, s, 4)` | `(bs, 4)` | fp16 / bf16 / fp32 | 每条支路的门控 |
| `comb` | `(b, s, 4, 4)` | `(bs, 4, 4)` | 与 `post` 相同 | 4×4 混合矩阵 |
| `y` | `(b, s, 4, d)` | `(bs, 4, d)` | **与 `x` 相同** | 4 份新的 hidden state |

**没有任何属性（attr）**——`hc_mult` 是从 `residual` 的形状里读出来的（[hc_post_tiling.cpp L98-L106](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.cpp#L98-L106)）。这是和 `hc_pre` 的一个明显差异。

### 1.4 数学公式

$$
y[b,j,h] \;=\; \sum_{i=0}^{3} residual[b,i,h]\cdot comb[b,i,j] \;+\; x[b,h]\cdot post[b,j]
$$

逐个符号解释：

- `b`：第几个 token
- `i`：**输入**支路下标（0..3），来自 `residual` 的第 2 维
- `j`：**输出**支路下标（0..3），对应 `y` 的第 2 维
- `h`：hidden 维下标（0..d-1）
- `comb[b,i,j]`：第 i 条旧支路对第 j 条新支路的贡献权重
- `post[b,j]`：当前层输出 `x` 对第 j 条新支路的门控

**注意下标顺序**：`comb` 的第 2 维是 `i`（源），第 3 维是 `j`（目标）。写成矩阵形式就是：

$$
Y = C^{\top} R + \mathbf{1}\,(post \otimes x), \qquad C\in\mathbb{R}^{4\times4},\; R\in\mathbb{R}^{4\times d}
$$

其中 $Y_j = \sum_i C_{ij} R_i$，即 $Y = C^{\top}R$。**这是最容易写错的地方**：如果写成 $CR$，就把「谁混合到谁」搞反了。

### 1.5 一个数值小例子

取 `d = 1`，`x = 10`，`post = [0.1, 0.2, 0.3, 0.4]`，

```text
residual = [1, 2, 3, 4]ᵀ
comb = ⎡0.5 0.2 0.2 0.1⎤    ← 第 0 行：旧支路 0 分给 4 条新支路的比例
       ⎢0.1 0.6 0.2 0.1⎥    ← 第 1 行
       ⎢0.2 0.1 0.6 0.1⎥    ← 第 2 行
       ⎣0.2 0.1 0.0 0.7⎦    ← 第 3 行（每行和为 1，因为 comb 是 Sinkhorn 出来的双随机矩阵）
```

逐项算：

```text
y[0] = 1×0.5 + 2×0.1 + 3×0.2 + 4×0.2 + 10×0.1 = 0.5+0.2+0.6+0.8+1.0 = 3.1
y[1] = 1×0.2 + 2×0.6 + 3×0.1 + 4×0.1 + 10×0.2 = 0.2+1.2+0.3+0.4+2.0 = 4.1
y[2] = 1×0.2 + 2×0.2 + 3×0.6 + 4×0.0 + 10×0.3 = 0.2+0.4+1.8+0.0+3.0 = 5.4
y[3] = 1×0.1 + 2×0.1 + 3×0.1 + 4×0.7 + 10×0.4 = 0.1+0.2+0.3+2.8+4.0 = 7.4
```

**小白总结**：`hc_post` = 4 次「标量 × 整行向量 + 累加」，共 5 条乘加链（4 条残差 + 1 条 x）。**没有归约、没有 exp、没有矩阵乘**，所以它比 `hc_pre` 简单得多，也注定是**纯 Vector 算子**。

---

## 2. Ascend C 必背 10 分钟（hc_post 版）

> 完整入门见 [1_npu_hc_pre_npu.md 第 2 章](1_npu_hc_pre_npu.md)。这里只保留 hc_post 真正用到的部分。

### 2.1 只用 AIV，不用 AIC

`hc_pre` 是 MIX 算子（Cube + Vector），入口有 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`。

`hc_post` 全是逐元素乘加，**没有一次矩阵乘**，所以它是**纯 Vector 算子**，入口没有 MIX 声明，`blockDim` 直接取 `GetCoreNumAiv()`（[hc_post_tiling.cpp L28](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.cpp#L28)）。

> 判断技巧：看 Host tiling 里拿的是 `GetCoreNumAiv()` 还是 `GetCoreNumAic()`。前者是纯 Vector，后者是纯 Cube，两个都拿才是 MIX。

### 2.2 GM → UB → 计算 → UB → GM

```text
GM: residual[b, 0..3, h]    ← 4 行、每行 d 个数
        │ DataCopyPad（一次搬 [4, dOnceDealing] 的小块）
        ▼
UB: residualUb[4, 4096]     ← 4 行对齐存放
        │ Brcb 把 comb[i][j] 广播成 8 份
        ▼
UB: sumReg = residual[0]*comb[0][j] + ... + x*post[j]
        │ Cast fp32 → bf16
        ▼
GM: y[b, j, h]
```

关键约束：**UB 只有 ~192KB**。`d = 7168`，若一次搬 4 行 bf16 需要 `4 × 7168 × 2B = 56KB`，看起来够；但 fp32 的中间结果 `sumTempBuf` 又是 `4 × 7168 × 4B = 112KB`，加上输入输出缓冲就爆了。所以必须**按 D 维切块**——这就是 tiling 里 `dOnceDealing` 的由来。

### 2.3 Tiling 只做两件事

`hc_post` 的 tiling 极其简单，只有两层切分：

```text
第一层：batch 维分给多个核          → usedCoreNum / batchOneCore / frontCore
第二层：每个核内部再按 D 维切块      → dSplitTime / dOnceDealing / dLastDealing
```

没有 Cube 的 M/K 二维切分，没有 MPMD，没有 workspace 复用（虽然申请了 16MB 但实际没用到）。**这是读懂本算子最大的省力点**。

### 2.4 `TQue`：带自动同步的缓冲区

```cpp
TQue<QuePosition::VECIN, 1> inputQue_;   // 输入队列
TQue<QuePosition::VECOUT, 1> outQue_;    // 输出队列

pipe_->InitBuffer(inputQue_, 2, size);   // 2 = 双缓冲（ping-pong）
LocalTensor<T1> t = inputQue_.AllocTensor<T1>();
DataCopyPad(t, gm[offset], params);      // GM → UB
inputQue_.EnQue<T1>(t);                  // 入队：隐含 SetFlag(MTE2 → V)
t = inputQue_.DeQue<T1>();               // 出队：隐含 WaitFlag(V ← MTE2)
// ... 计算 ...
inputQue_.FreeTensor(t);                 // 归还
```

`EnQue/DeQue` 帮你做了流水线同步，所以业务代码里看不到 `SetFlag/WaitFlag`。**但要注意**：`TBuf<QuePosition::VECCALC>`（比如 `tempSumBuf_`）是**裸缓冲**，没有队列语义，改完必须手动 `PipeBarrier<PIPE_V>()`。

### 2.5 两个必须看懂的指令

#### (1) `Brcb`：把一个数变成 8 个

Vector 指令的工作粒度是 **1 个 block = 32 字节 = 8 个 fp32**。而 `comb[i][j]` 是**一个标量**，要乘到整行 `d` 个元素上，怎么办？

`Brcb(dst, src, repeatTimes, {blockStride=1, repeatStride=8})` 的效果：

```text
src（1 个 block，8 个数）:  [c0 c1 c2 c3 c4 c5 c6 c7]
dst（repeatTimes=2）:
  repeat 0: [c0 c0 c0 c0 c0 c0 c0 c0]
  repeat 1: [c1 c1 c1 c1 c1 c1 c1 c1]
  ...
```

所以 `Brcb` 一次能把 8 个标量各复制 8 份，铺满 8 个 block。

#### (2) `src1BlkStride = 0`：让同一个 block 被重复读

`Mul(dst, src0, src1, calcCount, repeatTimes, params)` 里的 `params` 描述「每处理完一个 repeat，三个操作数的指针各往前跳几个 block」：

```cpp
BinaryRepeatParams p;
p.src0BlkStride = 1;   // src0 正常往后走
p.src1BlkStride = 0;   // ★ src1 原地不动 → 同一个权重一直复用
p.src1RepStride = 0;   // ★ 每轮 repeat 也不动
```

配合 `Brcb`，就实现了**「一个标量 × 一整行向量」**：

```text
src0: [x0 x1 x2 ... x63][x64 ...]      ← 每 repeat 前进
src1: [w  w  w  ... w  ][w  w  ...]    ← 永远读同一块
dst : [x0*w ... x63*w][x64*w ...]
```

### 2.6 MicroAPI（arch35）：更省事的写法

arch35（ascend950）上，同一件事只需一条指令：

```cpp
// 一次完成「读 1 个 comb 元素 + 广播到 256B 寄存器 + 转 fp32」
MicroAPI::DataCopy<T2, LoadDist::DIST_BRC_B16>(combRegFloat0, combAddr + hcIndex);
MicroAPI::Mul(sumRegFloat, residualRegFloat0, combRegFloat0, pMask);
MicroAPI::MulAddDst(sumRegFloat, residualRegFloat3, combRegFloat3, pMask);
```

| 经典写法（A2/A3） | MicroAPI（A5/950） |
| --- | --- |
| `DataCopyPad` + `Cast` + `Brcb` + `PipeBarrier`（4 步） | `DataCopy<DIST_BRC_B16>`（1 步） |
| 手算 `repeatTimes / numRepeatPerLine` | `RegTensor` 宽度固定 256B，`UpdateMask<float>(n)` 管尾巴 |
| 显式 `PipeBarrier<PIPE_V>` | `__VEC_SCOPE__` 块内由编译器/硬件保证 |
| `src1BlkStride=0` 广播 | `LoadDist::DIST_BRC_*` 广播 |

`LoadDist` 的几种取值在源码里都能看到：

| 枚举 | 作用 |
| --- | --- |
| `DIST_BRC_B16` / `DIST_BRC_B32` | 读一个元素并广播（B16 = 16bit，B32 = 32bit） |
| `DIST_UNPACK_B16` | 读 16bit 数据并「解包」成 32bit 布局（配合后面 Cast） |

### 2.7 两套实现（又是 2 套，不是 3 套）

| SoC | 字符串 | 走的 kernel | 编译宏 |
| --- | --- | --- | --- |
| A2 / 910B | `ascend910b` | `HcPostKernelDSplit`（经典 LocalTensor） | 无 |
| A3 / 910_93 | `ascend910_93` | `HcPostKernelDSplit`（同上） | 无 |
| A5 / 950 | `ascend950` | `HcPostRegBaseFloat32` 或 `HcPostRegBaseBfloat16`（MicroAPI） | `__DAV_C310__` |

判据同样是 SoC 字符串：[hc_post_tiling.cpp L213-L217](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.cpp#L213-L217) 里 `socVersion == ASCEND950` 走 `HcPostTilingRegbase`；kernel 侧用 `__DAV_C310__` 选择（[hc_post.cpp L17-L21](../../../../csrc/moe/hc_post/op_kernel/hc_post.cpp#L17-L21)）。

**小白总结**：hc_post 是纯 Vector 算子；只需要理解「GM↔UB 搬运 + 双缓冲 + 一个标量乘一整行」；A2/A3 用 `Brcb + src1BlkStride=0`，A5 用 `DIST_BRC_*`。

---

## 3. 代码地图

```text
csrc/moe/hc_post/
├── op_host/                              ← Host 侧（CPU）
│   ├── hc_post_def.cpp            55 行  算子原型：4 输入 1 输出、9 种 dtype、3 个 SoC
│   ├── hc_post_proto.cpp          64 行  InferShape / InferDtype（图模式用）
│   ├── hc_post_tiling.cpp        231 行  通用版 Tiling（A2/A3）+ SoC 分流
│   ├── hc_post_tiling.h          112 行  TilingData 结构 + 两个 Tiling 类声明
│   └── hc_post_tiling_arch35.h   234 行  arch35 Tiling（A5）
└── op_kernel/                            ← Device 侧（NPU）
    ├── hc_post.cpp                58 行  内核入口 + 两套实现分派
    ├── hc_post_d_split.h         394 行  通用版 kernel（A2/A3，经典 LocalTensor）
    ├── hc_post_bfloat16.h        351 行  arch35 bf16 kernel（MicroAPI）
    └── hc_post_float32.h         322 行  arch35 fp32 kernel（MicroAPI）
```

推荐阅读顺序：

1. `hc_post_def.cpp` —— 接口长什么样（特别是 9 列 dtype）
2. `hc_post_proto.cpp` —— 输出形状怎么推
3. `hc_post_tiling.h` —— Host 要算哪 10 个参数
4. `hc_post_tiling.cpp` —— 两行核心切分公式
5. `hc_post.cpp` —— 怎么按 SoC/dtype 分派
6. `hc_post_d_split.h` —— 经典写法，最适合入门（同步关系最完整）
7. `hc_post_bfloat16.h` / `hc_post_float32.h` —— 新写法

---

## 4. Host 侧：10 个参数搞定

### 4.1 `hc_post_def.cpp`：9 种 dtype 组合怎么读

先看 [L24-L48](../../../../csrc/moe/hc_post/op_host/hc_post_def.cpp#L24-L48)。`DataType({...})` 里的列表**按列对应**，每一列是一种合法的 dtype 组合：

| 列 | `x` / `residual` / `y` | `post` / `comb` |
| --- | --- | --- |
| 0 | fp16 | fp16 |
| 1 | fp16 | **bf16** |
| 2 | fp16 | **fp32** |
| 3 | bf16 | fp16 |
| 4 | bf16 | bf16 |
| 5 | bf16 | fp32 |
| 6 | fp32 | fp16 |
| 7 | fp32 | bf16 |
| 8 | fp32 | fp32 |

**这就是 9 种组合的由来：3 × 3 的笛卡尔积。** 注意：

- `x`、`residual`、`y` 三者**必须同 dtype**（`residual` 复制 `x` 的列表，`y` 也复制 `x` 的列表）
- `post`、`comb` 二者**必须同 dtype**（`comb` 复制 `post` 的列表）
- 但 `x` 和 `post` 之间**可以不同**（列 1、2、3... 就是交叉组合）

tiling 里的校验把这条规则写得明明白白（[hc_post_tiling.cpp L114-L153](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.cpp#L114-L153)）：

```cpp
residualDtype != xDtype   → 报错
combDtype != postDtype    → 报错
```

**为什么允许 x 和 post 不同 dtype？** 因为 `x/residual/y` 是大张量（`bs × 4 × 7168`），用 bf16 省带宽；`post/comb` 是小张量（`bs × 4` 和 `bs × 16`），用 fp32 保精度更划算。两者解耦后可以自由搭配。

**SoC 注册**（[L49-L51](../../../../csrc/moe/hc_post/op_host/hc_post_def.cpp#L49-L51)）：

```cpp
this->AICore().AddConfig("ascend910b");
this->AICore().AddConfig("ascend910_93");
this->AICore().AddConfig("ascend950");     // 没有 regbaseCfg，用默认配置
```

对比 `hc_pre`：`hc_pre` 给 950 传了 `OpAICoreConfig regbaseCfg`（因为它是 MIX 算子，需要定制 opFile），`hc_post` 是纯 Vector，不需要。

### 4.2 `hc_post_proto.cpp`：两行推断

这是 `hc_pre` 没有的文件（`hc_pre_proto.cpp` 是空的）。逻辑极简（[L29-L59](../../../../csrc/moe/hc_post/op_host/hc_post_proto.cpp#L29-L59)）：

```cpp
*yShape = *residualShape;                  // 输出形状 = residual 形状
context->SetOutputDataType(0, xDtype);     // 输出 dtype = x 的 dtype
```

就这么两行。因为 `y` 的形状必须和 `residual` 完全一致（都是 `[.., 4, d]`），dtype 跟 `x` 走。

> 为什么 `hc_pre` 不需要 proto 而 `hc_post` 需要？因为 `hc_pre` 在 vLLM 里只走 aclnn 直调（输出在 C++ binding 里 `at::empty` 建好），而 `hc_post` 要同时支持图模式（GE 会调用 InferShape/InferDtype）。

### 4.3 `hc_post_tiling.h`：10 个字段

```cpp
BEGIN_TILING_DATA_DEF(HcPostTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);        // 实际用几个核
TILING_DATA_FIELD_DEF(int64_t, bsParam);            // 总 token 数（3D 时 b*s）
TILING_DATA_FIELD_DEF(int64_t, hcParam);            // 支路数（通常 4）
TILING_DATA_FIELD_DEF(int64_t, dParam);             // hidden size
TILING_DATA_FIELD_DEF(int64_t, batchOneCore);       // 前 frontCore 个核各处理多少 token
TILING_DATA_FIELD_DEF(int64_t, batchOneCoreTail);   // 其余核各处理多少（= batchOneCore - 1）
TILING_DATA_FIELD_DEF(int64_t, frontCore);          // 有几个「多干一个」的核
TILING_DATA_FIELD_DEF(int64_t, dSplitTime);         // D 维切了几块
TILING_DATA_FIELD_DEF(int64_t, dOnceDealing);       // 每块多少元素
TILING_DATA_FIELD_DEF(int64_t, dLastDealing);       // 尾块多少元素（0 表示整除）
END_TILING_DATA_DEF;
```

见 [L27-L39](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.h#L27-L39)。这 10 个字段在 Kernel 侧被逐个读出（[hc_post_d_split.h L113-L122](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L113-L122)），名字必须完全一致。

### 4.4 Tiling 主流程：两行核心公式

`DoOpTiling()`（[L155-L174](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.cpp#L155-L174)）真的很短：

```cpp
int64_t useCoreNum   = batchSize < coreNum_ ? batchSize : coreNum_;   // ① 核数不能超过 token 数
int64_t batchOneCore = CeilDiv(batchSize, useCoreNum);                // ② 向上取整
int64_t batchOneCoreTail = batchOneCore - 1;
int64_t frontCore    = batchSize - batchOneCoreTail * useCoreNum;     // ③ 有几个核需要多干 1 个

int64_t dSplitTime = dParam_ / DEFAULT_DEAL_DPARAM;                   // ④ 2048 一块
tilingData_.set_dLastDealing(dParam_ - dSplitTime * DEFAULT_DEAL_DPARAM);
context_->SetBlockDim(useCoreNum);
```

#### `frontCore` 到底怎么算的？

假设 `bs = 10`，`coreNum = 4`：

```text
useCoreNum    = 4
batchOneCore  = ceil(10/4) = 3
tail          = 2
frontCore     = 10 - 2*4 = 2      ← 前 2 个核各干 3 个，后 2 个核各干 2 个
                                               3+3+2+2 = 10 ✓
```

Kernel 侧就是用 `blkIdx_ < frontCore` 判断自己属于哪一档（[hc_post_d_split.h L120](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L120)、[L129-L135](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L129-L135)）：

```cpp
int64_t xOffset = blkIdx_ * batchOneCore_ * dParam_;
if (!isFrontCore_) {
    xOffset = (blkIdx_ * batchOneCoreTail_ + frontCore) * dParam_;   // 把前面多出来的补上
}
```

**为什么不用 `blkIdx * batchOneCore` 一刀切？** 因为那样后面的核会越界（`bs=10` 时第 3 个核的起始 offset 是 9，但它要处理 3 个 token → 读到第 12 个，越界）。`frontCore` 公式保证每段起点严格连续。

#### D 维切块：2048 还是 4096？

| 版本 | 常量 | 值 | 依据 |
| --- | --- | --- | --- |
| 通用版（A2/A3） | `DEFAULT_DEAL_DPARAM` | **2048** | [hc_post_tiling.cpp L20](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.cpp#L20) |
| arch35（A5） | `ONCE_DEAL_DPARAM` | **4096** | [hc_post_tiling_arch35.h L34](../../../../csrc/moe/hc_post/op_host/hc_post_tiling_arch35.h#L34) |

为什么 A5 能一次处理两倍？因为 MicroAPI 的寄存器和 UB 使用效率更高，且 arch35 的 kernel 不再需要 `postCastBuf/combCastBuf/outCastBuf` 那一堆额外 fp32 中转（`DIST_BRC` 和 `DIST_UNPACK` 直接搞定）。

`dLastDealing = 0` 是一个**哨兵值**：kernel 里用 `if (dLastDealing_ != 0)` 判断要不要跑尾块（[hc_post_d_split.h L384-L387](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L384-L387)）。

### 4.5 arch35 Tiling：只多了一个 TilingKey

arch35 的 tiling 代码和通用版几乎一样，只多了两件事（[hc_post_tiling_arch35.h L186-L206](../../../../csrc/moe/hc_post/op_host/hc_post_tiling_arch35.h#L186-L206)）：

**(1) 按 dtype 选 TilingKey**（[L178-L182](../../../../csrc/moe/hc_post/op_host/hc_post_tiling_arch35.h#L178-L182)）：

```cpp
if (xDtype == ge::DT_FLOAT) { tilingKey_ = 0; }   // HC_POST_FLOAT
else                        { tilingKey_ = 1; }   // HC_POST_BFLOAT16
```

注意判据是 **`x` 的 dtype**，不是 post：

- `x` 是 fp32 → key 0 → `HcPostRegBaseFloat32`
- `x` 是 bf16 或 **fp16** → key 1 → `HcPostRegBaseBfloat16`

也就是说 **fp16 输入复用了 bf16 的 kernel**。这安全吗？安全，因为 kernel 里所有计算都会先 `Cast` 成 fp32，fp16/bf16 在 UB 里只是「16 位存储」，位宽相同，`DIST_UNPACK_B16` / `DIST_BRC_B16` 对两者都成立。

**(2) D 块大小改成 4096**（[L200-L203](../../../../csrc/moe/hc_post/op_host/hc_post_tiling_arch35.h#L200-L203)）。

另外通用版在 `PostTiling()` 里固定 `SetTilingKey(0)`（[hc_post_tiling.cpp L178](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.cpp#L178)），因为经典 kernel 只有一个模板类，不需要分派。

### 4.6 通用版 vs arch35 对比

| 维度 | 通用版（A2/A3） | arch35（A5/950） |
| --- | --- | --- |
| TilingKey | 固定 0 | 0（x=fp32）/ 1（x=16bit） |
| D 块大小 | 2048 | 4096 |
| 计算核 | AIV | AIV |
| workspace | 16MB（申请但不使用） | 16MB（同上） |
| Kernel 类 | `HcPostKernelDSplit<T1,T2>` | `HcPostRegBaseFloat32<T>` / `HcPostRegBaseBfloat16<T1,T2>` |
| 线程模型 | 经典 `LocalTensor` + `TQue` | `RegTensor` + `__VEC_SCOPE__` |

**小白总结**：Host 侧只做「token 怎么分给核 + D 怎么切块 + 选哪个 kernel」。没有 Cube，没有 workspace 魔法，10 个 int64 就是一封完整的施工说明。

---

## 5. Kernel 侧：三种实现

### 5.1 入口 `hc_post.cpp`

```cpp
extern "C" __global__ __aicore__ void hc_post(GM_ADDR x, GM_ADDR residual, GM_ADDR post,
    GM_ADDR comb, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
#if defined(__DAV_C310__)
    GET_TILING_DATA_WITH_STRUCT(HcPostTilingData, tilingData, tiling);
    if (TILING_KEY_IS(HC_POST_FLOAT)) {           // 0
        HcPostRegBaseFloat32<DTYPE_POST> op;      // ← 模板参数是 post 的 dtype！
        op.Init(...); op.Process(); return;
    } else if (TILING_KEY_IS(HC_POST_BFLOAT16)) { // 1
        HcPostRegBaseBfloat16<DTYPE_X, DTYPE_POST> op;
        op.Init(...); op.Process(); return;
    }
#else
    HcPostKernelDSplit<DTYPE_X, DTYPE_POST> op;  // A2/A3：唯一路径
    op.Init(...); op.Process(); return;
#endif
}
```

见 [hc_post.cpp L32-L58](../../../../csrc/moe/hc_post/op_kernel/hc_post.cpp#L32-L58)。

三个细节：

1. **`DTYPE_X` / `DTYPE_POST` 是编译期宏**，由算子编译框架按 `def.cpp` 里的 dtype 组合展开。所以同一个 `hc_post` 源文件会被编译成 9 个二进制变体。
2. `HcPostRegBaseFloat32<DTYPE_POST>` 只吃一个模板参数——因为 x 一定是 fp32 时，`T1` 就固定是 float，只需知道 `post` 的 dtype。
3. **没有 `GetUserWorkspace`**，因为根本不用 workspace（`workspace` 参数只是签名占位）。

### 5.2 通用版：`hc_post_d_split.h`

#### 类结构

```cpp
template <typename T1, typename T2>
class HcPostKernelDSplit {
    // 5 个 GlobalTensor：x / residual / post / comb / y
    // 4 个 TQue：inputQue_(x 和 residual 共用) / postQue_ / combQue_ / outQue_
    // 7 个 TBuf：inputCastBuf_ / postCastBuf_ / combCastBuf_ / outCastBuf_ / tempSumBuf_ / postBrcbBuf_ / combBrcbBuf_
};
```

见 [L51-L101](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L51-L101)。

**「DSplit」的含义**：这个 kernel 同时做了两件事——`D` 维分块（每个 token 的 7168 个元素分多次搬）+ 多核切 `batch`。名字里的 `D_Split` 指的是前者。

#### `Init`：先算地址偏移，再分配 UB

[L103-L158](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L103-L158)：

```cpp
blkIdx_ = GetBlockIdx();
if (blkIdx_ >= tilingData->usedCoreNum) return;   // ★ 多余的核直接退出

// ① 每个核负责的 batch 段起点（见 4.4 的 frontCore 分析）
int64_t xOffset = blkIdx_ * batchOneCore_ * dParam_;
if (!isFrontCore_) xOffset = (blkIdx_ * batchOneCoreTail_ + frontCore) * dParam_;

// ② 把 GlobalTensor 的基址挪到自己的段上，后续代码就只用相对下标
xGm_.SetGlobalBuffer((__gm__ T1 *)x + xOffset);

// ③ UB 分配（全部按 dOnceDealing 预算）
pipe_->InitBuffer(inputQue_,  2, hcParam_ * dOnceDealing_ * sizeof(T1));   // x 和 residual 交替复用
pipe_->InitBuffer(outQue_,    2, hcParam_ * dOnceDealing_ * sizeof(T1));
pipe_->InitBuffer(postQue_,   2, hcParam_ * sizeof(T2));                   // 小得可怜
pipe_->InitBuffer(combQue_,   2, hcParam_ * hcParamAlign_ * sizeof(T2));
if constexpr (sizeof(T1) == 2) {   // bf16/fp16 才需要 fp32 中转缓冲
    pipe_->InitBuffer(outCastBuf_,   hcParam_ * dOnceDealing_ * sizeof(float));
    pipe_->InitBuffer(inputCastBuf_, hcParam_ * dOnceDealing_ * sizeof(float));
}
if constexpr (sizeof(T2) == 2) {
    pipe_->InitBuffer(postCastBuf_, hcParam_ * sizeof(float));
    pipe_->InitBuffer(combCastBuf_, hcParam_ * RoundUp<T2>(hcParam_) * sizeof(float));
}
pipe_->InitBuffer(postBrcbBuf_, 64 * sizeof(float));   // ★ 固定 64：8 个 fp32 × 8 次 repeat
pipe_->InitBuffer(combBrcbBuf_, 64 * sizeof(float));
pipe_->InitBuffer(tempSumBuf_, hcParam_ * dOnceDealing_ * sizeof(float));
```

两个值得注意的点：

- **`inputQue_` 被 `x` 和 `residual` 共用**。因为 `x` 只有 1 行、`residual` 有 4 行，但两者不会同时存活（先用完 `x` 再搬 `residual`）。共用能省一块 UB。
- **`if constexpr`** 是关键：fp32 输入时 `inputCastBuf_` / `outCastBuf_` 根本不会被分配，省下 `2 × 4 × 2048 × 4B = 64KB`。这是编译期分支，没有运行时开销。

#### `DoProcess`：三层循环

[L361-L391](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L361-L391)：

```cpp
for (batchIndex = 0; batchIndex < batchSize; batchIndex++) {   // ① 本核负责的 token
    DataCopyInPost(batchIndex);      // post[b, 0..3]      → 4 个标量
    DataCopyInComb(batchIndex);      // comb[b, 0..3, 0..3] → 16 个标量
    for (dLoop = 0; dLoop < dSplitTime_; dLoop++) {             // ② D 维满块
        DoCompute(..., dOnceDealing_);
    }
    if (dLastDealing_ != 0) {                                   // ③ D 维尾块
        DoCompute(..., dLastDealing_);
    }
}
```

`post` 和 `comb` 是**每个 token 只搬一次**，然后在 D 维循环里反复使用——这是一个很重要的优化：`post/comb` 的数据量只有 `4 + 16 = 20` 个数，但要在 `7168` 个元素上重复使用 3584 倍。

#### `DoCompute`：公式的 1:1 翻译

[L298-L359](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L298-L359)：

```cpp
// 第 1 步：out[j] = x * post[j]        （j = 0..3，x 只有 1 行）
DataCopyInX(batchIndex, dLoop, dealNum);
Cast(inputCastBuf, xUb, ...);                                  // bf16 → fp32
DoMal<float>(inputCastBuf, postBrcb, outBuf, hcParam_, dealNum);

// 第 2 步：搬 4 行 residual
DataCopyInResidual(batchIndex, dLoop, dealNum);
for (i = 0; i < hcParam_; i++) Cast(inputCastBuf[i*align], residualUb[i*align], ...);

// 第 3 步：out[j] += Σ_i residual[i][j] * comb[i][j]
for (hcIndex = 0; hcIndex < hcParam_; hcIndex++) {
    Brcb(comBrcb, combUb[hcIndex * RoundUp(hcParam_)], ...);    // 把 comb[hcIndex][*] 广播
    DoMal<float>(inputCastBuf[hcIndex*align], comBrcb, sumTempBuf, hcParam_, dealNum);
    DoAdd<float>(outBuf, sumTempBuf, outBuf, hcParam_, dealNum); // out += sum
}
```

**注意 `DoMal` 里 `src0Local[0]` 的写法**（[L272-L274](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L272-L274)）：

```cpp
for (i = 0; i < curRowNum; i++)
    Mul(dstLocal[i*align], src0Local[0], src1Local[i*elemInOneBlock], ...);
```

`src0Local[0]` 恒等于第 0 行。所以第 1 步里 `x`（只有 1 行）能被乘到全部 4 行上——**这是「x 广播到 4 条支路」的实现**，不需要真的把 `x` 复制 4 份。

第 3 步则相反：`src0Local[hcIndex*align]` 是 `residual` 的第 `hcIndex` 行，`src1` 是 `comb[hcIndex][j]` 广播后的向量，结果 `sumTempBuf[j] = residual[hcIndex][j] * comb[hcIndex][j]`，累加 4 次就得到 $\sum_i$。

#### `DataCopyInComb` 的 padding 技巧

[L215-L228](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L215-L228)：

```cpp
uint8_t padNum = BLOCK_SIZE / sizeof(T2) - hcParam_;   // 32/4 - 4 = 4
DataCopyPadExtParams<T2> p{true, 0, padNum, 0};        // isPad=true, leftPad=0, rightPad=4
DataCopyPad(combUb, combGm_[...], copyParams, p);
```

`comb` 是 `[4, 4]`，每行只有 16 字节，**不满一个 32 字节 block**。如果直接搬，后面的 `Brcb` 会读到脏数据。所以右边补 4 个元素（补到 8 个），保证 `RoundUp(hcParam_)=8` 成立。

这就是 Host 侧 `hcParamAlign_` 和 UB 分配 `hcParam_ * hcParamAlign_` 的由来——**一切为了 32 字节对齐**。

### 5.3 arch35 之 bf16：`hc_post_bfloat16.h`

#### 和通用版最大的结构差异：循环顺序反了

```text
通用版：  for dLoop { out[0..3] = x*post; for i { out += residual[i]*comb[i] } }   ← D 外层
arch35 bf16：for hcIndex(j) { for dChunk { sum_j = Σ_i residual[i]*comb[i][j] + x*post[j] } }  ← 通道外层
```

看 [L250-L307](../../../../csrc/moe/hc_post/op_kernel/hc_post_bfloat16.h#L250-L307)：

```cpp
for (hcIndex = 0; hcIndex < hcTimes; hcIndex++) {          // 输出通道 j
    // 一次 load 就完成「读 comb[i][j] + 广播 + 转 fp32」
    DataCopy<T2, DIST_BRC_B16>(combReg0, combAddr + hcIndex);
    DataCopy<T2, DIST_BRC_B16>(combReg1, combAddr + hcParam_ + hcIndex);
    ...
    for (j = 0; j < repeatTimes; j++) {                    // 256B 一块
        DataCopy<T1, DIST_UNPACK_B16>(xReg, xAddr + j*vfLen);
        DataCopy<T1, DIST_UNPACK_B16>(residualReg0, residualAddr + j*vfLen);
        DataCopy<T1, DIST_UNPACK_B16>(residualReg1, residualAddr + xDealNumAlign + j*vfLen);
        ...
        Mul(sumTempReg0, residualRegFloat0, combRegFloat0, pMask);         // 第一条：赋值
        MulAddDst(sumTempReg0, residualRegFloat3, combRegFloat3, pMask);   // 后三条：累加
        MulAddDst(sumTempReg0, residualRegFloat1, combRegFloat1, pMask);
        MulAddDst(sumTempReg0, residualRegFloat2, combRegFloat2, pMask);
        MulAddDst(sumTempReg0, xRegFloat, postRegFloat, pMask);            // 最后加 x*post[j]
        DataCopy(sumAddr + hcIndex*xDealNumAlign + j*vfLen, sumTempReg0, pMask);
    }
}
```

**为什么用 `Mul` 起头、`MulAddDst` 收尾？** 因为寄存器没有「清零再累加」的便宜操作。第一条 `Mul` 同时完成「初始化累加器」，比 `Muls(sum, 0)` + 4 次 `MulAddDst` 少一条指令。

**注意累加顺序是 0,3,1,2**（[L282-L285](../../../../csrc/moe/hc_post/op_kernel/hc_post_bfloat16.h#L282-L285)）。数学上无影响（加法交换律），但源码这么写通常是为了匹配某个寄存器分配/流水调度顺序，不要以为是 bug。

**`vfLen = 256 / sizeof(float) = 64`**：一个 `RegTensor` 是 256 字节，装 64 个 fp32。所以内层循环一次处理 64 个 `h`。

**尾块处理**（[L289-L306](../../../../csrc/moe/hc_post/op_kernel/hc_post_bfloat16.h#L289-L306)）：

```cpp
uint16_t repeatTimes  = dOnceDealing / vfLen;             // 完整 64 元素块的个数
uint16_t tailLoopTimes = (dOnceDealing % vfLen == 0) ? 0 : 1;
```

尾块单独跑一次，`DataCopy` 会多读/写最多 63 个元素，但：

- UB 缓冲区是按 `RoundUp<32B>` 对齐分配的，多写不会越界；
- 最后 `DataCopyOut` 只写 `dDealing` 个元素（[L192-L201](../../../../csrc/moe/hc_post/op_kernel/hc_post_bfloat16.h#L192-L201)），多算的垃圾被丢弃。

**`d = 7168` 能整除 64 吗？** `7168 / 64 = 112`，正好整除，所以实际推理时 `tailLoopTimes = 0`，走不到尾块分支。

#### 写回：4 行一次搬走

[L311-L325](../../../../csrc/moe/hc_post/op_kernel/hc_post_bfloat16.h#L311-L325)：

```cpp
DoMulAndAdd(...);                                          // 4 个通道的结果都在 sumTempBuf
LocalTensor<T1> sumUb = sumQue_.AllocTensor<T1>();
AscendC::Cast(sumUb, sumTempBuf, RoundMode::CAST_RINT, hcParam_ * dOnceDealing_);  // ★ 一次 Cast 4×D
sumQue_.EnQue<T1>(sumUb);
DataCopyOut(batchIndex, dDealing, dOffset);                 // ★ 一次搬 4 行
```

`CAST_RINT` 是「四舍六入五取偶」的舍入模式（round to nearest even），和 PyTorch 的 bf16 cast 行为一致。用 `CAST_NONE`（截断）会引入系统性偏差。

### 5.4 arch35 之 fp32：`hc_post_float32.h`

结构几乎和 bf16 版一样（同样的 `__VEC_SCOPE__`、同样的 `DIST_BRC`、同样的 4 路 `MulAddDst`），差异只有两点：

**(1) 不需要 Cast**

`if constexpr (sizeof(T2) == 2)` 的分支全部不成立，直接 `DIST_BRC_B32` 读进 `RegTensor<float>`：

```cpp
DataCopy<T2, DIST_BRC_B32>(combRegFloat0, combAddr + hcIndex);
```

（[L255-L261](../../../../csrc/moe/hc_post/op_kernel/hc_post_float32.h#L255-L261)）

**(2) 逐通道写出，而不是 4 行一起**

看 `DoCompute`（[L280-L296](../../../../csrc/moe/hc_post/op_kernel/hc_post_float32.h#L280-L296)）：

```cpp
for (hc1Index = 0; hc1Index < hcParam_; hc1Index++) {
    DoMulAndAdd(..., hc1Index, dDealing);
    LocalTensor<float> sumUb = sumQue_.AllocTensor<float>();
    AscendC::Copy(sumUb, sumTempBuf, dDealing);       // 只拷当前通道
    sumQue_.EnQue<float>(sumUb);
    DataCopyOut(batchIndex, hc1Index, dDealing, dOffset);   // 只写 [b, hc1Index, :]
}
```

**为什么 fp32 要逐通道写、bf16 却能一次写 4 行？** 因为 fp32 的 `sumTempBuf` 是 `4 × 4096 × 4B = 64KB`，加上 4 行 residual（`4×4096×4B=64KB`）和 x（16KB），再要一块 64KB 的输出缓冲就接近 UB 上限了。逐通道写可以复用同一块 `sumUb`，把输出缓冲从 64KB 降到 16KB。

bf16 版则因为 `sumUb` 是 bf16（`4 × 4096 × 2B = 32KB`），可以承担一次性写 4 行。

### 5.5 三套 kernel 对比

| 维度 | `HcPostKernelDSplit` | `HcPostRegBaseBfloat16` | `HcPostRegBaseFloat32` |
| --- | --- | --- | --- |
| 平台 | A2 / A3 | A5 / 950 | A5 / 950 |
| 编程接口 | `LocalTensor` | `RegTensor` | `RegTensor` |
| 广播 | `Brcb` + `src1BlkStride=0` | `DIST_BRC_B16` | `DIST_BRC_B32` |
| 循环顺序 | D 外层 / 通道内层 | 通道外层 / D 内层 | 通道外层 / D 内层 |
| 累加方式 | 先 `Mul` 到 `sumTempBuf`，再 `Add` 到 `outBuf`（两趟） | `Mul` + 4×`MulAddDst`（一趟） | 同左 |
| Cast 次数 | 每轮都 cast（输入/输出各一次） | 输入 unpack + 输出一次 `CAST_RINT` | 无 |
| 输出粒度 | 一次 4 行 | 一次 4 行 | 一次 1 行 |
| UB 峰值 | 低（fp32 时会跳过 cast buffer） | 中 | 高 |
| TilingKey | 0（唯一） | 1 | 0 |

**小白总结**：三套 kernel 的数学完全一致，差别只在「用哪套指令」和「循环怎么排」。通用版是「一个 D 块算完 4 个通道」，arch35 是「一个通道算完整个 D」，这样能把 `comb[i][j]` 的广播提到最外层，减少重复 load。

---

## 6. 一次调用的完整旅程

```text
① Python（V4.1）：vllm_ascend/models/deepseek_v41/model.py#L871-L877
     torch.ops._C_ascend.npu_hc_post(
         x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0))
     ※ unsqueeze(0) 把 [T,d] 变 [1,T,d]，走 3D 分支（bs = 1*T）
        │
        ▼
② 算子注册：csrc/torch_binding.cpp#L3534
     ops.impl("npu_hc_post", torch::kPrivateUse1, &npu_hc_post_npu)
        │
        ▼
③ C++ binding：csrc/torch_binding.cpp#L1527-L1538
     ├── check_hc_post_shape_and_dtype(...)
     ├── construct_hc_post_output_tensor(residual)   → y 形状 = residual 形状
     └── EXEC_NPU_CMD(aclnnHcPost, x, residual, post, comb, out)
        │
        ▼
④ InferShape / InferDtype：hc_post_proto.cpp#L29-L59
     y.shape = residual.shape;  y.dtype = x.dtype
        │
        ▼
⑤ Host Tiling：hc_post_tiling.cpp#L203-L220
     ├── SoC == ASCEND950 ? HcPostTilingRegbase : HcPostTiling
     ├── GetShapeInfo     → bs / d / hc（hc 从 residual 形状读，不靠 attr）
     ├── GetPlatformInfo  → GetCoreNumAiv()、UB size
     └── DoOpTiling       → usedCoreNum / batchOneCore / frontCore / dSplitTime
        │
        ▼
⑥ Device Kernel（blockDim = usedCoreNum 个 AIV 核）
     hc_post.cpp#L32
     ├── [A2/A3]  HcPostKernelDSplit<T1,T2>::Process()
     │     for token: 搬 post/comb → for d块: out=x*post; out+=Σ residual*comb → 写回
     └── [A5]     HcPostRegBaseBfloat16 / Float32::Process()
           for token: 搬 post/comb → for d块: MicroAPI 5 条乘加 → Cast → 写回
        │
        ▼
⑦ 返回 y：[b, s, 4, d]（与 residual 同形状）
```

**文件汇总表**：

| 层 | 文件 | 职责 |
| --- | --- | --- |
| Python | [model.py](../model.py#L871-L877) | `unsqueeze(0)` 后调 `npu_hc_post` |
| Binding | [torch_binding.cpp](../../../../csrc/torch_binding.cpp#L1527-L1538) | 校验、建输出、`EXEC_NPU_CMD` |
| 原型 | [hc_post_def.cpp](../../../../csrc/moe/hc_post/op_host/hc_post_def.cpp) | 9 组 dtype、3 个 SoC |
| 形状推导 | [hc_post_proto.cpp](../../../../csrc/moe/hc_post/op_host/hc_post_proto.cpp) | `y.shape = residual.shape` |
| Host | [hc_post_tiling.cpp](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.cpp) / [hc_post_tiling_arch35.h](../../../../csrc/moe/hc_post/op_host/hc_post_tiling_arch35.h) | 两层切分 |
| Device | [hc_post.cpp](../../../../csrc/moe/hc_post/op_kernel/hc_post.cpp) | 入口 + 分派 |
| Device | [hc_post_d_split.h](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h) | A2/A3 实现 |
| Device | [hc_post_bfloat16.h](../../../../csrc/moe/hc_post/op_kernel/hc_post_bfloat16.h) / [hc_post_float32.h](../../../../csrc/moe/hc_post/op_kernel/hc_post_float32.h) | A5 实现 |

---

## 7. 性能设计清单

| 优化 | 做法 | 位置 |
| --- | --- | --- |
| **算子融合** | 4×4 混合 + 门控 + 累加 + 类型转换一次完成，省 4 次 GM 往返 | 整体 |
| **batch 多核** | `useCoreNum` 个核各领一段连续 token，`frontCore` 保证无越界 | `DoOpTiling` |
| **D 维分块** | 每核内按 2048（A5 为 4096）切块，避免 UB 溢出 | `DoOpTiling` |
| **post/comb 只搬一次** | 每个 token 只搬 20 个数，在 D 循环里反复用 | `DoProcess` |
| **双缓冲** | `InitBuffer(que, 2, size)`，`EnQue/DeQue` 自动 ping-pong | `Init` |
| **共用 inputQue_** | `x` 和 `residual` 顺序使用，不额外分配 UB | `Init` |
| **`if constexpr` 裁剪** | fp32 输入时不分配 cast 缓冲，省 64KB UB | `Init` L147-L154 |
| **标量广播** | `Brcb` + `src1BlkStride=0`，权重不重复 load | `DoMal` |
| **`src0Local[0]` 复用** | `x` 只有 1 行，靠这个写法广播到 4 行，避免复制 | `DoMal` |
| **一次 Cast 4 行** | arch35 bf16 把 `4×D` 的 fp32 一次转 bf16、一次写回 | `DoCompute` L319-L322 |
| **一次 load 带广播** | `DIST_BRC_B16/B32` 省掉 `Brcb` + `PipeBarrier` | arch35 |
| **`Mul` 起头** | 第一条 `Mul` 兼做累加器初始化，少一条 `Muls` | `DoMulAndAdd` |
| **UB 对齐 padding** | `comb` 右边补 4 个元素凑满 32B block | `DataCopyInComb` |

---

## 8. FAQ

**Q1：为什么 `hc_post` 没有属性（attr），`hc_mult` 从哪来？**
从 `residual` 的形状读：2D 时是 `residual.size(1)`，3D 时是 `residual.size(2)`（[hc_post_tiling.cpp L98-L106](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.cpp#L98-L106)）。这样算子不用关心模型配置，形状即事实。

**Q2：`comb` 的索引到底是 `[i][j]` 还是 `[j][i]`？**
`[i][j]`，`i` 是**源**（residual 支路），`j` 是**目标**（y 支路）。代码里 `combUb[hcIndex * align(hc)]` 取第 `hcIndex` 行，乘到 `residual[hcIndex]` 上，累加进 `out` 的全部 4 行 → 对应 $C_{ij}$。见 [hc_post_d_split.h L329-L343](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L329-L343)。

**Q3：为什么 `x` 只有 1 行，却要乘 4 个 `post`？**
这是公式设计的：当前层输出 `x` 对 4 条新支路各有独立门控 `post[j]`。实现上靠 `DoMal` 里的 `src0Local[0]` 把同一行 `x` 复用 4 次。

**Q4：fp16 输入走 bf16 kernel，精度会不会丢？**
不会额外丢。kernel 内部所有计算都 `Cast` 成 fp32，fp16 只是「搬运时的存储格式」，和 bf16 位宽相同，`DIST_UNPACK_B16` 对两者都适用。真正的精度损失只发生在最后 `CAST_RINT` 回写时，这与 PyTorch 行为一致。

**Q5：`dLastDealing = 0` 是什么意思？**
`d` 能被 2048（或 4096）整除时，没有尾块，这个字段就是 0。kernel 用 `if (dLastDealing_ != 0)` 跳过尾块处理。`d = 7168 = 3×2048 + 1024`，所以通用版会有尾块 1024；arch35 下 `7168 = 1×4096 + 3072`，尾块 3072。

**Q6：为什么 `blkIdx_ >= usedCoreNum` 要直接 return？**
`blockDim` 已经设成 `usedCoreNum` 了，理论上不会有空闲核。但某些调度模式下框架可能启动更多物理核，这个判断是防御性的（[hc_post_d_split.h L107-L110](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L107-L110)）。

**Q7：`workspace` 申请了 16MB，但代码里完全没用？**
对，`workspace` 在 `hc_post.cpp` 里连 `GetUserWorkspace` 都没调用。16MB 是历史遗留的保守占位，不影响正确性，只是多占一点显存。

**Q8：为什么 `Init` 里要 `SetGlobalBuffer` 加偏移，而不是每次地址都手算？**
把「第几个核负责哪一段」这件事在 `Init` 里一次性解决，后面所有 `DataCopy` 都只写相对下标（如 `batchIndex * dParam_`），代码可读性大幅提升，也不容易算错。

**Q9：`Brcb` 的 `repeatTimes` 为什么是 `CeilDiv(hcParam_, 64)` 而不是 `CeilDiv(hcParam_, 8)`？**
`Brcb` 的第三个参数是「处理几个 block」，而 `REPEAT_NUM = 64` 是常量（[hc_post_d_split.h L29](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L29)）。`hcParam_=4` 时 `CeilDiv(4,64)=1`，即只处理 1 个 block（8 个数），足够覆盖 4 个值。这个写法在 `hcParam_` 很小时有点「过度保守」，但不会错。

**Q10：`hc_post` 会不会和多线程/多卡冲突？**
不会。每个核只写自己负责的 `batch` 段，段之间严格不重叠（`frontCore` 公式保证），天然无写冲突，不需要跨核同步。

---

## 9. 术语表

| 术语 | 含义 |
| --- | --- |
| **hc_mult / hcParam** | 支路数，DeepSeek-V4 固定 4 |
| **comb** | 4×4 混合矩阵，来自 `hc_pre` 的 Sinkhorn 输出 |
| **post** | 每条输出支路的门控，来自 `hc_pre` |
| **AIV** | AI Vector，hc_post 只用这一种计算单元 |
| **UB** | Unified Buffer，Vector 工作内存（~192KB） |
| **Tiling** | Host 侧切分决策 |
| **TilingData** | Host → Device 的 10 个 int64 参数 |
| **TQue** | 带自动同步的缓冲区（`EnQue`/`DeQue`） |
| **TBuf** | 无队列语义的裸缓冲，需手动 `PipeBarrier` |
| **Brcb** | 广播指令，把标量复制 8 份铺满 block |
| **`src1BlkStride=0`** | 让同一个权重 block 在 repeat 间重复读 |
| **`RoundUp<T>(n)`** | 向上对齐到 32 字节（`32/sizeof(T)` 个元素） |
| **`DIST_BRC_B16/B32`** | MicroAPI 的「读并广播」模式 |
| **`DIST_UNPACK_B16`** | 读 16bit 数据并解包成 32bit 布局 |
| **`__VEC_SCOPE__`** | MicroAPI 的寄存器直算代码块 |
| **`CAST_RINT`** | 就近取偶舍入，与 PyTorch cast 一致 |
| **frontCore** | 需要多处理 1 个 token 的核的数量 |
| **arch35** | ascend950（A5）regbase 架构 |
| **`__DAV_C310__`** | arch35 编译宏 |

---

## 10. 核对说明

| 结论 | 核对位置 |
| --- | --- |
| 4 输入 1 输出、9 种 dtype 列、3 个 SoC | [hc_post_def.cpp L24-L51](../../../../csrc/moe/hc_post/op_host/hc_post_def.cpp#L24-L51) |
| `y.shape = residual.shape`、`y.dtype = x.dtype` | [hc_post_proto.cpp L45-L46](../../../../csrc/moe/hc_post/op_host/hc_post_proto.cpp#L45-L46)、[L55-L56](../../../../csrc/moe/hc_post/op_host/hc_post_proto.cpp#L55-L56) |
| TilingData 10 字段 | [hc_post_tiling.h L27-L39](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.h#L27-L39) |
| `DEFAULT_DEAL_DPARAM = 2048` | [hc_post_tiling.cpp L20](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.cpp#L20) |
| batch 切分与 `frontCore` 公式 | [hc_post_tiling.cpp L155-L174](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.cpp#L155-L174) |
| 通用版 TilingKey 固定 0、workspace 16MB | [hc_post_tiling.cpp L176-L184](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.cpp#L176-L184) |
| SoC 分流 | [hc_post_tiling.cpp L203-L220](../../../../csrc/moe/hc_post/op_host/hc_post_tiling.cpp#L203-L220) |
| arch35 `ONCE_DEAL_DPARAM = 4096` | [hc_post_tiling_arch35.h L34](../../../../csrc/moe/hc_post/op_host/hc_post_tiling_arch35.h#L34) |
| arch35 TilingKey 0/1 判据 | [hc_post_tiling_arch35.h L178-L182](../../../../csrc/moe/hc_post/op_host/hc_post_tiling_arch35.h#L178-L182) |
| `__DAV_C310__` 与两套分派 | [hc_post.cpp L17-L21](../../../../csrc/moe/hc_post/op_kernel/hc_post.cpp#L17-L21)、[L36-L57](../../../../csrc/moe/hc_post/op_kernel/hc_post.cpp#L36-L57) |
| 通用版 Init 地址偏移与 buffer 分配 | [hc_post_d_split.h L103-L158](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L103-L158) |
| `DoMal` 的 `src0Local[0]` 广播 | [hc_post_d_split.h L272-L274](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L272-L274) |
| 公式实现（4 次乘加 + x*post） | [hc_post_d_split.h L329-L343](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L329-L343) |
| `comb` 的 padding | [hc_post_d_split.h L215-L228](../../../../csrc/moe/hc_post/op_kernel/hc_post_d_split.h#L215-L228) |
| arch35 bf16 的 `Mul` + 4×`MulAddDst` | [hc_post_bfloat16.h L282-L287](../../../../csrc/moe/hc_post/op_kernel/hc_post_bfloat16.h#L282-L287) |
| arch35 bf16 一次 Cast 4 行 | [hc_post_bfloat16.h L319-L322](../../../../csrc/moe/hc_post/op_kernel/hc_post_bfloat16.h#L319-L322) |
| arch35 fp32 逐通道写出 | [hc_post_float32.h L287-L293](../../../../csrc/moe/hc_post/op_kernel/hc_post_float32.h#L287-L293) |
| Python 侧 `unsqueeze(0)` | [model.py L789-L793](../model.py#L789-L793) |

> 若后续分支改动导致行号漂移，以上「文件 + 函数名」仍可作为稳定锚点。
