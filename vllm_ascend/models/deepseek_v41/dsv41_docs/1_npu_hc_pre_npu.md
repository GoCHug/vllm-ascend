# `hc_pre` 算子：从零开始读懂 Ascend C 实现

> 面向读者：完全没写过 Ascend C 算子，但看得懂 Python 和一点 C++。
> 阅读方式：**第 1、2 章是从零铺垫，不要跳**；第 3 章之后才是源码精读。每一章结尾都有「小白总结」，赶时间只看总结也能串起来。

> **V4.1 适配说明（务必先读这一节，再读下文）**：本文原为 DeepSeek V4 编写。所述算子的 **C++ / Ascend C 实现位于 `csrc/moe/hc_pre/`，在 DeepSeek V4.1 中一行未改**，所以下文第 3~17 章对 csrc 的精读、行号、TilingKey、性能清单**全部仍然有效**。V4.1 的差异只在「Python 入口」和「系数怎么交接」两处：

| 差异 | DeepSeek V4 | DeepSeek V4.1 |
|------|-------------|---------------|
| Python 入口 | `npu_hc_pre_v2`（返回 3 元组） | **`npu_hc_pre_v3`（返回 4 元组，多一个 `pre`）** |
| `pre_mix` 入参 | 无（binding 传 `c10::nullopt`） | **有**（可选张量，`(T, hc_mult)` fp32） |
| 调用点 | `models/deepseek_v4/model.py:L783` | **`models/deepseek_v41/model.py:L858-L869`** |
| `pre` 的消费者 | 仅 KV-cache 场景自用 | **下一层的注意力**（延迟系数交接，见 §0.1） |
| `hidden_size` | 7168 | **5120**（下文以 7168 举例的形状请等比理解） |
| `mix_hc = (2+hc_mult)*hc_mult` | 24 | **24（不变）** |
| `hc_mult` / `hc_sinkhorn_iters` / `hc_eps` | 4 / 20 / 1e-6 | **4 / 20 / 1e-6（不变）** |

> 一句话：**算子没变，变的是「谁来算 `pre`、`pre` 给谁用」**。V4 里 `pre` 是可选输出（只在 KV-cache 场景被缓存复用，见下文 Q8）；V4.1 里 `pre` 是**每层必用的产物**，本层 FFN 算出的 `pre` 直接喂给**下一层的注意力**。

## 0.1 V4.1 的调用契约（`npu_hc_pre_v3`）

### 0.1.1 签名与返回值

`csrc/torch_binding.cpp` 的注册（L3546-L3552）与 C++ 实现（L1656-L1668）：

```cpp
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_v3_npu(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    const c10::optional<at::Tensor>& pre_mix,      // ← v3 新增：可选的「上一轮算好的 pre」
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult);
    return run_hc_pre_fusion(x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
}
```

| | `npu_hc_pre_v2`（V4） | `npu_hc_pre_v3`（V4.1） |
|---|---|---|
| `pre_mix` | 无 | **有**（`c10::optional`，`nullopt` 时等价于 v2） |
| 返回值 | `(y, post, comb_frag)` | **`(y, post, comb_frag, pre)`** |
| `check_hc_pre_shape_and_dtype` | 校验时把 `pre_mix` 固定为 `nullopt` | 校验 `pre_mix` 的形状/dtype（`(bs, 4)` fp32） |
| 底层 | 同一个 `run_hc_pre_fusion` | **同一个 `run_hc_pre_fusion`** |

**关键结论**：v3 不是「新算子」，而是同一个融合算子的**四输出 + 可选入参**版本。传入 `pre_mix` 时，Kernel 会走 `PREMIX_TILING_KEY` 短路掉 Part1 的矩阵乘（见下文 §4.5 与 §7 的「premix 短路」一行）；不传时行为与 v2 完全一致。

### 0.1.2 V4.1 的两处调用点

`DeepseekV41DecoderLayer`（`models/deepseek_v41/model.py`）：

```python
# L858-L869：薄封装，把「常量参数」固化在方法里
def hc_pre(self, x, hc_fn, hc_scale, hc_base, pre_mix=None):
    return torch.ops._C_ascend.npu_hc_pre_v3(
        x, hc_fn, hc_scale, hc_base, pre_mix,
        hc_mult=self.hc_mult,
        hc_sinkhorn_iters=self.hc_sinkhorn_iters,
        norm_eps=self.norm_eps,
        hc_eps=self.hc_eps)

# L854-L856：模型出口用的「只读坍缩」，等价于 hc_pre 的 Part2-y 部分
@staticmethod
def hc_collapse(x, pre_mix):
    return (pre_mix.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)
```

```python
# L889-L895：注意力分支 —— 读上一层的 ffn_pre
x, attn_post, attn_comb, attn_pre = self.hc_pre(
    hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, pre_mix)

# L905-L911：FFN 分支 —— 读本次注意力刚算出的 attn_pre
x, ffn_post, ffn_comb, ffn_pre = self.hc_pre(
    hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, attn_pre)

# L920：把 ffn_pre 交给下一层
return hidden_states, ffn_pre
```

模型层的循环把 `pre_mix` 一路传下去（L1563-L1565 初始化 one-hot、L1611 逐层传递、L1621 用最后一层的 `ffn_pre` 做坍缩）：

```python
pre_mix = hidden_states.new_zeros(hidden_states.shape[0], self.hc_mult, dtype=torch.float32)
pre_mix[:, 0] = 1.0                                  # 第 0 层：等价于「只读第 0 条流」
...
hidden_states, pre_mix = layer(positions, hidden_states, pre_mix, None, input_ids=input_ids)
...
hidden_states = last_layer.hc_collapse(hidden_states, pre_mix)
```

### 0.1.3 为什么 V4.1 要这样改

V4 的每次 `hc_pre` 都要先算 `mixes = F.linear(flatten(x), hc_fn) * rsqrt`，**再用 `mixes` 里的 pre 段坍缩 x**。这意味着：

```
x ──► Cube(hc_fn 投影) ──► Vector(rsqrt/sigmoid/Sinkhorn) ──► Vector(坍缩 y) ──► LN ──► Attention
```

`hc_fn` 投影是 **24×20480 的矩阵乘**（V4.1 下 `hc_dim = 4*5120 = 20480`），必须等本层拿到 `x` 才能开始，于是它卡在关键路径最前面。

V4.1 让「上一层 FFN」顺手把下一层要用的 `pre` 算出来：

```
层 N 的 FFN:  hc_pre(hc_ffn_fn, pre_mix=attn_pre) ──► y ──► MoE ──► hc_post ──► x_{N+1}
                    └────────────────────────────► ffn_pre ─────────────┐
                                                                          ▼
层 N+1 的注意力:  hc_pre(hc_attn_fn, pre_mix=ffn_pre) → 传了 pre_mix → 短路 Part1 → 直接坍缩
```

好处有三点：

1. **关键路径少一次依赖**：注意力不必等本层的 `hc_attn_fn` 投影，`pre` 早在上一层就算好了。
2. **Cube 利用率更高**：`hc_ffn_fn` 的投影与 `hc_attn_fn` 的投影可以错峰（前者在上一层 FFN 段，后者即使仍要算也不在等 `x`）。
3. **数值上更省一次归一化**：`pre` 已经包含了 `rsqrt` 归一化的效果，直接用它坍缩即可（Kernel 里走 `PREMIX_TILING_KEY`）。

> **注意**：`hc_attn_fn` 的 `post`/`comb` **仍然要算**（它们依赖本层的 `x`），所以 v3 在 `pre_mix` 模式下不是「完全不进 Cube」，而是「Part1 里跳过 pre 那一段」。具体短路范围见下文 §4.5 的 `PREMIX_TILING_KEY`。

### 0.1.4 与下文的对齐指引

| 下文位置 | 在 V4.1 下的读法 |
|---------|-----------------|
| §1.2 输入输出表 | `d` 从 7168 改成 5120；`pre` 从「可选输出」变成「必有输出」 |
| §1.3 `hc_fn` 的 24 行切分 | **不变**（24 = (2+4)×4） |
| §4.5 通用版 TilingKey / §4.6 arch35 TilingKey | **不变**；`hasPreMix` 分支在 V4.1 是**常态**而非 KV-cache 特例 |
| §6 「一次调用的完整旅程」 | 把 `npu_hc_pre_v2` 换成 `npu_hc_pre_v3`，把 `L783` 换成 `model.py#L858` |
| §8 Q8「为什么 `pre` 是可选输出」 | 在 V4.1 下应读成「`pre` 是**必有**输出，且由下一层消费」 |


## 0. 这份文档怎么读

| 你的问题 | 看哪里 |
| --- | --- |
| hc_pre 到底在算什么？数学公式是什么？ | 第 1 章 |
| Ascend C 是什么？AIC/AIV、UB、Tiling 都是啥？ | 第 2 章 |
| 这个算子有哪些文件？先看哪个？ | 第 3 章 |
| 为什么 Host 侧要算那么多参数？ | 第 4 章 |
| Kernel 里那一堆 `Brcb`、`MulAddDst` 在干嘛？ | 第 5 章 |
| 从 Python 调一次，到底发生了什么？ | 第 6 章 |
| 为什么它跑得快？ | 第 7 章 |
| 我卡在某个概念上了 | 第 8 章 FAQ、第 9 章术语表 |
| `m_k_split` 的 m / k 是什么？ | 4.4.1 |
| 某个具体函数到底在干嘛？ | 第 10 章导读 → 第 11~17 章「函数级详解」 |
| Host tiling 的每个函数 | 第 11 章（通用版）、第 12 章（arch35） |
| Vector 计算原语的每个函数 | 第 13 章（通用版）、第 14 章（arch35） |
| Kernel 主流程的每个函数 | 第 15 章（通用版）、第 17 章（arch35） |
| Cube 矩阵乘的每个函数 | 第 16 章 |

---

## 1. 先搞懂 hc_pre 在算什么

### 1.1 一句话版本

DeepSeek-V4 的 **mHC（Manifold-Constrained Hyper-Connections）** 里，每个 token 不是只有 1 份 hidden state，而是有 **4 份副本**（`hc_mult = 4`）。

`hc_pre` 干的事就是：

> **看这 4 份副本，算出一套权重，把它们压成 1 份；顺手把后面写回时要用的两个门控（`post`、`comb`）也算好。**

它是「读」的那一半，`hc_post` 是「写」的那一半（见 [2_npu_hc_post_npu.md](2_npu_hc_post_npu.md)）。

### 1.2 输入输出长什么样

假设一个 batch 有 `bs` 个 token，隐藏维 `d = 7168`，`hc_mult = 4`：

| 名字 | 形状 | dtype | 小白解释 |
| --- | --- | --- | --- |
| `x` | `(b, s, 4, d)` 或 `(bs, 4, d)` | bf16 | 4 份 hidden state 副本 |
| `hc_fn` | `(24, 4*d)` | fp32 | **权重矩阵**，24 行是关键（见 1.3） |
| `hc_scale` | `(3,)` | fp32 | 3 个缩放系数：pre / post / comb 各一个 |
| `hc_base` | `(24,)` | fp32 | 24 个偏置，和 `hc_fn` 的 24 行一一对应 |
| `pre_mix` | `(b, s, 4)` | fp32 | 可选。KV-cache 场景下用上一轮算好的 pre，省一次矩阵乘 |
| `y` | `(b, s, d)` | bf16 | 压缩后的 1 份 hidden state |
| `post` | `(b, s, 4)` | fp32 | 写回时 4 条支路的门控 |
| `comb_frag` | `(b, s, 4, 4)` | fp32 | 4×4 混合矩阵（Sinkhorn 归一化后） |
| `pre` | `(b, s, 4)` | fp32 | 可选输出，读到 Python 侧做缓存 |

形状校验在 [torch_binding.cpp](../../../../csrc/torch_binding.cpp#L1582-L1627)：`hc_mult` 只能是 4，`hc_fn.shape[0]` 只能是 24，`hc_scale` 长度只能是 3。

### 1.3 关键：`hc_fn` 的 24 行是怎么切分的

这是整套逻辑的「密码本」，记住这张图后面就不会迷路：

```text
hc_fn 的 24 行 = 4 + 4 + 16
┌──────────┬──────────┬──────────────────────────────┐
│ 行 0..3  │ 行 4..7  │ 行 8..23                     │
│  pre     │  post    │  comb（4×4 = 16 个）         │
│ 决定怎么  │ 决定怎么  │ 决定 4 份残差之间怎么互相混合  │
│ 读 4 份   │ 写回 4 份 │                              │
└──────────┴──────────┴──────────────────────────────┘
   用 scale[0]  用 scale[1]   用 scale[2]
   base[0:4]    base[4:8]     base[8:24]
```

对应源码里的三处偏移：

- pre 用 `hcBaseGm[0 .. hcMult)`、`hcScaleGm(0)` — [hc_pre_m_k_split_core.h](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L662-L671)
- post 用 `hcBaseGm[hcMult .. 2*hcMult)`、`hcScaleGm(1)`
- comb 用 `hcBaseGm[2*hcMult .. 24)`、`hcScaleGm(2)` — [hc_pre_m_k_split_core.h](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L851-L852)

### 1.4 四步数学公式

记：
- `x_j ∈ R^d`：第 `j` 份副本（`j = 0..3`）
- `k = 4 * d`：4 份拼起来的长度
- `m = hc_fn @ x̂`，其中 `x̂` 是归一化后的拼接向量，得到 24 个数 `m_0 .. m_23`

**Step 1：InvRMS 归一化**

$$
r=\frac{1}{\sqrt{\frac{1}{k}\sum_{i=1}^{k}x_i^2+\varepsilon_{\text{norm}}}}
$$

**Step 2：线性投影**

$$
m=\big(hc\_fn\ \cdot\ (r\cdot x)\big)\in\mathbb{R}^{24}
$$

**Step 3：门控（sigmoid）**

$$
pre_j=\sigma\!\left(m_j\cdot r\cdot scale_0+base_j\right)+\varepsilon,\quad j=0..3
$$

$$
post_j=2\,\sigma\!\left(m_{4+j}\cdot r\cdot scale_1+base_{4+j}\right),\quad j=0..3
$$

**Step 4：压缩 + 混合矩阵**

$$
y=\sum_{j=0}^{3}pre_j\,x_j
$$

$$
comb=\operatorname{Sinkhorn}_{20}\!\Big(\operatorname{softmax}\big(m_{8:24}\cdot r\cdot scale_2+base_{8:24}\big)+\varepsilon\Big)\in\mathbb{R}^{4\times4}
$$

三个 `ε` 是不同用途的：`norm_eps` 防 RMS 除零，`hc_eps` 防止门控被压成 0 后 Sinkhorn 无法归一化。

**小白总结**：`hc_pre` = 一次 RMSNorm + 一次小矩阵乘（`24 × 4d`）+ 若干 sigmoid + 一个 4×4 的 Sinkhorn 迭代 + 一次加权求和。

---

## 2. Ascend C 零基础铺垫

### 2.1 为什么不能像写 Python 那样写算子

Python 里 `y = (x * w).sum(-2)` 一行搞定，因为框架帮你调度好了。写成**算子**时要自己回答三个问题：

1. **数据现在在哪？** —— 在 GM（Global Memory，显存/HBM）里，计算单元碰不到它。
2. **搬到哪里算？** —— 搬到核内的 UB / L1 / L0，容量只有几百 KB。
3. **怎么切？** —— 一次搬不完，得分块循环，块之间还要做流水和同步。

这三件事分别对应 Ascend C 里的 **DataCopy**、**LocalTensor**、**Tiling**。

### 2.2 硬件：AI Core = AIC + AIV

一颗 Ascend 芯片里有几十个 **AI Core**，每个 AI Core 内部又分两种计算单元：

| 单元 | 全称 | 擅长 | 类比 |
| --- | --- | --- | --- |
| **AIC** | AI Cube | 矩阵乘（`Mmad`） | CPU 的矩阵加速卡 |
| **AIV** | AI Vector | 逐元素运算（加、乘、exp、cast） | CPU 的 SIMD 向量单元 |

`hc_pre` 是一个 **MIX 算子**（Cube 和 Vector 都要用），入口第一行就声明了配比：

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
```

`1_2` 意思是 **1 个 AIC 配 2 个 AIV**。所以你会看到源码里 `GetBlockIdx()` 同时被 Cube 侧和 Vector 侧使用，以及 `CrossCoreSetFlag/CrossCoreWaitFlag` 这种「两个单元握手」的指令。

相关代码：[hc_pre.cpp](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp#L38)。

### 2.3 内存层级：越靠近计算单元越快越小

```text
       容量大、速度慢
┌──────────────────────────────┐
│ GM（Global Memory，几十 GB）  │  ← 输入输出张量都在这
└───────────┬──────────────────┘
            │ DataCopy / DataCopyPad（搬运）
┌───────────▼──────────────────┐
│ L1（Cube 的缓存，~1 MB）      │  ← Cube 的输入 A/B 先放这
└───────────┬──────────────────┘
            │ LoadData
┌───────────▼──────────────────┐
│ L0A / L0B / L0C（几十 KB）    │  ← 真正做 Mmad 的寄存器阵列
└──────────────────────────────┘

┌──────────────────────────────┐
│ UB（Unified Buffer，~192 KB） │  ← Vector 的所有运算都在这
└──────────────────────────────┘
       容量小、速度快
```

两个直接后果：

- **UB 只有 ~192KB**，所以 `d = 7168` 一行都放不下 4 份（`4 × 7168 × 2B = 56KB`，加上 fp32 中间结果就更紧张）→ 必须**按 D 维切块**（源码里的 `dFactor_`、`dLoop_`）。
- **GM 到 UB 的搬运有固定开销**，所以要**一次搬一大块**再算，而不是一个元素一个元素搬。

### 2.4 Host 与 Device 的分工

Ascend C 算子永远由两部分组成：

| 角色 | 跑在哪 | 能不能看到真实数据 | 干什么 |
| --- | --- | --- | --- |
| **Host**（Tiling） | CPU | ❌ 只知道 shape/dtype | 算「怎么切」：几个核、每核几行、UB 放多大块 |
| **Device**（Kernel） | NPU | ✅ | 按 Host 给的参数执行搬运和计算 |

Host 算出来的这堆参数，会打包成一个 **TilingData 结构体**，序列化成一段二进制，通过 `tiling` 指针传给 Kernel。Kernel 侧用 `GET_TILING_DATA` 反序列化：

```cpp
GET_TILING_DATA_WITH_STRUCT(HcPreTilingData, tiling_data_in, tiling);
```

这就是为什么 Host 侧那些 `set_xxx()` 和 Kernel 侧的 `tilingData->xxx` 名字必须一模一样。

### 2.5 一个 Ascend C 算子的「五件套」

任何自定义算子都逃不出这五样东西，`hc_pre` 也不例外：

| # | 概念 | 作用 | hc_pre 里对应文件 |
| --- | --- | --- | --- |
| 1 | **OpDef（算子原型）** | 声明有几个输入/输出/属性、什么 dtype | [hc_pre_def.cpp](../../../../csrc/moe/hc_pre/op_host/hc_pre_def.cpp) |
| 2 | **InferShape / InferDtype** | 图模式下推导输出形状和类型 | [hc_pre_proto.cpp](../../../../csrc/moe/hc_pre/op_host/hc_pre_proto.cpp)（**空文件**，见 4.2） |
| 3 | **Tiling** | Host 侧算切分参数 | [hc_pre_tiling.cpp](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp)、[hc_pre_tiling_arch35.h](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h) |
| 4 | **TilingKey** | 用整数「选路」，一套算子挂多套实现 | 0/1/2 或 1000~1003 |
| 5 | **Kernel** | Device 侧真正干活的函数 | [hc_pre.cpp](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp) + 各 `*.h` |

**TilingKey 是新手最容易懵的地方**，先记住一句话：

> **TilingKey 就是一个 switch-case 的分支号。** Host 侧根据 shape 决定走哪条路，把号写进 tiling；Kernel 侧用 `TILING_KEY_IS(n)` 分派到不同的模板类。

### 2.6 经典写法：`LocalTensor` + `TPipe/TQue/TBuf`

A2（910B）、A3（910_93）走的是这套「经典 Vector 编程」：

```cpp
GlobalTensor<T> gm;        // 指到 GM 的句柄
LocalTensor<T> ub;         // 指到 UB 的句柄
TPipe pipe;                // 资源总管，负责分配 UB
TQue<QuePosition::VECIN, 1> inQue;   // 带队列语义的缓冲区（自动同步）
TBuf<QuePosition::VECCALC> calcBuf;  // 纯计算缓冲区（手动同步）

pipe.InitBuffer(inQue, 2, size);     // 2 = 双缓冲（ping-pong）
LocalTensor<T> t = inQue.AllocTensor<T>();
DataCopyPad(t, gm[offset], params);  // GM → UB
inQue.EnQue(t);                      // 入队（隐含 SetFlag）
t = inQue.DeQue<T>();                // 出队（隐含 WaitFlag）
// ... 计算 ...
inQue.FreeTensor(t);
```

必背的几个指令家族（本文会反复出现）：

| 指令 | 干什么 | 类比 |
| --- | --- | --- |
| `DataCopyPad` | GM ↔ UB 搬运，支持非 32B 对齐 | `memcpy` + 自动 padding |
| `DataCopyExtParams` | 描述搬运形状：`blockCount` 行、每行 `blockLen` 字节、行间 `srcStride` | 带 stride 的 `memcpy2d` |
| `Cast` | 类型转换（bf16 ↔ fp32） | `.to(torch.float32)` |
| `Muls` / `Adds` | 张量 × 标量 / 张量 + 标量 | `x * s` / `x + s` |
| `Mul` / `Add` / `Div` | 逐元素二元运算 | `a * b` |
| `MulAddDst` | `dst += a * b`，一条指令完成乘加 | `a * b + d` |
| `Brcb` | 把 1 个数复制成 8 个相同值，铺满一个 block | `torch.repeat_interleave` 的硬件版 |
| `ReduceSum` / `WholeReduceSum` | 沿某一维求和 | `.sum(dim)` |
| `GatherMask` | 按 bit-mask 抽取元素 | 高级 `gather` |
| `PipeBarrier<PIPE_V>` | 等 Vector 流水线跑完 | 同步点 |
| `SetFlag` / `WaitFlag` | 跨流水线（MTE2/V/MTE3）握手 | 事件同步 |
| `CrossCoreSetFlag` / `CrossCoreWaitFlag` | AIC ↔ AIV 跨核握手 | 跨进程信号量 |

**为什么到处都是 `Brcb`？** 因为 Vector 指令是「一个 block = 32 字节 = 8 个 fp32」为粒度工作的，而 `pre_j` 这种标量要乘到整行 `d` 个元素上。`Brcb` 把一个标量变成 8 个相同值，再配合 `src1BlkStride = 0`（同一个 block 重复读），就能把标量广播到整行——这就是 [hc_pre_base.h](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L648-L691) 里 `ProcessYHc4` 的核心技巧。

### 2.7 新写法：MicroAPI（arch35 / ascend950）

A5（ascend950）引入了 **regbase 架构**，Vector 侧新增了一套更接近汇编的接口，源码里叫 **MicroAPI**：

| 经典写法 | MicroAPI 写法 | 说明 |
| --- | --- | --- |
| `LocalTensor<T>` | `RegTensor<T>` | 直接映射到向量寄存器（256B） |
| 无 | `MaskReg` + `UpdateMask<float>(n)` | 显式掩码，处理尾巴 |
| 无 | `__VEC_SCOPE__ { ... }` | 标记「这段是寄存器直算」 |
| `DataCopy(t, gm)` | `MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B16>` | 搬运 + 广播一次完成 |
| `Cast(dst, src, ...)` | `MicroAPI::Cast<float, T, castB16ToB32>` | 带 layout/sat 参数的 Cast |
| `Mul` / `Add` | `MicroAPI::Mul` / `MicroAPI::MulAddDst` | 一条指令一个寄存器宽度 |

MicroAPI 的关键优势是 **`DIST_BRC_B16` 这种「带广播语义的 load」**：一次就把 `comb[i][j]` 广播到 256B 寄存器里，省掉了经典写法里 `Brcb` + `PipeBarrier` 的往返。

`hc_pre` 里用编译宏区分两套实现：

```cpp
#if defined(__DAV_C310__)     // arch35，ascend950（950PR / 950DT）
  #include "hc_pre_m_k_split_core_arch35.h"      // MicroAPI
#else                          // A2 / A3
  #include "hc_pre_m_k_split_core.h"           // LocalTensor
#endif
```

见 [hc_pre.cpp](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp#L19-L30)。

### 2.8 两套实现不是三套

新手常问「A2、A3、A5 是不是三套代码？」——**不是**，是两套：

| 代号 | SoC 字符串 | 走哪套 | 编译宏 |
| --- | --- | --- | --- |
| A2 / 910B | `ascend910b` | 通用版（经典 LocalTensor + 手写 Mmad） | 无 |
| A3 / 910_93 | `ascend910_93` | 通用版（同上） | 无 |
| A5 / 950PR / 950DT | `ascend950` | arch35（MicroAPI + regbase） | `__DAV_C310__` |

判据是 SoC 字符串，不是产品名：[hc_pre_tiling.cpp](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L736-L742) 里 `socVersion == ASCEND950` 就走 `HcPreTilingRegbase`，其余走 `HcPreTiling`；注册见 [hc_pre_def.cpp](../../../../csrc/moe/hc_pre/op_host/hc_pre_def.cpp#L75-L83)。

**小白总结**：AIC 算矩阵、AIV 算逐元素；数据在 GM，必须搬到 UB/L1；Host 只算「怎么切」，Device 才碰数据；TilingKey 就是 switch 分支号；A2/A3 一套老写法，A5 一套新写法。

---

## 3. 代码地图

先把所有文件摆在桌面上（行数为当前分支实测）：

```text
csrc/moe/hc_pre/
├── op_host/                          ← Host 侧（跑在 CPU）
│   ├── hc_pre_def.cpp          87 行  算子原型：IO + attr + SoC 注册
│   ├── hc_pre_proto.cpp         0 行  ★ 空文件（见 4.2）
│   ├── hc_pre_tiling.cpp      749 行  通用版 Tiling（A2/A3）
│   ├── hc_pre_tiling.h        194 行  TilingData 结构 + 类声明
│   └── hc_pre_tiling_arch35.h 813 行  arch35 Tiling（A5）
└── op_kernel/                        ← Device 侧（跑在 NPU）
    ├── hc_pre.cpp             140 行  内核入口 + TilingKey 分派
    ├── hc_pre_base.h          929 行  通用版公共算子（ProcessPre/Y/Post/Sinkhorn/Softmax）
    ├── hc_pre_base_arch35.h  1418 行  arch35 公共算子（VFProcessPre/Y/Sinkhorn）
    ├── hc_pre_cube_compute.h  551 行  通用版手写 Cube（Mmad）
    ├── hc_pre_cube_compute_arch35.h 241 行  arch35 Cube
    ├── hc_pre_m_k_split_core.h 1460 行  通用版 Part1/Part2 主流程
    ├── hc_pre_m_k_split_core_arch35.h 471 行  arch35 K 切分
    ├── hc_pre_m_k_split_core_premix_arch35.h 445 行  arch35 K 切分 + premix
    ├── hc_pre_m_split_core_arch35.h 478 行  arch35 M 切分
    └── hc_pre_m_split_core_premix_arch35.h 539 行  arch35 M 切分 + premix
```

**推荐阅读顺序**（也是本文顺序）：

1. `hc_pre_def.cpp` —— 先知道接口长什么样
2. `hc_pre_tiling.h` —— 看 Host 到底算了哪些参数
3. `hc_pre_tiling.cpp` 的 `DoOpTiling` —— 看 Host 主流程
4. `hc_pre.cpp` —— 看 Kernel 怎么按 key 分派
5. `hc_pre_base.h` 里的 `ProcessPre/ProcessY/ProcessPost/SoftmaxFP32Perf` —— 看数学怎么落地
6. `hc_pre_m_k_split_core.h` 的 `ProcessPrePostBatch` —— 看整体串起来
7. 最后再碰 cube 和 arch35

---

## 4. Host 侧：先决定「怎么切」

### 4.1 算子原型 `hc_pre_def.cpp`

核心就三块：

**(1) 输入输出声明**（[L24-L68](../../../../csrc/moe/hc_pre/op_host/hc_pre_def.cpp#L24-L68)）

```cpp
this->Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16})...;
this->Input("hc_fn").ParamType(REQUIRED).DataType({ge::DT_FLOAT})...;
this->Input("hc_scale")...;
this->Input("hc_base")...;
this->Input("pre_mix").ParamType(OPTIONAL).DataType({ge::DT_FLOAT})...;

this->Output("y").DataType({ge::DT_BF16})...;
this->Output("post").DataType({ge::DT_FLOAT})...;
this->Output("comb_frag").DataType({ge::DT_FLOAT})...;
this->Output("pre").ParamType(OPTIONAL).DataType({ge::DT_FLOAT})...;
```

注意：`DataType({...})` 列表里**只有一个元素**，说明这个算子在原型层面只支持一种 dtype 组合。`hc_post` 就不一样了，它列了 9 组（见第二篇）。

**(2) 属性**（[L70-L73](../../../../csrc/moe/hc_pre/op_host/hc_pre_def.cpp#L70-L73)）

```cpp
this->Attr("hc_mult").AttrType(OPTIONAL).Int(4);
this->Attr("hc_sinkhorn_iters").AttrType(OPTIONAL).Int(20);
this->Attr("hc_eps").AttrType(OPTIONAL).Float(1e-6f);
this->Attr("norm_eps").AttrType(OPTIONAL).Float(1e-6f);
```

括号里的数字是默认值。`OPTIONAL` 表示上层不传也能跑。

**(3) SoC 注册**（[L75-L83](../../../../csrc/moe/hc_pre/op_host/hc_pre_def.cpp#L75-L83)）

```cpp
this->AICore().AddConfig("ascend910b");
this->AICore().AddConfig("ascend910_93");

OpAICoreConfig regbaseCfg;
regbaseCfg.DynamicCompileStaticFlag(true)
    .DynamicRankSupportFlag(true)
    .DynamicShapeSupportFlag(true)
    .ExtendCfgInfo("opFile.value", "hc_pre");
this->AICore().AddConfig("ascend950", regbaseCfg);
```

- A2/A3 用默认配置。
- A5 额外传了 `regbaseCfg`：开启动静态 shape 支持，并用 `opFile.value` 把算子二进制文件名钉成 `hc_pre`（regbase 编译产物命名规则不同，得显式指定）。

### 4.2 为什么 `hc_pre_proto.cpp` 是空文件

`proto` 文件负责 **InferShape / InferDtype**，也就是「图模式下推导输出形状」。

`hc_pre` 没写，因为它在 vLLM 里是**单算子直调（aclnn）模式**：输出 tensor 由 Python 侧 C++ binding 直接 `at::empty(...)` 建好，形状在 [torch_binding.cpp](../../../../csrc/torch_binding.cpp#L1543-L1569) 里算：

```cpp
y_size        = {bs, d};
post_size     = {bs, hc_mult};
comb_frag_size= {bs, hc_mult, hc_mult};
```

不需要图编译器再推导一次。**这是一个很重要的观察**：不是所有算子都必须有 proto；只有走图模式（GE）时才必需。对比 `hc_post` 就有完整的 [hc_post_proto.cpp](../../../../csrc/moe/hc_post/op_host/hc_post_proto.cpp)，因为它要支持 `y.shape = residual.shape` 的推导。

### 4.3 Tiling 主线：五个步骤

入口在 [hc_pre_tiling.cpp L725-L743](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L725-L743)，先按 SoC 分流，然后 `DoOpTiling()`：

```text
DoOpTiling()                                   # L657
 ├── GetPlatformInfo()                         # L68   拿 AIV/AIC 核数、UB 大小、SoC 版本
 ├── GetShapeAttrsInfoInner()                  # L108  解析 shape + attr + 可选 IO 是否存在
 ├── CalcOpTiling()                            # L611  算 cube 切分 + vector 切分
 │    └── CalcMKSplitCoreMembasePart2Tiling()  # L207  算 Part2 的行/列切分（含 premix 分支）
 │         ├── CalcVectorCoreNum()             # L173  MPMD 下怎么把核分给 pre/post 和 sinkhorn
 │         └── CalcSinkhornTiling()            # L550  Sinkhorn 部分单独的行切分
 ├── GetWorkspaceSize()                        # L682  算 workspace 字节数
 └── PostTiling()                              # L702  SetTilingKey + SetBlockDim + 存 tiling
```

### 4.4 关键 tiling 参数

这些参数名在 Kernel 侧会原样出现，必须对上：

| 参数 | 含义 | 来源 |
| --- | --- | --- |
| `k` | `hc_mult * d`，即拼接后的长度（7168×4 = 28672） | `CalcOpTiling` L612 |
| `cubeBlockDimM` / `cubeBlockDimK` | Cube 在 M/K 两个方向分了几块 | L615-L637 |
| `multCoreSplitMSize` | 每个 Cube 核负责多少行 | L632 |
| `multCoreSplitKSize` | 每个 Cube 核负责多长 K | L634 |
| `mL1Size` / `kL1Size` | L1 中 M/K 分块大小 | L639-L642 |
| `cvLoopKSize` | AIC↔AIV 同步时 K 轴一次走多长（固定 1024） | L644 |
| `stage1MFactor` | Part1 里 Vector 一次处理多少行 | L650-L652 |
| `rowFactor` / `rowOfFormerBlock` / `rowOfTailBlock` | Part2 每核每次几行、前 N 核几行、尾核几行 | L207-L414 |
| `dFactor` / `dLoop` / `tailDFactor` | D 维切块大小、循环次数、尾块大小 | L274-L303 |
| `prePostRowFactor` / `sinkhornRowFactor` | pre/post 与 sinkhorn 各自的行批大小 | L340-L370 / L577-L588 |

表中 `cubeBlockDimM` / `cubeBlockDimK` / `multCoreSplitMSize` / `multCoreSplitKSize` 这四个字段就是 Cube 切核的全部信息，也是 `hc_pre_m_k_split_core.h` 这个文件名的由来。下面 4.4.1~4.4.5 专门讲它。

#### 4.4.1 先分清：这里的 m / k 不是 mHC 的 m

文件名 `hc_pre_m_k_split_core.h` 里的 `m`、`k` 是**矩阵乘的 M 维和 K 维**，和 mHC 的 "m"（manifold）毫无关系。hc_pre 的核心计算是：

$$
\underbrace{x}_{[M,\,K]}\ \times\ \underbrace{hc\_fn_{[N,\,K]}^{T}}_{[K,\,N]}\ \rightarrow\ \underbrace{m}_{[M,\,24]}
$$

| 符号 | 含义 | 数值 | 代码落点 |
| --- | --- | --- | --- |
| **M** | 行数，即 token 数 `bs = batch × seq` | 随 batch 变（prefill 几千，decode 几十） | `bs_` → `multCoreSplitMSize` |
| **K** | 收缩维，4 份 hidden 拼起来的长度 `hc_mult × d` | `4 × 7168 = 28672` | `set_k(kSize)` [L612-L613](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L612-L613) |
| **N** | 输出列数，即 24（pre 4 + post 4 + comb 16） | 固定 24 | `N_SIZE`，**不切** |

所以 "m k split" = **把这次矩阵乘的 M 维和 K 维分别沿 AI Cube 核数切开**，把 AIC 排成一个 `mDimNum × kDimNum` 的二维网格。类名 [HcPreMembaseKSplitCorePart1](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L26) 也印证了这一点：`Membase`（member base）= 以 M（token）为基本分配单位，一个核负责一段连续的 M 行；`KSplit` = 在这段 M 行上把 K 维再横向分给多个核；`Part1` = 第一阶段（算 `mix[24]` 与 `x` 的平方和）。

#### 4.4.2 Host 侧怎么定这个网格

[CalcOpTiling L611-L654](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L611-L654)。先定 M 方向分几核，再把剩下的核全给 K：

```cpp
uint64_t mDimNum = min(aicCoreNum_, CeilDiv(bs_, M_L1_MAX_SIZE));  // M 一块最多 256 行
uint64_t kDimNum = aicCoreNum_ / mDimNum;
if (kDimNum == 1) {
    mDimNum = aicCoreNum_;          // K 用不上剩余的核，就把 M 铺满所有核
} else if (aicCoreNum_ % mDimNum != 0) {
    // ★ 把 mDimNum 上调到 aicCoreNum_ 的因子，保证 mDimNum * kDimNum == aicCoreNum_
}

uint64_t singleCoreM = RoundUp(CeilDiv(bs_, mDimNum), BLOCK_CUBE);   // M 块大小，对齐 16 行
uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), 256);         // K 块大小，对齐 256
tilingData_.set_cubeBlockDimM(mDimNum);
tilingData_.set_cubeBlockDimK(CeilDiv(kSize, splitKSize));
tilingData_.set_multCoreSplitMSize(singleCoreM);
tilingData_.set_multCoreSplitKSize(splitKSize);
```

第二段那个「对齐到因子」是**反直觉的修正**，注释里写得很清楚：不这么做，`bs = 2024` 会算出 `mDimNum=8, kDimNum=2`，20 个 AIC 里只有 16 个在干活（整数除法静默截断，尾巴核空转）。

#### 4.4.3 Kernel 侧：核号 → (m 块, k 块)

每个核从自己的 `blockIdx` 反推网格坐标，见 [GetMkBlockInfo L74-L96](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L74-L96)：

```cpp
uint64_t mBlkDimIdx = curBlockIdx / tilingData->cubeBlockDimK;   // 第几行块
uint64_t kBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimK;   // 第几列块
kGmStartOffset = tilingData->multCoreSplitKSize * kBlkDimIdx;    // 我负责的 K 区间 [start, end)
mGmBaseOffset  = tilingData->multCoreSplitMSize * mBlkDimIdx;    // 我负责的 M 区间 [base, end)
```

注意 `kBlkDimIdx` 用取模得到，所以 **K 是快变维**——同一行块里的核在编号上连续：

```text
             kBlkDimIdx = 0        1        ...   kDimNum-1
mBlkDimIdx
    0        core 0              core 1          core k-1      ┐ 每个格子独立算一块
    1        core k              core k+1        core 2k-1     │ [M块, K块] 的子乘
   ...                                                        │
   m-1                                                        ┘
```

每个格子里做的事就是 `[M块, K块] × [K块, 24]`，在 [MmadA2 L489-L509](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L489-L509) 里落成 `MmadParams{m=16, n=16, k=kL0Size}` 的分形循环，K 方向逐步累加。

#### 4.4.4 算一遍：`bs=2024`、20 个 AIC

| 步骤 | 计算 | 结果 |
| --- | --- | --- |
| M 方向核数 | `min(20, ceil(2024/256)) = min(20,8)` → 上调到 20 的因子 | `mDimNum = 10` |
| K 方向核数 | `20 / 10` | `kDimNum = 2` |
| 每核 M 行数 | `RoundUp(ceil(2024/10)=203, 16)` | `208` 行（尾块 2024-9×208 = 152 行） |
| 每核 K 长度 | `RoundUp(ceil(28672/2)=14336, 256)` | `14336` |
| 每核实际做 | `[208, 14336] × [14336, 24]` | 20 个核刚好铺满 |

#### 4.4.5 为什么这两维都要切、代价是什么

| | 不切的后果 |
| --- | --- |
| **M 必须切** | `bs` 可能几千，一个核算完严重串行 |
| **K 要切** | `K = 28672` 是主要计算量；M 方向摊不开时（如 `bs` 小但对核数多）只能靠切 K 让所有核有活干 |
| **N 不能切** | 只有 24 列，塞不满 16×16 分形，切了纯浪费——这正是这里手写 `Mmad` 而不用 `AscendC::Matmul` 的原因 |

切 K 的代价是**要跨核累加**：每个 `kBlkDimIdx` 只能算出部分和，不能直接写最终结果。所以 workspace 里开的是 `cubeBlockDimK × bs × 24` 份浮点，各核先写自己那一份（`mmOutFp32WsGm[kBlkDimIdx_ * mmOutFp32BufSize_ + ...]`，[L146](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L146)），再由第二阶段把 `kDimNum` 份加起来。这也是 [GetWorkspaceSize L688-L689](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L688-L689) 里 `workspaceSize2` 要乘一个 `cubeBlockDimK` 系数的原因。

正因如此，Host 只在 `bs` 够大时才会真的切 K；`bs` 小时会退化成 `kDimNum = 1`。arch35 把这套思想抽成了一个布尔开关 `isKSplit`（[hc_pre_tiling_arch35.h L143-L180](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L143-L180)），非 KSplit 时直接按 M 分核，Kernel 也随之换成 `HcPreMSplitCoreArch35` 那套（见 4.6 的 TilingKey 表）。

### 4.5 通用版：三个 TilingKey 怎么选

常量定义在 [L62-L65](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L62-L65)：

```cpp
constexpr int64_t SMALL_BS_THRESHOLD = 32;
constexpr uint64_t MPMD_TILING_KEY     = 0;   // 大 batch，MPMD 调度
constexpr uint64_t ORIGINAL_TILING_KEY = 1;   // 小 batch，简单调度
constexpr uint64_t PREMIX_TILING_KEY   = 2;   // 带 pre_mix 输入
```

决策树（[L207-L225](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L207-L225)）：

```text
有没有 pre_mix？
 ├── 有  → key = 2（premix 专用流程）           L210-L212 / L527
 └── 没有
      ├── bs <= 32 → key = 1（ORIGINAL）
      └── bs >  32 → key = 0（MPMD，需要 >= 2 个 AIV 核）
```

**MPMD 是什么？** MPMD = Multiple Program Multiple Data，允许**不同核跑不同代码**。`hc_pre` 用它把核分成两拨：

- 一部分核做 `pre / y / post`（重活，和 D 维有关）
- 一部分核做 `comb / Sinkhorn`（只有 4×4，迭代 20 次，计算密度完全不同）

怎么分最均衡？`CalcVectorCoreNum()`（[L173-L205](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L173-L205)）直接暴力枚举：

```cpp
for (candidatePrePostCoreNum = 1; candidatePrePostCoreNum < totalCoreNum; ++candidate) {
    sinkhornCoreNum = totalCoreNum - candidate;
    prePostLatency  = ceil(bs / candidate)     * (3 * d);
    sinkhornLatency = ceil(bs / sinkhornCoreNum) * (64 * 4 * (iters + 4));
    latency = max(...); imbalance = |...|;
    // 选 latency 最小、平手时选 imbalance 最小的
}
```

这里的 `3 * d` 和 `64 * 4 * (iters + 4)` 是**经验代价模型**，来自常量 `PRE_POST_ROW_WORK_SCALE = 3`、`SINKHORN_ROW_WORK_SCALE = 64`（[L59-L61](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L59-L61)）。不是精确测量，只是为了让核分配不偏得太离谱。

MPMD 还需要 Host 侧显式开调度模式（[L706-L711](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L706-L711)）：

```cpp
if (tilingKey_ == MPMD_TILING_KEY) {
    context_->SetScheduleMode(1);
}
```

### 4.6 arch35：四个 TilingKey

arch35 的 key 不是硬编码 if-else，而是**算出来的**（[hc_pre_tiling_arch35.h L755](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L755)）：

```cpp
tilingKey_ = (hasPreMix ? PREMIX_MK_KEY : NO_PREMIX_MK_KEY) + (isKSplit ? 0 : 1);
// PREMIX_MK_KEY = 1002, NO_PREMIX_MK_KEY = 1000
```

展开就是：

| hasPreMix | isKSplit | key | 策略 | Kernel 类 |
| --- | --- | --- | --- | --- |
| ✗ | ✓ | **1000** | M×K 二维切分 | `HcPreMKSplitCorePart1/Part2` |
| ✗ | ✗ | **1001** | 只切 M | `HcPreMSplitCoreArch35` |
| ✓ | ✓ | **1002** | M×K 切分 + premix | `HcPreMKSplitCorePremixPart1/Part2` |
| ✓ | ✗ | **1003** | 只切 M + premix | `HcPreMSplitCorePremixArch35` |

`isKSplit` 的判据是 `kDimNum != 1`（[L747](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L747)），即「K 轴大到必须多个 Cube 核协作」。

另外 arch35 在 `!isKSplit && hasPreMix` 时会把 `mDimNum` 直接拉满到 `aicCoreNum_`，并重新选一个更合适的 `mL1Size`（`SelectPremixML1Size`），因为 premix 省掉了 Part1，UB 预算更宽松（[L748-L754](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L748-L754)）。

### 4.7 Workspace：GM 上的「中转站」

`hc_pre` 被拆成 Part1 和 Part2 两个 kernel 阶段，中间结果必须落到 GM，这就是 workspace。

**通用版**（[L682-L700](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L682-L700)）：

```text
workspace = ws1 + ws2 + ws3 + 16MB
  ws1 = cubeCoreNum × 2 × (mL1Size × align(cvLoopKSize,128)) × 4B   ← x 转 fp32 的缓冲
  ws2 = align(cubeBlockDimK × bs × align(hcMix,128) × 4B, 512)      ← mixes 全量结果
  ws3 = align(cubeBlockDimK × align(bs,16) × 16 × 4B, 512)          ← 平方和部分和
然后取 max(当前需求, 208MB)                                            ← 兜底
```

最后一行 `defaultSize = 16MB + 192MB = 208MB` 是保守兜底，保证各种边界 shape 不会越界。

**arch35** 只有 K 切分时才需要额外 workspace（[L793-L800](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L793-L800)）：

```cpp
workspaceSize_ = kBlockFactor * bs * hcMix * 4 + kBlockFactor * bs * 4 + 16MB;
```

### 4.8 通用版 vs arch35 对比

| 维度 | 通用版（A2/A3） | arch35（A5/950） |
| --- | --- | --- |
| 入口 | `HcPreTiling::DoOpTiling` | `HcPreTilingRegbase::DoOpTiling` |
| TilingKey | 0 / 1 / 2（硬编码 if-else） | 1000 / 1001 / 1002 / 1003（公式推导） |
| 调度模式 | MPMD + `SetScheduleMode(1)` | 普通调度，Part1/Part2 顺序执行 |
| 核分拨 | pre/post 与 sinkhorn 分不同核 | 不分拨，K 切分的多个核各自算部分和再 Reduce |
| workspace | ws1 + ws2 + ws3 + 16MB，兜底 208MB | 仅 K 切分时按 kBlockFactor 计算 + 16MB |
| Kernel 写法 | `LocalTensor` / `Brcb` / `PipeBarrier` | `RegTensor` / `MaskReg` / `__VEC_SCOPE__` |

**小白总结**：Host 侧不做任何数学，只做「分块决策」。它的产出是一张 Excel 表（TilingData），Device 侧照着表干活。TilingKey 是这张表的「工作表编号」。

---

## 5. Kernel 侧：照着表干活

### 5.1 入口 `hc_pre.cpp`

```cpp
extern "C" __global__ __aicore__ void hc_pre(GM_ADDR x, GM_ADDR hc_fn, ..., GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);   // 1 个 Cube 配 2 个 Vector
    if (workspace == nullptr) return;
    GM_ADDR userWs = GetUserWorkspace(workspace);         // 跳过系统预留区
    TPipe pipe;
#if defined(__DAV_C310__)
    if (TILING_KEY_IS(1000)) { ... Part1; pipe.Destroy(); Part2 ... }
    else if (TILING_KEY_IS(1002)) { ... }
    else if (TILING_KEY_IS(1001)) { ... }
    else if (TILING_KEY_IS(1003)) { ... }
#else
    if (TILING_KEY_IS(0)) { ... }
    else if (TILING_KEY_IS(1)) { ... }
    else if (TILING_KEY_IS(2)) { ... }
#endif
}
```

三个新手必知细节：

1. `extern "C"`：算子符号不能 C++ name mangling，框架按名字找入口。
2. `__global__ __aicore__`：`__global__` 是 kernel 函数，`__aicore__` 表示跑在 AI Core 上。
3. **`pipe.Destroy()` 出现在两个阶段之间**，然后新建 `TPipe pipeStage2`。这是因为 Part1 和 Part2 的 UB 需求完全不同，复用同一个 TPipe 会让分配器保守预留，直接重建更省 UB。

`GetUserWorkspace(workspace)` 也很关键：框架给的是「系统 workspace + 用户 workspace」的拼接地址，必须跳过前面一段才能给算子自己用。

### 5.2 通用版 Part1：InvRMS + 矩阵乘

Part1 只做一件事：**算出 24 个 mixes**（以及顺带把平方和存进 workspace）。

```text
HcPreMembaseKSplitCorePart1<T>::Process()      hc_pre_m_k_split_core.h#L189
 ├── AIV: x(bf16) → Cast → fp32 → 写 workspace1   ← 给 Cube 吃
 ├── AIV: x² 逐元素 → 归约 → 16 路部分和 → workspace3
 ├── AIC: DataCopy x_fp32 → L1，ND2NZ 重排
 ├── AIC: hc_fn → L1，ND2NZ 重排
 ├── AIC: 逐 K 块 LoadAToL0A / LoadBToL0B / MmadAB
 ├── AIC: L0C → fixpipe → workspace2（mixes 结果）
 └── CrossCoreSetFlag / CrossCoreWaitFlag 做 AIC↔AIV 握手
```

**这里最大的反直觉点：没用 `AscendC::Matmul` 高层 API，而是手写 `Mmad`。**

看 [hc_pre_cube_compute.h](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h) 的函数名就知道：

- `CopyInA1` / `CopyInB1`：GM → L1，带 `nd2nz` 重排
- `LoadAToL0A` / `LoadBToL0B`：L1 → L0A/L0B
- `MmadA2` / `MmadAB`：真正的矩阵乘
- `Fixp`：L0C → GM 的 fixpipe

为什么不用高层 API？因为这个矩阵乘形状极其特殊：

- M = `bs`（可能只有 1，也可能是 8192）
- N = 24（**极小**）
- K = 28672（**极长**）

N=24 连一个 16×16 分形都填不满，高层 API 的分形调度会浪费大量 Cube 算力。手写可以精细控制 L1 里 B 矩阵的驻留和 K 方向流水（`PreloadNextB1` 就是提前把下一块 B 搬进来）。

**ND2NZ 是什么？** Cube 只认 `NZ` 格式（分形布局：16×16 小块内部转置排列）。GM 里的张量是 `ND`（行主序），所以搬进 L1 时必须用 `nd2nz` 重排。参数 `srcDValue`、`dstNzC0Stride` 就是在描述这个转换（[L191-L244](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L191-L244)）。

### 5.3 通用版 Part2：pre / y / post / comb

Part2 是主战场，入口 `ProcessPrePost`（[L747-L770](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L747-L770)），外层按行循环：

```cpp
for (rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; ++rowOuterIdx) {
    curRowFactor     = 最后一批 ? tailRowFactor : prePostRowFactor;
    globalRowOffset  = blockIdx * prePostRowOfFormerBlock + rowOuterIdx * prePostRowFactor;
    ProcessPrePostBatch(...);
}
```

`ProcessPrePostBatch`（[L674-L745](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L674-L745)）的执行顺序非常值得记：

```text
1. CalcRsqrt()                 ← 从 workspace3 取平方和，算 r = 1/sqrt(mean+eps)
2. 从 workspace2 取 mixes[0:4] 和 mixes[4:8]
3. ReduceSumARAPerf()          ← 多个 K 核的部分 mixes 相加
4. ProcessPre()                ← pre = sigmoid(mix*r*scale0+base0)+eps
5. for rowIdx in rows:         ← 逐行做 y = Σ pre_j * x_j
     for dLoopIdx in dLoops:
        CopyIn x → ProcessY → CopyOut y
6. ProcessPost()               ← post = 2*sigmoid(mix*r*scale1+base1)
7. 逐行 CopyOut post
```

**几个巧妙设计**：

- **pre/post 的批量和 y 的批量分开**：`prePostRowFactor` 最大 4，而 `rowFactor` 可能只有 1。代码注释说得很清楚——pre/post 只需要「行大小」的状态，而 y 需要「整行 D」的缓冲，混在一起会让 sigmoid 被迫按行串行（[L337-L370](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L337-L370)）。
- **pre 行批做完，再逐行做 y**：因为 `y` 的 UB 占用和 `d` 成正比，先把所有行的 `pre` 都算完（每行只要 4 个数），再一行一行搬 `x`。寄存器的复用效率最高。
- **post 放到行循环外批量算**：注释「在行循环外批量处理 post，减少 sigmoid/mul 链的逐行同步开销」（[L729-L734](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L729-L734)）。

### 5.4 四个核心计算函数

这四个函数就是第 1.4 节公式的逐行翻译：

**(a) `ProcessPre`**（[hc_pre_base.h L489-L502](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L489-L502)）

```cpp
MulABLastDimBrcInline<float,true>(mixLocal, mixLocal, rsqrtLocal, ...); // mix *= r
Muls(mixLocal, mixLocal, scale, ...);                                   // mix *= scale
AddBAFirstDimBrcInline<float>(mixLocal, mixLocal, hcBaseLocal, ...);    // mix += base
SigmoidPerf(preLocal, mixLocal, ...);                                   // pre = sigmoid(mix)
Adds(preLocal, preLocal, eps, ...);                                     // pre += eps
```

`sigmoid` 没有硬件指令，`SigmoidPerf`（[L451](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L451)）用 `1/(1+exp(-x))` 实现：先 `Muls(-1)` 取负，`Exp`，`Adds(1)`，最后 `Div`。

**(b) `ProcessYHc4`**（[L626-L694](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L626-L694)）—— 专门为 `hc_mult = 4` 手写的融合版本：

```cpp
CastTwoDim(xCastLocal, xLocal, rows, dim2, false);  // bf16 4 份全转 fp32
Brcb(hcBrcbLocal1, mix01Local, ...);                // 4 个 pre 值各广播成 8 份

// Pass 0: y = x0 * w0
Mul(yCastLocal[...], xCastLocal[...], hcBrcbLocal1[...], ...);
// Pass 1..3: y += xk * wk
MulAddDst(yCastLocal[...], xCastLocal[...], hcBrcbLocal1[...], ...);
```

**为什么 `src1BlkStride = 0`？** 让同一个权重 block 在整行内反复读，不用为每个元素准备一份权重。这就是「广播」在硬件层面的实现。

**注意：`y` 用的是原始 `x`，不是归一化后的 `x`。** 归一化只进入了门控分支（`mixes`），值通路保持原始激活。这是一个容易看错的地方。

**(c) `ProcessPost`**（[L727-L740](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L727-L740)）

和 `ProcessPre` 结构完全一样，只有两处不同：
- `needBrc = false`：`rsqrtLocal` 已经在 `ProcessPre` 阶段广播好了，不重复 `Brcb`
- 结尾是 `Muls(post, post, 2.0f)`，不是 `Adds(eps)`

**(d) `SoftmaxFP32Perf` + Sinkhorn**（[L773-L787](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L773-L787)）

```text
1. LastDimReduceMaxPerf   → 每行最大值
2. SubABLastDimBrcInline  → 减去最大值（数值稳定，防 exp 溢出）
3. Exp
4. LastDimReduceSumPerf   → 每行和
5. DivABLastDimBrcInline  → 归一化
6. Adds(eps)              → 防止后续 Sinkhorn 除零
```

### 5.5 Sinkhorn 到底在迭代什么

这是全篇最「数学」的一段，但代码其实很短（[ProcessCompactSinkhorn L627-L660](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L627-L660)）：

```cpp
Brcb(hcBrcbLocal1, hcBase2Local, ...);                 // base[8:24] 广播
Mul(data, data, rsqrtLocal, ...);                      // data *= r
Muls(data, data, hcScale, ...);                        // data *= scale2
Add(data, data, hcBrcbLocal1, ...);                    // data += base2
SoftmaxCompactRows(data, curRowFactor);                // 行 softmax
NormalizeCompactCols(data, curRowFactor);              // 列归一化（第 1 次）
for (iter = 0; iter < iterTimes - 1; ++iter) {         // 再迭代 19 次
    NormalizeCompactRows(data, curRowFactor);
    NormalizeCompactCols(data, curRowFactor);
}
```

**Sinkhorn 的直觉**：想让一个 4×4 矩阵同时「每行和为 1」且「每列和为 1」（叫双随机矩阵 / Birkhoff 多面体）。但一般做不到一步到位，于是**交替归一化**：

```text
行归一化 → 列和变乱 → 列归一化 → 行和又变乱 → 行归一化 → ...
```

迭代足够多次后会收敛。`hc_sinkhorn_iters = 20` 就是迭代次数上限。

**为什么先 softmax 再列归一化？** softmax 保证所有元素非负且行和为 1，这是 Sinkhorn 收敛的前提（负元素会让交替归一化震荡）。

**为什么每步都加 `eps`？** 如果某个元素被归一到 0，下一轮除法就会出现 `0/0`。`eps = 1e-6` 是安全垫。

**为什么用 `COMPACT_MATRIX_SIZE` 和 `GatherMask`？** 因为数据在 UB 里是「行优先、行距对齐到 8」的布局（`hcMultAlign = 8`），4×4 矩阵实际占 4×8。`GatherCompactToElementMajor` / `GatherElementMajorToCompact`（[L447-L470](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L447-L470)）负责在「紧凑 4×4」和「对齐 4×8」之间搬来搬去，让向量指令能一次处理多行。

### 5.6 arch35 的四套 kernel

arch35 把「K 是否切分」和「有没有 premix」做成了 2×2 的模板组合：

| 文件 | 类 | 场景 |
| --- | --- | --- |
| `hc_pre_m_k_split_core_arch35.h` | `HcPreMKSplitCorePart1/Part2` | K 切分，无 premix（key 1000） |
| `hc_pre_m_split_core_arch35.h` | `HcPreMSplitCoreArch35` | 只切 M，无 premix（key 1001） |
| `hc_pre_m_k_split_core_premix_arch35.h` | `HcPreMKSplitCorePremixPart1/Part2` | K 切分 + premix（key 1002） |
| `hc_pre_m_split_core_premix_arch35.h` | `HcPreMSplitCorePremixArch35` | 只切 M + premix（key 1003） |

和通用版最大的差异：

1. **Part1/Part2 都在同一个核上顺序跑**（key 1000/1002 里显式 `pipe.Destroy()` 再建 `pipeStage2`），不再依赖 MPMD 分核。
2. **Vector 计算全部换成 MicroAPI**：`VFProcessPre`、`VFProcessY`、`VFProcessPremixYCast`（[hc_pre_base_arch35.h](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L678)、[L1252](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1252)、[L387](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L387)）。
3. **一次 load 自带广播**：`DIST_BRC_B16/B32` 省掉了 `Brcb` + `PipeBarrier`。
4. **`premix` 场景直接读 `pre_mix` 跳过 Part1 的矩阵乘**，只做 `VFProcessPremixYFromCastSegments` 的加权求和。

### 5.7 通用版 vs arch35 内核对比

| 维度 | 通用版 | arch35 |
| --- | --- | --- |
| Vector 数据类型 | `LocalTensor<T>` | `RegTensor<T>` |
| 广播方式 | `Brcb` + `src1BlkStride=0` | `LoadDist::DIST_BRC_*` |
| 同步 | 大量显式 `PipeBarrier<PIPE_V>` | `__VEC_SCOPE__` 内由编译器插入 |
| 尾块处理 | `RoundUp` + 逐行循环 | `UpdateMask<float>(n)` 一条指令 |
| 跨核同步 | `CrossCoreSetFlag/WaitFlag` | 主要靠 workspace + 顺序执行 |
| 代码量 | `hc_pre_base.h` 929 行 | `hc_pre_base_arch35.h` 1418 行（多了 layout 处理） |

**小白总结**：经典写法像「手动挡」，每一步搬、算、同步都自己写；MicroAPI 像「自动挡」，一条指令解决 load+广播，掩码处理尾巴。两者数学完全等价，只是编译器后端不同。

---

## 6. 一次调用的完整旅程

以 Python 侧 `hc_pre(hidden_states, hc_attn_fn, hc_attn_scale, hc_attn_base)` 为例：

```text
① Python（V4.1）
   vllm_ascend/models/deepseek_v41/model.py#L858-L869
   torch.ops._C_ascend.npu_hc_pre_v3(x, fn, scale, base, pre_mix, hc_mult, iters, norm_eps, hc_eps)
        │
        ▼
② PyTorch 自定义算子注册
   csrc/torch_binding.cpp#L3552  ops.impl("npu_hc_pre_v3", ...)
        │
        ▼
③ C++ binding
   csrc/torch_binding.cpp#L1656  npu_hc_pre_v3_npu()
     ├── check_hc_pre_shape_and_dtype()          # L1582 形状/dtype 校验
     ├── construct_hc_pre_output_tensor()        # L1543 建 y/post/comb_frag
     └── EXEC_NPU_CMD(aclnnHcPre, ...)           # L1639 调 aclnn 单算子
        │
        ▼
④ aclnn 框架
   - 调用 InferShape（本算子无 proto，跳过）
   - 调用 Tiling4HcPre                 ← Host 侧，CPU
        │
        ▼
⑤ Host Tiling（CPU）
   hc_pre_tiling.cpp#L725
     ├── SoC == 950 ? HcPreTilingRegbase : HcPreTiling
     ├── CalcOpTiling → 算 mDim/kDim/mL1/kL1/rowFactor/dFactor...
     ├── GetWorkspaceSize → 208MB 兜底
     └── PostTiling → SetTilingKey + SetBlockDim + 存 TilingData
        │
        ▼
⑥ Device Kernel 启动（NPU），blockDim 个 AI Core
   hc_pre.cpp#L34  hc_pre(...)
     ├── Part1：x→fp32、算平方和、Cube 算 mixes → workspace
     ├── pipe.Destroy()
     └── Part2：读 workspace → rsqrt → pre/y/post/comb+Sinkhorn → 输出
        │
        ▼
⑦ 结果回到 Python
   y (bf16, [bs,d]) / post (fp32, [bs,4]) / comb_frag (fp32, [bs,4,4]) / pre (fp32, [bs,4])
```

**文件汇总表**：

| 层 | 文件 | 职责 |
| --- | --- | --- |
| Python | [model.py](../model.py#L858-L869) | 调 `torch.ops._C_ascend.npu_hc_pre_v3`（V4.1；V4 为 `npu_hc_pre_v2`） |
| Binding | [torch_binding.cpp](../../../../csrc/torch_binding.cpp#L1527-L1654) | 校验、建输出、`EXEC_NPU_CMD` |
| 原型 | [hc_pre_def.cpp](../../../../csrc/moe/hc_pre/op_host/hc_pre_def.cpp) | IO/attr/SoC 注册 |
| Host | [hc_pre_tiling.cpp](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp) / [hc_pre_tiling_arch35.h](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h) | 切分决策 |
| Device | [hc_pre.cpp](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp) | 入口 + key 分派 |
| Device | [hc_pre_base.h](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h) / [hc_pre_base_arch35.h](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h) | 数学计算原语 |
| Device | [hc_pre_m_k_split_core.h](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h) | 主流程 |
| Device | [hc_pre_cube_compute.h](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h) | 手写 Mmad |

---

## 7. 性能设计清单

| 优化 | 做法 | 位置 |
| --- | --- | --- |
| **算子融合** | 一次调用完成 RMSNorm + 矩阵乘 + sigmoid + Sinkhorn + 加权和，省 5 次 GM 往返 | 整体设计 |
| **两级切分** | Cube 按 M×K 二维切（`mDimNum × kDimNum = aicCoreNum`），Vector 按行切 | `CalcOpTiling` |
| **MPMD 分核** | pre/post 与 Sinkhorn 用不同核并行，按代价模型选最均衡比例 | `CalcVectorCoreNum` |
| **双缓冲** | `InitBuffer(que, 2, size)`，搬下一块时算当前块 | 各 `DataCopyIn*` |
| **行批量分离** | pre/post 用 `prePostRowFactor`（≤4），y 用 `rowFactor`，互不拖累 | `CalcMKSplitCoreMembasePart2Tiling` |
| **省一次归一化** | 不预先归一化 `x`，而是在 Part2 把 `r` 乘到 `mixes` 上（数学等价，省一遍全量乘） | `CalcRsqrt` + `ProcessPre` |
| **融合乘加** | `MulAddDst` 一条指令完成 `y += x*w`，4 路只需 1 次 Mul + 3 次 MulAddDst | `ProcessYHc4` |
| **广播复用** | `src1BlkStride = 0` 让权重 block 在整行复用 | `ProcessYHc4` |
| **屏障合并** | 多个 dst 不重叠的指令并行发射，只留一个 `PipeBarrier` | `CastTwoDim` 注释 |
| **Sinkhorn 专用归约** | 4×4 用 `ReduceSum4x4Perf`，省掉首行 DataCopy 和每加数 barrier | `hc_pre_base.h#L556` |
| **TPipe 重建** | Part1/Part2 之间 `pipe.Destroy()`，避免 UB 按两阶段峰值分配 | `hc_pre.cpp` |
| **premix 短路** | 有 `pre_mix` 时跳过 Part1 矩阵乘 | `PREMIX_TILING_KEY` |

---

## 8. FAQ：小白最容易卡住的 10 个问题

**Q1：`TilingKey` 和 `blockDim` 有什么区别？**
`TilingKey` 选**代码路径**（if-else 分支），`blockDim` 选**用几个核**。前者是「走哪条路」，后者是「派多少人」。

**Q2：为什么 Part1 和 Part2 要拆开？**
因为中间结果（`mixes`、平方和）要跨核交换。Part1 的 Cube 核和 Part2 的 Vector 核不是同一批，只能通过 GM（workspace）传话。

**Q3：`CrossCoreSetFlag` 和 `SetFlag` 有什么区别？**
`SetFlag/WaitFlag` 是**同一个核内**不同流水线（MTE2 搬运 / V 计算 / MTE3 回写）之间握手；`CrossCoreSetFlag/CrossCoreWaitFlag` 是**AIC 和 AIV 之间**握手。

**Q4：为什么到处是 `RoundUp(x, 8)` 或 `RoundUp(x, 128)`？**
Vector 指令以 32 字节（8 个 fp32 / 16 个 bf16）为最小粒度，Cube 的 `nd2nz` 以 16×16 分形为最小粒度。不对齐会读越界或算错。

**Q5：`Brcb` 不是有 `repeatTimes` 吗，为什么还要配 `src1BlkStride=0`？**
`Brcb` 负责「把 1 个标量变成 8 个相同值」；`src1BlkStride=0` 负责「这 8 个值在整行内重复使用，不跳到下一个 block」。两者配合才是完整广播。

**Q6：`softmax` 前为什么要减最大值？**
`exp(1000)` 会溢出成 `inf`。减去每行最大值后，最大项变成 `exp(0)=1`，永远不会溢出。这不改变 softmax 结果（分子分母同乘一个常数）。

**Q7：Sinkhorn 迭代 20 次够吗？**
4×4 矩阵收敛很快，20 次远超实际需要。这个值是模型训练时定的（`hc_sinkhorn_iters=20`），推理必须与训练一致，不能用「收敛了」来提前退出——否则数值会和训练对不上。

**Q8：为什么 `pre` 是可选输出？**
V4 里 `pre` 是可选输出：KV-cache 场景下它只和当前 token 有关，可以缓存起来，下一轮通过 `pre_mix` 直接传入，省掉 Part1；默认路径（`npu_hc_pre_v2`）不输出 `pre`，只有 `npu_hc_pre_v3` 才返回四元组。
**V4.1 起 `npu_hc_pre_v3` 成为唯一入口**，`pre` 必有输出，且固定由**下一层的注意力**消费（延迟系数交接，见 §0.1）。

**Q9：`hc_fn` 为什么是 fp32 而 `x` 是 bf16？**
权重对精度敏感（24×28672 个参数，任何量化误差都会放大），激活可以用 bf16 省带宽。矩阵乘时 `x` 会先 Cast 成 fp32。

**Q10：为什么 `hc_pre_proto.cpp` 是空的还能跑？**
因为走的是 aclnn 单算子直调，输出形状在 C++ binding 里已经建好，不需要图编译器推导。只有走 GE 图模式才必须有 proto。

---

## 9. 术语表

| 术语 | 全称 / 含义 |
| --- | --- |
| **mHC** | Manifold-Constrained Hyper-Connections，DeepSeek-V4 的多副本残差连接 |
| **hc_mult** | 副本数，固定 4 |
| **AIC** | AI Cube，矩阵计算单元 |
| **AIV** | AI Vector，向量计算单元 |
| **GM** | Global Memory，片上之外的大容量显存 |
| **UB** | Unified Buffer，Vector 的工作内存（~192KB） |
| **L1 / L0A / L0B / L0C** | Cube 的多级缓存/寄存器阵列 |
| **Tiling** | Host 侧把大张量切成能塞进 UB/L1 的小块 |
| **TilingKey** | 选路编号，等价于 switch-case 的 case |
| **TilingData** | Host → Device 传参的结构体 |
| **Workspace** | GM 上给算子自己用的临时缓冲 |
| **MPMD** | Multiple Program Multiple Data，不同核跑不同代码 |
| **ND2NZ** | 把行主序张量转成 Cube 要求的分形格式 |
| **Mmad** | Cube 的矩阵乘指令 |
| **Brcb** | Broadcast，把标量复制成 8 份铺满 block |
| **Sinkhorn** | 交替行列归一化，把矩阵投影到双随机矩阵集合 |
| **InvRMS** | `1/sqrt(mean(x²)+eps)` |
| **arch35** | ascend950（A5）的 regbase 架构代号 |
| **MicroAPI** | arch35 的寄存器级 Vector 编程接口 |
| **RegTensor / MaskReg** | MicroAPI 的向量寄存器 / 掩码寄存器 |
| **`__DAV_C310__`** | 编译宏，标识 arch35 目标 |

---

---

## 10. 函数级详解：怎么用这一组章节

前 1~10 章讲「整体怎么运转」，从第 11 章起换成「逐个函数看」。

- 每章对应一到两个源文件，条目按**源码出现顺序**排列，方便对照阅读。
- 每条固定四栏：**职责**（干什么）、**入参/出参**（谁进谁出、单位是什么）、**实现**（关键步骤与 API）、**注意**（易错点与反直觉处）。
- 「注意」栏来自逐行核对源码。凡写成「可能 / 疑似 / 与注释不一致」的，都是**代码事实层面的观察**，不代表已确认的线上 bug，请以实际运行为准。
- 行号锚点为当前分支实测值；代码若变动，用函数名检索即可。

| 章 | 文件 | 函数条目数 |
| --- | --- | --- |
| 11 | `hc_pre_def.cpp`、`hc_pre_tiling.h`、`hc_pre_tiling.cpp` | 22 |
| 12 | `hc_pre_tiling_arch35.h` | 35 |
| 13 | `hc_pre_base.h` | 26 |
| 14 | `hc_pre_base_arch35.h` | 42 |
| 15 | `hc_pre.cpp`、`hc_pre_m_k_split_core.h` | 42 |
| 16 | `hc_pre_cube_compute.h`、`hc_pre_cube_compute_arch35.h` | 29 |
| 17 | 四套 arch35 Kernel 主流程 | 20 |
| — | 合计 | 216 |

---

## 11. 函数级详解：Host 侧 tiling（`hc_pre_def.cpp` / `hc_pre_tiling.h` / `hc_pre_tiling.cpp`）

#### hc_pre_def.cpp（算子原型注册，共 2 条）

> 这个文件只做一件事：告诉框架「HcPre 这个算子长什么样」——有几个输入、什么 dtype、什么形状格式、有哪些属性、能在哪些芯片上跑。它不参与任何数值计算。
> `OpDef` 是昇腾算子原型基类，`OP_ADD` 把它注册进全局算子库；`AICore()` 描述的是 AI Core 侧（kernel）的编译配置。

- **`HcPre::HcPre(const char *name)`** [hc_pre_def.cpp L22-L84](../../../../csrc/moe/hc_pre/op_host/hc_pre_def.cpp#L22-L84)
  - 职责：构造函数里声明 HcPre 的 5 个输入、4 个输出、4 个属性与 3 套芯片配置
  - 入参/出参：`name`=算子名字符串（由 `OP_ADD` 传入 `"HcPre"`）；返回=无（构造函数，无返回值）
  - 实现：① 依次 `this->Input(...).ParamType(...).DataType({...}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND})`，顺序为 x(bf16, REQUIRED)、hc_fn(f32)、hc_scale(f32)、hc_base(f32)、pre_mix(f32, OPTIONAL)；② 输出 y(bf16)、post(f32)、comb_frag(f32)、pre(f32, OPTIONAL)；③ 属性 `Attr("hc_mult").Int(4)`、`Attr("hc_sinkhorn_iters").Int(20)`、`Attr("hc_eps").Float(1e-6f)`、`Attr("norm_eps").Float(1e-6f)`，全部 `AttrType(OPTIONAL)`；④ `AICore().AddConfig("ascend910b")`、`AddConfig("ascend910_93")`，950 单独用 `OpAICoreConfig regbaseCfg`（开 DynamicCompileStatic/DynamicRank/DynamicShape，`ExtendCfgInfo("opFile.value","hc_pre")`）后 `AddConfig("ascend950", regbaseCfg)`
  - 注意：声明的顺序就是 tiling/kernel 侧取张量的索引 0..4，`pre_mix` 必须排第 5 个（tiling 里写死 `GetOptionalInputTensor(4)`）；这里的属性默认值必须与 `HcPreTiling::GetAttr()` 的兜底值完全一致，否则「属性没传」时两处语义会分叉；950 之所以单独一份 config，是因为它编译的是 regbase（arch35）kernel。

- **`OP_ADD(HcPre)`** [hc_pre_def.cpp L86](../../../../csrc/moe/hc_pre/op_host/hc_pre_def.cpp#L86)
  - 职责：把上面定义的 `HcPre` 原型注册进算子原型库
  - 入参/出参：`HcPre`=刚定义的类名（宏内部同时把它当作算子名）；返回=无（宏展开为静态注册动作）
  - 实现：① 宏在静态初始化期构造一个 `HcPre("HcPre")` 实例并登记到 `OpDefFactory`；② 运行时框架按算子名 `"HcPre"` 反查原型，据此做 dtype/shape/可选输入的合法性检查；③ 只有原型注册成功后，`IMPL_OP_OPTILING` 里的 tiling 注册才会被关联上
  - 注意：算子名由类名隐式决定，改类名等于改算子名，Python/ACLNN 侧调用会一起失效；本文件只注册「原型」，tiling 回调在 `hc_pre_tiling.cpp` 末尾注册。

#### hc_pre_tiling.h（数据结构与类声明，共 3 条）

> 这个头文件有三块内容：两个描述「输入/输出张量句柄」的小结构体、`HcPreTilingData`（Host 与 Device 之间传递 tiling 参数的结构体，用宏生成字段和 getter/setter）、以及 `HcPreTiling` 类的声明。
> `TilingData` 会被序列化成一块裸内存 buffer 交给 kernel，所以字段顺序 = 内存布局，host/kernel 必须用同一份头文件。

- **`REGISTER_TILING_DATA_CLASS(HcPre, HcPreTilingData)`** [hc_pre_tiling.h L125](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.h#L125)
  - 职责：把 `HcPreTilingData` 与算子 `HcPre` 绑定，供框架反序列化出结构体
  - 入参/出参：`HcPre`=算子名，`HcPreTilingData`=`BEGIN_TILING_DATA_DEF` 生成的结构体；返回=无（注册宏）
  - 实现：① 注册一个工厂，kernel 侧可用 `GetTilingData<HcPreTilingData>()` 拿到指针；② 每个 `TILING_DATA_FIELD_DEF(type, name)` 按声明顺序生成 `set_xxx/get_xxx` 与固定偏移；③ `SaveToBuffer/GetDataSize` 由宏基类提供，`HcPreTiling::PostTiling()` 正是调它们
  - 注意：字段只能「在末尾追加」，中间插入或删字段会整体错位，导致 kernel 读到的 `bs/d` 变成别的值；`double` 不能出现在 TilingData 里（这里 eps 在结构体里是 `float`，host 侧成员却是 `double`）。

- **`HcPreTiling::HcPreTiling(gert::TilingContext* tilingContext)`** [hc_pre_tiling.h L136-L138](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.h#L136-L138)
  - 职责：保存 tiling 上下文指针，构造即可用
  - 入参/出参：`tilingContext`=框架传入的 tiling 上下文（shape、attr、平台信息、tilingdata buffer 全从这里取）；返回=无（构造函数）
  - 实现：① 初始化列表 `: context_(tilingContext)`；② 其余成员靠类内默认值（指针 `nullptr`、数值 `0`、`socVersion_` 默认 `ASCEND910B`）；③ 构造阶段不做任何校验，校验推迟到 `GetShapeAttrsInfoInner()`
  - 注意：`context_` 只是借用，不能 `delete`；`context_` 为 `nullptr` 时构造函数不报错，第一个 `context_->GetPlatformInfo()` 才会崩，所以入口函数必须先判空。

- **`HcPreTiling::~HcPreTiling()`** [hc_pre_tiling.h L139](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.h#L139)
  - 职责：默认析构，不释放任何外部资源
  - 入参/出参：无；返回=无
  - 实现：① `= default`；② 成员 `tilingData_` 是值对象，随对象一起销毁；③ tiling 结果在 `PostTiling()` 已拷进框架 buffer，析构不参与提交
  - 注意：析构「不提交」tiling 是反直觉点——新建了对象但没调 `DoOpTiling()`，framework 侧什么都不会发生。

#### hc_pre_tiling.cpp（Host 侧 tiling 实现，共 17 条）

> 这个文件是算子的「切分大脑」：先拿到芯片参数（核数、UB 容量），再校验 shape/attr，然后把 bs×K 的矩阵切给各核、把每核的数据切成能塞进 UB 的小块，最后把结果写进 `HcPreTilingData` 并告诉框架用多少核、用哪个 tilingKey、要多大 workspace。
> 阅读顺序建议：`DoOpTiling` → `GetPlatformInfo`/`GetShapeAttrsInfoInner` → `CalcOpTiling` → `CalcMKSplitCoreMembasePart2Tiling` → `GetWorkspaceSize`/`PostTiling`。

- **`optiling::{anonymous}::CeilDiv`** [hc_pre_tiling.cpp L24-L30](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L24-L30)
  - 职责：整数向上取整除法，整个文件最基础的算术工具
  - 入参/出参：`x`=被除数（非负数，行数/元素数）、`y`=除数（核数/块大小）；返回=⌈x/y⌉
  - 实现：① `(x + y - 1) / y` 实现向上取整；② `y != 0` 才做除法；③ `y == 0` 时原样返回 `x`（兜底而非报错）
  - 注意：`y == 0` 返回 `x` 是反直觉的「静默兜底」，除数为 0 的 bug 不会在这里暴露，只会在后续尺寸算错时体现；负数入参语义未定义。

- **`optiling::{anonymous}::DownAlign`** [hc_pre_tiling.cpp L31-L36](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L31-L36)
  - 职责：把数值向下对齐到指定粒度的整数倍
  - 入参/出参：`x`=待对齐值、`y`=对齐粒度（如 32）；返回=`(x / y) * y`
  - 实现：① `y == 0` 直接返回 `x`；② 否则整除再乘回，舍掉余数；③ 主要用于把试出来的 `dFactor_` 对齐到 32，避免非对齐搬运
  - 注意：会把值「变小」，所以 `dLoop_ = CeilDiv(d_, dFactor_)` 之后尾块必须单独用 `tailDFactor_` 表示，不能假设每块都是 `dFactor_`。

- **`optiling::{anonymous}::RoundUp`** [hc_pre_tiling.cpp L37-L39](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L37-L39)
  - 职责：向上对齐（与 `DownAlign` 方向相反）
  - 入参/出参：`x`=待对齐值、`y`=对齐粒度；返回=`CeilDiv(x,y)*y`
  - 实现：① 复用 `CeilDiv`；② 用于「分配 buffer 尺寸」和「元素数按 8/16 对齐」两类场景；③ 例如 `hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float))` 即按 8 个 float 对齐
  - 注意：UB buffer 估算必须用 `RoundUp` 而不是 `DownAlign`，否则会低估占用导致越界；对齐粒度单位要和被对齐量一致（元素数 vs 字节数）。

- **`HcPreTiling::GetPlatformInfo()`** [hc_pre_tiling.cpp L68-L87](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L68-L87)
  - 职责：取 AIV 核数、AIC 核数、UB 大小和 SoC 版本
  - 入参/出参：无显式入参；写出 `aivCoreNum_`、`aicCoreNum_`、`ubSize_`、`socVersion_`；返回=`ge::GRAPH_SUCCESS`/`GRAPH_FAILED`
  - 实现：① 先 `context_->GetPlatformInfo()`；② 非空则 `platform_ascendc::PlatformAscendC(platformInfo)` 后调 `GetCoreNumAiv()`、`GetCoreNumAic()`、`GetCoreMemSize(CoreMemType::UB, ubSize_plaf)`、`GetSocVersion()`；③ 为空则退回 `context_->GetCompileInfo<HcPreCompileInfo>()`，只填 `coreNum` 和 `ubSize`
  - 注意：compileInfo 分支**只写了 `aivCoreNum_`，没写 `aicCoreNum_`**，而后续 `CalcOpTiling` 要拿 `aicCoreNum_` 当除数（L616）——一旦真走这个分支就是除 0；再叠加 `TilingPrepareForHcPre` 是空实现（compileInfo 从未被填充），结论是这个 fallback 实际不可用。

- **`HcPreTiling::GetAttr()`** [hc_pre_tiling.cpp L89-L106](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L89-L106)
  - 职责：读取 4 个可选属性并给缺省值
  - 入参/出参：无显式入参；写出 `hcMult_`、`iterTimes_`、`hcEps_`、`normEps_`（后两者是 `double` 成员）；返回=状态码
  - 实现：① `auto* attrs = context_->GetAttrs()`，空指针直接报错返回；② `attrs->GetAttrPointer<int64_t>(HC_MULT_ATTR_IDX=0)`、`ITER_TIMES_ATTR_IDX=1`；③ `GetAttrPointer<float>(2)`、`(3)` 取两个 eps；④ 指针为空分别兜底 4 / 20 / 1e-6 / 1e-6
  - 注意：属性索引是「声明顺序的硬编码」0..3，与 def.cpp 顺序强绑定；`GetAttrPointer<float>` 读 float 再隐式提升到 `double hcEps_`，所以 TilingData 里回写的仍是 float（无损，因为源就是 float）。

- **`HcPreTiling::GetShapeAttrsInfoInner()`** [hc_pre_tiling.cpp L108-L170](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L108-L170)
  - 职责：解析输入 shape、做全部合法性校验、判定可选输入/输出是否存在
  - 入参/出参：读 `context_` 的 shape/attr；写出 `bs_`、`hcMix_`、`hcMult_`、`d_`、`tilingData_.hasPreMix/hasPreOut`；返回=状态码
  - 实现：① x 支持 3 维 `(bs, hc_mult, d)` 与 4 维 `(b, s, hc_mult, d)`，4 维时 `bs_ = b * s`；② 校验 `shapeHcMult == hcMult_`、`hcMult_ == SUPPORTED_HC_MULT(4)`、`hc_fn[1] == d_ * hcMult_`、`hc_scale[0] == 3`、`hc_base[0] == hcMix_`；③ 可选输入用 `context_->GetOptionalInputTensor(4) != nullptr` 判存在（注释说明 gen_placeholder 模式下 `GetInputShape` 恒非空）；④ 可选输出没有对应 API，用 `GetOutputShape(3)` 首维 `> 0` 判存在
  - 注意：x 既不是 3 维也不是 4 维时**没有 else 报错分支**，`bs_`/`d_` 保持 0，最后会在 `shapeHcMult(0) != hcMult_` 处失败，报的却是「hc_mult 不匹配」这种误导性错误；`hc_scale`/`hc_base` 的 shape 指针也未判空（L147/L154），而 x/hc_fn 判了。

- **`HcPreTiling::CalcVectorCoreNum()`** [hc_pre_tiling.cpp L173-L205](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L173-L205)
  - 职责：MPMD 模式下在「pre/post 核」与「sinkhorn 核」之间找最优核数切分
  - 入参/出参：读 `bs_`、`d_`、`hcMult_`、`iterTimes_`、`aivCoreNum_`；写出 `prePostCoreNum_`、`sinkhornCoreNum_`；返回=无（void）
  - 实现：① 单行成本模型 `prePostRowWork = 3 * d_`（PRE_POST_ROW_WORK_SCALE）、`sinkhornRowWork = 64 * hcMult_ * (max(iters,1) + 4)`（SINKHORN_ROW_WORK_SCALE 与 SETUP_ITER_EQUIVALENT）；② 枚举 `prePost = 1 .. totalCore-1`，剩余核给 sinkhorn；③ 每轮算 `latency = max(ceil(bs/p)*pw, ceil(bs/s)*sw)` 与 `imbalance = |两者之差|`；④ 选 latency 最小者，并列时取 imbalance 更小者并记录核数
  - 注意：循环上界是严格 `< totalCoreNum`，即 prePost 永远不会拿走全部核（保证 sinkhorn 至少 1 核）；成本模型是纯经验权重（3 / 64 / +4）而非实测，所以这是「启发式」而不是「精确解」；若 `bestLatency` 一直没人赋值（`hasCandidate` 为假）而 totalCoreNum 为 1，两个核数会留在 0，调用方 L219 的 `aivCoreNum_ < 2` 检查才是真正的兜底。

- **`HcPreTiling::CalcMKSplitCoreMembasePart2Tiling()`** [hc_pre_tiling.cpp L207-L414](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L207-L414)
  - 职责：主 tiling 分发 + UB 预算驱动的行/列切分（三个分支：premix、小 bs 原始模板、MPMD）
  - 入参/出参：读 `bs_`、`d_`、`hcMult_`、`aivCoreNum_`、`ubSize_`、已写好的 `cubeBlockDimK`；写出 `tilingKey_`、`rowOf*/rowLoopOf*/tailRowFactorOf*`、`dLoop_/dFactor_/tailDFactor_`、`prePost*` 及大批 `tilingData_` 字段；返回=状态码
  - 实现：① 分流：`hasPreMix` → `CalcPremixPart2Tiling()`；`bs <= 32` → `ORIGINAL_TILING_KEY(1)` 并按 `aivCoreNum_` 均分行；否则 → `MPMD_TILING_KEY(0)` + `CalcVectorCoreNum()`，且要求 `aivCoreNum_ >= 2`；② 行切核后 `usedAivCoreNums_ = min(ceil(bs/row), stage2CoreNum)`，`rowOfTailBlock_ = bs - (used-1)*row`；③ 列切分：先按 1 行试算全部 UB buffer（mix0/mix1/mix0Origin/mix1Origin/squareSum/squareReduce/rsqrt/x/y/post/base0/base1/xCast/yCast/rowBrcb/hcBrcb/mask + original 额外项），超了就令 `base` 从 2 递增、`dFactor_ = CeilDiv(d_, base)` 重试，出来后 `>32` 再 `DownAlign` 到 32；④ d 能全载时用 `while (rowFactor_ <= rowOfFormerBlock_)` 逐步加行直到超 UB 再回退 1；⑤ 为 pre/post 单独算一个行 tile（`prePostRowFactor`，容量允许时上限截到 4）；⑥ 算 `rowLoopOf*/tailRowFactorOf*` 并 `set_*` 落盘，`useOriginal` 时提前 return
  - 注意：`rowOnceLoop = std::min(rowOfFormerBlock_, 1)` 恒等于 1（`minRowPerCore=1`），所以 UB 预算是「按单行起步」的；`set_stage2RowFactor(rowFactor_)` 存的是 stage2 行因子，而 `set_prePostRowOfFormerBlock(rowOfFormerBlock_)` 复用同一个块行数——两个不同粒度共用了同名字段，看 kernel 时要区分；原始模板分支里 `prePostCoreNum_`/`sinkhornCoreNum_` 仍是 0（照样被 set 进 tilingData）。

- **`HcPreTiling::CalcPremixPart2Tiling()`** [hc_pre_tiling.cpp L417-L548](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L417-L548)
  - 职责：pre_mix（tilingKey=2）场景的按行切分，UB 预算额外含 mix2/combFrag/base2/reduceBuf
  - 入参/出参：读 `bs_`、`d_`、`hcMult_`、`aivCoreNum_`、`ubSize_`、`cubeBlockDimK`；写出 `rowOf*/dLoop_/dFactor_/tilingKey_` 与公共 tilingData 字段；返回=状态码
  - 实现：① `rowOfFormerBlock_ = CeilDiv(bs, aivCoreNum_)`，`usedAivCoreNums_` 截断到实际用到的核；② 与主线同构的两级试探：先假设 1 行 d 全载算总占用，超了用 `base++` 收缩 `dFactor_`（`>32` 时 DownAlign 到 32）；③ d 全载则用 while 涨 `rowFactor_` 直到超 UB 再回退；④ `tilingKey_ = PREMIX_TILING_KEY(2)` 后只 set 公共字段（bs/hcMix/hcMult/d/hcMultAlign/row*/d*/iterTimes/eps）
  - 注意：premix 不走 MPMD，所以**不写** `prePostCoreNum/sinkhorn*` 字段，kernel 必须靠 tilingKey 区分；这里 `rsqrtSize` 多乘了一个 `DOUBLE_BUFFER`（L435，主线 L242 没有），两份预算不可直接对比；`maskPatternSize` 初值 `BLOCK_SIZE*16`、涨行后变 `BLOCK_SIZE`（L448 与 L508），同一变量两种语义。

- **`HcPreTiling::CalcSinkhornTiling()`** [hc_pre_tiling.cpp L550-L609](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L550-L609)
  - 职责：为 MPMD 的 sinkhorn 阶段分配核数，并求 UB 能容纳的最大行因子
  - 入参/出参：读 `bs_`、`hcMult_`、`sinkhornCoreNum_`、`ubSize_`、`cubeBlockDimK`；写出 `sinkhornUsedCoreNum_`、`sinkhornRowOf*/sinkhornRowLoopOf*/sinkhornRowFactor_/sinkhornTailRowFactorOf*`；返回=状态码
  - 实现：① `sinkhornRowOfFormerBlock_ = CeilDiv(bs, sinkhornCoreNum_)`，实际用核 `min(ceil(bs/row), sinkhornCoreNum_)`，再算尾块；② `maxSinkhornRowFactor = min(该核行数, 64)`；③ 从 1 开始试 `calcTotalSize(rowFactor) + base2Size <= ubSize_`，`++` 循环，失败时先 `--` 再 break；④ 把 8 个 sinkhorn 字段 set 进 tilingData
  - 注意：这里的矩阵规模是 `hcMult_ * hcMult_`（4×4=16）而不是对齐后的 `hcMultAlign_`，因为 sinkhorn 作用在 4×4 的 comb 矩阵上；`base2Size` 是 lambda 之外单独加的，改 lambda 时别忘了它；while 判断在顶部，若 rowFactor=1 就放不下会把 `sinkhornRowFactor_` 减到 0，靠 L589 的 `OPS_ERR_IF(< 1)` 报 "buffers exceed UB size"，所以「报错信息指向 UB 不够」而不是「第一次就失败」。

- **`HcPreTiling::CalcOpTiling()`** [hc_pre_tiling.cpp L611-L654](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L611-L654)
  - 职责：算 cube 侧 M/K 切核、L1 尺寸、vector stage1 的 M 因子，然后转入 Part2
  - 入参/出参：读 `hcMult_`、`d_`、`bs_`、`aicCoreNum_`、`ubSize_`；写出 `k/kL1Size/mL1Size/cubeBlockDimM/cubeBlockDimK/multCoreSplitM/K/Size/cvLoopKSize/cubeCoreNum/stage1MFactor`；返回=状态码
  - 实现：① `kSize = hcMult_ * d_`；`mDimNum = min(aicCoreNum_, ceil(bs / 256))`，`kDimNum = aicCoreNum_ / mDimNum`；② `kDimNum == 1` 时把 `mDimNum` 提到 `aicCoreNum_`（K 吃不下剩余核，就全给 M）；否则若 `aicCoreNum_ % mDimNum != 0`，从 `mDimNum` 向上找第一个能整除 `aicCoreNum_` 的因子（注释：否则会闲置尾部 AIC，如 bs=2024 只用 16/20 核）；③ `singleCoreM = RoundUp(ceil(bs/mDim), BLOCK_CUBE=16)`，`splitKSize = RoundUp(ceil(kSize/kDim), 256)`，`kL1Size = min(128*256/mL1, 1024)` 再按 128 DownAlign；④ `cvLoopKSize = 1024`；stage1 行因子 `stage1MFactor = ubSize / ((sizeof(int16)+sizeof(int32)) * 2 * cvLoopK)`；⑤ 最后一行 `return CalcMKSplitCoreMembasePart2Tiling();`
  - 注意：`set_cubeBlockDimK` 存的是 **K 块数** `ceil(kSize / splitKSize)`，而 `splitKSize` 本身存进 `multCoreSplitKSize`，两个名字都叫「size/dimK」很容易看反；`cubeBlockDimK` 被 Part2 和 workspace 复用，所以 CalcOpTiling 必须在 Part2 之前跑完（这就是为什么 Part2 只被它调用）；`BLOCK_CUBE` 来自 `AscendC` 命名空间（值为 16），不是本文件的 `BLOCK_SIZE=32`；`121*...` 这类 L1 上限（M_L1_MAX_SIZE=256、A_L1_SIZE=128*256、K_L1_MAX_SIZE=1024）全是硬编码常量。

- **`HcPreTiling::DoOpTiling()`** [hc_pre_tiling.cpp L657-L680](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L657-L680)
  - 职责：tiling 的 5 步主流程编排，框架唯一会直接调用的成员
  - 入参/出参：无显式入参（状态都在 `this` 上）；返回=任一阶段失败则 `GRAPH_FAILED`，否则 `GRAPH_SUCCESS`
  - 实现：① `GetPlatformInfo()`；② `GetShapeAttrsInfoInner()`；③ `CalcOpTiling()`；④ `GetWorkspaceSize()`；⑤ `PostTiling()`，每步都 `if (... == GRAPH_FAILED) return GRAPH_FAILED;`
  - 注意：顺序不可交换——workspace 依赖 `mL1Size/cvLoopKSize/cubeCoreNum/cubeBlockDimK`，所以必须在 `CalcOpTiling()` 之后；`PostTiling()` 必须最后（它要一次性把 tilingKey/blockDim/workspace/tilingData 提交）。这里用 `== GRAPH_FAILED` 判断而非 `!= GRAPH_SUCCESS`，语义上等价（该返回值只有两态）。

- **`HcPreTiling::GetWorkspaceSize()`** [hc_pre_tiling.cpp L682-L700](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L682-L700)
  - 职责：估算 device 侧 workspace 大小，并强制不低于 208MB
  - 入参/出参：读 `mL1Size`、`cvLoopKSize`、`cubeCoreNum`、`cubeBlockDimK`、`bs`、`hcMix`；写出 `workspaceSize_`（字节）；返回=状态码
  - 实现：① `xCastFp32BufSize = mL1Size * RoundUp(cvLoopKSize, 128)`（元素数），`workspaceSize1 = cubeCoreNum * 2 * xCastFp32BufSize * 4`（fp32 cast + 双缓冲）；② `workspaceSize2 = RoundUp(cubeBlockDimK * bs * RoundUp(hcMix,128) * 4, 512)`（mm 输出）；③ `squareSumSize = RoundUp(cubeBlockDimK * RoundUp(bs,16) * 16 * 4, 512)`；④ `requiredSize = 三者 + 16MB`，与 `defaultSize = 16MB + 192MB = 208MB` 取大
  - 注意：只要 `requiredSize` 没超过 208MB 就一律给 208MB（拿空间换确定性，不是 bug）；`workspaceSize_` 只是记账，真正写回框架发生在 `PostTiling()` 的 `workspaces[0] = workspaceSize_`；16MB 是「预留缓冲」，没有对应的显式 buffer 名。

- **`HcPreTiling::PostTiling()`** [hc_pre_tiling.cpp L702-L717](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L702-L717)
  - 职责：把 tilingKey、blockDim、workspace 大小和序列化后的 tilingData 交给框架
  - 入参/出参：读 `tilingKey_`、`aicCoreNum_`、`workspaceSize_`、`tilingData_`；返回=状态码
  - 实现：① `context_->SetTilingKey(tilingKey_)`；② `context_->SetBlockDim(aicCoreNum_)`；③ `tilingKey_ == MPMD_TILING_KEY(0)` 时额外 `SetScheduleMode(1)`（MPMD 混合调度），失败即返回；④ `GetWorkspaceSizes(1)[0] = workspaceSize_`，再 `tilingData_.SaveToBuffer(raw->GetData(), raw->GetCapacity())` + `SetDataSize(tilingData_.GetDataSize())`
  - 注意：`SetBlockDim` 传的是 **AIC** 核数，而算子算子在 AIV（vector）上跑，910B 上 AIC:AIV = 1:2，kernel 侧的 AIV 数由这个 blockDim 映射出来，别想当然以为它等于 `aivCoreNum_`；`SetScheduleMode(1)` 只对 MPMD 设，漏设会导致 pre/post 与 sinkhorn 两段无法并行调度；`SaveToBuffer` 不检查容量，`GetCapacity()` 必须足够放下 70 个字段。

- **`TilingPrepareForHcPre(gert::TilingParseContext *context)`** [hc_pre_tiling.cpp L719-L723](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L719-L723)
  - 职责：`TilingParse` 回调，当前是空实现（空壳）
  - 入参/出参：`context`=tiling parse 上下文（编译/加载期传入，本实现直接 `(void)context` 丢弃）；返回=固定 `ge::GRAPH_SUCCESS`
  - 实现：① 仅 `(void)context;` 抑制未使用告警；② 直接返回成功；③ 框架在 `IMPL_OP_OPTILING(...).TilingParse<HcPreCompileInfo>(...)` 绑定时调用一次
  - 注意：正因为这里没填 `HcPreCompileInfo.coreNum/ubSize`，一旦 `GetPlatformInfo()` 走到 compileInfo 兜底分支，拿到的核数/UB 都是构造时的 0；要支持离线编译（ACL 编译态）必须在这里真的解析并写回 compileInfo。

- **`TilingForHcPre(gert::TilingContext *context)`** [hc_pre_tiling.cpp L725-L743](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L725-L743)
  - 职责：tiling 入口，按 SoC 版本把请求分派给 arch35 或主线实现
  - 入参/出参：`context`=框架 tiling 上下文；返回=对应实现的 `DoOpTiling()` 结果
  - 实现：① `context == nullptr` 报错返回；② `context->GetPlatformInfo()` 为空报错返回；③ `PlatformAscendC::GetSocVersion()`，若为 `ASCEND950` 则构造 `HcPreTilingRegbase::HcPreTilingRegbase` 并 `DoOpTiling()`（打 INFO 日志 "Using arch35 tiling"）；④ 否则构造 `HcPreTiling` 并 `DoOpTiling()`
  - 注意：L732 的错误串写成了 `"TilingForMoeGatingTopKHash"`（复制粘贴遗留），按日志搜算子名会搜不到；950 分支用的是完全另一个类（成员少得多，见 `hc_pre_tiling_arch35.h`），两套 tiling 的 tilingKey 语义也不同（主线 0/1/2，regbase 1000/1002/1001/1003）。

- **`IMPL_OP_OPTILING(HcPre)`** [hc_pre_tiling.cpp L745-L747](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L745-L747)
  - 职责：把 HcPre 的 tiling 回调与 compileInfo 类型注册到框架
  - 入参/出参：`HcPre`=算子名；`.Tiling(...)` 与 `.TilingParse<...>(...)` 链式配置；返回=无（宏展开为静态注册）
  - 实现：① `.Tiling(TilingForHcPre)` 指定 host tiling 入口；② `.TilingParse<HcPreCompileInfo>(TilingPrepareForHcPre)` 指定编译期解析回调及其产出的结构体类型；③ 框架按算子名把这份注册与 def.cpp 的原型配对
  - 注意：算子名必须与 `hc_pre_def.cpp` 的 `OP_ADD(HcPre)` 完全一致，否则原型找不到 tiling；`TilingParse` 的模板参数类型必须与 `GetCompileInfo<HcPreCompileInfo>()` 的类型一致，否则是未定义行为。

### 附一：`HcPreTilingData` 全部字段（70 个，声明顺序即内存布局）

> 说明：「单位」一栏中，行=bs 方向的行数，元素=向量元素个数，块=K 轴块数，byte=字节数，次=循环次数，核=AI Core 个数。
> 「由谁计算」一栏中，`Part2`=`CalcMKSplitCoreMembasePart2Tiling`，`Premix`=`CalcPremixPart2Tiling`，`Sinkhorn`=`CalcSinkhornTiling`，`Calc`=`CalcOpTiling`，`Regbase`=`HcPreTilingRegbase::SaveCommonTiling`（arch35 路径），`未使用`=本两套 tiling 都未赋值（预留字段）。
> kernel 侧是「按字段名取值」的（`tilingData.get_bs()` 之类），所以**任何没被 `set_` 过的字段在 device 上读到的就是 0**，而 0 会一路被 `CeilDiv` 静默兜底，真正的错误常常在很远的地方才炸。

#### A. 形状与精度（host 解析层）

| 字段 | 含义 | 单位 | 由谁计算 |
|---|---|---|---|
| `bs` | batch×seq 展平后的总行数 | 行 | `GetShapeAttrsInfoInner` |
| `hcMix` | hc_fn 第 0 维，comb 矩阵的行数（mix 组数） | 个 | `GetShapeAttrsInfoInner`（取 hc_fn dim0） |
| `hcMult` | x 的 hc_mult 维，当前只支持 4 | 个 | `GetAttr` + `GetShapeAttrsInfoInner` 校验 |
| `d` | 每个 hc_mult 头内的 hidden 长度 | 元素 | `GetShapeAttrsInfoInner`（x 末维） |
| `hcMultAlign` | hcMult 按 8 个 float（32B）向上对齐 | 元素 | `Part2`/`Premix` 的 `RoundUp(hcMult,8)` |
| `iterTimes` | sinkhorn 迭代次数（attr hc_sinkhorn_iters） | 次 | `GetAttr`（默认 20） |
| `hcEps` | sinkhorn 归一化的数值稳定 eps | 无量纲 | `GetAttr`（默认 1e-6） |
| `normEps` | RMSNorm 的 eps | 无量纲 | `GetAttr`（默认 1e-6） |
| `hasPreMix` | 是否存在可选输入 pre_mix（0/1） | 布尔 | `GetShapeAttrsInfoInner` |
| `hasPreOut` | 是否存在可选输出 pre（0/1） | 布尔 | `GetShapeAttrsInfoInner` |

#### B. 行（bs）方向切核

| 字段 | 含义 | 单位 | 由谁计算 |
|---|---|---|---|
| `rowOfFormerBlock` | 每个核负责的行数（非尾核） | 行 | `Part2`：`ceil(bs/核数)` |
| `rowOfTailBlock` | 最后一个核负责的行数 | 行 | `Part2`：`bs-(used-1)*former` |
| `rowLoopOfFormerBlock` | 非尾核内按 rowFactor 的行循环次数 | 次 | `Part2` |
| `rowLoopOfTailBlock` | 尾核的行循环次数 | 次 | `Part2` |
| `rowFactor` | 原始模板（M 不切）单次搬入行数 | 行 | 仅 `Regbase` 的 M 切分写；主线不写 |
| `rowInnerFactor` | regbase 非 K 切分时的行内因子（=rowFactor） | 行 | `Regbase` |
| `tailRowFactorOfFormerBlock` | 非尾核最后一次循环实际行数 | 行 | `Part2` |
| `tailRowFactorOfTailBlock` | 尾核最后一次循环实际行数 | 行 | `Part2` |
| `secondUsedCoreNum` | 第二阶段实际使用核数 | 核 | `Part2`：`usedAivCoreNums_`；`Regbase`：`usedCoreNum` |
| `firstUsedCoreNum` | 第一阶段用核数（预留） | 核 | 未使用 |
| `stage2RowFactor` | 第二阶段（vector）单次行数 | 行 | `Part2`（=rowFactor_）；`Regbase` K 切分时 |

#### C. d（最后一维）方向切分

| 字段 | 含义 | 单位 | 由谁计算 |
|---|---|---|---|
| `dLoop` | d 轴循环次数 | 次 | `Part2`/`Premix`：`ceil(d/dFactor)` |
| `dFactor` | 单次处理的 d 长度 | 元素 | `Part2`/`Premix` 试探得到 |
| `tailDFactor` | d 轴最后一次的长度 | 元素 | `Part2`/`Premix` |
| `kUbSize` | UB 单次处理的 K 长度（arch35） | 元素 | `Regbase` |
| `mUbSize` | UB 单次处理的 M 行数（arch35 premix） | 行 | `Regbase` |

#### D. cube 切核与 L1/UB 尺寸

| 字段 | 含义 | 单位 | 由谁计算 |
|---|---|---|---|
| `k` | `hcMult * d`，即 hc_fn 的 K 总长 | 元素 | `Calc` |
| `kL1Size` | L1 上单次加载的 K 长度（128 对齐） | 元素 | `Calc` |
| `mL1Size` | L1 上单次加载的 M 行数（上限 256） | 行 | `Calc` |
| `cubeBlockDimM` | cube M 方向切核数 | 核 | `Calc` |
| `cubeBlockDimK` | cube K 方向块数 `ceil(k/splitKSize)` | 块 | `Calc` |
| `cubeCoreNum` | cube 总核数（=aicCoreNum） | 核 | `Calc` |
| `multCoreSplitMSize` | 多核切 M 后单核 M 行数（16 对齐） | 行 | `Calc` |
| `multCoreSplitKSize` | 多核切 K 后单核 K 长度（256 对齐） | 元素 | `Calc` |
| `cvLoopKSize` | CV（cube-vector）同步使用的 K 轴长度，固定 1024 | 元素 | `Calc` |
| `stage1MFactor` | stage1 一次处理的 M 行数 | 行 | `Calc`（ubSize/行字节） |
| `kBlockFactor` | K 轴块因子（arch35 取 cubeBlockDimK） | 块 | `Regbase` |
| `bufferPool0Size` | arch35 UB 池 0 大小（整块 UB） | byte | `Regbase` |
| `bufferPool1Size` | arch35 UB 池 1 剩余可用大小 | byte | `Regbase` |

#### E. K 轴预留字段（两套 tiling 均未赋值）

| 字段 | 含义 | 单位 | 由谁计算 |
|---|---|---|---|
| `kFactor` | K 轴单次处理长度 | 元素 | 未使用 |
| `tailKFactor` | K 轴尾块长度 | 元素 | 未使用 |
| `kLoop` | K 轴循环次数 | 次 | 未使用 |
| `kLoopOfFormerBlock` | 非尾块 K 循环次数 | 次 | 未使用 |
| `kLoopOfTailBlock` | 尾块 K 循环次数 | 次 | 未使用 |
| `stage1KFactor` | stage1 的 K 因子 | 元素 | 未使用 |
| `kFactorOfFormerBlock` | 非尾块 K 因子 | 元素 | 未使用 |
| `tailKSizeOfFormerBlock` | 非尾块 K 尾长度 | 元素 | 未使用 |
| `tailKSizeOfTailBlock` | 尾块 K 尾长度 | 元素 | 未使用 |

#### F. M 轴预留字段

| 字段 | 含义 | 单位 | 由谁计算 |
|---|---|---|---|
| `mLoopOfFormerBlock` | 非尾块 M 循环次数 | 次 | 未使用 |
| `mLoopOfTailBlock` | 尾块 M 循环次数 | 次 | 未使用 |
| `formerMSize` | 非尾块 M 大小 | 行 | 未使用 |
| `tailMSizeOfFormerBlock` | 非尾块 M 尾长度 | 行 | 未使用 |
| `tailMSizeOfTailBlock` | 尾块 M 尾长度 | 行 | 未使用 |

#### G. MPMD：pre/post 段

| 字段 | 含义 | 单位 | 由谁计算 |
|---|---|---|---|
| `prePostCoreNum` | pre/post 段占用的核数 | 核 | `CalcVectorCoreNum`（原始模板下为 0） |
| `prePostRowOfFormerBlock` | pre/post 段每核行数（复用 `rowOfFormerBlock_`） | 行 | `Part2` |
| `prePostRowOfTailBlock` | pre/post 段尾核行数（复用 `rowOfTailBlock_`） | 行 | `Part2` |
| `prePostRowLoopOfFormerBlock` | pre/post 段行循环次数 | 次 | `Part2` |
| `prePostRowLoopOfTailBlock` | pre/post 段尾块行循环次数 | 次 | `Part2` |
| `prePostRowFactor` | pre/post 段单次处理行数（上限 4） | 行 | `Part2` |
| `prePostTailRowFactorOfFormerBlock` | pre/post 段尾循环行数 | 行 | `Part2` |
| `prePostTailRowFactorOfTailBlock` | pre/post 段尾块尾循环行数 | 行 | `Part2` |

#### H. MPMD：sinkhorn 段

| 字段 | 含义 | 单位 | 由谁计算 |
|---|---|---|---|
| `sinkhornCoreNum` | sinkhorn 段分到的核数 | 核 | `CalcVectorCoreNum`（原始模板下为 0） |
| `sinkhornUsedCoreNum` | sinkhorn 段实际使用核数 | 核 | `Sinkhorn` |
| `sinkhornRowOfFormerBlock` | sinkhorn 段每核行数 | 行 | `Sinkhorn`：`ceil(bs/sinkhornCoreNum)` |
| `sinkhornRowOfTailBlock` | sinkhorn 段尾核行数 | 行 | `Sinkhorn` |
| `sinkhornRowLoopOfFormerBlock` | sinkhorn 段行循环次数 | 次 | `Sinkhorn` |
| `sinkhornRowLoopOfTailBlock` | sinkhorn 段尾块行循环次数 | 次 | `Sinkhorn` |
| `sinkhornRowFactor` | sinkhorn 单次处理行数（≤64） | 行 | `Sinkhorn` |
| `sinkhornTailRowFactorOfFormerBlock` | sinkhorn 尾循环行数 | 行 | `Sinkhorn` |
| `sinkhornTailRowFactorOfTailBlock` | sinkhorn 尾块尾循环行数 | 行 | `Sinkhorn` |

> 合计：A(10) + B(11) + C(5) + D(13) + E(9) + F(5) + G(8) + H(9) = 70 个字段，与头文件 [hc_pre_tiling.h L45-L121](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.h#L45-L121) 的 `TILING_DATA_FIELD_DEF` 条数一致。

### 附二：`HcPreTiling` 的成员变量（37 个，[hc_pre_tiling.h L154-L190](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.h#L154-L190)）

> 这些成员是「Host 侧的临时草稿纸」：函数之间靠它们传递中间结果，最后在 `Part2` 里统一 `set_*` 到 `tilingData_`。没有它们，`PostTiling` 也无从写回。

| 成员 | 职责 | 由谁写入 |
|---|---|---|
| `context_` | 框架传入的 tiling 上下文指针（借用，不释放） | 构造函数 |
| `tilingKey_` | 模板选择开关：0=MPMD、1=原始、2=premix | `Part2`/`Premix` |
| `tilingData_` | 最终下发给 kernel 的参数结构体（`HcPreTilingData`） | 各计算函数 + `PostTiling` 序列化 |
| `aivCoreNum_` | 平台 vector 核数（tiling 的主力核数） | `GetPlatformInfo` |
| `aicCoreNum_` | 平台 cube 核数（=BlockDim、cube 切核基数） | `GetPlatformInfo` |
| `workspaceSize_` | 估算出的 device workspace 字节数 | `GetWorkspaceSize` |
| `usedAivCoreNums_` | 第二阶段真正会用到的核数（尾核可能不满） | `Part2`/`Premix` |
| `ubSize_` | 单核 Unified Buffer 字节数，所有 UB 预算的硬上限 | `GetPlatformInfo` |
| `bs_` | 总行数（b*s 展平） | `GetShapeAttrsInfoInner` |
| `hcMix_` | comb 矩阵行数（hc_fn dim0） | `GetShapeAttrsInfoInner` |
| `hcMult_` | hc_mult，固定 4 | `GetAttr` |
| `d_` | 单头 hidden 长度 | `GetShapeAttrsInfoInner` |
| `hcMultAlign_` | hcMult 的 32B 对齐值 | `Part2`/`Premix` |
| `rowOfFormerBlock_` | 每核行数（非尾核） | `Part2`/`Premix` |
| `rowOfTailBlock_` | 尾核行数 | `Part2`/`Premix` |
| `rowLoopOfFormerBlock_` | 非尾核行循环次数 | `Part2`/`Premix` |
| `rowLoopOfTailBlock_` | 尾核行循环次数 | `Part2`/`Premix` |
| `rowFactor_` | 单次搬入行数（UB 预算试出来的结果） | `Part2`/`Premix` |
| `tailRowFactorOfFormerBlock_` | 非尾核最后一跳的行数 | `Part2`/`Premix` |
| `tailRowFactorOfTailBlock_` | 尾核最后一跳的行数 | `Part2`/`Premix` |
| `dLoop_` | d 轴循环次数 | `Part2`/`Premix` |
| `dFactor_` | 单次 d 长度 | `Part2`/`Premix` |
| `tailDFactor_` | d 轴最后一跳长度 | `Part2`/`Premix` |
| `iterTimes_` | sinkhorn 迭代次数 | `GetAttr` |
| `hcEps_` | sinkhorn eps（double 存 float 值） | `GetAttr` |
| `normEps_` | RMSNorm eps（double 存 float 值） | `GetAttr` |
| `socVersion_` | SoC 版本，默认 `ASCEND910B` | `GetPlatformInfo` |
| `prePostCoreNum_` | MPMD 中 pre/post 段核数 | `CalcVectorCoreNum` |
| `sinkhornCoreNum_` | MPMD 中 sinkhorn 段核数 | `CalcVectorCoreNum` |
| `sinkhornUsedCoreNum_` | sinkhorn 段实际用核数 | `CalcSinkhornTiling` |
| `sinkhornRowOfFormerBlock_` | sinkhorn 每核行数 | `CalcSinkhornTiling` |
| `sinkhornRowOfTailBlock_` | sinkhorn 尾核行数 | `CalcSinkhornTiling` |
| `sinkhornRowLoopOfFormerBlock_` | sinkhorn 行循环次数 | `CalcSinkhornTiling` |
| `sinkhornRowLoopOfTailBlock_` | sinkhorn 尾块行循环次数 | `CalcSinkhornTiling` |
| `sinkhornRowFactor_` | sinkhorn 单次行数（≤64） | `CalcSinkhornTiling` |
| `sinkhornTailRowFactorOfFormerBlock_` | sinkhorn 尾循环行数 | `CalcSinkhornTiling` |
| `sinkhornTailRowFactorOfTailBlock_` | sinkhorn 尾块尾循环行数 | `CalcSinkhornTiling` |

### 附三：`HcPreTilingRegbase` 的成员变量（15 个，[hc_pre_tiling_arch35.h L386-L400](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L386-L400)）

> 该类是 `ASCEND950`（arch35）专用实现，方法体也写在同一头文件 [hc_pre_tiling_arch35.h L403-L811](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L403-L811) 里（不在本次三个文件范围内，故不逐函数展开）。它复用同一份 `HcPreTilingData`，但把「按核切行 → 按 UB 切行/列」重写为「先定 M/K 切分 → 选 UB 池 → 再切行」，tilingKey 也换成 1000/1002（K 切分）与 1001/1003（M 切分）。
> 与 `HcPreTiling` 相比，它**没有**任何 `row*_/d*_/sinkhorn*_` 中间量——这些全部改为函数内的局部结构体（`CoreRowTiling`/`RowAndDTiling`/`PremixUbTiling`）传参，成员只保留跨阶段共享的输入量。
> 它的公开方法与主线同名同职责（`GetPlatformInfo`/`GetAttr`/`GetShapeAttrsInfoInner`/`CalcOpTiling`/`GetWorkspaceSize`/`PostTiling`/`DoOpTiling` 一套，外加 `CalcRegbaseCommonTiling`/`CalcRegbaseOpTiling`/`CalcMKSplitCorePart2Tiling`/`CalcPremixMSplitTiling`/`CalcPremixMKSplitTiling`/`InitPremixUbConfig`/`SelectPremixUbTiling`/`CalcPremixRowTiling`/`CalcNoPremixRowTiling`/`SaveCommonTiling`），差别只在切分算法。

| 成员 | 职责 | 由谁写入 |
|---|---|---|
| `context_` | tiling 上下文指针（借用） | 构造函数 |
| `tilingKey_` | arch35 模板键：1000/1001/1002/1003 | `CalcOpTiling` |
| `tilingData_` | 与主线共用的下发结构体 | 各计算函数、`SaveCommonTiling` |
| `aivCoreNum_` | vector 核数（arch35 的行切分基数） | `GetPlatformInfo` |
| `aicCoreNum_` | cube 核数（同时用于 `SetBlockDim`） | `GetPlatformInfo` |
| `workspaceSize_` | 仅 K 切分模板需要预留，M 切分为 0 | `GetWorkspaceSize` |
| `ubSize_` | UB 字节数（`bufferPool0Size` 初值） | `GetPlatformInfo` |
| `bs_` | 总行数 | `GetShapeAttrsInfoInner` |
| `hcMix_` | comb 矩阵行数 | `GetShapeAttrsInfoInner` |
| `hcMult_` | hc_mult（arch35 从 shape 读，而不是先读 attr） | `GetShapeAttrsInfoInner` |
| `d_` | 单头 hidden 长度 | `GetShapeAttrsInfoInner` |
| `iterTimes_` | sinkhorn 迭代次数 | `GetAttr` |
| `hcEps_` | sinkhorn eps | `GetAttr` |
| `normEps_` | RMSNorm eps | `GetAttr` |
| `socVersion_` | SoC 版本 | `GetPlatformInfo` |

### 附四：`hc_pre_tiling.cpp` 匿名命名空间中的常量（理解切分策略的钥匙）

| 常量 | 值 | 作用 |
|---|---|---|
| `BLOCK_SIZE` | 32 | 32 字节对齐粒度（UB/cube 搬运最小块） |
| `DOUBLE_BUFFER` | 2 | 双缓冲系数，所有流式 buffer 都乘它 |
| `M_L1_MAX_SIZE` | 256 | L1 上 M 方向单次最大行数 |
| `A_L1_SIZE` | 128*256 | 用来反推 `kL1Size` 的 L1 容量模型值 |
| `K_L1_MAX_SIZE` | 1024 | `kL1Size` 上限 |
| `K_L1_ALIGN_SIZE` | 128 | `kL1Size` 向下对齐粒度 |
| `CV_LOOP_K_SIZE` | 1024 | cube-vector 同步粒度的 K 长度 |
| `K_MULIT_CORE_SPLIT_BASE_SIZE` | 256 | 多核切 K 的粒度基数 |
| `SQUARE_SUM_SIZE` | 16 | 平方和归约的桶大小（16 个 float） |
| `MASK_PATTERN_DIM_SIZE` | 16 | 生成 mask pattern 的维度基数 |
| `HC_MULT_ATTR_IDX` / `ITER_TIMES_ATTR_IDX` / `HC_EPS_ATTR_IDX` / `NORM_EPS_ATTR_IDX` | 0 / 1 / 2 / 3 | 属性在 `GetAttrs()` 中的下标，与 def.cpp 声明顺序绑定 |
| `DEFAULT_ITER_TIMES` | 20 | 属性缺失时的迭代次数 |
| `SUPPORTED_HC_MULT` | 4 | 唯一支持的 hc_mult |
| `PRE_POST_ROW_WORK_SCALE` | 3 | pre/post 单行工作量权重（÷d 的成本模型） |
| `SINKHORN_ROW_WORK_SCALE` | 64 | sinkhorn 单行工作量权重 |
| `SINKHORN_SETUP_ITER_EQUIVALENT` | 4 | sinkhorn 初始化折算成 4 次迭代 |
| `SMALL_BS_THRESHOLD` | 32 | `bs <= 32` 走原始模板（不分 MPMD） |
| `MPMD_TILING_KEY` | 0 | 多核 MPMD 模板 |
| `ORIGINAL_TILING_KEY` | 1 | 原始单段模板 |
| `PREMIX_TILING_KEY` | 2 | pre_mix 模板 |
| `WORKSPACE_SIZE` | 32 | 本文件中未使用（遗留常量） |
| `REPEAT_SIZE` / `UB_RESEVED_SIZE` | 256 / 8192 | 本文件中未使用（arch35 侧才用 `REPEAT_SIZE`） |

### 附五：调用链速览（把这 22 条串起来）

```
框架
 └─ TilingForHcPre                      入口，判空 + 按 SoC 分流
     ├─ (ASCEND950) HcPreTilingRegbase::DoOpTiling      → arch35 另一套
     └─ (其他)      HcPreTiling::DoOpTiling
                       ├─ 1. GetPlatformInfo        → aiv/aic/ub/soc
                       ├─ 2. GetShapeAttrsInfoInner → bs/hcMix/hcMult/d + 5 项校验
                       │      └─ GetAttr            → hcMult/iterTimes/2 个 eps
                       ├─ 3. CalcOpTiling           → k/kL1/mL1/cubeBlockDimM,K/cvLoopK/stage1MFactor
                       │      └─ CalcMKSplitCoreMembasePart2Tiling
                       │            ├─ hasPreMix        → CalcPremixPart2Tiling
                       │            ├─ bs<=32           → tilingKey=1（原始）
                       │            ├─ else             → CalcVectorCoreNum + tilingKey=0（MPMD）
                       │            │                     └─ CalcSinkhornTiling
                       │            └─ 公共：UB 预算 → rowFactor/dFactor/dLoop/tail* → set_*
                       ├─ 4. GetWorkspaceSize       → 至少 208MB
                       └─ 5. PostTiling             → SetTilingKey/SetBlockDim/(SetScheduleMode)/
                                                        workspaces[0]/SaveToBuffer
```

- 一句话总结：`CalcOpTiling` 决定「怎么分给 cube 核」，`Part2` 决定「怎么分给 vector 核并把数据塞进 UB」，`GetWorkspaceSize`/`PostTiling` 决定「怎么把结论告诉框架」。

| 路径 | tilingKey | 触发条件 | 含义 |
|---|---|---|---|
| 主线 | 0 (`MPMD_TILING_KEY`) | 无 pre_mix 且 `bs > 32` | 多核 MPMD：pre/post 与 sinkhorn 分核并行 |
| 主线 | 1 (`ORIGINAL_TILING_KEY`) | 无 pre_mix 且 `bs <= 32` | 原始单段模板，按核均分整行 |
| 主线 | 2 (`PREMIX_TILING_KEY`) | 存在 pre_mix（优先级最高） | 按行切分，UB 多算 mix2/combFrag |
| arch35 | 1000 | 无 pre_mix，`kDimNum != 1`（K 切分） | M 不切、K 多核切分 |
| arch35 | 1001 | 无 pre_mix，`kDimNum == 1`（M 切分） | K 不切、M 多核切分 |
| arch35 | 1002 | 有 pre_mix，K 切分 | pre_mix + K 多核切分 |
| arch35 | 1003 | 有 pre_mix，M 切分 | pre_mix + M 多核切分 |

### 附六：术语速查（给完全不懂 Ascend C 的读者）

| 术语 | 含义 |
|---|---|
| Tiling | 「切分方案」。Host 侧算出怎么把大张量切成小块分给各核，算出来的结果就是 `*TilingData` |
| tilingKey | 模板编号，kernel 侧 `if (tilingKey == ...)` 选择不同的 kernel 实现分支 |
| AIC / AIV | AI Core 的 cube（矩阵）单元 / vector（向量）单元；910B 上 AIV 数量是 AIC 的 2 倍 |
| UB | Unified Buffer，每个 vector 核私有的片上高速缓冲（约 192KB 级），所有 vector 运算的输入输出都必须先搬进 UB |
| L1 | cube 侧片上缓冲，`mL1Size`/`kL1Size` 描述 L1 上装载的 M/K 分块 |
| BlockDim | 下发给 kernel 的核数（`SetBlockDim`），决定有多少个核并行跑这个 kernel |
| MPMD | 多模板多数据流：同一次 kernel 里不同核跑不同模板（这里 pre/post 与 sinkhorn 分核并行），需要 `SetScheduleMode(1)` |
| Workspace | device 侧的临时全局内存，由 host 在 `GetWorkspaceSizes` 处申报大小 |
| Double Buffer | 双缓冲：准备下一块数据的同时计算当前块，所有 buffer 尺寸都要 ×2 |
| Brcb | Broadcast 类指令（把一行数据广播成矩阵），代码里的 `rowBrcb/hcBrcb` 是为它准备的 pattern buffer |
| ND2NZ | 把普通 ND 布局转成 cube 需要的 NZ 布局，需要额外一块 UB 空间 |
| OPS_ERR_IF / OPS_LOG_E | 昇腾算子错误处理宏：条件成立则打日志并执行后面的 return 语句 |
| graphStatus | `ge::GRAPH_SUCCESS` / `GRAPH_FAILED`，算子接口的统一返回类型 |
| hcMix / hcMult / d / bs | 本算子的四个形状变量：comb 矩阵行数 / 4 个头 / 头内 hidden / 总行数 |
| cube / vector | 芯片上两类计算单元：cube 做矩阵乘（MM），vector 做逐元素/归约运算 |
| RMSNorm | 按行做「平方均值 → 开方 → 归一化」的归一化操作；本算子用 `norm_eps` 控制它的数值稳定性 |
| sinkhorn | 迭代式行列归一化算法（把方阵反复按行、按列归一），`iterTimes` 即迭代次数，作用在 `hcMult × hcMult` 的 comb 小矩阵上 |
| pre / post / comb_frag | 三个输出：`pre` 与 `post` 是缩放系数（`pre` 可选，`hcMult` 列），`comb_frag` 是 sinkhorn 的中间矩阵结果 |
| pre_mix | 可选输入，预混合系数；它的存在会切换到 `tilingKey=2` 的专用模板 |
| `GetOptionalInputTensor` | 取「可选输入」的 API，不存在时返回 `nullptr`；不能用 `GetInputShape` 判存在与否 |
| 32B 对齐 | 昇腾 DMA/UB 的最小搬运单位是 32 字节，所以所有 buffer 尺寸都要按 32B（8 个 float）向上对齐 |
| `AscendC::BLOCK_CUBE` | cube 的 block 常量，值为 16（M 方向对齐用它，别和 `BLOCK_SIZE=32` 混） |


---

## 12. 函数级详解：arch35 Host 侧 tiling（`hc_pre_tiling_arch35.h`）

源码：`vllm-ascend/csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h`（共 813 行）

### 0. 给"完全不懂 Ascend C"的读者：名词速查

- **AIC（Cube 核）**：做矩阵乘的核；**AIV（Vector 核）**：做逐元素、归约、sigmoid、rsqrt 等。
- **L1**：Cube 侧的片上大缓冲。本文件用 `A_L1_SIZE = 128*256 = 32KB` 估算"一次能放多大的 A 矩阵块"。
- **UB**：Vector 侧统一缓冲。arch35 直接用 `ubSize_`（字节）做精细预算，预算不够就缩小分块。
- **形状**：`bs` 行数（token 数）、`hcMult` 每行分几头、`d` 每头维度、`hcMix` 混合头数；K 轴 = `hcMult * d`。
- **切 K**：把 K 轴拆给多个 AIC，需要跨核 workspace 累加；本文件用它决定 4 种模板中的哪一种。
- **TilingKey**：Host 写给 Kernel 的整数分支号，Kernel 靠它选实现模板。
- **tiling 的本质**：算出"每核分几行、每块多大、每段缓冲多少字节"，塞进 `HcPreTilingData` 交给 Kernel。

本文件分两层：文件顶部匿名 namespace 里的纯计算辅助函数（内部链接，等价于 static），以及 `HcPreTilingRegbase` 类（真正被 `TilingForHcPre` 在 950 上实例化）。

---

### 1. 匿名 namespace 辅助函数（按文件出现顺序）

- **`CeilDiv`** [hc_pre_tiling_arch35.h L24-L30](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L24-L30)
  - 职责：向上取整除法，全文件最基础的算式
  - 入参/出参：`x` 被除数、`y` 除数 → `int64_t` 商
  - 实现：`y != 0` 时返回 `(x + y - 1) / y`；`y == 0` 时直接返回 `x`
  - 注意：除零时静默返回 `x` 而非报错，调用方必须自己保证 `y > 0`

- **`DownAlign`** [hc_pre_tiling_arch35.h L31-L36](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L31-L36)
  - 职责：向下对齐到 `y` 的整数倍
  - 入参/出参：`x` 待对齐值、`y` 对齐粒度 → `int64_t`
  - 实现：`(x / y) * y`；`y == 0` 时原样返回 `x`
  - 注意：与 `RoundUp` 方向相反；只用于 `dFactor` 收尾对齐到 32 和余量取整

- **`RoundUp`** [hc_pre_tiling_arch35.h L37-L39](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L37-L39)
  - 职责：向上对齐到 `y` 的整数倍
  - 入参/出参：`x` 待对齐值、`y` 粒度 → `int64_t`
  - 实现：`CeilDiv(x, y) * y`，复用上面的 `CeilDiv`
  - 注意：—

- **`CalcCoreRowTiling`** [hc_pre_tiling_arch35.h L87-L96](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L87-L96)
  - 职责：按 AIV 核数均分行，得出每核行数与实际用核数
  - 入参/出参：`bs`、`coreNum`、`CoreRowTiling&` → `bool`
  - 实现：`rowOfFormerBlock = CeilDiv(bs, coreNum)`；`usedCoreNum = min(CeilDiv(bs, rowOfFormerBlock), coreNum)`；`rowOfTailBlock = bs - (usedCoreNum-1)*rowOfFormerBlock`
  - 注意：`bs` 或 `coreNum ≤ 0` 直接返回 false；只填 3 个字段，循环信息留给 `CompleteCoreRowTiling`

- **`CompleteCoreRowTiling`** [hc_pre_tiling_arch35.h L98-L110](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L98-L110)
  - 职责：给定 `rowFactor` 补齐行循环次数与尾块行数
  - 入参/出参：`CoreRowTiling&`、`rowFactor` → `bool`
  - 实现：`rowLoop* = CeilDiv(row*, rowFactor)`；`tailRowFactor* = row* - (rowLoop*-1) * rowFactor`
  - 注意：`rowFactor ≤ 0` 返回 false；必须在 `rowFactor` 定稿后调用，否则所有 loop 数都错

- **`SelectPremixML1Size`** [hc_pre_tiling_arch35.h L112-L132](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L112-L132)
  - 职责：premix 且不切 K 时，选开销最小的 mL1
  - 入参/出参：`bs`、`mDimNum`（M 轴核数）、`kSize` → `uint64_t` mL1
  - 实现：`mL1` 自 256 每次减 16 到 `BLOCK_CUBE`；配 `kL1 = min(A_L1_SIZE/mL1, 1024)` 再下取整 128；剔除 `kL1/2 >= 256`（即 kL1 ≥ 512）或 `kSize % kL1 != 0` 的候选；取 `cost = CeilDiv(CeilDiv(bs, mL1), mDimNum) * mL1` 最小者
  - 注意：返回值再 `min(bestM, 208)`；调用方会用 208 重算 kL1，故实际 kL1 常只剩 128

- **`CalcNoPremixUbBufferSize`** [hc_pre_tiling_arch35.h L157-L174](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L157-L174)
  - 职责：估算非 premix 一组 `rowFactor × dFactor` 的 UB 总字节
  - 入参/出参：`UbBufferConfig`、`rowFactor`、`dFactor` → `int64_t`
  - 实现：mixes（fp32）+ x/y（bf16 ×2 双缓冲）+ pre/post（fp32 ×2 ×2 张）+ combFrag（`hcMult²`）；`isKSplit` 再追加 kBlock 份 mm、rms 与 bases
  - 注意：x 用 2 字节、pre/post 用 4 字节；`dFactor` 先 `RoundUp(…, 16)` 再计入

- **`IsPremixMFused`** [hc_pre_tiling_arch35.h L176-L182](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L176-L182)
  - 职责：判定 premix 的 M 轴能否整块融合
  - 入参/出参：`PremixUbConfig`、`mUbSize`、`kUbSize` → `bool`
  - 实现：需同时满足 `!isKSplit`、`hasPreMix`、`kUbSize > 0`、`mUbSize >= maxMRows`、`multCoreSplitKSize == hcMult*d`、`d % kUbSize == 0`
  - 注意：任一不成立即退化为分块累加，phase1/phase2 的预算公式整体换分支

- **`CalcPremixPhase1Size`** [hc_pre_tiling_arch35.h L184-L219](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L184-L219)
  - 职责：估算 premix 第一阶段（cast + preMix 计算）的 UB 占用
  - 入参/出参：`config`、`mUbSize`、`kUbSize` → `int64_t`
  - 实现：三分支——fusedY（x + hcMult 段 cast + preMix + y）、`isKSplit`（x + preMix + y + cast + yAcc 切片）、其它（x + cast）；末尾统一加 ND2NZ 双缓冲，切 K 再加 rms
  - 注意：yAcc 切片数 `CeilDiv(maxGroupK, kUbSize)`；ND2NZ 用 `RoundUp(mUb, C0_SIZE_HOST=8)`

- **`CalcMBufferPool1Size`** [hc_pre_tiling_arch35.h L221-L229](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L221-L229)
  - 职责：从 UB 扣掉固定开销得到 M 轴可用池
  - 入参/出参：`ubSize`、`maxMRows`、`hcMix`、`hcMult`、`hcMultAlign` → `int64_t`
  - 实现：`ubSize - mmX(maxMRows*hcMixAlign*4) - rms - bases((2+hcMult)*hcMultAlign*4)`，再 `DownAlign(…, 32)`
  - 注意：返回值就是后续所有 `fitsInUb` 的上限；为负说明固定块已撑爆 UB

- **`CalcPremixMPhase2Size`** [hc_pre_tiling_arch35.h L231-L248](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L231-L248)
  - 职责：估算 premix 不切 K 时第二阶段的 UB 占用
  - 入参/出参：`config`、`mUbSize`、`kUbSize`、`rowFactor` → `int64_t`
  - 实现：mixes + post（+ 可选 pre）；非 fused 时再加 x/y（d 对齐到 `BF16_BYTES*C0_SIZE_HOST`）与 preMix
  - 注意：是否 fused 决定要不要再分配 x/y，这是 `rowFactor` 能放大多少的主要分水岭

- **`CalcPremixMKPhase2Size`** [hc_pre_tiling_arch35.h L250-L264](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L250-L264)
  - 职责：估算 premix 切 K 时第二阶段的 UB 占用
  - 入参/出参：`config`、`rowFactor` → `int64_t`
  - 实现：`kBlockNum` 份 mm + rms 双缓冲，再加 mixes + post + bases + 可选 pre
  - 注意：没有 x/y 项（切 K 时 x/y 走跨核累加缓冲）；与 phase1 的变量口径要成对使用

- **`FindMaxFactor`** [hc_pre_tiling_arch35.h L266-L282](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L266-L282)
  - 职责：模板函数；二分出满足 UB 约束的最大因子
  - 入参/出参：`maxFactor` 上界、`fitsInUb` 谓词 → `int64_t`（0 表示无解）
  - 实现：在 `[1, maxFactor]` 二分；`fitsInUb(mid)` 成立则记录并右移，否则左移
  - 注意：要求谓词随因子单调（因子越大越放不下），否则二分结果无意义

- **`TryPremixUbCandidate`** [hc_pre_tiling_arch35.h L284-L314](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L284-L314)
  - 职责：试算一个 kUb 候选并把更优结果写入 ubTiling
  - 入参/出参：`config`、`availableSize`、`kUbSize`、`PremixUbTiling&`、`bestChunkCount&` → `void`
  - 实现：先卡 `kUbSize ∈ (0, kL1Size]`；切 K 时用 `FindMaxFactor` 反解 mUb，否则要求已 fused 且 phase1 放得下；`chunkCount = CeilDiv(maxMRows,mUb) * CeilDiv(maxGroupK,kUb) * vectorBlocks`
  - 注意：择优顺序为 chunk 数更小 → mUb 更大 → kUb 更大；`vectorBlocks` 仅 fused 时按 `REPEAT_SIZE/4` 切

- **`FindFirstDFactorDivisor`** [hc_pre_tiling_arch35.h L316-L330](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L316-L330)
  - 职责：模板函数；二分最小 d 切分份数，使单块 d 能进 UB
  - 入参/出参：`d`、`fitsInUb(rows, dChunk)` → `int64_t` 份数
  - 实现：区间 `[HALF_SPLIT=2, d]`，`fitsInUb(1, CeilDiv(d, mid))` 成立则 `right = mid`，否则 `left = mid + 1`
  - 注意：名字叫 Divisor，实际只按 `CeilDiv` 均分，并不保证真正整除

- **`CalcRowAndDTiling`** [hc_pre_tiling_arch35.h L332-L354](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L332-L354)
  - 职责：模板函数；一次定出 rowFactor / dFactor / dLoop / tailDFactor
  - 入参/出参：`d`、`maxRowFactor`、`fitsInUb`、`RowAndDTiling&` → `bool`
  - 实现：先探 `fitsInUb(1,1)`；能整 D 放下则 `FindMaxFactor` 扩行（dFactor=d）；否则先求切分份数，`dFactor = CeilDiv(d, 份数)`，>32 时 `DownAlign` 到 32；最后补 `dLoop` 与尾块
  - 注意：dFactor 对齐到 32 是为向量长度；尾块要区分整除与非整除两种写法

---

### 2. 类 `HcPreTilingRegbase`（L357-L401）

- **`HcPreTilingRegbase`** [hc_pre_tiling_arch35.h L359-L361](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L359-L361)
  - 职责：构造函数；保存 tiling context 指针
  - 入参/出参：`gert::TilingContext*` → 无
  - 实现：初始化列表只赋 `context_(tilingContext)`，其余成员用类内默认值
  - 注意：`explicit` 修饰；不接管指针生命周期，context 由框架持有

- **`~HcPreTilingRegbase`** [hc_pre_tiling_arch35.h L362](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L362)
  - 职责：析构函数；默认析构，不释放任何资源
  - 入参/出参：无 → 无
  - 实现：`= default`
  - 注意：—

- **`GetPlatformInfo`** [hc_pre_tiling_arch35.h L403-L423](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L403-L423)
  - 职责：取 AIV/AIC 核数、UB 字节数与 SoC 版本
  - 入参/出参：无 → `graphStatus`
  - 实现：有 `platformInfo` 时走 `PlatformAscendC` 的 `GetCoreNumAiv/GetCoreNumAic/GetCoreMemSize(UB)/GetSocVersion`；否则回退 `HcPreCompileInfo`，且 `aicCoreNum_ = aivCoreNum_ = coreNum`
  - 注意：回退分支不写 `socVersion_`；两分支的 `ubSize_` 口径不同，易导致同一 shape 在编译期/运行期结果不一致

- **`GetAttr`** [hc_pre_tiling_arch35.h L425-L442](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L425-L442)
  - 职责：读 4 个属性，缺省填默认值
  - 入参/出参：无 → `graphStatus`
  - 实现：idx0 `hcMult` 默认 4、idx1 `iterTimes` 默认 20、idx2 `hcEps` 默认 1e-6、idx3 `normEps` 默认 1e-6
  - 注意：eps 取的是 `float*` 却赋给 `double` 成员，存在精度截断；属性缺失不报错

- **`GetShapeAttrsInfoInner`** [hc_pre_tiling_arch35.h L444-L497](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L444-L497)
  - 职责：解析输入 shape、做一致性校验、记录可选输入输出标志
  - 入参/出参：无 → `graphStatus`
  - 实现：x 支持 3 维 `(bs,hcMult,d)` 与 4 维 `(b,s,hcMult,d)`（折成 `bs=b*s`）；校验 `HcFn.dim1 == d*hcMult`、`hc_scale.dim0 == 3`、`hc_base.dim0 == hcMix`；`hasPreMix = GetInputShape(4) != nullptr`、`hasPreOut = GetOutputShape(3) != nullptr`；最后调 `GetAttr`
  - 注意：用 `GetInputShape` 判可选输入而非 `GetOptionalInputTensor`（通用版注释已警告占位 shape 恒非空）；且末尾 `GetAttr` 会用属性值覆盖 shape 里解析出的 `hcMult_`，两者不校验一致性

- **`CalcPremixRowTiling`** [hc_pre_tiling_arch35.h L500-L532](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L500-L532)
  - 职责：premix 路径定 `rowFactor`，D 轴恒定不切
  - 入参/出参：`isKSplit`、`CoreRowTiling&`、`RowAndDTiling&`、`PremixUbTiling&` → `graphStatus`
  - 实现：先 `SelectPremixUbTiling`；`availableSize = 切K ? bufferPool0Size : bufferPool1Size`；`maxRowFactor = 切K ? rowOfFormerBlock : config.maxMRows`；用 phase2 预算做 `FindMaxFactor`；最后写 `dLoop=1, dFactor=d, tailDFactor=d`
  - 注意：`rowFactor == 0` 直接失败（UB 放不下哪怕一行）；premix 不走 `CalcRowAndDTiling`

- **`CalcNoPremixRowTiling`** [hc_pre_tiling_arch35.h L534-L558](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L534-L558)
  - 职责：非 premix 路径算 rowFactor 与 dFactor
  - 入参/出参：`isKSplit`、`CoreRowTiling`、`RowAndDTiling&`、`mUbSize`、`bufferPool1Size&` → `graphStatus`
  - 实现：`bufferPool1Size = 切K ? 0 : CalcMBufferPool1Size(ubSize_, mUbSize, …)`；`availableSize = 切K ? ubSize_ : bufferPool1Size`；组 `UbBufferConfig` 后调 `CalcRowAndDTiling(d_, maxRowFactor, fitsInUb, …)`
  - 注意：入参 `mUbSize` 来自调用方的 `mL1Size/2`，不是从 UB 反推；`bufferPool1Size` 靠引用回写给 `SaveCommonTiling`

- **`SaveCommonTiling`** [hc_pre_tiling_arch35.h L560-L601](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L560-L601)
  - 职责：把四条路径的结果统一写进 `HcPreTilingData`
  - 入参/出参：`hasPreMix`、`isKSplit`、`CoreRowTiling`、`RowAndDTiling`、`PremixUbTiling` → `void`
  - 实现：先写 bs/hcMix/hcMult/d/hcMultAlign 与 row*/d* 系列；再按是否切 K 分写 `stage2RowFactor + secondUsedCoreNum` 或 `rowFactor + rowInnerFactor`；`hasPreMix || !isKSplit` 时才写 bufferPool0/1；`kBlockFactor` 取 `cubeBlockDimK`
  - 注意：不写通用版的 prePost/sinkhorn/stage1 字段；未切 K 时 `rowInnerFactor` 与 `rowFactor` 同值

- **`CalcRegbaseCommonTiling`** [hc_pre_tiling_arch35.h L603-L628](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L603-L628)
  - 职责：四条 arch35 路径共用的行/UB 分配主流程
  - 入参/出参：`hasPreMix`、`isKSplit` → `graphStatus`
  - 实现：`CalcCoreRowTiling(bs_, aivCoreNum_)`；预置 `mUbSize=mL1/2`、`kUbSize=kL1/2`、`bufferPool0Size=ubSize_`；按 `hasPreMix` 二选一 → `CompleteCoreRowTiling(rowFactor)` → `SaveCommonTiling`
  - 注意：核分配用 `aivCoreNum_`，而 `PostTiling` 的 BlockDim 用 `aicCoreNum_`，二者不等时正确性依赖 Kernel 模板

- **`CalcRegbaseOpTiling`** [hc_pre_tiling_arch35.h L630-L633](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L630-L633)
  - 职责：非 premix + 不切 K 分支入口
  - 入参/出参：无 → `graphStatus`
  - 实现：直接 `return CalcRegbaseCommonTiling(false, false)`
  - 注意：纯转发，不做任何额外校验

- **`CalcMKSplitCorePart2Tiling`** [hc_pre_tiling_arch35.h L635-L638](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L635-L638)
  - 职责：非 premix + 切 K 分支入口
  - 入参/出参：无 → `graphStatus`
  - 实现：`return CalcRegbaseCommonTiling(false, true)`
  - 注意：与上一个仅布尔位不同，正好对应 1000/1001 两个 key

- **`InitPremixUbConfig`** [hc_pre_tiling_arch35.h L640-L674](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L640-L674)
  - 职责：填 premix UB 估算所需的配置
  - 入参/出参：`isKSplit`、`PremixUbConfig&` → `graphStatus`
  - 实现：填 hcMix/hcMult/hcMultAlign/kBlockNum/`maxMRows=mL1/2`/d/kL1Size/multCoreSplitKSize/hasPreMix/hasPreOut；切 K 时校验 `multCoreSplitKSize % hcMult == 0` 并取 `dPerCore = min(d, multCoreSplitKSize/hcMult)`；`maxGroupK = min(kL1Size, dPerCore)`
  - 注意：`hasPreMix/hasPreOut` 在此重新查 shape，与 `tilingData_` 里的 flag 是两套来源

- **`SelectPremixUbTiling`** [hc_pre_tiling_arch35.h L676-L710](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L676-L710)
  - 职责：枚举 kUb 候选，选出 premix 的 mUb/kUb 组合
  - 入参/出参：`isKSplit`、`PremixUbConfig&`、`PremixUbTiling&` → `graphStatus`
  - 实现：先 `InitPremixUbConfig`；`bufferPool0Size=ubSize_`、`bufferPool1Size=切K?0:CalcMBufferPool1Size(...)`；候选 `kUb = {halfKUb, fullKUb}`（按 C0=8 对齐）逐个 `TryPremixUbCandidate`；`mUbSize==0` 时报 UB 不足
  - 注意：只试两个候选（maxGroupK 的 1/2 与全量），不扫其它切法；两候选相等时靠判断去重跳过

- **`CalcPremixMSplitTiling`** [hc_pre_tiling_arch35.h L712-L715](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L712-L715)
  - 职责：premix + 不切 K 分支入口
  - 入参/出参：无 → `graphStatus`
  - 实现：`return CalcRegbaseCommonTiling(true, false)`
  - 注意：对应 TilingKey 1003

- **`CalcPremixMKSplitTiling`** [hc_pre_tiling_arch35.h L717-L720](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L717-L720)
  - 职责：premix + 切 K 分支入口
  - 入参/出参：无 → `graphStatus`
  - 实现：`return CalcRegbaseCommonTiling(true, true)`
  - 注意：对应 TilingKey 1002，与 1000 同属需要跨核 workspace 的 K 切分分支

- **`CalcOpTiling`** [hc_pre_tiling_arch35.h L722-L765](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L722-L765)
  - 职责：定 Cube 侧 M/K 切分、L1 预算与 TilingKey 并分发
  - 入参/出参：无 → `graphStatus`
  - 实现：`k = hcMult*d`；`mDimNum = min(aicCoreNum, CeilDiv(bs, 256))`、`singleCoreM = RoundUp(CeilDiv(bs,mDimNum), 16)`、`kDimNum = aicCoreNum/mDimNum`、`splitKSize = RoundUp(CeilDiv(k,kDimNum), 256)`、`kBlockNum = CeilDiv(k, splitKSize)`；`mL1 = min(256, singleCoreM)`、`kL1 = min(A_L1_SIZE/mL1, 1024)/128*128`、`cvLoopKSize = 1024`；`isKSplit = (kDimNum != 1)`；不切 K 且有 premix 时令 `mDimNum = aicCoreNum` 并用 `SelectPremixML1Size` 重算 mL1/kL1；`key = (1002|1000) + (isKSplit?0:1)`；最后按 4 分支分发
  - 注意：没有通用版那段"把 mDimNum 上抬到 aicCoreNum 的因子"的修正，`aicCoreNum % mDimNum != 0` 时 `kDimNum` 会被截断、尾核闲置

- **`DoOpTiling`** [hc_pre_tiling_arch35.h L768-L791](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L768-L791)
  - 职责：tiling 总编排，五步串行
  - 入参/出参：无 → `graphStatus`
  - 实现：`GetPlatformInfo → GetShapeAttrsInfoInner → CalcOpTiling → GetWorkspaceSize → PostTiling`，任一步 `GRAPH_FAILED` 即短路返回
  - 注意：顺序不可换，`GetWorkspaceSize` 依赖 `CalcOpTiling` 定出的 `tilingKey_` 与各类 size

- **`GetWorkspaceSize`** [hc_pre_tiling_arch35.h L793-L800](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L793-L800)
  - 职责：只为两个 K 切分模板申请跨核累加 workspace
  - 入参/出参：无 → `graphStatus`（写 `workspaceSize_`）
  - 实现：`key == 1000 || key == 1002` 时 `kBlockFactor*bs*hcMix*4 + kBlockFactor*bs*4 + 16MB`；其余 key 保持 0
  - 注意：两项都不做 `RoundUp` 对齐；也没有通用版 `max(required, 208MB)` 的兜底，M-only 分支 workspace=0

- **`PostTiling`** [hc_pre_tiling_arch35.h L802-L811](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L802-L811)
  - 职责：把 key、BlockDim、workspace 与 tilingData 下发框架
  - 入参/出参：无 → `graphStatus`
  - 实现：`SetTilingKey(tilingKey_)`、`SetBlockDim(aicCoreNum_)`、`GetWorkspaceSizes(1)[0] = workspaceSize_`、`SaveToBuffer(...GetData(), ...GetCapacity())` 后 `SetDataSize(GetDataSize())`
  - 注意：BlockDim 固定取 AIC 核数，且没有通用版 MPMD 用的 `SetScheduleMode(1)`；tilingData 超出 capacity 会被静默截断

---

### 3. arch35 与通用版（`hc_pre_tiling.cpp`）算法逐项差异

| 维度 | 通用版 `hc_pre_tiling.cpp` | arch35 `hc_pre_tiling_arch35.h` |
|---|---|---|
| 切分维度 | Cube 侧 M×K 二维；Vector 侧再分 Part2（rowFactor×dFactor）、prePost 行批、sinkhorn 行批三套独立 tiling | 统一 M 行切 + D 切（`RowAndDTiling`）；premix 额外做 mUb×kUb 两级块切 |
| K 切分 | `kDimNum = aicCoreNum/mDimNum`，并上抬 mDimNum 到 aicCoreNum 的因子保证整除；`splitKSize` RoundUp 到 256 | 同公式但**无整除修正**，`kDimNum` 向下截断即用；premix 切 K 时另需 `multCoreSplitKSize % hcMult == 0` |
| L1 预算 | `mL1 = min(256, singleCoreM)`，`kL1 = min(32K/mL1,1024)↓128`，无 premix 特化 | 新增 `SelectPremixML1Size`：不切 K 且有 premix 时按最小 cost 选 mL1 并 cap 208，再重算 kL1 |
| 核分配 | MPMD 时把 AIV 拆成 prePostCoreNum / sinkhornCoreNum，按行工作量估延迟择优；`bs ≤ 32` 走 ORIGINAL | 无 MPMD：固定 `usedCoreNum = min(CeilDiv(bs, rowOfFormerBlock), coreNum)`；但 `PostTiling` 的 BlockDim 仍取 AIC 核数 |
| UB 预算 | 逐 buffer 手写累加（mix0/1/2、squareSum、brcb、mask…），超 UB 后回溯缩小 dFactor | 抽象成 bufferPool0/1 + 多个 `Calc*Size` 函数，用 `FindMaxFactor` 二分搜索 |
| workspace | 三段精确估算后 `max(required, 208MB)` | 仅 K 切分两个 key 申请：`kBlockFactor*bs*(hcMix+1)*4 + 16MB`；其余为 0 |
| TilingKey | 0 = MPMD、1 = ORIGINAL、2 = PREMIX | 1000/1001（无 premix：切K/不切K）、1002/1003（premix：切K/不切K） |
| Part2 阶段 | 有 prePost/sinkhorn 完整字段与 `CalcSinkhornTiling`、`CalcVectorCoreNum` | 无这两套，全部收敛为 rowFactor/dFactor + mUb/kUb |

---

### 4. `DoOpTiling()` 内部调用时间线与关键产出

```
DoOpTiling()
├─ GetPlatformInfo()            → aivCoreNum_, aicCoreNum_, ubSize_, socVersion_
├─ GetShapeAttrsInfoInner()     → bs_, hcMix_, hcMult_, d_, iterTimes_, hcEps_, normEps_,
│                                 tilingData_.hasPreMix / hasPreOut
├─ CalcOpTiling()               → k, cubeBlockDimM, cubeBlockDimK, multCoreSplitMSize,
│   │                             multCoreSplitKSize, mL1Size, kL1Size, cvLoopKSize,
│   │                             isKSplit, tilingKey_
│   └─ 四选一分发
│       ├─ CalcRegbaseOpTiling()        (hasPreMix=false, isKSplit=false)
│       ├─ CalcMKSplitCorePart2Tiling() (false, true)
│       ├─ CalcPremixMSplitTiling()     (true, false)
│       └─ CalcPremixMKSplitTiling()    (true, true)
│            └─ CalcRegbaseCommonTiling(hasPreMix, isKSplit)
│                ├─ CalcCoreRowTiling          → rowOfFormerBlock, usedCoreNum, rowOfTailBlock
│                ├─ [premix] SelectPremixUbTiling
│                │    ├─ InitPremixUbConfig    → maxMRows, maxGroupK, kL1Size, multCoreSplitKSize
│                │    └─ TryPremixUbCandidate  → mUbSize, kUbSize, phase1Size, bufferPool1Size
│                │   CalcPremixRowTiling       → rowFactor, dLoop=1, dFactor=d
│                ├─ [no premix] CalcNoPremixRowTiling
│                │                             → bufferPool1Size, rowFactor, dFactor,
│                │                               dLoop, tailDFactor
│                ├─ CompleteCoreRowTiling      → rowLoopOf*, tailRowFactorOf*
│                └─ SaveCommonTiling           → tilingData_ 全部字段落库
├─ GetWorkspaceSize()           → workspaceSize_（仅 key 1000/1002 非零）
└─ PostTiling()                 → SetTilingKey, SetBlockDim, workspaces[0], tilingData 序列化
```


---

## 13. 函数级详解：通用版计算原语（`hc_pre_base.h`）

> 目标读者：完全没写过 Ascend C 的人。
> 讲解对象（只读，未修改仓库）：`c:\Users\89517\Desktop\github\vllm-npu\vllm-ascend\csrc\moe\hc_pre\op_kernel\hc_pre_base.h`
> 全部行号均指该文件。共 26 个函数（含 1 个重载），按文件中出现顺序排列。

### 0. 先补 6 个前置概念（不然后面每个函数都看不懂）

#### 0.1 UB 与 GM

AI Core 上的向量（Vector/AIV）单元不能直接算显存。数据必须先从 **GM**（Global Memory，就是 HBM 显存）
用 `DataCopy`/`DataCopyPad` 搬进 **UB**（Unified Buffer，片上向量缓存，几 MB 级），算完再搬回去。
本文件里 `GlobalTensor<T>` 一律指 GM 上的指针，`LocalTensor<T>` 一律指 UB 上的指针，二者不能互换传参。

#### 0.2 Block（32B）与 Repeat（256B）

- 1 **Block** = 32 字节。fp32 一个 Block 装 8 个元素，fp16/bf16 装 16 个。
- 1 **Repeat** = 256 字节 = 8 个 Block。一条向量指令一次迭代（一个 repeat）最多处理 256B。
- 所有"对齐"都是把长度补到 Block 的倍数；文件里的 `BLOCK_SIZE=32`、`REPEAT_SIZE=256`、`ONE_REPEAT_BLOCK_NUMS=8` 就是这三个事实。

#### 0.3 两级 API 与 BinaryRepeatParams

- **二级（count 级）API**：`Mul(dst, src0, src1, count)`——只给元素总数，硬件/编译器自己切 repeat。
  它的前提是数据在 UB 上**连续紧凑**，没法表达"行间留空"。本文件里 `curColNum <= elemInOneBlock`、以及各种尾块都用它。
- **零级（repeat 级）API**：要显式给 `repeatTimes` 和 `BinaryRepeatParams`，6 个字段全是**以 Block 为单位**的步长：
  - `dstBlkStride` / `src0BlkStride` / `src1BlkStride`：一个 repeat 内部，相邻 Block 之间跳几个 Block。
  - `dstRepStride` / `src0RepStride` / `src1RepStride`：相邻两个 repeat 之间跳几个 Block。
  - 取值为 **0 表示"不动"**，这正是"广播"的实现方式（同一块数据被反复读）。
- 地址公式：某 repeat `r` 里第 `b` 个 Block 的偏移 = `r * RepStride + b * BlkStride`（Block 数）。

#### 0.4 本文件三种"广播"长得完全不同

1. **Brcb 预广播**：`Brcb` 每个 repeat 从源取 8 个元素，把每个元素复制满一个 32B Block 写出。
   结果是"每个值占满一个 Block"的稀疏布局，不能当成紧凑数组。
2. **BlkStride=0 原地广播**：不复制数据，让 `src1BlkStride=0` 反复读同一块（见 `AddBAFirstDimBrcInline`）。
3. **RepStride=0 跨行广播**：行方向开 repeat 时让 `src1RepStride=0`，同一行权重对所有行复用。
   看懂这三者的前提差异，才知道为什么要区分 `needBrc` 参数。

#### 0.5 同步：PipeBarrier 与队列事件

同一条 PIPE_V（向量流水）上的指令默认是流水化的，**对同一块 UB 读改写时**必须 `PipeBarrier<PIPE_V>()`，
否则后一条指令可能在写回前就读到旧值。跨流水（MTE2 搬入 / V 计算 / MTE3 搬出）由 Queue 的 EnQue/DeQue 事件保证，
所以本文件里很多函数把 `syncAfter` 设成 `false`，把同步点合并到调用方，减少流水排空（见 `ProcessYHc4`）。

#### 0.6 DataCopyPad 的三个步长单位（最容易错的地方）

`DataCopyExtParams` 里：`blockLen` 是**字节**，`srcStride` 是**字节**，`dstStride` 是**Block 数**（三者单位都不同！）。
本文件的 `CopyIn`/`CopyOut` 在包装它时对参数做了换算，换算方向见各函数的"注意"。
另外 `dstStride` 只能是整 Block：UB 行距若比有效行多出 4 个 float（16 字节），这个空隙无法用一次 `DataCopyPad` 表达，
整数除法会被抹成 0——本文件所有 UB 行距都取 `RoundUp` 后的整 Block，正是为了绕开这条限制。

#### 0.7 这些原语在本算子里的组合关系（先有个地图）

- `ProcessPre`：`MulABLastDimBrcInline`(乘 rsqrt) → `Muls`(乘 scale) → `AddBAFirstDimBrcInline`(加 base) → `SigmoidPerf` → `Adds`(加 eps)。
- `ProcessY` / `ProcessYHc4`：`CastTwoDim`(bf16→fp32) → 乘权重（`Mul`+`MulAddDst` 或 `MulABLastDimBrcInline`）→ `ReduceSumARAPerf`(沿 hc_mult 归约) → `CastTwoDim`(fp32→bf16)。
- `ProcessPost`：与 `ProcessPre` 同构，只是 `needBrc=false`（复用上一步已广播的 rsqrt）且结尾 `Muls(*2)` 而不是 `Adds(eps)`。
- `SoftmaxFP32Perf`：`LastDimReduceMaxPerf` → `SubABLastDimBrcInline` → `Exp` → `LastDimReduceSumPerf` → `DivABLastDimBrcInline` → `Adds(eps)`。
- Sinkhorn 迭代：`ReduceSum4x4Perf`（4×4 行和）+ `DivABABrcInline`（沿 dim1 归一）。
- RMSNorm 平方和对角线：`SetGatherMaskPattern` + `GatherMaskByDiagonal`。
- 所有进出 GM 的动作：`CopyIn` / `CopyInWithOuterFor`（3D、4D 两个重载）/ `CopyOut`。
- 尺寸与对齐：`CeilDiv` / `CeilAlign` / `RoundUp<T>`，它们被上面几乎所有函数调用。

#### 0.8 一张"输出布局"对照表（后面反复用到）

| 产出者 | 在 UB 上的存放形态 | 谁能直接消费 |
| --- | --- | --- |
| `Brcb` | 每个值复制满 1 个 32B Block（fp32 即 8 个相同 float） | `Mul/Sub/Div` 里 `src1BlkStride=0` 的形态 |
| `LastDimReduceMaxPerf` / `LastDimReduceSumPerf` | 每行 1 个值，落在 `out[i*8]`，其余是填充 | `Brcb`（它正好按 Block 首元素取数） |
| `ReduceSumARAPerf` | 紧凑的两维数组，行距 `dim2Align` | `CastTwoDim`、`DataCopy`、count 级 API |
| `GatherMaskByDiagonal` | 紧凑 `dim0` 个 float | count 级 API / Cast |
| `CastTwoDim` | 行距为目标类型 `RoundUp` 后的值 | 上面的广播 / 归约原语 |

一句话记法："**Brcb 与 Reduce 排的是 Block，其余都是紧凑行**"。

#### 0.9 手算一次末维广播乘（`rows=8, cols=64, fp32`）

1. `Brcb`：`repeatTimes = CeilDiv(8, 8) = 1`，一次 repeat 吃掉 8 个权重、产出 8 个 Block；
   `tmpBuffer` 第 `i` 个 Block（`tmpBuffer[i*8]`）里是 8 个相同的 `w[i]`。
2. 常量：`elemInOneBlock=8`、`elemInOneRepeat=64`、`colAlign=RoundUp(64)=64`、
   `numRepeatPerLine = 64/64 = 1`、`remain = 0`、`dstRepStridePerLine = 64/8 = 8`。
3. 策略选择：`8 > 255` 不成立、`rows(8) < numRepeatPerLine(1)` 也不成立 → 走 **Row 方向**：
   `Mul(output[0], input0[0], tmpBuffer, mask=64, repeatTimes=8, {dstRep=8, src0Rep=8, src1Rep=1})`。
4. 地址自检：repeat `r` 的 dst 首个 Block = `r*8`，即 `output[r*64]`，正好是第 r 行行首；
   src1 每个 repeat 前进 1 个 Block，即 `w[r]`——这就是"逐行换权重"的关键，`src1RepStride` 必须是 1。
5. 若 `cols=100`：`numRepeatPerLine=1`、`remain=36`，先按上面的方式算前 64 列，再单独补一次 36 列的
   `Mul`，起始偏移是 `numRepeatPerLine*elemInOneRepeat = 64`（按元素，不是按行对齐值）。

#### 0.10 本文件出现的 Ascend C API 速查

| API | 类别 | 语义要点（一句话） | 本文件用在哪 |
| --- | --- | --- | --- |
| `Duplicate` | 造数 | 往 UB 写同一标量值 `count` 个 | 掩码行、常量 1.0 |
| `Brcb` | 广播 | 每 repeat 取 8 个源元素，各复制满 1 个 Block | 所有 `needBrc=true` 的乘法 / 除法 |
| `Mul` / `Add` / `Sub` / `Div` | 二元逐元素 | 既可 count 级，也可 repeat 级（配 `BinaryRepeatParams`） | 广播运算、累加、归一 |
| `MulAddDst` | 融合乘加 | `dst = dst + src0 * src1`，省一次读改写 | `ProcessYHc4` 的 Pass1..3 |
| `Muls` / `Adds` | 标量逐元素 | 与标量做乘 / 加 | `-x`、`*scale`、`+1`、`+eps`、`*2` |
| `Exp` | 超越函数 | 逐元素 `e^x` | sigmoid 分母、softmax 分子 |
| `Cast` | 类型转换 | 带 `RoundMode`（本文件用 `CAST_NONE` / `CAST_RINT`） | `CastTwoDim` |
| `WholeReduceMax` / `WholeReduceSum` | 归约 | 整 repeat 归约成 1 个值，结果按 Block 存放 | 末维 max / sum |
| `GatherMask` | 归约/收集 | 按位掩码挑元素，`reduceMode=true` 时稀疏输出 | 平方和对角线 |
| `DataCopy` | 搬运 | 按 `DataCopyParams`（Block 粒度，无 padding） | 归约首切片 |
| `DataCopyPad` | 搬运 | 按 `DataCopyExtParams`（字节粒度，支持非对齐） | `CopyIn` / `CopyOut` |
| `PipeBarrier<PIPE_V>` | 同步 | 排空向量流水，保证同 PIPE 的读改写顺序 | 几乎每个原语尾部 |

#### 0.11 命名与常量的读法（省一半查表时间）

- 后缀 `Perf`：性能优化版，行为与朴素版一致但减少了屏障或中间缓冲（`SigmoidPerf`、`ReduceSumARAPerf`）。
- 后缀 `Inline`：调用点内联展开，避免函数调用开销（`...BrcInline`）。
- `AB` / `BA` 是"末维/首维"的简写：`ABLastDim` = 沿列广播（每行都要完整一行权重）；
  `BAFirstDim` = 沿行广播（同一行对多行复用）。
- 模板参 `T0` / `T1`：`T0` 是 Cast 的目的类型、`T1` 是源类型，别写反。
- 常量含义（[hc_pre_base.h L26-L43](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L26-L43)）：`BLOCK_SIZE=32`、`REPEAT_SIZE=256`、`ONE_REPEAT_BLOCK_NUMS=8`、
  `MAX_REPEAT_STRIDE=255`（`BinaryRepeatParams` 的步长是 8 位无符号，超过 255 就要改按行循环）、
  `WORKSPACE_ALIGN_SIZE=512`（GM 地址对齐）、`SQUARE_SUM_SIZE=16`（NZ 16×16 对角块）。

### 1. 函数逐个讲解（按文件出现顺序）

- **`CeilDiv`** [hc_pre_base.h L48-L54](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L48-L54)
  - 职责：向上取整除法，返回 `(a + b - 1) / b`。
  - 入参/出参：`a` 被除数（长度/元素数，标量）、`b` 粒度（标量）；返回值为整型商，不返回余数。
  - 实现：①纯标量整数运算，不调用任何 Ascend C API；②`b == 0` 时直接返回 `a`，避免除零；③负数场景未定义，本文件不涉及。
  - 注意：结果既可能是"块数"也可能是"元素数"，取决于调用点传的 `b`（8/64/255 各有用例），别跨函数套用同一个含义。

- **`CeilAlign`** [hc_pre_base.h L57-L60](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L57-L60)
  - 职责：把 `a` 向上对齐到 `b` 的整数倍。
  - 入参/出参：`a` 任意长度、`b` 对齐粒度（均为标量）；返回 ≥ `a` 的最小对齐值。
  - 实现：①内部只调 `CeilDiv(a, b) * b` 一行；②不校验 `b` 是否为 2 的幂。
  - 注意：`b` 不是 2 的幂时数学上仍然"对齐"，但硬件要求的是 32B 的倍数，地址类对齐不要用它替代 `RoundUp`。

- **`RoundUp`** [hc_pre_base.h L63-L68](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L63-L68)
  - 职责：按类型 `T` 的元素个数，把 `num` 对齐到一个 32B Block。
  - 入参/出参：`num` 元素个数（标量）；模板参 `T` 决定 `32 / sizeof(T)`；返回对齐后的元素个数。
  - 实现：①`elemNum = BLOCK_SIZE / sizeof(T)`（fp32=8，fp16/bf16=16）；②交给 `CeilAlign(num, elemNum)`。
  - 注意：单位是"元素"不是字节；本文件里 fp32 张量的行 stride 一律用它算，而 `CastTwoDim` 因为源/目的类型不同会同时算两个对齐值，混用会错位。

- **`SetGatherMaskPattern`** [hc_pre_base.h L75-L82](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L75-L82)
  - 职责：预生成"循环对角"位掩码，供 `GatherMask` 抽取方阵对角线。
  - 入参/出参：`maskPattern` 是 uint32 掩码缓冲（目的），`patternRows` 要生成多少行（标量）。
  - 实现：①第 `i` 行的基值 `base = 1 << (i % 16)`，即把"第 i 个候选元素"标出来；②`Duplicate(maskPattern[i * 8], base, 8)` 把同一个 uint32 写满 8 个，正好填满 1 个 32B Block；③结尾 `PipeBarrier<PIPE_V>()`。
  - 注意：行 i 固定占 1 个 Block（8 个 uint32），所以缓冲需 `patternRows * 8` 个元素且 32B 对齐；调用方要按 `相位偏移 + 15` 多备 15 行，否则 `GatherMask` 的靠后相位会读到未初始化掩码。

- **`GatherMaskByDiagonal`** [hc_pre_base.h L90-L100](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L90-L100)
  - 职责：从 NZ 格式的 16×16 平方和块中，一次 `GatherMask` 抽出一条对角线（每行挑 1 个元素）。
  - 入参/出参：`output` 目的（dim0 个 float）、`input` 源（NZ 平方和）、`maskPattern` 掩码、`dim0` 行数、`diagonalOffset` 当前 16×16 块内的对角相位；`rsvdCnt` 是出参但被丢弃。
  - 实现：①`patternIdx = diagonalOffset % 16` 选起始掩码块；②`GatherMask(output, input, maskPattern[patternIdx * 8], true, 16, {1, dim0, 2, 1}, rsvdCnt)`：`src0BlockStride=1`、`repeatTimes=dim0`、`src0RepeatStride=2`、`src1RepeatStride=1`；③第 `r` 个 repeat 选中第 `(r + phase) % 16` 列，于是结果正好是对角元素；④结尾 `PipeBarrier<PIPE_V>()`。
  - 注意：`src0RepeatStride=2` 是硬编码"每行 16 个 fp32 = 2 个 Block"，源布局不是 NZ 16×16 就会错位；输出是紧凑 dim0 个 float，和 `Brcb` 期望的"每值一个 Block"布局不同，不能直接喂给 `needBrc=true` 的接口。

- **`MulABLastDimBrcInline`** [hc_pre_base.h L111-L193](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L111-L193)
  - 职责：`out[i,j] = in0[i,j] * brc(in1[j])`，末维（列）广播乘法。
  - 入参/出参：`output` 目的、`input0` 源矩阵、`input1` 一维权重（只有一行）、`tmpBuffer` 存 Brcb 广播结果；`curRowNum`/`curColNum` 为标量；模板参 `needBrc` 决定是否重新广播。
  - 实现：①`Brcb(tmpBuffer, input1, CeilDiv(rows, 8), {1, 8})`——每个 repeat 取 8 个权重、各铺满 1 个 Block；②列 ≤ 1 Block 时用二级 `Mul`，count = `rows * colAlign`；③列 > 1 Block 时按"完整 Repeat 段 + 尾段"两段处理，并按 `dstRepStridePerLine > 255 或 rows < numRepeatPerLine` 在两种策略间选：**按行循环（Col 方向开 repeat，src1 固定 / `src1RepStride=0`）** 或 **`repeatTimes=curRowNum`（Row 方向开 repeat，`src1RepStride=1` 逐行换一个权重 Block）**；④两种策略都令 `src1BlkStride=0`，让一个权重 Block 在整行 8 个 Block 内复用。
  - 注意：`tmpBuffer` 必须按 Brcb 的行数补齐到 `CeilDiv(rows,8)*8` 行（否则尾 repeat 越界写）；尾段起始偏移是 `numRepeatPerLine * elemInOneRepeat`（不是按行对齐的 `i * colAlign`）；`needBrc=false` 时调用方传入的必须是 Brcb 产出的 block 布局，传原始一维数组会静默算错。

- **`SubABLastDimBrcInline`** [hc_pre_base.h L198-L273](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L198-L273)
  - 职责：`out[i,j] = in0[i,j] - brc(in1[j])`，末维广播减法。
  - 入参/出参：与 `MulABLastDimBrcInline` 完全一致（目的、源矩阵、一行被减数、Brcb 缓存、行/列标量）。
  - 实现：①`Brcb` 预广播（`needBrc` 时）；②列 ≤ 1 Block 走二级 `Sub`；③否则同样两段 + 两种 repeat 策略；④唯一区别是把 `Mul` 换成 `Sub`，其余 stride 参数逐字相同。
  - 注意：`input1` 通常直接来自 `LastDimReduceMaxPerf` 的行最大值，而该归约把结果"一个占一个 Block"存放，`Brcb` 恰好按 Block 首元素取数——若换成手写紧凑数组就不成立；三份拷贝（Mul/Sub/Div）改一处必须同步改三处。

- **`DivABLastDimBrcInline`** [hc_pre_base.h L279-L354](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L279-L354)
  - 职责：`out[i,j] = in0[i,j] / brc(in1[j])`，末维广播除法。
  - 入参/出参：同 Mul/Sub 版本；`input1` 是每行共用的分母向量。
  - 实现：①`Brcb` 预广播（`needBrc` 时）；②列 ≤ 1 Block 用二级 `Div`；③列 > 1 Block 按完整段 + 尾段，`dstRepStridePerLine > 255` 时按行循环；④其余 stride 与 Mul 版本一致。
  - 注意：除数来自 `LastDimReduceSumPerf`，必须保证非 0（Softmax 末尾已 `+eps`）；本文件没有除法保护，分母为 0 会得到 inf/NaN 并污染后续 `Adds`。

- **`AddBAFirstDimBrcInline`** [hc_pre_base.h L363-L427](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L363-L427)
  - 职责：`out[i,j] = in0[i,j] + brc(in1[j])`，首维（行）广播加法。
  - 入参/出参：`output` 目的、`input0` 源矩阵、`input1` **紧凑一维向量**（不需要 Brcb 缓存，也没有 tmp 参数）；`curRowNum`/`curColNum` 标量。
  - 实现：①不做 Brcb，直接令 `src1BlkStride=1`：一个 repeat 内 src1 的 8 个 Block 依次是同一行的 8 个连续 Block；②行方向策略再令 `src1RepStride=0`，于是同一行对所有行复用；③同样分完整段与尾段，尾段在 `dstRepStridePerLine > 255` 时按行循环。
  - 注意：**该函数尾段且在 `dstRepStridePerLine > 255` 的分支里，循环 `curRowNum` 次却没有加 `i * curColNumAlign` 的行偏移**（对比 `Sub`/`Div` 同名分支都有），等于只反复重算第 0 行尾块、其余行尾块未处理——落到该配置时结果是错的；另外 `input1` 必须按 Block 补齐，否则最后一块会读越界。

- **`CalcDenominator`** [hc_pre_base.h L434-L444](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L434-L444)
  - 职责：算出 sigmoid 的分母 `1 + exp(-x)`。
  - 入参/出参：`output` 既作目的又作中间结果、`input` 源、`calCount` 元素数（标量）。
  - 实现：①`Muls(output, input, -1.0, calCount)` → `-x`；②`Exp(output, output, calCount)` → `exp(-x)`（原地）；③`Adds(output, output, 1.0, calCount)` → `1 + exp(-x)`；④三步之间各一次 `PipeBarrier<PIPE_V>()`，因为都是对同一块读改写。
  - 注意：三次都是 count 级整段 API，`calCount` 不必是 Repeat 的倍数；但它只产出分母，名字容易让人以为就是 sigmoid。

- **`SigmoidPerf`** [hc_pre_base.h L450-L476](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L450-L476)
  - 职责：`sigmoid(x) = 1 / (1 + exp(-x))`，用一个常量 Block 当被除数来省掉逐元素造数。
  - 入参/出参：`output` 目的、`input` 源、`tmpBuffer` 只存标量 1.0、`calCount` 元素数。
  - 实现：①`Duplicate(tmpBuffer, 1.0, elemInOneBlock)` 只写 1 个 Block（8 个 float）；②`CalcDenominator` 得到分母写进 `output`；③`Div(output, tmpBuffer, output, 64, numRepeatPerLine, instrParams)`，其中 `src0BlkStride=0 / src0RepStride=0` 让这 8 个 1.0 被所有 repeat 复用；④余数 `numRemainPerLine` 用 `repeatTimes=1` 再补一次 `Div(rem, 1, instrParams)`。
  - 注意：`tmpBuffer` 只需 32B 但必须 32B 对齐；函数头已声明"暂不处理 repeat 超限"，`calCount > 255*64` 会传非法 `repeatTimes`；尾段复用同一 `instrParams`，靠 `src0RepStride=0` 才能让常量继续生效，改参数时要连带检查尾段。

- **`ProcessPre`** [hc_pre_base.h L489-L502](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L489-L502)
  - 职责：`pre = sigmoid(mix * rsqrt * scale + base) + eps`。
  - 入参/出参：`preLocal` 目的；`mixLocal` 既是源又被原地改写（中间结果）；`rsqrtLocal`（沿行广播的 RMSNorm 系数）、`hcBaseLocal`（沿行广播的 base）；两个 tmp；`scale`/`eps` 标量。
  - 实现：①`MulABLastDimBrcInline<float,true>` 乘 rsqrt；②`Muls(mix, mix, scale, rows * colAlign)`；③`AddBAFirstDimBrcInline` 加 base；④`SigmoidPerf(pre, mix, tmpBuffer1, ...)`；⑤`Adds(pre, pre, eps, ...)`。
  - 注意：`mixLocal` 被覆盖，调用方若要保留原始 mix 必须自留副本；显式 barrier 只在 `Muls` 之后，其余依赖各原语自带的尾部 `PipeBarrier`，改动原语会破坏这里的正确性。

- **`ReduceSumARAPerf`** [hc_pre_base.h L510-L552](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L510-L552)
  - 职责：`out[d0,d2] = Σ_j in[d0,j,d2]`，3D 张量沿中间轴（K-split 核数）求和。
  - 入参/出参：`output` 目的（dim0×dim2）、`input` 源（dim0×dim1×dim2，按 `dim2Align` 补齐）、`dim0`/`dim1`/`dim2` 标量。
  - 实现：①先用一次 `DataCopy` 把 `j=0` 的切片搬成初值：`blockCount=dim0`、`blockLen=dim2Align/8` 个 Block、`srcStride=(dim1-1)*dim2Align/8` 个 Block（跳过同 batch 的其余切片）、`dstStride=0`（目的逐 burst 紧凑连排）；②对每个 batch、每个 `j≥1` 用 `Add(repeatTimes=dim2/64, ...)` 累加，余数单独一次 `Add(rem, 1, ...)`；③每加完一层 `PipeBarrier<PIPE_V>()`，因为是对 `output` 的读改写。
  - 注意：`dstStride=0` 依赖 `DataCopy` 的 burst 连排，所以 `output` 必须是紧凑的 `dim2Align` 行距；输入行距必须是补齐后的 `dim2Align` 而不是 `dim2`；`dim2 < 8` 会让 `blockLen` 变成 0（4×4 场景改用 `ReduceSum4x4Perf`）。

- **`ReduceSum4x4Perf`** [hc_pre_base.h L556-L570](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L556-L570)
  - 职责：Sinkhorn 专用，把 4×4 的每行 4 个元素按左到右顺序求和。
  - 入参/出参：`output`（dim0×8，前 4 个有效）、`input`（dim0×4×8）、`dim0` 标量。
  - 实现：①`dim2` 硬编码 4，`dim2Align = RoundUp<float>(4) = 8`；②第一次 `Add(dst, in+0, in+1, 4)` 直接写目的（省掉首行 `DataCopy`）；③再 `Add(dst, dst, in+2*align, 4)`、`Add(dst, dst, in+3*align, 4)`；④末尾只 `PipeBarrier<PIPE_V>()` 一次。
  - 注意：省掉了每步累加的 barrier，依赖同一 dst 上指令按发射顺序完成读改写，只在 4 个元素这种小块上被认为安全；`input` 的行距必须是 8 个 float，按 4 算会读串行。

- **`CastTwoDim`** [hc_pre_base.h L580-L613](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L580-L613)
  - 职责：二维张量 `T1 → T0` 类型转换（典型用法 bf16→fp32、fp32→bf16）。
  - 入参/出参：`output`（T0 目的）、`input`（T1 源）、`dim0` 行数、`dim1` 列数、`syncAfter` 是否补尾部 `PipeBarrier<PIPE_V>()`。
  - 实现：①先分别算 `dim1AlignT0 = RoundUp<T0>(dim1)` 与 `dim1AlignT1 = RoundUp<T1>(dim1)`；②当两者都等于 `dim1` 时，用一次 `Cast(output, input, mode, dim0*dim1)` 完成整块转换（快路径）；③否则逐行 `Cast`，dst 用 T0 的行距、src 用 T1 的行距；④`bf16→fp32` 用 `CAST_NONE`（无损扩展），其他组合用 `CAST_RINT`（就近偶数舍入）。
  - 注意：快路径判据是"**两种类型都对齐**"，例如 `dim1=8` 且源是 bf16 时 8 个 bf16 只有 16B，不算对齐，必须走逐行路径；`syncAfter=false` 是性能手段，调用方必须用同一流水的后续指令屏障或 `EnQue` 的 V_MTE3 事件兜底。

- **`ProcessYHc4`** [hc_pre_base.h L625-L694](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L625-L694)
  - 职责：`hc_mult = 4` 特化，`y = Σ_{k<4} w_k * x_k`，用 `Mul` + `MulAddDst` 一次成型。
  - 入参/出参：`yLocal` 目的（T，非 fp32）、`xLocal` 源、`mix01Local` 4 路权重（每 batch 4 个）、`hcBrcbLocal1` 广播缓存、`xCastLocal`/`yCastLocal` fp32 中转、`dim0`/`dim2` 标量、`needBrc`。
  - 实现：①`CastTwoDim(xCastLocal, xLocal, dim0*4, dim2, false)` 转换但不立刻同步；②`Brcb(hcBrcbLocal1, mix01Local, CeilDiv(rows,8), {1,8})`，与 Cast 并行发射后统一一次 `PipeBarrier<PIPE_V>()`；③Pass0 用 `Mul` 直接写 `y`（所以不需要清零），Pass1..3 用 `MulAddDst(mask=64 或 rem, repeatTimes=nRep)` 累加，pass 之间各一次 barrier；④最后 `CastTwoDim(yLocal, yCastLocal, dim0, dim2, false)` 转回 T。
  - 注意：权重按 `[batch][k]` 交错，偏移是 `i*32 + k*8` 个 float（4 个 Block 一组）；`src1BlkStride=0 / src1RepStride=0` 让一个权重 Block 铺满整行；两次 `syncAfter=false`，`yLocal` 对 GM 的可见性完全靠之后 `yQue.EnQue` 的 V_MTE3 事件，调整语句顺序会导致静默错数。

- **`ProcessY`** [hc_pre_base.h L696-L712](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L696-L712)
  - 职责：通用出口——`dim1==4` 走 `ProcessYHc4`，否则走"转换→乘权重→归约→转回"四步。
  - 入参/出参：`yLocal`、`xLocal`、`mix01Local`（每行一个权重）、`hcBrcbLocal1`、`xCastLocal`、`yCastLocal`、`dim0`/`dim1`/`dim2`、`needBrc`。
  - 实现：①`dim1 == 4` 时直接转调特化版并返回；②`CastTwoDim(xCast, x, dim0*dim1, dim2)`（默认 `syncAfter=true`）；③`MulABLastDimBrcInline<float,true>(xCast, xCast, mix01, hcBrcb, dim0*dim1, dim2)` 按行乘权重；④`ReduceSumARAPerf(yCast, xCast, dim0, dim1, dim2)` 沿 hc_mult 归约；⑤`CastTwoDim(yLocal, yCast, dim0, dim2)`。
  - 注意：这里两个 Cast 都用默认 `syncAfter=true`，比 `ProcessYHc4` 多两次屏障（性能差异点）；权重条数分别是 `dim0*4`（hc4 路径）与 `dim0*dim1`（通用路径），且 hc4 是"batch 内交错"布局、通用路径是"一行一个"，语义不同不能互换缓冲。

- **`ProcessPost`** [hc_pre_base.h L727-L740](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L727-L740)
  - 职责：`post = sigmoid(mix * rsqrt * scale + base) * 2`。
  - 入参/出参：`postLocal` 目的、`mixLocal` 被原地改写、`hcBaseLocal`（base1）、`rsqrtLocal`（**已广播过的** block 布局）、两个 tmp、`scale` 标量（无 `eps`）。
  - 实现：①`MulABLastDimBrcInline<float,false>` 直接复用以广播好的 rsqrt（跳过 Brcb）；②`Muls(mix, mix, scale, rows*colAlign)`；③`AddBAFirstDimBrcInline` 加 base；④`SigmoidPerf(post, mix, tmpBuffer1, ...)`；⑤`Muls(post, post, 2.0f, ...)`。
  - 注意：`needBrc=false` 要求传入的是 `ProcessPre` 中 `Brcb` 产生的 `tmpBuffer0`（每 Block 8 个相同值），传原始一维 rsqrt 不会报错但结果错；两个阶段的 tmp 复用顺序不能颠倒。

- **`LastDimReduceMaxPerf`** [hc_pre_base.h L746-L753](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L746-L753)
  - 职责：`out[r] = max_j in[r, j]`，沿最后一维取最大值。
  - 入参/出参：`output`（每行 1 个结果）、`input`、`curRowNum`/`curColNum` 标量。
  - 实现：①`WholeReduceMax(output, input, curColNum, curRowNum, 1, 1, CeilDiv(curColNum, 8), ORDER_ONLY_VALUE)`：mask=列数、repeatTimes=行数、`dstRepStride=1`、`srcBlkStride=1`、`srcRepStride=CeilDiv(cols,8)` 个 Block；②`ORDER_ONLY_VALUE` 只取值不要索引；③尾部 `PipeBarrier<PIPE_V>()`。
  - 注意：结果是"一个 repeat 的 1 个值占 1 个 Block"，即 `out[i*8]` 有效而不是紧凑数组——下游用 `Brcb` 能直接吃这种布局，用紧凑数组接口就会读到填充值；`curColNum > 255*8` 会超 `srcRepStride` 上限。

- **`LastDimReduceSumPerf`** [hc_pre_base.h L757-L763](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L757-L763)
  - 职责：`out[r] = Σ_j in[r, j]`，沿最后一维求和。
  - 入参/出参：`output`（每行 1 个结果）、`input`、`curRowNum`/`curColNum` 标量。
  - 实现：①`WholeReduceSum(output, input, curColNum, curRowNum, 1, 1, CeilDiv(curColNum, 8))`；②尾部 `PipeBarrier<PIPE_V>()`。
  - 注意：输出布局同样是"每 Block 1 个值"的稀疏形态，和 `ReduceSumARAPerf` 的紧凑行布局不同，两者不能混用；`Dim` 超过一个 repeat 时靠 `srcRepStride` 顺序读，行距必须是 `RoundUp` 后的值。

- **`SoftmaxFP32Perf`** [hc_pre_base.h L773-L787](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L773-L787)
  - 职责：按行（末维）做 softmax 并 `+eps`，供 Sinkhorn 迭代使用。
  - 入参/出参：`output` 目的、`input` 源、`tmpReduceBuffer`（存行归约结果）、`tmpBrcbBuffer`（Brcb 缓存）、`curRowNum`/`curColNum` 标量、`eps` 标量。
  - 实现：①`LastDimReduceMaxPerf` 求每行最大值；②`SubABLastDimBrcInline<float,true>` 减最大值（数值稳定）；③`Exp(output, output, rows * colAlign)` 原位求指数 + `PipeBarrier`；④`LastDimReduceSumPerf` 求每行和；⑤`DivABLastDimBrcInline<float,true>` 除以行和；⑥`Adds(out, out, eps, ...)`。
  - 注意：函数头明确"只支持末维 < 64"（约一个 Repeat），列数一旦变大会同时破坏"行和占 1 Block"和 repeat 步长假设；两个 tmp 尺寸不同——`tmpReduceBuffer` 需 `rows` 个 Block，`tmpBrcbBuffer` 需 `CeilDiv(rows,8)*64` 个 float；对齐补出的填充区也会被 `Exp` 处理，别把对齐区当数据读。

- **`DivABABrcInline`** [hc_pre_base.h L795-L844](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L795-L844)
  - 职责：`out[d0,d1,d2] = in0[d0,d1,d2] / in1[d0,1,d2]`，沿 dim1 广播的 3D 除法。
  - 入参/出参：`output`/`input0` 为 3D（d0×d1×d2 按 d2 对齐）、`input1` 为每 d0 一行（d0×d2Align）、`dim0`/`dim1`/`dim2` 标量。
  - 实现：①`dim1 >= dim2/64` 时在 dim1 方向开 repeat：`dstRepStride = src0RepStride = CeilDiv(dim2,8)`、`src1RepStride=0`，让该 d0 的行向量在 dim1 各层复用；②否则逐 dim1 行用二级 `Div(dst, src0, src1, dim2)`，src1 固定为该 d0 的行向量；③余数列单独一次 `Div(rem, dim1, ...)`；④末尾 `PipeBarrier<PIPE_V>()`。
  - 注意：分支判据把"dim1 行数"与"dim2 的 Repeat 数"直接比大小，两者量纲不同，属经验阈值；`input1` 的布局是 `[dim0][dim2Align]`（没有 dim1 这一维），若按 `[dim0][dim1][dim2]` 算偏移会读到别的 batch。

- **`CopyIn`** [hc_pre_base.h L851-L868](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L851-L868)
  - 职责：GM→UB 单次搬运，用 `DataCopyPad` 支持非对齐长度。
  - 入参/出参：`inputGm` 源、`inputTensor` 目的、`nBurst` 行数、`copyLen` 每行元素数、`srcStride`/`dstStride`（**元素**为单位的间隙）。
  - 实现：①`DataCopyPadExtParams` 全部置 0 且 `isPad=false`；②`DataCopyExtParams`：`blockCount=nBurst`、`blockLen=copyLen*sizeof(T)`（字节）、`srcStride=srcStride*sizeof(T)`（字节）、`dstStride=dstStride/elemInOneBlock`（Block）；③调 `DataCopyPad`，无内部屏障。
  - 注意：函数注释写 `dstStride`"以 Block 为单位"，但代码除以 `elemInOneBlock`，实际语义是**元素**（4D 重载的调用证实这一点），按注释传 Block 会静默错位；`srcStride` 是字节间隙（不含本 burst 长度）；`isPad=false` 时 UB 右侧对齐区保留旧数据。

- **`CopyInWithOuterFor`** [hc_pre_base.h L873-L884](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L873-L884)（3D 重载）
  - 职责：外层循环搬 `outerLoop × nBurst` 行，GM 行距与 UB 对齐行距不同。
  - 入参/出参：`inputGm`/`inputTensor`、`outerLoop` 外层次数、`nBurst` 每次行数、`copyLen` 每行元素数、`gmLastDim` GM 每行长度（标量）。
  - 实现：①`ubLastDimAlign = RoundUp<T>(copyLen)`；②循环 `i < outerLoop`，以 `i*nBurst*gmLastDim`（GM）和 `i*nBurst*ubLastDimAlign`（UB）为基址调用 `CopyIn`；③`srcStride`/`dstStride` 全用默认 0。
  - 注意：默认 0 步长意味着假定 GM 每行紧凑且 `gmLastDim == copyLen`，此时 `gmLastDim` 只参与基址计算；GM 行内存在 padding 时必须改用 4D 重载。

- **`CopyInWithOuterFor`** [hc_pre_base.h L890-L911](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L890-L911)（4D 重载）
  - 职责：4D 源按两种遍历策略搬到 `[outerLoop][nBurst][align]` 的 UB。
  - 入参/出参：`inputGm`/`inputTensor`、`outerLoop`、`nBurst`、`copyLen`、`gmFirstDim`、`gmLastDim`（后三个为标量）。
  - 实现：①`outerLoop <= nBurst`：外层循环，每层一次 `CopyIn` 搬 `nBurst` 行，`srcStride = gmLastDim - copyLen` 跳过行内 padding；②`outerLoop > nBurst`：按 `nBurst` 循环，每次搬 `outerLoop` 段，`srcStride = (gmLastDim - copyLen) + (gmFirstDim - 1) * gmLastDim`、`dstStride = (nBurst - 1) * ubLastDimAlign`；③两条路径产出相同的 `[outerLoop][nBurst][align]` 布局，因为 `dstStride` 是"间隙"而不是绝对步长。
  - 注意：第一条路径要求 `gmFirstDim >= nBurst` 且外层切片连续；第二条把 `srcStride` 按"跨 `gmFirstDim` 层"跳，若该 2D 切片之外还有更高维 padding 就会读错；两分支的 `dstStride` 单位是元素（由 `CopyIn` 内部再换算）。

- **`CopyOut`** [hc_pre_base.h L915-L925](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L915-L925)
  - 职责：UB→GM 单次搬出。
  - 入参/出参：`outputTensor` 源（UB）、`outputGm` 目的、`nBurst` 行数、`copyLen` 每行元素数、`dstStride`（**元素**为单位的行间间隙）。
  - 实现：①`DataCopyExtParams`：`blockCount=nBurst`、`blockLen=copyLen*sizeof(T)`、`srcStride=0`、`dstStride=dstStride*sizeof(T)`（转字节）；②`DataCopyPad` 直接调用，函数内**没有** `PipeBarrier`。
  - 注意：`srcStride` 硬编码 0，要求 UB 侧"每行恰好 `copyLen` 个元素紧凑排列"；如果 UB 是按 `RoundUp` 对齐存的（行距 8/16），必须先把数据紧凑化或按行分多次调用，否则搬出去的内容错位；注释说 `dstStride` 是字节，代码却乘了 `sizeof(T)`，实际单位是元素；GM 可见性靠调用方 `EnQue` 事件。

### 2. 分类表（26 个函数，主/副分类）

| 函数 | 行号 | 主分类 | 副分类 |
| --- | --- | --- | --- |
| `CeilDiv` | L48-L54 | 缓冲管理 | — |
| `CeilAlign` | L57-L60 | 缓冲管理 | — |
| `RoundUp<T>` | L63-L68 | 缓冲管理 | 类型转换（依赖 sizeof(T)） |
| `SetGatherMaskPattern` | L75-L82 | 归约 | 缓冲管理（掩码预置） |
| `GatherMaskByDiagonal` | L90-L100 | 归约 | 基础搬运（按掩码收集） |
| `MulABLastDimBrcInline` | L111-L193 | 广播 | 逐元素 |
| `SubABLastDimBrcInline` | L198-L273 | 广播 | 逐元素 |
| `DivABLastDimBrcInline` | L279-L354 | 广播 | 逐元素 |
| `AddBAFirstDimBrcInline` | L363-L427 | 广播 | 逐元素 |
| `CalcDenominator` | L434-L444 | 激活 | 逐元素 |
| `SigmoidPerf` | L450-L476 | 激活 | 广播（常量 Block 复用） |
| `ProcessPre` | L489-L502 | 逐元素 | 激活 / 广播（组合流程） |
| `ReduceSumARAPerf` | L510-L552 | 归约 | 基础搬运（首切片 DataCopy） |
| `ReduceSum4x4Perf` | L556-L570 | 归约 | 逐元素 |
| `CastTwoDim` | L580-L613 | 类型转换 | 基础搬运（按行搬运式 Cast） |
| `ProcessYHc4` | L625-L694 | 逐元素 | 广播 / 类型转换（MulAddDst 融合） |
| `ProcessY` | L696-L712 | 逐元素 | 广播 / 归约 / 类型转换（调度） |
| `ProcessPost` | L727-L740 | 逐元素 | 激活 / 广播（组合流程） |
| `LastDimReduceMaxPerf` | L746-L753 | 归约 | — |
| `LastDimReduceSumPerf` | L757-L763 | 归约 | — |
| `SoftmaxFP32Perf` | L773-L787 | Softmax | 归约 / 广播（组合流程） |
| `DivABABrcInline` | L795-L844 | 广播 | 逐元素（Sinkhorn 归一） |
| `CopyIn` | L851-L868 | 基础搬运 | —（GM→UB） |
| `CopyInWithOuterFor`（3D） | L873-L884 | 基础搬运 | —（GM→UB） |
| `CopyInWithOuterFor`（4D） | L890-L911 | 基础搬运 | 缓冲管理（stride 重排） |
| `CopyOut` | L915-L925 | 基础搬运 | —（UB→GM） |

分类计数（按主分类）：基础搬运 4、类型转换 1、归约 6、逐元素 4、广播 5、激活 2、Softmax 1、缓冲管理 3，合计 26 条（其中 `CopyInWithOuterFor` 的 3D/4D 两个重载各算一条）。

#### 2.1 按类走读（每类的共同套路）

- **基础搬运**（`CopyIn` / `3D` / `4D` / `CopyOut`）：唯一直接和 GM 打交道的 4 个函数。
  套路都是"把元素数换算成字节给 `blockLen`、把行间空隙换算成 `srcStride` 字节 / `dstStride` Block"。
  3D 与 4D 两个重载的区别只在"GM 行内是否连续"：连续就走简单的按外层循环，不连续才用 `srcStride` 跳。
- **类型转换**（`CastTwoDim`）：bf16 与 fp32 的 Block 元素数不同（16 vs 8），所以任何"混类型"的张量
  都天然有两种行距。所有逐行 Cast 的 stride 分歧都源于这一点，对齐判断必须同时看源和目的。
- **归约**（`SetGatherMaskPattern` / `GatherMaskByDiagonal` / `ReduceSumARAPerf` / `ReduceSum4x4Perf` /
  `LastDimReduceMaxPerf` / `LastDimReduceSumPerf`）：分两类——"沿末维归约"用 `WholeReduce*`
  （一行一指令、结果占 1 Block），"沿中间维归约"用 `DataCopy` + 多层 `Add`（结果紧凑）。
  混用这两类输出布局是本文件最常见的坑。
- **逐元素**（本表把四个 `*BrcInline` 记在"广播"下；这里的 4 个是编排函数 `ProcessPre` / `ProcessPost` /
  `ProcessYHc4` / `ProcessY`，它们本身不含新指令，只是把广播、激活、Cast、归约串起来）：
  真正的逐元素模板在 `*BrcInline` 里，共同点是"两种 repeat 策略 + 尾段"。判断"按行循环"还是
  "行方向开 repeat"的依据是 `dstRepStridePerLine > 255`（硬件上限）与 `curRowNum < numRepeatPerLine`（行太少时并行度不够）。
- **广播**（四个 `*BrcInline` + `DivABABrcInline` + `SigmoidPerf` 的常量块）：
  三种实现手段已在 0.4 列出；看函数时先找"哪一种 stride 被置 0"，就知道广播发生在哪个维度。
- **激活**（`CalcDenominator` / `SigmoidPerf`）：`sigmoid(x)=1/(1+exp(-x))`，故意不做 `exp(x)/(1+exp(x))`
  以避开溢出；数值稳定性靠"分母恒 ≥ 1"，所以不需要 max 平移（softmax 才需要）。
- **Softmax**（`SoftmaxFP32Perf`）：`max` 平移 + `exp` + 行和归一 + `eps`，是归约与广播原语的编排，
  本身不含新指令；限制（末维 ≤ 64）来自"行和只占 1 个 Block"这一布局假设。
- **缓冲管理**（`CeilDiv` / `CeilAlign` / `RoundUp<T>`）：没有搬运动作，只产出整数。
  但它们决定了上面所有函数的 row/col stride，改错一处会让整条链路错位而不报错。

#### 2.2 热路径与屏障密度（读代码时容易忽略的成本）

- `ProcessYHc4` 是唯一刻意把两次 `Cast` 的 `syncAfter` 置 `false`、把同步合并成一次的函数，
  换来的是少两次向量流水排空；`ProcessY` 仍走默认屏障，属于"未优化路径"。
- `SigmoidPerf` 与 `SoftmaxFP32Perf` 里都用"一个常量 Block + `src0RepStride=0`"替代逐元素造数，
  减少的是 `Duplicate` 的写带宽，不是指令条数。
- `ReduceSumARAPerf` 在 `j` 的每一层都 `PipeBarrier`，是正确性必需（对 `output` 读改写）；
  `ReduceSum4x4Perf` 省掉了它，仅因为 4 元素小块下发射顺序可控——这是特化路径的隐含前提。
- 四个 `CopyIn/CopyOut` 内部都没有屏障，全靠调用方 Queue 的 `EnQue`/`DeQue` 事件；
  在它们后面立刻读 UB 是典型错误。
- 三份 `Mul/Sub/Div` 的 `ABLastDimBrcInline` 是逐字复制的，新增一类逐元素运算（例如 `Max`）时
  必须三处同改；只改一处会让编译通过但语义分叉。
- `needBrc` 参数不是"性能开关"而是"数据形态契约"：`false` 只表示"上游已经按 Brcb 布局准备好"，
  并不等价于"跳过了一次计算"。

### 3. 三条最容易看错的点

1. **步长单位三种混用**：`CopyIn`/`CopyOut` 的 `srcStride` 是字节间隙、`dstStride` 实际是元素（注释写 Block/字节，与代码不符），而 `BinaryRepeatParams` 的 6 个 stride 全是以 Block 为单位；传错量纲不会报错，只会错数据。
2. **两类输出布局不能互换**：`LastDimReduce*` 的结果是"一个值占一个 Block"，能被 `Brcb` 直接消费；`ReduceSumARAPerf` 是紧凑行；反之 `AddBAFirstDimBrcInline` 要求的是**紧凑 dense 向量**，把它接到 Brcb 缓冲上必错。
3. **`AddBAFirstDimBrcInline` 的尾块分支漏了行偏移**：[hc_pre_base.h L411-L413](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L411-L413) 循环里没加 `i * curColNumAlign`，只有第 0 行尾块被反复重算；同结构的 `Sub`/`Div` 版本（L256-L258、L336-L339）都带该偏移，可对照确认这是笔误而非设计。


---

## 14. 函数级详解：arch35 计算原语（`hc_pre_base_arch35.h`）

> 对象文件：`vllm-ascend/csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h`（共 1418 行，命名空间 `HcPreNs`）
> 对照文件：`vllm-ascend/csrc/moe/hc_pre/op_kernel/hc_pre_base.h`（通用版，命名空间 `HcPre`，A3 架构）
> 阅读前提：读者不需要懂 Ascend C。只需要记住三条：`LocalTensor` 是 UB（片上统一缓冲）里的一块内存；`__local_mem__ T*` 是它的裸指针；`RegTensor<T>` 是向量寄存器，一次覆盖 `VL_FP32 = 64` 个 fp32 通道（lane），`MaskReg` 是按 lane 的位掩码。
> 本文按源码出现顺序排列，共 42 个函数，无遗漏（含模板函数、内联工具、类外自由函数）。

---

### 零、函数索引（按源码顺序）

| # | 函数 | 行号 | 一行职责 |
| --- | --- | --- | --- |
| 1 | `Align` | L30-L36 | 向上对齐到 b 的倍数 |
| 2 | `CeilDiv` | L38-L44 | 向上取整除法 |
| 3 | `CeilAlign` | L46-L49 | 向上对齐值 |
| 4 | `RoundUp<T>` | L51-L56 | 按 32B Block 元素数对齐 |
| 5 | `SetWaitFlag<event>` | L72-L78 | Set+Wait 事件排空 |
| 6 | `LoadInputData<T>` | L80-L90 | UB → 寄存器（含 16bit 升精度） |
| 7 | `StoreOutputData<T>` | L92-L103 | 寄存器 → UB（含降精度打包） |
| 8 | `LoadInputDataWithBrc<T>` | L105-L116 | 标量广播加载 |
| 9 | `LoadInputDataUnalign<T>` | L118-L131 | 非对齐流式加载 |
| 10 | `VFSigmoid` | L134-L140 | 逐 lane sigmoid |
| 11 | `VFTransND2NZ` | L142-L194 | UB 内 ND → NZ 重排 |
| 12 | `VFProcessCast<T>` | L196-L233 | 二维 Cast 到 fp32 |
| 13 | `VFProcessCastAndInvRmsPart1` | L235-L314 | Cast + 行平方和 |
| 14 | `VFProcessCastInvRmsAndPremixYPart1` | L316-L384 | Cast + 平方和 + premix 累加 |
| 15 | `VFProcessPremixYCast<T>` | L386-L410 | premix 结果转回 T |
| 16 | `VFProcessInvRmsPart1` | L412-L484 | fp32 就地平方和 |
| 17 | `VFProcessInvRmsPart2` | L488-L513 | 1/sqrt(行和+eps) |
| 18 | `VFProcessInvRmsPart3WithGroupReduce` | L518-L639 | groupK 路归约 + rsqrt 加权 |
| 19 | `VFProcessInvRmsPart3` | L642-L674 | 单 k 版 rsqrt 加权 |
| 20 | `VFProcessPre` | L678-L730 | 产出 pre |
| 21 | `VFProcessPost` | L732-L790 | 产出 post |
| 22 | `VFProcessCombFragRLessVL` | L793-L867 | comb_frag softmax + Sinkhorn |
| 23 | `VFProcessIteration` | L869-L877 | Sinkhorn 单步归一累加 |
| 24 | `VFProcessCombFragRLessVLUseFourUnfold` | L879-L961 | 4 行展开版 Sinkhorn |
| 25 | `RowGroupMaxBcast` | L963-L972 | 4 lane 组内 max 广播 |
| 26 | `RowGroupSumBcast` | L975-L984 | 4 lane 组内 sum 广播 |
| 27 | `SoftmaxRowBcastInplace` | L987-L998 | 就地行 softmax |
| 28 | `RowNormAccum` | L1001-L1010 | 行归一 + 列和累加 |
| 29 | `VFProcessCombFragPacked` | L1012-L1096 | 打包布局的 comb_frag |
| 30 | `VFElementMajorSum<Pairwise>` | L1098-L1111 | 4 寄存器求和 |
| 31 | `VFElementMajorNormalize<Pairwise>` | L1113-L1124 | 4 寄存器归一 |
| 32 | `VFElementMajorSoftmax` | L1126-L1152 | 4 寄存器 softmax |
| 33 | `VFElementMajorLoad<Element>` | L1154-L1164 | gather + 偏置广播 + scale |
| 34 | `VFProcessCombFragPremixElementMajor` | L1167-L1248 | 元素主序 comb_frag |
| 35 | `VFProcessY<T>` | L1251-L1303 | 加权求和产出 y |
| 36 | `CopyIn<T>` | L1305-L1321 | GM → UB |
| 37 | `CopyToL1<T>` | L1323-L1327 | UB → L1 |
| 38 | `CopyInWithLoopMode<T>` | L1329-L1357 | LoopMode 3D 搬入 |
| 39 | `CopyOut<T>`（GM） | L1359-L1369 | UB → GM |
| 40 | `CopyInWithUbStride<T>` | L1371-L1389 | 带 UB 行距的搬入 |
| 41 | `UbRowGapBlocks` | L1391-L1398 | 算 UB 侧空块数 |
| 42 | `CopyOut<T>`（UB） | L1400-L1413 | UB → UB 压实 |

---

### 一、逐函数讲解（按出现顺序）

- **`Align`** [hc_pre_base_arch35.h L30-L36](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L30-L36)
  - 职责：把 a 向上对齐到 b 的倍数，b 为 0 时原样返回。
  - 入参/出参：a 待对齐值、b 对齐粒度；返回 `uint64_t` 对齐结果。
  - 实现：纯标量 `(a + b - 1) / b * b`，不涉及任何 MicroAPI，用于算 UB 偏移与行距。
  - 注意：通用版 `hc_pre_base.h` 没有此函数（只有 `CeilDiv`/`CeilAlign`）；易错点：b 必须非 0，否则靠函数内特判兜住，调用方不要依赖除零行为。

- **`CeilDiv`** [hc_pre_base_arch35.h L38-L44](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L38-L44)
  - 职责：向上取整除法，b 为 0 时返回 a。
  - 入参/出参：a 被除数、b 除数；返回 `uint64_t` 商。
  - 实现：标量 `(a + b - 1) / b`，全文件用它算 `loopCount = CeilDiv(col, VL_FP32)` 与流水步数。
  - 注意：通用版同名函数是 `int32_t` 版（L48-L54）；arch35 提升为 `uint64_t` 以免大 offset 溢出，但也意味着传负数会回绕成大数（易错）。

- **`CeilAlign`** [hc_pre_base_arch35.h L46-L49](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L46-L49)
  - 职责：`CeilDiv` 的结果再乘 b，得到对齐值。
  - 入参/出参：a 值、b 粒度；返回 `uint64_t`。
  - 实现：`return CeilDiv(a, b) * b;`，是 `RoundUp` 与 `VFTransND2NZ` 的底座。
  - 注意：与通用版一致（仅返回类型由 `int32_t` 变 `uint64_t`）。

- **`RoundUp<T>`** [hc_pre_base_arch35.h L51-L56](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L51-L56)
  - 职责：按类型 T 在 32B Block 内的元素数向上对齐元素个数。
  - 入参/出参：num 元素数；返回对齐后的元素数（仍是元素单位，不是字节）。
  - 实现：`elemNum = BLOCK_SIZE / sizeof(T)`，再 `CeilAlign(num, elemNum)`；fp32 得 8 元素对齐，bf16/fp16 得 16 元素对齐。
  - 注意：与通用版一致；易错：它对齐的是"元素个数"，`DataCopyPad` 的 `dstStride` 却是 32B 块单位，两者不能混用。

- **`SetWaitFlag<event>`** [hc_pre_base_arch35.h L72-L78](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L72-L78)
  - 职责：立即 Set 并 Wait 同一个硬件事件，实现"就地排空"同步。
  - 入参/出参：模板参数 `event`（`HardEvent`）；函数内 `FetchEventID` 取 eventId，无返回值。
  - 实现：`SetFlag<event>(eventId)` 紧跟 `WaitFlag<event>(eventId)`；不在 `__VEC_SCOPE__` 内，属纯流水控制指令。
  - 注意：通用版无（通用版统一用 `PipeBarrier<PIPE_V>()`）；易错：等价于"当前流水全部排空"，放在热循环里会切断向量指令并行。

- **`LoadInputData<T>`** [hc_pre_base_arch35.h L80-L90](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L80-L90)
  - 职责：从 UB 取 64 lane 到 `RegTensor<float>`，16bit 输入自动升精度。
  - 入参/出参：dst 值寄存器、src UB 裸指针、pregLoop 有效 lane 掩码、srcOffset 元素偏移。
  - 实现：① `T=float` 走 `DataCopy(dst, src + offset)`；② `T=half/bfloat16_t` 先 `DataCopy<T, LoadDist::DIST_UNPACK_B16>` 到 `RegTensor<T>` 再 `Cast<float, T, castTraitB162B32Even>`；③ 语义为 `dst[l] = src[srcOffset + l]`。
  - 注意：通用版无此层（对应 `CastTwoDim`）；易错：掩码只负责"哪些 lane 有效"，不足 64 的尾部必须靠行距多留 8 个元素 padding 才读得安全。

- **`StoreOutputData<T>`** [hc_pre_base_arch35.h L92-L103](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L92-L103)
  - 职责：把 `RegTensor<float>` 按掩码写回 UB，16bit 输出先 Cast 再打包。
  - 入参/出参：dst UB 指针、src 值寄存器、pregLoop 掩码、dstOffset 元素偏移。
  - 实现：① fp32 走 `DataCopy(dst + off, src, pregLoop)`；② 16bit 先 `Cast<T, float, castTraitB322B16Even>`（`CAST_RINT`）再 `DataCopy<T, StoreDist::DIST_PACK_B32>`；③ 未被掩码选中的 lane 不落盘。
  - 注意：通用版无（对应 `CastTwoDim` 的末段）；易错：`DIST_PACK_B32` 的地址按元素算，写成字节地址会整段错位。

- **`LoadInputDataWithBrc<T>`** [hc_pre_base_arch35.h L105-L116](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L105-L116)
  - 职责：把单个标量广播到 64 个 lane（行内广播加载）。
  - 入参/出参：dst、src、pregLoop、srcOffset 指向被广播的那一个元素。
  - 实现：① fp32 用 `DataCopy<float, LoadDist::DIST_BRC_B32>`；② 16bit 用 `DIST_BRC_B16` 再 `Cast`；③ 语义 `dst[l] = src[srcOffset]`（与 l 无关）。
  - 注意：通用版对应 `Brcb(tmpBuffer, input1, repeatTimes, {...})`；易错：BRC 只重复"一个元素"，拿它去读连续 64 个值会得到 64 份相同数据。

- **`LoadInputDataUnalign<T>`** [hc_pre_base_arch35.h L118-L131](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L118-L131)
  - 职责：非对齐流式加载一拍 64 lane，并推进源指针。
  - 入参/出参：dst、src 指针引用（函数内被推进）、uSrc `UnalignReg` 状态、pregLoop、postUpdateStride 每次前进的元素数。
  - 实现：① `DataCopyUnAlign(dst, uSrc, src, postUpdateStride)`；② 16bit 先取 `RegTensor<T>`，`UnPack` 成 `uint32` 再 `Cast` 到 float；③ 跨不同行距（非 32B 整数倍）顺序读取靠它替代二维 Brcb。
  - 注意：通用版无；易错：必须先用 `DataCopyUnAlignPre` 预热 `UnalignReg`，且每轮要把 src 指针重置到"原点 + 列偏移"，否则步进漂移。

- **`VFSigmoid`** [hc_pre_base_arch35.h L134-L140](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L134-L140)
  - 职责：逐 lane 算 `1/(1+exp(-x))` 的 sigmoid。
  - 入参/出参：y 输出、x 输入兼临时（被就地改写）、one 常量 1 寄存器、pregLoop 掩码。
  - 实现：① `Muls(x, x, -1)`；② `Exp`；③ `Adds(x, x, 1)`；④ `Div(y, one, x)`；即 `y = 1/(1+e^{-x})`。
  - 注意：通用版拆成 `CalcDenominator` + `SigmoidPerf`（靠 `BinaryRepeatParams` 让常量 1 的 Blk/RepStride=0 复用）；此处 one 必须由调用方预置且宽度覆盖掩码；易错：one 只按 VL1 生成却用于 64 lane 会读到脏 lane。

- **`VFTransND2NZ`** [hc_pre_base_arch35.h L142-L194](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L142-L194)
  - 职责：UB 内把行主序 (row, col) 重排成 8x8 分形的 NZ 布局。
  - 入参/出参：xLocal 源、yLocal 目的、curRowNum 有效行、curColNum 有效列；无返回值，直接改写 yLocal。
  - 实现：① 以 8 行（`C0_SIZE`）x 8 元素（`c1Size = 32/sizeof(float)`）为一个 64 lane 的块，用 `DataCopyMode::DATA_BLOCK_COPY` 一拍取满；② `dataBlockStride = curColNumAlign/8 + 1` 用于跨过源行尾多出的 32B padding；③ 主体行块循环，剩余行用 `UpdateMask<float>(rowReminder*8)` 只搬需要的行；④ 目的偏移 `j * curRowNumAlign * c1Size`，形成"列块在外、行块在内"的 `(N/8, M/8, 8, 8)` 布局。
  - 注意：通用版无此函数（通用版 NZ 由 Matmul/fixpipe 侧完成）；易错：注释写"M,N 方向按 16 对齐"，实际对齐量是 `C0_SIZE = 8`，且源行距必须自带 +8 元素 padding，否则 `DATA_BLOCK_COPY` 会取到下一行的数据。

- **`VFProcessCast<T>`** [hc_pre_base_arch35.h L196-L233](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L196-L233)
  - 职责：二维 tile 逐行把 T 转成 float 写进 xCastLocal。
  - 入参/出参：xLocal 源、yLocal（float）目的、curRowNum 行数、curColNum 列数。
  - 实现：① `loopCount > 1`（列 > 64）时每行内用 `UpdateMask<float>(sreg)` 逐 64 列推进，否则一次 `CreateMask<float>()` 到底；② 每拍 `LoadInputData` + `StoreOutputData` 成对出现；③ 目的行距 `RoundUp<float>(col) + 8`，每行尾部多留一个 32B block。
  - 注意：通用版 `CastTwoDim` 是整块或逐行 `Cast`，无行尾 padding；易错：调用方之后读 xCast 的偏移必须按 `align + 8` 算，漏掉 +8 会把下一行读进当前行。

- **`VFProcessCastAndInvRmsPart1<T, WithUbReduce>`** [hc_pre_base_arch35.h L235-L314](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L235-L314)
  - 职责：Cast 到 fp32 的同时算行平方和乘 coeff，写入 rmsNorm。
  - 入参/出参：rmsNormLocal 输出每行一个值、xCastLocal 输出 fp32、xLocal 源、coeff 缩放、curRowNum/curColNum；`WithUbReduce` 决定是否与旧值相加。
  - 实现：① `Duplicate(sum, 0.0f)` 起始；② 每拍 `LoadInputData` -> `Mul(x1, x, x)` -> `Add(sum, sum, x1, pregMain)` 累加，同时 `StoreOutputData` 落 xCast；③ 收尾 `Muls(sum, sum, coeff)` 再 `ReduceSum` 得行内总和；④ `DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>` 只写 lane0；公式 `rms[i] = coeff * Σ_j x[i][j]^2`（可选 + 旧值）。
  - 注意：通用版是 `CastTwoDim` + `ReduceSumARAPerf`/`GatherMaskByDiagonal` 两趟往返 UB，此处一趟融合；易错：累加刻意用全掩码 `pregMain`，因为 `Mul` 已把无效 lane 置零，若改用 `pregLoop` 反而漏加尾数。

- **`VFProcessCastInvRmsAndPremixYPart1<T, WithRmsReduce, WithYReduce>`** [hc_pre_base_arch35.h L316-L384](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L316-L384)
  - 职责：Cast、行平方和、premix 加权累加三合一。
  - 入参/出参：rmsNormLocal 出 rms、yAccLocal 出累加结果、xCastLocal 出 fp32、preMixLocal 权重源、`preMixColIdx`/`preMixRowStride` 定位权重、`yAccRowStride`、coeff；两个 bool 决定是否叠加旧 rms / 旧 yAcc。
  - 实现：① `LoadInputData<T>` 取 x 后立刻 `StoreOutputData<float>` 存 cast 副本；② `Mul(xSquare, x, x)` + `Add` 累进 `squareSum`；③ `LoadInputDataWithBrc` 把 `mix[i][k]` 广播成 64 lane，`Mul(weightedX, mix, x)`，再 `Add(yAcc, yAcc, weightedX)` 回写；④ 行尾 `Muls(coeff)` + `ReduceSum`，可选 `Add(squareSum, rmsOld, squareSum)` 后 `DIST_FIRST_ELEMENT_B32` 写 rms。公式：`yAcc[i][d] += mix[i][k]*x[i][d]`，`rms[i] = coeff*Σx^2 (+old)`。
  - 注意：通用版用 `MulABLastDimBrcInline` + `ReduceSumARAPerf` 分多趟做；此处单趟；易错：第 ③ 步 BRC 用 `pregLoop` 作掩码，列数 < 64 时未选中 lane 是脏值，绝不能参与后面归约。

- **`VFProcessPremixYCast<T>`** [hc_pre_base_arch35.h L386-L410](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L386-L410)
  - 职责：把 fp32 的 yAcc 逐行转回 T 写入 yLocal。
  - 入参/出参：yLocal 目的（T）、yAccLocal 源（float）、`yAccRowStride`、curRowNum/curColNum。
  - 实现：① 每行 `UpdateMask<float>(sreg)` 逐 64 列；② `LoadInputData<float>` 取 fp32；③ `StoreOutputData<T>` 内部做 `Cast` + `DIST_PACK_B32`。
  - 注意：通用版对应 `CastTwoDim(yLocal, yCastLocal, ...)`；易错：目的行距是 `RoundUp<T>(col)`、源行距是 `yAccRowStride`，两者不同，不能复用同一个偏移表达式。

- **`VFProcessInvRmsPart1<WithUbReduce>`** [hc_pre_base_arch35.h L412-L484](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L412-L484)
  - 职责：对已是 fp32 的 x 就地平方求和，写 rms（可选累加）。
  - 入参/出参：yLocal 输出每行一个 rms、xLocal 源兼就地平方、coeff、curRowNum/curColNum、`WithUbReduce`。
  - 实现：① `Duplicate(sum, 0.0f)`；② `Mul(x, x, x)` 就地平方 + `Add(sum, sum, x)` 累加；③ `Muls(coeff)` -> `ReduceSum` -> `DIST_FIRST_ELEMENT_B32` 写 lane0；④ `WithUbReduce` 分支先用 BRC 取旧值 `Add` 再写回。
  - 注意：通用版对应 `ReduceSumARAPerf`（多核 K 轴归约）不属于同一职责，此处只做单核行内平方和；易错：来源行距是 `RoundUp<float>(col) + 8`，偏移漏 +8 会读到上一行 padding。

- **`VFProcessInvRmsPart2`** [hc_pre_base_arch35.h L488-L513](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L488-L513)
  - 职责：行求和后算 `1/sqrt(sum+eps)`，把 (bs, k) 压成 (bs, 1)。
  - 入参/出参：yLocal 输出、xLocal 输入、eps 防零、curRowNum 行、curColNum 列（k 分核数，必然 < 64）。
  - 实现：① 一次 `UpdateMask<float>(sreg)` 定列掩码（整函数不再分块）；② `ReduceSum(sum, x)` 得行和；③ `Adds(eps)` -> `Sqrt` -> `Div(y, one, sum)`；④ `DIST_FIRST_ELEMENT_B32` 写 lane0。公式 `y[i] = 1/sqrt(Σ_j x[i][j] + eps)`。
  - 注意：通用版无同名函数；易错：one 只用 `MaskPattern::VL1` 生成，参与 Div 的 lane 必须都落在这个窄掩码里，否则读到未初始化 lane。

- **`VFProcessInvRmsPart3WithGroupReduce`** [hc_pre_base_arch35.h L518-L639](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L518-L639)
  - 职责：把 groupK 个分核的 (k, bs, hcMix) 归约并做 rsqrt 加权。
  - 入参/出参：yLocal 输出 (bs, hcMix)、mmLocal 为 (k, bs, hcMix)、xLocal 为 (k, bs)、eps、groupK、bs、hcMix。
  - 实现：① `groupK < 4` 分支逐个 j：`LoadInputDataWithBrc` 取 `x[i + j*bsAlign]` 累加进 sum1，同时整行 `LoadInputData` 取 mm 累加进 sum2；② `groupK >= 4` 走 4 路展开（`fourLoopNum` + `tailLoopNum`），4 组 sumX/sumM 最后两两相加；③ `Adds(eps)` -> `Sqrt` -> `Div(rsqrt, one, sum1)` -> `Duplicate` 把标量铺满 64 lane -> `Mul(y, sumM, rsqrt)`；公式 `y[i][c] = Σ_k mm[k,i,c] / sqrt(Σ_k x[k,i] + eps)`。
  - 注意：通用版由 `ReduceSumARAPerf(dim1=k)` + 后续归一化分步完成；此处 eps 加在 sqrt 之前；易错：4 路展开的 tail 循环只回写 `sumX1/sumM1`，且 `fourLoopNum=0` 时依赖初始 0 值，别把 sumX2..4 当成有效输入。

- **`VFProcessInvRmsPart3`** [hc_pre_base_arch35.h L642-L674](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L642-L674)
  - 职责：单 k 版本，`y[i][c] = mm[i][c] / sqrt(x[i] + eps)`。
  - 入参/出参：yLocal 输出、mmLocal (bs, hcMix)、xLocal (bs)、eps、bs、hcMix。
  - 实现：① `LoadInputDataWithBrc` 取 `x[i]` 一个标量；② `Adds(eps)` -> `Sqrt` -> `Div(rsqrt, one, x)`；③ `Mul(y, mm, rsqrt)` 后整行 `StoreOutputData`（无行尾回写）。
  - 注意：通用版无（等价 `ReduceSum4x4Perf` 后接 `DivABABrcInline`）；易错：`one` 用 `pregLoop` 生成，隐含 hcMix < 64 的前提，超 64 必须改分块。

- **`VFProcessPre`** [hc_pre_base_arch35.h L678-L730](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L678-L730)
  - 职责：`pre = sigmoid(mix*scale + base) + eps`。
  - 入参/出参：preLocal 输出、mixLocal 输入（行距 `hcMixAlign`）、hcBaseLocal 偏置、scale、eps、curRowNum/curColNum(=hcMix)、hcMix。
  - 实现：① 列 > 64 时外层按 64 列分块、内层按行，否则行主循环；② 每行 `Muls(scale)` -> `Add(base)` -> `VFSigmoid` -> `Adds(eps)`；③ base 每个列块只 `LoadInputData` 一次，内层多行复用。
  - 注意：通用版 `ProcessPre` 先 `mix *= rsqrt` 再 `*= scale`；arch35 的 mix 已预乘 rsqrt，故少了这一步；易错：函数里声明的 `curColNumAlign` 从未使用，行距实际是 `hcMixAlign`，照抄会算错地址。

- **`VFProcessPost`** [hc_pre_base_arch35.h L732-L790](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L732-L790)
  - 职责：`post = 2 * sigmoid(mix*scale + base)`。
  - 入参/出参：postLocal 输出、mixLocal 输入（行距 `hcMixAlign`）、hcBaseLocal、scale、eps（入参透传）、curRowNum/curColNum、hcMix。
  - 实现：① `DataCopyUnAlignPre<float>(uMix, mixLocalAddr)` 预热，逐行 `LoadInputDataUnalign(mix, ..., hcMixAlign)` 顺序推进；② `Muls(scale)` -> `Add(base)` -> `VFSigmoid` -> `Muls(2.0)`；③ 目的行距 `curColNumAlign` 写回。
  - 注意：通用版 `ProcessPost` 用 `MulABLastDimBrcInline<..., false>` 复用已广播的 rsqrt，无 Unalign；易错：每轮必须把 `mixLocalAddr` 重置为 `mixOriginLocalAddr + 列块偏移`，否则指针一路滑到别的行。

- **`VFProcessCombFragRLessVL`** [hc_pre_base_arch35.h L793-L867](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L793-L867)
  - 职责：R 轴 < hcMult 的 comb_frag：行 softmax -> 列归一 -> iters 轮 Sinkhorn。
  - 入参/出参：combFragLocal 输出、mixLocal 输入（行距 hcMixAlign）、hcBaseLocal、scale、eps、iters 迭代数、dim0=bs、dim1=hc_mult(行)、dim2=hc_mult(R 轴)、hcMix。
  - 实现：① 行 softmax：`ReduceMax` -> `Duplicate(max)` -> `Sub` -> `Exp` -> `ReduceSum` -> `Duplicate(sum)` -> `Div` -> `Adds(eps)`，同时 `Add(sum1, sum1, mix)` 累列和；② `LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>()` 后 `Adds(sum1, sum1, eps)`，逐行 `Div(mix, mix, sum1)`；③ iters 轮：每行 `ReduceSum` 归一并累列和再统一除。
  - 注意：通用版用 `SoftmaxFP32Perf` + `DivABABrcInline` 走 UB 二维 API；此处全寄存器；易错：`VEC_STORE -> VEC_LOAD` 的 `LocalMemBar` 不能省，否则回读到旧值。

- **`VFProcessIteration`** [hc_pre_base_arch35.h L869-L877](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L869-L877)
  - 职责：单行 Sinkhorn 一步：`mix /= (行和+eps)` 并累加列和。
  - 入参/出参：sum0 列和累加器、sum1 行和临时、mix 数据（就地）、eps、pregLoop。
  - 实现：① `ReduceSum(sum1, mix, pregLoop)`；② `Duplicate(sum1, sum1, pregLoop)` 广播；③ `Adds(sum1, sum1, eps)` 后 `Div(mix, mix, sum1)`；④ `Add(sum0, sum0, mix)`。
  - 注意：通用版无此函数（从头文件里抽出的公共片段）；易错：sum0 需调用方每轮清零，函数本身不做清零。

- **`VFProcessCombFragRLessVLUseFourUnfold`** [hc_pre_base_arch35.h L879-L961](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L879-L961)
  - 职责：dim1 恰为 4 时把 Sinkhorn 尾段 4 行展开的版本。
  - 入参/出参：同 `VFProcessCombFragRLessVL`，dim1 必须为 4；输出仍写回 combFragLocal。
  - 实现：① 首段与 RLessVL 完全一致（行 softmax + 列归一）；② 一次性 `LoadInputData` 取 4 行到 mix1..mix4 常驻寄存器；③ iters 轮内 4 次 `VFProcessIteration` 累进 sum，再对 4 行统一 `Div(sum+eps)`；④ 循环结束后 4 次 `StoreOutputData` 落盘，中间无 UB 往返。
  - 注意：通用版用 `ReduceSum4x4Perf` 省掉首行的 `DataCopy`；此处省掉的是"每轮 UB 来回"；易错：函数硬编码读 4 行，dim1 != 4 会越界读相邻数据。

- **`RowGroupMaxBcast`** [hc_pre_base_arch35.h L963-L972](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L963-L972)
  - 职责：在连续 4 lane 一组内求 max 并广播回这 4 个 lane。
  - 入参/出参：in 输入、out 输出、t0..t3 临时寄存器、preg 掩码。
  - 实现：① `DeInterleave(t0, t1, in, in)` 拆出偶/奇 lane 两两配对；② `Max(t0, t0, t1)` 再 `DeInterleave` + `Max`，得到 lane0..3 为每组 max；③ `Interleave` 两次把结果铺成 `[m0,m0,m0,m0,m1,...]`，即每行 max 广播到 4 个 lane。
  - 注意：通用版用 `LastDimReduceMaxPerf`（`WholeReduceMax`）在 UB 上做；易错：`DeInterleave` 把 in 同时当两个源是"原地配对"技巧，临时寄存器不可与 in 复用。

- **`RowGroupSumBcast`** [hc_pre_base_arch35.h L975-L984](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L975-L984)
  - 职责：4 lane 一组求和并广播回这 4 个 lane。
  - 入参/出参：同上，只是把 `Max` 换成 `Add`；语义 `out[4r+c] = Σ_c M[r][c]`。
  - 实现：① `DeInterleave` + `Add` 两次；② `Interleave` 两次广播；③ 全程单掩码、无 `ReduceSum` 指令。
  - 注意：通用版对应 `LastDimReduceSumPerf`；易错：必须先 `DeInterleave` 再 `Add`，顺序反了就变成跨组求和。

- **`SoftmaxRowBcastInplace`** [hc_pre_base_arch35.h L987-L998](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L987-L998)
  - 职责：寄存器内按 4 lane 一组做 softmax 并 +eps，就地完成。
  - 入参/出参：m 数据（就地读写）、red 归约临时、t0..t3、eps、preg。
  - 实现：① `RowGroupMaxBcast` -> `Sub(m, m, red)`；② `Exp`；③ `RowGroupSumBcast` -> `Div(m, m, red)`；④ `Adds(m, m, eps)`。
  - 注意：通用版 `SoftmaxFP32Perf` 是 UB 二维 tile + `tmpReduceBuffer`/`tmpBrcbBuffer` 两个临时块；易错：red 先后承载 max 与 sum，函数返回后不能指望里面还是 max。

- **`RowNormAccum`** [hc_pre_base_arch35.h L1001-L1010](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1001-L1010)
  - 职责：就地 `m /= (行和+eps)`，并把结果累加进列和 csum。
  - 入参/出参：m 就地、red 临时、csum 列和累加器（读写）、t0..t3、eps、preg。
  - 实现：① `RowGroupSumBcast(red, m, ...)`；② `Adds(red, red, eps)`；③ `Div(m, m, red)`；④ `Add(csum, csum, m)`。
  - 注意：通用版把这两步散在 `ReduceSumARAPerf` + `DivABABrcInline` 里；易错：eps 加在分母，而 `SoftmaxRowBcastInplace` 的 eps 加在结果上，两处语义不同。

- **`VFProcessCombFragPacked`** [hc_pre_base_arch35.h L1012-L1096](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1012-L1096)
  - 职责：把 comb_frag 按"每 4 lane 一行"打包进 64 lane 处理 softmax + Sinkhorn。
  - 入参/出参：combFragLocal 输出（行距 `R*R`）、mixLocal 输入（行距 hcMix）、hcBaseLocal、scale、eps、iters、dim0=bs、hcMult=R、hcMix。
  - 实现：① `Arange(vL, 0)` 造 lane id，`ShiftRight(vbs, vL, 2)` + `And(vc, vL, R-1)` 拆行/列，得 gather 索引 `gIdx = vbs*hcMix + vc` 与 scatter 索引 `sIdx = vbs*(R*R) + vc`；② 用 `DataCopyGather` 按 `gIdx` 取 4 个 mix 寄存器、按 `vc` 取 4 组 base 广播；③ 4 次 `SoftmaxRowBcastInplace`，累 `csum` 后统一 `Div`；④ iters 轮 `RowNormAccum` 后 `DataCopyScatter` 写回。数学上仍是 `softmax(mix*scale+base)+eps` 再接 Sinkhorn 行列交替归一。
  - 注意：通用版无（是 `SoftmaxFP32Perf`+`DivABABrcInline` 的寄存器打包重写）；易错：`Duplicate(cShift, 2)` 硬编码 `log2(R)=2`，R 必须为 4；`pack = 64/R`，dim0 非整倍数时靠 `act = dim0*R` 尾掩码，不能传 full mask。

- **`VFElementMajorSum<Pairwise>`** [hc_pre_base_arch35.h L1098-L1111](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1098-L1111)
  - 职责：4 个寄存器逐 lane 求和，`Pairwise` 决定结合顺序。
  - 入参/出参：sum 输出、a..d 输入、mask 掩码。
  - 实现：① `Add(sum, a, b)`；② 模板分支：`Pairwise=true` 先 `Add(tail, c, d)` 再相加，`false` 则顺序 `Add` c、d；③ 保持与串行参考实现一致的浮点累加顺序。
  - 注意：通用版无；易错：顺序不同会带来末位偏差，精度对齐用例会因此挂掉，改分支前先确认参考顺序。

- **`VFElementMajorNormalize<Pairwise>`** [hc_pre_base_arch35.h L1113-L1124](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1113-L1124)
  - 职责：4 个寄存器按 lane 归一：`x/(Σ+eps)`。
  - 入参/出参：a..d 就地、eps、mask。
  - 实现：① 调 `VFElementMajorSum<Pairwise>` 得 sum；② `Adds(sum, sum, eps)`；③ 对 a..d 各做一次 `Div`。
  - 注意：通用版对应 `DivABABrcInline`（用 `src1RepStride=0` 做行广播）；易错：此处 eps 在分母，和 softmax 结尾 +eps 不是一回事。

- **`VFElementMajorSoftmax`** [hc_pre_base_arch35.h L1126-L1152](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1126-L1152)
  - 职责：跨 4 个寄存器（同一 R 轴位置）做 softmax 再 +eps。
  - 入参/出参：a..d 就地、eps、mask。
  - 实现：① `Max` 三次取 4 者最大（`Max(maximum,a,b)`、`Max(tail,c,d)`、再合并）；② `Sub` x4 去最大值保数值稳定；③ `Exp` x4 -> `VFElementMajorSum<true>` -> `Div` x4 -> `Adds(eps)` x4。
  - 注意：通用版 `SoftmaxFP32Perf` 沿物理末维归约；此处"末维"散落在 4 个寄存器里；易错：只对一个寄存器取 max 会算错，最大值必须跨 4 个寄存器合成。

- **`VFElementMajorLoad<Element>`** [hc_pre_base_arch35.h L1154-L1164](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1154-L1164)
  - 职责：按固定 `Element` 取一列混合值，乘 scale 并加偏置。
  - 入参/出参：value 输出、input 基址、bias 基址、index（lane -> 行偏移表）、scale、mask。
  - 实现：① `Gather(value, input + Element, index, mask)`，即 `value[l] = input[Element + l*inputStride]`；② `LoadAlign<float, LoadDist::DIST_BRC_B32>(base, bias + Element/4*C0_SIZE + Element%4)`；③ `Muls(scale)` -> `Add(base)`。
  - 注意：通用版无；易错：偏置表是"4 组 x 8 元素"的打包布局，`Element/4*C0_SIZE + Element%4` 这个换算不能简化成线性下标。

- **`VFProcessCombFragPremixElementMajor`** [hc_pre_base_arch35.h L1167-L1248](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1167-L1248)
  - 职责：premix 版 comb_frag，按 4x4 元素主序做 softmax + Sinkhorn 并压实成 16 列。
  - 入参/出参：mixesLocal 每行 24 个 float（前 8 个是 pre/post，comb 从 `+8` 开始）、bias、scale、eps、iters、rows、hcMix；输出就地复用同一块 UB（行宽 16）。
  - 实现：① `Arange` 造 lane，`Muls(inputIndex, lane, inputStride)`、`Muls(outputIndex, lane, 16)` 两张偏移表；② 16 次 `VFElementMajorLoad<0..15>` 取满 4x4（注释：先读满再压实）；③ 4 次 `VFElementMajorSoftmax`（按行 j）-> 4 次 `VFElementMajorNormalize<false>`（按列 i）-> iters 轮交替；④ 16 次 `Scatter(output + k, m.., outputIndex)` 压实。
  - 注意：通用版无（对应 `SoftmaxFP32Perf`+`DivABABrcInline` 的元素主序重写）；易错：`input = mixAddr + batch*64*inputStride + 8` 的 `+8` 是 pre/post 占位；rows 非 64 整倍数必须用 `UpdateMask<float>(remaining)`。

- **`VFProcessY<T>`** [hc_pre_base_arch35.h L1251-L1303](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1251-L1303)
  - 职责：`y[i][d] = Σ_k mix[i][k] * x[i][k][d]`。
  - 入参/出参：yLocal 输出（T）、mixLocal 权重（行距 hcMixAlign）、xLocal 输入（行距 `RoundUp<T>(d)`）、bs、hcMult、d、hcMix。
  - 实现：① `loopCount > 1`（d > 64）时按列分块，否则单块；② 每 k 一次 `LoadInputDataWithBrc` 广播 `mix[i][k]`，一次 `LoadInputData<T>` 取 `x[i][k][:]`；③ `Mul(x, mix, x)` + `Add(sum, sum, x)` 累加；④ `StoreOutputData<T>` 转回 T。
  - 注意：通用版 `ProcessYHc4` 用 `Brcb` + `MulAddDst` + `BinaryRepeatParams`，且对 hcMult=4 单开特化分支；此处统一走 `Mul`+`Add` 逐 k 累加；易错：x 偏移是 `i*hcMult*dAlign + j*VL + k*dAlign`，漏乘 hcMult 会串到下一行。

- **`CopyIn<T>`** [hc_pre_base_arch35.h L1305-L1321](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1305-L1321)
  - 职责：GM -> UB 的非对齐搬运，按 nBurst 行。
  - 入参/出参：inputGm 源、inputTensor 目的、nBurst 行数、copyLen 每行元素、srcStride 源行间隔（元素）。
  - 实现：① `DataCopyPadExtParams` 全 0（不 pad）；② `DataCopyExtParams`：`blockLen = copyLen*sizeof(T)`、`srcStride = srcStride*sizeof(T)`、`dstStride = 0`；③ `DataCopyPad` 一次下发。
  - 注意：通用版 `CopyIn` 多一个 `dstStride` 入参；arch35 去掉了（UB 侧强制紧凑），需要行距时改用 `CopyInWithUbStride`；易错：`srcStride` 单位是元素，`dstStride` 单位是 32B block。

- **`CopyToL1<T>`** [hc_pre_base_arch35.h L1323-L1327](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1323-L1327)
  - 职责：UB -> L1 的 `DataCopy` 直通封装。
  - 入参/出参：srcTensor、dstTensor、`DataCopyParams`（blockLen/stride 均以 32B 为单位）。
  - 实现：单条 `DataCopy(dstTensor, srcTensor, dataCopyXParams)`，无 MicroAPI、无 barrier。
  - 注意：通用版无；易错：参数已是 block 单位，调用方不要再乘 `sizeof(T)`。

- **`CopyInWithLoopMode<T>`** [hc_pre_base_arch35.h L1329-L1357](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1329-L1357)
  - 职责：用硬件 LoopMode 一次搬完 3D 数据 `(outerLoop, nBurst, copyLen)`。
  - 入参/出参：outerLoop 外层数、nBurst、copyLen、gmLastDim 源最内维长度、srcStride。
  - 实现：① `LoopModeParams`：`loop1Size = outerLoop`、`loop1SrcStride = gmLastDim*sizeof(T)`、`loop1DstStride = nBurst*copyLenAlign*sizeof(T)`、loop2 全置 1/0；② `SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB)`；③ 一次 `DataCopyPad`；④ `ResetLoopModePara`。
  - 注意：通用版用 `CopyInWithOuterFor` 双层 for 循环；此处把循环交给 MTE2；易错：`Set/Reset` 必须成对，漏 Reset 会污染后续所有搬运的 stride。

- **`CopyOut<T>`（UB -> GM）** [hc_pre_base_arch35.h L1359-L1369](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1359-L1369)
  - 职责：UB -> GM 的非对齐搬出。
  - 入参/出参：outputTensor、outputGm、nBurst、copyLen（元素）、dstStride（元素）。
  - 实现：① `DataCopyExtParams`：`blockCount = nBurst`、`blockLen = copyLen*sizeof(T)`、`srcStride = 0`、`dstStride = dstStride*sizeof(T)`；② `DataCopyPad`。
  - 注意：与通用版一致；易错：本文件还有同名 UB->UB 重载（L1403），第三个 stride 参数语义完全不同（元素 vs 32B block），传错不报错但会写花。

- **`CopyInWithUbStride<T>`** [hc_pre_base_arch35.h L1371-L1389](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1371-L1389)
  - 职责：GM -> UB 且 UB 侧按 32B block 预留行距。
  - 入参/出参：`ubDstStrideBlock` 为 UB 侧相邻 block 间隔（32B 单位）、copyLen 元素、srcStride 元素。
  - 实现：① `blockLen = copyLen*sizeof(T)`；② `dstStride = ubDstStrideBlock`（注释强调单位是 32B）；③ `DataCopyPad`，用于把行首 hcMult 个元素铺到 `hcMixAlign` 行距布局。
  - 注意：通用版无（通用版用 `CopyIn` 的 dstStride 按元素换算）；常与 `UbRowGapBlocks` 搭配；易错：UB 侧一块的足迹是 `ceil32(blockLen)`，算间距要按对齐后长度减。

- **`UbRowGapBlocks`** [hc_pre_base_arch35.h L1391-L1398](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1391-L1398)
  - 职责：算出把每行 hcMult 个 float 铺到 hcMixAlign 行距时需要空出几个 32B block。
  - 入参/出参：hcMult 每行有效元素、hcMix 行宽；返回 block 数（给 `CopyInWithUbStride` 用）。
  - 实现：① `rowPitchBytes = RoundUp<float>(hcMix) * sizeof(float)`；② `rowFootprintBytes = CeilAlign(hcMult*sizeof(float), 32)`；③ 返回 `(rowPitch - rowFootprint) / 32`。
  - 注意：通用版无；易错：两个量都是 `uint32_t`，若 hcMult 足迹大于 hcMix 行距会无符号下溢成大数，调用前要保证 hcMult <= RoundUp(hcMix)。

- **`CopyOut<T>`（UB -> UB）** [hc_pre_base_arch35.h L1400-L1413](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h#L1400-L1413)
  - 职责：UB 内对齐搬运，把 `hcMixAlign` 行距的行首 hcMult 元素压实成紧凑布局。
  - 入参/出参：outputTensor 目的、outputGm 源（UB 侧同名参数）、nBurst、copyLen、dstStride/srcStride 均以 32B block 计。
  - 实现：① `blockLen = CeilDiv(copyLen*sizeof(T), BLOCK_SIZE)` 换算成 block 数；② 组装 `DataCopyParams`（`srcStride`/`dstStride` 原样透传）；③ 用 `DataCopy`（非 `DataCopyPad`）下发，配合 VECOUT 队列做异步拷出。
  - 注意：通用版只有 GM 版 `CopyOut`；易错：与 L1359 的重载仅靠实参类型区分（`GlobalTensor` vs `LocalTensor`），stride 语义不同，改调用点时最容易出错。

---

### 二、VF* 函数与通用版 `hc_pre_base.h` 对照表

| 本文件 arch35 函数 | 通用版 `hc_pre_base.h` 对应函数 | 差异要点 |
| --- | --- | --- |
| `VFSigmoid` L134 | `SigmoidPerf` L451 + `CalcDenominator` L435 | RegTensor 上 4 条指令就地完成；`one` 由调用方预置，通用版靠 `Duplicate` + `Div(BinaryRepeatParams)` 的 src0BlkStride=0 复用常量 |
| `VFTransND2NZ` L142 | 无（NZ 在 Matmul/fixpipe 侧完成） | 新增 UB 内 8x8 分形重排，用 `DATA_BLOCK_COPY` 一拍 64 lane；源行距必须自带 +8 padding |
| `VFProcessCast<T>` L197 | `CastTwoDim` L581 | 改为逐 64 lane 掩码 Cast；目的行距 `align+8`，通用版无行尾 padding |
| `VFProcessCastAndInvRmsPart1` L236 | `CastTwoDim` + `ReduceSumARAPerf` L510 / `GatherMaskByDiagonal` L90 | 一趟融合 Cast + 平方和 + 行归约，省掉中间 UB 往返 |
| `VFProcessCastInvRmsAndPremixYPart1` L317 | `CastTwoDim` + `MulABLastDimBrcInline` L112 + `ReduceSumARAPerf` | 三合一；权重用 `DIST_BRC_B32` 单元素广播，替代 `Brcb` |
| `VFProcessPremixYCast<T>` L387 | `CastTwoDim`（反向）L581 | 结构一致，仅行距不同（源按 `yAccRowStride`、目的按 `RoundUp<T>`） |
| `VFProcessInvRmsPart1` L413 | `ReduceSumARAPerf`（单核行内平方和情形）L510 | 就地平方 + `WithUbReduce` 取代多核 K 轴归约；不再需要 mask pattern |
| `VFProcessInvRmsPart2` L488 | 无 | 新增 `1/sqrt(sum+eps)` 的行级收尾，`ReduceSum`+`Adds`+`Sqrt`+`Div` |
| `VFProcessInvRmsPart3WithGroupReduce` L518 | `ReduceSumARAPerf`(dim1=groupK) + `DivABABrcInline` L796 | 归约与归一融合；eps 加在 sqrt 内；groupK>=4 走 4 路展开 |
| `VFProcessInvRmsPart3` L642 | `ReduceSum4x4Perf` L556 + `DivABABrcInline` | 单 k 版，全寄存器；无 `dim2Align` 地址推算 |
| `VFProcessPre` L678 | `ProcessPre` L489 | 不再乘 rsqrt（mix 已预乘）；base 每列块只 Load 一次；行距取 `hcMixAlign` 而非 `curColNumAlign` |
| `VFProcessPost` L732 | `ProcessPost` L727 | 用 `DataCopyUnAlign` 流式取 mix，替代 `needBrc=false` 的广播复用；目的行距 `curColNumAlign` |
| `VFProcessCombFragRLessVL` L793 | `SoftmaxFP32Perf` L773 + `DivABABrcInline` + 迭代 | 全寄存器实现 softmax/列归一/Sinkhorn；靠 `LocalMemBar<VEC_STORE,VEC_LOAD>` 保证回读 |
| `VFProcessIteration` L869 | 无（由循环体抽出） | `ReduceSum`+`Duplicate`+`Adds`+`Div`+`Add` 单行归一累加 |
| `VFProcessCombFragRLessVLUseFourUnfold` L879 | `ReduceSum4x4Perf` + `DivABABrcInline` | 4 行常驻寄存器，迭代过程零 UB 往返；要求 dim1==4 |
| `VFProcessCombFragPacked` L1012 | 无（`SoftmaxFP32Perf`/`DivABABrcInline` 的打包重写） | `Arange` 造索引 + `DataCopyGather/Scatter`；每 4 lane 一行；R 必须为 4 |
| `VFProcessCombFragPremixElementMajor` L1167 | 无 | 元素主序版本，16 个寄存器覆盖 4x4；原地 Scatter 压实到 16 列 |
| `VFProcessY<T>` L1251 | `ProcessY` L696 / `ProcessYHc4` L626 | `Mul`+`Add` 取代 `MulAddDst`；不再按 hcMult==4 分特化路径 |
| `RowGroupMaxBcast` L963 / `RowGroupSumBcast` L975 | `LastDimReduceMaxPerf` L746 / `LastDimReduceSumPerf` L757 | 4 lane 组内归约 + 广播，靠 `DeInterleave/Interleave`，无 `WholeReduceMax/Sum` |
| `SoftmaxRowBcastInplace` L987 | `SoftmaxFP32Perf` L773 | 就地、寄存器内完成，无 tmpReduce/tmpBrcb 两个临时 buffer |
| `RowNormAccum` L1001 | `DivABABrcInline` L796（片段） | 归一与列和累加合成一条链路，省一次 UB 往返 |
| `VFElementMajorSum/Normalize/Softmax/Load` L1098-L1164 | 无 | 元素主序辅助原语：跨 4 个寄存器做 sum/normalize/softmax/gather+bias |

---

### 三、MicroAPI 编程范式速查

#### 3.1 `__VEC_SCOPE__` 里能做什么

- 能写：所有 `RegTensor<T>` / `MaskReg` / `UnalignReg` 的声明，以及以它们为操作数的指令——`Add`/`Sub`/`Mul`/`Div`/`Muls`/`Adds`/`Max`/`Exp`/`Sqrt`、`Duplicate`、`Arange`、`ReduceSum`、`ReduceMax`、`Cast`、超长向量搬运 `DataCopy`/`DataCopyUnAlign`/`DataCopyUnAlignPre`/`DataCopyGather`/`DataCopyScatter`、位置重排 `DeInterleave`/`Interleave`/`UnPack`。
- 能做：把 `LocalTensor.GetPhyAddr()` 转成 `__local_mem__ T*` 后在 scope 内做指针算术；scope 内允许普通 `for` 循环与 `if constexpr`（本文件大量使用模板分支做 fast path）。
- 不能做：不能 `new`/`AllocTensor`/`EnQue`/`DeQue`/`FreeTensor`——这些要在 scope 外的 TPipe 流程里完成；不能调用 Matmul / fixpipe / `DataCopyPad` 之类跨流水接口；不能把 `RegTensor` 声明在 scope 外再进去用（寄存器生命周期绑定 scope，出 scope 即失效）；不能做标量读取（把 UB 值读到标量再参与分支），需要标量判断时必须先落 UB 再出 scope。
- 同步：scope 内对同一块 UB 先写后读（例如把中间结果落盘再读回来做第二轮）必须显式插 `LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>()`；scope 结束时的流水排空交给外层的队列/事件（本文件用 `SetWaitFlag` 或 TPipe 的 EnQue 事件）。

#### 3.2 掩码怎么用

- `CreateMask<float>()`：fp32 全 64 lane 有效，用于"整拍都参与"的场景（如 `pregMain`）。
- `CreateMask<float, MaskPattern::VL1>()`：只有 lane0 有效，专用于"标量搬运"——`Duplicate(one, 1.0f, pregMerge)` 生成常量,`Adds/Sqrt/Div` 只算 lane0，`StoreDist::DIST_FIRST_ELEMENT_B32` 只取 lane0 落盘。
- `UpdateMask<float>(sreg)`：按剩余元素数 `sreg` 生成掩码，并**消耗** `sreg`（每次调用向前推进一个 VL）。所以内层列循环每轮都要"重置 sreg = 列数"再 UpdateMask，本文件在所有 `for (j ...)` 里都把 `sreg = curColNum;` 写在进入内层循环之前。
- 掩码语义：未选中的 lane 在 `Mul/Muls/Add/Adds` 等指令里被置 0（因此累加可以安全地用全掩码，见 `VFProcessCastAndInvRmsPart1`），但 `DataCopy` 的未选中 lane 不会被写；读取侧则可能读到 UB 里的历史脏值，所以"窄掩码取数 + 全掩码归约"的组合是错的。
- 尾部保护：本文件统一用"行距 = `RoundUp<T>(col) + 8`"给每行尾部多留一个 32B block，配合 64 lane 的全掩码读不会越界踩到下一行数据。

#### 3.3 寄存器宽度与分块怎么定

- 宽度常量：`VL_FP32 = 64`（fp32 一拍 64 lane）、`BLOCK_SIZE = 32`（字节）、`C0_SIZE = 8`（fp32 一个 32B block 的元素数）、`FOUR_UNFOLD = 4`、`DOUBLE_BUFFER = 2`。
- 分块原则：凡是"列方向"可变的维度都按 64 分块，`loopCount = CeilDiv(col, VL_FP32)`；列 <= 64 时走单块 fast path（`CreateMask` 一次到位），列 > 64 时走 `UpdateMask` 分块（本文件每个 Process 函数都有 `if (loopCount > 1)` 两套分支，逻辑必须保持一致）。
- 哪些维度被假定 < 64：`hcMix`（组合混合宽度，实际 24）、`hcMult`（=4）、`groupK`（K 轴分核数）、Sinkhorn 的 R 轴 `hcMult`——这些都在注释里点名"小于 64 因此不需要回写 UB / 不需要内层循环"。改 tiling 超过 64 就必须补分块，否则掩码只覆盖一部分。
- 打包技巧：`VFProcessCombFragPacked` 用 `pack = VL_FP32 / R` 把 `pack` 个 token 塞进一拍（每 4 lane 一个 token）；`VFProcessCombFragPremixElementMajor` 用 `64` 个 lane 对应 64 个 token、每个 token 用一个寄存器的一个 (i,j) 位置。选择哪种布局取决于后续归约是"跨 lane 4 个"还是"跨寄存器 4 个"。
- 数据流：GM -> UB（`CopyIn*`）-> 寄存器（`LoadInputData*`）-> 计算 -> UB（`StoreOutputData`）-> GM（`CopyOut*`）。寄存器之间不能直接搬张量，Any 中间结果必须经过 UB，中间结果跨行复用就需要 `LocalMemBar`。

#### 3.4 常见坑（本文件内真实存在的）

- `UpdateMask` 的 `sreg` 会被消耗：内层循环忘记重置，第二行开始掩码全 0。
- `curColNumAlign` 在 `VFProcessPre`/`VFProcessPost` 里声明却未参与地址计算（行距实际是 `hcMixAlign`），照抄会错。
- `DIST_PACK_B32` / `DIST_UNPACK_B16` 必须与 `castTraitB322B16Even` / `castTraitB162B32Even` 成对使用（前者 `SatMode::NO_SAT`，后者 `SatMode::UNKNOWN`），混用会出饱和或舍入差异。
- `DataCopyUnAlign` 之前必须有 `DataCopyUnAlignPre`；且每次换列块都要把源指针重置回原点，不能让指针跨块漂移。
- `DataCopyPad` 的 `srcStride`/`dstStride` 单位不一致（GM 侧元素、UB 侧 32B block），本文件为此专门拆出 `CopyInWithUbStride` + `UbRowGapBlocks`。
- `VFTransND2NZ` 的注释写"16 对齐"但代码用 `C0_SIZE = 8`，以代码为准；同时源行距必须自带 +8 元素 padding。
- 浮点累加顺序会影响精度用例：`VFElementMajorSum<Pairwise>` 与 4 路展开的 tail 分支都刻意固定了顺序，改动前先跑对齐用例。

#### 3.5 典型走读：`VFProcessInvRmsPart2`（L488-L513）

- 第 1 步（scope 外）：`yLocal.GetPhyAddr()` / `xLocal.GetPhyAddr()` 转成裸指针，再算行距 `curColNumAlign = RoundUp<float>(curColNum)`。所有地址与步长必须在进 scope 前算好，scope 内只做寄存器算术。
- 第 2 步（scope 内声明）：`RegTensor<float> x, sum, one, y;` 与两个掩码。注意 `pregLoop` 由 `UpdateMask<float>(sreg)` 在循环外一次性算好——因为这个函数每行只有一拍、不存在"内层循环被消耗"的问题。
- 第 3 步：`pregMerge = CreateMask<float, MaskPattern::VL1>()` 只留 lane0；`Duplicate(one, 1.0f, pregMerge)` 因此只有 lane0 是 1，其余 lane 是脏值或 0——后续 `Div(y, one, sum, pregMerge)` 也只在 lane0 计算，所以脏 lane 不会污染结果。
- 第 4 步：行循环里 `LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign)` 取一整行；`ReduceSum(sum, x, pregLoop)` 把 64 lane 归约成标量并广播回所有 lane（MicroAPI 的 Reduce 是"归约 + 广播"，不是只留一个元素）。
- 第 5 步：`Adds(sum, sum, eps)` → `Sqrt` → `Div(y, one, sum)`，三拍都是窄掩码，最后 `DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(yLocalAddr + i, y, pregMerge)` 只把 lane0 写到 `yLocal[i]`。
- 易错对照：如果把 `pregMerge` 换成 `pregLoop`，`one` 会在所有有效 lane 上都是 1（`Duplicate` 会广播），结果 `y[i]` 仍然只写 lane0，看起来"没坏"；但一旦列数 < 64，`pregLoop` 的无效 lane 参与 `Div` 就可能触发无效值传播，属于潜在坑。

#### 3.6 一次前向的完整数据流（函数调用顺序）

- 搬入：`CopyIn` / `CopyInWithLoopMode` / `CopyInWithUbStride`（GM → UB）。
- Stage1（x 量化与 RMS）：`VFProcessCastAndInvRmsPart1` 一趟出 xCast 和平方和；需要 NZ 布局时接 `VFTransND2NZ`；premix 路线改走 `VFProcessCastInvRmsAndPremixYPart1`，之后用 `VFProcessPremixYCast` 落盘。
- Stage2（跨核的 K 轴归约）：`VFProcessInvRmsPart2` 先把每个核的 (bs, k) 压成 rsqrt，再由 `VFProcessInvRmsPart3WithGroupReduce`（groupK > 1）或 `VFProcessInvRmsPart3`（单核）合成 (bs, hcMix)。
- Stage3（产出 pre / y / post）：`VFProcessPre` → `VFProcessY` → `VFProcessPost`；三者的输入都依赖 Stage2 的 rsqrt 与 hcMix 行距布局，所以"行距 = `hcMixAlign`"这条约定贯穿全程。
- Stage4（comb_frag / Sinkhorn）：`VFProcessCombFragRLessVL`（通用）、`VFProcessCombFragRLessVLUseFourUnfold`（dim1=4 展开）、`VFProcessCombFragPacked`（4 lane 一行打包）、`VFProcessCombFragPremixElementMajor`（元素主序），内部共用 `RowGroupMaxBcast` / `RowGroupSumBcast` / `SoftmaxRowBcastInplace` / `RowNormAccum` / `VFProcessIteration` / `VFElementMajor*`。
- 搬出：`CopyOut`（UB → GM）或用 UB → UB 重载 + `UbRowGapBlocks` 先压实再异步拷出。

#### 3.7 通用版（UB 二维 API）→ arch35（MicroAPI）心智映射

| 通用版写法 | arch35 写法 | 关键差别 |
| --- | --- | --- |
| `Cast(output, input, RoundMode, count)` | `LoadInputData` + `Cast` trait | arch35 用 `CastTrait` 指定 RegLayout/SatMode/MergeMode/RoundMode，一次只处理 64 lane |
| `Brcb(tmp, src, repeatTimes, {1,8})` | `LoadInputDataWithBrc`（`DIST_BRC_B32/B16`） | Brcb 在 UB 里铺开，BRC 直接在加载时广播，省一块临时 buffer |
| `Mul/Add/Exp` 带 `calCount`、`BinaryRepeatParams` | `Mul/Add/Exp` 带 `MaskReg` | 二维 API 靠 blk/rep stride 描述"形状"，MicroAPI 靠掩码描述"有效 lane" |
| `WholeReduceMax/WholeReduceSum` + `ReduceOrder` | `ReduceMax/ReduceSum`（归约+广播） | 前者把结果写 UB（一个值），后者留在寄存器且自动广播到整拍 |
| `PipeBarrier<PIPE_V>()` | `__VEC_SCOPE__` 边界 + `LocalMemBar` | scope 内不再需要逐指令 barrier，只在"写 UB 再读回"处加 `VEC_STORE→VEC_LOAD` |
| `GatherMask` + mask pattern buffer | `Gather` / `DataCopyGather` + 索引寄存器 | 索引由 `Arange` + 整数运算现算，不再需要预生成掩码 pattern |
| `DataCopyParams` 的 block 单位 | `DataCopy` 元素偏移 + `MaskReg` | 寄存器搬运的偏移直接按元素加，不再换算 block |

#### 3.8 新增或改造一个 VF 函数的检查清单

1. 先确定哪些维度 < 64、哪些维度需要 `CeilDiv(col, 64)` 分块，两套分支都要写，且结果必须一致。
2. 所有行距在进 scope 前算好；凡是"后续会被 64 lane 全掩码读"的 buffer，行距统一 `RoundUp<T>(col) + 8`。
3. 每层内层循环前重置 `sreg`；归约前确认无效 lane 已被置零（`Mul/Muls`）或已被掩码排除。
4. 需要"取值/存值"的标量结果一律 `MaskPattern::VL1` + `DIST_FIRST_ELEMENT_B32` 配对使用。
5. 跨行/跨轮复用的中间结果必须经过 UB，并在写后读之间插 `LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>()`。
6. 16bit 场景的两套 `CastTrait`（`B162B32Even` / `B322B16Even`）要与 `DIST_UNPACK_B16` / `DIST_PACK_B32` 严格配对。
7. 与通用版对拍时先比 `align+8` 的行距假设，再比累加顺序（浮点末位差异基本都来自这里）。

#### 3.9 术语小抄

- lane：向量寄存器的最小并行单位，fp32 下一拍 64 个。
- VL / `VL_FP32`：一拍能处理的元素数，本文件固定 64。
- Block：32 字节，`BLOCK_SIZE`；fp32 一个 Block 装 8 个元素（`C0_SIZE`）。
- UB：片上统一缓冲，`LocalTensor` 所在处；GM 与寄存器之间的唯一中转。
- 掩码（MaskReg）：按 lane 的位置位图，决定"这一拍哪些 lane 参与"。
- trait：`CastTrait` / `LoadDist` / `StoreDist` 这类"编译期策略参数"，决定搬运时的布局解释方式。

#### 3.10 数值与性能上的取舍（代码里能看到的证据）

- `ReduceSum` 是"归约 + 广播"，所以本文件大量出现 `ReduceSum` 紧跟 `Duplicate` 的组合（如 `VFProcessCombFragRLessVL`、`VFProcessIteration`）——其实 `ReduceSum` 已经把结果广播了，`Duplicate` 属于冗余但无害的显式化；改代码时删不删都不影响结果。
- `Muls`/`Adds` 传标量比先 `Duplicate` 成一个寄存器再 `Mul`/`Add` 少一条指令，本文件在 `scale`、`eps`、`2.0`、`-1` 这些常数上一律用 `Muls`/`Adds`。
- `VFProcessCombFragRLessVLUseFourUnfold` 与 `VFProcessInvRmsPart3WithGroupReduce` 的 4 路展开都是为了"把独立的寄存器链填满指令级并行"，属于架构相关优化，不是数学变换，输出必须逐位一致。
- `VFProcessCombFragPacked` / `VFProcessCombFragPremixElementMajor` 用 `Gather`/`Scatter` 换来"行距无关"的布局自由度，代价是每拍要先算索引寄存器；索引是常量表，务必在列/行循环外算一次。
- 所有 `StoreOutputData` 都带掩码，因此"写回"天然只写有效列；但 `DataCopy` 的写粒度仍是 32B block，行距不足时相邻行会互相覆盖，这是 `RoundUp` 必须成对出现的原因。

#### 3.11 搬运类参数的统一规律

| 接口 | srcStride 单位 | dstStride 单位 | 说明 |
| --- | --- | --- | --- |
| `CopyIn` L1305 | 元素 | —（固定 0） | UB 侧紧排 |
| `CopyInWithUbStride` L1371 | 元素 | 32B block | UB 侧留行距 |
| `CopyInWithLoopMode` L1329 | 元素（loop1 用 `gmLastDim*sizeof(T)`） | 字节（`nBurst*align*sizeof(T)`） | LoopMode 参数是字节 |
| `CopyOut`(GM) L1359 | —（固定 0） | 元素 | 注意与 UB 版重载区分 |
| `CopyOut`(UB) L1400 | 32B block | 32B block | `blockLen` 也是 block 数 |
| `CopyToL1` L1323 | `DataCopyParams` 原样 | 同左 | 全部按 block 传入 |

#### 3.12 与调用方约定的隐含契约

- 行距：凡是被本文件函数写过、又被本文件函数按 64 lane 全掩码读回的 buffer，行距一律 `RoundUp<T>(col)`（16bit）或 `RoundUp<float>(col) + 8`（fp32 出参）。
- 尺寸：`hcMix`、`hcMult`、`groupK`、Sinkhorn 的 R 轴都必须在 64 以内；`dim1 == 4` 是 `VFProcessCombFragRLessVLUseFourUnfold` 与 `VFProcessCombFragPacked` 的硬前提。
- 布局：`mix` 行内排布为 `pre(hcMult) | post(hcMult) | comb(hcMult*hcMult)`，premix 元素主序路径从 `+8` 处开始读 comb 段（`hcMix = 24` 时成立）。
- 指针：`__local_mem__` 指针不能在跨函数调用之间保存，所有函数都在入口重新用 `GetPhyAddr()` 取，避免 UB 地址空间重排后失效。


---

## 15. 函数级详解：通用版主流程（`hc_pre.cpp` + `hc_pre_m_k_split_core.h`）

> 本文件名里的 `m` / `k` 指矩阵乘的 M 维（token 数 `bs`）与 K 维（`hc_mult × d`），**不是** mHC 的 m。先读 4.4.1 再往下看，否则 `kBlkDimIdx_`、`multCoreSplitMSize` 这些字段会一直对不上号。

目标读者：完全没写过 Ascend C。下文先给最小必要背景，再按源码出现顺序逐个函数讲，最后给成员划分、端到端时序和三块 workspace 的交接。

全文条目：39 个函数 + 3 个 TilingKey 分支段 = 42 条。

范围说明：本篇只讲 `hc_pre.cpp` 与 `hc_pre_m_k_split_core.h` 这两个文件，即 A2/A3 的通用（非 `__DAV_C310__`）路径。`HcCubeCompute`（`hc_pre_cube_compute.h`）与所有数学辅助（`hc_pre_base.h`）只在调用点按语义说明，不逐函数展开；950 系列的 `*_arch35.h` 不在本路径内。行号以当前仓库文件为准。

理解成本最高的三处（先有个预期）：一是 Part1 里 Cube/Vector 双生产者-消费者循环（旗标 8/9 + ws1 双缓冲）；二是 Part2 里同一个类运行期两套 UB 布局；三是 Sinkhorn 的"元素主序 ↔ 矩阵序"两次重排（`[16][row]` 与 `[row][16]`）。其余函数基本是"搬入 → 归约 → 广播运算 → 搬出"的重复套路。

---

### 0. 背景：这个算子与这些名词

算子在做什么（一句话）：先算 `mix = x · hc_fnᵀ`（Cube 做矩阵乘，N=24 列），再把 24 列切成 mix0(4)/mix1(4)/comb(4×4) 三段，分别产出 `pre`、`y`、`post`、`comb_frag`，其中 comb 还要做 Sinkhorn 的行/列交替归一化。

算子 I/O 一览（`bs` 是本次 token 行数，`d` 是 hidden 维，`hcMult=4`）：

| 张量 | 形状 | 类型 | 角色 |
|---|---|---|---|
| `x` | `[bs, hcMult, d]` | bf16 | 输入。三维排布，所以行距是 `hcMult*d`，同一行的 4 个 hc 分量在内存里是 4 段长度为 d 的连续数据 |
| `hc_fn` | `[hcMix=24, d*hcMult]` | float | 权重矩阵，即 `mix = x·fnᵀ` 里的 fn；K 维是 `d*hcMult` |
| `hc_scale` | `[3]` | float | 三个标量：pre 的 scale、post 的 scale、comb 的 scale（`GetValue(0/1/2)`） |
| `hc_base` | `[24, 4]` | float | 行 0~3 = base0（给 pre）、4~7 = base1（给 post）、8~23 = base2（给 comb 的 4×4） |
| `pre_mix` | `[bs, hcMult]` | float | 仅 TK2 使用：外部给的 pre 权重，会替代本算子算出的 pre 去加权 y |
| `y` | `[bs, d]` | bf16 | 主输出：对每行把 4 个 hc 分量按 pre 权重加权求和 |
| `post` | `[bs, hcMult]` | float | 输出：sigmoid 类变换后再 ×2 |
| `comb_frag` | `[bs, hcMult, hcMult]` | float | 输出：4×4 转移矩阵，经 softmax + Sinkhorn 行/列交替归一化 |
| `pre` | `[bs, hcMult]` | float | 仅 TK2 且 `hasPreOut!=0` 时输出 |
| `workspace` | 由 tiling 决定 | float | 框架传入，进程内切 ws1/ws2/ws3 三段（见第 6 节） |

两个 Stage 的分工：Stage1（Part1）用 Cube 算 `mix`，把结果和 x 的平方和落 workspace；Stage2（Part2）用 Vector 把 24 列拆开，做 sigmoid/softmax/Sinkhorn 这些"逐元素或逐行"的数学。

| 名词 | 含义 | 在本文里的落点 |
|---|---|---|
| AIC / AIV | Cube 核 / Vector 核。本算子声明 `KERNEL_TYPE_MIX_AIC_1_2`，1 Cube 配 2 Vector | `CV_RATIO=2`；AIV 的 blockIdx 要 `/2` 才等于对应 Cube 编号 |
| GM | 全局显存，`GlobalTensor<T>`，算子入参与 workspace 都在这里 | `xGm/yGm/workspaceGm/...` |
| UB | Vector 核片上暂存，`TBuf`/`TQue` 从这里划 | `reduceBuf/rsqrtBuf/xQue/...` |
| L1 / L0A / L0B / L0C | Cube 核的多级片上缓存，矩阵乘的 A/B 与累加器 | `cubeCompute_` 内部（不在本篇两个文件里） |
| TQue | 带双缓冲的搬运队列，仪式是 `AllocTensor→EnQue→DeQue→FreeTensor` | `xQue/mmInQue/mixesQue01/mixesQue2/...` |
| TBuf | 纯计算缓冲，`InitBuffer` 后 `Get<float>()` 一次拿句柄 | `rsqrtBuf/rowBrcbBuf0/...` |
| TPipe | UB 分配器。一个 TPipe 对应一套 UB 划分 | 入口里 `pipe`（Stage1）与 `pipeStage2`（Stage2） |
| CrossCore 旗标 | AIC 与 AIV 之间的显式同步，`CrossCoreSetFlag/WaitFlag` | 号 8（AIV→AIC）、9（AIC→AIV） |
| `SyncAll<false>()` | 同簇内全 cv 同步，是 Part1→Part2 的数据可见性闸门 | `Process` 末尾 |
| tiling | Host 算好、随 kernel 传入的尺寸/切分参数 | `HcPreTilingData`，字段如 `mL1Size/cvLoopKSize/cubeBlockDimK` |
| bf16 / fp32 | 输入输出用 bf16，中间计算统一 fp32 | `T` 与 `float` |
| NZ 格式 | Cube 输出常见的 16×16 分块布局，对角元素散落在块内 | 平方和 16×16 → 需要 `GatherMaskByDiagonal` |
| mmLastAxisSize | ws2 的行距，`CeilAlign(hcMix, 128)` 个 float（128 float = 512B） | Part2 的所有 ws2 寻址 |

关键常量（`hc_pre_base.h`）：`hcMult=4`、`hcMultAlign=8`、`hcMix=24`（=2×4+4×4）、`N_SIZE=24`、`SQUARE_SUM_SIZE=16`、`COMPACT_HC_MULT=4`、`COMPACT_MATRIX_SIZE=16`、`BLOCK_SIZE=32B`、`MM_CACHE_LINE_BYTES=512B`、`WORKSPACE_ALIGN_SIZE=512B`、`DOUBLE_BUFFER=2`、`CV_RATIO=2`、`MASK_PATTERN_DIM_SIZE=16`。

读代码时最容易懵的三件事：

1. "同一个函数里 AIC 和 AIV 走完全不同的分支"——`if ASCEND_IS_AIC { ... } else { ... }` 是编译期/运行期双重分流，两段代码几乎无共享逻辑。
2. "偏移一律是元素数，只有 `CopyIn/CopyOut` 的 stride 参数是字节数"——`dstStride` 单位是字节，`mmLastAxisSize` 一类是 float 元素数，混用必错。
3. "Stage1 与 Stage2 之间只有 workspace 和 SyncAll"——没有中间 kernel，也没有隐式 flush。

---

### 1. hc_pre.cpp：入口与三个 TilingKey 分支

- **`hc_pre`** [hc_pre.cpp L34-L140](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp#L34-L140)
  - 职责：Kernel 入口，按 TilingKey 分派两段式 Stage1/Stage2。
  - 入参/出参：10 个 `GM_ADDR`（x、hc_fn、hc_scale、hc_base、pre_mix、y、post、comb_frag、pre、workspace）+ tiling；无返回值。
  - 实现：①先判空 `workspace`、再 `GetUserWorkspace(workspace)` 取用户区 `userWs`（框架在头部留了簿记区，直接拿 `workspace` 当用户区会整体错位）；②`TPipe pipe` 是 Stage1 的 UB 分配入口；③`TILING_KEY_IS(0/1/2)` 三个 A2/A3 分支，每个分支都是"Part1 跑完 → `pipe.Destroy()` → 新建 `pipeStage2` → Part2 跑完 → 再 Destroy"；④`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 声明核型，框架据此按 1 Cube : 2 Vector 起核。
  - 注意：必须两个 TPipe——Stage1/Stage2 的 UB 预算完全不同，用同一 pipe 二次 `InitBuffer` 会撞地址；`__DAV_C310__` 的 1000~1003 分支属于 950 系列，不是本篇的 A2/A3 路径，读的时候别串了。

- **分支 `TILING_KEY_IS(0)`** [hc_pre.cpp L92-L106](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp#L92-L106)
  - 职责：默认两段式，Part2 走"按核角色二分"的紧凑路径。
  - 入参/出参：Part1(x, hc_fn, userWs)；Part2(x, hc_scale, hc_base, y, post, comb_frag, pre_mix, pre, userWs)。
  - 实现：①`GET_TILING_DATA_WITH_STRUCT` 把 tiling 拷到栈上结构体，取 `const __restrict` 指针传给算子，避免每条指令重复读 GM；②`HcPreMembaseKSplitCorePart1<DTYPE_X>` 负责算 matmul 中间量并写 ws；③`HcPreMembaseKSplitCorePart2<DTYPE_X, false>` 的 `false` 是 `useOriginal` 模板参数，决定走"pre/post 核 + sinkhorn 核"的二分实现。
  - 注意：此分支 `pre_mix` 只是占位实参，Part2 模板的 `InitGlobalBuffers` 根本不绑定它，别以为 TK0 会回写 pre_mix；`pre` 同理（TK0 的 Part2 也不写 pre）。

- **分支 `TILING_KEY_IS(1)`** [hc_pre.cpp L107-L121](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp#L107-L121)
  - 职责：同 TK0，但 Part2 用 `useOriginal=true` 的单核全流程实现。
  - 入参/出参：参数与 TK0 完全一致，唯一差别是模板实参 `HcPreMembaseKSplitCorePart2<DTYPE_X, true>`。
  - 实现：①Part1 代码与 TK0 逐字相同（Stage1 不感知下游走哪条路）；②模板参数在 `Init` 里选中 `InitOriginalBuffers`，在 `Process` 里直接转 `ProcessOriginal`；③comb 走 `hcMult×hcMult`（4×4）通用形状，而不是 compact 的 16 元素重排路径。
  - 注意：TK1 不按核角色切分，`secondUsedCoreNum` 个 AIV 各自把 pre/y/post/comb 全算一遍，并行度只靠"按行切批"，因此单核 UB 压力最大、适合小 batch。

- **分支 `TILING_KEY_IS(2)`** [hc_pre.cpp L122-L138](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp#L122-L138)
  - 职责：preMix 场景，Part1 不变 + 专用 `HcPreMembaseKSplitCorePremixPart2`。
  - 入参/出参：Part2 的 `preMix`、`pre` 是真实输入/输出指针（与前两个分支相反），加上 x/hc_scale/hc_base/y/post/comb_frag/userWs。
  - 实现：①Part1 与 TK0 是同一份代码，完全不感知 premix；②Part2 换成 premix 类，`Init` 里多绑两个 GM；③`Process` 是单核全流程，按 `hasPreOut` 决定是否写 `preGm`、按 `hasPreMix` 决定 y 的权重取自 `preMixLocal` 还是算出来的 `mixes01ReduceLocal`。
  - 注意：三个分支共用同一份 tiling 结构体，只靠 tilingKey 区分语义；`hasPreMix/hasPreOut` 是运行期字段而非模板参数，所以两种模式编译进同一个 kernel 二进制，靠取值分流。

---

### 2. hc_pre_m_k_split_core.h

Stage1 只做一件事：`mix = x · hc_fnᵀ`，同时把每行 x 的平方和（RMSNorm 的分子）一起算出来。它的多核切法是"m 方向按行切、k 方向按归约维切"：`cubeBlockDimK` 个核平分 K，每个核把自己的部分和写进 ws2/ws3，Stage2 再把 k 块维加回去。

具体切分：总核数 `cubeCoreNum = cubeBlockDimM * cubeBlockDimK`；每个核拿 `multCoreSplitMSize` 行、`multCoreSplitKSize` 列（尾核 clamp）。M 方向还要再按 `mL1Size` 切一层内循环（一个 m-chunk 一次 MMAD 的 L1 容量），K 方向按 `cvLoopKSize` 切"cast 粒度"（一次 Vector 能处理多少 K）。所以 Part1 是 **三层循环**：`m-chunk`（外层，`Process`）→ `k tile`（中层，`ProcessMChunk` 的 for）→ `L1/L0 tile`（内层，在 `HcCubeCompute` 里）。

因为 Cube 不能直接吃 bf16 的 x（要 fp32 且要 NZ 布局），中间的"bf16→fp32 + 落 GM"由同一簇的 2 个 AIV 完成，这就是 ws1 存在的理由：**它是 Vector 写给 Cube 的交接板**。

#### 2.1 HcPreMembaseKSplitCorePart1（Stage1）

- **`HcPreMembaseKSplitCorePart1()`** [hc_pre_m_k_split_core.h L28-L30](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L28-L30)
  - 职责：默认构造函数，占位，不做任何初始化。
  - 入参/出参：无。
  - 实现：①函数体为空；②所有成员（含 `pipe`、`tilingData`）都靠 `Init` 赋值；③`cvLoopIdx_`、`kBlkDimIdx_` 依赖类内默认初始化 `=0`。
  - 注意：构造期间不能碰 `tilingData`，此时是未初始化指针；也不要在构造里 `InitBuffer`，那时 pipe 还没绑定。

- **`Init`** [hc_pre_m_k_split_core.h L32-L72](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L32-L72)
  - 职责：绑 GM、按 Cube 核切 ws1/ws2/ws3、AIV 与 AIC 分别建资源。
  - 入参/出参：x、hcFn、workspace（已去掉框架头的用户区）、tiling 指针、TPipe 指针；出参是全部成员状态。
  - 实现：①`curCubeBlockIdx=GetBlockIdx()`，若是 AIV 再 `/CV_RATIO`，保证 2 个 AIV 与 1 个 AIC 指向同一片 ws1；②`xCastFp32BufSize_ = mL1Size * CeilAlign(cvLoopKSize,128)`，`ws1 = workspace + curCubeBlockIdx * 该值 * DOUBLE_BUFFER`（每核两槽，供 `cvLoopIdx_%2` 交替）；③`ws2` 起点 = `cubeCoreNum * 单核ws1`，大小 = `cubeBlockDimK * bs * CeilAlign(N_SIZE,128)`，再按 512B 向上对齐；④`ws3` 紧接 ws2，大小 = `CeilAlign(bs,16) * 16`；⑤AIC 只 `cubeCompute_.Init(ws1, hcFn, pipe)` 后 `return`，AIV 才建 `xQue`（`QuePosition::VECIN`）与 `mmInQue`（`VECOUT`），各 2 槽。
  - 注意：ws1 的核偏移按 Cube 编号算，两个 AIV 共用同一块，靠行奇偶拆开写；ws2/ws3 不给单核偏移，Part2 用"全局基址 + globalRowOffset"自行寻址；`xQue` 容量是 `stage1MFactor * RoundUp<T>(cvLoopKSize)`，与 ws1 单槽的 `mL1Size` 无关。

- **`GetMkBlockInfo`** [hc_pre_m_k_split_core.h L74-L96](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L74-L96)
  - 职责：把一维 blockIdx 折成 (m 块, k 块) 网格，输出 GM 行列范围。
  - 入参/出参：出参 5 个引用——`mGmBaseOffset/mGmEndOffset/kGmStartOffset/kGmEndOffset/curVectorBlockIdx`。
  - 实现：①AIV 视角先 `/NUM_TWO` 折成 Cube 视角，同时把原始 AIV 编号留给调用者（后面拆半行要用）；②`mBlkDimIdx = idx / cubeBlockDimK`、`kBlkDimIdx = idx % cubeBlockDimK`，后者写入成员供 ws2/ws3 寻址；③k 范围 = `multCoreSplitKSize * kBlkDimIdx`，尾块 clamp 到 `k`；④m 范围 = `multCoreSplitMSize * mBlkDimIdx`，尾块 clamp 到 `bs`。
  - 注意：k 是"除/模"的低维（连续核号对应连续 k 块），m 是高位；`kGmEnd` 用 `k`、`mGmEnd` 用 `bs`，两个维度名字相似但含义完全不同，写反不会报错只会算错。

- **`ProcessMChunk`** [hc_pre_m_k_split_core.h L100-L187](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L100-L187)
  - 职责：处理一个 m-chunk 内的整个 k 循环；AIC 做 MMAD，AIV 做 cast 落 ws1。
  - 入参/出参：m 起止、k 起止、AIV 原始 blockIdx；无返回值，副作用是写 ws1/ws2/ws3。
  - 实现：①`realMSize = min(mL1Size, mEnd-mOffset)`，k 以 `cvLoopKSize` 步进，`realKGmSize` 同样对尾块 clamp；②AIC 分支填 `MmParams`（`curML1/curKL1/curNL1=N_SIZE/singleCoreK/xWsKSize=cvLoopKSize/nOutSize/kGmSize=k/isFirstK/isLastK`），首块先 `WaitBL1Mte1ToMte2Flag()+CopyInB1`，再 `CrossCoreWaitFlag(SYNC_AIV_TO_AIC_FLAG)` 等 AIV 写完本块 fp32，调 `ComputeDecode(ws1[cvLoopIdx_%2], ws1[(cvLoopIdx_+1)%2], ws3[…], ws2[…], …)`，收尾 `SetBL1Mte1ToMte2Flag()` 并 `CrossCoreSetFlag<SYNC_MODE2,PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG)`；③AIV 分支先 `CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG)`，把 `realMSize` 按奇偶拆半（偶数核取前 `(n+1)/2` 行、奇数核取剩余行），再在 `stage1MFactor` 行块内 `Alloc xQue → CopyIn（行距 k、行内 realK、尾部 pad k-realK）→ EnQue → DeQue → CastTwoDim 到 mmInQue → DeQue → CopyOut 到 ws1`（dst 行距 = `cvLoopKSize`，把 realK 之后的空洞补齐）；④每轮末 `cvLoopIdx_++`。
  - 注意：ws1 槽位由 `cvLoopIdx_` 跨 m-chunk 连续递增决定，所以 `Process` 的循环里绝不能重置它；`b1Prefetched/a1Prefetched` 每进 chunk 重置为 false 表示"本 chunk 的首块还没预取"，置位后写回的是 `preloadNextB1` 而非常量；AIV 拆半必须满足两半之和恰为 realMSize（`mVectorLength` 先算上取整的半，再让奇数核取差）。

- **`Process`** [hc_pre_m_k_split_core.h L189-L214](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L189-L214)
  - 职责：Part1 总控：发前置旗标 → m 两层循环 → 收尾同步。
  - 入参/出参：无；依赖 `Init` 后的成员。
  - 实现：①AIC 先连发**两次** `CrossCoreSetFlag<SYNC_MODE2,PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG)`，与 AIV 每轮一次的 Wait 配对（首批需要"预置信用"，否则 AIV 第一轮就卡住）；②`GetMkBlockInfo` 只算一次块范围；③`for mGmOffset` 以 `mL1Size` 步进调 `ProcessMChunk`；④AIC 调 `cubeCompute_.End()` 冲干尾批 MMAD，AIV 连 `CrossCoreWaitFlag` 两次把残留旗标吃掉，最后 `SyncAll<false>()` 做全 cv 同步。
  - 注意：前置 2 次 Set 与收尾 2 次 Wait 必须严格成对，少一次死锁、多一次会把旗标"溢出"到 Part2；`SyncAll<false>()` 是 Part1→Part2 唯一的数据可见性保证，删掉它 ws 可能还没落地就被读。

#### 2.2 HcPreMembaseKSplitCorePart2（Stage2）

这个类同时承担两种完全不同的工作，靠运行期核编号区分：`blockIdx < prePostCoreNum` 的核算 pre/y/post，其余的核算 4×4 的 comb + Sinkhorn。`useOriginal=true` 时两者合一（TK1）。

- **`HcPreMembaseKSplitCorePart2()`** [hc_pre_m_k_split_core.h L251-L253](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L251-L253)
  - 职责：默认构造，占位。
  - 入参/出参：无。
  - 实现：①函数体为空；②UB 布局必须等 `Init` 知道自己是 pre/post 核还是 sinkhorn 核后才能定；③成员 `LocalTensor` 只在对应 Init 分支里 Get，另一分支里保持未绑定。
  - 注意：不要以为"一个类一套 UB"——同一个类在运行期有两套互斥布局，同时 `InitBuffer` 同名 TBuf 会失败。

- **`InitGlobalBuffers`** [hc_pre_m_k_split_core.h L255-L266](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L255-L266)
  - 职责：把 7 个 GM 地址绑到 `GlobalTensor`。
  - 入参/出参：x/hcScale/hcBase/y/post/combFrag/preMix/pre/workspace；出参为 GM 成员。
  - 实现：①逐个 `SetGlobalBuffer((__gm__ T*)…)`，x/y 用模板类型 T（bf16），其余按 float；②`preMix/pre` 两个形参被忽略，此模板路径不产这两个输出；③`workspaceGm` 是纯 float 视图，Part1 写的也是 float，无类型转换。
  - 注意：workspace 必须传"用户区"基址（entry 里的 `userWs`），否则与 Stage1 的偏移计算整体错位；这里不校验空指针，依赖 entry 的前置判空。

- **`InitCommonBuffers`** [hc_pre_m_k_split_core.h L268-L283](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L268-L283)
  - 职责：建 pre/post 与 sinkhorn 两条路径共用的归约/广播/掩码缓冲。
  - 入参/出参：`stage1UsedCoreNum`（k 块数，即 Part2 的归约轴长度）、`rowFactor`（本路径的批行数上限）。
  - 实现：①`squareSumQue` 建 2 槽，容量 `stage1UsedCoreNum*rowFactor*SQUARE_SUM_SIZE` 个 float；②`squareReduceBuf/rsqrtBuf/rowBrcbBuf0/maskPatternBuf` 四个 TBuf；③`rowBrcbBuf0` 按 `RoundUp<float>(rowFactor)*BLOCK_SIZE` 字节建，是"每行一个 32B block"的广播原料；④`SetGatherMaskPattern(mask, rowFactor+15)` 生成循环对角掩码并 `PipeBarrier`。
  - 注意：掩码行数必须是 `rowFactor + MASK_PATTERN_DIM_SIZE - 1`，因为 `GatherMaskByDiagonal` 用 `globalRowOffset % 16` 选相位，少分配 15 行就会读越界；这也是三条路径都先调本函数的原因。

- **`InitPrePostBuffers`** [hc_pre_m_k_split_core.h L285-L318](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L285-L318)
  - 职责：pre/post 核的 UB 布局（行向量批量 + y 单行批量）。
  - 入参/出参：`stage1UsedCoreNum`。
  - 实现：①`rowFactor = prePostRowFactor`、`yRowFactor = stage2RowFactor`，两条流的 batch 不同；②`InitCommonBuffers(rowFactor)` 复用公共缓冲；③`mixesQue01` 2 槽容 `stage1UsedCoreNum*rowFactor*hcMultAlign*2`（mix0/mix1 并存），`xQue` 容 `yRowFactor*hcMult*RoundUp(dFactor)`，`yQue` 2 槽；④`postQue` 按 `prePostRowFactor*hcMultAlign` 建，因为 `ProcessPost` 是整批处理；⑤`hcBaseBuf0/1`、`hcBrcbBuf1`、`mixes01ReduceBuf`、`xCastBuf`、`yCastBuf` 走 TBuf 并立刻 Get。
  - 注意：x/y 缓冲按 `stage2RowFactor`（通常 1）建、pre/post 按 `prePostRowFactor` 建，这是"D 维常驻一行、只把行向量批量化"的设计；把 y batch 也调大等于 UB 预算翻倍且与 `ProcessY(dim0=1)` 的用法冲突。

- **`InitSinkhornBuffers`** [hc_pre_m_k_split_core.h L320-L340](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L320-L340)
  - 职责：sinkhorn 核的 UB 布局（以 4×4 compact 为单位）。
  - 入参/出参：`stage1UsedCoreNum`。
  - 实现：①`rowFactor = sinkhornRowFactor`，`rowFactorAlign = RoundUp<float>(rowFactor)`；②`InitCommonBuffers` 复用公共缓冲；③`mixesQue2`、`combFragQue` 各 `DOUBLE_BUFFER` 槽，元素数按 `COMPACT_MATRIX_SIZE=16`；④`hcBaseBuf2` 只 16 个 float（4×4 base），`hcBrcbBuf1` 16 个 block，`reduceBuf` 容 `COMPACT_HC_MULT*rowAlign`，`mixes02ReduceBuf` 容 `16*rowAlign`。
  - 注意：`hcBrcbLocal1/reduceLocal` 与 pre/post 路径同名成员被复用，两条 Init 分支只能走一条；本路径刻意只给 16 个 base（不是 4×8），因为 comb 走到这里是 4×4 而非 4×8。

- **`InitOriginalBuffers`** [hc_pre_m_k_split_core.h L342-L385](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L342-L385)
  - 职责：`useOriginal=true` 时"一个核干全流程"的 UB 布局。
  - 入参/出参：`stage1UsedCoreNum`。
  - 实现：①`rowFactor = stage2RowFactor`，`xQue` 容 `rowFactor*hcMult*RoundUp(dFactor)`；②`mixesQue01`/`mixesQue2`/`xQue`/`yQue`/`postQue`/`combFragQue` 全部 2 槽，其中 mix2 与 combFrag 是 `hcMult*hcMultAlign`；③`hcBaseBuf2` 按 `hcMult*hcMultAlign`（4×8）建，不是 16；④`reduceBuf` 与 `mixes02ReduceBuf` 同尺寸，最后统一 Get 出 10 个 LocalTensor。
  - 注意：这条路径没有角色分核，UB 必须"pre + post + comb 三项同时装下"，所以它的 `rowFactor` 通常比另两条路径小；`mixesQue2` 的元素数是 `hcMult*hcMultAlign`（32）而非 16，把 compact 路径的常量搬过来会少分配一半。

- **`Init`** [hc_pre_m_k_split_core.h L387-L407](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L387-L407)
  - 职责：绑 GM 后按核类型/核角色选一套 UB 方案。
  - 入参/出参：9 个 GM + tiling + TPipe；出参与成员状态。
  - 实现：①先 `InitGlobalBuffers`；②`if ASCEND_IS_AIC { return; }`——Part2 纯 Vector，Cube 核不分配任何 UB；③`stage1UsedCoreNum = cubeBlockDimK`（Part1 的 k 块数就是 Part2 的归约轴长度）；④`useOriginal` 走 Original，否则 `GetBlockIdx() < prePostCoreNum` 走 pre/post 套、其余走 sinkhorn 套。
  - 注意：这里用原始 `GetBlockIdx()`（AIV 编号）而不是折半的 Cube 编号，因为 Part2 只有 AIV 在跑；`prePostCoreNum` 是 tiling 给的"角色分界"，必须与 `secondUsedCoreNum` 相容，否则会有核既不算 pre 也不算 sinkhorn。

- **`CalcRsqrt`** [hc_pre_m_k_split_core.h L409-L431](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L409-L431)
  - 职责：从 ws3 读各 k 块平方和，归约成行，算 `1/sqrt(sum/k + eps)`。
  - 入参/出参：`stage1UsedCoreNum`（归约行数）、`globalRowOffset`、`curRowFactor`、`workspaceSize1/2`（定位 ws3 起点）；出参 `rsqrtLocal`。
  - 实现：①`squareSumQue.AllocTensor` 后 `CopyIn(ws3 + ws1+ws2 + globalRowOffset*16, nBurst=stage1UsedCoreNum, len=curRowFactor*16, srcStride=CeilAlign(bs,16)*16 - curRowFactor*16)`，把 `[kBlock][row][16]` 的 k 维当 burst 维搬；②`ReduceSumARAPerf(squareReduceLocal, …, dim0=1, dim1=kBlocks, dim2=curRowFactor*16)` 沿 k 块求和；③`GatherMaskByDiagonal(rsqrtLocal, …, curRowFactor, globalRowOffset%16)` 用循环掩码取 16×16 NZ 块的对角，得到每行的 4×4 平方和；④`Muls(1/k) → Adds(normEps) → Sqrt → Duplicate(1.0) → Div` 得 `1/sqrt(mean+eps)`。
  - 注意：`squareSumOutLocal` 在本函数里 Alloc、却由**调用者**在整批结束才 Free（`ProcessPrePostBatch`/`ProcessSinkhornBatch` 末尾），提前 Free 会与下一批 Alloc 撞地址；相位必须用 `globalRowOffset % 16` 而不是 `% curRowFactor`。

- **`CopyInPartialCombCompact`** [hc_pre_m_k_split_core.h L433-L445](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L433-L445)
  - 职责：把 ws2 每行偏移 `2*hcMult` 起的 4×4 comb 块搬进 UB，k 块维保留为最高维。
  - 入参/出参：`stage1UsedCoreNum`、`workspaceSize1`、`mmLastAxisSize`、`globalRowOffset`、`curRowFactor`；出参 `mixes2Local`。
  - 实现：①`combOffset = hcMult * NUM_TWO`（跳过 mix0、mix1）；②`gmOffset = ws1 + kBlockIdx*bs*mmLastAxisSize + globalRowOffset*mmLastAxisSize + combOffset`；③`ubOffset = kBlockIdx * curRowFactor * COMPACT_MATRIX_SIZE`，每块紧凑排列；④逐 k 块 `CopyIn(nBurst=curRowFactor, len=16, dstStride=mmLastAxisSize-16)`。
  - 注意：dst 行距要保留被跳过的空白（`mmLastAxisSize-16`），否则 16 个元素被写成连续行、后续 gather 全错位；循环按 k 块走，不做 k 块内的行合并。

- **`GatherCompactToElementMajor`** [hc_pre_m_k_split_core.h L447-L461](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L447-L461)
  - 职责：把 `[16 元素][row]` 的"矩阵元素主序"重排为 `[元素][row]` 连续段。
  - 入参/出参：output `mixes02ReduceLocal`、input `combFragLocal`、`curRowFactor`。
  - 实现：①`Duplicate(output, 0, 16*rowAlign)` 先清位；②对 16 个矩阵元素各做一次 `GatherMask(output[element*rowAlign], input, mask[element*8], repeat=16, {1, curRowFactor, 2, 0})`，即"每个元素横跨所有行各取一份"；③末尾一次 `PipeBarrier<PIPE_V>` 统一同步。
  - 注意：`repeatTimes` 固定 16，与实际行数无关，靠 pattern 逐行位移实现选择；`{1, curRowFactor, 2, 0}` 必须与逆函数严格互逆（src0RepStride=2 表示"每两行取一次数据"，因为 16 个元素占两行）。

- **`GatherElementMajorToCompact`** [hc_pre_m_k_split_core.h L463-L484](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L463-L484)
  - 职责：上一步的逆变换，`[16][row]` → `[4][4][row]` 的 compact 矩阵序。
  - 入参/出参：output `combFragLocal`、input `mixes02ReduceLocal`、`curRowFactor`。
  - 实现：①逐 rowIdx 算它落在哪个 16-行组：`rowGroupOffset = rowIdx/16*16`、`rowInGroup = rowIdx%16`；②`rowsInGroup` 先取剩余行再 clamp 到 16；③`GatherMask(output[rowIdx*16], input[rowGroupOffset], mask[rowInGroup*8], repeat=rowsInGroup, {1, COMPACT_MATRIX_SIZE, rowStride, 0})`。
  - 注意：源码注释直指历史 bug——pattern 每 16 元素循环，`rowFactor>16` 时源指针必须前移到 `rowGroupOffset`，否则行 16 会别名回行 0；`rowStride = rowAlign/(BLOCK_SIZE/sizeof(float))` 是"以 block 为单位的行距"，不是元素数。

- **`ReduceCompactRows<reduceMax>`** [hc_pre_m_k_split_core.h L486-L516](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L486-L516)
  - 职责：对 4×4 矩阵的每一行（4 个元素）求 max 或 sum，得 `[4][row]`。
  - 入参/出参：output、input、`curRowFactor`；模板参数选 max 还是 sum。
  - 实现：①先按 `dstRepStride=rowStride`、`src0RepStride=COMPACT_HC_MULT*rowStride` 做 `[i] op [i+1]`（跨过 4 个行段）；②随后把 `src0RepStride` 改回 `rowStride`，再与 `input[2*rowAlign]`、`input[3*rowAlign]` 累加；③`reduceMax` 走 `Max`，否则 `Add`；④每步 `PipeBarrier<PIPE_V>`。
  - 注意：两次调用的 `src0RepStride` 不同是刻意的（第一次源是"下一个 4 元素块的同一行"，之后是"本块内下一列"），抄错就会串列；hc 维在这里被硬编码为 4（`COMPACT_HC_MULT`）。

- **`ReduceCompactCols`** [hc_pre_m_k_split_core.h L518-L540](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L518-L540)
  - 职责：沿 4 个矩阵"行"归约同一列，得 4 个列和（`[4][row]`）。
  - 入参/出参：output、input、`curRowFactor`。
  - 实现：①`dstRepStride/src0RepStride/src1RepStride` 全为 `rowStride`；②依次加 `input[4*rowAlign]`、`input[2*4*rowAlign]`、`input[3*4*rowAlign]`；③三步之间各一次 `PipeBarrier`。
  - 注意：与 `ReduceCompactRows` 只差 src 偏移与 stride，是最容易抄混的一对——"行归约跨块、列归约块内"；累加顺序必须保持左到右，否则与固定参考实现的浮点结果不一致。

- **`ApplyCompactRowValues<divide>`** [hc_pre_m_k_split_core.h L542-L567](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L542-L567)
  - 职责：把 `[4][row]` 的行值广播到 4 列，做除（归一化）或减（softmax）。
  - 入参/出参：output、input、`rowValues`、`curRowFactor`；模板参数 `divide` 选 Div/Sub。
  - 实现：①`dstRepStride = COMPACT_HC_MULT*rowStride`、`src1RepStride = rowStride`（行值不随列走）；②`for colIdx 0..3` 对 `output[colIdx*rowAlign]` 调 `Div/Sub(..., curRowFactor, COMPACT_HC_MULT, params)`；③末尾一次 `PipeBarrier`。
  - 注意：`dstRepStride` 必须是 `4*rowStride`，因为循环每次处理"一列段"、repeat 再跨 4 个元素块，写 `rowStride` 会把结果覆盖到相邻列；`divide` 用 `if constexpr` 而非运行期分支。

- **`ApplyCompactColValues`** [hc_pre_m_k_split_core.h L569-L589](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L569-L589)
  - 职责：把 4 个列值广播到每个矩阵块的 4 行做除法。
  - 入参/出参：output、input、`colValues`、`curRowFactor`。
  - 实现：①三个 stride 都是 `rowStride`；②`for matrixRow 0..3` 处理 `output[matrixRow*4*rowAlign]`；③`Div(..., curRowFactor, COMPACT_HC_MULT, params)`；④末尾一次 `PipeBarrier`。
  - 注意：本函数没有 `divide` 模板参数，只做除法——Sinkhorn 的列归一化只有这一个方向，所以没有对称的 Sub 版本。

- **`NormalizeCompactRows`** [hc_pre_m_k_split_core.h L591-L600](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L591-L600)
  - 职责：行归一化 `data /= (rowSum + hcEps)`。
  - 入参/出参：data（原地）、`curRowFactor`。
  - 实现：①`ReduceCompactRows<false>` 累加到 `reduceLocal`；②`Adds(reduceLocal, reduceLocal, hcEps, 4*rowAlign)`；③`ApplyCompactRowValues<true>` 原地除。
  - 注意：eps 用 `hcEps`（不是 `normEps`），且必须加在除之前；`Adds` 的 count 覆盖 4 行全 padding，不能只写 `curRowFactor`。

- **`NormalizeCompactCols`** [hc_pre_m_k_split_core.h L602-L611](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L602-L611)
  - 职责：列归一化 `data /= (colSum + hcEps)`。
  - 入参/出参：data（原地）、`curRowFactor`。
  - 实现：①`ReduceCompactCols` 得列和；②`Adds(hcEps, 4*rowAlign)`；③`ApplyCompactColValues` 原地除。
  - 注意：列和为 0 时完全靠 eps 兜底，没有"零行跳过"之类的判断；本函数与 `NormalizeCompactRows` 不共享中间量，各自算各自的。

- **`SoftmaxCompactRows`** [hc_pre_m_k_split_core.h L613-L625](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L613-L625)
  - 职责：每行 4 个元素做数值稳定 softmax，最后加 eps。
  - 入参/出参：data（原地）、`curRowFactor`。
  - 实现：①`ReduceCompactRows<true>` 求行最大；②`ApplyCompactRowValues<false>` 减最大值；③`Exp` 全量（16*rowAlign）；④再 `ReduceCompactRows<false>` 求 exp 和并用 `<true>` 除；⑤`Adds(hcEps, 16*rowAlign)`。
  - 注意：减 max 与除和用同一族广播指令但模板参数相反，写反会变成"加 max / 乘和"；最后 `Adds` 覆盖 16 个元素（含 padding），不是只加 4 行，因此后面读 padding 也非零（无害但影响逐位比对）。

- **`ProcessCompactSinkhorn`** [hc_pre_m_k_split_core.h L627-L660](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L627-L660)
  - 职责：一批 4×4 compact 矩阵的缩放/偏置/softmax/交替行列表归一化。
  - 入参/出参：data（原地）、`curRowFactor`、标量 `hcScale`。
  - 实现：①`Brcb(hcBrcbLocal1, hcBase2Local, NUM_TWO, …)` 把 4×4 base 广播成"按行重复"的布局；②`Mul(data, data, rsqrtLocal, curRowFactor, 16, mulParams)`（`src1RepStride=0` 表示 rsqrt 沿行广播）→ `Muls(hcScale, 16*rowAlign)` → `Add(base, src1BlkStride=0/src1RepStride=1)`；③`SoftmaxCompactRows` 后先 `NormalizeCompactCols`；④再 `for iter < iterTimes-1`：`NormalizeCompactRows` + `NormalizeCompactCols`。
  - 注意：迭代首步只做**列**归一，因为 softmax 已保证行和为 1+eps；若改成先做行归一会把 softmax 结果再缩放一次，数值偏；`Muls` 的 count 用 `16*rowAlign`（含 padding）。

- **`PrePostLoadBase`** [hc_pre_m_k_split_core.h L662-L671](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L662-L671)
  - 职责：载入 base0/base1 到 UB，并取 `hc_scale[0]/[1]` 两个标量。
  - 入参/出参：出参两个 float 引用 `hcScale0/hcScale1`。
  - 实现：①`CopyIn(hcBaseGm, hcBase0Local, 1, hcMult)`、`CopyIn(hcBaseGm[hcMult], hcBase1Local, 1, hcMult)` 分两段搬（每段 4 个 float）；②`FetchEventID(HardEvent::MTE2_V)` + `SetFlag/WaitFlag` 显式同步搬运与向量流水；③`hcScaleGm.GetValue(0/1)` 直接标量读 GM。
  - 注意：`GetValue` 绕开搬运引擎做标量直读，会带来一次同步开销，所以只在批循环外调一次；两次 CopyIn 共用同一 eventId，只保证"搬完"，不保证跨核可见。

- **`ProcessPrePostBatch`** [hc_pre_m_k_split_core.h L673-L745](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L673-L745)
  - 职责：处理一批 rows 的 pre 行向量、y 主输出、post；是 pre/post 核的批内主体。
  - 入参/出参：`stage1UsedCoreNum`、`workspaceSize1/2`、`mmLastAxisSize`、`globalRowOffset`、`curRowFactor`、两个 scale；输出 yGm/postGm。
  - 实现：①先 `CalcRsqrt` 得到 `rsqrtLocal`；②`Alloc mixesQue01`，用 `CopyInWithOuterFor(ws2+mixBase, …, outer=kBlocks, nBurst=curRowFactor, len=hcMult, gmFirstDim=bs, gmLastDim=mmLastAxisSize)` 搬 mix0，再以 `hcMult` 偏移搬 mix1，UB 内两段以 `kBlocks*curRowFactor*hcMultAlign` 分界；③`ReduceSumARAPerf(dim0=2, dim1=kBlocks, dim2=row*hcMultAlign)` 消掉 k 维，`ProcessPre`（×rsqrt→×scale→+base→sigmoid→+eps）得 pre 行向量，随后 `Free mixesQue01` 与 `squareSumOutLocal`；④`for rowIdx` × `for dLoopIdx`：逐行 `CopyIn` x（GM 行距 `hcMult*d`、行内 `curDFactor`、尾部 pad `d-curDFactor`）→ `ProcessY`（cast、乘权重、沿 hcMult 归约、cast 回 T）→ `CopyOut` 到 yGm（行距 d）；⑤行循环外一次 `Alloc postQue`，`ProcessPost` 整批算 post（`*2` 而非 `+eps`），再逐行 `CopyOut` 到 postGm（GM 行距 hcMult、UB 行距 hcMultAlign）。
  - 注意：y 缓冲按 `stage2RowFactor` 建，这里却按 `prePostRowFactor` 行跑，所以 ProcessY 只能 `dim0=1` 逐行走；post 的 UB/GM 行距不成 block 整数倍，故保留逐行搬出；`Free squareSumOutLocal` 必须在这里做（CalcRsqrt 只负责 Alloc）；`workspaceSize2` 仅用于定位 ws3，本函数不再单独用。

- **`ProcessPrePost`** [hc_pre_m_k_split_core.h L747-L771](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L747-L771)
  - 职责：pre/post 核入口，按 blockIdx 领行、外层切批。
  - 入参/出参：与 Batch 相同的四个尺寸参数；输出 yGm/postGm。
  - 实现：①`blockIdx >= secondUsedCoreNum` 直接 return（多起的核空转）；②`PrePostLoadBase` 取两个标量；③末核用 `prePostRowLoopOfTailBlock / prePostTailRowFactorOfTailBlock`，其余核用 `*FormerBlock`；④`globalRowOffset = blockIdx*prePostRowOfFormerBlock + rowOuterIdx*prePostRowFactor`，逐批调 Batch。
  - 注意：行偏移一律用 `prePostRowOfFormerBlock` 作核步长，即使尾核行数少也用同一式（tiling 保证不越界）；只有"最后一批的行数"取 tail 值，两个 tail 字段别混。

- **`ProcessSinkhornBatch`** [hc_pre_m_k_split_core.h L773-L801](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L773-L801)
  - 职责：处理一批行的 4×4 comb 矩阵（含 Sinkhorn 迭代），输出 comb_frag。
  - 入参/出参：同前 + `hcScale2`；输出 combFragGm。
  - 实现：①`CalcRsqrt`；②`Alloc mixesQue2` + `CopyInPartialCombCompact`；③`ReduceSumARAPerf(dim1=kBlocks)` 消 k 维 → `GatherCompactToElementMajor` → `ProcessCompactSinkhorn` → `GatherElementMajorToCompact` 还原；④`Free mixesQue2` 与 `squareSumOutLocal`；⑤`combFragQue` 出队后 `CopyOut(combFragGm[globalRowOffset*16], nBurst=1, len=curRowFactor*16)`。
  - 注意：UB 里是"元素主序"、GM 里是"行主序扁平"，所以输出必须 `nBurst=1` 一次连续写——这正是两次 gather 变换存在的理由；`combFragLocal` 在 sinkhorn 迭代前被当作临时量复用，别以为它只装输出。

- **`ProcessSinkhorn`** [hc_pre_m_k_split_core.h L803-L830](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L803-L830)
  - 职责：sinkhorn 核入口。
  - 入参/出参：四个尺寸参数；输出 combFragGm。
  - 实现：①`blockIdx = GetBlockIdx() - prePostCoreNum`，`<0` 或 `>= sinkhornUsedCoreNum` 即 return；②`CopyIn(hcBaseGm[COMPACT_HC_MULT*NUM_TWO], hcBase2Local, 1, 16)` 搬 4×4 base 并做 `MTE2_V` 同步；③`hcScale2 = hcScaleGm.GetValue(NUM_TWO)`；④末核取 `sinkhornRowLoopOfTailBlock / sinkhornTailRowFactorOfTailBlock`，其余取 Former；⑤`globalRowOffset = blockIdx*sinkhornRowOfFormerBlock + rowOuterIdx*sinkhornRowFactor`，逐批调 Batch。
  - 注意：这里的 `blockIdx` 是减去 pre/post 核数后的**局部**编号，不能拿去算全局行或 ws 偏移（行步长已由 tiling 写进 `sinkhornRowOfFormerBlock`）；base2 的 GM 偏移是 `2*hcMult`，因为前两段被 base0/base1 占用。

- **`ProcessOriginal`** [hc_pre_m_k_split_core.h L832-L990](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L832-L990)
  - 职责：`useOriginal=true` 的整段实现——一个核把 pre/y/post/comb 全算完。
  - 入参/出参：无入参；输出 yGm/postGm/combFragGm。
  - 实现：①`AIC return`、`blockIdx>=secondUsedCoreNum return`；②自己重算 `workspaceSize1 = cubeCoreNum*DOUBLE_BUFFER*xCastFp32BufSize`、`workspaceSize2 = CeilAlign(kBlocks*bs*mmLastAxisSize*4, 512)/4`；③一次搬 base0/base1/base2（base2 是 `hcMult×hcMult` 二维）与三个 scale；④外层切批，每批：`CalcRsqrt` → 搬 mix0/mix1 → `ReduceSum` → `ProcessPre` → `dLoop` 内 `ProcessY/CopyOut` → `ProcessPost/CopyOut` → 搬 comb 做 softmax + Sinkhorn 行列迭代 → `CopyOut`；⑤迭代 `iterTimes-1` 次，每轮先末维归约除、再列归约除。
  - 注意：这是拆分前的老路径，与 `ProcessPrePostBatch`/`ProcessSinkhornBatch` 的算式重复，改公式必须三处同步；mix1 的 UB 偏移用 `stage2RowFactor` 而不是 `curRowFactor`，尾批留空洞但地址仍对齐；comb 的 GM 行距是 `hcMult*hcMult`（16），而 PrePost 路径输出 UB 行距是 `hcMultAlign`，两者不能混用。

- **`Process`** [hc_pre_m_k_split_core.h L992-L1013](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L992-L1013)
  - 职责：Part2 总调度：Original 直通，否则按核号二分。
  - 入参/出参：无；输出由子函数落 GM。
  - 实现：①`if constexpr (useOriginal)` 直接 `ProcessOriginal()` 并 return；②`AIC return`；③重算 `workspaceSize1/2`（必须与 Part1 的 Init 同式，否则 ws3 读错位）；④`blockIdx < prePostCoreNum` 调 `ProcessPrePost`，否则 `ProcessSinkhorn`。
  - 注意：`prePostCoreNum` 是两条路径的唯一分界；Part1 完全不分核，真正按角色切核只发生在 Part2；`if constexpr` 让未选中的分支不参与编译。

#### 2.3 HcPreMembaseKSplitCorePremixPart2（TK2）

- **`HcPreMembaseKSplitCorePremixPart2()`** [hc_pre_m_k_split_core.h L1083-L1085](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L1083-L1085)
  - 职责：默认构造，占位。
  - 入参/出参：无。
  - 实现：①函数体为空；②UB 在 `Init` 里一次性分配（不区分子路径）；③成员 LocalTensor 统一在 `GetLocalTensors` 取。
  - 注意：本类没有 `useOriginal` 模板参数，构造不涉及任何分支，也没有"角色二分"。

- **`InitGlobalBuffers`** [hc_pre_m_k_split_core.h L1087-L1100](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L1087-L1100)
  - 职责：绑定 9 个真实 GM（比 Part2 多 `preMixGm`、`preGm`，少两个未用成员）。
  - 入参/出参：x/hcScale/hcBase/y/post/combFrag/preMix/pre/workspace；出参为 GM 成员。
  - 实现：①逐个 `SetGlobalBuffer`；②`preMix` 与 `pre` 在这里是真实指针，缺一不可；③workspace 同样是 float 视图，供读 ws2/ws3。
  - 注意：本类不做 pre/post/sinkhorn 分核，因此 `secondUsedCoreNum` 个 AIV 都持有全量指针，行范围靠 blockIdx 自切；`preMix` 为空但 `hasPreMix==0` 时不会解引用，是合法调用。

- **`InitQueBuffers`** [hc_pre_m_k_split_core.h L1102-L1121](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L1102-L1121)
  - 职责：按固定批 `stage2RowFactor` 建 7 个 TQue。
  - 入参/出参：`stage1UsedCoreNum`、`xQueNum2`（x 队列元素数，由 `Init` 算好）。
  - 实现：①`mixesQue01` 2 槽容 `kBlocks*stage2RowFactor*hcMultAlign*2`；②`mixesQue2` 2 槽容 `kBlocks*stage2RowFactor*hcMult*hcMultAlign`；③`squareSumQue` 2 槽容 `kBlocks*stage2RowFactor*16`；④`xQue` 用 `xQueNum2`，`yQue/postQue/combFragQue` 各 2 槽。
  - 注意：全部用 `stage2RowFactor` 而非 prePost/sinkhorn 的 rowFactor，因为本类不切批；`xQueNum2` 必须与 Process 里实际搬入的行数一致，否则 UB 越界。

- **`InitTBufBuffers`** [hc_pre_m_k_split_core.h L1123-L1152](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L1123-L1152)
  - 职责：建 14 个 TBuf（base/brcb/reduce/cast/掩码/preMix）。
  - 入参/出参：`xQueNum2`。
  - 实现：①`hcBaseBuf0/1` 各 `hcMultAlign`，`hcBaseBuf2` 为 `hcMult*hcMultAlign`；②`rowBrcbBuf0` 按 `RoundUp(stage2RowFactor)*32B`，`hcBrcbBuf1` 按 `RoundUp(stage2RowFactor*hcMultAlign)*32B`；③`reduceBuf/mixes01ReduceBuf/mixes02ReduceBuf/squareReduceBuf` 按行批尺寸；④`xCastBuf/yCastBuf/rsqrtBuf/maskPatternBuf/preMixBuf`，其中掩码只建固定 16 行模式。
  - 注意：掩码按 `MASK_PATTERN_DIM_SIZE*MASK_PATTERN_REPEAT_SIZE` 个 uint32 建（=16×8），配合 `GatherMaskByDiagonal` 的取模相位；`preMixBuf` 只容一个批的 `hcMult` 列，与 `preMixLocal` 一一对应。

- **`GetLocalTensors`** [hc_pre_m_k_split_core.h L1154-L1171](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L1154-L1171)
  - 职责：一次性把 15 个 TBuf 转成 LocalTensor，并生成对角掩码。
  - 入参/出参：无。
  - 实现：①按固定顺序 `Get<float>()` / `Get<uint32_t>()`；②`SetGatherMaskPattern(maskPatternLocal, MASK_PATTERN_DIM_SIZE)`；③`preMixLocal` 最后取。
  - 注意：掩码只 16 行，行批大于 16 时靠 `diagonalOffset` 取模复用相位，不会重新生成；`Get()` 必须在所有 `InitBuffer` 之后调用，早调拿到的是空句柄。

- **`Init`** [hc_pre_m_k_split_core.h L1173-L1186](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L1173-L1186)
  - 职责：绑 GM、算 x 队列容量、建 UB、取 Tensor。
  - 入参/出参：11 个 GM + tiling + TPipe。
  - 实现：①`InitGlobalBuffers`；②`xQueNum2 = stage2RowFactor*hcMult*RoundUp<T>(dFactor)`；③依次 `InitQueBuffers → InitTBufBuffers → GetLocalTensors`；④没有 AIC 早退。
  - 注意：AIC 上也会执行这一整套 `InitBuffer`，真正跳过 AIC 的是 `Process` 里的 `if ASCEND_IS_AIV`；也因此本类的 UB 预算在 AIC 上是"算了但不用"。

- **`Process`** [hc_pre_m_k_split_core.h L1188-L1391](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L1188-L1391)
  - 职责：premix 一核到底：rsqrt → pre（可输出）→ preMix 加权 y → post → comb。
  - 入参/出参：无；输出 yGm/postGm/combFragGm，可选 preGm。
  - 实现：①`if ASCEND_IS_AIV` 包住全部，`stage2BlockIdx>=secondUsedCoreNum` return；②重算 `workspaceSize1/2` 与 `mmLastAxisSize`，搬 base0/base1/base2 并 `MTE2_V` 同步；③外层切批：`CopyIn(ws3 + 核偏移 + 批偏移)` → `ReduceSumARAPerf` → `GatherMaskByDiagonal` → `1/sqrt`；④搬 mix0/mix1（ws2，步长用 `CeilAlign(hcMix, WORKSPACE_ALIGN_SIZE/sizeof(float))`）→ `ReduceSumARAPerf(dim0=2)` → `ProcessPre`；⑤`hasPreOut!=0` 时先 `SetFlag/WaitFlag(V_MTE3)` 再 `CopyOut` 到 preGm，`hasPreMix!=0` 时 `CopyIn(preMixGm → preMixLocal)` 并 `MTE2_V` 同步；⑥`dLoop` 内 `ProcessY(yLocal, xLocal, hasPreMix?preMixLocal:mixes01ReduceLocal, …)` 后写 yGm；⑦`ProcessPost` 写 postGm；⑧comb：逐 `(k,row)` 搬 4×4 → `ReduceSum` → ×rsqrt → ×scale → +base → `SoftmaxFP32Perf` → 列归一 → `iterTimes-1` 次行列交替 → 写 combFragGm。
  - 注意：mix 基址步长这里写成 `CeilAlign(hcMix,512/4)=128`，与 Part2 的 `mmLastAxisSize` 表达式不同但当前取值相同（hcMix=24），改 tiling 时极易错；x 的 `CopyIn` 固定按 `stage2RowFactor` 行搬，只有 `curRowFactor` 行参与计算，尾批会多读（依赖 x GM 尾部可读）；`xGmBlockBaseOffsetPart2/xGmBsBaseOffsetPart2` 是"核基址 + 批基址"两段相加，漏一段整行错位。

---

### 3. 三个类的成员变量与职责划分

| 类 | 成员 | 类型/规模 | 职责 |
|---|---|---|---|
| Part1 | `workspaceGm` | `GlobalTensor<float>` | ws 全量视图，只在 Init 里做指针算术的基准 |
| Part1 | `xCastFp32WsGm` | `GlobalTensor<float>` | ws1：fp32 版 x，每 Cube 核 2 槽，单槽 `mL1Size*align(cvLoopKSize,128)` |
| Part1 | `mmOutFp32WsGm` | `GlobalTensor<float>` | ws2：MMAD 结果，布局 `[kBlock][bs][128]` |
| Part1 | `squareSumFp32WsGm` | `GlobalTensor<float>` | ws3：平方和，布局 `[kBlock][align(bs,16)][16]` |
| Part1 | `hcFnGm` / `xGm` | `GlobalTensor<float>` / `<T>` | 权重矩阵（`hcMix×d*hcMult`）与输入 x |
| Part1 | `xQue` / `mmInQue` | `TQue<VECIN,1>` / `<VECOUT,1>`，各 2 槽 | x 原始 T 的搬入；cast 成 fp32 后的搬出 |
| Part1 | `xLocal` / `xCastLocal` | `LocalTensor<T>` / `<float>` | 上面两队列的句柄 |
| Part1 | `yCastLocal` / `mmOutLocal` | `LocalTensor<float>` | 预留未用（真实数据都走函数参数里的 GlobalTensor） |
| Part1 | `cubeCompute_` | `HcCubeCompute<false>` | Cube 子算子：A1/B1 搬入、L1/L0 调度、MMAD、Fixp 写 ws2/ws3 |
| Part1 | `SYNC_AIV_TO_AIC_FLAG=8` / `SYNC_AIC_TO_AIV_FLAG=9` / `SYNC_MODE2=2` | 常量 | 双向 CrossCore 旗标号与同步模式 |
| Part1 | `cvLoopIdx_` | uint64 | k 迭代全局计数，决定 ws1 双缓冲槽位（跨 m-chunk 连续） |
| Part1 | `kBlkDimIdx_` | uint64 | 本核 k 块号，决定 ws2/ws3 的 k 维偏移 |
| Part1 | `xCastFp32BufSize_` / `mmOuterInnerSize_` / `mmOutFp32BufSize_` / `squareSumFp32BufSize_` | uint64 | ws1 单槽、ws2 行距(128)、ws2 单 k 块、ws3 单 k 块 |
| Part2 | `workspaceGm` | `GlobalTensor<float>` | ws2/ws3 的读取入口 |
| Part2 | `xGm` / `yGm` | `GlobalTensor<T>` | 输入 x / 主输出 y |
| Part2 | `hcScaleGm` / `hcBaseGm` | `GlobalTensor<float>` | 3 个标量 scale 与 `[3*hcMult, hcMult]` 的 base |
| Part2 | `postGm` / `combFragGm` | `GlobalTensor<float>` | post 输出；comb_frag 输出（行主序扁平） |
| Part2 | `mixesGm` / `rsqrtGm` | `GlobalTensor<float>` | 声明但从未使用（历史残留） |
| Part2 | `mixesQue01` / `mixesQue2` / `squareSumQue` | `TQue<VECIN,1>` | mix0+mix1、comb 4×4、ws3 平方和的搬入 |
| Part2 | `xQue` / `yQue` / `postQue` / `combFragQue` | `TQue` | x 搬入；y/post/combFrag 搬出 |
| Part2 | `hcBaseBuf0/1/2` | `TBuf<VECCALC>` | base0(4)、base1(4)、base2（4×8 或 4×4，随路径） |
| Part2 | `rowBrcbBuf0` / `hcBrcbBuf1` | `TBuf<VECCALC>` | rsqrt 的行广播；hcMult 权重的 block 广播 |
| Part2 | `reduceBuf` / `squareReduceBuf` | `TBuf<VECCALC>` | 通用中间归约；k 维归约后的 4×4 平方和 |
| Part2 | `mixes01ReduceBuf` / `mixes02ReduceBuf` | `TBuf<VECCALC>` | pre/post 行向量（2×row×8）；comb 矩阵（16×rowAlign 或 4×8×row） |
| Part2 | `xCastBuf` / `yCastBuf` | `TBuf<VECCALC>` | ProcessY 的 fp32 中间体 |
| Part2 | `maskPatternBuf` | `TBuf<VECCALC>` | 循环对角 GatherMask 掩码（`rowFactor+15` 行或固定 16 行） |
| Part2 | 20 个 `LocalTensor` | 见上 | TBuf/TQue 的句柄；`squareSumOutLocal` 跨 CalcRsqrt 与批末尾存活 |
| Part2 | `COMPACT_HC_MULT=4` / `COMPACT_MATRIX_SIZE=16` | 常量 | Sinkhorn 专用 4×4 线索的维度常量 |
| Premix | 9 个真实 `GlobalTensor` + `mixesGm/rsqrtGm` | 同 Part2 + `preMixGm/preGm` | 唯一真正读写 preMix/pre 的类 |
| Premix | `mixesQue01/2`、`xQue/yQue/postQue/combFragQue`、`squareSumQue` | `TQue` 均 2 槽 | 固定按 `stage2RowFactor` 批建，不随角色变化 |
| Premix | 14 个 `TBuf`（含 `preMixBuf`） | `TBuf<VECCALC>` | 在 Part2 基础上多一份 preMix 的 UB 缓存 |
| Premix | `preMixLocal` | `LocalTensor<float>` | preMix 权重批（`stage2RowFactor*hcMultAlign`），供 ProcessY 当权重 |
| Premix | 21 个 `LocalTensor` | 见上 | 20 个与 Part2 同名 + `preMixLocal` |

职责划分一句话总结：**Part1 只对 Cube 负责**（把 x 转 fp32、跑 MMAD、落 ws1/ws2/ws3），**Part2 只对 Vector 负责**（把 ws 归约回单核视野，再算 sigmoid 类逐行变换与 Sinkhorn）。

三类成员的实际归属可以再压缩成三句：

- 跨函数共享：`workspaceGm`、`tilingData`、`pipe`——生命周期覆盖整个类实例。
- 路径独占：Part2 的 `mixesQue01/yQue/postQue` 只属于 pre/post 套，`mixesQue2/combFragQue` 只属于 sinkhorn 套；premix 类则两种都有（它不切角色）。
- 跨批存活：只有 `squareSumOutLocal`（CalcRsqrt Alloc、批末尾 Free）；其余 LocalTensor 都在单个函数作用域内封闭。


---

### 4. 调用树（谁调谁）

- `hc_pre`（entry）
  - TK0：`Part1::Init` → `Part1::Process` → `[GetMkBlockInfo → ProcessMChunk×N]` → `Part2<false>::Init` → `Part2::Process`
    - `Part2::Process` → `ProcessPrePost` → `PrePostLoadBase` → `ProcessPrePostBatch` → `CalcRsqrt` / `ProcessPre` / `ProcessY` / `ProcessPost`
    - `Part2::Process` → `ProcessSinkhorn` → `ProcessSinkhornBatch` → `CalcRsqrt` / `CopyInPartialCombCompact` / `GatherCompactToElementMajor` / `ProcessCompactSinkhorn` / `GatherElementMajorToCompact`
      - `ProcessCompactSinkhorn` → `SoftmaxCompactRows` / `NormalizeCompactRows` / `NormalizeCompactCols`
        - `NormalizeCompactRows` → `ReduceCompactRows<false>` / `ApplyCompactRowValues<true>`
        - `NormalizeCompactCols` → `ReduceCompactCols` / `ApplyCompactColValues`
        - `SoftmaxCompactRows` → `ReduceCompactRows<true>` / `ApplyCompactRowValues<false>` / `ReduceCompactRows<false>` / `ApplyCompactRowValues<true>`
  - TK1：`Part1::Init` → `Part1::Process` → `Part2<true>::Init` → `ProcessOriginal`（内部不再有二层拆分）
  - TK2：`Part1::Init` → `Part1::Process` → `PremixPart2::Init` → `PremixPart2::Process`
- 三个 `Init` 的共同点：`InitGlobalBuffers`（绑 GM）→ 分支选择 UB 方案（`InitCommonBuffers` 被三条分支共用）→ Get LocalTensor。
- 三个 `Process` 的共同点：先拿 `workspaceSize1/2` 定位 ws2/ws3（必须与 Part1 同式），再按 blockIdx 切行。

---

### 5. 端到端时序（一次 kernel 调用）

Stage1（`pipe`）：

1. `hc_pre` 入口：`GetUserWorkspace` 得到 `userWs`，建 `TPipe pipe`；此时所有数据都在 **GM**。（[hc_pre.cpp L34-L47](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp#L34-L47)）
2. `Part1::Init`：只算偏移、建 `xQue/mmInQue`（**UB**）与 `cubeCompute_`（L1/L0）；数据仍在 **GM**。（`:32-72`）
3. `Part1::Process`：AIC 预置 2 个 `SYNC_AIC_TO_AIV_FLAG` 旗标（**同步单元**，不是内存）。（`:189-195`）
4. `GetMkBlockInfo`：纯整数运算，无数据移动。（`:74-96`）
5. `ProcessMChunk` 每轮 k：
   - AIV：x 从 **GM** → **UB**（`xQue`）→ cast fp32 → **UB**（`mmInQue`）→ **ws1（GM）**；
   - AIC：**ws1（GM）** → L1（A1）→ L0A；`hc_fn` 从 **GM** → L1（B1，可预取下一块）→ L0B；
   - MMAD 结果在 L0C，`Fixp` 搬出到 **ws2（GM）**，平方和写 **ws3（GM）**；
   - 双向旗标 8/9 串起"cast 完→可算"与"算完→可 cast 下一块"。（`:100-187`）
6. `Process` 收尾：`cubeCompute_.End()` 冲干 MMAD，AIV 清旗标，`SyncAll<false>()` 保证 **ws1/ws2/ws3 在 GM 上全局可见**。（`:207-213`）
7. `pipe.Destroy()`；新建 `pipeStage2`（**UB 重新划分**）。（[hc_pre.cpp L99-L101](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp#L99-L101)）

Stage2（`pipeStage2`）：

8. `Part2::Init`：AIC 立即 return；AIV 按 `blockIdx` 选 pre/post 套或 sinkhorn 套（**UB**）。（`:387-407`）
9. `Part2::Process`：重算 `workspaceSize1/2`，按 `prePostCoreNum` 分流。（`:992-1013`）
10. pre/post 核：`PrePostLoadBase`（**GM**→**UB** base0/base1）；每批 `CalcRsqrt`（**ws3(GM)** → **UB** → rsqrt）；搬 mix0/mix1（**ws2(GM)** → **UB**）→ `ProcessPre` 得 pre 行向量（**UB**）；逐行 `CopyIn` x（**GM**→**UB**）→ `ProcessY`（**UB**）→ `CopyOut` y（**UB**→**GM**）；`ProcessPost`（**UB**）→ post（**UB**→**GM**）。（`:673-745`）
11. sinkhorn 核：`CalcRsqrt` 同上；搬 comb（**ws2(GM)** → **UB**）→ 沿 k 归约（**UB**）→ 元素主序重排（**UB**）→ 缩放/偏置/softmax/行列迭代（**UB**）→ 逆重排（**UB**）→ `CopyOut` comb_frag（**UB**→**GM**）。（`:773-801`）
12. 角色二分返回后 Stage2 结束；`pipeStage2.Destroy()`，kernel 退出。此时 y/post/comb_frag（TK2 还有 pre）全部在 **GM**。

TK1 的差异：第 10、11 步合并成 `ProcessOriginal` 一个循环体，**GM/UB 的往返次数不变**，只是同一个核依次做 pre→y→post→comb。

TK2 的差异：第 10 步的 `ProcessPre` 之后多一个"可选写 pre / 可选读 preMix"的旁路，`ProcessY` 的权重可能来自 **GM**（preMix）而不是 UB 内的 pre。

#### 5.1 Stage1 的单轮 k（`ProcessMChunk` 内部，按时间顺序）

1. AIV 侧 `CrossCoreWaitFlag(9)`——等 AIC 说"上一块我算完了"。**同步单元**，无数据。
2. AIV 侧算好本 AIV 负责的半区行范围（`mVectorOffset/mVectorLength`）。**寄存器**。
3. AIV 逐 `stage1MFactor` 行块 `Alloc`：x 从 **GM** 搬到 **UB**（`xQue`，带 pad 的二维 strided 搬运）。
4. AIV `CastTwoDim` bf16→fp32：**UB→UB**（`xQue` 槽 → `mmInQue` 槽）。
5. AIV `CopyOut` 到 ws1：**UB→GM**，行距 `cvLoopKSize`，realK 之后补空。
6. AIV `CrossCoreSetFlag(8)`——告诉 AIC"本块 fp32 已就绪"。
7. AIC 首块：`WaitBL1Mte1ToMte2Flag()` 后 `CopyInB1`，把 `hc_fn` 的 K 片从 **GM** 搬到 **L1**（ND→NZ）；后续块的 B1 由 `ComputeDecode` 内部的 `PreloadNextB1` 预取。
8. AIC `CrossCoreWaitFlag(8)`——等 AIV 的 fp32。
9. AIC `ComputeDecode`：ws1（**GM**）→ L1（A1）→ L0A，B1 已在 L1 → L0B，MMAD 累加到 L0C；同时预取下一块 B1（**GM→L1**）。
10. AIC `Fixp`：L0C → **ws2（GM）**（行距 128 float），并把 x 的平方和写 **ws3（GM）**。
11. AIC `SetBL1Mte1ToMte2Flag` + `CrossCoreSetFlag(9)`，`cvLoopIdx_++`，进入下一轮。

要点：这是典型的 **producer-consumer 双缓冲**——AIV 与 AIC 交替用旗标 8/9 拉手，ws1 的两槽让"AIV 准备第 n+1 块"与"AIC 计算第 n 块"在时序上重叠；B1 的预取在 AIC 内部用 `MTE1_MTE2/MTE2_MTE1` 事件管理，与 cv 旗标是两条独立的同步链。

#### 5.2 Stage2 的单个批（pre/post 核，`ProcessPrePostBatch`）

1. `CalcRsqrt`：ws3 的 `[kBlock][row][16]` 从 **GM→UB**，沿 kBlock 求和（**UB**），GatherMask 取对角（**UB**），`1/sqrt(mean+eps)`（**UB**）→ `rsqrtLocal`。
2. `Alloc mixesQue01`，搬 mix0：ws2 里 `[kBlock][row][0..4)` → **UB**；再搬 mix1：`[kBlock][row][4..8)` → **UB** 的第二段。
3. `ReduceSumARAPerf(dim0=2, dim1=kBlocks)`：沿 kBlock 把两份 mix 各自求和（**UB**），得到 `[row][8]`。
4. `ProcessPre`：×rsqrt → ×scale0 → +base0 → sigmoid → +eps（全在 **UB**），得到 pre 行向量。
5. `Free mixesQue01` 与 `squareSumOutLocal`（本批不再用）。
6. 逐行逐 D 块：x 的 4 个 hc 分量从 **GM→UB**（`nBurst=4`，行内 `curDFactor`）。
7. `ProcessY`：cast fp32 → 乘 pre 权重 → 沿 4 个 hc 分量归约 → cast 回 bf16（**UB**）。
8. `CopyOut` y（**UB→GM**，行距 `d`）。
9. `ProcessPost`：整批算 `[row][4]`（**UB**），再逐行 `CopyOut` 到 post（**UB→GM**，行距 `hcMult`）。

#### 5.3 Stage2 的单个批（sinkhorn 核，`ProcessSinkhornBatch`）

1. `CalcRsqrt`：同上（**ws3→UB**）。
2. `CopyInPartialCombCompact`：ws2 里每行偏移 8 起的 16 个 float，按 kBlock 堆叠进 **UB**。
3. `ReduceSumARAPerf(dim1=kBlocks)`：沿 kBlock 求和（**UB**），得 `[row][4][4]`。
4. `GatherCompactToElementMajor`：重排成 `[16][row]`（**UB**）。
5. `ProcessCompactSinkhorn`：×rsqrt → ×scale2 → +base2 → softmax → 列归一 → `iterTimes-1` 次(行归一, 列归一)（全 **UB**）。
6. `GatherElementMajorToCompact`：还原成 `[row][4][4]`（**UB**）。
7. `CopyOut` 一次连续写 comb_frag（**UB→GM**，`nBurst=1`）。
8. `Free mixesQue2`、`squareSumOutLocal`、`combFragQue`。

---

### 6. Part1 与 Part2 如何交接：workspace 三块

`userWs` 被切成"三段顺序拼接"，偏移全部由 `Part1::Init`（`:48-60`）算出，`Part2::Process`（`:1001-1007`）用同样式子重算一遍——**两处必须逐字一致**。

| 块 | 起点（float 偏移） | 大小 | 写方 | 读方 | 布局 |
|---|---|---|---|---|---|
| ws1 | `cubeBlockIdx * (mL1Size*align(cvLoopKSize,128)) * 2` | 每核 2 槽 | AIV（`ProcessMChunk` 的 `CopyOut`，cast 后的 x） | AIC（`ComputeDecode` 的 A1，`srcDValue=cvLoopKSize`） | `[core][2][mL1Size][cvLoopKSize]`，行距 `cvLoopKSize` 个 float，槽位 = `cvLoopIdx_ % 2` |
| ws2 | `cubeCoreNum * 单核ws1` | `align(cubeBlockDimK * bs * align(24,128) * 4B, 512)/4` | AIC（`Fixp`，MMAD 的 fp32 结果） | AIV Part2（mix0/mix1 与 comb） | `[kBlock][bs][128]`；行内前 `2*hcMult` 列给 pre/post，紧随 16 列是 4×4 comb，其后为 padding |
| ws3 | `ws2 起点 + ws2 大小` | `align(bs,16) * 16` | AIC（`ComputeDecode` 内写平方和） | AIV Part2（`CalcRsqrt`） | `[kBlock][align(bs,16)][16]`，每个 `[row]` 段是 16×16 NZ 块沿对角的 4×4（4 个有效数 + 12 个空位） |

交接要点：

- 写序：AIV 先写 ws1（cast），AIC 才能算；AIC 算完写 ws2/ws3，Part2 才能读。这条链由旗标 8/9 与最终 `SyncAll<false>()` 保证，不依赖任何一次 `pipe.Destroy()`（Destroy 只回收 UB 分配，不刷 GM）。
- 并发：ws1 是 per-Cube-core 私有暂存（两 AIV 各写一半行），ws2/ws3 是 per-kBlock 的公共结果（`cubeBlockDimK` 份），所以 Part2 的归约轴 `dim1` 恰好等于 `cubeBlockDimK`。
- 双缓冲：ws1 的两槽用 `cvLoopIdx_` 交替；AIC 的 `ComputeDecode` 同时收到当前槽与 `(+1)%2` 槽——前者本块算、后者被预取为下一块的 A1，`a1Prefetched/b1Prefetched` 传递"下一块是否已就绪"。
- 行距陷阱：Part1 用 `mmOuterInnerSize_ = CeilAlign(N_SIZE,128) = 128`，Part2 用 `mmLastAxisSize = CeilAlign(hcMix,128) = 128`。当前 `hcMix == N_SIZE == 24` 所以相等；一旦 tiling 放开 `hcMix`，两式必须同步改，否则 mix/comb 全错位。
- ws3 的相位：Part2 读 ws3 时 burst 维是 kBlock、行内长度是 `curRowFactor*16`，源步长是 `align(bs,16)*16 - curRowFactor*16`；随后 `GatherMaskByDiagonal` 的 `diagonalOffset = globalRowOffset % 16`，说明 ws3 的布局隐含"行号在 16×16 NZ 块内的相位"，不能随意重排。
- 越界边界：ws2 的 `bs` 行是逻辑行；`align(bs,16)*16` 只出现在 ws3，所以 ws2 尾部不做 16 行补齐，读 mix 时靠 `CopyInWithOuterFor` 的 `gmFirstDim=bs` 精确控制。

#### 6.1 用一元素对账（最有效的验证方式）

取"第 `r` 行、第 `kb` 个 kBlock 的 mix0 第 0 列"这一个 float，两边算式应当完全相等：

- Part2 读：`ws1起点 + ws2起点 + kb*bs*128 + r*128 + 0`
- Part1 写（Cube 核 `kb`，该行在其 m-chunk 内偏移 `mi`，`r = mGmOffset + mi`）：`ws2起点 + kb*(bs*128) + mGmOffset*128 + mi*128 + 0`
- 两者相等 ⇔ `r == mGmOffset + mi` 且行距同为 128。

ws3 同理：Part1 写 `ws3 + kb*align(bs,16)*16 + m*16`；Part2 用 `nBurst=kb`、`srcStride=align(bs,16)*16 - rows*16` 一次把 `kb` 份 `[row][16]` 读进来。若某个 kBlock 的平方和对不上，先查 `squareSumFp32BufSize_`（= `align(bs,16)*16`）有没有在两边算成一致的值。

对齐的传递链：`CeilAlign(cvLoopKSize,128)` 保证 ws1 单槽是 128 float 的整数倍 → ws1 总大小是 512B 的整数倍 → ws2 起点天然 512B 对齐；ws2 大小再 `CeilAlign(...,512)/sizeof(float)` 抬一次 → ws3 起点也对齐。**这条链上任何一处少对齐，后面全错。**

---

### 7. 三条 TilingKey 路径对照

| 维度 | TK0（Part2<false>） | TK1（Part2<true>） | TK2（PremixPart2） |
|---|---|---|---|
| Stage1 | Part1，相同 | Part1，相同 | Part1，相同 |
| Stage2 分工 | 核 0..prePostCoreNum-1 算 pre/y/post；其余算 comb | 每核都算全流程 | 每核都算全流程 |
| comb 形状 | 4×4 compact（16 元素，两次 gather 变换） | 4×4 通用（`hcMult×hcMultAlign`） | 4×4 通用（同 TK1） |
| UB 套数 | 2 套（互斥） | 1 套（三项合一） | 1 套 |
| 主要 rowFactor | `prePostRowFactor` / `sinkhornRowFactor` | `stage2RowFactor` | `stage2RowFactor` |
| pre/preMix | 不用 | 不用 | `hasPreOut` 写 pre / `hasPreMix` 读 preMix |
| 归一化实现 | `NormalizeCompact*`（4×4 专用） | `SoftmaxFP32Perf` + 通用归约 | `SoftmaxFP32Perf` + 通用归约 |
| 适用 | 大 batch，追求核间分工 | 小/中 batch，代码最直白 | 需要 premix 输入或 pre 输出 |

怎么快速判断一次运行走哪条：看 tilingKey（0/1/2）即可确定分支；再看 `prePostCoreNum` 与 `sinkhornUsedCoreNum` 是否都非零，可确认 TK0 真的把核切开（极端小 batch 下 tiling 可能把 sinkhorn 核数配成 0）。TK1/TK2 下 `prePostCoreNum` 不参与分流，只影响 tiling 里无用的字段。

三个分支共用同一份 Part1，所以 **Stage1 的任何改动对三条路径同时生效**；Stage2 的改动则要判断是否三处都要改（TK0 的两套、TK1 的原版、TK2 的 premix 版）。

---

### 8. 易错点汇总（读改代码前必看）

| 位置 | 陷阱 | 后果 |
|---|---|---|
| `ProcessMChunk` | 误把 `cvLoopIdx_` 当局部变量在 m-chunk 内重置 | ws1 双缓冲退化成单缓冲，AIC 可能读到半写数据 |
| `ProcessMChunk` | 只给偶数核加 `(realMSize+1)/2` 偏移，忘了奇数核取差值 | 两 AIV 行数之和 ≠ realMSize，尾行漏算 |
| `Part1::Process` | Set/Wait 旗标次数不配平 | 死锁或多出的旗标被 Part2 误消费 |
| `Part1::Init` | ws2 大小忘记按 512B 对齐 | ws3 起点错位，`CalcRsqrt` 读到垃圾 |
| `InitCommonBuffers` | 掩码只生成 `rowFactor` 行 | `GatherMaskByDiagonal` 的相位索引越界 |
| `CalcRsqrt` | 相位用 `% curRowFactor` 代替 `% 16` | 尾批取到错误的对角线 |
| `CalcRsqrt` | 在函数内 Free `squareSumOutLocal` | 与下一批 Alloc 冲突/重复释放 |
| `CopyInPartialCombCompact` | dst 行距写成 `0` 或 `16` | 4×4 块被压成连续行，gather 全错 |
| `GatherElementMajorToCompact` | rowFactor>16 时不做 `rowGroupOffset` 前移 | 行 16 别名回行 0（注释里点名的历史 bug） |
| `ReduceCompactRows` | 第二次调用沿用 `src0RepStride=4*rowStride` | 跨块串列 |
| `ApplyCompactRowValues` | `dstRepStride` 写成 `rowStride` | 列结果互相覆盖 |
| `ProcessCompactSinkhorn` | 首步改成先做行归一 | softmax 结果被二次缩放 |
| `ProcessPrePostBatch` | 让 `ProcessY` 一次吃 `prePostRowFactor` 行 | 超出按 `stage2RowFactor` 分配的 y UB |
| `ProcessPrePost` | 尾核行偏移用 tail 步长 | 行区间重叠/空洞 |
| `ProcessOriginal` | 改公式只改这一处 | 与二分路径结果不一致（三处需同步） |
| `ProcessSinkhorn` | 把局部 `blockIdx` 当全局核号 | 行偏移二次加 base |
| `PremixPart2::Process` | mix 步长用 `mmLastAxisSize` 或反过来 | 当前值相同、改 tiling 后才爆的隐性 bug |
| `PremixPart2::Process` | 未做 `V_MTE3` 同步就写 preGm | 写出未算完的 pre |
| `PremixPart2::Init` | 以为 AIC 不会 `InitBuffer` | AIC 上的 UB 预算被无谓占用 |
| `Part2::Process` | `workspaceSize1/2` 与 Part1 式不同 | ws3 起点错位（最隐蔽的一类） |

---

### 9. 建议的阅读顺序（给第一次读这段代码的人）

1. 先只看 [hc_pre.cpp L34-L47](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp#L34-L47) 与 `L92-L138`，建立"一个 kernel、两个 Stage、三个 TilingKey"的骨架，暂时忽略所有 `Init` 细节。
2. 打开 `Part1::Init`（L32-L72），把 ws1/ws2/ws3 的三个起点写成一张纸上的三行算式——后面所有寻址都从这三行长出来。
3. 读 `GetMkBlockInfo`（L74-L96），确认"核号 → (m 块, k 块)"的映射；不理解这一步，后面会一直搞不清 `kBlkDimIdx_` 从哪来。
4. 读 `ProcessMChunk` 的 AIC 分支（L111-L151），只跟三件事：`MmParams` 填了什么、`CrossCoreWaitFlag` 在哪、`ComputeDecode` 的两个 ws1 槽参数是什么。
5. 再读同一函数的 AIV 分支（L152-L184），重点看"按奇偶拆半行"和"cast 后 CopyOut 的 dst 行距"。
6. 跳到 `Part2::Process`（L992-L1013），先只看分流那 4 行；确认自己认同"prePostCoreNum 是分界"。
7. 精读 `ProcessPrePostBatch`（L673-L745），它是全文件最典型的"读 ws → 归约 → 逐行搬 y → 整批算 post"结构。
8. 再读 `CalcRsqrt`（L409-L431）与 `GatherMaskByDiagonal`（在 `hc_pre_base.h`），理解 16×16 NZ 对角相位的必要性。
9. Sinkhorn 链（L433-L660）：先只看数据形状的演变 `[k][row][4][4] → [row][4][4] → [16][row] → 归一化 → [row][4][4]`，再看指令级参数。
10. `ProcessOriginal`（L832-L990）可以最后看，它是拆分前的老实现，价值在于对照"哪些算式被复制到了两个新函数里"。
11. 辅助函数都在 `hc_pre_base.h`：`CopyIn/CopyOut/CopyInWithOuterFor`（搬运）、`ProcessPre/ProcessPost/ProcessY`（数学）、`ReduceSumARAPerf/LastDimReduceSumPerf`（归约）、`SetGatherMaskPattern/GatherMaskByDiagonal`（对角）、`MulABLastDimBrcInline/AddBAFirstDimBrcInline/DivABABrcInline`（广播）。本篇只按调用点解释它们的语义，细节建议按需回查。

调试建议（不改代码也能定位大半问题）：

- 把 tiling 固定成"1 个 Cube、2 个 AIV、1 个 kBlock"，所有多核交互退化成单核顺序执行，旗标对不齐的问题会立刻暴露成死锁或数据错。
- 对不上数时，先按第 6.1 节的算式手算一个元素的地址，再打印 ws 对应位置，比盲看 kernel 快得多。
- 只差最后几位小数：优先怀疑 `hcEps/normEps` 用错、`Muls` 的 count 少乘了 padding、或 `SoftmaxCompactRows` 与 `NormalizeCompact*` 的先后顺序。


---

## 16. 函数级详解：手写 Cube 矩阵乘（`hc_pre_cube_compute*.h`）

两个文件是同一算子在不同硬件路径上的 Cube 实现：`hc_pre_cube_compute.h` 是通用版（模板类 + TPipe 管缓冲），`hc_pre_cube_compute_arch35.h` 是 arch35（新一代）路径（手工算 L0 地址 + 单路输出）。按代码出现顺序逐个函数说明；行号取自当前仓库版本。

### 零、先把名词认全

读这两个文件之前，只需要知道下面这些词；它们会在后面每个函数里反复出现。

| 名词 | 一句话解释 | 代码里对应 |
| --- | --- | --- |
| GM | 全局显存，矩阵原始数据所在地 | `GlobalTensor<float>`、`xGm`、`fnGm` |
| L1 | Cube 旁边的大缓存（KB 级），放整块 A/B tile | `TPosition::A1/B1`、`l1a_`、`l1b_` |
| L0A / L0B | 更小更快的左/右矩阵缓存，Mmad 直接从这里取数 | `TPosition::A2/B2`、`l0a_`、`l0b_` |
| L0C | 累加结果缓存，Mmad 的输出、Fixpipe 的输入 | `TPosition::CO1`、`l0c_` |
| ND | 普通行优先二维数组，GM 上的默认格式 | nd2nz 的源、`nz2ndEn` 的目的 |
| NZ | 按 16×16（fp32 为 16×8）小块切开的"分形"格式，Cube 计算的标准布局 | `dstNzC0Stride`、`L1_BUF_OFFSET` |
| 分形 (fractal) | NZ 里的一个 16×16 小方块，是搬运算力的最小单位 | `BLOCK_CUBE=16` |
| C0 | 一个分形在 K 方向的元素数，fp32 为 8 | `FLOAT_C0_SIZE=8`、`C0_SIZE` |
| MTE1 | 负责 L1 → L0 搬运的单元 | `HardEvent::MTE1_M`、`M_MTE1` |
| MTE2 | 负责 GM → L1 搬运的单元 | `HardEvent::MTE2_MTE1` |
| M 单元 | Cube 里真正做乘加的硬件 | `Mmad` |
| Fixpipe | 把 L0C 结果搬出去并做格式/量化转换的单元 | `DataCopy`、`Fixpipe` |
| unitFlag | Mmad 的"本次之后是否关闭累加"标记，3=关闭 | `UNIT_FLAG_ENABLE(_AUTO_CLOSE)` |
| ping-pong | 同一类缓存开两块轮转，一边算一边搬 | `% L1_BUF_NUM`、`l0PingPongID_` |
| SetFlag/WaitFlag | 两条硬件流水之间的生产者/消费者握手 | `SetFlag<HardEvent::...>` |
| 跨核标志 | AIC 与 AIV 两个核之间的同步 | `CrossCoreSetFlag/WaitFlag` |
| HF32 | fp32 走 19-bit 尾数的高吞吐乘法模式 | `SetHF32Mode`、`SetHF32TransMode` |
| 尾块 | 不是 16/C0 整数倍的最后一段 K 或 M | `curKL1Size`、`min(baseK_, kL1_-kOffset)` |

### 零之二、一次计算的调用顺序

配合上面的名词，先看整条链路，再看单个函数会轻松很多。

- 通用版：`WaitBL1Mte1ToMte2Flag` → `CopyInB1`（B1 进 L1）→ `ComputeDecode`（内含 `CopyInA1` 预取）→ `ComputeL0Pairs`（`LoadAToL0A`/`LoadAToL0B`/`LoadBToL0B`/`MmadA2`/`MmadAB`）→ `Fixp` → `CopyOut` → `SetBL1Mte1ToMte2Flag` → `End`。
- arch35：`CopyInB1Nd2Nz` → `Process`（`CopyInA2`/`CopyInB2`/`MmadBase` 循环）→ `CopyOut` → `End`。
- 两条链路的分岔点在"每轮算几路结果"：通用版一轮出两路，arch35 一轮只出一路。
- 建议阅读顺序：先看 `Init` 弄清有哪些缓冲和事件，再看 `Mmad*`/`CopyOut` 弄清单步动作，最后回看 `ComputeDecode`/`Process` 的循环。

### 一、通用版 `hc_pre_cube_compute.h`

- **`MmadParams`（结构体，非函数）** [hc_pre_cube_compute.h L35-L47](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L35-L47)
  - 职责：打包一次 cube 调用的尺寸与 GM 偏移。
  - 入参/出参：curML1/curKL1/curNL1 为 L1 上 M/K/N 切片；singleCoreK 本核 K 总量；xWsKSize 是 A 的 workspace 行 stride；kGmSize 是原始 K 行 stride；nOutSize 是输出 ND 行 stride；isFirstK/isLastK 控制清零与关累加。
  - 实现：纯 POD 结构，无逻辑；由调用方 [hc_pre_m_k_split_core.h L112-L123](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L112-L123) 逐字段填写。
  - 注意：带 Gm 后缀的是元素偏移/stride，不是字节；curNL1 固定为 N_SIZE=24，本身不是 16 的倍数。

- **`HcCubeCompute`** [hc_pre_cube_compute.h L55](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L55)
  - 职责：构造函数，只声明对象，不申请资源。
  - 入参/出参：无参；成员 blkIdx_=-1、各 LoopIdx_=0 由类内默认值给出。
  - 实现：空函数体；所有缓冲和事件在 Init 中建立。
  - 注意：未调用 Init 就执行其它成员函数属于调用方责任，会导致空 Tensor。

- **`Init`** [hc_pre_cube_compute.h L153-L188](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L153-L188)
  - 职责：一次性分配 L1/L0 缓冲并预置同步事件。
  - 入参/出参：xGm 实际未使用（保留接口）；fnGm 缓存为成员 fnGm_，作为 B1 的 GM 源；tpipe 用于 InitBuffer。
  - 实现：① InitBuffer 分配 A1、B1 各 256KB，A2、B2、CO1 各 64KB；② L1 双缓冲靠常量 L1_BUF_OFFSET=128*256 个 float 切两半；③ 预置 M_MTE1（3/4 给 L0A、5/6 给 L0B）与 MTE1_MTE2（2/3 给 A1、4/5 给 B1）为"已释放"；④ SetHF32TransMode(1) 打开 HF32 转置输入模式。
  - 注意：L0C 仅 64KB，平方和结果占前 256*16 个元素、主结果排在其后（L0C_A2_BUF_OFFSET），两路不能越界；不同 pipe 的事件号可以重号，同一 pipe 内不能。

- **`CopyInA1`** [hc_pre_cube_compute.h L190-L205](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L190-L205)
  - 职责：把 A（x 的 fp32 workspace）从 GM 搬进 L1 并转 NZ。
  - 入参/出参：kL1Size 本次搬运的 K 长度；aGlobal 已偏移到 kGmOffset 的 GM 源；al1Local 目标 L1 块（调用方已算好 buffer 偏移）；mmParams 提供 curML1 与 xWsKSize。
  - 实现：① 填 Nd2NzParams：nValue=curML1 行、dValue=kL1Size、srcDValue=xWsKSize；② dstNzC0Stride=align(curML1,16)，形成 16×C0 分形；③ DataCopy(al1Local, aGlobal, nd2nzParams) 由 MTE2 执行，ND→NZ 在搬运中完成。
  - 注意：srcDValue 必须是 workspace 的真实行 stride（cvLoopKSize），写错每行都会错位；本函数不设 Flag，由调用方补 SetFlag(MTE2_MTE1)。

- **`CopyInB1`** [hc_pre_cube_compute.h L207-L222](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L207-L222)
  - 职责：把权重 fn（ND，N×K）搬进 L1 并转 NZ。
  - 入参/出参：mGmOffset 传了但函数体内未使用（N=24 整块加载，无 M 分块）；kGmOffset 是 GM 上 K 起点；kL1Size 是 K 长度；curNL1=24、kGmSize 来自 mmParams。
  - 实现：① nValue=curNL1(24)、dValue=kL1Size、srcDValue=kGmSize（原始 k）；② dstNzC0Stride=align(24,16)=32，即 NZ 里 N 方向补到 32 行；③ 写入 l1b_[l1bLoopIdx_%2 * L1_BUF_OFFSET]；④ SetFlag(MTE2_MTE1, B_MTE2_MTE1_EVENT+槽号)。
  - 注意：dstNzC0Stride 用 16 对齐而非 curNL1，后续 LoadBToL0B 的 repeat 次数必须同样按 16 向上取整，否则读到 padding。

- **`PreloadNextB1`** [hc_pre_cube_compute.h L224-L242](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L224-L242)
  - 职责：把下一个 K 块的 B1 预取进另一块 L1 buffer。
  - 入参/出参：kGmOffset/kL1Size 指向下一块 K；mmParams 提供 curNL1 与 kGmSize；目标是 nextB1Idx 槽。
  - 实现：① 先 WaitFlag(MTE1_MTE2, next 槽) 确认该槽已被 MTE1 用完；② 与 CopyInB1 相同的 nd2nz 参数组合（这里用 CeilAlign 代替手写对齐式）；③ DataCopy 后 SetFlag(MTE2_MTE1, next 槽)。
  - 注意：只搬数据、不推进 l1bLoopIdx_，槽位轮转统一由 SetBL1Mte1ToMte2Flag 收尾；同一次预取不能重复发。

- **`CopyOut`** [hc_pre_cube_compute.h L244-L265](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L244-L265)
  - 职责：用 Fixpipe 把 L0C 的一块搬到 GM，可选 NZ→ND。
  - 入参/出参：workspaceGlobal 目的 GM；c1Local L0C 源；baseM/baseN 输出块大小；enableNz2Nd 选格式；N 为 ND 模式下的整行 stride。
  - 实现：① 填 DataCopyCO12DstParams，quantPre=NoQuant（fp32 不量化，cast 留给 UB）；② ND 分支 dstStride=N、srcStride=align(baseM,16)，并 SetFixpipeNz2ndFlag(1,1,1)；③ NZ 分支 dstStride=align(nSize,16)；④ unitFlag=3，搬完自动关闭 unitFlag。
  - 注意：srcStride 的单位是 16 行 × C_T 的 CO 块，必须 align(baseM,16)；NZ/ND 两支 dstStride 含义完全不同，不能混用。

- **`Fixp`** [hc_pre_cube_compute.h L267-L285](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L267-L285)
  - 职责：一次调出两路结果（X·Xᵀ 与 X·W）。
  - 入参/出参：workspaceGlobalA2 平方和输出（NZ，M×16）；workspaceGlobalAB 主输出（ND，M×24）；mmParams 提供 curML1/curNL1/nOutSize。
  - 实现：① 第一路取 L0C 槽起点，baseN=BLOCK_CUBE(16)，enableNz2Nd=false；② 第二路取槽起点 + L0C_A2_BUF_OFFSET(256*16 元素)，baseN=curNL1，Nz2Nd=true，行 stride=nOutSize（已按 512B 对齐）；③ 两次都经 CopyOut。
  - 注意：两路 L0C 区必须不重叠，A2 区固定占 16 列；第二路 dstStride 由调用方保证对齐，算子侧不再补。

- **`SetBL1Mte1ToMte2Flag`** [hc_pre_cube_compute.h L287-L292](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L287-L292)
  - 职责：释放当前 B1 槽并切到另一槽。
  - 入参/出参：无参；读写成员 l1bLoopIdx_。
  - 实现：① SetFlag(MTE1_MTE2, B_MTE1_MTE2_EVENT + l1bLoopIdx_%2)；② l1bLoopIdx_++。
  - 注意：必须在 ComputeDecode 之后调用，否则 MTE2 会覆盖 MTE1 正在读的 B1。

- **`WaitBL1Mte1ToMte2Flag`** [hc_pre_cube_compute.h L294-L298](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L294-L298)
  - 职责：等当前 B1 槽空出来才允许写入。
  - 入参/出参：无参；只读 l1bLoopIdx_。
  - 实现：WaitFlag(MTE1_MTE2, B_MTE1_MTE2_EVENT + l1bLoopIdx_%2)，不推进索引。
  - 注意：与 CopyInB1 配对使用（调用方 126-127 行）；它比 Set 版少一个自增，写反会死等或越界。

- **`ComputeL0Pairs`** [hc_pre_cube_compute.h L300-L354](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L300-L354)
  - 职责：在一个 L1 K 块上按 2×32 展开 L0 流水。
  - 入参/出参：l1BaseKOffset 是 L1 内起始 K（GM 绝对 K）；curKL1Size 本块 K 长度；closeAccumulator 标记最后一块 K（令 unitFlag=3）；mmParams 提供 M/N。
  - 实现：① 外层 kL1Offset 每次 +2*K_L0_SIZE(=64)，第二半用 `secondKL1Offset >= curKL1Size` 判空跳过（奇数个 32-K 时）；② 每个 32-K：Wait M_MTE1 → LoadAToL0A、LoadAToL0B（A 转置）→ MmadA2 出 16×16 平方和；③ 再 LoadBToL0B 后 SetHF32Mode(1)/MmadAB/SetHF32Mode(0)；④ 每次 Mmad 后 SetFlag(M_MTE1) 把 L0 槽还给 MTE1。
  - 注意：L0A 与 L0B 的槽索引各自独立递增；L0B 两个槽被"Aᵀ"和"B"轮流占用，顺序错会算错；HF32 只包住 MmadAB。

- **`ComputeDecode`** [hc_pre_cube_compute.h L356-L415](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L356-L415)
  - 职责：单块 m/k 的主循环，驱动 A1/B1 预取与 L0 计算。
  - 入参/出参：xGm 本块 A 的 workspace；nextXGm 下一块 A（跨块预取）；workspaceGlobalA2/AB 两路输出；a1Preloaded 表示上一轮已预取 A1；preloadNextTile/nextB1Offset/nextB1Size 控制跨块 B1+A1 预取。
  - 实现：① 等 B1 到达，未预载则 CopyInA1 并 SetFlag(MTE2_MTE1)；② while 循环把下一块 A1 预取进另一个 L1 槽，最后一块处顺带 PreloadNextB1；③ 对已到达的 A1 调 ComputeL0Pairs(不关累加)，再 SetFlag(MTE1_MTE2) 释放槽；④ 收尾 CrossCoreWaitFlag 等 AIV 写完 nextX 后预取，再 ComputeL0Pairs(isLastK) 关累加，isLastK 时调 Fixp。
  - 注意：curKL1Size 以 singleCoreK 截断尾块；A1 与 B1 共用 MTE2 队列，源码注释要求把"下一个 B1 + 首块 A1"的次序压住；flag 编号必须与 Init 预置一致。

- **`End`** [hc_pre_cube_compute.h L417-L428](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L417-L428)
  - 职责：排空流水，等所有 MTE1/M 事件回落。
  - 入参/出参：无参。
  - 实现：① 对 2 个 L0A/L0B 槽 WaitFlag(M_MTE1, 3/4 与 5/6)；② 对 2 个 A1/B1 槽 WaitFlag(MTE1_MTE2, 2/3 与 4/5)。
  - 注意：必须与 Init 中预置的 SetFlag 一一配对且个数相等，否则结束时死等；HF32 不在此关闭，而是围绕 MmadAB 开关。

- **`LoadAToL0A`** [hc_pre_cube_compute.h L430-L466](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L430-L466)
  - 职责：L1A(NZ) → L0A，作为 Mmad 的左矩阵。
  - 入参/出参：kL1Offset L1 内 K 偏移；kL0Size 本次 K(=32)；l1LoopIdx 选 L1 槽；curML1 来自 mmParams；目标 l0a_[l0aLoopIdx_%2]。
  - 实现：① 用 LoadData3DParamsV2 模拟滑窗：l1H=ceil(curML1/16)、l1W=16、channelSize=kL0Size；② padList[3]=255 让尾部 padding 不参与结果；③ mExtension=align(curML1,16)、kExtension=kL0Size、enTranspose=0；④ LoadData<float,{true,true}> 从 L1 偏移 align(curML1,16)*kL1Offset 处搬。
  - 注意：IsResetLoad3dConfig={true,true} 表示每次都重设 FMatrix，不能省；curML1 非 16 倍数时靠 pad 补齐，l1H 必须向上取整。

- **`LoadAToL0B`** [hc_pre_cube_compute.h L468-L486](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L468-L486)
  - 职责：把同一份 A 转置搬进 L0B，用于算 X·Xᵀ。
  - 入参/出参：kL1Offset/kL0Size/l1LoopIdx 同上；curML1 决定循环次数；目标 l0b_[l0bLoopIdx_%2]。
  - 实现：① 按 16 行一段循环 mL0Offset，共 curML1/16 次；② LoadData2DParams：repeatTimes=ceil(kL0Size,8)=4、srcStride=ceil(curML1,16)、dstGap=0；③ 目的偏移 mL0Offset*align(kL0Size,8)，源偏移 mL0Offset*8；④ 得到 K×M 的分形块（源码注释 "mk nz -> m,k zz"）。
  - 注意：这里步长用 8（fp32 的 C0）而非 BLOCK_CUBE；与 LoadBToL0B 写同一块 l0b_ 的不同槽，槽号顺序必须对齐。

- **`MmadA2`** [hc_pre_cube_compute.h L488-L509](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L488-L509)
  - 职责：算 X·Xᵀ，输出 M×16 的平方和矩阵。
  - 入参/出参：kGmOffset GM 绝对 K 起点（用于判断首次清零）；kL0Size=32；isLastK 决定 unitFlag；curML1 为 M；输出写 l0c_[l0cLoopIdx_%2] 的 A2 区。
  - 实现：① SetFlag 紧跟 WaitFlag(MTE1_M)，确保 L0A/L0B 搬运完成；② 按 16 行一段循环，每次 Mmad m=n=16、k=32；③ cmatrixInitVal = isFirstK && kGmOffset==0（首块清零，否则累加）；④ unitFlag=isLastK?3:2，目的偏移 mL0Offset*BLOCK_CUBE。
  - 注意：Set 后立刻 Wait 是"先保正确、牺牲并行"的写法；MTE1_M 事件号复用 l0bLoopIdx_，因为 Aᵀ 占的就是 L0B 槽。

- **`LoadBToL0B`** [hc_pre_cube_compute.h L511-L526](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L511-L526)
  - 职责：L1B(NZ) → L0B，作为 Mmad 的右矩阵。
  - 入参/出参：kL1Offset L1 内 K 偏移；kL0Size=32；l1LoopIdx 选 L1 槽；curNL1 来自 mmParams；目标 l0b_[l0bLoopIdx_%2]。
  - 实现：① repeatTimes=ceil(curNL1,16)*ceil(kL0Size,8)=2*4=8；② srcStride=1、dstGap=0，即 NZ 连续搬、布局不变；③ 源偏移 align(curNL1,16)*kL1Offset（"n,k nz → n,k nz"）。
  - 注意：curNL1=24 非 16 倍数，repeat 向上取整会多读末尾一个分形，依赖 L1 已按 32 补齐；srcStride 若改成 16 会跨块错位。

- **`MmadAB`** [hc_pre_cube_compute.h L528-L547](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L528-L547)
  - 职责：一次 Mmad 出 M×24 主结果。
  - 入参/出参：kGmOffset 绝对 K 起点；kL0Size=32；isLastK 控制 unitFlag；curML1/curNL1 为 M/N；输出写 L0C 的 AB 区。
  - 实现：① 同样 SetFlag+WaitFlag(MTE1_M) 同步；② mmadParams m=align(curML1,16)、n=curNL1(24)、k=32；③ cmatrixInitVal 与 MmadA2 一致，跨 K 累加；④ 目的地址 = L0C 槽 + L0C_A2_BUF_OFFSET，源取 L0A/L0B 槽首址（"mk zz @ nk nz"）。
  - 注意：n=24 直接交给硬件、不做 16 对齐，这是手写 Mmad 的核心价值；m 已向上取整，多算的行由 L0C 丢弃。

### 二、arch35 版 `hc_pre_cube_compute_arch35.h`

- **`HcPreCubeCompute`** [hc_pre_cube_compute_arch35.h L57-L59](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute_arch35.h#L57-L59)
  - 职责：构造函数，成员全部由类内初值给出。
  - 入参/出参：无参；m_/n_/k_/baseM_/baseN_/baseK_/kL1_=0，三个 pingpong id=0。
  - 实现：函数体为空；真正的资源绑定在 Init。
  - 注意：L1（al1/bl1）不由本类分配，由调用方传入，生命周期由外部保证。

- **`GetBL1BufferId`** [hc_pre_cube_compute_arch35.h L61-L64](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute_arch35.h#L61-L64)
  - 职责：返回当前 B1 的 ping/pong 槽号。
  - 入参/出参：无参；出参 0 或 1（bL1BufferID_）。
  - 实现：直接返回成员，无逻辑。
  - 注意：该值在 Process 结尾翻转，调用方需在 CopyInB1Nd2Nz 后立即读取才有意义。

- **`Init`** [hc_pre_cube_compute_arch35.h L66-L90](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute_arch35.h#L66-L90)
  - 职责：手工切 L0A/L0B/L0C 双缓冲并预置事件。
  - 入参/出参：无参；产出 aL0Ping_/Pong_、bL0Ping_/Pong_、cL0Ping_/Pong_ 六个 LocalTensor。
  - 实现：① L0A/L0B 每块 256*32 元素（32KB），Pong 的地址偏移用 aL0OneBuffer*sizeof(float) 按字节算；② L0C 每块 256*128 元素（128KB）；③ 预置 MTE1_MTE2(0/1) 给 B1、M_MTE1(3/4) 给 L0A/L0B 共用；④ SetHF32Mode(1) 与 SetHF32TransMode(1) 全程常开。
  - 注意：LocalTensor 第二个参数是字节偏移、第三个是元素个数，单位不同；绕过 TPipe 后越界不会被框架检查。

- **`CopyInB1Nd2Nz`** [hc_pre_cube_compute_arch35.h L92-L111](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute_arch35.h#L92-L111)
  - 职责：把 B1 从 GM 搬到 L1 并转 NZ。
  - 入参/出参：k 原始 K 行 stride；currentK 本次 K 长度；baseN=N；bGlobal GM 源；bl1Local 目标 L1 块。
  - 实现：① 记录 k_/kL1_/baseN_ 供 Process 使用；② WaitFlag(MTE1_MTE2, bL1BufferID_) 防覆盖；③ nd2nz：nValue=baseN、dValue=currentK、srcDValue=k、dstNzC0Stride=align(baseN,16)；④ DataCopy 后 SetFlag 并立刻 WaitFlag(MTE2_MTE1)，同步在函数内闭合。
  - 注意：阻塞式等待让 B1 搬运失去异步重叠，是 arch35 的简化取舍；baseN 必须与 Mmad 的 n 完全一致。

- **`Process`** [hc_pre_cube_compute_arch35.h L113-L138](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute_arch35.h#L113-L138)
  - 职责：在一个 L1 K 段上跑 L0 主循环。
  - 入参/出参：m/n 输出尺寸；baseM/baseK L0 单块大小；isFirstKL1/isLastKL1 标识该段在整 K 中的位置；al1Local/bl1Local 是已备好的 L1 A/B。
  - 实现：① kb 以 baseK_ 步进，isLastKL0=kb+baseK_>=kL1_；② Wait M_MTE1 → CopyInA2/CopyInB2 → Set+Wait MTE1_M → MmadBase；③ SetFlag(M_MTE1, l0PingPongID_+3) 释放 L0A/L0B 并翻转 l0PingPongID_；④ 循环结束 SetFlag(MTE1_MTE2) 交还 B1 并翻转 bL1BufferID_。
  - 注意：l0PingPongID_ 同时索引 L0A、L0B、L0C 三块，必须同拍切换；113 行注释给出约束：baseM*baseK ≤ 256*32、baseN*baseK ≤ 256*32、baseM*baseN ≤ 256*128。

- **`CopyInA2`** [hc_pre_cube_compute_arch35.h L140-L156](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute_arch35.h#L140-L156)
  - 职责：L1A(NZ) → L0A，用 2D 参数搬运。
  - 入参/出参：kOffset L1 内 K 偏移；kAL1Offset L1 中该段起始 K；al1Local_ L1A；目标按 l0PingPongID_ 选 aL0Ping_/aL0Pong_。
  - 实现：① mAL1=Align(baseM,16)，offsetAL1=Align(kAL1Offset, C0_SIZE=8)*mAL1；② LoadData2DParamsV2：mStep=ceil(currM,16)、kStep=ceil(currK,8)、srcStride=ceil(currM,16)、dstStride=mStep、ifTranspose=false；③ LoadData 搬入。
  - 注意：K 偏移按 C0=8 对齐而不是 16；currK=min(baseK_, kL1_-kOffset) 处理尾块，不能漏截断。

- **`CopyInB2`** [hc_pre_cube_compute_arch35.h L158-L174](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute_arch35.h#L158-L174)
  - 职责：L1B(NZ) → L0B，参数结构与 A 对称。
  - 入参/出参：kOffset/kBL1Offset K 偏移；bl1Local_ L1B；目标 bL0Ping_/bL0Pong_。
  - 实现：① nBL1=Align(baseN,16)，offsetBL1=Align(kBL1Offset,8)*nBL1；② mStep=ceil(currN,16)、kStep=ceil(currK,8)、srcStride=ceil(currN,16)、dstStride=mStep；③ ifTranspose=false，NZ 的 B 直接可用。
  - 注意：N 方向按 16 对齐跨块，baseN 小于 16 时会读到相邻分形；与 CopyInA2 的唯一差别是用 N 还是 M 算 stride。

- **`MmadBase`** [hc_pre_cube_compute_arch35.h L176-L190](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute_arch35.h#L176-L190)
  - 职责：发起一次 M×N×k 的 Mmad 累积。
  - 入参/出参：kOffset L1 内 K 偏移；isFirstKL1/isLastKL1/isLastKL0 决定清零与 unitFlag；结果写 cl0PingPongID_ 选中的 C 块。
  - 实现：① mmadK=min(baseK_, kL1_-kOffset)；② m=baseM_、n=baseN_、k=mmadK；③ disableGemv=true 关闭 GEMV 特殊路径，cmatrixInitVal=(isFirstKL1 && kOffset==0)；④ unitFlag=(isLastKL1&&isLastKL0)?FINAL_ACCUMULATION(3):NON_FINAL(2)，A/B/C 三块按同一 id 选 ping/pong。
  - 注意：cmatrixInitVal 只在整条 K 的第一个 L0 块为真，否则会把已有累加结果冲掉；unitFlag=3 是 Fixpipe 依赖的"搬完即关"。

- **`CopyOut`（UB 重载）** [hc_pre_cube_compute_arch35.h L192-L212](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute_arch35.h#L192-L212)
  - 职责：Fixpipe 把 L0C 搬到 UB，交给 AIV 后处理。
  - 入参/出参：dstLocal UB 目的地址；输出尺寸取 baseM_/baseN_，C 块取 cl0PingPongID_。
  - 实现：① FixpipeParamsC310<CO2Layout::ROW_MAJOR>，默认使能 NZ→ND；② c0=AuxGetC0Size<float>()，nSize=Align(baseN,c0)、mSize=splitM_?Align(baseM,2):baseM_；③ dstStride=nSize、srcStride=Align(baseM,16)、quantPre=NoQuant；④ dualDstCtl=DUAL_DST_SPLIT_M（CV 1:2，按 M 拆给两个 AIV），unitFlag=3；⑤ Fixpipe<float,float,CFG_ROW_MAJOR_UB> 后翻转 cl0PingPongID_。
  - 注意：splitM 时 mSize 必须 2 对齐（SPLIT_M_ALIGN=2）；srcStride 单位是 16 行 CO 块，别写成 mSize。

- **`CopyOut`（GM 重载）** [hc_pre_cube_compute_arch35.h L214-L229](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute_arch35.h#L214-L229)
  - 职责：Fixpipe 把 L0C 直接搬到 GM，ND 行 stride 取整行 n_。
  - 入参/出参：cGlobal 目的 GM；同样用 baseM_/baseN_ 与 C 槽。
  - 实现：① DataCopyCO12DstParams：nSize=baseN_、mSize=baseM_、dstStride=n_、srcStride=Align(baseM,16)；② quantPre=NoQuant、nz2ndEn=true、unitFlag=3；③ SetFixpipeNz2ndFlag(1,1,1) 后 DataCopy，再翻转 cl0PingPongID_。
  - 注意：dstStride 用完整 N 而不是 baseN，写错会整体行错位；两个 CopyOut 重载共用同一 C 槽并各自翻转，调用路径只能二选一。

- **`End`** [hc_pre_cube_compute_arch35.h L231-L238](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute_arch35.h#L231-L238)
  - 职责：收尾等事件并复位 HF32。
  - 入参/出参：无参。
  - 实现：① WaitFlag(MTE1_MTE2, 0/1) 排空 B1 流水；② WaitFlag(M_MTE1, 3/4) 排空 L0A/L0B；③ SetHF32Mode(0) 关闭。
  - 注意：只等 Init 预置的 4 个事件，Process 未跑满会挂住；每个 SetFlag 都必须有配对 WaitFlag。

### 三、Cube 数据通路

| 阶段 | 执行单元 | 指令 / 接口 | 格式变化 | 对应实现 |
| --- | --- | --- | --- | --- |
| GM → L1（A/x） | MTE2 | `DataCopy` + `Nd2NzParams` | ND(M×K) → NZ（M 补到 16，K 按 C0 分块） | `CopyInA1`；arch35 的 A1 由外部/AIV 备好 |
| GM → L1（B/fn） | MTE2 | `DataCopy` + `Nd2NzParams` | ND(N×K) → NZ（N 补到 16 的倍数） | `CopyInB1`、`PreloadNextB1`、`CopyInB1Nd2Nz` |
| L1 → L0A | MTE1 | `LoadData`（3D 或 2D V2 参数） | NZ → 分形 ZZ（fp32 C0=8） | `LoadAToL0A`、`CopyInA2` |
| L1 → L0B | MTE1 | `LoadData` 2D（可选转置） | NZ → NZ，或 A 转置成 K×M | `LoadBToL0B`、`LoadAToL0B`、`CopyInB2` |
| L0A × L0B → L0C | Cube（M 单元） | `Mmad` | 每次 16×16 分形块点积累加 | `MmadA2`/`MmadAB`、`MmadBase` |
| L0C → GM | Fixpipe | `DataCopy` + `DataCopyCO12DstParams` | NZ → ND（`nz2ndEn`）或保持 NZ | `CopyOut`、GM 重载 |
| L0C → UB | Fixpipe | `Fixpipe<CFG_ROW_MAJOR_UB>` | NZ → ND，并按 M 拆成两半 | arch35 UB 重载 |
| 单元间同步 | 标量/事件 | `SetFlag`/`WaitFlag`、`CrossCoreSetFlag` | — | `Init`/`End`/`ComputeDecode` |

### 四、为什么 N=24 必须手写 Mmad

- Matmul 高层 API 以 baseN=16 的分形为单位做 tiling，N=24 会被切成 16+16（或补到 32），约 1/3 计算和一次多余的 L0C/Fixpipe 搬运全是浪费。
- hc_pre 要在同一份 A 上同时算出 X·Xᵀ（N=16）和 X·W（N=24），两路结果放在 L0C 不同区、输出格式也不同（NZ 与 ND），而一个 Matmul 对象只对应一个 C 矩阵，表达不了这种"一进两出"。
- 手写 Mmad 可以把 n=24 直接交给 Cube（fp32 的 C0=8，24=3*C0），并自行掌控 cmatrixInitVal、unitFlag、HF32 与 K 尾块行为。
- 代价是同步、分形对齐、buffer 复用全部要自己保证——这正是上面每个函数里"注意"条目存在的原因。

### 五、通用版 vs arch35 差异

| 维度 | 通用版 `hc_pre_cube_compute.h` | arch35 `hc_pre_cube_compute_arch35.h` |
| --- | --- | --- |
| 类形态 | 模板类 `HcPre::HcCubeCompute<enableSquareSum>` | 非模板 `HcPreNs::HcPreCubeCompute` |
| L1 分配 | `TPipe::InitBuffer`，A1/B1 各 256KB | L1 由调用方传入，本类只算偏移 |
| L0 分配 | `TPipe` 分配 A2/B2/CO1 各 64KB 单块 | 手工 `LocalTensor` + 字节偏移，A/B/C 均 pingpong |
| 每轮产出 | 同时出 A2（16 列）与 AB（24 列）两路 | 只出一路 C |
| L1→L0 指令 | `LoadData3DParamsV2` + `LoadData2D` | `LoadData2DParamsV2` |
| 同步风格 | 跨函数 pingpong 事件 + 预取下一 tile | 函数内 SetFlag 紧跟 WaitFlag 自闭合 |
| 输出 | 两次 `CopyOut` 到 GM（NZ + ND） | Fixpipe 到 UB（`dualDstCtl=DUAL_DST_SPLIT_M`，CV 1:2）或直接到 GM |
| HF32 | `MmadAB` 前后 `SetHF32Mode(1)/(0)` | `Init` 常开、`End` 关闭 |
| K 尾块 | `curKL1Size` 截断 + `secondKL1Offset` 提前 continue | `min(baseK_, kL1_ - kOffset)` 截断 |
| 核间协作 | `CrossCoreWaitFlag(AIV→AIC)` 等 x workspace | `dualDstCtl=DUAL_DST_SPLIT_M` 把结果直接交给两个 AIV |

### 六、最容易踩的三个坑

- 分形对齐：NZ 的 N 方向一律按 16 补，K 方向按 C0=8 走；`dstNzC0Stride` 与 `repeatTimes` 必须用同一套向上取整规则，否则读到的不是你想要的那一块。
- 事件配对：Init 里 SetFlag 了几个，End 就必须 WaitFlag 几个；`+0/+1`、`+2/+3` 这类偏移是 buffer 槽号，不是随手写的数字。
- buffer 复用：通用版里 Aᵀ 与 B 共用 L0B 的两个槽、两路结果共用 L0C 的两半，任何一处槽号写错都不会报错，只会算错。
- 共性：硬件不做越界检查，错误通常表现为"数值不对"而不是崩溃，所以读代码时应优先核对偏移算式。


---

## 17. 函数级详解：arch35 四套 Kernel 主流程

> 行号取自当前工作区：`vllm-ascend/csrc/moe/hc_pre/op_kernel/`。

### 0. 先建立 12 个最小概念（读正文时随时回查）

1. **AIC / AIV**：同一颗 AI Core 里有两类执行单元。AIC 跑 Cube（矩阵乘），AIV 跑 Vector（逐元素/规约）。一个 AIC 固定配 2 个 AIV，所以 AIV 的 `GetBlockIdx()` 是 `2*c + 0/1`，代码里到处出现 `curBlockIdx / 2` 或 `% 2`。
2. **GM / L1 / UB**：GM 是全局显存（`GlobalTensor`）；L1 是 Cube 的输入缓存；UB 是 Vector 的片上缓存（`LocalTensor`，由 `TPipe` 切分）。
3. **`TPipe` / `TBuf` / `TQue`**：`TPipe` 是 UB 分配器。`TBuf<TPosition::VECCALC>` 是纯计算缓冲（手动管同步）；`TQue<..., N>` 是带 N 级队列语义的缓冲，`AllocTensor/EnQue/DeQue/FreeTensor` 会自动插入搬运与计算的同步。`TBufPool` 是可 `Reset()` 复用的 UB 池。
4. **MTE1 / MTE3 / V / FIX 流水**：硬件是流水线，`SetFlag/WaitFlag<HardEvent::X_Y>` 表示「X 流水做完通知 Y 流水」。例如 `MTE3_V` 是搬出完成后才允许向量改写缓冲。
5. **`CrossCoreSetFlag/WaitFlag<SYNC_MODE4, PIPE_X>(id)`**：跨 AIC↔AIV 的核间同步，`id` 是硬件 flag 号。`id` 与 `id + FLAG_ID_MAX(16)` 是两组独立通道，可以一组传「数据就绪」一组传「缓冲可复用」。
6. **`DType T`**：输入 x 是 BF16/FP16，所以处处先 `cast` 成 float 算，最后再 cast 回 T。系数 `coeff = 1/(hcMult*d)`。
7. **ND2NZ / `CopyToL1`**：Cube 要的 L1 布局是 16×16 分形（NZ）。`VFTransND2NZ` 在 UB 里把行主序改成 NZ，再用 `DataCopyParams`（blockCount/blockLen/srcStride/dstStride）搬进 L1。
8. **K 切分**：把规约维 `hcMult*d` 拆给多个核，各核算部分和，第二阶段再跨核归约（`VFProcessInvRmsPart3WithGroupReduce`）。
9. **M 切分**：不拆 K，把 batch 行 `bs` 按核分，每核独立走完全 K，无跨核归约（`VFProcessInvRmsPart3`）。
10. **`multCoreSplitKSize`**：每核分到的 K 长度；`kBlockFactor`、`cubeBlockDimK/M` 是 K/M 方向核数；`mL1Size/kL1Size/mUbSize/kUbSize` 是 L1/UB 的分块尺寸，全部来自 tiling。
11. **`Part1/Part2`**：阶段一算 `mm = x·hcFnᵀ`（K 切分时落 workspace）与 `rms`；阶段二做 invRms、pre/post、sinkhorn、y。中间要 `pipe.Destroy()` + 新建 `TPipe`，因为两阶段 UB 需求不同。
12. **`CopyInWithLoopMode` / `CopyInWithUbStride`**：带行距的批量 GM→UB 拷贝；`UbRowGapBlocks(a,b)` 计算每行末尾需要跳过的 32B 块数。

### 1. 四套实现的定位

| 维度 | 说明 |
| --- | --- |
| TilingKey | 由 `op_host/hc_pre_tiling_arch35.h:755` 生成：`(hasPreMix ? 1002 : 1000) + (isKSplit ? 0 : 1)` |
| 分发 | `op_kernel/hc_pre.cpp:51-89` 按 1000/1002/1001/1003 依次 new 对应类 |
| 1000/1002 | K 切分（`isKSplit=true`）：Part1 + Part2 两个类、两个 TPipe |
| 1001/1003 | 不切 K，仅 M 切分：单类一次跑完，用 `tbufPool1.Reset()` 模拟「阶段重建」 |

### 2. 文件一：`hc_pre_m_k_split_core_arch35.h`（TilingKey 1000，K 切分、无 premix）

#### 2.1 文件概览

- 两个类：`HcPreMKSplitCorePart1`（L26-L262，阶段一）与 `HcPreMKSplitCorePart2`（L265-L468，阶段二）。
- 数据流：Part1 写 `workspace` 的两块 —— `mmGm`（前 `kBlockFactor*bs*hcMix` 个 float）与 `rmsGm`（紧随其后）。
- 分核：`idx / cubeBlockDimK` 取 M 块号、`idx % cubeBlockDimK` 取 K 块号；配对的 2 个 AIV 再把每轮 M 行对半切。
- 同步：4 个 `SYNC_AIC_AIV_FLAG(9)` / `9+16` 作为「信用」在 Init 里播种，Part1 末尾逐个 drain。

主线（一次 round，即一个 `mL1Size` 行块）：

```
AIC: CopyInB1Nd2Nz(hcFn 第 kBlk 段) ─┐
                                     ├─ 等 AIV×2 ─ Process ─(isLastKL1) CopyOut→mmGm
AIV: CopyIn(x 半行) → cast+invRms → ND2NZ → CopyToL1 → 通知 AIC
                                     └─ k 循环结束后 CopyOut(rms)→rmsGm
```

阶段一缓冲清单（阶段二另起一套）：

| 缓冲 | 尺寸来源 | 用途 |
| --- | --- | --- |
| `l1Buffer`(A1) | `L1_ALLOC_SIZE` | 劈成 `xL1_`（A 矩阵）与 `wL1_`（B1），各自按 `L1_BUF_OFFSET` 乒乓 |
| `xQue` | 2 × `mUbSize*RoundUp<T>(kUbSize)*sizeof(T)` | 原始 x 的 UB 入口 |
| `rmsQue` | 2 × `RoundUp<float>(mUbSize)*4` | 每行 invRms 的搬出口 |
| `castBuf` | `mUbSize*(RoundUp<float>(kUbSize)*4 + BLOCK_SIZE)` | float 化后的 x，尾部多 32B 给 ND2NZ |
| `nd2NzBuf` | `CeilAlign(mUbSize,C0)*RoundUp<float>(kUbSize)*4*2` | NZ 布局双缓冲 |
| `mmService_` 内 B1 | `GetBL1BufferId()` | cube 服务的权重缓存，不在本类显式管理 |

#### 2.2 函数逐条

- **`HcPreMKSplitCorePart1()`** [hc_pre_m_k_split_core_arch35.h L29-L30](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_arch35.h#L29-L30)
  - 职责：默认构造，无任何逻辑。
  - 入参/出参：无入参、无出参（成员由 `Init` 填充）。
  - 实现：①仅 `{}`；②`aL1BufferID_` 等靠类内初始化（L261）置 0；③AIC/AIV 共用同一模板实例，构造期不碰 GM。
  - 注意：不要在此加 `InitBuffer`，那时 `pipe` 还是空指针。

- **`Init`** [hc_pre_m_k_split_core_arch35.h L32-L66](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_arch35.h#L32-L66)
  - 职责：绑定 GM、切分 L1/UB、在 AIC 侧播种 4 个核间 flag。
  - 入参/出参：入 `x`、`hcFn`、`workspace`、`tilingDataPtr`、`pipePtr`；出参无，副作用是绑定 `xGm/hcFnGm/mmGm/rmsGm` 与各队列。
  - 实现：①`rmsGm = workspace + kBlockFactor*bs*hcMix`，即 mm 在前、rms 在后（L40）；②L1 用一块 `TBuf<A1>`，`xL1_` 是基址、`wL1_` 偏移 `L1_BUF_NUM*L1_BUF_OFFSET`（L42-L45）；③`xQue`/`rmsQue` 各 2 级、`castBuf` 多留 `BLOCK_SIZE`、`nd2NzBuf` 双份（L48-L55）；④AIC 连发 4 次 `CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>`（9、9+16 各两遍，L57-L63）。
  - 注意：这 4 个 flag 是「预算信用」，AIV 必须恰好消费 2 个（L222-L223），多 set 或少 wait 都会让 `SyncAll` 死等。

- **`Process`** [hc_pre_m_k_split_core_arch35.h L68-L229](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_arch35.h#L68-L229)
  - 职责：阶段一主循环 —— 边 cast/求 RMS，边做 `x·hcFnᵀ`，结果落 workspace。
  - 入参/出参：无入参；写 `mmGm`（每轮每 K 块一份）与 `rmsGm`。
  - 实现：①`singleCoreRound` 按「前 `mainCoreCount` 个 M 块多一轮」算（L77-L80），`mGmOffset` 用定长块公式（L84-L87）；②AIC 每个 `kL1Size` 块：`CopyInB1Nd2Nz` → 等 AIV 两个 flag → `mmService_.Process` → 仅 `isLastKL1` 时 `CopyOut` 到 `mmGm[mBlk 段 + kBlk 段 + roundIdx]`（L135-L153）；③AIV 把 K 再按 `kUbSize` 切，首个分片用 `VFProcessCastAndInvRmsPart1<T,false>` 建和、之后 `<T,true>` 累加，随即 ND2NZ + `CopyToL1` 写进 L1 的本核半区（L163-L202）；④k 循环后把 `rmsNormLocal` 写成 `rmsGm`，再 drain `MTE3_V` 0/1 与 2 个 AIC flag，AIC 侧 `mmService_.End()`，最后 `SyncAll<false>()`（L207-L228）。
  - 注意：奇偶 AIV 用 `mL1RealSize/2` 下取整（偶数核拿 floor、奇数核拿 ceil，L156-L162），与文件三/四的 `CeilDiv` 约定正好相反；`rmsGm` 的 K 段偏移是 `(blockIdx/2)%kBlockFactor*bs`（L208）。

- **`HcPreMKSplitCorePart2()`** [hc_pre_m_k_split_core_arch35.h L268-L269](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_arch35.h#L268-L269)
  - 职责：默认构造，无逻辑。
  - 入参/出参：无。
  - 实现：①空体；②`hasPreOut_` 默认 false（L466）；③所有缓冲在 `Init` 中按 tiling 分配。
  - 注意：`pre` 是否为空决定 `preQue` 是否分配，构造期无法判断。

- **`Init`** [hc_pre_m_k_split_core_arch35.h L271-L324](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_arch35.h#L271-L324)
  - 职责：绑定阶段二的 7+1 个 GM 并切分输出侧 UB。
  - 入参/出参：入 `x/hcScale/hcBase/y/post/combFrag/pre/workspace/tiling/pipe`；出参无。
  - 实现：①`pre` 可为 nullptr，`hasPreOut_` 控制后续分支（L284-L287）；②`xQue` 按 `stage2RowFactor*hcMult*RoundUp(dFactor)` 开双份，`rmsAndmmQue` 把「mm 的 kBlockFactor 份 + rms 的 kBlockFactor 份」拼在一个 tensor（L293-L298）；③`yQue/postQue/combFragQue` 与可选 `preQue`（L300-L312）；④`hcBase0/1/2` 与 `mixesBuf` 用 TBuf（L315-L323）。
  - 注意：`mixesLocal` 行距是 `hcMix`（非对齐），pre 输出前必须 `CopyOut(..., 0, ubRowGapBlocks_)` 压实成 `hcMultAlign` 行距（L371）。

- **`Process`** [hc_pre_m_k_split_core_arch35.h L326-L427](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_arch35.h#L326-L427)
  - 职责：阶段二 —— 行块级 invRms → pre/post/sinkhorn → y，并写 4 个输出。
  - 入参/出参：无入参；写 `yGm/postGm/combFragGm/preGm`。
  - 实现：①只跑 AIV，`secondUsedCoreNum` 分核，末核取 `*OfTailBlock` 的循环数与尾行数（L331-L338）；②每行块一次 `CopyInWithLoopMode` 搬 `kBlockFactor` 份 mm、再 `CopyIn` rms 到 `rmsAndmmLocal[mmLocalSize]`，`VFProcessInvRmsPart3WithGroupReduce` 做跨 K 核归约（L350-L362）；③`VFProcessPre` → 可选 pre 压实搬出 → `dFactor` 循环里 `VFProcessY` 写 y（L380-L399）；④`VFProcessPost` 与 `VFProcessCombFragPacked`，偏移按 `bs` 行铺开（L400-L424）。
  - 注意：Part2 会重新从 GM 读 x（Part1 只留了 mm/rms）；`pre` 复用 post 的 GM 偏移公式，但行距是 `hcMult` 而非 `hcMultAlign`。

### 3. 文件二：`hc_pre_m_k_split_core_premix_arch35.h`（TilingKey 1002，K 切分 + premix）

#### 3.1 文件概览

- 两个类：`HcPreMKSplitCorePremixPart1`（L28-L302）与 `HcPreMKSplitCorePremixPart2`（L304-L441）。
- 与文件一最大差别：**y 在阶段一就写完**（`yAccLocal` 在 UB 累加 `preMix·x`），阶段二不再读 x、不再有 d 循环。
- K 迭代顺序被重排成「hc 段 × d 窗口」，因为 `y` 需要同一 d 窗口在所有 `hcMult` 段上的 cast 结果。
- 契约写在 L21-L27：`dPerCore = multCoreSplitKSize/hcMult`、`cubeBlockDimK = CeilDiv(d, dPerCore)`。

主线（一次 round）：

```
AIV: 先整块搬 preMix（局部半行）→ 留 UB
     for dGroup(=kL1 窗口) in 本核 d 区间:
       for hcIdx in 0..hcMult-1:          ← K 的「段」维，段内再按 kUbSize 分子窗
         AIC: CopyInB1Nd2Nz(hcFn 第 hcIdx 段) → 等 AIV → Process ─(isLastK) CopyOut mm
         AIV: cast+invRms+premix 累加(写 yAcc) → ND2NZ → CopyToL1
              若 hcIdx 是最后一段: yAcc→cast→yGm
     最后 CopyOut(rms)→rmsGm
```

阶段一缓冲清单：

| 缓冲 | 尺寸来源 | 用途 |
| --- | --- | --- |
| `xQue` | 2 × `mUbSize*RoundUp<T>(kUbSize)` | 原始 x 入口 |
| `preMixQue` | 1 × `maxMRows*RoundUp<float>(hcMix)` | preMix 的 `[mReal, hcMult]` 行，`maxMRows=CeilDiv(mL1Size,2)` |
| `rmsQue` | 2 × `RoundUp<float>(maxMRows)` | invRms 输出 |
| `yQue` | 2 × `mUbSize*RoundUp<T>(kUbSize)` | 阶段一就写出的 y |
| `castBuf` | 同文件一 | float 化 x |
| `nd2NzBuf` | 同文件一 | NZ 双缓冲 |
| `yAccBuf` | `maxMRows*RoundUp<float>(kUbSize)*CeilDiv(maxGroupK,kUbSize)` | 每子窗一份 y 累加器，按 `subIdx` 切片 |

#### 3.2 函数逐条

- **`HcPreMKSplitCorePremixPart1()`** [hc_pre_m_k_split_core_premix_arch35.h L30-L30](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_premix_arch35.h#L30-L30)
  - 职责：默认构造。
  - 入参/出参：无。
  - 实现：①空体；②`aL1BufferID_` 类内初始化为 0（L301）；③无 GM 绑定。
  - 注意：—

- **`Init`** [hc_pre_m_k_split_core_premix_arch35.h L32-L75](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_premix_arch35.h#L32-L75)
  - 职责：绑定含 `preMix`、`y` 的 GM，并额外切出 y 累加缓冲。
  - 入参/出参：入 `x/hcFn/preMix/y/workspace/tiling/pipe`；出参无。
  - 实现：①多绑 `preMixGm`、`yGm`（阶段一即出 y）；②`yAccBuf = maxMRows × RoundUp(kUbSize) × CeilDiv(maxGroupK,kUbSize)`，`maxGroupK=min(kL1Size,min(dPerCore,d))`（L51-L63）；③`preMixQue` 1 级、`rmsQue`/`yQue`/`nd2NzBuf` 各 2 级（L50-L61）；④AIC 同样播种 4 个 flag（L68-L74）。
  - 注意：`yQue` 只在本文件出现；`maxMRows=CeilDiv(mL1Size,2)` 是按「配对 AIV 各半」估的上界，若 tiling 改了半切约定必须同步改这里。

- **`Process`** [hc_pre_m_k_split_core_premix_arch35.h L77-L257](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_premix_arch35.h#L77-L257)
  - 职责：阶段一 —— 按「hc 段 × d 窗口」算 mm，同时累加 premix 得到 y 并直接写出。
  - 入参/出参：无入参；写 `mmGm/rmsGm/yGm`。
  - 实现：①越界核走 `FinishStage1 + SyncAll + return`（L84-L89）；`mGmOffset` 用「长核优先」紧凑公式（L97-L102），与文件一定长块公式不同；②`dPerCore/dCoreStart/dCoreSize/dGroupCount` 切 d 维，组内再按 `kUbSize` 分 `subCount`（L104-L107、L137-L141）；③三层 `hcIdx×mInner×subIdx` 循环，按「首子块 / hc 首段 / 其余」三选一调用 `VFProcessCastInvRmsAndPremixYPart1`，最后一个 hc 段再把 `yAcc` cast 写 `yGm`（L167-L231）；④`rms/preMix` 用带行距的 `CopyInWithUbStride`，末尾先 drain `MTE3_V` 0/1、再对 event 2 做一次自 set/自 wait，然后 `FinishStage1 + SyncAll`（L240-L256）。
  - 注意：K 的「段」维是 `hcIdx`，`groupHcIdx = groupIdx*hcMult + hcIdx` 同时决定 `isFirstK/isLastK`（L144-L146），只看 `groupIdx` 会漏掉段边界。

- **`FinishStage1`** [hc_pre_m_k_split_core_premix_arch35.h L260-L268](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_premix_arch35.h#L260-L268)
  - 职责：阶段一的收尾（AIV 排空信用 / AIC 结束 cube 服务）。
  - 入参/出参：无入参、无出参。
  - 实现：①AIV 两次 `CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(9)`；②AIC 调 `mmService_.End()`；③被正常路径与越界早退路径共用。
  - 注意：文件一把这段内联在 `Process` 末尾（L220-L227），本文件抽成成员函数所以两条路径都必须调用，漏掉早退分支就会与 AIC 信用不匹配。

- **`HcPreMKSplitCorePremixPart2()`** [hc_pre_m_k_split_core_premix_arch35.h L306-L306](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_premix_arch35.h#L306-L306)
  - 职责：默认构造。
  - 入参/出参：无。
  - 实现：①空体；②`hasPreOut_` 默认 false（L439）；③缓冲在 `Init` 分配。
  - 注意：—

- **`Init`** [hc_pre_m_k_split_core_premix_arch35.h L310-L346](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_premix_arch35.h#L310-L346)
  - 职责：只绑阶段二必需的 GM 与 UB（不含 x/y 队列）。
  - 入参/出参：第 1、4、5 个 GM 形参匿名（`GM_ADDR` 无名字），实际只绑 `hcScale/hcBase/post/combFrag/pre/workspace`。
  - 实现：①`hcScaleGm/hcBaseGm/postGm/combFragGm/preGm` 与两块 workspace（L316-L327）；②`rmsAndmmQue` 双 buffer，大小由 `kBlockFactor` 与 `stage2RowFactor` 拼出（L329-L332）；③可选 `preQue` + `postQue`（L333-L337）；④`hcBase0/1/2` 与 `mixesBuf`（L338-L345）。
  - 注意：调用处（[hc_pre.cpp L74](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp#L74)）仍按 8 个位置传 `x, hc_scale, hc_base, pre_mix, y, post, comb_frag, pre`，形参匿名不会报错但错位会静默绑错指针。

- **`Process`** [hc_pre_m_k_split_core_premix_arch35.h L348-L411](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core_premix_arch35.h#L348-L411)
  - 职责：阶段二 —— invRms → pre/post → sinkhorn，combFrag 元素主序直出。
  - 入参/出参：无入参；写 `preGm/postGm/combFragGm`。
  - 实现：①AIV-only，`secondUsedCoreNum` + 末核尾行分支（L350-L365）；②mm 与 rms 合批搬入后 `VFProcessInvRmsPart3WithGroupReduce`（L370-L380）；③pre 可选（含压实），post 走 TQue（L383-L400）；④`VFProcessCombFragPremixElementMajor` 就地写 `mixesLocal`，`SetWaitFlag(V_MTE3)` 后一次 `CopyOut(1, rows*hcMult²)`，再 `SetWaitFlag(MTE3_V)`（L402-L408）。
  - 注意：本文件不用 `combFragQue`，同步靠手写 `SetWaitFlag`；`CopyOut` 是「1 行 × rows*hcMult²」整块搬，要求 UB 里是元素主序。

### 4. 文件三：`hc_pre_m_split_core_arch35.h`（TilingKey 1001，M 切分、无 premix）

#### 4.1 文件概览

- 只有一个类 `HcPreMSplitCoreArch35`（L26-L474），阶段一/二写在同一个 `Process` 里。
- 不再切 K：`kGmEndOffset = multCoreSplitKSize`，无 K 块号；mm 结果直接留在 UB 的 `mmXLocal`（`mmXBuf`），不落 workspace。
- 「pipe 重建」改为 `tbufPool1.Reset()` 后重新 `InitBuffer`（L280-L296），UB 在两阶段间复用。
- 阶段二重算 y 时**重新从 GM 读 x**，并复用 UB 中的 `rmsNormLocal`，因此不需要 group reduce。

主线（一个 round）：

```
AIV: tbufPool1.Reset() → 装 cast/ND2NZ 缓冲
     for kL1 in 本核全部 K:            ← 无 K 分核，kGmEndOffset=multCoreSplitKSize
       AIC: CopyInB1Nd2Nz → 等 AIV → Process ─(isLastKL1) CopyOut→mmXLocal(UB) + PRE_POST flag
       AIV: CopyIn x → cast+invRms → ND2NZ → CopyToL1
     等 PRE_POST flag → tbufPool1.Reset() → 装输出队列
     for 行块: invRms(UB) → pre → y(重读 x) → post → combFrag
```

两阶段 UB 布局（同一 `TBufPool` 复用）：

| 阶段 | 缓冲 | 尺寸来源 |
| --- | --- | --- |
| 共用 | `mmXBuf` | `CeilDiv(mL1Size, AIVS_PER_AIC)*RoundUp<float>(hcMix)` |
| 共用 | `rmsNormBuf` | `RoundUp<float>(CeilDiv(mL1Size,2))*4`（只有本核一半行） |
| 共用 | `hcBaseBuf0/1/2` | `hcMultAlign` / `hcMultAlign` / `hcMult*hcMultAlign` |
| 阶段一(`tbufPool1`) | `xQue` / `castBuf` / `nd2NzBuf` | 同文件一（按 `mUbSize/kUbSize`） |
| 阶段二(`tbufPool1` Reset 后) | `xQue`/`yQue` | `rowInnerFactor*hcMult*RoundUp<T>(dFactor)` / `rowInnerFactor*RoundUp<T>(dFactor)` |
| 阶段二 | `postQue`/`combFragQue`/`preQue`/`mixesBuf` | 按 `rowInnerFactor` 与 `hcMult/hcMultAlign` |

#### 4.2 函数逐条

- **`HcPreMSplitCoreArch35()`** [hc_pre_m_split_core_arch35.h L28-L28](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_split_core_arch35.h#L28-L28)
  - 职责：默认构造。
  - 入参/出参：无。
  - 实现：①空体；②`aL1BufferID_{0}` 类内初始化（L469）；③`TBufPool` 成员已声明但需 `InitBufPool` 才可用。
  - 注意：—

- **`Init`** [hc_pre_m_split_core_arch35.h L30-L78](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_split_core_arch35.h#L30-L78)
  - 职责：绑定 8 个 GM，用两级 `TBufPool` 预分配两阶段 UB，AIC 播种 flag。
  - 入参/出参：入 `x/hcFn/hcScale/hcBase/y/post/combFrag/pre/tiling/pipe`；出参无。
  - 实现：①`mmXBuf = CeilDiv(mL1Size, AIVS_PER_AIC)*RoundUp(hcMix)`，mm 结果留 UB（L54-L58）；②`tbufPool0` 按 `bufferPool0Size` 初始化，池内放 `rmsNormBuf`（半 M）、`hcBase` 三块，并再嵌 `tbufPool1`（L54、L66-L72）；③AIC 只 `mmService_.Init()` + 4 次 flag，AIV 才取 `hcBase` 的 LocalTensor（L60-L77）；④L1 仍是一整块 `TBuf<A1>` 劈成 x/w（L49-L52）。
  - 注意：`tbufPool0` 在 AIC/AIV 上分配的内容不同（AIC 只有 mmXBuf），改尺寸时两边都要核对 UB 总量。

- **`Process`** [hc_pre_m_split_core_arch35.h L80-L394](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_split_core_arch35.h#L80-L394)
  - 职责：单次跑完 —— cube 算 mm 留在 UB，随后重建 UB 布局做全部向量输出。
  - 入参/出参：无入参；写 `yGm/postGm/combFragGm/preGm`。
  - 实现：①`logicalBlockIdx` 越界即 drain/`End` 后返回（L83-L95）；M 轮划分同文件一，但 `mGmOffset` 用长核优先公式且判定写作 `curBlockIdx <= mainCoreCount`（L118-L134）；②k 循环内 AIC `CopyInB1Nd2Nz→Process`，`isLastKL1` 时 `CopyOut(mmXLocal)` 并用 `PIPE_FIX` 发 `SYNC_AIC_AIV_PRE_POST_FLAG(10)`（L185-L206）；③AIV 做 cast/invRms/ND2NZ/L1 搬运，k 循环后等 PRE_POST flag，`tbufPool1.Reset()` 重建为输出阶段队列（L275-L300）；④行内循环：`VFProcessInvRmsPart3`（rms 来自 UB，非 group reduce）→ pre → y（重读 x）→ post → combFrag（L302-L383）。
  - 注意：`mBlkDimIdx`/`kBlkDimIdx`（L108-L109）算出来后未被使用，是残留；`cvLoopKSize = kL1Size/kUbSize`（L147）固定，尾块 `kRealSize` 可能小于 `kUbSize`（L220），但 `CopyIn` 仍按 `kUbSize` 列搬，依赖 K 已按 `kUbSize` 对齐的 tiling 前提。

### 5. 文件四：`hc_pre_m_split_core_premix_arch35.h`（TilingKey 1003，M 切分 + premix）

#### 5.1 文件概览

- 一个自由函数 `VFProcessPremixYFromCastSegments`（L36-L66）+ 一个类 `HcPreMSplitCorePremixArch35`（L68-L535）。
- 在文件三基础上加 premix：同一 d 窗口的 `hcMult` 段连续算，cast 结果按段驻留 `castSegBuf`（4 段），段齐后一次寄存器链算出 y 并写 GM（L336-L357）。
- 因此阶段二彻底没有 y 与 d 循环，只剩 invRms/pre/post/combFrag。
- 常量 `PREMIX_YFUSE_CAST_SEG_NUM = 4`（L33）把「hcMult=4」写死。

主线（一个 round）：

```
AIV: tbufPool1.Reset() → 装 castSeg(4 段)/x/y/preMix/ND2NZ 缓冲
     for kChunkIdx in CeilDiv(K, kUbSize):
       segIdx = kChunkIdx % hcMult;  chunkInSeg = kChunkIdx / hcMult
       kGmOffset = segIdx*d + chunkInSeg*kUbSize
       AIC: CopyInB1Nd2Nz → 等 AIV → Process ─(isLastKL1) CopyOut mmX + PRE_POST flag
       AIV: CopyIn 一个 kUb 窗 → cast(每次只 cast 一次，落到第 segIdx 段) → ND2NZ → CopyToL1
            若 segIdx == hcMult-1: 4 段 cast + preMix → 寄存器链 → yGm
     等 PRE_POST → tbufPool1.Reset()(只装 post/mixes/pre)
     for 行块: invRms → pre → post → combFrag(元素主序)
```

两阶段 UB 布局（相对文件三的增量）：

| 阶段 | 缓冲 | 与文件三的差别 |
| --- | --- | --- |
| 共用 | `mmXBuf`/`rmsNormBuf`/`hcBaseBuf*` | 相同，仅 mmXBuf 除数写作 `DOUBLE_BUFFER` |
| 阶段一 | `castSegBuf` | 新增，`4 × mUbSize × (RoundUp<float>(kUbSize)*4 + BLOCK_SIZE)` |
| 阶段一 | `preMixQue` | 新增，1 级，`mUbSize*RoundUp<float>(hcMix)*4` |
| 阶段一 | `yQue` | 新增（文件三阶段一无 y） |
| 阶段二 | `postQue`/`mixesBuf`/`preQue` | 相同；但**没有** `xQue/yQue/combFragQue` |

#### 5.2 函数逐条

- **`VFProcessPremixYFromCastSegments`** [hc_pre_m_split_core_premix_arch35.h L36-L66](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_split_core_premix_arch35.h#L36-L66)
  - 职责：用 UB 内 4 段 cast 结果与 preMix 算 `y = Σ_j preMix[row,j]·x_j` 并 cast 成 T。
  - 入参/出参：入 `yOutLocal`、`preMixLocal`、`xCastSegLocal`、`rowNum/colNum/hcMult/pmRowStride/segRows`；出参 `yOutLocal`（原地写）。
  - 实现：①`__VEC_SCOPE__` 内「行 × VL_FP32 窗口」双层循环（L53-L64）；②每窗口 `Duplicate` 清零 `sum`，对 `j<hcMult` 做 `LoadInputDataWithBrc(preMix)` × `LoadInputData(seg j)` 的 `Mul+Add` 寄存器链（L55-L61）；③`StoreOutputData<T>` 单次写出，行距 `RoundUp<T>(colNum)`（L44、L62）；④段地址 = `j*segStride + i*xCastRowStride + w*VL_FP32`，`segStride = segRows*xCastRowStride`（L46-L47）。
  - 注意：`xCastRowStride` 额外加 `BLOCK_SIZE/sizeof(float)`，对应 `castSegBuf` 每段多留的 32B（L206-L208）；4 段必须同窗口、同 `hcIdx` 顺序，否则 y 会串段。

- **`HcPreMSplitCorePremixArch35()`** [hc_pre_m_split_core_premix_arch35.h L70-L70](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_split_core_premix_arch35.h#L70-L70)
  - 职责：默认构造。
  - 入参/出参：无。
  - 实现：①空体；②`aL1BufferID_{0}`（L530）；③UB 池未初始化。
  - 注意：—

- **`Init`** [hc_pre_m_split_core_premix_arch35.h L72-L121](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_split_core_premix_arch35.h#L72-L121)
  - 职责：绑定含 `preMix` 的 9 个 GM，预分配两阶段 UB 池。
  - 入参/出参：入 `x/hcFn/hcScale/hcBase/preMix/y/post/combFrag/pre/tiling/pipe`；出参无。
  - 实现：①`preMixGm` 常绑、`preGm` 条件绑（L82-L90）；②`mmXBuf = CeilDiv(mL1Size, DOUBLE_BUFFER)*RoundUp(hcMix)`（L99-L101）；③`tbufPool0` 放 `rmsNormBuf`（半 M）+ `hcBase` 三块，内嵌 `tbufPool1`（L98-L120）；④AIC `mmService_.Init()` + 4 次 flag；AIV 才取 hcBase 指针（L103-L120）。
  - 注意：与文件三 `Init` 几乎逐行相同，差异只有 `preMixGm` 与 mmXBuf 的除数写法（`DOUBLE_BUFFER` vs `AIVS_PER_AIC`，值都是 2）。

- **`Process`** [hc_pre_m_split_core_premix_arch35.h L123-L456](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_split_core_premix_arch35.h#L123-L456)
  - 职责：M 切分 + premix —— 阶段一算 mm、cast、y 一次完成，阶段二只出 pre/post/combFrag。
  - 入参/出参：无入参；写 `yGm`（阶段一）与 `preGm/postGm/combFragGm`（阶段二）。
  - 实现：①K 迭代改成 `kChunkIdx`，`segIdx = kChunkIdx%hcMult`、`chunkInSeg = kChunkIdx/hcMult`、`kGmOffset = segIdx*d + chunkInSeg*kUbSize`（L243-L248）；②每个 chunk 只做一次 cast（`for cvLoopIdx<1`，L287），结果落到 `xCastSegLocal` 的第 `segIdx` 段（L283-L286）；③`segIdx == hcMult-1` 时调 `VFProcessPremixYFromCastSegments` 并 `CopyOut` 到 `yGm[... + chunkInSeg*kUbSize]`（L336-L357）；④k 循环后等 PRE_POST flag、`tbufPool1.Reset()` 只重建 `postQue/mixesBuf/可选 preQue`，行内 `VFProcessInvRmsPart3`→pre→post→`VFProcessCombFragPremixElementMajor`（L368-L444）。
  - 注意：`isFirstKL1/isLastKL1` 用重排后的 `kGmOffset` 判断（L250-L252），`isLastKL1` 只在「最后一段的最后一窗」成立；preMix 只在阶段一读一次，Part2 完全不碰它。

### 6. TilingKey 2×2 矩阵

| | K 切分（`isKSplit=true`，key 尾数 0） | 不切 K / 仅 M 切分（`isKSplit=false`，key 尾数 1） |
| --- | --- | --- |
| **无 premix** | **1000**<br>`hc_pre_m_k_split_core_arch35.h`<br>`HcPreMKSplitCorePart1` + `HcPreMKSplitCorePart2`<br>两个类、两个 TPipe，mm/rms 落 workspace | **1001**<br>`hc_pre_m_split_core_arch35.h`<br>`HcPreMSplitCoreArch35`<br>单类，mm 留 UB，阶段二重读 x |
| **有 premix** | **1002**<br>`hc_pre_m_k_split_core_premix_arch35.h`<br>`HcPreMKSplitCorePremixPart1` + `HcPreMKSplitCorePremixPart2`<br>y 在阶段一出，阶段二无 x/y 队列 | **1003**<br>`hc_pre_m_split_core_premix_arch35.h`<br>`HcPreMSplitCorePremixArch35`<br>y 由 4 段 cast 融合出，阶段二无 y |

### 7. 四个类共有的骨架：Init → Part1 → pipe 重建 → Part2

```
[Host] SetTilingKey(1000|1001|1002|1003)
   │
[Kernel] KERNEL_TASK_TYPE_DEFAULT(MIX_AIC_1_2); TPipe pipe;
   │
   ├─ Init(...)
   │    绑定 GM ─ 切 L1(A1) ─ 切 UB(TBuf/TQue 或 TBufPool) ─ [AIC] mmService_.Init()+播 flag
   │
   ├─ Part1.Process()
   │    for round in M 轮          ← M 切分：bs 按核分轮
   │      for kL1 in K 块          ← K 切分：multCoreSplitKSize 按核分块
   │        [AIC] CopyInB1Nd2Nz → 等 AIV → cube Process → (isLastK) CopyOut
   │        [AIV] CopyIn x → cast+invRms → ND2NZ → CopyToL1 → set flag
   │      写 rms（+ premix 版本在此写 y）
   │    drain flag; [AIC] mmService_.End(); SyncAll<false>()
   │
   ├─ 阶段边界（pipe 重建）
   │    1000/1002：pipe.Destroy(); TPipe pipeStage2; Part2.Init();
   │                 （同进程不同 UB 布局 → 必须销毁重建）
   │    1001/1003：同一个类内 tbufPool1.Reset() + 重新 InitBuffer（UB 复用，等价于重建）
   │
   └─ Part2.Process()
        行块 loop：搬 mm(+rms) → invRms(±group reduce) → pre → [y] → post → combFrag
        （1001/1003 的 y 在阶段一或阶段二重算；1002/1003 的 y 已在阶段一完成）
```

各文件在此骨架上增减：

- **1000**：最标准的两段式。Part1 只产 `mm+rms`；Part2 负责 y 并重读 x。K 切分 → Part2 必须 group reduce。
- **1002**：把 y 提前到 Part1（`yAccBuf` + `yQue`），因此 Part2 去掉 x/y 队列与 d 循环，`combFrag` 改元素主序直出。
- **1001**：合并成一个类，去掉 workspace 的 mm 往返（`mmXLocal` 留 UB），无跨核归约，阶段二用 UB 里的 `rmsNormLocal`；k 循环结束后多一次 `PIPE_FIX` 的 PRE_POST 同步。
- **1003**：在 1001 之上把 y 塞进阶段一（4 段 cast 驻留 + 寄存器链），阶段二只剩 pre/post/combFrag。

### 8. 与通用版 `hc_pre_m_k_split_core.h` 的差异

| 维度 | 通用版 `hc_pre_m_k_split_core.h` | arch35 四套 |
| --- | --- | --- |
| 同步方式 | `CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>` + 无模板参数的 `CrossCoreWaitFlag` | 统一 `SYNC_MODE4`，`CrossCoreSetFlag/WaitFlag<SYNC_MODE4, PIPE_X>(id)`，并用 `id+16` 做第二条通道 |
| 缓冲 | 中间结果大量走 workspace：`xCastFp32Ws`（双 buffer）、`mmOutFp32Ws`、`squareSumFp32Ws`，按核切分 | workspace 只剩 mm/rms（1000/1002）；1001/1003 的 mm 留在 UB，无 workspace 往返 |
| L1 使用 | `cubeCompute_.CopyInB1` 内部管理，带 `WaitBL1Mte1ToMte2Flag/SetBL1Mte1ToMte2Flag` | 显式 `L1_BUF_NUM*L1_BUF_OFFSET` 劈 x/w，靠 `aL1BufferID_ ^= 1` 乒乓 |
| 循环结构 | `ProcessMChunk` 抽函数，内含 B1/A1 预取状态机（`b1Prefetched/a1Prefetched`），`cvLoopIdx_` 跨 chunk 连续 | 无预取状态机，每块重新 `CopyInB1Nd2Nz`；1002/1003 把 K 循环重排为「hc 段 × d 窗口」 |
| M 分核 | `GetMkBlockInfo` 用 `multCoreSplitMSize` 定长区间 + `mGmEndOffset` 截断 | 用 `mCnt/singleCoreMaxRound/mainCoreCount` 算轮数，尾核少一轮，再按 `mL1RealSize` 截断 |
| 尾块处理 | `realMSize/realKGmSize` 逐层取 min，`cvLoopKSize` 为 K 块 | 1000/1002 用 `kL1RealSize` + `cvLoopKSize=CeilDiv(kL1,kUb)`；1001/1003 用固定 `cvLoopKSize=kL1/kUb` + 尾块 `kRealSize` |
| 一对 AIV 的行切 | `mVectorLength=(realMSize+1)/2`，偶数核拿 ceil（`curVectorBlockIdx%2`） | 1000 偶数核拿 floor（`mL1RealSize/2`）；1002/1003 偶数核拿 ceil（`CeilDiv`） |
| 阶段切分 | Part1/Part2 两个类 + 两个 TPipe | 1000/1002 相同；1001/1003 单类 + `tbufPool1.Reset()` |
| UB 管理 | 直接 `pipe->InitBuffer`，无池复用 | 1001/1003 用 `TBufPool` 两阶段复用（`UB_POOL_BUFFER_COUNT` 12/16） |

### 9. 容易踩的坑（汇总）

1. 1000 的 AIV 半行切分是 floor/ceil，1001/1002/1003 的偶数核多拿一行 —— 混用会让 `rmsGm` 与 `mmGm` 行错位。
2. 1002/1003 的 `isLastK` 必须用 `groupIdx*hcMult+hcIdx` / 重排后的 `kGmOffset` 判断，不能只看窗口号。
3. 1000/1002 的 `CrossCoreSetFlag` 是 4 次、AIV 只 drain 2 次，这是刻意的双 buffer 信用，不是笔误，但改动循环次数时必须重算。
4. 1001 的 `mGmOffset` 用 `<=` 判长核、`singleCoreRound` 用 `<`，两者边界不一致（[hc_pre_m_split_core_arch35.h L113-L134](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_split_core_arch35.h#L113-L134)）。
5. 1001 中 `mBlkDimIdx/kBlkDimIdx`（L108-L109）是未使用变量；`init` 里 `mmXBuf` 的除数写成 `AIVS_PER_AIC`、1003 写成 `DOUBLE_BUFFER`，值相同但语义不同。
6. 1002 的 `Part2.Init` 有 3 个 GM 形参匿名（第 1、4、5 个），调用位置一错就会静默绑错。
7. 1003 的 y 依赖「4 段 cast 同窗口驻留」，`cvLoopIdx<1` 是刻意的单次循环；`PREMIX_YFUSE_CAST_SEG_NUM=4` 与 `hcMult` 强绑定。
8. 所有 `rmsGm` 偏移都假设 mm 占用 `kBlockFactor*bs*hcMix` 个 float，改 workspace 布局必须同时改文件一/二的 Init。

### 10. tiling 字段速查（读代码时最常拦路的 20 个）

| 字段 | 含义 | 主要出现位置 |
| --- | --- | --- |
| `bs` | batch 行数（M 总长） | 所有 M 循环 `CeilDiv(bs, mL1Size)` |
| `d` | 单头 hidden 维 | `hcMult*d` 即 K 总长 |
| `hcMult` / `hcMix` | 头数 / 混合维（sinkhorn 的 N） | cube 的 N、mixes 行距 |
| `hcMultAlign` | 头数按 8 对齐 | 输出队列行距、`ubRowGapBlocks_` |
| `mL1Size` / `kL1Size` | L1 上 M/K 分块 | 外/内层循环步长 |
| `mUbSize` / `kUbSize` | UB 上 M/K 分块 | cast/ND2NZ/CopyIn 的列宽 |
| `multCoreSplitKSize` | 每核 K 长度 | 1000/1002 的 K 分核；1001/1003 的 K 上界 |
| `cubeBlockDimM/K` | M/K 方向核数 | 1000/1002 两者都用；1001/1003 只用 M |
| `kBlockFactor` | 阶段二搬几份 mm | `rmsAndmmQue` 与 `rmsGm` 行距 |
| `stage2RowFactor` | 阶段二每次处理行数 | Part2 的队列尺寸 |
| `rowInnerFactor` | 1001/1003 阶段二行块 | 行内循环步长 |
| `dFactor` / `dLoop` / `tailDFactor` | 阶段二按 d 切片 | `VFProcessY` 的 d 循环 |
| `secondUsedCoreNum` | 阶段二实际用核数 | Part2 早退判断 |
| `rowOfFormerBlock` | 阶段二每核行数 | 三个 GM 的 base offset |
| `rowLoopOfFormerBlock` / `tailRowFactorOfFormerBlock` | 非末核循环数 / 尾行 | 与 `*OfTailBlock` 成对使用 |
| `iterTimes` | sinkhorn 迭代次数 | `VFProcessCombFrag*` 的 `iterTimes-1` |
| `normEps` / `hcEps` | RMS 归一化 / pre-post 的 eps | `VFProcessInvRmsPart3*`、`VFProcessPre/Post` |
| `bufferPool0Size` / `bufferPool1Size` | 两级 UB 池大小 | 1001/1003 的 `InitBufPool` |
| `kBlockFactor*bs*hcMix` | mm 在 workspace 的长度 | `rmsGm` 的起始偏移 |
| `AIVS_PER_AIC`=2 / `FLAG_ID_MAX`=16 | 常量 | 分核与 flag 通道 |

### 11. 建议阅读顺序

1. 先读 §7 的骨架图，再读文件三（1001）—— 它是「最少特例」的全流程单类实现。
2. 再读文件一（1000），对照理解「为什么 K 切分要把 mm/rms 落 workspace、为什么 Part2 还要 group reduce」。
3. 然后读文件二（1002）—— 只在文件一上加了 `yAcc` 与 K 重排，重点看 `hcIdx` 三层循环。
4. 最后读文件四（1003）—— 只把 1001 的 K 循环按段重排 + 融合 y，其余逐行对照即可。
5. 每读一个 `Process`，先只看 `if ASCEND_IS_AIC` / `else` 两支的对称点（谁 set flag、谁 wait flag），再回头看数据搬运。


---

## 18. 核对说明

本文所有结论均对照当前分支源码逐条核对：

| 结论 | 核对位置 |
| --- | --- |
| 输入/输出/attr 定义 | [hc_pre_def.cpp L24-L73](../../../../csrc/moe/hc_pre/op_host/hc_pre_def.cpp#L24-L73) |
| SoC 注册（910b / 910_93 / 950+regbase） | [hc_pre_def.cpp L75-L83](../../../../csrc/moe/hc_pre/op_host/hc_pre_def.cpp#L75-L83) |
| proto 为空文件 | `hc_pre_proto.cpp` 0 行 |
| 通用版 TilingKey 0/1/2 与阈值 32 | [hc_pre_tiling.cpp L62-L65](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L62-L65)、[L207-L225](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L207-L225) |
| arch35 TilingKey 公式 | [hc_pre_tiling_arch35.h L755](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h#L755) |
| SoC 分流判据 | [hc_pre_tiling.cpp L736-L742](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L736-L742) |
| MIX_AIC_1_2 | [hc_pre.cpp L38](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp#L38) |
| `__DAV_C310__` 分派 | [hc_pre.cpp L19-L30](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp#L19-L30)、[L50-L89](../../../../csrc/moe/hc_pre/op_kernel/hc_pre.cpp#L50-L89) |
| pre/post 公式与 eps/×2 | [hc_pre_base.h L489-L502](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L489-L502)、[L727-L740](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L727-L740) |
| y = Σ pre_j·x_j（用原始 x） | [hc_pre_base.h L626-L694](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_base.h#L626-L694) |
| InvRMS 公式 | [hc_pre_m_k_split_core.h L409-L431](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L409-L431) |
| Sinkhorn 迭代结构 | [hc_pre_m_k_split_core.h L627-L660](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L627-L660) |
| hc_fn 24 行的三段偏移 | [hc_pre_m_k_split_core.h L662-L671](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L662-L671)、[L851-L852](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h#L851-L852) |
| 手写 Mmad（非 Matmul API） | [hc_pre_cube_compute.h L83-L89](../../../../csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h#L83-L89) |
| workspace 公式 | [hc_pre_tiling.cpp L682-L700](../../../../csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp#L682-L700) |
| 形状/dtype 校验 | [torch_binding.cpp L1582-L1627](../../../../csrc/torch_binding.cpp#L1582-L1627) |

> 若后续分支改动导致行号漂移，以上「文件 + 函数名」仍可作为稳定锚点（函数名一般不改）。
