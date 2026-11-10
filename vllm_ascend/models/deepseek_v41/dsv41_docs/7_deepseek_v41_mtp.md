# DeepSeek-V4.1-Flash MTP / 投机草稿管线详解

> **代码版本冻结声明**：本文基于当前 `vllm-ascend` 工作区源码。目标模型在 [model.py](../model.py)，草稿模型在 [dspark.py](../dspark.py)；文中 `Lxxx` 均已逐条核对。上代 V4 见 [6_deepseek_v4_mtp.md](../../deepseek_v4/dsv4_docs/6_deepseek_v4_mtp.md)；DSpark 的块内结构（Markov/Confidence 头、非因果块注意力）在 [8_deepseek_v41_dspark.md](8_deepseek_v41_dspark.md)，本文只在"接线"层面引用。
>
> **配置基线**（FP8 权重 + FP4 专家，取自 [config.json](0_DeepSeek-V4.1-Flash-1017_config.json)）：

| 参数 | 值 | 与本文的关系 |
|------|-----|------------|
| `hidden_size` / `hc_mult` | **5120** / 4 | 单流宽度 `H`、残差流条数 `C`（**不是 V4 的 7168**） |
| `hc_mult*hidden_size` | **20480** | 4 流拼平宽度 `D` = `_mtp_hidden_buffer` 列数 |
| `num_hidden_layers` / `num_nextn_predict_layers` | **40** / 3 | 主模型层号 0~39；草稿层号 **40 / 41 / 42** |
| `dspark_target_layer_ids` | **[37,38,39]** | 取这 3 层**入口**残差做融合（checkpoint 零基编号） |
| `dspark_block_size` / `dspark_noise_token_id` | **5** / 128799 | 一次出一个 5-token 块；块内非首位填噪音 token |
| `dspark_markov_rank` / `dspark_n_routed_experts` | **256** / 128（top-3） | Markov 低秩瓶颈（V4 是 512）；草稿 MoE 比主模型（384 / 6）小得多 |
| `vocab_size` / `hc_sinkhorn_iters` | 129280 / 20 | full-vocab 草稿无 d2t 映射；sinkhorn 迭代数 |

> **先正名（最容易混的一点）**：V4.1 的草稿**方法**是 **DSpark**（`speculative_config.method="dspark"`），`mtp.*` 只是 checkpoint 里的**权重目录名**——V4.1 根本没有经典 MTP 草稿，上游对 `method="mtp"` 直接报错并要求改用 `dspark`（[speculative.py:L1366-L1376](../../../../../vllm/vllm/config/speculative.py#L1366)）。而 Ascend 的 `AscendDSparkProposer` **继承 `AscendDflashProposer`**，所以 DSpark 路径里会出现 `dflash_proposer.py` / `_dflash_hidden_states` 这类名字。三者的完整对照见 [8_deepseek_v41_dspark.md](8_deepseek_v41_dspark.md) §1.4。

**一句话**：V4.1 投机管线 = ① **延迟 mHC 系数交接**（子层用"上一个子层算出的 mix"做 collapse，把 Cube 依赖链拉长一个子层）+ ② **预 collapse 的 4 流隐态缓冲** + ③ **草稿层 40/41/42 与 checkpoint `mtp.{0,1,2}` 的重映射** + ④ **`mtp.*` 在目标 loader 里被显式跳过**。任何一条错位都不会崩，只会让接受率归零。

---

## 1. vLLM step 上下文：三个草稿层吃什么

### 1.1 异步调度：placeholder 与草稿延迟回填

一步 = 「① 调度 → ② 目标前向（`non_block=True` 发起）→ ③ 验证采样（第二次 RPC）→ ④ 草稿前向（嵌在 ③ 内）→ ⑤ 结算 + 草稿回写」。同步路径在 ③ 处原地阻塞等采样；异步路径把 `(采样Future, SchedulerOutput, 前向Future)` 塞进 `batch_queue` 后立刻排下一批。代价是调度器必须在**草稿 token 还没算出来**时先排下一步，于是用 `-1` 占位：

```python
# vllm/v1/core/sched/async_scheduler.py:18, :50
self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
request.spec_token_ids = self._spec_token_placeholders    # 先占住 K 个草稿位
```

`[-1]*K` 保证 decode 批形状恒为 `1+K`；真实草稿 id 留在 **worker 侧输入批**，由 worker 下一步消费——`post_step` 的草稿回写被整块跳过（`if ... and not self.async_scheduling`），`ModelRunnerOutput` 也不再携带 `spec_token_ids`。

| 维度 | 同步（`step`） | 异步（`step_with_batch_queue`） |
|------|------|------|
| 采样 RPC | 阻塞 `sample_tokens(...)` | `non_block=True` 入队，队满才结算最老批次 |
| 草稿回写 | `post_step → update_draft_token_ids` | **跳过**，草稿留在 worker 输入批 |

> 草稿前向照旧在 ③ 的 RPC 内跑完，故 §2~§5 的隐态/权重路径只与"批里有哪些 token"有关；但 ③ 取到的隐态必须是**本步刚写完**的。

### 1.2 调用栈与三条隐态通道

```
EngineCore.step / step_with_batch_queue
├─ ① 调度：装入上一步草稿 D(t-1)，新请求用 [-1]*K 补齐
├─ ② 目标前向 → DeepseekV41Model.forward                        # model.py:1517
│     · 4 流展开 + pre_mix one-hot 初始化                        # L1563-1565
│     · 逐层：aux 采集 → layer(...) → (h, pre_mix)               # L1579-1611
│     · ★ 先把 4 流残差写进 _mtp_hidden_buffer                   # L1613-1619
│     · 再 hc_collapse(最后一层 ffn_pre) → norm                  # L1621-1624
├─ ③ 验证采样，内嵌草稿前向（combine → 预写 KV → 噪音块 → 3 层 → markov/confidence）
└─ ⑤ 结算：接受/拒绝/重采样 + bonus token；草稿供下一步 ①
```

| 通道 | 数据 | 生产者 | 消费者 |
|------|------|--------|--------|
| **A 采样** | `norm(hc_collapse(...))` `(T,5120)` | `model.py:1621-1624` | `compute_logits` / 采样器 |
| **B 4 流缓冲** | `(T,20480)`，**pre-collapse** | `model.py:1619`（`copy_`） | `get_mtp_target_hidden_states()` → `get_drafter_hidden_states`（[utils.py:16-32](../../../../../vllm/vllm/v1/worker/gpu/spec_decode/utils.py#L16)） |
| **C aux 残差** | 3 层入口 4 流均值，`(T,15360)` | `model.py:1579-1583` | `torch.cat(..., -1)`（[model_runner.py:2187](../../../worker/v1/model_runner.py#L2187)）→ `combine_hidden_states` |

> **V4.1 与 V4 的第一个差异**：V4 的 aux 采集在**层输出后**（[deepseek_v4/model.py:1013-1017](../../deepseek_v4/model.py#L1013)），V4.1 在**层输入前**（`model.py:1579`，注释："DSpark consumes the residual stream **entering** its configured target layers"），与参考实现"the MTP head reads the attention input of its target layers, not their output"（[reference/model.py:1267-1269](reference/model.py#L1267)）一致。

---

## 2. 延迟 mHC 系数交接（delayed mHC coefficient handoff）

> V4.1 相对 V4 唯一的**结构性**改动，类注释即："V4.1 block with the checkpoint's delayed mHC coefficient handoff"（`model.py:769`）。

### 2.1 V4 基线：mix 当场算、当场用

```python
# deepseek_v4/model.py:783-787 —— V4
def hc_pre(self, x, hc_fn, hc_scale, hc_base):
    y = torch.ops._C_ascend.npu_hc_pre_v2(
        x, hc_fn, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.norm_eps, self.hc_eps)
    return y                 # 3 输出 y/post/comb；mix 算子内部自算自用
```

对应 `forward` 返回 `(hidden_states, residual)`（`deepseek_v4/model.py:829`），层间只传残差本体，系数不外传。

### 2.2 V4.1：算子多一个输入、多一个输出

```python
# model.py:858-869
def hc_pre(self, x, hc_fn, hc_scale, hc_base, pre_mix=None):
    return torch.ops._C_ascend.npu_hc_pre_v3(
        x, hc_fn, hc_scale, hc_base,
        pre_mix,                                  # ★ 外部传入的 collapse 系数
        hc_mult=self.hc_mult, hc_sinkhorn_iters=self.hc_sinkhorn_iters,
        norm_eps=self.norm_eps, hc_eps=self.hc_eps)
```

算子签名同步扩成 **4 输出**：返回 `std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>`，形参表在 v2 的四个张量之后插入 `const c10::optional<at::Tensor>& pre_mix`，函数体直接 `return run_hc_pre_fusion(...)`（[torch_binding.cpp:1656-1663](../../../../csrc/torch_binding.cpp#L1656)）。

| 输出 | 形状 | 含义 |
|------|------|------|
| `y` | `(T,H)` | 用**传入的** `pre_mix` collapse 出的单流（送进 attention/FFN） |
| `post` / `comb` | `(T,C)` / `(T,C,C)` | 子层输出重展开门控 / 残差回混矩阵（`hc_post` 用） |
| **`pre`** | **`(T,C)`** | **本次新算出的 collapse 系数**，本子层不用，只交给下游 |

`pre` 形状由 `construct_hc_pre_pre_output_tensor` 定：3 维输入为 `{x.size(0), hc_mult}`（`L1571-1580`）。`pre_mix` 被 `check_hc_pre_shape_and_dtype` 强校验为 **fp32、`x.dim()-1` 维、各轴与 `x` 前缀一致**（`L1618-1626`）。

### 2.3 三处交接点

```python
# model.py:879-920
residual = hidden_states
x, attn_post, attn_comb, attn_pre = self.hc_pre(
    hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
    pre_mix)                    # ★① attention 用【调用方】的 mix（上一层 FFN 算的）
x = self.self_attn(positions, self.input_layernorm(x), llama_4_scaling)
hidden_states = self.hc_post(x, residual, attn_post, attn_comb)

residual = hidden_states
x, ffn_post, ffn_comb, ffn_pre = self.hc_pre(
    hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
    attn_pre)                   # ★② FFN 用【本层 attention】刚算出的 mix
x, x_fp32 = self.rms_norm_cast(x)
x = self.mlp(x, input_ids=input_ids, hidden_states_fp32=x_fp32,
             already_sequence_parallel=use_sequence_parallel)
hidden_states = self.hc_post(x, residual, ffn_post, ffn_comb)
return hidden_states, ffn_pre   # ★③ 本层 FFN 的 mix 交给下一层 attention
```

`hc_post`（`model.py:871-877`）用本子层的 `post`/`comb` 把单流输出重展开成 C 条流并回混残差。串起来是一条跨两层的流水线：

```
layer N-1                      layer N                        layer N+1
FFN hc_pre ─► ffn_pre ────► attn hc_pre(pre_mix=ffn_pre)
                                  │ 顺带产出 attn_pre
                                  └────► FFN hc_pre(pre_mix=attn_pre)
                                                │ 顺带产出 ffn_pre
                                                └──────────► attn hc_pre(pre_mix=ffn_pre) → …
```

模型级入口初始化并逐层传递（`model.py:1563-1565`、`1611`）：`pre_mix` 是 fp32 的 `(T,C)` 零张量，只把 `[:,0]` 置 1 形成 **one-hot**——首层 attention 的 collapse 因此退化为"只取第 0 条流"（等价参考实现的 `make_identity_pre_mix`，[reference/model.py:1162-1166](reference/model.py#L1162)）。

结尾不再需要专用 `hc_head` 投影，而是**复用最后一层 FFN 的 `ffn_pre`** 做纯加权和（`model.py:1621`）：

```python
hidden_states = last_layer.hc_collapse(hidden_states, pre_mix)

# model.py:854-856
@staticmethod
def hc_collapse(x, pre_mix):
    return (pre_mix.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)
```

> **可核对**：V4.1 的 [weight index](0_DeepSeek-V4.1-Flash-1017_model_weights.index.json) 里**检索不到任何 `hc_head_*`**，`model.py` 也只有 `hc_collapse` 而无 `hc_head`。dspark 草稿侧仍保留 `hc_head`（`dspark.py:356-363`）与 `hc_head_*` 加载分支（`dspark.py:546`、`637`）——那是草稿自己的输出压缩。

### 2.4 为什么更快：把 Cube 依赖链拉长一个子层

> **类比**：原来每个子层"先算今天的菜单再下锅"，锅（Cube）得等菜单；现在"用昨天的菜单下锅，同时把今天的菜单算好留给明天"。

Ascend 上 Cube 做 MatMul、Vector 做 elementwise。`hc_pre` 里算 mix 的部分是一次 **`hc_fn @ x_flatten^T` 的 Cube GEMM**（`hc_fn: (mix_hc, D)`，`mix_hc=(2+C)*C=24`，`model.py:824-826`）。若它产出的 fp32 mix 立刻被同一子层的 collapse 消费，就形成 `Cube → Vector → Cube` 的**紧依赖**，中间没有别的活可填。V4.1 让 collapse 消费的是**上一个子层顺带算出的 mix**：生产者是"上一子层的 `hc_pre`"，消费者是"本子层 `hc_pre`"，之间隔着**一整个子层的 attention/FFN**；每子层仍只发**一次**融合算子（算 mix、算 post/comb、做 collapse 都在 `npu_hc_pre_v3` 内），只是本次 collapse 用的系数来自入参——**总算力不变，关键路径少一次"必须等自己算完"**。

| 维度 | V4（`npu_hc_pre_v2`） | V4.1（`npu_hc_pre_v3`） |
|------|------|------|
| 算子输出 / `pre_mix` 入参 | 3 输出 / 无 | 4 输出 / 有（可选，fp32 `(T,C)`） |
| collapse 系数来源 | 本子层现算现用 | **上一个子层**现算，本子层直接用 |
| 层间返回值 | `(hidden, residual)` | `(hidden, pre_mix)` |
| 末尾单流化 | 专用 `hc_head`（独立投影 + `hc_head_*` 权重） | `hc_collapse`（纯加权和，复用最后一层 `ffn_pre`） |

### 2.5 不变量（改这块代码前必读）

| 不变量 | 依据 | 破坏后果 |
|--------|------|---------|
| `pre_mix` 必须 fp32、`(T,C)`、行数 = `x.size(0)` | `torch_binding.cpp:1618-1626` | 直接抛断言 |
| 首层 `pre_mix` 是 one-hot `[:,0]=1` | `model.py:1564-1565` | 首层退化为 4 流均匀平均，HC 语义丢失 |
| `hc_collapse` 在 fp32 域加权后回落原 dtype | `model.py:856` | 精度下降 |
| SP 开启时 `hidden_states` 与 `pre_mix` 必须**同时** gather | `model.py:1615-1617` | token 轴错位（§7.2） |
| 末尾只能用**最后一层**的 `hc_collapse` | `model.py:1621` | 误用 `hc_head` 会缺权重 |

---

## 3. 草稿侧的隐态捕获

### 3.1 `_mtp_hidden_buffer`：按需分配

```python
# model.py:983-997
needs_mtp_hidden_states = spec_config is not None and (
    spec_config.use_eagle() or spec_config.uses_draft_model())
self._mtp_hidden_buffer = (
    torch.empty(vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_mult * config.hidden_size,        # 4*5120 = 20480
                dtype=vllm_config.model_config.dtype, device=self.device)
    if get_pp_group().is_last_rank and needs_mtp_hidden_states else None)
```

1. **只在最后一个 PP rank 分配**——只有它做 `hc_collapse`/`norm`，也只有它有完整 4 流；
2. **只在需要隐态的投机方法下分配**——`use_eagle()` 覆盖 `eagle/eagle3/mtp/dflash/dspark`（[speculative.py:1944-1948](../../../../../vllm/vllm/config/speculative.py#L1944)），`uses_draft_model()` 覆盖独立草稿模型（`L1963-1964`）；不开投机即 `None`，省下 `max_num_batched_tokens × 20480 × 2B`（每 4K token ≈ 160MB）级的常驻显存；
3. 第一维用 `max_num_batched_tokens`，保证 FULL graph 重放永远写同一地址。

### 3.2 写入点：`hc_collapse` **之前**

```python
# model.py:1611-1624
hidden_states, pre_mix = layer(positions, hidden_states, pre_mix, None, input_ids=input_ids)
# MTP needs full HC states
if self._mtp_hidden_buffer is not None:
    if use_sequence_parallel:
        hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        pre_mix      = sp_all_gather(pre_mix)[:full_num_tokens]
    num_tokens = hidden_states.shape[0]
    self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

hidden_states = last_layer.hc_collapse(hidden_states, pre_mix)
if use_sequence_parallel and self._mtp_hidden_buffer is None:
    hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
```

- **顺序不可交换**：`copy_` 必须在 `hc_collapse` 之前，否则拿到压成单流的 `(T,5120)`，草稿侧 `(T,C,H)` 直接对不上；
- `flatten(1)` 把 `(T,4,5120)` 摊成 `(T,20480)`，**C 在中间维**；草稿侧再 `view(-1,C,H)` 还原；
- 用 `copy_` 而非赋值：`forward` 必须留下"写固定地址"的副作用，图捕获后重放才落到同一张量；
- SP 的 gather 在这里做（要保 4 流），所以下面那次 gather 加了 `and ... is None` 条件，避免重复 gather。

### 3.3 `aux_hidden_states`：一层一号、只取入口、对 C 求均值

```python
# model.py:1576-1583
# DSpark consumes the residual stream entering its configured
# target layers. The runner expresses checkpoint IDs as one-based.
if layer.layer_idx + 1 in self.aux_hidden_state_layers:
    aux_hidden_state = hidden_states.mean(dim=1)
    if use_sequence_parallel:
        aux_hidden_state = sp_all_gather(aux_hidden_state)[:full_num_tokens]
    aux_hidden_states.append(aux_hidden_state)
```

1. **位置**：在 `layer(...)` **之前**（V4 是之后），拿到"进入该层的残差"，与 `main_hiddens.append(h.mean(dim=2))`（[reference/model.py:1267-1269](reference/model.py#L1267)）一致；
2. **编号**：`layer_idx` 零基，`+1` 才是一层基；runner 把 checkpoint 零基 id 加一后写入（`model_runner.py:827-833`）：`[37,38,39]` → `(38,39,40)` → 命中 `layer_idx ∈ {37,38,39}`。**最容易错一位的地方**（§7.7）；
3. **聚合**：`mean(dim=1)` 对 C 条流求均值得 `(T,5120)`——aux 是"4 流均值"，与 §3.2 的"4 流全量"是两种不同压缩。

层顺序即列表顺序，runner 沿最后一维拼接（`model_runner.py:2187`）：`(T,15360)`，列块顺序严格 **37→38→39**，与 `main_proj`（`ColumnParallelLinear(15360→5120)`，`dspark.py:223-231`）的输入列分块一一对应。**层序换了权重就全错位，但不会报错。**（PP 复现路径 `_run_replay_layers`（`model.py:1631-1637`）须保持同样三条约定。）

### 3.4 两条喂数路线

| 路线 | 来源 | 消费者 | 触发条件 |
|------|------|--------|---------|
| **A. aux** | 3 层入口均值 `(T,15360)` | `combine_hidden_states`（`dspark.py:312-313`）→ `main_proj/main_norm` → `main_x` | `method=="dspark"` 且开了 aux 输出 |
| **B. 4 流** | `_mtp_hidden_buffer` `(T,20480)` | v1：`hidden_states = mtp_hidden_states`，**仅当 `method=="mtp"`**（`model_runner.py:2177-2178`）；v2：`get_drafter_hidden_states`（[utils.py:16-32](../../../../../vllm/vllm/v1/worker/gpu/spec_decode/utils.py#L16)） | `use_eagle()`/`uses_draft_model()` 且方法确实要 pre-hc_head 残差 |

> 因此 V4.1 在 dspark 部署下 `_mtp_hidden_buffer` **会被分配也会被写**（`use_eagle()` 为真），但 v1 的 rebind 卡在 `method=="mtp"`，真正读它的是 v2 的 `get_drafter_hidden_states`。这是排查"buffer 有数据但草稿不变"的第一现场（§7.6）。

---

## 4. 草稿模型接线

### 4.1 三个 draft 层：40 / 41 / 42

```python
# dspark.py:160-183
self.block_size = int(config.dspark_block_size)                 # 5
self.target_layer_ids = list(config.dspark_target_layer_ids)    # [37,38,39]
self.num_dspark_layers = _get_dspark_num_mtp_layers(config)     # 3
self.mtp_start_layer_idx = config.num_hidden_layers             # 40
self.layers = torch.nn.ModuleDict({
    str(self.mtp_start_layer_idx + idx): DeepseekV41DSparkDecoderLayer(
        vllm_config, prefix=f"mtp.{idx}", config=config, is_draft_layer=True)
    for idx in range(self.num_dspark_layers)})
```

| 运行时层号 | 模块 prefix（=checkpoint 命名） | 备注 |
|-----------|------------------------------|------|
| `model.layers.40` | `mtp.0` | 挂 `main_proj` / `main_norm` |
| `model.layers.41` | `mtp.1` | 普通 draft 层 |
| `model.layers.42` | `mtp.2` | 挂 `norm` / `markov_head`；`confidence_head` 挂 model 级 |

层数优先级（`dspark.py:78-82`）：`n_mtp_layers` > `dspark_num_mtp_layers` > 3。

### 4.2 复用主模型的"延迟 mHC"块

`DeepseekV41DSparkDecoderLayer` 只换 attention 类，**forward 完全继承**（`dspark.py:145-148`）：`attention_cls = DeepseekV41DSparkAttention`。因此 §2 的 `pre_mix` 接力在草稿侧一模一样地跑（`dspark.py:291-304`）：`pre_mix` 同样 one-hot 初始化，循环体逐层 `hidden_states, pre_mix = layer(positions, hidden_states, pre_mix, llama_4_scaling=None, input_ids=input_ids)`，最后 `last_layer.hc_collapse(hidden_states, pre_mix)`。

草稿 attention 走 **SWA-only** 后端（`DeepseekV41DSparkAttention(DeepseekV41SWAAttention)`，`dspark.py:109-118`），构造时 `role` 固定为 `compress_ratio=0`、无 kv/index/candidate source（`dspark.py:121-137`）——草稿层不需要 Compressor/Indexer/LongKV，只有滑窗 KV。

### 4.3 入口类与 Eagle3 接口

```python
# dspark.py:390-394
@support_torch_compile
class DSparkDeepseekV41ForCausalLM(torch.nn.Module, DeepseekV41MixtureOfExperts, SupportsEagle3):
    draft_id_to_target_id = None
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
```

- `SupportsEagle3` 让 runner 把它当作"吃目标隐态"的草稿器，从而允许 `set_aux_hidden_state_layers`（否则 [model_runner.py:4751-4755](../../../worker/v1/model_runner.py#L4751) 直接抛 `Model does not support EAGLE3 interface`）；
- `draft_id_to_target_id = None` → full-vocab 草稿（`dspark.py:462-467`）；改成 reduced-vocab 会让 `llm_base_proposer` 退化成 greedy（[llm_base_proposer.py:1558-1566](../../../worker/v1/spec_decode/llm_base_proposer.py#L1558)）；
- `set_aux_hidden_state_layers`（`model.py:1839-1840`）转发到 `EagleModelMixin._set_aux_hidden_state_layers`，后者做 `tuple(sorted(set(layers)))`（[interfaces.py:1577-1581](../../../../../vllm/vllm/model_executor/models/interfaces.py#L1577)）——**排序**保证 aux 按层号升序，`main_proj` 的列分块顺序才不会飘。

### 4.4 draft KV cache 的层名

```python
# dspark.py:309-310
def get_draft_kv_cache_layer_names(self) -> list[str]:
    return [layer.self_attn.dsa_attn.swa_cache_layer.prefix for layer in self.layers.values()]
```

入口类转发（`dspark.py:491-492`）。proposer 用它从 `kv_cache_config` 挑出**属于草稿的** cache group（`_draft_attn_layer_names`，[dspark_proposer.py:138-146](../../../worker/v1/spec_decode/dspark_proposer.py#L138)），再建 per-group 的 block table / slot mapping。draft cache 的 spec 被改写成 `AscendSlidingWindowMLASpec`（`dspark.py:92-103`），与主模型 MLA cache 区分——这是草稿层能独立分 KV 显存的前提。

---

## 5. 权重映射：`mtp.{i}.*` → `model.layers.{40+i}`

### 5.1 目标模型：先跳过

```python
# model.py:334-337
def get_spec_layer_idx_from_weight_name(config: PretrainedConfig, weight_name: str) -> int | None:
    if weight_name.startswith("mtp."):
        return 0
    return None
```

调用点在 `_load_model_weights` 最前面（`model.py:1871-1873`）：`mtp.*` 一律 `continue`。**目标 loader 永不加载草稿权重**，同一份 checkpoint 被两个 loader 各读一遍。

### 5.2 草稿模型：三段式重映射

`_remap_checkpoint_name`（`dspark.py:623-662`）先按 stage 分派，再替换组件名：

```python
m = re.match(r"mtp\.(\d+)\.(.*)", name)                       # L624
stage, rest = int(m.group(1)), m.group(2)
if stage == self.model.num_dspark_layers - 1 and rest.startswith("confidence_head."):
    return f"model.{rest}"                                    # L630-631
if stage == 0 and rest == "embed.weight":   return "model.embed_tokens.weight"   # L633-634
if stage == self.model.num_dspark_layers - 1 and rest == "head.weight":
    return "lm_head.weight"                                   # L635-636
if rest.startswith(("hc_head_fn", "hc_head_base", "hc_head_scale")):
    return f"model.{rest}"                                    # L637-638

first_layer_idx = self.config.num_hidden_layers               # 40
last_layer_idx  = first_layer_idx + self.model.num_dspark_layers - 1   # 42
if rest.startswith(("main_proj.", "main_norm.")):  layer_idx = first_layer_idx   # L642-643
elif rest.startswith(("norm.", "markov_head.")):   layer_idx = last_layer_idx    # L644-645
else:                                             layer_idx = first_layer_idx + stage  # L646-647
name = f"model.layers.{layer_idx}.{rest}"
```

组件名替换表（`dspark.py:650-661`）：

| checkpoint 片段 | 运行时片段 | | checkpoint 片段 | 运行时片段 |
|----------------|-----------|---|----------------|-----------|
| `.attn.` | `.self_attn.` | | `.w1.` / `.w2.` / `.w3.` | `.gate_proj.` / `.down_proj.` / `.up_proj.` |
| `.ffn_norm.` | `.post_attention_layernorm.` | | `.mlp.gate.bias` | `.mlp.gate.e_score_correction_bias` |
| `.attn_norm.` | `.input_layernorm.` | | `.ffn.` | `.mlp.` |

外加 `_remap_dspark_name` 的三条 V4.1 专属改写（`dspark.py:419-431`）——checkpoint 用**算子名**、运行时用**参数名**：`.markov_head.embed.weight → .markov_head.markov_w1.weight`、`.markov_head.head.weight → .markov_head.markov_w2.weight`、`model.confidence_head.weight → model.confidence_head.proj.weight`。

### 5.3 真实 checkpoint 名 → 运行时名（已核对 weight index）

| checkpoint 名 | 运行时名 | 依据 |
|--------------|---------|------|
| `mtp.0.main_proj.weight` / `.scale` | `model.layers.40.main_proj.*` | `L642-643` + `.scale→.weight_scale`（`L559-560`） |
| `mtp.0.main_norm.weight` | `model.layers.40.main_norm.weight` | `L642-643` |
| `mtp.0.attn.wq_a.weight` | `model.layers.40.self_attn.wq_a.weight` | `L646-647` + `.attn.→.self_attn.` |
| `mtp.0.attn.wkv.weight` | `model.layers.40.self_attn.wkv.weight` | 同上；**另复制一份**进 `context_wkv_proj`（§5.4） |
| `mtp.0.hc_attn_fn` / `hc_ffn_fn` / `*_base` / `*_scale` | `model.layers.40.hc_*` | `L646-647`（不走 `hc_head_*` 分支） |
| `mtp.1.attn.wq_a.weight` | `model.layers.41.self_attn.wq_a.weight` | `L646-647` |
| `mtp.1.ffn.gate.bias` | `model.layers.41.mlp.gate.e_score_correction_bias` | `L658` |
| `mtp.2.norm.weight` | `model.layers.42.norm.weight` | `L644-645` |
| `mtp.2.markov_head.embed.weight` | `model.layers.42.markov_head.markov_w1.weight` | `L644-645` + `L428` |
| `mtp.2.markov_head.head.weight` | `model.layers.42.markov_head.markov_w2.weight` | `L644-645` + `L429` |
| `mtp.2.confidence_head.proj.weight` | `model.confidence_head.proj.weight` | `L630-631` |
| `embed.weight` / `head.weight`（顶层共享） | `model.embed_tokens.weight` / `lm_head.weight` | `L542-545`（非旋转量化时） |

> **`main_proj` 挂 stage 0、`norm`/`markov_head` 挂 stage 2**，解释了两处"看起来挂错位置"的代码：`first_layer.main_proj = self.main_proj`（`dspark.py:233-234`）与 `last_layer.norm/markov_head = ...`（`dspark.py:250-252`）——运行时是 model 级属性，只是**借某层做参数命名宿主**，以便与 checkpoint 的 stage 归属对齐。

### 5.4 两份 WKV：一份给上下文预计算

`_duplicate_context_wkv_weights`（[deepseek_v4/dspark.py:83-105](../../deepseek_v4/dspark.py#L83)，V4.1 直接复用，调用在 `dspark.py:539`）把每条 `mtp.{i}.attn.wkv.*` **额外复制一份**并打 `shard_id=i`，产出 `context_wkv_proj.{...}`；原权重保留给草稿 token 的单层前向。加载走 `param.weight_loader(param, loaded_weight, loaded_weight.shard_id)`（`dspark.py:562-566`）。用显存换一次合并投影：`precompute_and_store_context_kv` 用**一个** `MergedColumnParallelLinear` 把整段上下文 KV 算完（`dspark.py:338-342`），再逐层 RoPE + 散射进各自 SWA cache（`dspark.py:346-354`）。细节见 [8_deepseek_v41_dspark.md](8_deepseek_v41_dspark.md) §4.4。

---

## 6. 一次完整投机步（端到端）

```
目标前向（输入 = 上一步的 1+K 个 token）
├─ 逐层 0..39：hc_pre(上层 ffn_pre) → attn → hc_post → hc_pre(本层 attn_pre) → MoE → hc_post
│   层 37/38/39 入口 ★ aux[0..2] = mean_C(h) → (T,5120)×3
├─ 末尾：_mtp_hidden_buffer[:T] ← h.flatten(1)  (T,20480)  ← 预 collapse
│        h = last_layer.hc_collapse(h, pre_mix) → norm → lm_head → logits → 采样
└─ 草稿（同一 sample_tokens RPC 内）：
   ① combine: (T,15360) --main_proj--> (T,5120) --main_norm--> main_x
   ② context_wkv_proj 一次算 3 层 KV → 3 个 SWA cache
   ③ 噪音块 [anchor, N,N,N,N]（N=128799，5 列）→ embed (B,5,5120) → (B,5,4,5120)，pre_mix one-hot
   ④ 层 40（复用 main_x）→ 41 → 42，全程 SWA-only 注意力 + hc 接力
   ⑤ hc_collapse → norm → lm_head → logits (B,5,V)
   ⑥ 从左到右：bias = markov_w2 @ markov_w1[prev]；logits[:,i] += bias；argmax
        draft[:,0]=anchor，draft[:,1..5]=逐位置采样
   ⑦ confidence = sigmoid(proj([head_hidden ‖ markov_emb])) → (B,5) → 动态验证长度
验证：主模型因果并行验证，接受最长前缀，拒绝处按目标分布重采样 + bonus token
```

| 张量 | 形状 | 产生位置 |
|------|------|---------|
| 层间 4 流 `hidden_states` / `pre_mix` | `(T,4,5120)` / `(T,4)` fp32 | `model.py:1563` / `1564-1565` |
| `_mtp_hidden_buffer` | `(max_num_batched_tokens, 20480)` | `model.py:988-997` |
| aux（每层）/ 拼接后 | `(T,5120)` / `(T,15360)` | `model.py:1580` / `model_runner.py:2187` |
| `main_x` / 噪音块 | `(T,5120)` / `(B,5,4,5120)` | `dspark.py:313` / `dspark.py:280` |
| draft `head_hidden` / `logits` | `(B*5,5120)` / `(B,5,V)` | `dspark.py:304` / `llm_base_proposer.py:1575` |
| `markov_embed` / `markov_bias` | `(B,256)` / `(B,129280)` | `dspark.py:365-369` |
| `draft_token_ids` / `confidence` | `(B,6)` / `(B,5)` | `dspark_proposer.py:45-47` / `:108-125` |

> **块宽与 `K`**：`block_size=5` 决定一次前向出 5 列 logits；`sample_from_anchor=True` 时每请求只发 `num_query_per_req = num_speculative_tokens` 个 query，否则是 `1+K`（`dspark_proposer.py:39-43`）。若 `--num-speculative-tokens` 与 `dspark_block_size` 不一致，这里会出现"少一列/多一列"的隐性错位。

---

## 7. 常见坑 / FAQ

### 7.1 PP 下的 hc 缓冲

`_mtp_hidden_buffer` **只在最后一个 PP rank 分配**（`model.py:995`），中间 rank 是 `None`；而 37/38/39 可能落在前段。跨段接力靠 `IntermediateTensors` 的 aux 载荷（v1 补丁 [patch_eagle3_pp_aux.py:116-125](../../../patch/worker/patch_v1/patch_eagle3_pp_aux.py#L116) 把 `f"{_AUX_KEY_PREFIX}{i}"` 塞进 `IntermediateTensors`；v2 用 `add_pp_transport_tensors(..., PPTransportDataType.AUX_HIDDEN_STATES, ...)`）。

**现有约束**：v2 的 `SpecPPSupport` 白名单里 `dspark` 只登记了 `DeepseekV4ForCausalLM` / `GlmMoeDsaForCausalLM` / `KimiLinearForCausalLM` / `KimiK3*`（[pp_transport.py:53-64](../../../worker/v2/pp_transport.py#L53)），`mtp` 的 `needs_aux_hidden_states` 为 `False`（`L43`）。**V4.1 架构不在白名单内**，故 PP + dspark 必须确认 aux 接力走 v1 补丁那条路；否则表现是"最后一段一直用上一段的旧隐态"，接受率掉但无报错。

### 7.2 SP 下 MTP 缓冲的 all_gather

| 坑 | 后果 |
|----|------|
| 只 gather `hidden_states`、漏 `pre_mix`（`model.py:1615-1617`） | `pre_mix` 是本地分片、`x` 是全长 → 广播不匹配（报错）或越界静默 |
| gather 后不 `[:full_num_tokens]` | 结果按 padding 后长度返回，缓冲混进 padding 行 |
| 缓冲分支后又走 `L1622-1623` 的 gather | 多一次集合通信；判断写错则形状翻倍 |
| 写缓冲时 `num_tokens ≠ full_num_tokens` | 缓冲前 N 行与真实 token 错位 |

> SP 分片在**层内**（`sp_reduce_scatter`/`sp_all_gather` 成对出现在 `model.py:897-901`），4 流残差在层间是**分片态**——这正是 gather 必须放在写缓冲**之前**的原因。

### 7.3 为什么 MTP 要 **pre-hc_head**（pre-collapse）状态

**结构同构**：草稿层是 `DeepseekV41DecoderLayer` 的复用体，内部同样跑 `hc_pre → attn → hc_post → hc_pre → MoE → hc_post`，要 `(T,C,H)` 的 4 流输入（`dspark.py:280`、`291-302`）；喂单流就得在草稿侧再造一次"单流→4 流"展开，而这次展开没有任何学到的语义。

**collapse 是输入相关的动态加权**：系数来自 `hc_fn @ x_flatten` 的 sinkhorn 输出（参考 [reference/model.py:950-957](reference/model.py#L950)），每 token、每子层都不同。压成单流等于永久丢弃逐 token 门控，草稿再也分不清 4 条流各自的贡献；缓冲存 `(T,20480)` 而非 `(T,5120)` 正是为与训练期对齐。

> 例外：V4.1 实际走 §3.4 的 **A 路线**（aux 三层均值，已"压过一次 C"），所以"必须 4 流"只对 B 路线（eagle3/mtp 式 drafter）成立。

### 7.4 草稿 MoE 只有 128 专家、top-3

配置里只有 `dspark_n_routed_experts=128` / `dspark_num_experts_per_tok=3`，建模代码并不读它——改写在**草稿配置补丁**里：

```python
# patch_speculative_config.py:187-198 —— SpeculativeConfig 持私有副本，改它不影响目标模型
if is_deepseek_v41_model:
    draft_updates = {"n_routed_experts": draft_text_config.dspark_n_routed_experts,      # 128
                     "n_mtp_layers": getattr(draft_text_config, "num_nextn_predict_layers", 3)}
    if draft_experts_per_token is not None:
        draft_updates["num_experts_per_tok"] = draft_experts_per_token                   # 3
    draft_text_config.update(draft_updates)
```

`DeepseekV41MoE` 只读 `config.n_routed_experts` / `config.num_experts_per_tok`（`model.py:174-181`），所以必须在建草稿层**之前**改写草稿私有 config。推论：改的是**私有副本**，目标模型仍 384/6；草稿 `gate` 输出维因此是 128（`model.py:179-181`），加载 384 行权重会形状报错（反而是好事）；`set_moe_parameters` 用**最后一个** MoE 当样例（`dspark.py:443-449`），EP/EPLB 元数据按 128 登记。

### 7.5 fp8 `main_proj` 的 scale 名

两条规则叠加：

1. **必须用 fp8 量化配置构造** `main_proj`，否则不会有 `weight_scale` 参数：`dspark.py:210-231` 只在 `DeviceOperator.get_dsv41_packed_cache_ops() is not None` 且 checkpoint 的 `quantization_config.quant_method == "fp8"` 时才把 `vllm_config.quant_config` 传给 `ColumnParallelLinear`；注释直说后果："constructing a BF16 linear drops the scale parameter and makes the draft loader fail on `main_proj.scale`."（`dspark.py:210-213`）
2. **后缀统一改写**（`dspark.py:558-560`）：`.scale → .weight_scale`。于是 `mtp.0.main_proj.scale` → `model.layers.40.main_proj.weight_scale`。这条改写发生在 `_remap_dspark_name` **之后**，顺序不能换。

同族坑：`mtp.{i}.attn.attn_sink` 要按 TP 头段切片（`dspark.py:609-614`）；`e_score_correction_bias_vl` 被**故意忽略**而非 alias 到文本 correction bias（`dspark.py:569-575`），否则一开投机文本路由就变。

### 7.6 buffer 填了但草稿不读

```python
# model_runner.py:2174-2178
mtp_hidden_states = getattr(self.get_model(), "get_mtp_target_hidden_states", lambda: None)()
if self.speculative_config.method == "mtp" and mtp_hidden_states is not None:
    hidden_states = mtp_hidden_states
```

`method == "dspark"` 时这条不生效，草稿改吃 `aux_hidden_states`（A 路线），4 流缓冲只为 v2 的 `get_drafter_hidden_states` 服务。**不要为了"让它被用上"去改 rebind 条件**——A 路线要的是 3 层入口均值，与缓冲的"最后一层出口 4 流"语义不兼容。

### 7.7 其它零碎

| 坑 | 说明 |
|----|------|
| aux 层号差一 | `[37,38,39]`（零基）→ `(38,39,40)`（一层基，`model_runner.py:833`）；模型侧比 `layer_idx+1`（`model.py:1579`）。写 `(37,38,39)` 会命中 36/37/38，质量降但不报错 |
| aux 采集位置 | V4.1 在**层入口**（`model.py:1579`；V4 是出口 `deepseek_v4/model.py:1013`），挪位置等于换语义 |
| Markov rank | V4.1 是 **256**（V4 是 512），`confidence_head` 输入 `5120+256=5376`（`dspark.py:244-249`）；沿用 512 形状不匹配 |
| noise token | `128799` 经 `ptd_token_id` 传给 proposer（`patch_speculative_config.py:287`）；块内**第 0 列必须是 anchor** |
| `pre_mix` dtype / 图捕获 | 必须 fp32（`torch_binding.cpp:1618-1619`）；缓冲必须**编译前**创建、用 `copy_` 写（`model.py:1619`），改成每步 `torch.empty` 会让重放写到已释放地址 |

---

## 8. 小结

V4.1 的投机解码不是新模块，而是**四个接口约定**：

1. **延迟 mHC 系数交接**（`model.py:879-920`）：`hc_pre` 多一个可选 `pre_mix` 入参、多一个 `pre` 出参（`npu_hc_pre_v3` 四输出，`torch_binding.cpp:1656-1663`），层间返回值从 `(hidden, residual)` 变成 `(hidden, pre_mix)`；末尾用最后一层 `ffn_pre` 做一次纯加权 `hc_collapse`（`model.py:1621`），关键路径上不再有"算完立刻用"的 Cube 依赖，目标模型也不再需要 `hc_head_*` 权重。
2. **预 collapse 的 4 流缓冲**（`model.py:983-997`、`1613-1619`、`1833-1837`）：只在最后一个 PP rank、只在需要隐态的投机方法下分配，在 `hc_collapse` **之前** `copy_`。
3. **两层编号体系**（`model.py:334-337`、`dspark.py:160-183`、`model_runner.py:823-838`）：checkpoint 的 `mtp.{0,1,2}` 与零基 `[37,38,39]`，对上运行时 `model.layers.{40,41,42}` 与一层基 `(38,39,40)`。
4. **两个 loader 各管一段**（`model.py:1871-1873` 跳过 `mtp.*`；`dspark.py:509-621` + `623-662` 重映射）：加上 WKV 双份、`.scale → .weight_scale`、`markov_head.{embed,head} → {markov_w1,markov_w2}`、attn_sink 的 TP 切片。

四条对齐后，一次投机步就是：**目标前向在 37/38/39 入口留 3 份 aux 均值、末尾留 1 份 4 流残差 → 草稿融合 aux 成 `main_x`、预写 3 层 SWA KV、吃一个 5 列噪音块 → 3 个 draft 层沿用同一套延迟 mix 流水线 → 单流化后接 Markov 偏置逐位置采样 → confidence 估接受率决定验证长度 → 主模型因果验证**。这条链上任何一处 shape、编号或命名错位都不会崩，只会静默地让接受率归零。
