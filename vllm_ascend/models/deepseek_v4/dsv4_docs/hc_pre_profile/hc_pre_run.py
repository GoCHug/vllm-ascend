#!/usr/bin/env python3
"""单次运行 torch.ops._C_ascend.npu_hc_pre_v2，输入模拟 DeepSeek-V4-Pro（num_tokens=2048）。

输入构造对齐真实 forward：
    model.py:1004        hidden_states (T, H) -> unsqueeze(1).repeat(1, hc_mult, 1) -> x (T, hc_mult, H)
    model.py:763-770     mix_hc=(2+hc_mult)*hc_mult=24, hc_dim=hc_mult*H, hc_fn (24, hc_dim) fp32,
                         hc_base (24,) fp32, hc_scale (3,) fp32
    model.py:784-786     算子调用

运行前先 source 自定义算子环境：
    source /usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
    python hc_pre_run.py
"""

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()
torch_npu.npu.config.allow_internal_format = True

# ---- DeepSeek-V4-Pro 配置（0_DeepSeek-V4-Pro-0813-w4a8_config.json）----
NUM_TOKENS = 2048
HIDDEN_SIZE = 7168
HC_MULT = 4
HC_SINKHORN_ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6
MIX_HC = (2 + HC_MULT) * HC_MULT  # 24
HC_DIM = HC_MULT * HIDDEN_SIZE  # 28672

DEVICE = "npu:0"
torch.npu.set_device(0)
torch.manual_seed(1024)

# ---- 输入构造 ----
x = (torch.rand(NUM_TOKENS, HC_MULT, HIDDEN_SIZE, dtype=torch.float32) * 2).to(torch.bfloat16).npu()
hc_fn = (torch.rand(MIX_HC, HC_DIM, dtype=torch.float32) / HC_DIM).npu()
hc_scale = (torch.rand(3, dtype=torch.float32) * 2).npu()
hc_base = (torch.rand(MIX_HC, dtype=torch.float32) * 2).npu()

print("[inputs]")
print(f"  x        {tuple(x.shape)} {x.dtype}")
print(f"  hc_fn    {tuple(hc_fn.shape)} {hc_fn.dtype}")
print(f"  hc_scale {tuple(hc_scale.shape)} {hc_scale.dtype}")
print(f"  hc_base  {tuple(hc_base.shape)} {hc_base.dtype}")

# ---- 调用算子 ----
y, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
    x, hc_fn, hc_scale, hc_base, HC_MULT, HC_SINKHORN_ITERS, NORM_EPS, HC_EPS
)
torch.npu.synchronize()

# ---- 结果打印 ----
print("\n[outputs]")
print(f"  y     {tuple(y.shape)} {y.dtype}  abs_mean={y.float().abs().mean():.4f} abs_max={y.float().abs().max():.4f}")
print(f"  post  {tuple(post.shape)} {post.dtype}  min={post.min():.6f} max={post.max():.6f}")
print(f"  comb  {tuple(comb.shape)} {comb.dtype}  row_sum={comb.sum(-1).mean():.6f} col_sum={comb.sum(-2).mean():.6f}")
print(f"\n  y[0, :8]      = {y[0, :8].float().tolist()}")
print(f"  post[0]       = {post[0].tolist()}")
print(f"  comb[0]       = {comb[0].tolist()}")