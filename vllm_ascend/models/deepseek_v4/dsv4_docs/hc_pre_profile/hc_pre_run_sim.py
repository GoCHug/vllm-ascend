#!/usr/bin/env python3
"""仿真器（camodel）版本：直接构造 (T, hc_mult, H) 输入，不经 aclnnRepeat。

为什么单独写一份：
    原脚本用 `hidden.unsqueeze(1).repeat(1, HC_MULT, 1)` 构造 x，这会触发
    aclnnRepeat（kernel 名 Tile_xxx）。而在 `msopprof simulator` 下，Tile 算子
    需要 `aclnnRepeat_1_TileAiCore.json` 动态 shape 配置，当前 OPP 包里没有，
    直接报 errno 561000 把整个仿真跑挂。本脚本直接分配 (T, hc_mult, H)，
    避开 Tile，让仿真只跑真正的 HcPre kernel。

运行：
    source /usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
    python3 hc_pre_run_sim.py
"""

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()
torch_npu.npu.config.allow_internal_format = True

NUM_TOKENS = 2048
HIDDEN_SIZE = 7168
HC_MULT = 4
HC_SINKHORN_ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6
MIX_HC = (2 + HC_MULT) * HC_MULT  # 24
HC_DIM = HC_MULT * HIDDEN_SIZE  # 28672

torch.npu.set_device(0)
torch.manual_seed(1024)

# 直接分配 (T, hc_mult, H)，等价于“残差流沿 hc 维广播”后的结果，但不调 aclnnRepeat
x = (torch.rand(NUM_TOKENS, HC_MULT, HIDDEN_SIZE, dtype=torch.float32) * 2).to(torch.bfloat16).npu()
hc_fn = (torch.rand(MIX_HC, HC_DIM, dtype=torch.float32) / HC_DIM).npu()
hc_scale = (torch.rand(3, dtype=torch.float32) * 2).npu()
hc_base = (torch.rand(MIX_HC, dtype=torch.float32) * 2).npu()

print("[inputs]")
print(f"  x        {tuple(x.shape)} {x.dtype}")
print(f"  hc_fn    {tuple(hc_fn.shape)} {hc_fn.dtype}")
print(f"  hc_scale {tuple(hc_scale.shape)} {hc_scale.dtype}")
print(f"  hc_base  {tuple(hc_base.shape)} {hc_base.dtype}")

y, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
    x, hc_fn, hc_scale, hc_base, HC_MULT, HC_SINKHORN_ITERS, NORM_EPS, HC_EPS
)
torch.npu.synchronize()

print("\n[outputs]")
print(f"  y     {tuple(y.shape)} {y.dtype}")
print(f"  post  {tuple(post.shape)} {post.dtype}  row_sum={post.sum(-1).mean():.6f}")
print(f"  comb  {tuple(comb.shape)} {comb.dtype}  row_sum={comb.sum(-1).mean():.6f} col_sum={comb.sum(-2).mean():.6f}")