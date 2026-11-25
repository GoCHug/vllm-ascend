#!/usr/bin/env python3
"""hc_pre 单算子压测 / profiling 入口。

按 DSv4 的真实形状造输入，只调用一次 `torch.ops._C_ascend.npu_hc_pre_v2`，
用来喂给 `msprof op`（设备侧）或 `msprof op simulator`（指令级）。

    # 只看耗时
    python prof_hc_pre.py --t 2048

    # 被 msprof 包起来跑（由 run_prof.sh 调）
    python prof_hc_pre.py --t 2048 --iters 20

DSv4 flash 的参数：
    T = 2048          -> M = 2048，24 AIC 按 mDimNum=8 / kDimNum=3 切，singleCoreM = 256
    hidden = 4096     -> K = hc_mult * hidden = 16384
    hc_mult = 4, mix_hc = 24, sinkhorn = 20
"""

import argparse
import os
import sys
import time

import torch
import torch_npu

# 允许在容器里直接跑（vllm_ascend 已安装时这两行是必须的）
from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

HC_MULT = 4
MIX_HC = 24
NORM_EPS = 1e-6
HC_EPS = 1e-6
SINKHORN_ITERS = 20


def build_inputs(t: int, hidden: int, hc_mult: int, mix_hc: int, seed: int = 1024):
    """和 tests/e2e/.../test_hc_pre.py 保持一致的造数方式。"""
    torch.manual_seed(seed)
    fan_in = hc_mult * hidden
    x = (torch.rand((t, hc_mult, hidden), dtype=torch.float32) * 2).to(torch.bfloat16)
    hc_fn = torch.rand((mix_hc, fan_in), dtype=torch.float32) / fan_in
    hc_scale = torch.rand(3, dtype=torch.float32) * 2
    hc_base = torch.rand(mix_hc, dtype=torch.float32) * 2
    return (
        x.npu(),
        hc_fn.npu(),
        hc_scale.npu(),
        hc_base.npu(),
    )


def run_once(x, hc_fn, hc_scale, hc_base, hc_mult, iters, eps_norm, eps_hc):
    return torch.ops._C_ascend.npu_hc_pre_v2(
        x, hc_fn, hc_scale, hc_base, hc_mult, iters, eps_norm, eps_hc
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=2048, help="token 数，即算子的 M")
    ap.add_argument("--hidden", type=int, default=4096, help="DSv4 flash 的 hidden")
    ap.add_argument("--hc-mult", type=int, default=HC_MULT)
    ap.add_argument("--mix-hc", type=int, default=MIX_HC)
    ap.add_argument("--sinkhorn-iters", type=int, default=SINKHORN_ITERS)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    torch_npu.npu.set_device(args.device)

    x, hc_fn, hc_scale, hc_base = build_inputs(
        args.t, args.hidden, args.hc_mult, args.mix_hc
    )

    # M = T，K = hc_mult * hidden，N = mix_hc
    print(
        f"[hc_pre] T(M)={args.t}  K={args.hc_mult * args.hidden}  N={args.mix_hc}  "
        f"x={tuple(x.shape)} fn={tuple(hc_fn.shape)}",
        flush=True,
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            run_once(x, hc_fn, hc_scale, hc_base, args.hc_mult,
                     args.sinkhorn_iters, NORM_EPS, HC_EPS)
        torch.npu.synchronize()

        # ---- 计时（不依赖 msprof 的粗略值，只做自检）----
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            run_once(x, hc_fn, hc_scale, hc_base, args.hc_mult,
                     args.sinkhorn_iters, NORM_EPS, HC_EPS)
        end.record()
        torch.npu.synchronize()
        per_iter_ms = start.elapsed_time(end) / args.iters

        y, post, comb = run_once(x, hc_fn, hc_scale, hc_base, args.hc_mult,
                                 args.sinkhorn_iters, NORM_EPS, HC_EPS)
        torch.npu.synchronize()

    print(f"[hc_pre] avg {per_iter_ms * 1000:.1f} us  over {args.iters} iters", flush=True)
    print(f"[hc_pre] out y={tuple(y.shape)} {y.dtype}  post={tuple(post.shape)}  "
          f"comb={tuple(comb.shape)}", flush=True)
    print("[hc_pre] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
