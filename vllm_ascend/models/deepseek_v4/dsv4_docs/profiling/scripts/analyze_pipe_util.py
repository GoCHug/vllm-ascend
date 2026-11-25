#!/usr/bin/env python3
"""解析 msprof op 的 PipeUtilization.csv（AIC/AIV 分块列）。

    python analyze_pipe_util.py raw/op_t2048/PipeUtilization.csv

输出：
  1) AIC（24 个 cube0 行）各流水线平均/最大利用率
  2) AIV（vector0/vector1）的 vec / mte2 / mte3 利用率
  3) 每个 block 的 aic_time，看负载均衡
  4) 由利用率反推瓶颈流水线
"""

import csv
import sys
from collections import defaultdict


def load(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def stats(rows, key):
    vals = [fnum(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {
        "n": len(vals),
        "avg": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
    }


def show(rows, keys, title):
    print(f"\n== {title}（n={len(rows)}）")
    print(f"{'指标':<34}{'avg':>10}{'min':>10}{'max':>10}")
    for key in keys:
        s = stats(rows, key)
        if not s:
            continue
        print(f"{key:<34}{s['avg']:>10.3f}{s['min']:>10.3f}{s['max']:>10.3f}")


def main(path):
    rows = load(path)
    aic = [r for r in rows if (r.get("sub_block_id") or "").startswith("cube")]
    aiv = [r for r in rows if (r.get("sub_block_id") or "").startswith("vector")]

    print(f"文件: {path}")
    print(f"总行数 {len(rows)}：AIC {len(aic)} 行，AIV {len(aiv)} 行")

    show(aic, [
        "aic_time(us)", "aic_total_cycles",
        "aic_cube_time(us)", "aic_cube_ratio",
        "aic_mte1_time(us)", "aic_mte1_ratio",
        "aic_mte2_time(us)", "aic_mte2_ratio",
        "aic_mte3_time(us)", "aic_mte3_ratio",
        "aic_fixpipe_time(us)", "aic_fixpipe_ratio",
        "aic_scalar_time(us)", "aic_scalar_ratio",
        "aic_scalar_wait_time(us)",
        "aic_scalar_mte1_stall_time(us)",
        "aic_scalar_mte2_stall_time(us)",
        "aic_scalar_cube_stall_time(us)",
    ], "AIC（Cube 核）")

    if aiv:
        show(aiv, [
            "aiv_time(us)", "aiv_total_cycles",
            "aiv_vec_ratio", "aiv_vec_time(us)",
            "aiv_mte2_ratio", "aiv_mte2_time(us)",
            "aiv_mte3_ratio", "aiv_mte3_time(us)",
            "aiv_scalar_ratio",
        ], "AIV（Vector 核）")

    # 负载均衡：每个 block 的 aic_time
    per_block = defaultdict(float)
    for r in aic:
        t = fnum(r.get("aic_time(us)"))
        if t is not None:
            per_block[r.get("block_id")] += t
    if per_block:
        ts = sorted(per_block.values())
        print("\n== 每个 AIC block 的 aic_time(us)")
        print("   ", " ".join(f"{t:.1f}" for t in ts))
        print(f"    min {ts[0]:.1f} / max {ts[-1]:.1f} / 离散度 {(ts[-1]-ts[0])/ts[0]*100:.1f}%")

    # 瓶颈判断
    def avg_ratio(key):
        s = stats(aic, key)
        return s["avg"] if s else 0.0

    ratios = {
        "cube": avg_ratio("aic_cube_ratio"),
        "mte1": avg_ratio("aic_mte1_ratio"),
        "mte2": avg_ratio("aic_mte2_ratio"),
        "mte3": avg_ratio("aic_mte3_ratio"),
        "fixpipe": avg_ratio("aic_fixpipe_ratio"),
    }
    worst = max(ratios, key=ratios.get)
    print("\n== 瓶颈判断")
    for k, v in sorted(ratios.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<8} {v*100:6.2f}%")
    print(f"   最忙的流水线: {worst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "raw/op_t2048/PipeUtilization.csv"))
