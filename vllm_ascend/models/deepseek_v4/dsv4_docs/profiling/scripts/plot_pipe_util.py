#!/usr/bin/env python3
"""把 PipeUtilization.csv 画成两条柱状图（AIC / AIV），输出 SVG。

    python plot_pipe_util.py raw/op_t2048/PipeUtilization.csv report/pipe_util.svg
"""

import csv
import sys

FONT = "Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif"

AIC_KEYS = [
    ("aic_cube_ratio", "Cube"),
    ("aic_mte1_ratio", "MTE1  L1→L0"),
    ("aic_mte2_ratio", "MTE2  GM→L1"),
    ("aic_mte3_ratio", "MTE3"),
    ("aic_fixpipe_ratio", "Fixp"),
    ("aic_scalar_ratio", "Scalar"),
]
AIV_KEYS = [
    ("aiv_vec_ratio", "Vector"),
    ("aiv_mte2_ratio", "MTE2"),
    ("aiv_mte3_ratio", "MTE3"),
    ("aiv_scalar_ratio", "Scalar"),
]


def avg(rows, key):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key]))
        except (KeyError, TypeError, ValueError):
            pass
    return sum(vals) / len(vals) if vals else 0.0


def bars(x, y, w, h, items, color):
    out = []
    n = len(items)
    gap = 14
    bw = (w - gap * (n - 1)) / n
    for i, (label, val) in enumerate(items):
        bx = x + i * (bw + gap)
        bh = max(2.0, h * val)
        out.append(
            f'<rect x="{bx:.1f}" y="{y + h - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
            f'fill="{color}" rx="4"/>'
        )
        cx = bx + bw / 2
        out.append(
            f'<text x="{cx:.1f}" y="{y + h - bh - 6:.1f}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="11" font-weight="700" fill="#0f172a">{val * 100:.1f}%</text>'
        )
        out.append(
            f'<text x="{cx:.1f}" y="{y + h + 16:.1f}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="10" fill="#475569">{label}</text>'
        )
    return out


def main(src, dst):
    rows = list(csv.DictReader(open(src, encoding="utf-8", errors="replace")))
    aic = [r for r in rows if (r.get("sub_block_id") or "").startswith("cube")]
    aiv = [r for r in rows if (r.get("sub_block_id") or "").startswith("vector")]

    aic_items = [(label, avg(aic, key)) for key, label in AIC_KEYS]
    aiv_items = [(label, avg(aiv, key)) for key, label in AIV_KEYS]

    W, H = 900, 520
    aic_time = avg(aic, "aic_time(us)")
    aiv_time = avg(aiv, "aiv_time(us)")

    title = "hc_pre 各流水线利用率（T=2048，24 AIC / 48 AIV）"
    sub = f"占比按各流水线忙碌时间 / 核的总时长算；AIC {aic_time:.0f} us，AIV {aiv_time:.0f} us"
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W / 2}" y="30" text-anchor="middle" font-family="{FONT}" font-size="17" '
        f'font-weight="700" fill="#0f172a">{title}</text>',
        f'<text x="{W / 2}" y="52" text-anchor="middle" font-family="{FONT}" font-size="11" '
        f'fill="#64748b">{sub}</text>',
    ]

    o.append(f'<text x="40" y="88" font-family="{FONT}" font-size="12" font-weight="700" '
             f'fill="#1d4ed8">AIC（Cube 核，24 个）</text>')
    o += bars(40, 104, W - 80, 140, aic_items, "#60a5fa")
    o.append(f'<line x1="40" y1="252" x2="{W - 40}" y2="252" stroke="#e2e8f0" stroke-width="1"/>')

    o.append(f'<text x="40" y="296" font-family="{FONT}" font-size="12" font-weight="700" '
             f'fill="#7c3aed">AIV（Vector 核，48 个）</text>')
    o += bars(40, 312, W - 80, 140, aiv_items, "#a78bfa")
    o.append(f'<line x1="40" y1="460" x2="{W - 40}" y2="460" stroke="#e2e8f0" stroke-width="1"/>')
    note = ("AIC 三条搬运/计算流水都没打满：MTE2 最忙 33.8%，Cube 只有 22.4%。"
            "AIV 的 scalar 最忙（38.4%），其余三条各 30% 上下。")
    o.append(f'<text x="40" y="492" font-family="{FONT}" font-size="10.5" fill="#475569">{note}</text>')

    o.append("</svg>")
    open(dst, "w", encoding="utf-8").write("\n".join(o))
    print(f"written {dst}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raw/op_t2048/PipeUtilization.csv",
         sys.argv[2] if len(sys.argv) > 2 else "report/pipe_util.svg")
