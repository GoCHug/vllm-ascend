#!/usr/bin/env python3
"""把切片后的 trace JSON 画成流水甘特图（SVG）。

    python plot_trace_gantt.py report/trace_core0_steady_5us.json report/trace_gantt.svg

一个核一条时间轴，每行一条流水线（MTE2 / MTE1 / CUBE / FIXP），
色块就是一条指令，宽度是它的耗时。
"""

import json
import sys

FONT = "Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif"

PIPE_COLOR = {
    "MTE2": ("#e2e8f0", "#64748b"),
    "MTE1": ("#bbf7d0", "#15803d"),
    "CUBE": ("#bfdbfe", "#2563eb"),
    "FIXP": ("#fed7aa", "#ea580c"),
    "SCALAR": ("#f1f5f9", "#cbd5e1"),
}

# 想高亮的指令
NICE = {
    "LOAD_L1_TO_DST_3DV2": "LoadAToL0A",
    "LOAD_2D": "Load*L0B",
    "MMAD": "Mmad",
    "MOV_OUT_TO_L1_MULTI_ND2NZ": "CopyIn(Nd2Nz)",
    "SET_CROSS_CORE": "CrossCoreSet",
}


def main(src, dst, core="core0.cubecore0"):
    data = json.load(open(src, encoding="utf-8"))
    events = [e for e in data["traceEvents"] if e["ph"] == "X"]

    pids = {e["pid"]: e["name"] for e in data["traceEvents"]
            if e["ph"] == "M" and e["name"] == "process_name"}
    tids = {e["tid"]: e["args"]["name"] for e in data["traceEvents"]
            if e["ph"] == "M" and e["name"] == "thread_name"}

    want_pid = [p for p, n in pids.items() if n == core]
    if want_pid:
        events = [e for e in events if e["pid"] == want_pid[0]]

    if not events:
        print("没有事件", file=sys.stderr)
        return 1

    for e in events:
        e["pipe"] = tids.get(e["tid"], "?")

    t0 = min(e["ts"] for e in events)
    t1 = max(e["ts"] + e["dur"] for e in events)
    span = max(t1 - t0, 1e-9)

    order = ["MTE2", "MTE1", "CUBE", "FIXP"]
    pipes = [p for p in order if any(e["pipe"] == p for e in events)]
    pipes += sorted({e["pipe"] for e in events} - set(pipes))

    W = 1100
    left, right = 90, 30
    top, rowh = 70, 46
    H = top + rowh * len(pipes) + 78
    plot_w = W - left - right

    def X(t):
        return left + (t - t0) / span * plot_w

    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W / 2}" y="28" text-anchor="middle" font-family="{FONT}" font-size="16" '
        f'font-weight="700" fill="#0f172a">{core} 的指令级流水（窗口 {t1 - t0:.2f} us）</text>',
        f'<text x="{W / 2}" y="48" text-anchor="middle" font-family="{FONT}" font-size="10" '
        f'fill="#64748b">每个色块是一条指令，横向是时间；'
        f'同一列上出现多个颜色就说明这几条流水在并行</text>',
    ]

    # 时间刻度
    for k in range(0, 6):
        frac = k / 5
        x = left + frac * plot_w
        o.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{top + rowh * len(pipes) + 4}" '
                 f'stroke="#e2e8f0" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{top + rowh * len(pipes) + 20}" text-anchor="middle" '
                 f'font-family="{FONT}" font-size="9" fill="#94a3b8">{t0 + frac * span:.2f}</text>')

    for i, pipe in enumerate(pipes):
        y = top + i * rowh
        fill, stroke = PIPE_COLOR.get(pipe, ("#f1f5f9", "#94a3b8"))
        o.append(f'<rect x="{left}" y="{y}" width="{plot_w}" height="{rowh - 8}" fill="#fbfdff" rx="4"/>')
        o.append(f'<text x="{left - 10}" y="{y + rowh / 2 - 2}" text-anchor="end" font-family="{FONT}" '
                 f'font-size="11" font-weight="700" fill="#334155">{pipe}</text>')
        labels = []
        for e in sorted([x for x in events if x["pipe"] == pipe], key=lambda x: x["ts"]):
            x0 = X(e["ts"])
            w = max(X(e["ts"] + e["dur"]) - x0, 1.2)
            o.append(f'<rect x="{x0:.2f}" y="{y + 4}" width="{w:.2f}" height="{rowh - 16}" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="0.8" rx="1.5"/>')
            labels.append((x0, w, e["name"]))
        # 只给够宽的块标名字，放不下就留白
        for x0, w, name in labels:
            short = NICE.get(name, name)
            need = len(short) * 4.6
            if w >= need + 6:
                o.append(f'<text x="{x0 + w / 2:.1f}" y="{y + 18}" text-anchor="middle" '
                         f'font-family="{FONT}" font-size="7.2" fill="#0f172a">{short}</text>')

    # 图例：一行，按流水线着色
    ly = H - 34
    lx = left
    legend = [
        ("#bbf7d0", "#15803d", "MTE1 · LOAD_2D = Load*ToL0B 片"),
        ("#bbf7d0", "#15803d", "MTE1 · LOAD_L1_TO_DST_3DV2 = LoadAToL0A"),
        ("#bfdbfe", "#2563eb", "CUBE · MMAD"),
        ("#e2e8f0", "#64748b", "MTE2 · MOV_OUT_TO_L1_MULTI_ND2NZ = CopyIn"),
    ]
    for fill, stroke, text in legend:
        o.append(f'<rect x="{lx}" y="{ly - 9}" width="11" height="10" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="0.8" rx="2"/>')
        o.append(f'<text x="{lx + 15}" y="{ly}" font-family="{FONT}" font-size="8.6" '
                 f'fill="#475569">{text}</text>')
        lx += 24 + len(text) * 8.2

    o.append("</svg>")
    open(dst, "w", encoding="utf-8").write("\n".join(o))
    print(f"written {dst}  ({len(events)} 条指令，{span:.2f} us)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
