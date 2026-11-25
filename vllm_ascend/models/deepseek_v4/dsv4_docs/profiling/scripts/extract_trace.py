#!/usr/bin/env python3
"""从 msprof op simulator 的 trace.json 里抽一个核的指令时间线。

    python extract_trace.py <trace.json> <out.csv> [--core core0] [--limit 200000]

trace.json 是单行的大 JSON（几个 GB），这里用 ijson 流式解析，
只保留指定核的事件，写出 name/ts/dur/tid 四列。
"""

import argparse
import csv
import sys

import ijson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("--core", default="core0", help="只看这个核，如 core0 / core3")
    ap.add_argument("--limit", type=int, default=0, help="最多保留多少条（0=不限）")
    args = ap.parse_args()

    kept = 0
    with open(args.trace, "rb") as f, open(args.out, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["pid", "tid", "name", "ts", "dur"])
        for ev in ijson.items(f, "traceEvents.item"):
            pid = str(ev.get("pid") or "")
            if not pid.startswith(args.core):
                continue
            ph = ev.get("ph")
            if ph not in ("X", "B", "E"):
                continue
            w.writerow([pid, ev.get("tid"), ev.get("name"), ev.get("ts"), ev.get("dur")])
            kept += 1
            if args.limit and kept >= args.limit:
                break
    print(f"kept {kept} events -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
