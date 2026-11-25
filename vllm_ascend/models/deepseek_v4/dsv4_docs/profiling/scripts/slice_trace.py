#!/usr/bin/env python3
"""把 msprof 的指令级时间线（core*.csv）切成一小段 Chrome Trace JSON。

产出的 JSON 可以直接拖进 `chrome://tracing`、Perfetto（ui.perfetto.dev）或
MindStudio Insight 的 Timeline 视图。

    # 从 core0 的 24 万条指令里取前 5 us
    python slice_trace.py raw/trace_core0.csv.gz report/trace_core0_5us.json --window 5

    # 指定起点
    python slice_trace.py raw/trace_core0.csv.gz out.json --start 20 --window 5
"""

import argparse
import csv
import gzip
import json
import sys


def open_maybe_gz(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=None, help="窗口起点(us)，默认取第一条事件")
    ap.add_argument("--window", type=float, default=5.0, help="窗口长度(us)")
    ap.add_argument("--pipes", default="", help="只保留这些流水线，逗号分隔，如 MTE1,MTE2,CUBE,FIXP")
    args = ap.parse_args()

    events = []
    with open_maybe_gz(args.csv) as f:
        for r in csv.DictReader(f):
            try:
                events.append(
                    (r["pid"], r["tid"], r["name"], float(r["ts"]), float(r["dur"]))
                )
            except (KeyError, TypeError, ValueError):
                continue

    if not events:
        print("没有事件", file=sys.stderr)
        return 1

    start = args.start if args.start is not None else min(e[3] for e in events)
    end = start + args.window
    keep = {x.strip().upper() for x in args.pipes.split(",") if x.strip()}
    picked = []
    for e in events:
        pid, tid, name, ts, dur = e
        if not (start <= ts < end) or (keep and str(tid).upper() not in keep):
            continue
        # 跨出窗口的长指令截到窗口末尾，避免把时间轴拉长
        picked.append((pid, tid, name, ts, min(dur, end - ts)))
    if not picked:
        print(f"窗口 [{start}, {end}) 里没有事件", file=sys.stderr)
        return 1

    # pid -> 进程号，tid -> 轨道号（trace 里两者都要是数字）
    trace = []
    pids, tids = {}, {}
    for pid, tid, name, ts, dur in picked:
        if pid not in pids:
            pids[pid] = len(pids) + 1
            trace.append({"name": "process_name", "ph": "M", "pid": pids[pid],
                          "tid": 0, "args": {"name": pid}})
        key = (pid, tid)
        if key not in tids:
            tids[key] = len(tids) + 1
            trace.append({"name": "thread_name", "ph": "M", "pid": pids[pid],
                          "tid": tids[key], "args": {"name": str(tid)}})
        trace.append({
            "name": name,
            "cat": tid,
            "ph": "X",
            "ts": round(ts - start, 6),   # us（Chrome Trace 的时间单位）
            "dur": round(dur, 6),
            "pid": pids[pid],
            "tid": tids[key],
        })

    out = {"displayTimeUnit": "ns", "traceEvents": trace}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"{len(picked)} 个事件，窗口 [{start:.3f}, {end:.3f}) us -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
