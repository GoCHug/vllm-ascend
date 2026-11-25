#!/usr/bin/env python3
"""把 msprof op 的 op_summary.csv 整理成人能看的几行。

    python analyze_op_summary.py raw/*/OPPROF_*/op_summary*.csv

不同 CANN 版本列名不完全一样，这里按关键字匹配。
"""

import csv
import glob
import sys
from collections import defaultdict

# 关心的列：关键字 -> 显示名
COLUMN_HINTS = {
    "Task Duration": "op 耗时(us)",
    "task_duration": "op 耗时(us)",
    "aicore_time": "aicore_time(us)",
    "aic_total_time": "aic 总时间(us)",
    "aiv_total_time": "aiv 总时间(us)",
    "cube_ratio": "cube%",
    "mac_ratio": "mac%",
    "vec_ratio": "vector%",
    "vector_ratio": "vector%",
    "mte1_ratio": "mte1%",
    "mte2_ratio": "mte2%",
    "mte3_ratio": "mte3%",
    "fixpipe_ratio": "fixp%",
    "scalar_ratio": "scalar%",
}


def pick_columns(fieldnames):
    picked = {}
    for hint, label in COLUMN_HINTS.items():
        for name in fieldnames:
            if name == hint or hint in name:
                picked[name] = label
                break
    return picked


def main(paths):
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            cols = pick_columns(reader.fieldnames or [])
            if not cols:
                print(f"[skip] {path}: 没匹配到已知列，表头是 {reader.fieldnames}")
                continue
            for row in reader:
                rows.append((path, row, cols))

    if not rows:
        print("没有可分析的行")
        return 1

    print(f"共 {len(rows)} 行，来自 {len(set(p for p, _, _ in rows))} 个文件\n")

    # 1) 按 op name 聚合
    agg = defaultdict(lambda: defaultdict(list))
    for _, row, cols in rows:
        op = row.get("Op Name") or row.get("op_name") or row.get("OpName") or "op"
        for src, label in cols.items():
            try:
                agg[op][label].append(float(row[src]))
            except (TypeError, ValueError):
                pass

    for op, metrics in agg.items():
        print(f"== {op}")
        for label, vals in metrics.items():
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            if "%" in label:
                print(f"   {label:>16}: {avg:6.2f}  (n={len(vals)})")
            else:
                print(f"   {label:>16}: {avg:9.1f}  (n={len(vals)})")
        print()

    # 2) 把各流水线利用率并排打出来，方便看谁在等
    ratio_keys = [k for k in agg[next(iter(agg))] if "%" in k]
    if ratio_keys:
        print("== 流水线利用率一览（各核平均）")
        header = f"{'op':<24}" + "".join(f"{k:>10}" for k in ratio_keys)
        print(header)
        for op, metrics in agg.items():
            line = f"{op:<24}"
            for k in ratio_keys:
                vals = metrics.get(k, [])
                line += f"{(sum(vals) / len(vals) if vals else 0):>10.2f}"
            print(line)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = glob.glob("raw/**/op_summary*.csv", recursive=True)
    sys.exit(main(args))
