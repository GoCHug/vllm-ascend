#!/usr/bin/env python3
"""打印 HcPre 单算子的执行流程与全部 tiling 切块信息（核间 + 核内）。

数据来源
    OPPROF_*/dump/kernel_data/input_tiling.bin    host tiling 结果裸 buffer（唯一真实来源）
    OPPROF_*/dump/kernel_data/kernel_config.bin   block_dim / tiling_key
    OPPROF_*/OpBasicInfo.csv                      Block Dim / Mix Block Dim（交叉校验）

切块公式出处（镜像内与本地仓库一致）
    op_kernel/hc_pre.cpp:92-138                    tiling_key → Part1 + Part2 分支
    op_kernel/hc_pre_m_k_split_core.h:189-214      Part1 主循环（AIV cast → AIC mmad → SyncAll）
    op_kernel/hc_pre_m_k_split_core.h:75-96        stage1 网格 M×K 划分
    op_kernel/hc_pre_m_k_split_core.h:153-160      stage1 AIV 奇偶分半 M
    op_kernel/hc_pre_m_k_split_core.h:662-745      stage2/prePost 行处理（pre → y → post）
    op_kernel/hc_pre_m_k_split_core.h:747-830      prePost / sinkhorn 行分核
    op_kernel/hc_pre_m_k_split_core.h:998-1012     AIV 按 prePostCoreNum 分成两拨
    op_kernel/hc_pre_cube_compute.h:133/302        L0 成对 / mmad 基块
    csrc/torch_binding.cpp:1553-1562               输出张量形状

输出分节
    0 执行流程            1 输入与校验        2 总览
    3 Stage1 核间切块      4 Stage1 核内切层
    5 Stage2 核间切块      6 Stage2 核内切块
    7 Stage3 sinkhorn
    8 tiling 字段全量      9 校验与负载提示

用法
    python3 analysis_tiling.py                          # 自动找最新一次 OPPROF
    python3 analysis_tiling.py --opprof opprof_core0/OPPROF_xxx
    python3 analysis_tiling.py --only 0,2,3             # 只看指定分节
    python3 analysis_tiling.py --header /vllm-workspace/vllm-ascend/csrc/moe/hc_pre/op_host/hc_pre_tiling.h
    python3 analysis_tiling.py --json
"""

import argparse
import csv
import json
import re
import struct
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ───────────────────────────────────────────────────────────── 编译期常量
BLOCK_CUBE = 16            # AscendC::BLOCK_CUBE
K_L0_SIZE = 32             # hc_pre_cube_compute.h:133
L0_PAIR = K_L0_SIZE * 2    # ComputeL0Pairs 步长
MM_CACHE_ELEMS = 512 // 4  # MM_CACHE_LINE_BYTES / sizeof(float) = 128
DOUBLE_BUFFER = 2

# tiling_key → 内核分支（hc_pre.cpp 的 A3 段）
BRANCH = {
    0: "HcPreMembaseKSplitCorePart1 + Part2<useOriginal=false>",
    1: "HcPreMembaseKSplitCorePart1 + Part2<useOriginal=true>",
    2: "HcPreMembaseKSplitCorePart1 + HcPreMembaseKSplitCorePremixPart2",
}

# 兜底字段表；找到头文件时以头文件为准（换镜像请 --header 指向容器内那份）
BUILTIN_FIELDS = [
    ("bs", "i64"), ("hcMix", "i64"), ("hcMult", "i64"), ("d", "i64"), ("hcMultAlign", "i64"),
    ("rowOfFormerBlock", "i64"), ("rowOfTailBlock", "i64"),
    ("rowLoopOfFormerBlock", "i64"), ("rowLoopOfTailBlock", "i64"), ("rowFactor", "i64"),
    ("tailRowFactorOfFormerBlock", "i64"), ("tailRowFactorOfTailBlock", "i64"),
    ("dLoop", "i64"), ("dFactor", "i64"), ("tailDFactor", "i64"), ("iterTimes", "i64"),
    ("hcEps", "f32"), ("normEps", "f32"),
    ("kBlockFactor", "i64"), ("kFactor", "i64"), ("tailKFactor", "i64"), ("kLoop", "i64"),
    ("stage2RowFactor", "i64"), ("k", "i64"), ("kLoopOfFormerBlock", "i64"),
    ("kLoopOfTailBlock", "i64"), ("stage1KFactor", "i64"), ("kFactorOfFormerBlock", "i64"),
    ("kL1Size", "i64"), ("mL1Size", "i64"), ("cubeBlockDimM", "i64"), ("cubeBlockDimK", "i64"),
    ("kUbSize", "i64"), ("cvLoopKSize", "i64"), ("multCoreSplitKSize", "i64"),
    ("multCoreSplitMSize", "i64"), ("tailKSizeOfFormerBlock", "i64"), ("tailKSizeOfTailBlock", "i64"),
    ("mLoopOfFormerBlock", "i64"), ("mLoopOfTailBlock", "i64"), ("formerMSize", "i64"),
    ("tailMSizeOfFormerBlock", "i64"), ("tailMSizeOfTailBlock", "i64"),
    ("firstUsedCoreNum", "i64"), ("secondUsedCoreNum", "i64"), ("rowInnerFactor", "i64"),
    ("cubeCoreNum", "i64"), ("stage1MFactor", "i64"),
    ("bufferPool0Size", "i64"), ("bufferPool1Size", "i64"), ("mUbSize", "i64"),
    ("hasPreMix", "i64"), ("hasPreOut", "i64"),
    ("prePostCoreNum", "i64"), ("sinkhornCoreNum", "i64"),
    ("prePostRowOfFormerBlock", "i64"), ("prePostRowOfTailBlock", "i64"),
    ("prePostRowLoopOfFormerBlock", "i64"), ("prePostRowLoopOfTailBlock", "i64"),
    ("prePostRowFactor", "i64"), ("prePostTailRowFactorOfFormerBlock", "i64"),
    ("prePostTailRowFactorOfTailBlock", "i64"), ("sinkhornUsedCoreNum", "i64"),
    ("sinkhornRowOfFormerBlock", "i64"), ("sinkhornRowOfTailBlock", "i64"),
    ("sinkhornRowLoopOfFormerBlock", "i64"), ("sinkhornRowLoopOfTailBlock", "i64"),
    ("sinkhornRowFactor", "i64"), ("sinkhornTailRowFactorOfFormerBlock", "i64"),
    ("sinkhornTailRowFactorOfTailBlock", "i64"),
]

FIELD_NOTE = {
    "bs": "M 轴总行数（token 数）", "hcMix": "N 轴 = (2+hcMult)*hcMult，即 N_SIZE",
    "hcMult": "hyper-connection 分支数", "d": "hidden size", "hcMultAlign": "hcMult 在 UB 内的行对齐",
    "k": "GEMM 的 K 轴 = hcMult*d", "iterTimes": "sinkhorn 迭代次数",
    "hcEps": "hyper-connection eps", "normEps": "norm eps",
    "cubeBlockDimM": "stage1 网格 M 方向核数", "cubeBlockDimK": "stage1 网格 K 方向核数",
    "cubeCoreNum": "stage1 使用 AIC 总数 = M×K",
    "multCoreSplitMSize": "每 AIC 的 M 块", "multCoreSplitKSize": "每 AIC 的 K 块",
    "mL1Size": "L1 M 块（=单核 M 块）", "kL1Size": "L1 K 块",
    "cvLoopKSize": "AIC/AIV 握手 K 步长（UB 一次处理的 K）",
    "stage1MFactor": "AIV stage1 每次 UB 处理的行数",
    "stage2RowFactor": "stage2 每轮处理行数", "rowOfFormerBlock": "stage2 前 N-1 核每核行数",
    "rowOfTailBlock": "stage2 尾核行数", "rowLoopOfFormerBlock": "stage2 前 N-1 核循环轮数",
    "rowLoopOfTailBlock": "stage2 尾核循环轮数",
    "tailRowFactorOfFormerBlock": "stage2 前 N-1 核末轮行数",
    "tailRowFactorOfTailBlock": "stage2 尾核末轮行数",
    "dLoop": "d 轴循环数", "dFactor": "d 轴每轮长度", "tailDFactor": "d 轴末轮长度",
    "secondUsedCoreNum": "stage2 使用的 AIV 核数", "prePostCoreNum": "prePost 使用的 AIV 核数",
    "sinkhornUsedCoreNum": "sinkhorn 使用的 AIV 核数",
    "prePostRowFactor": "prePost 每轮行数", "sinkhornRowFactor": "sinkhorn 每轮行数",
    "hasPreMix": "是否走 premix 分支", "hasPreOut": "是否输出 pre",
}

FMT = {"i64": ("<q", 8), "i32": ("<i", 4), "f32": ("<f", 4), "f64": ("<d", 8)}
TYPE_MAP = {"int64_t": "i64", "int32_t": "i32", "float": "f32", "double": "f64"}
FIELD_RE = re.compile(r"TILING_DATA_FIELD_DEF\(\s*(\w+)\s*,\s*(\w+)\s*\)")


# ───────────────────────────────────────────────────────────── 基础工具
def dw(s):
    """显示宽度（中文占 2 列）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def pad(s, width, align="<"):
    gap = max(0, width - dw(s))
    if align == ">":
        return " " * gap + str(s)
    if align == "^":
        left = gap // 2
        return " " * left + str(s) + " " * (gap - left)
    return str(s) + " " * gap


def rng(a, b):
    return f"{a}..{b - 1}" if b > a else "-"


def shape(*dims):
    return "[" + ",".join(str(x) for x in dims) + "]"


def ceil_align(a, b):
    return (a + b - 1) // b * b


def human(nbytes):
    v = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024 or unit == "GB":
            return f"{v:.0f}{unit}" if unit == "B" else f"{v:.1f}{unit}"
        v /= 1024.0
    return f"{v:.1f}GB"


def table(headers, rows, aligns=None, indent=2):
    aligns = aligns or ["<"] * len(headers)
    widths = [max([dw(h)] + [dw(r[i]) for r in rows]) for i, h in enumerate(headers)]
    head = " " * indent + "  ".join(pad(h, widths[i], aligns[i]) for i, h in enumerate(headers))
    sep = " " * indent + "  ".join("-" * w for w in widths)
    body = [" " * indent + "  ".join(pad(r[i], widths[i], aligns[i]) for i in range(len(headers)))
            for r in rows]
    return "\n".join([head, sep] + body)


def title(no, text):
    return f"\n{'=' * 108}\n[{no}] {text}\n{'=' * 108}"


def kv(pairs, indent=2, key_w=18):
    return "\n".join(" " * indent + pad(k, key_w) + ": " + str(v) for k, v in pairs)
# ───────────────────────────────────────────────────────────── 解析
def load_fields(header):
    if header and Path(header).is_file():
        txt = Path(header).read_text(errors="ignore")
        found = [(n, TYPE_MAP[t]) for t, n in FIELD_RE.findall(txt) if t in TYPE_MAP]
        if found:
            return found, f"头文件 {Path(header).name}（{len(found)} 字段）"
    return BUILTIN_FIELDS, f"内置字段表（{len(BUILTIN_FIELDS)} 字段）"


def decode(buf, fields):
    vals, off = {}, 0
    for name, kind in fields:
        fmt, size = FMT[kind]
        off = ceil_align(off, size)
        if off + size > len(buf):
            raise ValueError(f"buffer {len(buf)}B 不足，解析 {name} 时越界（偏移 {off}）")
        vals[name] = struct.unpack_from(fmt, buf, off)[0]
        off += size
    return vals, off


def find_tiling(explicit_tiling, explicit_opprof):
    if explicit_tiling:
        p = Path(explicit_tiling)
        return p / "dump/kernel_data/input_tiling.bin" if p.is_dir() else p
    base = Path(explicit_opprof) if explicit_opprof else HERE
    cands = sorted(base.glob("**/dump/kernel_data/input_tiling.bin"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        sys.exit(f"[error] {base} 下没有 */dump/kernel_data/input_tiling.bin，用 --tiling 指定")
    return cands[0]


def read_meta_files(tiling_path):
    """读同目录的 kernel_config.bin 与 OPPROF 根目录的 OpBasicInfo.csv。"""
    out = {"opprof_dir": str(tiling_path.parent.parent.parent)}
    cfg = tiling_path.parent / "kernel_config.bin"
    if cfg.is_file():
        try:
            c = json.loads(cfg.read_text(errors="ignore"))
            out["block_dim"] = int(c.get("block_dim", 0)) or None
            out["tiling_key"] = c.get("tiling_key")
            out["kernel_name"] = c.get("kernel_name")
        except json.JSONDecodeError:
            pass
    best = tiling_path.parent.parent.parent / "OpBasicInfo.csv"
    if best.is_file():
        with best.open() as f:
            row = next(csv.DictReader(f))
        out.update({"op_name": row.get("Op Name"), "task_us": row.get("Task Duration(us)"),
                    "block_dim": int(row["Block Dim"]), "mix_dim": int(row["Mix Block Dim"])})
    return out


# ───────────────────────────────────────────────────────────── 派生模型
def split(loop_def, loop_tail, per_def, tail_former, tail_tail, is_tail):
    """复刻 kernel：轮数取 tail/former 版，每轮行数恒为 per_def，末轮取 tail 因子。"""
    loops = loop_tail if is_tail else loop_def
    tail = tail_tail if is_tail else tail_former
    return loops, per_def, tail, (loops - 1) * per_def + tail


def build_model(t, mix_dim):
    m = {
        "bs": t["bs"], "k": t["k"], "n": t["hcMix"], "hc_mult": t["hcMult"], "d": t["d"],
        "dim_m": t["cubeBlockDimM"], "dim_k": t["cubeBlockDimK"],
        "msz": t["multCoreSplitMSize"], "ksz": t["multCoreSplitKSize"],
        "ml1": t["mL1Size"], "kl1": t["kL1Size"], "cvk": t["cvLoopKSize"],
        "s1_m_factor": t["stage1MFactor"],
        "ws_last_axis": ceil_align(t["hcMix"], MM_CACHE_ELEMS),
        "compact": t["hcMult"] ** 2,       # COMPACT_HC_MULT = hcMult
        "n_stage2": t["secondUsedCoreNum"], "n_pre": t["prePostCoreNum"],
        "n_sink": t["sinkhornUsedCoreNum"], "iters": t["iterTimes"],
        "row_of_former": t["rowOfFormerBlock"], "row_of_tail": t["rowOfTailBlock"],
        "d_factor": t["dFactor"], "d_loop": t["dLoop"],
        "sk_row_of_former": t["sinkhornRowOfFormerBlock"],
        "sk_row_of_tail": t["sinkhornRowOfTailBlock"],
        "pp_per_rows": t["prePostRowFactor"], "pp_loops": t["prePostRowLoopOfFormerBlock"],
        "pp_tail_loops": t["prePostRowLoopOfTailBlock"],
    }
    m["aic"] = m["dim_m"] * m["dim_k"]
    m["aiv"] = mix_dim or m["aic"] * 2
    m["tail_k"] = m["k"] - m["ksz"] * (m["dim_k"] - 1)
    m["ws_slots"] = m["dim_k"]
    m["ws_bytes"] = m["ws_slots"] * m["bs"] * m["ws_last_axis"] * 4

    # Stage1 · AIC 逐单元
    aic = []
    for core in range(m["aic"]):
        mi, ki = divmod(core, m["dim_k"])
        m0, k0 = mi * m["msz"], ki * m["ksz"]
        m1, k1 = min(m0 + m["msz"], m["bs"]), min(k0 + m["ksz"], m["k"])
        aic.append({
            "core": core, "m_idx": mi, "k_idx": ki, "rows": rng(m0, m1), "cols": rng(k0, k1),
            "m": m1 - m0, "kk": k1 - k0, "x": [m1 - m0, k1 - k0], "b": [m["n"], k1 - k0],
            "out": [m1 - m0, m["n"]], "slot": ki, "tail": (k1 - k0) != m["ksz"],
            "x_bytes": (m1 - m0) * (k1 - k0) * 2,
        })
    m["aic_units"] = aic

    # Stage1 · AIV（两核共槽，各担 M 的一半）
    s1 = []
    for core in range(m["aiv"]):
        slot = core // 2
        mi, ki = divmod(slot, m["dim_k"])
        m0, k0 = mi * m["msz"], ki * m["ksz"]
        m1, k1 = min(m0 + m["msz"], m["bs"]), min(k0 + m["ksz"], m["k"])
        half = (m1 - m0 + 1) // 2
        a0, a1 = (m0, m0 + half) if core % 2 == 0 else (m0 + half, m1)
        s1.append({
            "core": core, "slot": slot, "m_idx": mi, "k_idx": ki,
            "half": "前半" if core % 2 == 0 else "后半",
            "rows": rng(a0, a1), "cols": rng(k0, k1), "x": [a1 - a0, k1 - k0],
        })
    m["aiv_s1_units"] = s1

    # Stage2 / Stage3 · AIV 行切分
    sb = []
    for core in range(m["aiv"]):
        if core < m["n_stage2"]:
            is_tail = core == m["n_stage2"] - 1
            loops, per, tail, nrows = split(t["rowLoopOfFormerBlock"], t["rowLoopOfTailBlock"],
                                            t["stage2RowFactor"], t["tailRowFactorOfFormerBlock"],
                                            t["tailRowFactorOfTailBlock"], is_tail)
            start = core * t["rowOfFormerBlock"]
            pre_loops, pre_per, _, _ = split(t["prePostRowLoopOfFormerBlock"],
                                             t["prePostRowLoopOfTailBlock"], t["prePostRowFactor"],
                                             t["prePostTailRowFactorOfFormerBlock"],
                                             t["prePostTailRowFactorOfTailBlock"],
                                             core == m["n_pre"] - 1)
            sb.append({
                "core": core, "role": "stage2+pre" if core < m["n_pre"] else "stage2",
                "rows_range": rng(start, start + nrows), "rows": nrows,
                "loops": loops, "per_loop": per, "tail_loop": tail,
                "x": [per * m["hc_mult"], t["dFactor"]], "d_loop": t["dLoop"],
                "ws_read": [m["ws_slots"], per, m["ws_last_axis"]],
                "comb": [pre_per, m["compact"]] if core < m["n_pre"] else None,
                "pre_loops": pre_loops, "pre_per": pre_per,
            })
        elif core < m["n_pre"] + m["n_sink"]:
            i = core - m["n_pre"]
            is_tail = i == m["n_sink"] - 1
            loops, per, tail, nrows = split(t["sinkhornRowLoopOfFormerBlock"],
                                            t["sinkhornRowLoopOfTailBlock"], t["sinkhornRowFactor"],
                                            t["sinkhornTailRowFactorOfFormerBlock"],
                                            t["sinkhornTailRowFactorOfTailBlock"], is_tail)
            start = i * t["sinkhornRowOfFormerBlock"]
            sb.append({
                "core": core, "role": "sinkhorn", "rows_range": rng(start, start + nrows),
                "rows": nrows, "loops": loops, "per_loop": per, "tail_loop": tail,
                "comb": [per, m["compact"]], "iters": m["iters"],
            })
        else:
            sb.append({"core": core, "role": "空闲", "rows_range": "-", "rows": 0})
    m["aiv_s2_units"] = sb

    # Stage1 · AIC 核内三层
    def layers(kk):
        return {
            "k": kk,
            "cv_steps": [ceil_align(kk, m["cvk"]) // m["cvk"], kk % m["cvk"]],
            "l1_blocks": ceil_align(kk, m["kl1"]) // m["kl1"],
            "l0_pairs": ceil_align(kk, L0_PAIR) // L0_PAIR,
            "mmad_seg": ceil_align(kk, K_L0_SIZE) // K_L0_SIZE,
        }
    m["aic_inner"] = {"full": layers(m["ksz"]), "tail": layers(m["tail_k"])}
    m["aic_inner_units"] = [dict(core=u["core"], k_idx=u["k_idx"], **layers(u["kk"])) for u in aic]

    # 核内 UB 轮
    s1_rows = (m["msz"] + 1) // 2
    f_m = m["s1_m_factor"]
    loops = ceil_align(s1_rows, f_m) // f_m
    m["inner"] = {
        "s1_rows": s1_rows, "s1_loops": loops, "s1_per_rows": f_m,
        "s1_tail_rows": s1_rows - (loops - 1) * f_m,
        "s2_rows": t["rowOfFormerBlock"], "s2_tail_rows": t["rowOfTailBlock"],
        "s2_per": [t["stage2RowFactor"] * m["hc_mult"], t["dFactor"]], "s2_dloop": t["dLoop"],
        "sk_rows": t["sinkhornRowOfFormerBlock"], "sk_tail_rows": t["sinkhornRowOfTailBlock"],
        "sk_loops": t["sinkhornRowLoopOfFormerBlock"], "sk_per": t["sinkhornRowFactor"],
        "sk_iters": t["iterTimes"],
    }
    return m
# ───────────────────────────────────────────────────────────── 分节
def sec0(m, meta):
    tk = meta.get("tiling_key")
    branch = BRANCH.get(int(tk), "未知分支") if tk is not None else "未知（缺 kernel_config）"
    x = shape(m["bs"], m["hc_mult"], m["d"])
    i = m["inner"]
    print(title(0, f"算子执行流程（tiling_key={tk} → {branch}）"))
    print(f"  输入   x{x} bf16    hc_fn{shape(m['n'], m['k'])} fp32    hc_scale[3] fp32    "
          f"hc_base{shape(m['n'])} fp32")
    print("   │")
    print(f"   ├─ Part1 · Stage1   AIC {m['aic']} 核 + AIV {m['aiv']} 核"
          f"（KERNEL_TYPE_MIX_AIC_1_2，2 AIV 共一槽）")
    print(f"   │   ① AIV 搬入+cast   每核 {i['s1_rows']} 行 / {i['s1_per_rows']} 行每轮"
          f" = {i['s1_loops']} 轮；每轮 {shape(i['s1_per_rows'], m['cvk'])} bf16 → cast fp32"
          f" → 写 xCast workspace")
    print(f"   │   ② AIV→AIC 握手    SYNC_AIV_TO_AIC_FLAG（跨核 SetFlag/WaitFlag）")
    print(f"   │   ③ AIC mmad        x {shape(m['ml1'], m['kl1'])} Nd2Nz(L1) × hc_fn "
          f"{shape(ceil_align(m['n'], BLOCK_CUBE), m['kl1'])}；L0 成对 k={L0_PAIR}，"
          f"mmad m={m['ml1']} n={m['n']} k={K_L0_SIZE} 累加")
    print(f"   │                     → partial {shape(m['ml1'], m['n'])} fp32 + squareSum"
          f" → workspace 第 kIdx 槽（共 {m['ws_slots']} 槽）")
    print(f"   │   ④ SyncAll<false>  全 CV 同步，Stage1 结束")
    print("   │")
    print(f"   ├─ Part2 · Stage2   AIV 0..{m['n_pre'] - 1}（{m['n_stage2']} 核，每核 "
          f"{i['s2_rows']} 行 / 尾核 {i['s2_tail_rows']}）")
    print(f"   │   ⑤ 收 partial      读 workspace {shape(m['ws_slots'], '<rows>', m['ws_last_axis'])}"
          f" → ReduceSum 去 k 维 → mix{shape(m['n'])}")
    print(f"   │   ⑥ CalcRsqrt+Pre   pre = f(mix, hc_base, hc_scale[0], hcEps)"
          f" → pre{shape(1, m['hc_mult'])} / rstd")
    print(f"   │   ⑦ 行循环          d 分 {i['s2_dloop']} 轮 × {i['s2_per'][1]}："
          f"x{shape(*i['s2_per'])} → ProcessY → y{shape(*i['s2_per'])} bf16 → 写 y")
    print(f"   │   ⑧ ProcessPost     post = g(mix, rstd, hc_scale[1]) → post{shape(m['hc_mult'])}/行")
    print("   │")
    print(f"   └─ Part2 · Stage3 sinkhorn   AIV {m['n_pre']}..{m['n_pre'] + m['n_sink'] - 1}"
          f"（{m['n_sink']} 核，每核 {i['sk_rows']} 行 / 尾核 {i['sk_tail_rows']}）")
    print(f"       ⑨ 收 partial→mix   compact {shape(i['sk_per'], m['compact'])} → "
          f"{i['sk_iters']} 次归一化迭代 → comb_frag{shape(m['hc_mult'], m['hc_mult'])}/行")
    print()
    print(f"  输出   y{x} bf16    post{shape(m['bs'], m['hc_mult'])} fp32    "
          f"comb_frag{shape(m['bs'], m['hc_mult'], m['hc_mult'])} fp32    "
          f"pre{shape(m['bs'], m['hc_mult'])} fp32（可选，hasPreOut）")
    print()
    print("  说明   Stage1 / Stage2 是同一 kernel 内先后两段（Part1 结束销毁 TPipe 后重建 Part2），"
          "不是两次 launch；")
    print("         Stage2 与 Stage3 由 blockIdx 静态划分到不同 AIV，同时并行执行。")


def sec1(meta):
    print(title(1, "输入与校验"))
    print(kv([
        ("input_tiling.bin", meta["tiling"]),
        ("文件大小", f"{meta['file_size']}B（结构体 {meta['struct_size']}B + "
                     f"{meta['file_size'] - meta['struct_size']}B 填充）"),
        ("字段表来源", meta["fields_desc"]),
        ("kernel_config", f"kernel={meta.get('kernel_name')}  block_dim={meta.get('block_dim')}  "
                          f"tiling_key={meta.get('tiling_key')}"),
        ("OpBasicInfo", f"{meta.get('op_name')}  Task={meta.get('task_us')}us  "
                        f"Block Dim={meta.get('block_dim')}  Mix={meta.get('mix_dim')}"),
    ]))


def sec2(m, t):
    print(title(2, "总览（全局切分）"))
    print(kv([
        ("问题规模", f"bs={m['bs']}  hcMix(N)={m['n']}  hcMult={m['hc_mult']}  d={m['d']}  k={m['k']}"),
        ("AIC 单元数", f"{m['aic']}  = cubeBlockDimM {m['dim_m']} × cubeBlockDimK {m['dim_k']}"
                       f"（cubeCoreNum={t['cubeCoreNum']}）"),
        ("AIV 单元数", f"{m['aiv']}  （Mix Block Dim，AIC:AIV = 1:{m['aiv'] // max(m['aic'], 1)}）"),
        ("AIC 网格", f"M 方向 {m['dim_m']} 块 × K 方向 {m['dim_k']} 块，每块 "
                     f"M={m['msz']} 行、K={m['ksz']}（尾块 {m['tail_k']}）"),
        ("AIV 分配", f"Stage1 用满 {m['aiv']} 核；Stage2 {m['n_stage2']} 核 + Stage3 {m['n_sink']} 核"),
        ("workspace", f"{m['ws_slots']} 份 partial × [bs={m['bs']}, {m['ws_last_axis']}] fp32 "
                      f"= {human(m['ws_bytes'])}（末轴 {m['ws_last_axis']} = CeilAlign(hcMix={m['n']}, "
                      f"512B/4B)={MM_CACHE_ELEMS}）"),
        ("行切分校验", f"Stage2 ({m['n_stage2']}−1)×{t['rowOfFormerBlock']}+{t['rowOfTailBlock']}"
                       f"={((m['n_stage2'] - 1) * t['rowOfFormerBlock'] + t['rowOfTailBlock'])}"
                       f" ; Stage3 ({m['n_sink']}−1)×{t['sinkhornRowOfFormerBlock']}"
                       f"+{t['sinkhornRowOfTailBlock']}"
                       f"={((m['n_sink'] - 1) * t['sinkhornRowOfFormerBlock'] + t['sinkhornRowOfTailBlock'])}"
                       f" ; bs={m['bs']}"),
    ]))


def sec3(m):
    print(title(3, f"Stage1 · 核间切块 —— AIC（{m['aic']} 单元，GEMM N={m['n']}）"))
    rows = [[u["core"], u["m_idx"], u["k_idx"], u["rows"], u["cols"], shape(*u["x"]),
             shape(*u["b"]), shape(*u["out"]), f"ws[{u['slot']}]",
             human(u["x_bytes"]) + (" ←K尾块" if u["tail"] else "")] for u in m["aic_units"]]
    print(table(["core", "mIdx", "kIdx", "M 行范围", "K 列范围", "x_tile", "b_tile(hc_fn)",
                 "out_partial", "写槽", "x 字节"], rows,
                ["<", ">", ">", ">", ">", "<", "<", "<", "<", "<"]))
    print()
    print(f"Stage1 · 核间切块 —— AIV（{m['aiv']} 单元，负责 cast；idx//2 定槽、idx%2 定 M 前后半段）")
    rows = [[u["core"], u["slot"], u["m_idx"], u["k_idx"], u["half"], u["rows"], u["cols"],
             shape(*u["x"])] for u in m["aiv_s1_units"]]
    print(table(["core", "槽", "mIdx", "kIdx", "半段", "M 行范围", "K 列范围", "x_tile"], rows,
                ["<", ">", ">", ">", "<", ">", ">", "<"]))


def sec4(m):
    print(title(4, "Stage1 · 核内切层（AIC: CV 步 → L1 → L0 → mmad；AIV: UB 轮）"))
    full, tail = m["aic_inner"]["full"], m["aic_inner"]["tail"]

    def qty(x):
        return f"{x[0]}(尾{x[1]})" if x[1] else str(x[0])

    rows = [
        ["CV 同步步", shape(m["ml1"], m["cvk"]), qty(full["cv_steps"]), qty(tail["cv_steps"]),
         "cvLoopKSize，AIC/AIV 握手粒度"],
        ["L1 单块", shape(m["ml1"], m["kl1"]), full["l1_blocks"], tail["l1_blocks"], "mL1Size × kL1Size"],
        [" └ A 搬入", f"Nd2Nz {shape(m['ml1'], m['kl1'])} 源 stride {m['cvk']}", "-", "-",
         "CopyInA1，源为 xCast workspace"],
        [" └ B 搬入", shape(ceil_align(m["n"], BLOCK_CUBE), m["kl1"]), "-", "-",
         f"CeilAlign(curNL1={m['n']}, {BLOCK_CUBE})"],
        ["L0 成对", f"K={L0_PAIR}（内含两段 {K_L0_SIZE}）", full["l0_pairs"], tail["l0_pairs"],
         "ComputeL0Pairs"],
        ["mmad", f"m={m['ml1']}, n={m['n']}, k={K_L0_SIZE}", full["mmad_seg"], tail["mmad_seg"],
         f"MmadA2 + MmadAB（每 {K_L0_SIZE}-K 两条）"],
    ]
    print(table(["层", "单块 shape", f"K={m['ksz']}", f"K={m['tail_k']}(尾块)", "依据"], rows,
                ["<", "<", ">", ">", "<"]))
    print()
    rows = [[u["core"], u["k_idx"], u["k"], u["l1_blocks"], qty(u["cv_steps"]), u["l0_pairs"],
             f"{u['mmad_seg']}段/{u['mmad_seg'] * 2}条", f"{m['ml1']}×{m['n']}×{u['k']}"]
            for u in m["aic_inner_units"]]
    print(table(["core", "kIdx", "K", "L1块(128)", "CV步(1024)", "L0对(64)", "mmad(32)", "单核计算量"],
                rows, ["<", ">", ">", ">", ">", ">", ">", "<"]))
    print()
    i = m["inner"]
    print(kv([
        ("AIV stage1", f"每核 {i['s1_rows']} 行 → {i['s1_loops']} 轮 × "
                       f"{shape(i['s1_per_rows'], m['cvk'])} bf16（末轮 "
                       f"{shape(i['s1_tail_rows'], m['cvk'])}），cast 后 fp32 同 shape 写 workspace"),
    ], key_w=13))
def sec5(m):
    print(title(5, f"Stage2 · 核间切块 —— AIV 0..{m['n_pre'] - 1}"
                   f"（{m['n_stage2']} 核，读 partial → pre → y → post）"))
    rows = [[u["core"], u["role"], u["rows_range"], u["rows"], f"{u['loops']}×{u['per_loop']}",
             f"{shape(*u['x'])}×{u['d_loop']}", shape(*u["ws_read"]),
             shape(*u["comb"]) if u["comb"] else "-"] for u in m["aiv_s2_units"]
            if u["role"] not in ("sinkhorn", "空闲")]
    print(table(["core", "角色", "行范围", "行数", "轮×行", "每轮 x/y tile", "每轮 workspace 读",
                 "每轮 comb"], rows, ["<", "<", ">", ">", "<", "<", "<", "<"]))


def sec6(m):
    i = m["inner"]
    u0 = m["aiv_s2_units"][0]
    print(title(6, "Stage2 · 核内切块（UB 层）"))
    print(kv([
        ("收 partial", f"每轮读 {shape(*u0['ws_read'])} fp32，沿 {m['ws_slots']} 个 k 槽求和"
                       f" → mix{shape(m['n'])}"),
        ("pre", f"ProcessPre → pre{shape(1, m['hc_mult'])} + rstd（每行 1 次 rsqrt）"),
        ("y 主体", f"每行 d 分 {u0['d_loop']} 轮 × {u0['x'][1]}：x{shape(*i['s2_per'])} bf16"
                   f" → ProcessY → y{shape(*i['s2_per'])} bf16，行内不再切"),
        ("post", f"ProcessPost → {shape(m['pp_per_rows'], m['compact'])} fp32/轮"
                 f"（{m['pp_per_rows']} 行），{m['pp_loops']} 轮（尾核 {m['pp_tail_loops']}）"),
        ("行循环", f"每核 {i['s2_rows']} 轮 × 每轮 {u0['per_loop']} 行（尾核 {i['s2_tail_rows']} 轮），"
                   f"每行 1 次 x 搬入 + 1 次 y 搬出"),
    ], key_w=12))


def sec7(m):
    i = m["inner"]
    sk = [u for u in m["aiv_s2_units"] if u["role"] == "sinkhorn"]
    print(title(7, f"Stage3 · sinkhorn —— AIV {m['n_pre']}..{m['n_pre'] + m['n_sink'] - 1}"
                   f"（{m['n_sink']} 核，{i['sk_iters']} 次迭代归一化）"))
    rows = [[u["core"], u["rows_range"], u["rows"],
             f"{u['loops']}×({'+'.join([str(u['per_loop'])] * (u['loops'] - 1) + [str(u['tail_loop'])])})",
             f"{shape(*u['comb'])} ×{u['iters']} 迭代"] for u in sk]
    print(table(["core", "行范围", "行数", "轮×行", "每轮 comb 块"], rows,
                ["<", ">", ">", "<", "<"]))
    tail_last = sk[-1]["tail_loop"] if sk else "-"
    print(kv([
        ("核内", f"{shape(i['sk_per'], m['compact'])} fp32/轮 → {i['sk_iters']} 次迭代，"
                 f"{i['sk_loops']} 轮（末轮 {i['sk_per']} → 前 N-1 核 {sk[0]['tail_loop']}，"
                 f"尾核 {tail_last}）"),
        ("紧凑态", f"{m['compact']} 元素 = COMPACT_HC_MULT(hcMult={m['hc_mult']})²，"
                   f"对应输出 comb_frag{shape(m['hc_mult'], m['hc_mult'])}/行"),
    ], key_w=8))


def sec8(t, fields):
    print(title(8, f"tiling 字段全量（{len(fields)} 项，按声明顺序；0 多为本路径未使用）"))
    rows = []
    for idx, (name, kind) in enumerate(fields):
        v = t[name]
        rows.append([idx, name, f"{v:.3g}" if kind == "f32" else v, FIELD_NOTE.get(name, "-")])
    print(table(["#", "字段", "值", "用途"], rows, [">", "<", ">", "<"]))


def sec9(m, t, meta):
    print(title(9, "校验与负载提示"))
    checks = [
        ("AIC 核数", f"{m['dim_m']}×{m['dim_k']} = {m['aic']}", f"cubeCoreNum={t['cubeCoreNum']}",
         m["aic"] == t["cubeCoreNum"]),
        ("Block Dim", f"tiling 推算 {m['aic']}", f"kernel/OpBasicInfo {meta.get('block_dim')}",
         meta.get("block_dim") in (None, m["aic"])),
        ("Mix Block Dim", f"AIC×2 = {m['aic'] * 2}", f"OpBasicInfo {meta.get('mix_dim')}",
         meta.get("mix_dim") in (None, m["aic"] * 2)),
        ("M 轴覆盖", f"{m['dim_m']}×{m['msz']} = {m['dim_m'] * m['msz']}", f"bs={m['bs']}",
         m["dim_m"] * m["msz"] == m["bs"]),
        ("K 轴覆盖", f"({m['dim_k']}−1)×{m['ksz']}+{m['tail_k']}", f"k={m['k']}",
         (m["dim_k"] - 1) * m["ksz"] + m["tail_k"] == m["k"]),
        ("Stage2 行覆盖", f"({m['n_stage2']}−1)×{t['rowOfFormerBlock']}+{t['rowOfTailBlock']}",
         f"bs={m['bs']}", (m["n_stage2"] - 1) * t["rowOfFormerBlock"] + t["rowOfTailBlock"] == m["bs"]),
        ("Stage3 行覆盖",
         f"({m['n_sink']}−1)×{t['sinkhornRowOfFormerBlock']}+{t['sinkhornRowOfTailBlock']}",
         f"bs={m['bs']}",
         (m["n_sink"] - 1) * t["sinkhornRowOfFormerBlock"] + t["sinkhornRowOfTailBlock"] == m["bs"]),
        ("AIV 分区", f"{m['n_pre']}+{m['n_sink']}", f"Mix Block Dim {m['aiv']}",
         m["n_pre"] + m["n_sink"] == m["aiv"]),
    ]
    print(table(["校验项", "推算", "参照", "结论"],
                [[c[0], c[1], c[2], "OK" if c[3] else "不一致"] for c in checks],
                ["<", ">", ">", "<"]))
    print()
    tail_cores = ", ".join(str(i) for i in range(m["dim_k"] - 1, m["aic"], m["dim_k"]))
    print(f"  1) Stage1 AIC K 尾块: kIdx={m['dim_k'] - 1} 只算 {m['tail_k']}"
          f"（少 {m['ksz'] - m['tail_k']}，{(m['ksz'] - m['tail_k']) / m['ksz']:.1%}），"
          f"涉及 core {tail_cores}")
    print(f"  2) Stage2/3 分区: {m['n_stage2']} 核 × {t['rowOfFormerBlock']} 行"
          f"（尾核 {t['rowOfTailBlock']}）vs {m['n_sink']} 核 × {t['sinkhornRowOfFormerBlock']} 行"
          f"（尾核 {t['sinkhornRowOfTailBlock']}），行数比 "
          f"{t['sinkhornRowOfFormerBlock'] / t['rowOfFormerBlock']:.1f}:1")
    print(f"  3) Stage1 AIV: 每核 {m['inner']['s1_rows']} 行 / {m['s1_m_factor']} = "
          f"{m['inner']['s1_loops']} 轮，末轮仅 {m['inner']['s1_tail_rows']} 行"
          f"（UB 利用率 {m['inner']['s1_tail_rows'] / m['s1_m_factor']:.0%}）")


# ───────────────────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser(description="打印 HcPre 的执行流程与全部 tiling 切块信息")
    ap.add_argument("--tiling", help="input_tiling.bin 或其所在 OPPROF 目录")
    ap.add_argument("--opprof", help="OPPROF_* 目录（自动拼 dump/kernel_data/input_tiling.bin）")
    ap.add_argument("--header", default=str(HERE.parents[4] / "csrc/moe/hc_pre/op_host/hc_pre_tiling.h"),
                    help="tiling 头文件（决定字段顺序），默认取仓库内路径")
    ap.add_argument("--only", help="只输出指定分节，如 0,2,3")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而不是表格")
    args = ap.parse_args()

    tiling = find_tiling(args.tiling, args.opprof)
    buf = tiling.read_bytes()
    fields, desc = load_fields(args.header)
    t, struct_size = decode(buf, fields)
    meta = read_meta_files(tiling)
    meta.update({"tiling": str(tiling), "file_size": len(buf), "struct_size": struct_size,
                 "fields_desc": desc})
    m = build_model(t, meta.get("mix_dim"))

    if args.json:
        print(json.dumps({"meta": meta, "tiling": t, "model": m}, ensure_ascii=False, indent=2))
        return

    want = {int(x) for x in args.only.split(",")} if args.only else None
    sections = {
        0: lambda: sec0(m, meta), 1: lambda: sec1(meta), 2: lambda: sec2(m, t),
        3: lambda: sec3(m), 4: lambda: sec4(m), 5: lambda: sec5(m),
        6: lambda: sec6(m), 7: lambda: sec7(m), 8: lambda: sec8(t, fields),
        9: lambda: sec9(m, t, meta),
    }
    for no, fn in sections.items():
        if want is None or no in want:
            fn()


if __name__ == "__main__":
    main()