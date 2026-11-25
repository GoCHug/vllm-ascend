# hc_pre 单算子 profiling 归档

对 `torch.ops._C_ascend.npu_hc_pre_v2`（DSv4 flash 形状：T=2048 / hidden=4096 / hc_mult=4 / mix_hc=24）
做单算子采集，验证 [mhc.md 4.2 节](../mhc.md) 关于 AIC 流水的几条论断。

## 结论速览

| 项 | 值 |
| --- | --- |
| op 总时长（T=2048） | **181.4 us** |
| AIC（24 核） | 109.2 us，核间离散度 0.7% |
| AIV（48 核） | **170.4 us**（占总时长 94%） |
| AIC 最忙的流水线 | MTE2 33.8%（GM→L1 搬运） |
| AIC 最闲的流水线 | Cube 22.4% |
| AIV 等待 AIC 的时间 | 82.3 us（`wait_id9`，占 AIV 48%） |
| AIC 等 Cube 的时间 | 55.2 us（`scalar_cube_stall`，占 AIC 50%） |

一句话：**AIV 决定总时长，AIC 内部的 Cube 只跑了 22%，瓶颈是搬运与同步而不是算力。**
详见 [`report/analysis.md`](report/analysis.md)。

## 目录

```text
profiling/
├── README.md              本文件
├── report/
│   ├── analysis.md        分析报告（数据 + 对照 + 建议）
│   ├── pipe_util.svg      各流水线利用率柱状图（设备侧）
│   ├── trace_gantt.svg    指令级流水甘特图（core0，3 us 窗口）
│   └── trace_core0_3us.json  可直接拖进 Perfetto / chrome://tracing 的 timeline
├── scripts/
│   ├── prof_hc_pre.py     单算子入口：造数 + warmup + 计时
│   ├── run_prof.sh        msprof 包装（plain / device / sim）
│   ├── analyze_pipe_util.py   解析 PipeUtilization.csv
│   ├── analyze_op_summary.py  解析 op_summary.csv（msprof 其他模式用）
│   ├── plot_pipe_util.py      画利用率柱状图
│   └── extract_trace.py       从 simulator 的 trace.json 按核抽指令时间线
└── raw/
    ├── op_t2048/          主数据 · T=2048（181.4 us）
    ├── op_t512/           规模扫描 · T=512（70.9 us）
    ├── op_t1024/          规模扫描 · T=1024（107.8 us）
    ├── op_t4096/          规模扫描 · T=4096（326.9 us）
    └── trace_core0.csv.gz     指令级时间线（core0，24.3 万条事件）
```

## 环境

容器：`itask start gchtest`（A3，2 卡，镜像 `vllm:release_0.32.0_1111`，CANN 9.2.1）

```bash
# 1. 注册自定义算子（vendors 下的 hc_pre 依赖它把算子库加进路径）
source /usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

# 2. 打开自定义算子（脚本内部已调用）
python3 -c "from vllm_ascend.utils import enable_custom_op; enable_custom_op()"
```

## 跑

```bash
bash scripts/run_prof.sh plain     # 只跑一遍看耗时
bash scripts/run_prof.sh device    # msprof op 设备侧：PipeUtilization.csv
SOC_VERSION=Ascend910_9362 ITERS=1 WARMUP=0 bash scripts/run_prof.sh sim   # 指令级（慢，约 11 分钟）
```

形状可覆盖：

```bash
T=4096 HIDDEN=4096 ITERS=20 bash scripts/run_prof.sh device
```

## 容器里的原始产物在哪

```bash
itask exec gchtest            # 交互进入
# 工作目录
cd /a3_inference/itask/workdir/gch02599191/hc_pre_prof
```

```text
hc_pre_prof/
├── prof_hc_pre.py  run_prof.sh  extract_trace.py      脚本
├── prof_out/
│   ├── OPPROF_20261124162337_VUSZDEXBBSYUFAOC/   设备侧 T=2048（2.4 MB）
│   │   ├── OpBasicInfo.csv
│   │   ├── PipeUtilization.csv
│   │   ├── visualize_data.bin        ← MindStudio Insight 导入这一层
│   │   └── dump/                     DeviceProf*.bin / L2Cache.bin / aicore_binary.o / duration.bin
│   └── OPPROF_20261124162503_AQIJUPESUEBIYXJK/   指令级仿真（2.8 GB）
│       ├── simulator/
│       │   ├── core0.cubecore0/      trace.json 27 MB + instr_exe.csv 125 KB + code_exe.csv
│       │   ├── core0.veccore0/  core0.veccore1/   …（共 60 个核目录）
│       │   ├── trace.json            976 MB，全核时间线
│       │   └── visualize_data.bin    984 MB
│       └── dump/
├── prof_out_t512/OPPROF_20261124163538_QBNDZQBHALCAWRSE/    设备侧 T=512（2.4 MB）
├── prof_out_t1024/OPPROF_20261124163551_DBUFRNUTLJODMINH/   设备侧 T=1024（2.4 MB）
└── prof_out_t4096/OPPROF_20261124163603_SNYWBVLVIOCZFCAE/   设备侧 T=4096（2.4 MB）
```

每个核目录里的 `*_instr_exe.csv` 是**按指令聚合**的耗时表（`instr, pipe, call_count, cycles, running_time(us), detail`），
比 trace 小得多，适合先看瓶颈。仓库里已归档 cube 核的这一份：`raw/core*.cubecore0/`。

## 看 timeline

两条路：

**① 设备侧（msprof op 原始产物）**——用 MindStudio Insight

`visualize_data.bin` 不能单独打开：它是 msprof 给 MindStudio Insight 的私有容器
（文件头是 `ZZ{"ad…"` 的 JSON 块），必须由工具读。

正确姿势：MindStudio Insight → **Import → 选整个 `OPPROF_*` 目录**（不是 .bin），
再进 Timeline / Operator 视图。

⚠️ 但**只有 `--aic-metrics=TimelineDetail` 采出来的目录才有时间线**。
`raw/op_t*/OPPROF_*/`（约 2.4 MB/个）是 `PipeUtilization` 模式，只有指标表，
导入后看不到 Timeline，这是文件小的原因。

要时间线就加这个参数重采（会走一遍指令级仿真，3 GB 左右）：

```bash
msprof op --application="python3 prof_hc_pre.py --t 2048" \
          --output=$PWD/prof_tl --aic-metrics=TimelineDetail \
          --kernel-name='HcPre*' --launch-count=1
```

产出的 `prof_tl/OPPROF_*/simulator/` 里是 24 个 cube 核 + 48 个 vector 核的
`trace.json` 与 `*_instr_exe.csv`，外加 1 GB 的 `visualize_data.bin`。
超过 `itask cp` 的 100 MB 上限，需要 OBS 中转或者按核抽小片。

**② 指令级（simulator）**——1 GB 的 `trace.json` 在容器里，仓库里放的是切片

- `report/trace_core0_3us.json`：core0 在 40~43 us 的一段，234 条指令事件；
  拖进 [Perfetto](https://ui.perfetto.dev) 或 `chrome://tracing` 即可，
  也能用 MindStudio Insight 的 Trace 视图打开；
- `report/trace_gantt.svg`：同一段数据的静态甘特图，直接看 MTE1/CUBE 怎么交错。

自己切别的窗口：

```bash
python3 scripts/slice_trace.py raw/trace_core0.csv.gz out.json --start 95 --window 5 \
        --pipes MTE1,MTE2,MTE3,CUBE,FIXP
python3 scripts/plot_trace_gantt.py out.json out.svg
```

注意：

- `--kernel-name` 必须是 `HcPre*`（实际 kernel 名形如 `HcPre_<hash>_0_mix_aic`）；
- `set_env.bash` 里有未定义变量，`set -u` 下 source 会报错，脚本里已临时关掉 nounset；
- `itask cp` 单次上限 100 MB，指令级产物（2.8 GB）留在容器里，按需用 `extract_trace.py` 抽小片。
