#!/usr/bin/env bash
# hc_pre 单算子 profiling 一键脚本（在 A3 容器里跑）
#
#   bash run_prof.sh device      # 设备侧 msprof op：op_summary.csv + 流水利用率
#   bash run_prof.sh sim         # 指令级 simulator：core*.instr_log.dump，喂 MindStudio Insight
#   bash run_prof.sh plain       # 不采集，只跑一遍看耗时
#
# 常用覆盖：
#   T=2048 HIDDEN=4096 ITERS=20 bash run_prof.sh device
#   SOC_VERSION=Ascend910_9382 bash run_prof.sh sim   # 设备实测就是这个 soc
set -euo pipefail

MODE=${1:-device}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT="$HERE/prof_hc_pre.py"

T=${T:-2048}
HIDDEN=${HIDDEN:-4096}
HC_MULT=${HC_MULT:-4}
MIX_HC=${MIX_HC:-24}
SINKHORN=${SINKHORN:-20}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-20}
OUT=${OUT:-$HERE/prof_out}
SOC_VERSION=${SOC_VERSION:-Ascend910_9382}
KERNEL=${KERNEL:-HcPre*}   # 实际 kernel 名形如 HcPre_<hash>_0_mix_aic

# 自定义算子环境（vendors 里注册的 hc_pre 依靠这个脚本把算子库加进路径）
VENDOR_ENV=/usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
if [[ -f "$VENDOR_ENV" ]]; then
    # set_env.bash 里有未定义变量，先关掉 nounset
    set +u
    # shellcheck disable=SC1090
    source "$VENDOR_ENV"
    set -u
else
    echo "[warn] vendor env not found: $VENDOR_ENV"
fi

# 单算子采集必须关掉异步下发，否则 msprof 抓到的是一次 batch 而不是逐 op
export ASCEND_LAUNCH_BLOCKING=1
export ASCEND_GLOBAL_LOG_LEVEL=3          # 少打日志，避免污染时间线
export ASCEND_SLOG_PRINT_TO_STDOUT=0

PY_ARGS="--t $T --hidden $HIDDEN --hc-mult $HC_MULT --mix-hc $MIX_HC \
--sinkhorn-iters $SINKHORN --warmup $WARMUP --iters $ITERS"

mkdir -p "$OUT"

echo "== hc_pre profiling =="
echo "   mode    : $MODE"
echo "   shape   : T=$T  hidden=$HIDDEN  K=$((HC_MULT * HIDDEN))  N=$MIX_HC"
echo "   iters   : warmup=$WARMUP / measure=$ITERS"
echo "   out     : $OUT"

case "$MODE" in
  plain)
    python3 "$SCRIPT" $PY_ARGS
    ;;

  device)
    command -v msprof >/dev/null || { echo "msprof not found in PATH"; exit 1; }
    msprof op \
        --application="python3 $SCRIPT $PY_ARGS" \
        --output="$OUT" \
        --aic-metrics=PipeUtilization \
        --kernel-name="$KERNEL"
    ;;

  sim)
    command -v msprof >/dev/null || { echo "msprof not found in PATH"; exit 1; }
    # 指令级仿真：不需要整卡跑满，但要求算子可被单次调用
    msprof op simulator \
        --soc-version="$SOC_VERSION" \
        --application="python3 $SCRIPT $PY_ARGS" \
        --output="$OUT" \
        --aic-metrics=PipeUtilization
    ;;

  *)
    echo "usage: bash run_prof.sh [device|sim|plain]"; exit 2
    ;;
esac

echo
echo "== 产物 =="
find "$OUT" -maxdepth 3 -name "*.csv" -o -maxdepth 3 -name "*.dump" | head -20
echo
echo "看这几项："
echo "  op_summary.csv    各核 aicore_time / aic_total_time / aiv_total_time"
echo "  op_statistic.csv  按算子聚合的耗时"
echo "  simulator/        (sim 模式) 指令级时间线，MindStudio Insight 直接打开"
