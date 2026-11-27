#!/bin/bash
# npu_hc_pre_v2 单算子流水采集
#
# 上板（真机）采不到 pipe 级流水，硬件 PMU 只有累计计数器。要看 AIC/AIV 时序必须走
# TimelineDetail，底层是 CAModel 周期精确仿真。
#
# 用法：bash run.sh            # 采 core0
#       CORE_ID=5 bash run.sh  # 采 core5
#
set -e
cd "$(dirname "$0")"
CORE_ID=${CORE_ID:-0}

# 自定义算子环境（换镜像要改路径：python3 -c "import os,vllm_ascend;print(os.path.dirname(vllm_ascend.__file__))"）
source /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

# 裸跑一次，确认算子功能正常
python3 hc_pre_run.py

# 只采 HcPre；--core-id 只解析 1 个 core（1 AIC + 2 AIV）
msopprof --output="opprof_core${CORE_ID}" \
         --aic-metrics=TimelineDetail \
         --kernel-name=HcPre \
         --core-id="${CORE_ID}" \
         python3 hc_pre_run.py

echo "[done] opprof_core${CORE_ID}/OPPROF_*/simulator/core${CORE_ID}.*"