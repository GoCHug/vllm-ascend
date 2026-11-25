#!/bin/bash
# 运行 npu_hc_pre_v2 单次调用测试
#
# 用法：
#   bash /a3_inference/itask/workdir/gch02599191/hc_pre/run.sh
#
set -e

ENV_FILE=/usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
else
    echo "[warn] 未找到自定义算子环境变量文件: ${ENV_FILE}" >&2
fi

cd "${SCRIPT_DIR}"
exec python3 "${SCRIPT_DIR}/hc_pre_run.py" "$@"