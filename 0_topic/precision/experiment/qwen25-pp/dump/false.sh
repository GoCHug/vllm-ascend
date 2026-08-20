export TASK_QUEUE_ENABLE=1
#export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_OP_EXPANSION_MODE="AIV"
nohup vllm serve /home/admin/model-csi/models/modelhub_99547_qwen2-5-7b-instruct-96200119_20251211191451/model/ \
    --served-model-name qwen2.5 \
    --trust-remote-code \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 1 \
    --max-model-len 1024 \
    --max-num-batched-tokens 25 \
    --port 8115 \
    --enforce-eager \
    --gpu-memory-utilization 0.9 \
    --additional-config '{
        "dump_config": {
        "task": "statistics",
        "level": "mix",
        "dump_path": "./false",
        "statistics": {
            "list": []
        }
        }
    }' > ./false.log 2>&1 &
