# 0.32.0rc 社区镜像流水线
# https://linkex.alipay.com/project/166200011/pipeline/725664220?tenant_path=alipay&jobId=2194800317

itask create --name gchtest \
--image hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.32.0rc1-a3-openeuler-20261111103136_aarch64 \
--model hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_79400015_deepseek-v4.1-flash-1017-w4a8:157400109_20261103190349 \
--16card \
-t a3

