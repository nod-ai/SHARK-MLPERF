#!/bin/bash
echo "always" | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /data/mlperf_sdxl/data:/data -v /data/mlperf_sdxl/models:/models -v `pwd`/SDXL_inference/:/mlperf/harness -w /mlperf/harness --name `whoami`micro_shortfin_v54 mlperf_rocm_sdxl:micro_shortfin_v54
