#!/bin/bash
docker build --no-cache --platform linux/amd64 --tag mlperf_rocm_sdxl:micro_shortfin_v56 --file SDXL_inference/sdxl_harness_rocm_shortfin_from_source_iree.dockerfile .
