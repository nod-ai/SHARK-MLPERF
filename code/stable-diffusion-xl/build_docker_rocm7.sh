#!/bin/bash
docker build --no-cache --platform linux/amd64 --tag mlperf_rocm_sdxl:micro_shortfin_rocm7_v1 --file SDXL_inference/sdxl_harness_rocm7_py311.dockerfile .
