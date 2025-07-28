#!/bin/bash
docker build --platform linux/amd64 --tag mlperf_rocm_sdxl:micro_shortfin_nogil_rocm7_v1 --file SDXL_inference/sdxl_harness_rocm7_py313t.dockerfile .
