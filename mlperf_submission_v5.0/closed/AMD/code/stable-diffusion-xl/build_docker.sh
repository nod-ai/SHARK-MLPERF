#!/bin/bash
docker build --platform linux/amd64 --tag mlperf_rocm_sdxl:micro_shortfin_v54 --file SDXL_inference/sdxl_harness_rocm_shortfin_from_source_iree.dockerfile .
