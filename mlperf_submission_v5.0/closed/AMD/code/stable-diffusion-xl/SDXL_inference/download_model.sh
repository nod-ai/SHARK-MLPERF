#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

source ./file_downloads.sh

MODEL_DIR=/models

if [ -e "${MODEL_DIR}/SDXL/official_pytorch/fp16" ]
then
    echo "Model zip for SDXL already exists!"
else
    # Download the fp16 raw weights of MLCommon hosted HF checkpoints
    download_file ${MODEL_DIR} SDXL/official_pytorch/fp16 \
        https://cloud.mlcommons.org/index.php/s/LCdW5RM6wgGWbxC/download \
        stable_diffusion_fp16.zip
fi

if [ -e "${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/" ]
then
    echo "Unzipped model folder for SDXL already exists!"
else
    unzip ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16.zip \
    -d ${MODEL_DIR}/SDXL/official_pytorch/fp16/
fi

md5sum ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/text_encoder/model.safetensors | grep "81b87e641699a4cd5985f47e99e71eeb"
if [ $? -ne 0 ]; then
    echo "SDXL CLIP1 fp16 model md5sum mismatch"
    exit -1
fi

md5sum ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/text_encoder_2/model.safetensors | grep "5e540a9d92f6f88d3736189fd28fa6cd"
if [ $? -ne 0 ]; then
    echo "SDXL CLIP2 fp16 model md5sum mismatch"
    exit -1
fi

if [ -e "${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/scheduled_unet/diffusion_pytorch_model.fp16.safetensors" ]
then
    echo "unet already renamed to match shark expected path"
else
    # Note: we need this for scheduled_unet
    ln -s ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/unet/diffusion_pytorch_model.safetensors \
          ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/diffusion_pytorch_model.fp16.safetensors
    ln -s ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/unet/config.json \
          ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/config.json
    echo "Renamed unet to match shark expected path"
fi

md5sum ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/diffusion_pytorch_model.fp16.safetensors | grep "edfa956683fb6121f717d095bf647f53"
if [ $? -ne 0 ]; then
    echo "SDXL UNet fp16 model md5sum mismatch"
    exit -1
fi

if [ -e "${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/vae/diffusion_pytorch_model.fp16.safetensors" ]
then
    echo "vae already renamed to match shark expected path"
else
    ln -s ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/vae/diffusion_pytorch_model.safetensors \
          ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/vae/diffusion_pytorch_model.fp16.safetensors
    echo "Renamed vae to match shark expected path"
fi

md5sum ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/vae/diffusion_pytorch_model.fp16.safetensors | grep "25fe90074af9a0fe36d4a713ad5a3a29"
if [ $? -ne 0 ]; then
    echo "SDXL VAE fp16 model md5sum mismatch"
    exit -1
fi

