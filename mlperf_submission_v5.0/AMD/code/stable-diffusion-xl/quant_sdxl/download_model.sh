#!/bin/bash
#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################

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
