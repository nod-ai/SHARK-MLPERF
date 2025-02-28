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

DATA_DIR="/data"
CAPTIONS_DIR="coco/SDXL/captions"
CAPTIONS_FILE="captions.tsv"
LATENTS_DIR="coco/SDXL/latents"
LATENTS_FILE="latents.pt"

if [ -e "${DATA_DIR}/${CAPTIONS_DIR}/${CAPTIONS_FILE}" ]
then
    echo "Dataset for SDXL already exists!"
else
    download_file ${DATA_DIR} ${CAPTIONS_DIR} https://raw.githubusercontent.com/mlcommons/inference/master/text_to_image/coco2014/captions/captions_source.tsv ${CAPTIONS_FILE}
fi

if [ -e ${DATA_DIR}/${LATENTS_DIR}/${LATENTS_FILE} ]
then
    echo "Fix latent for SDXL already exists!"
else
    download_file ${DATA_DIR} ${LATENTS_DIR} https://github.com/mlcommons/inference/raw/master/text_to_image/tools/latents.pt ${LATENTS_FILE}
fi

md5sum ${DATA_DIR}/${CAPTIONS_DIR}/${CAPTIONS_FILE} | grep "24ba91c1e0fc04e70895385b4a57dca1"
if [ $? -ne 0 ]; then
    echo "SDXL data md5sum mismatch"
    exit -1
fi

md5sum ${DATA_DIR}/${LATENTS_DIR}/${LATENTS_FILE} | grep "58f4165e574452b9ce6a1a32c2fb3908"
if [ $? -ne 0 ]; then
    echo "SDXL fix latent md5sum mismatch"
    exit -1
fi
