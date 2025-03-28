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

DATA_DIR="/data"
CAPTIONS_DIR="coco/SDXL/captions"
CAPTIONS_FILE="captions.tsv"
LATENTS_DIR="coco/SDXL/latents"
LATENTS_FILE="latents.pt"
ACCURACY_DIR="coco/SDXL/tools"
ACCURACY_FILE="val2014.npz"

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

if [ -e "${DATA_DIR}/${ACCURACY_FILE}" ]
then
    echo "Dataset for SDXL already exists!"
else
    download_file ${DATA_DIR} ${ACCURACY_DIR} https://raw.githubusercontent.com/mlcommons/inference/master/text_to_image/tools/val2014.npz ${ACCURACY_FILE}
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

md5sum ${DATA_DIR}/${ACCURACY_DIR}/${ACCURACY_FILE} | grep "b399b117828bedc564d0aaaf308bcc05"
if [ $? -ne 0 ]; then
    echo "SDXL fix latent md5sum mismatch"
    exit -1
fi
