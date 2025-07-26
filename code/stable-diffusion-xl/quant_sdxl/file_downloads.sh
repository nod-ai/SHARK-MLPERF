#!/bin/bash
#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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


# $1 indicates the subdirectory (i.e. models or data)
# $2 indicates the benchmark name (i.e. ResNet50, dlrm, etc.)
# $3 indicates the URL to download from
# $4 indicates the destination filename
function download_file {
    _SUB_DIR=$1/$2

    if [ ! -d ${_SUB_DIR} ]; then
        echo "Creating directory ${_SUB_DIR}"
        mkdir -p ${_SUB_DIR}
    fi
    echo "Downloading $2 $1..." \
        && wget $3 -O ${_SUB_DIR}/$4 \
        && echo "Saved $2 $1 to ${_SUB_DIR}/$4!"
}

# $1 indicates the subdirectory (i.e. models or data)
# $2 indicates the benchmark name (i.e. ResNet50, dlrm, etc.)
# $3 indicates the destination filename
function download_sdxl_file {
    _SUB_DIR=$1/$2

    if [ ! -d ${_SUB_DIR} ]; then
        echo "Creating directory ${_SUB_DIR}"
        mkdir -p ${_SUB_DIR}
    fi
    echo "Downloading $2 $1..." \
        && echo ${_SUB_DIR}/$3 \
        && sudo -v ; curl https://rclone.org/install.sh | sudo bash \
        && rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
    cd ${SUB_DIR} 
    sudo rclone copy mlc-inference:mlcommons-inference-wg-public/stable_diffusion_fp16 ${_SUB_DIR} -P --no-check-certificate
    echo "Saved $2 $1 to ${_SUB_DIR}/$3!"
    cd -
}
