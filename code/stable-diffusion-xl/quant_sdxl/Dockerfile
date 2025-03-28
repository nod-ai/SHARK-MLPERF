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

FROM rocm/dev-ubuntu-22.04:6.1.2

# ######################################################
# # Install MLPerf+Shark reference implementation
# ######################################################
ENV DEBIAN_FRONTEND=noninteractive

# apt dependencies
RUN apt-get update && apt-get install -y \
ffmpeg libsm6 libxext6 git wget unzip \
  software-properties-common git \
  build-essential curl cmake ninja-build clang lld vim nano python3.10-dev python3.10-venv && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/models/huggingface/

WORKDIR /workspace

COPY . .

RUN pip install setuptools==65.2.0
RUN pip install -r ./env/requirements.txt
