FROM rocm/pytorch-private:vllm-fp4-405b-250603-asm-rc2


# ######################################################
# # Install MLPerf+Shark reference implementation
# ######################################################
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# apt dependencies
RUN apt-get --fix-broken install -y
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git wget unzip \
    software-properties-common git \
    build-essential curl cmake ninja-build clang lld vim nano python3.11-dev python3.11-venv gfortran pkg-config libopenblas-dev libssl-dev openssl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


RUN curl -fsSL https://pyenv.run | bash

ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:${PATH}"
RUN echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
RUN eval "$(pyenv init --bash)"
RUN pyenv install 3.11.3
RUN pyenv global 3.11.3

RUN pip install --upgrade pip setuptools wheel && \
    pip install pybind11 'nanobind<2' numpy==1.* pandas && \
    pip install hip-python hip-python-as-cuda -i https://test.pypi.org/simple

# Rust requirements
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

#FROM rocm/dev-ubuntu-22.04:6.1.2
#
## ######################################################
## # Install MLPerf+Shark reference implementation
## ######################################################
#ENV DEBIAN_FRONTEND=noninteractive
#
## apt dependencies
#RUN apt-get update && apt-get install -y \
#ffmpeg libsm6 libxext6 git wget unzip \
#  software-properties-common git \
#  build-essential curl cmake ninja-build clang lld vim nano python3.11-dev python3.11-venv gfortran pkg-config libopenblas-dev && \
#  apt-get clean && rm -rf /var/lib/apt/lists/*
#RUN python3.11 -m pip install --upgrade pip setuptools wheel && \
#    python3.11 -m pip install pybind11 'nanobind<2' numpy==1.* pandas && \
#    python3.11 -m pip install hip-python hip-python-as-cuda -i https://test.pypi.org/simple
#
## Rust requirements
#RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#ENV PATH="/root/.cargo/bin:${PATH}"
#
# install loadgen
RUN mkdir /mlperf/ && cd /mlperf && \
    git clone --recursive https://github.com/mlcommons/inference.git && \
    cd inference/loadgen && \
    mkdir -p /mlperf/harness/ && \
    CFLAGS="-std=c++14" python3.11 setup.py install

RUN mkdir -p /mlperf/shark_reference/ && cp -r /mlperf/inference/text_to_image/* /mlperf/shark_reference/ && cp /mlperf/inference/mlperf.conf /mlperf/shark_reference/
RUN cd /mlperf/shark_reference/ && python3.11 -m pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /mlperf/quant_sdxl/
COPY ./quant_sdxl/* /mlperf/quant_sdxl/

######################################################
# Install iree tools
######################################################

SHELL ["/bin/bash", "-c"]

# Disable apt-key parse waring
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# Checkout and build IREE
RUN git clone https://github.com/iree-org/iree.git \
    && cd iree \
    && git submodule update --init

RUN cd iree && python3.11 -m pip install --force-reinstall -r runtime/bindings/python/iree/runtime/build_requirements.txt && \
  python3.11 -m pip uninstall -y numpy && \
  python3.11 -m pip install numpy==1.* && \
  cmake -S . -B build-release \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=`which clang` -DCMAKE_CXX_COMPILER=`which clang++` \
  -DIREE_HAL_DRIVER_CUDA=OFF \
  -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DPython3_EXECUTABLE="$(which python3.11)" \
  -DIREE_TARGET_BACKEND_ROCM=ON \
  -DIREE_HAL_DRIVER_HIP=ON && \
  cmake --build build-release/ --target tools/all && \
  cmake --build build-release/ --target install

# Make IREE tools discoverable in PATH
#ENV PATH=/iree/build-release/tools:$PATH
#ENV PYTHONPATH=/iree/build-release/runtime/bindings/python:/iree/build-release/compiler/bindings/python

ENV PATH=/usr/local/python_packages/iree_runtime/iree/_runtime_libs:/app/vllm/iree/build-release/tools:$PATH
ENV PYTHONPATH=/app/vllm/iree/build-release/runtime/bindings/python:/app/vllm/iree/build-release/compiler/bindings/python
######################################################
# Install shark-ai
######################################################

RUN git clone https://github.com/nod-ai/shark-ai.git -b shared/mlperf-v5.1-sdxl \
  && cd shark-ai \
  && python3.11 -m pip uninstall torch torchvision torchaudio -y \
  && python3.11 -m pip install torch==2.5.1+cpu torchvision --index-url https://download.pytorch.org/whl/cpu \
  && python3.11 -m pip install --pre iree-base-compiler==3.7.0rc20250723 iree-base-runtime==3.7.0rc20250723 iree-turbine==3.7.0rc20250723 -f https://iree.dev/pip-release-links.html \
  && python3.11 -m pip install -r requirements.txt -e sharktank/ -e shortfin/ \
  && pip uninstall iree-base-compiler iree-base-runtime -y

# enable RPD
RUN git clone https://github.com/ROCm/rocmProfileData.git \
  && cd rocmProfileData \
  && apt-get update && ./install.sh \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/models/huggingface/

# enable bandwith test and numa
RUN git clone https://github.com/ROCm/rocm_bandwidth_test --depth 1 rocm_bandwidth_test \
  && cd rocm_bandwidth_test \
  && mkdir build && cd build \
  && cmake -DCMAKE_PREFIX_PATH="/opt/rocm/" .. \
  && make -j && make install \
  && python3.11 -m pip install py-libnuma

# copy the harness code to the docker image
COPY SDXL_inference /mlperf/harness

RUN echo "alias python='python3.11'" >> ~/.bash_aliases

# initialization settings for CPX mode
ENV HSA_USE_SVM=0
ENV HSA_XNACK=0
 
