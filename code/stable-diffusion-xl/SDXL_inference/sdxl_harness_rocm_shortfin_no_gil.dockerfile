FROM rocm/dev-ubuntu-22.04:6.1.2

ENV DEBIAN_FRONTEND=noninteractive

# ######################################################
# # Set up pyenv and python 3.13t
# ######################################################

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    libgdbm-dev libgdbm-compat-dev libdb-dev
RUN git clone https://github.com/pyenv/pyenv.git /root/.pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}"
RUN source "$PYENV_ROOT/completions/pyenv.bash" && \
    pyenv install 3.13t-dev && \
    pyenv global 3.13t-dev && \
    pyenv rehash

# Rust requirements
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . "/root/.cargo/env"
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3
RUN git clone https://github.com/vinayakdsci/tokenizers.git -b fix-cp-nogil-build-failures \
    && cd tokenizers/bindings/python \
    && pip install -e .

# ######################################################
# # Install MLPerf+Shark reference implementation
# ######################################################

# apt dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 wget unzip software-properties-common \
    curl cmake ninja-build clang lld vim nano gfortran pkg-config libopenblas-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install pybind11 'nanobind<2' pandas

# install loadgen
RUN mkdir /mlperf/ && cd /mlperf && \
    git clone --recursive https://github.com/jinchen62/inference.git -b fix-nogil-build && \
    cd inference/loadgen && \
    mkdir -p /mlperf/harness/ && \
    CFLAGS="-std=c++14" python setup.py install

RUN mkdir -p /mlperf/shark_reference/ && cp -r /mlperf/inference/text_to_image/* /mlperf/shark_reference/ && cp /mlperf/inference/mlperf.conf /mlperf/shark_reference/
RUN pip install --pre scipy pycocotools
RUN cd /mlperf/shark_reference/ \
    && pip install --no-deps --no-cache-dir -r requirements.txt \
    && pip install ftfy timm
RUN mkdir -p /mlperf/quant_sdxl/
COPY ./quant_sdxl/* /mlperf/quant_sdxl/

######################################################
# Install iree tools
######################################################

# Disable apt-key parse waring
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# Checkout and build IREE
RUN git clone https://github.com/iree-org/iree.git \
    && cd iree \
    && git submodule update --init

RUN cd iree && pip install --force-reinstall -r runtime/bindings/python/iree/runtime/build_requirements.txt && \
    cmake -S . -B build-release \
    -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=`which clang` -DCMAKE_CXX_COMPILER=`which clang++` \
    -DIREE_HAL_DRIVER_CUDA=OFF \
    -DIREE_BUILD_PYTHON_BINDINGS=ON \
    -DPython3_EXECUTABLE="$(which python)" \
    -DIREE_TARGET_BACKEND_ROCM=ON \
    -DIREE_HAL_DRIVER_HIP=ON && \
    cmake --build build-release/ --target tools/all && \
    cmake --build build-release/ --target install

# Make IREE tools discoverable in PATH
ENV PATH=/iree/build-release/tools:$PATH
ENV PYTHONPATH=/iree/build-release/runtime/bindings/python:/iree/build-release/compiler/bindings/python

######################################################
# Install shark-ai
######################################################

ENV UNSAFE_PYO3_BUILD_FREE_THREADED=1

RUN git clone https://github.com/nod-ai/shark-ai.git -b sdxl-5.1-rebase \
    && cd shark-ai \
    && pip install aiohttp==3.9.5 \
    && pip install -r requirements.txt -r requirements-iree-pinned.txt -e sharktank/ -e shortfin/ \
    && pip uninstall -y iree-base-compiler iree-base-runtime fastapi

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
    && cmake -DCMAKE_MODULE_PATH="/rocm_bandwidth_test/cmake_modules" -DCMAKE_PREFIX_PATH="/opt/rocm/" .. \
    && make -j && make install \
    && pip install py-libnuma

# copy the harness code to the docker image
COPY SDXL_inference /mlperf/harness

# initialization settings for CPX mode
ENV HSA_USE_SVM=0
ENV HSA_XNACK=0
