FROM rocm/7.0-preview:rocm7.0_preview_ubuntu_22.04_vllm_0.8.5_mi35X_prealpha

ENV DEBIAN_FRONTEND=noninteractive


# ######################################################
# # Set up pyenv and python 3.13t
# ######################################################

SHELL ["/bin/bash", "-c"]

# apt dependencies
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get --fix-broken install -y
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git wget unzip \
    software-properties-common git \
    python3.11 \
    python3.11-dev python3.11-venv \
    build-essential curl cmake ninja-build clang lld vim nano python3.11-dev python3.11-venv gfortran pkg-config libopenblas-dev libssl-dev openssl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
    
   

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
RUN python3.11 -m pip install pybind11 'nanobind<2' pandas

# install loadgen
RUN mkdir /mlperf/ && cd /mlperf && \
    git clone --recursive https://github.com/jinchen62/inference.git -b fix-nogil-build && \
    cd inference/loadgen && \
    mkdir -p /mlperf/harness/ && \
    CFLAGS="-std=c++14" python setup.py install && \
    CFLAGS="-std=c++14" python3.11 setup.py install

RUN mkdir -p /mlperf/shark_reference/ && cp -r /mlperf/inference/text_to_image/* /mlperf/shark_reference/ && cp /mlperf/inference/mlperf.conf /mlperf/shark_reference/
RUN pip install --pre scipy pycocotools
RUN cd /mlperf/shark_reference/ \
    && pip install --no-deps --no-cache-dir -r requirements.txt \
    && python3.11 -m pip install --no-deps --no-cache-dir -r requirements.txt \
    && pip install ftfy timm \
    && python3.11 -m pip install ftfy timm
RUN mkdir -p /mlperf/quant_sdxl/
COPY ./quant_sdxl/* /mlperf/quant_sdxl/

######################################################
# Install iree tools
######################################################

# Disable apt-key parse waring
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

######################################################
# Install shark-ai
######################################################

ENV UNSAFE_PYO3_BUILD_FREE_THREADED=1

RUN git clone https://github.com/nod-ai/shark-ai.git -b shared/mlperf-v5.1-sdxl \
    && cd shark-ai \
    && pip install aiohttp==3.9.5 dataclasses_json \
    && pip install -r requirements-iree-pinned.txt -r sharktank/requirements.txt -r shortfin/requirements-tests-nogil.txt shortfin/ \
    && pip uninstall -y fastapi

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
  && python3.11 -m pip install py-libnuma \
  && python -m pip install py-libnuma

# copy the harness code to the docker image
COPY SDXL_inference /mlperf/harness

RUN echo "alias python='python3.13'" >> ~/.bash_aliases

# initialization settings for CPX mode
ENV HSA_USE_SVM=0
ENV HSA_XNACK=0
