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
RUN pip install --upgrade pip setuptools wheel && \
    pip install pybind11 'nanobind<2' numpy==1.* pandas && \
    pip install hip-python hip-python-as-cuda -i https://test.pypi.org/simple

# Rust requirements
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# install loadgen
RUN mkdir /mlperf && cd /mlperf && \
    git clone --recursive https://github.com/mlcommons/inference && \
    cd inference/loadgen && \
    CFLAGS="-std=c++14" python3 setup.py install

RUN mkdir -p /mlperf/shark_reference/ && cp -r /mlperf/inference/text_to_image/* /mlperf/shark_reference/ && cp /mlperf/inference/mlperf.conf /mlperf/shark_reference/
RUN cd /mlperf/shark_reference/ && pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /mlperf/quant_sdxl/
COPY ./quant_sdxl/* /mlperf/quant_sdxl/

######################################################
# Install iree tools
######################################################
SHELL ["/bin/bash", "-c"]

# Disable apt-key parse waring
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# Checkout and build IREE
RUN git clone https://github.com/iree-org/iree.git -b shared/sdxl_quantized\
  && cd iree \
  && git checkout 610592d2bb2cf7362a6e2a31d011f2e7371780d0 \
  && git submodule update --init --depth=1

# TODO: not sure if this is still needed
RUN pip install --force-reinstall 'nanobind<2'

RUN cd iree && cmake -S . -B build-release \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=`which clang` -DCMAKE_CXX_COMPILER=`which clang++` \
  -DIREE_HAL_DRIVER_CUDA=OFF \
  -DIREE_EXTERNAL_HAL_DRIVERS="rocm" \
  -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DPython3_EXECUTABLE="$(which python3)" && \
  cmake --build build-release/ --target tools/all && \
  cmake --build build-release/ --target install

# Make IREE tools discoverable in PATH
ENV PATH=/iree/build-release/tools:$PATH
ENV PYTHONPATH=/iree/build-release/runtime/bindings/python:/iree/build-release/compiler/bindings/python

######################################################
# Install SHARK-Turbine
######################################################
RUN git clone https://github.com/iree-org/iree-turbine -b sdxl-vae-fix \
  && cd iree-turbine \
  && git checkout 939bacf2be02ce2d2c5b7c77e4a1095e8f86c0aa \
  && pip install -r requirements.txt .

# Install turbine-models, where the SDXL pipeline is implemented.
# This also uninstalls the IREE pip packages from turbine setup, in favor of the
# python bindings from our source build.

RUN git clone https://github.com/nod-ai/sharktank \
  && cd sharktank \
  && git checkout 953c3582a2d117759d12f7d1fdf08016f1f54e3d \
  && pip install -e sharktank

RUN git clone https://github.com/nod-ai/SHARK-Turbine -b bump-punet \
  && cd SHARK-Turbine \
  && git checkout a74d98ecfe4932d1b1bdbcfac0265a966a409ce8 \
  && pip install --pre --upgrade -e models -r models/requirements.txt \
  && pip3 install --force-reinstall --pre torch==2.5.0.dev20240710+rocm6.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1 \
  && pip uninstall iree-compiler iree-runtime -y

RUN git clone https://github.com/nod-ai/sdxl-scripts \
  && cd sdxl-scripts \
  && git checkout b36985bc31febd83d9c335eda70a45a9e5966353

ENV HF_HOME=/models/huggingface/

# enable RPD
RUN git clone https://github.com/ROCm/rocmProfileData.git \
  && cd rocmProfileData \
  && apt-get update && ./install.sh \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

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
