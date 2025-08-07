# AMD MI325X SDXL

## Machine setup
1. Install latest public build of ROCM:

Follow the official instructions for installing rocm6.3.3: [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/amdgpu-installer-index.html) or [package manager](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager-index.html)

2. If you run into any issue, follow the official instructions for uninstalling in the above links, and try again.

3. Run following to see all the 64 devices in CPX mode
```shell
echo 'blacklist ast' | sudo tee /etc/modprobe.d/blacklist-ast.conf
sudo update-initramfs -u -k $(uname -r)
```
4. run: 'sudo reboot'
5. Once machine comes back up, you can run the following to make sure you have the right compute/memory partitioning and power/perf setup via rocm-smi:
MI325x:
```shell
sudo rocm-smi --setmemorypartition NPS1
sudo rocm-smi --setcomputepartition SPX
sudo rocm-smi --setperfdeterminism 2100
sudo rocm-smi --setcomputepartition CPX
```
6. run 'rocm-smi' to check your mode, ensuring CPX and NPS1 show in rocm-smi output.

7. power settings -- run exactly:
```shell
echo 3 | sudo tee /proc/sys/vm/drop_caches && \
sudo cpupower idle-set -d 2 && \
sudo cpupower frequency-set -g performance && \
echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog && \
echo 0 | sudo tee /proc/sys/kernel/numa_balancing && \
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space && \
echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/enabled && \
echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
```

## Submission Setup

### Quantization
NOTE: Running quantization will require 2 or more hours (on GPU) to complete, and much longer on CPU. As a matter of convenience, the weights that result from this quantization are also available from [huggingface](https://huggingface.co/amd-shark/sdxl-quant-models). To skip quantization and work from downloaded weights, please jump to the [AMD MLPerf Inference Docker Container Setup](#amd-mlperf-inference-docker-container-setup) section.

Create the container that will be used for dataset preparation and model quantization
```bash
cd quant_sdxl

# Build quantization container
docker build --tag  mlperf_rocm_sdxl:quant --file Dockerfile .
```

Run the quantization container; prepare data and models
```bash
docker run -it --network=host --device=/dev/kfd --device=/dev/dri   --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /data/mlperf_sdxl/data:/data \
  -v /data/mlperf_sdxl/models:/models \
  -v `pwd`/:/mlperf/harness \
  -w /mlperf/harness \
  mlperf_rocm_sdxl:quant

# Download data and base weights
./download_data.sh
./download_model.sh
```

Execute quantization

NOTE: additional quantization options are described in [SDXL Quantization](./quant_sdxl/README.md) documentation.
```bash
# Execute quantization
./run_quant.sh

# Exit the quantization container
exit
```

### AMD MLPerf Inference Docker Container Setup

From `code/stable-diffusion-xl/`:

Use the provided scripts:
```bash
./build_docker.sh
./build_docker_nogil.sh
./run_docker.sh
```

Preprocess data and prepare for run execution
```bash
python3.11 preprocess_data.py

# Process local checkpoint generated from quantization docker
python3.11 process_quant.py
```

## Precompile models
Run the commands below in the first container to reproduce full submission results.
Each submission run requires a certain set of precompiled model artifacts in the correct format.
The precompile_model_shortfin.sh script uses shark-ai tooling to export and compile SDXL for MI325x.

For best results, prepare the offline and server mode artifacts at once.

``` bash
# MI325x

# Compile the SHARK engines (Offline)
IREE_BUILD_DIR=/iree/build-offline/ IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --td_spec attention_and_matmul_spec_gfx942_MI325_bs32.mlir --model_json sdxl_config_fp8_sched_unet_bs32.json

# Compile the SHARK engines (Server)
IREE_BUILD_DIR=/iree/build-server/ IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --td_spec attention_and_matmul_spec_gfx942_MI325.mlir --model_json sdxl_config_fp8_sched_unet_bs2.json

```

## Run scenarios and reproduce results

### Server Mode

Now that you have successfully compiled artifacts for running SDXL, you can run the Server scenario in the current docker container:

```bash
./run_scenario_server_MI325x_cpx.sh
```

This will run:
 - Performance Run
 - Accuracy Run
 - Accuracy Validation
 - Compliance Test 01 and 04

for the server scenario, likewise for the offline script which we will run in a separate docker container as follows:

### Offline Mode (Using GIL-free python)
```bash
exit
./run_docker_nogil.sh
```
This will pick up the previous container's compiled artifacts automatically.

Once you are in this container, run:

``` bash
PYTHON_GIL=0 ./run_scenario_offline_MI325x_cpx.sh
```

By default, the scripts will save your submission results to `code/stable-diffusion-xl/SDXL_inference/Submission/...` including accuracy and compliance test results.

The offline scenario uses a large sample count and can take up to 3 hours to complete all scenario tests. Server mode is shorter and should take under 45 minutes.

