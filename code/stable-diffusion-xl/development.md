# AMD SDXL mlperf harness development documentation

This is currently validated for MI325x.

## Machine setup
1. Install latest public build of ROCM:

Follow the official instructions for installing rocm6.3.3: [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/amdgpu-installer-index.html) or [package manager](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager-index.html)

2. If you run into any issue, follow the official instructions for uninstalling in the above links, and try again.

3. Run following to see all the 64 devices in CPX mode
```
echo 'blacklist ast' | sudo tee /etc/modprobe.d/blacklist-ast.conf
sudo update-initramfs -u -k $(uname -r)
```
4. run: 'sudo reboot'
5. Once machine comes back up, you can run the following to make sure you have the right compute/memory partitioning:
MI325x:
```
sudo rocm-smi --setmemorypartition NPS1
sudo rocm-smi --setcomputepartition CPX 
```
8. run 'rocm-smi' to check your mode

## Submission Setup

Clone this repository with the staging-v5.1 branch:
```bash
git clone https://github.com/nod-ai/SHARK-MLPERF -b dev
```

### Quantization

#### Skip this if you don't want to test quantization from scratch.

Create the container that will be used for dataset preparation and model quantization
```bash
cd SHARK-MLPERF/code/stable-diffusion-xl/quant_sdxl

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

From `SHARK-MLPERF/code/stable-diffusion-xl/`:

Convenience shellscripts are provided in this directory. To precompile models, run `./build_docker.sh`, `./run_docker.sh`, and then follow the steps below to precompile.
If you are just running accuracy validation, you can use this docker container for the rest of your needs.

For best performance, once precompilation has finished, exit the container and run `./build_docker_nogil.sh` and `./run_docker_nogil.sh`, then run the scenario shellscript as shown below.

On MI355, use `./build_docker_mi355.sh` and `./run_docker_mi355.sh`.

NOTE: skip this step if quantization methods were executed above; necessary data and models will already be in place
```bash
# Download data and base weights
./download_data.sh
./download_model.sh
```

Preprocess data and prepare for run execution
```bash
python3.11 preprocess_data.py

# Process local checkpoint generated from quantization docker, if you generated quantized artifacts. Otherwise skip this step.
python3.11 process_quant.py
```

## Reproduce Results
Run the commands below in an inference container to reproduce full submission results.
Each submission run command is preceded by a specific precompilation command. If you encounter issues with the precompilation, please file an issue at [shark-ai/issues](https://github.com/nod-ai/shark-ai/issues)
The commands will execute performance, accuracy, and compliance tests for Offline and Server scenarios.

Best results are achieved through use of free-threaded python (3.13t) where GIL is optional
### NOTE (GIL-FREE): 
> Precompilation should be performed in the python3.11 container built via ../build_docker.sh, and the scenario can then be executed with the precompiled artifacts by the nogil docker build (../build_docker_nogil.sh) with the shellscript commands below. The environment variable has no effect if you are not using python3.13t, so we include it in the commands below as a convenience.

### NOTE (GIL-FREE):
> The precompile command and the run_scenario.sh should be executed in separate docker containers for best performance. Once you have finished precompiling, the artifacts will be saved to your disk, and picked up in the python3.13t docker container.

``` bash
# MI325x

# Compile the SHARK engines (Offline)
IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --td_spec attention_and_matmul_spec_gfx942_MI325_bs32_mod.mlir --model_json sdxl_config_fp8_sched_unet_bs32.json
# Run the offline scenario.

PYTHON_GIL=0 ./run_scenario_offline_MI325x_cpx.sh

# Compile the SHARK engines (Server)
IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --td_spec attention_and_matmul_spec_gfx942_MI325_mod.mlir --model_json sdxl_config_fp8_sched_unet_bs2.json
# Run the server scenario.

PYTHON_GIL=0 ./run_scenario_server_MI325x_cpx.sh
```
``` bash
# MI355:
# Requires a different (rocm7) docker image. See instructions above.

# Compile the SHARK engines (Offline)
IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --model_json sdxl_config_fp8_ocp_sched_unet_bs32.json --target gfx950 --flag_file "sdxl_flagfile_gfx950.txt" --td_spec "" --shortfin_dir /app/vllm/shark-ai/shortfin/python/shortfin_apps/sd
# Run the offline scenario.
PYTHON_GIL=0 ./run_scenario_offline_MI355.sh

# Compile the SHARK engines (Server)
IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --model_json sdxl_config_fp8_ocp_sched_unet_bs2.json --target gfx950 --flag_file "sdxl_flagfile_gfx950.txt" --td_spec "" --shortfin_dir /app/vllm/shark-ai/shortfin/python/shortfin_apps/sd
# Run the server scenario.
PYTHON_GIL=0 ./run_scenario_server_MI355.sh
```


## Test Accuracy Only
We provide a script for checking accuracy only, for validating codegen / IR changes to the SDXL models. It is essentially a pared-down version of the full scenario runners, designed for developer use.

To use this script, follow all instructions as normal up until the run_scenario_X_X.sh execution, considering the following:

Most likely, you will want to change the IREE or shark-ai checkout to test your changes. To do so, you may either edit the dockerfile at SHARK-MLPERF/code/stable-diffusion-xl/SDXL_inference/sdxl_harness_rocm_shortfin_from_source_iree.dockerfile and change the checkout for your build, OR you may build the images as-is and manually checkout the candidates you want to test e.g.:
```
./build_docker.sh
./run_docker.sh

cd /shark-ai/
git checkout my_branch_or_commit
pip install -e sharktank/ -e shortfin/

cd /iree/
git checkout my_branch_or_commit
git submodule update
cmake --build ./build-release/

cd /mlperf/harness/

<your_precompile_model_shortfin.sh_command> --force_export True (make sure you aren't caching old artifacts)

./run_accuracy_mi325x.sh
```
This script assumes you are running Batch size 32 with 64 CPX devices. You may edit the file to change the parameters, but changing the batch size will require the respective artifacts to have been generated ahead of time.
