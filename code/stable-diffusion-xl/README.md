# AMD MI325X SDXL

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
MI300x:
```
sudo rocm-smi --setmemorypartition NPS4
sudo rocm-smi --setcomputepartition CPX 
```
8. run 'rocm-smi' to check your mode

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
IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --td_spec attention_and_matmul_spec_gfx942_MI325_bs32.mlir --model_json sdxl_config_fp8_sched_unet_bs32.json

# Compile the SHARK engines (Server)
IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --td_spec attention_and_matmul_spec_gfx942_MI325.mlir --model_json sdxl_config_fp8_sched_unet_bs2.json

```

## Run scenario and reproduce results

Now that you have successfully compiled artifacts for running SDXL, you may exit the docker container and run:
```bash
./run_docker_nogil.sh
```
This will pick up the previous container's result artifacts automatically.

Once you are in this container, run:

``` bash
# MI325x

# Offline
PYTHON_GIL=0 ./run_scenario_offline_MI325x_cpx.sh

# Server
PYTHON_GIL=0 ./run_scenario_server_MI325x_cpx.sh
```

### Troubleshooting

When you see error
```bash
ValueError: shortfin_iree-src/runtime/src/iree/io/parameter_index.c:237: NOT_FOUND; no parameter found in index with key 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.out_q:rscale'
```
Please execute command
```bash
rm /models/SDXL/official_pytorch/fp16/stable_diffusion_fp16/genfiles/sdxl/stable_diffusion_xl_base_1_0_punet_dataset_i8.irpa
```
Then re-run the harness.py

