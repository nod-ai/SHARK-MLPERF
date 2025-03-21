# AMD MI300X/MI325X SDXL

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
```bash

# Build the container
docker build --platform linux/amd64 \
  --tag mlperf_rocm_sdxl:micro_sfin_harness \
  --file SDXL_inference/sdxl_harness_rocm_shortfin_from_source_iree.dockerfile .

# Run the container
docker run -it --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /data/mlperf_sdxl/data:/data \
  -v /data/mlperf_sdxl/models:/models \
  -v `pwd`/SDXL_inference/:/mlperf/harness \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63 \
  -w /mlperf/harness \
  mlperf_rocm_sdxl:micro_sfin_harness
```

NOTE: skip this step if quantization methods were executed above; necessary data and models will already be in place
```bash
# Download data and base weights
./download_data.sh
./download_model.sh
```

Preprocess data and prepare for run execution
```bash
python3.11 preprocess_data.py

# Process local checkpoint generated from quantization docker
python3.11 process_quant.py
```

## Reproduce Results
Run the commands below in an inference container to reproduce full submission results.
Each submission run command is preceded by a specific precompilation command. If you encounter issues with the precompilation, please file an issue at [shark-ai/issues](https://github.com/nod-ai/shark-ai/issues)
The commands will execute performance, accuracy, and compliance tests for Offline and Server scenarios.

NOTE: additional run commands and profiling options are described in [SDXL Inference](./SDXL_inference/README.md) documentation.
``` bash
# MI300x

# Compile the SHARK engines (Offline)
IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --td_spec attention_and_matmul_spec_gfx942_MI325.mlir --model_json sdxl_config_fp8_sched_unet_bs2.json
# Run the offline scenario.
./run_scenario_offline_MI300x_cpx.sh

# Compile the SHARK engines (Server)
IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --td_spec attention_and_matmul_spec_gfx942_MI325.mlir --model_json sdxl_config_fp8_sched_unet_bs1.json
# Run the server scenario.
./run_scenario_server_MI300x_cpx.sh
```
``` bash
# MI325x

# Compile the SHARK engines (Offline)
IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --td_spec attention_and_matmul_spec_gfx942_MI325.mlir --model_json sdxl_config_fp8_sched_unet_bs16.json
# Run the offline scenario.
./run_scenario_offline_MI300x_cpx.sh

# Compile the SHARK engines (Server)
IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --td_spec attention_and_matmul_spec_gfx942_MI325.mlir --model_json sdxl_config_fp8_sched_unet_bs2.json
# Run the server scenario.
./run_scenario_server_MI300x_cpx.sh
```

### Execute individual scenario tests
Alternatively, the scenario and test mode tests can be run separately.  To generate results for the Offline scenario only, run the command below in an inference container 
``` bash
# CPX: --devices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63"
# QPX: --devices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
# SPX: --devices "0,1,2,3,4,5,6,7"
python3.11 harness.py \
  --devices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63" \
  --gpu_batch_size 8 \
  --cores_per_devices 1 \
  --scenario Offline \
  --logfile_outdir output_offline \
  --test_mode SubmissionRun
```

To generate results for the Server scenario only, run the command below in an inference container 
``` bash
python3.11 harness.py \
  --devices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63" \
  --gpu_batch_size 8 \
  --cores_per_devices 1 \
  --scenario Server \
  --logfile_outdir output_server \
  --test_mode SubmissionRun
```
Output logs will write out to the location where the script was executed, with a directory name (or path) specified by `--logfile_outdir`.

Runs executed with `--test_mode SubmissionRun` will execute both `PerformanceOnly` and `AccuracyOnly` runs. Please note that either of these options can be executed independently; `--test_mode PerformanceOnly` or `--test_mode AccuracyOnly`.

Processing accuracy results requires additional steps. A run with `--test_mode AccuracyOnly` (or `SubmissionRun`) will create a `<logfile_outdir>/mlperf_log_accuracy.json` file (it should be ~30 GB).

To check accuracy, create an environment with the following
```bash
./setup_accuracy_env.sh
```

Finally, run the following script to generate accuracy scores
```bash
./check_accuracy_scores.sh <output_dir>/mlperf_log_accuracy.json
```
