# AMD MI300X SDXL

## Machine setup
[Setup Mi3xx machine and change to CPX/QPX/SPX modes](https://github.com/nod-ai/playbook/blob/main/HOWTO/Setup/mi3xx.md)
## Setup

### Quantization (Optional)
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
```bash
# TODO update this
cd 2024q2-sdxl-mlperf-sprint

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

# Compile the SHARK engines
./precompile_model_shortfin.sh --td_spec attention_and_matmul_spec_gfx942_MI325.mlir --model_json sdxl_config_fp8_sched_unet_all.json
```
The above precompile command will compile inference execution artifacts for all the batch sizes used in the official submission. (1, 2, 16)
If you have any questions or issues with this step, please file an issue on [nod-ai/SHARK-MLPERF](https://github.com/nod-ai/SHARK-MLPERF).

## Reproduce Results
Run the two commands below in an inference container to reproduce full submission results.
The commands will execute performance, accuracy, and compliance tests for Offline and Server scenarios.

NOTE: additional run commands and profiling options are described in [SDXL Inference](./SDXL_inference/README.md) documentation.
``` bash
# MI300x
./run_scenario_offline_MI300x_cpx.sh
./run_scenario_server_MI300x_cpx.sh
```
``` bash
# MI325x
./run_scenario_offline_MI325x_cpx.sh
./run_scenario_server_MI325x_cpx.sh
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