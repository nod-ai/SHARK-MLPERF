# SDXL Inference
The following documentation describes additional options for run execution and profiling of SDXL. This documentation assumes users have completed all steps within [AMD MI300X SDXL](../README.md), up to the point of [Reproduce Results](../README.md#reproduce-results), and starts from a position _within_ the running Inference Docker Container.

## Inference options
You can find the code for our SHARK inference stack which this project is built on here - https://github.com/nod-ai/shark-ai/tree/main/shortfin/python/shortfin_apps/sd

```bash
# Basic run template
python3 harness.py --devices <list the devices> --scenario Offline

# configure to run in CPX mode (64 gpus)
export DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63
```

## Run inference
NOTE: the arguments below are informed by the best current parameter search values available. CAUTION: these parameters will require update with any substantial change in code; not doing so may impose an artificial ceiling on throughput.


```bash
IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --gpu_batch_size 1 --td_spec attention_and_matmul_spec_gfx942_MI325.mlir --model_json sdxl_config_fp8_sched_unet_all.json

# Run Offline scenario (Perf)
ROCR_VISIBLE_DEVICES=$DEVICES HIP_VISIBLE_DEVICES=$DEVICES python3.11 harness.py \
  --devices "$DEVICES" \
  --gpu_batch_size 16 \
  --cores_per_devices 1 \
  --count 51200 \
  --qps 17  \
  --fibers_per_device 1 \
  --test_mode PerformanceOnly \
  --scenario Offline \
  --vae_batch_size 1 \
  --td_spec=attention_and_matmul_spec_gfx942_MI325.mlir \
  --model_json=sdxl_config_fp8_sched_unet.json \
  --logfile_outdir output_offline_perf
```

## Check Accuracy
Accuracy checks require execution of a separate run
```bash
# Run Offline scenario (Accuracy)
ROCR_VISIBLE_DEVICES=$DEVICES HIP_VISIBLE_DEVICES=$DEVICES python3.11 harness.py \
  --devices $DEVICES" \
  --gpu_batch_size 16 \
  --cores_per_devices 1 \
  --fibers_per_device 1 \
  --count 5000 \
  --qps 17  \
  --test_mode AccuracyOnly \
  --scenario Offline \
  --vae_batch_size 1 \
  --td_spec=attention_and_matmul_spec_gfx942_MI325.mlir \
  --model_json=sdxl_config_fp8_sched_unet.json \
  --logfile_outdir output_offline_acc

The `--test_mode AccuracyOnly` run will create a <output_dir>/accuracy.json. (It should be ~30Gb)

Next, create the environment with dependencies
```bash
./setup_accuracy_env.sh
```

Finally, run the following script to generate the scores
```bash
./check_accuracy_scores.sh <output_dir>/mlperf_log_accuracy.json
```

## Tracing
If you want to run with tracing enabled, set this env:
```bash
export ENABLE_TRACING=1

# You can disable it with
export ENABLE_TRACING=0
# or
unset ENABLE_TRACING
```

An example how to use tracing
```bash
# remove any db from previous run (this is the default name)
rm -f trace.rpd

# create an empty db with the proper schema
python3 -m rocpd.schema --create trace.rpd

# Run the harness (this will run 1 GPU Offline, BS=8 for 64 samples)
python3 harness.py --devices "0" --gpu_batch_size 8 --skip_warmup 1 --count 64 --time 1

# Create a tracing json for chrome://tracing
python3 /rocmProfileData/tools/rpd2tracing.py trace.rpd trace_sdxl_shark.json

# You can also use --start XX% --end YY% to check only certain sections
# The output json will be really large. The following command will remove
# the GPU/hip calls
sed -i -E '/(QueueDepth|KernelExecution|FillBuffer|Copy(Device|Host)To(Device|Host)|writeRows|hcc_activity_callback|hip[A-Za-z]+|\"ph\"\:\"s\"|\"bp\"\:\"e\")/d'  trace_sdxl_shark.json

# Note: you can check db with `sqlite3 trace.rpd`
#       an example query: `select * from top limit 10;`
#       (set `.mode columns` to have a better view)
```

## Tuning

The following arguments can be set to tweak better performance.

### devices

The number of GPUs used can be set with `--devices <listed device ids>`.

### gpu batch size

The created pipeline's batch side can be set with `--gpu_batch_size <num>`.

### cores per devices

By default, 1 pipeline is created per GPU. This can be increased with `--cores_per_devices <num>`.
The "core" means a bundle of pipelines here, see the next section.

### multiple pipelines

By default, 1 pipeline is created in a core. But that can be extended with multiple pipelines with more batch sizes. This can be enabled with `--multiple_pipelines <list of batch sizes>`. It must contain `gpu_batch_size`.

This is useful for e.g. Server, when the received sample is less then the max batch size. So instead of padding it, we will pick a smaller pipeline.

### optuner
A dedicated tuner script to find the best numbers for these arguments.
Executing `optuner.py` will run the tunable with `optuna` (pip install optuna).
There is an `objective` function where tunable values can be set.

## Useful args

### qps, count, and time

For a shorter Offline scenario experiment, reduce inputs from 5k to `--count <num>`.

The official min run time is 10 minutes. This can be further reduced via `--time <sec>`

The `qps` defaults to 1.0. Alternative values should be set in the `user.conf` file. For experimentation, this value can be set via `--qps <num>`.

NOTE: `qps` can also be used to increase the sample count (mentioned above); this may necessary to fill the entirety of the 10 minutes run (or the time set).

### saving images

In debugging, the user can use the `--save_images 1` to save the generated images into `harness_result_shark`

### detailed_logdir_name

If `--logfile_outdir` is not set, the default directory name will constructed from run params.
This can be disabled with `--detailed_logdir_name 0`.

### mocking

There is a way to skip SHARK entirely and test harness functionality only. With `--mock_timeout_ms <ms>`, the inference will wait for the specified amount of time, and return random data.

### skip warmup

There is a warmup(x2) with random data at pipeline creating. That can be disabled with `--skip_warmup 1` to speed up testing.
