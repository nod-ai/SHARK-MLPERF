# SDXL Inference
The following documentation describes additional options for run execution and profiling of SDXL. This documentation assumes users have completed all steps within [AMD MI300X SDXL](../README.md), up to the point of [Reproduce Results](../README.md#reproduce-results), and starts from a position _within_ the running Inference Docker Container.

## Inference options

```bash
# Basic run template
python3 harness.py --devices <list the devices> --scenario Offline # e.g. --devices "6,7"
# Use "--save_images" to save the generated image into ./harness_result_shark

# configure to run in CPX mode (64 gpus)
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63

# Alternative example: 2-GPU configuration, using the last two devices
export ROCR_VISIBLE_DEVICES=6,7
export HIP_VISIBLE_DEVICES=0,1

python3 precompile_models.py --batch_sizes "1"
```

## Run inference
NOTE: the arguments below are informed by the best current parameter search values available. CAUTION: these parameters will require update with any substantial change in code; not doing so may impose an artificial ceiling on throughput.


```bash
# Run Offline scenario (Perf)
python3 harness.py \
  --devices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63" \
  --gpu_batch_size 1 \
  --cores_per_devices 2 \
  --qps 12  \
  --scenario Offline \
  --logfile_outdir output_offline_perf

# Run Server scenario, PerformanceOnly, 8 GPUs
python3 harness.py \
  --devices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63" \
  --gpu_batch_size 1 \
  --cores_per_devices 2 \
  --qps 12  \
  --scenario Server \
  --logfile_outdir output_server_perf

# Method to run a smaller test 
# Smaller batch sizes can be specified with `--gpu_batch_size <num>`, providing the model was 
# built with that batch size
# `--infer_timeout` is in seconds
python3 infer.py --devices "0" --save_images 1 --infer_timeout 20 --verbose 1 --gpu_batch_size 4

# There should be 10 images, check them manually
ls harness_result_shark/
```

## Check Accuracy
Accuracy checks require execution of a separate run
```bash
# Run Offline scenario (Accuracy)
python3 harness.py \
  --devices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63" \
  --gpu_batch_size 1 \
  --cores_per_devices 2 \
  --scenario Offline \
  --test_mode AccuracyOnly \
  --logfile_outdir output_offline_acc

# Run Server scenario (Accuracy)
python3 harness.py \
  --devices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63" \
  --gpu_batch_size 1 \
  --cores_per_devices 2 \
  --scenario Server \
  --test_mode AccuracyOnly \
  --logfile_outdir output_server_acc
```
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
