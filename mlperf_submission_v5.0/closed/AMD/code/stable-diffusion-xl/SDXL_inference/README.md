# SDXL Inference
The following documentation describes additional options for run execution and profiling of SDXL. This documentation assumes users have completed all steps within [AMD MI300X SDXL](../README.md), up to the point of [Reproduce Results](../README.md#reproduce-results), and starts from a position _within_ the running Inference Docker Container.

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
