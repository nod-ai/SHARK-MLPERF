# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import logging
from pathlib import Path
import multiprocessing as mp
import sys
import pprint
import gc
import json

import logging

from dataset import Dataset
from utilities import rpd_trace
from sdxl_backend import SDXLShortfinService

try:
    import mlperf_loadgen as lg
except:
    logging.warning("Loadgen Python bindings are not installed. Install it!")
    raise RuntimeError("Missing loadgen lib")

scenario_map = {
    "Offline": lg.TestScenario.Offline,
    "SingleStream": lg.TestScenario.SingleStream,
    "Server": lg.TestScenario.Server,
}
test_mode_map = {
    "PerformanceOnly": lg.TestMode.PerformanceOnly,
    "AccuracyOnly": lg.TestMode.AccuracyOnly,
    "SubmissionRun": lg.TestMode.SubmissionRun,
}
log_mode_map = {
    "AsyncPoll": lg.LoggingMode.AsyncPoll,
    "EndOfTestOnly": lg.LoggingMode.EndOfTestOnly,
    "Synchronous": lg.LoggingMode.Synchronous,
}


def get_args():

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("true", "1"):
            return True
        elif v.lower() in ("false", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    # Test args
    parser.add_argument(
        "--scenario",
        choices=["Offline", "Server"],
        default="Offline",
    )
    parser.add_argument(
        "--test_mode",
        choices=["PerformanceOnly", "AccuracyOnly", "SubmissionRun"],
        default="PerformanceOnly",
    )

    # QSL args
    parser.add_argument(
        "--tensor_path",
        type=str,
        default="/data/coco2014-tokenized-sdxl/5k_dataset_final/",
    )
    parser.add_argument("--performance_sample_count", type=int, default=5000)

    # SUT args
    parser.add_argument(
        "--devices", type=str, required=True, help="Comma-separated numbered devices"
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default="/models/SDXL/official_pytorch/fp16/stable_diffusion_fp16",
    )
    parser.add_argument(
        "--gpu_batch_size",
        type=int,
        default=1,
        help="Max Batch size to use for all devices and engines",
    )
    parser.add_argument(
        "--verbose", type=str2bool, default=False, help="SUT verbose logging"
    )
    parser.add_argument(
        "--batcher_limit",
        type=float,
        default=3,
        help="SDXL harness server scenario batcher time out threashold in seconds",
    )
    parser.add_argument(
        "--fibers_per_device",
        type=int,
        default=2,
        help="Configure the number of fibers for each device",
    )
    parser.add_argument(
        "--workers_per_device",
        type=int,
        default=1,
        help="Configure the number of workers for each device",
    )
    parser.add_argument(
        "--num_sample_loops",
        type=int,
        default=2,
        help="Number of sample_loop threads to run at once",
    )
    parser.add_argument(
        "--vae_batch_size",
        type=int,
        help="If set, changes the VAE batch size, otherwise matches gpu_batch_size",
    )
    parser.add_argument(
        "--model_json",
        type=str,
        default="sdxl_config_fp8_sched_unet.json",
        help="The config file for the model",
    )
    parser.add_argument(
        "--td_spec",
        type=str,
        default=None,
        help="The tuning specfile to apply during artifact generation",
    )
    parser.add_argument(
        "--force_export",
        type=str2bool,
        default=False,
        help="Force re-export of artifacts"
    )
    parser.add_argument(
        "--cores_per_devices",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--multiple_pipelines",
        type=str,
        default=None,
        help="Comma-separated batch sizes",
    )
    parser.add_argument("--use_response_pipes", type=str2bool, default=False)
    parser.add_argument("--send_latents_once", type=str2bool, default=False)
    parser.add_argument("--skip_warmup", type=str2bool, default=False)
    parser.add_argument(
        "--log_sample_get",
        type=str2bool,
        default=False,
        help="Log time taken for each `sample_queue.get` in a file corresponding to the device_id for the process.",
    )

    # Config args
    parser.add_argument(
        "--mlperf_conf_path",
        help="Path to mlperf.conf",
        default="/mlperf/inference/mlperf.conf",
    )
    parser.add_argument(
        "--user_conf_path", help="Path to user.conf", default="./user.conf"
    )
    parser.add_argument("--audit_conf_path", help="Path to audit.conf", default=None)

    # Log args
    parser.add_argument(
        "--log_mode", type=str, default="AsyncPoll", help="Logging mode for Loadgen"
    )
    parser.add_argument(
        "--log_mode_async_poll_interval_ms",
        type=int,
        default=1000,
        help="Specify the poll interval for asynchronous logging",
    )
    parser.add_argument(
        "--logfile_outdir",
        type=str,
        default="output",
        help="Specify the existing output directory for the LoadGen logs",
    )
    parser.add_argument(
        "--logfile_prefix",
        type=str,
        default="mlperf_log_",
        help="Specify the filename prefix for the LoadGen log files",
    )
    parser.add_argument(
        "--logfile_suffix",
        type=str,
        default="",
        help="Specify the filename suffix for the LoadGen log files",
    )
    parser.add_argument(
        "--detailed_logdir_name",
        type=str2bool,
        default=True,
        help="Make the default log names more verbose. Only applied if every log* name is default.",
    )

    # Debug
    parser.add_argument(
        "--save_images",
        type=str2bool,
        default=False,
        help="Save generated images",
    )
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument("--debug", type=str2bool, default=False, help="Reduce number of devices to 1 and synchronously run one sample_loop to debug harness breakages")
    parser.add_argument("--qps", type=str, default="")
    parser.add_argument(
        "--mock_timeout_ms",
        type=int,
        default=None,
        help="Mock test running without GPU, delay the result with the provided time in milliseconds",
    )
    parser.add_argument(
        "--enable_numa",
        type=str2bool,
        default=True,
        help="Find best matching CPU-GPU pairs based on the NUMA.",
    )
    parser.add_argument(
        "--enable_batcher",
        type=str2bool,
        default=False,
        help="Batch requests in server mode for gpu_batch_size>1.",
    )

    parser.add_argument(
        "--shark_engine",
        choices=["iree_python_api", "micro_shortfin", "shortfin"],
        default="iree_python_api",
    )

    args = parser.parse_args()
    return args


@rpd_trace()
def mlperf(args):
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    command = "python3.11 " + " ".join(sys.argv)
    logging.info(f"{'#'*180}\nRunning {command}\n{'#'*190}")
    logging.debug(f"\n{pprint.pformat(vars(args))}\n")

    devices = [int(x) for x in args.devices.split(",")]

    multiple_pipelines = (
        [int(x) for x in args.multiple_pipelines.split(",")]
        if args.multiple_pipelines
        else [args.gpu_batch_size]
    )
    assert (
        args.gpu_batch_size in multiple_pipelines
    ), f"{args.gpu_batch_size=} not in {multiple_pipelines=}"

    test_settings = lg.TestSettings()
    test_settings.scenario = scenario_map[args.scenario]
    test_settings.mode = test_mode_map[args.test_mode]

    test_settings.FromConfig(
        args.mlperf_conf_path, "stable-diffusion-xl", args.scenario
    )
    #test_settings.FromConfig(args.user_conf_path, "stable-diffusion-xl", args.scenario)
    if args.audit_conf_path:
        test_settings.FromConfig(
            args.audit_conf_path, "stable-diffusion-xl", args.scenario
        )
    test_settings.server_coalesce_queries = True
    if args.count:
        logging.warning(f"Override count with {args.count}")
        test_settings.min_query_count = args.count
        test_settings.max_query_count = args.count

    if args.qps:
        qps = float(args.qps)
        logging.warning(f"Override qps with {qps}")
        test_settings.server_target_qps = qps
        test_settings.offline_expected_qps = qps
    else:
        args.qps = (
            test_settings.server_target_qps
            if args.scenario == "Server"
            else test_settings.offline_expected_qps
        )

    if args.time:
        time = args.time * 1000
        # override the time we want to run
        logging.warning(f"Override min duration with {time} ms")
        test_settings.min_duration_ms = time
        test_settings.max_duration_ms = time

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.copy_summary_to_stdout = True

    if (
        args.detailed_logdir_name
        and args.logfile_outdir == "output"
        and args.logfile_prefix == "mlperf_log_"
        and args.logfile_suffix == ""
    ):
        scenario = args.scenario.lower()
        mode = args.test_mode[:4].lower()
        dev_count = f"x{len(devices)}"
        batch = f"bs{args.gpu_batch_size}"
        cores = f"cpd{args.cores_per_devices}"
        qps = f"qps{args.qps}"
        mp = "mp-" + "-".join(map(str, multiple_pipelines))
        args.logfile_outdir = "_".join(
            ["output", scenario, mode, dev_count, batch, cores, qps, mp]
        )
    with open(args.model_json, "r") as json_file:
        model_json_string = json.load(json_file)

    model_json_string["batch_sizes"]["clip"] = [args.gpu_batch_size]
    model_json_string["batch_sizes"]["scheduled_unet"] = [args.gpu_batch_size]
    model_json_string["batch_sizes"]["vae"] = (
        [args.gpu_batch_size] if args.vae_batch_size is None else [args.vae_batch_size]
    )

    with open(f"{args.model_json}", "w") as json_file:
        json.dump(model_json_string, json_file, indent=4)

    log_output_settings.outdir = args.logfile_outdir
    log_output_settings.prefix = args.logfile_prefix
    log_output_settings.suffix = args.logfile_suffix
    Path(args.logfile_outdir).mkdir(parents=True, exist_ok=True)

    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.log_mode = log_mode_map[args.log_mode]
    log_settings.log_mode_async_poll_interval_ms = args.log_mode_async_poll_interval_ms

    server = SDXLShortfinService(
        devices=[int(x) for x in args.devices.split(",")],
        model_weights=args.model_weights,
        dataset=Dataset(args.tensor_path),
        gpu_batch_size=args.gpu_batch_size,
        verbose=args.verbose,
        enable_batcher=args.enable_batcher,
        batch_timeout_threashold=3 if args.enable_batcher else -1,
        cores_per_devices=args.cores_per_devices,
        save_images=args.save_images,
        performance_sample_count=args.performance_sample_count,
        skip_warmup=args.skip_warmup,
        skip_complete=False,
        use_response_pipes=False,
        send_latents_once=False,
        mock_timeout=None,
        enable_numa=True,
        workers_per_device=args.workers_per_device,
        fibers_per_device=args.fibers_per_device,
        isolation="per_fiber",
        num_sample_loops=args.num_sample_loops,
        model_json=args.model_json,
        td_spec=args.td_spec,
        log_sample_get=args.log_sample_get,
        debug=args.debug,
        force_export=args.force_export,
    )

    with open(Path(args.logfile_outdir, "command.txt"), "w") as f:
        f.writelines(command)

    logging.info("Start Test!")
    lg.StartTestWithLogSettings(server.sut, server.qsl, test_settings, log_settings)
    server.finish_test()
    logging.info("Test Done!")

    logging.info("Destroying SUT...")
    lg.DestroySUT(server.sut)

    logging.info("Destroying QSL...")
    lg.DestroyQSL(server.qsl)
    logging.info(f"Check {args.logfile_outdir}")


def main():
    gc.disable()
    # Needed for tracing
    mp.set_start_method("spawn")
    args = get_args()
    mlperf(args)


if __name__ == "__main__":
    main()
