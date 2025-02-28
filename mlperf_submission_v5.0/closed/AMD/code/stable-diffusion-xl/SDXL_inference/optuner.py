import argparse
import logging
import optuna
import multiprocessing as mp
import sys
import shutil

from harness import mlperf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1"):
        return True
    elif v.lower() in ("false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--num_trials", type=int, required=True)

    parser.add_argument(
        "--scenario", type=str, choices=["Offline", "Server"], default="Offline"
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        choices=["PerformanceOnly", "AccuracyOnly", "SubmissionRun"],
        default="PerformanceOnly",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default="/models/SDXL/official_pytorch/fp16/stable_diffusion_fp16/",
    )
    parser.add_argument(
        "--tensor_path",
        type=str,
        default="/data/coco2014-tokenized-sdxl/5k_dataset_final/",
    )
    parser.add_argument("--performance_sample_count", type=int, default=5000)
    parser.add_argument("--mock_timeout_ms", type=int, default=None)
    parser.add_argument("--devices", type=str, required=True)
    parser.add_argument("--verbose", type=str2bool, default=False)
    parser.add_argument("--save_images", type=str2bool, default=False)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument("--skip_warmup", type=str2bool, default=False)
    parser.add_argument("--detailed_logdir_name", type=str2bool, default=False)

    # laodgen settings
    parser.add_argument(
        "--mlperf_conf_path", type=str, default="/mlperf/inference/mlperf.conf"
    )
    parser.add_argument("--user_conf_path", type=str, default="./user.conf")
    parser.add_argument("--audit_conf_path", type=str, default="audit.conf")
    parser.add_argument("--log_mode", type=str, default="AsyncPoll")
    parser.add_argument(
        "--log_mode_async_poll_interval_ms",
        type=int,
        default=1000,
        help="Specify the poll interval for asynchrounous logging",
    )
    parser.add_argument(
        "--logfile_outdir",
        type=str,
        default="output_optuna",
        help="Specify the existing output directory for the LoadGen logs",
    )
    parser.add_argument(
        "--logfile_prefix",
        type=str,
        default="",
        help="Specify the filename prefix for the LoadGen log files",
    )
    parser.add_argument(
        "--logfile_suffix",
        type=str,
        default="",
        help="Specify the filename suffix for the LoadGen log files",
    )

    # Tunable arguments for hyper parameter tuning
    parser.add_argument("--gpu_batch_size", type=int, default=1)
    parser.add_argument("--cores_per_devices", type=int, default=1)
    parser.add_argument("--use_response_pipes", type=str2bool, default=False)
    parser.add_argument("--qps", type=str, default="")
    parser.add_argument("--batcher_limit", type=float, default=3)
    parser.add_argument("--multiple_pipelines", type=str, default=None)

    args = parser.parse_args()
    return args


def rewrite_target_qps(user_conf_file, target_qps, scenario):
    configs = []
    with open(user_conf_file, "r") as file:
        configs = file.readlines()

    for i in range(len(configs)):
        if f"{scenario}.target_qps" in configs[i]:
            configs[i] = f"*.{scenario}.target_qps = {target_qps}\n"

    with open(user_conf_file, "w") as file:
        file.writelines(configs)


def collect_results_from_log(log_file):
    result = 0.0
    valid = ""
    with open(log_file) as file:
        for line in file.readlines():
            # S|s
            if "amples per second" in line:
                result = float(line.split(":")[-1])
            if "Result is" in line:
                valid = line.split(":")[-1]
                break
    return result, valid


def write_gpu_stats(scenario, result, valid, stats_file, args):
    with open(stats_file, "a") as file:
        file.write(f"{scenario=}, {result=}, {valid=}, {args=}\n")


def copy_and_clear_log(args, trial_number):
    new_file = f"{args.logfile_outdir}/{args.study_name}-trial{trial_number}.log"
    shutil.copy(f"{args.logfile_outdir}/summary.txt", new_file)


def objective(trial):
    args = get_args()
    args.gpu_batch_size = trial.suggest_int("gpu_batch_size", 1, 8, step=1)
    args.cores_per_devices = trial.suggest_int("cores_per_devices", 1, 4, step=1)
    # TODO fix this
    # args.multiple_pipelines = trial.suggest_categorical('multiple_pipelines', [None, ','.join([str(i) for i in range(1, args.gpu_batch_size+1)])])
    args.qps = trial.suggest_float("qps", 8.3, 9.0, step=0.1)

    mlperf(args)

    result, valid = collect_results_from_log(f"{args.logfile_outdir}/summary.txt")
    copy_and_clear_log(args, trial.number)
    log.info(f"Samples per sec: {result}")
    write_gpu_stats(
        args.scenario,
        result,
        valid,
        f"{args.logfile_outdir}/{args.study_name}-stats.log",
        args,
    )
    return result - (args.qps if "INVALID" in valid else 0)


def main():
    mp.set_start_method("spawn")
    args = get_args()
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = args.study_name  # Unique identifier of the study.
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.num_trials)
    log.info(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
