import argparse
import logging
import time
import multiprocessing as mp
import gc

import logging

from dataset import Dataset
from utilities import rpd_trace
from sdxl_backend import SDXLServer

try:
    import mlperf_loadgen as lg
except:
    logging.warning("Loadgen Python bindings are not installed. Install it!")
    raise RuntimeError("Missing loadgen lib")


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
        choices=["Offline", "Server", "SingleStream"],
        default="Offline",
    )
    # QSL args
    parser.add_argument(
        "--tensor_path",
        type=str,
        default="/data/coco2014-tokenized-sdxl/5k_dataset_final/",
    )
    # SUT args
    parser.add_argument(
        "--devices", type=str, required=True, help="Comma-separated numbered devices"
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default="/models/SDXL/official_pytorch/fp16/stable_diffusion_fp16/",
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
        "--cores_per_devices",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--infer_timeout",
        type=int,
        default=20,
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
    # Debug
    parser.add_argument(
        "--save_images",
        type=str2bool,
        default=False,
        help="Save generated images",
    )

    parser.add_argument(
        "--mock_timeout_ms",
        type=int,
        default=None,
        help="Mock test running without GPU, delay the result with the provided time in milliseconds",
    )

    args = parser.parse_args()
    return args


@rpd_trace()
def infer(args):
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    multiple_pipelines = (
        [int(x) for x in args.multiple_pipelines.split(",")]
        if args.multiple_pipelines
        else None
    )
    assert (
        multiple_pipelines is None or args.gpu_batch_size in multiple_pipelines
    ), f"{args.gpu_batch_size=} not in {multiple_pipelines=}"
    server = SDXLServer(
        devices=[int(x) for x in args.devices.split(",")],
        model_weights=args.model_weights,
        dataset=Dataset(args.tensor_path),
        gpu_batch_size=args.gpu_batch_size,
        verbose=args.verbose,
        enable_batcher=(args.scenario == "Server"),
        batch_timeout_threashold=args.batcher_limit,
        cores_per_devices=args.cores_per_devices,
        save_images=args.save_images,
        multiple_pipelines=multiple_pipelines,
        skip_complete=True,
        skip_warmup=args.skip_warmup,
        use_response_pipes=args.use_response_pipes,
        send_latents_once=args.send_latents_once,
        mock_timeout=args.mock_timeout_ms,
    )
    # TODO from args (sample_ids.txt)
    sample_ids = [4655, 2569, 1303, 109, 4509, 3009, 2179, 1826, 2094, 3340]
    server.issue_queries(
        [lg.QuerySample(index, id) for index, id in enumerate(sample_ids)]
    )
    # We don't know when it finishes
    time.sleep(args.infer_timeout)
    server.finish_test()


def main():
    gc.disable()
    # Needed for tracing
    mp.set_start_method("spawn")
    args = get_args()
    infer(args)


if __name__ == "__main__":
    main()
