import argparse
from utilities import CONFIG
import shutil
import os
import hashlib

def hashfile(file, verbose_log, golden_hash, destination_dir):
    if not os.path.isfile(file):
        return
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
    with open(file, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    hash = sha256.hexdigest()
    if hash != golden_hash:
        verbose_log(f"WARNING: Generated quantized artifacts ({str(file).split('/')[-1]}) differ from golden, tested weights. Results may slightly vary. Clear out /mlperf/quant_sdxl/ and {destination_dir}. Then, rerun if you would like to use golden weights from huggingface.")
    return hash

# Before creating the sd pipeline, move quant artifacts to appropriate spot
def move_quant_file(source, destination, verbose_log):
    if not os.path.isfile(source):
        verbose_log("Please run the quantization setup docker before the SDXL inference docker")
    else:
        shutil.copy(source, destination)

def move_quant_artifacts(args, destination_dir, verbose_log):
    # Get the directory of this file
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    ml_perf_dir = os.path.dirname(current_dir)
    quant_dir = os.path.join(ml_perf_dir, "quant_sdxl")
    os.makedirs(destination_dir, exist_ok=True)
    verbose_log(f"Moving quant artifacts to {destination_dir} for sd pipeline")
    move_quant_file(os.path.join(quant_dir, "config.json"), os.path.join(destination_dir, "config.json"), verbose_log)
    hashfile(os.path.join(quant_dir, "config.json"), verbose_log, "9a2fe50e083fb0418a07f7f59d30abac582d8c30fe69f698b9893fb1198874c3", destination_dir)
    move_quant_file(os.path.join(quant_dir, "params.safetensors"), os.path.join(destination_dir, "params.safetensors"), verbose_log)
    hashfile(os.path.join(quant_dir, "params.safetensors"), verbose_log, "7e0756aa50578c07b4db426fc3c97b01a7ae211650b6187d6c2057d47ee95717", destination_dir)
    move_quant_file(os.path.join(quant_dir, "quant_params.json"), os.path.join(destination_dir, "quant_params.json"), verbose_log)
    hashfile(os.path.join(quant_dir, "quant_params.json"), verbose_log, "97a452a4f96252a5bc1c686f6e58c90a8fd460f4aedff565fff50a61f69b6675", destination_dir)
    move_quant_file(os.path.join(quant_dir, "vae.safetensors"), os.path.join(destination_dir, "vae.safetensors"), verbose_log)
    hashfile(os.path.join(quant_dir, "vae.safetensors"), verbose_log, "d5f4a2c48ac98e9ad4054823d5a9a8b9370f4d23b8ed5ce21d5463c36ffc28fe", destination_dir)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_weights",
        type=str,
        default="/models/SDXL/official_pytorch/fp16/stable_diffusion_fp16",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    move_quant_artifacts(args, os.path.join(args.model_weights, "safetensors_quant"), print) 


if __name__ == "__main__":
    main()
