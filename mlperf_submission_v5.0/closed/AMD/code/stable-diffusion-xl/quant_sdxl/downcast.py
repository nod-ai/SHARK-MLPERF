from safetensors import safe_open
from safetensors.torch import save_file
import argparse
import os
import torch

def main(args):
    if args.dtype != "float32":
        print("Downcast not neccesary. Weights are quantized and ready to use.")
        return
    
    input_dir = args.input_dir
    if args.output_dir is None:
        output_dir = args.input_dir + '_fp16'
    else:
        output_dir = args.output_dir
    os.mkdir(output_dir)

    os.rename(os.path.join(input_dir, "params.safetensors"), os.path.join(input_dir, "params_fp32.safetensors"))
    os.rename(os.path.join(input_dir, "vae.safetensors"), os.path.join(input_dir, "vae_fp32.safetensors"))

    safetensor_files = ['params_fp32.safetensors', 'vae_fp32.safetensors']

    for file in safetensor_files:
        safetensor_path = os.path.join(input_dir, file)

        new_dict = {}
        with safe_open(safetensor_path, framework="pt") as f:
            for k in f.keys():
                new_dict[k] = f.get_tensor(k).to(torch.float16)

        output_file_name = file.split("_")[0] + ".safetensors"
        save_file(new_dict, os.path.join(output_dir, output_file_name))
        print(f"Saved weights to {output_file_name}")
    
    print(f"Downcasting of weight artifacts has been completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='conversion')
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Path or name of the model.')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Path or name of the model.')
    parser.add_argument(
        '--dtype',
        default='float32',
        choices=['float32', 'float16', 'bfloat16'],
        help='Downcast quant artifacts input dtype')
    args = parser.parse_args()
    main(args)
