#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################

import torch
import argparse
import json
import math
import os
import time
import requests
import shutil
from safetensors.torch import save_file

from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.quant.experimental.float_quant_fnuz import Fp8e4m3FNUZActPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas_examples.common.generative.nn import LoRACompatibleQuantConv2d, LoRACompatibleQuantLinear
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
from diffusers.models.attention_processor import Attention
import pandas as pd
from torch import nn
from tqdm import tqdm
import brevitas.nn as qnn

from brevitas.graph.base import ModuleToModuleByClass
from brevitas.graph.calibrate import bias_correction_mode, load_quant_model_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.quantize import layerwise_quantize
from brevitas.inject.enum import StatsOp
from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.utils.torch_utils import KwargsForwardHook
import brevitas.config as config

from brevitas_examples.common.parse_utils import add_bool_arg
from brevitas_examples.stable_diffusion.sd_quant.export import export_quant_params
from brevitas_examples.stable_diffusion.sd_quant.nn import QuantAttention, AttnProcessor

TEST_SEED = 123456
VAE_FIX_SCALE = 128

torch.manual_seed(TEST_SEED)

class WeightQuant(ShiftedUint8WeightPerChannelFloat):
    narrow_range = False
    scaling_min_val = 1e-4
    quantize_zero_point = True
    scaling_impl_type = 'parameter_from_stats'
    zero_point_impl = ParameterFromStatsFromParameterZeroPoint

class WeightQuantSym(Int8WeightPerChannelFloat):
    narrow_range = False
    scaling_min_val = 1e-4
    scaling_impl_type = 'parameter_from_stats'

class InputQuant(Int8ActPerTensorFloat):
    scaling_stats_op = StatsOp.MAX

class FP8Quant(Fp8e4m3FNUZActPerTensorFloat):
    scaling_stats_op = StatsOp.MAX

NEGATIVE_PROMPTS = ["normal quality, low quality, worst quality, low res, blurry, nsfw, nude"]

def load_calib_prompts(calib_data_path, sep="\t"):
    df = pd.read_csv(calib_data_path, sep=sep)
    lst = df["caption"].tolist()
    return lst

def run_val_inference(
        pipe,
        prompts,
        guidance_scale,
        total_steps,
        test_latents=None):
    with torch.no_grad():
        for prompt in tqdm(prompts):
            # We don't want to generate any image, so we return only the latent encoding pre VAE
            pipe(
                prompt,
                negative_prompt=NEGATIVE_PROMPTS[0],
                latents=test_latents,
                output_type='latent',
                guidance_scale=guidance_scale,
                num_inference_steps=total_steps)


def main(args):

    dtype = getattr(torch, args.dtype)

    calibration_prompts = load_calib_prompts(args.calibration_prompt_path)
    assert args.calibration_prompts <= len(calibration_prompts) , f"--calibration-prompts must be <= {len(calibration_prompts)}"
    calibration_prompts = calibration_prompts[:args.calibration_prompts]
    latents = torch.load(args.path_to_latents).to(dtype)

    # Create output dir.
    output_dir = os.path.join(args.output_path, "sdxl_quant_artifacts")
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print(f"Saving results in {output_dir}")

    # Download config from sdxl-quant-models
    url = "https://huggingface.co/amd-shark/sdxl-quant-models/resolve/main/unet/int8/config.json"
    response = requests.get(url)
    config_json_data = response.json()

    with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
        json.dump(config_json_data, fp, indent=4)

    # Dump args to json
    with open(os.path.join(output_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp)

    # Load model from float checkpoint
    print(f"Loading model from {args.model}...")
    variant = 'fp16' if dtype == torch.float16 else None
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype, variant=variant, use_safetensors=True)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    layer_whitelist = [
        "decoder.up_blocks.2.upsamplers.0.conv",
        "decoder.up_blocks.3.resnets.0.conv2",
        "decoder.up_blocks.3.resnets.1.conv2",
        "decoder.up_blocks.3.resnets.2.conv2"]

    corrected_layers = []
    with torch.no_grad():
        for name, module in pipe.vae.named_modules():
            if name in layer_whitelist:
                corrected_layers.append(name)
                module.weight /= VAE_FIX_SCALE
                if module.bias is not None:
                    module.bias /= VAE_FIX_SCALE
    print(f"Corrected layers in VAE: {corrected_layers}")
    pipe.vae.config.force_upcast = False # If VAE is rescaled, we DO NOT need upcast

    # Combine QKV/Q+KV layers for shared quantization
    pipe.fuse_qkv_projections()

    print(f"Model loaded from {args.model}.")

    # Move model to target device
    print(f"Moving model to {args.device}...")
    pipe = pipe.to(args.device)

    # Enable attention slicing
    if args.attention_slicing:
        pipe.enable_attention_slicing()

    # Extract list of layers to avoid
    blacklist = []
    non_blacklist = dict()
    blacklist = []
    for name, module in pipe.unet.named_modules():
        if any(map(lambda x: x in name, ['time_emb', 'conv_in', 'conv_out', 'add_embedding'])):
            blacklist.append(name)
    print(f"Blacklisted layers: {set(blacklist)}")

    # Make sure there all LoRA layers are fused first, otherwise raise an error
    for m in pipe.unet.modules():
        if hasattr(m, 'lora_layer') and m.lora_layer is not None:
            raise RuntimeError("LoRA layers should be fused in before calling into quantization.")

    pipe.set_progress_bar_config(disable=True)

    if args.load_checkpoint is not None:
        # Don't run full activation equalization if we're loading a quantized checkpoint
        num_ae_prompts = 2
    else:
        num_ae_prompts = len(calibration_prompts)
    with torch.no_grad(), activation_equalization_mode(
            pipe.unet,
            alpha=args.act_eq_alpha,
            layerwise=True,
            blacklist_layers=blacklist if args.exclude_blacklist_act_eq else None,
            add_mul_node=True):
        # Workaround to expose `in_features` attribute from the Hook Wrapper
        for m in pipe.unet.modules():
            if isinstance(m, KwargsForwardHook) and hasattr(m.module, 'in_features'):
                m.in_features = m.module.in_features
        total_steps = args.calibration_steps
        run_val_inference(
            pipe,
            calibration_prompts[:num_ae_prompts],
            total_steps=total_steps,
            test_latents=latents,
            guidance_scale=args.guidance_scale)

    # Workaround to expose `in_features` attribute from the EqualizedModule Wrapper
    for m in pipe.unet.modules():
        if isinstance(m, EqualizedModule) and hasattr(m.layer, 'in_features'):
            m.in_features = m.layer.in_features

    quant_layer_kwargs = {
    'input_quant': InputQuant, 'weight_quant': WeightQuantSym, 'dtype': dtype, 'device': args.device, 'input_dtype': dtype, 'input_device': args.device}
    quant_linear_kwargs = {
    'input_quant': InputQuant, 'weight_quant': WeightQuantSym, 'dtype': dtype, 'device': args.device, 'input_dtype': dtype, 'input_device': args.device}

    if args.quantize_sdp:
        rewriter = ModuleToModuleByClass(
            Attention,
            QuantAttention,
            matmul_input_quant=FP8Quant,
            query_dim=lambda module: module.to_qkv.in_features if hasattr(module, 'to_qkv') else module.to_q.in_features,
            dim_head=lambda module: math.ceil(1 / (module.scale ** 2)),
            processor=AttnProcessor(),
            is_equalized=True,
            fuse_qkv=True,
            cross_attention_dim=lambda module: module.cross_attention_dim if module.is_cross_attention else None )
        config.IGNORE_MISSING_KEYS = True
        pipe.unet = rewriter.apply(pipe.unet)
        config.IGNORE_MISSING_KEYS = False
        pipe.unet = pipe.unet.to(args.device)
        pipe.unet = pipe.unet.to(dtype)

    layer_map = {
    nn.Linear: (qnn.QuantLinear, quant_linear_kwargs),
    nn.Conv2d: (qnn.QuantConv2d, quant_layer_kwargs),
    'diffusers.models.lora.LoRACompatibleLinear':
        (LoRACompatibleQuantLinear, quant_linear_kwargs),
    'diffusers.models.lora.LoRACompatibleConv': (LoRACompatibleQuantConv2d, quant_layer_kwargs)}

    pipe.unet = layerwise_quantize(
        model=pipe.unet, compute_layer_map=layer_map, name_blacklist=blacklist)
    print("Model quantization applied.")


    with torch.no_grad():
        run_val_inference(
            pipe,
            [calibration_prompts[0]],
            total_steps=2,
            test_latents=latents,
            guidance_scale=args.guidance_scale)

    if args.load_checkpoint is not None:
        with load_quant_model_mode(pipe.unet):
            config.IGNORE_MISSING_KEYS = True
            print(f"Loading checkpoint: {args.load_checkpoint}... ", end="")
            sd = torch.load(args.load_checkpoint)
            pipe.unet.load_state_dict(sd)
            pipe.unet.eval()
            print(f"Checkpoint loaded!")
            config.IGNORE_MISSING_KEYS = False
        pipe = pipe.to(args.device)

    pipe.set_progress_bar_config(disable=True)

    # After fusing QKV (for self-attention) or KV (for cross-attention) we can remove the other dangling layers
    list_of_quant_attention = [m for m in pipe.unet.modules() if isinstance(m, QuantAttention)]
    for quant_att in list_of_quant_attention:
        if quant_att.is_cross_attention:
            del quant_att.to_v
            del quant_att.to_k
        else:
            del quant_att.to_v
            del quant_att.to_k
            del quant_att.to_q


    if args.load_checkpoint is None:
        print("Applying activation calibration")
        with torch.no_grad(), calibration_mode(pipe.unet):
            run_val_inference(
                pipe,
                calibration_prompts,
                total_steps=args.calibration_steps,
                test_latents=latents,
                guidance_scale=args.guidance_scale)

        print("Applying bias correction")
        with torch.no_grad(), bias_correction_mode(pipe.unet):
            run_val_inference(
                pipe,
                calibration_prompts,
                total_steps=args.calibration_steps,
                test_latents=latents,
                guidance_scale=args.guidance_scale)

        if args.checkpoint_name is not None:
            torch.save(pipe.unet.state_dict(), os.path.join(output_dir, args.checkpoint_name))
            torch.save(pipe.vae.state_dict(), os.path.join(output_dir, 'VAE_' + args.checkpoint_name))

    if args.export_target:
        pipe.unet.to('cpu').to(dtype)
        export_quant_params(pipe.unet, output_dir, '')
        save_file(pipe.vae.state_dict(), os.path.join(output_dir, 'vae.safetensors'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stable Diffusion quantization')
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default=None,
        help='Path or name of the model.')
    parser.add_argument(
        '-d', '--device', type=str, default='cuda:0', help='Target device for quantized model.')
    parser.add_argument(
        '--calibration-prompt-path', type=str, default=None, required=True, help='Path to calibration prompt')
    parser.add_argument(
        '--calibration-prompts',
        type=int,
        default=500,
        help='Number of prompts to use for calibration. Default: %(default)s')
    parser.add_argument(
        '--checkpoint-name',
        type=str,
        default=None,
        help=
        'Name to use to store the checkpoint in the output dir. If not provided, no checkpoint is saved.'
    )
    parser.add_argument(
        '--load-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to load. If provided, PTQ techniques are skipped.')
    parser.add_argument(
        '--path-to-latents',
        type=str,
        required=True,
        help=
        'Path to pre-defined latents.')
    parser.add_argument('--guidance-scale', type=float, default=8., help='Guidance scale.')
    parser.add_argument(
        '--calibration-steps', type=float, default=10, help='Steps used during calibration')
    add_bool_arg(
        parser,
        'output-path',
        str_true=True,
        default='.',
        help='Path where to generate output folder.')
    parser.add_argument(
        '--dtype',
        default='float16',
        choices=['float32', 'float16', 'bfloat16'],
        help='Model Dtype, choices are float32, float16, bfloat16. Default: float16')
    add_bool_arg(
        parser,
        'attention-slicing',
        default=False,
        help='Enable attention slicing. Default: Disabled')
    add_bool_arg(
        parser,
        'export-target',
        default=True,
        help='Export flow.')
    parser.add_argument(
        '--act-eq-alpha',
        type=float,
        default=0.9,
        help='Alpha for activation equalization. Default: 0.9')
    add_bool_arg(parser, 'quantize-sdp', default=False, help='Quantize SDP. Default: Disabled')
    add_bool_arg(
        parser,
        'exclude-blacklist-act-eq',
        default=False,
        help='Exclude unquantized layers from activation equalization. Default: Disabled')
    args = parser.parse_args()
    print("Args: " + str(vars(args)))
    main(args)
