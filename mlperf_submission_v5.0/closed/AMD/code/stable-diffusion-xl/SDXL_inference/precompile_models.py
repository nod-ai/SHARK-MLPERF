# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import torch
from iree.turbine.aot import *
from iree.compiler.ir import Context

from diffusers import EulerDiscreteScheduler

from turbine_models.custom_models.sd_inference import utils

import argparse
from shark_process_samples import create_sd_pipeline
from utilities import CONFIG
import shutil
import os
import hashlib


def hashfile(file, verbose_log, golden_hash, destination_dir):
    if not os.path.isfile(file):
        return
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
    with open(file, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    hash = sha256.hexdigest()
    if hash != golden_hash:
        verbose_log(
            f"WARNING: Generated quantized artifacts ({str(file).split('/')[-1]}) differ from golden, tested weights. Results may slightly vary. Clear out /mlperf/quant_sdxl/ and {destination_dir}. Then, rerun if you would like to use golden weights from huggingface."
        )
    return hash


# Before creating the sd pipeline, move quant artifacts to appropriate spot
def move_quant_file(source, destination, verbose_log):
    if not os.path.isfile(source):
        verbose_log(
            "Please run the quantization setup docker before the SDXL inference docker"
        )
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
    move_quant_file(
        os.path.join(quant_dir, "config.json"),
        os.path.join(destination_dir, "config.json"),
        verbose_log,
    )
    hashfile(
        os.path.join(quant_dir, "config.json"),
        verbose_log,
        "9a2fe50e083fb0418a07f7f59d30abac582d8c30fe69f698b9893fb1198874c3",
        destination_dir,
    )
    move_quant_file(
        os.path.join(quant_dir, "params.safetensors"),
        os.path.join(destination_dir, "params.safetensors"),
        verbose_log,
    )
    hashfile(
        os.path.join(quant_dir, "params.safetensors"),
        verbose_log,
        "7e0756aa50578c07b4db426fc3c97b01a7ae211650b6187d6c2057d47ee95717",
        destination_dir,
    )
    move_quant_file(
        os.path.join(quant_dir, "quant_params.json"),
        os.path.join(destination_dir, "quant_params.json"),
        verbose_log,
    )
    hashfile(
        os.path.join(quant_dir, "quant_params.json"),
        verbose_log,
        "97a452a4f96252a5bc1c686f6e58c90a8fd460f4aedff565fff50a61f69b6675",
        destination_dir,
    )
    move_quant_file(
        os.path.join(quant_dir, "vae.safetensors"),
        os.path.join(destination_dir, "vae.safetensors"),
        verbose_log,
    )
    hashfile(
        os.path.join(quant_dir, "vae.safetensors"),
        verbose_log,
        "d5f4a2c48ac98e9ad4054823d5a9a8b9370f4d23b8ed5ce21d5463c36ffc28fe",
        destination_dir,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_weights",
        type=str,
        default="/models/SDXL/official_pytorch/fp16/stable_diffusion_fp16",
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        required=True,
        help='Comma-separated batch sizes, e.g. "1,2,4,8"',
    )
    args = parser.parse_args()
    return args


def compile(args):
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    for batch_size in batch_sizes:
        pipe = create_sd_pipeline(
            engine_dir=args.model_weights,
            precision="fp16",
            unet_precision="i8",
            gpu_batch_size=batch_size,
            steps=CONFIG.STEPS,
            scheduler_id=CONFIG.SCHEDULER,
            device="hip",
            verbose_log=print,
        )
        pipe.prepare_all({}, {}, {})
        _ = export_scheduler_model(
            "stabilityai/stable-diffusion-xl-base-1.0",
            CONFIG.SCHEDULER,
            batch_size,
            1024,
            1024,
            CONFIG.STEPS,
            "fp16",
            "vmfb",
            "hip",
            "gfx942",
            pipeline_dir=os.path.join(args.model_weights, "checkpoint_pipe"),
            exit_on_vmfb=False,
        )
        # pipe.load_scheduler(CONFIG.SCHEDULER, CONFIG.STEPS)


class SchedulingModel(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        scheduler,
        height,
        width,
        batch_size,
        dtype,
    ):
        super().__init__()
        # For now, assumes SDXL implementation. May not need parametrization for other models,
        # but keeping hf_model_name in case.
        self.model = scheduler
        self.height = height
        self.width = width
        self.is_sd3 = False
        if "stable-diffusion-3" in hf_model_name:
            self.is_sd3 = True
        self.batch_size = batch_size
        # Whether this will be used with CFG-enabled pipeline.
        self.do_classifier_free_guidance = True
        timesteps = [torch.empty((100), dtype=dtype, requires_grad=False)] * 100
        sigmas = [torch.empty((100), dtype=torch.float32, requires_grad=False)] * 100
        for i in range(1, 100):
            self.model.set_timesteps(i)
            timesteps[i] = torch.nn.functional.pad(
                self.model.timesteps.clone().detach(), (0, 100 - i), "constant", 0
            )
            sigmas[i] = torch.nn.functional.pad(
                self.model.sigmas.clone().detach(), (0, 100 - (i + 1)), "constant", 0
            )
        self.timesteps = torch.stack(timesteps, dim=0).clone().detach()
        self.sigmas = torch.stack(sigmas, dim=0).clone().detach()
        self.model.is_scale_input_called = True
        self.dtype = dtype

    # TODO: Make steps dynamic here
    def initialize(self, sample, num_inference_steps):
        height = self.height
        width = self.width
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=self.dtype)
        if self.do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
            add_time_ids = add_time_ids.repeat(self.batch_size, 1).type(self.dtype)
        max_sigma = self.sigmas[num_inference_steps].max()
        init_noise_sigma = (max_sigma**2 + 1) ** 0.5
        sample = sample * init_noise_sigma
        return (
            sample.type(self.dtype),
            add_time_ids,
            self.timesteps[num_inference_steps].squeeze(0),
            self.sigmas[num_inference_steps].squeeze(0),
        )

    def prepare_model_input(self, sample, i, timesteps, sigmas):
        sigma = sigmas[i]
        next_sigma = sigmas[i + 1]
        t = timesteps[i]
        latent_model_input = sample / ((sigma**2 + 1) ** 0.5)
        self.model.is_scale_input_called = True
        return (
            latent_model_input.type(self.dtype),
            t.type(self.dtype),
            sigma.type(self.dtype),
            next_sigma.type(self.dtype),
        )

    def step(self, noise_pred, sample, sigma, next_sigma):
        sample = sample.to(torch.float32)
        noise_pred = noise_pred.to(torch.float32)
        pred_original_sample = sample - sigma * noise_pred
        deriv = (sample - pred_original_sample) / sigma
        dt = next_sigma - sigma
        prev_sample = sample + deriv * dt
        return prev_sample.type(self.dtype)


@torch.no_grad()
def export_scheduler_model(
    hf_model_name: str,
    scheduler_id: str,
    batch_size: int = 1,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 30,
    precision: str = "fp16",
    compile_to: str = "torch",
    device: str = None,
    target: str = None,
    ireec_flags: str = None,
    exit_on_vmfb: bool = False,
    pipeline_dir: str = None,
    input_mlir: str = None,
    attn_spec: str = None,
    external_weights: str = None,
    external_weight_path: str = None,
    upload_ir=False,
):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    iree_dtype = "float16" if precision == "fp16" else "float32"
    scheduler = get_scheduler(hf_model_name, scheduler_id)
    scheduler_module = SchedulingModel(
        hf_model_name, scheduler, height, width, batch_size, dtype
    )

    vmfb_names = [
        scheduler_id + "Scheduler",
        f"bs{batch_size}",
        f"{height}x{width}",
        precision,
    ]
    vmfb_name = "_".join(vmfb_names)
    safe_name = utils.create_safe_name(hf_model_name, "_" + vmfb_name)
    if pipeline_dir:
        safe_name = os.path.join(pipeline_dir, safe_name)

    if input_mlir:
        vmfb_path = utils.compile_to_vmfb(
            input_mlir,
            device,
            target,
            ireec_flags,
            safe_name,
            mlir_source="file",
            return_path=not exit_on_vmfb,
        )
        return vmfb_path

    sample = (
        batch_size,
        4,
        height // 8,
        width // 8,
    )
    noise_pred_shape = (
        batch_size,
        4,
        height // 8,
        width // 8,
    )
    example_init_args = (
        torch.empty(sample, dtype=dtype),
        torch.empty(1, dtype=torch.int64),
    )
    example_prep_args = (
        torch.empty(sample, dtype=dtype),
        torch.empty(1, dtype=torch.int64),
        torch.empty(100, dtype=torch.float32),
        torch.empty(100, dtype=torch.float32),
    )
    example_step_args = [
        torch.empty(noise_pred_shape, dtype=dtype),
        torch.empty(sample, dtype=dtype),
        torch.empty(1, dtype=dtype),
        torch.empty(1, dtype=dtype),
    ]

    fxb = FxProgramsBuilder(scheduler_module)

    @fxb.export_program(
        args=(example_init_args,),
    )
    def _initialize(module, sample):
        return module.initialize(*sample)

    @fxb.export_program(
        args=(example_prep_args,),
    )
    def _scale(module, inputs):
        return module.prepare_model_input(*inputs)

    @fxb.export_program(
        args=(example_step_args,),
    )
    def _step(module, inputs):
        return module.step(*inputs)

    decomp_list = []
    # if decomp_attn == True:
    #     decomp_list.extend(
    #         [
    #             torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
    #             torch.ops.aten._scaled_dot_product_flash_attention.default,
    #         ]
    #     )
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
    ):

        class CompiledScheduler(CompiledModule):
            run_initialize = _initialize
            run_scale = _scale
            run_step = _step

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledScheduler(context=Context(), import_to=import_to)

    module = CompiledModule.get_mlir_module(inst)
    metadata_modelname = "_".join(
        [hf_model_name, scheduler_id, "scheduler", str(num_inference_steps)]
    )
    model_metadata_init = {
        "model_name": metadata_modelname,
        "input_shapes": [sample],
        "input_dtypes": [iree_dtype],
    }
    model_metadata_prep = {
        "model_name": metadata_modelname,
        "input_shapes": [sample, (1,), (1,)],
        "input_dtypes": [iree_dtype, "int64", "int64"],
    }
    model_metadata_step = {
        "model_name": metadata_modelname,
        "input_shapes": [noise_pred_shape, (1,), sample, (1,)],
        "input_dtypes": [iree_dtype, iree_dtype, "int64", "int64"],
    }
    module = AddMetadataPass(module, model_metadata_init, "run_initialize").run()
    module = AddMetadataPass(module, model_metadata_prep, "run_scale").run()
    module = AddMetadataPass(module, model_metadata_step, "run_step").run()
    module_str = str(module)
    if compile_to != "vmfb":
        return module_str
    elif compile_to == "vmfb":
        vmfb = utils.compile_to_vmfb(
            module_str,
            device,
            target,
            ireec_flags,
            safe_name,
            return_path=True,
        )
        return vmfb


def get_scheduler(model_id, scheduler_id):
    # TODO: switch over to turbine and run all on GPU
    print(f"\n[LOG] Initializing schedulers from model id: {model_id}")
    if scheduler_id in SCHEDULER_MAP.keys():
        scheduler = SCHEDULER_MAP[scheduler_id].from_pretrained(
            model_id, subfolder="scheduler"
        )
    else:
        raise ValueError(f"Scheduler {scheduler_id} not found.")
    if "Karras" in scheduler_id:
        scheduler.config.use_karras_sigmas = True

    return scheduler


SCHEDULER_MAP = {
    "EulerDiscrete": EulerDiscreteScheduler,
}


def main():
    args = get_args()
    move_quant_artifacts(
        args, os.path.join(args.model_weights, "safetensors_quant"), print
    )
    compile(args)


if __name__ == "__main__":
    main()
