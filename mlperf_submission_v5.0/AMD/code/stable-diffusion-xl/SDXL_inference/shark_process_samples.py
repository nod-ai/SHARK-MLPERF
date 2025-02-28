import os
import numpy as np

from base_process_samples import BaseProcessSamples
from sample_processor import SampleRequest, SampleResponse
from utilities import CONFIG

from preprocess_data import prepare_tokenizer, encode_prompts
from pathlib import Path
from utilities import rpd_trace, measure


@rpd_trace()
@measure
def create_sd_pipeline(
    engine_dir,
    precision="fp16",
    unet_precision="i8",
    gpu_batch_size=1,
    steps=CONFIG.STEPS,
    scheduler_id=CONFIG.SCHEDULER,
    device="hip",
    verbose_log=lambda _: None,
):
    from turbine_models.custom_models.sdxl_inference.sdxl_compiled_pipeline import (
        EMPTY_FLAGS,
    )
    from turbine_models.custom_models.sd_inference.sd_pipeline import SharkSDPipeline

    verbose_log(
        f"Create pipeline {gpu_batch_size=} with {device=}, {precision=}, {unet_precision=}"
    )
    use_i8_punet = unet_precision == "i8"
    vmfbs_path = os.path.join(engine_dir, "turbine_sdxl_quant", f"bs{gpu_batch_size}")
    os.makedirs(vmfbs_path, exist_ok=True)
    spec_path = "/iree/build_tools/pkgci/external_test_suite/attention_and_matmul_spec_punet_mi300.mlir"
    safetensors_path = os.path.join(engine_dir, "safetensors_quant")
    os.makedirs(safetensors_path, exist_ok=True)
    pipe = SharkSDPipeline(
        hf_model_name=f"{engine_dir}/checkpoint_pipe",
        height=1024,
        width=1024,
        batch_size=gpu_batch_size,
        max_length=CONFIG.MAX_PROMPT_LENGTH,
        precision=precision,
        device=device,
        target="gfx942",
        ireec_flags=EMPTY_FLAGS,
        attn_spec=spec_path,
        decomp_attn={"text_encoder": True, "unet": False, "vae": True},
        pipeline_dir=vmfbs_path,
        external_weights_dir=safetensors_path,
        external_weights="safetensors",
        num_inference_steps=steps,
        cpu_scheduling=False,
        scheduler_id=scheduler_id,
        shift=None,
        use_i8_punet=use_i8_punet,  # use int8 unet vs fp16
        benchmark=True,
        verbose=False,
        batch_prompts=True,  # batch prompt encoder inputs
        punet_quant_paths={
            "config": f"{safetensors_path}/config.json",
            "params": f"{safetensors_path}/params.safetensors",
            "quant_params": f"{safetensors_path}/quant_params.json",
        },
        vae_weight_path=f"{safetensors_path}/vae.safetensors",
        vae_harness=True,
    )
    return pipe


class SharkProcessSamples(BaseProcessSamples):

    @rpd_trace()
    def get_asdevicearray():
        import iree.runtime as ireert

        return ireert.asdevicearray

    @rpd_trace()
    def create(
        device_id,
        core_id,
        as_device_array_fn,
        init_queue,
        model_weights,
        precision,
        unet_precision,
        init_noise_latent,
        gpu_batch_size,
        multiple_pipelines,
        verbose_log,
    ):
        verbose_log(f"Initializing with pid {os.getpid()}")
        try:
            # Note: These moved here deliberately, so we should fail even when we are not printing them
            rocr_devs = os.environ["ROCR_VISIBLE_DEVICES"]
            hip_devs = os.environ["ROCR_VISIBLE_DEVICES"]
            verbose_log(
                f"ROCR_VISIBLE_DEVICES={rocr_devs} HIP_VISIBLE_DEVICES={hip_devs}"
            )
        except KeyError:
            init_queue.put((-1, device_id, core_id))
            raise RuntimeError(
                f"[Device {device_id}:{core_id}] Please set 'ROCR_VISIBLE_DEVICES' and 'HIP_VISIBLE_DEVICES' envs"
            )

        device = f"hip://{device_id}"
        batch_size_options = (
            multiple_pipelines if multiple_pipelines else [gpu_batch_size]
        )

        # Pipeline
        verbose_log(f"load pipeline from {model_weights}")
        # Use only batch sizes which smaller then gpu_batch_size
        # TODO: Further reduce the batch_sizes if not enough VRAM
        try:
            pipelines = __class__._create_pipelines(
                device_id=device_id,
                core_id=core_id,
                as_device_array_fn=as_device_array_fn,
                batch_sizes=list(
                    filter(
                        lambda batch_size: batch_size <= gpu_batch_size,
                        batch_size_options,
                    )
                ),
                engine_dir=model_weights,
                precision=precision,
                unet_precision=unet_precision,
                init_noise_latent=init_noise_latent,
                steps=CONFIG.STEPS,
                guidance_scale=CONFIG.GUIDANCE,
                device=device,
                verbose_log=verbose_log,
            )
        except RuntimeError:
            init_queue.put((-1, device_id, core_id))
            raise

        verbose_log(f"{len(pipelines)} pipeline(s) created")
        return pipelines

    @rpd_trace()
    def warmup(
        device_id,
        core_id,
        pipelines,
        model_weights,
        verbose_log,
    ):
        tokenizer_clip1 = prepare_tokenizer(
            Path(model_weights, "checkpoint_pipe/tokenizer")
        )
        tokenizer_clip2 = prepare_tokenizer(
            Path(model_weights, "checkpoint_pipe/tokenizer_2")
        )
        syntetic_strs = [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua",
        ]
        # The last should be the largest
        largest_batch_size = pipelines[-1]["size"]
        input_ids_clip1 = encode_prompts(tokenizer_clip1, syntetic_strs)
        input_ids_clip2 = encode_prompts(tokenizer_clip2, syntetic_strs)
        prompt_tokens_clip1 = np.concatenate(
            [[input_ids_clip1[0]]] * largest_batch_size
        )
        prompt_tokens_clip2 = np.concatenate(
            [[input_ids_clip2[0]]] * largest_batch_size
        )
        negative_prompt_tokens_clip1 = np.concatenate(
            [[input_ids_clip1[1]]] * largest_batch_size
        )
        negative_prompt_tokens_clip2 = np.concatenate(
            [[input_ids_clip2[1]]] * largest_batch_size
        )

        for pipeline in pipelines:
            batch_size = pipeline["size"]
            # create samples
            dummy_request = SampleRequest(
                sample_ids=list(range(batch_size)),
                sample_indices=list(range(batch_size)),
                prompt_tokens_clip1=prompt_tokens_clip1[
                    :batch_size, : CONFIG.MAX_PROMPT_LENGTH
                ],
                prompt_tokens_clip2=prompt_tokens_clip2[
                    :batch_size, : CONFIG.MAX_PROMPT_LENGTH
                ],
                negative_prompt_tokens_clip1=negative_prompt_tokens_clip1[
                    :batch_size, : CONFIG.MAX_PROMPT_LENGTH
                ],
                negative_prompt_tokens_clip2=negative_prompt_tokens_clip2[
                    :batch_size, : CONFIG.MAX_PROMPT_LENGTH
                ],
                init_noise_latent=None,
            )
            for _ in range(2):
                __class__.generate_images(
                    device_id=device_id,
                    core_id=core_id,
                    pipelines=pipelines,
                    request=dummy_request,
                    verbose_log=verbose_log,
                )

    @rpd_trace()
    def generate_images(
        device_id,
        core_id,
        pipelines,
        request,
        verbose_log,
    ):
        verbose_log("generate images started")
        sample_indices = request.sample_indices
        with rpd_trace(f"generate_images:init"):
            selected_pipeline = __class__._select_pipeline(
                pipelines, len(sample_indices)
            )
            assert selected_pipeline is not None

            pipe = selected_pipeline["pipe"]
            gpu_batch_size = selected_pipeline["size"]
            latents_input = selected_pipeline["latents_input"]
            guidance_scale = selected_pipeline["guidance_scale"]
            steps = selected_pipeline["steps"]

            actual_batch_size = len(sample_indices)
            verbose_log(
                f"generate_images batch_size: {gpu_batch_size} samples: {actual_batch_size}"
            )
            sample_indices = request.sample_indices
            sample_ids = request.sample_ids
            actual_batch_size = len(sample_indices)
            verbose_log(
                f"generate_images batch_size: {gpu_batch_size} samples: {actual_batch_size}"
            )
            verbose_log(
                f"Running inference on sample {sample_indices} {sample_ids} with batch size {actual_batch_size}"
            )
        with rpd_trace(f"generate_images:sample data"):
            # TODO: This uses numpy, but torch has pin_memory. We should check if np has similar
            prompt_tokens_clip1 = request.prompt_tokens_clip1
            prompt_tokens_clip2 = request.prompt_tokens_clip2
            negative_prompt_tokens_clip1 = request.negative_prompt_tokens_clip1
            negative_prompt_tokens_clip2 = request.negative_prompt_tokens_clip2
            # Pad to gpu_batch_size with zeros
            verbose_log(
                f"GPU batch size: {gpu_batch_size} actual batch size: {actual_batch_size}"
            )
            assert prompt_tokens_clip1.shape[1] == CONFIG.MAX_PROMPT_LENGTH
            padding = np.zeros(
                (gpu_batch_size - actual_batch_size, prompt_tokens_clip1.shape[1]),
                dtype=prompt_tokens_clip1.dtype,
            )
            prompt_tokens_clip1 = np.concatenate((prompt_tokens_clip1, padding)).astype(
                np.int64
            )
            prompt_tokens_clip2 = np.concatenate((prompt_tokens_clip2, padding)).astype(
                np.int64
            )
            negative_prompt_tokens_clip1 = np.concatenate(
                (negative_prompt_tokens_clip1, padding)
            ).astype(np.int64)
            negative_prompt_tokens_clip2 = np.concatenate(
                (negative_prompt_tokens_clip2, padding)
            ).astype(np.int64)
            if request.init_noise_latent is not None:
                latents_input = np.concatenate(
                    [request.init_noise_latent.astype(np.float16)] * gpu_batch_size
                )

        with rpd_trace(f"generate_images:encode prompts"):
            prompt_embeds, add_text_embeds = pipe.text_encoder(
                "encode_prompts",
                [
                    prompt_tokens_clip1,
                    prompt_tokens_clip2,
                    negative_prompt_tokens_clip1,
                    negative_prompt_tokens_clip2,
                ],
            )

        with rpd_trace(f"generate_images:unet loop"):
            latents = pipe._produce_latents_sdxl(
                latents_input, prompt_embeds, add_text_embeds, steps, guidance_scale
            )

        with rpd_trace(f"generate_images:vae decoe"):
            vae_out = pipe.vae("decode", [latents])

        with rpd_trace(f"generate_images:create response"):
            # Report back to loadgen use sample_ids
            response = SampleResponse(
                sample_ids=sample_ids,
                sample_indices=sample_indices,
                generated_images=vae_out,
            )

        return response

    @rpd_trace()
    def _create_pipelines(
        device_id,
        core_id,
        as_device_array_fn,
        batch_sizes,
        engine_dir,
        precision,
        unet_precision,
        init_noise_latent,
        steps,
        guidance_scale,
        device,
        verbose_log,
    ):
        pipelines = []
        for batch_size in sorted(batch_sizes):
            pipe = __class__._init_pipeline(
                engine_dir=engine_dir,
                device_id=device_id,
                core_id=core_id,
                precision=precision,
                unet_precision=unet_precision,
                steps=steps,
                gpu_batch_size=batch_size,
                device=device,
                verbose_log=verbose_log,
            )

            verbose_log(f"create pipeline for batch_size {batch_size}")
            noise_latent = np.concatenate(
                [init_noise_latent.astype(np.float16)] * batch_size
            )
            latents_input = as_device_array_fn(pipe.unet.device, noise_latent)
            pipelines.append(
                {
                    "size": batch_size,
                    "pipe": pipe,
                    "latents_input": latents_input,
                    "guidance_scale": guidance_scale,
                    "steps": steps,
                }
            )

        return pipelines

    @rpd_trace()
    def _select_pipeline(pipelines, batch_size):
        for pipeline in pipelines:
            if pipeline["size"] < batch_size:
                continue

            return pipeline

    @rpd_trace()
    def _init_pipeline(
        engine_dir,
        device_id,
        core_id,
        precision="fp16",
        unet_precision="i8",
        gpu_batch_size=1,
        steps=CONFIG.STEPS,
        scheduler_id=CONFIG.SCHEDULER,
        device="hip",
        verbose_log=lambda _: None,
    ):
        verbose_log(
            f"Create pipeline for device: {device_id}:{core_id} batch_size: {gpu_batch_size}"
        )
        pipe = create_sd_pipeline(
            engine_dir=engine_dir,
            precision=precision,
            unet_precision=unet_precision,
            gpu_batch_size=gpu_batch_size,
            steps=steps,
            scheduler_id=scheduler_id,
            device=device,
            verbose_log=verbose_log,
        )
        ready = pipe.is_prepared({}, {})
        if not ready:
            # We need this to avoid multiprocess compile conflicts
            raise RuntimeError(
                "Models are not compiled. Use precompile_models.py before running harness!"
            )
        pipe.prepare_all()
        pipe.load_map()
        pipe.load_scheduler(scheduler_id, steps)
        return pipe
