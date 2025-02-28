import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from sys import exec_prefix
import numpy as np
import asyncio

from base_process_samples import BaseProcessSamples
from pathlib import Path
from preprocess_data import prepare_tokenizer, encode_prompts
from sample_processor import SampleRequest, SampleResponse
from utilities import CONFIG, rpd_trace, ArgHolder, gen_input_ids, sample_request
from PIL import Image

import shortfin.array as sfnp
import shortfin as sf

import time

from shortfin_apps.sd.python_pipe import create_service
from shortfin_apps.sd.components.messages import InferencePhase, InferenceExecRequest
from shortfin_apps.sd.components.service import (
    InferenceExecutorProcess,
)

from threading import Lock

class MicroSDXLHarnessExecutor:

    def __init__(self, args, service, idx, logger):
        self.service = service

        self.args = args
        self.exec = None
        self.imgs = None
        self.fiber_idx = idx
        self.logger = logger

    async def run(self):

        runner = Runner(self.args, self.service, self.fiber_idx, self.logger)
        await asyncio.gather(runner.launch())
        # self.imgs = runner.imgs
        return runner.imgs


class Runner(sf.Process):

    def __init__(self, args, service, fiber_idx, logger):
        super().__init__(fiber=service.meta_fibers[fiber_idx].fiber)
        self.args = args
        self.imgs = None
        self.service = service
        self.fiber_idx = fiber_idx

    async def run(self):
        args = self.args
        self.exec = InferenceExecRequest(
            prompt=None,
            neg_prompt=None,
            input_ids=args.input_ids,
            sample=args.sample,
            height=1024,
            width=1024,
            steps=20,
            guidance_scale=8,
        )
        self.exec.batch_size = args.input_ids[0][0].shape[0]

        self.exec.phases[InferencePhase.POSTPROCESS]["required"] = False
        self.exec.phases[InferencePhase.PREPARE]["required"] = False

        while len(self.service.idle_meta_fibers) == 0:
            print("All fibers busy...")

        fiber = self.service.idle_meta_fibers.pop()
        exec_process = InferenceExecutorProcess(self.service, fiber)

        exec_process.exec_request = self.exec
        exec_process.launch()

        await asyncio.gather(exec_process)
        await self.exec.done

        if self.service.prog_isolation != sf.ProgramIsolation.PER_FIBER:
            self.service.idle_meta_fibers.append(fiber)

        self.imgs = exec_process.exec_request.image_array
        return exec_process.exec_request.image_array


class SharkMicroShortfinProcessSamples(BaseProcessSamples):

    def __init__(self, service, init_noise_latent):
        self.service = None
        self.num_exec_procs = 1
        self.imgs = []
        self.procs = []
        self.init_noise_latent = init_noise_latent
        self.service = service

    @rpd_trace()
    def get_asdevicearray():
        pass

    @rpd_trace()
    def create(
        self,
        device_id,
        core_id,
        init_queue,
        model_config,
        vmfbs,
        params,
        init_noise_latent,
        verbose_log,
    ):
        pass
        # verbose_log(f"Initializing with pid {os.getpid()}")
        # try:
        #     # Note: These moved here deliberately, so we should fail even when we are not printing them
        #     rocr_devs = os.environ["ROCR_VISIBLE_DEVICES"]
        #     hip_devs = os.environ["ROCR_VISIBLE_DEVICES"]
        #     verbose_log(
        #         f"ROCR_VISIBLE_DEVICES={rocr_devs} HIP_VISIBLE_DEVICES={hip_devs}"
        #     )
        # except KeyError:
        #     init_queue.put((-1, device_id, core_id))
        #     raise RuntimeError(
        #         f"[Device {device_id}:{core_id}] Please set 'ROCR_VISIBLE_DEVICES' and 'HIP_VISIBLE_DEVICES' envs"
        #     )
        # try:
        #     init_noise_latent = init_noise_latent.astype(np.float16)
        # except RuntimeError:
        #     init_queue.put((-1, device_id, core_id))
        #     raise
        # return

    @rpd_trace()
    def start_service(self, model_config, vmfbs, params, noise, device_idx=None):
        pass
        # model_params, tokenizers, vmfbs, params = self.prepare_service()
        # fibers_per_device = 1
        # isolation = "per_fiber"
        # trace_execution = False
        #
        # service = create_service(
        #     model_params=model_config,
        #     device="amdgpu",
        #     tokenizers=[],
        #     vmfbs=vmfbs,
        #     params=params,
        #     device_idx=0,
        #     device_ids=[],
        #     fibers_per_device=fibers_per_device,
        #     isolation=isolation,
        #     trace_execution=trace_execution,
        # )
        # self.init_noise_latent = []
        # for _ in range(self.num_exec_procs):
        #     # NOTE: This assumes each service knows one device! Otherwise, we need a mapping of init latents to fibers explicitly.
        #     latent_dev = sfnp.device_array.for_device(
        #         service.meta_fibers[0].device(0), noise.shape, model_config.unet_dtype
        #     )
        #     sample_host = latent_dev.for_transfer()
        #     with sample_host.map(discard=True) as m:
        #         m.fill(noise)

        #     latent_dev.copy_from(sample_host)
        #     self.init_noise_latent = latent_dev
        # return service

    @rpd_trace()
    def warmup(
        self,
        device_id,
        core_id,
        pipelines,
        model_weights,
        verbose_log,
        service,  # TODO: Remove this
        batch_size=1,
        url=None,
    ):
        # pass
        # # self.service.load_infere
        tokenizers = service.tokenizers
        synthetic_strs = [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua",
        ]
        input_ids_clip1 = tokenizers[0].encode(synthetic_strs[0])
        input_ids_clip2 = tokenizers[1].encode(synthetic_strs[1])

        prompt_tokens_clip1 = np.concatenate(
            [[input_ids_clip1["input_ids"]]] * batch_size
        ).reshape(batch_size, -1)
        prompt_tokens_clip2 = np.concatenate(
            [[input_ids_clip2["input_ids"]]] * batch_size
        ).reshape(batch_size, -1)
        negative_prompt_tokens_clip1 = np.concatenate(
            [[input_ids_clip1["input_ids"]]] * batch_size
        ).reshape(batch_size, -1)
        negative_prompt_tokens_clip2 = np.concatenate(
            [[input_ids_clip2["input_ids"]]] * batch_size
        ).reshape(batch_size, -1)

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
            init_noise_latent=self.init_noise_latent,
            shark_engine=None,
        )

        data = gen_input_ids(dummy_request, batch_size, self)
        data["sample"] = self.init_noise_latent

        worker = self.service.sysman.ls.create_worker("Warmup")
        for _ in range(2):
            _ = self.generate_images([dummy_request], [data], print, worker=worker)

    @rpd_trace()
    # TODO: Add a batch size arg
    def generate_images(
        self,
        samples,
        datas,
        verbose_log,
        worker,
        url=None,
        idx=0,
    ):
        with rpd_trace(f"generate_images:send request"):
            executors = [
                MicroSDXLHarnessExecutor(
                    ArgHolder(**data), service=self.service, idx=idx, logger=verbose_log
                )
                for data in datas
            ]
            fut = asyncio.run_coroutine_threadsafe(executors[0].run(), loop=worker.loop)
            results = None
            for fut in concurrent.futures.as_completed([fut]):
                results = fut.result()
            assert results is not None
            self.imgs = results
            return results
