from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor
import concurrent.futures as futures
import gc
import asyncio
import multiprocessing as mp
import multiprocessing.queues
import os
import signal

import time

from multiprocessing import JoinableQueue, Process

import queue
import numa_helpers
import shortfin.array as sfnp

from shortfin_apps.sd.components.config_struct import ModelParams
from shortfin_apps.sd.components.generate import GenerateImageProcess
from shortfin_apps.sd.components.io_struct import GenerateReqInput
from shortfin_apps.sd.components.manager import SystemManager
from shortfin_apps.sd.components.messages import InferenceExecRequest, InferencePhase
from shortfin_apps.sd.components.service import (
    GenerateService,
    InferenceExecutorProcess,
)
from shortfin_apps.sd.components.tokenizer import Tokenizer

from shortfin_apps.sd.python_pipe import *
from transformers import CLIPTokenizer
from utilities import CONFIG, ArgHolder, rpd_trace, gen_input_ids

from threading import Thread

import time
import random


def create_service(
    sysman,
    model_params,
    device,
    tokenizers,
    vmfbs,
    params,
    device_idx=None,
    device_ids=[],
    fibers_per_device=2,
    workers_per_device=1,
    isolation="per_call",
    trace_execution=False,
    amdgpu_async_allocations=True,
):
    sdxl_service = GenerateService(
        name="sd",
        sysman=sysman,
        tokenizers=tokenizers,
        model_params=model_params,
        fibers_per_device=fibers_per_device,
        workers_per_device=workers_per_device,
        prog_isolation=isolation,
        show_progress=False,
        trace_execution=trace_execution,
    )
    for key, bs in vmfbs.items():
        for bs_int, vmfb_list in bs.items():
            for vmfb in vmfb_list:
                sdxl_service.load_inference_module(
                    vmfb, component=key, batch_size=bs_int
                )
    for key, datasets in params.items():
        sdxl_service.load_inference_parameters(
            *datasets, parameter_scope="model", component=key
        )
    print("Starting service")
    sdxl_service.start()
    print("Started service")
    return sdxl_service


class SampleRequest:

    def __init__(
        self,
        sample_ids,
        sample_indices,
        prompt_tokens_clip1,
        prompt_tokens_clip2,
        negative_prompt_tokens_clip1,
        negative_prompt_tokens_clip2,
        init_noise_latent,
        shark_engine,
    ):
        self.sample_ids = sample_ids
        self.sample_indices = sample_indices
        self.prompt_tokens_clip1 = prompt_tokens_clip1
        self.prompt_tokens_clip2 = prompt_tokens_clip2
        self.negative_prompt_tokens_clip1 = negative_prompt_tokens_clip1
        self.negative_prompt_tokens_clip2 = negative_prompt_tokens_clip2
        self.init_noise_latent = init_noise_latent
        self.shark_engine = shark_engine

    @staticmethod
    def create(samples, dataset, skip_latents=False, gpu_batch_size=1):
        pad = 0
        if len(samples) < gpu_batch_size:
            pad = gpu_batch_size - len(samples)
        sample_indices = [q.index for q in samples]
        # If required, pad the dataset to match the batch size.
        sample_indices += [-1 for _ in range(pad)]

        assert len(sample_indices) == gpu_batch_size

        return SampleRequest(
            sample_ids=[q.id for q in samples],
            sample_indices=sample_indices,
            prompt_tokens_clip1=dataset.prompt_tokens_clip1[
                sample_indices, : CONFIG.MAX_PROMPT_LENGTH
            ],
            prompt_tokens_clip2=dataset.prompt_tokens_clip2[
                sample_indices, : CONFIG.MAX_PROMPT_LENGTH
            ],
            negative_prompt_tokens_clip1=dataset.negative_prompt_tokens_clip1[
                sample_indices, : CONFIG.MAX_PROMPT_LENGTH
            ],
            negative_prompt_tokens_clip2=dataset.negative_prompt_tokens_clip2[
                sample_indices, : CONFIG.MAX_PROMPT_LENGTH
            ],
            init_noise_latent=None if skip_latents else dataset.init_noise_latent,
            shark_engine=None,
        )

    def __len__(self):
        return len(self.sample_ids)


class SampleResponse:

    def __init__(self, sample_ids, sample_indices, generated_images):
        self.sample_ids = sample_ids
        self.sample_indices = sample_indices
        self.generated_images = generated_images


class SampleProcessor(Process):

    def __init__(
        self,
        device_id,
        core_id,
        init_queue,
        sample_queue,
        response_comm,
        model_weights,
        init_noise_latent,
        gpu_batch_size,
        # precision,
        # unet_precision,
        verbose,
        enable_numa: bool,
        implementation,
        multiple_pipelines=None,
        skip_warmup=False,
        vmfbs=[],
        params=[],
        fibers_per_device: int = 2,
        workers_per_device: int = 1,
        num_sample_loops: int = 2,
        model_json: str = "sdxl_config_fp8_sched_unet.json",
        log_sample_gets: bool = False,
        debug: bool = False
    ):
        super(Process, self).__init__()
        self.device_id = 0
        self.real_device_id = device_id
        self.core_id = core_id
        self.init_queue = init_queue
        self.sample_queue = sample_queue
        self.response_comm = response_comm
        self.model_weights = model_weights
        self.init_noise_latent = init_noise_latent
        self.gpu_batch_size = gpu_batch_size
        self.fibers_per_device = fibers_per_device
        self.workers_per_device = workers_per_device
        self.model_json = model_json
        self.num_sample_loops = num_sample_loops
        self.verbose = verbose
        self.enable_numa = enable_numa
        self.multiple_pipelines = multiple_pipelines
        self.skip_warmup = skip_warmup
        self.verbose_log = print
        self.pipelines = {}
        self.service = None
        self.implementation_holder = implementation
        self.implementation = None
        self.vmfbs = vmfbs
        self.params = params
        self.model_params = None
        self.log_sample_gets = log_sample_gets
        self.debug = debug

        os.environ["ROCR_VISIBLE_DEVICES"] = f"{device_id}"

        if isinstance(self.response_comm, multiprocessing.queues.JoinableQueue):
            self.send_response = self.response_comm.put_nowait
        elif isinstance(self.response_comm, mp.connection.Connection):
            self.send_response = self.response_comm.send
        else:
            raise RuntimeError(f"Unsupported response comm {type(self.response_comm)}")

        def signal_handler(sig, frame):
            print(f'Sample Processor {self.pid} received signal {sig}')
            # Perform cleanup actions here, if needed
            if self.service is not None:
                self.sysman.shutdown()
                self.service.shutdown()
            if self.implementation is not None:
                del self.implementation
            gc.collect()

            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)

    @rpd_trace()
    def init_processor(self):
        if self.enable_numa:
            numa_helpers.set_affinity_by_device(self.device_id)

        # This can't be in the __init__ because the process spawn complains about the lambda
        self.verbose_log = lambda i: (
            print(f"[Device {self.real_device_id}:{self.core_id}] {i}", flush=True)
            if self.verbose
            else lambda _: None
        )

        self.sysman = SystemManager("amdgpu", [self.device_id], True)

        # Load here to avoid pickling errors.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, self.model_json)
        self.model_params = ModelParams.load_json(script_path)

        self.service = self.start_service()

        if not self.skip_warmup:
            self.warmup()

        self.init_queue.put((0, self.device_id, self.core_id))

    def start_service(self):
        tokenizers = []
        tokenizers.append(
            Tokenizer.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer"
            )
        )
        tokenizers.append(
            Tokenizer.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2"
            )
        )

        sdxl_service = create_service(
            self.sysman,
            self.model_params,
            "amdgpu",
            tokenizers,
            self.vmfbs,
            self.params,
            self.real_device_id,
            fibers_per_device=self.fibers_per_device,
            workers_per_device=self.workers_per_device,
            isolation="per_call",
            amdgpu_async_allocations=True,
        )
        self.implementation = self.implementation_holder(
            sdxl_service, self.init_noise_latent
        )

        return sdxl_service

    @staticmethod
    def log_times(fn_name, device_id, worker_id, start, end):
        logfile_name = f"time_log_{fn_name}_device{device_id}_worker{worker_id}.log"
        with open(logfile_name, "a") as logfile:
            print(f"[{device_id}:{worker_id}][{fn_name}] Ran in {end-start} sec.", file=logfile, end="\n")

    @rpd_trace()
    def warmup(self):
        self.verbose_log("Warming up...")
        self.implementation.warmup(
            service=self.service,
            device_id=self.device_id,
            core_id=self.core_id,
            pipelines=self.pipelines,
            model_weights=self.model_weights,
            verbose_log=self.verbose_log,
            batch_size=self.gpu_batch_size,
        )
        self.verbose_log("Warmup done")

    @rpd_trace()
    def wait_for_sample(self):
        sample = self.sample_queue.get()
        self.sample_queue.task_done()
        return sample

    @rpd_trace()
    def sample_loop(self, fiber_idx, worker):
        self.verbose_log(f"process sample loop started")
        while True:
            samples = [self.wait_for_sample()]
            if samples[0] is None:
                # None in the queue indicates the SUT want us to exit
                break
            data_ls = [gen_input_ids(sample, self.implementation) for sample in samples]
            responses = self.implementation.generate_images(
                samples=samples,
                datas=data_ls,
                verbose_log=self.verbose_log,
                idx=fiber_idx,
                worker=worker,
            )
            self.send_response(
                SampleResponse(
                    samples[0].sample_ids, samples[0].sample_indices, responses
                )
            )
            self.implementation.imgs = []
        self.verbose_log(f"process sample loop finished")

    async def distributor(self):
        loop = asyncio.get_running_loop()
        workers = []
        for idx in range(self.fibers_per_device):
            worker = self.sysman.ls.create_worker(f"distributor_{idx}")
            workers.append(worker)

        with ThreadPoolExecutor(max_workers=16) as executor:
            futs = []
            for idx in range(self.fibers_per_device):
                futs.append(
                    loop.run_in_executor(executor, self.sample_loop, idx, workers[idx])
                )

            for f in asyncio.as_completed(futs):
                await f
                self.verbose_log("Finished executor loop")

    @rpd_trace()
    def run(self):
        gc.disable()
        self.init_processor()
        asyncio.run(self.distributor())
