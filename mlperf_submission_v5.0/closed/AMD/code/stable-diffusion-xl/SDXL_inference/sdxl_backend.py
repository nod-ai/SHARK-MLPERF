from dataset import Dataset
from typing import List, Optional
import logging
import time
import numpy as np
import array
import os
import requests
import subprocess
import sys
import queue
import socket
import signal

import multiprocessing.queues

from multiprocessing import JoinableQueue

from transformers import CLIPTokenizer

try:
    import mlperf_loadgen as lg
except:
    logging.warning(
        "Loadgen Python bindings are not installed. Installing Loadgen Python bindings!"
    )
    raise RuntimeError("Missing loadgen lib")

# Shared between
from queue import Empty as QueueEmpty
from threading import Thread
from sample_processor import SampleRequest, SampleProcessor

# from shortfin_process_samples import ShortfinProcessSamples
from mock_process_samples import MockProcessSamples
from utilities import rpd_trace
from shortfin_apps.sd.python_pipe import *
from shortfin_apps.sd.components.config_struct import ModelParams


from shark_micro_shortfin_process_samples import SharkMicroShortfinProcessSamples


class SDXLShortfinService:

    @rpd_trace()
    def __init__(
        self,
        devices: List[int],
        model_weights: List[str],
        dataset: Dataset,
        gpu_batch_size: int,
        verbose: bool = False,
        enable_batcher: bool = False,
        batch_timeout_threashold: float = -1,
        cores_per_devices: int = 1,
        save_images: bool = False,
        performance_sample_count: int = 5000,
        skip_warmup: bool = False,
        skip_complete: bool = False,
        use_response_pipes: bool = False,
        send_latents_once: bool = False,
        mock_timeout: Optional[int] = None,
        enable_numa: bool = True,
        workers_per_device: int = 1,
        fibers_per_device: int = 1,
        isolation: str = "per_fiber",
        num_sample_loops: int = 2,
        model_json: str = "sdxl_config_fp8_sched_unet.json",
        td_spec: str | None = None,
        log_sample_get: bool = False,
        debug: bool = False,
        force_export: bool = False
    ):
        self.devices = devices if not debug else [0]
        self.gpu_batch_size = gpu_batch_size
        self.dataset = dataset
        self.verbose = verbose
        self.enable_batcher = enable_batcher and batch_timeout_threashold > 0
        self.use_response_pipes = use_response_pipes
        self.send_latents_once = send_latents_once
        self.num_sample_loops = num_sample_loops
        self.force_export = force_export

        self.verbose_log = lambda i: (
            print(f"[Server] {i}", flush=True) if self.verbose else lambda _: None
        )
        self.verbose_log(f"init with {devices}x{cores_per_devices}")

        try:
            # Note: These moved here deliberately, so we should fail even when we are not printing them
            rocr_devs = os.environ["ROCR_VISIBLE_DEVICES"]
            hip_devs = os.environ["ROCR_VISIBLE_DEVICES"]
            self.verbose_log(
                f"ROCR_VISIBLE_DEVICES={rocr_devs} HIP_VISIBLE_DEVICES={hip_devs}"
            )
        except KeyError:
            raise RuntimeError(
                "Please set 'ROCR_VISIBLE_DEVICES' and 'HIP_VISIBLE_DEVICES' envs"
            )

        # start shortfin server
        self.workers_per_device = workers_per_device
        self.fibers_per_device = fibers_per_device
        self.model_json = model_json
        self.td_spec = td_spec
        self.isolation = isolation
        self.model_weights = model_weights

        _, vmfbs, params = self.prepare_service()

        # Server components
        self.init_queue = JoinableQueue()
        self.sample_queue = JoinableQueue()  # sample sync queue
        if self.use_response_pipes:
            self.response_pipes = []
        else:
            self.response_queue = JoinableQueue()
        self.sample_count = 0
        self.core_jobs = []
        self.response_jobs = []
        self.interrupt = False

        def signal_handler(sig, frame):
            self.interrupt = True
            print(f'Shortfin service received signal{sig}. Cleaning up child processes...')
            time.sleep(1)
            for job in self.core_jobs:
                if job.is_alive():
                    print(f"Terminating job {job}..")
                    job.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        # Start the cores
        for core_id in range(cores_per_devices):
            for device_id in self.devices:
                self.verbose_log(
                    f"Server process job init started for {device_id}:{core_id} device"
                )
                if use_response_pipes:
                    self.response_pipes.append(JoinableQueue())

                # TODO: Refactor to nicer solution
                implementation = SharkMicroShortfinProcessSamples
                if mock_timeout is not None:
                    implementation = MockProcessSamples
                    import mock_process_samples

                    mock_process_samples.DELAY_MS = mock_timeout

                job = SampleProcessor(
                    device_id=device_id,
                    core_id=core_id,
                    init_queue=self.init_queue,
                    sample_queue=self.sample_queue,
                    response_comm=self.response_queue if not self.use_response_pipes else self.response_pipes[-1],
                    model_weights=model_weights,
                    gpu_batch_size=self.gpu_batch_size,
                    verbose=self.verbose,
                    enable_numa=enable_numa,
                    implementation=SharkMicroShortfinProcessSamples,
                    skip_warmup=skip_warmup if not debug else True,
                    init_noise_latent=dataset.init_noise_latent,
                    vmfbs=vmfbs,
                    params=params,
                    fibers_per_device=fibers_per_device,
                    workers_per_device=workers_per_device,
                    num_sample_loops=num_sample_loops,
                    log_sample_gets=log_sample_get,
                    debug=debug
                )

                self.core_jobs.append(job)
                if not self.interrupt:
                    job.start()

            # We are initializing the engines parallel, but making sure that
            # only one engines is getting compiled in each process
            ready_count = len(self.devices)
            while ready_count:
                status, device, core = self.init_queue.get()
                assert (
                    status == 0
                ), f"Something went wrong during init with {device}:{core}"
                self.init_queue.task_done()
                self.verbose_log(
                    f"Server process job is ready for {device}:{core} device"
                )
                ready_count -= 1

        if self.enable_batcher:
            self.verbose_log(f"Server enabling batcher")
            self.batcher_threshold = (
                batch_timeout_threashold  # maximum seconds to form a batch
            )
            self.batcher_queue = JoinableQueue()
            self.batcher_job = Thread(target=self.batch_samples_loop, args=())
            self.batcher_job.start()

        response_thread_count = len(self.core_jobs) if self.use_response_pipes else 1
        for idx in range(response_thread_count):
            response_job = Thread(
                target=self.process_response_loop,
                args=(
                    (
                        self.response_pipes[idx]
                        if self.use_response_pipes
                        else self.response_queue
                    ),
                    save_images,
                    skip_complete,
                ),
                daemon=True,
            )
            response_job.start()
            self.response_jobs.append(response_job)

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

        self.qsl = lg.ConstructQSL(
            dataset.caption_count,
            performance_sample_count,
            dataset.load_query_samples,
            dataset.unload_query_samples,
        )

        self.verbose_log(f"Server init with {devices}x{cores_per_devices} finished")

    @rpd_trace()
    def prepare_service(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, self.model_json)
        flagfile = os.path.join(script_dir, "sdxl_flagfile_gfx942.txt")
        td_spec = (
            os.path.join(script_dir, self.td_spec) if self.td_spec is not None else None
        )
        model_params = ModelParams.load_json(script_path)
        vmfbs, params = get_modules(
            target="gfx942",
            device="amdgpu",
            model_config=script_path,
            artifacts_dir=self.model_weights,
            force_update=self.force_export,
            flagfile=flagfile,
            td_spec=td_spec,
            build_preference="export",
        )
        return None, vmfbs, params

    @rpd_trace()
    def batch_samples_loop(self):

        @rpd_trace()
        def _wait_for_batches():
            return self.batcher_queue.get(timeout=self.batcher_threshold)

        batched_samples = self.batcher_queue.get()
        timeout_stamp = time.time()
        while True:
            if len(batched_samples) != 0 and (
                len(batched_samples) >= self.gpu_batch_size
                or time.time() - timeout_stamp >= self.batcher_threshold
            ):  # max batch or time limit exceed
                self.verbose_log(
                    f"Formed batch of {len(batched_samples[:self.gpu_batch_size])} samples"
                )
                self.sample_queue.put(
                    SampleRequest.create(
                        batched_samples[: self.gpu_batch_size],
                        self.dataset,
                        skip_latents=self.send_latents_once,
                        gpu_batch_size=self.gpu_batch_size,
                    )
                )
                batched_samples = batched_samples[self.gpu_batch_size :]
                timeout_stamp = time.time()

            try:
                samples = _wait_for_batches()
            except QueueEmpty:
                continue

            if samples is None:  # None in the queue indicates the SUT want us to exit
                break
            batched_samples += samples

    @rpd_trace()
    def issue_queries(self, query_samples):
        num_samples = len(query_samples)
        self.verbose_log(f"[Server] Received {num_samples} samples")
        self.sample_count += num_samples
        for i in range(0, num_samples, self.gpu_batch_size):
            # Construct batches
            actual_batch_size = (
                self.gpu_batch_size
                if num_samples - i > self.gpu_batch_size
                else num_samples - i
            )
            if self.enable_batcher:
                self.batcher_queue.put(query_samples[i : i + actual_batch_size])
            else:
                self.sample_queue.put(
                    SampleRequest.create(
                        query_samples[i : i + actual_batch_size],
                        self.dataset,
                        skip_latents=self.send_latents_once,
                        gpu_batch_size=self.gpu_batch_size,
                    )
                )

    @rpd_trace()
    def process_response_loop(
        self, response_comm, save_images=False, skip_complete=False
    ):
        self.verbose_log(f"process response loop started")
        if isinstance(response_comm, multiprocessing.queues.JoinableQueue):
            recv_response = response_comm.get
            task_done = response_comm.task_done
        elif isinstance(response_comm, socket.socket):
            recv_response = response_comm.recv
            task_done = lambda *args, **kwargs: None
        else:
            raise RuntimeError(f"Unsupported response comm {type(response_comm)}")

        @rpd_trace()
        def _wait_for_response():
            return recv_response()

        @rpd_trace()
        def _process_response(response):
            qsr = []
            response_array_refs = []
            actual_batch_size = len(response.sample_ids)
            self.verbose_log(f"Reporting back {actual_batch_size} samples")

            response.generated_images = (
                (
                    response.generated_images.transpose(0, 2, 3, 1).astype(np.float32)
                    * 255
                )
                .clip(0, 255)
                .round()
                .astype(np.uint8)
            )
            self.verbose_log(
                f"{response.generated_images.shape=} {response.generated_images.dtype=}"
            )
            if not skip_complete:
                for idx, sample_id in enumerate(response.sample_ids):
                    response_array = array.array(
                        "B",
                        np.array(response.generated_images[idx], np.uint8).tobytes(),
                    )
                    response_array_refs.append(response_array)
                    bi = response_array.buffer_info()
                    qsr.append(lg.QuerySampleResponse(sample_id, bi[0], bi[1]))

                self.verbose_log(f"Call QuerySamplesComplete...")
                lg.QuerySamplesComplete(qsr)
                self.verbose_log(f"QuerySamplesComplete {actual_batch_size} samples")

            if save_images:
                self.save_response_as_image(response)

        while True:
            response = _wait_for_response()
            if response is None:
                # None in the queue indicates the parent want us to exit
                task_done()
                break
            _process_response(response)
            task_done()
        self.verbose_log(f"process response loop finished")

    @rpd_trace()
    def save_response_as_image(self, response):
        self.verbose_log(f"Save images...")
        from PIL import Image
        import os

        os.makedirs("harness_result_shark", exist_ok=True)
        for idx, sample_index in enumerate(response.sample_indices):
            Image.fromarray(response.generated_images[idx]).save(
                f"harness_result_shark/response_{sample_index}.jpg"
            )
            self.verbose_log(
                f"Image saved to harness_result_shark/response_{sample_index}.jpg"
            )

    @rpd_trace()
    def flush_queries(self):
        pass

    @rpd_trace()
    def finish_test(self):
        # exit all jobs
        self.verbose_log(f"SUT finished!")
        logging.info(f"[Server] Received {self.sample_count} total samples")
        for _ in self.core_jobs:
            for _ in range(self.fibers_per_device):
                self.sample_queue.put(None)
        self.sample_queue.join()
        if self.use_response_pipes:
            for pipe in self.response_pipes:
                pipe.put(None)
        else:
            for _ in self.response_jobs:
                self.response_queue.put(None)
            self.response_queue.join()
        if self.enable_batcher:
            self.batcher_queue.put(None)
            self.batcher_job.join()
        for job in self.core_jobs:
            # self.sample_queue.join()
            job.join()
        for job in self.response_jobs:
            job.join()
