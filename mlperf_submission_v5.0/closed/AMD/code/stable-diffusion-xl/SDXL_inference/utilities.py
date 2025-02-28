# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cuda import cuda, cudart
import logging
import time
from functools import wraps
import copy
import numpy as np

import os

ENABLE_TRACING = bool(int(os.getenv("ENABLE_TRACING", 0)))

sample_request = {
    "prompt": None,
    "neg_prompt": None,
    "height": [1024],
    "width": [1024],
    "steps": 20,
    "guidance_scale": 8,
    "seed": [0],
    "output_type": ["base64"],
    "rid": ["string"],
}


class ArgHolder:
    def __init__(
        self,
        prompt,
        neg_prompt,
        height,
        width,
        steps,
        guidance_scale,
        seed,
        output_type,
        rid,
        input_ids,
        sample,
    ):
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.height = height
        self.width = width
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.output_type = output_type
        self.rid = rid
        self.input_ids = input_ids
        self.sample = sample


class CONFIG:
    GUIDANCE = 8
    STEPS = 20
    MAX_PROMPT_LENGTH = 64
    SCHEDULER = "EulerDiscrete"


# tracing helper
class rpd_trace:

    def __init__(self, name="", args=None, skip=False):
        self.skip = skip
        if ENABLE_TRACING and not self.skip:
            from rpdTracerControl import rpdTracerControl

            self.rpd = rpdTracerControl()
            self.name = name
            self.args = args if args else ""

    def _recreate_cm(self):
        return self

    def __call__(self, func):
        if ENABLE_TRACING and not self.skip:
            if self.name:
                self.name += f":{func.__name__}"
            else:
                self.name = f"{func.__qualname__}"

            @wraps(func)
            def inner(*args, **kwds):
                with self._recreate_cm():
                    return func(*args, **kwds)

            return inner
        return func

    def __enter__(self):
        if ENABLE_TRACING and not self.skip:
            self.rpd.start()
            self.rpd.rangePush("python", f"{self.name}", f"{self.args}")
        return self

    def __exit__(self, *exc):
        if ENABLE_TRACING and not self.skip:
            self.rpd.rangePop()
            self.rpd.stop()
        return False


def gen_input_ids(request, batch_size=1, implementation=None):
    prompt_tokens_clip1 = request.prompt_tokens_clip1
    prompt_tokens_clip2 = request.prompt_tokens_clip2
    negative_prompt_tokens_clip1 = request.negative_prompt_tokens_clip1
    negative_prompt_tokens_clip2 = request.negative_prompt_tokens_clip2
    assert prompt_tokens_clip1.shape[1] == CONFIG.MAX_PROMPT_LENGTH

    data = copy.deepcopy(sample_request)
    prompt_tokens = [prompt_tokens_clip1.tolist(), prompt_tokens_clip2.tolist()]
    negative_prompt_tokens = [
        negative_prompt_tokens_clip1.tolist(),
        negative_prompt_tokens_clip2.tolist(),
    ]
    data["input_ids"] = [[]]
    for arr in [*prompt_tokens, *negative_prompt_tokens]:
        data["input_ids"][0].append(np.asarray(arr, dtype=np.int64))

    if request.init_noise_latent is not None:
        data["sample"] = request.init_noise_latent
    return data


# measurement helper
def measure(fn):

    @wraps(fn)
    def measure_ms(*args, **kwargs):
        logger = kwargs.get("verbose_log", print)
        start_time = time.perf_counter_ns()
        result = fn(*args, **kwargs)
        end_time = time.perf_counter_ns()
        logger(
            f"Elapsed time for {fn.__qualname__}: {(end_time - start_time) * 1e-6:.4f} ms",
        )
        return result

    return measure_ms
