import numpy as np
from time import sleep

from base_process_samples import BaseProcessSamples
from sample_processor import SampleResponse
from utilities import rpd_trace

# 1 second delay by default
DELAY_MS = 1000


# TODO: Refactor to receive the mock parameters
class MockProcessSamples(BaseProcessSamples):

    @rpd_trace()
    def get_asdevicearray():
        return None

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
        return [
            {
                "size": gpu_batch_size,
            }
        ]

    @rpd_trace()
    def warmup(
        device_id,
        core_id,
        pipelines,
        model_weights,
        verbose_log,
    ):
        pass

    @rpd_trace()
    def generate_images(
        device_id,
        core_id,
        pipelines,
        request,
        verbose_log,
    ):
        actual_batch_size = len(request.sample_indices)
        pipeline_batch_size = next(iter(pipelines))["size"]
        verbose_log(
            f"got {actual_batch_size} samples for {pipeline_batch_size} pipeline"
        )
        response = SampleResponse(
            sample_ids=request.sample_ids,
            sample_indices=request.sample_indices,
            generated_images=np.random.rand(pipeline_batch_size, 3, 1024, 1024).astype(
                np.float16
            ),
        )

        # TODO: Not precise, it is including the time of creating the response!
        verbose_log(f"response delayed with {DELAY_MS}ms")
        sleep(DELAY_MS / 1000)
        verbose_log(f"delay done")
        return response
