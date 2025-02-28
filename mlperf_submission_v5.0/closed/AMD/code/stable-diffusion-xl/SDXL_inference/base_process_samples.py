from abc import ABC, abstractmethod


class BaseProcessSamples(ABC):

    @staticmethod
    @abstractmethod
    def get_asdevicearray():
        raise NotImplementedError("BaseProcessSamples create function not implemented")

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError("BaseProcessSamples create function not implemented")

    @staticmethod
    @abstractmethod
    def warmup(
        device_id,
        core_id,
        pipelines,
        model_weights,
        verbose_log,
    ):
        raise NotImplementedError("BaseProcessSamples warmup function not implemented")

    @staticmethod
    @abstractmethod
    def generate_images(
        device_id,
        core_id,
        pipelines,
        request,
        verbose_log,
    ):
        raise NotImplementedError(
            "BaseProcessSamples generate_images function not implemented"
        )
