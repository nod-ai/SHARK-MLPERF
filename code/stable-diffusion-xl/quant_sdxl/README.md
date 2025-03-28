# SDXL Quantization
The following documentation describes additional options for SDXL quantization. This documentation assumes the user has created the `mlperf_rocm_sdxl:quant` container described in the [Quantization](../README.md/#quantization) section, and has completed data and model pull operations, up to the point of executing quantization. Documentation starts from a position _within_ the running quantization container.


### Fetching and Pre-Processing Datasets
Information on fetching and pre-processing the data can be found [here](https://github.com/mlcommons/inference/tree/master/text_to_image).
Note, the latents generated here will be used in the next step.

### Quantize Int8 / FP16 Model
```bash
python quant_sdxl.py --model stabilityai/stable-diffusion-xl-base-1.0 --device <device> --calibration-prompt-path ./captions.tsv --checkpoint-name unet.ckpt  --path-to-latents <path/to/latents/latents.pt> --guidance-scale 8 --exclude-blacklist-act-eq [--path-to-coco <path/to/coco> --validation-prompts 5000]
```

### Quantize Int8 / FP8 Model
```bash
python quant_sdxl.py --model stabilityai/stable-diffusion-xl-base-1.0 --device <device> --calibration-prompt-path ./captions.tsv --checkpoint-name unet.ckpt  --path-to-latents <path/to/latents/latents.pt> --guidance-scale 8 --quantize-sdp --exclude-blacklist-act-eq [--path-to-coco <path/to/coco> --validation-prompts 5000]
```
