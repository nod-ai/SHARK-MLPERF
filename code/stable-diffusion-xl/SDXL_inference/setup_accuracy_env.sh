#!/bin/bash

set -x
set -e

python3.11 -m venv accuracy_venv
source accuracy_venv/bin/activate
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/rocm6.1
python -m pip install -r /mlperf/inference/text_to_image/requirements.txt
python -m pip install pandas ijson numpy==1.26.4 py-libnuma # these are missing from the reqs
deactivate