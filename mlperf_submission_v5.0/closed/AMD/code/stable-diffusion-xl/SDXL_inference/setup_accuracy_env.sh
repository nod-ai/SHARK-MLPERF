#!/bin/bash

set -x
set -e

python3.11 -m venv accuracy_venv
source accuracy_venv/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/rocm6.1
pip install -r /mlperf/inference/text_to_image/requirements.txt
pip install pandas ijson numpy==1.26.4 # these are missing from the reqs
deactivate
