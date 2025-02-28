#!/bin/bash

# set -x
set -e

PYTHON3_BIN_PATH=/mlperf/harness/accuracy_venv/bin
PYTHON3_PATH=${PYTHON3_BIN_PATH}/python3.11
ACTIVATE_PATH=${PYTHON3_BIN_PATH}/activate
CAPTIONS_PATH=/data/coco/SDXL/captions/captions.tsv
COCO_SCRIPT_PATH=/mlperf/inference/text_to_image/tools/accuracy_coco.py

if [ ! -f ${PYTHON3_PATH} ]; then
    echo "venv not found, run ./setup_accuracy_env.sh"
    exit 1
fi

if [ ! -f ${CAPTIONS_PATH} ]; then
    echo "captions not found, run ./download_data.sh"
    exit 1
fi

if [ ! -f ${COCO_SCRIPT_PATH} ]; then
    echo "accuracy_coco.py not found"
    exit 1
fi

source ${ACTIVATE_PATH}

if [ ${PYTHON3_PATH} != `which python3.11` ]; then
    echo "incorrect python3.11 is used"
    exit 1
fi

ACCURACY_JSON=${1}

if [ -z ${ACCURACY_JSON} ]; then
    echo "incorrect accuracy path, set it with ${0} <path>"
    deactivate
    exit 1
fi

if [ ! -f ${ACCURACY_JSON} ]; then
    echo "incorrect accuracy path, set it with ${0} <path>"
    deactivate
    exit 1
fi

OUTPUT_DIR=$(dirname ${ACCURACY_JSON})
COMPLIANCE_IMAGE_DIR=${OUTPUT_DIR}/images/
RESULT_JSON_PATH=${OUTPUT_DIR}/coco-results.json
RESULT_TXT=${OUTPUT_DIR}/accuracy.txt

python3.11 ${COCO_SCRIPT_PATH} --mlperf-accuracy-file ${ACCURACY_JSON}          \
                            --caption-path ${CAPTIONS_PATH} --verbose        \
                            --compliance-images-path ${COMPLIANCE_IMAGE_DIR} \
                            --device gpu                                     \
                            --output-file ${RESULT_JSON_PATH} > ${RESULT_TXT}

deactivate

echo "Check $RESULT_JSON_PATH for the accuracy scores"
echo "Check $COMPLIANCE_IMAGE_DIR for the compliance images"
