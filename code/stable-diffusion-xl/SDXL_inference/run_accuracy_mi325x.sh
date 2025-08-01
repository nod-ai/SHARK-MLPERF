#!/bin/bash
shopt -s expand_aliases
source ~/.bash_aliases
set -euxo pipefail
RESULT_DIR="/mlperf/harness/testSubmission/"
SCENARIO="Offline"
BATCH_SIZE=32
COUNT=0
QPS=17
FPD=1
CPD=1
SYSTEM_CONFIG_ID="8xMI325x_2xEPYC-9655"

# constants
OUTPUT_ROOT=$RESULT_DIR/closed/AMD
DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63"

function run_scenario {
	# cleanup audit.config just in case
	if test -f "audit.config"; then
		rm -v audit.config
	fi
	

	RESULTS_ROOT=${OUTPUT_ROOT}/results/${SYSTEM_CONFIG_ID}/stable-diffusion-xl
	COMP_ROOT=${OUTPUT_ROOT}/compliance/${SYSTEM_CONFIG_ID}/stable-diffusion-xl
	echo "Run $SCENARIO accuracy test"

	ROCR_VISIBLE_DEVICES=$DEVICES HIP_VISIBLE_DEVICES=$DEVICES python harness.py \
		--devices "$DEVICES" \
		--gpu_batch_size $BATCH_SIZE \
		--cores_per_devices $CPD \
		--fibers_per_device $FPD \
		--scenario ${SCENARIO} \
		--count 5000 \
		--test_mode AccuracyOnly \
		--logfile_outdir ${RESULTS_ROOT}/${SCENARIO}/accuracy \
  		--vae_batch_size 1 \
		--model_json=sdxl_config_fp8_sched_unet_bs$BATCH_SIZE.json 

	echo "Finished accuracy test."

	./setup_accuracy_env.sh
	./check_accuracy_scores.sh ${RESULTS_ROOT}/${SCENARIO}/accuracy/mlperf_log_accuracy.json
	
  echo "Required scores: FID ∈ (23.0108, 23.9501), CLIP ∈ (31.686, 31.813)"
}

# TODO: this could be updated to run both scenarios but one is enough for now
# NOTE:  as of 2/27 each scenario takes about an hour to run
START_TS=$(date +%Y%m%dT%H%M%S)
echo "$START_TS started $SCENARIO $SYSTEM_CONFIG_ID"
run_scenario

DONE_TS=$(date +%Y%m%dT%H%M%S)
echo "$DONE_TS Completed $SCENARIO ( start at $START_TS )"
