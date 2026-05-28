#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/kxh/smx/homework/h2"
EXP_DIR="${ROOT}/scripts/experiments"
LOG_DIR="${ROOT}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

LOG_FILE="${LOG_DIR}/${STAMP}_all_experiments_gpu${CUDA_VISIBLE_DEVICES}.log"

echo "Logs will be written to ${LOG_FILE}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

nohup bash -c "
set -euo pipefail
export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'
export PYTHONUNBUFFERED=1
echo '[1/4] data check'
bash '${EXP_DIR}/run_data_check.sh'
echo '[2/4] CLIP zero-shot Flickr30k test'
bash '${EXP_DIR}/run_clip_f30k_test.sh' '${ROOT}/results/clip_flickr30k_test_metrics.json'
echo '[3/4] VSRN Flickr30k training'
bash '${EXP_DIR}/run_vsrn_train_f30k.sh' flickr_VSRN
echo '[4/4] VSRN Flickr30k test evaluation'
bash '${EXP_DIR}/run_vsrn_eval_f30k.sh' '${ROOT}/VSRN/runs/flickr_VSRN/model_best.pth.tar'
echo 'all experiments finished'
" > "${LOG_FILE}" 2>&1 &

echo "all_experiments pid=$! log=${LOG_FILE}"
