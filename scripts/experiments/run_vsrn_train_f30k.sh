#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/kxh/smx/homework/h2"
VSRN_DIR="${ROOT}/VSRN"
DATA_PATH="${ROOT}/datasets/scan/data"
LOG_DIR="${ROOT}/logs"
RUN_NAME="${1:-flickr_VSRN}"
shift || true

mkdir -p "${LOG_DIR}" "${VSRN_DIR}/runs"

source /mnt/kxh/miniconda3/bin/activate trl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DATA_PATH

cd "${VSRN_DIR}"

python train.py \
  --data_path "${DATA_PATH}" \
  --data_name f30k_precomp \
  --logger_name "runs/${RUN_NAME}" \
  --max_violation \
  --lr_update 10 \
  --max_len 60 \
  "$@"
