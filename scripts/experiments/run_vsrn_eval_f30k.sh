#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/kxh/smx/homework/h2"
VSRN_DIR="${ROOT}/VSRN"
DATA_PATH="${ROOT}/datasets/scan/data"
MODEL_PATH="${1:-${VSRN_DIR}/runs/flickr_VSRN/model_best.pth.tar}"

source /mnt/kxh/miniconda3/bin/activate vsrn_py37
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "${VSRN_DIR}"

python - <<PY
from vocab import Vocabulary
import evaluation

evaluation.evalrank(
    "${MODEL_PATH}",
    data_path="${DATA_PATH}",
    split="test",
    fold5=False,
)
PY
