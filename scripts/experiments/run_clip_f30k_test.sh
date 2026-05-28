#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/kxh/smx/homework/h2"
OUTPUT_JSON="${1:-${ROOT}/results/clip_flickr30k_test_metrics.json}"

mkdir -p "${ROOT}/results" "${ROOT}/logs"

source /mnt/kxh/miniconda3/bin/activate trl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "${ROOT}"

python scripts/evaluate_clip_flickr30k.py \
  --annotations-csv datasets/flickr30k_hf/flickr_annotations_30k.csv \
  --image-root datasets/flickr30k_images/Images \
  --split test \
  --model-name openai/clip-vit-base-patch32 \
  --batch-size 64 \
  --output-json "${OUTPUT_JSON}"
