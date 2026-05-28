#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/kxh/smx/homework/h2"

source /mnt/kxh/miniconda3/bin/activate trl
export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "${ROOT}"

python scripts/inspect_flickr30k.py \
  --annotations-csv datasets/flickr30k_hf/flickr_annotations_30k.csv \
  --image-root datasets/flickr30k_images/Images
