#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: scripts/run_train.sh <config> <train_manifest> <val_manifest> <output_dir> [device]"
  exit 1
fi

CONFIG="$1"
TRAIN_MANIFEST="$2"
VAL_MANIFEST="$3"
OUTPUT_DIR="$4"
DEVICE="${5:-cuda}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python -m atadd.train \
  --config "${CONFIG}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --val-manifest "${VAL_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --seed 42

