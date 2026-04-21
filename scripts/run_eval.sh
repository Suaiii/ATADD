#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: scripts/run_eval.sh <config> <checkpoint> <manifest> <output_dir> [device]"
  exit 1
fi

CONFIG="$1"
CHECKPOINT="$2"
MANIFEST="$3"
OUTPUT_DIR="$4"
DEVICE="${5:-cuda}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python -m atadd.eval \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}"

