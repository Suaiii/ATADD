#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: scripts/run_predict.sh <config> <checkpoint> <manifest_or_audio_dir> <output_dir> [device] [mode]"
  echo "mode: manifest (default) or audio-dir"
  exit 1
fi

CONFIG="$1"
CHECKPOINT="$2"
INPUT_PATH="$3"
OUTPUT_DIR="$4"
DEVICE="${5:-cuda}"
MODE="${6:-manifest}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

ARGS=(
  -m atadd.predict
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
)

if [ "${MODE}" = "audio-dir" ]; then
  ARGS+=(--audio-dir "${INPUT_PATH}")
else
  ARGS+=(--manifest "${INPUT_PATH}")
fi

python "${ARGS[@]}"
