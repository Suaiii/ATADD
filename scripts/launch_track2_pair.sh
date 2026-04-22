#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: scripts/launch_track2_pair.sh <train_manifest> <val_manifest> <output_root>"
  exit 1
fi

TRAIN_MANIFEST="$1"
VAL_MANIFEST="$2"
OUTPUT_ROOT="$3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

case "${TRAIN_MANIFEST}" in
  /*) ;;
  *) TRAIN_MANIFEST="${ROOT_DIR}/${TRAIN_MANIFEST}" ;;
esac

case "${VAL_MANIFEST}" in
  /*) ;;
  *) VAL_MANIFEST="${ROOT_DIR}/${VAL_MANIFEST}" ;;
esac

case "${OUTPUT_ROOT}" in
  /*) ;;
  *) OUTPUT_ROOT="${ROOT_DIR}/${OUTPUT_ROOT}" ;;
esac

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

launch_job() {
  local gpu_id="$1"
  local config_path="$2"
  local run_name="$3"
  local timestamp
  timestamp="$(date +"%Y%m%d_%H%M%S")"
  local output_dir="${OUTPUT_ROOT}/${run_name}"
  local log_path="${LOG_DIR}/${run_name}_${timestamp}.log"

  nohup bash -lc "
    set -euo pipefail
    cd '${ROOT_DIR}'
    CUDA_VISIBLE_DEVICES='${gpu_id}' bash scripts/run_train.sh \
      '${config_path}' \
      '${TRAIN_MANIFEST}' \
      '${VAL_MANIFEST}' \
      '${output_dir}' \
      cuda
    CUDA_VISIBLE_DEVICES='${gpu_id}' bash scripts/run_eval.sh \
      '${config_path}' \
      '${output_dir}/best.pt' \
      '${VAL_MANIFEST}' \
      '${output_dir}' \
      cuda
  " > "${log_path}" 2>&1 &

  echo "[launched] ${run_name}"
  echo "  gpu=${gpu_id}"
  echo "  config=${config_path}"
  echo "  output=${output_dir}"
  echo "  log=${log_path}"
}

launch_job 0 "configs/baselines/wavlm_base.yaml" "wavlm_base"
launch_job 1 "configs/baselines/xlsr_base.yaml" "xlsr_base"

echo
echo "Use the following commands to monitor progress:"
echo "  tail -f ${LOG_DIR}/wavlm_base_*.log"
echo "  tail -f ${LOG_DIR}/xlsr_base_*.log"
echo "  ps -ef | grep 'atadd.train' | grep -v grep"
