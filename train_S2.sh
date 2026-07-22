#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
OPT="${OPT:-options/pd_fs_ablation/train/baseline_stage2_prior256_500k.yml}"

usage() {
  cat <<'EOF'
Run DiffMSR Stage 2 with one checked-in YAML file.

Environment variables:
  OPT         training YAML to run
              default: options/pd_fs_ablation/train/baseline_stage2_prior256_500k.yml
  GPU_ID      visible GPU id, default: 0
  PYTHON_BIN  Python executable, default: python

Examples:
  GPU_ID=3 PYTHON_BIN=/opt/miniconda3/bin/python bash train_S2.sh
  OPT=options/pd_fs_ablation/train/timesteps_steps4_stage2_100k.yml bash train_S2.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --opt)
      OPT="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$OPT" ]]; then
  echo "Missing YAML: $OPT" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Stage 2 options: $OPT"
echo "GPU_ID: $GPU_ID"
echo "Python: $PYTHON_BIN"

exec "$PYTHON_BIN" -u DiffMSR_Main/train.py -opt "$OPT" --launcher none
