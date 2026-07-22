#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Train a DiffMSR latent/prior-size ablation.

Usage:
  bash scripts/wrap_train_latent_size.sh <dataset_root> <iterations> <prior_dim> [gpu_id]

Arguments:
  dataset_root   Prepared dataset root with train/, valid/, and dc_mask.mat
  iterations     Training iterations for each stage, for example 100000
  prior_dim      Latent/prior vector size, for example 64, 128, 256, or 512
  gpu_id         CUDA_VISIBLE_DEVICES id (default: GPU_ID env or 3)

Environment variables:
  PYTHON_BIN             Python executable                       (default: /opt/miniconda3/bin/python)
  CPUS                   CPU threads/workers                     (default: 4)
  TIMESTEPS              Stage-2 diffusion steps                 (default: 8)
  VAL_FREQ               Validation interval                     (default: 999999)
  SAVE_FREQ              Checkpoint interval                     (default: 50000)
  PRINT_FREQ             Log interval                            (default: 100)
  DRY_RUN                Generate options without starting train  (default: 0)
  SKIP_TRAIN_IF_EXISTS   Reuse existing net_g_latest checkpoints (default: 1)
  STAGE1_NAME            Optional Stage-1 experiment name override
  STAGE2_NAME            Optional Stage-2 experiment name override

Notes:
  Changing prior_dim changes the Stage-1 transformer/latent interface, so this
  wrapper trains Stage 1 and then trains Stage 2 from the matching Stage-1
  checkpoint.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 3 ]]; then
  usage
  exit 2
fi

DATA_ROOT="$1"
TOTAL_ITER="$2"
PRIOR_DIM="$3"
GPU_ID="${4:-${GPU_ID:-3}}"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/bin/python}"
CPUS="${CPUS:-4}"
TIMESTEPS="${TIMESTEPS:-8}"
VAL_FREQ="${VAL_FREQ:-999999}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
PRINT_FREQ="${PRINT_FREQ:-100}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_TRAIN_IF_EXISTS="${SKIP_TRAIN_IF_EXISTS:-1}"

if [[ ! "$TOTAL_ITER" =~ ^[0-9]+$ ]] || (( TOTAL_ITER <= 0 )); then
  echo "iterations must be a positive integer: ${TOTAL_ITER}" >&2
  exit 2
fi
if [[ ! "$PRIOR_DIM" =~ ^[0-9]+$ ]] || (( PRIOR_DIM <= 0 )); then
  echo "prior_dim must be a positive integer: ${PRIOR_DIM}" >&2
  exit 2
fi
if [[ ! "$TIMESTEPS" =~ ^[0-9]+$ ]] || (( TIMESTEPS <= 0 )); then
  echo "TIMESTEPS must be a positive integer: ${TIMESTEPS}" >&2
  exit 2
fi
if [[ ! -d "$DATA_ROOT/train" || ! -d "$DATA_ROOT/valid" || ! -f "$DATA_ROOT/dc_mask.mat" ]]; then
  echo "dataset_root must contain train/, valid/, and dc_mask.mat: ${DATA_ROOT}" >&2
  exit 2
fi

iter_tag() {
  local value="$1"
  if (( value % 1000 == 0 )); then
    echo "$((value / 1000))k"
  else
    echo "$value"
  fi
}

ITER_TAG="$(iter_tag "$TOTAL_ITER")"
STAGE1_NAME="${STAGE1_NAME:-pd_fs_stage1_x4_prior${PRIOR_DIM}_${ITER_TAG}}"
STAGE2_NAME="${STAGE2_NAME:-pd_fs_stage2_x4_prior${PRIOR_DIM}_${ITER_TAG}}"
STAGE1_CKPT="experiments/${STAGE1_NAME}/models/net_g_latest.pth"
STAGE2_CKPT="experiments/${STAGE2_NAME}/models/net_g_latest.pth"

echo "Training latent-size ablation"
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  Stage 1 run=${STAGE1_NAME}"
echo "  Stage 2 run=${STAGE2_NAME}"
echo "  iterations per stage=${TOTAL_ITER}"
echo "  prior_dim=${PRIOR_DIM}"
echo "  timesteps=${TIMESTEPS}"
echo "  GPU_ID=${GPU_ID}"

if [[ "$SKIP_TRAIN_IF_EXISTS" == "1" && -s "$STAGE1_CKPT" ]]; then
  echo "Stage 1 checkpoint exists; skipping Stage 1: ${STAGE1_CKPT}"
else
  DATA_ROOT="$DATA_ROOT" \
  GPU_ID="$GPU_ID" \
  CPUS="$CPUS" \
  PYTHON_BIN="$PYTHON_BIN" \
  STAGE=1 \
  PRIOR_DIM="$PRIOR_DIM" \
  TOTAL_ITER="$TOTAL_ITER" \
  VAL_FREQ="$VAL_FREQ" \
  SAVE_FREQ="$SAVE_FREQ" \
  PRINT_FREQ="$PRINT_FREQ" \
  RUN_NAME="$STAGE1_NAME" \
  RUN_TAG="prior${PRIOR_DIM}_stage1_${ITER_TAG}" \
  bash scripts/run_diffmsr_nohup.sh
fi

if [[ ! -s "$STAGE1_CKPT" && "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; Stage-1 checkpoint was not created, continuing to Stage-2 option generation."
elif [[ ! -s "$STAGE1_CKPT" ]]; then
  echo "Missing Stage-1 checkpoint after Stage 1: ${STAGE1_CKPT}" >&2
  exit 1
fi

if [[ "$SKIP_TRAIN_IF_EXISTS" == "1" && -s "$STAGE2_CKPT" ]]; then
  echo "Stage 2 checkpoint exists; skipping Stage 2: ${STAGE2_CKPT}"
else
  DATA_ROOT="$DATA_ROOT" \
  GPU_ID="$GPU_ID" \
  CPUS="$CPUS" \
  PYTHON_BIN="$PYTHON_BIN" \
  STAGE=2 \
  PRIOR_DIM="$PRIOR_DIM" \
  TIMESTEPS="$TIMESTEPS" \
  PRETRAIN_S1="$STAGE1_CKPT" \
  TOTAL_ITER="$TOTAL_ITER" \
  VAL_FREQ="$VAL_FREQ" \
  SAVE_FREQ="$SAVE_FREQ" \
  PRINT_FREQ="$PRINT_FREQ" \
  RUN_NAME="$STAGE2_NAME" \
  RUN_TAG="prior${PRIOR_DIM}_stage2_${ITER_TAG}" \
  bash scripts/run_diffmsr_nohup.sh
fi

echo "Done. Checkpoints:"
echo "  ${STAGE1_CKPT}"
echo "  ${STAGE2_CKPT}"
echo "Tail logs:"
echo "  tail -f logs/diffmsr_prior${PRIOR_DIM}_stage1_${ITER_TAG}.log"
echo "  tail -f logs/diffmsr_prior${PRIOR_DIM}_stage2_${ITER_TAG}.log"
