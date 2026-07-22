#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Train a Stage-2 DiffMSR denoising-step ablation.

Usage:
  bash scripts/wrap_train_denoising_steps.sh <dataset_root> <iterations> <denoising_steps> [gpu_id]

Arguments:
  dataset_root      Prepared dataset root with train/, valid/, and dc_mask.mat
  iterations        Stage-2 training iterations, for example 100000
  denoising_steps   Diffusion denoising timesteps, for example 1, 2, 4, 6, or 8
  gpu_id            CUDA_VISIBLE_DEVICES id (default: GPU_ID env or 3)

Environment variables:
  PYTHON_BIN        Python executable                         (default: /opt/miniconda3/bin/python)
  CPUS              CPU threads/workers                       (default: 4)
  PRIOR_DIM         Latent/prior vector size                  (default: 256)
  STAGE1_CKPT       Stage-1 checkpoint to reuse               (default: experiments/pd_fs_stage1_x4/models/net_g_330000.pth)
  TOTAL_ITER_S1     Stage-1 iterations if AUTO_TRAIN_STAGE1=1 (default: iterations)
  AUTO_TRAIN_STAGE1 Train Stage 1 if STAGE1_CKPT is missing   (default: 0)
  VAL_FREQ          Validation interval                       (default: 999999)
  SAVE_FREQ         Checkpoint interval                       (default: 50000)
  PRINT_FREQ        Log interval                              (default: 100)
  DRY_RUN           Generate options without starting training (default: 0)
  RUN_NAME          Optional Stage-2 experiment name override
  RUN_TAG           Optional Stage-2 log/generated-yaml tag

Notes:
  Changing denoising steps changes Stage 2 only. The Stage-1 encoder/decoder
  should stay fixed for a clean timestep ablation.
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
TIMESTEPS="$3"
GPU_ID="${4:-${GPU_ID:-3}}"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/bin/python}"
CPUS="${CPUS:-4}"
PRIOR_DIM="${PRIOR_DIM:-256}"
USER_SET_STAGE1_CKPT="${STAGE1_CKPT+x}"
STAGE1_CKPT="${STAGE1_CKPT:-experiments/pd_fs_stage1_x4/models/net_g_330000.pth}"
AUTO_TRAIN_STAGE1="${AUTO_TRAIN_STAGE1:-0}"
TOTAL_ITER_S1="${TOTAL_ITER_S1:-$TOTAL_ITER}"
VAL_FREQ="${VAL_FREQ:-999999}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
PRINT_FREQ="${PRINT_FREQ:-100}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! "$TOTAL_ITER" =~ ^[0-9]+$ ]] || (( TOTAL_ITER <= 0 )); then
  echo "iterations must be a positive integer: ${TOTAL_ITER}" >&2
  exit 2
fi
if [[ ! "$TIMESTEPS" =~ ^[0-9]+$ ]] || (( TIMESTEPS <= 0 )); then
  echo "denoising_steps must be a positive integer: ${TIMESTEPS}" >&2
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
STAGE1_ITER_TAG="$(iter_tag "$TOTAL_ITER_S1")"
DEFAULT_STAGE1_NAME="pd_fs_stage1_x4_prior${PRIOR_DIM}_${STAGE1_ITER_TAG}"
DEFAULT_STAGE1_CKPT="experiments/${DEFAULT_STAGE1_NAME}/models/net_g_latest.pth"

if [[ ! -s "$STAGE1_CKPT" && -z "$USER_SET_STAGE1_CKPT" && -s "$DEFAULT_STAGE1_CKPT" ]]; then
  echo "Default Stage-1 checkpoint is missing; using existing generated Stage-1 checkpoint:"
  echo "  ${DEFAULT_STAGE1_CKPT}"
  STAGE1_CKPT="$DEFAULT_STAGE1_CKPT"
fi

if [[ ! -s "$STAGE1_CKPT" && "$AUTO_TRAIN_STAGE1" == "1" ]]; then
  echo "Stage-1 checkpoint not found; training shared Stage 1 first."
  DATA_ROOT="$DATA_ROOT" \
  GPU_ID="$GPU_ID" \
  CPUS="$CPUS" \
  PYTHON_BIN="$PYTHON_BIN" \
  STAGE=1 \
  PRIOR_DIM="$PRIOR_DIM" \
  TOTAL_ITER="$TOTAL_ITER_S1" \
  VAL_FREQ="$VAL_FREQ" \
  SAVE_FREQ="$SAVE_FREQ" \
  PRINT_FREQ="$PRINT_FREQ" \
  RUN_NAME="$DEFAULT_STAGE1_NAME" \
  RUN_TAG="prior${PRIOR_DIM}_stage1_${STAGE1_ITER_TAG}" \
  bash scripts/run_diffmsr_nohup.sh
  STAGE1_CKPT="$DEFAULT_STAGE1_CKPT"
fi

if [[ ! -s "$STAGE1_CKPT" && "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; Stage-1 checkpoint is missing but option generation can still be tested:"
  echo "  ${STAGE1_CKPT}"
elif [[ ! -s "$STAGE1_CKPT" ]]; then
  echo "Missing Stage-1 checkpoint: ${STAGE1_CKPT}" >&2
  echo "Either set STAGE1_CKPT=... or rerun with AUTO_TRAIN_STAGE1=1." >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-pd_fs_stage2_x4_steps${TIMESTEPS}_${ITER_TAG}}"
RUN_TAG="${RUN_TAG:-steps${TIMESTEPS}_${ITER_TAG}_gpu${GPU_ID}}"

echo "Training denoising-step ablation"
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  Stage 1 checkpoint=${STAGE1_CKPT}"
echo "  Stage 2 run=${RUN_NAME}"
echo "  iterations=${TOTAL_ITER}"
echo "  denoising steps=${TIMESTEPS}"
echo "  prior_dim=${PRIOR_DIM}"
echo "  GPU_ID=${GPU_ID}"

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
RUN_NAME="$RUN_NAME" \
RUN_TAG="$RUN_TAG" \
bash scripts/run_diffmsr_nohup.sh

echo "Done. Checkpoint:"
echo "  experiments/${RUN_NAME}/models/net_g_latest.pth"
echo "Tail log:"
echo "  tail -f logs/diffmsr_${RUN_TAG}.log"
