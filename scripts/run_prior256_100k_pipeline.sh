#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-3}"
CPUS="${CPUS:-4}"
DATA_ROOT="${DATA_ROOT:-mri_data_complex/mc_knee_pd_fs}"
TOTAL_ITER="${TOTAL_ITER:-100000}"
TIMESTEPS="${TIMESTEPS:-8}"
VAL_FREQ="${VAL_FREQ:-999999}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
PRINT_FREQ="${PRINT_FREQ:-100}"
SKIP_TRAIN_IF_EXISTS="${SKIP_TRAIN_IF_EXISTS:-1}"
RUN_TEST="${RUN_TEST:-1}"

STAGE1_NAME="pd_fs_stage1_x4_prior256_100k"
STAGE2_NAME="pd_fs_stage2_x4_prior256_100k"
STAGE1_CKPT="experiments/${STAGE1_NAME}/models/net_g_latest.pth"
STAGE2_CKPT="experiments/${STAGE2_NAME}/models/net_g_latest.pth"

echo "Started prior256 100k pipeline: $(date)"
echo "GPU_ID=${GPU_ID}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "TOTAL_ITER=${TOTAL_ITER}"
echo "TIMESTEPS=${TIMESTEPS}"
echo "VAL_FREQ=${VAL_FREQ}"
echo "SAVE_FREQ=${SAVE_FREQ}"

if [[ "$SKIP_TRAIN_IF_EXISTS" == "1" && -s "$STAGE1_CKPT" ]]; then
  echo "Stage 1 checkpoint exists; skipping Stage 1: ${STAGE1_CKPT}"
else
  echo "Running Stage 1: ${STAGE1_NAME}"
  GPU_ID="$GPU_ID" \
  CPUS="$CPUS" \
  PYTHON_BIN="$PYTHON_BIN" \
  DATA_ROOT="$DATA_ROOT" \
  STAGE=1 \
  PRIOR_DIM=256 \
  TOTAL_ITER="$TOTAL_ITER" \
  VAL_FREQ="$VAL_FREQ" \
  SAVE_FREQ="$SAVE_FREQ" \
  PRINT_FREQ="$PRINT_FREQ" \
  RUN_NAME="$STAGE1_NAME" \
  RUN_TAG="prior256_stage1_100k" \
  bash scripts/run_diffmsr_nohup.sh
fi

if [[ ! -s "$STAGE1_CKPT" ]]; then
  echo "Missing Stage 1 checkpoint after Stage 1: ${STAGE1_CKPT}" >&2
  exit 1
fi

if [[ "$SKIP_TRAIN_IF_EXISTS" == "1" && -s "$STAGE2_CKPT" ]]; then
  echo "Stage 2 checkpoint exists; skipping Stage 2: ${STAGE2_CKPT}"
else
  echo "Running Stage 2: ${STAGE2_NAME}"
  GPU_ID="$GPU_ID" \
  CPUS="$CPUS" \
  PYTHON_BIN="$PYTHON_BIN" \
  DATA_ROOT="$DATA_ROOT" \
  STAGE=2 \
  PRIOR_DIM=256 \
  TIMESTEPS="$TIMESTEPS" \
  PRETRAIN_S1="$STAGE1_CKPT" \
  TOTAL_ITER="$TOTAL_ITER" \
  VAL_FREQ="$VAL_FREQ" \
  SAVE_FREQ="$SAVE_FREQ" \
  PRINT_FREQ="$PRINT_FREQ" \
  RUN_NAME="$STAGE2_NAME" \
  RUN_TAG="prior256_stage2_100k" \
  bash scripts/run_diffmsr_nohup.sh
fi

if [[ ! -s "$STAGE2_CKPT" ]]; then
  echo "Missing Stage 2 checkpoint after Stage 2: ${STAGE2_CKPT}" >&2
  exit 1
fi

if [[ "$RUN_TEST" == "1" ]]; then
  echo "Running full held-out validation/test for prior256 100k."
  GPU_ID="$GPU_ID" \
  CPUS="$CPUS" \
  PYTHON_BIN="$PYTHON_BIN" \
  DATA_ROOT="$DATA_ROOT" \
  TEST_GROUP=latent256 \
  SKIP_EXISTING=0 \
  bash scripts/run_pd_fs_full_tests.sh
else
  echo "RUN_TEST=${RUN_TEST}; skipping final test."
fi

echo "Finished prior256 100k pipeline: $(date)"
