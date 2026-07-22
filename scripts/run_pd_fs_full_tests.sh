#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/bin/python}"
GPU_ID="${GPU_ID:-3}"
CPUS="${CPUS:-4}"
TEST_GROUP="${TEST_GROUP:-latent}"
DATA_ROOT="${DATA_ROOT:-mri_data_complex/mc_knee_pd_fs}"
BASE_OPT="${BASE_OPT:-options/test_DiffMSR_S2_x4_pd_fs_full_one.yml}"
GENERATED_OPT_DIR="${GENERATED_OPT_DIR:-logs/generated_test_options}"
LOG_DIR="${LOG_DIR:-logs}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
BASELINE_STAGE1_CKPT="${BASELINE_STAGE1_CKPT:-experiments/pd_fs_stage1_x4/models/net_g_330000.pth}"
BASELINE_STAGE2_CKPT="${BASELINE_STAGE2_CKPT:-experiments/pd_fs_stage2_x4/models/net_g_latest.pth}"
TIMESTEP_STAGE1_CKPT="${TIMESTEP_STAGE1_CKPT:-$BASELINE_STAGE1_CKPT}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export NUMEXPR_NUM_THREADS="$CPUS"
export VECLIB_MAXIMUM_THREADS="$CPUS"

usage() {
  cat <<'EOF'
Run full held-out PD/PD-FS tests sequentially.

Environment variables:
  TEST_GROUP      latent | latent256 | baseline | timesteps | all   (default: latent)
  DATA_ROOT       prepared dataset root with valid/ and dc_mask.mat  (default: mri_data_complex/mc_knee_pd_fs)
  GPU_ID          visible GPU id                         (default: 3)
  PYTHON_BIN      Python executable                       (default: /opt/miniconda3/bin/python)
  SKIP_EXISTING   skip result folders with test logs       (default: 1)
  DRY_RUN         generate test YAMLs without running test.py
  BASELINE_STAGE1_CKPT  Stage-1 checkpoint for baseline/timestep tests
  BASELINE_STAGE2_CKPT  Stage-2 checkpoint for baseline test
  TIMESTEP_STAGE1_CKPT  Shared Stage-1 checkpoint for timestep tests

Examples:
  TEST_GROUP=latent GPU_ID=3 bash scripts/run_pd_fs_full_tests.sh
  TEST_GROUP=all GPU_ID=3 bash scripts/run_pd_fs_full_tests.sh

Notes:
  - The generated YAML files are written to logs/generated_test_options/.
  - This repository's validation code saves one large .mat per full valid run.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "$GENERATED_OPT_DIR" "$LOG_DIR"

if [[ ! -d "$DATA_ROOT/valid" || ! -f "$DATA_ROOT/dc_mask.mat" ]]; then
  echo "DATA_ROOT must contain valid/ and dc_mask.mat: ${DATA_ROOT}" >&2
  exit 1
fi

run_test() {
  local name="$1"
  local prior_dim="$2"
  local timesteps="$3"
  local stage1_ckpt="$4"
  local stage2_ckpt="$5"

  local opt_path="${GENERATED_OPT_DIR}/${name}.yml"
  local test_log="${LOG_DIR}/test_${name}.log"
  local result_dir="results/${name}"

  if [[ "$SKIP_EXISTING" == "1" ]] && find "$result_dir" -maxdepth 1 -type f -name "test_${name}_*.log" 2>/dev/null | grep -q .; then
    echo "Skipping existing test result: ${result_dir}"
    return 0
  fi

  BASE_OPT="$BASE_OPT" \
  OUT_OPT="$opt_path" \
  TEST_NAME="$name" \
  PRIOR_DIM="$prior_dim" \
  TIMESTEPS="$timesteps" \
  PRETRAIN_S1="$stage1_ckpt" \
  PRETRAIN_S2="$stage2_ckpt" \
  DATA_ROOT="$DATA_ROOT" \
  "$PYTHON_BIN" - <<'PY'
import os
import yaml

with open(os.environ["BASE_OPT"], "r", encoding="utf-8") as f:
    opt = yaml.safe_load(f)

name = os.environ["TEST_NAME"]
prior_dim = int(os.environ["PRIOR_DIM"])
timesteps = int(os.environ["TIMESTEPS"])
data_root = os.environ["DATA_ROOT"].rstrip("/")

opt["name"] = name

dataset = opt["datasets"]["test_1"]
dataset["name"] = "KneePDHeldout"
dataset["dataroot_gt"] = f"{data_root}/valid"
dataset["dataroot_lq"] = f"{data_root}/valid"
dataset["dataroot_mask"] = f"{data_root}/dc_mask.mat"
dataset.pop("meta_info", None)

opt["network_g"]["prior_dim"] = prior_dim
opt["network_g"]["timesteps"] = timesteps
opt["network_g"]["sample_timesteps"] = timesteps
opt["network_g"]["sample_timestep_mode"] = "uniform"

opt["network_S1"]["prior_dim"] = prior_dim

opt["path"]["pretrain_network_S1"] = os.environ["PRETRAIN_S1"]
opt["path"]["pretrain_network_g"] = os.environ["PRETRAIN_S2"]
opt["path"]["strict_load_g"] = False
opt["path"]["ignore_resume_networks"] = "network_S1"

opt.setdefault("val", {})["save_img"] = False
opt["val"]["pbar"] = True

with open(os.environ["OUT_OPT"], "w", encoding="utf-8") as f:
    yaml.safe_dump(opt, f, sort_keys=False)
PY

  echo "============================================================" | tee -a "$test_log"
  echo "Started: $(date)" | tee -a "$test_log"
  echo "Name: ${name}" | tee -a "$test_log"
  echo "Prior dim: ${prior_dim}" | tee -a "$test_log"
  echo "Timesteps: ${timesteps}" | tee -a "$test_log"
  echo "Stage 1: ${stage1_ckpt}" | tee -a "$test_log"
  echo "Stage 2: ${stage2_ckpt}" | tee -a "$test_log"
  echo "Data root: ${DATA_ROOT}" | tee -a "$test_log"
  echo "Options: ${opt_path}" | tee -a "$test_log"
  echo "GPU_ID: ${GPU_ID}" | tee -a "$test_log"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1; generated options only. Test was not started." | tee -a "$test_log"
    return 0
  fi

  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" -u DiffMSR_Main/test.py \
    -opt "$opt_path" \
    --launcher none \
    2>&1 | tee -a "$test_log"

  echo "Finished: $(date)" | tee -a "$test_log"
}

run_baseline() {
  run_test \
    "pd_fs_test_baseline_prior256_500k" \
    256 \
    8 \
    "$BASELINE_STAGE1_CKPT" \
    "$BASELINE_STAGE2_CKPT"
}

run_latent() {
  for dim in 64 128 256 512; do
    run_test \
      "pd_fs_test_prior${dim}_100k" \
      "$dim" \
      8 \
      "experiments/pd_fs_stage1_x4_prior${dim}_100k/models/net_g_latest.pth" \
      "experiments/pd_fs_stage2_x4_prior${dim}_100k/models/net_g_latest.pth"
  done
}

run_latent256() {
  run_test \
    "pd_fs_test_prior256_100k" \
    256 \
    8 \
    "experiments/pd_fs_stage1_x4_prior256_100k/models/net_g_latest.pth" \
    "experiments/pd_fs_stage2_x4_prior256_100k/models/net_g_latest.pth"
}

run_timesteps() {
  for steps in 1 2 4 6 8; do
    run_test \
      "pd_fs_test_steps${steps}_100k" \
      256 \
      "$steps" \
      "$TIMESTEP_STAGE1_CKPT" \
      "experiments/pd_fs_stage2_x4_steps${steps}_100k/models/net_g_latest.pth"
  done
}

case "$TEST_GROUP" in
  baseline)
    run_baseline
    ;;
  latent)
    run_latent
    ;;
  latent256)
    run_latent256
    ;;
  timesteps)
    run_timesteps
    ;;
  all)
    run_baseline
    run_latent
    run_timesteps
    ;;
  *)
    echo "Unknown TEST_GROUP=${TEST_GROUP}" >&2
    usage >&2
    exit 2
    ;;
esac
