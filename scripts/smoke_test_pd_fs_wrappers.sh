#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/.conda/envs/diffmsr/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.conda/envs/diffmsr/bin/python"
  else
    PYTHON_BIN="/opt/miniconda3/bin/python"
  fi
fi
export PYTHON_BIN

GPU_ID="${GPU_ID:-0}"
CPUS="${CPUS:-2}"
DATA_ROOT="${DATA_ROOT:-mri_data_complex/mc_knee_pd_fs}"
RAW_DB="${RAW_DB:-data/knee_mri_clinical_seq_batch2}"
RUN_PREPROCESS_SMOKE="${RUN_PREPROCESS_SMOKE:-auto}"
ALLOW_ARCHIVE_EXTRACT="${ALLOW_ARCHIVE_EXTRACT:-0}"
SMOKE_MAX_STUDIES="${SMOKE_MAX_STUDIES:-2}"
SMOKE_TRAIN_RATIO="${SMOKE_TRAIN_RATIO:-0.5}"
SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-}"
SMOKE_DATA_ROOT="${SMOKE_DATA_ROOT:-/tmp/diffmsr_pd_fs_smoke_dataset_${USER:-user}_$$}"

echo "DiffMSR PD/PD-FS smoke tests"
echo "  PROJECT_ROOT=${PROJECT_ROOT}"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  RAW_DB=${RAW_DB}"
echo "  ALLOW_ARCHIVE_EXTRACT=${ALLOW_ARCHIVE_EXTRACT}"
echo "  SMOKE_MAX_STUDIES=${SMOKE_MAX_STUDIES}"
echo "  SMOKE_TRAIN_RATIO=${SMOKE_TRAIN_RATIO}"

echo
echo "[1/5] Shell syntax checks"
for script in \
  scripts/run_diffmsr_nohup.sh \
  scripts/run_pd_fs_full_tests.sh \
  scripts/run_prior256_100k_pipeline.sh \
  scripts/wrap_prepare_pd_fs_dataset.sh \
  scripts/wrap_train_denoising_steps.sh \
  scripts/wrap_train_latent_size.sh; do
  bash -n "$script"
  echo "  ok: ${script}"
done

echo
echo "[2/5] Help/argument paths"
bash scripts/wrap_prepare_pd_fs_dataset.sh --help >/dev/null
bash scripts/wrap_train_denoising_steps.sh --help >/dev/null
bash scripts/wrap_train_latent_size.sh --help >/dev/null
bash scripts/run_pd_fs_full_tests.sh --help >/dev/null
echo "  ok: help commands"

EFFECTIVE_DATA_ROOT="$DATA_ROOT"

echo
echo "[3/5] Optional tiny preprocessing conversion"
should_preprocess=0
if [[ "$RUN_PREPROCESS_SMOKE" == "1" ]]; then
  should_preprocess=1
elif [[ "$RUN_PREPROCESS_SMOKE" == "auto" && -d "$RAW_DB" ]]; then
  should_preprocess=1
fi

if [[ "$should_preprocess" == "1" && -f "$RAW_DB" && "$ALLOW_ARCHIVE_EXTRACT" != "1" ]]; then
  echo "  skipped: RAW_DB is an archive/file, and extracting it may unpack the whole database"
  echo "  set ALLOW_ARCHIVE_EXTRACT=1 if you intentionally want to test archive extraction"
elif [[ "$should_preprocess" == "1" ]]; then
  preprocess_env=(
    MAX_STUDIES="$SMOKE_MAX_STUDIES"
    TRAIN_RATIO="$SMOKE_TRAIN_RATIO"
    OVERWRITE=1
    PYTHON_BIN="$PYTHON_BIN"
  )
  if [[ -n "$SMOKE_MAX_SAMPLES" ]]; then
    preprocess_env+=(MAX_SAMPLES="$SMOKE_MAX_SAMPLES")
  fi
  env "${preprocess_env[@]}" \
  bash scripts/wrap_prepare_pd_fs_dataset.sh "$RAW_DB" "$SMOKE_DATA_ROOT"
  EFFECTIVE_DATA_ROOT="$SMOKE_DATA_ROOT"
  smoke_train_count="$(find "$EFFECTIVE_DATA_ROOT/train" -maxdepth 1 -type f -name '*.mat' | wc -l)"
  smoke_valid_count="$(find "$EFFECTIVE_DATA_ROOT/valid" -maxdepth 1 -type f -name '*.mat' | wc -l)"
  if [[ "$smoke_train_count" -eq 0 || "$smoke_valid_count" -eq 0 ]]; then
    echo "  failed: smoke preprocessing needs at least one train and one valid .mat file" >&2
    echo "  train=${smoke_train_count}, valid=${smoke_valid_count}" >&2
    echo "  try without SMOKE_MAX_SAMPLES, or increase SMOKE_MAX_STUDIES." >&2
    exit 1
  fi
  echo "  ok: tiny dataset written to ${SMOKE_DATA_ROOT}"
else
  echo "  skipped: set RUN_PREPROCESS_SMOKE=1 and RAW_DB=/path/to/dicoms to enable"
fi

echo
echo "[4/5] Dry-run training wrapper YAML generation"
if [[ -d "$EFFECTIVE_DATA_ROOT/train" && -d "$EFFECTIVE_DATA_ROOT/valid" && -f "$EFFECTIVE_DATA_ROOT/dc_mask.mat" ]]; then
  DRY_RUN=1 \
  DATA_ROOT="$EFFECTIVE_DATA_ROOT" \
  GPU_ID="$GPU_ID" \
  CPUS="$CPUS" \
  PYTHON_BIN="$PYTHON_BIN" \
  STAGE=1 \
  PRIOR_DIM=256 \
  TOTAL_ITER=2 \
  VAL_FREQ=999999 \
  SAVE_FREQ=50000 \
  PRINT_FREQ=100 \
  RUN_NAME=smoke_denoise_shared_stage1 \
  RUN_TAG=smoke_denoise_stage1 \
  bash scripts/run_diffmsr_nohup.sh

  DRY_RUN=1 \
  STAGE1_CKPT=experiments/smoke_denoise_shared_stage1/models/net_g_latest.pth \
  DATA_ROOT="$EFFECTIVE_DATA_ROOT" \
  GPU_ID="$GPU_ID" \
  CPUS="$CPUS" \
  PYTHON_BIN="$PYTHON_BIN" \
  RUN_NAME=smoke_steps2 \
  RUN_TAG=smoke_steps2 \
  bash scripts/wrap_train_denoising_steps.sh "$EFFECTIVE_DATA_ROOT" 2 2 "$GPU_ID"

  DRY_RUN=1 \
  SKIP_TRAIN_IF_EXISTS=0 \
  DATA_ROOT="$EFFECTIVE_DATA_ROOT" \
  GPU_ID="$GPU_ID" \
  CPUS="$CPUS" \
  PYTHON_BIN="$PYTHON_BIN" \
  STAGE1_NAME=smoke_prior64_stage1 \
  STAGE2_NAME=smoke_prior64_stage2 \
  bash scripts/wrap_train_latent_size.sh "$EFFECTIVE_DATA_ROOT" 2 64 "$GPU_ID"

  echo "  ok: train dry-runs"
else
  echo "  skipped: no prepared DATA_ROOT with train/, valid/, and dc_mask.mat"
fi

echo
echo "[5/5] Dry-run full-test YAML generation"
if [[ -d "$EFFECTIVE_DATA_ROOT/valid" && -f "$EFFECTIVE_DATA_ROOT/dc_mask.mat" ]]; then
  DRY_RUN=1 \
  TEST_GROUP=latent256 \
  DATA_ROOT="$EFFECTIVE_DATA_ROOT" \
  GENERATED_OPT_DIR=logs/generated_test_options_smoke \
  LOG_DIR=logs \
  SKIP_EXISTING=0 \
  GPU_ID="$GPU_ID" \
  CPUS="$CPUS" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_pd_fs_full_tests.sh

  VALIDATE_DATA_ROOT="$EFFECTIVE_DATA_ROOT" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import yaml

root = os.environ["VALIDATE_DATA_ROOT"].rstrip("/")

checks = [
    (
        Path("logs/generated_options/smoke_denoise_stage1.yml"),
        lambda opt: opt["name"] == "smoke_denoise_shared_stage1"
        and opt["network_g"]["prior_dim"] == 256
        and opt["train"]["total_iter"] == 2
        and opt["datasets"]["train"]["dataroot_gt"] == f"{root}/train"
        and opt["datasets"]["val_1"]["dataroot_gt"] == f"{root}/valid",
    ),
    (
        Path("logs/generated_options/smoke_steps2.yml"),
        lambda opt: opt["network_g"]["timesteps"] == 2
        and opt["network_g"]["prior_dim"] == 256
        and opt["train"]["total_iter"] == 2
        and opt["datasets"]["train"]["dataroot_gt"] == f"{root}/train"
        and opt["path"]["pretrain_network_g"] == "experiments/smoke_denoise_shared_stage1/models/net_g_latest.pth"
        and opt["path"]["pretrain_network_S1"] == "experiments/smoke_denoise_shared_stage1/models/net_g_latest.pth",
    ),
    (
        Path("logs/generated_options/prior64_stage1_2.yml"),
        lambda opt: opt["network_g"]["prior_dim"] == 64
        and opt["train"]["total_iter"] == 2
        and opt["datasets"]["train"]["dataroot_gt"] == f"{root}/train",
    ),
    (
        Path("logs/generated_options/prior64_stage2_2.yml"),
        lambda opt: opt["network_g"]["prior_dim"] == 64
        and opt["network_S1"]["prior_dim"] == 64
        and opt["network_g"]["timesteps"] == 8
        and opt["train"]["total_iter"] == 2,
    ),
    (
        Path("logs/generated_test_options_smoke/pd_fs_test_prior256_100k.yml"),
        lambda opt: opt["network_g"]["prior_dim"] == 256
        and opt["datasets"]["test_1"]["dataroot_gt"] == f"{root}/valid",
    ),
]

for path, predicate in checks:
    if not path.exists():
        raise SystemExit(f"missing generated YAML: {path}")
    with path.open("r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)
    if not predicate(opt):
        raise SystemExit(f"generated YAML failed validation: {path}")
    print(f"  ok: {path}")
PY
else
  echo "  skipped: no prepared DATA_ROOT with valid/ and dc_mask.mat"
fi

echo
echo "Smoke tests finished."
