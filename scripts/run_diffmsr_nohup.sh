#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

STAGE="${STAGE:-2}"
GPU_ID="${GPU_ID:-1}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/bin/python}"
CPUS="${CPUS:-4}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_TAG="${RUN_TAG:-stage${STAGE}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/diffmsr_${RUN_TAG}.log}"
TOTAL_ITER="${TOTAL_ITER:-}"
TIMESTEPS="${TIMESTEPS:-}"
VAL_FREQ="${VAL_FREQ:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
PRINT_FREQ="${PRINT_FREQ:-}"
RUN_NAME="${RUN_NAME:-}"

if [[ "$STAGE" == "1" ]]; then
  OPT="options/train_DiffMSR_S1_x4_pd_fs.yml"
elif [[ "$STAGE" == "2" ]]; then
  OPT="options/train_DiffMSR_S2_x4_pd_fs.yml"
else
  echo "Unknown STAGE=$STAGE; use STAGE=1 or STAGE=2" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

if [[ -n "$TOTAL_ITER$TIMESTEPS$VAL_FREQ$SAVE_FREQ$PRINT_FREQ$RUN_NAME" ]]; then
  GENERATED_OPT_DIR="${LOG_DIR}/generated_options"
  GENERATED_OPT="${GENERATED_OPT_DIR}/${RUN_TAG}.yml"
  mkdir -p "$GENERATED_OPT_DIR"
  IN_OPT="$OPT" OUT_OPT="$GENERATED_OPT" TOTAL_ITER="$TOTAL_ITER" TIMESTEPS="$TIMESTEPS" VAL_FREQ="$VAL_FREQ" \
    SAVE_FREQ="$SAVE_FREQ" PRINT_FREQ="$PRINT_FREQ" RUN_NAME="$RUN_NAME" RUN_TAG="$RUN_TAG" \
    "$PYTHON_BIN" - <<'PY'
import os
import yaml

in_opt = os.environ["IN_OPT"]
out_opt = os.environ["OUT_OPT"]
with open(in_opt, "r", encoding="utf-8") as f:
    opt = yaml.safe_load(f)

run_name = os.environ.get("RUN_NAME") or ""
total_iter = os.environ.get("TOTAL_ITER") or ""
timesteps = os.environ.get("TIMESTEPS") or ""
val_freq = os.environ.get("VAL_FREQ") or ""
save_freq = os.environ.get("SAVE_FREQ") or ""
print_freq = os.environ.get("PRINT_FREQ") or ""
run_tag = os.environ.get("RUN_TAG") or "override"

if run_name:
    opt["name"] = run_name
elif total_iter:
    opt["name"] = f"{opt['name']}_it{int(total_iter)}_{run_tag}"

if total_iter:
    opt["train"]["total_iter"] = int(total_iter)
if timesteps:
    opt["network_g"]["timesteps"] = int(timesteps)
if val_freq:
    opt.setdefault("val", {})["val_freq"] = float(val_freq)
if save_freq:
    opt.setdefault("logger", {})["save_checkpoint_freq"] = float(save_freq)
if print_freq:
    opt.setdefault("logger", {})["print_freq"] = int(print_freq)

with open(out_opt, "w", encoding="utf-8") as f:
    yaml.safe_dump(opt, f, sort_keys=False)
PY
  OPT="$GENERATED_OPT"
fi

exec > >(tee -a "$LOG_FILE") 2>&1

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export NUMEXPR_NUM_THREADS="$CPUS"
export VECLIB_MAXIMUM_THREADS="$CPUS"

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Stage: $STAGE"
echo "Options: $OPT"
echo "Run name override: ${RUN_NAME:-none}"
echo "Total iter override: ${TOTAL_ITER:-none}"
echo "Timesteps override: ${TIMESTEPS:-none}"
echo "Val freq override: ${VAL_FREQ:-none}"
echo "Save freq override: ${SAVE_FREQ:-none}"
echo "GPU_ID: $GPU_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python: $PYTHON_BIN"
echo "Log: $LOG_FILE"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

"$PYTHON_BIN" - <<'PY'
import importlib
import sys

required = ["torch", "torchvision", "numpy", "cv2", "PIL", "tqdm", "scipy", "timm", "yaml", "einops", "tensorboard"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}: {exc}")

if missing:
    print("Missing or broken Python modules:")
    for item in missing:
        print(f"  {item}")
    sys.exit(1)

import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    print("CUDA is not available to PyTorch in this session; refusing to start training.")
    sys.exit(1)
print("cuda_device", torch.cuda.get_device_name(0))
PY

PYTHONPATH="$SCRIPT_DIR" \
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/DiffMSR_Main/train.py" \
  -opt "$OPT" \
  --launcher none
