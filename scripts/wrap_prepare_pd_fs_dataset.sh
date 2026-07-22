#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Prepare a PD/PD fat-sat DICOM database for DiffMSR.

Usage:
  bash scripts/wrap_prepare_pd_fs_dataset.sh <dicom_folder_or_archive> [output_dataset_root]

Arguments:
  dicom_folder_or_archive  DICOM database folder, .zip, .tar, .tar.gz, .tgz, .tar.bz2, or .tar.xz
  output_dataset_root      Prepared DiffMSR dataset root (default: mri_data_complex/mc_knee_pd_fs)

Environment variables:
  PYTHON_BIN        Python executable                 (default: /opt/miniconda3/bin/python)
  EXTRACT_ROOT      Where archives are extracted       (default: data/extracted_pd_fs)
  TRAIN_RATIO       Study-level train split            (default: 0.8)
  SEED              Split seed                         (default: 0)
  HR_SIZE           Stored HR size / T1,T2 size         (default: 256)
  MID_SIZE          Stored T1_128,T2_128 size           (default: 128)
  LR_SIZE           Stored T1_64,T2_64 size             (default: 64)
  MASK_SRC          Mask copied to dc_mask.mat          (default: complex_data_demo/dc_mask/lr_4x.mat)
  OVERWRITE         Replace existing generated .mat     (default: 0)
  COMPRESS          Save compressed .mat files          (default: 0)
  DRY_RUN           Detect pairs without writing data   (default: 0)
  MAX_STUDIES       Optional quick-test study limit
  MAX_SAMPLES       Optional quick-test sample limit

Output format:
  output_dataset_root/train/*.mat
  output_dataset_root/valid/*.mat
  output_dataset_root/dc_mask.mat
  output_dataset_root/manifest.csv
  output_dataset_root/summary.json
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

SOURCE="$1"
OUT_ROOT="${2:-mri_data_complex/mc_knee_pd_fs}"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/bin/python}"
EXTRACT_ROOT="${EXTRACT_ROOT:-data/extracted_pd_fs}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
SEED="${SEED:-0}"
HR_SIZE="${HR_SIZE:-256}"
MID_SIZE="${MID_SIZE:-128}"
LR_SIZE="${LR_SIZE:-64}"
MASK_SRC="${MASK_SRC:-complex_data_demo/dc_mask/lr_4x.mat}"
OVERWRITE="${OVERWRITE:-0}"
COMPRESS="${COMPRESS:-0}"
DRY_RUN="${DRY_RUN:-0}"
MAX_STUDIES="${MAX_STUDIES:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"

if [[ -d "$SOURCE" ]]; then
  INPUT_ROOT="$SOURCE"
elif [[ -f "$SOURCE" ]]; then
  mkdir -p "$EXTRACT_ROOT"
  archive_name="$(basename "$SOURCE")"
  archive_stem="${archive_name%.tar.gz}"
  archive_stem="${archive_stem%.tar.bz2}"
  archive_stem="${archive_stem%.tar.xz}"
  archive_stem="${archive_stem%.tgz}"
  archive_stem="${archive_stem%.tbz2}"
  archive_stem="${archive_stem%.txz}"
  archive_stem="${archive_stem%.tar}"
  archive_stem="${archive_stem%.zip}"
  INPUT_ROOT="${EXTRACT_ROOT}/${archive_stem}"
  mkdir -p "$INPUT_ROOT"

  echo "Extracting ${SOURCE} -> ${INPUT_ROOT}"
  case "$SOURCE" in
    *.zip)
      if ! command -v unzip >/dev/null 2>&1; then
        echo "unzip is required for .zip archives." >&2
        exit 1
      fi
      unzip -q "$SOURCE" -d "$INPUT_ROOT"
      ;;
    *.tar|*.tar.gz|*.tgz|*.tar.bz2|*.tbz2|*.tar.xz|*.txz)
      tar -xf "$SOURCE" -C "$INPUT_ROOT"
      ;;
    *)
      echo "Unsupported archive type: ${SOURCE}" >&2
      exit 2
      ;;
  esac
else
  echo "Input database does not exist: ${SOURCE}" >&2
  exit 2
fi

args=(
  scripts/preprocess_pd_fs_dicom.py
  --input-root "$INPUT_ROOT"
  --out-root "$OUT_ROOT"
  --train-ratio "$TRAIN_RATIO"
  --seed "$SEED"
  --hr-size "$HR_SIZE"
  --mid-size "$MID_SIZE"
  --lr-size "$LR_SIZE"
  --mask-src "$MASK_SRC"
  --progress-every "$PROGRESS_EVERY"
)

if [[ "$OVERWRITE" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ "$COMPRESS" == "1" ]]; then
  args+=(--compress)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi
if [[ -n "$MAX_STUDIES" ]]; then
  args+=(--max-studies "$MAX_STUDIES")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  args+=(--max-samples "$MAX_SAMPLES")
fi

echo "Preparing DiffMSR dataset"
echo "  Input root: ${INPUT_ROOT}"
echo "  Output root: ${OUT_ROOT}"
valid_ratio="$("$PYTHON_BIN" -c 'import sys; print(f"{1.0 - float(sys.argv[1]):.3g}")' "$TRAIN_RATIO")"
echo "  Split: train=${TRAIN_RATIO}, valid=${valid_ratio}"

"$PYTHON_BIN" "${args[@]}"

if [[ "$DRY_RUN" != "1" ]]; then
  train_count="$(find "$OUT_ROOT/train" -maxdepth 1 -type f -name '*.mat' | wc -l)"
  valid_count="$(find "$OUT_ROOT/valid" -maxdepth 1 -type f -name '*.mat' | wc -l)"
  echo "Prepared dataset is ready for training:"
  echo "  DATA_ROOT=${OUT_ROOT}"
  echo "  train .mat files: ${train_count}"
  echo "  valid .mat files: ${valid_count}"
fi
