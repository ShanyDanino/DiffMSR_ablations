#!/bin/bash

#SBATCH --job-name=DiffMSR
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mem=0
#SBATCH --partition=long
#SBATCH --nodelist=argus04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=14-00:00:00

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

STAGE="${STAGE:-1}"
CONDA_ENV="${CONDA_ENV:-base}"

if [[ "$STAGE" == "1" ]]; then
  OPT="options/train_DiffMSR_S1_x4_pd_fs.yml"
elif [[ "$STAGE" == "2" ]]; then
  OPT="options/train_DiffMSR_S2_x4_pd_fs.yml"
else
  echo "Unknown STAGE=$STAGE; use STAGE=1 or STAGE=2" >&2
  exit 2
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Stage: $STAGE"
echo "Options: $OPT"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Limit CPU thread libraries so Python/PyTorch/NumPy do not use too many CPUs.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}

# Optional but useful for debugging oversubscription.
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"

if [[ -f /opt/miniconda3/etc/profile.d/conda.sh ]]; then
  source /opt/miniconda3/etc/profile.d/conda.sh
elif [[ -f /tcmldrive/lib/miniconda3/etc/profile.d/conda.sh ]]; then
  source /tcmldrive/lib/miniconda3/etc/profile.d/conda.sh
else
  echo "Could not find conda.sh" >&2
  exit 1
fi

conda activate "$CONDA_ENV"

conda info

# Optional: show GPU status if nvidia-smi exists.
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

python scripts/check_diffmsr_env.py

PYTHONPATH="$SCRIPT_DIR" \
python -u "$SCRIPT_DIR/DiffMSR_Main/train.py" \
  -opt "$OPT" \
  --launcher none

echo "END"
