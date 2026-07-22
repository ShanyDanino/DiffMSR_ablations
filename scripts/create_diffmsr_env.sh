#!/bin/bash

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${1:-${PROJECT_ROOT}/.conda/envs/diffmsr}"
CONDA_ROOT="${CONDA_ROOT:-${PROJECT_ROOT}/.conda}"
OFFLINE="${OFFLINE:-0}"

mkdir -p "${CONDA_ROOT}/pkgs" "${CONDA_ROOT}/envs"
export CONDA_PKGS_DIRS="${CONDA_ROOT}/pkgs"
export CONDA_ENVS_PATH="${CONDA_ROOT}/envs"
export CONDA_NOTICES=false

if [[ -f /opt/miniconda3/etc/profile.d/conda.sh ]]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
elif [[ -f /tcmldrive/lib/miniconda3/etc/profile.d/conda.sh ]]; then
    source /tcmldrive/lib/miniconda3/etc/profile.d/conda.sh
else
    echo "Could not find conda.sh" >&2
    exit 1
fi

if [[ -d "${ENV_PREFIX}" && ! -x "${ENV_PREFIX}/bin/python" ]]; then
    BROKEN_PREFIX="${ENV_PREFIX}.incomplete.$(date +%Y%m%d_%H%M%S)"
    echo "Moving incomplete environment aside:"
    echo "  ${ENV_PREFIX}"
    echo "  -> ${BROKEN_PREFIX}"
    mv "${ENV_PREFIX}" "${BROKEN_PREFIX}"
fi

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
    echo "Environment already exists: ${ENV_PREFIX}"
else
    echo "Creating ${ENV_PREFIX} by cloning the working base environment..."
    CLONE_ARGS=(create -y -p "${ENV_PREFIX}" --clone base)
    if [[ "${OFFLINE}" == "1" ]]; then
        CLONE_ARGS+=(--offline)
    fi
    conda "${CLONE_ARGS[@]}"
fi

echo "Registering this repository on the environment Python path..."
PROJECT_ROOT_FOR_PTH="${PROJECT_ROOT}" conda run -p "${ENV_PREFIX}" python - <<'PY'
import os
from pathlib import Path
import site

project_root = os.environ["PROJECT_ROOT_FOR_PTH"]
site_packages = next(
    (Path(path) for path in site.getsitepackages() if path.endswith("site-packages")),
    Path(site.getusersitepackages()),
)
site_packages.mkdir(parents=True, exist_ok=True)
pth_path = site_packages / "diffmsr_repo.pth"

# Force the checked-out repository to win over any pip-installed BasicSR.
line = (
    "import sys; p = "
    + repr(project_root)
    + "; sys.path[:] = [x for x in sys.path if x != p]; sys.path.insert(0, p)"
)
pth_path.write_text(line + "\n", encoding="utf-8")
print(pth_path)
PY

echo "Checking DiffMSR imports..."
if ! conda run -p "${ENV_PREFIX}" python "${PROJECT_ROOT}/scripts/check_diffmsr_env.py"
then
    echo "Some imports are missing. Installing repo requirements plus DiffMSR extras..."
    conda run -p "${ENV_PREFIX}" python -m pip install \
        -r "${PROJECT_ROOT}/requirements.txt"
fi

echo "Final import check..."
conda run -p "${ENV_PREFIX}" python "${PROJECT_ROOT}/scripts/check_diffmsr_env.py"

echo "Done. Use this with Slurm:"
echo "  CONDA_ENV=${ENV_PREFIX}"
