#!/usr/bin/env bash
#SBATCH --job-name=fabrag-ingest
#SBATCH --output=logs/fabrag-ingest-%j.out
#SBATCH --error=logs/fabrag-ingest-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Submit from the repository root with:
#   mkdir -p logs
#   export FABRAG_PYTHON="$PWD/ingestion-worker/.venv/bin/python"
#   sbatch scripts/ingest_all.sbatch
#
# If FABRAG_PYTHON is not provided, this script uses the repository's .venv
# when it exists; otherwise it falls back to a conda environment called
# "fabrag". Override FABRAG_CONDA_ENV for a differently named conda env.
set -euo pipefail

# Slurm executes a copied script from /tmp; SLURM_SUBMIT_DIR preserves the
# repository directory from which sbatch was called.
FABRAG_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FABRAG_WORKER_ROOT="${FABRAG_PROJECT_ROOT}/ingestion-worker"

if [[ -n "${FABRAG_PYTHON:-}" ]]; then
    FABRAG_INGEST_PYTHON="${FABRAG_PYTHON}"
elif [[ -x "${FABRAG_WORKER_ROOT}/.venv/bin/python" ]]; then
    FABRAG_INGEST_PYTHON="${FABRAG_WORKER_ROOT}/.venv/bin/python"
else
    : "${FABRAG_CONDA_BASE:?Set FABRAG_PYTHON or FABRAG_CONDA_BASE.}"
    FABRAG_CONDA_ENV="${FABRAG_CONDA_ENV:-fabrag}"
    source "${FABRAG_CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${FABRAG_CONDA_ENV}"
    FABRAG_INGEST_PYTHON="$(command -v python)"
fi

cd "${FABRAG_WORKER_ROOT}"
"${FABRAG_INGEST_PYTHON}" -m src.ingest --all
