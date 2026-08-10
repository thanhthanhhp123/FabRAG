#!/usr/bin/env bash
#SBATCH --job-name=fabrag-e2e-ingest
#SBATCH --output=logs/fabrag-e2e-ingest-%j.out
#SBATCH --error=logs/fabrag-e2e-ingest-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# End-to-end HPC smoke test for the ingestion pipeline. It starts an isolated
# PostgreSQL + pgvector server on the allocated compute node, ingests the
# corpus, prints row counts, then stops the server. The database files remain
# in .runtime/postgres so a later job can reuse them.
#
# This is deliberately a development/test database, not a public deployment.
# Submit from the repository root:
#   mkdir -p logs
#   export FABRAG_PYTHON="$PWD/ingestion-worker/.venv/bin/python"
#   sbatch scripts/ingest_with_local_pg.sbatch
set -euo pipefail

# Slurm executes a copied script from /tmp; SLURM_SUBMIT_DIR preserves the
# repository directory from which sbatch was called.
FABRAG_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FABRAG_WORKER_ROOT="${FABRAG_PROJECT_ROOT}/ingestion-worker"
FABRAG_IMAGE_DIR="${FABRAG_PROJECT_ROOT}/.containers"
FABRAG_IMAGE="${FABRAG_IMAGE_DIR}/pgvector-pg16.sif"
FABRAG_PGDATA="${FABRAG_PROJECT_ROOT}/.runtime/postgres"
FABRAG_SCHEMA="${FABRAG_WORKER_ROOT}/src/schema.sql"
FABRAG_PG_PORT="${FABRAG_PG_PORT:-15432}"
FABRAG_DB_USER="${FABRAG_DB_USER:-fabrag}"
FABRAG_DB_NAME="${FABRAG_DB_NAME:-fabrag}"
FABRAG_DB_PASSWORD="${FABRAG_DB_PASSWORD:-fabrag}"

if [[ -n "${FABRAG_PYTHON:-}" ]]; then
    FABRAG_INGEST_PYTHON="${FABRAG_PYTHON}"
elif [[ -x "${FABRAG_WORKER_ROOT}/.venv/bin/python" ]]; then
    FABRAG_INGEST_PYTHON="${FABRAG_WORKER_ROOT}/.venv/bin/python"
else
    : "${FABRAG_CONDA_BASE:?Set FABRAG_PYTHON or FABRAG_CONDA_BASE.}"
    source "${FABRAG_CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${FABRAG_CONDA_ENV:-fabrag}"
    FABRAG_INGEST_PYTHON="$(command -v python)"
fi

mkdir -p "${FABRAG_IMAGE_DIR}" "${FABRAG_PGDATA}"
module load singularity/3.9.5

if [[ ! -f "${FABRAG_IMAGE}" ]]; then
    singularity pull "${FABRAG_IMAGE}" docker://pgvector/pgvector:pg16
fi

export POSTGRES_USER="${FABRAG_DB_USER}"
export POSTGRES_DB="${FABRAG_DB_NAME}"
export POSTGRES_PASSWORD="${FABRAG_DB_PASSWORD}"
export SINGULARITYENV_POSTGRES_USER="${FABRAG_DB_USER}"
export SINGULARITYENV_POSTGRES_DB="${FABRAG_DB_NAME}"
export SINGULARITYENV_POSTGRES_PASSWORD="${FABRAG_DB_PASSWORD}"

singularity exec \
    --bind "${FABRAG_PGDATA}:/var/lib/postgresql/data" \
    --bind "${FABRAG_SCHEMA}:/docker-entrypoint-initdb.d/01_schema.sql:ro" \
    "${FABRAG_IMAGE}" \
    docker-entrypoint.sh postgres -p "${FABRAG_PG_PORT}" &
FABRAG_PG_PID=$!

cleanup() {
    kill "${FABRAG_PG_PID}" 2>/dev/null || true
    wait "${FABRAG_PG_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
    if singularity exec "${FABRAG_IMAGE}" pg_isready -h 127.0.0.1 -p "${FABRAG_PG_PORT}" \
        -U "${FABRAG_DB_USER}" -d "${FABRAG_DB_NAME}" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

singularity exec "${FABRAG_IMAGE}" pg_isready -h 127.0.0.1 -p "${FABRAG_PG_PORT}" \
    -U "${FABRAG_DB_USER}" -d "${FABRAG_DB_NAME}"

export DATABASE_URL="postgresql+psycopg://${FABRAG_DB_USER}:${FABRAG_DB_PASSWORD}@127.0.0.1:${FABRAG_PG_PORT}/${FABRAG_DB_NAME}"
cd "${FABRAG_WORKER_ROOT}"
"${FABRAG_INGEST_PYTHON}" -m src.ingest --all

singularity exec "${FABRAG_IMAGE}" psql \
    -h 127.0.0.1 -p "${FABRAG_PG_PORT}" -U "${FABRAG_DB_USER}" -d "${FABRAG_DB_NAME}" \
    -c "SELECT COUNT(*) AS documents FROM documents; SELECT COUNT(*) AS chunks FROM chunks;"
