# FabRAG contributor guide

This file is the practical map for humans and coding agents working in this
repository. It intentionally explains the project from first principles.

## What FabRAG is

FabRAG will answer questions about electronics-manufacturing documents, such
as component datasheets. It is a **RAG** system:

```text
PDF document
  -> parse text and tables
  -> split into small, page-aware chunks
  -> turn each chunk into an embedding (a list of numbers representing meaning)
  -> store text + vector + source page in PostgreSQL/pgvector
  -> later: retrieve relevant chunks for a question
  -> later: LLM writes an answer using those chunks and cites the source page
```

An LLM is a text-generating model. It can write a convincing answer even when
it is wrong. RAG reduces that risk by supplying relevant passages from the
actual datasheet and requiring the eventual answer to cite them. It does not
replace engineering review or the latest official document revision.

## Current implementation status

Implemented:

- PDF text/table extraction (`ingestion-worker/src/parse.py`)
- fixed-size, overlapping chunks with source-page tracking (`chunk.py`)
- BGE-M3 embedding generation (`embed.py`)
- idempotent PostgreSQL + pgvector storage (`db.py`, `ingest.py`, `schema.sql`)
- vector and PostgreSQL full-text hybrid retrieval (`retrieve.py`,
  `hybrid_retrieve.py`)
- BGE cross-encoder reranking (`rerank.py`)
- local, evidence-only answer generation with source IDs (`answer.py`)
- typed health and answer HTTP endpoints (`api-service/fabrag_api/main.py`)
- fail-closed API-key authentication and single-process rate limiting
  (`api-service/fabrag_api/security.py`)
- request correlation and JSON access logs (`api-service/fabrag_api/observability.py`)
- offline source/page recall and MRR evaluation (`evaluation/fabrag_eval/evaluate.py`)

Tested without heavyweight PDF/model/database dependencies:

- 41 unit tests covering parsing, chunking, embedding helpers, ingestion error
  handling, retrieval validation, hybrid fusion, reranking, and generation
- 11 API contract/security/observability tests
- 13 evaluation loader, matching, and metric tests
- Ruff lint and format checks

Real PDF/model/database checks have also covered ingestion, retrieval, reranking,
and a Qwen answer-generation smoke test. Not implemented yet: an LLM router,
shared/distributed rate limiting, UI, offline RAG evaluation, CI/CD, and
deployment. Do not describe the repository as a deployed or production-ready
chatbot until those parts exist.

## Directory map

```text
datasheets/                       Input PDFs (public documents only)
api-service/fabrag_api/main.py    HTTP health and grounded-answer endpoints
api-service/fabrag_api/security.py API-key auth and local rate limiting
api-service/fabrag_api/observability.py Request IDs and JSON access logs
evaluation/fabrag_eval/evaluate.py Offline recall and MRR evaluation CLI
evaluation/questions.example.jsonl Schema example, not a reviewed benchmark
ingestion-worker/src/parse.py     Step 1: PDF -> Page(number, text)
ingestion-worker/src/chunk.py     Step 2: Page -> ChunkRecord
ingestion-worker/src/embed.py     Step 3: chunk text -> normalized vectors
ingestion-worker/src/db.py        SQLAlchemy models and document upsert
ingestion-worker/src/ingest.py    Command-line pipeline coordinator
ingestion-worker/src/schema.sql   Postgres tables, pgvector and indexes
ingestion-worker/tests/           Fast unit tests; no database/model required
scripts/                          Helpers, including the SLURM batch template
```

## Run it locally

FabRAG requires Python **3.11+**. Create an isolated environment instead of
installing packages into the system Python:

```bash
conda create -n fabrag python=3.11
conda activate fabrag
cd ingestion-worker
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
ruff check src tests
ruff format --check src tests
```

Copy `.env.example` to `.env` at the repository root before real ingestion.
`DATABASE_URL` must point to a reachable Postgres server with the pgvector
extension. The default `localhost` URL only works when Postgres runs on the
same machine as the ingestion process.

For a disposable local database, start it from the repository root:

```bash
docker compose up -d postgres
```

On this HPC cluster, Docker is not available. For an end-to-end development
test, `scripts/ingest_with_local_pg.sbatch` runs a temporary pgvector
container on the allocated compute node and preserves its database files in
the ignored `.runtime/postgres` directory. It is useful to prove ingestion,
but a managed Postgres service is still needed later for a deployed API.

Then ingest one small document first:

```bash
cd ingestion-worker
python -m src.ingest ../datasheets/16_LM358_datasheet.pdf
```

Only use `python -m src.ingest --all` after that succeeds. The first run
downloads BGE-M3 model weights and is CPU/RAM intensive.

## Run ingestion on an HPC cluster

Do not run the full corpus on a login node. First create the `fabrag` conda
environment above, configure a reachable database URL, then submit the batch
template:

```bash
mkdir -p logs
export FABRAG_PYTHON="$PWD/ingestion-worker/.venv/bin/python"
sbatch scripts/ingest_all.sbatch
```

The template requests 8 CPUs and 32 GB RAM; adapt its `#SBATCH` lines to the
cluster's queue policy. It uses `FABRAG_PYTHON` when set, otherwise the local
`ingestion-worker/.venv`, then finally a conda environment. A GPU is optional:
Sentence Transformers will use one when PyTorch can see one, but CPU ingestion
is valid and simply slower.

Before a full batch, this command checks PDF parsing only (no database and no
embedding model download):

```bash
cd ingestion-worker
python -m src.parse ../datasheets/16_LM358_datasheet.pdf
```

## Important operating rules

- Never commit `.env`, database passwords, API keys, or private PDFs.
- The vector dimension in `.env` and `schema.sql` must match. BGE-M3 uses
  1024 dimensions. Changing the embedding model/dimension requires a full
  re-index and possibly a schema migration.
- Re-ingesting the same filename deletes and recreates that document's chunks;
  it is safe for chunking/embedding experiments but replaces existing rows.
- `CHUNK_SIZE_TOKENS` and `CHUNK_OVERLAP_TOKENS` are currently approximate
  **word** counts, not tokenizer-accurate token counts. Keep the names for
  configuration compatibility and document this limitation in experiments.
- Keep a document's filename, page range and future section heading with every
  chunk. These are what make citations possible.
- Add tests for any change to parsing or chunking. Retrieval/prompt changes
  must eventually include before/after evaluation metrics.

## Useful definitions

- **Chunk:** a short overlapping slice of a PDF. Smaller chunks are more
  precise; larger chunks contain more context. The project will measure the
  trade-off later.
- **Embedding:** a numeric vector produced by a model. Similar meanings are
  placed near each other, enabling semantic search.
- **pgvector:** a Postgres extension that stores and compares embeddings.
- **Idempotent ingestion:** running the same input again gives a clean updated
  result rather than duplicate chunks.
- **Hybrid retrieval:** a future combination of vector similarity and keyword
  search. This matters for exact part numbers such as `STM32F103C8`.
- **Reranker:** a future second model that reorders a small set of retrieved
  chunks before the LLM sees them.
