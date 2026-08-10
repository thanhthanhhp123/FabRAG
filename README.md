# FabRAG

FabRAG is a production-oriented Retrieval-Augmented Generation (RAG) project for electronics manufacturing documentation. It is designed to help engineers find grounded answers in component datasheets, equipment manuals, SOPs, work instructions, and public standards excerpts—with citations back to the source document and page.

> **Project status:** active development. The PDF ingestion pipeline and a 50-document starter corpus are in the repository. The query API, agentic router, hybrid retrieval, answer generation, evaluation harness, and web UI are on the roadmap and are not implemented yet.

## Why FabRAG?

Technical questions in electronics manufacturing often depend on exact values, package variants, operating conditions, and document revisions. Generic LLM answers are not reliable enough for that work. FabRAG is intended to combine:

- semantic retrieval for concept-level matches;
- keyword search for exact part numbers and specification codes;
- reranking for better context selection;
- page-aware citations for traceability; and
- an LLM router that can choose between single-hop retrieval, multi-hop comparison, general knowledge, or an out-of-domain rejection.

The project is also a practical study of the LLM application layer: ingestion, retrieval, routing, evaluation, deployment, and observability.

## Current capabilities

- A starter corpus of 50 public component datasheets from STMicroelectronics and Texas Instruments.
- PDF text and table extraction with `pdfplumber`.
- Overlapping fixed-size chunking while preserving source page spans.
- Normalized embeddings with `BAAI/bge-m3` through Sentence Transformers.
- Idempotent ingestion into PostgreSQL with `pgvector`.
- Document, chunk, embedding-model, and chunking-strategy metadata.
- PostgreSQL full-text and vector-ready schema.
- Unit tests for chunk sizing, overlap, page tracking, and validation.
- Docker Compose setup for the local PostgreSQL/pgvector service.

## Architecture

The ingestion path is implemented today. The online question-answering path shown below is the target MVP architecture.

```mermaid
flowchart LR
    subgraph Implemented[Implemented]
        PDF[PDF corpus] --> Parse[Text and table parsing]
        Parse --> Chunk[Page-aware chunking]
        Chunk --> Embed[BGE-M3 embeddings]
        Embed --> DB[(PostgreSQL + pgvector)]
    end

    subgraph Planned[Planned MVP]
        User[Web or API client] --> API[FastAPI]
        API --> Router{LLM query router}
        Router -->|single-hop| Retrieve[Hybrid retrieval]
        Router -->|multi-hop| Multi[Multiple retrieval passes]
        Router -->|general knowledge| Generate[Answer generation]
        Router -->|out of domain| Reject[Polite rejection]
        DB --> Retrieve
        DB --> Multi
        Retrieve --> Rerank[Cross-encoder reranker]
        Multi --> Rerank
        Rerank --> Generate
        Generate --> Answer[Grounded answer + citations]
    end
```

The planned services are independently deployable:

- **Ingestion worker:** parse, chunk, embed, and store documents asynchronously.
- **API:** route questions, retrieve and rerank evidence, then return cited answers.
- **LLM server:** serve routing and generation separately so the model runtime can be scaled or replaced without changing the API.

## Repository layout

```text
FabRAG/
├── datasheets/                    # 50-document starter corpus
├── ingestion-worker/
│   ├── src/
│   │   ├── parse.py               # PDF text and table extraction
│   │   ├── chunk.py               # Fixed-size overlapping chunks
│   │   ├── embed.py               # Sentence Transformer embeddings
│   │   ├── db.py                  # SQLAlchemy models and DB helpers
│   │   ├── ingest.py              # End-to-end ingestion CLI
│   │   └── schema.sql             # PostgreSQL + pgvector schema
│   ├── tests/
│   └── pyproject.toml
├── scripts/
│   └── bulk_download.py           # Corpus collection helper
├── .env.example
├── docker-compose.yml
└── CLAUDE.md                      # Product scope and engineering brief
```

## Quick start

### Prerequisites

- Python 3.11 or newer
- Docker with Docker Compose
- Git
- Enough RAM and disk space to run `BAAI/bge-m3`
- Internet access on the first run so Sentence Transformers can download the embedding model

### 1. Clone and configure

```bash
git clone https://github.com/thanhthanhhp123/FabRAG.git
cd FabRAG
cp .env.example .env
```

On Windows PowerShell, replace the last command with:

```powershell
Copy-Item .env.example .env
```

The example credentials are intended for local development only. Change them before exposing PostgreSQL beyond your machine.

### 2. Start PostgreSQL with pgvector

```bash
docker compose up -d postgres
docker compose ps
```

`ingestion-worker/src/schema.sql` is applied automatically when the database volume is created for the first time.

### 3. Install the ingestion worker

```bash
cd ingestion-worker
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install the package and development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 4. Ingest documents

Ingest a single datasheet:

```bash
python -m src.ingest ../datasheets/16_LM358_datasheet.pdf
```

Ingest the complete starter corpus:

```bash
python -m src.ingest --all
```

Re-ingesting a file replaces its existing chunks, which makes experiments with chunk sizes or embedding models repeatable without accumulating stale rows.

The first ingestion run may take substantially longer because it downloads and loads the embedding model. Processing all 50 PDFs is CPU- and memory-intensive.

### 5. Inspect the result

From the repository root:

```bash
docker compose exec postgres psql -U fabrag -d fabrag -c \
  "SELECT d.filename, COUNT(c.id) AS chunks FROM documents d LEFT JOIN chunks c ON c.document_id = d.id GROUP BY d.id ORDER BY d.filename;"
```

## Configuration

Copy `.env.example` to `.env` and adjust these values as needed:

| Variable | Default | Purpose |
|---|---:|---|
| `POSTGRES_USER` | `fabrag` | Local PostgreSQL user |
| `POSTGRES_PASSWORD` | `fabrag` | Local PostgreSQL password |
| `POSTGRES_DB` | `fabrag` | Local database name |
| `DATABASE_URL` | `postgresql+psycopg://...` | SQLAlchemy connection URL used by the ingestion worker |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence Transformer model used for indexing |
| `EMBEDDING_DIM` | `1024` | Vector size expected by the database schema |
| `CHUNK_SIZE_TOKENS` | `400` | Approximate chunk window; currently implemented as words |
| `CHUNK_OVERLAP_TOKENS` | `50` | Approximate overlap; currently implemented as words |

Changing the embedding model or dimension requires a full re-index. If the vector dimension changes, update both `.env` and `ingestion-worker/src/schema.sql` before creating the database schema.

## Development

Run the tests and static checks from `ingestion-worker/`:

```bash
pytest
ruff check src tests
ruff format --check src tests
```

You can also inspect PDF extraction without connecting to PostgreSQL:

```bash
python -m src.parse ../datasheets/16_LM358_datasheet.pdf
```

The Docker initialization script only runs against a new database volume. When the schema changes, apply the migration manually or recreate the local development volume if its data is disposable.

## Evaluation plan

The MVP will be evaluated with a manually reviewed set of 30–50 questions covering router decisions, single-hop lookup, multi-document comparison, general knowledge, and out-of-domain requests.

Planned metrics:

- router classification accuracy;
- retrieval recall@k and mean reciprocal rank (MRR);
- fixed-size versus heading-aware chunking;
- hybrid retrieval with and without reranking; and
- RAGAS faithfulness and answer relevancy.

Results will be reported as measured values rather than estimated product-impact claims.

## Roadmap

- [x] Define the product scope and architecture.
- [x] Collect a 50-document starter corpus.
- [x] Implement PDF parsing, fixed-size chunking, embedding, and storage.
- [x] Add an initial chunking test suite.
- [ ] Add semantic/heading-aware chunking.
- [ ] Implement vector + keyword hybrid retrieval and reranking.
- [ ] Build the function-calling query router.
- [ ] Add cited answer generation and out-of-domain guardrails.
- [ ] Expose a rate-limited, API-key-protected FastAPI service.
- [ ] Add a minimal web interface and feedback capture.
- [ ] Build the offline evaluation harness and publish results.
- [ ] Add structured observability, CI/CD, and deployment.

## Data and responsible use

The starter corpus contains publicly available manufacturer datasheets. Copyright and trademarks remain with their respective owners. Do not add or redistribute paid IPC standards, private SOPs, confidential work instructions, or other restricted documents without permission.

FabRAG is an engineering information-retrieval project, not a substitute for checking the latest official document revision or completing the review and validation required for manufacturing decisions.

## Contributing

Keep configuration in environment variables and never commit secrets or local `.env` files. Changes to retrieval, chunking, reranking, or prompts should include before/after evaluation results once the evaluation harness is available.
