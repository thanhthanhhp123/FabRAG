# FabRAG implementation progress

This document records what has actually been implemented and verified. It is
intentionally more conservative than the target architecture: a component is
only marked end-to-end verified when it has run against a real PDF/model/database.

## Current pipeline

```text
PDF
  -> page-aware text/table parsing
  -> overlapping fixed-size chunks
  -> BGE-M3 embeddings
  -> PostgreSQL + pgvector storage
  -> vector + keyword hybrid retrieval
  -> cross-encoder reranking
  -> grounded answer generation (implemented, local-model smoke test pending)
```

The database currently contains one development document
(`32_L293_datasheet.pdf`), not the complete 50-document corpus.

## Development environment

- Managed Python: `3.11.15`
- Virtual environment: `ingestion-worker/.venv`
- PyTorch: `2.9.1+cpu`
- PostgreSQL: 16 in Docker
- pgvector: `0.8.6`
- Local model cache: `.runtime/huggingface` (ignored by Git)

Activate the environment:

```bash
cd ingestion-worker
source .venv/bin/activate
```

Run quality checks:

```bash
pytest -q
ruff check src tests
ruff format --check src tests
```

## Step 1: PDF parsing

Implementation: `ingestion-worker/src/parse.py`

The parser converts a PDF into a list of records shaped like:

```python
Page(number=1, text="page text and rendered tables")
```

What was added or improved:

1. Validate that the input exists and has a `.pdf` suffix.
2. Detect tables using `pdfplumber`.
3. Remove objects inside detected table bounding boxes from prose extraction.
4. Render each table once as pipe-separated text.
5. Preserve empty pages and human-facing, one-based page numbers.

Removing table regions matters because normal PDF text extraction often already
contains table text. Appending `extract_tables()` output without filtering would
duplicate facts before chunking and distort retrieval.

Real verification:

```text
32_L293_datasheet.pdf
pages: 19
non-empty pages: 19
characters: 30,824
```

Example rendered table:

```text
PARTNUMBER | PACKAGE | BODYSIZE(NOM)
L293NE | PDIP(16) | 19.80mm×6.35mm
L293DNE | PDIP(16) | 19.80mm×6.35mm
```

## Step 2: page-aware chunking

Implementation: `ingestion-worker/src/chunk.py`

The fixed-size baseline flattens page text into `(word, page_number)` pairs and
slides a window over them. Defaults are 400 words with 50 words of overlap.
Despite the environment-variable names, these are approximate word counts, not
embedding-tokenizer token counts.

Overlap keeps boundary facts together. For a 100-word window with 20-word
overlap, the next chunk starts at word 80.

A boundary bug was fixed: when a chunk already contained the document's final
word, the old loop could create another trailing chunk containing only repeated
overlap. Tests now cover exact-size and one-word-over-size documents.

Real verification on L293:

```text
19 pages -> 13 chunks
maximum chunk size: 400 words
chunk 0: pages 1-3
chunk 1: pages 3-4
last chunk: page 19, 68 words
overlap check: passed
```

## Step 3: embeddings

Implementation: `ingestion-worker/src/embed.py`

`BAAI/bge-m3` maps each chunk to a normalized 1024-dimensional vector. Similar
meaning should produce nearby vector directions even when wording differs.

Important behavior:

- lazy model loading;
- module-level model reuse;
- batched encoding;
- normalized embeddings for cosine similarity;
- empty input does not load the model.

Real verification:

```text
13 L293 chunks -> 13 vectors
dimension: 1024
minimum vector norm: 1.000000
maximum vector norm: 1.000000
```

For the question `What is the supply voltage range of the L293?`, an in-memory
cosine comparison selected chunk 0, which contains `4.5 V to 36 V`.

## Step 4: PostgreSQL and pgvector storage

Implementations:

- `ingestion-worker/src/db.py`
- `ingestion-worker/src/ingest.py`
- `ingestion-worker/src/schema.sql`

The local PostgreSQL service is started with:

```bash
docker compose up -d postgres
```

Each document row owns many chunk rows. A chunk stores text, page range,
chunking strategy, embedding model and a `vector(1024)` embedding.

Ingestion is transactional and idempotent by filename. Re-ingestion updates the
document, deletes its old chunks and inserts the new chunks in one transaction.

Verified by ingesting L293 twice:

```text
after first ingest:  1 document, 13 chunks
after second ingest: 1 document, 13 chunks
document ID: 1
missing embeddings: 0
```

The CLI was also fixed to return exit status 1 when any input fails. Previously
it printed `[FAILED]` but exited successfully, which could make CI or SLURM jobs
report false success.

## Step 5: vector retrieval

Implementation: `ingestion-worker/src/retrieve.py`

The query is embedded once with BGE-M3. pgvector orders stored chunks by cosine
distance and returns text plus filename/page metadata.

Real query:

```text
What is the supply voltage range of the L293?
```

Top result:

```text
score: 0.6806
source: 32_L293_datasheet.pdf
pages: 1-3
chunk: 0
evidence: Wide Supply-Voltage Range: 4.5 V to 36 V
```

## Step 6: hybrid retrieval

Implementation: `ingestion-worker/src/hybrid_retrieve.py`

Hybrid retrieval combines:

1. BGE-M3 semantic vector ranking.
2. PostgreSQL full-text ranking for exact terms and part numbers.
3. Weighted Reciprocal Rank Fusion (RRF).

PostgreSQL uses `ts_rank_cd`; this is full-text ranking, not strict BM25. Keyword
candidate generation uses OR between query terms to avoid losing all candidates
when PDF extraction joins tokens such as `4.5Vto36V`.

Current fusion weights are vector 2 and keyword 1. These are baseline values,
not evaluated optimums. Equal weighting was observed to promote repeated L293
page headers over the actual voltage evidence.

Verified query:

```text
L293 supply voltage range 4.5V 36V
```

The correct page 1-3 chunk ranked first after weighted fusion.

## Step 7: cross-encoder reranking

Implementation: `ingestion-worker/src/rerank.py`

Hybrid retrieval cheaply selects candidates. `BAAI/bge-reranker-base` then reads
each `(question, passage)` pair jointly and reorders only that small set. This is
more precise but too expensive to run against every stored chunk.

Real CPU verification over 10 candidates took about four seconds:

```text
rank 1: previous hybrid rank 1, pages 1-3, score 0.999519
rank 2: previous hybrid rank 2, pages 11-14, score 0.995517
rank 3: previous hybrid rank 5, pages 3-4, score 0.972176
```

The third result contains the more specific pin-level evidence
`Power VCC for drivers: 4.5 V to 36 V`. Reranker scores are ranking signals, not
calibrated probabilities that an answer is correct.

## Step 8: grounded answer generation

Implementation: `ingestion-worker/src/answer.py`

The local MVP packages reranked evidence into source blocks:

```text
[S1] File: 32_L293_datasheet.pdf; pages: 1-3
<chunk text>
```

The system prompt requires the model to:

- use only supplied evidence;
- cite factual claims with source IDs such as `[S1]`;
- say `I don't have enough evidence.` when context is insufficient;
- never invent filenames, pages or source IDs.

The deterministic no-evidence path does not load an LLM. Prompt construction and
generation-token slicing are covered by unit tests using a fake generator.

Configured smoke-test model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Status: code and unit tests are complete, but the first real model download was
interrupted near completion. Therefore local answer generation is **not yet
end-to-end verified**. The 0.5B model is intended only as a CPU smoke test; the
target architecture should serve a stronger model behind a separate LLM server.

## Test status

At the time this document was written:

```text
40 tests passed
Ruff lint passed
Ruff format check passed
```

Tests cover parsing, chunk boundaries, overlap, page metadata, embedding helper
behavior, ingestion exit codes, retrieval validation, RRF fusion, reranking and
grounded prompt/generation behavior without requiring heavyweight models in the
unit suite.

## Known limitations and next work

1. Finish the Qwen smoke-test download and run answer generation end to end.
2. Move online retrieval/reranking/generation out of `ingestion-worker` into a
   FastAPI service; the current location is an incremental prototype.
3. Ingest the complete corpus only after the single-document path is stable.
4. Remove recurring headers/footers and improve PDF whitespace reconstruction.
5. Add heading-aware/tokenizer-aware chunking and section metadata.
6. Add a manually reviewed evaluation set and report recall@k, MRR, reranker
   before/after metrics, citation correctness and groundedness.
7. Add a query router, out-of-domain handling and multi-document/multi-hop flow.
8. Add API authentication, rate limiting, structured logs, feedback and CI/CD.

FabRAG remains an active-development RAG prototype. It must not yet be described
as a deployed or production-ready chatbot.
