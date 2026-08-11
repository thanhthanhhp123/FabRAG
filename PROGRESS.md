# FabRAG implementation progress

This document records what has actually been implemented and verified. It is
intentionally more conservative than the target architecture: a component is
only marked end-to-end verified when it has run against a real PDF/model/database.

## Work log

### 2026-08-11 — Repository handoff and baseline audit

- Cloned `https://github.com/thanhthanhhp123/FabRAG` on branch `main` at commit
  `98e75c3` (`Implement retrieval and grounded RAG pipeline`). The checkout was
  clean and matched `origin/main` before this work log was added.
- Read the contributor guide, README, package configuration, recent Git history,
  source/test inventory, and this progress record before changing implementation.
- Confirmed that the implementation is ahead of parts of the documentation:
  retrieval, hybrid fusion, reranking, and grounded answer generation exist in
  `ingestion-worker/src`, while README/AGENTS still describe several of them as
  future work. Documentation alignment is therefore a tracked follow-up rather
  than evidence that those components are absent.
- Selected the first concrete continuation task from the existing "Known
  limitations and next work" list: establish a clean test/lint baseline, then
  complete the real Qwen answer-generation smoke test if the local environment
  and model cache permit it. Results and any code changes will be recorded below
  as they happen.
- First baseline attempt could not start the tests: the documented
  `ingestion-worker/.venv` is correctly not committed, and the fresh workspace's
  system Python reported `No module named pytest`. This is an environment setup
  gap, not a failing project test. The next action is to create the local virtual
  environment and install the declared development dependencies.
- Environment follow-up found only Python 3.10.12 on the host, while the package
  correctly requires Python 3.11 or newer. Docker 28.1.1 was available, so the
  checks were moved to the official `python:3.11-slim` image instead of weakening
  the project's Python constraint.
- A normal unconstrained dependency resolution selected a new PyTorch build with
  the complete CUDA 13 stack. That download was stopped because this workspace is
  CPU-only and unit tests mock model execution. A CPU/minimal test dependency set
  was used for the baseline; future real-model verification must explicitly use
  CPU PyTorch to avoid downloading several GB of unused GPU libraries.
- Baseline verification on Python 3.11 completed successfully:

  ```text
  pytest -q:                 40 passed in 0.78s
  ruff check src tests:      All checks passed
  ruff format --check:       18 files already formatted
  ```

  This establishes that commit `98e75c3` is internally consistent before the
  next implementation step. The next check is a real Qwen generation using
  supplied evidence, isolated from the database so it tests only the still
  unverified tokenizer/model/generation path.
- Installed the model runtime under ignored `.runtime/model-venv` with explicit
  `torch==2.9.1+cpu` and `transformers==4.57.1`. Hugging Face model files are
  stored under ignored `.runtime/huggingface`; no generated environment or model
  weights are part of the Git diff.
- The first real `Qwen/Qwen2.5-0.5B-Instruct` smoke test completed without a
  runtime error and preserved the supplied source metadata, but its answer was
  behaviorally incorrect. Given explicit evidence that the range is `4.5 V to
  36 V`, it claimed the evidence was insufficient and emitted no `[S1]` citation.
  Therefore answer generation is **not** considered verified merely because the
  model loaded and generated text.
- Root cause at this stage is weak instruction following by the 0.5B smoke-test
  model, exposed by a prompt that did not explicitly tell it to accept directly
  stated evidence over prior beliefs. The system prompt is being strengthened to
  make evidence the sole source of truth, require direct extraction of stated
  values, and require each factual sentence to end in a source ID. A unit test is
  added for these invariants before repeating the real-model check.
- After that prompt-only change, all 41 tests and both Ruff checks passed, but the
  second real-model run still returned only `I don't have enough evidence.` This
  rules out the earlier contradiction but still fails extraction and citation.
  The next refinement adds one short few-shot conversation demonstrating the
  exact desired transformation: a value stated in `[S1]` becomes a concise answer
  ending in `[S1]`. The example uses unrelated temperature data so it teaches the
  response pattern without leaking the L293 test answer.
- The third real-model run, now with the few-shot example, passed:

  ```text
  Question: What is the supply voltage range of the L293?
  Answer: The supply voltage range for the L293 device is 4.5 V to 36 V [S1].
  Source metadata: 32_L293_datasheet.pdf, pages 1-3
  ```

  This closes the previously pending local Qwen smoke test for a controlled
  evidence input. It does not establish broad answer quality; that still requires
  the planned reviewed evaluation set.
- Final verification after the few-shot implementation:

  ```text
  pytest -q:                 41 passed in 0.75s
  ruff check src tests:      All checks passed
  ruff format --check:       18 files already formatted
  git diff --check:          passed
  ```

- Updated `README.md` and `AGENTS.md` so repository-facing status, architecture,
  directory map, and roadmap match the already implemented retrieval, reranking,
  and answer-generation modules. The remaining next milestone is the FastAPI
  boundary (or the evaluation harness before it), not reimplementing those core
  modules.

### 2026-08-11 — FastAPI service baseline

- Started after committing the verified Qwen work as commit `08d180b`
  (`Verify grounded Qwen answer generation`). The new milestone follows item 1
  in "Known limitations and next work": expose the online RAG pipeline through a
  separate FastAPI service rather than adding HTTP concerns to the ingestion CLI.
- Initial scope is deliberately small and testable without a model/database:
  `GET /health` for process health and `POST /v1/answers` for question → retrieval
  → reranking → generation. The answer response will include explicit filename,
  page, chunk, and ranking metadata so API clients do not need to parse citations
  from prose.
- Request validation will bound `candidate_k` and `top_n`, with `top_n <=
  candidate_k`. Backend/model/database failures will become a generic HTTP 503
  response rather than exposing internal exception strings. API-key auth, rate
  limiting, routing, and deployment remain separate hardening milestones and are
  not claimed by this baseline.
- Added `api-service` as a separate Python package. `GET /health` returns only
  process liveness and intentionally does not load a model or touch PostgreSQL.
  `POST /v1/answers` delegates to the existing `answer_question` pipeline and
  returns the generated text plus `S1`, `S2`, ... records containing filename,
  page span, chunk index, reranker score, original retrieval rank, vector rank,
  and keyword rank.
- Added request constraints: non-blank questions up to 2,000 characters,
  `candidate_k` from 1 to 50, `top_n` from 1 to 10, and `top_n <= candidate_k`.
  These limits prevent obviously accidental or disproportionately expensive
  requests before a full rate limiter exists.
- Added five HTTP contract tests covering liveness, successful source mapping,
  cross-field validation, whitespace-only questions, and sanitized 503 errors.
  A fake backend exception deliberately contains a database password-like string;
  the test confirms it is absent from the HTTP response.
- Verification on Python 3.11:

  ```text
  api-service:       5 passed in 0.98s
  API Ruff lint:     passed
  API format check:  3 files already formatted
  ingestion-worker:  41 passed in 0.77s
  worker Ruff lint:  passed
  worker format:     18 files already formatted
  ```

  FastAPI's TestClient emitted one upstream `StarletteDeprecationWarning` about
  the `httpx` compatibility layer. It does not affect these tests, but dependency
  compatibility should be revisited during API hardening rather than suppressing
  the warning.
- Updated README setup, curl examples, repository layout, status, and roadmap.
  The documentation explicitly warns that the baseline has no authentication or
  rate limiting and must not be exposed to an untrusted network.

## Current pipeline

```text
PDF
  -> page-aware text/table parsing
  -> overlapping fixed-size chunks
  -> BGE-M3 embeddings
  -> PostgreSQL + pgvector storage
  -> vector + keyword hybrid retrieval
  -> cross-encoder reranking
  -> grounded answer generation (implemented and local-model smoke tested)
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

Status: code, unit tests, and a real CPU model smoke test are complete. A single
few-shot example was needed because the 0.5B model incorrectly selected the
no-evidence fallback when given a directly stated value. With that example, the
real output was:

```text
The supply voltage range for the L293 device is 4.5 V to 36 V [S1].
```

The returned source metadata remained `32_L293_datasheet.pdf`, pages 1-3. This
verifies the local generation path for a controlled input, not general answer
quality. The 0.5B model is intended only as a CPU smoke test; the target
architecture should serve a stronger model behind a separate LLM server.

## Test status

At the time this document was written:

```text
41 tests passed
Ruff lint passed
Ruff format check passed
```

Tests cover parsing, chunk boundaries, overlap, page metadata, embedding helper
behavior, ingestion exit codes, retrieval validation, RRF fusion, reranking and
grounded prompt/generation behavior without requiring heavyweight models in the
unit suite.

## Known limitations and next work

1. Add API-key authentication, rate limiting, request IDs, and structured logs
   before exposing the FastAPI service beyond a trusted development network.
2. Ingest the complete corpus only after the single-document path is stable.
3. Remove recurring headers/footers and improve PDF whitespace reconstruction.
4. Add heading-aware/tokenizer-aware chunking and section metadata.
5. Add a manually reviewed evaluation set and report recall@k, MRR, reranker
   before/after metrics, citation correctness and groundedness.
6. Add a query router, out-of-domain handling and multi-document/multi-hop flow.
7. Add API authentication, rate limiting, structured logs, feedback and CI/CD.

FabRAG remains an active-development RAG prototype. It must not yet be described
as a deployed or production-ready chatbot.
