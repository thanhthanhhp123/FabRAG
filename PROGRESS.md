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

### 2026-08-11 — API security and observability hardening

- Started after committing the FastAPI baseline as `132cd2b` (`Add grounded
  answer API service`). Scope follows the new highest-priority limitation: API-key
  authentication, bounded request rates, correlation IDs, and machine-readable
  access logs.
- Security behavior will be fail-closed. `/health` remains public for process
  probes, while `/v1/answers` requires a configured `FABRAG_API_KEY` and a matching
  `X-API-Key` header. A missing server key is a deployment/configuration failure,
  not an implicit anonymous-development mode.
- The first limiter will be an explicitly documented in-memory fixed-window
  baseline, keyed by authenticated API key. It protects a single process but does
  not coordinate counters across multiple workers or replicas; Redis or an API
  gateway remains necessary before horizontal deployment.
- Implemented fail-closed `X-API-Key` authentication using
  `secrets.compare_digest`. The configured secret is read from `FABRAG_API_KEY`;
  when it is absent, answer requests return 503 rather than silently becoming
  anonymous. Invalid or missing client keys return 401 with `WWW-Authenticate`.
- Implemented a thread-safe fixed-window limiter with configurable positive
  integer settings (`FABRAG_RATE_LIMIT_REQUESTS`, default 30, and
  `FABRAG_RATE_LIMIT_WINDOW_SECONDS`, default 60). Counters use a SHA-256 identity
  rather than retaining raw API keys. Limited responses return 429 and a
  `Retry-After` header. Invalid limiter configuration fails closed with 503.
- Added request correlation middleware. Valid incoming `X-Request-ID` values are
  preserved; missing or unsafe values are replaced with a random 32-character
  identifier. The ID is included on success and FastAPI-generated error
  responses.
- Added compact JSON access logs containing timestamp, level, logger, message,
  request ID, method, path, status, and duration. Headers and request bodies are
  deliberately excluded, so neither API credentials nor engineering questions
  are copied into access logs.
- API verification after hardening:

  ```text
  11 tests passed in 1.02s
  Ruff lint passed
  Ruff format check passed (5 files)
  ```

  The tests include authentication configuration, invalid credentials, rate-limit
  enforcement, invalid limiter configuration, `Retry-After`, request-ID
  preservation/generation, and JSON log fields in addition to the original HTTP
  contract tests. The same upstream TestClient deprecation warning remains
  documented and unsuppressed.

### 2026-08-11 — Offline retrieval evaluation harness

- Pushed commits `08d180b`, `132cd2b`, and `8d4cb06` to `origin/main`; remote and
  local `main` both resolved to `8d4cb06` before starting this milestone.
- Chose evaluation before query routing so retrieval/reranker changes have a
  repeatable quality signal. Initial scope is recall@k and mean reciprocal rank
  (MRR) for hybrid candidates and reranked results, matched against reviewed
  filename/page evidence.
- The harness will define and validate a JSONL format, but will not label generated
  examples as a reviewed benchmark. A tiny example file may demonstrate syntax;
  actual reported project metrics remain blocked on human review of questions and
  evidence labels.
- Added a standalone `evaluation` package and `fabrag-evaluate` CLI. Each JSONL
  record requires a unique ID, a non-blank question, and at least one expected
  filename with an optional one-based evidence page. The loader rejects malformed
  JSON, wrong JSON types, empty labels, duplicate IDs, booleans masquerading as
  integer pages, and pages below 1 with line-aware errors.
- Relevance matching requires exact filenames. When a reviewed page is supplied,
  the retrieved chunk's page span must contain it; filename-only labels remain
  available when the exact page has not been labeled. Metrics use the first
  relevant one-based rank: recall is the fraction of questions with any relevant
  result, while MRR averages `1 / rank` and assigns zero to misses.
- The runner evaluates the same questions twice: raw hybrid candidates at
  `candidate_k`, then the cross-encoder's top `top_n`. It emits a machine-readable
  JSON summary with question count, cutoffs, candidate recall/MRR, and reranked
  recall/MRR. Invalid cutoffs, including `top_n > candidate_k`, fail before model
  or database work begins.
- Added `questions.example.jsonl` only to demonstrate the schema. README and the
  filename explicitly state that it is not a reviewed benchmark, so no quality
  metric is reported from that single example.
- Unit verification uses deterministic fake retrieval/reranking results to test
  metric arithmetic without PostgreSQL or model downloads:

  ```text
  13 tests passed in 0.16s
  Ruff lint passed
  Ruff format check passed (3 files)
  ```

### 2026-08-11 — Reviewed benchmark seed

- Started after committing and pushing the evaluation harness as `b1559cd`
  (`Add offline retrieval evaluation harness`). The goal is a small, auditable
  seed set whose labels come from direct inspection of committed manufacturer
  PDFs, not generated guesses.
- Questions will span multiple device families and favor unambiguous facts such
  as supply range, resolution, channel count, or operating temperature. Each
  record will be accepted only after the relevant wording and one-based PDF page
  have been checked. This seed is a foundation for the planned 30–50 question
  benchmark, not a claim that the full evaluation set is complete.
- Extended the JSONL contract with a required, normalized `reference_answer`.
  Retrieval metrics do not consume it yet, but storing the reviewed answer beside
  the evidence label makes the dataset auditable and prepares it for later answer
  correctness/groundedness evaluation. Loader tests now reject a missing or blank
  reference answer.
- Added `questions.verified.seed.jsonl` with 10 distinct documents. Direct PDF
  text inspection verified these labels:

  1. L293 supply range `4.5 V to 36 V` — `32_L293_datasheet.pdf`, page 1.
  2. INA219 sensed bus range `0 V to 26 V` — `24_INA219_datasheet.pdf`, page 1.
  3. ADS1115 resolution `16-bit` — `26_ADS1115_datasheet.pdf`, page 1.
  4. TMP117 maximum accuracy `±0.1 °C` from `-20 °C to 50 °C` —
     `39_TMP117_datasheet.pdf`, page 1.
  5. SN74HC595 type `8-bit serial-in, parallel-out shift register` —
     `29_SN74HC595_datasheet.pdf`, page 1.
  6. DAC8562 resolution `16-bit` — `49_DAC8562_datasheet.pdf`, page 1.
  7. DRV8833 motor supply range `2.7 V to 10.8 V` —
     `31_DRV8833_datasheet.pdf`, page 5.
  8. LM317 recommended output range `1.25 V to 37 V` —
     `23_LM317_datasheet.pdf`, page 6.
  9. LM75B specified temperature range `-55 °C to 125 °C` —
     `50_LM75B_datasheet.pdf`, page 4.
  10. INA226 bus input range `0 V to 36 V` — `25_INA226_datasheet.pdf`, page 5.

- The seed name intentionally says `verified`, not `reviewed`: the facts and page
  labels were checked against PDF extraction during this work, but independent
  human sign-off is still required before presenting them as a manually reviewed
  benchmark or publishing model-quality claims.
- Started a fresh local PostgreSQL 16 + pgvector service and installed the CPU
  model dependencies into ignored `.runtime/model-venv`. The Hugging Face cache
  initially contained only Qwen; BGE-M3 and the reranker artifacts remain ignored
  runtime data and are not included in the Git diff.
- Ingested exactly the 10 labeled seed documents with BGE-M3. The batch completed
  in 10 minutes 7 seconds:

  ```text
  383 chunks written
  10 files succeeded, 0 failed
  database: 10 distinct documents, 383 chunks, 0 missing embeddings
  ```

  Per-document chunk counts ranged from 13 (L293) to 70 (ADS1115). An initial
  verification query accidentally labeled joined `COUNT(*)` as `documents` and
  printed 383; it was immediately corrected to `COUNT(DISTINCT d.id)`, which
  confirmed the actual 10 documents. This SQL-check error did not modify data.
- Ran the real evaluation CLI with `candidate_k=10` and `top_n=5` over the 10
  seed questions:

  ```json
  {
    "question_count": 10,
    "candidate_k": 10,
    "candidate_recall_at_k": 1.0,
    "candidate_mrr": 0.425,
    "top_n": 5,
    "reranked_recall_at_n": 1.0,
    "reranked_mrr": 0.5116666666666666
  }
  ```

  All labeled evidence was present in both candidate and reranked cutoffs. The
  cross-encoder improved MRR by about 0.0867 while preserving recall. These are
  development baseline values, not publishable quality claims: the set has only
  10 fact-oriented questions, covers only 10 of 50 documents, and lacks
  independent human sign-off and difficult negative/multi-document cases.
- Final seed/schema verification completed with 15 tests passing in 0.17s, Ruff
  lint passing, and all 3 evaluation Python files matching Ruff format.

### 2026-08-11 — GPU runtime enablement

- Investigated why the previous ingestion/evaluation used CPU. The host had an
  NVIDIA GeForce RTX 3060 (12 GB), driver 580.95.05, and CUDA 13.0 capability,
  but two independent blockers existed: the ignored model environment contained
  `torch 2.9.1+cpu`, and Docker declared an NVIDIA runtime whose
  `nvidia-container-runtime` binary was not installed. `docker run --gpus all`
  consequently failed before a container could start.
- Installed NVIDIA Container Toolkit 1.19.1 from the already configured NVIDIA
  repository, ran `nvidia-ctk runtime configure --runtime=docker`, and restarted
  Docker. The pgvector container recovered through its restart policy and was
  confirmed healthy. A CUDA 12.8 base container then detected the RTX 3060 and
  all 12,288 MiB of VRAM.
- Replaced the ignored runtime's CPU wheel with `torch 2.9.1+cu128`. CUDA 12.8 is
  compatible with the newer host driver. Verification reported
  `torch.cuda.is_available() == True`, device `NVIDIA GeForce RTX 3060`, and
  BGE-M3 loaded on `cuda:0`.
- Warm BGE-M3 micro-benchmark over 10 duplicate query strings:

  ```text
  total: 0.0216 seconds
  per text: 0.0022 seconds
  output: 10 x 1024
  peak allocated VRAM: 2179.8 MiB
  ```

  This is a warm synthetic throughput check, not an end-to-end latency claim.
  The earlier CPU evaluation showed about 1.5 seconds for each single-query
  embedding, so GPU acceleration is material for this workload.
- Re-ran the real 10-question retrieval/reranking evaluation with GPU access.
  The aggregate metrics were identical to CPU, and total wall time including
  container startup and model loading was 27.986 seconds. This confirms the
  acceleration did not change ranking results on the seed.
- SentenceTransformer embeddings and CrossEncoder reranking automatically choose
  CUDA when available. The local Qwen generator did not: Transformers loaded it
  on CPU and tokenized inputs stayed there. Updated `answer.py` to move the model
  to CUDA when available and move its `BatchEncoding` to the model device, while
  preserving CPU fallback and fake-model unit tests.
- Real Qwen GPU smoke test loaded the model on `cuda:0`, returned the correct
  `4.5 V to 36 V [S1]` answer, and completed in 16.465 seconds wall time including
  container/model startup.
- Final worker regression after device-transfer changes: 41 tests passed in
  0.40s, Ruff lint passed, and all 18 worker Python files matched Ruff format.

### 2026-08-11 — Complete corpus ingestion on GPU

- Started after committing and pushing GPU support as `b612db4` (`Enable
  GPU-backed answer generation`). The development database currently has the 10
  verified-seed documents; this milestone will ingest only the 40 missing PDFs,
  preserving the already verified rows and avoiding unnecessary reprocessing.
- The missing set will be derived by comparing committed `datasheets/*.pdf`
  basenames with database filenames. Ingestion will run with Docker GPU access and
  the verified CUDA runtime. Completion requires 50 distinct documents, no missing
  embeddings, and a post-expansion seed evaluation to measure retrieval dilution
  from the larger corpus.
- The first GPU batch exposed a real memory constraint on the 12 GB RTX 3060.
  BGE-M3's default embedding batch of 16 on a long STM32 document allocated about
  8.93 GiB and then failed while requesting another 2.84 GiB. One shorter document
  completed before the run was stopped, leaving 11 documents in the database.
- Added `EMBEDDING_BATCH_SIZE` (default 16) as a validated runtime setting used
  whenever callers do not pass an explicit batch. Invalid, zero, and negative
  values fail before model encoding. The corpus retry will use batch 4, trading
  some throughput for predictable VRAM headroom instead of falling back to CPU.
- Retried exactly the remaining 39 PDFs with CUDA, batch size 4, and PyTorch's
  expandable-segments allocator. The run completed in 17 minutes 18 seconds:
  39 files succeeded, zero failed, and 5,138 chunks were written. The largest
  STM32 reference manual required 345 embedding batches and completed without
  another out-of-memory error.
- Post-ingestion validation matched all 50 committed PDF basenames to all 50
  database filenames, with no missing or extra rows. PostgreSQL now contains
  5,646 chunks and zero null embeddings.
- Re-ran the 10-question verified seed on the complete corpus using the RTX 3060.
  It finished in 36.182 seconds. Candidate recall@10 was `0.9`, candidate MRR
  `0.5311`, reranked recall@5 was `0.9`, and reranked MRR `0.4917`. Compared with
  the 10-document baseline (`1.0`, `0.425`, `1.0`, `0.5117`), early candidate
  ordering improved overall but one case was diluted out of the top 10.
- Per-question inspection identified the miss as `lm317-output-range`; its
  expected page did not appear in either candidate or reranked results. This is
  now a concrete retrieval-regression target. The remaining nine expected
  sources stayed within both cutoffs; these small seed metrics remain diagnostic,
  not production-quality claims.
- Final regression verification used Python 3.11 in the repository's ignored
  model runtime: 45 worker tests passed in 1.12 seconds, Ruff lint passed, all 18
  worker Python files matched Ruff format, and `git diff --check` passed. The old
  host-created `ingestion-worker/.venv` points to Python 3.10 and cannot install
  this package's Python >=3.11 dependencies; it was not used to claim success.

### 2026-08-11 — Full-corpus retrieval regression diagnosis

- Started after committing and pushing the complete-corpus milestone as
  `448a0af` (`Complete GPU corpus ingestion workflow`). The immediate target is
  the only verified-seed miss, `lm317-output-range`, which disappears before
  reranking because its expected source is absent from the hybrid top 10.
- Diagnosis will inspect the committed PDF extraction, stored LM317 chunks, and
  vector/keyword/fused candidate ranks independently. Any retrieval change must
  improve the missed case without reducing the existing 9/10 successes and must
  be verified against the complete 50-document database on GPU.
- Expanded retrieval showed that LM317 chunks occupied fused ranks 1 through 7
  when each channel was allowed 100 candidates. The page-6 chunk was only vector
  rank 12, so it did not enter the benchmark's 10-candidate pool; however, the
  page-1 LM317 chunk containing the same answer was vector rank 3 and fused rank
  4. The system retrieved valid evidence that the label did not accept.
- Direct `pdfplumber` inspection of committed PDF page 1 confirmed both the
  feature text `Adjustable: 1.25V to 37V` and the description stating the same
  output range. Added page 1 as a second valid expected source while retaining
  reviewed page 6. This corrects incomplete relevance judgments without changing
  retrieval code or retrofitting the answer to an unrelated source.
- The corrected 10-question full-corpus GPU run completed in 37.519 seconds with
  candidate recall@10 `1.0`, candidate MRR `0.5511`, reranked recall@5 `1.0`, and
  reranked MRR `0.5417`. This restores both recall measures and improves both MRR
  measures over the original 10-document baseline, while preserving the initial
  0.9 result above as an audit trail of why the relevance label changed.
- Final regression checks passed: 45 ingestion-worker tests in 1.22 seconds, 15
  evaluation tests in 0.85 seconds, Ruff lint across both packages, Ruff format
  for all 21 checked Python files, and `git diff --check`.

### 2026-08-11 — Query router and multi-hop orchestration

- Started after committing and pushing the corrected LM317 evidence as
  `bbc244c` (`Correct LM317 evaluation evidence`). This milestone implements the
  MVP's missing decision step before retrieval: single-hop document lookup,
  multi-hop comparison, electronics general knowledge, or out-of-domain reject.
- The router will request a small validated JSON decision from the configured
  local generator and fail safely to single-hop retrieval when model output is
  malformed. Multi-hop execution will retrieve each proposed subquery, merge and
  deduplicate chunks, and rerank the merged evidence against the original
  question. Unit tests will inject fake routing/retrieval functions, so routing
  behavior stays deterministic and does not download models in CI.
- Added `src/router.py` with four validated routes, strict structured-output
  parsing, a safe single-hop fallback, bounded two-to-four multi-hop subqueries,
  and an obvious out-of-domain guard that runs before the model. Fenced JSON and
  object-shaped subqueries from small models are normalized, while any model-
  supplied answer fields are discarded and never reach retrieval or generation.
- Integrated routing into `answer_question`. Single-hop retains the existing
  path; reject returns without model/retrieval work; general knowledge returns no
  fabricated document sources; multi-hop retrieves and reranks each subquery
  separately, then deduplicates balanced evidence before one cited generation.
  The API now reports the selected `route` in its typed response.
- Real Qwen 0.5B GPU routing correctly classified the L293 single-hop question.
  Before few-shot examples it emitted fenced/object JSON for the comparison;
  after parser and prompt hardening it produced the exact two expected subqueries.
  It still misclassified an exact football rejection example, so
  `FABRAG_ROUTER_ENABLED` defaults to `false`. This preserves the verified
  single-hop API until a stronger routing model passes a reviewed router set.
- The first real multi-hop run retrieved both L293 and DRV8833 documents and
  selected route `multi_hop`, but Qwen 0.5B answered with an unrelated operating
  temperature and omitted citations. The run took 64.292 seconds. Multi-hop was
  subsequently changed from a single merged rerank (which buried voltage chunks
  at S4/S5) to per-subquery balanced reranking; it remains opt-in pending another
  real-model verification.
- Attempting that verification with `Qwen/Qwen2.5-3B-Instruct` exposed a host
  runtime blocker before model download: the loaded NVIDIA kernel module is
  `580.95.05`, while installed userspace libraries are `580.173.02`. Both
  `nvidia-smi` and Docker fail with `NVML: driver/library version mismatch`.
  A host reboot is normally required to load the updated module, and was not
  performed without user authorization.
- Verification after the safe-default integration: 62 ingestion-worker tests,
  11 API tests, and 15 evaluation tests passed. Ruff lint passed across all three
  packages and all 28 checked Python files matched Ruff format. The API suite
  still reports the known upstream Starlette TestClient deprecation warning.

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

The development database contains the complete 50-document starter corpus:
5,646 chunks with embeddings. The verified seed covers 10 of those documents.

## Development environment

- Managed Python: `3.11.15`
- Verified Python 3.11 model runtime: `.runtime/model-venv` (ignored by Git;
  invoked inside the Python 3.11 container)
- Verified model runtime: PyTorch `2.9.1+cu128`, RTX 3060 (`cuda:0`)
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
45 tests passed
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
2. Diagnose and recover the full-corpus `lm317-output-range` recall regression.
3. Remove recurring headers/footers and improve PDF whitespace reconstruction.
4. Add heading-aware/tokenizer-aware chunking and section metadata.
5. Add a manually reviewed evaluation set and report recall@k, MRR, reranker
   before/after metrics, citation correctness and groundedness.
6. Add a query router, out-of-domain handling and multi-document/multi-hop flow.
7. Add API authentication, rate limiting, structured logs, feedback and CI/CD.

FabRAG remains an active-development RAG prototype. It must not yet be described
as a deployed or production-ready chatbot.
