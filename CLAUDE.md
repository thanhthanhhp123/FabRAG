# FabRAG — Technical Docs Assistant for Electronics Manufacturing

> RAG system that answers questions over manufacturing technical documents
> (component datasheets, IPC standards, SOPs, work instructions) with cited
> sources. Built as a portfolio project to demonstrate the LLM
> application-layer (retrieval, agent, production deployment) alongside
> existing CV/deployment work.

## 1. Context & Motivation

- Author is an AI Engineer at USI (PCB defect inspection: YOLOv8, TensorRT,
  ONNX, multi-camera production line) and final-year AI & Robotics student
  at Phenikaa University.
- Existing portfolio strength: computer vision + model deployment
  (PCB defect inspection, VisionOCR with Qwen2.5-VL + LoRA + vLLM, 3D Gaussian
  Splatting, strawberry weight estimation, automatic checkout system).
- Gap: no project demonstrating the LLM **application layer**
  (retrieval, agents, RAG evaluation) — increasingly expected in AI Engineer
  JDs alongside CV/backend skills.
- Narrative goal: "I built CV to detect PCB defects → I built FabRAG so
  engineers can instantly look up the standard/procedure for handling that
  defect." Keeps the whole portfolio coherent around one domain
  (electronics manufacturing) instead of scattering across unrelated demos.
- Target: not a local demo notebook — a **production system** with a live
  URL, tests, CI/CD, and basic observability, because that's what
  differentiates this project in interviews.

## 2. Scope

**In scope (MVP):**
- Ingest PDFs (component datasheets, IPC standards excerpts, equipment
  manuals) → parsed, chunked, embedded, stored with metadata.
- Query router agent (1 LLM call, function-calling) run before retrieval:
  decides retrieval vs. general-knowledge answer, single-hop vs. multi-hop
  (e.g. comparing two components requires two separate retrieval passes
  merged into one answer), and out-of-domain rejection (decline politely
  instead of hallucinating on non-electronics-manufacturing questions).
  A real decision-making step, not a fixed chain — this is the project's
  "agent" claim.
- Hybrid retrieval (vector + BM25) with reranking.
- Answer generation with mandatory source citation (doc, page, section).
- REST API (FastAPI) + minimal web UI for demo.
- Deployed, publicly reachable, running continuously (not "starts when I run
  it locally").
- Quantitative evaluation (retrieval + generation metrics), not just "it
  works when I tried it."

**Out of scope (for now — see Section 8 for stretch goals):**
- Multi-turn conversational memory / chat history.
- Writing back to source documents.
- Multi-language support beyond English (Vietnamese optional stretch).
- Fine-tuning the LLM (use an off-the-shelf instruct model; fine-tuning is
  already demonstrated in the VisionOCR project — no need to repeat it here).

## 3. Architecture

Three independently deployable services:

```
┌─────────────┐      ┌──────────────────────────────┐      ┌─────────────────┐
│   ingestion  │      │             api               │      │    llm-server    │
│   worker     │      │   (FastAPI)                   │      │ (vLLM / llama.cpp)│
│ (Celery+Redis)│─────▶│  query router → retrieve →   │─────▶│   separate scaling│
└─────────────┘      │  (single/multi-hop/reject) →  │      └─────────────────┘
       │              │   rerank → generate            │
       │              └──────────────────────────────┘
       ▼                       │
┌─────────────────────────────────────┐
│   Postgres + pgvector (Neon/Supabase) │
│   documents / chunks / embeddings /   │
│   feedback / eval_runs tables         │
└─────────────────────────────────────┘
```

- **ingestion-worker**: parses PDFs, chunks, embeds, writes to Postgres.
  Runs async via Celery so large-document ingestion never blocks the API.
- **api**: stateless FastAPI service. First runs the query router (LLM
  function-call: retrieval-needed? single- vs. multi-hop? in-domain?),
  then handles retrieval (vector + BM25 + rerank) accordingly — multi-hop
  questions get two+ retrieval passes merged before generation,
  out-of-domain questions short-circuit to a polite decline without
  calling retrieval or the generation LLM. Prompts the LLM, returns cited
  answers. Rate-limited, API-key gated.
- **llm-server**: generation model served separately from the API so it can
  be scaled/swapped independently (e.g. move from CPU llama.cpp to a GPU
  vLLM instance later without touching the API code). Serves both the
  router's function-calling call and the final answer-generation call.

## 4. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| PDF parsing | pdfplumber / unstructured.io | Handles tables + text in datasheets |
| Chunking | Custom — fixed-size AND semantic/heading-based (compare both) | Evaluation should show which wins for this domain |
| Embeddings | BAAI/bge-m3 (or multilingual-e5) | Strong open-source retrieval embeddings |
| Query router | 1 LLM call, function-calling (same model as generation) | Real agentic decision step: retrieval-vs-general-knowledge, single- vs. multi-hop, in- vs. out-of-domain — cheap (~2-3 days) but genuine "agent" claim for interviews |
| Vector DB | Postgres + pgvector (Neon/Supabase free tier) | Reuses existing Postgres skill, no separate vector DB service to operate |
| Keyword search | BM25 (e.g. via Postgres full-text or rank_bm25) | Hybrid retrieval outperforms vector-only on exact part numbers/codes |
| Reranker | bge-reranker (cross-encoder) | Cheap accuracy boost after hybrid retrieval |
| LLM | Qwen2.5-Instruct (3B/7B, GGUF quantized) | Same model family as VisionOCR project — consistent narrative; runs on CPU via llama.cpp for a cheap always-on deployment |
| LLM serving | llama.cpp (CPU VPS) or vLLM (GPU spot instance) | Explicit cost/latency trade-off — worth explaining in interviews |
| Backend | FastAPI + Celery + Redis | Same stack as VisionOCR project |
| Frontend | Streamlit or minimal React | Just enough to demo; not the focus |
| CI/CD | GitHub Actions: lint (ruff) → test → build → push | Signals "production," not just "shippable code" |
| Eval | RAGAS (faithfulness, answer relevancy) + custom recall@k / MRR | Quantifiable claims for the CV, not "it works well" |
| Observability | Structured logs (latency, tokens, retrieved docs) → Postgres or Grafana+Prometheus | Answers "how do you know it's healthy in production" |

## 5. Data

**Sources (public, freely available for engineering use):**
- Component datasheets: manufacturer sites (ST, TI, Analog Devices,
  Microchip) — MCUs, op-amps, passives, connectors.
- IPC standards: publicly available summaries/excerpts (full standards are
  paid — do not redistribute paid IPC documents).
- Equipment manuals: AOI/SMT/reflow oven manuals where manufacturers
  publish them publicly.

**Target corpus size:** 50–100 documents for a meaningful demo.

**Bulk collection:** Claude's sandboxed environment cannot crawl these
domains at scale (network is restricted to package registries + GitHub).
Use `scripts/bulk_download.py` locally (full internet access) — seed URLs
already verified, add more by searching `"<part> datasheet site:st.com"` /
`site:ti.com` / etc. and copying the direct PDF link.

**Metadata to keep per chunk:** source filename, page number, section
heading — required for citations in generated answers.

## 6. Evaluation Plan

Do not skip this — it's the part that differentiates the project.

1. Write 30–50 test questions by hand (or LLM-generate + manually review —
   do not ship ungenerated/unreviewed eval questions). Include cases for
   all three router branches: needs-retrieval vs. general-knowledge,
   single- vs. multi-hop, in-domain vs. out-of-domain — router accuracy on
   these is a reportable metric, not just retrieval/generation quality.
2. Retrieval metrics: recall@k, MRR — compare fixed-size vs. semantic
   chunking, and with/without reranking.
3. Generation metrics: RAGAS faithfulness + answer relevancy.
4. Log real numbers. Use them as the "impact metrics" on the CV instead of
   fabricated business-impact figures.

## 7. Production Requirements (MVP bar)

- [ ] Deployed and reachable at a stable URL, running continuously.
- [ ] Dockerized services + docker-compose for local dev.
- [ ] CI pipeline: lint → unit tests → integration test (fixed question set,
      end-to-end) → build → push image.
- [ ] API key auth + basic rate limiting.
- [ ] Input validation (query length limits, basic prompt-injection
      guardrails).
- [ ] Structured logging per request: retrieval latency, generation
      latency, token count, which docs were retrieved.
- [ ] Feedback capture: thumbs up/down per answer, stored — this is the
      direct answer to "how do you know it's good in production."
- [ ] Incremental ingestion (adding a new document does not require
      re-embedding the whole corpus).
- [ ] Embedding model version tracked (changing embedding models requires
      full re-index — document this as a known operational cost).

## 8. Stretch Goals (only after MVP + Section 7 checklist is done)

- Query caching for repeated questions (Redis).
- A/B test chunking strategies on real traffic instead of only offline eval.
- Simple Grafana dashboard instead of raw logs.

## 9. Non-goals / Explicit Trade-offs to Be Ready to Explain

- CPU-served LLM (llama.cpp) instead of always-on GPU: cost vs. latency
  trade-off for a side project — explain this clearly rather than treating
  it as a limitation to hide.
- No fine-tuning here: fine-tuning is already demonstrated elsewhere
  (VisionOCR); this project's job is to prove the retrieval/application
  layer, not to repeat a skill already shown.
- Small eval set (30–50 questions): enough to be directionally meaningful
  and honestly reported, not claimed as statistically rigorous.

## 10. Repo Conventions (for Claude Code / contributors)

- Python 3.11+, `ruff` for lint/format, `pytest` for tests.
- One `Dockerfile` per service (`ingestion-worker/`, `api/`, `llm-server/`),
  shared `docker-compose.yml` at repo root for local dev.
- Config via environment variables (`.env.example` checked in, `.env`
  gitignored) — never hardcode API keys or DB URLs.
- All retrieval/generation prompts live in a single `prompts/` directory,
  version-controlled — makes prompt changes reviewable and evaluable.
- Every PR that touches retrieval or prompting must show before/after eval
  numbers (recall@k, RAGAS scores) in the PR description.

## 11. Current Status

- [x] Project scoped and named.
- [x] Starter corpus: 2 real datasheets fetched and parsed
      (STM32F103C8 — ST, LM358 — TI).
- [x] Bulk corpus collection: 50/50 datasheets downloaded to `datasheets/`.
- [ ] Ingestion pipeline.
- [ ] Retrieval + generation.
- [ ] Deployment.
- [ ] Evaluation harness.
- [ ] Production hardening (Section 7 checklist).

## 12. Timeline (rough)

3–4 weekend/evening weeks for MVP (Sections 3–6, now including the query
router agent, +~2-3 days), +1–2 weeks for production hardening (Section 7)
if time allows before job applications need the portfolio link.