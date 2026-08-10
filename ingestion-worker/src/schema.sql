-- FabRAG ingestion schema.
-- Runs automatically on first `docker compose up` via docker-entrypoint-initdb.d.
-- NOTE: only fires on an empty data volume. If you change this file after the
-- volume already exists, apply it manually (psql -f) or `docker compose down -v`
-- to reset — a proper migration tool (Alembic) is a stretch goal, not needed
-- for the MVP ingestion pipeline.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source PDF.
CREATE TABLE IF NOT EXISTS documents (
    id           SERIAL PRIMARY KEY,
    filename     TEXT NOT NULL UNIQUE,   -- e.g. "16_LM358_datasheet.pdf"
    title        TEXT,                   -- e.g. "LM358 Low-Power Dual Op-Amp"
    manufacturer TEXT,                   -- e.g. "Texas Instruments"
    source_url   TEXT,                   -- original download URL, for provenance
    num_pages    INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per chunk of text extracted from a document.
-- This is what gets embedded and what retrieval searches over.
CREATE TABLE IF NOT EXISTS chunks (
    id                 SERIAL PRIMARY KEY,
    document_id        INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index        INTEGER NOT NULL,       -- order within the document
    page_start         INTEGER,
    page_end           INTEGER,
    section            TEXT,                   -- nearest heading, if known
    chunking_strategy  TEXT NOT NULL DEFAULT 'fixed',  -- 'fixed' | 'heading' — lets eval compare strategies later
    text               TEXT NOT NULL,
    embedding          vector(1024),           -- dim must match EMBEDDING_DIM in .env
    embedding_model    TEXT,                   -- model name+version that produced this vector
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- Full-text index for BM25-style keyword search (exact part numbers etc.)
-- Added now so the retrieval step (later) doesn't require a migration.
CREATE INDEX IF NOT EXISTS idx_chunks_text_fts
    ON chunks USING GIN (to_tsvector('english', text));

-- NOTE: no ANN index (ivfflat/hnsw) on `embedding` yet — those need real data
-- to build well. Create it after the first full ingest, e.g.:
--   CREATE INDEX idx_chunks_embedding ON chunks
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
