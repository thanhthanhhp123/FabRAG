"""
Entry point: ingest one or more PDFs end-to-end into Postgres.

    python -m src.ingest ../datasheets/16_LM358_datasheet.pdf
    python -m src.ingest ../datasheets/*.pdf
    python -m src.ingest --all          # every PDF in ../datasheets

Re-running on the same file replaces its chunks (delete-then-insert) so
tweaking chunk size / embedding model during development doesn't pile up
stale rows.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from tqdm import tqdm

from .chunk import chunk_pages
from .db import EMBEDDING_DIM, Chunk, get_session, upsert_document
from .embed import EMBEDDING_MODEL_NAME, embed_texts
from .parse import parse_pdf

DATASHEETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasheets"


def ingest_file(path: str) -> int:
    """Parse, chunk, embed and store one PDF. Returns the number of chunks written."""
    filename = os.path.basename(path)
    pages = parse_pdf(path)
    chunks = chunk_pages(pages)

    if not chunks:
        print(f"  [skip] {filename}: no extractable text")
        return 0

    vectors = embed_texts([c.text for c in chunks])
    if len(vectors[0]) != EMBEDDING_DIM:
        raise ValueError(
            f"embedding dim {len(vectors[0])} != EMBEDDING_DIM {EMBEDDING_DIM} "
            "(update .env or the schema if the model changed)"
        )

    with get_session() as session:
        doc = upsert_document(session, filename=filename, num_pages=len(pages))

        # Replace old chunks for this doc (idempotent re-ingest).
        session.query(Chunk).filter_by(document_id=doc.id).delete()

        for c, vector in zip(chunks, vectors):
            session.add(
                Chunk(
                    document_id=doc.id,
                    chunk_index=c.chunk_index,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    chunking_strategy=c.chunking_strategy,
                    text=c.text,
                    embedding=vector,
                    embedding_model=EMBEDDING_MODEL_NAME,
                )
            )

        session.commit()

    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="PDF file(s) or glob(s) to ingest")
    parser.add_argument("--all", action="store_true", help="ingest every PDF in datasheets/")
    args = parser.parse_args()

    if args.all:
        targets = sorted(str(p) for p in DATASHEETS_DIR.glob("*.pdf"))
    else:
        targets = []
        for p in args.paths:
            matched = glob.glob(p)
            targets.extend(matched if matched else [p])

    if not targets:
        print("No files to ingest. Use --all or pass file path(s).")
        sys.exit(1)

    print(f"Ingesting {len(targets)} file(s)...")
    total_chunks = 0
    for path in tqdm(targets):
        try:
            n = ingest_file(path)
            total_chunks += n
        except Exception as e:  # noqa: BLE001 - continue processing the remaining PDFs
            print(f"  [FAILED] {os.path.basename(path)}: {type(e).__name__}: {e}")

    print(f"\nDone. {total_chunks} chunks written across {len(targets)} file(s).")


if __name__ == "__main__":
    main()
