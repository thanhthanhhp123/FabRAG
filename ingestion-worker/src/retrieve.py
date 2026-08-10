"""Vector retrieval: question -> nearest stored chunks with source metadata."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from sqlalchemy import select

from .db import EMBEDDING_DIM, Chunk, Document, get_session
from .embed import embed_texts


@dataclass(frozen=True)
class SearchResult:
    text: str
    filename: str
    page_start: int | None
    page_end: int | None
    chunk_index: int
    score: float


def retrieve(query: str, top_k: int = 5) -> list[SearchResult]:
    """Return the chunks with the highest cosine similarity to ``query``."""
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_vector = embed_texts([query])[0]
    if len(query_vector) != EMBEDDING_DIM:
        raise ValueError(
            f"query embedding dim {len(query_vector)} != EMBEDDING_DIM {EMBEDDING_DIM}"
        )

    distance = Chunk.embedding.cosine_distance(query_vector)
    statement = (
        select(Chunk, Document.filename, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )

    with get_session() as session:
        rows = session.execute(statement).all()

    return [
        SearchResult(
            text=chunk.text,
            filename=filename,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chunk_index=chunk.chunk_index,
            score=1.0 - float(row_distance),
        )
        for chunk, filename, row_distance in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="question or search text")
    parser.add_argument("--top-k", type=int, default=5, help="number of chunks to return")
    args = parser.parse_args()

    for rank, result in enumerate(retrieve(args.query, args.top_k), start=1):
        pages = (
            str(result.page_start)
            if result.page_start == result.page_end
            else f"{result.page_start}-{result.page_end}"
        )
        print(
            f"\n[{rank}] score={result.score:.4f} "
            f"source={result.filename} pages={pages} chunk={result.chunk_index}"
        )
        print(result.text[:800])


if __name__ == "__main__":
    main()
