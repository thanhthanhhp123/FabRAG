"""Hybrid retrieval using dense vectors, PostgreSQL full text, and RRF fusion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from sqlalchemy import func, literal_column, select

from .db import EMBEDDING_DIM, Chunk, Document, get_session
from .embed import embed_texts


@dataclass(frozen=True)
class HybridSearchResult:
    text: str
    filename: str
    page_start: int | None
    page_end: int | None
    chunk_index: int
    rrf_score: float
    vector_rank: int | None
    keyword_rank: int | None


def _disjunctive_web_query(query: str) -> str:
    """Ask Postgres to match any query term; ranking rewards multiple matches."""
    return " OR ".join(query.split())


def _fuse_rankings(
    vector_rows: list[tuple[Chunk, str]],
    keyword_rows: list[tuple[Chunk, str]],
    *,
    rrf_k: int,
    top_k: int,
    vector_weight: float = 2.0,
    keyword_weight: float = 1.0,
) -> list[HybridSearchResult]:
    candidates: dict[int, dict[str, object]] = {}

    rankings = (
        ("vector_rank", vector_rows, vector_weight),
        ("keyword_rank", keyword_rows, keyword_weight),
    )
    for source, rows, weight in rankings:
        for rank, (chunk, filename) in enumerate(rows, start=1):
            candidate = candidates.setdefault(
                chunk.id,
                {
                    "chunk": chunk,
                    "filename": filename,
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "keyword_rank": None,
                },
            )
            candidate[source] = rank
            candidate["rrf_score"] = float(candidate["rrf_score"]) + weight / (rrf_k + rank)

    ordered = sorted(candidates.values(), key=lambda item: float(item["rrf_score"]), reverse=True)
    results: list[HybridSearchResult] = []
    for candidate in ordered[:top_k]:
        chunk = candidate["chunk"]
        assert isinstance(chunk, Chunk)
        results.append(
            HybridSearchResult(
                text=chunk.text,
                filename=str(candidate["filename"]),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                chunk_index=chunk.chunk_index,
                rrf_score=float(candidate["rrf_score"]),
                vector_rank=candidate["vector_rank"],
                keyword_rank=candidate["keyword_rank"],
            )
        )
    return results


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
    rrf_k: int = 60,
) -> list[HybridSearchResult]:
    """Fuse semantic and exact-term rankings and return the best chunks."""
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")

    candidate_k = max(candidate_k, top_k)
    query_vector = embed_texts([query])[0]
    if len(query_vector) != EMBEDDING_DIM:
        raise ValueError(
            f"query embedding dim {len(query_vector)} != EMBEDDING_DIM {EMBEDDING_DIM}"
        )

    vector_distance = Chunk.embedding.cosine_distance(query_vector)
    vector_statement = (
        select(Chunk, Document.filename)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.embedding.is_not(None))
        .order_by(vector_distance)
        .limit(candidate_k)
    )

    english = literal_column("'english'::regconfig")
    document_terms = func.to_tsvector(english, Chunk.text)
    query_terms = func.websearch_to_tsquery(english, _disjunctive_web_query(query))
    keyword_rank = func.ts_rank_cd(document_terms, query_terms)
    keyword_statement = (
        select(Chunk, Document.filename)
        .join(Document, Chunk.document_id == Document.id)
        .where(document_terms.op("@@")(query_terms))
        .order_by(keyword_rank.desc())
        .limit(candidate_k)
    )

    with get_session() as session:
        vector_rows = list(session.execute(vector_statement).tuples())
        keyword_rows = list(session.execute(keyword_statement).tuples())

    return _fuse_rankings(
        vector_rows,
        keyword_rows,
        rrf_k=rrf_k,
        top_k=top_k,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="question or search text")
    parser.add_argument("--top-k", type=int, default=5, help="number of chunks to return")
    parser.add_argument(
        "--candidate-k", type=int, default=20, help="candidates from each retrieval method"
    )
    args = parser.parse_args()

    for rank, result in enumerate(
        hybrid_retrieve(args.query, args.top_k, args.candidate_k), start=1
    ):
        pages = (
            str(result.page_start)
            if result.page_start == result.page_end
            else f"{result.page_start}-{result.page_end}"
        )
        print(
            f"\n[{rank}] rrf={result.rrf_score:.6f} "
            f"vector_rank={result.vector_rank} keyword_rank={result.keyword_rank} "
            f"source={result.filename} pages={pages} chunk={result.chunk_index}"
        )
        print(result.text[:800])


if __name__ == "__main__":
    main()
