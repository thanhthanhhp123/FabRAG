"""Rerank hybrid-retrieval candidates with a question-passage cross-encoder."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .hybrid_retrieve import HybridSearchResult, hybrid_retrieve

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")

_model: CrossEncoder | None = None


@dataclass(frozen=True)
class RerankedResult:
    text: str
    filename: str
    page_start: int | None
    page_end: int | None
    chunk_index: int
    reranker_score: float
    retrieval_rank: int
    vector_rank: int | None
    keyword_rank: int | None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(RERANKER_MODEL_NAME)
    return _model


def rerank(
    query: str,
    candidates: list[HybridSearchResult],
    top_n: int = 5,
) -> list[RerankedResult]:
    """Score query-passage pairs jointly and return the best passages."""
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")
    if not candidates:
        return []

    pairs = [(query, candidate.text) for candidate in candidates]
    scores = get_reranker().predict(pairs, show_progress_bar=len(pairs) > 8)
    if len(scores) != len(candidates):
        raise ValueError(f"reranker returned {len(scores)} scores for {len(candidates)} candidates")

    ranked = sorted(
        enumerate(zip(candidates, scores, strict=True), start=1),
        key=lambda item: float(item[1][1]),
        reverse=True,
    )
    return [
        RerankedResult(
            text=candidate.text,
            filename=candidate.filename,
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            chunk_index=candidate.chunk_index,
            reranker_score=float(score),
            retrieval_rank=retrieval_rank,
            vector_rank=candidate.vector_rank,
            keyword_rank=candidate.keyword_rank,
        )
        for retrieval_rank, (candidate, score) in ranked[:top_n]
    ]


def retrieve_and_rerank(
    query: str,
    candidate_k: int = 10,
    top_n: int = 5,
) -> list[RerankedResult]:
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero")
    candidates = hybrid_retrieve(query, top_k=candidate_k, candidate_k=candidate_k)
    return rerank(query, candidates, top_n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="question or search text")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    for rank, result in enumerate(
        retrieve_and_rerank(args.query, args.candidate_k, args.top_n), start=1
    ):
        pages = (
            str(result.page_start)
            if result.page_start == result.page_end
            else f"{result.page_start}-{result.page_end}"
        )
        print(
            f"\n[{rank}] reranker={result.reranker_score:.6f} "
            f"previous_rank={result.retrieval_rank} "
            f"source={result.filename} pages={pages} chunk={result.chunk_index}"
        )
        print(result.text[:800])


if __name__ == "__main__":
    main()
