"""Evaluate hybrid retrieval and reranking against reviewed source labels."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from src.hybrid_retrieve import HybridSearchResult, hybrid_retrieve
from src.rerank import RerankedResult, rerank


class SourceResult(Protocol):
    filename: str
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class ExpectedSource:
    filename: str
    page: int | None = None


@dataclass(frozen=True)
class EvaluationQuestion:
    question_id: str
    question: str
    expected_sources: tuple[ExpectedSource, ...]


@dataclass(frozen=True)
class EvaluationSummary:
    question_count: int
    candidate_k: int
    top_n: int
    candidate_recall_at_k: float
    candidate_mrr: float
    reranked_recall_at_n: float
    reranked_mrr: float


def _required_string(record: dict[str, object], field: str, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {field} must be a non-empty string")
    return value.strip()


def load_questions(path: str | Path) -> list[EvaluationQuestion]:
    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise TypeError(f"line {line_number}: each record must be a JSON object")

            question_id = _required_string(record, "id", line_number)
            if question_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate id {question_id!r}")
            seen_ids.add(question_id)
            question = _required_string(record, "question", line_number)

            raw_sources = record.get("expected_sources")
            if not isinstance(raw_sources, list) or not raw_sources:
                raise ValueError(f"line {line_number}: expected_sources must be a non-empty list")
            sources: list[ExpectedSource] = []
            for source_index, raw_source in enumerate(raw_sources, start=1):
                if not isinstance(raw_source, dict):
                    raise TypeError(
                        f"line {line_number}: expected source {source_index} must be an object"
                    )
                filename = _required_string(raw_source, "filename", line_number)
                page = raw_source.get("page")
                if page is not None and (
                    not isinstance(page, int) or isinstance(page, bool) or page < 1
                ):
                    raise ValueError(
                        f"line {line_number}: expected source {source_index} page must be >= 1"
                    )
                sources.append(ExpectedSource(filename=filename, page=page))

            questions.append(EvaluationQuestion(question_id, question, tuple(sources)))
    if not questions:
        raise ValueError("evaluation file contains no questions")
    return questions


def source_matches(result: SourceResult, expected: ExpectedSource) -> bool:
    if result.filename != expected.filename:
        return False
    if expected.page is None:
        return True
    if result.page_start is None or result.page_end is None:
        return False
    return result.page_start <= expected.page <= result.page_end


def first_relevant_rank(
    results: Sequence[SourceResult], expected_sources: Sequence[ExpectedSource]
) -> int | None:
    for rank, result in enumerate(results, start=1):
        if any(source_matches(result, expected) for expected in expected_sources):
            return rank
    return None


def _aggregate(ranks: Sequence[int | None]) -> tuple[float, float]:
    recall = sum(rank is not None for rank in ranks) / len(ranks)
    mrr = sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks)
    return recall, mrr


def evaluate_questions(
    questions: Sequence[EvaluationQuestion],
    *,
    candidate_k: int = 10,
    top_n: int = 5,
    retrieve_fn: Callable[[str, int, int], list[HybridSearchResult]] = hybrid_retrieve,
    rerank_fn: Callable[[str, list[HybridSearchResult], int], list[RerankedResult]] = rerank,
) -> EvaluationSummary:
    if not questions:
        raise ValueError("questions must not be empty")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero")
    if top_n <= 0 or top_n > candidate_k:
        raise ValueError("top_n must be between 1 and candidate_k")

    candidate_ranks: list[int | None] = []
    reranked_ranks: list[int | None] = []
    for item in questions:
        candidates = retrieve_fn(item.question, candidate_k, candidate_k)
        reranked = rerank_fn(item.question, candidates, top_n)
        candidate_ranks.append(first_relevant_rank(candidates, item.expected_sources))
        reranked_ranks.append(first_relevant_rank(reranked, item.expected_sources))

    candidate_recall, candidate_mrr = _aggregate(candidate_ranks)
    reranked_recall, reranked_mrr = _aggregate(reranked_ranks)
    return EvaluationSummary(
        question_count=len(questions),
        candidate_k=candidate_k,
        top_n=top_n,
        candidate_recall_at_k=candidate_recall,
        candidate_mrr=candidate_mrr,
        reranked_recall_at_n=reranked_recall,
        reranked_mrr=reranked_mrr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", help="reviewed JSONL question/evidence file")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    summary = evaluate_questions(
        load_questions(args.questions), candidate_k=args.candidate_k, top_n=args.top_n
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
