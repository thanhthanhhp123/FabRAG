import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from fabrag_eval.evaluate import (
    EvaluationQuestion,
    ExpectedSource,
    evaluate_questions,
    first_relevant_rank,
    load_questions,
    source_matches,
)


@dataclass(frozen=True)
class Result:
    filename: str
    page_start: int | None
    page_end: int | None


def test_load_questions_validates_and_normalizes(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": " voltage ",
                "question": " What is the voltage? ",
                "reference_answer": " 4.5 V to 36 V. ",
                "expected_sources": [{"filename": " L293.pdf ", "page": 2}],
            }
        ),
        encoding="utf-8",
    )

    assert load_questions(path) == [
        EvaluationQuestion(
            "voltage",
            "What is the voltage?",
            "4.5 V to 36 V.",
            (ExpectedSource("L293.pdf", 2),),
        )
    ]


@pytest.mark.parametrize(
    "contents, message",
    [
        ("", "contains no questions"),
        (
            '{"id":"x","question":"q","expected_sources":[{"filename":"a"}]}',
            "reference_answer",
        ),
        (
            '{"id":"x","question":"q","reference_answer":"a","expected_sources":[]}',
            "non-empty list",
        ),
        (
            '{"id":"x","question":"q","reference_answer":"a","expected_sources":[{"filename":"a","page":0}]}',
            "page",
        ),
    ],
)
def test_load_questions_rejects_invalid_records(tmp_path, contents, message):
    path = tmp_path / "questions.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_questions(path)


def test_load_questions_rejects_duplicate_ids(tmp_path):
    record = (
        '{"id":"same","question":"q","reference_answer":"a","expected_sources":[{"filename":"a"}]}'
    )
    path = tmp_path / "questions.jsonl"
    path.write_text(f"{record}\n{record}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate id"):
        load_questions(path)


@pytest.mark.parametrize(
    "contents, message",
    [
        ('["not an object"]', "each record must be a JSON object"),
        (
            '{"id":"x","question":"q","reference_answer":"a","expected_sources":["not an object"]}',
            "expected source 1 must be an object",
        ),
    ],
)
def test_load_questions_rejects_wrong_json_types(tmp_path, contents, message):
    path = tmp_path / "questions.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(TypeError, match=message):
        load_questions(path)


def test_verified_seed_has_ten_distinct_questions_with_reference_answers():
    path = Path(__file__).parents[1] / "questions.verified.seed.jsonl"

    questions = load_questions(path)

    assert len(questions) == 10
    assert len({question.question_id for question in questions}) == 10
    assert all(question.reference_answer for question in questions)
    assert (
        len({source.filename for question in questions for source in question.expected_sources})
        == 10
    )


def test_source_matching_uses_filename_and_page_overlap():
    result = Result("doc.pdf", 2, 4)

    assert source_matches(result, ExpectedSource("doc.pdf", 3))
    assert source_matches(result, ExpectedSource("doc.pdf"))
    assert not source_matches(result, ExpectedSource("other.pdf", 3))
    assert not source_matches(result, ExpectedSource("doc.pdf", 5))


def test_first_relevant_rank_returns_one_based_rank():
    results = [Result("wrong.pdf", 1, 1), Result("right.pdf", 4, 5)]

    assert first_relevant_rank(results, [ExpectedSource("right.pdf", 5)]) == 2
    assert first_relevant_rank(results, [ExpectedSource("missing.pdf")]) is None


def test_evaluate_questions_reports_candidate_and_reranked_metrics():
    questions = [
        EvaluationQuestion("q1", "first", "answer 1", (ExpectedSource("a.pdf", 2),)),
        EvaluationQuestion("q2", "second", "answer 2", (ExpectedSource("b.pdf", 1),)),
    ]

    def retrieve(question, top_k, candidate_k):
        assert (top_k, candidate_k) == (3, 3)
        if question == "first":
            return [Result("wrong.pdf", 1, 1), Result("a.pdf", 1, 3)]
        return [Result("wrong.pdf", 1, 1)]

    def rerank_results(question, candidates, top_n):
        assert top_n == 2
        return list(reversed(candidates))

    summary = evaluate_questions(
        questions,
        candidate_k=3,
        top_n=2,
        retrieve_fn=retrieve,
        rerank_fn=rerank_results,
    )

    assert summary.question_count == 2
    assert summary.candidate_recall_at_k == 0.5
    assert summary.candidate_mrr == 0.25
    assert summary.reranked_recall_at_n == 0.5
    assert summary.reranked_mrr == 0.5


@pytest.mark.parametrize(("candidate_k", "top_n"), [(0, 1), (2, 0), (2, 3)])
def test_evaluate_questions_rejects_invalid_limits(candidate_k, top_n):
    question = EvaluationQuestion("q", "question", "answer", (ExpectedSource("a.pdf"),))

    with pytest.raises(ValueError):
        evaluate_questions([question], candidate_k=candidate_k, top_n=top_n)
