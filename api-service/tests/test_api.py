from fastapi.testclient import TestClient
from src.answer import GeneratedAnswer
from src.rerank import RerankedResult

from fabrag_api import main

client = TestClient(main.app)


def source() -> RerankedResult:
    return RerankedResult(
        text="The supply range is 4.5 V to 36 V.",
        filename="32_L293_datasheet.pdf",
        page_start=1,
        page_end=3,
        chunk_index=0,
        reranker_score=0.99,
        retrieval_rank=1,
        vector_rank=1,
        keyword_rank=2,
    )


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_answer_returns_citation_metadata(monkeypatch):
    def fake_answer(question, candidate_k, top_n):
        assert (question, candidate_k, top_n) == ("What is the voltage?", 8, 2)
        return GeneratedAnswer(
            question=question,
            text="The range is 4.5 V to 36 V [S1].",
            sources=(source(),),
        )

    monkeypatch.setattr(main, "answer_question", fake_answer)

    response = client.post(
        "/v1/answers",
        json={"question": "  What is the voltage?  ", "candidate_k": 8, "top_n": 2},
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "What is the voltage?",
        "answer": "The range is 4.5 V to 36 V [S1].",
        "sources": [
            {
                "source_id": "S1",
                "filename": "32_L293_datasheet.pdf",
                "page_start": 1,
                "page_end": 3,
                "chunk_index": 0,
                "reranker_score": 0.99,
                "retrieval_rank": 1,
                "vector_rank": 1,
                "keyword_rank": 2,
            }
        ],
    }


def test_create_answer_rejects_invalid_limits():
    response = client.post(
        "/v1/answers",
        json={"question": "voltage", "candidate_k": 2, "top_n": 3},
    )

    assert response.status_code == 422


def test_create_answer_rejects_blank_question():
    response = client.post("/v1/answers", json={"question": "   "})

    assert response.status_code == 422


def test_create_answer_hides_backend_error(monkeypatch):
    def fail(*args):
        raise RuntimeError("postgresql://user:secret@database/fabrag")

    monkeypatch.setattr(main, "answer_question", fail)

    response = client.post("/v1/answers", json={"question": "voltage"})

    assert response.status_code == 503
    assert response.json() == {"detail": "answer pipeline unavailable"}
    assert "secret" not in response.text
