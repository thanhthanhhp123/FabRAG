import json
import logging

import pytest
from fastapi.testclient import TestClient
from src.answer import GeneratedAnswer
from src.rerank import RerankedResult

from fabrag_api import main
from fabrag_api.observability import JsonFormatter
from fabrag_api.security import rate_limiter

client = TestClient(main.app)
API_HEADERS = {"X-API-Key": "test-secret-key"}


@pytest.fixture(autouse=True)
def configure_security(monkeypatch):
    monkeypatch.setenv("FABRAG_API_KEY", "test-secret-key")
    monkeypatch.setenv("FABRAG_RATE_LIMIT_REQUESTS", "30")
    monkeypatch.setenv("FABRAG_RATE_LIMIT_WINDOW_SECONDS", "60")
    rate_limiter.reset()


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
        headers=API_HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "What is the voltage?",
        "answer": "The range is 4.5 V to 36 V [S1].",
        "route": "single_hop",
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
        headers=API_HEADERS,
    )

    assert response.status_code == 422


def test_create_answer_rejects_blank_question():
    response = client.post("/v1/answers", json={"question": "   "}, headers=API_HEADERS)

    assert response.status_code == 422


def test_create_answer_hides_backend_error(monkeypatch):
    def fail(*args):
        raise RuntimeError("postgresql://user:secret@database/fabrag")

    monkeypatch.setattr(main, "answer_question", fail)

    response = client.post("/v1/answers", json={"question": "voltage"}, headers=API_HEADERS)

    assert response.status_code == 503
    assert response.json() == {"detail": "answer pipeline unavailable"}
    assert "secret" not in response.text


def test_answer_requires_configured_api_key(monkeypatch):
    monkeypatch.delenv("FABRAG_API_KEY")

    response = client.post("/v1/answers", json={"question": "voltage"})

    assert response.status_code == 503


def test_answer_rejects_invalid_api_key():
    response = client.post(
        "/v1/answers",
        json={"question": "voltage"},
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "ApiKey"


def test_rate_limit_returns_retry_after(monkeypatch):
    monkeypatch.setenv("FABRAG_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setattr(
        main,
        "answer_question",
        lambda question, candidate_k, top_n: GeneratedAnswer(question, "answer", ()),
    )

    first = client.post("/v1/answers", json={"question": "one"}, headers=API_HEADERS)
    limited = client.post("/v1/answers", json={"question": "two"}, headers=API_HEADERS)

    assert first.status_code == 200
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1


def test_invalid_rate_limit_configuration_fails_closed(monkeypatch):
    monkeypatch.setenv("FABRAG_RATE_LIMIT_REQUESTS", "unlimited")

    response = client.post("/v1/answers", json={"question": "voltage"}, headers=API_HEADERS)

    assert response.status_code == 503
    assert response.json() == {"detail": "API security configuration invalid"}


def test_request_id_is_preserved_or_generated():
    supplied = client.get("/health", headers={"X-Request-ID": "request_1234"})
    generated = client.get("/health", headers={"X-Request-ID": "bad id"})

    assert supplied.headers["X-Request-ID"] == "request_1234"
    assert len(generated.headers["X-Request-ID"]) == 32
    assert generated.headers["X-Request-ID"] != "bad id"


def test_json_formatter_emits_correlation_and_http_fields():
    record = logging.LogRecord("fabrag.access", logging.INFO, "", 0, "done", (), None)
    record.request_id = "request_1234"
    record.method = "POST"
    record.path = "/v1/answers"
    record.status_code = 200
    record.duration_ms = 12.5

    payload = json.loads(JsonFormatter().format(record))

    assert payload["request_id"] == "request_1234"
    assert payload["method"] == "POST"
    assert payload["path"] == "/v1/answers"
    assert payload["status_code"] == 200
    assert payload["duration_ms"] == 12.5
