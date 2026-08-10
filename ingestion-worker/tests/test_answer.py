from types import SimpleNamespace

import pytest

from src import answer
from src.rerank import RerankedResult


def evidence(index=0, start=1, end=1):
    return RerankedResult(
        text="The supply range is 4.5 V to 36 V.",
        filename="L293.pdf",
        page_start=start,
        page_end=end,
        chunk_index=index,
        reranker_score=0.9,
        retrieval_rank=1,
        vector_rank=1,
        keyword_rank=1,
    )


def test_build_evidence_prompt_assigns_source_ids_and_pages():
    prompt = answer.build_evidence_prompt("What is the voltage?", [evidence(end=2)])

    assert "Question: What is the voltage?" in prompt
    assert "[S1] File: L293.pdf; pages: 1-2" in prompt
    assert "The supply range is 4.5 V to 36 V." in prompt


def test_generate_without_evidence_does_not_load_model(monkeypatch):
    monkeypatch.setattr(answer, "get_generator", lambda: pytest.fail("model should not load"))

    result = answer.generate_from_evidence("Unknown specification?", [])

    assert result.text == "I don't have enough evidence."
    assert result.sources == ()


def test_generate_decodes_only_new_tokens(monkeypatch):
    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"] == answer.SYSTEM_PROMPT
            return "rendered prompt"

        def __call__(self, rendered, **kwargs):
            return {"input_ids": SimpleNamespace(shape=(1, 3))}

        def decode(self, tokens, **kwargs):
            assert tokens == [7, 8]
            return "The range is 4.5 V to 36 V [S1]."

    class FakeModel:
        def generate(self, **kwargs):
            return [[1, 2, 3, 7, 8]]

    monkeypatch.setattr(answer, "get_generator", lambda: (FakeTokenizer(), FakeModel()))

    result = answer.generate_from_evidence("What is the voltage?", [evidence()])

    assert result.text == "The range is 4.5 V to 36 V [S1]."
    assert len(result.sources) == 1
