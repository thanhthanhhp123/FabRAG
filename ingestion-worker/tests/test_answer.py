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


def test_system_prompt_prioritizes_direct_evidence_and_requires_citations():
    assert "only source of truth" in answer.SYSTEM_PROMPT
    assert "directly states the requested value" in answer.SYSTEM_PROMPT
    assert "Every factual sentence must end with" in answer.SYSTEM_PROMPT


def test_generate_without_evidence_does_not_load_model(monkeypatch):
    monkeypatch.setattr(answer, "get_generator", lambda: pytest.fail("model should not load"))

    result = answer.generate_from_evidence("Unknown specification?", [])

    assert result.text == "I don't have enough evidence."
    assert result.sources == ()


def test_generate_decodes_only_new_tokens(monkeypatch):
    class FakeInputs(dict):
        def __init__(self):
            super().__init__(input_ids=SimpleNamespace(shape=(1, 3)))
            self.moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

    class FakeTokenizer:
        eos_token_id = 0

        def __init__(self):
            self.inputs = FakeInputs()

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"] == answer.SYSTEM_PROMPT
            assert messages[1]["content"] == answer.EXAMPLE_USER_PROMPT
            assert messages[2]["content"] == answer.EXAMPLE_ASSISTANT_ANSWER
            assert messages[3]["content"].startswith("Question: What is the voltage?")
            return "rendered prompt"

        def __call__(self, rendered, **kwargs):
            return self.inputs

        def decode(self, tokens, **kwargs):
            assert tokens == [7, 8]
            return "The range is 4.5 V to 36 V [S1]."

    class FakeModel:
        device = "cuda:0"

        def generate(self, **kwargs):
            return [[1, 2, 3, 7, 8]]

    tokenizer = FakeTokenizer()
    monkeypatch.setattr(answer, "get_generator", lambda: (tokenizer, FakeModel()))

    result = answer.generate_from_evidence("What is the voltage?", [evidence()])

    assert result.text == "The range is 4.5 V to 36 V [S1]."
    assert len(result.sources) == 1
    assert tokenizer.inputs.moved_to == "cuda:0"
