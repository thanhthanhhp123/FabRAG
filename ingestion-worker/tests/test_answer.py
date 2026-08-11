from types import SimpleNamespace

import pytest

from src import answer
from src.hybrid_retrieve import HybridSearchResult
from src.rerank import RerankedResult
from src.router import Route, RouteDecision


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


def candidate(filename="L293.pdf", index=0, score=0.1):
    return HybridSearchResult(
        text="candidate evidence",
        filename=filename,
        page_start=1,
        page_end=1,
        chunk_index=index,
        rrf_score=score,
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


def test_answer_question_routes_single_hop_through_retrieval_and_reranking():
    calls = []

    def retrieve(query, top_k, candidate_k):
        calls.append(("retrieve", query, top_k, candidate_k))
        return [candidate()]

    def rerank(query, candidates, top_n):
        calls.append(("rerank", query, len(candidates), top_n))
        return [evidence()]

    result = answer.answer_question(
        "voltage?",
        8,
        2,
        route_fn=lambda _: RouteDecision(Route.SINGLE_HOP),
        retrieve_fn=retrieve,
        rerank_fn=rerank,
        evidence_generate_fn=lambda question, sources: answer.GeneratedAnswer(
            question, "answer [S1].", tuple(sources)
        ),
    )

    assert result.route is Route.SINGLE_HOP
    assert calls == [
        ("retrieve", "voltage?", 8, 8),
        ("rerank", "voltage?", 1, 2),
    ]


def test_answer_question_merges_and_deduplicates_multi_hop_candidates():
    queries = []

    def retrieve(query, top_k, candidate_k):
        queries.append(query)
        shared_score = 0.1 if query == "part A" else 0.2
        return [candidate("shared.pdf", 1, shared_score), candidate(f"{query}.pdf", 2)]

    def rerank(query, candidates, top_n):
        assert query in {"part A", "part B"}
        assert top_n == 3
        return [evidence()]

    result = answer.answer_question(
        "compare parts",
        10,
        3,
        route_fn=lambda _: RouteDecision(Route.MULTI_HOP, ("part A", "part B")),
        retrieve_fn=retrieve,
        rerank_fn=rerank,
        evidence_generate_fn=lambda question, sources: answer.GeneratedAnswer(
            question, "comparison [S1].", tuple(sources)
        ),
    )

    assert queries == ["part A", "part B"]
    assert result.route is Route.MULTI_HOP


def test_answer_question_disables_model_router_by_default(monkeypatch):
    monkeypatch.delenv("FABRAG_ROUTER_ENABLED", raising=False)
    result = answer.answer_question(
        "voltage?",
        retrieve_fn=lambda *args: [candidate()],
        rerank_fn=lambda *args: [evidence()],
        evidence_generate_fn=lambda question, sources: answer.GeneratedAnswer(
            question, "answer", tuple(sources)
        ),
    )

    assert result.route is Route.SINGLE_HOP


def test_answer_question_rejects_out_of_domain_without_loading_pipeline():
    result = answer.answer_question(
        "Who won the match?",
        route_fn=lambda _: RouteDecision(Route.REJECT),
        retrieve_fn=lambda *args: pytest.fail("retrieval should not run"),
        general_generate_fn=lambda _: pytest.fail("generation should not run"),
    )

    assert result.route is Route.REJECT
    assert result.sources == ()


def test_answer_question_uses_general_knowledge_generator_without_retrieval():
    result = answer.answer_question(
        "What is a resistor?",
        route_fn=lambda _: RouteDecision(Route.GENERAL_KNOWLEDGE),
        retrieve_fn=lambda *args: pytest.fail("retrieval should not run"),
        general_generate_fn=lambda question: answer.GeneratedAnswer(question, "general", ()),
    )

    assert result.route is Route.GENERAL_KNOWLEDGE
    assert result.text == "general"
