from src import rerank
from src.hybrid_retrieve import HybridSearchResult


class FakeScores(list):
    pass


class FakeReranker:
    def __init__(self, scores):
        self.scores = FakeScores(scores)
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores


def candidate(index):
    return HybridSearchResult(
        text=f"chunk {index}",
        filename="doc.pdf",
        page_start=index + 1,
        page_end=index + 1,
        chunk_index=index,
        rrf_score=0.01,
        vector_rank=index + 1,
        keyword_rank=None,
    )


def test_rerank_sorts_scores_and_preserves_original_retrieval_rank(monkeypatch):
    model = FakeReranker([0.2, 0.9, -0.1])
    monkeypatch.setattr(rerank, "_model", model)

    results = rerank.rerank("supply voltage", [candidate(0), candidate(1), candidate(2)], 2)

    assert [result.chunk_index for result in results] == [1, 0]
    assert [result.retrieval_rank for result in results] == [2, 1]
    assert model.calls[0][0] == [
        ("supply voltage", "chunk 0"),
        ("supply voltage", "chunk 1"),
        ("supply voltage", "chunk 2"),
    ]
    assert model.calls[0][1] == {"show_progress_bar": False}


def test_rerank_empty_candidates_does_not_load_model(monkeypatch):
    monkeypatch.setattr(rerank, "get_reranker", lambda: None)

    assert rerank.rerank("supply voltage", []) == []
