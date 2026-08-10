import pytest

from src import retrieve


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_retrieve_rejects_empty_query(query):
    with pytest.raises(ValueError, match="query must not be empty"):
        retrieve.retrieve(query)


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retrieve.retrieve("supply voltage", top_k=top_k)


def test_retrieve_rejects_wrong_embedding_dimension(monkeypatch):
    monkeypatch.setattr(retrieve, "embed_texts", lambda texts: [[0.1, 0.2]])

    with pytest.raises(ValueError, match="query embedding dim 2"):
        retrieve.retrieve("supply voltage")
