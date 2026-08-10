import pytest

from src import embed


class FakeVectors:
    def tolist(self):
        return [[0.6, 0.8]]


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return FakeVectors()


def test_embed_texts_normalizes_and_batches(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(embed, "_model", model)

    vectors = embed.embed_texts(["supply voltage"], batch_size=8)

    assert vectors == [[0.6, 0.8]]
    assert model.calls == [
        (
            ["supply voltage"],
            {"batch_size": 8, "normalize_embeddings": True, "show_progress_bar": False},
        )
    ]


def test_embed_texts_empty_input_does_not_load_model(monkeypatch):
    monkeypatch.setattr(embed, "get_model", lambda: pytest.fail("model should not be loaded"))

    assert embed.embed_texts([]) == []


def test_embed_texts_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        embed.embed_texts(["a chunk"], batch_size=0)
