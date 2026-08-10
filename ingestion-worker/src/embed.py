"""
Step 3 of the pipeline: chunk text -> embedding vector.

Loads BAAI/bge-m3 once (module-level cache) and encodes in batches — loading
the model per-call would re-pay a multi-second startup cost for every chunk.
"""

from __future__ import annotations

import os

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Encode a list of chunk texts into vectors, normalized for cosine similarity."""
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > batch_size,
    )
    return vectors.tolist()


if __name__ == "__main__":
    vecs = embed_texts(["Supply voltage 3V to 32V", "Maximum output current 40mA"])
    print(f"model: {EMBEDDING_MODEL_NAME}")
    print(f"{len(vecs)} vectors, dim={len(vecs[0])}")
