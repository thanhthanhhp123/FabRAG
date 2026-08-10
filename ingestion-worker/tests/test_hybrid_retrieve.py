from unittest.mock import Mock

import pytest

from src import hybrid_retrieve


def make_chunk(chunk_id, index):
    chunk = Mock(spec=hybrid_retrieve.Chunk)
    chunk.id = chunk_id
    chunk.chunk_index = index
    chunk.text = f"chunk {index}"
    chunk.page_start = index + 1
    chunk.page_end = index + 1
    return chunk


def test_fuse_rankings_rewards_results_found_by_both_methods():
    shared = make_chunk(1, 0)
    vector_only = make_chunk(2, 1)
    keyword_only = make_chunk(3, 2)

    results = hybrid_retrieve._fuse_rankings(
        [(vector_only, "doc.pdf"), (shared, "doc.pdf")],
        [(keyword_only, "doc.pdf"), (shared, "doc.pdf")],
        rrf_k=60,
        top_k=3,
    )

    assert results[0].chunk_index == 0
    assert results[0].vector_rank == 2
    assert results[0].keyword_rank == 2
    assert {result.chunk_index for result in results[1:]} == {1, 2}


def test_fuse_rankings_uses_vector_as_the_stronger_signal():
    vector_first = make_chunk(1, 0)
    keyword_first = make_chunk(2, 1)

    results = hybrid_retrieve._fuse_rankings(
        [(vector_first, "doc.pdf"), (keyword_first, "doc.pdf")],
        [(keyword_first, "doc.pdf"), (vector_first, "doc.pdf")],
        rrf_k=60,
        top_k=2,
    )

    assert results[0].chunk_index == 0


def test_disjunctive_web_query_preserves_terms_and_adds_or():
    assert (
        hybrid_retrieve._disjunctive_web_query("L293 supply voltage 36V")
        == "L293 OR supply OR voltage OR 36V"
    )


@pytest.mark.parametrize(
    ("query", "top_k", "candidate_k", "rrf_k", "message"),
    [
        (" ", 5, 20, 60, "query"),
        ("voltage", 0, 20, 60, "top_k"),
        ("voltage", 5, 0, 60, "candidate_k"),
        ("voltage", 5, 20, 0, "rrf_k"),
    ],
)
def test_hybrid_retrieve_rejects_invalid_input(query, top_k, candidate_k, rrf_k, message):
    with pytest.raises(ValueError, match=message):
        hybrid_retrieve.hybrid_retrieve(query, top_k, candidate_k, rrf_k)
