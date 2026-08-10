from src.chunk import chunk_pages
from src.parse import Page


def test_chunk_pages_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    pages = [Page(number=1, text=text)]

    chunks = chunk_pages(pages, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    # step = chunk_size - overlap = 80, so chunk 1 should start at word80
    assert chunks[1].text.split()[0] == "word80"
    # last word of chunk 0 should reappear as part of chunk 1's overlap
    assert chunks[0].text.split()[-1] in chunks[1].text.split()


def test_chunk_pages_tracks_page_span():
    pages = [
        Page(number=1, text=" ".join(f"a{i}" for i in range(60))),
        Page(number=2, text=" ".join(f"b{i}" for i in range(60))),
    ]

    chunks = chunk_pages(pages, chunk_size=100, overlap=10)

    # first chunk spans page 1 into page 2
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2


def test_chunk_pages_empty_input():
    assert chunk_pages([]) == []


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_pages([Page(number=1, text="a b c")], chunk_size=10, overlap=10)
