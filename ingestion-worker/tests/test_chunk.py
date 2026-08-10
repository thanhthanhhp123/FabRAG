import pytest

from src.chunk import chunk_pages
from src.parse import Page


def test_chunk_pages_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    pages = [Page(number=1, text=text)]

    chunks = chunk_pages(pages, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    # step = chunk_size - overlap = 80, so chunk 1 should start at word80
    assert chunks[1].text.split()[0] == "word80"
    # The final 20 words of chunk 0 must be the first 20 words of chunk 1.
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]
    assert all(len(chunk.text.split()) <= 100 for chunk in chunks)


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


def test_chunk_pages_does_not_create_overlap_only_trailing_chunk():
    text = " ".join(f"word{i}" for i in range(100))

    chunks = chunk_pages([Page(number=1, text=text)], chunk_size=100, overlap=20)

    assert len(chunks) == 1
    assert chunks[0].text.split() == text.split()


def test_chunk_pages_final_chunk_contains_new_content_and_sequential_index():
    text = " ".join(f"word{i}" for i in range(101))

    chunks = chunk_pages([Page(number=1, text=text)], chunk_size=100, overlap=20)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert chunks[1].text.split() == text.split()[80:]
    assert chunks[1].text.split()[-1] == "word100"


def test_chunk_pages_skips_empty_pages_and_keeps_source_page_number():
    pages = [
        Page(number=4, text="   "),
        Page(number=5, text="one two three"),
    ]

    chunks = chunk_pages(pages, chunk_size=10, overlap=0)

    assert len(chunks) == 1
    assert chunks[0].text == "one two three"
    assert (chunks[0].page_start, chunks[0].page_end) == (5, 5)


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [
        (0, 0, "chunk_size"),
        (-1, 0, "chunk_size"),
        (10, -1, "negative"),
        (10, 10, "smaller"),
        (10, 11, "smaller"),
    ],
)
def test_chunk_pages_rejects_invalid_sizes(chunk_size, overlap, message):
    with pytest.raises(ValueError, match=message):
        chunk_pages([Page(number=1, text="a b c")], chunk_size=chunk_size, overlap=overlap)
