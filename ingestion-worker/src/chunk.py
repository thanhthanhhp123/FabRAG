"""
Step 2 of the pipeline: pages -> chunks.

Fixed-size strategy (baseline): slide a fixed-width window over the words of
the whole document, with overlap so a fact split across a window boundary
still appears whole in at least one chunk. Word count is a rough proxy for
token count (~0.75 tokens/word for English) — close enough for chunk sizing;
we don't tokenize with the embedding model's own tokenizer here to keep this
step model-agnostic.

`chunking_strategy="fixed"` is tagged per chunk so the eval step (CLAUDE.md
section 6) can later compare this against a heading-based strategy without
a schema change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .parse import Page

DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE_TOKENS", "400"))
DEFAULT_OVERLAP = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "50"))


@dataclass
class ChunkRecord:
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    chunking_strategy: str = "fixed"


def chunk_pages(
    pages: list[Page],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[ChunkRecord]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Flatten to (word, page_number) so a sliding window can span page
    # breaks while each chunk still knows which pages it touched.
    words: list[tuple[str, int]] = []
    for page in pages:
        for word in page.text.split():
            words.append((word, page.number))

    if not words:
        return []

    chunks: list[ChunkRecord] = []
    step = chunk_size - overlap
    start = 0
    index = 0

    while start < len(words):
        window = words[start : start + chunk_size]
        text = " ".join(w for w, _ in window)
        page_start = window[0][1]
        page_end = window[-1][1]

        chunks.append(
            ChunkRecord(
                chunk_index=index,
                text=text,
                page_start=page_start,
                page_end=page_end,
            )
        )

        index += 1
        start += step

    return chunks


if __name__ == "__main__":
    import sys

    from .parse import parse_pdf

    target = sys.argv[1] if len(sys.argv) > 1 else "../datasheets/16_LM358_datasheet.pdf"
    pages = parse_pdf(target)
    chunks = chunk_pages(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    print("--- chunk 0 ---")
    print(f"pages {chunks[0].page_start}-{chunks[0].page_end}")
    print(chunks[0].text[:500])
