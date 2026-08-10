"""
Step 1 of the pipeline: PDF -> plain text, per page.

Datasheets are mostly tables (electrical characteristics, pin descriptions),
so we extract tables explicitly and render them as simple pipe-separated text
inline with the prose — losing the visual grid but keeping every value
associated with its row/column labels, which is what the embedding model and
the LLM actually need.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    number: int  # 1-indexed, matches what a human would cite ("page 12")
    text: str


def _render_table(table: list[list[str | None]]) -> str:
    rows = []
    for row in table:
        cells = [(cell or "").strip().replace("\n", " ") for cell in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_pdf(path: str | Path) -> list[Page]:
    """Extract text + tables from every page of a PDF, in reading order."""
    # Keep this import inside the function.  Chunking tests only need the
    # lightweight Page data class; they should not need to install a PDF
    # parser or open an actual PDF.
    import pdfplumber

    pages: list[Page] = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            parts: list[str] = []

            body_text = page.extract_text() or ""
            if body_text.strip():
                parts.append(body_text.strip())

            for table in page.extract_tables():
                rendered = _render_table(table)
                if rendered.strip():
                    parts.append(rendered)

            pages.append(Page(number=i, text="\n\n".join(parts)))

    return pages


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "../datasheets/16_LM358_datasheet.pdf"
    result = parse_pdf(target)
    print(f"{len(result)} pages parsed")
    print("--- page 1 preview ---")
    print(result[0].text[:800])
