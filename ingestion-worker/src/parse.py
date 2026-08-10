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
from typing import Any


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


def _outside_tables(obj: dict[str, Any], table_bboxes: list[tuple[float, ...]]) -> bool:
    """Return false for PDF objects whose center lies inside a detected table."""
    x = (float(obj["x0"]) + float(obj["x1"])) / 2
    y = (float(obj["top"]) + float(obj["bottom"])) / 2
    return not any(x0 <= x <= x1 and top <= y <= bottom for x0, top, x1, bottom in table_bboxes)


def _extract_page(page: Any) -> str:
    """Extract prose and tables once, avoiding table text duplicated as prose."""
    tables = page.find_tables()
    table_bboxes = [table.bbox for table in tables]
    prose_page = page.filter(lambda obj: _outside_tables(obj, table_bboxes))

    parts: list[str] = []
    body_text = prose_page.extract_text() or ""
    if body_text.strip():
        parts.append(body_text.strip())

    for table in tables:
        rendered = _render_table(table.extract())
        if rendered.strip():
            parts.append(rendered)

    return "\n\n".join(parts)


def parse_pdf(path: str | Path) -> list[Page]:
    """Extract prose and tables from every page of a PDF."""
    # Keep this import inside the function.  Chunking tests only need the
    # lightweight Page data class; they should not need to install a PDF
    # parser or open an actual PDF.
    import pdfplumber

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file: {pdf_path}")

    pages: list[Page] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            pages.append(Page(number=i, text=_extract_page(page)))

    return pages


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "../datasheets/16_LM358_datasheet.pdf"
    result = parse_pdf(target)
    print(f"{len(result)} pages parsed")
    print("--- page 1 preview ---")
    print(result[0].text[:800])
