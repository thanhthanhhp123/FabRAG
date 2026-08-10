import sys
from types import SimpleNamespace

import pytest

from src.parse import _extract_page, _render_table, parse_pdf


class FakeFilteredPage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakeTable:
    bbox = (10, 20, 100, 80)

    def extract(self):
        return [["Parameter", "Value"], ["Supply\nvoltage", None]]


class FakePage:
    def __init__(self, text="Body text"):
        self.text = text
        self.predicate = None

    def find_tables(self):
        return [FakeTable()]

    def filter(self, predicate):
        self.predicate = predicate
        return FakeFilteredPage(self.text)


def test_render_table_normalizes_cells():
    assert _render_table([[" Supply\nvoltage ", None]]) == "Supply voltage | "


def test_extract_page_keeps_prose_and_renders_each_table_once():
    page = FakePage()

    text = _extract_page(page)

    assert text == "Body text\n\nParameter | Value\nSupply voltage | "
    assert page.predicate({"x0": 20, "x1": 30, "top": 30, "bottom": 40}) is False
    assert page.predicate({"x0": 110, "x1": 120, "top": 30, "bottom": 40}) is True


def test_extract_page_preserves_empty_page():
    page = FakePage(text=None)
    page.find_tables = list

    assert _extract_page(page) == ""


def test_parse_pdf_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        parse_pdf(tmp_path / "missing.pdf")


def test_parse_pdf_rejects_non_pdf_file(tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("not a PDF")

    with pytest.raises(ValueError, match=r"Expected a \.pdf file"):
        parse_pdf(path)


def test_parse_pdf_numbers_pages_and_keeps_empty_pages(tmp_path, monkeypatch):
    path = tmp_path / "document.pdf"
    path.touch()
    fake_pages = [FakePage("First page"), FakePage(None)]
    fake_pages[1].find_tables = list

    class FakePdf:
        pages = fake_pages

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    fake_pdfplumber = SimpleNamespace(open=lambda opened_path: FakePdf())
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)

    pages = parse_pdf(path)

    assert [(page.number, page.text) for page in pages] == [
        (1, "First page\n\nParameter | Value\nSupply voltage | "),
        (2, ""),
    ]
