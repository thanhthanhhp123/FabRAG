import sys

import pytest

from src import ingest


def test_main_reports_success(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ingest", "document.pdf"])
    monkeypatch.setattr(ingest, "ingest_file", lambda path: 3)

    ingest.main()

    output = capsys.readouterr().out
    assert "3 chunks written" in output
    assert "1 file(s) succeeded, 0 failed" in output


def test_main_returns_nonzero_when_any_file_fails(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ingest", "good.pdf", "bad.pdf"])

    def fake_ingest(path):
        if path == "bad.pdf":
            raise RuntimeError("broken PDF")
        return 3

    monkeypatch.setattr(ingest, "ingest_file", fake_ingest)

    with pytest.raises(SystemExit) as exc_info:
        ingest.main()

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "[FAILED] bad.pdf: RuntimeError: broken PDF" in output
    assert "1 file(s) succeeded, 1 failed" in output
