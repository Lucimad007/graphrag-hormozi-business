from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.ingestion.chunk import chunk_document
from agent.ingestion.models import ChunkingConfig, ParsedDocument, ParsedSection
from agent.ingestion.normalize import normalize_text
from agent.ingestion.parse import UnsupportedDocumentError, parse_bytes, parse_path
from agent.ingestion.pipeline import parse_and_chunk


def test_normalize_text_collapses_whitespace() -> None:
    raw = "Low   close rate.\r\n\r\n\r\nNext   paragraph.\n"
    assert normalize_text(raw) == "Low close rate.\n\nNext paragraph."


def test_parse_markdown_sections() -> None:
    md = b"""# Diagnosis

Lots of leads, low close rate.

## Offer

The offer may be unclear.
"""
    parsed = parse_bytes(md, source="notes.md", suffix=".md", document_id="notes")
    assert parsed.media_type == "text/markdown"
    assert [s.title for s in parsed.sections] == ["Diagnosis", "Offer"]
    assert "low close rate" in parsed.sections[0].text.lower()


def test_parse_txt_single_section() -> None:
    parsed = parse_bytes(b"Hello world", source="a.txt", suffix=".txt", document_id="a")
    assert len(parsed.sections) == 1
    assert parsed.sections[0].title is None
    assert parsed.sections[0].text == "Hello world"


def test_unsupported_type() -> None:
    with pytest.raises(UnsupportedDocumentError):
        parse_bytes(b"{}", source="x.json", suffix=".json")


def test_chunk_ids_are_idempotent() -> None:
    doc = ParsedDocument(
        document_id="notes:abc",
        source="notes.md",
        media_type="text/markdown",
        sections=[ParsedSection(title="Offer", text="A " * 50 + "B " * 50)],
        metadata={"filename": "notes.md"},
    )
    config = ChunkingConfig(max_chars=40, overlap_chars=10)
    first = chunk_document(doc, config)
    second = chunk_document(doc, config)
    assert [c.id for c in first] == [c.id for c in second]
    assert first[0].id == "notes:abc:0"
    assert all(c.document == "notes.md" for c in first)
    assert all(c.section == "Offer" for c in first)


def test_chunk_windows_overlap() -> None:
    text = "alpha " * 30
    doc = ParsedDocument(
        document_id="d",
        source="d.txt",
        media_type="text/plain",
        sections=[ParsedSection(text=text)],
        metadata={"filename": "d.txt"},
    )
    chunks = chunk_document(doc, ChunkingConfig(max_chars=40, overlap_chars=8))
    assert len(chunks) >= 2
    assert chunks[0].text[:10] in text


def test_parse_and_chunk_from_path(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Metric\n\nClose rate is the conversion metric.\n", encoding="utf-8")
    chunks = parse_and_chunk(path, document_id="guide")
    assert chunks[0].section == "Metric"
    assert "conversion" in chunks[0].text
    assert chunks[0].id == "guide:0"


def test_parse_path_uses_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "plain.txt"
    path.write_text("same", encoding="utf-8")
    a = parse_path(path)
    b = parse_path(path)
    assert a.document_id == b.document_id
    assert a.document_id.startswith("plain:")


def test_parse_pdf_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    page1 = SimpleNamespace(extract_text=lambda: "Page one text about leads.")
    page2 = SimpleNamespace(extract_text=lambda: "Page two text about close rate.")
    reader = SimpleNamespace(pages=[page1, page2])
    monkeypatch.setattr("agent.ingestion.parse.PdfReader", lambda *_args, **_kw: reader)
    parsed = parse_bytes(b"%PDF-fake", source="book.pdf", suffix=".pdf", document_id="book")
    assert parsed.media_type == "application/pdf"
    assert [s.title for s in parsed.sections] == ["page 1", "page 2"]
    assert "close rate" in parsed.sections[1].text


def test_invalid_chunk_config() -> None:
    doc = ParsedDocument(
        document_id="d",
        source="d.txt",
        media_type="text/plain",
        sections=[ParsedSection(text="x")],
    )
    with pytest.raises(ValueError):
        chunk_document(doc, ChunkingConfig(max_chars=10, overlap_chars=10))
