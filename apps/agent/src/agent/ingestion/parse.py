from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from agent.ingestion.models import ParsedDocument, ParsedSection
from agent.ingestion.normalize import document_id_for, normalize_text

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


class UnsupportedDocumentError(ValueError):
    pass


def parse_path(path: Path, *, document_id: str | None = None) -> ParsedDocument:
    raw = path.read_bytes()
    return parse_bytes(
        raw,
        source=str(path),
        suffix=path.suffix,
        document_id=document_id,
    )


def parse_bytes(
    raw: bytes,
    *,
    source: str,
    suffix: str,
    document_id: str | None = None,
) -> ParsedDocument:
    ext = suffix.lower()
    doc_id = document_id or document_id_for(source, raw)
    if ext in {".txt", ".md", ".markdown"}:
        text = raw.decode("utf-8")
        media = "text/markdown" if ext in {".md", ".markdown"} else "text/plain"
        sections = (
            _sections_from_markdown(text) if ext in {".md", ".markdown"} else _single_section(text)
        )
        return ParsedDocument(
            document_id=doc_id,
            source=source,
            media_type=media,
            sections=sections,
            metadata={"filename": Path(source).name},
        )
    if ext == ".pdf":
        return _parse_pdf(raw, source=source, document_id=doc_id)
    raise UnsupportedDocumentError(f"Unsupported document type: {suffix}")


def _single_section(text: str) -> list[ParsedSection]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [ParsedSection(title=None, text=normalized)]


def _sections_from_markdown(text: str) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    current_title: str | None = None
    buf: list[str] = []

    def flush() -> None:
        body = normalize_text("\n".join(buf))
        if body:
            sections.append(ParsedSection(title=current_title, text=body))
        buf.clear()

    for line in text.replace("\r\n", "\n").split("\n"):
        match = _HEADING.match(line.strip())
        if match:
            flush()
            current_title = normalize_text(match.group(2))
            continue
        buf.append(line)
    flush()
    return sections


def _parse_pdf(raw: bytes, *, source: str, document_id: str) -> ParsedDocument:
    from io import BytesIO

    reader = PdfReader(BytesIO(raw))
    sections: list[ParsedSection] = []
    for index, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        normalized = normalize_text(extracted)
        if not normalized:
            continue
        sections.append(ParsedSection(title=f"page {index}", text=normalized))
    return ParsedDocument(
        document_id=document_id,
        source=source,
        media_type="application/pdf",
        sections=sections,
        metadata={"filename": Path(source).name, "page_count": len(reader.pages)},
    )
