from __future__ import annotations

from agent.ingestion.models import ChunkingConfig, ParsedDocument, TextChunk


def chunk_document(
    document: ParsedDocument,
    config: ChunkingConfig | None = None,
) -> list[TextChunk]:
    cfg = config or ChunkingConfig()
    if cfg.max_chars < 1:
        raise ValueError("max_chars must be a positive int")
    if cfg.overlap_chars < 0 or cfg.overlap_chars >= cfg.max_chars:
        raise ValueError("overlap_chars must be >= 0 and < max_chars")

    chunks: list[TextChunk] = []
    index = 0
    filename = str(document.metadata.get("filename") or document.source)
    for section in document.sections:
        for piece in _window(section.text, cfg.max_chars, cfg.overlap_chars):
            chunks.append(
                TextChunk(
                    id=f"{document.document_id}:{index}",
                    document_id=document.document_id,
                    source=document.source,
                    document=filename,
                    section=section.title,
                    index=index,
                    text=piece,
                    metadata={"media_type": document.media_type},
                )
            )
            index += 1
    return chunks


def _window(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            cut = _prefer_break(text[start:end])
            if cut is not None:
                end = start + cut
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= length:
            break
        nxt = end - overlap
        start = end if nxt <= start else nxt
    return parts


def _prefer_break(window: str) -> int | None:
    min_pos = max(len(window) // 3, 1)
    para = window.rfind("\n\n")
    if para >= min_pos:
        return para + 2
    sent = window.rfind(". ")
    if sent >= min_pos:
        return sent + 2
    return None
