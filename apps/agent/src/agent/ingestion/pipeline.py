from pathlib import Path

from agent.ingestion.chunk import chunk_document
from agent.ingestion.models import ChunkingConfig, TextChunk
from agent.ingestion.parse import parse_path


def parse_and_chunk(
    path: Path,
    *,
    document_id: str | None = None,
    config: ChunkingConfig | None = None,
) -> list[TextChunk]:
    parsed = parse_path(path, document_id=document_id)
    return chunk_document(parsed, config)
