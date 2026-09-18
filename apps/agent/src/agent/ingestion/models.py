from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_chars: int = 1200
    overlap_chars: int = 200


class ParsedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    text: str


class ParsedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    source: str
    media_type: str
    sections: list[ParsedSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextChunk(BaseModel):
    """Chunk prior to embedding. Vector payload is added during later ingest."""

    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str
    source: str
    document: str
    section: str | None = None
    index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
