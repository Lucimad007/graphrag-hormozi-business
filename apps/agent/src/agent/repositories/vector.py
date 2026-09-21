from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ChunkRecord(BaseModel):
    """A text chunk with embedding and provenance metadata."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    vector: list[float]
    source: str | None = None
    document: str | None = None
    section: str | None = None
    entity_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    document: str | None = None
    section: str | None = None
    entity_ids: list[str] = Field(default_factory=list)


class VectorHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: ChunkRecord
    score: float


class VectorStore(Protocol):
    def ensure_collection(self, vector_size: int) -> None: ...

    def upsert_chunks(self, chunks: Sequence[ChunkRecord]) -> None: ...

    def list_chunks(self) -> list[ChunkRecord]: ...

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 8,
        filters: VectorFilter | None = None,
    ) -> list[VectorHit]: ...
