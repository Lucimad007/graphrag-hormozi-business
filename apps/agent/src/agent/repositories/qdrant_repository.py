from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from agent.config.settings import Settings
from agent.repositories.vector import ChunkRecord, VectorFilter, VectorHit

_NAMESPACE = uuid.UUID("8b2c0d3e-4f5a-46b7-8c9d-0e1f2a3b4c5d")


class QdrantVectorStore:
    """Qdrant-backed vector store. Callers depend on VectorStore, not this class."""

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection
        self._vector_size: int | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> QdrantVectorStore:
        api_key = (
            settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        )
        client = QdrantClient(url=settings.qdrant_url, api_key=api_key)
        return cls(client, settings.qdrant_collection)

    def ensure_collection(self, vector_size: int) -> None:
        if vector_size < 1:
            raise ValueError("vector_size must be a positive int")
        self._vector_size = vector_size
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qdrant_models.VectorParams(
                size=vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    def upsert_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        if not chunks:
            return
        points = [_to_point(chunk) for chunk in chunks]
        self._client.upsert(collection_name=self._collection, points=points)

    def list_chunks(self) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                payload = point.payload or {}
                vector = _payload_vector(point, payload)
                records.append(chunk_from_payload(payload, vector=vector))
            if offset is None:
                break
        return records

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 8,
        filters: VectorFilter | None = None,
    ) -> list[VectorHit]:
        if limit < 1:
            raise ValueError("limit must be a positive int")
        query_filter = _to_qdrant_filter(filters)
        results = self._client.query_points(
            collection_name=self._collection,
            query=list(query_vector),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        points = getattr(results, "points", results)
        hits: list[VectorHit] = []
        for point in points:
            payload = point.payload or {}
            vector = _payload_vector(point, payload)
            chunk = chunk_from_payload(payload, vector=vector)
            hits.append(VectorHit(chunk=chunk, score=float(point.score)))
        return hits

    def close(self) -> None:
        closer = getattr(self._client, "close", None)
        if closer is not None:
            closer()


def point_id_for(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def chunk_from_payload(payload: dict[str, Any], *, vector: list[float]) -> ChunkRecord:
    metadata = dict(payload.get("metadata") or {})
    return ChunkRecord(
        id=payload["chunk_id"],
        text=payload.get("text") or "",
        vector=vector,
        source=payload.get("source"),
        document=payload.get("document"),
        section=payload.get("section"),
        entity_ids=list(payload.get("entity_ids") or []),
        metadata=metadata,
    )


def _to_point(chunk: ChunkRecord) -> qdrant_models.PointStruct:
    return qdrant_models.PointStruct(
        id=point_id_for(chunk.id),
        vector=chunk.vector,
        payload={
            "chunk_id": chunk.id,
            "text": chunk.text,
            "source": chunk.source,
            "document": chunk.document,
            "section": chunk.section,
            "entity_ids": chunk.entity_ids,
            "metadata": chunk.metadata,
        },
    )


def _to_qdrant_filter(filters: VectorFilter | None) -> qdrant_models.Filter | None:
    if filters is None:
        return None
    must: list[qdrant_models.FieldCondition] = []
    if filters.source is not None:
        must.append(
            qdrant_models.FieldCondition(
                key="source",
                match=qdrant_models.MatchValue(value=filters.source),
            )
        )
    if filters.document is not None:
        must.append(
            qdrant_models.FieldCondition(
                key="document",
                match=qdrant_models.MatchValue(value=filters.document),
            )
        )
    if filters.section is not None:
        must.append(
            qdrant_models.FieldCondition(
                key="section",
                match=qdrant_models.MatchValue(value=filters.section),
            )
        )
    if filters.entity_ids:
        must.append(
            qdrant_models.FieldCondition(
                key="entity_ids",
                match=qdrant_models.MatchAny(any=list(filters.entity_ids)),
            )
        )
    if not must:
        return None
    return qdrant_models.Filter(must=must)


def _payload_vector(point: Any, payload: dict[str, Any]) -> list[float]:
    vector = getattr(point, "vector", None)
    if isinstance(vector, list):
        return [float(v) for v in vector]
    stored = payload.get("vector")
    if isinstance(stored, list):
        return [float(v) for v in stored]
    return []
