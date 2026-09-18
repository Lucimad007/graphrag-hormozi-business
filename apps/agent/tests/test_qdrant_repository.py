from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from agent.config.settings import Settings
from agent.repositories.qdrant_repository import (
    QdrantVectorStore,
    chunk_from_payload,
    point_id_for,
)
from agent.repositories.vector import ChunkRecord, VectorFilter


def _chunk(*, cid: str = "doc-1:0", document: str = "playbook.md") -> ChunkRecord:
    return ChunkRecord(
        id=cid,
        text="Low close rate often traces to offer or sales process, not lead volume.",
        vector=[0.1, 0.2, 0.3],
        source="synthetic",
        document=document,
        section="diagnosis",
        entity_ids=["problem:low-close-rate"],
        metadata={"chapter": "1"},
    )


def test_ensure_collection_creates_once() -> None:
    client = MagicMock()
    client.collection_exists.return_value = False
    store = QdrantVectorStore(client, "business_chunks")
    store.ensure_collection(3)
    client.create_collection.assert_called_once()
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "business_chunks"
    assert kwargs["vectors_config"].size == 3
    client.collection_exists.return_value = True
    store.ensure_collection(3)
    assert client.create_collection.call_count == 1


def test_ensure_collection_rejects_invalid_size() -> None:
    store = QdrantVectorStore(MagicMock(), "c")
    with pytest.raises(ValueError, match="vector_size"):
        store.ensure_collection(0)


def test_upsert_maps_stable_point_ids_and_payload() -> None:
    client = MagicMock()
    store = QdrantVectorStore(client, "business_chunks")
    chunk = _chunk()
    store.upsert_chunks([chunk])
    points = client.upsert.call_args.kwargs["points"]
    assert len(points) == 1
    point = points[0]
    assert point.id == point_id_for(chunk.id)
    assert point.vector == chunk.vector
    assert point.payload["chunk_id"] == chunk.id
    assert point.payload["document"] == "playbook.md"
    assert point.payload["entity_ids"] == ["problem:low-close-rate"]
    assert chunk.text not in str(client.upsert.call_args.kwargs["collection_name"])


def test_upsert_empty_is_noop() -> None:
    client = MagicMock()
    QdrantVectorStore(client, "c").upsert_chunks([])
    client.upsert.assert_not_called()


def test_search_maps_hits_and_filters() -> None:
    client = MagicMock()
    payload = {
        "chunk_id": "doc-1:0",
        "text": "hello",
        "source": "synthetic",
        "document": "playbook.md",
        "section": "diagnosis",
        "entity_ids": ["problem:low-close-rate"],
        "metadata": {"chapter": "1"},
    }
    point = SimpleNamespace(payload=payload, score=0.91, vector=[0.1, 0.2, 0.3])
    client.query_points.return_value = SimpleNamespace(points=[point])
    store = QdrantVectorStore(client, "business_chunks")
    hits = store.search(
        [0.1, 0.2, 0.3],
        limit=5,
        filters=VectorFilter(document="playbook.md", entity_ids=["problem:low-close-rate"]),
    )
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["limit"] == 5
    assert kwargs["query"] == [0.1, 0.2, 0.3]
    assert kwargs["query_filter"] is not None
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(0.91)
    assert hits[0].chunk.id == "doc-1:0"
    assert hits[0].chunk.metadata["chapter"] == "1"


def test_search_rejects_invalid_limit() -> None:
    store = QdrantVectorStore(MagicMock(), "c")
    with pytest.raises(ValueError, match="limit"):
        store.search([0.1], limit=0)


def test_from_settings_passes_url_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_client(*, url: str, api_key: str | None) -> MagicMock:
        captured["url"] = url
        captured["api_key"] = api_key
        return MagicMock()

    monkeypatch.setattr("agent.repositories.qdrant_repository.QdrantClient", fake_client)
    settings = Settings(
        _env_file=None,
        qdrant_url="http://qdrant:6333",
        qdrant_api_key=SecretStr("qk"),
        qdrant_collection="cols",
    )
    store = QdrantVectorStore.from_settings(settings)
    assert captured == {"url": "http://qdrant:6333", "api_key": "qk"}
    assert store._collection == "cols"


def test_chunk_from_payload_roundtrip() -> None:
    chunk = chunk_from_payload(
        {
            "chunk_id": "x",
            "text": "t",
            "source": None,
            "entity_ids": [],
            "metadata": {},
        },
        vector=[1.0],
    )
    assert chunk.id == "x"
    assert chunk.vector == [1.0]
