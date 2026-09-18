import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from agent.embedding import OpenAICompatibleEmbeddings
from agent.extraction.client import LlmError
from agent.extraction.extractor import OntologyExtractor
from agent.ingestion.models import ChunkingConfig
from agent.ingestion.service import IngestionService
from agent.repositories.memory import InMemoryGraphRepository, InMemoryVectorStore


class FakeLlm:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = json.dumps(payload)
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        return self.payload


class HashEmbeddings:
    dimensions = 4

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            seed = sum(ord(ch) for ch in text) or 1
            vectors.append([((seed * (i + 1)) % 100) / 100.0 for i in range(self.dimensions)])
        return vectors


def _extractor() -> OntologyExtractor:
    return OntologyExtractor(
        FakeLlm(
            {
                "entities": [
                    {
                        "type": "Problem",
                        "name": "Low close rate",
                        "description": "Leads do not convert",
                        "aliases": [],
                    },
                    {"type": "Metric", "name": "Close rate", "aliases": []},
                ],
                "relationships": [
                    {
                        "type": "MEASURED_BY",
                        "source_type": "Problem",
                        "source_name": "Low close rate",
                        "target_type": "Metric",
                        "target_name": "Close rate",
                    }
                ],
            }
        )
    )


def _service(
    llm: OntologyExtractor | None = None,
) -> tuple[IngestionService, InMemoryGraphRepository, InMemoryVectorStore]:
    graph = InMemoryGraphRepository()
    vectors = InMemoryVectorStore()
    service = IngestionService(
        extractor=llm or _extractor(),
        embeddings=HashEmbeddings(),
        graph=graph,
        vectors=vectors,
        chunking=ChunkingConfig(max_chars=400, overlap_chars=40),
    )
    return service, graph, vectors


def test_ingest_path_writes_graph_and_vectors(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text(
        "# Problem\n\nLots of leads and a low close rate.\n\n# Metric\n\nTrack close rate.\n",
        encoding="utf-8",
    )
    service, graph, vectors = _service()
    result = service.ingest_path(path, document_id="notes")
    assert result.chunk_count >= 1
    assert result.entity_count == 2
    assert result.relationship_count == 1
    assert graph.schema_ready
    assert "problem:low-close-rate" in graph.entities
    assert any(rel.type == "MEASURED_BY" for rel in graph.relationships.values())
    assert any(entity.type == "Source" for entity in graph.entities.values())
    chunk = next(iter(vectors.chunks.values()))
    assert len(chunk.vector) == 4
    assert "problem:low-close-rate" in chunk.entity_ids
    assert chunk.document == "notes.md"


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("Low close rate is measured by close rate.\n", encoding="utf-8")
    service, graph, vectors = _service()
    first = service.ingest_path(path, document_id="notes")
    entity_ids = set(graph.entities)
    chunk_ids = set(vectors.chunks)
    second = service.ingest_path(path, document_id="notes")
    assert first.document_id == second.document_id
    assert set(graph.entities) == entity_ids
    assert set(vectors.chunks) == chunk_ids
    assert len(graph.relationships) == 1


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")
    service, graph, vectors = _service()
    result = service.ingest_path(path, document_id="empty")
    assert result.chunk_count == 0
    assert graph.entities == {}
    assert vectors.chunks == {}


def test_embeddings_client(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "data": [
                    {"index": 1, "embedding": [0.2, 0.1]},
                    {"index": 0, "embedding": [0.0, 1.0]},
                ]
            }

    def fake_post(
        url: str,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("agent.embedding.httpx.post", fake_post)
    client = OpenAICompatibleEmbeddings(
        base_url="https://example.test/v1",
        api_key="sk",
        model="text-embedding-test",
        dimensions=2,
    )
    vectors = client.embed(["a", "b"])
    assert captured["url"].endswith("/embeddings")
    assert captured["json"]["input"] == ["a", "b"]
    assert vectors == [[0.0, 1.0], [0.2, 0.1]]


def test_embeddings_require_api_key() -> None:
    from agent.config.settings import Settings

    with pytest.raises(LlmError):
        OpenAICompatibleEmbeddings.from_settings(Settings(_env_file=None, llm_api_key=None))
