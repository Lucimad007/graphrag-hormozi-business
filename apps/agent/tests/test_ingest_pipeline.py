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
from agent.repositories.vector import ChunkRecord


class FakeLlm:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = json.dumps(payload)
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        return self.payload


class HashEmbeddings:
    dimensions = 4

    def embed(self, texts: Sequence[str], *, input_type: str | None = None) -> list[list[float]]:
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


def test_ingest_tree_skips_non_text(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# Metric\n\nClose rate matters.\n", encoding="utf-8")
    (tmp_path / "skip.mp4").write_bytes(b"not-a-video")
    nested = tmp_path / "playbooks"
    nested.mkdir()
    (nested / "offer.md").write_text("# Offer\n\nClarify the offer.\n", encoding="utf-8")
    service, graph, vectors = _service()
    result = service.ingest_tree(tmp_path)
    assert result.chunk_count >= 2
    assert "notes.md" in result.documents
    assert "offer.md" in result.documents
    assert any(name.endswith("skip.mp4") for name in result.skipped_files)
    assert len(vectors.chunks) >= 2


def test_retry_missing_extractions_fills_empty_entity_ids() -> None:
    service, graph, vectors = _service()
    vectors.upsert_chunks(
        [
            ChunkRecord(
                id="notes:0",
                text="Low close rate is measured by close rate.",
                vector=[0.1, 0.2, 0.3, 0.4],
                source="notes.md",
                document="notes.md",
                section=None,
                entity_ids=[],
                metadata={"document_id": "notes"},
            )
        ]
    )
    result = service.retry_missing_extractions()
    assert result.chunk_count == 1
    assert result.entity_count == 2
    assert vectors.chunks["notes:0"].entity_ids
    assert "problem:low-close-rate" in graph.entities


def test_ingest_keeps_file_when_one_chunk_is_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text(
        "# Problem\n\nLots of leads and a low close rate.\n\n# Metric\n\nTrack close rate.\n",
        encoding="utf-8",
    )
    good = FakeLlm(
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

    class MixedLlm:
        def complete(self, *, system: str, user: str) -> str:
            del system
            if "Track close rate" in user:
                return "this is not json"
            return good.payload

    service, _graph, vectors = _service(OntologyExtractor(MixedLlm()))
    result = service.ingest_path(path, document_id="notes")
    assert result.chunk_count >= 2
    assert result.skipped
    assert result.documents == ["notes.md"]
    assert vectors.chunks


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
    vectors = client.embed(["a", "b"], input_type="document")
    assert captured["url"].endswith("/embeddings")
    assert captured["json"]["input"] == ["a", "b"]
    assert captured["json"]["input_type"] == "document"
    assert vectors == [[0.0, 1.0], [0.2, 0.1]]


def test_embeddings_require_api_key() -> None:
    from agent.config.settings import Settings
    from agent.embedding import embeddings_from_settings

    with pytest.raises(LlmError, match="EMBEDDING_API_KEY"):
        OpenAICompatibleEmbeddings.from_settings(Settings(_env_file=None, llm_api_key=None))

    with pytest.raises(LlmError, match="VOYAGE"):
        embeddings_from_settings(
            Settings(_env_file=None, embedding_provider="voyage", embedding_api_key=None)
        )

    local = embeddings_from_settings(
        Settings(_env_file=None, embedding_provider="local", llm_api_key=None)
    )
    vectors = local.embed(["close rate", "close rate"])
    assert len(vectors[0]) == 384
    assert vectors[0] == vectors[1]


def test_bge_m3_flagembedding(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    from agent.config.settings import Settings
    from agent.embedding import embeddings_from_settings

    class FakeM3:
        def __init__(self, model: str, use_fp16: bool = False) -> None:
            self.model = model
            self.use_fp16 = use_fp16

        def encode(self, texts: list[str], **kwargs: object) -> dict[str, list[list[float]]]:
            del kwargs
            return {"dense_vecs": [[0.1, 0.2] for _ in texts]}

    monkeypatch.setitem(sys.modules, "FlagEmbedding", SimpleNamespace(BGEM3FlagModel=FakeM3))
    emb = embeddings_from_settings(
        Settings(_env_file=None, embedding_provider="bge_m3", embedding_model="BAAI/bge-m3")
    )
    vectors = emb.embed(["offer"])
    assert vectors == [[0.1, 0.2]]
    assert emb.dimensions == 2
