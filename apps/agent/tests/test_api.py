import json
from collections.abc import Sequence
from pathlib import Path

from fastapi.testclient import TestClient

from agent.api.app import create_app
from agent.extraction.extractor import OntologyExtractor
from agent.graph.workflow import QueryWorkflow
from agent.ingestion.models import ChunkingConfig
from agent.ingestion.service import IngestionService
from agent.repositories.memory import InMemoryGraphRepository, InMemoryVectorStore
from agent.repositories.vector import ChunkRecord
from agent.retrieval.graph import GraphRetriever
from agent.retrieval.vector import VectorRetriever
from ontology import Entity, EntityType, Relationship, RelationType, entity_id, relationship_id


class ScriptedLlm:
    def complete(self, *, system: str, user: str) -> str:
        if "Restate" in system:
            return '{"restated_query": "What causes a low close rate?"}'
        if "Classify" in system:
            return '{"intent": "diagnosis"}'
        if "evidence paths" in system:
            return '{"reasoning_summary": "Problem MEASURED_BY Metric (close.md)."}'
        if "Answer using only" in system:
            return '{"answer": "Close rate, not lead volume, is the bottleneck (close.md)."}'
        return "{}"


class ExtractLlm:
    def complete(self, *, system: str, user: str) -> str:
        return json.dumps(
            {
                "entities": [
                    {"type": "Problem", "name": "Low close rate", "aliases": []},
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


class QueryEmbeddings:
    dimensions = 2

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class IngestEmbeddings:
    dimensions = 4

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.25, 0.25, 0.25, 0.25] for _ in texts]


def _query_app() -> TestClient:
    store = InMemoryVectorStore()
    store.ensure_collection(2)
    store.upsert_chunks(
        [
            ChunkRecord(
                id="close:0",
                text="Close rate measures conversion from lead to customer.",
                vector=[1.0, 0.0],
                source="synthetic",
                document="close.md",
                section="Metric",
                entity_ids=["metric:close-rate"],
            )
        ]
    )
    graph = InMemoryGraphRepository()
    problem = Entity(
        id=entity_id("Problem", "Low close rate"),
        type=EntityType.PROBLEM,
        name="Low close rate",
    )
    metric = Entity(
        id=entity_id("Metric", "Close rate"),
        type=EntityType.METRIC,
        name="Close rate",
    )
    graph.upsert_entity(problem)
    graph.upsert_entity(metric)
    graph.upsert_relationship(
        Relationship(
            id=relationship_id("MEASURED_BY", problem.id, metric.id),
            type=RelationType.MEASURED_BY,
            source_id=problem.id,
            target_id=metric.id,
            source_type=problem.type,
            target_type=metric.type,
        )
    )
    workflow = QueryWorkflow(
        llm=ScriptedLlm(),
        vector=VectorRetriever(store, QueryEmbeddings()),
        graph=GraphRetriever(graph),
    )
    return TestClient(create_app(compiled_query=workflow.compile()))


def test_health() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_grounded_payload() -> None:
    response = _query_app().post(
        "/query",
        json={"question": "lots of leads but a low close rate"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "close.md" in body["answer"]
    assert body["sources"][0]["document"] == "close.md"
    assert body["graph_evidence"]["relationships"][0]["type"] == "MEASURED_BY"
    assert body["reasoning"]["intent"] == "diagnosis"
    assert body["reasoning"]["summary"]
    assert "chain-of-thought" not in body["reasoning"]["summary"].lower()
    assert body["retrieval"]["vector_hit_count"] >= 1
    assert body["retrieval"]["skipped_retrieval"] is False


def test_query_unconfigured() -> None:
    response = TestClient(create_app()).post("/query", json={"question": "hello"})
    assert response.status_code == 503


def test_ingest_and_missing_file(tmp_path: Path) -> None:
    graph = InMemoryGraphRepository()
    vectors = InMemoryVectorStore()
    service = IngestionService(
        extractor=OntologyExtractor(ExtractLlm()),
        embeddings=IngestEmbeddings(),
        graph=graph,
        vectors=vectors,
        chunking=ChunkingConfig(max_chars=400, overlap_chars=40),
    )
    client = TestClient(create_app(ingestion=service))
    missing = client.post("/ingest", json={"path": str(tmp_path / "nope.md")})
    assert missing.status_code == 404
    path = tmp_path / "notes.md"
    path.write_text("# Problem\n\nLow close rate with many leads.\n", encoding="utf-8")
    ok = client.post("/ingest", json={"path": str(path), "document_id": "notes"})
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["document_id"] == "notes"
    assert payload["chunk_count"] >= 1
    assert payload["entity_count"] == 2
    assert "problem:low-close-rate" in graph.entities
