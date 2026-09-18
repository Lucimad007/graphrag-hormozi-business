from collections.abc import Sequence
from typing import Any

from agent.graph.workflow import QueryWorkflow
from agent.repositories.memory import InMemoryGraphRepository, InMemoryVectorStore
from agent.repositories.vector import ChunkRecord
from agent.retrieval.graph import GraphRetriever
from agent.retrieval.vector import VectorRetriever
from ontology import Entity, EntityType, Relationship, RelationType, entity_id, relationship_id


class ScriptedLlm:
    def __init__(self, intent: str = "diagnosis") -> None:
        self.intent = intent
        self.calls: list[str] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append(system)
        if "Restate" in system:
            return '{"restated_query": "What causes a low close rate despite many leads?"}'
        if "Classify" in system:
            return f'{{"intent": "{self.intent}"}}'
        if "evidence paths" in system:
            return (
                '{"reasoning_summary": '
                '"Problem low close rate MEASURED_BY Metric close rate (close.md)."}'
            )
        if "Answer using only" in system:
            return (
                '{"answer": "Low close rate is a conversion metric issue, '
                'not lead volume (close.md)."}'
            )
        return "{}"


class RecordingVectorRetriever(VectorRetriever):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]:
        self.calls += 1
        return super().retrieve(query, **kwargs)


class FixedEmbeddings:
    dimensions = 2

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _vector_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.ensure_collection(2)
    store.upsert_chunks(
        [
            ChunkRecord(
                id="close:0",
                text="Close rate measures conversion; lead volume is the wrong diagnosis.",
                vector=[1.0, 0.0],
                source="synthetic",
                document="close.md",
                section="Metric",
                entity_ids=["metric:close-rate"],
            )
        ]
    )
    return store


def _graph() -> InMemoryGraphRepository:
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
    return graph


def test_workflow_retrieves_then_answers_with_sources() -> None:
    llm = ScriptedLlm()
    vector = RecordingVectorRetriever(_vector_store(), FixedEmbeddings())
    app = QueryWorkflow(llm=llm, vector=vector, graph=GraphRetriever(_graph())).compile()
    result = app.invoke({"query": "lots of leads but a low close rate"})
    assert result["intent"] == "diagnosis"
    assert result["skip_retrieval"] is False
    assert vector.calls == 1
    assert result["source_documents"] == ["close.md"]
    assert "close.md" in result["answer"]
    assert "MEASURED_BY" in result["reasoning_summary"]
    assert result["graph"]["seed_ids"]
    assert any(hit["chunk"]["id"] == "close:0" for hit in result["vector_hits"])


def test_out_of_scope_skips_retrieval() -> None:
    llm = ScriptedLlm(intent="out_of_scope")
    vector = RecordingVectorRetriever(_vector_store(), FixedEmbeddings())
    app = QueryWorkflow(llm=llm, vector=vector, graph=GraphRetriever(_graph())).compile()
    result = app.invoke({"query": "write a poem about cats"})
    assert result["intent"] == "out_of_scope"
    assert vector.calls == 0
    assert result["source_documents"] == []
    assert "outside" in result["answer"].lower()


def test_nodes_are_wired_in_order() -> None:
    app = QueryWorkflow(
        llm=ScriptedLlm(),
        vector=RecordingVectorRetriever(_vector_store(), FixedEmbeddings()),
        graph=GraphRetriever(_graph()),
    ).compile()
    names = set(app.get_graph().nodes)
    for node in (
        "understand_query",
        "classify_intent",
        "retrieve_graph_context",
        "retrieve_vector_context",
        "evaluate_evidence",
        "reason",
        "generate_answer",
    ):
        assert node in names
