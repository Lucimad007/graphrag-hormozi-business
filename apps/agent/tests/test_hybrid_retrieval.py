from collections.abc import Sequence

from agent.repositories.memory import InMemoryGraphRepository, InMemoryVectorStore
from agent.repositories.vector import ChunkRecord, VectorFilter
from agent.retrieval.graph import GraphRetriever
from agent.retrieval.hybrid import HybridRetriever
from agent.retrieval.vector import VectorRetriever
from ontology import Entity, EntityType, Relationship, RelationType, entity_id, relationship_id


class FixedEmbeddings:
    dimensions = 2

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(self.mapping[text]) for text in texts]


def _vector_store() -> InMemoryVectorStore:
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
            ),
            ChunkRecord(
                id="ads:0",
                text="Buying more ads increases lead volume.",
                vector=[0.0, 1.0],
                source="synthetic",
                document="ads.md",
                section="LeadSource",
                entity_ids=["leadsource:ads"],
            ),
        ]
    )
    return store


def _graph() -> InMemoryGraphRepository:
    graph = InMemoryGraphRepository()
    metric = Entity(
        id=entity_id("Metric", "Close rate"),
        type=EntityType.METRIC,
        name="Close rate",
    )
    strategy = Entity(
        id=entity_id("Strategy", "Clarify the offer"),
        type=EntityType.STRATEGY,
        name="Clarify the offer",
    )
    ads = Entity(
        id=entity_id("LeadSource", "Ads"),
        type=EntityType.LEAD_SOURCE,
        name="Ads",
    )
    graph.upsert_entity(metric)
    graph.upsert_entity(strategy)
    graph.upsert_entity(ads)
    graph.upsert_relationship(
        Relationship(
            id=relationship_id("IMPROVES", strategy.id, metric.id),
            type=RelationType.IMPROVES,
            source_id=strategy.id,
            target_id=metric.id,
            source_type=strategy.type,
            target_type=metric.type,
        )
    )
    return graph


def _retriever() -> HybridRetriever:
    return HybridRetriever(
        VectorRetriever(
            _vector_store(),
            FixedEmbeddings(
                {
                    "why is close rate low": [1.0, 0.0],
                    "volume play": [0.0, 1.0],
                }
            ),
        ),
        GraphRetriever(_graph()),
    )


def test_hybrid_uses_vector_entity_ids_as_graph_seeds() -> None:
    result = _retriever().retrieve("why is close rate low", hops=1)
    assert [hit.chunk.id for hit in result.vector_hits][0] == "close:0"
    assert "metric:close-rate" in result.graph.seed_ids
    ids = {entity.id for entity in result.graph.subgraph.entities}
    assert "metric:close-rate" in ids
    assert "strategy:clarify-the-offer" in ids
    assert result.vector_hits[0].chunk.document == "close.md"


def test_hybrid_expands_neighbors_from_chunk_entities() -> None:
    result = _retriever().retrieve("why is close rate low", hops=1)
    assert any(rel.type == "IMPROVES" for rel in result.graph.subgraph.relationships)


def test_hybrid_respects_vector_filters() -> None:
    result = _retriever().retrieve(
        "volume play",
        filters=VectorFilter(document="ads.md"),
        hops=1,
    )
    assert [hit.chunk.id for hit in result.vector_hits] == ["ads:0"]
    assert result.graph.seed_ids == ["leadsource:ads"]
    assert {entity.id for entity in result.graph.subgraph.entities} == {"leadsource:ads"}


def test_hybrid_empty_query() -> None:
    result = _retriever().retrieve("  ")
    assert result.vector_hits == []
    assert result.graph.seed_ids == []
