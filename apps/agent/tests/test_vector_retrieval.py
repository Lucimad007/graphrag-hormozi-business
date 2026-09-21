from collections.abc import Sequence

from agent.repositories.memory import InMemoryVectorStore
from agent.repositories.vector import ChunkRecord, VectorFilter
from agent.retrieval.vector import VectorRetriever


class FixedEmbeddings:
    dimensions = 2

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed(self, texts: Sequence[str], *, input_type: str | None = None) -> list[list[float]]:
        return [list(self.mapping[text]) for text in texts]


def _store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.ensure_collection(2)
    store.upsert_chunks(
        [
            ChunkRecord(
                id="a:0",
                text="Close rate measures conversion from lead to customer.",
                vector=[1.0, 0.0],
                source="synthetic",
                document="close.md",
                section="Metric",
                entity_ids=["metric:close-rate"],
            ),
            ChunkRecord(
                id="b:0",
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


def test_vector_retriever_ranks_similar_chunk_first() -> None:
    retriever = VectorRetriever(
        _store(),
        FixedEmbeddings({"low close rate": [0.95, 0.05]}),
    )
    hits = retriever.retrieve("low close rate", limit=2)
    assert [hit.chunk.id for hit in hits] == ["a:0", "b:0"]
    assert hits[0].score > hits[1].score
    assert hits[0].chunk.document == "close.md"
    assert hits[0].chunk.section == "Metric"
    assert "metric:close-rate" in hits[0].chunk.entity_ids


def test_vector_retriever_applies_metadata_filter() -> None:
    retriever = VectorRetriever(
        _store(),
        FixedEmbeddings({"query": [1.0, 0.0]}),
    )
    hits = retriever.retrieve("query", filters=VectorFilter(document="ads.md"))
    assert [hit.chunk.id for hit in hits] == ["b:0"]


def test_vector_retriever_filters_entity_ids() -> None:
    retriever = VectorRetriever(
        _store(),
        FixedEmbeddings({"query": [1.0, 0.0]}),
    )
    hits = retriever.retrieve(
        "query",
        filters=VectorFilter(entity_ids=["leadsource:ads"]),
    )
    assert hits[0].chunk.id == "b:0"


def test_empty_query_returns_no_hits() -> None:
    retriever = VectorRetriever(_store(), FixedEmbeddings({}))
    assert retriever.retrieve("   ") == []


def test_retrieve_union_merges_queries_and_keeps_best_score() -> None:
    retriever = VectorRetriever(
        _store(),
        FixedEmbeddings(
            {
                "close rate declining": [1.0, 0.0],
                "what is close rate": [0.2, 0.8],
            }
        ),
    )
    hits = retriever.retrieve_union(
        ["close rate declining", "what is close rate", "close rate declining"],
        limit=2,
    )
    assert [hit.chunk.id for hit in hits] == ["a:0", "b:0"]
    assert hits[0].score > hits[1].score
