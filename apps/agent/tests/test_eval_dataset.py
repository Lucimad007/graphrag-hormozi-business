import json
from pathlib import Path

from agent.eval.dataset import load_eval_dataset
from agent.eval.lexical import LexicalEmbeddings
from agent.extraction.extractor import OntologyExtractor
from agent.ingestion.models import ChunkingConfig
from agent.ingestion.service import IngestionService
from agent.repositories.memory import InMemoryGraphRepository, InMemoryVectorStore
from agent.retrieval.graph import GraphRetriever
from agent.retrieval.hybrid import HybridRetriever
from agent.retrieval.vector import VectorRetriever

_REPO = Path(__file__).resolve().parents[3]
_DATASET = _REPO / "knowledge" / "eval" / "dataset.json"
_SAMPLES = _REPO / "knowledge" / "samples"

_CLOSE_RATE = {
    "entities": [
        {
            "type": "Problem",
            "name": "Low close rate",
            "description": "Leads exist but conversion is weak",
            "aliases": [],
        },
        {"type": "Metric", "name": "Close rate", "aliases": []},
        {"type": "Strategy", "name": "Clarify the offer", "aliases": []},
    ],
    "relationships": [
        {
            "type": "MEASURED_BY",
            "source_type": "Problem",
            "source_name": "Low close rate",
            "target_type": "Metric",
            "target_name": "Close rate",
        },
        {
            "type": "IMPROVES",
            "source_type": "Strategy",
            "source_name": "Clarify the offer",
            "target_type": "Metric",
            "target_name": "Close rate",
        },
        {
            "type": "SOLVES",
            "source_type": "Strategy",
            "source_name": "Clarify the offer",
            "target_type": "Problem",
            "target_name": "Low close rate",
        },
    ],
}

_OFFER = {
    "entities": [
        {"type": "Offer", "name": "Unclear offer", "aliases": []},
        {
            "type": "Problem",
            "name": "Confusion about what is being bought",
            "aliases": [],
        },
        {
            "type": "Strategy",
            "name": "Rewrite the offer around a specific outcome",
            "aliases": [],
        },
    ],
    "relationships": [
        {
            "type": "SOLVES",
            "source_type": "Offer",
            "source_name": "Unclear offer",
            "target_type": "Problem",
            "target_name": "Confusion about what is being bought",
        }
    ],
}


class CorpusExtractLlm:
    def complete(self, *, system: str, user: str) -> str:
        text = user.lower()
        if "unclear offer" in text or "confusion about what is being bought" in text:
            return json.dumps(_OFFER)
        if "close rate" in text or "inbound leads" in text:
            return json.dumps(_CLOSE_RATE)
        return json.dumps({"entities": [], "relationships": []})


def _ingest_corpus() -> HybridRetriever:
    graph = InMemoryGraphRepository()
    vectors = InMemoryVectorStore()
    service = IngestionService(
        extractor=OntologyExtractor(CorpusExtractLlm()),
        embeddings=LexicalEmbeddings(),
        graph=graph,
        vectors=vectors,
        chunking=ChunkingConfig(max_chars=2000, overlap_chars=100),
    )
    for path in sorted(_SAMPLES.glob("*.md")):
        service.ingest_path(path, document_id=path.stem)
    return HybridRetriever(VectorRetriever(vectors, LexicalEmbeddings()), GraphRetriever(graph))


def test_eval_dataset_schema() -> None:
    dataset = load_eval_dataset(_DATASET)
    assert dataset.version
    assert len(dataset.cases) >= 3
    ids = [case.id for case in dataset.cases]
    assert len(ids) == len(set(ids))
    for case in dataset.cases:
        assert case.query.strip()
        assert case.expect_documents or case.expect_entity_ids


def test_eval_dataset_hybrid_retrieval() -> None:
    dataset = load_eval_dataset(_DATASET)
    retriever = _ingest_corpus()
    for case in dataset.cases:
        result = retriever.retrieve(case.query, vector_limit=6, hops=2)
        documents = {hit.chunk.document for hit in result.vector_hits}
        entity_ids = {entity.id for entity in result.graph.subgraph.entities}
        for hit in result.vector_hits:
            entity_ids.update(hit.chunk.entity_ids)
        rel_types = {rel.type for rel in result.graph.subgraph.relationships}
        missing_docs = set(case.expect_documents) - documents
        missing_entities = set(case.expect_entity_ids) - entity_ids
        missing_rels = set(case.expect_relationship_types) - rel_types
        assert not missing_docs, (
            f"{case.id} missing documents {missing_docs} in {documents}"
        )
        assert not missing_entities, (
            f"{case.id} missing entities {missing_entities} in {entity_ids}"
        )
        assert not missing_rels, f"{case.id} missing rels {missing_rels} in {rel_types}"
