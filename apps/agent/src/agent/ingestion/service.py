from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agent.config.settings import Settings
from agent.domain.models import DomainEntity, DomainRelationship
from agent.embedding import Embeddings, OpenAICompatibleEmbeddings
from agent.extraction.client import OpenAICompatibleClient
from agent.extraction.extractor import ExtractionResult, OntologyExtractor
from agent.ingestion.models import ChunkingConfig, TextChunk
from agent.ingestion.pipeline import parse_and_chunk
from agent.repositories.graph import GraphRepository
from agent.repositories.vector import ChunkRecord, VectorStore
from ontology import Entity, entity_id


class IngestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    chunk_count: int
    entity_count: int
    relationship_count: int
    skipped: list[str] = Field(default_factory=list)


class IngestionService:
    def __init__(
        self,
        *,
        extractor: OntologyExtractor,
        embeddings: Embeddings,
        graph: GraphRepository,
        vectors: VectorStore,
        chunking: ChunkingConfig | None = None,
    ) -> None:
        self._extractor = extractor
        self._embeddings = embeddings
        self._graph = graph
        self._vectors = vectors
        self._chunking = chunking

    @classmethod
    def from_settings(cls, settings: Settings) -> IngestionService:
        from agent.repositories.neo4j_repository import Neo4jGraphRepository
        from agent.repositories.qdrant_repository import QdrantVectorStore

        extractor = OntologyExtractor(OpenAICompatibleClient.from_settings(settings))
        embeddings = OpenAICompatibleEmbeddings.from_settings(settings)
        return cls(
            extractor=extractor,
            embeddings=embeddings,
            graph=Neo4jGraphRepository.from_settings(settings),
            vectors=QdrantVectorStore.from_settings(settings),
        )

    def ingest_path(self, path: Path, *, document_id: str | None = None) -> IngestResult:
        chunks = parse_and_chunk(path, document_id=document_id, config=self._chunking)
        if not chunks:
            return IngestResult(
                document_id=document_id or path.stem,
                chunk_count=0,
                entity_count=0,
                relationship_count=0,
            )
        doc_id = chunks[0].document_id
        extraction = _merge_extractions([self._extractor.extract_chunk(chunk) for chunk in chunks])
        source = Entity(
            id=entity_id("Source", doc_id),
            type="Source",
            name=path.name,
            source_ids=[doc_id],
            properties={"path": str(path)},
        )
        self._graph.ensure_schema()
        self._graph.upsert_entity(source)
        for item in extraction.entities:
            entity = item.entity
            if doc_id not in entity.source_ids:
                entity = entity.model_copy(update={"source_ids": [*entity.source_ids, doc_id]})
            self._graph.upsert_entity(entity)
        for item in extraction.relationships:
            self._graph.upsert_relationship(item.relationship)

        vectors = self._embeddings.embed([chunk.text for chunk in chunks])
        self._vectors.ensure_collection(self._embeddings.dimensions)
        records = [
            _chunk_record(chunk, vector, extraction.entities)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._vectors.upsert_chunks(records)
        return IngestResult(
            document_id=doc_id,
            chunk_count=len(chunks),
            entity_count=len(extraction.entities),
            relationship_count=len(extraction.relationships),
            skipped=extraction.skipped,
        )


def _chunk_record(
    chunk: TextChunk,
    vector: Sequence[float],
    entities: Sequence[DomainEntity],
) -> ChunkRecord:
    entity_ids = [item.entity.id for item in entities if chunk.id in item.chunk_ids]
    return ChunkRecord(
        id=chunk.id,
        text=chunk.text,
        vector=list(vector),
        source=chunk.source,
        document=chunk.document,
        section=chunk.section,
        entity_ids=entity_ids,
        metadata={"document_id": chunk.document_id, **chunk.metadata},
    )


def _merge_extractions(results: Sequence[ExtractionResult]) -> ExtractionResult:
    entities: dict[str, DomainEntity] = {}
    relationships: dict[str, DomainRelationship] = {}
    skipped: list[str] = []
    for result in results:
        skipped.extend(result.skipped)
        for item in result.entities:
            current = entities.get(item.entity.id)
            if current is None:
                entities[item.entity.id] = item
                continue
            chunk_ids = list(dict.fromkeys(current.chunk_ids + item.chunk_ids))
            source_ids = list(dict.fromkeys(current.entity.source_ids + item.entity.source_ids))
            aliases = list(dict.fromkeys(current.entity.aliases + item.entity.aliases))
            entities[item.entity.id] = DomainEntity(
                entity=current.entity.model_copy(
                    update={"source_ids": source_ids, "aliases": aliases}
                ),
                chunk_ids=chunk_ids,
                confidence=current.confidence,
            )
        for item in result.relationships:
            current = relationships.get(item.relationship.id)
            if current is None:
                relationships[item.relationship.id] = item
                continue
            chunk_ids = list(dict.fromkeys(current.chunk_ids + item.chunk_ids))
            relationships[item.relationship.id] = DomainRelationship(
                relationship=current.relationship,
                chunk_ids=chunk_ids,
                confidence=current.confidence,
            )
    return ExtractionResult(
        entities=list(entities.values()),
        relationships=list(relationships.values()),
        skipped=skipped,
    )
