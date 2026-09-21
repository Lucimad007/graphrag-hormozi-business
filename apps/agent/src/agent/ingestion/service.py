from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agent.config.settings import Settings
from agent.domain.models import DomainEntity, DomainRelationship
from agent.embedding import Embeddings
from agent.extraction.client import LlmError, OpenAICompatibleClient
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
    documents: list[str] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)


class IngestionService:
    def __init__(
        self,
        *,
        extractor: OntologyExtractor,
        embeddings: Embeddings,
        graph: GraphRepository,
        vectors: VectorStore,
        chunking: ChunkingConfig | None = None,
        extract_concurrency: int = 1,
    ) -> None:
        self._extractor = extractor
        self._embeddings = embeddings
        self._graph = graph
        self._vectors = vectors
        self._chunking = chunking
        self._extract_concurrency = max(1, extract_concurrency)

    @classmethod
    def from_settings(cls, settings: Settings) -> IngestionService:
        from agent.embedding import embeddings_from_settings
        from agent.repositories.neo4j_repository import Neo4jGraphRepository
        from agent.repositories.qdrant_repository import QdrantVectorStore

        extractor = OntologyExtractor(OpenAICompatibleClient.from_settings(settings))
        embeddings = embeddings_from_settings(settings)
        return cls(
            extractor=extractor,
            embeddings=embeddings,
            graph=Neo4jGraphRepository.from_settings(settings),
            vectors=QdrantVectorStore.from_settings(settings),
            chunking=ChunkingConfig(
                max_chars=settings.ingest_max_chars,
                overlap_chars=settings.ingest_overlap_chars,
            ),
            extract_concurrency=settings.ingest_extract_concurrency,
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
        extraction = _merge_extractions(self._extract_chunks(chunks))
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

        vectors = self._embeddings.embed(
            [chunk.text for chunk in chunks],
            input_type="document",
        )
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
            documents=[path.name],
        )

    def _extract_chunks(self, chunks: Sequence[TextChunk]) -> list[ExtractionResult]:
        if len(chunks) == 1 or self._extract_concurrency == 1:
            return [self._extract_one(chunk) for chunk in chunks]
        workers = min(self._extract_concurrency, len(chunks))
        results: list[ExtractionResult | None] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._extract_one, chunk): index
                for index, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [item for item in results if item is not None]

    def _extract_one(self, chunk: TextChunk) -> ExtractionResult:
        try:
            return self._extractor.extract_chunk(chunk)
        except (LlmError, ValueError) as exc:
            return ExtractionResult(skipped=[f"{chunk.id}: {exc}"])

    def ingest_tree(self, root: Path, *, suffixes: set[str] | None = None) -> IngestResult:
        from agent.ingestion.parse import TEXT_SUFFIXES, iter_ingestible_files
        from ontology import slug

        allowed = {item.lower() for item in (suffixes or TEXT_SUFFIXES)}
        files = iter_ingestible_files(root, suffixes=allowed)
        skipped_files = sorted(
            str(path.relative_to(root)).replace("\\", "/")
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() not in allowed
        )
        totals = IngestResult(
            document_id=slug(root.name),
            chunk_count=0,
            entity_count=0,
            relationship_count=0,
            skipped_files=skipped_files,
        )
        for path in files:
            relative = path.relative_to(root).with_suffix("").as_posix()
            doc_id = slug(relative.replace("/", "--"))
            try:
                result = self.ingest_path(path, document_id=doc_id)
            except Exception as exc:
                totals.skipped.append(f"{path.name}: {exc}")
                continue
            totals.chunk_count += result.chunk_count
            totals.entity_count += result.entity_count
            totals.relationship_count += result.relationship_count
            totals.skipped.extend(result.skipped)
            totals.documents.extend(result.documents)
        return totals

    def retry_missing_extractions(
        self, *, chunk_ids: set[str] | None = None
    ) -> IngestResult:
        pending = [
            record
            for record in self._vectors.list_chunks()
            if not record.entity_ids and (chunk_ids is None or record.id in chunk_ids)
        ]
        totals = IngestResult(
            document_id="retry-missing-extractions",
            chunk_count=len(pending),
            entity_count=0,
            relationship_count=0,
        )
        if not pending:
            return totals
        text_chunks = [_text_chunk_from_record(record) for record in pending]
        extraction = _merge_extractions(self._extract_chunks(text_chunks))
        self._graph.ensure_schema()
        for item in extraction.entities:
            self._graph.upsert_entity(item.entity)
        for item in extraction.relationships:
            self._graph.upsert_relationship(item.relationship)
        by_id = {record.id: record for record in pending}
        updated: list[ChunkRecord] = []
        for chunk in text_chunks:
            record = by_id[chunk.id]
            entity_ids = [item.entity.id for item in extraction.entities if chunk.id in item.chunk_ids]
            updated.append(record.model_copy(update={"entity_ids": entity_ids}))
        self._vectors.upsert_chunks(updated)
        totals.entity_count = len(extraction.entities)
        totals.relationship_count = len(extraction.relationships)
        totals.skipped = extraction.skipped
        totals.documents = sorted({record.document or "" for record in pending if record.document})
        return totals


def _text_chunk_from_record(record: ChunkRecord) -> TextChunk:
    document_id = str(record.metadata.get("document_id") or record.id.rsplit(":", 1)[0])
    suffix = record.id.rsplit(":", 1)[-1]
    index = int(suffix) if suffix.isdigit() else 0
    return TextChunk(
        id=record.id,
        document_id=document_id,
        source=record.source or record.document or document_id,
        document=record.document or document_id,
        section=record.section,
        index=index,
        text=record.text,
        metadata=dict(record.metadata),
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
