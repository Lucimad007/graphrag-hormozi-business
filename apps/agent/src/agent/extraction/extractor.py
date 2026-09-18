from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.domain.models import DomainEntity, DomainRelationship
from agent.extraction.client import parse_json_object
from agent.extraction.prompt import extraction_system_prompt, extraction_user_prompt
from agent.extraction.schemas import ExtractionDraft, LlmCompletion
from agent.ingestion.models import TextChunk
from ontology import (
    CORE_ONTOLOGY,
    Entity,
    OntologyError,
    OntologyRegistry,
    Relationship,
    entity_id,
    relationship_id,
)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[DomainEntity] = Field(default_factory=list)
    relationships: list[DomainRelationship] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)


class OntologyExtractor:
    def __init__(
        self,
        llm: LlmCompletion,
        registry: OntologyRegistry | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry or CORE_ONTOLOGY

    def extract_chunk(self, chunk: TextChunk) -> ExtractionResult:
        raw = self._llm.complete(
            system=extraction_system_prompt(self._registry),
            user=extraction_user_prompt(chunk.text, source=chunk.source),
        )
        payload = parse_json_object(raw)
        try:
            draft = ExtractionDraft.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Extraction payload does not match schema: {exc}") from exc
        return self._materialize(draft, chunk_id=chunk.id, source_id=chunk.document_id)

    def _materialize(
        self,
        draft: ExtractionDraft,
        *,
        chunk_id: str,
        source_id: str,
    ) -> ExtractionResult:
        entities: dict[str, DomainEntity] = {}
        skipped: list[str] = []
        for item in draft.entities:
            eid = entity_id(item.type, item.name)
            entity = Entity(
                id=eid,
                type=item.type,
                name=item.name.strip(),
                description=item.description,
                aliases=item.aliases,
                source_ids=[source_id],
            )
            try:
                self._registry.validate_entity(entity)
            except OntologyError as exc:
                skipped.append(str(exc))
                continue
            existing = entities.get(eid)
            if existing is None:
                entities[eid] = DomainEntity(
                    entity=entity,
                    chunk_ids=[chunk_id],
                    confidence=1.0,
                )
            elif chunk_id not in existing.chunk_ids:
                existing.chunk_ids.append(chunk_id)

        relationships: dict[str, DomainRelationship] = {}
        for item in draft.relationships:
            src_id = entity_id(item.source_type, item.source_name)
            dst_id = entity_id(item.target_type, item.target_name)
            if src_id not in entities or dst_id not in entities:
                skipped.append(
                    f"Relationship {item.type} skipped; missing endpoint "
                    f"{item.source_name} or {item.target_name}"
                )
                continue
            rel = Relationship(
                id=relationship_id(item.type, src_id, dst_id),
                type=item.type,
                source_id=src_id,
                target_id=dst_id,
                source_type=item.source_type,
                target_type=item.target_type,
                source_ids=[source_id],
            )
            try:
                self._registry.validate_relationship(rel)
            except OntologyError as exc:
                skipped.append(str(exc))
                continue
            relationships[rel.id] = DomainRelationship(
                relationship=rel,
                chunk_ids=[chunk_id],
                confidence=1.0,
            )
        return ExtractionResult(
            entities=list(entities.values()),
            relationships=list(relationships.values()),
            skipped=skipped,
        )
