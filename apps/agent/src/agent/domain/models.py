"""Agent-facing domain models wrapping the shared ontology."""

from pydantic import BaseModel, ConfigDict, Field

from ontology import Entity, Relationship


class DomainEntity(BaseModel):
    """Ontology entity plus retrieval/ingest provenance used by the agent."""

    model_config = ConfigDict(extra="forbid")

    entity: Entity
    chunk_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None


class DomainRelationship(BaseModel):
    """Ontology relationship plus provenance used by the agent."""

    model_config = ConfigDict(extra="forbid")

    relationship: Relationship
    chunk_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None
