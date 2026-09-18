from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from ontology import Entity, Relationship


class Subgraph(BaseModel):
    """Entities and relationships retrieved from the knowledge graph."""

    model_config = ConfigDict(extra="forbid")

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


class GraphRepository(Protocol):
    def ensure_schema(self) -> None: ...

    def upsert_entity(self, entity: Entity) -> Entity: ...

    def upsert_relationship(self, relationship: Relationship) -> Relationship: ...

    def get_entity(self, entity_id: str) -> Entity | None: ...

    def related_subgraph(
        self,
        seed_ids: Sequence[str],
        *,
        hops: int = 2,
    ) -> Subgraph: ...

    def close(self) -> None: ...
