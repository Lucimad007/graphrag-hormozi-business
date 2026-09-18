from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ontology.types import EntityType, RelationType


class Entity(BaseModel):
    """A typed node in the business knowledge graph."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: str
    name: str
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)

    def typed(self) -> EntityType | None:
        try:
            return EntityType(self.type)
        except ValueError:
            return None


class Relationship(BaseModel):
    """A typed directed edge between two entities."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: str
    source_id: str
    target_id: str
    source_type: str
    target_type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)

    def typed(self) -> RelationType | None:
        try:
            return RelationType(self.type)
        except ValueError:
            return None
