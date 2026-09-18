from __future__ import annotations

from dataclasses import dataclass, field

from ontology.edges import CORE_ALLOWED_EDGES
from ontology.models import Entity, Relationship
from ontology.types import ONTOLOGY_VERSION, EntityType, RelationType


class OntologyError(ValueError):
    pass


@dataclass(frozen=True)
class OntologyRegistry:
    """Versioned catalog of entity types, relation types, and allowed edges."""

    version: str
    entity_types: frozenset[str]
    relation_types: frozenset[str]
    allowed_edges: frozenset[tuple[str, str, str]] = field(default_factory=frozenset)
    strict_edges: bool = True

    @classmethod
    def core(cls) -> OntologyRegistry:
        return cls(
            version=ONTOLOGY_VERSION,
            entity_types=frozenset(t.value for t in EntityType),
            relation_types=frozenset(t.value for t in RelationType),
            allowed_edges=CORE_ALLOWED_EDGES,
            strict_edges=True,
        )

    def register_entity_type(self, name: str) -> OntologyRegistry:
        _require_token(name, "entity type")
        if name in self.entity_types:
            return self
        return OntologyRegistry(
            version=self.version,
            entity_types=self.entity_types | {name},
            relation_types=self.relation_types,
            allowed_edges=self.allowed_edges,
            strict_edges=self.strict_edges,
        )

    def register_relation_type(self, name: str) -> OntologyRegistry:
        _require_token(name, "relation type")
        if name in self.relation_types:
            return self
        return OntologyRegistry(
            version=self.version,
            entity_types=self.entity_types,
            relation_types=self.relation_types | {name},
            allowed_edges=self.allowed_edges,
            strict_edges=self.strict_edges,
        )

    def allow_edge(self, source_type: str, relation: str, target_type: str) -> OntologyRegistry:
        self.require_entity_type(source_type)
        self.require_entity_type(target_type)
        self.require_relation_type(relation)
        edge = (source_type, relation, target_type)
        if edge in self.allowed_edges:
            return self
        return OntologyRegistry(
            version=self.version,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
            allowed_edges=self.allowed_edges | {edge},
            strict_edges=self.strict_edges,
        )

    def require_entity_type(self, name: str) -> None:
        if name not in self.entity_types:
            raise OntologyError(f"Unknown entity type: {name}")

    def require_relation_type(self, name: str) -> None:
        if name not in self.relation_types:
            raise OntologyError(f"Unknown relation type: {name}")

    def validate_entity(self, entity: Entity) -> Entity:
        self.require_entity_type(entity.type)
        return entity

    def validate_relationship(self, relationship: Relationship) -> Relationship:
        self.require_entity_type(relationship.source_type)
        self.require_entity_type(relationship.target_type)
        self.require_relation_type(relationship.type)
        edge = (relationship.source_type, relationship.type, relationship.target_type)
        if self.strict_edges and edge not in self.allowed_edges:
            raise OntologyError(
                "Relationship not allowed by ontology: "
                f"{relationship.source_type} -{relationship.type}-> {relationship.target_type}"
            )
        return relationship


def _require_token(name: str, kind: str) -> None:
    if not name or name != name.strip():
        raise OntologyError(f"Invalid {kind}: {name!r}")
    if kind == "relation type" and (not name.isupper() or not name.replace("_", "").isalnum()):
        raise OntologyError(f"Relation types must be UPPER_SNAKE: {name!r}")
    if kind == "entity type" and (not name[0].isupper() or " " in name):
        raise OntologyError(f"Entity types must be PascalCase without spaces: {name!r}")


CORE_ONTOLOGY = OntologyRegistry.core()
