"""Extensible business ontology for GraphRAG."""

from ontology.ids import entity_id, relationship_id, slug
from ontology.models import Entity, Relationship
from ontology.registry import CORE_ONTOLOGY, OntologyError, OntologyRegistry
from ontology.types import ONTOLOGY_VERSION, EntityType, RelationType

__version__ = "0.1.0"

__all__ = [
    "CORE_ONTOLOGY",
    "ONTOLOGY_VERSION",
    "Entity",
    "EntityType",
    "OntologyError",
    "OntologyRegistry",
    "RelationType",
    "Relationship",
    "__version__",
    "entity_id",
    "relationship_id",
    "slug",
]
