from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from neo4j import Driver, GraphDatabase

from agent.config.settings import Settings
from agent.repositories.graph import Subgraph
from ontology import CORE_ONTOLOGY, Entity, OntologyRegistry, Relationship

_IDENT = re.compile(r"^[A-Z][A-Z0-9_]*$")
_MAX_HOPS = 5

_UPSERT_ENTITY = """
MERGE (e:Entity {id: $id})
SET e.type = $type,
    e.name = $name,
    e.description = $description,
    e.aliases = $aliases,
    e.source_ids = $source_ids,
    e.properties_json = $properties_json
RETURN e
"""

_FIND_ENTITIES = """
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($needle)
   OR any(alias IN coalesce(e.aliases, []) WHERE toLower(alias) CONTAINS toLower($needle))
   OR any(term IN $terms WHERE toLower(e.name) CONTAINS term)
RETURN e
LIMIT $limit
"""

_GET_ENTITY = """
MATCH (e:Entity {id: $id})
RETURN e
"""

_ENSURE_ID_CONSTRAINT = """
CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE
"""


class Neo4jGraphRepository:
    """Neo4j-backed graph store. Labels are fixed; relationship types are registry-whitelisted."""

    def __init__(
        self,
        driver: Driver,
        registry: OntologyRegistry | None = None,
    ) -> None:
        self._driver = driver
        self._registry = registry or CORE_ONTOLOGY

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        registry: OntologyRegistry | None = None,
    ) -> Neo4jGraphRepository:
        if settings.neo4j_password is None:
            raise RuntimeError("NEO4J_PASSWORD is required to connect to Neo4j")
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
        )
        return cls(driver, registry=registry)

    def ensure_schema(self) -> None:
        with self._driver.session() as session:
            session.run(_ENSURE_ID_CONSTRAINT)

    def upsert_entity(self, entity: Entity) -> Entity:
        self._registry.validate_entity(entity)
        params = _entity_params(entity)
        with self._driver.session() as session:
            session.run(_UPSERT_ENTITY, params)
        return entity

    def upsert_relationship(self, relationship: Relationship) -> Relationship:
        self._registry.validate_relationship(relationship)
        rel_type = _cypher_rel_type(relationship.type, self._registry)
        query = (
            "MATCH (s:Entity {id: $source_id}) "
            "MATCH (t:Entity {id: $target_id}) "
            f"MERGE (s)-[r:{rel_type} {{id: $id}}]->(t) "
            "SET r.type = $type, "
            "    r.source_id = $source_id, "
            "    r.target_id = $target_id, "
            "    r.source_type = $source_type, "
            "    r.target_type = $target_type, "
            "    r.source_ids = $source_ids, "
            "    r.properties_json = $properties_json "
            "RETURN r"
        )
        params = {
            "id": relationship.id,
            "source_id": relationship.source_id,
            "target_id": relationship.target_id,
            "type": relationship.type,
            "source_type": relationship.source_type,
            "target_type": relationship.target_type,
            "source_ids": relationship.source_ids,
            "properties_json": json.dumps(relationship.properties),
        }
        with self._driver.session() as session:
            session.run(query, params)
        return relationship

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._driver.session() as session:
            result = session.run(_GET_ENTITY, {"id": entity_id})
            record = result.single()
        if record is None:
            return None
        return entity_from_node(record["e"])

    def find_entities(self, query: str, *, limit: int = 16) -> list[Entity]:
        needle = query.strip()
        if not needle:
            return []
        if limit < 1:
            raise ValueError("limit must be a positive int")
        terms = _search_terms(needle)
        with self._driver.session() as session:
            result = session.run(
                _FIND_ENTITIES,
                {"needle": needle, "terms": terms, "limit": limit},
            )
            return [entity_from_node(record["e"]) for record in result]

    def related_subgraph(self, seed_ids: Sequence[str], *, hops: int = 2) -> Subgraph:
        depth = _validated_hops(hops)
        query = (
            "MATCH (seed:Entity) "
            "WHERE seed.id IN $ids "
            f"OPTIONAL MATCH path = (seed)-[*1..{depth}]-(other:Entity) "
            "RETURN seed, path"
        )
        entities: dict[str, Entity] = {}
        relationships: dict[str, Relationship] = {}
        with self._driver.session() as session:
            for record in session.run(query, {"ids": list(seed_ids)}):
                seed = record["seed"]
                entities[seed["id"]] = entity_from_node(seed)
                path = record["path"]
                if path is None:
                    continue
                for node in path.nodes:
                    entities[node["id"]] = entity_from_node(node)
                for rel in path.relationships:
                    mapped = relationship_from_rel(rel)
                    relationships[mapped.id] = mapped
        return Subgraph(
            entities=list(entities.values()),
            relationships=list(relationships.values()),
        )

    def close(self) -> None:
        self._driver.close()


def entity_from_node(node: Mapping[str, Any]) -> Entity:
    data = dict(node)
    properties = json.loads(data.get("properties_json") or "{}")
    return Entity(
        id=data["id"],
        type=data["type"],
        name=data["name"],
        description=data.get("description"),
        aliases=list(data.get("aliases") or []),
        properties=properties,
        source_ids=list(data.get("source_ids") or []),
    )


def relationship_from_rel(rel: Any) -> Relationship:
    props = dict(rel)
    source_id = props.get("source_id") or rel.start_node["id"]
    target_id = props.get("target_id") or rel.end_node["id"]
    rel_type = props.get("type") or rel.type
    properties = json.loads(props.get("properties_json") or "{}")
    return Relationship(
        id=props["id"],
        type=rel_type,
        source_id=source_id,
        target_id=target_id,
        source_type=props.get("source_type") or rel.start_node["type"],
        target_type=props.get("target_type") or rel.end_node["type"],
        properties=properties,
        source_ids=list(props.get("source_ids") or []),
    )


def _entity_params(entity: Entity) -> dict[str, Any]:
    return {
        "id": entity.id,
        "type": entity.type,
        "name": entity.name,
        "description": entity.description,
        "aliases": entity.aliases,
        "source_ids": entity.source_ids,
        "properties_json": json.dumps(entity.properties),
    }


def _cypher_rel_type(rel_type: str, registry: OntologyRegistry) -> str:
    registry.require_relation_type(rel_type)
    if not _IDENT.fullmatch(rel_type):
        raise ValueError(f"Unsafe relationship type for Cypher: {rel_type!r}")
    return rel_type


def _search_terms(query: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) >= 4]


def _validated_hops(hops: int) -> int:
    if not isinstance(hops, int) or isinstance(hops, bool) or hops < 1 or hops > _MAX_HOPS:
        raise ValueError(f"hops must be an int between 1 and {_MAX_HOPS}")
    return hops


