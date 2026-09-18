from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest
from pydantic import SecretStr

from agent.config.settings import Settings
from agent.repositories.neo4j_repository import (
    Neo4jGraphRepository,
    entity_from_node,
    relationship_from_rel,
)
from ontology import (
    Entity,
    EntityType,
    OntologyError,
    Relationship,
    RelationType,
    entity_id,
    relationship_id,
)


class FakeResult:
    def __init__(self, records: Sequence[Mapping[str, Any]] | None = None) -> None:
        self._records = list(records or [])

    def single(self) -> Mapping[str, Any] | None:
        return self._records[0] if self._records else None

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return iter(self._records)


class FakeSession:
    def __init__(self, results: Sequence[FakeResult] | None = None) -> None:
        self.calls: list[tuple[str, Mapping[str, Any] | None]] = []
        self._results = list(results or [])

    def run(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> FakeResult:
        params = parameters if parameters is not None else kwargs
        self.calls.append((query, params))
        if self._results:
            return self._results.pop(0)
        return FakeResult()

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeDriver:
    def __init__(self, session: FakeSession) -> None:
        self._session = session
        self.closed = False

    def session(self) -> FakeSession:
        return self._session

    def close(self) -> None:
        self.closed = True


class FakePath:
    def __init__(self, nodes: Sequence[Mapping[str, Any]], relationships: Sequence[Any]) -> None:
        self.nodes = nodes
        self.relationships = relationships


class FakeRel(dict[str, Any]):
    def __init__(
        self,
        props: Mapping[str, Any],
        *,
        type: str,
        start_node: Mapping[str, Any],
        end_node: Mapping[str, Any],
    ) -> None:
        super().__init__(props)
        self.type = type
        self.start_node = start_node
        self.end_node = end_node


def _problem() -> Entity:
    return Entity(
        id=entity_id("Problem", "Low close rate"),
        type=EntityType.PROBLEM,
        name="Low close rate",
        description="Leads exist but conversion is weak",
        source_ids=["doc:sample"],
        properties={"severity": "high"},
    )


def _metric() -> Entity:
    return Entity(
        id=entity_id("Metric", "Close rate"),
        type=EntityType.METRIC,
        name="Close rate",
    )


def test_upsert_entity_uses_parameters_not_inline_values() -> None:
    session = FakeSession()
    repo = Neo4jGraphRepository(FakeDriver(session))
    entity = _problem()
    repo.upsert_entity(entity)
    query, params = session.calls[0]
    assert "MERGE (e:Entity {id: $id})" in query
    assert entity.name not in query
    assert params is not None
    assert params["id"] == entity.id
    assert params["name"] == entity.name
    assert '"severity": "high"' in params["properties_json"]


def test_upsert_relationship_whitelists_rel_type() -> None:
    session = FakeSession()
    repo = Neo4jGraphRepository(FakeDriver(session))
    problem = _problem()
    metric = _metric()
    rel = Relationship(
        id=relationship_id("MEASURED_BY", problem.id, metric.id),
        type=RelationType.MEASURED_BY,
        source_id=problem.id,
        target_id=metric.id,
        source_type=problem.type,
        target_type=metric.type,
    )
    repo.upsert_relationship(rel)
    query, params = session.calls[0]
    assert "MERGE (s)-[r:MEASURED_BY {id: $id}]->(t)" in query
    assert params is not None
    assert params["source_id"] == problem.id
    assert problem.id not in query


def test_upsert_relationship_rejects_injection_payload() -> None:
    session = FakeSession()
    repo = Neo4jGraphRepository(FakeDriver(session))
    rel = Relationship(
        id="bad",
        type="SOLVES",
        source_id="a",
        target_id="b",
        source_type=EntityType.STRATEGY,
        target_type=EntityType.PROBLEM,
    )
    rel.type = "SOLVES]-(n) DELETE n //"
    with pytest.raises(OntologyError):
        repo.upsert_relationship(rel)
    assert session.calls == []


def test_related_subgraph_parameterizes_ids_and_bounds_hops() -> None:
    seed = {
        "id": "problem:low-close-rate",
        "type": "Problem",
        "name": "Low close rate",
        "description": None,
        "aliases": [],
        "source_ids": [],
        "properties_json": "{}",
    }
    other = {
        "id": "metric:close-rate",
        "type": "Metric",
        "name": "Close rate",
        "description": None,
        "aliases": [],
        "source_ids": [],
        "properties_json": "{}",
    }
    rel = FakeRel(
        {
            "id": "measured_by|problem:low-close-rate|metric:close-rate",
            "type": "MEASURED_BY",
            "source_id": seed["id"],
            "target_id": other["id"],
            "source_type": "Problem",
            "target_type": "Metric",
            "source_ids": [],
            "properties_json": "{}",
        },
        type="MEASURED_BY",
        start_node=seed,
        end_node=other,
    )
    path = FakePath([seed, other], [rel])
    session = FakeSession([FakeResult([{"seed": seed, "path": path}])])
    repo = Neo4jGraphRepository(FakeDriver(session))
    subgraph = repo.related_subgraph([seed["id"]], hops=2)
    query, params = session.calls[0]
    assert "WHERE seed.id IN $ids" in query
    assert "[*1..2]" in query
    assert seed["id"] not in query
    assert params == {"ids": [seed["id"]]}
    assert {e.id for e in subgraph.entities} == {seed["id"], other["id"]}
    assert subgraph.relationships[0].type == "MEASURED_BY"


def test_invalid_hops_rejected() -> None:
    repo = Neo4jGraphRepository(FakeDriver(FakeSession()))
    with pytest.raises(ValueError, match="hops"):
        repo.related_subgraph(["x"], hops=99)


def test_get_entity_maps_node() -> None:
    node = {
        "id": "problem:low-close-rate",
        "type": "Problem",
        "name": "Low close rate",
        "description": "x",
        "aliases": ["conversion"],
        "source_ids": ["doc:1"],
        "properties_json": '{"k": 1}',
    }
    session = FakeSession([FakeResult([{"e": node}])])
    repo = Neo4jGraphRepository(FakeDriver(session))
    entity = repo.get_entity("problem:low-close-rate")
    assert entity is not None
    assert entity.name == "Low close rate"
    assert entity.properties == {"k": 1}


def test_from_settings_requires_password() -> None:
    settings = Settings(_env_file=None, neo4j_password=None)
    with pytest.raises(RuntimeError, match="NEO4J_PASSWORD"):
        Neo4jGraphRepository.from_settings(settings)


def test_from_settings_builds_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_driver(uri: str, auth: tuple[str, str]) -> FakeDriver:
        captured["uri"] = uri
        captured["auth"] = auth
        return FakeDriver(FakeSession())

    monkeypatch.setattr("agent.repositories.neo4j_repository.GraphDatabase.driver", fake_driver)
    settings = Settings(
        _env_file=None,
        neo4j_uri="bolt://example:7687",
        neo4j_user="neo",
        neo4j_password=SecretStr("secret"),
    )
    repo = Neo4jGraphRepository.from_settings(settings)
    assert captured["uri"] == "bolt://example:7687"
    assert captured["auth"] == ("neo", "secret")
    repo.close()


def test_entity_and_relationship_mappers() -> None:
    node = {
        "id": "a",
        "type": "Strategy",
        "name": "Nurture",
        "properties_json": "{}",
    }
    entity = entity_from_node(node)
    assert entity.type == "Strategy"
    other = {"id": "b", "type": "Problem", "name": "No show"}
    rel = FakeRel(
        {
            "id": "r1",
            "type": "SOLVES",
            "source_id": "a",
            "target_id": "b",
            "source_type": "Strategy",
            "target_type": "Problem",
            "properties_json": "{}",
        },
        type="SOLVES",
        start_node=node,
        end_node=other,
    )
    mapped = relationship_from_rel(rel)
    assert mapped.source_id == "a"
    assert mapped.type == "SOLVES"
