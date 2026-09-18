from agent.repositories.memory import InMemoryGraphRepository
from agent.retrieval.graph import GraphRetriever
from ontology import Entity, EntityType, Relationship, RelationType, entity_id, relationship_id


def _entity(type_: EntityType, name: str, *, aliases: list[str] | None = None) -> Entity:
    return Entity(
        id=entity_id(type_, name),
        type=type_,
        name=name,
        aliases=aliases or [],
    )


def _graph() -> InMemoryGraphRepository:
    graph = InMemoryGraphRepository()
    problem = _entity(EntityType.PROBLEM, "Low close rate")
    metric = _entity(EntityType.METRIC, "Close rate")
    strategy = _entity(EntityType.STRATEGY, "Clarify the offer")
    constraint = _entity(EntityType.CONSTRAINT, "Unrelated budget cap")
    graph.upsert_entity(problem)
    graph.upsert_entity(metric)
    graph.upsert_entity(strategy)
    graph.upsert_entity(constraint)
    graph.upsert_relationship(
        Relationship(
            id=relationship_id("MEASURED_BY", problem.id, metric.id),
            type=RelationType.MEASURED_BY,
            source_id=problem.id,
            target_id=metric.id,
            source_type=problem.type,
            target_type=metric.type,
        )
    )
    graph.upsert_relationship(
        Relationship(
            id=relationship_id("IMPROVES", strategy.id, metric.id),
            type=RelationType.IMPROVES,
            source_id=strategy.id,
            target_id=metric.id,
            source_type=strategy.type,
            target_type=metric.type,
        )
    )
    return graph


def test_graph_retriever_seeds_from_query_text() -> None:
    result = GraphRetriever(_graph()).retrieve(
        "lots of leads but a low close rate",
        hops=1,
    )
    assert "problem:low-close-rate" in result.seed_ids
    ids = {entity.id for entity in result.subgraph.entities}
    assert "problem:low-close-rate" in ids
    assert "metric:close-rate" in ids
    assert "constraint:unrelated-budget-cap" not in ids
    assert any(rel.type == "MEASURED_BY" for rel in result.subgraph.relationships)


def test_graph_retriever_respects_hops() -> None:
    retriever = GraphRetriever(_graph())
    one = retriever.retrieve(seed_ids=["problem:low-close-rate"], hops=1)
    two = retriever.retrieve(seed_ids=["problem:low-close-rate"], hops=2)
    one_ids = {entity.id for entity in one.subgraph.entities}
    two_ids = {entity.id for entity in two.subgraph.entities}
    assert "strategy:clarify-the-offer" not in one_ids
    assert "strategy:clarify-the-offer" in two_ids


def test_explicit_seeds_do_not_require_query() -> None:
    result = GraphRetriever(_graph()).retrieve(seed_ids=["metric:close-rate"], hops=1)
    assert result.seed_ids == ["metric:close-rate"]
    names = {entity.name for entity in result.subgraph.entities}
    assert "Close rate" in names


def test_empty_query_without_seeds() -> None:
    result = GraphRetriever(_graph()).retrieve("   ")
    assert result.seed_ids == []
    assert result.subgraph.entities == []
