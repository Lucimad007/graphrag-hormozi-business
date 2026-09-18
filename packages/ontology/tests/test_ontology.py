import pytest

from ontology import (
    CORE_ONTOLOGY,
    ONTOLOGY_VERSION,
    Entity,
    EntityType,
    OntologyError,
    Relationship,
    RelationType,
    entity_id,
    relationship_id,
)


def test_core_entity_types_cover_required_catalog() -> None:
    required = {
        "Business",
        "Customer",
        "Offer",
        "Problem",
        "Metric",
        "Strategy",
        "Tactic",
        "Concept",
        "Constraint",
        "Prerequisite",
        "Outcome",
        "LeadSource",
        "BusinessStage",
        "Evidence",
        "Source",
    }
    assert required <= CORE_ONTOLOGY.entity_types
    assert CORE_ONTOLOGY.entity_types == {t.value for t in EntityType}


def test_core_relation_types_cover_required_catalog() -> None:
    required = {
        "SOLVES",
        "IMPROVES",
        "REQUIRES",
        "TARGETS",
        "CAUSES",
        "RELATED_TO",
        "CONFLICTS_WITH",
        "DERIVED_FROM",
        "APPLIES_TO",
        "MEASURED_BY",
        "PART_OF",
    }
    assert required <= CORE_ONTOLOGY.relation_types
    assert CORE_ONTOLOGY.relation_types == {t.value for t in RelationType}


def test_version_is_semver() -> None:
    parts = ONTOLOGY_VERSION.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_entity_id_is_stable() -> None:
    assert entity_id("Problem", "Low Close Rate") == entity_id("Problem", "low close rate")
    assert entity_id("Problem", "Low Close Rate").startswith("problem:")


def test_validate_entity_and_core_edge() -> None:
    problem = Entity(
        id=entity_id("Problem", "Low close rate"),
        type=EntityType.PROBLEM,
        name="Low close rate",
    )
    metric = Entity(
        id=entity_id("Metric", "Close rate"),
        type=EntityType.METRIC,
        name="Close rate",
    )
    CORE_ONTOLOGY.validate_entity(problem)
    CORE_ONTOLOGY.validate_entity(metric)
    rel = Relationship(
        id=relationship_id("MEASURED_BY", problem.id, metric.id),
        type=RelationType.MEASURED_BY,
        source_id=problem.id,
        target_id=metric.id,
        source_type=problem.type,
        target_type=metric.type,
    )
    CORE_ONTOLOGY.validate_relationship(rel)


def test_unknown_entity_type_rejected() -> None:
    entity = Entity(id="x", type="Widget", name="x")
    with pytest.raises(OntologyError, match="Unknown entity type"):
        CORE_ONTOLOGY.validate_entity(entity)


def test_disallowed_edge_rejected() -> None:
    rel = Relationship(
        id="bad",
        type=RelationType.SOLVES,
        source_id="a",
        target_id="b",
        source_type=EntityType.METRIC,
        target_type=EntityType.SOURCE,
    )
    with pytest.raises(OntologyError, match="not allowed"):
        CORE_ONTOLOGY.validate_relationship(rel)


def test_registry_is_extensible() -> None:
    extended = (
        CORE_ONTOLOGY.register_entity_type("Channel")
        .register_relation_type("FEEDS")
        .allow_edge("Channel", "FEEDS", "LeadSource")
    )
    assert "Channel" in extended.entity_types
    assert "FEEDS" in extended.relation_types
    entity = Entity(id="c1", type="Channel", name="Outbound")
    extended.validate_entity(entity)
    rel = Relationship(
        id="r1",
        type="FEEDS",
        source_id="c1",
        target_id="l1",
        source_type="Channel",
        target_type="LeadSource",
    )
    extended.validate_relationship(rel)
    assert "Channel" not in CORE_ONTOLOGY.entity_types


def test_invalid_extension_names() -> None:
    with pytest.raises(OntologyError):
        CORE_ONTOLOGY.register_entity_type("notPascal")
    with pytest.raises(OntologyError):
        CORE_ONTOLOGY.register_relation_type("feeds")
