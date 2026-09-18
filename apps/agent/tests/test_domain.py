from agent.domain import DomainEntity, DomainRelationship
from ontology import Entity, EntityType, Relationship, RelationType, entity_id, relationship_id


def test_domain_entity_wraps_ontology_entity() -> None:
    entity = Entity(
        id=entity_id("Strategy", "Nurture sequence"),
        type=EntityType.STRATEGY,
        name="Nurture sequence",
        source_ids=["doc:sample"],
    )
    wrapped = DomainEntity(entity=entity, chunk_ids=["chunk-1"], confidence=0.9)
    assert wrapped.entity.type == EntityType.STRATEGY
    assert wrapped.chunk_ids == ["chunk-1"]


def test_domain_relationship_wraps_ontology_relationship() -> None:
    rel = Relationship(
        id=relationship_id("SOLVES", "s1", "p1"),
        type=RelationType.SOLVES,
        source_id="s1",
        target_id="p1",
        source_type=EntityType.STRATEGY,
        target_type=EntityType.PROBLEM,
    )
    wrapped = DomainRelationship(relationship=rel, chunk_ids=["chunk-2"])
    assert wrapped.relationship.type == RelationType.SOLVES
