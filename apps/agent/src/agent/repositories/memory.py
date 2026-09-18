from collections.abc import Sequence

from agent.repositories.graph import Subgraph
from agent.repositories.vector import ChunkRecord, VectorFilter, VectorHit
from ontology import Entity, Relationship


class InMemoryGraphRepository:
    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relationships: dict[str, Relationship] = {}
        self.schema_ready = False

    def ensure_schema(self) -> None:
        self.schema_ready = True

    def upsert_entity(self, entity: Entity) -> Entity:
        self.entities[entity.id] = entity
        return entity

    def upsert_relationship(self, relationship: Relationship) -> Relationship:
        self.relationships[relationship.id] = relationship
        return relationship

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def related_subgraph(self, seed_ids: Sequence[str], *, hops: int = 2) -> Subgraph:
        return Subgraph(
            entities=[self.entities[i] for i in seed_ids if i in self.entities],
            relationships=list(self.relationships.values()),
        )

    def close(self) -> None:
        return None


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.chunks: dict[str, ChunkRecord] = {}
        self.vector_size: int | None = None

    def ensure_collection(self, vector_size: int) -> None:
        self.vector_size = vector_size

    def upsert_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 8,
        filters: VectorFilter | None = None,
    ) -> list[VectorHit]:
        return []
