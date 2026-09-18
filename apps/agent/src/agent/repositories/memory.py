from collections.abc import Sequence
from math import sqrt

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
        if limit < 1:
            raise ValueError("limit must be a positive int")
        hits: list[VectorHit] = []
        for chunk in self.chunks.values():
            if not _matches_filter(chunk, filters):
                continue
            hits.append(VectorHit(chunk=chunk, score=_cosine(query_vector, chunk.vector)))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]


def _matches_filter(chunk: ChunkRecord, filters: VectorFilter | None) -> bool:
    if filters is None:
        return True
    if filters.source is not None and chunk.source != filters.source:
        return False
    if filters.document is not None and chunk.document != filters.document:
        return False
    if filters.section is not None and chunk.section != filters.section:
        return False
    if filters.entity_ids and not set(filters.entity_ids).intersection(chunk.entity_ids):
        return False
    return True


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_l = sqrt(sum(a * a for a in left))
    norm_r = sqrt(sum(b * b for b in right))
    if norm_l == 0.0 or norm_r == 0.0:
        return 0.0
    return dot / (norm_l * norm_r)
