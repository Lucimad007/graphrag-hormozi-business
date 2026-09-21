import re
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

    def find_entities(self, query: str, *, limit: int = 16) -> list[Entity]:
        needle = query.strip().lower()
        if not needle:
            return []
        if limit < 1:
            raise ValueError("limit must be a positive int")
        terms = [token for token in re.findall(r"[a-z0-9]+", needle) if len(token) >= 4]
        hits: list[Entity] = []
        for entity in self.entities.values():
            blob = " ".join([entity.name, *entity.aliases, entity.id]).lower()
            if needle in blob or any(term in blob for term in terms):
                hits.append(entity)
            if len(hits) >= limit:
                break
        return hits

    def related_subgraph(self, seed_ids: Sequence[str], *, hops: int = 2) -> Subgraph:
        if not isinstance(hops, int) or isinstance(hops, bool) or hops < 1 or hops > 5:
            raise ValueError("hops must be an int between 1 and 5")
        seen: set[str] = {entity_id for entity_id in seed_ids if entity_id in self.entities}
        frontier = set(seen)
        for _ in range(hops):
            nxt: set[str] = set()
            for rel in self.relationships.values():
                if rel.source_id in frontier and rel.target_id in self.entities:
                    nxt.add(rel.target_id)
                if rel.target_id in frontier and rel.source_id in self.entities:
                    nxt.add(rel.source_id)
            nxt -= seen
            if not nxt:
                break
            seen |= nxt
            frontier = nxt
        rels = [
            rel
            for rel in self.relationships.values()
            if rel.source_id in seen and rel.target_id in seen
        ]
        return Subgraph(
            entities=[self.entities[entity_id] for entity_id in seen],
            relationships=rels,
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

    def list_chunks(self) -> list[ChunkRecord]:
        return list(self.chunks.values())

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
