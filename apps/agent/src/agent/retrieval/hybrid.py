from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent.repositories.vector import VectorFilter, VectorHit
from agent.retrieval.graph import GraphRetrieval, GraphRetriever
from agent.retrieval.vector import VectorRetriever


class HybridResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vector_hits: list[VectorHit] = Field(default_factory=list)
    graph: GraphRetrieval = Field(default_factory=GraphRetrieval)


class HybridRetriever:
    """Dense hits plus a graph expansion seeded by the query and chunk entity ids."""

    def __init__(self, vector: VectorRetriever, graph: GraphRetriever) -> None:
        self._vector = vector
        self._graph = graph

    def retrieve(
        self,
        query: str,
        *,
        vector_limit: int = 8,
        hops: int = 2,
        filters: VectorFilter | None = None,
    ) -> HybridResult:
        hits = self._vector.retrieve(query, limit=vector_limit, filters=filters)
        seeds: list[str] = []
        for hit in hits:
            for entity_id in hit.chunk.entity_ids:
                if entity_id not in seeds:
                    seeds.append(entity_id)
        graph = self._graph.retrieve(query, seed_ids=seeds, hops=hops)
        return HybridResult(vector_hits=hits, graph=graph)
