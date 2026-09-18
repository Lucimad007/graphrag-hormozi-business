from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from agent.repositories.graph import GraphRepository, Subgraph


class GraphRetrieval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_ids: list[str] = Field(default_factory=list)
    subgraph: Subgraph = Field(default_factory=Subgraph)


class GraphRetriever:
    """Seed the knowledge graph from query text and/or ids, then expand a subgraph."""

    def __init__(self, graph: GraphRepository) -> None:
        self._graph = graph

    def retrieve(
        self,
        query: str = "",
        *,
        seed_ids: Sequence[str] | None = None,
        hops: int = 2,
        limit: int = 16,
    ) -> GraphRetrieval:
        seeds: list[str] = []
        for entity_id in seed_ids or []:
            if entity_id not in seeds:
                seeds.append(entity_id)
        text = query.strip()
        if text:
            for entity in self._graph.find_entities(text, limit=limit):
                if entity.id not in seeds:
                    seeds.append(entity.id)
        if not seeds:
            return GraphRetrieval()
        subgraph = self._graph.related_subgraph(seeds, hops=hops)
        return GraphRetrieval(seed_ids=seeds, subgraph=subgraph)
