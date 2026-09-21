from __future__ import annotations

from collections.abc import Sequence

from agent.embedding import Embeddings
from agent.repositories.vector import VectorFilter, VectorHit, VectorStore


class VectorRetriever:
    """Dense retrieval: embed the query, then similarity-search the vector store."""

    def __init__(self, store: VectorStore, embeddings: Embeddings) -> None:
        self._store = store
        self._embeddings = embeddings

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 8,
        filters: VectorFilter | None = None,
    ) -> list[VectorHit]:
        text = query.strip()
        if not text:
            return []
        if limit < 1:
            raise ValueError("limit must be a positive int")
        vector = self._embeddings.embed([text], input_type="query")[0]
        return self._store.search(vector, limit=limit, filters=filters)

    def retrieve_union(
        self,
        queries: Sequence[str],
        *,
        limit: int = 8,
        filters: VectorFilter | None = None,
    ) -> list[VectorHit]:
        if limit < 1:
            raise ValueError("limit must be a positive int")
        deduped: list[str] = []
        seen: set[str] = set()
        for item in queries:
            text = item.strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(text)
        if not deduped:
            return []
        best: dict[str, VectorHit] = {}
        per_query = max(limit, 8)
        for query in deduped:
            for hit in self.retrieve(query, limit=per_query, filters=filters):
                current = best.get(hit.chunk.id)
                if current is None or hit.score > current.score:
                    best[hit.chunk.id] = hit
        ranked = sorted(best.values(), key=lambda hit: hit.score, reverse=True)
        return ranked[:limit]
