from __future__ import annotations

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
        vector = self._embeddings.embed([text])[0]
        return self._store.search(vector, limit=limit, filters=filters)
