from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

import httpx

from agent.config.settings import Settings
from agent.extraction.client import LlmError


class Embeddings(Protocol):
    dimensions: int

    def embed(
        self, texts: Sequence[str], *, input_type: str | None = None
    ) -> list[list[float]]: ...


class OpenAICompatibleEmbeddings:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int = 1536,
        timeout: float = 60.0,
        extra: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self.dimensions = dimensions
        self._timeout = timeout
        self._extra = extra or {}

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleEmbeddings:
        api_key = settings.embedding_api_key or settings.llm_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise LlmError("EMBEDDING_API_KEY is required for embeddings")
        base_url = settings.embedding_base_url or settings.llm_base_url
        return cls(
            base_url=base_url,
            api_key=api_key.get_secret_value(),
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    def embed(self, texts: Sequence[str], *, input_type: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self._base_url}/embeddings"
        payload: dict[str, object] = {"model": self._model, "input": list(texts), **self._extra}
        if input_type:
            payload["input_type"] = input_type
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
            rows = list(data["data"])
            if rows and "index" in rows[0]:
                rows = sorted(rows, key=lambda row: int(row["index"]))
            vectors = [list(map(float, row["embedding"])) for row in rows]
        except httpx.HTTPStatusError as exc:
            raise LlmError(f"Embedding request failed: {exc} {exc.response.text[:400]}") from exc
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise LlmError(f"Embedding request failed: {exc}") from exc
        if len(vectors) != len(texts):
            raise LlmError("Embedding response size does not match inputs")
        if vectors and len(vectors[0]) != self.dimensions:
            self.dimensions = len(vectors[0])
        return vectors


class LocalHashEmbeddings:
    """Deterministic local vectors so ingest works without a cloud embedding API."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str], *, input_type: str | None = None) -> list[list[float]]:
        del input_type
        return [_hash_vector(text, self.dimensions) for text in texts]


class BgeM3Embeddings:
    """Local BAAI/bge-m3 dense vectors via FlagEmbedding."""

    def __init__(self, model: str = "BAAI/bge-m3", dimensions: int = 1024) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self._backend = BGEM3FlagModel(model, use_fp16=False)
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str], *, input_type: str | None = None) -> list[list[float]]:
        del input_type
        if not texts:
            return []
        encoded = self._backend.encode(
            list(texts),
            batch_size=8,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = encoded["dense_vecs"]
        vectors = [list(map(float, vector)) for vector in dense]
        if vectors and len(vectors[0]) != self.dimensions:
            self.dimensions = len(vectors[0])
        return vectors


def embeddings_from_settings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "local":
        return LocalHashEmbeddings()
    if settings.embedding_provider == "bge_m3":
        return BgeM3Embeddings(
            model=settings.embedding_model or "BAAI/bge-m3",
            dimensions=settings.embedding_dimensions or 1024,
        )
    if settings.embedding_provider == "voyage":
        api_key = settings.embedding_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise LlmError("EMBEDDING_API_KEY or VOYAGE_API_KEY is required for Voyage")
        return OpenAICompatibleEmbeddings(
            base_url=(settings.embedding_base_url or "https://api.voyageai.com/v1"),
            api_key=api_key.get_secret_value(),
            model=settings.embedding_model or "voyage-4",
            dimensions=settings.embedding_dimensions or 1024,
        )
    return OpenAICompatibleEmbeddings.from_settings(settings)


def _hash_vector(text: str, dimensions: int) -> list[float]:
    values = [0.0] * dimensions
    tokens = text.lower().split() or [text]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]
