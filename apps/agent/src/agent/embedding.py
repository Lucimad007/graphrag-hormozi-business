from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx

from agent.config.settings import Settings
from agent.extraction.client import LlmError


class Embeddings(Protocol):
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbeddings:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int = 1536,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self.dimensions = dimensions
        self._timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleEmbeddings:
        if settings.llm_api_key is None:
            raise LlmError("LLM_API_KEY is required for embeddings")
        return cls(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.embedding_model,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self._base_url}/embeddings"
        payload = {"model": self._model, "input": list(texts)}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
            items = sorted(data["data"], key=lambda row: int(row["index"]))
            vectors = [list(map(float, row["embedding"])) for row in items]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise LlmError(f"Embedding request failed: {exc}") from exc
        if len(vectors) != len(texts):
            raise LlmError("Embedding response size does not match inputs")
        return vectors
