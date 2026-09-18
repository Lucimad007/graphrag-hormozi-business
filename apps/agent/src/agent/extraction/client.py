from __future__ import annotations

import json

import httpx

from agent.config.settings import Settings


class LlmError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Minimal chat-completions client. Tests should inject a fake instead."""

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleClient:
        if settings.llm_api_key is None:
            raise LlmError("LLM_API_KEY is required for extraction")
        return cls(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
        )

    def complete(self, *, system: str, user: str) -> str:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise LlmError(f"LLM request failed: {exc}") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError("LLM response missing message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise LlmError("LLM returned empty content")
        return content


def parse_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LlmError("LLM did not return valid JSON") from exc
    if not isinstance(parsed, dict):
        raise LlmError("LLM JSON must be an object")
    return parsed
