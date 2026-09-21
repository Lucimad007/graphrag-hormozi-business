from __future__ import annotations

import json
from uuid import uuid4

import httpx

from agent.config.settings import Settings


class LlmError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Minimal chat-completions client. Tests should inject a fake instead."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 180.0,
        max_tokens: int = 8192,
        json_mode: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._json_mode = json_mode
        self._session_id = str(uuid4())
        self._user_agent = "graphrag-agent/0.1"

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleClient:
        if settings.llm_api_key is None or not settings.llm_api_key.get_secret_value().strip():
            raise LlmError("LLM_API_KEY is required for extraction")
        return cls(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            json_mode=settings.llm_json_mode,
        )

    def complete(self, *, system: str, user: str) -> str:
        url = f"{self._base_url}/chat/completions"
        payload: dict[str, object] = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": self._user_agent,
            "x-opencode-session": self._session_id,
        }
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise LlmError(f"LLM request failed: {exc} {detail}") from exc
        except httpx.HTTPError as exc:
            raise LlmError(f"LLM request failed: {exc}") from exc
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError("LLM response missing message content") from exc
        content = _message_text(message)
        if not content.strip():
            raise LlmError("LLM returned empty content")
        return content


def _message_text(message: object) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
        ]
        joined = "".join(parts).strip()
        if joined:
            return joined
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str):
        return reasoning
    return ""


def parse_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")].strip()
    decoder = json.JSONDecoder()
    found: dict[str, object] | None = None
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(stripped, index)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        found = parsed
        if any(
            key in parsed
            for key in (
                "entities",
                "relationships",
                "intent",
                "answer",
                "restated_query",
                "step_back_query",
                "situation",
            )
        ):
            return parsed
    if found is not None:
        return found
    raise LlmError("LLM did not return valid JSON")
