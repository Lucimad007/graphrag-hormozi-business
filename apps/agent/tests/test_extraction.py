import json
from typing import Any

import pytest
from pydantic import SecretStr

from agent.config.settings import Settings
from agent.extraction.client import LlmError, OpenAICompatibleClient, parse_json_object
from agent.extraction.extractor import OntologyExtractor
from agent.extraction.prompt import extraction_system_prompt
from agent.ingestion.models import TextChunk


class FakeLlm:
    def __init__(self, payload: str | dict[str, Any]) -> None:
        self.payload = payload if isinstance(payload, str) else json.dumps(payload)
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self.payload


def _chunk() -> TextChunk:
    return TextChunk(
        id="notes:0",
        document_id="notes",
        source="notes.md",
        document="notes.md",
        section="Diagnosis",
        index=0,
        text=(
            "Many inbound leads with a low close rate often point to offer "
            "or sales-process issues."
        ),
    )


def test_extracts_valid_ontology_graph() -> None:
    llm = FakeLlm(
        {
            "entities": [
                {
                    "type": "Problem",
                    "name": "Low close rate",
                    "description": "Conversion from lead to customer is weak",
                    "aliases": ["poor conversion"],
                },
                {"type": "Metric", "name": "Close rate", "description": None, "aliases": []},
            ],
            "relationships": [
                {
                    "type": "MEASURED_BY",
                    "source_type": "Problem",
                    "source_name": "Low close rate",
                    "target_type": "Metric",
                    "target_name": "Close rate",
                }
            ],
        }
    )
    result = OntologyExtractor(llm).extract_chunk(_chunk())
    assert {e.entity.type for e in result.entities} == {"Problem", "Metric"}
    assert result.relationships[0].relationship.type == "MEASURED_BY"
    assert result.skipped == []
    assert "notes:0" in result.entities[0].chunk_ids
    assert llm.calls[0]["user"].startswith("Source: notes.md")


def test_skips_unknown_types_and_edges() -> None:
    llm = FakeLlm(
        {
            "entities": [
                {"type": "Widget", "name": "Gadget", "aliases": []},
                {"type": "Problem", "name": "No shows", "aliases": []},
                {"type": "Metric", "name": "Show rate", "aliases": []},
            ],
            "relationships": [
                {
                    "type": "SOLVES",
                    "source_type": "Metric",
                    "source_name": "Show rate",
                    "target_type": "Problem",
                    "target_name": "No shows",
                }
            ],
        }
    )
    result = OntologyExtractor(llm).extract_chunk(_chunk())
    assert [e.entity.name for e in result.entities] == ["No shows", "Show rate"]
    assert result.relationships == []
    assert result.skipped


def test_skips_relationship_without_endpoints() -> None:
    llm = FakeLlm(
        {
            "entities": [{"type": "Strategy", "name": "Nurture", "aliases": []}],
            "relationships": [
                {
                    "type": "SOLVES",
                    "source_type": "Strategy",
                    "source_name": "Nurture",
                    "target_type": "Problem",
                    "target_name": "Missing problem",
                }
            ],
        }
    )
    result = OntologyExtractor(llm).extract_chunk(_chunk())
    assert len(result.entities) == 1
    assert result.relationships == []
    assert any("missing endpoint" in s.lower() for s in result.skipped)


def test_prompt_lists_ontology_not_a_vendor() -> None:
    prompt = extraction_system_prompt()
    assert "Problem" in prompt
    assert "MEASURED_BY" in prompt
    assert "Hormozi" not in prompt


def test_parse_json_fences() -> None:
    parsed = parse_json_object('```json\n{"entities": []}\n```')
    assert parsed == {"entities": []}


def test_parse_json_invalid() -> None:
    with pytest.raises(LlmError, match="valid JSON"):
        parse_json_object("not json")


def test_openai_client_posts_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    def fake_post(
        url: str,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr("agent.extraction.client.httpx.post", fake_post)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1/",
        api_key="sk-test",
        model="gpt-test",
    )
    content = client.complete(system="sys", user="usr")
    assert content == '{"ok": true}'
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["json"]["messages"][0]["content"] == "sys"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_from_settings_requires_key() -> None:
    settings = Settings(_env_file=None, llm_api_key=None)
    with pytest.raises(LlmError, match="LLM_API_KEY"):
        OpenAICompatibleClient.from_settings(settings)


def test_from_settings_builds_client() -> None:
    settings = Settings(_env_file=None, llm_api_key=SecretStr("k"), llm_model="m")
    client = OpenAICompatibleClient.from_settings(settings)
    assert client._model == "m"
