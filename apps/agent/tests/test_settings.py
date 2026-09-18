import pytest

from agent.config.settings import Settings, get_settings


def test_settings_defaults_without_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LLM_API_KEY",
        "QDRANT_API_KEY",
        "NEO4J_PASSWORD",
        "LLM_MODEL",
        "QDRANT_URL",
        "NEO4J_URI",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)
    assert settings.llm_api_key is None
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.llm_model == "gpt-4o-mini"


def test_settings_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_chunks")
    settings = Settings(_env_file=None)
    assert settings.log_level == "DEBUG"
    assert settings.qdrant_collection == "test_chunks"


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()
