from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_COMPOSE = _REPO / "docker-compose.yml"


def test_compose_defines_qdrant_neo4j_and_agent() -> None:
    text = _COMPOSE.read_text(encoding="utf-8")
    assert "qdrant:" in text
    assert "neo4j:" in text
    assert "agent:" in text
    assert "docker/agent/Dockerfile" in text
    assert "6333:6333" in text
    assert "7687:7687" in text
    assert "8000:8000" in text
