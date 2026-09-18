# Local setup

Python 3.12+, [uv](https://docs.astral.sh/uv/), and Docker Compose.

## 1. Configure environment

```bash
cp .env.example .env
```

Set `NEO4J_PASSWORD` (the compose file defaults to `changeme` if unset). Set `LLM_API_KEY` for ingest and query. Leave it empty only if you are running unit tests, which mock the model.

## 2. Start Qdrant, Neo4j, and the API

```bash
docker compose up --build
```

- API: http://localhost:8000/health
- Qdrant: http://localhost:6333
- Neo4j Browser: http://localhost:7474 (`neo4j` / your password)

Infrastructure only (no agent image):

```bash
docker compose up qdrant neo4j
uv sync
uv run uvicorn agent.api.main:app --reload --host 0.0.0.0 --port 8000
```

Health-only app without stores (unit-test style):

```bash
uv run uvicorn agent.api.app:app --port 8000
```

## 3. Ingest a synthetic sample

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d "{\"path\": \"/app/knowledge/samples/low-close-rate.md\", \"document_id\": \"low-close-rate\"}"
```

If the API runs on the host instead of Compose, use the repo path to that file.

## 4. Query

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is probably causing a business with lots of leads but a low close rate?\"}"
```

## Tests

```bash
uv sync
uv run pytest
uv run ruff check .
```

Do not commit copyrighted books. See [knowledge/README.md](../knowledge/README.md).
