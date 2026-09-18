# Agentic GraphRAG (business strategy)

Open-source GraphRAG for business strategy. The first intended knowledge domain is Hormozi-style frameworks; **copyrighted books are not in this repository**. Users ingest material they are legally entitled to use.

See [docs/architecture.md](docs/architecture.md) for the system design and [docs/local-setup.md](docs/local-setup.md) for Docker.

## Local setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and Docker Compose.

```bash
cp .env.example .env
uv sync
uv run pytest
uv run ruff check .
docker compose up --build
```

`GET http://localhost:8000/health` should return `{"status":"ok"}` once the agent container is up.
