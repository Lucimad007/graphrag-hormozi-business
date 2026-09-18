# Agentic GraphRAG (business strategy)

Open-source GraphRAG for business strategy. The first intended knowledge domain is Hormozi-style frameworks; **copyrighted books are not in this repository**. Users ingest material they are legally entitled to use.

See [docs/architecture.md](docs/architecture.md) for the system design.

## Local Python setup (M1)

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
uv run pytest
uv run ruff check .
```

Infrastructure (Qdrant, Neo4j) and the query API arrive in later milestones.
