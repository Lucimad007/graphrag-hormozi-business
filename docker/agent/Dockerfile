FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir uv==0.8.15

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
COPY apps ./apps
COPY packages ./packages
COPY knowledge ./knowledge

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
