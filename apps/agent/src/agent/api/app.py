from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from agent.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    query_response_from_state,
)
from agent.config.logging import configure_logging
from agent.config.settings import get_settings
from agent.graph.workflow import QueryWorkflow
from agent.ingestion.service import IngestionService


def create_app(
    *,
    compiled_query: Any | None = None,
    ingestion: IngestionService | None = None,
) -> FastAPI:
    application = FastAPI(
        title="GraphRAG Agent",
        description="Evidence-grounded business-strategy GraphRAG. Not a generic chatbot.",
        version="0.1.0",
    )
    application.state.compiled_query = compiled_query
    application.state.ingestion = ingestion

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest) -> QueryResponse:
        compiled = application.state.compiled_query
        if compiled is None:
            raise HTTPException(status_code=503, detail="Query workflow is not configured")
        state = compiled.invoke({"query": body.question.strip()})
        return query_response_from_state(state)

    @application.post("/ingest", response_model=IngestResponse)
    def ingest(body: IngestRequest) -> IngestResponse:
        service = application.state.ingestion
        if service is None:
            raise HTTPException(status_code=503, detail="Ingestion is not configured")
        path = Path(body.path).expanduser()
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Document not found: {path}")
        result = service.ingest_path(path, document_id=body.document_id)
        return IngestResponse(
            document_id=result.document_id,
            chunk_count=result.chunk_count,
            entity_count=result.entity_count,
            relationship_count=result.relationship_count,
            skipped=result.skipped,
        )

    return application


def create_app_from_settings() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    workflow = QueryWorkflow.from_settings(settings)
    ingestion = IngestionService.from_settings(settings)
    return create_app(compiled_query=workflow.compile(), ingestion=ingestion)


app = create_app()
