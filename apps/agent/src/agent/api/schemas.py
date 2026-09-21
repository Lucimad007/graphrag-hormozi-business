from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)


class RetrievedSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document: str | None = None
    section: str | None = None
    source: str | None = None
    score: float
    entity_ids: list[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    source_id: str
    target_id: str


class GraphEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    relationships: list[GraphEdge] = Field(default_factory=list)


class ReasoningMetadata(BaseModel):
    """Safe-to-expose evidence summary. Not model chain-of-thought."""

    model_config = ConfigDict(extra="forbid")

    intent: str | None = None
    restated_query: str | None = None
    step_back_query: str | None = None
    situation: str | None = None
    summary: str = ""
    evidence_paths: list[str] = Field(default_factory=list)


class RetrievalInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vector_hit_count: int = 0
    skipped_retrieval: bool = False


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[RetrievedSource] = Field(default_factory=list)
    graph_evidence: GraphEvidence = Field(default_factory=GraphEvidence)
    reasoning: ReasoningMetadata = Field(default_factory=ReasoningMetadata)
    retrieval: RetrievalInfo = Field(default_factory=RetrievalInfo)


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    document_id: str | None = None


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    chunk_count: int
    entity_count: int
    relationship_count: int
    skipped: list[str] = Field(default_factory=list)
    documents: list[str] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)


def query_response_from_state(state: dict[str, Any]) -> QueryResponse:
    hits = state.get("vector_hits") or []
    graph = state.get("graph") or {}
    subgraph = graph.get("subgraph") or {}
    sources = [
        RetrievedSource(
            chunk_id=hit["chunk"]["id"],
            document=hit["chunk"].get("document"),
            section=hit["chunk"].get("section"),
            source=hit["chunk"].get("source"),
            score=float(hit.get("score") or 0),
            entity_ids=list(hit["chunk"].get("entity_ids") or []),
        )
        for hit in hits
    ]
    relationships = [
        GraphEdge(
            type=rel["type"],
            source_id=rel["source_id"],
            target_id=rel["target_id"],
        )
        for rel in subgraph.get("relationships") or []
    ]
    return QueryResponse(
        answer=state.get("answer") or "",
        sources=sources,
        graph_evidence=GraphEvidence(
            seed_ids=list(graph.get("seed_ids") or []),
            entity_ids=[entity["id"] for entity in subgraph.get("entities") or []],
            relationships=relationships,
        ),
        reasoning=ReasoningMetadata(
            intent=state.get("intent"),
            restated_query=state.get("restated_query"),
            step_back_query=state.get("step_back_query") or state.get("restated_query"),
            situation=state.get("situation"),
            summary=state.get("reasoning_summary") or "",
            evidence_paths=list(state.get("evidence_notes") or []),
        ),
        retrieval=RetrievalInfo(
            vector_hit_count=len(sources),
            skipped_retrieval=bool(state.get("skip_retrieval")),
        ),
    )
