from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.config.settings import Settings
from agent.extraction.client import parse_json_object
from agent.extraction.schemas import LlmCompletion
from agent.graph.state import QueryState
from agent.repositories.graph import Subgraph
from agent.repositories.vector import VectorHit
from agent.retrieval.graph import GraphRetrieval, GraphRetriever
from agent.retrieval.vector import VectorRetriever

_VALID_INTENTS = {
    "diagnosis",
    "how_to",
    "definition",
    "comparison",
    "out_of_scope",
    "unknown",
}


class QueryWorkflow:
    """LangGraph workflow: retrieve, then reason, then generate a grounded answer."""

    def __init__(
        self,
        *,
        llm: LlmCompletion,
        vector: VectorRetriever,
        graph: GraphRetriever,
        hops: int = 2,
        vector_limit: int = 8,
        min_score: float = 0.05,
    ) -> None:
        self._llm = llm
        self._vector = vector
        self._graph = graph
        self._hops = hops
        self._vector_limit = vector_limit
        self._min_score = min_score

    @classmethod
    def from_settings(cls, settings: Settings) -> QueryWorkflow:
        from agent.embedding import OpenAICompatibleEmbeddings
        from agent.extraction.client import OpenAICompatibleClient
        from agent.repositories.neo4j_repository import Neo4jGraphRepository
        from agent.repositories.qdrant_repository import QdrantVectorStore

        embeddings = OpenAICompatibleEmbeddings.from_settings(settings)
        vectors = QdrantVectorStore.from_settings(settings)
        graph = Neo4jGraphRepository.from_settings(settings)
        return cls(
            llm=OpenAICompatibleClient.from_settings(settings),
            vector=VectorRetriever(vectors, embeddings),
            graph=GraphRetriever(graph),
        )

    def compile(self) -> Any:
        graph = StateGraph(QueryState)
        graph.add_node("understand_query", self.understand_query)
        graph.add_node("classify_intent", self.classify_intent)
        graph.add_node("retrieve_graph_context", self.retrieve_graph_context)
        graph.add_node("retrieve_vector_context", self.retrieve_vector_context)
        graph.add_node("evaluate_evidence", self.evaluate_evidence)
        graph.add_node("reason", self.reason)
        graph.add_node("generate_answer", self.generate_answer)
        graph.add_edge(START, "understand_query")
        graph.add_edge("understand_query", "classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            self.route_after_intent,
            {
                "retrieve_graph_context": "retrieve_graph_context",
                "generate_answer": "generate_answer",
            },
        )
        graph.add_edge("retrieve_graph_context", "retrieve_vector_context")
        graph.add_edge("retrieve_vector_context", "evaluate_evidence")
        graph.add_edge("evaluate_evidence", "reason")
        graph.add_edge("reason", "generate_answer")
        graph.add_edge("generate_answer", END)
        return graph.compile()

    def understand_query(self, state: QueryState) -> QueryState:
        query = state["query"]
        raw = self._llm.complete(
            system=(
                "Restate the user's business question in one clear sentence. "
                'Return JSON {"restated_query": "..."} only.'
            ),
            user=query,
        )
        payload = _safe_json(raw)
        restated = str(payload.get("restated_query") or query).strip() or query
        return {"restated_query": restated}

    def classify_intent(self, state: QueryState) -> QueryState:
        raw = self._llm.complete(
            system=(
                "Classify the business question intent. "
                "Allowed: diagnosis, how_to, definition, comparison, out_of_scope, unknown. "
                'Return JSON {"intent": "..."} only.'
            ),
            user=state.get("restated_query") or state["query"],
        )
        payload = _safe_json(raw)
        intent = str(payload.get("intent") or "unknown").strip()
        if intent not in _VALID_INTENTS:
            intent = "unknown"
        skip = intent == "out_of_scope"
        return {"intent": intent, "skip_retrieval": skip}

    def route_after_intent(self, state: QueryState) -> str:
        if state.get("skip_retrieval"):
            return "generate_answer"
        return "retrieve_graph_context"

    def retrieve_graph_context(self, state: QueryState) -> QueryState:
        query = state.get("restated_query") or state["query"]
        retrieval = self._graph.retrieve(query, hops=self._hops)
        return {"graph": retrieval.model_dump()}

    def retrieve_vector_context(self, state: QueryState) -> QueryState:
        query = state.get("restated_query") or state["query"]
        hits = self._vector.retrieve(query, limit=self._vector_limit)
        return {"vector_hits": [hit.model_dump() for hit in hits]}

    def evaluate_evidence(self, state: QueryState) -> QueryState:
        hits = [
            VectorHit.model_validate(item)
            for item in state.get("vector_hits") or []
            if float(item.get("score") or 0) >= self._min_score
        ]
        graph = GraphRetrieval.model_validate(state.get("graph") or {})
        extra_seeds = [
            entity_id
            for hit in hits
            for entity_id in hit.chunk.entity_ids
            if entity_id not in graph.seed_ids
        ]
        if extra_seeds:
            extra = self._graph.retrieve("", seed_ids=extra_seeds, hops=self._hops)
            graph = _merge_graph(graph, extra)
        notes: list[str] = []
        if not hits:
            notes.append("No vector chunks met the relevance threshold.")
        if not graph.subgraph.entities:
            notes.append("No graph entities were retrieved.")
        for hit in hits:
            notes.append(
                f"chunk {hit.chunk.id} ({hit.chunk.document}/{hit.chunk.section}) "
                f"score={hit.score:.3f}"
            )
        for rel in graph.subgraph.relationships:
            notes.append(f"{rel.source_id} -{rel.type}-> {rel.target_id}")
        return {
            "vector_hits": [hit.model_dump() for hit in hits],
            "graph": graph.model_dump(),
            "evidence_notes": notes,
        }

    def reason(self, state: QueryState) -> QueryState:
        raw = self._llm.complete(
            system=(
                "Summarize retrieved evidence as short evidence paths. "
                "Do not invent sources. Do not include hidden chain-of-thought. "
                'Return JSON {"reasoning_summary": "..."} only.'
            ),
            user=json.dumps(
                {
                    "query": state.get("restated_query") or state["query"],
                    "intent": state.get("intent"),
                    "evidence_notes": state.get("evidence_notes") or [],
                }
            ),
        )
        payload = _safe_json(raw)
        summary = str(payload.get("reasoning_summary") or "").strip()
        if not summary:
            summary = "; ".join(state.get("evidence_notes") or []) or "No evidence paths."
        return {"reasoning_summary": summary}

    def generate_answer(self, state: QueryState) -> QueryState:
        if state.get("skip_retrieval"):
            return {
                "answer": (
                    "This question is outside the ingested business-strategy knowledge. "
                    "Ingest relevant documents and ask again."
                ),
                "source_documents": [],
                "reasoning_summary": state.get("reasoning_summary") or "Retrieval skipped.",
            }
        hits = [VectorHit.model_validate(item) for item in state.get("vector_hits") or []]
        documents = list(
            dict.fromkeys(hit.chunk.document for hit in hits if hit.chunk.document)
        )
        raw = self._llm.complete(
            system=(
                "Answer using only the evidence. Cite document names. "
                "If evidence is insufficient, say so. "
                'Return JSON {"answer": "..."} only.'
            ),
            user=json.dumps(
                {
                    "query": state.get("restated_query") or state["query"],
                    "reasoning_summary": state.get("reasoning_summary"),
                    "evidence_notes": state.get("evidence_notes") or [],
                    "chunks": [
                        {
                            "document": hit.chunk.document,
                            "section": hit.chunk.section,
                            "text": hit.chunk.text,
                        }
                        for hit in hits
                    ],
                }
            ),
        )
        payload = _safe_json(raw)
        answer = str(payload.get("answer") or "").strip()
        if not answer:
            answer = "I could not produce a grounded answer from the retrieved evidence."
        return {"answer": answer, "source_documents": documents}


def _safe_json(text: str) -> dict[str, Any]:
    try:
        parsed = parse_json_object(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _merge_graph(base: GraphRetrieval, extra: GraphRetrieval) -> GraphRetrieval:
    entities = {entity.id: entity for entity in base.subgraph.entities}
    rels = {rel.id: rel for rel in base.subgraph.relationships}
    seeds = list(base.seed_ids)
    for entity_id in extra.seed_ids:
        if entity_id not in seeds:
            seeds.append(entity_id)
    for entity in extra.subgraph.entities:
        entities[entity.id] = entity
    for rel in extra.subgraph.relationships:
        rels[rel.id] = rel
    return GraphRetrieval(
        seed_ids=seeds,
        subgraph=Subgraph(entities=list(entities.values()), relationships=list(rels.values())),
    )
