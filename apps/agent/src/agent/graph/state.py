from typing import Literal, TypedDict

Intent = Literal[
    "diagnosis",
    "how_to",
    "definition",
    "comparison",
    "out_of_scope",
    "unknown",
]


class QueryState(TypedDict, total=False):
    query: str
    restated_query: str
    step_back_query: str
    situation: str
    intent: str
    skip_retrieval: bool
    vector_hits: list[dict]
    graph: dict
    evidence_notes: list[str]
    reasoning_summary: str
    answer: str
    source_documents: list[str]
