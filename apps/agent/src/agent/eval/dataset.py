from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    expect_documents: list[str] = Field(default_factory=list)
    expect_entity_ids: list[str] = Field(default_factory=list)
    expect_relationship_types: list[str] = Field(default_factory=list)


class EvalDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    description: str = ""
    cases: list[EvalCase]


def load_eval_dataset(path: Path) -> EvalDataset:
    return EvalDataset.model_validate_json(path.read_text(encoding="utf-8"))
