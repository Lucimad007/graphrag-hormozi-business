from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class LlmCompletion(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...


class ExtractedEntityDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    name: str
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)


class ExtractedRelationshipDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    source_type: str
    source_name: str
    target_type: str
    target_name: str


class ExtractionDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entities: list[ExtractedEntityDraft] = Field(default_factory=list)
    relationships: list[ExtractedRelationshipDraft] = Field(default_factory=list)
