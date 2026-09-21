from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-based application settings. Secrets never have defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "graphrag-agent"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://opencode.ai/zen/go/v1"
    llm_model: str = "deepseek-v4.1-flash"
    llm_max_tokens: int = 8192
    llm_json_mode: bool = False

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "business_chunks"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr | None = None

    embedding_provider: Literal["local", "openai", "voyage", "bge_m3"] = "bge_m3"
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_API_KEY", "VOYAGE_API_KEY"),
    )
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_dimensions: int = 1024
    ingest_max_chars: int = 2500
    ingest_overlap_chars: int = 200
    ingest_extract_concurrency: int = 8
    ingest_pdf_ocr: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
