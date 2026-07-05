"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the local clinical documentation demo."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    whisper_model: str = "small"
    whisper_device: str = "auto"
    embedding_model: str = "BAAI/bge-m3"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "clinical_corpus"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "mistral:7b"
    chunk_size_tokens: int = Field(default=500, ge=1)
    chunk_overlap_tokens: int = Field(default=80, ge=0)
    top_k: int = Field(default=5, ge=1)
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    max_validation_retries: int = Field(default=2, ge=0)
    uvicorn_workers: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_chunk_window(self) -> Self:
        """Ensure the overlap is smaller than the chunk size."""
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            msg = "CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS"
            raise ValueError(msg)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings, mainly for tests."""
    get_settings.cache_clear()
