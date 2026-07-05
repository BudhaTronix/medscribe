"""Tests for environment driven configuration."""

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


def test_config_defaults() -> None:
    settings = Settings()

    assert settings.whisper_model == "small"
    assert settings.whisper_device == "auto"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "clinical_corpus"
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_api_key == "ollama"
    assert settings.llm_model == "mistral:7b"
    assert settings.chunk_size_tokens == 500
    assert settings.chunk_overlap_tokens == 80
    assert settings.top_k == 5
    assert settings.score_threshold == 0.35
    assert settings.max_validation_retries == 2
    assert settings.uvicorn_workers == 1


def test_config_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("SCORE_THRESHOLD", "0.42")

    settings = get_settings()

    assert settings.qdrant_collection == "test_collection"
    assert settings.top_k == 9
    assert settings.score_threshold == 0.42


def test_config_rejects_overlap_equal_to_size() -> None:
    with pytest.raises(ValidationError):
        Settings(chunk_size_tokens=10, chunk_overlap_tokens=10)
