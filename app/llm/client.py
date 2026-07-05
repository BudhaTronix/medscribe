"""OpenAI-compatible local LLM client for Ollama."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.config import Settings, get_settings


class LlmUnavailableError(RuntimeError):
    """Raised when the local LLM endpoint is unavailable."""


class ChatClient(Protocol):
    """Protocol for chat clients used by extraction and RAG."""

    def generate(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        """Generate text from chat messages."""


@dataclass(frozen=True)
class LlmStatus:
    """Status from the local LLM endpoint."""

    reachable: bool
    detail: str


class OllamaOpenAIClient:
    """Small wrapper around the OpenAI client pointed at Ollama."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: object | None = None

    def generate(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        """Generate a response from the configured local model."""
        try:
            client = self._load_client()
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
        except Exception as exc:
            msg = f"LLM endpoint is unavailable at {self.settings.llm_base_url}: {exc}"
            raise LlmUnavailableError(msg) from exc
        if content is None:
            msg = "LLM returned an empty response"
            raise LlmUnavailableError(msg)
        return str(content).strip()

    def model_status(self) -> LlmStatus:
        """Check whether the model-list endpoint responds."""
        try:
            client = self._load_client()
            client.models.list()
        except Exception as exc:
            return LlmStatus(reachable=False, detail=str(exc))
        return LlmStatus(reachable=True, detail="model list reachable")

    def _load_client(self) -> object:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                msg = "openai is not installed in this environment"
                raise LlmUnavailableError(msg) from exc
            self._client = OpenAI(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
            )
        return self._client
