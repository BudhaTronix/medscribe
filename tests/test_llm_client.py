"""Tests for local LLM client housekeeping."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from app.config import Settings
from app.llm.client import OllamaOpenAIClient


def test_ollama_unload_posts_keep_alive_zero(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    fake_httpx = ModuleType("httpx")

    def post(url: str, *, json: dict[str, Any], timeout: int) -> FakeResponse:
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()

    fake_httpx.post = post  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    settings = Settings(llm_base_url="http://localhost:11434/v1", llm_model="mistral:7b")
    OllamaOpenAIClient(settings).unload_model()

    assert calls == [
        {
            "url": "http://localhost:11434/api/generate",
            "json": {
                "model": "mistral:7b",
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
            "timeout": 10,
        }
    ]


def test_generate_unloads_model_after_completion(monkeypatch: Any) -> None:
    unload_calls = 0

    class FakeChat:
        class completions:
            @staticmethod
            def create(**kwargs: Any) -> Any:
                del kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="done"))]
                )

    client = OllamaOpenAIClient(Settings(cleanup_model_memory_after_use=True))
    monkeypatch.setattr(client, "_load_client", lambda: SimpleNamespace(chat=FakeChat()))

    def unload_model() -> None:
        nonlocal unload_calls
        unload_calls += 1

    monkeypatch.setattr(client, "unload_model", unload_model)

    assert client.generate([{"role": "user", "content": "hi"}]) == "done"
    assert unload_calls == 1
