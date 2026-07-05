"""Grounded question answering over retrieved corpus chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from app.config import Settings, get_settings
from app.llm.client import ChatClient, LlmUnavailableError, OllamaOpenAIClient
from app.retrieval.search import QdrantSearcher, SearchResponse, SearchResult

REFUSAL_TEXT = "The local corpus does not contain enough information to answer this question."


class Searcher(Protocol):
    """Protocol for retrieval dependencies used by RAG."""

    def search(
        self,
        question: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        apply_default_threshold: bool = True,
    ) -> SearchResponse:
        """Search for relevant chunks."""


class Citation(BaseModel):
    """Citation for a retrieved corpus chunk."""

    doc_id: str
    title: str
    chunk_index: int
    score: float


class RagResponse(BaseModel):
    """Grounded answer response."""

    answer: str
    citations: list[Citation]
    refused: bool
    timings_ms: dict[str, float]


@dataclass(frozen=True)
class ContextBlock:
    """Context text and citation list for generation."""

    text: str
    citations: list[Citation]


def answer_question(
    question: str,
    *,
    searcher: Searcher | None = None,
    client: ChatClient | None = None,
    settings: Settings | None = None,
    generate: bool = True,
) -> RagResponse:
    """Answer a question from retrieved context, refusing below threshold."""
    active_settings = settings or get_settings()
    active_searcher = searcher or QdrantSearcher(active_settings)
    search_response = active_searcher.search(
        question,
        top_k=active_settings.top_k,
        score_threshold=None,
        apply_default_threshold=False,
    )
    timings = dict(search_response.timings_ms)

    if _should_refuse(search_response.results, active_settings.score_threshold):
        return RagResponse(answer=REFUSAL_TEXT, citations=[], refused=True, timings_ms=timings)

    context = build_context(search_response.results)
    if not generate:
        return RagResponse(
            answer="",
            citations=context.citations,
            refused=False,
            timings_ms=timings,
        )

    llm_client = client or OllamaOpenAIClient(active_settings)
    started = time.perf_counter()
    try:
        answer = llm_client.generate(_messages(question, context.text), temperature=0.0)
    except LlmUnavailableError as exc:
        answer = f"LLM endpoint is unavailable: {exc}"
    timings["generate"] = _elapsed_ms(started)
    return RagResponse(
        answer=answer,
        citations=context.citations,
        refused=False,
        timings_ms=timings,
    )


def build_context(results: list[SearchResult]) -> ContextBlock:
    """Build the grounded context block and citations."""
    context_parts: list[str] = []
    citations: list[Citation] = []
    for result in results:
        citation_id = f"{result.doc_id}:{result.chunk_index}"
        context_parts.append(
            f"[{citation_id}] {result.title} (score {result.score:.3f})\n{result.text}"
        )
        citations.append(
            Citation(
                doc_id=result.doc_id,
                title=result.title,
                chunk_index=result.chunk_index,
                score=result.score,
            )
        )
    return ContextBlock(text="\n\n".join(context_parts), citations=citations)


def _should_refuse(results: list[SearchResult], score_threshold: float) -> bool:
    if not results:
        return True
    return max(result.score for result in results) < score_threshold


def _messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer only from the supplied context. If the context is insufficient, "
                "say that the context is insufficient. Include concise source references."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}",
        },
    ]


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
