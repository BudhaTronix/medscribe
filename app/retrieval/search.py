"""Dense retrieval over the synthetic clinical corpus."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.ingestion.pipeline import IngestionUnavailableError, SentenceTransformerEmbedder


@dataclass(frozen=True)
class SearchResult:
    """A scored Qdrant result with citation metadata."""

    doc_id: str
    title: str
    chunk_index: int
    text: str
    language: str
    score: float


@dataclass(frozen=True)
class SearchResponse:
    """Search results with stage timings."""

    results: list[SearchResult]
    timings_ms: dict[str, float]


class QdrantSearcher:
    """Search Qdrant using a sentence-transformers embedding model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = SentenceTransformerEmbedder(self.settings.embedding_model)

    def search(self, question: str, *, top_k: int | None = None) -> SearchResponse:
        """Return scored chunks above the configured score threshold."""
        if not question.strip():
            return SearchResponse(results=[], timings_ms={"embed": 0.0, "retrieve": 0.0})

        timings: dict[str, float] = {}
        started = time.perf_counter()
        self._ensure_collection_available()
        timings["qdrant_check"] = _elapsed_ms(started)

        started = time.perf_counter()
        vector = self.embedder.embed([question])[0]
        timings["embed"] = _elapsed_ms(started)

        started = time.perf_counter()
        points = self._query(vector, top_k or self.settings.top_k)
        timings["retrieve"] = _elapsed_ms(started)

        results = [_point_to_result(point) for point in points]
        return SearchResponse(results=results, timings_ms=timings)

    def _query(self, vector: list[float], top_k: int) -> list[Any]:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            msg = "qdrant-client is not installed in this environment"
            raise IngestionUnavailableError(msg) from exc

        try:
            client = QdrantClient(url=self.settings.qdrant_url)
            response = client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=vector,
                limit=top_k,
                score_threshold=self.settings.score_threshold,
                with_payload=True,
            )
            return list(response.points)
        except Exception as exc:
            msg = f"Qdrant is unavailable at {self.settings.qdrant_url}: {exc}"
            raise IngestionUnavailableError(msg) from exc

    def _ensure_collection_available(self) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            msg = "qdrant-client is not installed in this environment"
            raise IngestionUnavailableError(msg) from exc

        try:
            client = QdrantClient(url=self.settings.qdrant_url)
            if not client.collection_exists(self.settings.qdrant_collection):
                msg = f"Qdrant collection {self.settings.qdrant_collection} does not exist"
                raise IngestionUnavailableError(msg)
        except IngestionUnavailableError:
            raise
        except Exception as exc:
            msg = f"Qdrant is unavailable at {self.settings.qdrant_url}: {exc}"
            raise IngestionUnavailableError(msg) from exc


def format_results(results: list[SearchResult]) -> str:
    """Format search results for CLI display."""
    if not results:
        return "No chunks met the score threshold."
    lines: list[str] = []
    for result in results:
        lines.append(
            f"{result.score:.3f} | {result.doc_id} | chunk {result.chunk_index} | {result.title}"
        )
        lines.append(result.text)
        lines.append("")
    return "\n".join(lines).strip()


def _point_to_result(point: Any) -> SearchResult:
    payload = point.payload or {}
    return SearchResult(
        doc_id=str(payload.get("doc_id", "")),
        title=str(payload.get("title", "")),
        chunk_index=int(payload.get("chunk_index", 0)),
        text=str(payload.get("text", "")),
        language=str(payload.get("language", "")),
        score=float(point.score),
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
