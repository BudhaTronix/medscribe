"""Tests for grounded RAG refusal and citation assembly."""

from app.config import Settings
from app.llm.rag import REFUSAL_TEXT, answer_question, build_context
from app.retrieval.search import SearchResponse, SearchResult


class FakeSearcher:
    """Fake searcher for RAG tests."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(
        self,
        question: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        apply_default_threshold: bool = True,
    ) -> SearchResponse:
        """Return configured results."""
        del question, top_k, score_threshold, apply_default_threshold
        return SearchResponse(results=self.results, timings_ms={"embed": 1.0, "retrieve": 2.0})


class FakeClient:
    """Fake LLM client for RAG generation."""

    def generate(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        """Return a stable grounded answer."""
        del messages, temperature
        return "Grounded answer from context."


def _result(score: float) -> SearchResult:
    return SearchResult(
        doc_id="doc-1",
        title="Synthetic title",
        chunk_index=0,
        text="Synthetic context",
        language="en",
        score=score,
    )


def test_refusal_threshold_blocks_generation() -> None:
    response = answer_question(
        "outside corpus",
        searcher=FakeSearcher([_result(0.2)]),
        client=FakeClient(),
        settings=Settings(score_threshold=0.35),
    )

    assert response.refused is True
    assert response.answer == REFUSAL_TEXT
    assert response.citations == []


def test_rag_generate_false_returns_citations_only() -> None:
    response = answer_question(
        "inside corpus",
        searcher=FakeSearcher([_result(0.8)]),
        client=FakeClient(),
        settings=Settings(score_threshold=0.35),
        generate=False,
    )

    assert response.refused is False
    assert response.answer == ""
    assert response.citations[0].doc_id == "doc-1"


def test_citation_assembly_from_results() -> None:
    context = build_context([_result(0.7)])

    assert "doc-1:0" in context.text
    assert context.citations[0].score == 0.7
