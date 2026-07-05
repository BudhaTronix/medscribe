"""Evaluate corpus retrieval quality with synthetic questions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.config import get_settings
from app.ingestion.pipeline import IngestionUnavailableError
from app.retrieval.search import QdrantSearcher

QUESTIONS_PATH = Path("eval/questions.yaml")
RESULTS_PATH = Path("eval/results/retrieval_results.md")


@dataclass(frozen=True)
class RetrievalRow:
    """Per-question retrieval result."""

    question_id: str
    relevant_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    rank: int | None
    refused_correctly: bool | None


def evaluate_retrieval() -> list[RetrievalRow]:
    """Evaluate hit rates, MRR, and refusal correctness."""
    questions = _load_questions()
    settings = get_settings()
    searcher = QdrantSearcher(settings)
    rows: list[RetrievalRow] = []

    try:
        for item in questions:
            relevant = list(item.get("relevant_doc_ids", []))
            response = searcher.search(
                str(item["question"]),
                top_k=5,
                score_threshold=None,
                apply_default_threshold=False,
            )
            retrieved = [result.doc_id for result in response.results]
            if item.get("out_of_corpus"):
                best_score = max((result.score for result in response.results), default=0.0)
                rows.append(
                    RetrievalRow(
                        question_id=str(item["id"]),
                        relevant_doc_ids=relevant,
                        retrieved_doc_ids=retrieved,
                        rank=None,
                        refused_correctly=best_score < settings.score_threshold,
                    )
                )
                continue
            rows.append(
                RetrievalRow(
                    question_id=str(item["id"]),
                    relevant_doc_ids=relevant,
                    retrieved_doc_ids=retrieved,
                    rank=_rank(relevant, retrieved),
                    refused_correctly=None,
                )
            )
    except IngestionUnavailableError as exc:
        _write_unavailable_results(str(exc))
        return rows

    _write_results(rows)
    return rows


def main() -> None:
    """Run retrieval evaluation."""
    evaluate_retrieval()


def _load_questions() -> list[dict[str, Any]]:
    raw = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        msg = "eval/questions.yaml must contain a list"
        raise ValueError(msg)
    return raw


def _rank(relevant_doc_ids: list[str], retrieved_doc_ids: list[str]) -> int | None:
    for index, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return index
    return None


def _write_unavailable_results(reason: str) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        "# Retrieval Results\n\n"
        "Retrieval evaluation was not run because Qdrant is unavailable.\n\n"
        f"Reason: `{reason}`\n",
        encoding="utf-8",
    )


def _write_results(rows: list[RetrievalRow]) -> None:
    answerable = [row for row in rows if row.refused_correctly is None]
    refusal = [row for row in rows if row.refused_correctly is not None]
    hit_at_1 = _hit_at(answerable, 1)
    hit_at_3 = _hit_at(answerable, 3)
    hit_at_5 = _hit_at(answerable, 5)
    mrr = _mrr(answerable)
    refusal_correctness = (
        sum(1 for row in refusal if row.refused_correctly) / len(refusal) if refusal else 0.0
    )

    lines = [
        "# Retrieval Results",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| hit@1 | {hit_at_1:.3f} |",
        f"| hit@3 | {hit_at_3:.3f} |",
        f"| hit@5 | {hit_at_5:.3f} |",
        f"| MRR | {mrr:.3f} |",
        f"| refusal correctness | {refusal_correctness:.3f} |",
        "",
        "| question | relevant | retrieved | rank | refusal correct |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        rank = "" if row.rank is None else str(row.rank)
        refusal_value = "" if row.refused_correctly is None else str(row.refused_correctly)
        lines.append(
            f"| {row.question_id} | {', '.join(row.relevant_doc_ids)} | "
            f"{', '.join(row.retrieved_doc_ids)} | {rank} | {refusal_value} |"
        )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _hit_at(rows: list[RetrievalRow], k: int) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.rank is not None and row.rank <= k) / len(rows)


def _mrr(rows: list[RetrievalRow]) -> float:
    if not rows:
        return 0.0
    return sum(1 / row.rank for row in rows if row.rank is not None) / len(rows)


if __name__ == "__main__":
    main()
