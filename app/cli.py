"""Command line interface for local demo operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from app.asr.transcriber import TranscriptionUnavailableError, WhisperTranscriber
from app.config import get_settings
from app.ingestion.pipeline import IngestionUnavailableError, ingest_corpus
from app.llm.client import LlmUnavailableError
from app.llm.extraction import extract_clinical_note
from app.llm.rag import answer_question
from app.retrieval.search import QdrantSearcher, format_results

app = typer.Typer(help="Clinical voice note assistant CLI.")


@app.command()
def ingest() -> None:
    """Ingest the synthetic reference corpus into Qdrant."""
    try:
        report = ingest_corpus(get_settings())
    except IngestionUnavailableError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report.__dict__, indent=2, ensure_ascii=False))


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to retrieve corpus chunks for.")],
    top_k: Annotated[int | None, typer.Option("--top-k", help="Override TOP_K.")] = None,
    generate: Annotated[bool, typer.Option("--generate/--no-generate")] = True,
) -> None:
    """Answer a question with grounded RAG, or return retrieval-only citations."""
    settings = get_settings()
    if top_k is not None:
        settings = settings.model_copy(update={"top_k": top_k})
    try:
        response = answer_question(question, settings=settings, generate=generate)
    except IngestionUnavailableError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(response.model_dump_json(indent=2))


@app.command()
def retrieve(
    question: Annotated[str, typer.Argument(help="Question to retrieve corpus chunks for.")],
    top_k: Annotated[int | None, typer.Option("--top-k", help="Override TOP_K.")] = None,
) -> None:
    """Retrieve scored corpus chunks for a clinical question."""
    try:
        response = QdrantSearcher(get_settings()).search(question, top_k=top_k)
    except IngestionUnavailableError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(format_results(response.results))
    typer.echo(json.dumps({"timings_ms": response.timings_ms}, indent=2))


@app.command()
def transcribe(
    audio: Annotated[Path, typer.Argument(help="Path to wav, mp3, or m4a audio.")],
    language: Annotated[str | None, typer.Option("--language", help="Optional de or en.")] = None,
) -> None:
    """Transcribe an audio file with faster-whisper."""
    if language not in {None, "de", "en"}:
        typer.secho("Language must be de or en when provided.", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    try:
        result = WhisperTranscriber(get_settings()).transcribe(audio, language=language)
    except TranscriptionUnavailableError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(result, default=lambda value: value.__dict__, indent=2, ensure_ascii=False)
    )


@app.command()
def structure(
    text_file: Annotated[Path, typer.Argument(help="Path to a transcript text file.")],
) -> None:
    """Extract a structured ClinicalNote from transcript text."""
    transcript = text_file.read_text(encoding="utf-8")
    try:
        result = extract_clinical_note(transcript, settings=get_settings())
    except LlmUnavailableError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(result.model_dump_json(indent=2))


def main() -> None:
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
