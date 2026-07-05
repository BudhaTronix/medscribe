"""Command line interface for local demo operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from app.asr.transcriber import TranscriptionUnavailableError, WhisperTranscriber
from app.config import get_settings
from app.ingestion.pipeline import IngestionUnavailableError, ingest_corpus
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


def main() -> None:
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
