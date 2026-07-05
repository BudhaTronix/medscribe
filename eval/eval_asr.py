"""Evaluate ASR recordings against synthetic reference transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.asr.transcriber import TranscriptionUnavailableError, WhisperTranscriber
from app.asr.wer import compute_error_rates

RECORDINGS_DIR = Path("data/audio/recordings")
REFERENCES_DIR = Path("data/audio/references")
RESULTS_PATH = Path("eval/results/asr_results.md")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}


@dataclass(frozen=True)
class AsrRow:
    """One ASR evaluation row."""

    file: str
    language: str
    duration: float
    wer: float
    cer: float
    real_time_factor: float


def evaluate_asr() -> list[AsrRow]:
    """Evaluate all available recordings with matching references."""
    rows: list[AsrRow] = []
    recordings = [
        path
        for path in sorted(RECORDINGS_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not recordings:
        _write_empty_results()
        return rows

    transcriber = WhisperTranscriber()
    for recording in recordings:
        reference_path = REFERENCES_DIR / f"{recording.stem}.txt"
        if not reference_path.exists():
            continue
        language = _language_from_stem(recording.stem)
        forced_language = language if language in {"de", "en"} else None
        reference = reference_path.read_text(encoding="utf-8")
        result = transcriber.transcribe(recording, language=forced_language)
        rates = compute_error_rates(reference, result.text, language=forced_language)
        rows.append(
            AsrRow(
                file=recording.name,
                language=language,
                duration=result.duration_seconds,
                wer=rates.wer,
                cer=rates.cer,
                real_time_factor=result.real_time_factor,
            )
        )
    _write_results(rows)
    return rows


def main() -> None:
    """Run the ASR evaluation and write markdown results."""
    try:
        evaluate_asr()
    except TranscriptionUnavailableError as exc:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        content = f"# ASR Results\n\nASR evaluation failed: {exc}\n"
        RESULTS_PATH.write_text(content, encoding="utf-8")
        raise SystemExit(1) from exc


def _write_empty_results() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        "# ASR Results\n\nNo recordings found in `data/audio/recordings`.\n",
        encoding="utf-8",
    )


def _write_results(rows: list[AsrRow]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ASR Results",
        "",
        "| file | language | duration | WER | CER | real-time factor |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.file} | {row.language} | {row.duration:.2f} | "
            f"{row.wer:.3f} | {row.cer:.3f} | {row.real_time_factor:.3f} |"
        )
    if rows:
        mean_wer = sum(row.wer for row in rows) / len(rows)
        mean_cer = sum(row.cer for row in rows) / len(rows)
        mean_rtf = sum(row.real_time_factor for row in rows) / len(rows)
        mean_duration = sum(row.duration for row in rows) / len(rows)
        lines.append(
            f"| mean | all | {mean_duration:.2f} | {mean_wer:.3f} | "
            f"{mean_cer:.3f} | {mean_rtf:.3f} |"
        )
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _language_from_stem(stem: str) -> str:
    parts = stem.split("_")
    for part in parts:
        if part in {"de", "en"}:
            return part
    return "unknown"


if __name__ == "__main__":
    main()
