"""Speech transcription with faster-whisper."""

from __future__ import annotations

import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.config import Settings, get_settings

LanguageCode = Literal["de", "en"]


class TranscriptionUnavailableError(RuntimeError):
    """Raised when ASR dependencies or audio decoding are unavailable."""


@dataclass(frozen=True)
class TranscriptionSegment:
    """A transcribed audio segment with timestamps."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    """Transcription output with timing metadata."""

    text: str
    segments: list[TranscriptionSegment]
    detected_language: str
    duration_seconds: float
    processing_seconds: float
    real_time_factor: float
    timings_ms: dict[str, float]


class WhisperTranscriber:
    """Lazy faster-whisper transcriber configured from environment settings."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any | None = None

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: LanguageCode | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file and report wall-clock timing plus real-time factor."""
        path = Path(audio_path)
        if not path.exists():
            msg = f"Audio file does not exist: {path}"
            raise TranscriptionUnavailableError(msg)

        timings: dict[str, float] = {}
        started = time.perf_counter()
        model = self._load_model()
        timings["load_model"] = _elapsed_ms(started)

        started = time.perf_counter()
        try:
            segments_iter, info = model.transcribe(str(path), language=language)
            segments = [
                TranscriptionSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=str(segment.text).strip(),
                )
                for segment in segments_iter
            ]
        except Exception as exc:
            msg = f"Unable to transcribe {path}: {exc}"
            raise TranscriptionUnavailableError(msg) from exc
        timings["transcribe"] = _elapsed_ms(started)

        text = " ".join(segment.text for segment in segments).strip()
        processing_seconds = timings["transcribe"] / 1000
        duration_seconds = _audio_duration_seconds(path, segments)
        real_time_factor = (
            round(duration_seconds / processing_seconds, 3) if processing_seconds > 0 else 0.0
        )

        return TranscriptionResult(
            text=text,
            segments=segments,
            detected_language=str(getattr(info, "language", language or "unknown")),
            duration_seconds=round(duration_seconds, 3),
            processing_seconds=round(processing_seconds, 3),
            real_time_factor=real_time_factor,
            timings_ms=timings,
        )

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                msg = "faster-whisper is not installed in this environment"
                raise TranscriptionUnavailableError(msg) from exc
            device = (
                None if self.settings.whisper_device == "auto" else self.settings.whisper_device
            )
            try:
                self._model = WhisperModel(self.settings.whisper_model, device=device or "auto")
            except Exception as exc:
                msg = f"Unable to load Whisper model {self.settings.whisper_model}: {exc}"
                raise TranscriptionUnavailableError(msg) from exc
        return self._model


def _audio_duration_seconds(path: Path, segments: list[TranscriptionSegment]) -> float:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as audio:
                frames = audio.getnframes()
                rate = audio.getframerate()
                return frames / float(rate) if rate else 0.0
        except (wave.Error, OSError):
            pass

    ffprobe_duration = _ffprobe_duration_seconds(path)
    if ffprobe_duration is not None:
        return ffprobe_duration
    if segments:
        return max(segment.end for segment in segments)
    return 0.0


def _ffprobe_duration_seconds(path: Path) -> float | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
