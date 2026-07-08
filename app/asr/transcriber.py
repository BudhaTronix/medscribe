"""Speech transcription with faster-whisper."""

from __future__ import annotations

import ctypes
import gc
import logging
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.config import Settings, get_settings

LanguageCode = Literal["de", "en"]
logger = logging.getLogger(__name__)


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
        try:
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
        finally:
            if self.settings.cleanup_model_memory_after_use:
                self.cleanup_model_memory()

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
            if device != "cpu":
                _preload_cuda_libraries()
            try:
                self._model = WhisperModel(self.settings.whisper_model, device=device or "auto")
            except Exception as exc:
                msg = f"Unable to load Whisper model {self.settings.whisper_model}: {exc}"
                raise TranscriptionUnavailableError(msg) from exc
        return self._model

    def cleanup_model_memory(self) -> None:
        """Release the loaded Whisper model and ask CUDA allocators to free cached memory."""
        self._model = None
        gc.collect()
        _empty_torch_cuda_cache()


_cuda_libraries_loaded = False


def _preload_cuda_libraries() -> None:
    """Load pip-packaged cuBLAS and cuDNN so ctranslate2 can run on the GPU.

    ctranslate2 links against CUDA 12 libraries, which are usually absent from
    the system loader path (torch ships its own CUDA under a different soname).
    The nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels provide them, but only
    preloading makes them visible: LD_LIBRARY_PATH is read once at process
    start, so setting it here would have no effect. Best effort — without the
    wheels or a GPU, faster-whisper falls back to CPU.
    """
    global _cuda_libraries_loaded
    if _cuda_libraries_loaded:
        return
    _cuda_libraries_loaded = True
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
    except ImportError:
        return
    for module in (nvidia.cublas.lib, nvidia.cudnn.lib):
        lib_dir = Path(next(iter(module.__path__)))
        for library in sorted(lib_dir.glob("*.so*")):
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue


def _empty_torch_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        logger.warning("Unable to empty torch CUDA cache after transcription: %s", exc)


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
