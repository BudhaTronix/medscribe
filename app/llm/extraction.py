"""Schema-validated clinical note extraction."""

from __future__ import annotations

import json
import logging
import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.config import Settings, get_settings
from app.llm.client import ChatClient, LlmUnavailableError, OllamaOpenAIClient
from app.observability import EXTRACTION_RETRY_COUNT

logger = logging.getLogger(__name__)


class Medication(BaseModel):
    """Medication entry in a structured clinical note."""

    name: str
    dose: str | None = None
    frequency: str | None = None


class ClinicalNote(BaseModel):
    """Schema-validated clinical note generated from a transcript."""

    language: Literal["de", "en"]
    chief_complaint: str = Field(min_length=3)
    history_of_present_illness: str
    medications: list[Medication] = []
    allergies: list[str] = []
    assessment: str
    plan: list[str] = Field(min_length=1)
    red_flags: list[str] = []
    uncertainty_notes: str = ""


class ExtractionFailure(BaseModel):
    """Typed extraction failure returned instead of raising to callers."""

    success: Literal[False] = False
    raw_output: str
    errors: list[str]
    attempts: int
    timings_ms: dict[str, float]


class ExtractionSuccess(BaseModel):
    """Successful structured extraction response."""

    success: Literal[True] = True
    note: ClinicalNote
    attempts: int
    timings_ms: dict[str, float]


ExtractionResponse = ExtractionSuccess | ExtractionFailure


def extract_clinical_note(
    transcript: str,
    *,
    client: ChatClient | None = None,
    settings: Settings | None = None,
) -> ExtractionResponse:
    """Extract a ClinicalNote from transcript text with validation retries."""
    active_settings = settings or get_settings()
    llm_client = client or OllamaOpenAIClient(active_settings)
    timings: dict[str, float] = {}
    errors: list[str] = []
    raw_output = ""
    messages = _initial_messages(transcript)
    total_attempts = active_settings.max_validation_retries + 1

    for attempt in range(1, total_attempts + 1):
        started = time.perf_counter()
        try:
            raw_output = llm_client.generate(messages, temperature=0.0)
        except LlmUnavailableError as exc:
            errors.append(str(exc))
            timings["generate"] = timings.get("generate", 0.0) + _elapsed_ms(started)
            return ExtractionFailure(
                raw_output=raw_output,
                errors=errors,
                attempts=attempt,
                timings_ms=timings,
            )
        timings["generate"] = timings.get("generate", 0.0) + _elapsed_ms(started)

        started = time.perf_counter()
        try:
            note = ClinicalNote.model_validate_json(_strip_json_fence(raw_output))
        except ValidationError as exc:
            timings["validate"] = timings.get("validate", 0.0) + _elapsed_ms(started)
            error_text = str(exc)
            errors.append(error_text)
            if attempt < total_attempts:
                EXTRACTION_RETRY_COUNT.inc()
                messages.append({"role": "assistant", "content": raw_output})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The JSON failed validation. Fix these errors and return only "
                            f"corrected JSON:\n{error_text}"
                        ),
                    }
                )
            continue
        timings["validate"] = timings.get("validate", 0.0) + _elapsed_ms(started)
        logger.info("ClinicalNote extraction succeeded on attempt %s", attempt)
        return ExtractionSuccess(note=note, attempts=attempt, timings_ms=timings)

    return ExtractionFailure(
        raw_output=raw_output,
        errors=errors,
        attempts=total_attempts,
        timings_ms=timings,
    )


def _initial_messages(transcript: str) -> list[dict[str, str]]:
    schema = json.dumps(ClinicalNote.model_json_schema(), ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": (
                "You extract structured clinical notes from synthetic dictation. "
                "Return JSON only. Do not include markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Transcript:\n{transcript}\n\n"
                f"ClinicalNote JSON schema:\n{schema}\n\n"
                "Return only one valid JSON object matching the schema."
            ),
        },
    ]


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
