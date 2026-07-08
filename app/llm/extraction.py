"""Schema-validated clinical note extraction."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings, get_settings
from app.ingestion.pipeline import IngestionUnavailableError
from app.llm.client import ChatClient, LlmUnavailableError, OllamaOpenAIClient
from app.llm.rag import build_context
from app.observability import EXTRACTION_RETRY_COUNT
from app.retrieval.search import QdrantSearcher, SearchResponse, SearchResult

logger = logging.getLogger(__name__)


class Searcher(Protocol):
    """Protocol for retrieval dependencies used during note extraction."""

    def search(
        self,
        question: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        apply_default_threshold: bool = True,
    ) -> SearchResponse:
        """Search for relevant reference chunks."""


class SuggestionReference(BaseModel):
    """Source document used for the AI suggestion."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)


class AISuggestion(BaseModel):
    """Grounded next-step suggestion with source references."""

    model_config = ConfigDict(extra="forbid")

    suggestion: str = Field(min_length=1)
    referenced_documents: list[SuggestionReference] = Field(default_factory=list)


class PatientStatus(BaseModel):
    """Current patient status broken into clinically useful components."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
        str_strip_whitespace=True,
    )

    overall_description: str = Field(
        alias="Overall description",
        description="Brief overall description of the patient's current status.",
        min_length=1,
    )
    chief_complaint: str = Field(
        alias="Chief Complaint",
        description="Main presenting complaint, such as chest pain or cough.",
        min_length=1,
    )
    hpi: str = Field(
        alias="HPI",
        description="History of present illness from the transcript.",
        min_length=1,
    )
    red_flags: list[str] = Field(
        alias="Red Flags",
        description="Concerning symptoms or warning signs, such as shortness of breath.",
    )


class ClinicalNote(BaseModel):
    """Schema-validated clinical note generated from a transcript."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
        str_strip_whitespace=True,
    )

    patient_id: str = Field(
        alias="Patient ID",
        description="Patient identifier from the transcript; use 'unknown' if absent.",
        min_length=1,
    )
    current_status: PatientStatus = Field(
        alias="Current status of the patient",
        description="Current patient status split into description, complaint, HPI, and red flags.",
    )
    medicine_and_dosage: list[str] = Field(
        alias="Medicine and its dosage",
        description=(
            "Medicines with dosage and frequency when available; use an empty list if none "
            "are mentioned."
        ),
    )
    further_tests: list[str] = Field(
        alias="Further tests if needed",
        description=(
            "Further tests or follow-up investigations explicitly mentioned as needed; use "
            "an empty list if none are mentioned."
        ),
    )
    ai_suggestion: AISuggestion = Field(
        alias="AI suggestion",
        description=(
            "A next-step suggestion grounded in retrieved Qdrant reference context. If no "
            "relevant context is available, suggest consulting the treating doctor."
        ),
    )


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


@dataclass(frozen=True)
class SuggestionContext:
    """Retrieved context used to ground the AI suggestion."""

    text: str
    references: list[SuggestionReference]
    timings_ms: dict[str, float]
    unavailable_reason: str | None = None


def extract_clinical_note(
    transcript: str,
    *,
    client: ChatClient | None = None,
    searcher: Searcher | None = None,
    settings: Settings | None = None,
) -> ExtractionResponse:
    """Extract a ClinicalNote from transcript text with validation retries."""
    active_settings = settings or get_settings()
    llm_client = client or OllamaOpenAIClient(active_settings)
    timings: dict[str, float] = {}
    errors: list[str] = []
    raw_output = ""
    suggestion_context = _suggestion_context(transcript, active_settings, searcher)
    timings.update(suggestion_context.timings_ms)
    messages = _initial_messages(transcript, suggestion_context)
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
            note = _validate_note_from_output(raw_output)
            note = _ground_note_in_transcript(note, transcript)
            note = _with_retrieved_references(note, suggestion_context.references)
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
                        "content": _retry_prompt(error_text),
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


def _suggestion_context(
    transcript: str,
    settings: Settings,
    searcher: Searcher | None,
) -> SuggestionContext:
    active_searcher = searcher or QdrantSearcher(settings)
    try:
        response = active_searcher.search(
            transcript,
            top_k=settings.top_k,
            score_threshold=None,
            apply_default_threshold=False,
        )
    except IngestionUnavailableError as exc:
        logger.warning("Qdrant reference retrieval failed during extraction: %s", exc)
        return SuggestionContext(text="", references=[], timings_ms={}, unavailable_reason=str(exc))

    relevant_results = [
        result for result in response.results if result.score >= settings.score_threshold
    ][:2]
    if not relevant_results:
        return SuggestionContext(
            text="",
            references=[],
            timings_ms=response.timings_ms,
            unavailable_reason="No Qdrant reference chunk met the relevance threshold.",
        )

    context = build_context(relevant_results)
    return SuggestionContext(
        text=context.text,
        references=_references_from_results(relevant_results),
        timings_ms=response.timings_ms,
    )


def _references_from_results(results: list[SearchResult]) -> list[SuggestionReference]:
    references: list[SuggestionReference] = []
    seen: set[tuple[str, int]] = set()
    for result in results:
        key = (result.doc_id, result.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        references.append(
            SuggestionReference(
                doc_id=result.doc_id,
                title=result.title,
                chunk_index=result.chunk_index,
            )
        )
    return references


def _with_retrieved_references(
    note: ClinicalNote,
    references: list[SuggestionReference],
) -> ClinicalNote:
    suggestion = note.ai_suggestion.model_copy(update={"referenced_documents": references})
    return note.model_copy(update={"ai_suggestion": suggestion})


def _ground_note_in_transcript(note: ClinicalNote, transcript: str) -> ClinicalNote:
    transcript_text = _clean_text(transcript)
    patient_id = _extract_patient_id(transcript_text)
    status = PatientStatus(
        overall_description=transcript_text or note.current_status.overall_description,
        chief_complaint=_extract_chief_complaint(transcript_text),
        hpi=transcript_text or note.current_status.hpi,
        red_flags=_extract_red_flags(transcript_text),
    )
    return note.model_copy(
        update={
            "patient_id": patient_id,
            "current_status": status,
            "medicine_and_dosage": _extract_medicines(transcript_text),
            "further_tests": _extract_further_tests(transcript_text),
        }
    )


def _extract_patient_id(transcript: str) -> str:
    match = re.search(
        r"\b(?:patient\s+)?id(?:\s+of)?\s*(?:is\s*)?[:#-]?\s*([A-Za-z0-9-]+)",
        transcript,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return "unknown"


def _extract_chief_complaint(transcript: str) -> str:
    patterns = [
        r"\b(?:is\s+)?having\s+([^.;]+)",
        r"\b(?:has|had|presents\s+with|complains\s+of)\s+([^.;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, transcript, flags=re.IGNORECASE)
        if match:
            return _trim_phrase(match.group(1))
    first_sentence = re.split(r"[.;]", transcript, maxsplit=1)[0].strip()
    return first_sentence or "unknown"


def _extract_red_flags(transcript: str) -> list[str]:
    patterns = [
        r"shortness of breath",
        r"chest pain",
        r"severe pain(?:\s+in\s+the\s+[A-Za-z ]+)?",
        r"problem with urination",
        r"unable to urinate",
        r"not able to take (?:any kind of )?medication",
        r"supposed to be admitted",
    ]
    red_flags: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, transcript, flags=re.IGNORECASE):
            _append_unique(red_flags, _trim_phrase(match.group(0)))
    return red_flags


def _extract_medicines(transcript: str) -> list[str]:
    medicines: list[str] = []
    dose_pattern = (
        r"\b([A-Za-z][A-Za-z-]+)\s+(\d+(?:\.\d+)?)\s*"
        r"(milligrams?|mg|grams?|g|micrograms?|mcg)\b"
        r"((?:\s+(?:once|twice|three\s+times|four\s+times))?"
        r"(?:\s+(?:daily|per\s+day|a\s+day|weekly|monthly))?)"
    )
    for match in re.finditer(dose_pattern, transcript, flags=re.IGNORECASE):
        name = match.group(1)
        dose = f"{match.group(2)} {match.group(3)}"
        frequency = _trim_phrase(match.group(4))
        medicine = " ".join(part for part in (name, dose, frequency) if part)
        _append_unique(medicines, medicine)
    return medicines


def _extract_further_tests(transcript: str) -> list[str]:
    tests: list[str] = []
    test_terms = {
        "blood": "blood tests",
        "urine": "urine tests",
        "urination": "urine tests",
        "ecg": "ECG",
        "x-ray": "X-ray",
        "xray": "X-ray",
        "ultrasound": "ultrasound",
    }
    for match in re.finditer(r"\btests?\s+(?:of|for)\s+([^.;]+)", transcript, re.IGNORECASE):
        phrase = match.group(1).lower()
        for term, label in test_terms.items():
            if term in phrase:
                _append_unique(tests, label)
    if re.search(r"\bblood\b", transcript, re.IGNORECASE) and re.search(
        r"\btests?\b", transcript, re.IGNORECASE
    ):
        _append_unique(tests, "blood tests")
    return tests


def _clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _trim_phrase(text: str) -> str:
    return _clean_text(text).strip(" ,;:.")


def _initial_messages(
    transcript: str,
    suggestion_context: SuggestionContext,
) -> list[dict[str, str]]:
    schema = json.dumps(ClinicalNote.model_json_schema(), ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": (
                "You are a JSON-only clinical extraction engine. Your whole response must be "
                "one valid JSON object that starts with { and ends with }. Do not include "
                "markdown, code fences, prose, examples, or schema explanations."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Transcript:\n{transcript}\n\n"
                "Use the transcript above as the only source for Patient ID, Current status, "
                "Medicine and dosage, and Further tests. Do not copy symptoms, medicines, "
                "tests, or diagnoses from the Qdrant context into those fields.\n\n"
                f"Qdrant reference context for AI suggestion only:\n"
                f"{_format_suggestion_context(suggestion_context)}\n\n"
                f"ClinicalNote JSON schema:\n{schema}\n\n"
                f"Required output shape:\n{_output_template()}\n\n"
                "Return only one valid JSON object matching the schema. For "
                "'AI suggestion.suggestion', use only the Qdrant reference context "
                "when it is present. "
                "If no Qdrant context is present, suggest consulting the treating doctor "
                "or clinical team. Set 'referenced_documents' to [] because the application "
                "will attach the actual Qdrant document references."
            ),
        },
    ]


def _format_suggestion_context(suggestion_context: SuggestionContext) -> str:
    if not suggestion_context.text:
        reason = suggestion_context.unavailable_reason or "No relevant context was retrieved."
        return f"{reason}\nAllowed referenced_documents: []"

    allowed_references = "\n".join(
        (
            f"- doc_id={reference.doc_id}, title={reference.title}, "
            f"chunk_index={reference.chunk_index}"
        )
        for reference in suggestion_context.references
    )
    return (
        "Allowed referenced_documents:\n"
        f"{allowed_references}\n\n"
        "Retrieved context:\n"
        f"{suggestion_context.text}"
    )


def _retry_prompt(error_text: str) -> str:
    return (
        "The previous response was rejected. Return a corrected JSON object only. "
        "Start with { and end with }. Do not include prose, markdown, code fences, "
        "schemas, examples, or comments. Do not use snake_case keys. Use exactly this "
        f"shape:\n{_output_template()}\n\nValidation errors:\n{error_text}"
    )


def _output_template() -> str:
    return json.dumps(
        {
            "Patient ID": "patient identifier or unknown",
            "Current status of the patient": {
                "Overall description": "transcript-only current status",
                "Chief Complaint": "transcript-only main complaint",
                "HPI": "transcript-only history",
                "Red Flags": [],
            },
            "Medicine and its dosage": [],
            "Further tests if needed": [],
            "AI suggestion": {
                "suggestion": "grounded next step or consult the treating doctor",
                "referenced_documents": [],
            },
        },
        indent=2,
        ensure_ascii=False,
    )


def _validate_note_from_output(raw_output: str) -> ClinicalNote:
    last_error: ValidationError | None = None
    for candidate in _json_candidates(raw_output):
        try:
            return ClinicalNote.model_validate_json(candidate)
        except ValidationError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return ClinicalNote.model_validate_json(raw_output)


def _json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    _append_unique(candidates, _strip_json_fence(text))

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        _append_unique(candidates, text[index : index + end])
    return candidates


def _append_unique(items: list[str], value: str) -> None:
    stripped = value.strip()
    if stripped and stripped not in items:
        items.append(stripped)


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
