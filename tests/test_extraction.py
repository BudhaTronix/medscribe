"""Tests for schema-validated LLM extraction."""

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.llm.extraction import (
    ClinicalNote,
    ExtractionFailure,
    ExtractionSuccess,
    extract_clinical_note,
)
from app.retrieval.search import SearchResponse, SearchResult


class FakeClient:
    """Fake LLM client returning configured outputs."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.seen_messages: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        """Return the next configured output."""
        del temperature
        self.seen_messages.append(messages)
        output = self.outputs[self.calls]
        self.calls += 1
        return output


class FakeSearcher:
    """Fake Qdrant searcher returning configured results."""

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
        """Return configured search results."""
        del question, top_k, score_threshold, apply_default_threshold
        return SearchResponse(results=self.results, timings_ms={"embed": 1.0, "retrieve": 2.0})


VALID_NOTE_JSON = """
{
  "Patient ID": "P-001",
  "Current status of the patient": {
    "Overall description": "Synthetic patient with improving cough.",
    "Chief Complaint": "Cough",
    "HPI": "Cough is improving after observation.",
    "Red Flags": ["worsening breathlessness"]
  },
  "Medicine and its dosage": ["metformin 500 mg twice daily"],
  "Further tests if needed": ["repeat oxygen saturation if symptoms worsen"],
  "AI suggestion": {
    "suggestion": "Follow the local reference and consult the treating doctor if symptoms worsen.",
    "referenced_documents": []
  }
}
"""


def test_clinical_note_validation_happy_path() -> None:
    note = ClinicalNote.model_validate_json(VALID_NOTE_JSON)

    assert note.patient_id == "P-001"
    assert note.current_status.chief_complaint == "Cough"
    assert note.medicine_and_dosage == ["metformin 500 mg twice daily"]
    assert set(note.model_dump()) == {
        "Patient ID",
        "Current status of the patient",
        "Medicine and its dosage",
        "Further tests if needed",
        "AI suggestion",
    }
    assert set(note.model_dump()["Current status of the patient"]) == {
        "Overall description",
        "Chief Complaint",
        "HPI",
        "Red Flags",
    }


def test_clinical_note_validation_failure() -> None:
    with pytest.raises(ValidationError):
        ClinicalNote.model_validate({"Patient ID": "P-001"})


def test_clinical_note_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ClinicalNote.model_validate(
            {
                "Patient ID": "P-001",
                "Current status of the patient": {
                    "Overall description": "Stable",
                    "Chief Complaint": "Cough",
                    "HPI": "Cough is improving.",
                    "Red Flags": [],
                },
                "Medicine and its dosage": [],
                "Further tests if needed": [],
                "AI suggestion": {"suggestion": "Consult the doctor.", "referenced_documents": []},
                "Assessment": "This extra field is not part of the output contract.",
            }
        )


def test_current_status_requires_nested_fields() -> None:
    with pytest.raises(ValidationError):
        ClinicalNote.model_validate(
            {
                "Patient ID": "P-001",
                "Current status of the patient": {
                    "Overall description": "Stable",
                    "Chief Complaint": "Cough",
                    "Red Flags": [],
                },
                "Medicine and its dosage": [],
                "Further tests if needed": [],
                "AI suggestion": {"suggestion": "Consult the doctor.", "referenced_documents": []},
            }
        )


def test_extraction_retry_succeeds_on_second_attempt() -> None:
    settings = Settings(max_validation_retries=2)
    client = FakeClient(["{}", VALID_NOTE_JSON])

    result = extract_clinical_note(
        "synthetic transcript",
        client=client,
        searcher=FakeSearcher([]),
        settings=settings,
    )

    assert isinstance(result, ExtractionSuccess)
    assert result.attempts == 2
    assert client.calls == 2
    retry_user_messages = [
        message["content"] for message in client.seen_messages[1] if message["role"] == "user"
    ]
    assert any("Do not include prose" in content for content in retry_user_messages)


def test_extraction_accepts_json_embedded_in_prose_and_fence() -> None:
    client = FakeClient(
        [
            (
                "Here is the structured note:\n"
                "```json\n"
                f"{VALID_NOTE_JSON}\n"
                "```\n"
                "This follows the requested schema."
            )
        ]
    )

    result = extract_clinical_note(
        "The patient ID P-001 has improving cough.",
        client=client,
        searcher=FakeSearcher([]),
        settings=Settings(),
    )

    assert isinstance(result, ExtractionSuccess)
    assert result.attempts == 1
    assert result.note.patient_id == "P-001"


def test_extraction_replaces_hallucinated_clinical_fields_from_transcript() -> None:
    hallucinated_json = """
    {
      "Patient ID": "123456",
      "Current status of the patient": {
        "Overall description": "The patient presents with shortness of breath, cough, and fever.",
        "Chief Complaint": "Shortness of breath, cough, and fever",
        "HPI": "Symptoms worsened yesterday.",
        "Red Flags": ["shortness of breath"]
      },
      "Medicine and its dosage": [
        "Albuterol inhaler: 2 puffs every 4 hours",
        "Ibuprofen 600 mg every 8 hours"
      ],
      "Further tests if needed": ["Chest X-ray"],
      "AI suggestion": {
        "suggestion": "Consult the treating doctor.",
        "referenced_documents": []
      }
    }
    """
    transcript = (
        "The patient ID of 12304 is having kidney problem with severe pain in the stomach. "
        "The medication has been given of ibuphoin 500 milligram twice daily. "
        "The patient is having problem with urination and is supposed to be admitted. "
        "Further tests of blood are requested."
    )

    result = extract_clinical_note(
        transcript,
        client=FakeClient([hallucinated_json]),
        searcher=FakeSearcher([]),
        settings=Settings(),
    )

    assert isinstance(result, ExtractionSuccess)
    assert result.note.patient_id == "12304"
    assert result.note.current_status.chief_complaint == (
        "kidney problem with severe pain in the stomach"
    )
    assert "shortness of breath" not in result.note.current_status.red_flags
    assert result.note.medicine_and_dosage == ["ibuphoin 500 milligram twice daily"]
    assert result.note.further_tests == ["blood tests"]


def test_extraction_exhausts_retries() -> None:
    settings = Settings(max_validation_retries=1)
    client = FakeClient(["{}", "{}"])

    result = extract_clinical_note(
        "synthetic transcript",
        client=client,
        searcher=FakeSearcher([]),
        settings=settings,
    )

    assert isinstance(result, ExtractionFailure)
    assert result.attempts == 2
    assert len(result.errors) == 2


def test_extraction_uses_qdrant_context_for_ai_suggestion_references() -> None:
    settings = Settings(score_threshold=0.35)
    client = FakeClient([VALID_NOTE_JSON])
    result = extract_clinical_note(
        "synthetic transcript about asthma",
        client=client,
        searcher=FakeSearcher([_result(0.82)]),
        settings=settings,
    )

    assert isinstance(result, ExtractionSuccess)
    suggestion = result.note.ai_suggestion
    assert suggestion.referenced_documents[0].doc_id == "doc-1"
    assert suggestion.referenced_documents[0].title == "Synthetic asthma guidance"
    assert "Synthetic asthma guidance" in client.seen_messages[0][1]["content"]


def _result(score: float) -> SearchResult:
    return SearchResult(
        doc_id="doc-1",
        title="Synthetic asthma guidance",
        chunk_index=2,
        text="Check inhalation technique and arrange follow-up if control is poor.",
        language="en",
        score=score,
    )
