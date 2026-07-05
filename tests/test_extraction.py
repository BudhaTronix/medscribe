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


class FakeClient:
    """Fake LLM client returning configured outputs."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def generate(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        """Return the next configured output."""
        del messages, temperature
        output = self.outputs[self.calls]
        self.calls += 1
        return output


VALID_NOTE_JSON = """
{
  "language": "en",
  "chief_complaint": "Cough",
  "history_of_present_illness": "Synthetic patient with improving cough.",
  "medications": [{"name": "metformin", "dose": "500 mg", "frequency": "twice daily"}],
  "allergies": [],
  "assessment": "Improving respiratory infection.",
  "plan": ["Continue observation"],
  "red_flags": ["Worsening breathlessness"],
  "uncertainty_notes": ""
}
"""


def test_clinical_note_validation_happy_path() -> None:
    note = ClinicalNote.model_validate_json(VALID_NOTE_JSON)

    assert note.language == "en"
    assert note.medications[0].name == "metformin"


def test_clinical_note_validation_failure() -> None:
    with pytest.raises(ValidationError):
        ClinicalNote.model_validate({"language": "en", "chief_complaint": "No"})


def test_extraction_retry_succeeds_on_second_attempt() -> None:
    settings = Settings(max_validation_retries=2)
    client = FakeClient(["{}", VALID_NOTE_JSON])

    result = extract_clinical_note("synthetic transcript", client=client, settings=settings)

    assert isinstance(result, ExtractionSuccess)
    assert result.attempts == 2
    assert client.calls == 2


def test_extraction_exhausts_retries() -> None:
    settings = Settings(max_validation_retries=1)
    client = FakeClient(["{}", "{}"])

    result = extract_clinical_note("synthetic transcript", client=client, settings=settings)

    assert isinstance(result, ExtractionFailure)
    assert result.attempts == 2
    assert len(result.errors) == 2
