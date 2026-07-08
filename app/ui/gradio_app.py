"""Gradio user interface for the local demo."""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

from app.asr.transcriber import TranscriptionUnavailableError, WhisperTranscriber
from app.config import get_settings
from app.llm.extraction import ClinicalNote, ExtractionFailure, extract_clinical_note
from app.llm.rag import answer_question

DISCLAIMER = (
    "Synthetic data only, not a medical device, not medical advice, "
    "demo for engineering evaluation."
)
SAVED_NOTES_DIR = Path("data/notes")
NoteFormValues = tuple[str, str, str, str, str, str, str, str, str]
DICTATION_CSS = """
.dictation-compact {
  --accent: #f97316;
  --accent-hover: #fb923c;
  --panel: rgba(255, 255, 255, 0.045);
  --panel-border: rgba(255, 255, 255, 0.10);
  --compact-gap: 12px;
}
.dictation-compact .gap {
  gap: var(--compact-gap) !important;
}
.dictation-compact .form {
  border-radius: 8px !important;
}
.dictation-top-row {
  align-items: stretch !important;
  gap: var(--compact-gap) !important;
  margin-bottom: 12px !important;
}
.dictation-card,
.compact-note-fields {
  background: var(--panel) !important;
  border: 1px solid var(--panel-border) !important;
  border-radius: 8px !important;
  padding: 14px !important;
  margin: 0 0 12px 0 !important;
}
.dictation-card > div,
.compact-note-fields > div {
  background: transparent !important;
}
.compact-audio {
  min-height: 76px !important;
}
.compact-audio > div,
.compact-audio .wrap,
.compact-audio .container {
  min-height: 56px !important;
}
.compact-audio audio {
  max-height: 38px !important;
}
.compact-audio button,
.compact-action button,
.compact-save-action button {
  min-height: 42px !important;
  border-radius: 8px !important;
}
.compact-action {
  align-self: stretch !important;
  justify-content: end !important;
}
.compact-action button,
.compact-save-action button {
  width: 100%;
}
.compact-action button,
.compact-save-action button {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #111827 !important;
  font-weight: 700 !important;
}
.compact-action button:hover,
.compact-save-action button:hover {
  background: var(--accent-hover) !important;
  border-color: var(--accent-hover) !important;
}
.compact-transcript textarea {
  min-height: 158px !important;
  max-height: 180px !important;
}
.compact-section-title h3 {
  margin: 0 0 10px 0 !important;
  font-size: 1rem !important;
}
.compact-note-fields textarea {
  min-height: 70px !important;
}
.compact-note-fields .field-short textarea {
  min-height: 44px !important;
  max-height: 56px !important;
}
.compact-note-fields .field-medium textarea {
  min-height: 82px !important;
  max-height: 98px !important;
}
.compact-note-fields .field-tall textarea {
  min-height: 112px !important;
  max-height: 132px !important;
}
.compact-save-row {
  align-items: center !important;
  justify-content: flex-end !important;
  margin-top: 4px !important;
}
.compact-save-status {
  min-height: 28px !important;
}
.compact-latency {
  margin-top: 4px !important;
}
.compact-latency > label,
.compact-latency pre {
  font-size: 0.82rem !important;
}
.compact-disclaimer p {
  font-size: 0.82rem !important;
  opacity: 0.75;
  margin-top: 8px !important;
}
@media (max-width: 780px) {
  .dictation-top-row,
  .compact-fields-row,
  .compact-save-row {
    flex-direction: column !important;
  }
  .dictation-card,
  .compact-note-fields {
    padding: 12px !important;
  }
}
"""


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks UI."""
    with gr.Blocks(title="Clinical Voice Note Assistant") as demo:
        gr.Markdown("# Clinical Voice Note Assistant")
        with gr.Tabs():
            with gr.Tab("Dictation to Note", elem_classes=["dictation-compact"]):
                with gr.Row(elem_classes=["dictation-top-row"]):
                    audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Audio",
                        min_width=240,
                        scale=6,
                        elem_classes=["compact-audio"],
                    )
                    language = gr.Dropdown(
                        choices=[("Auto", ""), ("German", "de"), ("English", "en")],
                        value="",
                        label="Language",
                        min_width=160,
                        scale=3,
                    )
                    with gr.Column(
                        min_width=190,
                        scale=4,
                        elem_classes=["compact-action"],
                    ):
                        run_note = gr.Button("Transcribe and Structure", variant="primary")

                with gr.Group(elem_classes=["dictation-card"]):
                    transcript = gr.Textbox(
                        label="Transcript",
                        lines=7,
                        elem_classes=["compact-transcript"],
                    )
                with gr.Group(elem_classes=["compact-note-fields"]):
                    gr.Markdown(
                        "### Structured Note Fields",
                        elem_classes=["compact-section-title"],
                    )
                    with gr.Row(elem_classes=["compact-fields-row"]):
                        with gr.Column(scale=1, min_width=320):
                            patient_id = gr.Textbox(
                                label="Patient ID",
                                elem_classes=["field-short"],
                            )
                            overall_description = gr.Textbox(
                                label="Overall description",
                                lines=2,
                                elem_classes=["field-medium"],
                            )
                            chief_complaint = gr.Textbox(
                                label="Chief Complaint",
                                elem_classes=["field-short"],
                            )
                            hpi = gr.Textbox(
                                label="HPI",
                                lines=3,
                                elem_classes=["field-tall"],
                            )
                        with gr.Column(scale=1, min_width=320):
                            red_flags = gr.Textbox(
                                label="Red Flags",
                                lines=2,
                                elem_classes=["field-medium"],
                            )
                            medicine_and_dosage = gr.Textbox(
                                label="Medicine and its dosage",
                                lines=2,
                                elem_classes=["field-medium"],
                            )
                            further_tests = gr.Textbox(
                                label="Further tests if needed",
                                lines=2,
                                elem_classes=["field-medium"],
                            )
                            ai_suggestion = gr.Textbox(
                                label="AI suggestion",
                                lines=2,
                                elem_classes=["field-medium"],
                            )
                            referenced_documents = gr.Textbox(
                                label="Referenced documents",
                                lines=2,
                                elem_classes=["field-medium"],
                            )
                    with gr.Row(elem_classes=["compact-save-row"]):
                        save_status = gr.Markdown(
                            scale=3,
                            elem_classes=["compact-save-status"],
                        )
                        with gr.Column(
                            scale=1,
                            min_width=180,
                            elem_classes=["compact-save-action"],
                        ):
                            save_note = gr.Button("Save Note", variant="primary")
                        saved_file = gr.File(label="Saved text file", visible=False)
                latency = gr.JSON(label="Latency", elem_classes=["compact-latency"])
                gr.Markdown(DISCLAIMER, elem_classes=["compact-disclaimer"])
                run_note.click(
                    fn=_dictation_to_note,
                    inputs=[audio, language],
                    outputs=[
                        transcript,
                        patient_id,
                        overall_description,
                        chief_complaint,
                        hpi,
                        red_flags,
                        medicine_and_dosage,
                        further_tests,
                        ai_suggestion,
                        referenced_documents,
                        latency,
                    ],
                )
                save_note.click(
                    fn=_save_structured_note,
                    inputs=[
                        transcript,
                        patient_id,
                        overall_description,
                        chief_complaint,
                        hpi,
                        red_flags,
                        medicine_and_dosage,
                        further_tests,
                        ai_suggestion,
                        referenced_documents,
                    ],
                    outputs=[save_status, saved_file],
                )
            with gr.Tab("Ask the Guidelines"):
                question = gr.Textbox(label="Question", lines=3)
                ask_button = gr.Button("Ask", variant="primary")
                answer = gr.Markdown(label="Answer")
                citations = gr.JSON(label="Citations")
                refusal = gr.Markdown(label="Refusal State")
                ask_latency = gr.JSON(label="Latency")
                gr.Markdown(DISCLAIMER)
                ask_button.click(
                    fn=_ask_guidelines,
                    inputs=question,
                    outputs=[answer, citations, refusal, ask_latency],
                )
            with gr.Tab("Evaluation"):
                refresh = gr.Button("Refresh Results")
                results = gr.Markdown(value=_evaluation_markdown())
                gr.Markdown(DISCLAIMER)
                refresh.click(fn=_evaluation_markdown, outputs=results)
    return demo


def _dictation_to_note(
    audio_path: str | None,
    language: str,
) -> tuple[str, str, str, str, str, str, str, str, str, str, dict[str, float]]:
    if not audio_path:
        return ("", "", "", "", "", "", "", "", "", "", {})
    if _api_base_url():
        return _dictation_to_note_via_api(audio_path, language)
    forced_language = language or None
    timings: dict[str, float] = {}
    try:
        transcription = WhisperTranscriber(get_settings()).transcribe(
            audio_path,
            language=forced_language,
        )
    except TranscriptionUnavailableError as exc:
        return ("", "", str(exc), "", "", "", "", "", "", "", timings)
    timings.update(transcription.timings_ms)
    extraction = extract_clinical_note(transcription.text, settings=get_settings())
    timings.update(extraction.timings_ms)
    if isinstance(extraction, ExtractionFailure):
        return (
            transcription.text,
            "",
            "Extraction failed validation.",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            timings,
        )
    return (transcription.text, *_note_form_values(extraction.note), timings)


def _ask_guidelines(question: str) -> tuple[str, list[dict[str, Any]], str, dict[str, float]]:
    if _api_base_url():
        return _ask_guidelines_via_api(question)
    response = answer_question(question, settings=get_settings())
    refusal = "Refused: below retrieval threshold." if response.refused else "Answered from corpus."
    citations = [citation.model_dump() for citation in response.citations]
    return response.answer, citations, refusal, response.timings_ms


def _dictation_to_note_via_api(
    audio_path: str,
    language: str,
) -> tuple[str, str, str, str, str, str, str, str, str, str, dict[str, float]]:
    base_url = _api_base_url()
    if base_url is None:
        return ("", "", "API base URL is not configured.", "", "", "", "", "", "", "", {})
    with Path(audio_path).open("rb") as handle:
        files = {"audio": (Path(audio_path).name, handle)}
        data = {"language": language} if language else {}
        transcription = httpx.post(f"{base_url}/transcribe", data=data, files=files, timeout=120)
    if transcription.status_code >= 400:
        return ("", "", transcription.text, "", "", "", "", "", "", "", {})
    transcription_payload = transcription.json()
    transcript = str(transcription_payload.get("text", ""))
    response = httpx.post(
        f"{base_url}/notes/structure",
        data={"transcript": transcript},
        timeout=120,
    )
    if response.status_code >= 400:
        return (transcript, "", response.text, "", "", "", "", "", "", "", {})
    payload = response.json()
    timings = dict(transcription_payload.get("timings_ms", {}))
    if "Patient ID" in payload:
        return (transcript, *_note_form_values(payload), timings)
    else:
        timings.update(payload.get("timings_ms", {}))
        return (
            transcript,
            "",
            "Extraction failed validation.",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            timings,
        )


def _note_form_values(note: ClinicalNote | dict[str, Any]) -> NoteFormValues:
    payload = note.model_dump() if isinstance(note, ClinicalNote) else note
    status = payload.get("Current status of the patient", {})
    if not isinstance(status, dict):
        status = {}
    return (
        str(payload.get("Patient ID", "")),
        str(status.get("Overall description", "")),
        str(status.get("Chief Complaint", "")),
        str(status.get("HPI", "")),
        _plain_list(status.get("Red Flags", [])),
        _plain_list(payload.get("Medicine and its dosage", [])),
        _plain_list(payload.get("Further tests if needed", [])),
        _ai_suggestion_text(payload.get("AI suggestion", {})),
        _references_plain_text(payload.get("AI suggestion", {})),
    )


def _save_structured_note(
    transcript: str,
    patient_id: str,
    overall_description: str,
    chief_complaint: str,
    hpi: str,
    red_flags: str,
    medicine_and_dosage: str,
    further_tests: str,
    ai_suggestion: str,
    referenced_documents: str,
) -> tuple[str, str | None]:
    if not any(
        value.strip()
        for value in (
            transcript,
            patient_id,
            overall_description,
            chief_complaint,
            hpi,
            red_flags,
            medicine_and_dosage,
            further_tests,
            ai_suggestion,
            referenced_documents,
        )
    ):
        return "Nothing to save yet.", None

    SAVED_NOTES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_patient_id = _safe_filename_part(patient_id or "unknown")
    path = SAVED_NOTES_DIR / f"note_{safe_patient_id}_{timestamp}.txt"
    path.write_text(
        _structured_note_text(
            transcript=transcript,
            patient_id=patient_id,
            overall_description=overall_description,
            chief_complaint=chief_complaint,
            hpi=hpi,
            red_flags=red_flags,
            medicine_and_dosage=medicine_and_dosage,
            further_tests=further_tests,
            ai_suggestion=ai_suggestion,
            referenced_documents=referenced_documents,
        ),
        encoding="utf-8",
    )
    return f"Saved to `{path}`.", str(path)


def _structured_note_text(
    *,
    transcript: str,
    patient_id: str,
    overall_description: str,
    chief_complaint: str,
    hpi: str,
    red_flags: str,
    medicine_and_dosage: str,
    further_tests: str,
    ai_suggestion: str,
    referenced_documents: str,
) -> str:
    return (
        f"Patient ID: {patient_id.strip()}\n\n"
        "Current status of the patient\n"
        f"Overall description: {overall_description.strip()}\n"
        f"Chief Complaint: {chief_complaint.strip()}\n"
        f"HPI: {hpi.strip()}\n"
        f"Red Flags:\n{_bulleted_text(red_flags)}\n\n"
        f"Medicine and its dosage:\n{_bulleted_text(medicine_and_dosage)}\n\n"
        f"Further tests if needed:\n{_bulleted_text(further_tests)}\n\n"
        f"AI suggestion:\n{ai_suggestion.strip()}\n\n"
        f"Referenced documents:\n{_bulleted_text(referenced_documents)}\n\n"
        f"Transcript:\n{transcript.strip()}\n"
    )


def _bulleted_text(value: str) -> str:
    lines = [line.strip(" -") for line in value.splitlines() if line.strip(" -")]
    return "\n".join(f"- {line}" for line in lines) if lines else "- None"


def _safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip() or "unknown")[:80]


def _format_note_card(note: ClinicalNote | dict[str, Any]) -> str:
    payload = note.model_dump() if isinstance(note, ClinicalNote) else note
    return (
        f"### Patient {payload.get('Patient ID', 'unknown')}\n\n"
        f"**Current status:**\n{_status_section(payload.get('Current status of the patient', {}))}"
        "\n\n"
        f"**Medicine and dosage:**\n{_markdown_list(payload.get('Medicine and its dosage', []))}"
        f"\n\n**Further tests:**\n{_markdown_list(payload.get('Further tests if needed', []))}"
        f"\n\n**AI suggestion:** {_ai_suggestion_text(payload.get('AI suggestion', {}))}"
        f"\n\n**Referenced documents:**\n"
        f"{_reference_list(payload.get('AI suggestion', {}))}"
    )


def _status_section(value: Any) -> str:
    if not isinstance(value, dict):
        text = str(value).strip()
        return text or "None mentioned"
    red_flags = value.get("Red Flags", [])
    return (
        f"Overall description: {value.get('Overall description', '')}\n\n"
        f"Chief Complaint: {value.get('Chief Complaint', '')}\n\n"
        f"HPI: {value.get('HPI', '')}\n\n"
        f"Red Flags:\n{_markdown_list(red_flags)}"
    )


def _markdown_list(items: Any) -> str:
    if isinstance(items, list):
        return "\n".join(f"- {item}" for item in items) or "- None mentioned"
    text = str(items).strip()
    return f"- {text}" if text else "- None mentioned"


def _plain_list(items: Any) -> str:
    if isinstance(items, list):
        return "\n".join(str(item) for item in items)
    return str(items).strip()


def _ai_suggestion_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("suggestion", "")).strip() or "None mentioned"
    return str(value).strip() or "None mentioned"


def _references_plain_text(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    references = value.get("referenced_documents", [])
    if not isinstance(references, list):
        return ""
    lines = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        doc_id = reference.get("doc_id", "")
        title = reference.get("title", "")
        chunk_index = reference.get("chunk_index", "")
        lines.append(f"{doc_id} | {title} | chunk {chunk_index}")
    return "\n".join(lines)


def _reference_list(value: Any) -> str:
    if not isinstance(value, dict):
        return "- None"
    references = value.get("referenced_documents", [])
    if not isinstance(references, list) or not references:
        return "- None"
    lines = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        doc_id = reference.get("doc_id", "")
        title = reference.get("title", "")
        chunk_index = reference.get("chunk_index", "")
        lines.append(f"- {doc_id} | {title} | chunk {chunk_index}")
    return "\n".join(lines) or "- None"


def _ask_guidelines_via_api(
    question: str,
) -> tuple[str, list[dict[str, Any]], str, dict[str, float]]:
    base_url = _api_base_url()
    if base_url is None:
        return "API base URL is not configured.", [], "Refused.", {}
    response = httpx.post(f"{base_url}/ask", json={"question": question}, timeout=120)
    if response.status_code >= 400:
        return response.text, [], "Refused.", {}
    payload = response.json()
    refusal = (
        "Refused: below retrieval threshold."
        if payload.get("refused")
        else "Answered from corpus."
    )
    return (
        str(payload.get("answer", "")),
        list(payload.get("citations", [])),
        refusal,
        dict(payload.get("timings_ms", {})),
    )


def _evaluation_markdown() -> str:
    paths = sorted(Path("eval/results").glob("*_results.md"))
    if not paths:
        return "Run `make eval` to generate result tables."
    return "\n\n".join(path.read_text(encoding="utf-8") for path in paths)


def _api_base_url() -> str | None:
    value = os.getenv("API_BASE_URL", "").rstrip("/")
    return value or None


def main() -> None:
    """Launch the Gradio UI."""
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    launch_kwargs: dict[str, Any] = {
        "server_name": "0.0.0.0",
        "server_port": port,
        "prevent_thread_lock": True,
        "css": DICTATION_CSS,
    }
    certfile = os.getenv("GRADIO_SSL_CERTFILE", "certs/cert.pem")
    keyfile = os.getenv("GRADIO_SSL_KEYFILE", "certs/key.pem")
    if Path(certfile).is_file() and Path(keyfile).is_file():
        # Browsers only expose getUserMedia (microphone) on a secure context, so
        # anything reached over a network address (not localhost) needs TLS.
        launch_kwargs["ssl_certfile"] = certfile
        launch_kwargs["ssl_keyfile"] = keyfile
        launch_kwargs["ssl_verify"] = False
    build_app().launch(**launch_kwargs)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
