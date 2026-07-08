"""Gradio user interface for the local demo."""

from __future__ import annotations

import os
import time
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


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks UI."""
    with gr.Blocks(title="Clinical Voice Note Assistant") as demo:
        gr.Markdown("# Clinical Voice Note Assistant")
        with gr.Tabs():
            with gr.Tab("Dictation to Note"):
                audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio",
                )
                language = gr.Dropdown(
                    choices=[("Auto", ""), ("German", "de"), ("English", "en")],
                    value="",
                    label="Language",
                )
                run_note = gr.Button("Transcribe and Structure", variant="primary")
                transcript = gr.Textbox(label="Transcript", lines=8)
                note_card = gr.Markdown(label="Structured Note")
                raw_json = gr.JSON(label="Raw JSON")
                latency = gr.JSON(label="Latency")
                gr.Markdown(DISCLAIMER)
                run_note.click(
                    fn=_dictation_to_note,
                    inputs=[audio, language],
                    outputs=[transcript, note_card, raw_json, latency],
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
) -> tuple[str, str, dict[str, Any], dict[str, float]]:
    if not audio_path:
        return "", "No audio provided.", {}, {}
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
        return "", str(exc), {"error": str(exc)}, timings
    timings.update(transcription.timings_ms)
    extraction = extract_clinical_note(transcription.text, settings=get_settings())
    timings.update(extraction.timings_ms)
    if isinstance(extraction, ExtractionFailure):
        payload = extraction.model_dump()
        card = "Extraction failed validation."
    else:
        payload = extraction.note.model_dump()
        card = _format_note_card(extraction.note)
    return transcription.text, card, payload, timings


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
) -> tuple[str, str, dict[str, Any], dict[str, float]]:
    base_url = _api_base_url()
    if base_url is None:
        return "", "API base URL is not configured.", {}, {}
    with Path(audio_path).open("rb") as handle:
        files = {"audio": (Path(audio_path).name, handle)}
        data = {"language": language} if language else {}
        transcription = httpx.post(f"{base_url}/transcribe", data=data, files=files, timeout=120)
    if transcription.status_code >= 400:
        return "", transcription.text, {"error": transcription.text}, {}
    transcription_payload = transcription.json()
    transcript = str(transcription_payload.get("text", ""))
    response = httpx.post(
        f"{base_url}/notes/structure",
        data={"transcript": transcript},
        timeout=120,
    )
    if response.status_code >= 400:
        return "", response.text, {"error": response.text}, {}
    payload = response.json()
    timings = dict(transcription_payload.get("timings_ms", {}))
    if "Patient ID" in payload:
        card = _format_note_card(payload)
    else:
        card = "Extraction failed validation."
        timings.update(payload.get("timings_ms", {}))
    return transcript, card, payload, timings


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


def _ai_suggestion_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("suggestion", "")).strip() or "None mentioned"
    return str(value).strip() or "None mentioned"


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
