"""Gradio user interface for the local demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from app.asr.transcriber import TranscriptionUnavailableError, WhisperTranscriber
from app.config import get_settings
from app.llm.extraction import ExtractionFailure, extract_clinical_note
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
    payload = extraction.model_dump()
    if isinstance(extraction, ExtractionFailure):
        card = "Extraction failed validation."
    else:
        note = extraction.note
        card = (
            f"### {note.chief_complaint}\n\n"
            f"**Assessment:** {note.assessment}\n\n"
            f"**Plan:**\n" + "\n".join(f"- {item}" for item in note.plan)
        )
    return transcription.text, card, payload, timings


def _ask_guidelines(question: str) -> tuple[str, list[dict[str, Any]], str, dict[str, float]]:
    response = answer_question(question, settings=get_settings())
    refusal = "Refused: below retrieval threshold." if response.refused else "Answered from corpus."
    citations = [citation.model_dump() for citation in response.citations]
    return response.answer, citations, refusal, response.timings_ms


def _evaluation_markdown() -> str:
    paths = sorted(Path("eval/results").glob("*_results.md"))
    if not paths:
        return "Run `make eval` to generate result tables."
    return "\n\n".join(path.read_text(encoding="utf-8") for path in paths)


def main() -> None:
    """Launch the Gradio UI."""
    build_app().launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
