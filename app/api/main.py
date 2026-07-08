"""FastAPI application for the clinical voice note assistant."""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.asr.transcriber import TranscriptionUnavailableError, WhisperTranscriber
from app.config import get_settings
from app.ingestion.pipeline import (
    IngestionUnavailableError,
    SentenceTransformerEmbedder,
    ingest_corpus,
)
from app.llm.client import OllamaOpenAIClient
from app.llm.extraction import ExtractionFailure, extract_clinical_note
from app.llm.rag import RagResponse, answer_question
from app.observability import (
    READINESS_GAUGE,
    RequestIdMiddleware,
    metrics_response,
    observe_stage_timings,
)


class AskRequest(BaseModel):
    """Question request body."""

    question: str = Field(min_length=1)


class ReadyStatus(BaseModel):
    """Readiness body for dependencies."""

    qdrant: bool
    llm: bool
    embedding_model_loaded: bool
    detail: dict[str, str]


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Load the embedding model at startup for readiness reporting."""
    settings = get_settings()
    app.state.embedding_model_loaded = False
    try:
        SentenceTransformerEmbedder(settings.embedding_model).dimension()
        app.state.embedding_model_loaded = True
    except IngestionUnavailableError:
        app.state.embedding_model_loaded = False
    yield


app = FastAPI(title="Clinical Voice Note Assistant", lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)


@app.get("/health/live")
def live() -> dict[str, str]:
    """Return liveness while the API process is serving."""
    return {"status": "live"}


@app.get("/health/ready")
def ready() -> JSONResponse:
    """Return readiness only when Qdrant, the LLM, and embeddings are ready."""
    status = _ready_status()
    ready_value = status.qdrant and status.llm and status.embedding_model_loaded
    READINESS_GAUGE.set(1 if ready_value else 0)
    return JSONResponse(status_code=200 if ready_value else 503, content=status.model_dump())


@app.get("/metrics")
def metrics() -> Any:
    """Expose Prometheus metrics."""
    return metrics_response()


@app.post("/ingest")
def ingest() -> dict[str, Any]:
    """Run the corpus ingestion pipeline."""
    try:
        report = ingest_corpus(get_settings())
    except IngestionUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    observe_stage_timings(report.timings_ms)
    return report.__dict__


@app.post("/transcribe")
async def transcribe(
    audio: Annotated[UploadFile, File()],
    language: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Transcribe uploaded audio."""
    if language not in {None, "de", "en"}:
        raise HTTPException(status_code=422, detail="language must be de or en")
    temp_path = await _save_upload(audio)
    try:
        result = WhisperTranscriber(get_settings()).transcribe(temp_path, language=language)
    except TranscriptionUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
    observe_stage_timings(result.timings_ms)
    return _asdict(result)


@app.post("/notes/structure")
async def structure_note(
    transcript: Annotated[str | None, Form()] = None,
    audio: Annotated[UploadFile | None, File()] = None,
    language: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Structure a note from transcript text or uploaded audio."""
    timings: dict[str, float] = {}
    transcript_text = transcript or ""
    if audio is not None:
        temp_path = await _save_upload(audio)
        try:
            transcription = WhisperTranscriber(get_settings()).transcribe(
                temp_path,
                language=language,
            )
        except TranscriptionUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            temp_path.unlink(missing_ok=True)
        transcript_text = transcription.text
        timings.update(transcription.timings_ms)
    if not transcript_text.strip():
        raise HTTPException(status_code=422, detail="transcript text or audio file is required")
    extraction = extract_clinical_note(transcript_text, settings=get_settings())
    timings.update(extraction.timings_ms)
    observe_stage_timings(timings)
    if isinstance(extraction, ExtractionFailure):
        body = extraction.model_dump()
        body["transcript"] = transcript_text
        body["timings_ms"] = timings
        return body
    return extraction.note.model_dump()


@app.post("/ask")
def ask(
    request: AskRequest,
    generate: Annotated[bool, Query()] = True,
) -> RagResponse:
    """Answer a question from grounded corpus context."""
    try:
        response = answer_question(request.question, settings=get_settings(), generate=generate)
    except IngestionUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    observe_stage_timings(response.timings_ms)
    return response


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(await upload.read())
        return Path(handle.name)


def _ready_status() -> ReadyStatus:
    settings = get_settings()
    detail: dict[str, str] = {}
    qdrant_ready = False
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url)
        exists = client.collection_exists(settings.qdrant_collection)
        count = client.count(settings.qdrant_collection, exact=True).count if exists else 0
        qdrant_ready = exists and count > 0
        detail["qdrant"] = f"collection exists with {count} points"
    except Exception as exc:
        detail["qdrant"] = str(exc)

    llm_status = OllamaOpenAIClient(settings).model_status()
    detail["llm"] = llm_status.detail

    embedding_loaded = bool(getattr(app.state, "embedding_model_loaded", False))
    detail["embedding_model_loaded"] = str(embedding_loaded)
    return ReadyStatus(
        qdrant=qdrant_ready,
        llm=llm_status.reachable,
        embedding_model_loaded=embedding_loaded,
        detail=detail,
    )


def _asdict(value: Any) -> Any:
    if isinstance(value, list):
        return [_asdict(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _asdict(item) for key, item in value.__dict__.items()}
    return value
