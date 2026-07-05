# Clinical Voice Note Assistant

Local-first demo for clinical documentation workflows: spoken dictation -> transcript -> structured
clinical note, plus grounded Q&A over a small synthetic bilingual (DE/EN) corpus. No paid APIs.
Synthetic data only — not a medical device, not medical advice, engineering demo.

## Architecture

```
Audio -> faster-whisper (ASR) -> transcript
transcript -> local LLM (Ollama, OpenAI-compatible) -> Pydantic ClinicalNote (schema-validated)
corpus (data/corpus/*.md) -> chunk -> sentence-transformers embed -> Qdrant
question -> embed -> Qdrant search -> score-threshold gate -> local LLM -> grounded answer + citations
```

- `app/asr/` — faster-whisper transcription (`transcriber.py`) and WER/CER eval helpers (`wer.py`).
- `app/ingestion/` — corpus chunking (`chunker.py`, ~500 token chunks / 80 token overlap) and
  embed+upsert pipeline into Qdrant (`pipeline.py`).
- `app/retrieval/search.py` — Qdrant vector search with score threshold.
- `app/llm/` — `client.py` (Ollama via OpenAI-compatible endpoint), `extraction.py` (schema-validated
  note extraction with retry-on-validation-error, returns typed `ExtractionFailure` instead of raising),
  `rag.py` (grounded Q&A; refuses below `SCORE_THRESHOLD` before ever calling the LLM).
- `app/api/main.py` — FastAPI app: `/transcribe`, `/notes/structure`, `/ask`, `/ingest`,
  `/health/live`, `/health/ready` (checks Qdrant, LLM, embedding model), `/metrics` (Prometheus).
- `app/ui/gradio_app.py` — Gradio Blocks UI (Dictation-to-Note, Ask the Guidelines, Evaluation tabs).
  Calls the API over HTTP if `API_BASE_URL` is set, otherwise runs the pipeline in-process.
  Auto-launches over HTTPS when `certs/cert.pem` + `certs/key.pem` exist (`make certs` /
  `scripts/generate_certs.sh` — uses `mkcert` for a browser-trusted cert if installed, otherwise
  falls back to a plain self-signed pair) — needed because browsers only allow microphone capture
  (`getUserMedia`) on a secure context, and only `localhost` gets a free pass over plain HTTP.
- `app/config.py` — `pydantic-settings` `Settings`, env-var driven, cached via `get_settings()`.
- `app/cli.py` — Typer CLI: `ingest`, `ask`, `transcribe`, `structure`.
- `eval/` — ASR (WER/CER) and retrieval (hit@k, MRR, refusal correctness) evaluation scripts,
  results written to `eval/results/*.md` and stitched into `README.md`.
- `data/corpus/` — 10 synthetic bilingual clinical markdown docs. `data/audio/scripts` +
  `data/audio/recordings` — dictation scripts and matching recordings for ASR eval (paired by filename stem).

## Runtime / environment

- Python 3.11 only, conda env named `medscribe`. Install with `pip install -e ".[dev]"`.
- External deps: Qdrant (vector store, `docker compose up qdrant`) and Ollama (local LLM,
  default model `mistral:7b`, OpenAI-compatible endpoint at `LLM_BASE_URL`).
- Embedding model: `BAAI/bge-m3` (multilingual, chosen for German retrieval quality — see
  `DESIGN_DECISIONS.md`).
- Config is env-var driven (see `.env.example`); key vars: `QDRANT_URL`, `QDRANT_COLLECTION`,
  `EMBEDDING_MODEL`, `LLM_BASE_URL`, `LLM_MODEL`, `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`,
  `TOP_K`, `SCORE_THRESHOLD`, `GRADIO_SERVER_PORT` (default 7860), `API_BASE_URL` (UI -> API).

## Commands

```bash
make test              # pytest -m "not integration"
make lint              # ruff check .
make ingest            # embed + upsert corpus into Qdrant
make ask QUESTION="…"  # CLI retrieval/RAG query
make api               # uvicorn app.api.main:app --reload
make ui                # python -m app.ui.gradio_app
make eval              # eval-asr + eval-retrieval, writes eval/results/*.md
```

## Deployment

- Local dev: `docker-compose.yml` (Qdrant + optional `app` profile).
- Production demo: `docker-compose.prod.yml` — Qdrant, 3x stateless `api` replicas behind `nginx`
  (nginx published on 8080, API has no host port), Gradio `ui` published directly on 7860 pointed
  at nginx via `API_BASE_URL`, optional `ollama` profile. See `DEPLOYMENT.md`.
- `docker/nginx.conf` currently reverse-proxies only the API (`/` -> `api_backend:8000`); it does
  not front the Gradio UI, and neither service terminates TLS — everything is plain HTTP.
- `k8s/` manifests are illustrative only (`kubectl apply --dry-run=client -f k8s/`).
- Containers run as non-root `app` user; secrets are not baked into images.

## Design decisions worth knowing (full detail in `DESIGN_DECISIONS.md`)

- RAG refuses to answer (no LLM call) when the best Qdrant score is below `SCORE_THRESHOLD` —
  a hallucination guard, not a bug.
- Extraction failures are typed (`ExtractionFailure` with raw output + validation errors) rather
  than exceptions, after `MAX_VALIDATION_RETRIES` retries.
- Everything runs local (ASR, embeddings, vector search, LLM) by design, mirroring hospital
  data-residency constraints — don't introduce external paid APIs into the core pipeline.

## Testing

- `pytest -m "not integration"` for fast tests (config, chunker, extraction, RAG, WER — see `tests/`).
- Integration tests are marked `integration` and need real Qdrant/Ollama.
- Ruff lint rules: `E, F, I, B, UP, ANN` (see `pyproject.toml`), line length 100.
