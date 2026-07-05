# Clinical Voice Note Assistant

Clinical Voice Note Assistant is a local engineering demo for clinical documentation workflows. It transcribes spoken dictation, structures notes, and retrieves grounded reference snippets from a small bilingual corpus, all without paid APIs.

Synthetic data only, not a medical device, not medical advice, demo for engineering evaluation.

## Phase Status

Phase 1 provides the scaffold, configuration, synthetic reference corpus, chunking, ingestion, Qdrant retrieval, and CLI retrieval commands. Later phases add ASR, schema extraction, grounded generation, API endpoints, Gradio UI, evaluation, deployment, and observability.

## Local Checks

Use the `medscribe` conda environment:

```bash
conda run -n medscribe python -m pytest -m "not integration"
conda run -n medscribe python -m ruff check .
```

For retrieval, start Qdrant first, then run:

```bash
conda run -n medscribe make ingest
conda run -n medscribe make ask QUESTION="Welche erste Therapie ist bei Hypertonie beschrieben?"
```
