#!/usr/bin/env bash
# One-shot local startup: env check, deps, Qdrant, certs, ingest, API, UI.
# Mirrors the manual steps in README.md "Step By Step Execution".
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-medscribe}"
API_PORT="${API_PORT:-8000}"
GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:11434/v1}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

log() { printf '\n==> %s\n' "$1"; }

run() {
  conda run -n "$CONDA_ENV" --no-capture-output "$@"
}

log "Locating conda"
if ! command -v conda >/dev/null 2>&1; then
  # `conda init` only wires PATH/shell functions into the *interactive* section of
  # ~/.bashrc, so a script run non-interactively (e.g. via `make`) won't see it even
  # though the same conda works fine in a normal terminal. Find and source the hook
  # directly instead of assuming `conda` is already on PATH.
  for base in "${CONDA_EXE_BASE:-}" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
    "$HOME/mambaforge" /opt/conda /opt/miniconda3 /opt/anaconda3; do
    if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
      source "$base/etc/profile.d/conda.sh"
      break
    fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH and no install was found in the usual locations." >&2
  echo "Set CONDA_EXE_BASE=/path/to/your/conda (the install root, not the binary) and retry." >&2
  exit 1
fi

log "Checking conda environment '$CONDA_ENV'"
if ! conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
  echo "Conda environment '$CONDA_ENV' not found. Create it first, e.g.:" >&2
  echo "  conda create -n $CONDA_ENV python=3.11" >&2
  exit 1
fi

if [ "$SKIP_INSTALL" != "1" ]; then
  log "Installing/updating dependencies (set SKIP_INSTALL=1 to skip)"
  run python -m pip install -e ".[dev]"
fi

log "Starting Qdrant"
docker compose up -d qdrant
echo -n "Waiting for Qdrant at $QDRANT_URL"
for _ in $(seq 1 30); do
  if curl -fsS "$QDRANT_URL/readyz" >/dev/null 2>&1; then
    echo " - ready"
    break
  fi
  echo -n "."
  sleep 2
done
if ! curl -fsS "$QDRANT_URL/readyz" >/dev/null 2>&1; then
  echo "Qdrant did not become ready in time." >&2
  exit 1
fi

log "Checking TLS certificate for the UI (needed for microphone access off localhost)"
if [ ! -f certs/cert.pem ] || [ ! -f certs/key.pem ]; then
  make certs
else
  echo "certs/cert.pem and certs/key.pem already exist, skipping."
fi

log "Ingesting the synthetic corpus"
run python -m app.cli ingest

log "Checking Ollama at $LLM_BASE_URL"
if curl -fsS "${LLM_BASE_URL%/v1}/api/tags" >/dev/null 2>&1; then
  echo "Ollama is reachable."
else
  echo "Warning: Ollama not reachable at $LLM_BASE_URL. Note structuring and RAG answers need it;" >&2
  echo "retrieval-only search still works. Start it with 'ollama serve' and pull a model, e.g.:" >&2
  echo "  ollama pull mistral:7b" >&2
fi

API_LOG="$(mktemp)"
log "Starting the API on port $API_PORT (log: $API_LOG)"
run python -m uvicorn app.api.main:app --host 0.0.0.0 --port "$API_PORT" >"$API_LOG" 2>&1 &
API_PID=$!

cleanup() {
  log "Stopping API (pid $API_PID)"
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo -n "Waiting for API at http://localhost:$API_PORT/health/live"
for _ in $(seq 1 30); do
  if curl -fsS "http://localhost:$API_PORT/health/live" >/dev/null 2>&1; then
    echo " - ready"
    break
  fi
  echo -n "."
  sleep 1
done
if ! curl -fsS "http://localhost:$API_PORT/health/live" >/dev/null 2>&1; then
  echo "API did not become ready in time, see $API_LOG" >&2
  exit 1
fi

log "Starting the UI on port $GRADIO_SERVER_PORT (Ctrl+C stops both API and UI)"
scheme="http"
[ -f certs/cert.pem ] && [ -f certs/key.pem ] && scheme="https"
echo "UI:  $scheme://localhost:$GRADIO_SERVER_PORT"
echo "API: http://localhost:$API_PORT (docs at /docs)"
API_BASE_URL="http://localhost:$API_PORT" GRADIO_SERVER_PORT="$GRADIO_SERVER_PORT" \
  run python -m app.ui.gradio_app
