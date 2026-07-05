FROM python:3.11-slim AS builder

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
COPY eval ./eval
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HF_HOME=/models
ENV UVICORN_WORKERS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY . .
RUN mkdir -p /models && chown -R app:app /app /models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/live || exit 1

CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-1}"]
