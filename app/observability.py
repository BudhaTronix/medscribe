"""Request logging and Prometheus metrics."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

REQUEST_COUNT = Counter(
    "clinical_voice_note_requests_total",
    "Total HTTP requests by path and status.",
    ["path", "status"],
)
REQUEST_DURATION = Histogram(
    "clinical_voice_note_request_duration_ms",
    "HTTP request duration in milliseconds.",
    ["path"],
)
STAGE_DURATION = Histogram(
    "clinical_voice_note_stage_duration_ms",
    "Pipeline stage duration in milliseconds.",
    ["stage"],
)
READINESS_GAUGE = Gauge("clinical_voice_note_ready", "Readiness status, 1 ready and 0 not ready.")
EXTRACTION_RETRY_COUNT = Counter(
    "clinical_voice_note_extraction_validation_retries_total",
    "Total extraction validation retries.",
)

logger = logging.getLogger("clinical_voice_note")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach request IDs, log requests, and update request metrics."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process a request with structured logging."""
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            duration_ms = _elapsed_ms(started)
            _log_request(request_id, request.url.path, status_code, duration_ms)
            REQUEST_COUNT.labels(path=request.url.path, status=str(status_code)).inc()
            REQUEST_DURATION.labels(path=request.url.path).observe(duration_ms)
            raise
        duration_ms = _elapsed_ms(started)
        response.headers["X-Request-ID"] = request_id
        REQUEST_COUNT.labels(path=request.url.path, status=str(status_code)).inc()
        REQUEST_DURATION.labels(path=request.url.path).observe(duration_ms)
        _log_request(request_id, request.url.path, status_code, duration_ms)
        return response


def metrics_response() -> Response:
    """Return Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def observe_stage_timings(timings_ms: dict[str, float]) -> None:
    """Record stage timings in the Prometheus histogram."""
    for stage, duration_ms in timings_ms.items():
        if stage in {"transcribe", "embed", "retrieve", "generate"}:
            STAGE_DURATION.labels(stage=stage).observe(duration_ms)


def _log_request(request_id: str, path: str, status: int, duration_ms: float) -> None:
    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "path": path,
                "status": status,
                "duration_ms": duration_ms,
                "stage_timings": {},
            }
        )
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
