"""Asynchronous load test for the clinical voice note API."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

RESULTS_PATH = Path("eval/results/load_results.md")
QUESTIONS_PATH = Path("eval/questions.yaml")


async def run_load_test(args: argparse.Namespace) -> dict[str, Any]:
    """Run the configured load test and return summary metrics."""
    questions = _load_questions()
    url = args.url.rstrip("/")
    params = {"generate": "false"} if args.no_generate and args.endpoint == "/ask" else {}
    latencies: list[float] = []
    errors = 0
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=120) as client:
        for _ in range(args.warmup):
            await _send(client, url, args.endpoint, questions[0], params)

        started = time.perf_counter()

        async def worker(index: int) -> None:
            nonlocal errors
            question = questions[index % len(questions)]
            async with semaphore:
                request_started = time.perf_counter()
                ok = await _send(client, url, args.endpoint, question, params)
                latencies.append((time.perf_counter() - request_started) * 1000)
                if not ok:
                    errors += 1

        await asyncio.gather(*(worker(index) for index in range(args.requests)))
        duration = time.perf_counter() - started

    summary = {
        "url": url,
        "endpoint": args.endpoint,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "warmup": args.warmup,
        "no_generate": args.no_generate,
        "throughput": args.requests / duration if duration else 0.0,
        "p50": _percentile(latencies, 50),
        "p95": _percentile(latencies, 95),
        "p99": _percentile(latencies, 99),
        "error_rate": errors / args.requests if args.requests else 0.0,
    }
    _write_results(summary)
    return summary


def main() -> None:
    """Parse CLI args and run the load test."""
    args = _parse_args()
    summary = asyncio.run(run_load_test(args))
    for key, value in summary.items():
        print(f"{key}: {value}")


async def _send(
    client: httpx.AsyncClient,
    url: str,
    endpoint: str,
    question: str,
    params: dict[str, str],
) -> bool:
    try:
        if endpoint == "/ask":
            response = await client.post(
                f"{url}{endpoint}",
                params=params,
                json={"question": question},
            )
        else:
            response = await client.get(f"{url}{endpoint}", params=params)
    except httpx.HTTPError:
        return False
    return response.status_code < 500


def _load_questions() -> list[str]:
    raw = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = [
        str(item["question"])
        for item in raw
        if not item.get("out_of_corpus") and str(item["id"]).startswith("q_de")
    ]
    return questions or ["Welche Warnzeichen werden beschrieben?"]


def _write_results(summary: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Load Test Results",
        "",
        (
            "| url | endpoint | concurrency | requests | no generate | throughput | "
            "p50 | p95 | p99 | error rate |"
        ),
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary['url']} | {summary['endpoint']} | {summary['concurrency']} | "
            f"{summary['requests']} | {summary['no_generate']} | "
            f"{summary['throughput']:.2f} | {summary['p50']:.2f} | "
            f"{summary['p95']:.2f} | {summary['p99']:.2f} | "
            f"{summary['error_rate']:.3f} |"
        ),
    ]
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return float(statistics.quantiles(values, n=100)[percentile - 1])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--endpoint", default="/ask")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--no-generate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
