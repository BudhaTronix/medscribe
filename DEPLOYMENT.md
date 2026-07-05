# Deployment

```mermaid
flowchart LR
    U[User] --> N[nginx on 8080]
    N --> A1[api replica]
    N --> A2[api replica]
    N --> A3[api replica]
    A1 --> Q[Qdrant volume]
    A2 --> Q
    A3 --> Q
    A1 --> O[Ollama]
    A2 --> O
    A3 --> O
    G[Gradio UI on 7860] --> N
```

## Production Compose

Review `.env.prod.example`, or copy it to `.env.prod` for your own overrides, then run:

```bash
docker compose -f docker-compose.prod.yml up --build --scale api=3
```

The UI is published on `http://localhost:7860`, and the API is available through nginx on `http://localhost:8080`. The API service has no published host port, so traffic flows through nginx. nginx access logs include `upstream_addr`, which makes replica distribution visible when scaled.

## Stateless API Tier

The API tier is stateless. Request state lives only for the lifetime of a call. Durable state lives in the Qdrant named volume, and model artefacts live in container or host caches such as `HF_HOME=/models`. Because the API is stateless, horizontal scaling is done with replicas rather than increasing `UVICORN_WORKERS` inside one container.

## Scaling Story

ASR and embedding are CPU-bound in this demo, so additional API replicas can improve concurrent throughput. The LLM is the shared bottleneck when generation is enabled. The next scaling step would be a GPU node running vLLM, plus request queueing for async transcription and long-running note generation.

## Load Test Results

Run:

```bash
python scripts/load_test.py --url http://localhost:8080 --requests 20 --concurrency 1 --no-generate
python scripts/load_test.py --url http://localhost:8080 --requests 60 --concurrency 3 --no-generate
```

Generated results are written to `eval/results/load_results.md`.

## Security Notes

Containers run as a non-root `app` user. Secrets are not baked into images. The API service is internal-only in production compose, and nginx is the published entrypoint. Kubernetes secret manifests are examples only and contain no real secret values.

## Kubernetes

The manifests in `k8s/` are illustrative:

```bash
kubectl apply --dry-run=client -f k8s/
```

Qdrant is shown as a simple StatefulSet with a PVC. Production should use the official Helm chart or managed Qdrant. API pods use liveness and readiness probes, resource requests and limits, three replicas, and a rolling update strategy with `maxUnavailable: 0`.

## Next Steps

Use managed Qdrant or sharding for larger corpora, autoscale on queue depth, add canary deploys, and move generated note jobs to an async worker when audio duration grows.
