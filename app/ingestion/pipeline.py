"""Corpus ingestion pipeline for Qdrant vector search."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.ingestion.chunker import Chunk, chunk_text

CORPUS_DIR = Path("data/corpus")
POINT_NAMESPACE = uuid.UUID("6f1dd0d5-6165-4b5a-b079-2dfc996e7514")


class IngestionUnavailableError(RuntimeError):
    """Raised when an external ingestion dependency is unreachable."""


@dataclass(frozen=True)
class CorpusDocument:
    """A synthetic reference document loaded from disk."""

    doc_id: str
    title: str
    language: str
    text: str
    path: Path


@dataclass(frozen=True)
class PreparedChunk:
    """A chunk with its source document metadata."""

    doc_id: str
    title: str
    language: str
    chunk: Chunk


@dataclass(frozen=True)
class IngestionReport:
    """Counts and timings from a corpus ingestion run."""

    documents: int
    chunks: int
    collection: str
    timings_ms: dict[str, float]


class SentenceTransformerEmbedder:
    """Lazy wrapper around sentence-transformers with normalised embeddings."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def load(self) -> Any:
        """Load and cache the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                msg = "sentence-transformers is not installed in this environment"
                raise IngestionUnavailableError(msg) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        model = self.load()
        dimension = model.get_sentence_embedding_dimension()
        if dimension is None:
            sample = self.embed(["dimension probe"])[0]
            return len(sample)
        return int(dimension)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with normalised dense vectors."""
        if not texts:
            return []
        model = self.load()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[CorpusDocument]:
    """Load markdown corpus files with YAML frontmatter."""
    documents: list[CorpusDocument] = []
    for path in sorted(corpus_dir.glob("*.md")):
        metadata, body = _read_frontmatter(path)
        documents.append(
            CorpusDocument(
                doc_id=str(metadata["id"]),
                title=str(metadata["title"]),
                language=str(metadata["language"]),
                text=body.strip(),
                path=path,
            )
        )
    return documents


def prepare_chunks(
    settings: Settings | None = None,
    corpus_dir: Path = CORPUS_DIR,
) -> list[PreparedChunk]:
    """Load the corpus and split each document into retrieval chunks."""
    active_settings = settings or get_settings()
    prepared: list[PreparedChunk] = []
    for document in load_corpus(corpus_dir):
        chunks = chunk_text(
            document.text,
            active_settings.chunk_size_tokens,
            active_settings.chunk_overlap_tokens,
        )
        prepared.extend(
            PreparedChunk(
                doc_id=document.doc_id,
                title=document.title,
                language=document.language,
                chunk=chunk,
            )
            for chunk in chunks
        )
    return prepared


def ingest_corpus(
    settings: Settings | None = None,
    corpus_dir: Path = CORPUS_DIR,
) -> IngestionReport:
    """Embed corpus chunks and upsert them to Qdrant."""
    active_settings = settings or get_settings()
    timings: dict[str, float] = {}

    started = time.perf_counter()
    prepared = prepare_chunks(active_settings, corpus_dir)
    timings["chunk"] = _elapsed_ms(started)

    started = time.perf_counter()
    _ensure_qdrant_available(active_settings)
    timings["qdrant_check"] = _elapsed_ms(started)

    started = time.perf_counter()
    embedder = SentenceTransformerEmbedder(active_settings.embedding_model)
    texts = [item.chunk.text for item in prepared]
    vectors = embedder.embed(texts)
    dimension = len(vectors[0]) if vectors else embedder.dimension()
    timings["embed"] = _elapsed_ms(started)

    started = time.perf_counter()
    _upsert_chunks(active_settings, prepared, vectors, dimension)
    timings["upsert"] = _elapsed_ms(started)

    return IngestionReport(
        documents=len(load_corpus(corpus_dir)),
        chunks=len(prepared),
        collection=active_settings.qdrant_collection,
        timings_ms=timings,
    )


def _upsert_chunks(
    settings: Settings,
    prepared: list[PreparedChunk],
    vectors: list[list[float]],
    dimension: int,
) -> None:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except ImportError as exc:
        msg = "qdrant-client is not installed in this environment"
        raise IngestionUnavailableError(msg) from exc

    try:
        client = QdrantClient(url=settings.qdrant_url)
        if not client.collection_exists(settings.qdrant_collection):
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
        points = [
            PointStruct(
                id=str(_point_id(item.doc_id, item.chunk.chunk_index)),
                vector=vector,
                payload={
                    "doc_id": item.doc_id,
                    "title": item.title,
                    "chunk_index": item.chunk.chunk_index,
                    "text": item.chunk.text,
                    "language": item.language,
                },
            )
            for item, vector in zip(prepared, vectors, strict=True)
        ]
        if points:
            client.upsert(collection_name=settings.qdrant_collection, points=points)
    except Exception as exc:
        msg = f"Qdrant is unavailable at {settings.qdrant_url}: {exc}"
        raise IngestionUnavailableError(msg) from exc


def _ensure_qdrant_available(settings: Settings) -> None:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        msg = "qdrant-client is not installed in this environment"
        raise IngestionUnavailableError(msg) from exc

    try:
        QdrantClient(url=settings.qdrant_url).get_collections()
    except Exception as exc:
        msg = f"Qdrant is unavailable at {settings.qdrant_url}: {exc}"
        raise IngestionUnavailableError(msg) from exc


def _read_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    try:
        import yaml
    except ImportError as exc:
        msg = "PyYAML is not installed in this environment"
        raise IngestionUnavailableError(msg) from exc

    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        msg = f"{path} is missing YAML frontmatter"
        raise ValueError(msg)
    _, metadata_text, body = raw.split("---", maxsplit=2)
    metadata = yaml.safe_load(metadata_text) or {}
    for key in ("id", "title", "language"):
        if key not in metadata:
            msg = f"{path} frontmatter is missing {key}"
            raise ValueError(msg)
    return metadata, body


def _point_id(doc_id: str, chunk_index: int) -> uuid.UUID:
    digest = hashlib.sha256(f"{doc_id}:{chunk_index}".encode()).hexdigest()
    return uuid.uuid5(POINT_NAMESPACE, digest)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
