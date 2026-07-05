"""Text chunking utilities for synthetic clinical reference documents."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

TOKEN_WORD_RATIO = 0.75


@dataclass(frozen=True)
class Chunk:
    """A chunk of source text with approximate token metadata."""

    chunk_index: int
    text: str
    start_word: int
    end_word: int
    estimated_tokens: int


def estimate_tokens(text: str) -> int:
    """Estimate tokens as whitespace word count times 0.75."""
    words = text.split()
    if not words:
        return 0
    return max(1, math.ceil(len(words) * TOKEN_WORD_RATIO))


def chunk_text(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    *,
    paragraph_aware: bool = False,
) -> list[Chunk]:
    """Split text into overlapping chunks using an approximate token budget."""
    if chunk_size_tokens <= 0:
        msg = "chunk_size_tokens must be greater than zero"
        raise ValueError(msg)
    if overlap_tokens < 0:
        msg = "overlap_tokens cannot be negative"
        raise ValueError(msg)
    if overlap_tokens >= chunk_size_tokens:
        msg = "overlap_tokens must be smaller than chunk_size_tokens"
        raise ValueError(msg)

    stripped = text.strip()
    if not stripped:
        return []

    if paragraph_aware:
        return _chunk_paragraphs(stripped, chunk_size_tokens, overlap_tokens)
    return _chunk_words(stripped.split(), chunk_size_tokens, overlap_tokens)


def _words_for_tokens(tokens: int) -> int:
    return max(1, math.ceil(tokens / TOKEN_WORD_RATIO))


def _chunk_words(words: list[str], chunk_size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    chunk_word_limit = _words_for_tokens(chunk_size_tokens)
    overlap_words = (
        min(_words_for_tokens(overlap_tokens), chunk_word_limit - 1) if overlap_tokens else 0
    )
    step = chunk_word_limit - overlap_words
    chunks: list[Chunk] = []

    start = 0
    while start < len(words):
        end = min(start + chunk_word_limit, len(words))
        chunk_words = words[start:end]
        chunks.append(
            Chunk(
                chunk_index=len(chunks),
                text=" ".join(chunk_words),
                start_word=start,
                end_word=end,
                estimated_tokens=max(1, math.ceil(len(chunk_words) * TOKEN_WORD_RATIO)),
            )
        )
        if end == len(words):
            break
        start += step
    return chunks


def _chunk_paragraphs(text: str, chunk_size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    paragraphs = [_normalise_spaces(part) for part in re.split(r"\n\s*\n", text) if part.strip()]
    raw_chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        if paragraph_tokens > chunk_size_tokens:
            if current:
                raw_chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            paragraph_chunks = _chunk_words(paragraph.split(), chunk_size_tokens, 0)
            raw_chunks.extend(chunk.text for chunk in paragraph_chunks)
            continue
        if current and current_tokens + paragraph_tokens > chunk_size_tokens:
            raw_chunks.append("\n\n".join(current))
            current = [paragraph]
            current_tokens = paragraph_tokens
        else:
            current.append(paragraph)
            current_tokens += paragraph_tokens

    if current:
        raw_chunks.append("\n\n".join(current))

    overlap_words = _words_for_tokens(overlap_tokens) if overlap_tokens else 0
    chunks: list[Chunk] = []
    consumed_words = 0
    previous_words: list[str] = []
    for raw in raw_chunks:
        words = raw.split()
        prefix = previous_words[-overlap_words:] if overlap_words else []
        chunk_words = [*prefix, *words]
        chunks.append(
            Chunk(
                chunk_index=len(chunks),
                text=" ".join(chunk_words),
                start_word=max(0, consumed_words - len(prefix)),
                end_word=consumed_words + len(words),
                estimated_tokens=max(1, math.ceil(len(chunk_words) * TOKEN_WORD_RATIO)),
            )
        )
        consumed_words += len(words)
        previous_words = words
    return chunks


def _normalise_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.strip())
