"""Tests for fixed and paragraph-aware chunking."""

import pytest

from app.ingestion.chunker import chunk_text, estimate_tokens


def test_empty_input_returns_no_chunks() -> None:
    assert chunk_text(" \n ", 10, 2) == []


def test_estimate_tokens_uses_documented_ratio() -> None:
    assert estimate_tokens("one two three four") == 3


def test_fixed_chunking_creates_overlap() -> None:
    text = "one two three four five six seven eight nine ten"

    chunks = chunk_text(text, chunk_size_tokens=3, overlap_tokens=1)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "three four five six",
        "five six seven eight",
        "seven eight nine ten",
    ]
    assert chunks[1].start_word == 2


def test_fixed_chunking_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_tokens must be smaller"):
        chunk_text("one two", 2, 2)


def test_paragraph_aware_mode_keeps_short_paragraphs_together() -> None:
    text = "Alpha beta gamma.\n\nDelta epsilon zeta.\n\nEta theta iota."

    chunks = chunk_text(text, chunk_size_tokens=6, overlap_tokens=0, paragraph_aware=True)

    assert len(chunks) == 2
    assert chunks[0].text == "Alpha beta gamma. Delta epsilon zeta."
    assert chunks[1].text == "Eta theta iota."


def test_paragraph_aware_mode_adds_overlap_context() -> None:
    text = "Alpha beta gamma delta.\n\nEpsilon zeta eta theta."

    chunks = chunk_text(text, chunk_size_tokens=4, overlap_tokens=1, paragraph_aware=True)

    assert chunks[1].text.startswith("gamma delta. Epsilon")
