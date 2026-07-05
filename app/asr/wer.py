"""Word and character error rate helpers."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Literal

import jiwer

LanguageCode = Literal["de", "en"]


@dataclass(frozen=True)
class ErrorRates:
    """WER and CER scores for an ASR hypothesis."""

    wer: float
    cer: float


def normalise_for_wer(text: str, *, language: LanguageCode | None = None) -> str:
    """Lowercase, strip punctuation, collapse whitespace, and map German sharp s."""
    normalised = text.lower().replace("ß", "ss")
    punctuation = string.punctuation + "„“”‘’‚"
    normalised = normalised.translate(str.maketrans("", "", punctuation))
    normalised = re.sub(r"\s+", " ", normalised).strip()
    if language == "de":
        normalised = normalised.replace("ẞ".lower(), "ss")
    return normalised


def compute_error_rates(
    reference: str,
    hypothesis: str,
    *,
    language: LanguageCode | None = None,
) -> ErrorRates:
    """Compute WER and CER after demo normalisation."""
    reference_norm = normalise_for_wer(reference, language=language)
    hypothesis_norm = normalise_for_wer(hypothesis, language=language)
    if not reference_norm and not hypothesis_norm:
        return ErrorRates(wer=0.0, cer=0.0)
    if not reference_norm:
        return ErrorRates(wer=1.0, cer=1.0)
    return ErrorRates(
        wer=float(jiwer.wer(reference_norm, hypothesis_norm)),
        cer=float(jiwer.cer(reference_norm, hypothesis_norm)),
    )
