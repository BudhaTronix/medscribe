"""Tests for ASR text normalisation and error rates."""

from app.asr.wer import compute_error_rates, normalise_for_wer


def test_normalise_maps_german_sharp_s_to_ss() -> None:
    assert normalise_for_wer("Große Straße!", language="de") == "grosse strasse"


def test_error_rates_after_punctuation_normalisation() -> None:
    rates = compute_error_rates("Patient, stable.", "patient stable", language="en")

    assert rates.wer == 0.0
    assert rates.cer == 0.0
