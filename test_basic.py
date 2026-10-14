"""Unit tests that run without hailo-apps or a Hailo chip: pytest test_basic.py"""
import types

from hailo_whisper_language import force_language, language_token, START_OF_TRANSCRIPT
from hallucination_filter import scrub_hallucinations


def _fake_module(ids):
    m = types.ModuleType("fake_whisper_pipeline")
    m.FORCED_DECODER_IDS = ids
    return m


def test_force_language_swaps_only_the_language_token():
    m = _fake_module([START_OF_TRANSCRIPT, 50259, 50359, 50363])  # English
    assert force_language(m, "fr") is True
    assert m.FORCED_DECODER_IDS == [START_OF_TRANSCRIPT, language_token("fr"), 50359, 50363]
    assert language_token("fr") == 50265


def test_force_language_refuses_unexpected_shape():
    m = _fake_module([999, 50259])          # does not start with <|sot|>
    assert force_language(m, "de") is False
    assert m.FORCED_DECODER_IDS == [999, 50259]   # unchanged


def test_scrub_removes_hallucinations():
    assert scrub_hallucinations("[Musique] bonjour Monsieur") == "bonjour Monsieur"
    assert scrub_hallucinations("Hello [Applause]!") == "Hello !"
    assert scrub_hallucinations("Subtitles by Amara.org community") == ""


def test_scrub_pure_silence_is_empty():
    assert scrub_hallucinations("♪ ♪ ♪") == ""
    assert scrub_hallucinations("[Music]") == ""
    assert scrub_hallucinations("   ...  ") == ""
