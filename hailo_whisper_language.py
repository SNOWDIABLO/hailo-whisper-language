"""Force a non-English language on Hailo's Whisper speech-to-text, at runtime.

Why this exists
---------------
Hailo's official speech-recognition app (``hailo-apps``) hard-codes English in
its ``whisper_pipeline`` module::

    FORCED_DECODER_IDS = [<|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>]

and re-reads that list on every transcription. So even with a multilingual
Whisper HEF, the decoder is pinned to English and you get English (or garbled
mixed-language) output no matter what you speak.

The usual workaround is to hand-edit the installed ``whisper_pipeline.py``, but
that edit is lost on the next ``pip install`` / update, and silently flips you
back to English. This module instead swaps the language token **in memory, at
startup**, leaving the installed package untouched.

Usage
-----
    import hailo_apps.python.standalone_apps.speech_recognition.whisper_pipeline as wp
    from hailo_whisper_language import force_language

    if not force_language(wp, "fr"):      # "de", "es", "it", "pt", ...
        print("Unexpected FORCED_DECODER_IDS shape, language NOT forced")
    # ... now build your WhisperPipeline as usual; it decodes in the chosen language.

No dependencies: importable and testable without hailo-apps installed.

Note on models / hardware (community findings, late 2026)
---------------------------------------------------------
- Whisper **base** (multilingual) respects the forced language well and runs on
  Hailo-8 / 8L / 10H.
- Whisper **small/medium** (GenAI model zoo) target **Hailo-10H**; on Hailo-8L a
  self-compiled small encoder fails ("16x4 not supported"). The GenAI small HEF
  also has a reported bug where it ignores the language parameter, forcing the
  decoder token (what this module does) is the reliable lever.
"""

# <|startoftranscript|> in the Whisper *multilingual* tokenizer.
START_OF_TRANSCRIPT = 50258

# Language -> decoder token id (Whisper multilingual tokenizer; <|en|> = 50259,
# then one id per language in the fixed Whisper LANGUAGES order).
WHISPER_LANGUAGE_IDS = {
    "en": 50259, "zh": 50260, "de": 50261, "es": 50262, "ru": 50263,
    "ko": 50264, "fr": 50265, "ja": 50266, "pt": 50267, "tr": 50268,
    "pl": 50269, "ca": 50270, "nl": 50271, "ar": 50272, "sv": 50273,
    "it": 50274, "id": 50275, "hi": 50276, "fi": 50277, "vi": 50278,
    "he": 50279, "uk": 50280, "el": 50281, "ms": 50282, "cs": 50283,
    "ro": 50284, "da": 50285, "hu": 50286, "ta": 50287, "no": 50288,
    "th": 50289, "ur": 50290, "hr": 50291, "bg": 50292, "lt": 50293,
}


def language_token(lang: str) -> int:
    """Decoder token id for a 2-letter language code (e.g. 'fr'). Raises KeyError
    if unknown, see WHISPER_LANGUAGE_IDS or the Whisper tokenizer for more."""
    return WHISPER_LANGUAGE_IDS[lang.lower()]


def force_language(module, lang: str = "fr") -> bool:
    """Replace the language in ``module.FORCED_DECODER_IDS`` with ``lang``.

    Returns True on success. Returns False **without changing anything** if the
    list isn't the expected ``[<|sot|>, <|lang|>, ...]`` shape, so a future
    hailo-apps version is never patched blindly. Raises KeyError for an unknown
    language code.
    """
    ids = list(getattr(module, "FORCED_DECODER_IDS", None) or [])
    if len(ids) < 2 or ids[0] != START_OF_TRANSCRIPT:
        return False
    ids[1] = language_token(lang)
    module.FORCED_DECODER_IDS = ids
    return True
