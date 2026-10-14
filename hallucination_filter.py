"""Strip Whisper's non-speech hallucinations from a transcript.

On silence or near-silence Whisper tends to invent bracketed annotations
(``[Music]``, ``[Applause]``), music symbols (``♪``), or boilerplate credits
("Subtitles by Amara.org", "Thanks for watching", "merci d'avoir regardé"...).
These pollute anything downstream (an LLM, a command parser). This filter
removes them and returns "" when nothing real is left.

Language-agnostic by default; a few common FR/EN/DE phrases are included.
"""
import re

_BRACKET = re.compile(r"\[[^\]]*\]")                 # [Music], [Applause], [Musique]...
_PAREN = re.compile(
    r"\((?:[^)]*?)(?:music|musique|musik|applaus|applaud|laugh|rire|rires|"
    r"sigh|soupir|noise|bruit|silence|whistle|sifflement|cough|toux|breath|"
    r"respiration)(?:[^)]*?)\)", re.I)
_SYMBOLS = re.compile(r"[♪♫♬\U0001F3B5\U0001F3B6]")   # ♪ ♫ ♬ 🎵 🎶
_BOILERPLATE = re.compile(
    r"sub(?:title|titre)s?\s*(?:by|par|:)?[^.]*amara\.org[^.]*|"
    r"sous-?titrage[^.]*|"
    r"thank(?:s| you)[^.]*watching[^.]*|"
    r"merci d.?avoir regard\w*[^.]*|"
    r"(?:please\s+)?subscribe[^.]*|abonnez-?vous[^.]*|"
    r"untertitel[^.]*", re.I)

_HAS_WORD = re.compile(r"[0-9A-Za-zÀ-ɏ]")   # letters incl. accented


def scrub_hallucinations(text: str) -> str:
    """Return ``text`` with non-speech hallucinations removed, or "" if nothing
    real remains (pure silence that Whisper filled with junk)."""
    t = _BRACKET.sub(" ", text or "")
    t = _PAREN.sub(" ", t)
    t = _SYMBOLS.sub(" ", t)
    t = _BOILERPLATE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if _HAS_WORD.search(t) else ""
