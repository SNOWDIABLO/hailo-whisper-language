# hailo-whisper-language

Make **Hailo Whisper speak your language** (French, German, Spanish, ...), not just English.

Hailo's official speech-recognition app ([`hailo-apps`](https://github.com/hailo-ai/hailo-apps))
hard-codes English in its Whisper decoder:

```python
FORCED_DECODER_IDS = [<|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>]
```

and re-reads that list on every transcription. So even with a multilingual Whisper
HEF, you get English (or garbled mixed-language) output no matter what you speak.
The common workaround, hand-editing the installed `whisper_pipeline.py`, is lost on
the next `pip install` and silently flips you back to English.

This repo fixes it cleanly:

- **`hailo_whisper_language.py`** swaps the language token **in memory at startup**,
  so the installed package stays untouched and survives updates. Any Whisper
  language code works (`fr`, `de`, `es`, `it`, `pt`, `nl`, ...).
- **`hallucination_filter.py`** strips Whisper's non-speech junk on silence
  (`[Music]`, `♪`, "Subtitles by Amara.org", "merci d'avoir regardé", ...).
- **`example_sidecar.py`** is a tiny always-warm Unix-socket server (send a WAV path,
  get text back) so you load the model once and your app can fall back to CPU if it dies.

## Quick start

```python
import hailo_apps.python.standalone_apps.speech_recognition.whisper_pipeline as wp
from hailo_whisper_language import force_language

if not force_language(wp, "fr"):      # "de", "es", "it", "pt", ...
    print("Unexpected FORCED_DECODER_IDS shape, language NOT forced")

# build your WhisperPipeline as usual; it now decodes in the chosen language.
```

```python
from hallucination_filter import scrub_hallucinations
scrub_hallucinations("[Musique] bonjour")        # -> "bonjour"
scrub_hallucinations("♪ ♪")                       # -> ""  (pure silence)
```

`force_language` returns `False` without touching anything if a future
`hailo-apps` version changes the decoder-ids shape, so it never patches blindly.

## Which model / which Hailo chip?

Based on hands-on use and the Hailo community (late 2026):

| Model          | Hailo-8 / 8L | Hailo-10H | Language forcing |
|----------------|:------------:|:---------:|------------------|
| Whisper tiny   | yes          | yes       | works            |
| Whisper base   | **yes**      | yes       | **works well**   |
| Whisper small  | no*          | yes       | needed (small HEF tends to ignore the language param) |

\* On Hailo-8L a self-compiled small encoder fails to map (`16x4 not supported`),
and the GenAI small/medium HEFs target Hailo-10H. **On Hailo-8 / 8L, use `base`.**
Forcing the decoder token (what this repo does) is the reliable way to pin the
language, including for the small HEF's known "ignores language" bug.

## Notes

- Decoder token ids are from the Whisper **multilingual** tokenizer
  (`<|en|>` = 50259, one id per language in Whisper's fixed order).
- `hailo_whisper_language.py` and `hallucination_filter.py` have **no dependencies**
  and are unit-testable without hailo-apps installed (see `test_basic.py`).
- `example_sidecar.py` is a scaffold: adapt the import paths to your `hailo-apps` version.

## Author

Made by **[SnowDiablo](https://github.com/SNOWDIABLO)**, a solo dev. Extracted from a
self-hosted voice assistant running on a Raspberry Pi 5 with a Hailo-8L.

- GitHub: [@SNOWDIABLO](https://github.com/SNOWDIABLO)
- Twitch: [twitch.tv/snowdiablo](https://twitch.tv/snowdiablo)

Built on top of [hailo-ai/hailo-apps](https://github.com/hailo-ai/hailo-apps) and
OpenAI Whisper. Licensed MIT (see [LICENSE](LICENSE)), so you are free to use it,
just keep the copyright line.
