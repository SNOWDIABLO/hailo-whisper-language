#!/usr/bin/env python3
"""Minimal Hailo Whisper STT sidecar, in the language of your choice.

Loads the Whisper pipeline on a Hailo accelerator ONCE (kept warm in RAM), then
listens on a Unix socket: send it the path to a 16 kHz mono WAV, it returns the
transcribed text. Running it as a long-lived process means you pay the model
load cost only once, and a crash here never takes down your main app, if the
socket is gone, your app can fall back to CPU Whisper.

Run it with the Python env that has hailo-apps + torch installed, e.g.:

    WHISPER_LANG=fr WHISPER_VARIANT=base python example_sidecar.py

Then from any client:

    import socket
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect("/tmp/hailo_whisper.sock")
    c.sendall(b"/path/to/audio.wav")
    print(c.recv(65536).decode())

This is an example scaffold, adapt the import paths to your hailo-apps version.
"""
import os
import socket
import time
import traceback

SOCK = os.environ.get("WHISPER_SOCK", "/tmp/hailo_whisper.sock")
LANG = os.environ.get("WHISPER_LANG", "fr")
VARIANT = os.environ.get("WHISPER_VARIANT", "base")   # base = safe on Hailo-8/8L
ARCH = os.environ.get("WHISPER_ARCH", "hailo8l")

# hailo-apps pipeline + helpers (paths may differ across hailo-apps versions).
import hailo_apps.python.standalone_apps.speech_recognition.whisper_pipeline as whisper_pipeline
from hailo_apps.python.standalone_apps.speech_recognition.whisper_pipeline import (
    WhisperPipeline, load_audio, improve_audio, preprocess_audio, clean_transcription,
)

from hailo_whisper_language import force_language
from hallucination_filter import scrub_hallucinations

LANG_FORCED = force_language(whisper_pipeline, LANG)


def transcribe(path, pipeline, chunk_length):
    audio = load_audio(path)
    audio, start = improve_audio(audio)
    if start is None:
        return ""
    mels = preprocess_audio(audio, chunk_length=chunk_length,
                            chunk_offset=max(start - 0.2, 0))
    out = []
    for mel in mels:
        pipeline.send_data(mel)
        time.sleep(0.1)
        out.append(pipeline.get_transcription())
    return scrub_hallucinations(clean_transcription(" ".join(out)).strip())


def main():
    print("[hailo-whisper] loading pipeline (torch + Hailo)...", flush=True)
    pipeline = WhisperPipeline.from_variant(VARIANT, arch=ARCH)  # adapt to your version
    chunk_length = pipeline.get_chunk_length()
    print("[hailo-whisper] READY (variant=%s, lang=%s, forced=%s)"
          % (VARIANT, LANG, LANG_FORCED), flush=True)
    if not LANG_FORCED:
        print("[hailo-whisper] WARNING: unexpected FORCED_DECODER_IDS shape, "
              "language NOT forced (will decode in English).", flush=True)

    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    os.chmod(SOCK, 0o660)
    srv.listen(4)
    print("[hailo-whisper] listening on %s" % SOCK, flush=True)

    while True:
        try:
            conn, _ = srv.accept()
        except Exception:
            traceback.print_exc()
            continue
        try:
            path = conn.recv(8192).decode("utf-8", "replace").strip()
            txt = transcribe(path, pipeline, chunk_length) if (path and os.path.exists(path)) else ""
            conn.sendall((txt + "\n").encode("utf-8"))
        except Exception:
            traceback.print_exc()
        finally:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
