"""
Speak the reconstructed text out loud. Two backends:

  * speak_local(text) — say it on THIS machine via macOS `say` (offline, default
    for --tts). Fire-and-forget so it never blocks the caller.
  * broadcast(text)   — POST {"text": ...} to a remote gateway's /api/speak,
    which speaks it in a connected browser tab (opt-in via --tts-url).

Every failure is swallowed and logged — TTS is a nice-to-have and must never
interrupt the recognition loop.
"""

import os
import shutil
import subprocess
import sys

import requests

DEFAULT_GATEWAY = ("https://ais-dev-u6a6fsv7ggun5ksmsunvur-"
                   "13504627249.europe-west3.run.app")


def speak_local(text, voice=None):
    """Speak `text` on THIS machine with the OS's native TTS — offline and
    non-blocking (fire-and-forget), so it never stalls the caller. Works on
    macOS (`say`), Linux (espeak/spd-say), and Windows (System.Speech).

    The text is always passed as a subprocess ARGUMENT or environment variable,
    never interpolated into a shell string — so recognized text can't inject
    shell commands. Returns True if a speaker process was launched."""
    text = (text or "").strip()
    if not text:
        return False
    try:
        if sys.platform == "darwin":
            cmd = ["say"] + (["-v", voice] if voice else []) + [text]
            subprocess.Popen(cmd)
        elif sys.platform.startswith("linux"):
            engine = (shutil.which("espeak-ng") or shutil.which("espeak")
                      or shutil.which("spd-say"))
            if not engine:
                print("[tts] no Linux TTS engine (install espeak or speech-dispatcher).")
                return False
            subprocess.Popen([engine, text])
        elif sys.platform == "win32":
            # text goes via env var, NOT the command string -> no injection
            ps = ("Add-Type -AssemblyName System.Speech; "
                  "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                  ".Speak($env:GEMMA_TTS_TEXT)")
            subprocess.Popen(["powershell", "-NoProfile", "-Command", ps],
                             env={**os.environ, "GEMMA_TTS_TEXT": text})
        else:
            print(f"[tts] no local TTS backend for platform {sys.platform!r}.")
            return False
        print(f'[tts] speaking: "{text}"')
        return True
    except OSError as e:
        print(f"[tts] local speak error: {e}")
        return False


def broadcast(text, gateway_url=DEFAULT_GATEWAY, timeout=3):
    """POST `text` to the TTS gateway. Returns True on success, else False.
    Never raises."""
    text = (text or "").strip()
    if not text:
        return False
    try:
        r = requests.post(
            f"{gateway_url.rstrip('/')}/api/speak",
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if r.status_code == 200:
            print(f'[tts] spoke: "{text}"')
            return True
        print(f"[tts] gateway returned {r.status_code}: {r.text[:200]}")
        return False
    except requests.RequestException as e:
        print(f"[tts] connection error: {e.__class__.__name__}")
        return False


if __name__ == "__main__":
    import sys
    msg = sys.argv[1] if len(sys.argv) > 1 else "Sign language to speech test"
    speak_local(msg)
