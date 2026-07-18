"""
Gemma via the local Ollama server (http://localhost:11434).

Fingerspelling recognition is inherently noisy — dropped letters, near-miss
shapes, no explicit word boundaries. Gemma's job is to turn that raw letter
stream into the most plausible natural-language message.

Setup (see README):
    brew install ollama
    ollama serve                 # background server
    ollama pull gemma4:e2b-it-qat  # 4.3GB QAT build — fits 8GB RAM alongside the app

The client degrades gracefully: if the server or model isn't available it falls
back to naively joining the letters, so the vision demo still runs offline.
"""

import json

import requests

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:e2b-it-qat"

SYSTEM_PROMPT = (
    "You reconstruct text from American Sign Language fingerspelling. "
    "The input is a raw sequence of recognized letters; '/' marks a word "
    "boundary. Recognition may contain small errors, missing letters, or "
    "duplicates. Infer the single most likely intended message in natural, "
    "correctly spelled English. Reply with ONLY that message — no quotes, no "
    "explanation, no preamble."
)


class GemmaClient:
    def __init__(self, model=DEFAULT_MODEL, host=DEFAULT_HOST, timeout=(10, 180)):
        # timeout is (connect, read). The read budget is generous because the
        # first request pays a one-time cold model-load (tens of seconds for a
        # 4B model); after that Ollama keeps it warm.
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    # -- health ----------------------------------------------------------
    def available(self):
        """Return (ok, message). ok=True only if the model is pulled & ready."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.RequestException as e:
            return False, f"Ollama not reachable at {self.host} ({e.__class__.__name__})"
        tags = [m.get("name", "") for m in r.json().get("models", [])]
        # match 'gemma3', 'gemma3:latest', 'gemma3:4b', etc.
        if any(t == self.model or t.startswith(self.model + ":") for t in tags):
            return True, f"{self.model} ready"
        return False, (f"model '{self.model}' not pulled. "
                       f"Run: ollama pull {self.model}. Have: {tags or 'none'}")

    # -- generation ------------------------------------------------------
    def interpret(self, raw_letters, on_token=None):
        """Reconstruct a message from a raw letter string.

        `raw_letters` e.g. "H E L L O / W O R L D". If `on_token` is given it's
        called with each streamed text chunk for live rendering. Returns the
        full reconstructed string (falls back to a naive join on any failure).
        """
        raw_letters = (raw_letters or "").strip()
        if not raw_letters:
            return ""

        prompt = f"Recognized letters: {raw_letters}\nMessage:"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": on_token is not None,
            "options": {"temperature": 0.2},
        }
        try:
            if on_token is not None:
                return self._stream(payload, on_token, raw_letters)
            r = requests.post(f"{self.host}/api/generate", json=payload,
                              timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("response", "").strip() or _fallback(raw_letters)
        except requests.RequestException:
            return _fallback(raw_letters)

    def _stream(self, payload, on_token, raw_letters):
        chunks = []
        try:
            with requests.post(f"{self.host}/api/generate", json=payload,
                               stream=True, timeout=self.timeout) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    tok = obj.get("response", "")
                    if tok:
                        chunks.append(tok)
                        on_token(tok)
                    if obj.get("done"):
                        break
        except (requests.RequestException, json.JSONDecodeError):
            pass
        # fall back on the raw letters (never the prompt) if nothing came back
        return "".join(chunks).strip() or _fallback(raw_letters)

    def warmup(self):
        """Trigger the one-time model load so the first real request is fast."""
        try:
            requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": "hi", "stream": False,
                      "options": {"num_predict": 1}},
                timeout=(10, 300))
            return True
        except requests.RequestException:
            return False


def _fallback(raw_letters):
    """No LLM available: strip separators and lowercase into rough words."""
    words = [w.replace(" ", "") for w in raw_letters.split("/")]
    return " ".join(w.capitalize() for w in words if w).strip()


if __name__ == "__main__":
    # Quick manual check: python src/gemma_client.py "W O L F / F O O D"
    import sys
    client = GemmaClient()
    ok, msg = client.available()
    print(f"[gemma] {msg}")
    sample = sys.argv[1] if len(sys.argv) > 1 else "H E L L O / W O R L D"
    print("input :", sample)
    print("output:", client.interpret(sample, on_token=lambda t: None))
