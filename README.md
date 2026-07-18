# Sign-to-Text with Gemma

A recreation of Analytics Vidhya's
[hand-tracking system with OpenCV + MediaPipe](https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/),
extended with an **AI layer**: recognized ASL fingerspelling is reconstructed
into fluent English by **Gemma**, running locally via [Ollama](https://ollama.com).

```
 webcam ─▶ MediaPipe hand tracking ─▶ ASL letter recognition ─▶ word buffer ─▶ Gemma ─▶ fluent text
          (handDetector module)       (rules or trained model)   (+ smoothing)   (Ollama)
```

The original tutorial is pure computer vision — MediaPipe does the hand-landmark
ML and there's no LLM. Gemma is the new "AI section": it turns the inherently
noisy letter stream (dropped letters, near-miss shapes, no word boundaries) into
the most likely intended message.

> Model: this project uses **Gemma 4** (Google's open model family, released
> April 2026). The default is the `gemma4:e2b-it-qat` tag — a 4.3 GB
> quantization-aware build that fits in 8 GB of RAM *while the camera + MediaPipe
> app is also running*. On a bigger machine, override with e.g.
> `--gemma-model gemma4:e4b` or `--gemma-model gemma4:12b`.

## Project layout

| File | Role |
|------|------|
| [src/hand_tracking_module.py](src/hand_tracking_module.py) | The tutorial's `handDetector` (`findHands`/`findPosition`), ported to MediaPipe's current **Tasks** API. Adds `fingersUp()`, `getHandedness()`, bbox. |
| [src/gesture_features.py](src/gesture_features.py) | Landmarks → a translation/rotation/scale-invariant 42-D feature vector. |
| [src/asl_rules.py](src/asl_rules.py) | Rule-based recognizer — **zero-setup** subset of the alphabet. |
| [src/collect_data.py](src/collect_data.py) | Capture labeled samples → `data/landmarks.csv`. |
| [src/train_classifier.py](src/train_classifier.py) | Train a KNN/MLP over the samples → `models/sign_model.joblib`. |
| [src/sign_recognizer.py](src/sign_recognizer.py) | Dispatch rules/model + temporal smoothing (commit a letter only when held steady). |
| [src/gemma_client.py](src/gemma_client.py) | Talk to Gemma via Ollama; streams tokens, falls back gracefully offline. |
| [src/app.py](src/app.py) | The full pipeline + on-screen HUD. |

## Setup

MediaPipe doesn't ship wheels for Python 3.14, so use **Python 3.11**:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The MediaPipe hand-landmark model (`hand_landmarker.task`, ~8 MB) **auto-downloads**
to `models/` on first run.

### Gemma via Ollama

```bash
brew install ollama
brew services start ollama
ollama pull gemma4:e2b-it-qat
```

`brew services start ollama` runs the server in the background (stop it later
with `brew services stop ollama`). Alternatively run `ollama serve` in its own
terminal — it's a blocking foreground process, so it can't share a line with the
`pull`. The default model is ~4.3 GB (Gemma 4 e2b, QAT 4-bit), chosen to fit an
8 GB machine alongside the app; larger tags like `gemma4:e4b` (9.6 GB) or
`gemma4:12b` need more RAM.

> zsh tip: don't paste inline `# comments` after a command — unlike bash, an
> interactive zsh treats `#` as a literal argument, not a comment.

The app checks `http://localhost:11434` at startup. If Ollama/Gemma isn't ready,
the vision demo still runs and Gemma output falls back to a naive letter-join.

## Run

**Plain hand tracking** (closest to the original article — landmarks + FPS):

```bash
python src/hand_tracking_module.py
```

**Full sign-to-text app** (rule-based letters, Gemma on):

```bash
python src/app.py
```

Then fingerspell a word like `WOLF` or `FOOD` and press `SPACE` to send it to
Gemma.

Controls (focus the video window):

| Key / gesture | Action |
|---|---|
| hold a letter's handshape | commits that letter once it's steady |
| open palm | inserts a word break |
| `SPACE` | send the spelled buffer to Gemma |
| `Backspace` | delete last letter |
| `c` | clear buffer + output |
| `q` | quit |

### Rule mode vs. trained mode

**Rule mode** (default, no training) reliably covers a well-separated subset:

```
A B D F I L O U V W Y  + open-palm word break
```

Enough to fingerspell demo words like **WOLF**, **FOOD**, **BUD**, then let Gemma
expand/correct. Letters that look alike from one frame (and motion letters J, Z)
are left to the trained model.

**Trained mode** covers all 26. Collect ~30–50 samples per letter, train, run:

```bash
python src/collect_data.py --out data/landmarks.csv   # press each letter key to capture
python src/train_classifier.py --data data/landmarks.csv --out models/sign_model.joblib
python src/app.py --mode model
```

## What's verified vs. needs your machine

Verified headless in this repo (`python -m py_compile` + unit checks):
- feature extraction is invariant to translate/rotate/scale (max Δ ≈ 1e-15);
- all rule-classifier cases (A B D F I L O U V W Y + SPACE);
- collect → train → model-mode inference (synthetic data, CV acc 1.0);
- temporal smoothing commits a held letter once, and re-commits after a release
  gap so double letters (the "LL" in HELLO) work;
- MediaPipe's Tasks graph initializes and processes frames (Apple M1 / Metal).

Needs your hardware to exercise live:
- a real **webcam** (the `cv2.imshow` window can't run in a headless session);
- **Ollama + Gemma** pulled locally for real reconstruction (offline fallback
  otherwise).

## Notes

- This folder sits inside a git repo rooted at your home directory. Everything
  here is scoped to `gdg-hackathon/`; `.venv/`, `data/*.csv`, and the model
  binaries are git-ignored.
# gemma-signlanguage
