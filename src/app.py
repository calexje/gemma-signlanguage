"""
Sign-to-Text with Gemma — the full pipeline.

    webcam --> MediaPipe hand tracking --> ASL letter recognition
           --> word buffer --> Gemma (via Ollama) --> fluent English

Controls (focus the video window):
    (spell by holding each letter's handshape until it "commits")
    open palm     end the word: send it to Gemma, speak the result, wipe buffer
    devil horns   delete the last committed letter
    SPACE key     same as the open-palm gesture (manual trigger)
    c             clear the buffer and Gemma output
    q             quit

Usage:
    python src/app.py                         # rule-based letters, Gemma on
    python src/app.py --mode model            # trained classifier (all 26)
    python src/app.py --no-gemma              # vision only, naive join
    python src/app.py --gemma-model gemma4:e4b # bigger Gemma 4 tag (needs more RAM)
"""

import argparse
import queue
import threading
import time

import cv2

from hand_tracking_module import handDetector
from sign_recognizer import SignRecognizer
import asl_rules


# --------------------------------------------------------------------------- #
# Text buffer: completed words + the in-progress word.
# --------------------------------------------------------------------------- #
class TextBuffer:
    def __init__(self):
        self.words = []
        self.current = ""

    def add_letter(self, ch):
        self.current += ch

    def add_space(self):
        if self.current:
            self.words.append(self.current)
            self.current = ""

    def backspace(self):
        if self.current:
            self.current = self.current[:-1]
        elif self.words:
            self.current = self.words.pop()[:-1]

    def clear(self):
        self.words = []
        self.current = ""

    def is_empty(self):
        return not self.words and not self.current

    def display(self):
        parts = self.words + ([self.current] if self.current else [])
        return " ".join(parts)

    def raw_for_gemma(self):
        parts = self.words + ([self.current] if self.current else [])
        return " / ".join(" ".join(w) for w in parts)


# --------------------------------------------------------------------------- #
# Two-stage pipeline, both off the video thread so it never stalls:
#   GemmaWorker  — a queue of letter-strings -> reconstructed phrases
#   AudioPlayer  — a queue of phrases -> spoken one at a time, in order
# Because they're separate stages, Gemma can already be reconstructing the next
# word while the current one is still being spoken.
# --------------------------------------------------------------------------- #
class AudioPlayer:
    """Speak queued phrases sequentially (no overlap) on a worker thread."""
    def __init__(self, speak_fn):
        self.speak_fn = speak_fn          # blocking speak (returns when audio done)
        self.q = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def enqueue(self, text):
        if text and text.strip():
            self.q.put(text)

    def _loop(self):
        while True:
            text = self.q.get()
            try:
                self.speak_fn(text)
            except Exception as e:                      # never kill the audio thread
                print(f"[tts] play error: {e}")
            finally:
                self.q.task_done()


class GemmaWorker:
    """Reconstruct queued letter-strings with Gemma, one at a time. Streams into
    `self.text` for the HUD and hands each finished phrase to `on_final`."""
    def __init__(self, client, on_final=None):
        self.client = client
        self.on_final = on_final          # called with each finished phrase (-> audio)
        self.lock = threading.Lock()
        self.text = ""
        self.thinking = False
        self.q = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, raw_letters):
        """Queue a letter-string for reconstruction (non-blocking; never dropped)."""
        if self.client is None or not raw_letters or not raw_letters.strip():
            return
        self.q.put(raw_letters)

    def _loop(self):
        while True:
            raw = self.q.get()
            self.thinking = True
            with self.lock:
                self.text = ""

            def on_token(tok):
                with self.lock:
                    self.text += tok

            final = None
            try:
                final = self.client.interpret(raw, on_token=on_token)
                with self.lock:
                    self.text = final
            except Exception as e:
                print(f"[gemma] error: {e}")
            finally:
                self.thinking = False
                self.q.task_done()
            if self.on_final and final:
                self.on_final(final)      # hand off to the audio stage

    def snapshot(self):
        with self.lock:
            return self.text, self.thinking


# --------------------------------------------------------------------------- #
# Simple on-frame text panel with word wrap.
# --------------------------------------------------------------------------- #
def draw_panel(img, lines, org=(10, 0), color=(255, 255, 255)):
    x, y = org
    for text, scale, col in lines:
        y += int(28 * scale)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale,
                    col, 1, cv2.LINE_AA)


def wrap(text, width=48):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def draw_joint_labels(img, lmList):
    """Debug: mark every landmark with its id and (x, y) pixel coordinates."""
    for idx, x, y in lmList:
        cv2.circle(img, (x, y), 3, (0, 180, 255), cv2.FILLED)
        txt = f"{idx}:{x},{y}"
        cv2.putText(img, txt, (x + 5, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (60, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Sign-to-Text with Gemma.")
    ap.add_argument("--mode", choices=["rules", "model"], default="rules")
    ap.add_argument("--model-path", default="models/sign_model.joblib")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--gemma-model", default="gemma4:e2b-it-qat")
    ap.add_argument("--gemma-host", default="http://localhost:11434")
    ap.add_argument("--no-gemma", action="store_true")
    ap.add_argument("--stable-frames", type=int, default=8)
    ap.add_argument("--tts", action="store_true",
                    help="Speak Gemma's output aloud (local macOS `say` by default).")
    ap.add_argument("--tts-url", default=None,
                    help="Use a remote TTS gateway at this URL instead of local speech.")
    ap.add_argument("--debug", action="store_true",
                    help="Overlay raw handedness / fingersUp / prediction.")
    args = ap.parse_args()

    recognizer = SignRecognizer(
        mode=args.mode,
        model_path=args.model_path if args.mode == "model" else None,
        stable_frames=args.stable_frames,
    )

    client = None
    gemma_status = "gemma: disabled"
    if not args.no_gemma:
        from gemma_client import GemmaClient
        client = GemmaClient(model=args.gemma_model, host=args.gemma_host)
        ok, msg = client.available()
        gemma_status = f"gemma: {msg}"
        if ok:
            # Preload the model so the first SPACE press isn't a cold-start wait.
            threading.Thread(target=client.warmup, daemon=True).start()
        else:
            print(f"[warn] {msg}\n       Running with naive fallback until Ollama is ready.")

    on_final = None
    if args.tts:
        from tts_broadcast import speak_local
        sinks = []
        if args.tts_url:                            # optional web gateway (non-blocking)
            from tts_broadcast import broadcast
            sinks.append(lambda text: broadcast(text, gateway_url=args.tts_url))
            print(f"[tts] local speech + gateway -> {args.tts_url}")
        else:
            print("[tts] local speech (native OS TTS)")
        # blocking local speak LAST so the audio queue serializes clips in order
        sinks.append(lambda text: speak_local(text, blocking=True))

        def speak_all(text):
            for sink in sinks:
                sink(text)

        player = AudioPlayer(speak_all)
        on_final = player.enqueue               # Gemma hands finished phrases here

    buffer = TextBuffer()
    gemma = GemmaWorker(client, on_final=on_final)
    detector = handDetector(maxHands=1, detectionCon=0.7, trackCon=0.6)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    pTime = 0
    def send_word():
        """End of word: queue the current buffer for Gemma + speech, then wipe
        it so the next word can start immediately (both run off the video thread)."""
        raw = buffer.raw_for_gemma()
        if raw.strip():
            gemma.submit(raw)
            buffer.clear()

    print("Running. Focus the video window. Open palm = speak the word, q = quit.")

    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.flip(img, 1)  # selfie view
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)
        fingers = detector.fingersUp() if lmList else []

        current, committed = recognizer.update(fingers, lmList)

        if committed == asl_rules.SPACE:
            send_word()                     # open palm: send -> speak -> wipe buffer
        elif committed == asl_rules.DELETE:
            buffer.backspace()
        elif committed is not None:
            buffer.add_letter(committed)

        # big live label above the hand (letter, "_" for space, "DEL" for delete)
        if current and bbox:
            label = {asl_rules.SPACE: "_", asl_rules.DELETE: "DEL"}.get(current, current)
            cv2.putText(img, label, (bbox[0] - 10, bbox[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(img, label, (bbox[0] - 10, bbox[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 2, cv2.LINE_AA)

        # HUD
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime else 0
        pTime = cTime
        gtext, thinking = gemma.snapshot()

        h, w, _ = img.shape
        cv2.rectangle(img, (0, h - 110), (w, h), (0, 0, 0), cv2.FILLED)

        header = [
            (f"FPS {int(fps)}   mode:{args.mode}   {gemma_status}", 0.9, (180, 180, 180)),
        ]
        draw_panel(img, header, org=(10, 0))

        spelled = "spelled: " + (buffer.display() or "-")
        gemma_line = "gemma:   " + ("thinking..." if thinking else (gtext or "-"))
        lines = [(spelled, 1.0, (0, 255, 0))]
        for i, ln in enumerate(wrap(gemma_line, 60)):
            lines.append((ln, 1.0, (0, 200, 255)))
        draw_panel(img, lines, org=(10, h - 95))

        if args.debug:
            hand = detector.getHandedness() or "-"
            dbg = [(f"[debug] hand:{hand}  fingers:{fingers}  reads:{current or '-'}",
                    0.8, (0, 180, 255))]
            if lmList:
                draw_joint_labels(img, lmList)
                m = asl_rules.metrics(lmList)
                if m:
                    dbg.append(
                        (f"[debug] span:{m['tip_span']:.2f}/{asl_rules._C_MAX_TIP_SPREAD}"
                         f"  gap:{m['gap']:.2f}(C>{asl_rules._C_MIN_THUMB_GAP}/E<{asl_rules._E_MAX_THUMB_GAP})"
                         f"  thumb:{m['thumb']}  ext(imrp):{m['ext']}", 0.8, (0, 180, 255)))
            draw_panel(img, dbg, org=(10, h - 140))  # above the info band

        cv2.imshow("Sign-to-Text with Gemma", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            send_word()        # same as the open-palm gesture (manual trigger)
        elif key in (8, 127):  # backspace / delete
            buffer.backspace()
        elif key == ord("c"):
            buffer.clear()
            with gemma.lock:
                gemma.text = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
