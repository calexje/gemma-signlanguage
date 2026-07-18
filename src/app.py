"""
Sign-to-Text with Gemma — the full pipeline.

    webcam --> MediaPipe hand tracking --> ASL letter recognition
           --> word buffer --> Gemma (via Ollama) --> fluent English

Controls (focus the video window):
    (spell by holding each letter's handshape until it "commits")
    open palm     insert a word break  (also mappable to SPACE by the model)
    SPACE key     send the buffer to Gemma for reconstruction
    BACKSPACE     delete the last committed letter
    c             clear the buffer and Gemma output
    q             quit

Usage:
    python src/app.py                         # rule-based letters, Gemma on
    python src/app.py --mode model            # trained classifier (all 26)
    python src/app.py --no-gemma              # vision only, naive join
    python src/app.py --gemma-model gemma4:e4b # bigger Gemma 4 tag (needs more RAM)
"""

import argparse
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
# Gemma runs off-thread so the video never stalls.
# --------------------------------------------------------------------------- #
class GemmaWorker:
    def __init__(self, client, on_final=None):
        self.client = client
        self.on_final = on_final          # called with the finished text (e.g. TTS)
        self.lock = threading.Lock()
        self.text = ""
        self.thinking = False
        self.status = ""

    def start(self, raw_letters):
        if self.client is None or self.thinking or not raw_letters.strip():
            return
        self.thinking = True
        with self.lock:
            self.text = ""
        threading.Thread(target=self._run, args=(raw_letters,), daemon=True).start()

    def _run(self, raw_letters):
        def on_token(tok):
            with self.lock:
                self.text += tok
        final = None
        try:
            final = self.client.interpret(raw_letters, on_token=on_token)
            with self.lock:
                self.text = final
        finally:
            self.thinking = False
        # Pipe the finished reconstruction onward (TTS) — already off the video
        # thread, so a slow gateway can't stall the loop.
        if self.on_final and final:
            self.on_final(final)

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

    tts_fn = None
    if args.tts:
        if args.tts_url:
            from tts_broadcast import broadcast
            tts_fn = lambda text: broadcast(text, gateway_url=args.tts_url)
            print(f"[tts] gateway -> {args.tts_url}")
        else:
            from tts_broadcast import speak_local
            tts_fn = speak_local
            print("[tts] local speech (macOS `say`)")

    buffer = TextBuffer()
    gemma = GemmaWorker(client, on_final=tts_fn)
    detector = handDetector(maxHands=1, detectionCon=0.7, trackCon=0.6)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    pTime = 0
    print("Running. Focus the video window. SPACE = ask Gemma, q = quit.")

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
            buffer.add_space()
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
            gemma.start(buffer.raw_for_gemma())
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
