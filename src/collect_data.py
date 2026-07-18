"""
Collect labeled hand-shape samples for the trainable ASL classifier.

Run it, make a letter's handshape at the camera, and press that letter key to
capture a sample. Each capture stores the 42-D normalized feature vector plus
its label to a CSV. Aim for ~30-50 samples per letter, varying distance and
slight rotation so the model generalizes.

    python src/collect_data.py --out data/landmarks.csv

Keys:
    A-Z  capture one sample for that letter
    0    capture a "SPACE" (open-palm) sample   [optional]
    U    (undo) delete the most recently captured sample
    q    quit and save
"""

import argparse
import csv
import os

import cv2

from hand_tracking_module import handDetector
from gesture_features import landmarks_to_features, FEATURE_DIM


def load_counts(path):
    counts = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if row:
                    counts[row[0]] = counts.get(row[0], 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser(description="Collect ASL landmark samples.")
    ap.add_argument("--out", default="data/landmarks.csv",
                    help="CSV to append samples to.")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    counts = load_counts(args.out)
    last_written = []  # byte offsets, for undo

    cap = cv2.VideoCapture(args.camera)
    detector = handDetector(maxHands=1, detectionCon=0.7)

    print(f"Appending to {args.out}. Press a letter to capture, 'q' to quit.")

    with open(args.out, "a", newline="") as fh:
        writer = csv.writer(fh)
        while True:
            ok, img = cap.read()
            if not ok:
                break
            img = cv2.flip(img, 1)
            img = detector.findHands(img)
            lmList, _ = detector.findPosition(img, draw=False)

            total = sum(counts.values())
            cv2.putText(img, f"samples: {total}  labels: {len(counts)}",
                        (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0), 2)
            cv2.putText(img, "press a letter to capture, 'q' to quit",
                        (10, img.shape[0] - 15), cv2.FONT_HERSHEY_PLAIN, 1.2,
                        (200, 200, 200), 1)
            cv2.imshow("Collect ASL samples", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("u"):  # undo last capture
                if last_written:
                    fh.flush()
                    fh.seek(last_written.pop())
                    fh.truncate()
                    # recompute counts from disk after truncation
                    fh.flush()
                    counts = load_counts(args.out)
                    print("undid last sample")
                continue

            label = None
            if ord("a") <= key <= ord("z"):
                label = chr(key).upper()
            elif key == ord("0"):
                label = "SPACE"

            if label is not None:
                feats = landmarks_to_features(lmList)
                if feats is None or len(feats) != FEATURE_DIM:
                    print("  (no hand detected — sample skipped)")
                    continue
                last_written.append(fh.tell())
                writer.writerow([label, *["%.6f" % v for v in feats]])
                fh.flush()
                counts[label] = counts.get(label, 0) + 1
                print(f"  captured {label}  (now {counts[label]})")

    cap.release()
    cv2.destroyAllWindows()
    print("\nCounts per label:")
    for lab in sorted(counts):
        print(f"  {lab}: {counts[lab]}")


if __name__ == "__main__":
    main()
