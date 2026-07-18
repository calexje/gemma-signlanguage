"""
Turn a stream of per-frame hand shapes into a stream of *committed* letters.

Raw frame-by-frame classification is jittery, so this wraps either recognizer
(rule-based or a trained model) with temporal smoothing:

  * a prediction must stay stable for `stable_frames` frames to be "committed";
  * the same letter won't fire twice in a row until the hand releases (a short
    run of no-detection), so real double letters like the "LL" in HELLO still
    work — you just drop your hand briefly between them.

``update(fingers, lmList)`` returns ``(current, committed)``:
  current   = the shape being shown right now (for on-screen feedback), or None
  committed = a letter/SPACE that just became final this frame, or None
"""

from collections import Counter, deque

import numpy as np

import asl_rules


class SignRecognizer:
    def __init__(self, mode="rules", model_path=None,
                 stable_frames=8, release_frames=4, min_conf=0.6):
        self.mode = mode
        self.stable_frames = stable_frames
        self.release_frames = release_frames
        self.min_conf = min_conf

        self.history = deque(maxlen=stable_frames)
        self.locked = None          # last committed letter still being held
        self.empty_run = 0          # consecutive frames with no confident shape

        self.model = None
        self.labels = []
        if mode == "model":
            if not model_path:
                raise ValueError("mode='model' requires model_path")
            bundle = joblib_load(model_path)
            self.model = bundle["model"]
            self.labels = bundle["labels"]

    # -- raw single-frame prediction -------------------------------------
    def _predict_raw(self, fingers, lmList):
        if self.mode == "model":
            from gesture_features import landmarks_to_features
            feats = landmarks_to_features(lmList)
            if feats is None:
                return None
            X = feats.reshape(1, -1)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)[0]
                i = int(np.argmax(proba))
                if proba[i] < self.min_conf:
                    return None
                return self.model.classes_[i]
            return self.model.predict(X)[0]
        # rule-based
        return asl_rules.classify(fingers, lmList)

    # -- smoothed update --------------------------------------------------
    def update(self, fingers, lmList):
        raw = self._predict_raw(fingers, lmList) if lmList else None
        self.history.append(raw)

        if raw is None:
            self.empty_run += 1
            # a sustained gap "releases" the lock so a letter can repeat
            if self.empty_run >= self.release_frames:
                self.locked = None
            return None, None
        self.empty_run = 0

        # dominant prediction across the window
        counts = Counter(p for p in self.history if p is not None)
        if not counts:
            return raw, None
        current, n = counts.most_common(1)[0]

        committed = None
        stable = (len(self.history) == self.stable_frames
                  and n >= self.stable_frames)
        if stable and current != self.locked:
            committed = current
            self.locked = current

        return current, committed


def joblib_load(path):
    import joblib
    return joblib.load(path)
