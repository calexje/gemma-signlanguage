"""
Hand-tracking module — the reusable ``handDetector`` from the Analytics Vidhya
tutorial, ported to MediaPipe's current **Tasks** API.

    https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/

Why not the tutorial's code verbatim: the article uses the legacy
``mp.solutions.hands`` API, which recent MediaPipe (0.10.x) has removed in favour
of ``mediapipe.tasks.python.vision.HandLandmarker``. The public surface here is
kept identical to the tutorial (``findHands`` / ``findPosition``) so downstream
code reads the same; only the internals changed. Extras added for sign
recognition: ``fingersUp()``, ``getHandedness()``, and a bounding box.

The Tasks API needs a model file (``hand_landmarker.task``). It's auto-downloaded
to ``models/`` on first use if missing.
"""

import math
import os
import urllib.request

# Quiet MediaPipe's C++ (glog) chatter for a clean demo console. Must be set
# before mediapipe is imported.
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "hand_landmarker.task")

# The 21-landmark skeleton edges (previously mp.solutions.hands.HAND_CONNECTIONS).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


def ensure_model(path=DEFAULT_MODEL_PATH):
    """Download the hand-landmarker model to `path` if it isn't there yet."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[handDetector] downloading hand landmarker model -> {path}")
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


class handDetector:
    TIP_IDS = [4, 8, 12, 16, 20]

    def __init__(self, mode=False, maxHands=2, modelComplexity=1,
                 detectionCon=0.5, trackCon=0.5, model_path=None):
        # `mode`/`modelComplexity` are kept for signature-compatibility with the
        # tutorial; the Tasks API expresses the streaming case via RunningMode.
        self.maxHands = maxHands
        model_path = ensure_model(model_path or DEFAULT_MODEL_PATH)

        running_mode = (vision.RunningMode.IMAGE if mode
                        else vision.RunningMode.VIDEO)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_hand_presence_confidence=detectionCon,
            min_tracking_confidence=trackCon,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._video = not mode
        self._ts = 0                 # monotonic timestamp (ms) for VIDEO mode
        self.result = None
        self.lmList = []

    def _detect(self, imgRGB):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        if self._video:
            self._ts += 33           # ~30 fps; must be strictly increasing
            return self.landmarker.detect_for_video(mp_image, self._ts)
        return self.landmarker.detect(mp_image)

    def findHands(self, img, draw=True):
        """Run detection on a BGR frame; draw the skeleton if requested."""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self._detect(imgRGB)

        if draw and self.result and self.result.hand_landmarks:
            h, w, _ = img.shape
            for lms in self.result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(img, pts[a], pts[b], (255, 255, 255), 2)
                for p in pts:
                    cv2.circle(img, p, 4, (255, 0, 255), cv2.FILLED)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """Return (lmList, bbox); lmList = [[id, x, y], ...] in pixel coords."""
        xList, yList = [], []
        bbox = ()
        self.lmList = []

        if self.result and self.result.hand_landmarks and \
                handNo < len(self.result.hand_landmarks):
            h, w, _ = img.shape
            for idx, lm in enumerate(self.result.hand_landmarks[handNo]):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([idx, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)
            if xList and yList:
                bbox = (min(xList), min(yList), max(xList), max(yList))
                if draw:
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        return self.lmList, bbox

    def getHandedness(self, handNo=0):
        """Return 'Left' / 'Right' for the requested hand, or None.

        Note: labels reflect the original (un-mirrored) frame.
        """
        if self.result and self.result.handedness and \
                handNo < len(self.result.handedness):
            return self.result.handedness[handNo][0].category_name
        return None

    def fingersUp(self, handNo=0):
        """List of 5 ints (thumb..pinky), 1 = extended.

        The thumb test is orientation-independent: it does NOT depend on
        handedness, palm-facing (front vs back of hand), or whether the frame is
        mirrored — all of which flip the thumb's left/right sense and were the
        original source of misreads. It only uses relative distances between
        landmarks on the same hand. The four fingers use a tip-above-joint test,
        which assumes a roughly upright hand (the norm for fingerspelling)."""
        if not self.lmList or len(self.lmList) < 21:
            return []

        def d(a, b):
            return math.hypot(self.lmList[a][1] - self.lmList[b][1],
                              self.lmList[a][2] - self.lmList[b][2])

        fingers = []
        # Thumb is "open" when its tip sits farther from the far (pinky-base)
        # side of the hand than its own base joint does. An abducted thumb
        # sticks out sideways in both palm-in and palm-out views, so this holds
        # regardless of how the hand faces the camera.
        fingers.append(1 if d(4, 17) > d(2, 17) else 0)
        # Other four: tip above its PIP joint (smaller y) => extended.
        for tip in self.TIP_IDS[1:]:
            fingers.append(1 if self.lmList[tip][2] < self.lmList[tip - 2][2] else 0)
        return fingers


def main():
    """Standalone demo matching the tutorial: webcam + landmarks + FPS."""
    import time

    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector(maxHands=1)

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)  # selfie view
        img = detector.findHands(img)
        lmList, _ = detector.findPosition(img, draw=False)
        if lmList:
            print(lmList[4])  # thumb tip, as in the article

        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime else 0
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
