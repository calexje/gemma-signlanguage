"""
Turn the 21 hand landmarks into a feature vector that is invariant to where the
hand is in the frame (translation), how big it appears (scale), and how it's
rotated in-plane. That invariance is what lets a simple KNN/MLP learn ASL
fingerspelling shapes from only a handful of samples per letter.

Pipeline, given pixel landmarks [[id, x, y], ...] (21 points):
  1. translate  — move the wrist (landmark 0) to the origin
  2. rotate     — align the wrist -> middle-finger-MCP vector to "up"
  3. scale      — divide by that vector's length (a stable hand-size proxy)
  4. flatten    — 21 (x, y) pairs -> a 42-D vector
"""

import numpy as np

WRIST = 0
MIDDLE_MCP = 9  # base knuckle of the middle finger — a stable size reference


def landmarks_to_xy(lmList):
    """Extract an (21, 2) float array of pixel coords from a handDetector lmList."""
    if not lmList or len(lmList) < 21:
        return None
    return np.array([[p[1], p[2]] for p in lmList[:21]], dtype=np.float64)


def normalize(pts):
    """Translation/rotation/scale-normalize an (21, 2) landmark array.

    Returns the normalized (21, 2) array, or None if the hand is degenerate.
    """
    if pts is None:
        return None

    # 1. translate so the wrist sits at the origin
    pts = pts - pts[WRIST]

    # reference vector: wrist -> middle-finger base knuckle
    ref = pts[MIDDLE_MCP]
    scale = np.linalg.norm(ref)
    if scale < 1e-6:
        return None

    # 2. rotate so that reference vector points "up" (-y in image coords)
    theta = np.arctan2(ref[1], ref[0])          # current angle of ref
    target = -np.pi / 2                          # up in image space
    phi = target - theta
    c, s = np.cos(phi), np.sin(phi)
    rot = np.array([[c, -s], [s, c]])
    pts = pts @ rot.T

    # 3. scale to unit reference length
    pts = pts / scale
    return pts


def landmarks_to_features(lmList):
    """Full pipeline: handDetector lmList -> 42-D feature vector (or None)."""
    pts = normalize(landmarks_to_xy(lmList))
    if pts is None:
        return None
    return pts.flatten()


FEATURE_DIM = 42
