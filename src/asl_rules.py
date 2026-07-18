"""
Rule-based fingerspelling recognizer — built incrementally to a specific spec.

We're defining letters one small batch at a time and tuning them against the
live camera before adding more. Index order for the finger array is:

    [Thumb, Index, Middle, Ring, Pinky]

Letters implemented so far (exact definitions being matched):

    A = All fingers down; thumb OUTSIDE the fist, up (index between thumb & middle).
    B = All four fingers up, thumb tucked across (tip between index and pinky).
    C = Fingers together & curved; thumb held well OFF the fingertips (wide gap).
    D = Index up; others down; thumb touching the folded middle finger.
    E = All fingers down & SPREAD; thumb horizontal, below them, tucked close.
    F = Middle/ring/pinky up, index folded down to meet the thumb tip.
    G = Sideways; index + thumb extended horizontally, other three curled.
    H = Sideways; index + middle extended together, thumb tip x between joints 9,10.
    I = Pinky up, rest down, thumb wrapped so its tip is near the ring DIP (15).
    J = Pinky extended, pointing DOWN — the only shape with tip 20 below wrist 0.
    K = Index + middle up in a spread V (upright), thumb tip ABOVE both knuckles (5,9).
    L = Index up, others down, thumb horizontal, sticking out (away from middle).
    M = All fingers down; thumb tip in the PIP..DIP band, x between joints 10, 18.
    N = All fingers down; thumb tip above >=2 finger DIPs, x between joints 10, 18.
    O = Like C (fingers together & curved) but thumb tip CLOSE to the fingertips.
    P = Sideways K: index out to the side, middle angled down (spread), thumb between.
    Q = Sideways G angled DOWN: index + thumb extended diagonally down, rest curled.
    R = Index + middle up and CROSSED, ring/pinky down (fingers-crossed).
    T = All fingers down, thumb sticking UP between the index and middle fingers.
    U = Index + middle up TOGETHER (upright), ring/pinky down.
    V = Spread V like K but thumb tip BELOW both knuckles (tucked into the palm).
    W = Index + middle + ring up, pinky + thumb down.
    X = Index raised but CURLED into a claw (not straight), others down.
    Y = Thumb + pinky out, the middle three curled.
    S = (demo swap) Index up straight, others down, thumb on TOP of the folded
        knuckles. This is really the Z handshape, emitted as S: the true S (thumb
        in FRONT of a fist) needs depth the 2D landmarks don't give, and the demo
        needs an S, not a Z.

Plus two non-letter gestures: an open palm emits SPACE (word break) and a "devil
horns" (index + pinky up, middle + ring down) emits DELETE (backspace).

Up/down (from handDetector.fingersUp) assumes an upright hand, so the sideways
letters use an orientation-independent "extended" test. All `_*` thresholds are
tuning knobs, expressed relative to hand size.
"""

import numpy as np

from gesture_features import landmarks_to_xy

# Landmark ids
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
PIP_IDS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]     # 6, 10, 14, 18
DIP_IDS = [INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP]     # 7, 11, 15, 19

# Non-letter gestures. app.py maps a committed SPACE to buffer.add_space() and a
# committed DELETE to buffer.backspace().
SPACE = "SPACE"     # open palm — all fingers up, spread, thumb out
DELETE = "DELETE"   # "devil horns" — index + pinky up, middle + ring down

# --- tunables (as multiples of hand scale = wrist->middle-MCP length) -----
_T_TOUCH_MIDDLE = 0.55    # D: max thumb-tip -> middle-finger distance to be "touching"
_F_TOUCH_INDEX = 0.40     # F: max thumb-tip -> index-tip distance to be "touching"
_I_NEAR_RING = 0.40       # I: max thumb-tip -> ring-DIP (15) distance ("wrapped")
_V_SPREAD_MIN = 0.30      # index/middle fingertip gap above this = spread V (K/V), below = together (U/H)
_C_MIN_THUMB_GAP = 0.50   # C: min thumb-tip -> nearest-fingertip gap (wide C opening)
_O_MAX_GAP = 0.40         # O: max thumb-tip -> nearest-fingertip gap (thumb closes the ring)
_E_MAX_THUMB_GAP = 0.25   # E: max thumb-tip -> nearest-fingertip gap (thumb tucked to fingers)
_C_MAX_TIP_SPREAD = 0.50  # fingertip span below this = fingers "together"; above = "apart" (E)
_EXT_MIN = 0.50           # extended needs tip->own-MCP > this (not a claw) AND tip farther from wrist than MCP


def _scale(pts):
    """Stable hand-size reference: wrist -> middle-finger base knuckle."""
    return float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST])) + 1e-6


def _d(pts, a, b):
    return float(np.linalg.norm(pts[a] - pts[b]))


def _horizontal(vec):
    return abs(vec[0]) > abs(vec[1])


def _between(v, a, b):
    return min(a, b) <= v <= max(a, b)


def _extended(pts, tip, mcp, s):
    """A finger is 'extended' only when its tip is BOTH far from its own knuckle
    (so it isn't bent into a claw) AND farther from the wrist than the knuckle
    (so it isn't folded straight down — tip far from the knuckle but back toward
    the wrist). The two conditions together handle curled and folded fingers."""
    return (_d(pts, tip, mcp) / s > _EXT_MIN
            and _d(pts, tip, WRIST) > _d(pts, mcp, WRIST))


def metrics(lmList):
    """Debug helper: the scale-relative quantities the rules threshold on."""
    pts = landmarks_to_xy(lmList)
    if pts is None:
        return None
    s = _scale(pts)
    tips = [pts[t] for t in FINGER_TIPS]
    tip_span = max(float(np.linalg.norm(tips[i] - tips[j]))
                   for i in range(4) for j in range(i + 1, 4)) / s
    gap = min(float(np.linalg.norm(pts[THUMB_TIP] - t)) for t in tips) / s
    v = pts[THUMB_TIP] - pts[THUMB_MCP]
    thumb = "vert" if abs(v[1]) > abs(v[0]) else "horiz"
    ext = "".join(str(int(_extended(pts, tip, mcp, s)))
                  for tip, mcp in ((INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP),
                                   (RING_TIP, RING_MCP), (PINKY_TIP, PINKY_MCP)))
    return {"tip_span": tip_span, "gap": gap, "thumb": thumb, "ext": ext}


def classify(fingers, lmList):
    """Map (fingers, lmList) to a letter, or None."""
    if not fingers or len(fingers) != 5:
        return None
    pts = landmarks_to_xy(lmList)
    if pts is None:
        return None
    s = _scale(pts)
    t4 = pts[THUMB_TIP]

    tv = t4 - pts[THUMB_MCP]
    thumb_vertical = abs(tv[1]) > abs(tv[0])
    thumb_up = thumb_vertical and tv[1] < 0
    thumb_horizontal = not thumb_vertical

    _, index_up, middle_up, ring_up, pinky_up = (bool(f) for f in fingers)
    others_down = not (index_up or middle_up or ring_up or pinky_up)
    others_up = index_up and middle_up and ring_up and pinky_up

    index_ext = _extended(pts, INDEX_TIP, INDEX_MCP, s)
    middle_ext = _extended(pts, MIDDLE_TIP, MIDDLE_MCP, s)
    ring_ext = _extended(pts, RING_TIP, RING_MCP, s)
    pinky_ext = _extended(pts, PINKY_TIP, PINKY_MCP, s)

    tips = [pts[t] for t in FINGER_TIPS]
    tip_span = max(float(np.linalg.norm(tips[i] - tips[j]))
                   for i in range(4) for j in range(i + 1, 4)) / s
    fingers_together = tip_span < _C_MAX_TIP_SPREAD
    gap = min(float(np.linalg.norm(t4 - t)) for t in tips) / s

    thumb_x_in_10_18 = _between(t4[0], pts[MIDDLE_PIP][0], pts[PINKY_PIP][0])

    # ============ letter checks (specific -> general) ============

    # J: pinky extended, pointing DOWN.
    if (pinky_ext and pts[PINKY_TIP][1] > pts[PINKY_MCP][1]
            and pts[PINKY_TIP][1] > pts[WRIST][1]):
        return "J"

    # F: middle+ring+pinky up, index folded to meet the thumb tip.
    if middle_up and ring_up and pinky_up and not index_up:
        if _d(pts, THUMB_TIP, INDEX_TIP) / s < _F_TOUCH_INDEX:
            return "F"

    # W: index + middle + ring up, pinky down (thumb down).
    if index_up and middle_up and ring_up and not pinky_up:
        return "W"

    # DELETE: "devil horns" — index + pinky up, middle + ring down (thumb tucked).
    if index_up and pinky_up and not (middle_up or ring_up):
        return DELETE

    # I / Y: pinky up, rest down. I = thumb wrapped near ring DIP; Y = thumb out.
    if pinky_up and not (index_up or middle_up or ring_up):
        if _d(pts, THUMB_TIP, RING_DIP) / s < _I_NEAR_RING:
            return "I"
        return "Y"

    # R / U / K / V / P / H — index + middle extended, ring + pinky curled.
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        idx = pts[INDEX_TIP] - pts[INDEX_MCP]
        upright = not _horizontal(idx)
        spread = _d(pts, INDEX_TIP, MIDDLE_TIP) / s > _V_SPREAD_MIN
        crossed = ((pts[INDEX_MCP][0] - pts[MIDDLE_MCP][0])
                   * (pts[INDEX_TIP][0] - pts[MIDDLE_TIP][0])) < 0
        if crossed:
            return "R"
        if upright:
            if not spread:
                return "U"
            # K = thumb tip pokes UP, above both finger knuckles (5 and 9), into
            # the base of the V. V = thumb tip BELOW both, tucked into the palm.
            if t4[1] < pts[INDEX_MCP][1] and t4[1] < pts[MIDDLE_MCP][1]:
                return "K"
            return "V"
        else:                    # sideways
            if spread:
                return "P"       # sideways K
            if _between(t4[0], pts[MIDDLE_MCP][0], pts[MIDDLE_PIP][0]):
                return "H"

    # B / SPACE: all four fingers up.
    if others_up:
        if thumb_horizontal and _between(t4[0], pts[INDEX_TIP][0], pts[PINKY_TIP][0]):
            return "B"
        if not fingers_together:
            return SPACE

    # Q / G: index (and thumb) extended, other three curled.
    if index_ext and not (middle_ext or ring_ext or pinky_ext):
        idx = pts[INDEX_TIP] - pts[INDEX_MCP]
        if idx[1] > 0 and idx[1] > 0.3 * abs(idx[0]):     # index angled downward
            return "Q"
        if _horizontal(idx) and thumb_horizontal:          # index roughly level
            return "G"

    # T: all fingers down, thumb sticking UP between the index and middle fingers.
    if (others_down and _between(t4[0], pts[INDEX_PIP][0], pts[MIDDLE_PIP][0])
            and t4[1] < pts[MIDDLE_PIP][1]):
        return "T"

    # M / N: all fingers down, thumb tucked INSIDE (x between joints 10 and 18).
    if others_down and thumb_x_in_10_18:
        pip_y = [pts[i][1] for i in PIP_IDS]
        dip_y = [pts[i][1] for i in DIP_IDS]
        if min(pip_y) <= t4[1] <= max(dip_y):
            return "M"
        if sum(1 for d in dip_y if t4[1] < d) >= 2:
            return "N"

    # C / O: fingers together & curved. C = thumb off (wide gap); O = ring closed.
    if fingers_together:
        if not thumb_up and gap > _C_MIN_THUMB_GAP:
            return "C"
        if gap < _O_MAX_GAP:
            return "O"

    # X: index raised but CURLED into a claw (not extended), others down.
    if not (middle_up or ring_up or pinky_up):
        if not index_ext and pts[INDEX_TIP][1] < pts[MIDDLE_TIP][1] - 0.2 * s:
            return "X"

    # Z / D / L: index up straight, others down.
    if index_up and not (middle_up or ring_up or pinky_up):
        knuckle_y = min(pts[MIDDLE_MCP][1], pts[RING_MCP][1], pts[PINKY_MCP][1])
        if t4[1] < knuckle_y and _between(t4[0], pts[MIDDLE_MCP][0], pts[PINKY_MCP][0]):
            return "S"           # demo swap: Z handshape emitted as S (see docstring)
        touch = min(_d(pts, THUMB_TIP, j)
                    for j in (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)) / s
        if touch < _T_TOUCH_MIDDLE:
            return "D"
        if thumb_horizontal:
            return "L"

    # A / E: all fingers down.
    if others_down:
        if thumb_up and _between(pts[INDEX_MCP][0], t4[0], pts[MIDDLE_MCP][0]):
            return "A"
        tips_y = float(np.mean([t[1] for t in tips]))
        if (not fingers_together and thumb_horizontal
                and t4[1] > tips_y and gap < _E_MAX_THUMB_GAP):
            return "E"

    return None
