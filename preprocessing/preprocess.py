"""SignBridge preprocessing: low-light handling -> hand detection -> crop -> (optional
background removal) -> resize -> normalise.

Up to MAX_HANDS (2) hands are detected. The crop is one square that contains every detected
hand, so one-hand and two-hand gestures can be mixed in the same vocabulary.

Quick use from the CNN / demo code (this is the function to hand to Pooja):

    from preprocessing.preprocess import preprocess_frame
    img = preprocess_frame(frame_bgr)          # uint8 RGB, 224x224x3  (or None if no hand)
    probs = model.predict(img[None].astype("float32"))   # model rescales internally

The saved Keras model contains its own Rescaling layer, so it expects pixel values
in 0-255. Use `normalize(img, "mobilenet")` only if you feed a different network.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import (BACKLIT_GAP, BRIGHT_THRESHOLD, CROP_PADDING, IMG_SIZE,
                    LOW_LIGHT_THRESHOLD)

# Maximum number of hands MediaPipe is asked to find (1 = old single-hand behaviour).
MAX_HANDS = 2

# Landmark index pairs of the 21-point MediaPipe hand model (the "bones").
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# --------------------------------------------------------------------------- hand helpers
def split_hands(pts):
    """(K*21, 2) landmark array -> (K, 21, 2). A single (21, 2) hand also works."""
    return np.asarray(pts, dtype=np.float32).reshape(-1, 21, 2)


def draw_bones(canvas, pts, thickness, value=1):
    """Draw the finger bones of every hand in `pts` onto a single-channel canvas."""
    for hand in split_hands(pts).astype(np.int32):
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, tuple(int(v) for v in hand[a]), tuple(int(v) for v in hand[b]),
                     value, thickness)


# --------------------------------------------------------------------------- lighting
def detect_low_light(frame_bgr, threshold=LOW_LIGHT_THRESHOLD):
    """Return (is_dark, mean_brightness)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return brightness < threshold, brightness


def enhance_low_light(frame_bgr, target=120.0, max_gamma=3.0, mean=None, clahe=True):
    """Brighten a dark frame: adaptive gamma towards `target` mean, then CLAHE on lightness.

    Gamma is derived from the frame's own brightness so a dim frame is lifted a lot and a
    slightly dark one only a little. Very dark frames amplify noise; that is a real limit.
    """
    if mean is None:     # `mean` can be passed in (e.g. brightness of the hand only)
        mean = float(np.mean(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)))
    if mean >= target:
        return frame_bgr
    mean = max(mean, 1.0)
    gamma = np.log(mean / 255.0) / np.log(target / 255.0)
    gamma = float(np.clip(gamma, 1.0, max_gamma))
    table = (((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255).astype("uint8")
    out = cv2.LUT(frame_bgr, table)
    if not clahe:
        return out

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def detect_lighting(frame_bgr):
    """Return ('low' | 'bright' | 'normal', mean_brightness) for the whole frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    b = float(np.mean(gray))
    return ("low" if b < LOW_LIGHT_THRESHOLD else "bright" if b > BRIGHT_THRESHOLD else "normal"), b


def reduce_overexposure(frame_bgr, target=140.0, min_gamma=0.15):
    """Darken an overexposed / washed-out frame with a gamma curve (inverse of the low-light fix).
    Pixels that are already clipped to pure white cannot be recovered - a real limit."""
    mean = float(np.mean(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)))
    if mean <= target:
        return frame_bgr
    mean = min(mean, 254.0)
    gamma = float(np.clip(np.log(mean / 255.0) / np.log(target / 255.0), min_gamma, 1.0))
    table = (((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255).astype("uint8")
    return cv2.LUT(frame_bgr, table)


def boost_local_contrast(frame_bgr):
    """CLAHE on the lightness channel only; helps shadowed / backlit hands."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------- detection
class MediaPipeHandDetector:
    """Thin wrapper around MediaPipe Hands.

    detect() returns the landmarks of every detected hand (up to `max_hands`) stacked into one
    (K*21, 2) array of pixel coordinates, hands ordered left to right in the image. With a single
    hand this is the same (21, 2) array as before.
    """

    def __init__(self, static_image_mode=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, max_hands=MAX_HANDS):
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This mediapipe version has no `mp.solutions` API. "
                "Run: pip install mediapipe==0.10.14")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        res = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return None
        hands = [np.array([[p.x * w, p.y * h] for p in hl.landmark], dtype=np.float32)
                 for hl in res.multi_hand_landmarks]
        hands.sort(key=lambda a: float(a[0, 0]))          # left-to-right by wrist position
        return np.concatenate(hands, axis=0)

    def close(self):
        self._hands.close()


# --------------------------------------------------------------------------- geometry
def square_crop(frame, pts, pad=CROP_PADDING):
    """Square crop centred on all hands with `pad` margin. Pads (edge-replicate) if it
    runs past the image border. Returns (crop, pts_in_crop, (x1, y1, x2, y2))."""
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    side = max(int(round(max(x_max - x_min, y_max - y_min) * (1 + 2 * pad))), 16)
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = x1 + side, y1 + side

    h, w = frame.shape[:2]
    top, left = max(0, -y1), max(0, -x1)
    bottom, right = max(0, y2 - h), max(0, x2 - w)
    src, ox, oy = frame, 0, 0
    if top or left or bottom or right:
        src = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_REPLICATE)
        ox, oy = left, top
    crop = src[y1 + oy:y2 + oy, x1 + ox:x2 + ox]
    return crop, pts - np.array([x1, y1], dtype=np.float32), (x1, y1, x2, y2)


# --------------------------------------------------------------------------- background
def hand_mask(crop_bgr, pts_crop, iterations=4):
    """GrabCut seeded by the landmarks (better than a plain bounding box). Works for 1 or 2 hands.

    sure foreground      = thick lines along the finger bones of every hand
    probable foreground  = convex hull of each hand
    probable background  = ring around the hulls
    sure background      = everything else
    Returns a uint8 mask (0/255) or None if GrabCut fails.
    """
    h, w = crop_bgr.shape[:2]
    side = max(h, w)

    hull = np.zeros((h, w), np.uint8)
    for hand in split_hands(pts_crop).astype(np.int32):      # one hull per hand, not one for both
        cv2.fillConvexPoly(hull, cv2.convexHull(hand), 1)
    k = max(3, int(0.10 * side)) | 1
    ring = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    bones = np.zeros((h, w), np.uint8)
    draw_bones(bones, pts_crop, max(3, int(0.035 * side)))

    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[ring == 1] = cv2.GC_PR_BGD
    mask[hull == 1] = cv2.GC_PR_FGD
    mask[bones == 1] = cv2.GC_FGD

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop_bgr, mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # Drop stray background patches: keep every blob that contains finger-bone pixels, so two
    # separate hands both survive (the old "largest blob only" rule would delete the smaller hand).
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out)
    if n > 1:
        keep = set(int(v) for v in np.unique(labels[(bones > 0) & (out > 0)]) if v > 0)
        if not keep:
            keep = {1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))}
        out = np.where(np.isin(labels, list(keep)), 255, 0).astype("uint8")
    return out if out.any() else None


# --------------------------------------------------------------------------- pipeline
@dataclass
class PreprocessResult:
    image_bgr: np.ndarray          # size x size x 3 uint8 (what gets saved to disk)
    bbox: tuple                    # (x1, y1, x2, y2) of the crop in the original frame
    landmarks: np.ndarray          # (K*21, 2) in original-frame pixels, K = number of hands
    brightness: float
    was_enhanced: bool
    mask: Optional[np.ndarray] = None   # size x size uint8 0/255 when remove_bg=True
    lighting: str = "normal"       # normal | low | bright | backlit

    @property
    def num_hands(self):
        return len(self.landmarks) // 21

    @property
    def image_rgb(self):
        return cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)

    def as_bgra(self):
        """Transparent-background version (needs remove_bg=True)."""
        out = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2BGRA)
        if self.mask is not None:
            out[:, :, 3] = self.mask
        return out


class HandPreprocessor:
    def __init__(self, size=IMG_SIZE, remove_bg=False, enhance=True, pad=CROP_PADDING,
                 detector=None, static_image_mode=True, min_detection_confidence=0.5,
                 bg_value=0, max_hands=MAX_HANDS):
        self.size, self.remove_bg, self.enhance = size, remove_bg, enhance
        self.pad, self.bg_value = pad, bg_value
        self._detector = detector
        self._static, self._conf, self._max_hands = (static_image_mode, min_detection_confidence,
                                                     max_hands)

    @property
    def detector(self):
        if self._detector is None:
            self._detector = MediaPipeHandDetector(self._static, self._conf,
                                                   max_hands=self._max_hands)
        return self._detector

    def process(self, frame_bgr) -> Optional[PreprocessResult]:
        lighting, brightness = detect_lighting(frame_bgr)
        frame, enhanced = frame_bgr, False
        if self.enhance and lighting == "low":
            frame, enhanced = enhance_low_light(frame_bgr), True

        # Detect on the original frame first. A bright frame can be a dark hand in front of a
        # window (backlit) - darkening the whole frame up front would crush that hand.
        pts = self.detector.detect(frame)
        if pts is None and self.enhance and lighting != "low":
            fixes = ([reduce_overexposure] if lighting == "bright" else []) + [boost_local_contrast]
            for fix in fixes:                      # retry: find washed-out / shadowed hands
                alt = fix(frame_bgr)
                pts = self.detector.detect(alt)
                if pts is not None:
                    frame, enhanced = alt, True
                    break
        if pts is None:
            return None

        crop, pts_crop, bbox = square_crop(frame, pts, self.pad)
        if crop.shape[0] < 16 or crop.shape[1] < 16:
            return None
        scale = self.size / crop.shape[0]
        crop = cv2.resize(crop, (self.size, self.size),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        pts_crop = pts_crop * scale

        if self.enhance and lighting != "low":
            # hand brightness measured along the finger bones of all hands (certainly skin,
            # unlike the hull, which also contains background between spread fingers)
            bones = np.zeros(crop.shape[:2], np.uint8)
            draw_bones(bones, pts_crop, max(3, int(0.03 * self.size)))
            hand_b = (float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[bones > 0].mean())
                      if bones.any() else brightness)
            if hand_b < LOW_LIGHT_THRESHOLD and brightness - hand_b > BACKLIT_GAP:
                # Backlit: hand much darker than its surroundings -> brighten the hand itself.
                # Gamma only: CLAHE on a bright-background/dark-hand crop darkens the hand.
                crop = enhance_low_light(crop, mean=hand_b, clahe=False)
                lighting, enhanced = "backlit", True
            elif lighting == "bright" and not enhanced:
                crop = reduce_overexposure(crop)    # genuine overexposure: darken the hand crop
                enhanced = True

        mask = None
        if self.remove_bg:
            mask = hand_mask(crop, pts_crop)
            if mask is not None:       # if GrabCut fails we fall back to the plain crop
                crop = np.where(mask[:, :, None] > 0, crop, self.bg_value).astype(np.uint8)

        return PreprocessResult(crop, bbox, pts, brightness, enhanced, mask, lighting)

    def close(self):
        if self._detector is not None and hasattr(self._detector, "close"):
            self._detector.close()


# --------------------------------------------------------------------------- public API
def preprocessor_from_json(path, **kwargs):
    """Build a HandPreprocessor with the SAME settings the training data was prepared with
    (written by prepare_dataset.py). Prevents train/serve skew, e.g. forgetting --remove_bg."""
    cfg = {"size": IMG_SIZE, "remove_bg": False, "enhance": True, "bg_value": 0}
    path = Path(path)
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    cfg.update(kwargs)
    return HandPreprocessor(**cfg)


def normalize(img_uint8, mode="mobilenet"):
    """mode 'mobilenet' -> float32 in [-1, 1]; 'unit' -> float32 in [0, 1]."""
    x = img_uint8.astype(np.float32)
    return x / 127.5 - 1.0 if mode == "mobilenet" else x / 255.0


_default_pre = None


def preprocess_frame(frame_bgr, size=IMG_SIZE, remove_bg=False):
    """One-call helper for the real-time pipeline. Returns an RGB uint8 (size, size, 3)
    array, or None if no hand was found. Keeps one shared MediaPipe instance."""
    global _default_pre
    if (_default_pre is None or _default_pre.size != size or _default_pre.remove_bg != remove_bg):
        _default_pre = HandPreprocessor(size=size, remove_bg=remove_bg, static_image_mode=False)
    res = _default_pre.process(frame_bgr)
    return None if res is None else res.image_rgb
