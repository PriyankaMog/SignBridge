"""Offline tests (no webcam, no MediaPipe, no TensorFlow needed).
Run from the project root:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_utils import assign_splits  # noqa: E402
from preprocessing.preprocess import (HAND_CONNECTIONS, HandPreprocessor,  # noqa: E402
                                      detect_lighting, detect_low_light,
                                      enhance_low_light, normalize,
                                      reduce_overexposure, square_crop)

LANDMARKS = np.array([
    (320, 400),
    (270, 370), (240, 330), (225, 295), (215, 265),
    (285, 320), (280, 270), (278, 235), (276, 205),
    (315, 310), (315, 255), (315, 215), (315, 185),
    (345, 318), (350, 268), (352, 235), (354, 208),
    (372, 335), (385, 295), (393, 268), (399, 245)], dtype=np.float32)


def synthetic_frame(brightness=1.0, shift=(0, 0), bg=(150, 160, 170), skin=(120, 150, 215)):
    rng = np.random.default_rng(0)
    frame = np.full((480, 640, 3), bg, np.uint8)
    frame = np.clip(frame + rng.integers(-8, 8, frame.shape), 0, 255).astype(np.uint8)
    pts = LANDMARKS + np.array(shift, dtype=np.float32)
    cv2.fillConvexPoly(frame, cv2.convexHull(pts[[0, 1, 5, 9, 13, 17]].astype(np.int32)), skin)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), skin, 26)
    return (frame * brightness).astype(np.uint8), pts


class FakeDetector:
    def __init__(self, pts):
        self.pts = pts

    def detect(self, frame):
        return self.pts


class LightingTests(unittest.TestCase):
    def test_detects_dark_and_bright(self):
        bright, _ = synthetic_frame(1.0)
        dark, _ = synthetic_frame(0.3)
        self.assertFalse(detect_low_light(bright)[0])
        self.assertTrue(detect_low_light(dark)[0])

    def test_enhancement_brightens_dark_only(self):
        dark, _ = synthetic_frame(0.3)
        self.assertGreater(enhance_low_light(dark).mean(), dark.mean() + 25)
        bright, _ = synthetic_frame(1.0)
        self.assertTrue(np.array_equal(enhance_low_light(bright), bright))


class ExposureTests(unittest.TestCase):
    def test_overexposed_detected_and_darkened(self):
        over, pts = synthetic_frame(bg=(235, 238, 240), skin=(200, 215, 245))
        self.assertEqual(detect_lighting(over)[0], "bright")
        self.assertLess(reduce_overexposure(over).mean(), over.mean() - 30)
        res = HandPreprocessor(detector=FakeDetector(pts)).process(over)
        self.assertEqual(res.lighting, "bright")
        self.assertTrue(res.was_enhanced)

    def test_normal_frame_untouched_by_overexposure_fix(self):
        frame, _ = synthetic_frame(0.8)          # mean ~128: normal, below the 140 target
        self.assertEqual(detect_lighting(frame)[0], "normal")
        self.assertTrue(np.array_equal(reduce_overexposure(frame), frame))

    def test_backlit_hand_is_brightened(self):
        # bright background, dark hand
        frame, pts = synthetic_frame(bg=(215, 220, 225), skin=(25, 30, 40))
        res = HandPreprocessor(detector=FakeDetector(pts)).process(frame)
        self.assertEqual(res.lighting, "backlit")
        plain = HandPreprocessor(detector=FakeDetector(pts), enhance=False).process(frame)
        hand = plain.image_bgr.mean(axis=2) < 80          # the (dark) hand pixels
        self.assertGreater(hand.sum(), 1000)
        self.assertGreater(res.image_bgr[hand].mean(), plain.image_bgr[hand].mean() + 40)

    def test_retry_with_local_contrast_when_first_detection_fails(self):
        frame, pts = synthetic_frame()

        class FlakyDetector:
            calls = 0

            def detect(self, f):
                FlakyDetector.calls += 1
                return None if FlakyDetector.calls == 1 else pts
        res = HandPreprocessor(detector=FlakyDetector()).process(frame)
        self.assertIsNotNone(res)
        self.assertEqual(FlakyDetector.calls, 2)


class CropTests(unittest.TestCase):
    def test_square_and_padded_at_border(self):
        frame, pts = synthetic_frame()
        near_edge = pts + np.array([-200, 60], dtype=np.float32)   # runs off the left edge
        crop, pc, _ = square_crop(frame, near_edge)
        self.assertEqual(crop.shape[0], crop.shape[1])
        self.assertTrue(np.all(pc >= -1) and np.all(pc <= crop.shape[0] + 1))


class PreprocessorTests(unittest.TestCase):
    def test_output_shape_and_dtype(self):
        frame, pts = synthetic_frame()
        res = HandPreprocessor(size=224, detector=FakeDetector(pts)).process(frame)
        self.assertEqual(res.image_bgr.shape, (224, 224, 3))
        self.assertEqual(res.image_rgb.dtype, np.uint8)
        self.assertFalse(res.was_enhanced)

    def test_dark_frame_is_enhanced(self):
        frame, pts = synthetic_frame(0.3)
        res = HandPreprocessor(detector=FakeDetector(pts)).process(frame)
        self.assertTrue(res.was_enhanced)

    def test_no_hand_returns_none(self):
        frame, _ = synthetic_frame()
        self.assertIsNone(HandPreprocessor(detector=FakeDetector(None)).process(frame))

    def test_background_removal_keeps_hand_drops_background(self):
        frame, pts = synthetic_frame()
        res = HandPreprocessor(remove_bg=True, detector=FakeDetector(pts)).process(frame)
        self.assertIsNotNone(res.mask)
        frac = (res.mask > 0).mean()
        self.assertTrue(0.10 < frac < 0.75, f"mask covers {frac:.2f} of crop")
        # background pixels are blanked, hand pixels are kept
        self.assertEqual(int(res.image_bgr[res.mask == 0].max()), 0)
        self.assertGreater(int(res.image_bgr[res.mask > 0].mean()), 40)
        self.assertEqual(res.as_bgra().shape, (224, 224, 4))

    def test_normalize_ranges(self):
        img = np.array([[[0, 127, 255]]], np.uint8)
        n = normalize(img, "mobilenet")
        self.assertAlmostEqual(float(n.min()), -1.0)
        self.assertAlmostEqual(float(n.max()), 1.0)
        self.assertAlmostEqual(float(normalize(img, "unit").max()), 1.0)


class SplitTests(unittest.TestCase):
    rows = [{"signer": f"s{i}", "session": f"s{i}_a"} for i in range(1, 5)]

    def test_groups_never_shared(self):
        m = assign_splits(self.rows, by="signer")
        self.assertEqual(set(m.values()), {"train", "val", "test"})
        self.assertEqual(len(m), 4)

    def test_explicit_assignment(self):
        m = assign_splits(self.rows, by="signer", val_groups=["s3"], test_groups=["s4"])
        self.assertEqual((m["s3"], m["s4"], m["s1"]), ("val", "test", "train"))

    def test_too_few_groups(self):
        with self.assertRaises(ValueError):
            assign_splits(self.rows[:2], by="signer")

    def test_deterministic(self):
        self.assertEqual(assign_splits(self.rows, seed=1), assign_splits(self.rows, seed=1))


if __name__ == "__main__":
    unittest.main()