"""Step 3 webcam test: see low-light enhancement + hand crop + background removal live.

Run from the project root:   python preprocessing/background_removal.py
Keys: S = save transparent PNG to processed_hands/hand.png,  Q = quit
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from preprocessing.preprocess import HandPreprocessor  # noqa: E402

OUT_DIR = Path("processed_hands")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Camera not working")
    pre = HandPreprocessor(remove_bg=True, static_image_mode=False,
                           min_detection_confidence=0.6, bg_value=255)
    last = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera not working")
                break
            frame = cv2.flip(frame, 1)          # mirror view, same as collect_data/demo
            res = pre.process(frame)

            display = frame.copy()
            if res is not None:
                last = res
                x1, y1, x2, y2 = res.bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, "Hand Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Background Removed", res.image_bgr)
                cv2.imshow("Transparent Hand (alpha mask)", res.mask if res.mask is not None
                           else res.image_bgr[:, :, 0])
                if res.mask is None:
                    cv2.putText(display, "Segmentation failed - plain crop used", (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                cv2.putText(display, "No Hand Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            light = "Low Lighting (enhanced)" if (res and res.was_enhanced) else "Normal Lighting"
            cv2.putText(display, light, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("SignBridge Camera", display)

            key = cv2.waitKey(1) & 0xFF      # single waitKey per loop so keypresses aren't lost
            if key == ord("q"):
                break
            if key == ord("s") and last is not None:
                cv2.imwrite(str(OUT_DIR / "hand.png"), last.as_bgra())
                print("Transparent hand saved to", OUT_DIR / "hand.png")
    finally:
        cap.release()
        pre.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
