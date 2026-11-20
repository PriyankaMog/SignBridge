"""Real-time demo: webcam -> low-light fix -> MediaPipe hand crop -> model -> text (+ speech).

    python demo.py [--camera 0] [--no_speech]

A gesture is ACCEPTED only when it stays the top prediction, above the confidence
threshold, for HOLD_FRAMES frames in a row (prevents flicker). Accepted phrases build a
sentence. Keys:  S = speak sentence   C = clear   Q = quit
"""
import argparse
import json
import threading
import time
from collections import deque

import cv2
import numpy as np

import config
from preprocessing.preprocess import preprocessor_from_json


class Speaker:
    def __init__(self, enabled=True):
        self.ok = False
        self._lock = threading.Lock()
        if enabled:
            try:
                import pyttsx3  # noqa: F401
                self.ok = True
            except Exception:
                print("pyttsx3 not available - speech disabled (pip install pyttsx3)")

    def say(self, text):
        if not (self.ok and text):
            return

        def run():
            with self._lock:
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print("Speech failed:", e)
        threading.Thread(target=run, daemon=True).start()


def put(img, text, y, color=(255, 255, 255), scale=0.7):
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(config.MODELS_DIR / "signbridge.keras"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=config.CONF_THRESHOLD)
    ap.add_argument("--hold", type=int, default=config.HOLD_FRAMES)
    ap.add_argument("--no_speech", action="store_true")
    args = ap.parse_args()

    from tensorflow import keras
    model = keras.models.load_model(args.model)
    names = json.loads((config.MODELS_DIR / "class_names.json").read_text(encoding="utf-8"))
    pre = preprocessor_from_json(config.MODELS_DIR / "preprocess_config.json",
                                 static_image_mode=False, min_detection_confidence=0.6)
    speaker = Speaker(not args.no_speech)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")

    sentence, streak_cls, streak_n, last_accepted, missing = [], None, 0, None, 0
    lat = deque(maxlen=30)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)       # same mirror view used when collecting data
            t0 = time.perf_counter()
            res = pre.process(frame)
            label, conf = None, 0.0
            if res is not None:
                probs = model(res.image_rgb[None].astype(np.float32), training=False).numpy()[0]
                k = int(probs.argmax())
                label, conf = names[k], float(probs[k])
            lat.append((time.perf_counter() - t0) * 1000)

            missing = 0 if res is not None else missing + 1
            if missing >= 10:
                last_accepted = None        # hand left the frame: the same sign may repeat
            if res is not None and conf >= args.threshold:
                streak_n = streak_n + 1 if label == streak_cls else 1
                streak_cls = label
                if streak_n >= args.hold and label != last_accepted:
                    last_accepted = label
                    info = config.PHRASES.get(label, {"speech": label, "emoji": ""})
                    sentence.append(info["speech"])
                    sentence[:] = sentence[-4:]
                    print(f"{info.get('emoji', '')}  {label}: {info['speech']}  ({conf:.0%})")
            else:
                streak_cls, streak_n = None, 0

            if res is not None:
                x1, y1, x2, y2 = res.bbox
                good = conf >= args.threshold
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if good else (0, 165, 255), 2)
                put(frame, f"{label}  {conf:.0%}" if good else f"uncertain ({label} {conf:.0%})",
                    35, (0, 255, 0) if good else (0, 165, 255), 0.9)
                th = cv2.resize(res.image_bgr, (96, 96))
                frame[10:106, frame.shape[1] - 106:frame.shape[1] - 10] = th
            else:
                put(frame, "No hand", 35, (0, 0, 255), 0.9)
            if res is not None and res.was_enhanced:
                put(frame, "low light - enhanced", 65, (255, 255, 0), 0.6)
            put(frame, "  ".join(sentence) or "(sentence empty)", frame.shape[0] - 45)
            put(frame, f"{np.mean(lat):.0f} ms/frame   S speak  C clear  Q quit",
                frame.shape[0] - 15, (200, 200, 200), 0.55)
            cv2.imshow("SignBridge", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                sentence.clear()
            if key == ord("s"):
                speaker.say(". ".join(sentence))
    finally:
        cap.release()
        pre.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
