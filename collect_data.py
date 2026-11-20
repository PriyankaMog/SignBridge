"""Webcam data collection with the variation-matrix metadata from the briefing.

    python collect_data.py --signer signer01 --lighting dim --background cluttered
    python collect_data.py --signer signer01 --lighting normal --background plain --min_hands 2

One run = one *session* (signer + lighting + background). Images go to
    <dataset>/raw/<session>/<class>/<class>_<timestamp>.jpg
and every image is logged in <dataset>/metadata.csv.

Up to 2 hands are detected. Use --min_hands 2 in a session where every gesture is a two-hand
gesture, so frames where only one hand is seen are not saved. (Mixed one-hand/two-hand
vocabularies: collect them in separate runs with --classes and the matching --min_hands.)

Keys:  SPACE = shoot a burst for the current class   N / P = next / previous class
       Q = quit
Tips: during a burst slowly turn the hands and move them slightly nearer/further.
Only frames in which MediaPipe actually sees enough hands are saved.
Keep other people out of the camera view: a bystander's hand would be recorded too.
"""
import argparse
import re
import time
from collections import Counter

import cv2

import config
from dataset_utils import append_metadata, read_metadata
from preprocessing.preprocess import MediaPipeHandDetector, detect_low_light

SAFE = re.compile(r"^[A-Za-z0-9\-]+$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--signer", required=True, help="e.g. signer01 (letters/digits/hyphen only)")
    ap.add_argument("--lighting", required=True, choices=config.LIGHTING)
    ap.add_argument("--background", required=True, choices=config.BACKGROUNDS)
    ap.add_argument("--classes", default="all", help="comma list or 'all'")
    ap.add_argument("--burst", type=int, default=40, help="images per burst")
    ap.add_argument("--interval", type=float, default=0.2, help="seconds between saved frames")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--min_hands", type=int, default=1, choices=[1, 2],
                    help="save a frame only if at least this many hands are visible (default 1)")
    args = ap.parse_args()

    for v in (args.signer,):
        if not SAFE.match(v):
            ap.error("--signer may only contain letters, digits and '-'")
    classes = config.CLASSES if args.classes == "all" else args.classes.split(",")
    bad = [c for c in classes if c not in config.CLASSES]
    if bad:
        ap.error(f"unknown classes {bad}; valid: {config.CLASSES}")

    session = f"{args.signer}_{args.lighting}_{args.background}"
    counts = Counter(r["class"] for r in read_metadata(config.METADATA_CSV) if r["session"] == session)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")
    detector = MediaPipeHandDetector(static_image_mode=False, min_detection_confidence=0.6,
                                     max_hands=2)

    idx, shooting, countdown_end, saved_in_burst, last_save = 0, False, 0.0, 0, 0.0
    print(f"Session: {session}  -> {config.RAW_DIR / session}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)                  # mirror view (same in demo.py)
            raw = frame.copy()
            cls = classes[idx]
            g = config.PHRASES.get(cls, {})
            pts = detector.detect(frame)                # (K*21, 2) for K hands, or None
            n_hands = 0 if pts is None else len(pts) // 21
            _, bright = detect_low_light(frame)
            now = time.time()

            status, color = "SPACE = start burst", (255, 255, 255)
            if shooting:
                if now < countdown_end:
                    status, color = f"Get ready... {countdown_end - now:0.1f}s", (0, 200, 255)
                elif n_hands < args.min_hands:
                    status = (f"Need {args.min_hands} hand(s), seeing {n_hands} - not saving")
                    color = (0, 0, 255)
                else:
                    status, color = f"Recording {saved_in_burst}/{args.burst}", (0, 255, 0)
                    if now - last_save >= args.interval:
                        out_dir = config.RAW_DIR / session / cls
                        out_dir.mkdir(parents=True, exist_ok=True)
                        name = f"{cls}_{int(now * 1000)}.jpg"
                        cv2.imwrite(str(out_dir / name), raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        append_metadata(config.METADATA_CSV, {
                            "filepath": f"raw/{session}/{cls}/{name}", "class": cls,
                            "signer": args.signer, "lighting": args.lighting,
                            "background": args.background, "session": session})
                        counts[cls] += 1
                        saved_in_burst += 1
                        last_save = now
                        if saved_in_burst >= args.burst:
                            shooting = False
                            print(f"{cls}: burst done (total {counts[cls]})")

            if pts is not None:
                for x, y in pts:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            lines = [f"[{idx + 1}/{len(classes)}] {cls.upper()}  ({g.get('hand', '')})",
                     status, f"saved this class: {counts[cls]}   hands: {n_hands}   "
                     f"brightness: {bright:.0f}"
                     + ("  (LOW LIGHT)" if bright < config.LOW_LIGHT_THRESHOLD else ""),
                     "N/P class  SPACE burst  Q quit"]
            for i, t in enumerate(lines):
                cv2.putText(frame, t, (15, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            color if i == 1 else (255, 255, 0), 2)
            cv2.imshow("SignBridge - collect", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n") and not shooting:
                idx = (idx + 1) % len(classes)
            elif key == ord("p") and not shooting:
                idx = (idx - 1) % len(classes)
            elif key == ord(" ") and not shooting:
                shooting, saved_in_burst, countdown_end = True, 0, time.time() + 3.0
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
    print("Per-class counts this session:", dict(counts))


if __name__ == "__main__":
    main()