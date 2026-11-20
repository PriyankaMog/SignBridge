"""raw/ -> processed/{train,val,test}/<class>/  (hand crop, low-light fix, resize)
and update metadata.csv with the split of every image.

    python prepare_dataset.py                         # split by signer (recommended)
    python prepare_dataset.py --split_by session      # only if you have < 3 signers
    python prepare_dataset.py --val_signers signer03 --test_signers signer04
    python prepare_dataset.py --remove_bg             # also apply GrabCut background removal

Splitting is done on whole signers (or sessions), never on random images, so
near-identical frames from one burst cannot leak between train and test.
"""
import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2

import config
from dataset_utils import (assign_splits, read_metadata, split_coverage,
                           write_metadata)
from preprocessing.preprocess import HandPreprocessor


def scan_raw():
    """Rebuild metadata from folder names raw/<signer>_<lighting>_<background>/<class>/."""
    rows = []
    for img in sorted(config.RAW_DIR.glob("*/*/*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        session, cls = img.parent.parent.name, img.parent.name
        parts = session.split("_")
        signer, lighting, bg = (parts + ["unknown"] * 3)[:3]
        rows.append({"filepath": img.relative_to(config.DATASET_DIR).as_posix(), "class": cls,
                     "signer": signer, "lighting": lighting, "background": bg, "session": session})
    return rows


def parse_list(s):
    return [x for x in (s or "").split(",") if x]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split_by", choices=["signer", "session"], default="signer")
    ap.add_argument("--val_signers", default="", help="comma list (or sessions if --split_by session)")
    ap.add_argument("--test_signers", default="")
    ap.add_argument("--size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--remove_bg", action="store_true")
    ap.add_argument("--no_enhance", action="store_true", help="disable low-light enhancement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_conf", type=float, default=0.4, help="MediaPipe detection confidence")
    args = ap.parse_args()

    rows = read_metadata(config.METADATA_CSV) or scan_raw()
    if not rows:
        raise SystemExit(f"No images found under {config.RAW_DIR}. Run collect_data.py first.")

    mapping = assign_splits(rows, by=args.split_by, seed=args.seed,
                            val_groups=parse_list(args.val_signers),
                            test_groups=parse_list(args.test_signers))
    print(f"Split by {args.split_by}:")
    for g, s in sorted(mapping.items(), key=lambda kv: (kv[1], kv[0])):
        print(f"  {s:5s} <- {g}")

    if config.PROCESSED_DIR.exists():
        shutil.rmtree(config.PROCESSED_DIR)
    pre = HandPreprocessor(size=args.size, remove_bg=args.remove_bg, enhance=not args.no_enhance,
                           static_image_mode=True, min_detection_confidence=args.min_conf)

    kept, failed = [], []
    for i, r in enumerate(rows, 1):
        src = config.DATASET_DIR / r["filepath"]
        img = cv2.imread(str(src))
        res = pre.process(img) if img is not None else None
        r["split"] = mapping[r[args.split_by]]
        if res is None:
            r["split"], r["processed"] = "skipped", ""
            failed.append(r["filepath"])
        else:
            rel = Path("processed") / r["split"] / r["class"] / f"{r['session']}__{src.stem}.jpg"
            out = config.DATASET_DIR / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), res.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            r["processed"] = rel.as_posix()
            kept.append(r)
        if i % 200 == 0:
            print(f"  processed {i}/{len(rows)}")
    pre.close()

    write_metadata(config.METADATA_CSV, rows)
    (config.DATASET_DIR / "preprocess_config.json").write_text(json.dumps(
        {"size": args.size, "remove_bg": args.remove_bg, "enhance": not args.no_enhance,
         "bg_value": 0}, indent=2), encoding="utf-8")
    if failed:
        (config.DATASET_DIR / "failed_images.txt").write_text("\n".join(failed), encoding="utf-8")

    print(f"\nKept {len(kept)} / {len(rows)} images; {len(failed)} had no detectable hand "
          f"(listed in failed_images.txt)")
    cov = split_coverage(rows)
    all_classes = sorted({r["class"] for r in rows})
    print(f"\n{'class':12s} {'train':>6s} {'val':>6s} {'test':>6s}")
    for c in all_classes:
        print(f"{c:12s} {cov['train'][c]:6d} {cov['val'][c]:6d} {cov['test'][c]:6d}")
    for s in ("train", "val", "test"):
        missing = [c for c in all_classes if cov[s][c] == 0]
        if missing:
            print(f"WARNING: split '{s}' has no images for: {missing}")

    by_light = defaultdict(Counter)
    for r in kept:
        by_light[r["split"]][r["lighting"]] += 1
    print("\nImages per split by lighting condition:")
    for s in ("train", "val", "test"):
        print(f"  {s:5s} {dict(by_light[s])}")
    print("\nPoster numbers: classes =", len(all_classes), "| total images =", len(kept),
          "| signers =", len({r['signer'] for r in kept}))


if __name__ == "__main__":
    main()
