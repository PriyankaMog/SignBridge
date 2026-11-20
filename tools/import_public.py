"""Import a public folder-per-class image dataset (e.g. Kaggle ISL 36-class) so you can
build and debug the CNN pipeline before your own footage exists.

    python tools/import_public.py --src /path/to/ISL --per_class 200 [--classes A,B,C]

Images are copied into raw/public-NN_normal_plain/<class>/ and logged in metadata.csv.
Consecutive images are grouped into pseudo-"signers" (public-00, public-01, ...) of
--chunk images per class, because public sets carry no signer id. That keeps the split
leak-resistant for *pipeline testing*. These results say nothing about low-light or
unseen-user performance: report them as public-data pipeline tests only.

Note: the class names come from the folder names, so the model trained on this data
will have those classes (not config.GESTURES).
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from dataset_utils import append_metadata  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--per_class", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--classes", default="")
    args = ap.parse_args()

    src = Path(args.src)
    wanted = set(x for x in args.classes.split(",") if x)
    n_total = 0
    for cdir in sorted(p for p in src.iterdir() if p.is_dir()):
        if wanted and cdir.name not in wanted:
            continue
        imgs = sorted(p for p in cdir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for i, img in enumerate(imgs[:args.per_class]):
            signer = f"public-{i // args.chunk:02d}"
            session = f"{signer}_normal_plain"
            out = config.RAW_DIR / session / cdir.name / img.name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out)
            append_metadata(config.METADATA_CSV, {
                "filepath": out.relative_to(config.DATASET_DIR).as_posix(), "class": cdir.name,
                "signer": signer, "lighting": "normal", "background": "plain", "session": session})
            n_total += 1
        print(f"{cdir.name}: {min(len(imgs), args.per_class)} images")
    print(f"Imported {n_total} images into {config.RAW_DIR}")


if __name__ == "__main__":
    main()
