"""Evaluate the trained model on the held-out test split and write the poster numbers.

    python evaluate.py                        # accuracy, confusion matrix, by lighting/background/signer, latency
    python evaluate.py --simulate_lowlight 150   # extra experiment on up to 150 raw test images

Outputs in results/: results.md, results.json, confusion_matrix.png, confusion_matrix.csv
Everything in results.md is MEASURED on your data. The low-light experiment is SIMULATED
(darkened + noisy copies of normal-light photos) and is labelled as such - report it as that.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config
from dataset_utils import read_metadata
from preprocessing.preprocess import preprocessor_from_json


def simulate_low_light(img_bgr, gamma=2.0, gain=0.6, noise_sigma=4.0, seed=0):
    out = ((img_bgr / 255.0) ** gamma) * 255.0 * gain
    out = out + np.random.default_rng(seed).normal(0, noise_sigma, img_bgr.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def group_accuracy(rows, correct, key):
    agg = defaultdict(lambda: [0, 0])
    for r, c in zip(rows, correct):
        k = r.get(key) or "unknown"
        agg[k][0] += int(c)
        agg[k][1] += 1
    return {k: {"n": n, "accuracy": hit / n} for k, (hit, n) in sorted(agg.items())}


def timing(fn, n):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
            "fps_at_median": float(1000 / np.median(ts))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=str(config.MODELS_DIR / "signbridge.keras"))
    ap.add_argument("--data", default=str(config.PROCESSED_DIR))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--latency_runs", type=int, default=100)
    ap.add_argument("--simulate_lowlight", type=int, default=0, metavar="N")
    args = ap.parse_args()

    from tensorflow import keras
    model = keras.models.load_model(args.model)
    class_names = json.loads((config.MODELS_DIR / "class_names.json").read_text(encoding="utf-8"))
    size = model.input_shape[1]
    n_cls = len(class_names)

    try:
        ds = keras.utils.image_dataset_from_directory(
            Path(args.data) / "test", image_size=(size, size), batch_size=args.batch,
            label_mode="int", shuffle=False, class_names=class_names)
    except ValueError as e:
        raise SystemExit(f"Test split problem: {e}\nEvery class needs test images.")
    files = [str(Path(p).resolve()) for p in ds.file_paths]
    y_true = np.concatenate([y.numpy() for _, y in ds])
    probs = model.predict(ds, verbose=0)
    y_pred = probs.argmax(1)
    conf = probs.max(1)
    correct = y_pred == y_true

    meta = {str((config.DATASET_DIR / r["processed"]).resolve()): r
            for r in read_metadata(config.METADATA_CSV) if r["processed"]}
    rows = [meta.get(f, {}) for f in files]

    cm = np.zeros((n_cls, n_cls), int)
    np.add.at(cm, (y_true, y_pred), 1)
    per_class = {}
    for i, c in enumerate(class_names):
        tp, support, predicted = cm[i, i], cm[i].sum(), cm[:, i].sum()
        per_class[c] = {"n": int(support), "recall": float(tp / support) if support else None,
                        "precision": float(tp / predicted) if predicted else None}

    off = cm.copy()
    np.fill_diagonal(off, 0)
    top_conf = [(class_names[i], class_names[j], int(off[i, j]))
                for i, j in zip(*np.unravel_index(np.argsort(off, axis=None)[::-1][:5], off.shape))
                if off[i, j] > 0]

    thr = config.CONF_THRESHOLD
    mask = conf >= thr
    report = {
        "n_test_images": int(len(y_true)),
        "overall_accuracy": float(correct.mean()),
        "confidence_threshold": thr,
        "coverage_at_threshold": float(mask.mean()),
        "accuracy_at_threshold": float(correct[mask].mean()) if mask.any() else None,
        "by_lighting": group_accuracy(rows, correct, "lighting"),
        "by_background": group_accuracy(rows, correct, "background"),
        "by_signer": group_accuracy(rows, correct, "signer"),
        "per_class": per_class,
        "top_confusions": top_conf,
    }

    # ---- latency (model only, then full pipeline)
    x1 = np.zeros((1, size, size, 3), np.float32)
    for _ in range(10):
        model(x1, training=False)
    report["latency_model_only"] = timing(lambda: model(x1, training=False).numpy(),
                                          args.latency_runs)

    pre = preprocessor_from_json(config.MODELS_DIR / "preprocess_config.json",
                                 static_image_mode=True)
    test_raw = [r for r in read_metadata(config.METADATA_CSV) if r["split"] == "test"]
    imgs = [cv2.imread(str(config.DATASET_DIR / r["filepath"])) for r in test_raw[:60]]
    imgs = [i for i in imgs if i is not None]
    if imgs:
        it = iter(range(10 ** 9))

        def e2e():
            frame = imgs[next(it) % len(imgs)]
            res = pre.process(frame)
            if res is not None:
                model(res.image_rgb[None].astype(np.float32), training=False).numpy()
        e2e()
        report["latency_full_pipeline"] = timing(e2e, min(args.latency_runs, 3 * len(imgs)))
        report["latency_note"] = ("full pipeline = low-light fix + MediaPipe (detection every "
                                  "frame, conservative) + crop/resize + model, on THIS machine")

    # ---- simulated low light (optional)
    if args.simulate_lowlight and test_raw:
        sub = test_raw[:args.simulate_lowlight]
        res = {"label": "SIMULATED (darkened + noise copies of existing photos)", "n": len(sub)}
        for name, enhance, dark in (("normal_light", True, False),
                                    ("dark_with_enhancement", True, True),
                                    ("dark_without_enhancement", False, True)):
            p = preprocessor_from_json(config.MODELS_DIR / "preprocess_config.json",
                                       static_image_mode=True, enhance=enhance)
            hits = found = 0
            for r in sub:
                img = cv2.imread(str(config.DATASET_DIR / r["filepath"]))
                if img is None:
                    continue
                if dark:
                    img = simulate_low_light(img)
                out = p.process(img)
                if out is None:
                    continue
                found += 1
                pred = int(model(out.image_rgb[None].astype(np.float32), training=False)
                           .numpy().argmax())
                hits += class_names[pred] == r["class"]
            p.close()
            res[name] = {"hand_detected": found / len(sub),
                         "accuracy_on_detected": hits / found if found else None}
        report["simulated_low_light"] = res
    pre.close()

    # ---- write outputs
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savetxt(config.RESULTS_DIR / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(class_names), comments="")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(6, n_cls * 0.7),) * 2)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n_cls), class_names, rotation=60, ha="right")
        ax.set_yticks(range(n_cls), class_names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(n_cls):
            for j in range(n_cls):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        fig.tight_layout()
        fig.savefig(config.RESULTS_DIR / "confusion_matrix.png", dpi=150)
    except Exception as e:
        print("Could not plot confusion matrix:", e)

    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}%"

    md = ["# SignBridge results (measured on the held-out test split)", "",
          f"- Test images: **{report['n_test_images']}**",
          f"- Overall accuracy: **{pct(report['overall_accuracy'])}**",
          f"- At confidence >= {thr:.0%}: accuracy {pct(report['accuracy_at_threshold'])} "
          f"on {pct(report['coverage_at_threshold'])} of frames", ""]
    for title, key in (("Accuracy by lighting condition", "by_lighting"),
                       ("Accuracy by background", "by_background"),
                       ("Accuracy by signer", "by_signer")):
        md += [f"## {title}", "", "| group | n | accuracy |", "|---|---|---|"]
        md += [f"| {k} | {v['n']} | {pct(v['accuracy'])} |" for k, v in report[key].items()] + [""]
    md += ["## Per class", "", "| class | n | recall | precision |", "|---|---|---|---|"]
    md += [f"| {c} | {v['n']} | {pct(v['recall'])} | {pct(v['precision'])} |"
           for c, v in per_class.items()]
    md += ["", "Most frequent confusions (true -> predicted): "
           + (", ".join(f"{a}->{b} x{n}" for a, b, n in top_conf) or "none"), "",
           "## Latency (this machine)", "",
           f"- Model only: {report['latency_model_only']['median_ms']:.1f} ms median, "
           f"{report['latency_model_only']['p95_ms']:.1f} ms p95"]
    if "latency_full_pipeline" in report:
        lf = report["latency_full_pipeline"]
        md.append(f"- Full pipeline: {lf['median_ms']:.1f} ms median, {lf['p95_ms']:.1f} ms p95 "
                  f"(~{lf['fps_at_median']:.0f} FPS)")
    if "simulated_low_light" in report:
        s = report["simulated_low_light"]
        md += ["", f"## Low-light experiment - {s['label']}", "",
               "| condition | hand detected | accuracy (detected) |", "|---|---|---|"]
        md += [f"| {k} | {pct(s[k]['hand_detected'])} | {pct(s[k]['accuracy_on_detected'])} |"
               for k in ("normal_light", "dark_with_enhancement", "dark_without_enhancement")]
    (config.RESULTS_DIR / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
