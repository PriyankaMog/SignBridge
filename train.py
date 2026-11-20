"""Train the SignBridge classifier: MobileNetV2 transfer learning on the processed crops.

    python train.py                      # uses <dataset>/processed/{train,val,test}
    python train.py --epochs_head 5 --epochs_ft 20 --batch 32

Phase 1: backbone frozen, train the new softmax head.
Phase 2: unfreeze the last --unfreeze backbone layers (BatchNorm stays frozen), lr 1e-4.
Augmentation (rotation +-25 deg, zoom, shift, contrast, brightness jitter darker AND brighter)
is part of the model graph, so it is active ONLY while training - val/test stay untouched.
The saved model has a Rescaling layer inside: feed it float pixels in 0-255.
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=str(config.PROCESSED_DIR))
    ap.add_argument("--size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs_head", type=int, default=10)
    ap.add_argument("--epochs_ft", type=int, default=30)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=1e-4)
    ap.add_argument("--unfreeze", type=int, default=30, help="backbone layers to fine-tune")
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--rotation", type=float, default=25.0, help="max augmentation rotation, degrees")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.utils.set_random_seed(args.seed)
    root, size = Path(args.data), (args.size, args.size)
    for s in ("train", "val"):
        if not (root / s).is_dir():
            raise SystemExit(f"{root / s} not found. Run prepare_dataset.py first.")

    def load(split, shuffle):
        return keras.utils.image_dataset_from_directory(
            root / split, image_size=size, batch_size=args.batch, label_mode="int",
            shuffle=shuffle, seed=args.seed)

    train_ds, val_ds = load("train", True), load("val", False)
    class_names = train_ds.class_names
    if val_ds.class_names != class_names:
        raise SystemExit(f"train/val classes differ: {class_names} vs {val_ds.class_names}. "
                         "Every split needs every class - collect more data or change the split.")
    n_classes = len(class_names)
    print(f"{n_classes} classes: {class_names}")

    counts = Counter(Path(p).parent.name for p in train_ds.file_paths)
    print("train images per class:", dict(counts))
    class_weight = None
    if not args.no_class_weights:
        total = sum(counts.values())
        class_weight = {i: total / (n_classes * counts[c]) for i, c in enumerate(class_names)}

    autotune = tf.data.AUTOTUNE
    train_pf, val_pf = train_ds.prefetch(autotune), val_ds.cache().prefetch(autotune)

    augment = keras.Sequential([
        layers.RandomRotation(args.rotation / 360.0, fill_mode="nearest"),
        layers.RandomZoom((-0.15, 0.10)),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.3),
        layers.RandomBrightness(factor=(-0.4, 0.4), value_range=(0, 255)),
    ], name="augmentation")

    base = keras.applications.MobileNetV2(input_shape=size + (3,), include_top=False,
                                          weights="imagenet")
    base.trainable = False
    inputs = keras.Input(shape=size + (3,), name="image")
    x = augment(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1.0, name="rescale")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="gesture")(x)
    model = keras.Model(inputs, outputs, name="signbridge")

    def compile_(lr):
        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    config.MODELS_DIR.mkdir(exist_ok=True)
    best_path = config.MODELS_DIR / "best.keras"

    # ---- Phase 1: train the head
    compile_(args.lr_head)
    h1 = model.fit(train_pf, validation_data=val_pf, epochs=args.epochs_head,
                   class_weight=class_weight, callbacks=[
                       keras.callbacks.EarlyStopping("val_accuracy", patience=4,
                                                     restore_best_weights=True),
                       keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_accuracy",
                                                       save_best_only=True)])
    best1 = max(h1.history["val_accuracy"])

    # ---- Phase 2: fine-tune the top of the backbone
    base.trainable = True
    for layer in base.layers[:-args.unfreeze]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_(args.lr_ft)
    h2 = model.fit(train_pf, validation_data=val_pf, epochs=args.epochs_ft,
                   class_weight=class_weight, callbacks=[
                       keras.callbacks.EarlyStopping("val_accuracy", patience=6,
                                                     restore_best_weights=True),
                       keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=3,
                                                         min_lr=1e-6),
                       keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_accuracy",
                                                       save_best_only=True,
                                                       initial_value_threshold=best1)])

    # ---- Keep the best model seen across both phases
    best = keras.models.load_model(str(best_path))
    final_path = config.MODELS_DIR / "signbridge.keras"
    best.save(str(final_path))
    best_path.unlink(missing_ok=True)
    (config.MODELS_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2),
                                                        encoding="utf-8")
    pre_cfg = config.DATASET_DIR / "preprocess_config.json"
    if pre_cfg.exists():
        shutil.copy(pre_cfg, config.MODELS_DIR / "preprocess_config.json")

    history = {k: [float(v) for v in h1.history[k] + h2.history[k]] for k in h1.history
               if k in h2.history}
    (config.RESULTS_DIR / "history.json").write_text(json.dumps(history), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a, key in zip(ax, ("accuracy", "loss")):
            a.plot(history[key], label="train")
            a.plot(history["val_" + key], label="val")
            a.axvline(len(h1.history["loss"]) - 0.5, ls="--", c="gray", label="fine-tune starts")
            a.set_title(key)
            a.set_xlabel("epoch")
            a.legend()
        fig.tight_layout()
        fig.savefig(config.RESULTS_DIR / "training_curves.png", dpi=150)
    except Exception as e:  # plotting must never break training
        print("Could not plot curves:", e)

    tr_acc, va_acc = max(history["accuracy"]), max(history["val_accuracy"])
    print(f"\nbest train acc {tr_acc:.3f} | best val acc {va_acc:.3f}")
    if tr_acc - va_acc > 0.15:
        print("WARNING: large train/val gap = overfitting. Collect more varied data or "
              "augment more; do not just add epochs.")
    print(f"Saved {final_path}. Now run: python evaluate.py")


if __name__ == "__main__":
    main()
