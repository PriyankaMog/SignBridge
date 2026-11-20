# SignBridge

Real-time static hand-gesture recognition from a standard webcam: no gloves, no sensors.
A gesture is converted to text and, optionally, speech. Works in reduced lighting thanks to a
low-light enhancement + hand-isolation preprocessing stage.

Mini project - Sahyadri College of Engineering & Management, CSE 2026-27.
Team: Likhitha, Samiksha Naik, Pooja A S, Priyanka. Guide: Mrs. Suketha.

> **Scope (matches the written proposal):** *static single-frame* recognition of a fixed
> vocabulary. Not continuous signing, no LSTM / video modelling. See `docs/TEAM_PLAN.md`
> for the decisions this implements.

## Pipeline

```
webcam frame
  -> low-light detect + enhance (adaptive gamma + CLAHE)
  -> MediaPipe Hands (21 landmarks)
  -> square crop with padding   [-> optional GrabCut background removal, landmark-seeded]
  -> resize 224x224
  -> MobileNetV2 (transfer learning, softmax head)
  -> confidence threshold + hold-for-N-frames -> text -> optional TTS
```

## Folder layout

```
SignBridge/
  config.py              vocabulary, paths, thresholds  (edit GESTURES here)
  dataset_utils.py       metadata.csv + signer/session-aware splitting
  preprocessing/
    preprocess.py        THE preprocessing module; preprocess_frame() is the CNN-side entry point
    background_removal.py  live webcam test of the preprocessing (your "Step 3")
  collect_data.py        webcam capture with lighting/background/signer metadata
  prepare_dataset.py     raw -> cropped/resized train|val|test, split BY SIGNER
  train.py               MobileNetV2: freeze head -> fine-tune, early stopping, checkpoint
  evaluate.py            accuracy, confusion matrix, by lighting/background/signer, latency
  demo.py                real-time recognition + text + speech
  tools/import_public.py use a public dataset to debug the pipeline early
  tests/                 offline tests (no camera/GPU needed)
  docs/TEAM_PLAN.md      decisions, collection checklist, poster panel template
  models/  results/      outputs (model file is git-ignored; share via Drive)
```

The **dataset is not in Git.** Keep it in the shared Drive folder and point to it:
`set SIGNBRIDGE_DATA=D:\path\signbridge-dataset` (Windows) or `export SIGNBRIDGE_DATA=...`.
Default location is `./signbridge-dataset/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v                # should print OK
```

Use Python 3.9-3.12. `mediapipe` is pinned to 0.10.14 because newer releases removed the
`mp.solutions.hands` API used here. If you see *"no mp.solutions"*, reinstall that version.

## Workflow

1. **Check preprocessing live** - `python preprocessing/background_removal.py`
   (S saves a transparent PNG, Q quits). Try dim light, cluttered background, spread fingers.
2. **Collect data** - one session per signer x lighting x background:
   ```bash
   python collect_data.py --signer signer01 --lighting normal --background plain
   python collect_data.py --signer signer01 --lighting dim    --background cluttered
   ```
   SPACE shoots a burst (only frames where a hand is detected are kept), N/P switch class.
   Target 150-300 images/class overall. Cover every cell of the variation matrix
   (lighting: normal/dim/backlit; background: plain/cluttered/outdoor) for **every signer**.
3. **Prepare** - `python prepare_dataset.py` (add `--remove_bg` to also strip backgrounds).
   Splits by whole signer, prints per-class/per-split counts and lighting mix, and records the
   preprocessing settings so training and the demo can never disagree.
4. **Train** - `python train.py` (Colab free GPU is enough; see below).
5. **Evaluate** - `python evaluate.py` -> `results/results.md` (+ confusion matrix PNG).
   Optional: `--simulate_lowlight 150`.
6. **Demo** - `python demo.py` (S speak, C clear, Q quit).

### Getting going before your own data exists
Download the Kaggle ISL set and run
`python tools/import_public.py --src path/to/ISL --per_class 200`, then steps 3-6.
Treat these numbers as **pipeline tests only** (see "Reporting honestly").

### Colab
Put `signbridge-dataset/` on Drive, then in Colab:
```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/likhitha763/SignBridge && cd SignBridge && pip -q install -r requirements.txt
import os; os.environ['SIGNBRIDGE_DATA'] = '/content/drive/MyDrive/signbridge-dataset'
!cd SignBridge && python prepare_dataset.py && python train.py && python evaluate.py
```
(Collection needs a local webcam, so run `collect_data.py` on a laptop, not in Colab.)
Download `models/signbridge.keras`, `class_names.json`, `preprocess_config.json` for the demo.

## For Pooja (CNN side)

```python
from preprocessing.preprocess import preprocess_frame
img = preprocess_frame(frame_bgr)       # uint8 RGB 224x224x3, or None when no hand
probs = model(img[None].astype("float32"), training=False)   # the model rescales internally
```
Frames must be mirrored with `cv2.flip(frame, 1)` exactly as `collect_data.py` and `demo.py`
do - otherwise left/right hands look swapped relative to training data.

## Reporting honestly (poster / report)

- Say what each number is: *trained on X, validated on a held-out signer set*.
- By-lighting accuracy comes from **real** dim-light photos you shot (`metadata.csv`). The
  `--simulate_lowlight` experiment uses darkened copies; label it "simulated".
- With 4 signers a by-signer split is roughly 2 train / 1 val / 1 test, so the test number
  reflects a single unseen person. Recruit 2-3 outside volunteers if you can, and say how
  many signers the test set had.
- Anything not yet measured on the poster must read **"projected"**.

## Limitations

GrabCut is approximate: gaps between fingers or objects touching the hand can be mis-segmented.
That is why background removal is optional (`--remove_bg`); compare both on your validation
set and keep whichever generalises better. Very dark frames amplify sensor noise when brightened.
The vocabulary is fixed; adding a sign means collecting data and retraining.

## Note on the existing repo

The README on `main` describes a HaGRID/MobileNetV2 pipeline while the `train.py` there is a
20-sign landmark-LSTM. This folder implements one consistent design (static images, MobileNetV2,
own dataset). If you replace the repo contents with it, delete the old `collect_data.py`,
`common.py`, `demo.py`, `prepare_data.py`, `train.py` and `__pycache__/` first.
