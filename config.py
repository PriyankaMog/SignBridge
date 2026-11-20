"""Central settings for SignBridge. Edit GESTURES to change the vocabulary."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Paths ---------------------------------------------------------------
# Dataset is NOT stored in Git. Point SIGNBRIDGE_DATA at your Drive/local folder.
DATASET_DIR = Path(os.environ.get("SIGNBRIDGE_DATA", ROOT / "signbridge-dataset"))
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
METADATA_CSV = DATASET_DIR / "metadata.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# ---- Preprocessing ---------------------------------------------------------
IMG_SIZE = 224               # MobileNetV2 default input size
LOW_LIGHT_THRESHOLD = 110    # mean grey level (0-255) below which a frame is "dark"
BRIGHT_THRESHOLD = 180       # mean grey level above which a frame is "overexposed"
BACKLIT_GAP = 40             # frame brighter than the hand by this much (and hand dark) = backlit
CROP_PADDING = 0.25          # extra margin around the hand, as a fraction of hand size

# ---- Inference ---------------------------------------------------------------
CONF_THRESHOLD = 0.70        # only show/accept predictions above this confidence
HOLD_FRAMES = 8              # same prediction this many frames in a row -> accepted

# ---- Collection protocol (the "variation matrix" from the briefing) ------------
LIGHTING = ["normal", "dim", "backlit", "bright"]
BACKGROUNDS = ["plain", "cluttered", "outdoor"]

# ---- Vocabulary -------------------------------------------------------------------
# name = folder/class label, hand = how to form the gesture, speech = text read aloud.
# 12 classes (briefing recommends 15-20: add rows here to grow the vocabulary).
GESTURES = [
        {"name": "emergency", "emoji": "🆘", "hand": "both palms to camera", "speech": "This is an emergency"},
    {"name": "doctor",    "emoji": "🩺", "hand": "right C shape (thumb+index)", "speech": "I need a doctor"},
    {"name": "medicine",  "emoji": "💊", "hand": "both C shapes joined", "speech": "I need medicine"},
    {"name": "water",     "emoji": "💧", "hand": "pinch touching left palm", "speech": "I need water"},
    {"name": "washroom",  "emoji": "🚻", "hand": "both palms flat together", "speech": "I need the washroom"},
    {"name": "help",      "emoji": "🙋", "hand": "right fist on left palm", "speech": "I need help"},
    {"name": "pain",      "emoji": "🤕", "hand": "right index on left forearm", "speech": "I am in pain"},
    {"name": "stop",      "emoji": "✋", "hand": "right palm to camera", "speech": "Please stop"}
]
CLASSES = [g["name"] for g in GESTURES]
PHRASES = {g["name"]: g for g in GESTURES}
