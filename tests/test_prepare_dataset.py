"""End-to-end check of prepare_dataset.py on synthetic images (fake hand detector)."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import prepare_dataset  # noqa: E402
from dataset_utils import read_metadata  # noqa: E402
from preprocessing import preprocess  # noqa: E402
from tests.test_pipeline import FakeDetector, LANDMARKS, synthetic_frame  # noqa: E402


class PrepareDatasetTest(unittest.TestCase):
    def test_signer_split_has_no_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            patches = {"DATASET_DIR": tmp, "RAW_DIR": tmp / "raw",
                       "PROCESSED_DIR": tmp / "processed", "METADATA_CSV": tmp / "metadata.csv"}
            with mock.patch.multiple(config, **patches), \
                 mock.patch.object(preprocess, "MediaPipeHandDetector",
                                   lambda *a, **k: FakeDetector(LANDMARKS)), \
                 mock.patch.object(sys, "argv", ["prepare_dataset.py", "--remove_bg"]):
                frame, _ = synthetic_frame()
                for signer in ("signer01", "signer02", "signer03", "signer04"):
                    for light, b in (("normal", 1.0), ("dim", 0.3)):
                        session = f"{signer}_{light}_plain"
                        for cls in ("emergency", "price"):
                            d = tmp / "raw" / session / cls
                            d.mkdir(parents=True)
                            for i in range(3):
                                cv2.imwrite(str(d / f"{i}.jpg"), (frame * b).astype("uint8"))
                prepare_dataset.main()

                rows = read_metadata(tmp / "metadata.csv")
                self.assertEqual(len(rows), 4 * 2 * 2 * 3)
                split_of = {}
                for r in rows:
                    self.assertIn(r["split"], ("train", "val", "test"))
                    self.assertEqual(split_of.setdefault(r["signer"], r["split"]), r["split"])
                    self.assertTrue((tmp / r["processed"]).exists())
                self.assertEqual(set(split_of.values()), {"train", "val", "test"})
                img = cv2.imread(str(tmp / rows[0]["processed"]))
                self.assertEqual(img.shape, (224, 224, 3))


if __name__ == "__main__":
    unittest.main()
