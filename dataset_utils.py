"""Metadata CSV helpers and signer/session-aware train/val/test splitting."""
import csv
import random
from collections import defaultdict
from pathlib import Path

META_COLUMNS = ["filepath", "class", "signer", "lighting", "background",
                "session", "split", "processed"]


def read_metadata(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for c in META_COLUMNS:
            r.setdefault(c, "")
    return rows


def write_metadata(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=META_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def append_metadata(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=META_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in META_COLUMNS})


def assign_splits(rows, by="signer", seed=0, val_frac=0.15, test_frac=0.15,
                  val_groups=None, test_groups=None):
    """Return {group_name: 'train'|'val'|'test'}.

    Splitting is done on whole groups (all images of a signer, or of a session)
    so near-duplicate frames from one burst can never leak between splits.
    """
    if by not in ("signer", "session"):
        raise ValueError("by must be 'signer' or 'session'")
    groups = sorted({r[by] for r in rows})
    n = len(groups)
    if n < 3:
        raise ValueError(
            f"Only {n} distinct {by}(s) found; need at least 3 to build "
            f"train/val/test splits separated by {by}.")

    val_groups = set(val_groups or [])
    test_groups = set(test_groups or [])
    unknown = (val_groups | test_groups) - set(groups)
    if unknown:
        raise ValueError(f"Unknown {by}(s): {sorted(unknown)}. Available: {groups}")
    if val_groups & test_groups:
        raise ValueError("A group cannot be in both val and test.")

    if not (val_groups or test_groups):
        shuffled = groups[:]
        random.Random(seed).shuffle(shuffled)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        if n_test + n_val >= n:
            n_test = n_val = 1
        test_groups = set(shuffled[:n_test])
        val_groups = set(shuffled[n_test:n_test + n_val])

    return {g: ("test" if g in test_groups else "val" if g in val_groups else "train")
            for g in groups}


def split_coverage(rows, split_key="split"):
    """{split: {class: count}} for reporting and sanity checks."""
    out = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r[split_key] in ("train", "val", "test"):
            out[r[split_key]][r["class"]] += 1
    return out
