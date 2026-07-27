"""Building and checking the competition submission."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import IDX_TO_LABEL

SAMPLE_RE = re.compile(r"^img_\d{4}\.png$")


def build_submission(slide_ids, probabilities: np.ndarray,
                     out_path: str | None = None) -> pd.DataFrame:
    """Write `sample_index,label`, sorted by slide id."""
    probabilities = np.asarray(probabilities)
    if len(slide_ids) != len(probabilities):
        raise ValueError(f"{len(slide_ids)} slide ids for "
                         f"{len(probabilities)} probability rows")

    frame = pd.DataFrame({
        "sample_index": list(slide_ids),
        "label": [IDX_TO_LABEL[int(i)] for i in probabilities.argmax(axis=1)],
    }).sort_values("sample_index").reset_index(drop=True)

    if out_path:
        frame.to_csv(out_path, index=False)
        print(f"Submission written to {out_path}")
        print(frame["label"].value_counts().to_string())

    return frame


def check_submission(frame: pd.DataFrame, n_expected: int = 477,
                     first_id: int = 0, last_id: int = 476) -> None:
    """Fail loudly on the mistakes that cost a leaderboard slot.

    Every one of these is silent at write time: a duplicated id, a missing slide
    or a label spelled differently still produces a well-formed CSV.
    """
    problems = []

    if list(frame.columns) != ["sample_index", "label"]:
        problems.append(f"unexpected columns: {list(frame.columns)}")
    if len(frame) != n_expected:
        problems.append(f"{len(frame)} rows, expected {n_expected}")
    if frame["sample_index"].duplicated().any():
        problems.append("duplicated sample_index values")

    malformed = [s for s in frame["sample_index"] if not SAMPLE_RE.match(str(s))]
    if malformed:
        problems.append(f"malformed sample_index values, e.g. {malformed[:3]}")
    else:
        numbers = sorted(int(str(s)[4:8]) for s in frame["sample_index"])
        expected = list(range(first_id, last_id + 1))
        if numbers != expected:
            missing = sorted(set(expected) - set(numbers))[:5]
            extra = sorted(set(numbers) - set(expected))[:5]
            problems.append(f"index range wrong (missing {missing}, unexpected {extra})")

    unknown = set(frame["label"]) - set(IDX_TO_LABEL.values())
    if unknown:
        problems.append(f"unknown labels: {sorted(unknown)}")
    if frame["label"].isna().any():
        problems.append("missing labels")

    if problems:
        raise ValueError("Invalid submission:\n  - " + "\n  - ".join(problems))

    print(f"Submission OK: {len(frame)} rows.")
    print(frame["label"].value_counts().to_string())
