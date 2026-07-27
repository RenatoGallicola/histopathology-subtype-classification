"""Shared fixtures. Puts the repository root on sys.path so `src` imports work
without installing the package."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def oof(repo_root):
    """The committed out-of-fold slide probabilities."""
    data = np.load(repo_root / "artifacts" / "oof_slide_probs_5fold.npz",
                   allow_pickle=True)
    return {
        "resnet": data["probs_resnet"],
        "uni": data["probs_uni"],
        "y": data["labels"],
        "slide_ids": data["slide_ids"],
    }
