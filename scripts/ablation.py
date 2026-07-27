#!/usr/bin/env python3
"""Recompute the results table in docs/RESULTS.md from the committed out-of-fold
slide probabilities.

    python scripts/ablation.py

Reads only `artifacts/oof_slide_probs_5fold.npz`: no dataset, no GPU, no model
weights, no access to the gated UNI backbone. Needs numpy, pandas and
scikit-learn.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, ENSEMBLE          # noqa: E402
from src.ensemble import blend, score, tune_alpha     # noqa: E402
from src.submission import check_submission           # noqa: E402


def load_oof(root: Path = ROOT):
    data = np.load(root / "artifacts" / "oof_slide_probs_5fold.npz", allow_pickle=True)
    return (data["probs_resnet"], data["probs_uni"], data["labels"], data["slide_ids"])


def main() -> int:
    resnet, uni, y, slide_ids = load_oof()
    print(f"{len(y)} training slides, out-of-fold predictions from "
          f"{ENSEMBLE.n_splits}-fold slide-level cross-validation")
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    print("class counts: " + ", ".join(f"{n} {c}" for c, n in zip(CLASS_NAMES, counts)))

    rows = [
        ("ResNet18 tiles, alone", resnet),
        ("UNI linear head, alone", uni),
        (f"Ensemble, alpha = {ENSEMBLE.alpha_cv} (cross-validated)",
         blend(resnet, uni, ENSEMBLE.alpha_cv)),
        (f"Ensemble, alpha = {ENSEMBLE.alpha_shipped} (submitted)",
         blend(resnet, uni, ENSEMBLE.alpha_shipped)),
    ]

    print(f"\n{'':46s} {'macro-F1':>9s} {'weighted':>9s} {'accuracy':>9s}")
    for name, probs in rows:
        print(f"{name:46s} {score(probs, y):9.4f} "
              f"{score(probs, y, 'weighted'):9.4f} {score(probs, y, 'micro'):9.4f}")

    best_alpha, best_f1 = tune_alpha(resnet, uni, y)
    print(f"\nGrid search over alpha: best macro-F1 {best_f1:.4f} at alpha = {best_alpha:.2f}")

    shipped = blend(resnet, uni, ENSEMBLE.alpha_shipped)
    print(f"\nSubmitted ensemble (alpha = {ENSEMBLE.alpha_shipped}), per class:")
    print(classification_report(y, shipped.argmax(axis=1), target_names=list(CLASS_NAMES),
                                digits=4, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y, shipped.argmax(axis=1)))

    print("\nSubmitted predictions file:")
    import pandas as pd
    check_submission(pd.read_csv(ROOT / "submissions" / "submission_final.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
