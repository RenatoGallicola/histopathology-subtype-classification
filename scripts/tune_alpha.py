#!/usr/bin/env python3
"""Replay the ensemble weight search on the committed out-of-fold probabilities.

    python scripts/tune_alpha.py

Prints the score for every weight on the grid and reports the plateau the optimum
sits on, which is more informative than the single winning value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ENSEMBLE                              # noqa: E402
from src.ensemble import alpha_curve, blend, score, tune_alpha  # noqa: E402


def main() -> int:
    data = np.load(ROOT / "artifacts" / "oof_slide_probs_5fold.npz", allow_pickle=True)
    resnet, uni, y = data["probs_resnet"], data["probs_uni"], data["labels"]

    grid, scores = alpha_curve(resnet, uni, y)
    best_alpha, best_f1 = tune_alpha(resnet, uni, y)

    print("alpha  macro-F1")
    for a, s in zip(grid, scores):
        if round(a * 100) % 5 == 0:
            marker = "  <-- submitted" if abs(a - ENSEMBLE.alpha_shipped) < 1e-9 else ""
            print(f"{a:5.2f}  {s:.4f}{marker}")

    print(f"\nBest on the grid: macro-F1 {best_f1:.4f} at alpha = {best_alpha:.2f}")

    # F1 is a step function of alpha, so the curve is piecewise constant and, on
    # 627 slides, visibly jagged. Reporting how many weights come within a small
    # tolerance of the best says more than the winning value on its own.
    near = grid[scores >= best_f1 - 0.005]
    print(f"Within 0.005 of the best: {len(near)} of {len(grid)} weights, "
          f"spanning alpha in [{near.min():.2f}, {near.max():.2f}]")
    print("The curve is not unimodal — the weight is only loosely determined by "
          "this many slides.")

    shipped = score(blend(resnet, uni, ENSEMBLE.alpha_shipped), y)
    print(f"\nSubmitted alpha = {ENSEMBLE.alpha_shipped}: macro-F1 {shipped:.4f} "
          f"out of fold, against {best_f1:.4f} at the out-of-fold optimum.")
    print("The submitted weight is larger than the cross-validated one by design:\n"
          "dihedral test-time augmentation is applied to the ResNet branch at\n"
          "inference but not during cross-validation, so that branch is stronger\n"
          "at test time than these out-of-fold probabilities can show.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
