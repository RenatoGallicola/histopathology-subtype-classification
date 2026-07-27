"""Combining the two branches.

The two models disagree in a useful way: the ResNet18 reads local texture through
ImageNet features, UNI reads the same tiles through a representation built on
histopathology. A single scalar decides how much to trust each, and it is fitted
on out-of-fold predictions rather than picked by hand.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def blend(probs_a: np.ndarray, probs_b: np.ndarray, alpha: float) -> np.ndarray:
    """`alpha * probs_a + (1 - alpha) * probs_b`, with the weight validated."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    probs_a, probs_b = np.asarray(probs_a), np.asarray(probs_b)
    if probs_a.shape != probs_b.shape:
        raise ValueError(f"shape mismatch: {probs_a.shape} vs {probs_b.shape} — "
                         "align the two branches by slide id first")
    return alpha * probs_a + (1.0 - alpha) * probs_b


def score(probs: np.ndarray, y_true, average: str = "macro") -> float:
    """F1 of the arg-max prediction."""
    return f1_score(np.asarray(y_true), np.asarray(probs).argmax(axis=1),
                    average=average)


def tune_alpha(probs_a: np.ndarray, probs_b: np.ndarray, y_true,
               n_coarse: int = 101, refine: bool = True,
               average: str = "macro") -> tuple:
    """Grid-search the mixing weight, then refine around the winner.

    Returns ``(best_alpha, best_f1)``.

    Two things are worth knowing before trusting the number. The score is measured
    on the same out-of-fold probabilities the weight is fitted on, so it is
    optimistic. And F1 is a step function of alpha — the curve is piecewise
    constant, so a broad plateau can hold many alphas with identical scores; see
    `alpha_curve` for the shape.
    """
    grid = np.linspace(0.0, 1.0, n_coarse)
    scores = [score(blend(probs_a, probs_b, a), y_true, average) for a in grid]
    best_i = int(np.argmax(scores))
    best_alpha, best_f1 = float(grid[best_i]), float(scores[best_i])

    if refine:
        lo, hi = max(0.0, best_alpha - 0.05), min(1.0, best_alpha + 0.05)
        fine = np.linspace(lo, hi, n_coarse)
        fine_scores = [score(blend(probs_a, probs_b, a), y_true, average) for a in fine]
        j = int(np.argmax(fine_scores))
        if fine_scores[j] > best_f1:
            best_alpha, best_f1 = float(fine[j]), float(fine_scores[j])

    return best_alpha, best_f1


def alpha_curve(probs_a: np.ndarray, probs_b: np.ndarray, y_true,
                n: int = 101, average: str = "macro") -> tuple:
    """The whole score-versus-alpha curve, for plotting and for sanity."""
    grid = np.linspace(0.0, 1.0, n)
    return grid, np.array([score(blend(probs_a, probs_b, a), y_true, average)
                           for a in grid])
