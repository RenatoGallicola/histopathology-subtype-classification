#!/usr/bin/env python3
"""Render the documentation figures.

    python scripts/make_figures.py

Light and dark variants are written to assets/. The progression figure quotes the
leaderboard scores recorded in the report; the other two are computed from
`artifacts/oof_slide_probs_5fold.npz`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, ENSEMBLE     # noqa: E402
from src.ensemble import alpha_curve, blend      # noqa: E402

ASSETS = ROOT / "assets"

# One accent hue plus a de-emphasis grey: the "emphasis" form, where one mark is
# the point and the rest are context. Both accent steps clear the lightness band,
# the chroma floor and 3:1 contrast on their own surface.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  accent="#2a78d6", context="#898781",
                  ramp=["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", secondary="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", axis="#383835",
                 accent="#3987e5", context="#898781",
                 ramp=["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]),
}

FONT = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]

# Leaderboard scores as recorded in report/, one row per stage of the pipeline.
# These cannot be recomputed here: they are scored against labels we never had.
STAGES = [
    ("ResNet50, whole slides", 0.3768, False),
    ("ResNet18, tiles + flip TTA", 0.3859, False),
    ("+ UNI, hand-picked weight", 0.4020, False),
    ("+ 5-fold cross-validated weight", 0.4234, False),
    ("+ dihedral TTA  ·  submitted", 0.4304, True),
]


def _style(theme):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT,
        "figure.facecolor": theme["surface"], "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"], "text.color": theme["ink"],
        "axes.labelcolor": theme["secondary"], "xtick.color": theme["muted"],
        "ytick.color": theme["secondary"], "axes.edgecolor": theme["axis"],
    })


def _frame(ax, theme, xgrid=True):
    (ax.xaxis if xgrid else ax.yaxis).grid(True, color=theme["grid"], lw=0.8, zorder=0)
    (ax.yaxis if xgrid else ax.xaxis).grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=0)


def load_oof():
    d = np.load(ROOT / "artifacts" / "oof_slide_probs_5fold.npz", allow_pickle=True)
    return d["probs_resnet"], d["probs_uni"], d["labels"]


# --- Figure 1: pipeline progression ---------------------------------------
def figure_progression(mode: str) -> Path:
    """Dot plot, not bars: the scores span 0.377-0.430, so bars from zero would be
    indistinguishable and truncated bars would exaggerate the steps. Dots carry no
    area, so a non-zero axis is honest."""
    theme = THEMES[mode]
    _style(theme)

    labels = [s[0] for s in STAGES][::-1]
    scores = [s[1] for s in STAGES][::-1]
    final = [s[2] for s in STAGES][::-1]

    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    lo, hi = 0.368, 0.442

    for i, (v, is_final) in enumerate(zip(scores, final)):
        ax.plot([lo, v], [i, i], color=theme["grid"], lw=1.0, zorder=1,
                solid_capstyle="butt")
        ax.plot(v, i, "o", ms=11 if is_final else 9, zorder=3,
                color=theme["accent"] if is_final else theme["context"],
                markeredgecolor=theme["surface"], markeredgewidth=2)
        ax.text(v + 0.0022, i, f"{v:.4f}", va="center", ha="left", fontsize=10.5,
                color=theme["ink"] if is_final else theme["secondary"],
                fontweight="600" if is_final else "normal", zorder=4)

    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels(labels, fontsize=10.5)
    for tick, is_final in zip(ax.get_yticklabels(), final):
        tick.set_color(theme["ink"] if is_final else theme["secondary"])
        if is_final:
            tick.set_fontweight("600")

    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, len(scores) - 0.4)
    ax.set_xticks(np.arange(0.38, 0.441, 0.02))
    ax.set_xlabel("Test F1 on the competition leaderboard   ·   axis starts at 0.368",
                  fontsize=9.5, labelpad=9)
    _frame(ax, theme)
    # Centred on the figure rather than on the axes: the category labels push the
    # plot area to the right, so an axes-centred title reads as off-centre.
    fig.suptitle("Test F1 by pipeline stage", fontsize=13, fontweight="600",
                 color=theme["ink"], y=0.98)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ASSETS / f"progression-{mode}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# --- Figure 2: ensemble weight curve --------------------------------------
def figure_alpha(mode: str) -> Path:
    """A line: one continuous parameter against one score."""
    theme = THEMES[mode]
    _style(theme)
    resnet, uni, y = load_oof()
    grid, scores = alpha_curve(resnet, uni, y)

    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    ax.plot(grid, scores, lw=2.0, color=theme["accent"], zorder=3)

    best = int(np.argmax(scores))
    ax.plot(grid[best], scores[best], "o", ms=9, color=theme["accent"], zorder=4,
            markeredgecolor=theme["surface"], markeredgewidth=2)
    ax.set_ylim(scores.min() - 0.003, scores.max() + 0.009)
    # Each label sits just above and right of its own marker. The curve descends
    # to the right of both, so that corner is free; an opaque patch keeps the
    # gridlines from running through the text.
    label_bg = dict(facecolor=theme["surface"], edgecolor="none", pad=2.5)
    ax.annotate(f"cross-validated optimum  ·  alpha = {grid[best]:.2f}, F1 = {scores[best]:.4f}",
                xy=(grid[best], scores[best]),
                xytext=(grid[best] + 0.022, scores[best] + 0.0022),
                fontsize=9, color=theme["secondary"], va="bottom", ha="left",
                bbox=label_bg, zorder=5)

    shipped = float(np.interp(ENSEMBLE.alpha_shipped, grid, scores))
    ax.plot(ENSEMBLE.alpha_shipped, shipped, "o", ms=8, color=theme["context"],
            zorder=4, markeredgecolor=theme["surface"], markeredgewidth=2)
    ax.annotate(f"submitted  ·  alpha = {ENSEMBLE.alpha_shipped:.2f}",
                xy=(ENSEMBLE.alpha_shipped, shipped),
                xytext=(ENSEMBLE.alpha_shipped + 0.022, shipped + 0.0022),
                fontsize=9, color=theme["muted"], va="bottom", ha="left",
                bbox=label_bg, zorder=5)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("alpha   ·   0 = UNI only,   1 = ResNet18 only", fontsize=9.5, labelpad=9)
    ax.set_ylabel("Out-of-fold macro-F1", fontsize=9.5, labelpad=9)
    _frame(ax, theme, xgrid=False)
    ax.xaxis.grid(True, color=theme["grid"], lw=0.8, zorder=0)
    fig.suptitle("Out-of-fold macro-F1 across ensemble weights", fontsize=13,
                 fontweight="600", color=theme["ink"], y=0.98)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ASSETS / f"alpha-curve-{mode}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# --- Figure 3: confusion matrix -------------------------------------------
def figure_confusion(mode: str) -> Path:
    """Heatmap: sequential magnitude, one hue. Each cell also carries its count
    and row share, so nothing is encoded by colour alone."""
    theme = THEMES[mode]
    _style(theme)
    resnet, uni, y = load_oof()
    preds = blend(resnet, uni, ENSEMBLE.alpha_shipped).argmax(axis=1)

    counts = confusion_matrix(y, preds)
    shares = counts / counts.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("seq_blue", theme["ramp"])

    fig, ax = plt.subplots(figsize=(7.0, 5.9))
    im = ax.imshow(shares, cmap=cmap, vmin=0, vmax=1)

    n = len(CLASS_NAMES)
    ax.set_xticks(range(n), CLASS_NAMES, fontsize=10, color=theme["secondary"])
    ax.set_yticks(range(n), CLASS_NAMES, fontsize=10, color=theme["secondary"])
    ax.set_xlabel("predicted", fontsize=10, labelpad=9)
    ax.set_ylabel("true", fontsize=10, labelpad=9)

    for i in range(n):
        for j in range(n):
            red, green, blue, _ = cmap(shares[i, j])
            on_light = 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.5
            ax.text(j, i - 0.10, str(counts[i, j]), ha="center", va="center",
                    fontsize=15, fontweight="600",
                    color="#0b0b0b" if on_light else "#ffffff")
            ax.text(j, i + 0.20, f"{shares[i, j] * 100:.1f}%", ha="center",
                    va="center", fontsize=9,
                    color="#3d3d3b" if on_light else "#dcdcd6")

    ax.set_xticks(np.arange(0.5, n - 0.5, 1), minor=True)
    ax.set_yticks(np.arange(0.5, n - 0.5, 1), minor=True)
    ax.grid(which="minor", color=theme["surface"], linewidth=2.5)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.043, pad=0.04)
    cbar.set_label("share of the true class", fontsize=9, color=theme["secondary"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8.5, colors=theme["muted"])

    fig.suptitle("Confusion matrix of the submitted ensemble, out of fold",
                 fontsize=12.5, fontweight="600", color=theme["ink"], y=0.98)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ASSETS / f"confusion-matrix-{mode}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ASSETS.mkdir(exist_ok=True)
    for mode in ("light", "dark"):
        for path in (figure_progression(mode), figure_alpha(mode),
                     figure_confusion(mode)):
            print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
