# Artifacts

Per-slide probabilities produced by the submitted run. This is what makes the
model-selection stage reproducible without the dataset, a GPU, or the gated UNI
weights — `scripts/ablation.py` and `scripts/tune_alpha.py` read only this file.

## `oof_slide_probs_5fold.npz`

Out-of-fold predictions for the 627 training slides. Each slide was predicted by
models trained on the other four folds, so nothing here has seen the slide it
scores.

| Key | Shape | Contents |
|---|---|---|
| `slide_ids` | (627,) | Slide filenames, e.g. `img_0000.png` |
| `probs_resnet` | (627, 4) | ResNet18 tile classifier, tile probabilities averaged per slide |
| `probs_uni` | (627, 4) | Linear head over frozen UNI embeddings, averaged per slide |
| `labels` | (627,) | Ground truth: 0 `Luminal A`, 1 `Luminal B`, 2 `HER2(+)`, 3 `Triple negative` |

Rows are aligned across all four arrays: row *n* of `probs_resnet` and row *n* of
`probs_uni` describe the same slide, named in `slide_ids[n]`.

Each row sums to 1 — these are softmax probabilities averaged over the slide's
tiles, not logits. Blending them is therefore a convex combination, which is why
`alpha` is constrained to `[0, 1]` in `src/ensemble.py`.

## What is not here

The test-set probabilities of the two branches were not saved during the original
run; only the resulting predictions were, in `submissions/`. Reproducing the test
predictions therefore requires the trained weights and the tile archives — see
`models/README.md`. The weight search and every out-of-fold number in
`docs/RESULTS.md` need nothing beyond this directory.

## Alignment

Neither branch emits slides in a guaranteed order — the ResNet iterates the tile
archive, the UNI head iterates cached embeddings — so both were sorted by slide id
before being written here. If you regenerate either, align on `slide_ids` before
blending: a misalignment raises nothing, since the shapes still match, and simply
degrades the score.
