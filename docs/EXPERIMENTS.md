# Experiments

The eight notebooks in `notebooks/experiments/` are the branches that shaped the
final design. Four are the stages of the progression that led to the submission;
four are approaches that were tried and dropped. Both are kept, because the reason
something did not work is usually the more transferable half of the result.

All of them ran on Google Colab GPUs and keep their original outputs.

---

## The progression

### `01_resnet50_whole_slide.ipynb` — test F1 0.3768

Classify each slide as a single image. The preprocessing is the most elaborate in
the repository: green blob artifacts detected in HSV space and repainted with an
estimated background colour, tissue masks built by combining the provided
annotation masks with intensity-based detection, a bounding box around the tissue
with a 5% margin, and a resize to 512×512. ImageNet-pretrained ResNet50 with
`layer1`–`layer3` frozen for three warmup epochs, class-weighted cross-entropy
with label smoothing, AdamW and cosine annealing.

The ceiling here is not the model, it is the resize. A slide reduced to 512×512
has lost the nuclear detail that separates the subtypes, and no amount of careful
cleaning puts it back. That realisation is what produced the tile-based pipeline.

### `02_resnet18_tiles.ipynb` — test F1 0.3859

The pivot to multiple-instance learning. Each slide becomes a bag of 256×256 tiles
at native resolution, every tile is classified independently, and the slide
prediction is the mean of its tiles' softmax probabilities.

Note that the backbone got *smaller* — ResNet50 to ResNet18 — while the score went
up. The constraint was never capacity; it was that the network could not see what
it needed to see. Mean pooling was chosen over max: with a few dozen tiles per
slide, a max lets one confidently-wrong tile decide the slide.

### `03_resnet18_uni_ensemble.ipynb` — test F1 0.4020

Introduces [UNI](https://huggingface.co/MahmoodLab/UNI), a Vision Transformer
pretrained on roughly 100M histopathology images, as a second opinion. The
backbone is frozen and only a `Linear(1024→256) → ReLU → Dropout(0.4) →
Linear(256→4)` head is trained on its embeddings.

The largest single gain of the challenge. Freezing the backbone is what makes this
affordable: the embeddings are computed once and cached, so a fold costs minutes
of MLP fitting rather than hours of ViT fine-tuning — and with 627 slides,
fine-tuning a model that size would have overfitted anyway. The mixing weight is
still hand-picked at this stage.

### `04_5fold_alpha_tuning.ipynb` — test F1 0.4234

Replaces the hand-picked weight with a grid search over out-of-fold slide
probabilities from a slide-level stratified 5-fold split. Both branches are
trained inside the same folds, which is what makes their probabilities comparable
and therefore safe to mix.

Worth +0.0214 test F1 over the hand-picked weight — more than adding a whole
second model was worth in the previous step. This notebook also produced
`artifacts/oof_slide_probs_5fold.npz`, the file the reproducible scripts read.

### `notebooks/02_full_pipeline.ipynb` — test F1 0.4304

The submission: the above plus dihedral test-time augmentation on the ResNet
branch. Each tile is predicted under eight views — four rotations, each with and
without a horizontal flip — and the probabilities averaged before slide pooling.

A tissue tile has no canonical orientation, so all eight views are equally valid
presentations of the same tile. That is what makes this augmentation free of bias
rather than merely a variance reduction, and it is the same symmetry the training
augmentation exploits.

---

## Discarded approaches

### `05_convnext_focal_loss.ipynb`

ConvNeXt with `FocalLoss`, mask-guided attention and albumentations-based
augmentation, on whole slides.

Two bets, both lost. The stronger backbone had more capacity than 627 slides can
constrain, and focal loss — designed for extreme imbalance — mostly amplified
label noise on a 174/219/158/76 split that is not extreme at all. The final models
use class-weighted cross-entropy with label smoothing instead, a much gentler
correction for the same problem.

### `06_efficientnet_transfer.ipynb`

A systematic comparison on whole slides: a small CNN trained from scratch,
EfficientNet-B0 as a frozen feature extractor, then the same network fine-tuned.
Includes the exploratory analysis of class balance and image sizes.

The useful result is negative and consistent: every whole-slide approach lands in
the same band regardless of backbone or training strategy. That is what identified
the resize, rather than the architecture, as the binding constraint.

### `07_resnet18_whole_slide.ipynb`

Progressive unfreezing of a ResNet18 backbone on mask-cropped whole slides, with a
check on which parameters are trainable at each stage.

Same ceiling. Notable only because it is the same backbone that later succeeded on
tiles — the difference in outcome is entirely the input representation.

### `08_resnet50_focal_loss.ipynb`

ResNet50 with a custom `FocalLoss` and mask-aware preprocessing. Validation F1
peaked around 0.16, well below the plain cross-entropy baseline of
`01_resnet50_whole_slide.ipynb`. The clearest single piece of evidence against
focal loss on this dataset.

### Rotation-only test-time augmentation

Tried and dropped before the final run: four rotations without flips produced
results indistinguishable from flips without rotations. Neither half of the
dihedral group is redundant with the other, but neither alone captures the full
symmetry — only the complete group of eight views gave a measurable gain.

### Aggressive preprocessing on tiles

The blob removal and tissue cropping developed for the whole-slide phase were
carried over to the tile pipeline and did not help there. Tiles are already
filtered on tissue fraction and green ratio at extraction time, so the extra
cleaning had nothing left to remove.

---

## What we would do differently

* **Start from the data representation, not the model.** Four whole-slide
  notebooks across four architectures all landed within a few points of each
  other; the first tile-based notebook beat all of them with a smaller backbone.
  The tiling decision was worth more than every architecture comparison combined.
* **Reach for a domain foundation model earlier.** UNI frozen, with a two-layer
  head, outperformed every model trained here. On a dataset this size that should
  have been the first experiment rather than the third stage.
* **Treat the ensemble weight as a range, not a value.** The out-of-fold curve has
  17 of 101 grid points within 0.005 of the optimum and a second local peak; a
  weight reported to two decimals suggests a precision this data does not support.
* **Attack `Luminal A` versus `Luminal B` directly.** They account for the largest
  block of confusions and differ by proliferation rate, which is a nuclear-scale
  property — a higher-magnification crop or a proliferation-oriented feature would
  target it better than a fourth general-purpose backbone.
