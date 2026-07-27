# Data

The dataset belongs to the AN2DL course and is **not redistributed here**. The
images are also far too large for a git repository — around 1.9 GB of slides plus
masks.

## What is committed

| File | Description |
|---|---|
| `clean_train_labels.csv` | Slide-level labels after removing the contaminated slides. |

## What you need to add

Place the competition data here, or point `WSI_DATA_DIR` at it:

```
data/
  train_data/        691 image/mask pairs
  test_data/         477 image/mask pairs, unlabelled
  train_labels.csv   ground-truth molecular subtype per training slide
```

## Structure

Each sample is a low-magnification whole-slide image of human tissue paired with a
binary mask marking the regions most likely to contain diseased tissue. Images
vary in size. Labels are one of four molecular subtypes:

| Label | Meaning |
|---|---|
| `Luminal A` | Hormone-receptor positive, low proliferation |
| `Luminal B` | Hormone-receptor positive, higher proliferation |
| `HER2(+)` | HER2-enriched |
| `Triple negative` | Negative for all three receptors |

### Facts worth knowing before you model

* **The masks are optional but useful.** They are not needed for classification,
  but they identify which parts of a slide are worth tiling — the pipeline here
  uses them to skip tiles with less than 5% tumour coverage.
* **60 slides are contaminated** with an unrelated image blended into the tissue,
  and a further handful carry bright green blob artifacts from the staining or
  annotation process. `01_preprocessing_and_tiles.ipynb` removes the former and
  filters tiles dominated by the latter, leaving 627 slides with usable tiles.
* **Slide sizes vary widely**, so the tile count per slide does too. Any
  slide-level metric has to pool over a variable number of tiles, and any split
  has to be drawn over slides rather than tiles — an ungrouped split puts
  near-identical crops of one slide on both sides of the boundary.
* **Classes are imbalanced 174 / 219 / 158 / 76.** Always predicting `Luminal B`
  scores 0.349 accuracy and 0.129 macro-F1 — worth keeping in mind when reading
  any headline number, since macro-F1 around 0.46 is well above that floor even
  though it looks low in absolute terms.
* **No validation split is provided.** Build your own, stratified on the label and
  grouped by slide.
