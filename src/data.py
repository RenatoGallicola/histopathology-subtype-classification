"""Tile dataset, augmentation and the slide index the whole pipeline rests on."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import IMAGENET_MEAN, IMAGENET_STD, LABEL_TO_IDX


def load_tiles(path: str):
    """Read a tile archive written by `01_preprocessing_and_tiles.ipynb`.

    Returns ``(tiles, slide_ids, labels)`` where `tiles` is uint8 HWC, `slide_ids`
    says which slide each tile came from, and `labels` is None for the test set.
    """
    data = np.load(path, allow_pickle=True)
    tiles = data["tiles"]
    slide_ids = data["slide_ids"]
    labels = data["labels"] if "labels" in data.files else None
    return tiles, slide_ids, labels


def build_slide_index(slide_ids) -> dict:
    """Map each slide id to the positions of its tiles.

    Everything downstream depends on this: cross-validation splits over slides
    rather than tiles, and predictions are pooled per slide. Splitting over tiles
    instead would put near-duplicate crops of one slide on both sides of a fold.
    """
    index = defaultdict(list)
    for i, sid in enumerate(slide_ids):
        index[sid].append(i)
    return {sid: np.asarray(idxs) for sid, idxs in index.items()}


def slide_labels(slide_index: dict, tile_labels) -> dict:
    """One label per slide, taken from its tiles (which all share it)."""
    return {sid: int(tile_labels[idxs[0]]) for sid, idxs in slide_index.items()}


def encode_labels(label_series) -> np.ndarray:
    """Map the textual subtype names to their integer codes."""
    return np.asarray([LABEL_TO_IDX[str(v)] for v in label_series], dtype=np.int64)


def class_weights(tile_labels: np.ndarray, n_classes: int = 4) -> np.ndarray:
    """Inverse-frequency weights normalised to mean 1, as used in training."""
    counts = np.bincount(np.asarray(tile_labels, dtype=np.int64),
                         minlength=n_classes).astype(np.float64)
    w = counts.sum() / np.clip(counts, 1.0, None)
    return w / w.mean()


# --------------------------------------------------------------------------- #
# Torch-dependent pieces
# --------------------------------------------------------------------------- #
def _torch():
    import torch
    return torch


def normalize_imagenet(x_chw_float01):
    """Normalise a CHW float tensor in [0, 1] with the ImageNet statistics."""
    torch = _torch()
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (x_chw_float01 - mean) / std


def augment_train(x_chw_float01):
    """Random flips and a random multiple of 90 degrees.

    This is the whole augmentation policy, and it is chosen rather than inherited:
    a tissue tile has no canonical orientation, so every element of the dihedral
    group maps a valid tile to another valid tile of the same class. Crops or
    colour jitter would not be as safe — stain intensity carries signal.
    """
    torch = _torch()
    import random

    x = x_chw_float01
    if random.random() < 0.5:
        x = torch.flip(x, dims=[2])
    if random.random() < 0.5:
        x = torch.flip(x, dims=[1])
    k = random.randint(0, 3)
    if k:
        x = torch.rot90(x, k, dims=[1, 2])
    return normalize_imagenet(x)


def make_tile_dataset(tiles_uint8_hwc, labels_int=None, idxs=None, train_aug=False):
    """Build the `TileDataset` used for both branches.

    Defined inside a function so importing `src.data` does not require torch.
    """
    torch = _torch()
    from torch.utils.data import Dataset

    class TileDataset(Dataset):
        def __init__(self, tiles, labels, subset, augment):
            self.tiles = tiles
            self.labels = labels
            self.idxs = np.arange(len(tiles)) if subset is None else np.asarray(subset)
            self.augment = augment

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            idx = self.idxs[i]
            x = torch.from_numpy(self.tiles[idx]).permute(2, 0, 1).float() / 255.0
            x = augment_train(x) if self.augment else normalize_imagenet(x)
            if self.labels is None:
                return x
            return x, torch.tensor(int(self.labels[idx]), dtype=torch.long)

    return TileDataset(tiles_uint8_hwc, labels_int, idxs, train_aug)


def make_embedding_dataset(features, labels_int, idxs):
    """Dataset over cached UNI embeddings, for training the classifier head."""
    torch = _torch()
    from torch.utils.data import Dataset

    class EmbeddingDataset(Dataset):
        def __init__(self, X, y, subset):
            self.X = X[subset]
            self.y = np.asarray(y)[subset].astype(np.int64)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return (torch.tensor(self.X[i], dtype=torch.float32),
                    torch.tensor(int(self.y[i]), dtype=torch.long))

    return EmbeddingDataset(features, labels_int, np.asarray(idxs))
