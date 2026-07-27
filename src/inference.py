"""Pooling tile predictions up to slide predictions, with and without TTA."""
from __future__ import annotations

import numpy as np

from .config import ENSEMBLE, NUM_CLASSES


def pool_slide_probabilities(tile_probs: np.ndarray, slide_ids) -> tuple:
    """Average tile probabilities into one vector per slide.

    Mean rather than max. With a few dozen tiles per slide, a max lets a single
    confidently-wrong tile decide the slide; the mean asks what the tissue says on
    balance, which is the assumption multiple-instance learning is making here.

    Returns ``(slide_ids_sorted, probs)``.
    """
    tile_probs = np.asarray(tile_probs)
    slide_ids = np.asarray(slide_ids)
    order = sorted(set(slide_ids.tolist()))
    probs = np.vstack([tile_probs[slide_ids == sid].mean(axis=0) for sid in order])
    return np.asarray(order), probs


def predict_tiles(model, dataset, device, batch_size: int = 128) -> np.ndarray:
    """Softmax probabilities for every tile in `dataset`."""
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device, non_blocking=True)
            out.append(torch.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def predict_tiles_dihedral_tta(model, dataset, device, batch_size: int = 128) -> np.ndarray:
    """Tile probabilities averaged over the dihedral group.

    Eight views per tile: four rotations, each with and without a horizontal flip.
    Because a tissue tile has no canonical orientation, all eight are equally valid
    presentations of the same tile, so averaging over them reduces variance without
    biasing the prediction.
    """
    import torch
    import torchvision.transforms.functional as TF
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_views = len(ENSEMBLE.tta_rotations) * (2 if ENSEMBLE.tta_flip else 1)
    out = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device, non_blocking=True)
            acc = torch.zeros(x.size(0), NUM_CLASSES, device=device)
            for angle in ENSEMBLE.tta_rotations:
                xr = TF.rotate(x, angle)
                acc += torch.softmax(model(xr), dim=1)
                if ENSEMBLE.tta_flip:
                    acc += torch.softmax(model(TF.hflip(xr)), dim=1)
            out.append((acc / n_views).cpu().numpy())

    return np.concatenate(out, axis=0)


def predict_embeddings(classifier, features: np.ndarray, device,
                       batch_size: int = 1024) -> np.ndarray:
    """Softmax probabilities from the UNI head over cached embeddings."""
    import torch

    classifier.eval()
    X = torch.tensor(np.asarray(features), dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = classifier(X[i:i + batch_size].to(device))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)
