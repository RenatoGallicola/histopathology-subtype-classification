"""Hyperparameters of the submitted run, collected so the notebooks and the
modules cannot drift apart."""
from dataclasses import dataclass

SEED = 42
NUM_CLASSES = 4

# The mapping is fixed by the submission format. Getting it wrong permutes every
# prediction without raising anything, so it lives in exactly one place.
IDX_TO_LABEL = {0: "Luminal A", 1: "Luminal B", 2: "HER2(+)", 3: "Triple negative"}
LABEL_TO_IDX = {v: k for k, v in IDX_TO_LABEL.items()}
CLASS_NAMES = tuple(IDX_TO_LABEL[i] for i in range(NUM_CLASSES))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TileConfig:
    """Tile extraction. A tile inherits the label of the slide it came from, which
    is what turns a slide-level problem into a tile-level one."""
    size: int = 256
    stride: int = 128
    min_tissue_fraction: float = 0.20   # drop mostly-background tiles
    min_mask_fraction: float = 0.05     # drop tiles with almost no annotated tumour
    max_green_ratio: float = 0.05       # drop tiles dominated by staining artifacts
    green_hue_range: tuple = (35, 85)   # HSV hue band of the green blobs
    green_min_sat_val: int = 40


@dataclass(frozen=True)
class ResNetConfig:
    """ResNet18 tile classifier, ImageNet-pretrained."""
    backbone: str = "resnet18"
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1


@dataclass(frozen=True)
class UNIConfig:
    """Linear head over frozen UNI embeddings.

    UNI is a ViT pretrained on ~100M histopathology images; the backbone is never
    fine-tuned here, so a fold costs only the time to fit the head on cached
    1024-dimensional vectors.
    """
    hub_id: str = "hf-hub:MahmoodLab/uni"
    embed_dim: int = 1024
    hidden_dim: int = 256
    dropout: float = 0.4
    epochs: int = 25
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class EnsembleConfig:
    """Slide probability is `alpha * P_resnet + (1 - alpha) * P_uni`.

    Two alphas are recorded because they answer different questions. `alpha_cv` is
    what the grid search selected on out-of-fold probabilities. `alpha_shipped` is
    what produced `submissions/submission_final.csv`: dihedral test-time
    augmentation is applied to the ResNet branch at test time but not during
    cross-validation, so that branch is stronger at inference than the search saw.
    """
    n_splits: int = 5
    alpha_cv: float = 0.04
    alpha_shipped: float = 0.20
    tta_rotations: tuple = (0, 90, 180, 270)
    tta_flip: bool = True   # the full dihedral group: 4 rotations x {identity, hflip}


TILES = TileConfig()
RESNET = ResNetConfig()
UNI = UNIConfig()
ENSEMBLE = EnsembleConfig()


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and torch."""
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
