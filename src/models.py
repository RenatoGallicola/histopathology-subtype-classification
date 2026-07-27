"""The two branches of the ensemble.

Neither is large. The ResNet18 is a small ImageNet-pretrained CNN reading tiles at
native resolution; the UNI branch is a two-layer head over a frozen pathology
foundation model. Almost all the capacity in the second branch is pretrained and
never updated here, which is precisely why it works on 627 training slides.
"""
from __future__ import annotations

from .config import NUM_CLASSES, RESNET, UNI


def create_resnet(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """ResNet18 tile classifier."""
    import timm
    return timm.create_model(RESNET.backbone, pretrained=pretrained,
                             num_classes=num_classes)


def create_uni_backbone():
    """The frozen UNI Vision Transformer.

    Gated on the Hugging Face Hub: the weights download only from an account that
    has accepted the model's terms. Returns the backbone in eval mode with the
    classifier head removed, so it emits 1024-dimensional embeddings.
    """
    import timm
    backbone = timm.create_model(UNI.hub_id, pretrained=True, init_values=1e-5,
                                 dynamic_img_size=True, num_classes=0)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def create_uni_classifier(dropout: float = UNI.dropout):
    """The trainable head over UNI embeddings."""
    from torch import nn

    class UNIClassifier(nn.Module):
        def __init__(self, drop):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(UNI.embed_dim, UNI.hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(UNI.hidden_dim, NUM_CLASSES),
            )

        def forward(self, x):
            return self.net(x)

    return UNIClassifier(dropout)
