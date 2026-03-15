"""
ml/model.py
===========
Lightweight CNN classifier for CTHMP synthetic satellite scenes.

Architecture: MobileNetV2 backbone (pretrained=False for offline/CI use)
with a custom classification head.  The backbone is small enough to train
quickly on CPU with synthetic data.

For production you would swap in a pretrained backbone; keeping it random
here ensures the repo works fully offline and with no large model downloads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNetV2, mobilenet_v2


class SceneClassifier(nn.Module):
    """
    MobileNetV2-backbone scene classifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 3 — building, vehicle, open_area).
    pretrained : bool
        Load ImageNet weights.  Set False for offline/CI environments.
    dropout : float
        Dropout rate on the classification head.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Load backbone
        weights = None  # pretrained weights disabled by default for offline use
        backbone: MobileNetV2 = mobilenet_v2(weights=weights)

        # Keep all feature layers; replace the classifier
        self.features = backbone.features
        in_features = backbone.last_channel  # 1280

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  shape (B, 3, H, W)

        Returns
        -------
        logits : Tensor  shape (B, num_classes)
        """
        features = self.features(x)        # (B, 1280, H', W')
        logits = self.classifier(features)  # (B, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def build_model(num_classes: int = 3, pretrained: bool = False) -> SceneClassifier:
    """Factory — returns a freshly initialised SceneClassifier."""
    return SceneClassifier(num_classes=num_classes, pretrained=pretrained)


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
