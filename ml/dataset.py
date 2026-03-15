"""
ml/dataset.py
=============
PyTorch Dataset for the CTHMP synthetic COCO-style annotations.

Loads images from data/samples/images/ and their bounding-box annotations
from annotations.json.  Returns (image_tensor, target) pairs suitable for
object-detection or classification training.

For Part 2 we use a *classification* framing: the label for each scene is
the majority category_id among its annotations (1=building, 2=vehicle,
3=open_area).  This keeps the model simple while exercising the full
data → model → eval pipeline.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Transform = Callable[[np.ndarray], torch.Tensor]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SyntheticSceneDataset(Dataset[tuple[torch.Tensor, int]]):
    """
    Loads synthetic satellite scenes + majority-class labels.

    Parameters
    ----------
    annotations_path : Path
        Path to COCO-style annotations.json produced by synthetic_generator.py.
    images_dir : Path
        Directory containing the PNG images referenced in annotations.json.
    transform : callable, optional
        A function/transform that takes a (H,W,3) uint8 numpy array and
        returns a float32 torch.Tensor.  Defaults to ``default_transform``.
    """

    NUM_CLASSES: int = 3  # building, vehicle, open_area
    # category_id (1-indexed) → 0-indexed class label
    CATEGORY_TO_LABEL: dict[int, int] = {1: 0, 2: 1, 3: 2}
    LABEL_TO_NAME: dict[int, str] = {0: "building", 1: "vehicle", 2: "open_area"}

    def __init__(
        self,
        annotations_path: Path,
        images_dir: Path,
        transform: Optional[Transform] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transform = transform or default_transform

        with open(annotations_path, encoding="utf-8") as f:
            coco = json.load(f)

        # Build image_id → filename map
        id_to_file: dict[int, str] = {
            img["id"]: img["file_name"] for img in coco["images"]
        }

        # Build image_id → majority label
        votes: dict[int, list[int]] = {img["id"]: [] for img in coco["images"]}
        for ann in coco["annotations"]:
            votes[ann["image_id"]].append(ann["category_id"])

        self._samples: list[tuple[Path, int]] = []
        for image_id, filename in sorted(id_to_file.items()):
            cats = votes.get(image_id, [])
            if not cats:
                majority_cat = 1  # fallback
            else:
                majority_cat = Counter(cats).most_common(1)[0][0]
            label = self.CATEGORY_TO_LABEL[majority_cat]
            self._samples.append((self.images_dir / filename, label))

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self._samples[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        # BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image)
        return tensor, label

    def class_name(self, label: int) -> str:
        return self.LABEL_TO_NAME.get(label, "unknown")


# ---------------------------------------------------------------------------
# Default transform
# ---------------------------------------------------------------------------


def default_transform(image: np.ndarray) -> torch.Tensor:
    """
    Convert a (H, W, 3) uint8 numpy array → normalised (3, H, W) float32 tensor.

    Normalisation uses ImageNet mean/std so the backbone weights transfer cleanly.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = image.astype(np.float32) / 255.0          # [0, 1]
    img = (img - mean) / std                         # normalise
    tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC → CHW
    return tensor
