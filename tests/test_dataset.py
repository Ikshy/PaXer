"""
tests/test_dataset.py
======================
Unit tests for ml/dataset.py.

Tests cover:
- Dataset length matches image count in annotations.json
- __getitem__ returns correct (tensor, int) types and shapes
- Label values are within valid range
- default_transform produces correctly shaped/normalised tensor
- CATEGORY_TO_LABEL mapping is complete and consistent
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from data.synthetic_generator import SyntheticSceneGenerator
from ml.dataset import SyntheticSceneDataset, default_transform


# ---------------------------------------------------------------------------
# Shared fixture: generate a tiny dataset once per module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate 6 synthetic scenes into a temp directory."""
    out = tmp_path_factory.mktemp("tiny_data")
    gen = SyntheticSceneGenerator(seed=7, output_dir=out, num_images=6)
    gen.generate_all()
    return out


@pytest.fixture(scope="module")
def dataset(tiny_dataset_dir: Path) -> SyntheticSceneDataset:
    return SyntheticSceneDataset(
        annotations_path=tiny_dataset_dir / "annotations.json",
        images_dir=tiny_dataset_dir / "images",
    )


# ---------------------------------------------------------------------------
# Dataset length
# ---------------------------------------------------------------------------


class TestDatasetLength:
    def test_length_matches_images(
        self, tiny_dataset_dir: Path, dataset: SyntheticSceneDataset
    ) -> None:
        with open(tiny_dataset_dir / "annotations.json") as f:
            coco = json.load(f)
        assert len(dataset) == len(coco["images"])

    def test_length_is_six(self, dataset: SyntheticSceneDataset) -> None:
        assert len(dataset) == 6


# ---------------------------------------------------------------------------
# __getitem__ return types
# ---------------------------------------------------------------------------


class TestGetItem:
    def test_returns_tuple(self, dataset: SyntheticSceneDataset) -> None:
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_tensor_type(self, dataset: SyntheticSceneDataset) -> None:
        tensor, _ = dataset[0]
        assert isinstance(tensor, torch.Tensor)

    def test_label_type(self, dataset: SyntheticSceneDataset) -> None:
        _, label = dataset[0]
        assert isinstance(label, int)

    def test_tensor_shape(self, dataset: SyntheticSceneDataset) -> None:
        tensor, _ = dataset[0]
        # Should be (C, H, W) = (3, 256, 256)
        assert tensor.ndim == 3
        assert tensor.shape[0] == 3

    def test_tensor_dtype(self, dataset: SyntheticSceneDataset) -> None:
        tensor, _ = dataset[0]
        assert tensor.dtype == torch.float32

    def test_all_items_accessible(self, dataset: SyntheticSceneDataset) -> None:
        for i in range(len(dataset)):
            tensor, label = dataset[i]
            assert tensor is not None
            assert label is not None


# ---------------------------------------------------------------------------
# Label validity
# ---------------------------------------------------------------------------


class TestLabels:
    def test_labels_in_valid_range(self, dataset: SyntheticSceneDataset) -> None:
        valid_labels = set(SyntheticSceneDataset.CATEGORY_TO_LABEL.values())
        for i in range(len(dataset)):
            _, label = dataset[i]
            assert label in valid_labels, f"Invalid label {label} at index {i}"

    def test_class_name_lookup(self) -> None:
        ds = SyntheticSceneDataset.__new__(SyntheticSceneDataset)
        ds._samples = []  # empty — just testing the method
        # Monkeypatch LABEL_TO_NAME lookup
        assert SyntheticSceneDataset.LABEL_TO_NAME[0] == "building"
        assert SyntheticSceneDataset.LABEL_TO_NAME[1] == "vehicle"
        assert SyntheticSceneDataset.LABEL_TO_NAME[2] == "open_area"

    def test_category_to_label_keys(self) -> None:
        # All COCO category IDs 1,2,3 must be mapped
        for cat_id in (1, 2, 3):
            assert cat_id in SyntheticSceneDataset.CATEGORY_TO_LABEL


# ---------------------------------------------------------------------------
# default_transform
# ---------------------------------------------------------------------------


class TestDefaultTransform:
    def test_output_is_tensor(self) -> None:
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        out = default_transform(img)
        assert isinstance(out, torch.Tensor)

    def test_output_shape(self) -> None:
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        out = default_transform(img)
        assert out.shape == (3, 256, 256)

    def test_output_dtype_float32(self) -> None:
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        out = default_transform(img)
        assert out.dtype == torch.float32

    def test_white_image_normalised(self) -> None:
        """White image (255,255,255) should produce tensor near ImageNet shift."""
        img = np.full((256, 256, 3), 255, dtype=np.uint8)
        out = default_transform(img)
        # Values should not be in [0,1] range after normalisation
        assert out.max().item() > 1.5 or out.min().item() < -1.5

    def test_different_inputs_differ(self) -> None:
        img_a = np.zeros((64, 64, 3), dtype=np.uint8)
        img_b = np.full((64, 64, 3), 128, dtype=np.uint8)
        out_a = default_transform(img_a)
        out_b = default_transform(img_b)
        assert not torch.equal(out_a, out_b)
