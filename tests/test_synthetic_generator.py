"""
tests/test_synthetic_generator.py
===================================
Unit tests for data/synthetic_generator.py.

Validates:
- Output image count and file existence
- Image shape and dtype
- COCO annotation structure and field types
- Determinism (same seed → same output)
- BBox validity (non-negative, within image bounds)
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data.synthetic_generator import (
    Annotation,
    BBox,
    SyntheticSceneGenerator,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    CATEGORIES,
    CATEGORY_IDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tmp_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shared temporary output directory for the module."""
    return tmp_path_factory.mktemp("samples")


@pytest.fixture(scope="module")
def generated_coco(tmp_output: Path) -> dict[str, Any]:
    """Run the generator once and return the COCO dict."""
    gen = SyntheticSceneGenerator(seed=42, output_dir=tmp_output, num_images=10)
    return gen.generate_all()


# ---------------------------------------------------------------------------
# BBox tests
# ---------------------------------------------------------------------------


class TestBBox:
    def test_area(self) -> None:
        bb = BBox(x=0, y=0, w=10, h=20)
        assert bb.area() == 200

    def test_to_coco(self) -> None:
        bb = BBox(x=5, y=10, w=30, h=40)
        assert bb.to_coco() == [5, 10, 30, 40]

    def test_to_xyxy(self) -> None:
        bb = BBox(x=5, y=10, w=30, h=40)
        assert bb.to_xyxy() == (5, 10, 35, 50)

    def test_zero_size_area(self) -> None:
        bb = BBox(x=0, y=0, w=0, h=0)
        assert bb.area() == 0


# ---------------------------------------------------------------------------
# Annotation tests
# ---------------------------------------------------------------------------


class TestAnnotation:
    def test_to_dict_keys(self) -> None:
        bb = BBox(x=1, y=2, w=10, h=10)
        ann = Annotation(annotation_id=1, image_id=1, category_id=1, bbox=bb)
        d = ann.to_dict()
        for key in ("id", "image_id", "category_id", "bbox", "area", "segmentation", "iscrowd"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_area_matches_bbox(self) -> None:
        bb = BBox(x=0, y=0, w=12, h=8)
        ann = Annotation(annotation_id=1, image_id=1, category_id=2, bbox=bb)
        assert ann.to_dict()["area"] == 96


# ---------------------------------------------------------------------------
# Generator — file output tests
# ---------------------------------------------------------------------------


class TestGeneratorOutput:
    def test_image_files_created(self, generated_coco: dict, tmp_output: Path) -> None:
        images_dir = tmp_output / "images"
        png_files = list(images_dir.glob("*.png"))
        assert len(png_files) == 10, f"Expected 10 PNG files, got {len(png_files)}"

    def test_annotations_json_created(self, generated_coco: dict, tmp_output: Path) -> None:
        ann_path = tmp_output / "annotations.json"
        assert ann_path.exists(), "annotations.json not found"

    def test_annotations_json_valid(self, tmp_output: Path) -> None:
        ann_path = tmp_output / "annotations.json"
        with open(ann_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_image_count_in_coco(self, generated_coco: dict) -> None:
        assert len(generated_coco["images"]) == 10

    def test_annotations_non_empty(self, generated_coco: dict) -> None:
        assert len(generated_coco["annotations"]) > 0

    def test_categories_present(self, generated_coco: dict) -> None:
        assert generated_coco["categories"] == CATEGORIES

    def test_info_safety_note(self, generated_coco: dict) -> None:
        info = generated_coco["info"]
        assert "safety_note" in info
        assert "synthetic" in info["safety_note"].lower()


# ---------------------------------------------------------------------------
# Generator — image validity tests
# ---------------------------------------------------------------------------


class TestImageValidity:
    def test_image_dimensions(self, tmp_output: Path) -> None:
        import cv2

        images_dir = tmp_output / "images"
        for png_path in sorted(images_dir.glob("*.png")):
            img = cv2.imread(str(png_path))
            assert img is not None, f"Could not read {png_path}"
            assert img.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3), (
                f"{png_path.name}: expected ({IMAGE_HEIGHT},{IMAGE_WIDTH},3), got {img.shape}"
            )

    def test_image_dtype_uint8(self, tmp_output: Path) -> None:
        import cv2

        images_dir = tmp_output / "images"
        for png_path in sorted(images_dir.glob("*.png")):
            img = cv2.imread(str(png_path))
            assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# Generator — annotation validity tests
# ---------------------------------------------------------------------------


class TestAnnotationValidity:
    def test_all_category_ids_valid(self, generated_coco: dict) -> None:
        valid_ids = set(CATEGORY_IDS)
        for ann in generated_coco["annotations"]:
            assert ann["category_id"] in valid_ids, (
                f"Unknown category_id {ann['category_id']}"
            )

    def test_bboxes_within_image(self, generated_coco: dict) -> None:
        for ann in generated_coco["annotations"]:
            x, y, w, h = ann["bbox"]
            assert x >= 0 and y >= 0, f"Negative bbox origin: {ann['bbox']}"
            assert w > 0 and h > 0, f"Non-positive bbox size: {ann['bbox']}"
            assert x + w <= IMAGE_WIDTH, f"Bbox exceeds image width: {ann['bbox']}"
            assert y + h <= IMAGE_HEIGHT, f"Bbox exceeds image height: {ann['bbox']}"

    def test_unique_annotation_ids(self, generated_coco: dict) -> None:
        ids = [ann["id"] for ann in generated_coco["annotations"]]
        assert len(ids) == len(set(ids)), "Duplicate annotation IDs found"

    def test_image_ids_match(self, generated_coco: dict) -> None:
        valid_image_ids = {img["id"] for img in generated_coco["images"]}
        for ann in generated_coco["annotations"]:
            assert ann["image_id"] in valid_image_ids, (
                f"Annotation references unknown image_id {ann['image_id']}"
            )

    def test_area_matches_bbox(self, generated_coco: dict) -> None:
        for ann in generated_coco["annotations"]:
            x, y, w, h = ann["bbox"]
            assert ann["area"] == w * h, f"Area mismatch for annotation {ann['id']}"


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_annotations(self, tmp_path: Path) -> None:
        """Two runs with the same seed must produce identical COCO annotations."""
        out_a = tmp_path / "run_a"
        out_b = tmp_path / "run_b"

        gen_a = SyntheticSceneGenerator(seed=99, output_dir=out_a, num_images=5)
        gen_b = SyntheticSceneGenerator(seed=99, output_dir=out_b, num_images=5)

        coco_a = gen_a.generate_all()
        coco_b = gen_b.generate_all()

        assert coco_a["annotations"] == coco_b["annotations"], (
            "Generator is not deterministic with the same seed"
        )

    def test_different_seeds_differ(self, tmp_path: Path) -> None:
        """Two runs with different seeds should produce different annotations."""
        out_a = tmp_path / "seed1"
        out_b = tmp_path / "seed2"

        gen_a = SyntheticSceneGenerator(seed=1, output_dir=out_a, num_images=5)
        gen_b = SyntheticSceneGenerator(seed=2, output_dir=out_b, num_images=5)

        coco_a = gen_a.generate_all()
        coco_b = gen_b.generate_all()

        # Annotations should differ (overwhelmingly likely with different seeds)
        assert coco_a["annotations"] != coco_b["annotations"]
