"""
data/synthetic_generator.py
============================
Deterministic synthetic satellite-style imagery generator for CTHMP.

Generates small grayscale/RGB images containing simple geometric shapes that
represent humanitarian-relevant objects (buildings, vehicles, open areas).
Produces COCO-style JSON annotations alongside each image.

Usage
-----
    python data/synthetic_generator.py [--num-images N] [--seed S] [--output-dir PATH]

Safety note
-----------
All imagery is **fully synthetic**. No real satellite data is used.
Shapes and labels are coarse categories for humanitarian monitoring research only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
CHANNELS = 3

# Label map: category_id -> name
# Kept intentionally coarse — no targeting-relevant labels.
CATEGORIES: list[dict[str, Any]] = [
    {"id": 1, "name": "building", "supercategory": "structure"},
    {"id": 2, "name": "vehicle", "supercategory": "transport"},
    {"id": 3, "name": "open_area", "supercategory": "terrain"},
]
CATEGORY_IDS = [c["id"] for c in CATEGORIES]

# BGR colour palette per category
CATEGORY_COLOURS: dict[int, tuple[int, int, int]] = {
    1: (180, 180, 220),  # building — pale blue-grey
    2: (60, 180, 60),    # vehicle — green
    3: (200, 220, 180),  # open_area — light tan
}

# Background noise levels
BG_MEAN = 80
BG_STD = 15


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BBox:
    """Axis-aligned bounding box in COCO format [x, y, width, height]."""

    x: int
    y: int
    w: int
    h: int

    def area(self) -> int:
        return self.w * self.h

    def to_coco(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass
class Annotation:
    """Single object annotation (COCO-compatible)."""

    annotation_id: int
    image_id: int
    category_id: int
    bbox: BBox
    segmentation: list[list[int]] = field(default_factory=list)
    iscrowd: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.annotation_id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "bbox": self.bbox.to_coco(),
            "area": self.bbox.area(),
            "segmentation": self.segmentation,
            "iscrowd": self.iscrowd,
        }


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------


class SyntheticSceneGenerator:
    """
    Generates deterministic synthetic satellite-style scenes.

    Parameters
    ----------
    seed : int
        Random seed for full reproducibility.
    output_dir : Path
        Directory where images and annotations are saved.
    num_images : int
        Number of scenes to generate.
    image_width, image_height : int
        Pixel dimensions of each scene.
    """

    def __init__(
        self,
        seed: int = 42,
        output_dir: Path = Path("data/samples"),
        num_images: int = 10,
        image_width: int = IMAGE_WIDTH,
        image_height: int = IMAGE_HEIGHT,
    ) -> None:
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.num_images = num_images
        self.image_width = image_width
        self.image_height = image_height

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self) -> dict[str, Any]:
        """
        Generate all scenes and return a COCO-formatted annotation dict.

        Returns
        -------
        dict
            Full COCO dataset dict with keys: info, images, annotations, categories.
        """
        logger.info("Generating %d synthetic scenes (seed=%d)…", self.num_images, self.seed)

        coco: dict[str, Any] = {
            "info": {
                "description": "CTHMP Synthetic Satellite Dataset",
                "version": "1.0",
                "year": 2024,
                "contributor": "CTHMP synthetic generator",
                "url": "",
                "date_created": "2024-01-01",
                "safety_note": (
                    "Fully synthetic. No real imagery. "
                    "Humanitarian/transparency research use only."
                ),
            },
            "licenses": [],
            "categories": CATEGORIES,
            "images": [],
            "annotations": [],
        }

        annotation_id = 1
        for image_id in range(1, self.num_images + 1):
            filename = f"scene_{image_id:04d}.png"
            image_path = self.output_dir / "images" / filename

            image, annotations = self._generate_scene(image_id, annotation_id)

            # Save image
            cv2.imwrite(str(image_path), image)

            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": filename,
                    "width": self.image_width,
                    "height": self.image_height,
                    "license": 0,
                    "synthetic": True,
                }
            )
            for ann in annotations:
                coco["annotations"].append(ann.to_dict())
                annotation_id += 1

        # Save COCO JSON
        annotations_path = self.output_dir / "annotations.json"
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)

        logger.info("Saved %d images → %s/images/", self.num_images, self.output_dir)
        logger.info("Saved annotations → %s", annotations_path)
        return coco

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_background(self) -> np.ndarray:
        """Create a noisy grey background simulating bare ground."""
        noise = self._np_rng.normal(BG_MEAN, BG_STD, (self.image_height, self.image_width, CHANNELS))
        bg = np.clip(noise, 0, 255).astype(np.uint8)
        return bg

    def _random_bbox(self, min_size: int = 15, max_size: int = 50) -> BBox:
        """Sample a random bounding box that fits within the image."""
        w = self._rng.randint(min_size, max_size)
        h = self._rng.randint(min_size, max_size)
        x = self._rng.randint(0, self.image_width - w)
        y = self._rng.randint(0, self.image_height - h)
        return BBox(x=x, y=y, w=w, h=h)

    def _draw_building(self, image: np.ndarray, bbox: BBox) -> None:
        """Draw a rectangle (building footprint)."""
        x1, y1, x2, y2 = bbox.to_xyxy()
        colour = CATEGORY_COLOURS[1]
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness=-1)
        # Add a simple roof-line detail
        mid_x = (x1 + x2) // 2
        cv2.line(image, (mid_x, y1), (mid_x, y2), (150, 150, 190), 1)

    def _draw_vehicle(self, image: np.ndarray, bbox: BBox) -> None:
        """Draw a small ellipse (vehicle blob)."""
        cx = bbox.x + bbox.w // 2
        cy = bbox.y + bbox.h // 2
        axes = (max(bbox.w // 2, 4), max(bbox.h // 2, 4))
        colour = CATEGORY_COLOURS[2]
        cv2.ellipse(image, (cx, cy), axes, 0, 0, 360, colour, thickness=-1)

    def _draw_open_area(self, image: np.ndarray, bbox: BBox) -> None:
        """Draw a filled polygon (open area / clearing)."""
        x1, y1, x2, y2 = bbox.to_xyxy()
        pts = np.array(
            [[x1, y2], [x1 + bbox.w // 3, y1], [x2, y1], [x2, y2]],
            dtype=np.int32,
        )
        colour = CATEGORY_COLOURS[3]
        cv2.fillPoly(image, [pts], colour)

    _DRAW_FN = {
        1: _draw_building,
        2: _draw_vehicle,
        3: _draw_open_area,
    }

    def _generate_scene(
        self, image_id: int, start_annotation_id: int
    ) -> tuple[np.ndarray, list[Annotation]]:
        """
        Generate one synthetic scene image and its annotations.

        Returns
        -------
        image : np.ndarray  shape (H, W, 3) uint8
        annotations : list[Annotation]
        """
        image = self._make_background()
        annotations: list[Annotation] = []

        num_objects = self._rng.randint(3, 8)
        ann_id = start_annotation_id

        for _ in range(num_objects):
            category_id = self._rng.choice(CATEGORY_IDS)
            bbox = self._random_bbox()

            # Draw object on image
            draw_fn = self._DRAW_FN[category_id]
            draw_fn(self, image, bbox)

            # Add slight Gaussian blur to soften synthetic edges
            roi = image[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
            if roi.size > 0:
                blurred = cv2.GaussianBlur(roi, (3, 3), 0)
                image[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w] = blurred

            annotations.append(
                Annotation(
                    annotation_id=ann_id,
                    image_id=image_id,
                    category_id=category_id,
                    bbox=bbox,
                )
            )
            ann_id += 1

        return image, annotations


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic satellite-style scenes for CTHMP research."
    )
    parser.add_argument("--num-images", type=int, default=int(os.getenv("SYNTH_NUM_IMAGES", "10")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SYNTH_SEED", "42")))
    parser.add_argument(
        "--output-dir", type=str, default=os.getenv("SYNTH_OUTPUT_DIR", "data/samples")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = SyntheticSceneGenerator(
        seed=args.seed,
        output_dir=Path(args.output_dir),
        num_images=args.num_images,
    )
    coco = generator.generate_all()
    total_annotations = len(coco["annotations"])
    logger.info(
        "Done. %d images, %d annotations.", args.num_images, total_annotations
    )


if __name__ == "__main__":
    main()
