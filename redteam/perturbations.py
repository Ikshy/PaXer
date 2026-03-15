"""
redteam/perturbations.py
========================
Synthetic image perturbation library for CTHMP robustness evaluation.

Generates adversarial-style transformations on synthetic satellite imagery
to measure model fragility.  All perturbations operate on numpy uint8 arrays
and are deterministic given a fixed seed.

Available perturbations
-----------------------
  gaussian_noise    — additive Gaussian noise (sigma controllable)
  salt_pepper       — salt-and-pepper impulse noise
  gaussian_blur     — low-pass blurring (simulates sensor defocus)
  jpeg_compression  — lossy re-encode (simulates transmission artefacts)
  brightness_shift  — uniform brightness delta
  contrast_scale    — contrast multiplication
  occlusion_patch   — random black rectangle (simulates cloud / obstruction)
  rotation          — small-angle rotation (simulates sensor tilt)
  horizontal_flip   — mirror (tests rotation-invariance assumptions)

Safety note
-----------
These perturbations are for evaluating model robustness on synthetic data
only.  They must not be used to craft adversarial inputs against real
deployed systems.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

PerturbFn = Callable[[np.ndarray, random.Random], np.ndarray]


@dataclass
class PerturbationSpec:
    """Describes one perturbation and its parameter sweep."""

    name: str
    fn: PerturbFn
    # Human-readable severity levels → parameter dicts passed to fn
    severity_params: list[dict] = field(default_factory=list)

    def apply(self, image: np.ndarray, rng: random.Random, severity: int = 1) -> np.ndarray:
        """
        Apply this perturbation at a given severity level (0-indexed).

        Parameters
        ----------
        image    : np.ndarray  shape (H, W, 3) uint8
        rng      : random.Random  seeded RNG for reproducibility
        severity : int  index into severity_params (default 1 = medium)

        Returns
        -------
        np.ndarray  same shape and dtype as input
        """
        params = self.severity_params[min(severity, len(self.severity_params) - 1)]
        np_rng = np.random.default_rng(rng.randint(0, 2**31))
        return fn_with_params(self.fn, image, rng, np_rng, **params)


def fn_with_params(
    fn: PerturbFn,
    image: np.ndarray,
    rng: random.Random,
    np_rng: np.random.Generator,
    **params,
) -> np.ndarray:
    """Dispatch a perturbation function with keyword params injected."""
    import inspect

    sig = inspect.signature(fn)
    kwargs: dict = {}
    if "np_rng" in sig.parameters:
        kwargs["np_rng"] = np_rng
    kwargs.update(params)
    return fn(image, rng, **kwargs)


# ---------------------------------------------------------------------------
# Individual perturbation functions
# ---------------------------------------------------------------------------


def _gaussian_noise(
    image: np.ndarray,
    rng: random.Random,
    *,
    sigma: float = 15.0,
    np_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add zero-mean Gaussian noise with standard deviation `sigma`."""
    if np_rng is None:
        np_rng = np.random.default_rng()
    noise = np_rng.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _salt_pepper(
    image: np.ndarray,
    rng: random.Random,
    *,
    density: float = 0.05,
    np_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly set `density` fraction of pixels to 0 or 255."""
    if np_rng is None:
        np_rng = np.random.default_rng()
    out = image.copy()
    mask = np_rng.random(image.shape[:2])
    out[mask < density / 2] = 0
    out[(mask >= density / 2) & (mask < density)] = 255
    return out


def _gaussian_blur(
    image: np.ndarray,
    rng: random.Random,
    *,
    ksize: int = 5,
    **_kwargs,
) -> np.ndarray:
    """Apply Gaussian blur with kernel size `ksize` (must be odd)."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def _jpeg_compression(
    image: np.ndarray,
    rng: random.Random,
    *,
    quality: int = 30,
    **_kwargs,
) -> np.ndarray:
    """Re-encode as JPEG at `quality` (0–100) to introduce compression artefacts."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _brightness_shift(
    image: np.ndarray,
    rng: random.Random,
    *,
    delta: int = 40,
    **_kwargs,
) -> np.ndarray:
    """Shift all pixel values by `delta` (can be negative to darken)."""
    shifted = image.astype(np.int32) + delta
    return np.clip(shifted, 0, 255).astype(np.uint8)


def _contrast_scale(
    image: np.ndarray,
    rng: random.Random,
    *,
    factor: float = 0.5,
    **_kwargs,
) -> np.ndarray:
    """Multiply contrast by `factor` around mid-grey (128)."""
    scaled = (image.astype(np.float32) - 128) * factor + 128
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _occlusion_patch(
    image: np.ndarray,
    rng: random.Random,
    *,
    patch_frac: float = 0.25,
    **_kwargs,
) -> np.ndarray:
    """Black out a random rectangle covering `patch_frac` fraction of the image."""
    h, w = image.shape[:2]
    ph = int(h * patch_frac)
    pw = int(w * patch_frac)
    py = rng.randint(0, h - ph)
    px = rng.randint(0, w - pw)
    out = image.copy()
    out[py : py + ph, px : px + pw] = 0
    return out


def _rotation(
    image: np.ndarray,
    rng: random.Random,
    *,
    angle_deg: float = 15.0,
    **_kwargs,
) -> np.ndarray:
    """Rotate by `angle_deg` degrees (bilinear interpolation, black fill)."""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)


def _horizontal_flip(
    image: np.ndarray,
    rng: random.Random,
    **_kwargs,
) -> np.ndarray:
    """Flip the image horizontally."""
    return cv2.flip(image, 1)


# ---------------------------------------------------------------------------
# Registry — all available perturbations with severity sweeps
# ---------------------------------------------------------------------------

PERTURBATIONS: dict[str, PerturbationSpec] = {
    "gaussian_noise": PerturbationSpec(
        name="gaussian_noise",
        fn=_gaussian_noise,
        severity_params=[
            {"sigma": 5.0},    # mild
            {"sigma": 25.0},   # medium
            {"sigma": 60.0},   # severe
        ],
    ),
    "salt_pepper": PerturbationSpec(
        name="salt_pepper",
        fn=_salt_pepper,
        severity_params=[
            {"density": 0.01},
            {"density": 0.05},
            {"density": 0.15},
        ],
    ),
    "gaussian_blur": PerturbationSpec(
        name="gaussian_blur",
        fn=_gaussian_blur,
        severity_params=[
            {"ksize": 3},
            {"ksize": 7},
            {"ksize": 15},
        ],
    ),
    "jpeg_compression": PerturbationSpec(
        name="jpeg_compression",
        fn=_jpeg_compression,
        severity_params=[
            {"quality": 70},
            {"quality": 30},
            {"quality": 10},
        ],
    ),
    "brightness_shift": PerturbationSpec(
        name="brightness_shift",
        fn=_brightness_shift,
        severity_params=[
            {"delta": 20},
            {"delta": 60},
            {"delta": 100},
        ],
    ),
    "contrast_scale": PerturbationSpec(
        name="contrast_scale",
        fn=_contrast_scale,
        severity_params=[
            {"factor": 0.8},
            {"factor": 0.5},
            {"factor": 0.2},
        ],
    ),
    "occlusion_patch": PerturbationSpec(
        name="occlusion_patch",
        fn=_occlusion_patch,
        severity_params=[
            {"patch_frac": 0.10},
            {"patch_frac": 0.25},
            {"patch_frac": 0.40},
        ],
    ),
    "rotation": PerturbationSpec(
        name="rotation",
        fn=_rotation,
        severity_params=[
            {"angle_deg": 5.0},
            {"angle_deg": 15.0},
            {"angle_deg": 45.0},
        ],
    ),
    "horizontal_flip": PerturbationSpec(
        name="horizontal_flip",
        fn=_horizontal_flip,
        severity_params=[
            {},  # no params — flip is binary
            {},
            {},
        ],
    ),
}


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def list_perturbations() -> list[str]:
    """Return names of all registered perturbations."""
    return sorted(PERTURBATIONS.keys())


def apply_perturbation(
    image: np.ndarray,
    name: str,
    severity: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """
    Apply a named perturbation to an image.

    Parameters
    ----------
    image    : np.ndarray  (H, W, 3) uint8
    name     : str  one of list_perturbations()
    severity : int  0=mild, 1=medium, 2=severe
    seed     : int  for reproducibility

    Returns
    -------
    np.ndarray  perturbed image, same shape/dtype
    """
    if name not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation {name!r}. Available: {list_perturbations()}")
    spec = PERTURBATIONS[name]
    rng = random.Random(seed)
    return spec.apply(image, rng, severity=severity)
