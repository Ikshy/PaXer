"""
backend/services/store.py
==========================
Simple local-filesystem object store for ingested images.

In production this would be swapped for a MinIO/S3 client.  The interface
is identical so the routers never need to change — only this module.

Files are stored under LOCAL_STORE_DIR (default: data/store/) using the
pattern:  <store_dir>/<image_id>/<original_filename>
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from backend.config import get_settings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_upload(image_id: str, filename: str, data: bytes) -> tuple[str, str, int, int]:
    """
    Persist an uploaded image to the local store.

    Parameters
    ----------
    image_id : str   UUID for the new ImageRecord
    filename : str   Original filename
    data     : bytes Raw image bytes

    Returns
    -------
    store_path : str   Relative path (for DB storage)
    sha256     : str   Hex digest of the raw bytes
    width      : int   Image width in pixels
    height     : int   Image height in pixels
    """
    import cv2
    import numpy as np

    settings = get_settings()
    store_root = Path(settings.local_store_dir)
    dest_dir = store_root / image_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / filename
    dest_path.write_bytes(data)

    # Compute SHA-256
    sha256 = hashlib.sha256(data).hexdigest()

    # Read dimensions
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {filename}")
    height, width = img.shape[:2]

    store_path = str(dest_path.relative_to(store_root))
    return store_path, sha256, width, height


def load_image_bytes(store_path: str) -> bytes:
    """
    Load raw image bytes from the store.

    Parameters
    ----------
    store_path : str  Path relative to LOCAL_STORE_DIR

    Returns
    -------
    bytes
    """
    settings = get_settings()
    full_path = Path(settings.local_store_dir) / store_path
    if not full_path.exists():
        raise FileNotFoundError(f"Image not found in store: {store_path}")
    return full_path.read_bytes()


def delete_image(store_path: str) -> None:
    """Remove an image and its parent directory from the store."""
    settings = get_settings()
    full_path = Path(settings.local_store_dir) / store_path
    parent = full_path.parent
    if parent.exists():
        shutil.rmtree(parent, ignore_errors=True)
