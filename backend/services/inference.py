"""
backend/services/inference.py
==============================
Model inference service.

Loads the trained SceneClassifier checkpoint once (at module level),
runs forward pass, and returns predictions with full provenance metadata.

Safety note
-----------
Results from this service are *candidates* only.  They must not be treated
as authoritative until a human analyst calls the /verify endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from backend.config import get_settings
from ml.dataset import default_transform
from ml.model import build_model, SceneClassifier

logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "building", 1: "vehicle", 2: "open_area"}
NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Model loader (cached singleton)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_model() -> tuple[SceneClassifier, str, str]:
    """
    Load the checkpoint once and cache it.

    Returns
    -------
    model      : SceneClassifier  in eval mode on CPU/GPU
    checkpoint : str              path used
    sha256     : str              hex digest of the checkpoint file
    """
    settings = get_settings()
    checkpoint_path = Path(settings.model_checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. "
            "Run python ml/train.py first."
        )

    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    logger.info("Model loaded from %s (sha256=%s…)", checkpoint_path, sha256[:12])

    return model, str(checkpoint_path), sha256


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(image_bytes: bytes) -> dict:
    """
    Run the scene classifier on raw image bytes.

    Parameters
    ----------
    image_bytes : bytes  Raw PNG/JPEG bytes of the image

    Returns
    -------
    dict with keys:
        predicted_class  : int
        predicted_label  : str
        confidence       : float  (0–1, softmax probability of top class)
        logits_json      : str    (JSON array of raw logits)
        model_arch       : str
        model_checkpoint : str
        model_sha256     : str
    """
    import cv2

    model, checkpoint_path, model_sha256 = _load_model()
    settings = get_settings()

    # Decode image
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Cannot decode image bytes for inference.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Transform → tensor (1, 3, H, W)
    tensor = default_transform(img_rgb).unsqueeze(0)

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)                          # (1, NUM_CLASSES)
        probs = F.softmax(logits, dim=1)[0]             # (NUM_CLASSES,)

    predicted_class = int(probs.argmax().item())
    confidence = float(probs[predicted_class].item())
    logits_list = logits[0].cpu().tolist()

    return {
        "predicted_class": predicted_class,
        "predicted_label": LABEL_NAMES[predicted_class],
        "confidence": round(confidence, 6),
        "logits_json": json.dumps([round(v, 6) for v in logits_list]),
        "model_arch": settings.model_arch,
        "model_checkpoint": checkpoint_path,
        "model_sha256": model_sha256,
    }
