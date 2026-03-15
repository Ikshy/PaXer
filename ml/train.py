"""
ml/train.py
===========
Training script for the CTHMP scene classifier.

Usage
-----
    python ml/train.py [options]

Key flags
---------
    --data-dir        Root dir containing samples/ (default: data)
    --output-dir      Where to save checkpoints + exported model (default: ml/artifacts)
    --epochs          Number of training epochs (default: 10)
    --batch-size      Mini-batch size (default: 4)
    --lr              Learning rate (default: 1e-3)
    --seed            RNG seed (default: 42)
    --val-split       Fraction of data held out for validation (default: 0.2)
    --num-workers     DataLoader workers (default: 0 for Windows/Mac compat)

Outputs (all in --output-dir)
------------------------------
    best_model.pt         — state-dict of the best val-accuracy checkpoint
    last_model.pt         — state-dict after final epoch
    train_metrics.json    — per-epoch loss/accuracy history
    model_provenance.json — seed, hyperparams, data hash, timestamp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ml.dataset import SyntheticSceneDataset, default_transform
from ml.model import build_model, count_parameters

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Fix all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data hash (provenance)
# ---------------------------------------------------------------------------


def hash_annotations(annotations_path: Path) -> str:
    """Return SHA-256 of the annotations file for provenance tracking."""
    h = hashlib.sha256()
    with open(annotations_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Training & evaluation helpers
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one training epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float  (0–1)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate model on a DataLoader.

    Returns
    -------
    avg_loss : float
    accuracy : float  (0–1)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> dict[str, Any]:
    """
    Run the full train/val loop.

    Returns
    -------
    dict containing final metrics and provenance info.
    """
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = data_dir / "samples" / "annotations.json"
    images_dir = data_dir / "samples" / "images"

    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Annotations not found at {annotations_path}. "
            "Run: python data/synthetic_generator.py first."
        )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = SyntheticSceneDataset(
        annotations_path=annotations_path,
        images_dir=images_dir,
        transform=default_transform,
    )

    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    if n_train < 1:
        raise ValueError(
            f"Dataset too small ({n_total} samples) for val_split={args.val_split}."
        )

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info(
        "Dataset: %d train / %d val | classes: %d",
        n_train,
        n_val,
        SyntheticSceneDataset.NUM_CLASSES,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = build_model(
        num_classes=SyntheticSceneDataset.NUM_CLASSES, pretrained=False
    ).to(device)
    logger.info("Model parameters: %d", count_parameters(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_epoch = 0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_metrics)

        logger.info(
            "Epoch %02d/%02d  train_loss=%.4f  train_acc=%.3f  "
            "val_loss=%.4f  val_acc=%.3f",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    elapsed = time.time() - start_time

    # Save last checkpoint
    torch.save(model.state_dict(), output_dir / "last_model.pt")

    # ------------------------------------------------------------------
    # Provenance record (required by CTHMP audit policy)
    # ------------------------------------------------------------------
    provenance: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": round(elapsed, 2),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "val_split": args.val_split,
        "n_train": n_train,
        "n_val": n_val,
        "best_val_acc": round(best_val_acc, 6),
        "best_epoch": best_epoch,
        "annotations_sha256": hash_annotations(annotations_path),
        "model_arch": "SceneClassifier/MobileNetV2",
        "device": str(device),
        "num_classes": SyntheticSceneDataset.NUM_CLASSES,
        "safety_note": (
            "Trained on fully synthetic data only. "
            "All inferences require human analyst sign-off before use."
        ),
    }

    with open(output_dir / "model_provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info(
        "Training complete. Best val_acc=%.3f at epoch %d. "
        "Artifacts saved to %s/",
        best_val_acc,
        best_epoch,
        output_dir,
    )

    return provenance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CTHMP scene classifier.")
    parser.add_argument("--data-dir", default=os.getenv("SYNTH_OUTPUT_DIR", "data"), type=str)
    parser.add_argument("--output-dir", default="ml/artifacts", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--val-split", default=0.2, type=float)
    parser.add_argument("--num-workers", default=0, type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
