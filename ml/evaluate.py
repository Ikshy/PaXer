"""
ml/evaluate.py
==============
Standalone evaluation script for a trained CTHMP scene classifier.

Loads a saved model checkpoint, runs inference on the full synthetic dataset,
and prints a classification report + per-class accuracy.

Usage
-----
    python ml/evaluate.py --checkpoint ml/artifacts/best_model.pt

Outputs
-------
    Prints metrics to stdout.
    Saves eval_report.json to the same directory as the checkpoint.

Safety note
-----------
All evaluation is performed on synthetic data.  Results must be reviewed
by a human analyst before drawing any operational conclusions.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.dataset import SyntheticSceneDataset, default_transform
from ml.model import build_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-class metrics helper
# ---------------------------------------------------------------------------


def compute_per_class_metrics(
    all_preds: list[int],
    all_labels: list[int],
    num_classes: int,
) -> dict[int, dict[str, float]]:
    """
    Compute per-class precision, recall, and F1.

    Returns
    -------
    dict mapping class_id → {precision, recall, f1, support}
    """
    tp: dict[int, int] = defaultdict(int)
    fp: dict[int, int] = defaultdict(int)
    fn: dict[int, int] = defaultdict(int)

    for pred, label in zip(all_preds, all_labels):
        if pred == label:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[label] += 1

    support: dict[int, int] = defaultdict(int)
    for label in all_labels:
        support[label] += 1

    metrics: dict[int, dict[str, float]] = {}
    for cls in range(num_classes):
        prec = tp[cls] / max(tp[cls] + fp[cls], 1)
        rec = tp[cls] / max(tp[cls] + fn[cls], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        metrics[cls] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support[cls],
        }
    return metrics


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_dir: Path = Path("data"),
    batch_size: int = 4,
) -> dict:
    """
    Run full evaluation and return a metrics dict.

    Parameters
    ----------
    checkpoint_path : Path  Path to a .pt state-dict file.
    data_dir : Path         Root data directory (must contain samples/).
    batch_size : int

    Returns
    -------
    eval_report dict (also saved as eval_report.json next to checkpoint).
    """
    annotations_path = data_dir / "samples" / "annotations.json"
    images_dir = data_dir / "samples" / "images"

    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Annotations not found at {annotations_path}. "
            "Run python data/synthetic_generator.py first."
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SyntheticSceneDataset(
        annotations_path=annotations_path,
        images_dir=images_dir,
        transform=default_transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=SyntheticSceneDataset.NUM_CLASSES, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    overall_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)

    per_class = compute_per_class_metrics(
        all_preds, all_labels, SyntheticSceneDataset.NUM_CLASSES
    )

    # Re-key with class names for readability
    named_per_class = {
        SyntheticSceneDataset.LABEL_TO_NAME[cls]: metrics
        for cls, metrics in per_class.items()
    }

    report = {
        "checkpoint": str(checkpoint_path),
        "overall_accuracy": round(overall_acc, 4),
        "num_samples": len(all_labels),
        "per_class": named_per_class,
        "safety_note": (
            "Evaluated on synthetic data only. "
            "Human analyst sign-off required before operational use."
        ),
    }

    # Pretty print
    logger.info("=" * 50)
    logger.info("Evaluation Report")
    logger.info("  Checkpoint : %s", checkpoint_path)
    logger.info("  Samples    : %d", len(all_labels))
    logger.info("  Overall Acc: %.4f", overall_acc)
    logger.info("-" * 50)
    for cls_name, m in named_per_class.items():
        logger.info(
            "  %-12s  P=%.3f  R=%.3f  F1=%.3f  support=%d",
            cls_name,
            m["precision"],
            m["recall"],
            m["f1"],
            m["support"],
        )
    logger.info("=" * 50)

    # Save report alongside checkpoint
    report_path = checkpoint_path.parent / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved eval report → %s", report_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CTHMP checkpoint.")
    parser.add_argument(
        "--checkpoint", default="ml/artifacts/best_model.pt", type=str
    )
    parser.add_argument("--data-dir", default="data", type=str)
    parser.add_argument("--batch-size", default=4, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
    )
