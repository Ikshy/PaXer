"""
redteam/eval.py
===============
Robustness evaluation harness for the CTHMP scene classifier.

For each scenario in scenarios.py:
  1. Load the full synthetic dataset.
  2. Apply the scenario's perturbation(s) to every image in memory.
  3. Run model inference on the perturbed images.
  4. Compute per-class precision, recall, F1 and overall accuracy.
  5. Compute false-positive rate (FPR) and false-negative rate (FNR)
     per class.
  6. Emit a structured JSON report + a human-readable console summary.

Usage
-----
    python redteam/eval.py [--checkpoint ml/artifacts/best_model.pt]
                           [--data-dir data]
                           [--output-dir redteam/reports]
                           [--scenarios clean,mild_noise,severe_noise]
                           [--seed 42]

Safety note
-----------
All evaluation is performed on fully synthetic data.
This harness must never be pointed at real operational imagery.
Results require human review before informing any decisions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data.synthetic_generator import SyntheticSceneGenerator
from ml.dataset import SyntheticSceneDataset, default_transform
from ml.model import build_model
from redteam.perturbations import apply_perturbation
from redteam.scenarios import SCENARIOS, Scenario

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES = SyntheticSceneDataset.NUM_CLASSES
LABEL_NAMES = SyntheticSceneDataset.LABEL_TO_NAME


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------


def _compute_metrics(
    all_preds: list[int],
    all_labels: list[int],
) -> dict[str, Any]:
    """
    Compute accuracy, per-class precision/recall/F1/FPR/FNR.

    Returns
    -------
    dict with keys:
        overall_accuracy, per_class (dict label_name -> metrics)
    """
    n = len(all_labels)
    if n == 0:
        return {"overall_accuracy": 0.0, "per_class": {}}

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    overall_acc = correct / n

    tp: dict[int, int] = defaultdict(int)
    fp: dict[int, int] = defaultdict(int)
    fn: dict[int, int] = defaultdict(int)
    tn: dict[int, int] = defaultdict(int)

    for pred, label in zip(all_preds, all_labels):
        for cls in range(NUM_CLASSES):
            if pred == cls and label == cls:
                tp[cls] += 1
            elif pred == cls and label != cls:
                fp[cls] += 1
            elif pred != cls and label == cls:
                fn[cls] += 1
            else:
                tn[cls] += 1

    per_class: dict[str, dict[str, float]] = {}
    for cls in range(NUM_CLASSES):
        prec = tp[cls] / max(tp[cls] + fp[cls], 1)
        rec  = tp[cls] / max(tp[cls] + fn[cls], 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        # FPR = FP / (FP + TN)  — rate of false alarms
        fpr  = fp[cls] / max(fp[cls] + tn[cls], 1)
        # FNR = FN / (FN + TP)  — rate of missed detections
        fnr  = fn[cls] / max(fn[cls] + tp[cls], 1)
        support = tp[cls] + fn[cls]

        per_class[LABEL_NAMES[cls]] = {
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "fpr":       round(fpr, 4),
            "fnr":       round(fnr, 4),
            "support":   support,
        }

    return {
        "overall_accuracy": round(overall_acc, 4),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Single-scenario evaluation
# ---------------------------------------------------------------------------


def _eval_scenario(
    scenario: Scenario,
    images_dir: Path,
    dataset: SyntheticSceneDataset,
    model: torch.nn.Module,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """
    Run the model on all dataset images after applying scenario perturbations.

    Returns a metrics dict for this scenario.
    """
    all_preds: list[int] = []
    all_labels: list[int] = []

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        img_path, _ = dataset._samples[idx]

        # Load raw image (BGR)
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot load %s — skipping.", img_path)
            continue

        # Apply each perturbation in sequence
        for pert_name, severity in scenario.perturbations:
            img = apply_perturbation(img, pert_name, severity=severity, seed=seed + idx)

        # BGR -> RGB -> tensor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = default_transform(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            pred = int(logits.argmax(dim=1).item())

        all_preds.append(pred)
        all_labels.append(label)

    metrics = _compute_metrics(all_preds, all_labels)
    metrics["n_samples"] = len(all_labels)
    return metrics


# ---------------------------------------------------------------------------
# Main eval harness
# ---------------------------------------------------------------------------


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    """
    Run all requested scenarios and write a report.

    Returns the full report dict.
    """
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = data_dir / "samples" / "annotations.json"
    images_dir = data_dir / "samples" / "images"

    # Auto-generate synthetic data if not present
    if not annotations_path.exists():
        logger.info("Synthetic data not found — generating (seed=%d)…", args.seed)
        gen = SyntheticSceneGenerator(
            seed=args.seed,
            output_dir=data_dir / "samples",
            num_images=20,
        )
        gen.generate_all()

    # Load dataset (labels only; images loaded individually below)
    dataset = SyntheticSceneDataset(
        annotations_path=annotations_path,
        images_dir=images_dir,
    )
    logger.info("Dataset: %d samples, %d classes", len(dataset), NUM_CLASSES)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run python ml/train.py first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    logger.info("Model loaded from %s (device=%s)", checkpoint_path, device)

    # Determine which scenarios to run
    requested = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    if requested == ["all"]:
        scenarios_to_run = SCENARIOS
    else:
        from redteam.scenarios import SCENARIO_MAP
        scenarios_to_run = [SCENARIO_MAP[n] for n in requested if n in SCENARIO_MAP]
        unknown = [n for n in requested if n not in SCENARIO_MAP]
        if unknown:
            logger.warning("Unknown scenarios (skipped): %s", unknown)

    logger.info("Running %d scenario(s)…", len(scenarios_to_run))

    # ── Run scenarios ──────────────────────────────────────────────────────
    results: dict[str, Any] = {}
    baseline_acc: float | None = None

    for scenario in scenarios_to_run:
        t0 = time.time()
        logger.info("  [%s] %s", scenario.name, scenario.description)
        metrics = _eval_scenario(scenario, images_dir, dataset, model, device, args.seed)
        elapsed = time.time() - t0

        if scenario.name == "clean":
            baseline_acc = metrics["overall_accuracy"]

        # Accuracy drop relative to clean baseline
        acc = metrics["overall_accuracy"]
        drop = None if baseline_acc is None else round(baseline_acc - acc, 4)

        results[scenario.name] = {
            "description": scenario.description,
            "perturbations": [
                {"name": n, "severity": s} for n, s in scenario.perturbations
            ],
            "overall_accuracy": acc,
            "accuracy_drop_vs_clean": drop,
            "per_class": metrics["per_class"],
            "n_samples": metrics["n_samples"],
            "elapsed_s": round(elapsed, 2),
        }

        logger.info(
            "    → acc=%.3f  drop=%s  (%.1fs)",
            acc,
            f"{drop:+.3f}" if drop is not None else "n/a",
            elapsed,
        )

    # ── Build full report ──────────────────────────────────────────────────
    report: dict[str, Any] = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint_path),
            "data_dir": str(data_dir),
            "n_scenarios": len(scenarios_to_run),
            "seed": args.seed,
            "device": str(device),
            "safety_note": (
                "All evaluation performed on fully synthetic data only. "
                "Results require human review before informing any decisions. "
                "This harness must not be used with real operational imagery."
            ),
        },
        "summary": _build_summary(results, baseline_acc),
        "scenarios": results,
    }

    # Save JSON report
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = output_dir / f"robustness_report_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Also write a "latest" symlink-style copy
    latest_path = output_dir / "robustness_report_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Report saved → %s", report_path)

    # Console summary table
    _print_summary_table(results, baseline_acc)

    return report


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _build_summary(
    results: dict[str, Any],
    baseline_acc: float | None,
) -> dict[str, Any]:
    """Build a high-level summary across all scenarios."""
    accs = [v["overall_accuracy"] for v in results.values()]

    # Worst per-class FPR / FNR across all scenarios
    worst_fpr: dict[str, float] = defaultdict(float)
    worst_fnr: dict[str, float] = defaultdict(float)
    for scenario_data in results.values():
        for cls_name, m in scenario_data["per_class"].items():
            worst_fpr[cls_name] = max(worst_fpr[cls_name], m["fpr"])
            worst_fnr[cls_name] = max(worst_fnr[cls_name], m["fnr"])

    return {
        "baseline_accuracy": baseline_acc,
        "min_accuracy_across_scenarios": round(min(accs), 4) if accs else None,
        "max_accuracy_drop": (
            round(baseline_acc - min(accs), 4)
            if baseline_acc is not None and accs
            else None
        ),
        "worst_fpr_per_class": {k: round(v, 4) for k, v in worst_fpr.items()},
        "worst_fnr_per_class": {k: round(v, 4) for k, v in worst_fnr.items()},
    }


def _print_summary_table(
    results: dict[str, Any],
    baseline_acc: float | None,
) -> None:
    """Print a human-readable summary table to stdout."""
    sep = "─" * 72
    print(f"\n{sep}")
    print("  CTHMP Robustness Evaluation — Summary")
    print(sep)
    print(f"  {'Scenario':<30} {'Acc':>6}  {'Drop':>7}  {'Notes'}")
    print(sep)
    for name, data in results.items():
        acc = data["overall_accuracy"]
        drop = data["accuracy_drop_vs_clean"]
        drop_str = f"{drop:+.3f}" if drop is not None else "   n/a"
        pert_names = ", ".join(p["name"] for p in data["perturbations"]) or "none"
        print(f"  {name:<30} {acc:>6.3f}  {drop_str:>7}  [{pert_names}]")
    print(sep)
    if baseline_acc is not None:
        accs = [d["overall_accuracy"] for d in results.values()]
        worst_drop = baseline_acc - min(accs)
        print(f"  Baseline: {baseline_acc:.3f}  |  Worst drop: {worst_drop:+.3f}")
    print(f"{sep}\n")
    print(
        "  ⚠  Results are on synthetic data only. "
        "Human review required before any use.\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CTHMP robustness red-team evaluation harness."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("MODEL_CHECKPOINT", "ml/artifacts/best_model.pt"),
        help="Path to trained .pt checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("SYNTH_OUTPUT_DIR", "data"),
        help="Root data directory (must contain samples/).",
    )
    parser.add_argument(
        "--output-dir",
        default="redteam/reports",
        help="Directory to write reports.",
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help=(
            "Comma-separated list of scenario names, or 'all'. "
            f"Available: {', '.join(s.name for s in SCENARIOS)}"
        ),
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for perturbations.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_eval(_parse_args())
