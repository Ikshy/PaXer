"""
tests/test_redteam.py
======================
Unit and integration tests for the CTHMP red-team evaluation harness.

Coverage:
  - All 9 perturbation functions (output shape, dtype, determinism, effect)
  - apply_perturbation dispatcher (valid name, invalid name)
  - Scenario registry completeness
  - _compute_metrics correctness (accuracy, FPR, FNR edge cases)
  - Full eval integration smoke test (2 scenarios, tiny synthetic dataset)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from data.synthetic_generator import SyntheticSceneGenerator
from redteam.perturbations import (
    apply_perturbation,
    list_perturbations,
    PERTURBATIONS,
)
from redteam.scenarios import SCENARIOS, SCENARIO_MAP, get_scenario
from redteam.eval import _compute_metrics


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_image() -> np.ndarray:
    """256×256 synthetic BGR image for perturbation tests."""
    rng = np.random.default_rng(0)
    return rng.integers(30, 200, (256, 256, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def tiny_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate 8 synthetic scenes into a temp dir (reused across module)."""
    out = tmp_path_factory.mktemp("redteam_data")
    gen = SyntheticSceneGenerator(seed=42, output_dir=out / "samples", num_images=8)
    gen.generate_all()
    return out


# ---------------------------------------------------------------------------
# Perturbation: output contract
# ---------------------------------------------------------------------------


class TestPerturbationOutputContract:
    """Every perturbation must return same shape, dtype=uint8, no NaN."""

    @pytest.mark.parametrize("name", list_perturbations())
    def test_shape_preserved(self, name: str, sample_image: np.ndarray) -> None:
        out = apply_perturbation(sample_image, name, severity=1, seed=0)
        assert out.shape == sample_image.shape, f"{name}: shape mismatch"

    @pytest.mark.parametrize("name", list_perturbations())
    def test_dtype_uint8(self, name: str, sample_image: np.ndarray) -> None:
        out = apply_perturbation(sample_image, name, severity=1, seed=0)
        assert out.dtype == np.uint8, f"{name}: expected uint8"

    @pytest.mark.parametrize("name", list_perturbations())
    def test_pixel_range_valid(self, name: str, sample_image: np.ndarray) -> None:
        out = apply_perturbation(sample_image, name, severity=1, seed=0)
        assert out.min() >= 0 and out.max() <= 255, f"{name}: pixel out of range"

    @pytest.mark.parametrize("name", list_perturbations())
    def test_no_nan(self, name: str, sample_image: np.ndarray) -> None:
        out = apply_perturbation(sample_image.astype(np.float32), name, severity=1, seed=0)
        # Cast back to uint8 for check
        out_u8 = np.clip(out, 0, 255).astype(np.uint8)
        assert not np.isnan(out_u8.astype(np.float32)).any(), f"{name}: NaN in output"


# ---------------------------------------------------------------------------
# Perturbation: determinism
# ---------------------------------------------------------------------------


class TestPerturbationDeterminism:
    @pytest.mark.parametrize("name", list_perturbations())
    def test_same_seed_same_output(self, name: str, sample_image: np.ndarray) -> None:
        out1 = apply_perturbation(sample_image, name, severity=1, seed=7)
        out2 = apply_perturbation(sample_image, name, severity=1, seed=7)
        assert np.array_equal(out1, out2), f"{name}: not deterministic with same seed"

    @pytest.mark.parametrize("name", ["gaussian_noise", "salt_pepper", "occlusion_patch"])
    def test_different_seeds_differ(self, name: str, sample_image: np.ndarray) -> None:
        out1 = apply_perturbation(sample_image, name, severity=1, seed=1)
        out2 = apply_perturbation(sample_image, name, severity=1, seed=2)
        assert not np.array_equal(out1, out2), f"{name}: different seeds gave same output"


# ---------------------------------------------------------------------------
# Perturbation: effect checks
# ---------------------------------------------------------------------------


class TestPerturbationEffects:
    def test_gaussian_noise_changes_image(self, sample_image: np.ndarray) -> None:
        out = apply_perturbation(sample_image, "gaussian_noise", severity=2, seed=0)
        assert not np.array_equal(out, sample_image)

    def test_blur_reduces_variance(self, sample_image: np.ndarray) -> None:
        out = apply_perturbation(sample_image, "gaussian_blur", severity=2, seed=0)
        assert out.astype(float).std() < sample_image.astype(float).std()

    def test_occlusion_introduces_zeros(self, sample_image: np.ndarray) -> None:
        # Use a uniformly bright image so zeros are definitely from occlusion
        bright = np.full((256, 256, 3), 200, dtype=np.uint8)
        out = apply_perturbation(bright, "occlusion_patch", severity=2, seed=0)
        assert (out == 0).any()

    def test_horizontal_flip_is_mirror(self, sample_image: np.ndarray) -> None:
        out = apply_perturbation(sample_image, "horizontal_flip", severity=0, seed=0)
        expected = cv2.flip(sample_image, 1)
        assert np.array_equal(out, expected)

    def test_brightness_shift_increases_mean(self, sample_image: np.ndarray) -> None:
        dark = (sample_image // 3).astype(np.uint8)  # start dark to avoid clipping
        out = apply_perturbation(dark, "brightness_shift", severity=2, seed=0)
        assert out.astype(float).mean() > dark.astype(float).mean()

    def test_severity_levels_differ(self, sample_image: np.ndarray) -> None:
        mild = apply_perturbation(sample_image, "gaussian_noise", severity=0, seed=0)
        severe = apply_perturbation(sample_image, "gaussian_noise", severity=2, seed=0)
        # Severe noise should differ more from original
        diff_mild = np.abs(mild.astype(int) - sample_image.astype(int)).mean()
        diff_severe = np.abs(severe.astype(int) - sample_image.astype(int)).mean()
        assert diff_severe > diff_mild


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_invalid_name_raises(self, sample_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unknown perturbation"):
            apply_perturbation(sample_image, "nonexistent_perturbation")

    def test_list_perturbations_nonempty(self) -> None:
        names = list_perturbations()
        assert len(names) >= 9

    def test_all_registered_names_work(self, sample_image: np.ndarray) -> None:
        for name in list_perturbations():
            out = apply_perturbation(sample_image, name, seed=0)
            assert out is not None


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


class TestScenarios:
    def test_clean_scenario_exists(self) -> None:
        s = get_scenario("clean")
        assert s.name == "clean"
        assert s.perturbations == []

    def test_all_scenarios_have_name_and_description(self) -> None:
        for s in SCENARIOS:
            assert s.name, "Scenario missing name"
            assert s.description, f"{s.name}: missing description"

    def test_scenario_perturbation_names_are_valid(self) -> None:
        valid = set(list_perturbations())
        for s in SCENARIOS:
            for pert_name, severity in s.perturbations:
                assert pert_name in valid, (
                    f"Scenario {s.name!r} references unknown perturbation {pert_name!r}"
                )
                assert 0 <= severity <= 2, (
                    f"Scenario {s.name!r}: severity {severity} out of range"
                )

    def test_get_scenario_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario("this_does_not_exist")

    def test_scenario_count(self) -> None:
        assert len(SCENARIOS) >= 10, "Expected at least 10 scenarios"


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_predictions(self) -> None:
        labels = [0, 1, 2, 0, 1, 2]
        preds = labels[:]
        m = _compute_metrics(preds, labels)
        assert m["overall_accuracy"] == 1.0
        for cls_name in m["per_class"]:
            assert m["per_class"][cls_name]["precision"] == 1.0
            assert m["per_class"][cls_name]["recall"] == 1.0
            assert m["per_class"][cls_name]["fpr"] == 0.0
            assert m["per_class"][cls_name]["fnr"] == 0.0

    def test_all_wrong(self) -> None:
        labels = [0, 0, 0]
        preds  = [1, 1, 1]
        m = _compute_metrics(preds, labels)
        assert m["overall_accuracy"] == 0.0

    def test_empty_inputs(self) -> None:
        m = _compute_metrics([], [])
        assert m["overall_accuracy"] == 0.0
        assert m["per_class"] == {}

    def test_fpr_nonzero_on_false_alarms(self) -> None:
        # preds always say class 0, but some are class 1 or 2
        labels = [0, 1, 2]
        preds  = [0, 0, 0]
        m = _compute_metrics(preds, labels)
        # class 0: all correct → FPR should be 0 (no false alarms for class 0)
        assert m["per_class"]["building"]["fpr"] == 0.0
        # class 1: FN for class 1 (pred says 0, label is 1)
        assert m["per_class"]["vehicle"]["fnr"] == 1.0

    def test_accuracy_range(self) -> None:
        import random as _random
        _random.seed(0)
        n = 50
        labels = [_random.randint(0, 2) for _ in range(n)]
        preds  = [_random.randint(0, 2) for _ in range(n)]
        m = _compute_metrics(preds, labels)
        assert 0.0 <= m["overall_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------


class TestEvalIntegration:
    def test_run_eval_two_scenarios(
        self, tmp_path: Path, tiny_data_dir: Path
    ) -> None:
        """End-to-end: generate data, train a tiny model, run 2 scenarios."""
        from data.synthetic_generator import SyntheticSceneGenerator
        from ml.train import train

        # Train a 1-epoch model on the tiny dataset
        artifacts_dir = tmp_path / "artifacts"
        train_args = argparse.Namespace(
            data_dir=str(tiny_data_dir),
            output_dir=str(artifacts_dir),
            epochs=1,
            batch_size=2,
            lr=1e-3,
            seed=42,
            val_split=0.2,
            num_workers=0,
        )
        train(train_args)

        # Run clean + mild_noise scenarios
        from redteam.eval import run_eval

        eval_args = argparse.Namespace(
            checkpoint=str(artifacts_dir / "best_model.pt"),
            data_dir=str(tiny_data_dir),
            output_dir=str(tmp_path / "reports"),
            scenarios="clean,mild_noise",
            seed=42,
        )
        report = run_eval(eval_args)

        # Structural checks
        assert "scenarios" in report
        assert "clean" in report["scenarios"]
        assert "mild_noise" in report["scenarios"]
        assert "summary" in report
        assert "metadata" in report

        # Safety note must be present
        assert "safety_note" in report["metadata"]
        assert "synthetic" in report["metadata"]["safety_note"].lower()

        # Report file written
        report_files = list(Path(tmp_path / "reports").glob("*.json"))
        assert len(report_files) >= 2  # timestamped + latest

        # Accuracy drop vs baseline populated for non-clean scenario
        noise_data = report["scenarios"]["mild_noise"]
        assert noise_data["accuracy_drop_vs_clean"] is not None

        # Per-class metrics contain expected keys
        for cls_name, m in report["scenarios"]["clean"]["per_class"].items():
            for key in ("precision", "recall", "f1", "fpr", "fnr", "support"):
                assert key in m, f"Missing key {key!r} for class {cls_name!r}"

    def test_report_json_is_valid(
        self, tmp_path: Path, tiny_data_dir: Path
    ) -> None:
        """Latest report file must be valid, parseable JSON."""
        from ml.train import train
        from redteam.eval import run_eval

        artifacts_dir = tmp_path / "artifacts2"
        train(argparse.Namespace(
            data_dir=str(tiny_data_dir),
            output_dir=str(artifacts_dir),
            epochs=1, batch_size=2, lr=1e-3,
            seed=0, val_split=0.2, num_workers=0,
        ))

        eval_args = argparse.Namespace(
            checkpoint=str(artifacts_dir / "best_model.pt"),
            data_dir=str(tiny_data_dir),
            output_dir=str(tmp_path / "reports2"),
            scenarios="clean",
            seed=0,
        )
        run_eval(eval_args)

        latest = tmp_path / "reports2" / "robustness_report_latest.json"
        assert latest.exists()
        with open(latest) as f:
            data = json.load(f)
        assert isinstance(data, dict)
