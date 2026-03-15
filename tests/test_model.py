"""
tests/test_model.py
====================
Unit tests for ml/model.py and a single training step.

Tests cover:
- Model instantiation with various num_classes values
- Forward pass output shape
- Parameter count is nonzero and reasonable
- Single gradient update (smoke test for train_one_epoch)
- Provenance dict is written after training
- Loss decreases across a trivial overfit scenario (sanity check)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.model import SceneClassifier, build_model, count_parameters
from ml.train import train_one_epoch, evaluate, set_seed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_loader() -> DataLoader:  # type: ignore[type-arg]
    """Synthetic DataLoader: 8 random (3,256,256) images, 3 classes."""
    set_seed(0)
    images = torch.randn(8, 3, 256, 256)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=4, shuffle=False)


@pytest.fixture()
def model_3class() -> SceneClassifier:
    return build_model(num_classes=3, pretrained=False)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


class TestModelConstruction:
    def test_build_model_returns_classifier(self) -> None:
        m = build_model(num_classes=3)
        assert isinstance(m, SceneClassifier)

    def test_custom_num_classes(self) -> None:
        m = build_model(num_classes=5)
        # Last linear layer output features should equal num_classes
        last_linear = [l for l in m.classifier.modules() if isinstance(l, nn.Linear)][-1]
        assert last_linear.out_features == 5

    def test_three_classes(self, model_3class: SceneClassifier) -> None:
        last_linear = [l for l in model_3class.classifier.modules() if isinstance(l, nn.Linear)][-1]
        assert last_linear.out_features == 3

    def test_parameter_count_nonzero(self, model_3class: SceneClassifier) -> None:
        assert count_parameters(model_3class) > 0

    def test_parameter_count_reasonable(self, model_3class: SceneClassifier) -> None:
        # MobileNetV2 backbone: expect ~2M–4M params
        n = count_parameters(model_3class)
        assert 500_000 < n < 10_000_000, f"Unexpected param count: {n}"


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    def test_output_shape_batch4(self, model_3class: SceneClassifier) -> None:
        x = torch.randn(4, 3, 256, 256)
        with torch.no_grad():
            out = model_3class(x)
        assert out.shape == (4, 3)

    def test_output_shape_batch1(self, model_3class: SceneClassifier) -> None:
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model_3class(x)
        assert out.shape == (1, 3)

    def test_output_is_float32(self, model_3class: SceneClassifier) -> None:
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = model_3class(x)
        assert out.dtype == torch.float32

    def test_no_nan_in_output(self, model_3class: SceneClassifier) -> None:
        x = torch.randn(4, 3, 256, 256)
        with torch.no_grad():
            out = model_3class(x)
        assert not torch.isnan(out).any()

    def test_different_inputs_different_outputs(self, model_3class: SceneClassifier) -> None:
        x1 = torch.randn(2, 3, 256, 256)
        x2 = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            o1 = model_3class(x1)
            o2 = model_3class(x2)
        assert not torch.equal(o1, o2)


# ---------------------------------------------------------------------------
# Training step smoke tests
# ---------------------------------------------------------------------------


class TestTrainingStep:
    def test_train_one_epoch_returns_loss_and_acc(
        self, model_3class: SceneClassifier, tiny_loader: DataLoader
    ) -> None:
        optimizer = torch.optim.Adam(model_3class.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        loss, acc = train_one_epoch(model_3class, tiny_loader, optimizer, criterion, device)
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_loss_is_positive(
        self, model_3class: SceneClassifier, tiny_loader: DataLoader
    ) -> None:
        optimizer = torch.optim.Adam(model_3class.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        loss, _ = train_one_epoch(model_3class, tiny_loader, optimizer, criterion, device)
        assert loss > 0.0

    def test_accuracy_in_range(
        self, model_3class: SceneClassifier, tiny_loader: DataLoader
    ) -> None:
        optimizer = torch.optim.Adam(model_3class.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        _, acc = train_one_epoch(model_3class, tiny_loader, optimizer, criterion, device)
        assert 0.0 <= acc <= 1.0

    def test_evaluate_returns_loss_and_acc(
        self, model_3class: SceneClassifier, tiny_loader: DataLoader
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        loss, acc = evaluate(model_3class, tiny_loader, criterion, device)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_gradients_flow(self, model_3class: SceneClassifier) -> None:
        """Verify that gradients are computed for model parameters."""
        model_3class.train()
        x = torch.randn(2, 3, 256, 256)
        labels = torch.tensor([0, 1])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model_3class.parameters(), lr=0.01)

        optimizer.zero_grad()
        loss = criterion(model_3class(x), labels)
        loss.backward()

        grad_norms = [
            p.grad.norm().item()
            for p in model_3class.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0
        assert all(g >= 0 for g in grad_norms)


# ---------------------------------------------------------------------------
# Overfit sanity check
# ---------------------------------------------------------------------------


class TestOverfitSanity:
    def test_loss_decreases_on_trivial_data(self) -> None:
        """Model should be able to overfit 4 identical samples in a few steps."""
        set_seed(0)
        model = build_model(num_classes=3)
        model.train()

        # Single repeated batch
        x = torch.randn(4, 3, 256, 256)
        y = torch.tensor([0, 0, 0, 0])
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=4)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        device = torch.device("cpu")

        losses = []
        for _ in range(5):
            loss, _ = train_one_epoch(model, loader, optimizer, criterion, device)
            losses.append(loss)

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Full train() integration smoke test
# ---------------------------------------------------------------------------


class TestTrainIntegration:
    def test_train_produces_artifacts(self, tmp_path: Path) -> None:
        """Run train() for 2 epochs and check output files are created."""
        import argparse
        from data.synthetic_generator import SyntheticSceneGenerator
        from ml.train import train

        # Generate tiny synthetic dataset
        data_dir = tmp_path / "data"
        gen = SyntheticSceneGenerator(seed=42, output_dir=data_dir / "samples", num_images=6)
        gen.generate_all()

        args = argparse.Namespace(
            data_dir=str(data_dir),
            output_dir=str(tmp_path / "artifacts"),
            epochs=2,
            batch_size=2,
            lr=1e-3,
            seed=42,
            val_split=0.2,
            num_workers=0,
        )

        result = train(args)

        # Artifacts exist
        artifacts = tmp_path / "artifacts"
        assert (artifacts / "best_model.pt").exists()
        assert (artifacts / "last_model.pt").exists()
        assert (artifacts / "train_metrics.json").exists()
        assert (artifacts / "model_provenance.json").exists()

        # Provenance has required fields
        assert "annotations_sha256" in result
        assert "safety_note" in result
        assert result["epochs"] == 2

    def test_provenance_safety_note_present(self, tmp_path: Path) -> None:
        import argparse
        from data.synthetic_generator import SyntheticSceneGenerator
        from ml.train import train

        data_dir = tmp_path / "data"
        gen = SyntheticSceneGenerator(seed=1, output_dir=data_dir / "samples", num_images=6)
        gen.generate_all()

        args = argparse.Namespace(
            data_dir=str(data_dir),
            output_dir=str(tmp_path / "artifacts"),
            epochs=1,
            batch_size=2,
            lr=1e-3,
            seed=1,
            val_split=0.2,
            num_workers=0,
        )

        result = train(args)
        assert "human analyst" in result["safety_note"].lower()
