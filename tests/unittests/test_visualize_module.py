"""Execution tests for visualize module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import mlops_g116.visualize as visualize_module
from mlops_g116.model import TumorDetectionModelSimple


class DummyProfiler:
    """No-op profiler context manager."""

    def __enter__(self) -> "DummyProfiler":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def step(self) -> None:
        """Advance profiler step."""


class DummyRun:
    """Minimal W&B run stub."""

    id = "dummy-run"

    def log(self, *_args: object, **_kwargs: object) -> None:
        """No-op log."""

    def log_artifact(self, *_args: object, **_kwargs: object) -> None:
        """No-op log_artifact."""

    def link_artifact(self, *_args: object, **_kwargs: object) -> None:
        """No-op link_artifact."""


class DummyArtifact:
    """Minimal W&B artifact stub."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def add_file(self, *_args: object, **_kwargs: object) -> None:
        """No-op add_file."""


class DummyTSNE:
    """Fast TSNE stand-in."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Return a simple 2D projection."""
        if data.shape[1] >= 2:
            return data[:, :2]
        return np.concatenate([data, np.zeros((data.shape[0], 2 - data.shape[1]))], axis=1)


def _make_dataset(num_samples: int) -> torch.utils.data.TensorDataset:
    """Create a small tensor dataset for visualization."""
    images = torch.randn(num_samples, 1, 64, 64)
    labels = torch.tensor([i % 4 for i in range(num_samples)])
    return torch.utils.data.TensorDataset(images, labels)


class DummyProfile:
    """No-op cProfile.Profile stand-in."""

    def enable(self) -> None:
        """No-op enable."""

    def disable(self) -> None:
        """No-op disable."""

    def dump_stats(self, *_args: object, **_kwargs: object) -> None:
        """No-op dump_stats."""


def _patch_visualize_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Patch visualize runtime dependencies to use temp outputs."""
    output_dir = tmp_path / "outputs"
    runtime = SimpleNamespace(output_dir=str(output_dir))
    monkeypatch.setattr(visualize_module.HydraConfig, "get", lambda: SimpleNamespace(runtime=runtime))
    monkeypatch.setattr(visualize_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(visualize_module.cProfile, "Profile", DummyProfile)
    monkeypatch.setattr(
        visualize_module.torch.profiler,
        "profile",
        lambda *args, **kwargs: DummyProfiler(),
    )
    monkeypatch.setattr(
        visualize_module.torch.profiler,
        "tensorboard_trace_handler",
        lambda *args, **kwargs: lambda *_a, **_k: None,
    )
    monkeypatch.setattr(visualize_module.wandb, "init", lambda **_kwargs: DummyRun())
    monkeypatch.setattr(visualize_module.wandb, "log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(visualize_module.wandb, "finish", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(visualize_module.wandb, "Image", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(visualize_module.wandb, "Artifact", DummyArtifact)
    monkeypatch.setenv("RUN_SNAKEVIZ", "0")
    monkeypatch.setenv("RUN_TENSORBOARD", "0")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    return output_dir


def test_strip_classifier_replaces_fc1() -> None:
    """Ensure _strip_classifier replaces simple model classifier."""
    model = TumorDetectionModelSimple()
    visualize_module._strip_classifier(model)
    assert isinstance(model.fc1, torch.nn.Identity), "Expected fc1 to be replaced by Identity"


def test_visualize_raises_on_missing_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure visualize raises when required config keys are missing."""
    _patch_visualize_runtime(monkeypatch, tmp_path)
    config = OmegaConf.create(
        {
            "model": {"_target_": "mlops_g116.model.TumorDetectionModelSimple"},
        }
    )
    with pytest.raises(KeyError, match="Missing visualization configuration"):
        visualize_module.visualize.__wrapped__(config)


def test_visualize_writes_embedding_figure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure visualize writes the embedding figure."""
    _patch_visualize_runtime(monkeypatch, tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model = TumorDetectionModelSimple()
    checkpoint_path = models_dir / "model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    train_set = _make_dataset(4)
    test_set = _make_dataset(40)
    monkeypatch.setattr(visualize_module, "load_data", lambda: (train_set, test_set))
    monkeypatch.setattr(visualize_module, "TSNE", DummyTSNE)
    config = OmegaConf.create(
        {
            "model": {"_target_": "mlops_g116.model.TumorDetectionModelSimple"},
            "batch_size": 16,
            "checkpoint_path": str(checkpoint_path),
            "figure_name": "embeddings.png",
        }
    )
    visualize_module.visualize.__wrapped__(config)

    figure_path = tmp_path / "outputs" / "reports" / "figures" / "embeddings.png"
    assert figure_path.exists(), f"Expected visualization at {figure_path}"

    repo_figure_path = tmp_path / "reports" / "figures" / "embeddings.png"
    assert repo_figure_path.exists(), f"Expected repo visualization at {repo_figure_path}"


def test_visualize_raises_on_missing_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure visualize raises when checkpoint is missing."""
    _patch_visualize_runtime(monkeypatch, tmp_path)
    config = OmegaConf.create(
        {
            "model": {"_target_": "mlops_g116.model.TumorDetectionModelSimple"},
            "batch_size": 2,
            "checkpoint_path": str(tmp_path / "missing.pth"),
            "figure_name": "embeddings.png",
        }
    )
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        visualize_module.visualize.__wrapped__(config)
