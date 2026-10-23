"""Evaluate individual SED checkpoints and simple score-mean ensembles.

The evaluator consumes the saved, held-out validation predictions belonging to
each checkpoint.  This guarantees every candidate is scored against exactly
the same examples.  Temporal smoothing, when requested, is applied only to a
checkpoint's framewise predictions *within each validation recording* before
clip max-pooling; it never averages predictions between recordings.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.audio import AudioConfig, AudioPreprocessor
from src.dataset import LabelledWaveformDataset
from src.model import EfficientNetSED, SEDConfig
from src.train import (
    configure_accelerator,
    cuda_amp_enabled,
    data_loader_kwargs,
    macro_auc,
    resolve_device,
    waveform_batch_to_inputs,
)


DEFAULT_SMOOTHING_KERNEL = (0.25, 0.5, 0.25)


@dataclass(frozen=True)
class CheckpointPredictions:
    """Clip scores and provenance for one selected validation checkpoint."""

    path: Path
    epoch: int
    validation_auc: float
    targets: np.ndarray
    scores: np.ndarray

    @property
    def name(self) -> str:
        return self.path.parent.name


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _clip_prediction_path(path: Path, epoch: int) -> Path:
    return path.parent / f"validation_predictions_epoch_{epoch:02d}.npz"


def _framewise_prediction_path(path: Path, epoch: int) -> Path:
    return path.parent / f"validation_framewise_predictions_epoch_{epoch:02d}.npz"


def _read_npz(path: Path, required: Iterable[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as values:
        missing = set(required).difference(values.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        return {name: values[name] for name in required}


def load_checkpoint_predictions(path: Path) -> CheckpointPredictions:
    """Load the clip-level predictions saved at the selected checkpoint epoch."""
    checkpoint = _checkpoint_payload(path)
    epoch = int(checkpoint["epoch"])
    values = _read_npz(_clip_prediction_path(path, epoch), ("y_true", "y_score"))
    targets = np.asarray(values["y_true"], dtype=np.float32)
    scores = np.asarray(values["y_score"], dtype=np.float32)
    if targets.shape != scores.shape or targets.ndim != 2:
        raise ValueError(f"Expected matching two-dimensional validation arrays for {path}")
    return CheckpointPredictions(
        path=path,
        epoch=epoch,
        validation_auc=float(checkpoint["validation_macro_auc"]),
        targets=targets,
        scores=np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0),
    )


def temporal_smooth(framewise_probabilities: np.ndarray, kernel: Iterable[float] = DEFAULT_SMOOTHING_KERNEL) -> np.ndarray:
    """Apply an odd, normalized, edge-padded kernel over a model's time axis.

    Args:
        framewise_probabilities: ``(recordings, classes, frames)`` probabilities.
        kernel: Non-negative odd-length coefficients.  The default is a cheap
            three-frame triangular smoother.
    """
    scores = np.asarray(framewise_probabilities, dtype=np.float32)
    weights = np.asarray(tuple(kernel), dtype=np.float32)
    if scores.ndim != 3:
        raise ValueError("framewise probabilities must have shape (recordings, classes, frames)")
    if weights.ndim != 1 or len(weights) == 0 or len(weights) % 2 == 0 or np.any(weights < 0):
        raise ValueError("smoothing kernel must have non-negative odd length")
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("smoothing kernel must have a finite positive sum")
    weights /= weights.sum()
    radius = len(weights) // 2
    padded = np.pad(scores, ((0, 0), (0, 0), (radius, radius)), mode="edge")
    smoothed = np.zeros_like(scores)
    for offset, weight in enumerate(weights):
        smoothed += weight * padded[:, :, offset : offset + scores.shape[-1]]
    return smoothed


def _checkpoint_framewise_predictions(
    checkpoint_path: Path,
    config_path: Path,
    device_override: str | None,
    batch_size_override: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-run one selected checkpoint once and return deterministic frame scores."""
    root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = dict(config["training"])
    if batch_size_override is not None:
        training["batch_size"] = batch_size_override
    device = resolve_device(device_override or training.get("device", "auto"))
    configure_accelerator(device, bool(training.get("allow_tf32", True)))
    checkpoint = _checkpoint_payload(checkpoint_path)
    preprocessor = AudioPreprocessor(AudioConfig(**checkpoint["audio_config"]))
    paths = {name: root / value for name, value in config["paths"].items()}
    dataset = LabelledWaveformDataset(
        "validation", paths["split"], paths["label_mapping"], paths["train_audio_dir"], preprocessor
    )
    loader = DataLoader(dataset, shuffle=False, **data_loader_kwargs(training, device))
    model = EfficientNetSED(SEDConfig(**checkpoint["model_config"])).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    mixed_precision = cuda_amp_enabled(training, device)
    targets, probabilities = [], []
    with torch.no_grad():
        for chunks, batch_targets in loader:
            inputs = waveform_batch_to_inputs(chunks, preprocessor, device)
            with torch.autocast(device_type=device.type, enabled=mixed_precision):
                framewise_logits = model(inputs)
            targets.append(batch_targets.numpy())
            probabilities.append(torch.sigmoid(framewise_logits).cpu().float().numpy())
    return np.concatenate(targets).astype(np.float32), np.concatenate(probabilities).astype(np.float32)


def load_or_create_framewise_predictions(
    checkpoint: CheckpointPredictions,
    config_path: Path,
    device_override: str | None = None,
    batch_size_override: int | None = None,
    refresh: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or create a cache of deterministic framewise validation scores."""
    cache_path = _framewise_prediction_path(checkpoint.path, checkpoint.epoch)
    if cache_path.is_file() and not refresh:
        values = _read_npz(cache_path, ("y_true", "framewise_probabilities"))
        targets, framewise = values["y_true"], values["framewise_probabilities"]
    else:
        targets, framewise = _checkpoint_framewise_predictions(
            checkpoint.path, config_path, device_override, batch_size_override
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, y_true=targets, framewise_probabilities=framewise)
    if framewise.ndim != 3 or framewise.shape[:2] != checkpoint.scores.shape:
        raise ValueError(f"Framewise cache has incompatible shape: {cache_path}")
    if not np.array_equal(targets, checkpoint.targets):
        raise ValueError(f"Framewise cache targets do not match saved validation targets: {cache_path}")
    return targets.astype(np.float32), np.asarray(framewise, dtype=np.float32)


def _validate_alignment(checkpoints: list[CheckpointPredictions]) -> None:
    if len(checkpoints) < 1:
        raise ValueError("At least one checkpoint is required")
    reference = checkpoints[0].targets
    for checkpoint in checkpoints[1:]:
        if checkpoint.targets.shape != reference.shape or not np.array_equal(checkpoint.targets, reference):
            raise ValueError("All checkpoints must use identical held-out validation targets and ordering")


def _variant_record(
    name: str,
    members: list[CheckpointPredictions],
    scores_by_path: dict[Path, np.ndarray],
) -> dict[str, Any]:
    mean_scores = np.mean([scores_by_path[item.path] for item in members], axis=0, dtype=np.float64).astype(np.float32)
    return {
        "name": name,
        "checkpoint_names": [item.name for item in members],
        "checkpoint_paths": [str(item.path) for item in members],
        "weights": [round(1.0 / len(members), 8)] * len(members),
        "macro_roc_auc": macro_auc(members[0].targets, mean_scores),
    }


def _build_variants(
    checkpoints: list[CheckpointPredictions], scores_by_path: dict[Path, np.ndarray]
) -> list[dict[str, Any]]:
    variants = []
    for checkpoint in checkpoints:
        variants.append(_variant_record(f"single:{checkpoint.name}", [checkpoint], scores_by_path))
    for pair in itertools.combinations(checkpoints, 2):
        variants.append(_variant_record(f"pair:{pair[0].name}+{pair[1].name}", list(pair), scores_by_path))
    if len(checkpoints) >= 3:
        variants.append(_variant_record("three-model-mean", checkpoints, scores_by_path))
    return variants


def _summary(variants: list[dict[str, Any]]) -> dict[str, Any]:
    singles = [item for item in variants if item["name"].startswith("single:")]
    pairs = [item for item in variants if item["name"].startswith("pair:")]
    triples = [item for item in variants if item["name"] == "three-model-mean"]
    best_single = max(singles, key=lambda item: item["macro_roc_auc"])
    best_two = max(pairs, key=lambda item: item["macro_roc_auc"]) if pairs else None
    three = triples[0] if triples else None
    return {
        "best_single": best_single,
        "best_two": best_two,
        "three_checkpoint_mean": three,
        "best_two_gain_over_best_single": (
            None if best_two is None else best_two["macro_roc_auc"] - best_single["macro_roc_auc"]
        ),
        "three_checkpoint_gain_over_best_single": (
            None if three is None else three["macro_roc_auc"] - best_single["macro_roc_auc"]
        ),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Validation ensemble evaluation",
        "",
        "All scores use the same fixed held-out validation split and macro ROC-AUC.",
        "",
        "## Mean-score variants",
        "",
        "| Variant | Checkpoints | Macro ROC-AUC |",
        "| --- | --- | ---: |",
    ]
    for variant in report["without_smoothing"]["variants"]:
        lines.append(
            f"| {variant['name']} | {', '.join(variant['checkpoint_names'])} | {variant['macro_roc_auc']:.8f} |"
        )
    summary = report["without_smoothing"]["summary"]
    lines.extend([
        "",
        f"Best single: **{summary['best_single']['name']}** ({summary['best_single']['macro_roc_auc']:.8f}).",
    ])
    if summary["best_two"] is not None:
        lines.append(
            f"Best pair: **{summary['best_two']['name']}** ({summary['best_two']['macro_roc_auc']:.8f}; "
            f"Δ {summary['best_two_gain_over_best_single']:+.8f} vs best single)."
        )
    if summary["three_checkpoint_mean"] is not None:
        lines.append(
            f"Three-model mean: **{summary['three_checkpoint_mean']['macro_roc_auc']:.8f}** "
            f"(Δ {summary['three_checkpoint_gain_over_best_single']:+.8f} vs best single)."
        )
    smoothing = report.get("with_smoothing")
    lines.extend(["", "## Smoothing", ""])
    if smoothing is None:
        lines.append("Not evaluated. Re-run with framewise validation inference enabled.")
    else:
        raw = summary["three_checkpoint_mean"]
        smoothed = smoothing["summary"]["three_checkpoint_mean"]
        if raw is not None and smoothed is not None:
            delta = smoothed["macro_roc_auc"] - raw["macro_roc_auc"]
            lines.append(
                f"Kernel `{report['smoothing']['kernel']}` changes the three-model mean by **{delta:+.8f}** "
                f"({raw['macro_roc_auc']:.8f} → {smoothed['macro_roc_auc']:.8f})."
            )
            lines.append("Recommendation: " + ("retain smoothing." if delta > 0 else "do not retain smoothing."))
    lines.extend([
        "",
        "## Weight tuning",
        "",
        "Not performed: equal weights are retained because this single validation split is already used for checkpoint and ensemble selection; it is not independent evidence for fitted weights.",
    ])
    return "\n".join(lines) + "\n"


def evaluate_ensemble(
    checkpoint_paths: Iterable[str | Path],
    output_directory: str | Path,
    config_path: str | Path = Path("config/project_config.json"),
    smoothing: bool = True,
    smoothing_kernel: Iterable[float] = DEFAULT_SMOOTHING_KERNEL,
    device_override: str | None = None,
    batch_size_override: int | None = None,
    refresh_framewise: bool = False,
) -> dict[str, Any]:
    """Evaluate equal-weight single/pair/three-model candidates and write a report."""
    paths = [Path(path) for path in checkpoint_paths]
    if len(paths) != 3:
        raise ValueError("Phase 10 requires exactly three selected checkpoints")
    checkpoints = [load_checkpoint_predictions(path) for path in paths]
    _validate_alignment(checkpoints)
    raw_scores = {checkpoint.path: checkpoint.scores for checkpoint in checkpoints}
    report: dict[str, Any] = {
        "metric": "macro_roc_auc",
        "validation_examples": int(checkpoints[0].targets.shape[0]),
        "classes": int(checkpoints[0].targets.shape[1]),
        "checkpoints": [
            {
                "name": checkpoint.name,
                "path": str(checkpoint.path),
                "selected_epoch": checkpoint.epoch,
                "checkpoint_validation_macro_auc": checkpoint.validation_auc,
            }
            for checkpoint in checkpoints
        ],
        "without_smoothing": {"variants": _build_variants(checkpoints, raw_scores)},
        "weight_tuning": {
            "performed": False,
            "reason": "No independent validation set is available to provide reliable evidence for fitted ensemble weights.",
        },
    }
    report["without_smoothing"]["summary"] = _summary(report["without_smoothing"]["variants"])
    if smoothing:
        kernel = tuple(float(item) for item in smoothing_kernel)
        smoothed_scores: dict[Path, np.ndarray] = {}
        for checkpoint in checkpoints:
            _, framewise = load_or_create_framewise_predictions(
                checkpoint, Path(config_path), device_override, batch_size_override, refresh_framewise
            )
            smoothed_scores[checkpoint.path] = temporal_smooth(framewise, kernel).max(axis=-1)
        report["smoothing"] = {
            "applied_before": "clip-level maximum pooling",
            "kernel": list(kernel),
            "axis": "framewise temporal axis within each validation recording",
        }
        report["with_smoothing"] = {"variants": _build_variants(checkpoints, smoothed_scores)}
        report["with_smoothing"]["summary"] = _summary(report["with_smoothing"]["variants"])
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "ensemble_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "ensemble_evaluation.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def _default_checkpoint_paths(config_path: Path) -> list[Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.parent
    values = [root / path for path in config["ensemble"].get("checkpoint_paths", [])]
    if len(values) != 3:
        raise ValueError("config.ensemble.checkpoint_paths must contain the three selected checkpoints")
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate three SED checkpoints and their score-mean ensembles.")
    parser.add_argument("--config", type=Path, default=Path("config/project_config.json"))
    parser.add_argument("--checkpoint", type=Path, action="append", help="Provide exactly three paths; overrides config.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ensemble_evaluation"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0", "mps"))
    parser.add_argument("--batch-size", type=int, help="Validation batch size used only for framewise cache generation.")
    parser.add_argument("--no-smoothing", action="store_true", help="Skip framewise inference and smoothing comparison.")
    parser.add_argument("--refresh-framewise", action="store_true", help="Recreate cached framewise validation predictions.")
    arguments = parser.parse_args()
    paths = arguments.checkpoint or _default_checkpoint_paths(arguments.config)
    report = evaluate_ensemble(
        paths, arguments.output_dir, arguments.config, not arguments.no_smoothing,
        DEFAULT_SMOOTHING_KERNEL, arguments.device, arguments.batch_size, arguments.refresh_framewise,
    )
    print(json.dumps({
        "report": str(arguments.output_dir / "ensemble_evaluation.json"),
        "best_single": report["without_smoothing"]["summary"]["best_single"]["name"],
    }, indent=2))
