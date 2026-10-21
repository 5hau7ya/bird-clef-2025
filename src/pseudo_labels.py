"""Versioned five-second pseudo-label generation and inspection.

The stored ``five_second_scores.npz`` is time-resolved: every row contains a
source recording ID, segment boundaries, its 206 soft scores, and its
recording-level weighted-sampling score. `PseudoLabelStore.soft_target_for_crop`
max-pools four aligned five-second vectors into a 20-second soft target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

# Reports are generated in batch jobs (including headless Kaggle notebooks).
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.overlap_inference import OverlapAwareInferencer, load_checkpoint_inferencer


def checkpoint_identifier(path: Path) -> str:
    """Stable, content-addressed identifier used to isolate teacher versions."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"{path.stem}-{digest.hexdigest()[:12]}"


def transform_pseudo_scores(scores: np.ndarray, exponent: float) -> np.ndarray:
    """Apply the configured probability power once, before sampling/training."""
    if exponent <= 0 or not np.isfinite(exponent):
        raise ValueError("pseudo-label exponent must be finite and greater than zero")
    return np.power(np.clip(np.asarray(scores, dtype=np.float32), 0.0, 1.0), exponent).astype(np.float32)


class PseudoLabelStore:
    """Read stored segment scores and construct aligned 20-second soft targets."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        # `np.load` returns a lazy NpzFile backed by one ZipFile/decompressor.
        # A Dataset is then forked into multiple DataLoader workers on Linux;
        # concurrent reads from that inherited handle cause zlib "invalid block"
        # failures. Materialize every array and close the archive before workers
        # are created. Arrays are read-only afterwards and shared copy-on-write
        # by the normal Kaggle/Linux fork start method.
        with np.load(self.directory / "five_second_scores.npz", allow_pickle=False) as archive:
            self.values = {name: archive[name] for name in archive.files}
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))

    def soft_target_for_crop(self, recording_id: str, start_seconds: float) -> np.ndarray:
        """Classwise-max four contiguous five-second vectors for a 20-second crop."""
        crop_duration = self.metadata["pseudo_label_configuration"]["crop_duration_seconds"]
        stride = self.metadata["pseudo_label_configuration"]["crop_stride_seconds"]
        expected_starts = np.arange(start_seconds, start_seconds + crop_duration, stride)
        if len(expected_starts) != 4:
            raise ValueError("The initial pseudo-label recipe requires exactly four five-second scores")
        ids = self.values["recording_ids"].astype(str)
        starts = self.values["segment_start_seconds"]
        selected = []
        for expected in expected_starts:
            matching = np.flatnonzero((ids == recording_id) & np.isclose(starts, expected))
            if len(matching) != 1:
                raise ValueError(f"Missing or ambiguous pseudo-label segment: {recording_id} at {expected}s")
            selected.append(self.values["scores"][matching[0]])
        return np.max(np.stack(selected), axis=0)


def _write_report(directory: Path, mapping_path: Path) -> None:
    values = np.load(directory / "five_second_scores.npz")
    scores = values["scores"]
    recording_ids = values["recording_ids"].astype(str)
    starts = values["segment_start_seconds"]
    weights = values["sampling_weights"]
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))["ind2label"]

    # Sampling values make the score distribution legible without retaining an
    # additional copy of every 206-class score in the report.
    flattened = scores.ravel()
    sample = flattened[:: max(1, len(flattened) // 250_000)]
    top_count = min(20, len(flattened))
    top_indices = np.argpartition(flattened, -top_count)[-top_count:]
    top_indices = top_indices[np.argsort(flattened[top_indices])[::-1]]
    segment_indices, class_indices = np.unravel_index(top_indices, scores.shape)
    high_confidence = [
        {
            "recording_id": recording_ids[segment_index],
            "segment_start_seconds": float(starts[segment_index]),
            "label": mapping[str(class_index)],
            "score": float(scores[segment_index, class_index]),
        }
        for segment_index, class_index in zip(segment_indices, class_indices)
    ]
    selected_recording = recording_ids[int(np.argmax(weights))]
    selected_rows = np.flatnonzero(recording_ids == selected_recording)
    selected_scores = scores[selected_rows]
    top_classes = np.argsort(selected_scores.max(axis=0))[-5:][::-1]

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].hist(sample, bins=60, color="#4477aa")
    axes[0].set(title="Pseudo-label score distribution", xlabel="Soft score", ylabel="Count")
    axes[1].hist(weights[:: max(1, len(weights) // 100_000)], bins=50, color="#66aa55")
    axes[1].set(title="Segment sampling-weight distribution", xlabel="Recording weight", ylabel="Count")
    times = starts[selected_rows]
    for class_index in top_classes:
        axes[2].plot(times, selected_scores[:, class_index], marker="o", label=mapping[str(class_index)])
    axes[2].set(
        title=f"Segment variation: {selected_recording}",
        xlabel="Five-second segment start (s)",
        ylabel="Soft score",
        ylim=(0, 1),
    )
    axes[2].legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(directory / "pseudo_label_report.png", dpi=160)
    plt.close(figure)
    (directory / "pseudo_label_report.json").write_text(
        json.dumps(
            {
                "score_count": int(scores.size),
                "segment_count": int(len(scores)),
                "recording_count": int(len(np.unique(recording_ids))),
                "high_confidence_predictions": high_confidence,
                "segment_variation_recording": selected_recording,
                "segment_variation_labels": [mapping[str(index)] for index in top_classes],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def generate_pseudo_labels(
    checkpoint_path: Path,
    soundscape_directory: Path,
    output_root: Path,
    mapping_path: Path,
    batch_size: int = 16,
    exponent: float = 1.0,
) -> Path:
    """Run one teacher across soundscapes and persist a new immutable version directory."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    soundscapes = sorted(soundscape_directory.glob("*.ogg"))
    if not soundscapes:
        raise FileNotFoundError(f"No .ogg soundscapes in {soundscape_directory}")
    teacher_identifier = checkpoint_identifier(checkpoint_path)
    version = f"{teacher_identifier}-power-{exponent:.6g}"
    output_directory = output_root / version
    if output_directory.exists():
        raise FileExistsError(f"Pseudo-label version already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    inferencer = load_checkpoint_inferencer(checkpoint_path, batch_size=batch_size)
    all_ids, all_starts, all_ends, all_scores, all_weights = [], [], [], [], []
    manifest_path = output_directory / "recordings.jsonl"
    try:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for position, soundscape in enumerate(soundscapes, start=1):
                waveform = inferencer.preprocessor.load_audio(soundscape)
                scores, _ = inferencer.predict_waveform(waveform)
                scores = transform_pseudo_scores(scores, exponent)
                starts = np.arange(len(scores), dtype=np.float32) * inferencer.settings.stride_seconds
                ends = np.minimum(starts + inferencer.settings.stride_seconds, len(waveform) / inferencer.preprocessor.config.sample_rate)
                weight = float(scores.max(axis=0).sum())
                recording_id = soundscape.stem
                all_ids.extend([recording_id] * len(scores))
                all_starts.append(starts)
                all_ends.append(ends)
                all_scores.append(scores.astype(np.float32))
                all_weights.append(np.full(len(scores), weight, dtype=np.float32))
                manifest.write(json.dumps({
                    "recording_id": recording_id,
                    "source_filename": soundscape.name,
                    "duration_seconds": len(waveform) / inferencer.preprocessor.config.sample_rate,
                    "segment_count": len(scores),
                    "sampling_weight": weight,
                }) + "\n")
                if position % 25 == 0 or position == len(soundscapes):
                    print(f"Processed {position}/{len(soundscapes)} soundscapes", flush=True)
        np.savez_compressed(
            output_directory / "five_second_scores.npz",
            recording_ids=np.asarray(all_ids),
            segment_start_seconds=np.concatenate(all_starts),
            segment_end_seconds=np.concatenate(all_ends),
            scores=np.concatenate(all_scores),
            sampling_weights=np.concatenate(all_weights),
        )
        metadata: dict[str, Any] = {
            "version": version,
            "teacher_checkpoint": {"path": str(checkpoint_path), "identifier": teacher_identifier},
            "score_format": "five-second framewise-stitch-then-max-pooled class probabilities",
            "num_classes": int(all_scores[0].shape[1]),
            "pseudo_label_configuration": {
                "crop_duration_seconds": 20,
                "crop_stride_seconds": 5,
                "crop_target_operation": "classwise maximum of four contiguous five-second score vectors",
                "probability_power_exponent": exponent,
                "power_applied_at": "pseudo-label generation only",
                "sampling_weight": "sum_c max_t score[t, c] per recording",
            },
            "audio_preprocessing": asdict(inferencer.preprocessor.config),
            "overlap_inference": asdict(inferencer.settings),
            "recording_count": len(soundscapes),
            "segment_count": len(all_ids),
        }
        (output_directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _write_report(output_directory, mapping_path)
    except Exception:
        # Keep no partially valid version that could later be consumed by training.
        for path in output_directory.iterdir():
            path.unlink()
        output_directory.rmdir()
        raise
    return output_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a versioned five-second pseudo-label dataset.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--soundscapes", type=Path, default=Path("data/train_soundscapes"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/pseudo_labels"))
    parser.add_argument("--mapping", type=Path, default=Path("artifacts/label_mapping.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--exponent", type=float, default=1.0, help="Power applied once to teacher probabilities.")
    arguments = parser.parse_args()
    directory = generate_pseudo_labels(
        arguments.checkpoint, arguments.soundscapes, arguments.output_root, arguments.mapping,
        arguments.batch_size, arguments.exponent
    )
    print(f"Wrote pseudo-label version: {directory}")
