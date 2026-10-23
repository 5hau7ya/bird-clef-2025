"""Shared-window ensemble inference and competition submission generation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from src.audio import AudioConfig, AudioPreprocessor
from src.model import EfficientNetSED, SEDConfig
from src.overlap_inference import OverlapAwareInferencer, OverlapSettings
from src.train import configure_accelerator, resolve_device


@dataclass(frozen=True)
class SubmissionSettings:
    """Fixed final-inference choices selected by validation evaluation."""

    stride_seconds: int = 5
    batch_size: int = 16
    smoothing_enabled: bool = False


def parse_row_id(row_id: str, stride_seconds: int = 5) -> tuple[str, int]:
    """Return ``(recording_id, zero_based_interval_index)`` from a row ID."""
    try:
        recording_id, end_seconds_text = row_id.rsplit("_", maxsplit=1)
        end_seconds = int(end_seconds_text)
    except ValueError as error:
        raise ValueError(f"Invalid competition row ID: {row_id!r}") from error
    if not recording_id or end_seconds <= 0 or end_seconds % stride_seconds:
        raise ValueError(f"Row ID does not identify a positive {stride_seconds}-second interval: {row_id!r}")
    return recording_id, end_seconds // stride_seconds - 1


def read_submission_template(path: str | Path) -> tuple[list[str], list[str]]:
    """Read and validate the sample-submission header and row-ID order."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Submission template is empty: {path}") from error
        row_ids = [row[0] for row in reader]
    if not header or header[0] != "row_id" or len(header) != 207:
        raise ValueError("Submission template must contain row_id plus 206 class columns")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("Submission template contains duplicate row IDs")
    return header, row_ids


def label_order_from_mapping(mapping_path: str | Path) -> list[str]:
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    labels = [mapping["ind2label"][str(index)] for index in range(len(mapping["ind2label"]))]
    if len(labels) != 206 or len(labels) != len(set(labels)):
        raise ValueError("The label mapping must contain exactly 206 unique labels")
    return labels


def resolve_soundscapes(
    test_soundscapes_directory: str | Path,
    train_soundscapes_directory: str | Path | None = None,
    fallback_limit: int = 3,
) -> tuple[list[Path], bool]:
    """Return test soundscapes, or a small training fallback for notebook checks.

    The fallback mirrors the competition notebook pattern.  It is deliberately
    marked as non-competition inference because its generated row IDs are not
    the IDs from the hidden test-set template.
    """
    test_paths = sorted(Path(test_soundscapes_directory).glob("*.ogg"))
    if test_paths:
        return test_paths, False
    if train_soundscapes_directory is None:
        raise FileNotFoundError(f"No .ogg files in test soundscapes: {test_soundscapes_directory}")
    fallback_paths = sorted(Path(train_soundscapes_directory).glob("*.ogg"))[:fallback_limit]
    if not fallback_paths:
        raise FileNotFoundError(
            f"No .ogg files in test soundscapes or fallback train soundscapes: "
            f"{test_soundscapes_directory}, {train_soundscapes_directory}"
        )
    return fallback_paths, True


class SharedWindowEnsemble:
    """Ensemble that reads a waveform and computes its Mel windows exactly once."""

    def __init__(
        self,
        checkpoint_paths: Iterable[str | Path],
        weights: Iterable[float] | None = None,
        settings: SubmissionSettings | None = None,
        device_override: str | None = "cpu",
    ) -> None:
        self.paths = [Path(path) for path in checkpoint_paths]
        if not self.paths:
            raise ValueError("At least one checkpoint is required")
        self.settings = settings or SubmissionSettings()
        if device_override not in {None, "cpu"}:
            raise ValueError("Final submission inference is configured for CPU only")
        self.device = resolve_device("cpu")
        configure_accelerator(self.device, allow_tf32=True)
        payloads: list[dict[str, Any]] = []
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            payloads.append(torch.load(path, map_location="cpu", weights_only=False))
        audio_configs = [payload["audio_config"] for payload in payloads]
        if any(config != audio_configs[0] for config in audio_configs[1:]):
            raise ValueError("Selected checkpoints do not share identical audio preprocessing settings")
        model_configs = [payload["model_config"] for payload in payloads]
        if any(int(config["num_classes"]) != 206 for config in model_configs):
            raise ValueError("Every selected checkpoint must output 206 classes")
        self.preprocessor = AudioPreprocessor(AudioConfig(**audio_configs[0]))
        overlap_settings = OverlapSettings(
            window_seconds=self.preprocessor.config.chunk_duration,
            stride_seconds=self.settings.stride_seconds,
            batch_size=self.settings.batch_size,
        )
        self.members: list[OverlapAwareInferencer] = []
        for payload in payloads:
            model = EfficientNetSED(SEDConfig(**payload["model_config"])).to(self.device).eval()
            model.load_state_dict(payload["model_state_dict"])
            self.members.append(OverlapAwareInferencer(model, self.preprocessor, overlap_settings, self.device))
        raw_weights = np.ones(len(self.members), dtype=np.float64) if weights is None else np.asarray(tuple(weights), dtype=np.float64)
        if raw_weights.shape != (len(self.members),) or not np.isfinite(raw_weights).all() or np.any(raw_weights < 0):
            raise ValueError("Weights must be one finite, non-negative value per checkpoint")
        if raw_weights.sum() <= 0:
            raise ValueError("At least one ensemble weight must be positive")
        self.weights = raw_weights / raw_weights.sum()

    @torch.no_grad()
    def predict_waveform(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return equal- or configured-weight five-second ensemble scores."""
        windows, intervals = self.members[0].windows_from_waveform(waveform)
        all_scores, overlap_counts = [], None
        for member in self.members:
            scores, counts = member.predict_windows(windows, intervals)
            if overlap_counts is None:
                overlap_counts = counts
            elif not np.array_equal(overlap_counts, counts):
                raise RuntimeError("Ensemble checkpoints produced incompatible overlap offsets")
            all_scores.append(scores)
        combined = np.average(np.stack(all_scores), axis=0, weights=self.weights).astype(np.float32)
        if combined.shape != (intervals, 206) or not np.isfinite(combined).all():
            raise RuntimeError("Ensemble produced invalid five-second class scores")
        return combined, overlap_counts

    @torch.no_grad()
    def predict_file(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Load an audio file once, then run all checkpoints on shared windows."""
        waveform = self.preprocessor.load_audio(path)
        return self.predict_waveform(waveform)


def prediction_rows_for_recording(
    recording_id: str,
    interval_scores: np.ndarray,
    expected_row_ids: Iterable[str],
    stride_seconds: int = 5,
) -> dict[str, np.ndarray]:
    """Map a recording's interval scores onto its exact expected template IDs."""
    scores = np.asarray(interval_scores, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != 206 or not np.isfinite(scores).all():
        raise ValueError("Interval scores must be finite with shape (intervals, 206)")
    rows: dict[str, np.ndarray] = {}
    for row_id in expected_row_ids:
        row_recording_id, interval = parse_row_id(row_id, stride_seconds)
        if row_recording_id != recording_id:
            raise ValueError(f"Row ID {row_id!r} does not belong to recording {recording_id!r}")
        if interval >= len(scores):
            raise ValueError(
                f"{recording_id} has {len(scores)} inferred intervals but template requires interval {interval + 1}"
            )
        rows[row_id] = scores[interval]
    return rows


def verify_submission(path: str | Path, sample_submission_path: str | Path) -> dict[str, int]:
    """Raise on schema, row-ID, duplicate, or non-finite-score errors."""
    expected_header, expected_ids = read_submission_template(sample_submission_path)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            actual_header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Submission is empty: {path}") from error
        rows = list(reader)
    if actual_header != expected_header:
        raise ValueError("Submission columns do not exactly match the competition template")
    actual_ids = [row[0] for row in rows]
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Submission row IDs differ from the template or contain duplicates")
    values = np.asarray([[float(value) for value in row[1:]] for row in rows], dtype=np.float32)
    if values.shape != (len(expected_ids), 206) or not np.isfinite(values).all():
        raise ValueError("Submission scores must be finite and have 206 values per row")
    return {"rows": len(rows), "class_columns": values.shape[1]}


def create_submission(
    ensemble: SharedWindowEnsemble,
    test_soundscapes_directory: str | Path,
    train_soundscapes_directory: str | Path | None,
    sample_submission_path: str | Path,
    mapping_path: str | Path,
    output_path: str | Path,
    prefetched_interval_scores: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run shared-window inference over expected recordings and create a verified CSV."""
    header, template_ids = read_submission_template(sample_submission_path)
    mapping_labels = label_order_from_mapping(mapping_path)
    if set(header[1:]) != set(mapping_labels):
        raise ValueError("Competition class columns do not match the authoritative 206-label mapping")
    reorder = np.asarray([mapping_labels.index(label) for label in header[1:]], dtype=np.int64)
    soundscapes, fallback_mode = resolve_soundscapes(
        test_soundscapes_directory, train_soundscapes_directory
    )
    available_paths = {path.stem: path for path in soundscapes}
    if len(available_paths) != len(soundscapes):
        raise ValueError("Soundscape stems must be unique")
    rows_by_recording: dict[str, list[str]] = {}
    if fallback_mode:
        # The test template names hidden recordings, so create a schema-correct
        # diagnostic table for the fallback paths instead of fabricating a
        # supposedly valid competition submission.
        for path in soundscapes:
            rows_by_recording[path.stem] = []
        template_ids = []
    else:
        for row_id in template_ids:
            recording_id, _ = parse_row_id(row_id, ensemble.settings.stride_seconds)
            rows_by_recording.setdefault(recording_id, []).append(row_id)
    predictions: dict[str, np.ndarray] = {}
    prefetched_interval_scores = prefetched_interval_scores or {}
    for recording_id, row_ids in sorted(rows_by_recording.items()):
        if recording_id in prefetched_interval_scores:
            scores = prefetched_interval_scores[recording_id]
        else:
            audio_path = available_paths.get(recording_id)
            if audio_path is None:
                raise FileNotFoundError(f"Missing competition soundscape required by template: {recording_id}.ogg")
            scores, _ = ensemble.predict_file(audio_path)
        if fallback_mode:
            row_ids = [
                f"{recording_id}_{(index + 1) * ensemble.settings.stride_seconds}"
                for index in range(len(scores))
            ]
            rows_by_recording[recording_id] = row_ids
            template_ids.extend(row_ids)
        predictions.update(
            prediction_rows_for_recording(recording_id, scores, row_ids, ensemble.settings.stride_seconds)
        )
    if set(predictions) != set(template_ids):
        raise RuntimeError("Not every template row received exactly one prediction")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row_id in template_ids:
            writer.writerow([row_id, *predictions[row_id][reorder].tolist()])
    if fallback_mode:
        with output.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            actual_header = next(reader)
            output_rows = list(reader)
        values = np.asarray([[float(value) for value in row[1:]] for row in output_rows], dtype=np.float32)
        if actual_header != header or len(output_rows) != len(set(row[0] for row in output_rows)) or not np.isfinite(values).all():
            raise ValueError("Fallback prediction table failed schema, duplicate-ID, or finite-score validation")
        summary: dict[str, Any] = {
            "rows": len(output_rows), "class_columns": 206, "competition_submission": False,
            "mode": "train_soundscape_fallback_debug",
        }
    else:
        summary = verify_submission(output, sample_submission_path)
        summary["competition_submission"] = True
        summary["mode"] = "competition_test_soundscapes"
    summary["recordings"] = len(rows_by_recording)
    return summary


def smoke_test(
    ensemble: SharedWindowEnsemble, soundscapes: Iterable[Path], output_path: str | Path
) -> tuple[dict[str, int], np.ndarray]:
    """Run a single recording before full inference and persist its interval scores."""
    soundscapes = sorted(soundscapes)
    if not soundscapes:
        raise FileNotFoundError("No soundscapes supplied for smoke testing")
    selected = soundscapes[0]
    scores, counts = ensemble.predict_file(selected)
    if scores.shape[1] != 206 or not np.isfinite(scores).all():
        raise RuntimeError("Smoke-test prediction has invalid class scores")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, recording_id=selected.stem, scores=scores, overlap_counts=counts)
    return {
        "recording": selected.stem,
        "intervals": int(scores.shape[0]),
        "class_columns": int(scores.shape[1]),
    }, scores


def _configured_final_paths(config_path: Path) -> tuple[list[Path], list[float], SubmissionSettings, dict[str, Path]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.parent
    ensemble = config["ensemble"]
    checkpoint_paths = [root / value for value in ensemble["selected_checkpoint_paths"]]
    weights = [float(value) for value in ensemble.get("selected_weights", [])]
    inference = config["inference"]
    settings = SubmissionSettings(
        stride_seconds=int(inference["window_stride_seconds"]),
        batch_size=int(inference["batch_size"]),
        smoothing_enabled=bool(ensemble.get("smoothing_enabled", False)),
    )
    paths = {name: root / value for name, value in config["paths"].items()}
    return checkpoint_paths, weights, settings, paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a verified BirdCLEF competition submission.")
    parser.add_argument("--config", type=Path, default=Path("config/project_config.json"))
    parser.add_argument("--checkpoint", type=Path, action="append", help="Overrides configured final checkpoint paths.")
    parser.add_argument("--weight", type=float, action="append", help="One non-negative weight per --checkpoint.")
    parser.add_argument("--test-soundscapes", type=Path)
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/submission.csv"))
    smoke_group = parser.add_mutually_exclusive_group()
    smoke_group.add_argument("--smoke-test", action="store_true", help="Only score one recording and save its five-second scores.")
    smoke_group.add_argument(
        "--smoke-and-submit", action="store_true",
        help="Smoke-test one recording, reuse those scores, then write the full submission.",
    )
    parser.add_argument("--smoke-output", type=Path, default=Path("artifacts/submission_smoke_test.npz"))
    parser.add_argument("--device", choices=("cpu",), default="cpu", help="Submission inference is CPU-only.")
    parser.add_argument("--batch-size", type=int)
    arguments = parser.parse_args()
    configured_paths, configured_weights, settings, configured_data_paths = _configured_final_paths(arguments.config)
    if arguments.batch_size is not None:
        settings = SubmissionSettings(settings.stride_seconds, arguments.batch_size, settings.smoothing_enabled)
    checkpoint_paths = arguments.checkpoint or configured_paths
    weights = arguments.weight if arguments.weight is not None else configured_weights
    ensemble = SharedWindowEnsemble(checkpoint_paths, weights or None, settings, arguments.device)
    test_directory = arguments.test_soundscapes or configured_data_paths["test_soundscapes_dir"]
    soundscapes, fallback_mode = resolve_soundscapes(test_directory, configured_data_paths["unlabeled_soundscapes_dir"])
    if fallback_mode:
        print("test_soundscapes is empty; using the first three train_soundscapes for a debug-only run.", flush=True)
    if arguments.smoke_test:
        summary, _ = smoke_test(ensemble, soundscapes, arguments.smoke_output)
        summary["competition_submission"] = False
        summary["mode"] = "train_soundscape_fallback_debug" if fallback_mode else "competition_test_soundscapes"
        print(json.dumps(summary, indent=2))
    else:
        sample_submission = arguments.sample_submission or configured_data_paths["sample_submission"]
        mapping = arguments.mapping or configured_data_paths["label_mapping"]
        prefetched = None
        if arguments.smoke_and_submit:
            smoke_summary, smoke_scores = smoke_test(ensemble, soundscapes, arguments.smoke_output)
            prefetched = {smoke_summary["recording"]: smoke_scores}
        print(json.dumps(
            create_submission(
                ensemble, test_directory, configured_data_paths["unlabeled_soundscapes_dir"], sample_submission,
                mapping, arguments.output, prefetched,
            ), indent=2
        ))
