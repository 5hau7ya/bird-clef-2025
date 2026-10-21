"""First Noisy-Student iteration using labelled/pseudo-labelled waveform mixtures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.audio import AudioConfig, AudioPreprocessor
from src.dataset import LabelledWaveformDataset
from src.model import EfficientNetSED, SEDConfig
from src.pseudo_labels import PseudoLabelStore
from src.train import best_device, clip_logits, set_seed, validate, waveform_batch_to_inputs


def sampling_weights_from_scores(recording_ids: np.ndarray, scores: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Calculate ``sum_c max_t p[t,c]`` once per source soundscape.

    Invalid score rows contribute zero. If every score is invalid or zero, the
    returned weights are uniform so `WeightedRandomSampler` remains valid.
    """
    if scores.ndim != 2 or scores.shape[0] != len(recording_ids):
        raise ValueError("scores must have shape (segments, classes) matching recording_ids")
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for recording_id, score in zip(recording_ids.astype(str), scores):
        grouped[recording_id].append(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0))
    identifiers = sorted(grouped)
    weights = np.asarray([np.max(grouped[item], axis=0).sum() for item in identifiers], dtype=np.float64)
    weights[~np.isfinite(weights)] = 0.0
    if not np.any(weights > 0):
        weights.fill(1.0)
    else:
        weights = np.maximum(weights, np.finfo(np.float64).eps)
    return identifiers, weights


def weight_summary(weights: np.ndarray) -> dict[str, float]:
    probabilities = weights / weights.sum()
    return {
        "count": int(len(weights)),
        "minimum": float(weights.min()),
        "p01": float(np.quantile(weights, 0.01)),
        "median": float(np.median(weights)),
        "p99": float(np.quantile(weights, 0.99)),
        "maximum": float(weights.max()),
        "effective_sample_size": float(1.0 / np.square(probabilities).sum()),
        "top_1_percent_probability_mass": float(np.sort(probabilities)[-max(1, math.ceil(len(weights) * 0.01)) :].sum()),
    }


class PseudoLabelledWaveformDataset(Dataset):
    """One randomly aligned 20-second waveform crop per sampled soundscape."""

    def __init__(self, pseudo_directory: str | Path, soundscape_directory: str | Path, preprocessor: AudioPreprocessor) -> None:
        self.store = PseudoLabelStore(pseudo_directory)
        values = self.store.values
        self.preprocessor = preprocessor
        self.soundscape_directory = Path(soundscape_directory)
        ids = values["recording_ids"].astype(str)
        starts = values["segment_start_seconds"]
        self.recording_ids, self.weights = sampling_weights_from_scores(ids, values["scores"])
        self.crop_starts: dict[str, list[float]] = {}
        for recording_id in self.recording_ids:
            available = set(starts[ids == recording_id].tolist())
            candidates = [
                start for start in sorted(available)
                if all((start + offset) in available for offset in (0.0, 5.0, 10.0, 15.0))
            ]
            if candidates:
                self.crop_starts[recording_id] = candidates
        self.recording_ids = [item for item in self.recording_ids if item in self.crop_starts]
        retained = {item: weight for item, weight in zip(*sampling_weights_from_scores(ids, values["scores"]))}
        self.weights = np.asarray([retained[item] for item in self.recording_ids], dtype=np.float64)
        if not self.recording_ids:
            raise ValueError("Pseudo-label dataset has no recordings with four contiguous five-second scores")

    def __len__(self) -> int:
        return len(self.recording_ids)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, str, float]:
        recording_id = self.recording_ids[index]
        start_seconds = random.choice(self.crop_starts[recording_id])
        waveform = self.preprocessor.load_audio(self.soundscape_directory / f"{recording_id}.ogg")
        chunk = self.preprocessor.crop_or_pad(
            waveform,
            mode="validation",
            start_sample=round(start_seconds * self.preprocessor.config.sample_rate),
        )
        target = np.nan_to_num(
            self.store.soft_target_for_crop(recording_id, start_seconds), nan=0.0, posinf=0.0, neginf=0.0
        ).clip(0.0, 1.0).astype(np.float32)
        return chunk, target, recording_id, start_seconds


def mix_labelled_and_pseudo(
    labelled_chunks: torch.Tensor,
    labelled_targets: torch.Tensor,
    pseudo_chunks: torch.Tensor,
    pseudo_targets: torch.Tensor,
    coefficient: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix waveforms and targets using the same fixed Noisy-Student coefficient."""
    if not 0 <= coefficient <= 1:
        raise ValueError("mix coefficient must lie in [0, 1]")
    if labelled_chunks.shape != pseudo_chunks.shape or labelled_targets.shape != pseudo_targets.shape:
        raise ValueError("Labelled and pseudo-labelled batches must have matching shapes")
    return (
        coefficient * labelled_chunks + (1 - coefficient) * pseudo_chunks,
        coefficient * labelled_targets + (1 - coefficient) * pseudo_targets,
    )


def train_one_student_iteration(config_path: Path, pseudo_directory: Path, epochs_override: int | None = None) -> Path:
    root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training, self_training = config["training"], config["self_training"]
    set_seed(training["random_seed"])
    device = best_device()
    paths = {name: root / value for name, value in config["paths"].items()}
    preprocessor = AudioPreprocessor(AudioConfig.from_json(config_path), seed=training["random_seed"])
    labelled_train = LabelledWaveformDataset("train", paths["split"], paths["label_mapping"], paths["train_audio_dir"], preprocessor)
    labelled_validation = LabelledWaveformDataset("validation", paths["split"], paths["label_mapping"], paths["train_audio_dir"], preprocessor)
    pseudo_train = PseudoLabelledWaveformDataset(pseudo_directory, paths["unlabeled_soundscapes_dir"], preprocessor)
    sampler = WeightedRandomSampler(pseudo_train.weights, num_samples=len(labelled_train), replacement=True)
    loader_kwargs = {"batch_size": training["batch_size"], "num_workers": training["num_workers"]}
    labelled_loader = DataLoader(labelled_train, shuffle=True, **loader_kwargs)
    pseudo_loader = DataLoader(pseudo_train, sampler=sampler, **loader_kwargs)
    validation_loader = DataLoader(labelled_validation, shuffle=False, **loader_kwargs)

    teacher_path = root / training["checkpoint_dir"] / "best_supervised_teacher.pt"
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    # Keep ordinary head dropout from the teacher configuration; only stochastic
    # depth changes for the noisy student.
    student_config = replace(SEDConfig(**teacher["model_config"]), drop_path_rate=self_training["drop_path_rate"])
    student = EfficientNetSED(student_config).to(device)
    student.load_state_dict(teacher["model_state_dict"])
    optimizer = AdamW(student.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=training["scheduler_restart_epochs"], T_mult=1, eta_min=training["final_learning_rate"]
    )
    loss_function = nn.BCEWithLogitsLoss()
    output_directory = root / self_training["checkpoint_dir"]
    output_directory.mkdir(parents=True, exist_ok=True)
    summary = weight_summary(pseudo_train.weights)
    (output_directory / "pseudo_sampler_weight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    epochs = epochs_override or self_training["first_iteration_epochs"]
    best_auc = float("-inf")
    with (output_directory / "training_log.csv").open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "validation_loss", "validation_macro_auc", "learning_rate"])
        writer.writeheader()
        for epoch in range(epochs):
            student.train()
            losses = []
            for batch_index, (labelled_batch, pseudo_batch) in enumerate(zip(labelled_loader, pseudo_loader)):
                labelled_chunks, labelled_targets = labelled_batch
                pseudo_chunks, pseudo_targets, _, _ = pseudo_batch
                chunks, targets = mix_labelled_and_pseudo(
                    labelled_chunks, labelled_targets, pseudo_chunks, pseudo_targets, self_training["mix_coefficient"]
                )
                inputs = waveform_batch_to_inputs(chunks, preprocessor, device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(clip_logits(student(inputs)), targets)
                loss.backward()
                optimizer.step()
                scheduler.step(epoch + (batch_index + 1) / len(labelled_loader))
                losses.append(loss.item())
            validation_loss, validation_auc, y_true, y_score = validate(student, validation_loader, preprocessor, device, loss_function)
            record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss,
                      "validation_macro_auc": validation_auc, "learning_rate": optimizer.param_groups[0]["lr"]}
            writer.writerow(record)
            log_file.flush()
            np.savez_compressed(output_directory / f"validation_predictions_epoch_{epoch + 1:02d}.npz", y_true=y_true, y_score=y_score)
            if validation_auc > best_auc:
                best_auc = validation_auc
                torch.save({
                    "epoch": epoch + 1, "validation_macro_auc": validation_auc,
                    "model_state_dict": student.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(), "model_config": asdict(student_config),
                    "audio_config": asdict(preprocessor.config), "training_config": training,
                    "self_training_config": self_training, "teacher_checkpoint": str(teacher_path),
                    "pseudo_label_directory": str(pseudo_directory), "pseudo_sampler_weight_summary": summary,
                }, output_directory / "best_student_iteration_1.pt")
            print(json.dumps(record), flush=True)
    return output_directory / "best_student_iteration_1.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the first self-training student iteration.")
    parser.add_argument("pseudo_directory", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/project_config.json"))
    parser.add_argument("--epochs", type=int)
    arguments = parser.parse_args()
    print(train_one_student_iteration(arguments.config, arguments.pseudo_directory, arguments.epochs))
