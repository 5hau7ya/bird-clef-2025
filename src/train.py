"""Supervised teacher training: waveform MixUp, BCE loss, and validation AUC."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.audio import AudioConfig, AudioPreprocessor
from src.dataset import LabelledWaveformDataset
from src.model import EfficientNetSED, SEDConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def waveform_batch_to_inputs(
    chunks: torch.Tensor, preprocessor: AudioPreprocessor, device: torch.device
) -> torch.Tensor:
    """Normalize and transform already-MixUp'ed `(B, samples)` chunks to CNN inputs."""
    tensors = [
        preprocessor.waveform_to_tensor(chunk.numpy(), mode="validation")
        for chunk in chunks.cpu()
    ]
    return torch.stack(tensors).to(device, non_blocking=True)


def mixup_waveforms(
    chunks: torch.Tensor, targets: torch.Tensor, probability: float, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one Beta-sampled waveform-level MixUp coefficient to a batch."""
    if probability <= 0 or torch.rand(()) >= probability:
        return chunks, targets
    coefficient = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(chunks.shape[0])
    return (
        coefficient * chunks + (1.0 - coefficient) * chunks[permutation],
        coefficient * targets + (1.0 - coefficient) * targets[permutation],
    )


def clip_logits(framewise_logits: torch.Tensor) -> torch.Tensor:
    """Max-pool `(B, 206, T)` framewise logits to recording/crop logits `(B, 206)`."""
    return framewise_logits.max(dim=-1).values


def macro_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Macro ROC-AUC over classes with at least one positive and one negative."""
    valid = (targets.sum(axis=0) > 0) & (targets.sum(axis=0) < len(targets))
    if not valid.any():
        raise ValueError("Validation split has no classes suitable for ROC-AUC")
    return float(roc_auc_score(targets[:, valid], probabilities[:, valid], average="macro"))


@torch.no_grad()
def validate(
    model: EfficientNetSED,
    loader: DataLoader,
    preprocessor: AudioPreprocessor,
    device: torch.device,
    loss_function: nn.Module,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses, all_targets, all_probabilities = [], [], []
    for chunks, targets in loader:
        inputs = waveform_batch_to_inputs(chunks, preprocessor, device)
        targets = targets.to(device)
        logits = clip_logits(model(inputs))
        losses.append(loss_function(logits, targets).item())
        all_targets.append(targets.cpu().numpy())
        all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
    targets_array = np.concatenate(all_targets)
    probabilities_array = np.concatenate(all_probabilities)
    return float(np.mean(losses)), macro_auc(targets_array, probabilities_array), targets_array, probabilities_array


def train(config_path: Path, epochs_override: int | None = None) -> Path:
    root = config_path.parent.parent
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    training = config["training"]
    set_seed(training["random_seed"])
    device = best_device()
    audio_preprocessor = AudioPreprocessor(AudioConfig.from_json(config_path), seed=training["random_seed"])
    paths = {name: root / value for name, value in config["paths"].items()}
    train_dataset = LabelledWaveformDataset(
        "train", paths["split"], paths["label_mapping"], paths["train_audio_dir"], audio_preprocessor
    )
    validation_dataset = LabelledWaveformDataset(
        "validation", paths["split"], paths["label_mapping"], paths["train_audio_dir"], audio_preprocessor
    )
    loader_args = {"batch_size": training["batch_size"], "num_workers": training["num_workers"]}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_args)
    model_config = SEDConfig.from_json(config_path)
    model = EfficientNetSED(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=training["scheduler_restart_epochs"],
        T_mult=1,
        eta_min=training["final_learning_rate"],
    )
    loss_function = nn.BCEWithLogitsLoss()
    output_dir = root / training["checkpoint_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = epochs_override or training["epochs"]
    logs_path = output_dir / "training_log.csv"
    best_auc = float("-inf")
    with logs_path.open("w", newline="", encoding="utf-8") as log_file:
        fields = ["epoch", "train_loss", "validation_loss", "validation_macro_auc", "learning_rate"]
        logger = csv.DictWriter(log_file, fieldnames=fields)
        logger.writeheader()
        for epoch in range(epochs):
            model.train()
            losses = []
            for batch_index, (chunks, targets) in enumerate(train_loader):
                chunks, targets = mixup_waveforms(
                    chunks, targets, training["mixup_probability"], training["mixup_alpha"]
                )
                inputs = waveform_batch_to_inputs(chunks, audio_preprocessor, device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(clip_logits(model(inputs)), targets)
                loss.backward()
                optimizer.step()
                scheduler.step(epoch + (batch_index + 1) / len(train_loader))
                losses.append(loss.item())
            validation_loss, validation_auc, y_true, y_score = validate(
                model, validation_loader, audio_preprocessor, device, loss_function
            )
            record = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
                "validation_macro_auc": validation_auc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            logger.writerow(record)
            log_file.flush()
            np.savez_compressed(output_dir / f"validation_predictions_epoch_{epoch + 1:02d}.npz", y_true=y_true, y_score=y_score)
            if validation_auc > best_auc:
                best_auc = validation_auc
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "validation_macro_auc": validation_auc,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "model_config": asdict(model_config),
                        "audio_config": asdict(audio_preprocessor.config),
                        "training_config": training,
                        "label_mapping_path": str(paths["label_mapping"]),
                    },
                    output_dir / "best_supervised_teacher.pt",
                )
            print(json.dumps(record), flush=True)
    return output_dir / "best_supervised_teacher.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the initial supervised BirdCLEF teacher.")
    parser.add_argument("--config", type=Path, default=Path("config/project_config.json"))
    parser.add_argument("--epochs", type=int, help="Override configured epoch count for a smoke run.")
    arguments = parser.parse_args()
    checkpoint = train(arguments.config, arguments.epochs)
    print(f"Best checkpoint: {checkpoint}")
