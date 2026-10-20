"""Labelled waveform dataset for the initial supervised teacher."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from src.audio import AudioPreprocessor, CropMode


def _secondary_labels(value: str) -> list[str]:
    try:
        labels = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return [str(label) for label in labels] if isinstance(labels, list) else []


class LabelledWaveformDataset(Dataset):
    """Recording-level labels paired with cropped/padded 20-second waveforms.

    The target is a float32 multi-hot vector of shape ``(206,)``. Human-provided
    secondary labels are included when they are competition target classes.
    """

    def __init__(
        self,
        split: str,
        split_path: str | Path,
        mapping_path: str | Path,
        audio_dir: str | Path,
        preprocessor: AudioPreprocessor,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        with Path(mapping_path).open(encoding="utf-8") as handle:
            self.label2ind = json.load(handle)["label2ind"]
        with Path(split_path).open(newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows found for {split=}")
        self.audio_dir = Path(audio_dir)
        self.preprocessor = preprocessor
        self.crop_mode: CropMode = "train" if split == "train" else "validation"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.rows[index]
        waveform = self.preprocessor.load_audio(self.audio_dir / row["filename"])
        chunk = self.preprocessor.crop_or_pad(waveform, mode=self.crop_mode)
        target = np.zeros(len(self.label2ind), dtype=np.float32)
        for label in {row["primary_label"], *_secondary_labels(row["secondary_labels"])}:
            label_index = self.label2ind.get(label)
            if label_index is not None:
                target[label_index] = 1.0
        return chunk, target
