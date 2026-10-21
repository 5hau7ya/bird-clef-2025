"""Overlap-aware five-second inference for a single SED checkpoint.

The 20-second windows are centered on target five-second intervals. A 60-second
recording therefore has 12 windows, starts at -7.5, -2.5, ..., 47.5 seconds,
and is padded at both boundaries. Framewise window scores are stitched and
averaged before the artificial six-frame margins are removed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.audio import AudioConfig, AudioPreprocessor
from src.model import EfficientNetSED, SEDConfig
from src.train import best_device


@dataclass(frozen=True)
class OverlapSettings:
    window_seconds: float = 20.0
    stride_seconds: float = 5.0
    batch_size: int = 16


class OverlapAwareInferencer:
    """Create five-second class scores by overlap-stitching SED frame outputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        preprocessor: AudioPreprocessor,
        settings: OverlapSettings | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.settings = settings or OverlapSettings(
            window_seconds=preprocessor.config.chunk_duration,
        )
        self.device = device or best_device()
        if self.settings.window_seconds != preprocessor.config.chunk_duration:
            raise ValueError("Inference window duration must equal the preprocessing chunk duration")
        if self.settings.window_seconds <= self.settings.stride_seconds:
            raise ValueError("Window duration must exceed stride for overlap-aware inference")

    @property
    def context_seconds(self) -> float:
        return (self.settings.window_seconds - self.settings.stride_seconds) / 2

    def windows_from_waveform(self, waveform: np.ndarray) -> tuple[torch.Tensor, int]:
        """Compute reusable padded Mel power once, then extract normalized 20-second windows."""
        waveform = self.preprocessor._as_mono(waveform)
        interval_samples = int(self.settings.stride_seconds * self.preprocessor.config.sample_rate)
        intervals = math.ceil(len(waveform) / interval_samples)
        if intervals == 0:
            raise ValueError("Cannot infer from an empty waveform")
        context_samples = int(self.context_seconds * self.preprocessor.config.sample_rate)
        # n_fft samples of additional right padding cover the fractional
        # hop-length mismatch between 5 seconds and 128 spectrogram columns.
        padded = np.pad(
            waveform,
            (context_samples, context_samples + self.preprocessor.config.n_fft),
            mode="constant",
        )
        input_step = round(
            self.settings.stride_seconds
            * self.preprocessor.config.sample_rate
            / self.preprocessor.config.hop_length
        )
        width = self.preprocessor.config.spectrogram_width
        required_width = width + (intervals - 1) * input_step
        padded = self.preprocessor.normalize_waveform(padded)
        full_mel_power = self.preprocessor.mel_power_spectrogram(padded)
        if full_mel_power.shape[1] < required_width:
            # A source duration marginally greater than a five-second boundary
            # (for example, 60.001 s) needs one additional target interval.
            # Add only zero-valued right context, then recompute the shared
            # power Mel representation so every requested window is complete.
            missing_columns = required_width - full_mel_power.shape[1]
            padded = np.pad(
                padded,
                (0, missing_columns * self.preprocessor.config.hop_length + self.preprocessor.config.n_fft),
                mode="constant",
            )
            full_mel_power = self.preprocessor.mel_power_spectrogram(padded)
        if full_mel_power.shape[1] < required_width:
            raise RuntimeError("Unable to construct enough padded spectrogram columns")
        windows = []
        for index in range(intervals):
            start = index * input_step
            power_window = full_mel_power[:, start : start + width]
            unit_mel = self.preprocessor.power_to_unit_mel(power_window)
            windows.append(torch.from_numpy(np.repeat(unit_mel[None], 3, axis=0)))
        return torch.stack(windows), intervals

    @staticmethod
    def stitch_window_scores(window_scores: np.ndarray, step_frames: int) -> tuple[np.ndarray, np.ndarray]:
        """Average `(windows, classes, frames)` scores into one temporal sequence."""
        if window_scores.ndim != 3:
            raise ValueError("window_scores must have shape (windows, classes, frames)")
        windows, classes, frames = window_scores.shape
        if windows == 0 or step_frames <= 0:
            raise ValueError("Need at least one window and a positive frame offset")
        total_frames = frames + (windows - 1) * step_frames
        accumulation = np.zeros((total_frames, classes), dtype=np.float32)
        counts = np.zeros(total_frames, dtype=np.int32)
        for index, scores in enumerate(window_scores):
            start = index * step_frames
            accumulation[start : start + frames] += scores.T
            counts[start : start + frames] += 1
        return accumulation / counts[:, None], counts

    def pool_intervals(self, stitched: np.ndarray, output_frames: int, intervals: int) -> np.ndarray:
        """Remove context margins then maximum-pool each target five-second interval."""
        frames_per_interval = output_frames * self.settings.stride_seconds / self.settings.window_seconds
        margin_frames = output_frames * self.context_seconds / self.settings.window_seconds
        if not frames_per_interval.is_integer() or not margin_frames.is_integer():
            raise ValueError("Model frame count is incompatible with the configured timing")
        frames_per_interval, margin_frames = int(frames_per_interval), int(margin_frames)
        trimmed = stitched[margin_frames : stitched.shape[0] - margin_frames]
        expected_frames = intervals * frames_per_interval
        if trimmed.shape[0] != expected_frames:
            raise RuntimeError(f"Expected {expected_frames} post-margin frames, got {trimmed.shape[0]}")
        return trimmed.reshape(intervals, frames_per_interval, stitched.shape[1]).max(axis=1)

    @torch.no_grad()
    def predict_waveform(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return `(interval_scores, overlap_counts)` for one source waveform."""
        windows, intervals = self.windows_from_waveform(waveform)
        self.model.eval()
        score_batches = []
        for batch in windows.split(self.settings.batch_size):
            logits = self.model(batch.to(self.device))
            score_batches.append(torch.sigmoid(logits).cpu().numpy())
        window_scores = np.concatenate(score_batches, axis=0)
        output_frames = window_scores.shape[-1]
        step_frames = output_frames * self.settings.stride_seconds / self.settings.window_seconds
        if not step_frames.is_integer():
            raise ValueError("Model output frame count is incompatible with the configured stride")
        stitched, counts = self.stitch_window_scores(window_scores, int(step_frames))
        return self.pool_intervals(stitched, output_frames, intervals), counts

    def predict_file(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_waveform(self.preprocessor.load_audio(path))


def load_checkpoint_inferencer(checkpoint_path: Path, batch_size: int = 16) -> OverlapAwareInferencer:
    device = best_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EfficientNetSED(SEDConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return OverlapAwareInferencer(
        model,
        AudioPreprocessor(AudioConfig(**checkpoint["audio_config"])),
        OverlapSettings(batch_size=batch_size),
        device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate overlap-aware five-second predictions.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/overlap_prediction.npz"))
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()
    inferencer = load_checkpoint_inferencer(arguments.checkpoint, arguments.batch_size)
    scores, counts = inferencer.predict_file(arguments.audio)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, scores=scores, overlap_counts=counts)
    print(f"Wrote {arguments.output}: scores shape={scores.shape}")
