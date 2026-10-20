"""Consistent waveform-to-spectrogram preprocessing for BirdCLEF models.

Policy: both training and inference normalize each *final* 20-second waveform
chunk by its absolute maximum before computing a Mel power spectrogram. The
spectrogram is converted to dB with an 80 dB floor and mapped to ``[0, 1]``.
Inference will use this same function for every padded overlap window; it does
not normalize an entire recording before windowing.

Inputs:
  - a file path, or a mono/multichannel waveform shaped ``(samples,)`` or
    ``(channels, samples)``;
Outputs:
  - one model input tensor shaped ``(3, 224, 512)``;
  - a batch shaped ``(B, 3, 224, 512)``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


CropMode = Literal["train", "validation", "inference"]


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 32000
    chunk_duration: float = 20.0
    n_mels: int = 224
    n_fft: int = 4096
    hop_length: int = 1252
    spectrogram_width: int = 512
    fmin: float = 0.0
    fmax: float = 16000.0
    top_db: float = 80.0
    epsilon: float = 1e-8

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_duration)

    @classmethod
    def from_json(cls, path: str | Path = "config/project_config.json") -> "AudioConfig":
        with Path(path).open(encoding="utf-8") as handle:
            values = json.load(handle)["audio"]
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in valid})


class AudioPreprocessor:
    """Load, crop/pad, normalize, and transform audio into CNN inputs."""

    def __init__(self, config: AudioConfig | None = None, seed: int = 2025) -> None:
        self.config = config or AudioConfig()
        self._rng = np.random.default_rng(seed)
        self._window = torch.hann_window(self.config.n_fft, dtype=torch.float32)
        self._mel_filter = torch.from_numpy(self._make_mel_filter())

    def load_audio(self, path: str | Path) -> np.ndarray:
        """Load any supported file as a finite mono float32 waveform at 32 kHz."""
        waveform, source_rate = sf.read(path, always_2d=True, dtype="float32")
        waveform = waveform.mean(axis=1)
        if source_rate != self.config.sample_rate:
            divisor = math.gcd(source_rate, self.config.sample_rate)
            waveform = resample_poly(
                waveform, self.config.sample_rate // divisor, source_rate // divisor
            )
        waveform = np.asarray(waveform, dtype=np.float32)
        if not np.isfinite(waveform).all():
            raise ValueError(f"Audio contains non-finite samples: {path}")
        return waveform

    def crop_or_pad(
        self, waveform: np.ndarray, mode: CropMode = "train", start_sample: int | None = None
    ) -> np.ndarray:
        """Return exactly 20 seconds; validation/inference use a deterministic center crop."""
        waveform = self._as_mono(waveform)
        needed = self.config.chunk_samples
        if len(waveform) < needed:
            return np.pad(waveform, (0, needed - len(waveform)), mode="constant")
        if len(waveform) == needed:
            return waveform.copy()

        maximum_start = len(waveform) - needed
        if start_sample is not None:
            if not 0 <= start_sample <= maximum_start:
                raise ValueError(f"start_sample must be in [0, {maximum_start}]")
            start = start_sample
        elif mode == "train":
            start = int(self._rng.integers(0, maximum_start + 1))
        else:
            start = maximum_start // 2
        return waveform[start : start + needed].copy()

    def normalize_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Absmax-normalize a final chunk, retaining an all-zero waveform as zero."""
        waveform = self._as_mono(waveform)
        maximum = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if maximum <= self.config.epsilon:
            return np.zeros_like(waveform, dtype=np.float32)
        return (waveform / (maximum + self.config.epsilon)).astype(np.float32, copy=False)

    def mel_spectrogram(self, normalized_chunk: np.ndarray) -> np.ndarray:
        """Produce one normalized Mel image with shape ``(n_mels, width)``."""
        if len(normalized_chunk) != self.config.chunk_samples:
            raise ValueError("mel_spectrogram requires a final chunk of exactly chunk_duration")
        waveform = torch.from_numpy(normalized_chunk)
        spectrum = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            window=self._window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        mel_power = self._mel_filter @ spectrum.abs().square()
        maximum = mel_power.max()
        if maximum <= self.config.epsilon:
            unit_interval = np.zeros(tuple(mel_power.shape), dtype=np.float32)
        else:
            # Equivalent to a reference-at-maximum dB conversion, capped to
            # the configured 80 dB dynamic range before mapping to [0, 1].
            db = 10.0 * torch.log10(torch.clamp(mel_power, min=self.config.epsilon) / maximum)
            db = torch.clamp(db, min=-self.config.top_db, max=0.0)
            unit_interval = ((db + self.config.top_db) / self.config.top_db).numpy()
        if unit_interval.shape != (self.config.n_mels, self.config.spectrogram_width):
            raise RuntimeError(
                f"Unexpected Mel shape {unit_interval.shape}; expected "
                f"({self.config.n_mels}, {self.config.spectrogram_width})"
            )
        return unit_interval.astype(np.float32, copy=False)

    def waveform_to_tensor(
        self, waveform: np.ndarray, mode: CropMode = "train", start_sample: int | None = None
    ) -> torch.Tensor:
        """Convert a waveform to a three-channel EfficientNet-ready tensor."""
        chunk = self.crop_or_pad(waveform, mode=mode, start_sample=start_sample)
        normalized = self.normalize_waveform(chunk)
        mel = self.mel_spectrogram(normalized)
        return torch.from_numpy(np.repeat(mel[None, :, :], repeats=3, axis=0))

    def file_to_tensor(
        self, path: str | Path, mode: CropMode = "train", start_sample: int | None = None
    ) -> torch.Tensor:
        return self.waveform_to_tensor(self.load_audio(path), mode=mode, start_sample=start_sample)

    def batch_to_tensor(
        self, waveforms: Iterable[np.ndarray], mode: CropMode = "validation"
    ) -> torch.Tensor:
        """Stack independently preprocessed waveforms without changing channel geometry."""
        tensors = [self.waveform_to_tensor(waveform, mode=mode) for waveform in waveforms]
        if not tensors:
            raise ValueError("Cannot create a batch from zero waveforms")
        return torch.stack(tensors, dim=0)

    @staticmethod
    def _as_mono(waveform: np.ndarray) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
        if waveform.ndim != 1:
            raise ValueError("waveform must have shape (samples,) or (channels, samples)")
        if not np.isfinite(waveform).all():
            raise ValueError("waveform contains non-finite samples")
        return waveform

    def _make_mel_filter(self) -> np.ndarray:
        """Create a deterministic triangular HTK Mel filter bank once per instance."""
        def hz_to_mel(hz: np.ndarray) -> np.ndarray:
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel: np.ndarray) -> np.ndarray:
            return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)

        mel_points = np.linspace(
            hz_to_mel(np.array([self.config.fmin]))[0],
            hz_to_mel(np.array([self.config.fmax]))[0],
            self.config.n_mels + 2,
        )
        hz_points = mel_to_hz(mel_points)
        fft_frequencies = np.linspace(0, self.config.sample_rate / 2, self.config.n_fft // 2 + 1)
        filters = np.zeros((self.config.n_mels, len(fft_frequencies)), dtype=np.float32)
        for index in range(self.config.n_mels):
            left, center, right = hz_points[index : index + 3]
            rising = (fft_frequencies - left) / max(center - left, self.config.epsilon)
            falling = (right - fft_frequencies) / max(right - center, self.config.epsilon)
            filters[index] = np.maximum(0.0, np.minimum(rising, falling))
        return filters
