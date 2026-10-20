"""Load a supervised checkpoint and score one 20-second audio crop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.audio import AudioConfig, AudioPreprocessor
from src.model import EfficientNetSED, SEDConfig
from src.train import best_device, clip_logits


@torch.no_grad()
def predict_file(checkpoint_path: Path, audio_path: Path, mapping_path: Path) -> dict[str, float]:
    device = best_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EfficientNetSED(SEDConfig(**checkpoint["model_config"])).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    preprocessor = AudioPreprocessor(AudioConfig(**checkpoint["audio_config"]))
    tensor = preprocessor.file_to_tensor(audio_path, mode="inference").unsqueeze(0).to(device)
    probabilities = torch.sigmoid(clip_logits(model(tensor))).squeeze(0).cpu().tolist()
    with mapping_path.open(encoding="utf-8") as handle:
        index_to_label = json.load(handle)["ind2label"]
    return {index_to_label[str(index)]: score for index, score in enumerate(probabilities)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clip-level inference from a teacher checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--mapping", type=Path, default=Path("artifacts/label_mapping.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/clip_prediction.json"))
    arguments = parser.parse_args()
    scores = predict_file(arguments.checkpoint, arguments.audio, arguments.mapping)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
    print(f"Wrote {arguments.output}")
