# bird-clef-2025

## Introduction

- The project is a multi-label acoustic species detection system built around EfficientNet-B0, sound event detection, semi supervised learning, and ensemble inference for the [BirdCLEF+ 2025](https://www.kaggle.com/competitions/birdclef-2025) competition held on Kaggle.
- BirdCLEF+ 2025 involves identyfing species from environmental audio recordings. A recording may contain several overlapping vocalisations, substantial environmental noise, and species that are difficult to identify from a short segment.
- The overall architecture of the project is as follows:

<img width="1448" height="1086" alt="Audio Classification Training Pipeline" src="https://github.com/user-attachments/assets/bde6b3bd-c233-49af-b7d2-7fd2a3912650" />

- The structure of the repository is as follows:
  
```text
bird-clef-2025/
│
├── bird-clef-2025-train.ipynb
│
├── bird-clef-2025-inference.ipynb
│
├── README.md
│
├── config/
│   └── project_config.json
│
└── src/
    ├── __init__.py
    ├── audio.py
    ├── data_audit.py
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── inference.py
    ├── overlap_inference.py
    ├── pseudo_labels.py
    ├── self_train.py
    ├── self_training_pipeline.py
    ├── ensemble_evaluation.py
    └── submission.py
```

| Module | Purpose |
| --- | --- |
| `src/data_audit.py` | Official label mapping and deterministic group-safe train/validation split. |
| `src/audio.py` | Mono loading, crop/pad, normalization, Mel conversion, and model tensor creation. |
| `src/dataset.py` | Labelled waveform dataset backed by split metadata and audio files. |
| `src/model.py` | EfficientNet-B0 SED model with frequency-only GeM pooling. |
| `src/train.py` | Supervised teacher training, validation, checkpointing, and accelerator helpers. |
| `src/inference.py` | Ordinary clip-level inference for one 20-second crop. |
| `src/overlap_inference.py` | Overlap-aware conversion of long recordings to five-second scores. |
| `src/pseudo_labels.py` | Versioned five-second pseudo-label data and diagnostic reports. |
| `src/self_train.py` | One labelled/pseudo-labelled student-training iteration. |
| `src/self_training_pipeline.py` | Repeatable three-stage teacher–student orchestration and lineage. |
| `src/ensemble_evaluation.py` | Individual checkpoint, pair, three-model, and smoothing comparison. |
| `src/submission.py` | CPU-only selected-ensemble inference and verified CSV generation. |

`config/project_config.json`: keeps signal parameters, model choices, paths, training settings, self-training plan, and final ensemble choice together.

`bird-clef-2025-train.ipynb`: consolidated training notebook. The actual model was trained in parts and checkpoints were saved because of the 12 hour limit on Kaggle, this notebook is the concatenation of those small parts.

`bird-clef-2025-inference.ipynb`: Kaggle notebook used for generating the final `submission.csv`. It runs on CPU as per the competitions requirements.
