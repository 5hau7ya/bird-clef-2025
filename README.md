# bird-clef-2025

## Architecture

- The project is a multi-label acoustic species detection system built around EfficientNet-B0, sound event detection, semi supervised learning, and ensemble inference for the [BirdCLEF+ 2025](https://www.kaggle.com/competitions/birdclef-2025) competition held on Kaggle.
- BirdCLEF+ 2025 involves identyfing species from environmental audio recordings. A recording may contain several overlapping vocalisations, substantial environmental noise, and species that are difficult to identify from a short segment.

- The overall architecture of the project is as follows:

<img width="1448" height="1086" alt="Audio Classification Training Pipeline" src="https://github.com/user-attachments/assets/bde6b3bd-c233-49af-b7d2-7fd2a3912650" />

1. Dataset Preparation and Auditing (`data_audit.py` & `dataset.py`)
   
   1.1 Authoritative Label Mapping: using `sample_submission.csv`, we create two mappings — `label2ind` (maps species label to an integer) and `ind2label` (reverse mapping of `label2ind`) — inside the `label_mapping.json` file.
   
   1.2 Recording-Level Training and Validation Split: The split is grouped by source filename rather than by randomly selected audio crops. This prevents multiple rows originating from the same recording from being assigned to both training and validation. The code additionally preserves rare label representation.

2. Audio Preprocessing Architecture (`audio.py`)
   * Converts raw audio into a normalized spectrogram that the EfficientNet model will process:
    ```text
              Raw audio waveform
         Mono, resampled to 32,000 Hz
                      ↓
       20-second crop or zero-padding
            640,000 audio samples
                      ↓
      Waveform normalization and STFT
    Convert audio into time-frequency energy
                      ↓
       224-bin Mel power spectrogram
          80 dB range, scaled to [0, 1]
                      ↓
    ┌────────────────────────────────────────┐
    │           EfficientNet input           │
    │       Tensor shape: (3, 224, 512)      │
    └────────────────────────────────────────┘
    ```

3. Neural network architecture: EfficientNetSED (`audio.py`)
   <img width="3584" height="303" alt="sed_architecture" src="https://github.com/user-attachments/assets/e29c726d-d2a1-434c-b777-1a23773a319c" />

   * The network is an EfficientNet feature extractor followed by a temporal sound event detection head.

4. Training the supervised teacher (`train.py`)
   * We train the initial teacher which is the supervised baseline for the entire project
   * Waveform-level MixUp: training pipeline randomly applies MixUp to the audio waveforms. MixUp is applied with probability 0.5. Mixing is done before the spectrogram transformation. This allows the CNN to learn from blended acoustic signals rather than merely interpolating already-generated spectrograms.
   

      Given two waveforms $(x_i, x_j)$ and their corresponding label vectors $(y_i, y_j)$, Mixup generates a new training sample as follows:
      
      $$
      \tilde{x} = \lambda x_i + (1-\lambda)x_j
      $$
      
      $$
      \tilde{y} = \lambda y_i + (1-\lambda)y_j
      $$
      
      The mixing coefficient is sampled from a Beta distribution:
      
      $$
      \lambda \sim \beta(0.5, 0.5)
      $$

* Training Objective: We use binary cross entropy with logits. The optimiser is AdamW with an initial learning rate of $5 \times 10^{-4}$ and weight decay of $10^{-4}$. The learning rate follows a cosine annealing curve that triggers a warm restart every five epochs.
* After every epoch, macro ROC-AUC is evaluated on a fixed validation set and the best checkpoint is chosen accordingly. The best supervised teacher validation macro ROC-AUC was 0.84077.

## Repository Structure

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

- `config/project_config.json`: keeps signal parameters, model choices, paths, training settings, self-training plan, and final ensemble choice together.

- `bird-clef-2025-train.ipynb`: consolidated training notebook. The actual model was trained in parts and checkpoints were saved because of the 12 hour limit on Kaggle, this notebook is the concatenation of those small parts.

- `bird-clef-2025-inference.ipynb`: Kaggle notebook used for generating the final `submission.csv`. It runs on CPU as per the competitions requirements.

- The trained checkpoints are available [here](https://www.kaggle.com/datasets/shaury4/bird-clef-2025-dataset)

## Setup

- To train the model upload the `bird-clef-2025-train.ipynb` on Kaggle and add BirdCLEF+ 2025 dataset as input. Alternatively you can use this [notebook](https://www.kaggle.com/code/shaury4/bird-clef-2025-train) on Kaggle directly. If you are training the model locally then update the path of the dataset in `project_config.json` accordingly.
- To generate submission.csv upload the `bird-clef-2025-inference.ipynb` on Kaggle and add this [dataset](https://www.kaggle.com/datasets/shaury4/bird-clef-2025-dataset) along with the BirdCLEF+ 2025 dataset as input. Alternatively you can use this [notebook](https://www.kaggle.com/code/shaury4/bird-clef-2025-inference) on Kaggle directly. If you are inferring from the model locally then update the path of the dataset in the notebook and in `project_config.json` accordingly. In order to infer from the model on custom dataset update the path of the dataset in the notebook and in `project_config.json` accordingly.
- Note: `bird-clef-2025-train.ipynb` has been updated but `bird-clef-2025-inference.ipynb` is yet to be updated, hence if you train the model then you will have to make some changes in `bird-clef-2025-inference.ipynb` and `project_config.json` to match the paths of the required checkpoints. I will be updating `bird-clef-2025-inference.ipynb` soon. Thanks for your understanding.
