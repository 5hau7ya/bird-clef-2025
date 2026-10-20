"""Build reproducible metadata artifacts before any model training.

The submission header is the competition contract for both the set and order of
target columns.  This module deliberately does not derive target order from
alphabetical sorting or from the taxonomy table.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def target_labels_from_submission(path: Path) -> list[str]:
    """Return official target labels in exactly the sample-submission order."""
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    if not header or header[0] != "row_id":
        raise ValueError(f"Expected first submission column to be row_id: {path}")
    labels = header[1:]
    if len(labels) != len(set(labels)):
        raise ValueError("Sample-submission target columns are not unique")
    return labels


def parse_secondary_labels(value: str) -> list[str]:
    """Parse Kaggle's Python-list-like field without evaluating arbitrary code."""
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(label) for label in parsed if str(label).strip()]


def row_labels(row: dict[str, str], target_set: set[str]) -> set[str]:
    labels = {row["primary_label"]}
    labels.update(parse_secondary_labels(row.get("secondary_labels", "[]")))
    return labels & target_set


def grouped_multilabel_split(
    rows: list[dict[str, str]], target_set: set[str], validation_fraction: float, seed: int
) -> tuple[set[str], dict[str, Any]]:
    """Make a deterministic grouped holdout using recording filename as group.

    A group is the source recording, not an audio crop. The greedy assignment
    gives rare labels priority and never assigns a group to both partitions.
    Labels seen in only one group are retained in training, because a grouped
    split cannot represent them in both partitions.
    """
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["filename"]].append(row)

    group_labels = {
        group: set().union(*(row_labels(row, target_set) for row in grouped_rows))
        for group, grouped_rows in groups.items()
    }
    label_group_counts = Counter(label for labels in group_labels.values() for label in labels)
    eligible_labels = {label for label, count in label_group_counts.items() if count >= 2}
    desired = {
        label: max(1, round(label_group_counts[label] * validation_fraction))
        for label in eligible_labels
    }

    rng = random.Random(seed)
    ordered_groups = list(groups)
    rng.shuffle(ordered_groups)
    # Rarest-labelled recordings are allocated first; the random shuffle breaks ties stably.
    ordered_groups.sort(
        key=lambda group: min((label_group_counts[label] for label in group_labels[group]), default=10**9)
    )

    validation_groups: set[str] = set()
    validation_counts: Counter[str] = Counter()
    target_group_count = round(len(groups) * validation_fraction)
    for group in ordered_groups:
        labels = group_labels[group] & eligible_labels
        benefit = sum(max(0, desired[label] - validation_counts[label]) for label in labels)
        capacity = target_group_count - len(validation_groups)
        # Select a group only where it advances an unmet label target. This avoids
        # moving singleton-label groups and keeps the holdout close to the target size.
        preserves_training_coverage = all(
            validation_counts[label] < label_group_counts[label] - 1 for label in labels
        )
        if capacity > 0 and benefit > 0 and preserves_training_coverage:
            validation_groups.add(group)
            validation_counts.update(labels)

    # Fill any remaining capacity deterministically from groups that have labels
    # represented in at least two source recordings.
    for group in ordered_groups:
        if len(validation_groups) >= target_group_count:
            break
        labels = group_labels[group] & eligible_labels
        preserves_training_coverage = all(
            validation_counts[label] < label_group_counts[label] - 1 for label in labels
        )
        if group not in validation_groups and labels and preserves_training_coverage:
            validation_groups.add(group)
            validation_counts.update(labels)

    metadata = {
        "group_key": "filename",
        "num_groups": len(groups),
        "validation_groups": len(validation_groups),
        "singleton_group_labels_kept_in_train": sorted(
            label for label, count in label_group_counts.items() if count == 1
        ),
    }
    return validation_groups, metadata


def build_artifacts(config_path: Path) -> dict[str, Any]:
    root = config_path.parent.parent
    config = load_config(config_path)
    paths = {key: root / value for key, value in config["paths"].items()}
    rows = read_csv(paths["train_metadata"])
    taxonomy_rows = read_csv(paths["taxonomy"])
    labels = target_labels_from_submission(paths["sample_submission"])
    if len(labels) != config["model"]["num_classes"]:
        raise ValueError(f"Configured num_classes does not match submission targets: {len(labels)}")
    taxonomy_labels = {row["primary_label"] for row in taxonomy_rows}
    if set(labels) != taxonomy_labels:
        raise ValueError("Sample-submission target set does not match taxonomy primary_label set")

    label2ind = {label: index for index, label in enumerate(labels)}
    ind2label = {str(index): label for label, index in label2ind.items()}
    validation_groups, split_metadata = grouped_multilabel_split(
        rows,
        set(labels),
        config["training"]["validation_fraction"],
        config["training"]["random_seed"],
    )
    paths["label_mapping"].parent.mkdir(parents=True, exist_ok=True)
    with paths["label_mapping"].open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": "data/sample_submission.csv header (columns after row_id)",
                "num_classes": len(labels),
                "label2ind": label2ind,
                "ind2label": ind2label,
            },
            handle,
            indent=2,
            sort_keys=False,
        )

    split_rows = []
    for row in rows:
        split_rows.append(
            {
                "filename": row["filename"],
                "primary_label": row["primary_label"],
                "secondary_labels": row.get("secondary_labels", "[]"),
                "split": "validation" if row["filename"] in validation_groups else "train",
            }
        )
    with paths["split"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)

    missing_audio = [row["filename"] for row in rows if not (paths["train_audio_dir"] / row["filename"]).is_file()]
    split_counts = Counter(row["split"] for row in split_rows)
    labels_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    for row in rows:
        split = "validation" if row["filename"] in validation_groups else "train"
        labels_by_split[split].update(row_labels(row, set(labels)))
    source_partitions: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        source_partitions[row["filename"]].add(row["split"])
    leakage_sources = sorted(source for source, partitions in source_partitions.items() if len(partitions) > 1)
    unlabeled_dir = paths["unlabeled_soundscapes_dir"]
    unlabeled_files = list(unlabeled_dir.glob("*.ogg")) if unlabeled_dir.is_dir() else []
    audit = {
        "target_source": "sample_submission.csv header",
        "num_targets": len(labels),
        "num_training_rows": len(rows),
        "num_taxonomy_rows": len(taxonomy_rows),
        "split": {
            "seed": config["training"]["random_seed"],
            "validation_fraction_requested": config["training"]["validation_fraction"],
            "row_counts": dict(split_counts),
            "target_label_coverage": {
                split: len(split_labels) for split, split_labels in labels_by_split.items()
            },
            "source_recordings_in_both_partitions": len(leakage_sources),
            "source_recording_leakage_examples": leakage_sources[:20],
            **split_metadata,
        },
        "missing_train_audio_count": len(missing_audio),
        "missing_train_audio_examples": missing_audio[:20],
        "unlabeled_soundscape_count": len(unlabeled_files),
        "unlabeled_soundscape_directory_present": unlabeled_dir.is_dir(),
    }
    with paths["data_audit"].open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create BirdCLEF metadata audit artifacts.")
    parser.add_argument("--config", type=Path, default=Path("config/project_config.json"))
    arguments = parser.parse_args()
    print(json.dumps(build_artifacts(arguments.config), indent=2, sort_keys=True))
