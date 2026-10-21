"""Repeatable three-stage teacher–student self-training orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from src.pseudo_labels import checkpoint_identifier, generate_pseudo_labels
from src.self_train import train_one_student_iteration


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _pseudo_version_directory(output_root: Path, teacher: Path, exponent: float) -> Path:
    return output_root / f"{checkpoint_identifier(teacher)}-power-{exponent:.6g}"


def _valid_existing_pseudo_labels(directory: Path, teacher: Path, exponent: float) -> bool:
    metadata_path, values_path = directory / "metadata.json", directory / "five_second_scores.npz"
    if not metadata_path.is_file() or not values_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        metadata.get("teacher_checkpoint", {}).get("identifier") == checkpoint_identifier(teacher)
        and metadata.get("pseudo_label_configuration", {}).get("probability_power_exponent") == exponent
    )


def _checkpoint_metric(path: Path) -> float:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return float(payload["validation_macro_auc"])


def run_self_training_pipeline(
    config_path: Path,
    initial_teacher: Path,
    device: str | None = None,
    epochs_override: int | None = None,
    resume: bool = True,
) -> Path:
    """Generate labels and train up to three students with auditable lineage."""
    root = config_path.parent.parent
    initial_teacher = initial_teacher if initial_teacher.is_absolute() else root / initial_teacher
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = {name: root / value for name, value in config["paths"].items()}
    self_training = config["self_training"]
    plan = self_training["iteration_plan"][: self_training["iterations"]]
    if not initial_teacher.is_file():
        raise FileNotFoundError(initial_teacher)
    plan_digest = hashlib.sha256(_canonical_json(plan).encode()).hexdigest()[:10]
    run_directory = root / self_training["pipeline_runs_dir"] / f"{checkpoint_identifier(initial_teacher)}-plan-{plan_digest}"
    manifest_path = run_directory / "pipeline_lineage.json"
    if run_directory.exists() and not resume:
        raise FileExistsError(f"Pipeline run exists: {run_directory}; use --resume or a new teacher checkpoint.")
    run_directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {
        "initial_teacher": str(initial_teacher),
        "initial_teacher_identifier": checkpoint_identifier(initial_teacher),
        "iteration_plan": plan,
        "stages": [],
    }
    teacher = initial_teacher
    candidates = []
    for stage in plan:
        iteration, exponent = int(stage["iteration"]), float(stage["pseudo_label_exponent"])
        seed = int(config["training"]["random_seed"]) + iteration - 1
        pseudo_directory = _pseudo_version_directory(paths["pseudo_labels_root"], teacher, exponent)
        if not _valid_existing_pseudo_labels(pseudo_directory, teacher, exponent):
            pseudo_directory = generate_pseudo_labels(
                teacher, paths["unlabeled_soundscapes_dir"], paths["pseudo_labels_root"], paths["label_mapping"],
                config["inference"]["batch_size"], exponent,
            )
        stage_directory = run_directory / f"iteration_{iteration}"
        student_checkpoint = stage_directory / f"best_student_iteration_{iteration}.pt"
        if not (resume and student_checkpoint.is_file()):
            student_checkpoint = train_one_student_iteration(
                config_path, pseudo_directory, epochs_override, device, teacher, stage_directory,
                iteration, exponent, seed,
            )
        metric = _checkpoint_metric(student_checkpoint)
        metadata = json.loads((pseudo_directory / "metadata.json").read_text(encoding="utf-8"))
        stage_record = {
            "iteration": iteration,
            "teacher_checkpoint": str(teacher),
            "teacher_identifier": checkpoint_identifier(teacher),
            "pseudo_label_directory": str(pseudo_directory),
            "pseudo_label_version": metadata["version"],
            "pseudo_label_exponent": exponent,
            "training_seed": seed,
            "training_data_version": metadata["version"],
            "student_checkpoint": str(student_checkpoint),
            "validation_macro_auc": metric,
        }
        manifest["stages"] = [entry for entry in manifest["stages"] if entry["iteration"] != iteration] + [stage_record]
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        candidates.append(stage_record)
        teacher = student_checkpoint
    selected_count = int(config["ensemble"]["final_ensemble_checkpoints"])
    selected = sorted(candidates, key=lambda item: item["validation_macro_auc"], reverse=True)[:selected_count]
    (run_directory / "selected_ensemble.json").write_text(json.dumps({
        "selection_criterion": "highest fixed-validation macro ROC-AUC among student best checkpoints",
        "requested_count": selected_count,
        "selected_checkpoints": selected,
    }, indent=2), encoding="utf-8")
    return run_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run up to three auditable teacher-student iterations.")
    parser.add_argument("--config", type=Path, default=Path("config/project_config.json"))
    parser.add_argument(
        "--teacher", type=Path, default=Path("artifacts/supervised_teacher/best_supervised_teacher.pt")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0", "mps"))
    parser.add_argument("--epochs", type=int, help="Override student epochs for every pipeline stage.")
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()
    print(run_self_training_pipeline(arguments.config, arguments.teacher, arguments.device, arguments.epochs, not arguments.no_resume))
