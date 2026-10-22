#!/usr/bin/env bash
# Train the third self-training iteration in a later Kaggle session.
#
# Run this only after train_iteration_2.sh has written its COMPLETE marker.
# It generates a separate, power-transformed pseudo-label version from the
# completed iteration-2 teacher, then trains Student 3 from that teacher.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

CONFIG="${CONFIG:-$ROOT/config/project_config.json}"
RUN_ROOT="${RUN_ROOT:-$ROOT/artifacts/self_training/runs/best_supervised_teacher-7d26d7270da3-plan-8a75499211}"
ITERATION_2_DIR="${ITERATION_2_DIR:-$RUN_ROOT/iteration_2_session_2}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$ITERATION_2_DIR/best_student_iteration_2.pt}"
ITERATION_OUTPUT_DIR="${ITERATION_OUTPUT_DIR:-$RUN_ROOT/iteration_3_session_3}"
PSEUDO_ROOT="${PSEUDO_ROOT:-$ROOT/artifacts/pseudo_labels}"
SOUNDSCAPES="${SOUNDSCAPES:-/kaggle/input/competitions/birdclef-2025/train_soundscapes}"
MAPPING="${MAPPING:-$ROOT/artifacts/label_mapping.json}"
EPOCHS="${EPOCHS:-25}"
PSEUDO_BATCH_SIZE="${PSEUDO_BATCH_SIZE:-16}"
SEED="${SEED:-2027}"
POWER="1.8181818182"
COMPLETE_MARKER="$ITERATION_OUTPUT_DIR/COMPLETE"
CHECKPOINT="$ITERATION_OUTPUT_DIR/best_student_iteration_3.pt"

if [[ ! -f "$ITERATION_2_DIR/COMPLETE" ]]; then
  echo "Iteration 2 is not marked complete: $ITERATION_2_DIR/COMPLETE" >&2
  echo "Run scripts/train_iteration_2.sh successfully before starting iteration 3." >&2
  exit 1
fi

if [[ ! -f "$TEACHER_CHECKPOINT" ]]; then
  echo "Iteration-2 teacher checkpoint is missing: $TEACHER_CHECKPOINT" >&2
  exit 1
fi

if [[ -f "$COMPLETE_MARKER" ]]; then
  echo "Iteration 3 is already complete: $CHECKPOINT"
  exit 0
fi

if [[ -e "$CHECKPOINT" || -d "$ITERATION_OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite an unmarked iteration-3 output: $ITERATION_OUTPUT_DIR" >&2
  echo "Use a new ITERATION_OUTPUT_DIR for a deliberate retry; preserve this partial run for inspection." >&2
  exit 1
fi

PSEUDO_DIRECTORY="$(python3 - "$TEACHER_CHECKPOINT" "$PSEUDO_ROOT" "$POWER" <<'PY'
from pathlib import Path
import sys

from src.pseudo_labels import checkpoint_identifier

teacher = Path(sys.argv[1])
output_root = Path(sys.argv[2])
power = float(sys.argv[3])
print(output_root / f"{checkpoint_identifier(teacher)}-power-{power:.6g}")
PY
)"

if [[ -d "$PSEUDO_DIRECTORY" && ! -f "$PSEUDO_DIRECTORY/metadata.json" ]]; then
  echo "Pseudo-label directory is incomplete: $PSEUDO_DIRECTORY" >&2
  echo "Choose another PSEUDO_ROOT or remove it only after verifying it is disposable." >&2
  exit 1
fi

if [[ ! -f "$PSEUDO_DIRECTORY/metadata.json" ]]; then
  echo "Generating iteration-3 pseudo-labels from: $TEACHER_CHECKPOINT"
  python3 -m src.pseudo_labels "$TEACHER_CHECKPOINT" \
    --soundscapes "$SOUNDSCAPES" \
    --output-root "$PSEUDO_ROOT" \
    --mapping "$MAPPING" \
    --batch-size "$PSEUDO_BATCH_SIZE" \
    --exponent "$POWER"
else
  echo "Reusing verified pseudo-label version: $PSEUDO_DIRECTORY"
fi

mkdir -p "$ITERATION_OUTPUT_DIR"
echo "Training iteration 3 for $EPOCHS epochs on CUDA. Logs: $ITERATION_OUTPUT_DIR/console.log"
python3 -m src.self_train "$PSEUDO_DIRECTORY" \
  --config "$CONFIG" \
  --epochs "$EPOCHS" \
  --device cuda \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --output-dir "$ITERATION_OUTPUT_DIR" \
  --iteration 3 \
  --pseudo-label-exponent "$POWER" \
  --seed "$SEED" 2>&1 | tee "$ITERATION_OUTPUT_DIR/console.log"

if [[ ! -f "$CHECKPOINT" || ! -f "$ITERATION_OUTPUT_DIR/lineage.json" ]]; then
  echo "Iteration 3 finished without its required checkpoint or lineage file." >&2
  exit 1
fi

{
  echo "completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "iteration=3"
  echo "epochs=$EPOCHS"
  echo "teacher_checkpoint=$TEACHER_CHECKPOINT"
  echo "pseudo_label_directory=$PSEUDO_DIRECTORY"
  echo "checkpoint=$CHECKPOINT"
} > "$COMPLETE_MARKER"
echo "Iteration 3 complete: $CHECKPOINT"
