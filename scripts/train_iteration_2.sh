#!/usr/bin/env bash
# Train a fresh, complete second self-training iteration in one Kaggle session.
#
# This deliberately does not reuse the interrupted `iteration_2` directory:
# its best checkpoint was written after epoch 1, not after a complete run.
# Override RUN_ROOT or ITERATION_OUTPUT_DIR only when intentionally starting a
# different retry.  A completed run writes COMPLETE; an unmarked directory is
# treated as partial and is never overwritten.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

CONFIG="${CONFIG:-$ROOT/config/project_config.json}"
RUN_ROOT="${RUN_ROOT:-$ROOT/artifacts/self_training/runs/best_supervised_teacher-7d26d7270da3-plan-8a75499211}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$RUN_ROOT/iteration_1/best_student_iteration_1.pt}"
ITERATION_OUTPUT_DIR="${ITERATION_OUTPUT_DIR:-$RUN_ROOT/iteration_2_session_2}"
PSEUDO_ROOT="${PSEUDO_ROOT:-$ROOT/artifacts/pseudo_labels}"
SOUNDSCAPES="${SOUNDSCAPES:-/kaggle/input/competitions/birdclef-2025/train_soundscapes}"
MAPPING="${MAPPING:-$ROOT/artifacts/label_mapping.json}"
EPOCHS="${EPOCHS:-25}"
PSEUDO_BATCH_SIZE="${PSEUDO_BATCH_SIZE:-16}"
SEED="${SEED:-2026}"
POWER="1.5384615385"
COMPLETE_MARKER="$ITERATION_OUTPUT_DIR/COMPLETE"
CHECKPOINT="$ITERATION_OUTPUT_DIR/best_student_iteration_2.pt"

if [[ ! -f "$TEACHER_CHECKPOINT" ]]; then
  echo "Iteration-1 teacher checkpoint is missing: $TEACHER_CHECKPOINT" >&2
  exit 1
fi

if [[ -f "$COMPLETE_MARKER" ]]; then
  echo "Iteration 2 is already complete: $CHECKPOINT"
  exit 0
fi

if [[ -e "$CHECKPOINT" || -d "$ITERATION_OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite an unmarked iteration-2 output: $ITERATION_OUTPUT_DIR" >&2
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
  echo "Generating iteration-2 pseudo-labels from: $TEACHER_CHECKPOINT"
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
echo "Training iteration 2 for $EPOCHS epochs on CUDA. Logs: $ITERATION_OUTPUT_DIR/console.log"
python3 -m src.self_train "$PSEUDO_DIRECTORY" \
  --config "$CONFIG" \
  --epochs "$EPOCHS" \
  --device cuda \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --output-dir "$ITERATION_OUTPUT_DIR" \
  --iteration 2 \
  --pseudo-label-exponent "$POWER" \
  --seed "$SEED" 2>&1 | tee "$ITERATION_OUTPUT_DIR/console.log"

if [[ ! -f "$CHECKPOINT" || ! -f "$ITERATION_OUTPUT_DIR/lineage.json" ]]; then
  echo "Iteration 2 finished without its required checkpoint or lineage file." >&2
  exit 1
fi

{
  echo "completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "iteration=2"
  echo "epochs=$EPOCHS"
  echo "teacher_checkpoint=$TEACHER_CHECKPOINT"
  echo "pseudo_label_directory=$PSEUDO_DIRECTORY"
  echo "checkpoint=$CHECKPOINT"
} > "$COMPLETE_MARKER"
echo "Iteration 2 complete: $CHECKPOINT"

