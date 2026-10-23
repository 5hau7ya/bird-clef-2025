#!/usr/bin/env bash
# Run one shared-window ensemble smoke test, then create a verified submission.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

common_arguments=(
  --config "$ROOT/config/project_config.json"
  --device cpu
)
if [[ -n "${BATCH_SIZE:-}" ]]; then
  common_arguments+=(--batch-size "$BATCH_SIZE")
fi

python3 -m src.submission "${common_arguments[@]}" \
  --smoke-and-submit \
  --smoke-output "/kaggle/working/submission_smoke_test.npz" \
  --output "/kaggle/working/submission.csv"
