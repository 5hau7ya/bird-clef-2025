#!/usr/bin/env bash
# Evaluate the three selected students on the identical held-out split.
# Framewise outputs are cached beside their checkpoints; reruns reuse them.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

arguments=(
  --config "$ROOT/config/project_config.json"
  --output-dir "$ROOT/artifacts/ensemble_evaluation"
  --device "${DEVICE:-cuda}"
)
if [[ -n "${BATCH_SIZE:-}" ]]; then
  arguments+=(--batch-size "$BATCH_SIZE")
fi
python3 -m src.ensemble_evaluation "${arguments[@]}"
