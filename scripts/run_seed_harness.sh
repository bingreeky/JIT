#!/usr/bin/env bash
# One-command seed-harness evaluation.
#
#   bash scripts/run_seed_harness.sh <bench> [harness] [extra args]
#
# Defaults to plan_and_execute, the minimal ReAct baseline. List every harness with:
#   python -m scripts.run_seed_harness --bench xbench --list-harnesses
#
# Environment:
#   EXEC_MODEL   execution model id   (default: per config)
#   JUDGE_MODEL  judge model id       (default: per config)
#   OUTPUT       output directory     (default: runs/<bench>_<harness>_<ts>)
#
# Example:
#   bash scripts/run_seed_harness.sh officebench hiagent --max-samples 5
set -euo pipefail

BENCH="${1:?usage: bash scripts/run_seed_harness.sh <bench> [harness] [extra args]}"
HARNESS="${2:-plan_and_execute}"
shift || true
shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

ARGS=(--bench "$BENCH" --harness "$HARNESS")
[ -n "${OUTPUT:-}" ] && ARGS+=(--output "$OUTPUT")

exec "$PYTHON" -m scripts.run_seed_harness "${ARGS[@]}" "$@"
