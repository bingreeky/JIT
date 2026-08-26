#!/usr/bin/env bash
# One-command JIT best-of-N evaluation.
#
#   bash scripts/run_jit.sh <bench> [extra args passed to run_jit.py]
#
# Everything is overridable by environment variable; the defaults reproduce the
# published setup (3 harnesses per case at temperature 1, model-selected).
#
#   META_MODEL      served name of the JIT model            (default: jit)
#   META_API_BASE   its OpenAI-compatible base URL          (default: http://localhost:8000/v1)
#   META_API_KEY    its API key                             (default: EMPTY)
#   META_TOKENIZER  model dir / HF id for the logprob selector (optional)
#   EXEC_MODEL      execution model id                      (default: per config)
#   JUDGE_MODEL     judge model id                          (default: per config)
#   ROLLOUTS        candidates per case                     (default: 3)
#   OUTPUT          output directory                        (default: runs/<bench>_<ts>)
#
# Example:
#   META_API_BASE=http://127.0.0.1:8000/v1 META_MODEL=jit-ckpt70 \
#     bash scripts/run_jit.sh xbench --max-samples 5
set -euo pipefail

BENCH="${1:?usage: bash scripts/run_jit.sh <bench> [extra args]}"
shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

ARGS=(--bench "$BENCH" --rollouts "${ROLLOUTS:-3}")
[ -n "${META_TOKENIZER:-}" ] && ARGS+=(--tokenizer "$META_TOKENIZER")
[ -n "${OUTPUT:-}" ] && ARGS+=(--output "$OUTPUT")

exec "$PYTHON" -m scripts.run_jit "${ARGS[@]}" "$@"
