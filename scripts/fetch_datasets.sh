#!/usr/bin/env bash
# Fetch the benchmark assets too large to commit.
#
#   bash scripts/fetch_datasets.sh travel        # one benchmark
#   bash scripts/fetch_datasets.sh travel agentif  # several
#   bash scripts/fetch_datasets.sh all             # everything (~1 GB)
#
# Idempotent: anything already in place is skipped. Run from the repo root.
# Needs the Hugging Face CLI:  pip install -U "huggingface_hub[cli]"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DATA=dataset
CACHE=.cache/datasets

command -v hf >/dev/null 2>&1 || {
  echo "error: the 'hf' CLI is not on PATH."
  echo "       pip install -U \"huggingface_hub[cli]\""
  exit 1
}

have() { [ -e "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

fetch_travel() {
  if have "$DATA/deepplanning_travel/database/database_zh"; then
    echo "[travel] database already present -- skipping"; return
  fi
  echo "[travel] downloading DeepPlanning travel database (748 MB) ..."
  mkdir -p "$CACHE/deepplanning" "$DATA/deepplanning_travel/database"
  hf download Qwen/DeepPlanning --repo-type dataset \
    --include "travelplanning/database/database_zh.zip" \
    --include "travelplanning/database/database_en.zip" \
    --local-dir "$CACHE/deepplanning"
  cp "$CACHE"/deepplanning/travelplanning/database/database_*.zip \
     "$DATA/deepplanning_travel/database/"
  unzip -oq "$DATA/deepplanning_travel/database/database_zh.zip" \
     -d "$DATA/deepplanning_travel/database"
  unzip -oq "$DATA/deepplanning_travel/database/database_en.zip" \
     -d "$DATA/deepplanning_travel/database"
  echo "[travel] done"
}

fetch_shopping() {
  if have "$DATA/deepplanning_shopping/database_level1"; then
    echo "[shopping] databases already present -- skipping"; return
  fi
  echo "[shopping] downloading DeepPlanning shopping databases ..."
  mkdir -p "$CACHE/deepplanning" "$DATA/deepplanning_shopping/database_zip"
  hf download Qwen/DeepPlanning --repo-type dataset \
    --include "shoppingplanning/database_zip/database_level1.tar.gz" \
    --include "shoppingplanning/database_zip/database_level2.tar.gz" \
    --include "shoppingplanning/database_zip/database_level3.tar.gz" \
    --local-dir "$CACHE/deepplanning"
  for level in 1 2 3; do
    tar -xzf "$CACHE/deepplanning/shoppingplanning/database_zip/database_level${level}.tar.gz" \
        -C "$DATA/deepplanning_shopping"
  done
  echo "[shopping] done"
}

fetch_agentif() {
  if have "$DATA/agentif_oneday/Attachments"; then
    echo "[agentif] attachments already present -- skipping"; return
  fi
  echo "[agentif] downloading AgentIF-OneDay attachments (250 MB) ..."
  hf download xbench/AgentIF-OneDay --repo-type dataset \
    --local-dir "$DATA/agentif_oneday"
  echo "[agentif] done"
}

[ "$#" -gt 0 ] || { echo "usage: bash scripts/fetch_datasets.sh {travel|shopping|agentif|all}"; exit 1; }

TARGETS=("$@")
[ "${1}" = "all" ] && TARGETS=(travel shopping agentif)

for target in "${TARGETS[@]}"; do
  case "$target" in
    travel)   fetch_travel ;;
    shopping) fetch_shopping ;;
    agentif)  fetch_agentif ;;
    *) echo "unknown target '$target' (travel|shopping|agentif|all)"; exit 1 ;;
  esac
done

echo
python scripts/check_datasets.py
