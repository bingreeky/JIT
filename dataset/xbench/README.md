# XBench Dataset Setup

All commands below assume you are running from the repository root.

## 1) Download Source
- URL: `https://huggingface.co/datasets/xbench/DeepSearch`

## 2) Download Dataset
```bash
mkdir -p .cache/datasets/xbench
hf download xbench/DeepSearch \
  --repo-type dataset \
  --include "DeepSearch.csv" \
  --local-dir .cache/datasets/xbench
```

## 3) Place Files
```bash
mkdir -p dataset/xbench
cp .cache/datasets/xbench/DeepSearch.csv dataset/xbench/DeepSearch-2505.csv

# Optional canary file if your release includes one:
# cp .cache/datasets/xbench/canary.txt dataset/xbench/canary.txt
```

## 4) Quick Verification
```bash
test -f dataset/xbench/DeepSearch-2505.csv && echo OK
# [ -f dataset/xbench/canary.txt ] && echo CANARY_OK
```

## Running

Data placement is all this file covers. To evaluate on this benchmark:

```bash
# JIT best-of-N (meta model writes the scaffold)
python -m scripts.run_jit --bench xbench --meta-base http://localhost:8000/v1

# fixed reference scaffold baseline
python -m scripts.run_reference --bench xbench --scaffold react_list
```

See [`dataset/README.md`](../README.md) for the full data table and
[`jit/README.md`](../../jit/README.md) for the pipeline.
