# DeepPlanning (Travel) Dataset Setup

All commands below assume you are running from the repository root.

## 1) Download Source
- URL: `https://huggingface.co/datasets/Qwen/DeepPlanning`

## 2) Download Dataset
```bash
mkdir -p .cache/datasets/deepplanning
hf download Qwen/DeepPlanning \
  --repo-type dataset \
  --include "travelplanning/database/database_zh.zip" \
  --include "travelplanning/database/database_en.zip" \
  --local-dir .cache/datasets/deepplanning
```

## 3) Place Files
```bash
mkdir -p dataset/deepplanning_travel/database
cp .cache/datasets/deepplanning/travelplanning/database/database_zh.zip dataset/deepplanning_travel/database/
cp .cache/datasets/deepplanning/travelplanning/database/database_en.zip dataset/deepplanning_travel/database/

unzip -o dataset/deepplanning_travel/database/database_zh.zip -d dataset/deepplanning_travel/database
unzip -o dataset/deepplanning_travel/database/database_en.zip -d dataset/deepplanning_travel/database
```

## 4) Quick Verification
```bash
test -f dataset/deepplanning_travel/database/database_zh.zip && echo OK
test -f dataset/deepplanning_travel/database/database_en.zip && echo OK
```

## Running

Data placement is all this file covers. To evaluate on this benchmark:

```bash
# JIT best-of-N (meta model writes the scaffold)
python -m scripts.run_jit --bench travel --meta-base http://localhost:8000/v1

# fixed reference scaffold baseline
python -m scripts.run_reference --bench travel --scaffold react_list
```

See [`dataset/README.md`](../README.md) for the full data table and
[`jit/README.md`](../../jit/README.md) for the pipeline.
