# DeepPlanning (Shopping) Dataset Setup

All commands below assume you are running from the repository root.

## 1) Download Source
- URL: `https://huggingface.co/datasets/Qwen/DeepPlanning`

## 2) Download Dataset
```bash
mkdir -p .cache/datasets/deepplanning
hf download Qwen/DeepPlanning \
  --repo-type dataset \
  --include "shoppingplanning/database_zip/database_level1.tar.gz" \
  --include "shoppingplanning/database_zip/database_level2.tar.gz" \
  --include "shoppingplanning/database_zip/database_level3.tar.gz" \
  --local-dir .cache/datasets/deepplanning
```

## 3) Place Files
```bash
mkdir -p dataset/deepplanning_shopping/database_zip
cp .cache/datasets/deepplanning/shoppingplanning/database_zip/database_level1.tar.gz dataset/deepplanning_shopping/database_zip/
cp .cache/datasets/deepplanning/shoppingplanning/database_zip/database_level2.tar.gz dataset/deepplanning_shopping/database_zip/
cp .cache/datasets/deepplanning/shoppingplanning/database_zip/database_level3.tar.gz dataset/deepplanning_shopping/database_zip/

tar -xzf dataset/deepplanning_shopping/database_zip/database_level1.tar.gz -C dataset/deepplanning_shopping
tar -xzf dataset/deepplanning_shopping/database_zip/database_level2.tar.gz -C dataset/deepplanning_shopping
tar -xzf dataset/deepplanning_shopping/database_zip/database_level3.tar.gz -C dataset/deepplanning_shopping
```

## 4) Quick Verification
```bash
test -d dataset/deepplanning_shopping/database_level1 && echo OK
test -d dataset/deepplanning_shopping/database_level2 && echo OK
test -d dataset/deepplanning_shopping/database_level3 && echo OK
```

## Running

Data placement is all this file covers. To evaluate on this benchmark:

```bash
# JIT best-of-N (meta model writes the scaffold)
python -m scripts.run_jit --bench shopping --meta-base http://localhost:8000/v1

# fixed reference scaffold baseline
python -m scripts.run_reference --bench shopping --scaffold react_list
```

See [`dataset/README.md`](../README.md) for the full data table and
[`jit/README.md`](../../jit/README.md) for the pipeline.
