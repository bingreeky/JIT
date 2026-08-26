# DeepSearchQA Dataset Setup

All commands below assume you are running from the repository root.

## 1) Download Source
- URL: `https://huggingface.co/datasets/google/deepsearchqa`

## 2) Download Dataset
```bash
mkdir -p .cache/datasets/deepsearchqa
hf download google/deepsearchqa \
  --repo-type dataset \
  --include "*DSQA-full.csv" \
  --local-dir .cache/datasets/deepsearchqa
```

## 3) Place Files
```bash
mkdir -p dataset/deepsearchqa
find .cache/datasets/deepsearchqa -name '*DSQA-full.csv' -print -quit | xargs -I{} cp {} dataset/deepsearchqa/DSQA-full.csv
```

## 4) Quick Verification
```bash
test -f dataset/deepsearchqa/DSQA-full.csv && echo OK
head -n 2 dataset/deepsearchqa/DSQA-full.csv
```

## Running

Data placement is all this file covers. To evaluate on this benchmark:

```bash
# JIT best-of-N (meta model writes the scaffold)
python -m scripts.run_jit --bench deepsearchqa --meta-base http://localhost:8000/v1

# fixed reference scaffold baseline
python -m scripts.run_reference --bench deepsearchqa --scaffold react_list
```

See [`dataset/README.md`](../README.md) for the full data table and
[`jit/README.md`](../../jit/README.md) for the pipeline.
