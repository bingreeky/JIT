# OfficeBench Dataset Setup

All commands below assume you are in the repository root.

## Preparing the data

Nothing to download: `dataset/officebench/` ships complete, so you can run
tasks straight from the checkout.

## Running

Data placement is all this file covers. To evaluate on this benchmark:

```bash
# JIT best-of-N (meta model writes the scaffold)
python -m scripts.run_jit --bench officebench --meta-base http://localhost:8000/v1

# fixed reference scaffold baseline
python -m scripts.run_reference --bench officebench --scaffold react_list
```

See [`dataset/README.md`](../README.md) for the full data table and
[`jit/README.md`](../../jit/README.md) for the pipeline.
