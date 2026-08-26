# AgentIF-OneDay Dataset Setup

All commands below assume you are running from the repository root.

## 1) Download Source
- URL: `https://huggingface.co/datasets/xbench/AgentIF-OneDay`

## 2) Download Dataset
```bash
mkdir -p .cache/datasets/agentif_oneday
hf download xbench/AgentIF-OneDay \
  --repo-type dataset \
  --local-dir .cache/datasets/agentif_oneday
```

## 3) Place Files
```bash
mkdir -p dataset/agentif_oneday
rsync -av --delete .cache/datasets/agentif_oneday/ dataset/agentif_oneday/
```

## 4) Quick Verification
```bash
test -f dataset/agentif_oneday/data.jsonl && echo OK
test -d dataset/agentif_oneday/Attachments/Questions && echo OK
test -d dataset/agentif_oneday/Attachments/Reference_answer && echo OK
```

## Running

Data placement is all this file covers. To evaluate on this benchmark:

```bash
# JIT best-of-N (meta model writes the scaffold)
python -m scripts.run_jit --bench agentif --meta-base http://localhost:8000/v1

# fixed reference scaffold baseline
python -m scripts.run_reference --bench agentif --scaffold react_list
```

See [`dataset/README.md`](../README.md) for the full data table and
[`jit/README.md`](../../jit/README.md) for the pipeline.
