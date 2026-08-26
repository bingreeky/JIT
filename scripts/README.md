<div align="center">

# `scripts/`

**Shared code, and the two things you run**

[Main README](../README.md) · [`jit/`](../jit/) · [`harness_factory/`](../harness_factory/) · [`benchmark/`](../benchmark/) · [`dataset/`](../dataset/)

</div>

---

The runtime every harness executes on, the evaluation engine both runners share,
and the entry points themselves.

```text
scripts/
├── run_jit.py            ▶ JIT best-of-N: generate N harnesses, select one, execute, score
├── run_seed_harness.py   ▶ one fixed seed harness, same evaluation path
├── run_jit.sh              env-var wrapper around run_jit.py
├── run_seed_harness.sh     env-var wrapper around run_seed_harness.py
├── serve_meta_model.sh     vLLM launcher for a local JIT checkpoint
├── fetch_datasets.sh       download the large benchmark assets
├── check_datasets.py       report which datasets are present / partial / missing
├── env_config.py           .env loading and ${VAR} placeholder resolution
│
├── kernel/                 the agent runtime: harness loader, step loop, protocols, types
├── models/                 OpenAI-compatible chat client (meta, exec, judge, selector roles)
├── tools/                  every tool a harness can call, plus the registry and skills
├── eval/                   the shared evaluation engine: config, runner, metrics
└── workspaces/             transient: the harness under execution is installed here
```

**Contents** — [The two runners](#the-two-runners) · [`eval/`](#eval--the-shared-engine) ·
[`workspaces/`](#workspaces--where-a-harness-actually-runs) ·
[`kernel/`, `models/`, `tools/`](#kernel-models-tools)

---

## The two runners

```bash
# JIT: the meta model writes the framework
python -m scripts.run_jit --bench xbench \
    --meta-model jit-ckpt70 --meta-base http://localhost:8000/v1
```

```bash
# baseline: one hand-written framework
python -m scripts.run_seed_harness --bench xbench --harness plan_and_execute
```

They share `scripts/eval/` — the same per-unit runner, config wiring and score
aggregation, over the same `benchmark/config/` + `benchmark/registry.py` — so
their `summary.json` files are directly comparable.

> Details: [`jit/README.md`](../jit/README.md) and
> [`harness_factory/README.md`](../harness_factory/README.md).

---

## `eval/` — the shared engine

| Module | Owns |
|---|---|
| `config.py` | `REPO_ROOT`, `.env` loading, and wiring the three model roles into a run config |
| `runner.py` | one `(case, rollout)` unit: isolation, artifacts, resume, the thread pool |
| `metrics.py` | score extraction and aggregation across the three evaluator contracts |

A *unit* is one `(case, rollout)` pair. Running it means: build the adapter, get a
harness (generated, or installed from a preset directory), execute it against the
case, score the artifacts, and optionally repair-and-retry. Everything
JIT-specific — the meta agent, candidate reconstruction, best-of-N selection —
lives in [`jit/`](../jit/), not here.

---

## `workspaces/` — where a harness actually runs

No run executes a harness from its source directory. The runner copies the five
files into `scripts/workspaces/<workspace_name>/` and imports them from there,
whether the meta model just wrote them or they came from
`harness_factory/harnesses/`. One workspace per concurrent unit, removed when
the unit finishes; only `__init__.py` is committed.

> [!NOTE]
> That single install path is what keeps a JIT run and a reference run honest:
> both execute the same way, so a difference in score is a difference in the
> harness.

---

## `kernel/`, `models/`, `tools/`

The parts a harness is written against, and therefore the stable surface the meta
model targets:

| Surface | What it is |
|---|---|
| `kernel/protocols.py` | The four strategy protocols (`BaseMemory`, `BasePlanning`, `BaseAction`, `BaseToolPolicy`) a harness implements. |
| `kernel/types.py` | The shared data structures they pass around (`Message`, `StepRecord`, `MemoryView`, `ToolSelection`, `Directive`, …). |
| `kernel/runtime.py` | `AgentRuntime`: loads a harness and drives the think → act → observe loop under a step and model-call budget. |
| `models/` | One client for every model role; a role is just a base URL, an API key and a model id. |
| `tools/` | Web search, page crawl, code execution, and the per-benchmark action tools; `registry.py` decides which are exposed for a run. |

Generated code runs against exactly this surface, which is why the tool
descriptions rendered into the generation prompt come from the same registry that
executes them.
