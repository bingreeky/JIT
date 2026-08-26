<div align="center">

# `benchmark/`

**One interface per dataset**

[Main README](../README.md) · [`jit/`](../jit/) · [`scripts/`](../scripts/) · [`harness_factory/`](../harness_factory/) · [`dataset/`](../dataset/)

</div>

---

Everything a benchmark needs in order to be runnable: how to load its cases, how
to phrase one as a task, how to score the result, and what the execution model
and judge should be. The data itself is next door in [`dataset/`](../dataset/).

```text
benchmark/
├── registry.py             the suite: which benchmarks exist, and their run-level facts
├── config/<name>.yaml      one per benchmark: exec model, tools, step budget, judge
└── adapter/
    ├── base.py             the BenchmarkAdapter protocol every adapter implements
    ├── judge.py            the shared LLM-judge helper
    ├── <name>.py           the adapter: load_dataset / format_task / evaluate
    └── <name>_eval/        that benchmark's evaluator, vendored from upstream
```

`config/` is what you edit to change how a benchmark runs; `adapter/` is what you
edit to change what it means to run it. `registry.py` sits between the two —
see below.

Seven benchmarks ship: `xbench`, `deepsearchqa`, `agentif`, `officebench`,
`odyssey`, `shopping`, `travel`. The keys, case counts and download
requirements are tabulated in [`dataset/README.md`](../dataset/README.md).

**Contents** — [The two halves of a configuration](#the-two-halves-of-a-benchmarks-configuration) ·
[The adapter contract](#the-adapter-contract) · [Adding a benchmark](#adding-a-benchmark) ·
[Vendored evaluators](#vendored-evaluators)

---

## The two halves of a benchmark's configuration

| | Owns |
|---|---|
| **The YAML**<br>`config/<name>.yaml` | What a *run* needs: the execution model, the tool set, `max_steps`, the judge, and `benchmark.dataset_path`. It is what `--config` overrides point at. |
| **The registry entry**<br>`registry.py` | What the *runner* needs and the YAML cannot express: which of the three metric profiles the evaluator speaks, which item field to bucket results by, the full-dataset case count (so a truncated run is visible as truncated), and any environment variables the vendored harness needs. This is also where the short CLI key (`--bench travel`) is defined. |

> [!NOTE]
> The meta model is deliberately absent from both: it is CLI-only, because it
> needs a completely different `max_tokens` from the execution model and
> conflating the two silently truncates generated harnesses.

---

## The adapter contract

```python
class BenchmarkAdapter:
    def load_dataset(self, path: str) -> list[dict]: ...
    def format_task(self, item: dict) -> str: ...          # the prompt the harness sees
    def evaluate(self, item: dict, answer: str, **kw) -> dict: ...
```

`evaluate` returns the payload one of the three metric profiles in
[`scripts/eval/metrics.py`](../scripts/eval/metrics.py) knows how to flatten
(`generic`, `travel`, `shopping`). An adapter that needs a scratch directory per
case takes a `workspace_base` keyword — the runner passes one per `(case,
rollout)` so concurrent units cannot see each other's files.

Adapters are imported as `benchmark.adapter.<name>`, constructed by name
through `_load_benchmark_adapter` in
[`jit/harness_ops.py`](../jit/harness_ops.py) — the single lookup table every
entry point goes through.

---

## Adding a benchmark

1. Put the data under `dataset/<name>/` (or document the download in its README).
2. Write `benchmark/adapter/<name>.py` implementing the three methods above.
3. Register the class in `_load_benchmark_adapter` ([`jit/harness_ops.py`](../jit/harness_ops.py)).
4. Add `benchmark/config/<name>.yaml` — copy the closest existing one.
5. Add one `BenchmarkSpec` to [`benchmark/registry.py`](registry.py).

If its evaluator returns a shape none of the three metric profiles covers, add a
profile in `scripts/eval/metrics.py`. Nothing else has to change: both runners
pick the new benchmark up from the registry.

---

## Vendored evaluators

`adapter/<name>_eval/` directories are upstream evaluation code, kept in-tree so
a score is reproducible without a network fetch and without version drift. They
are edited only where they hardcoded paths or assumed a Docker sandbox.

> [!TIP]
> Several need file-format libraries beyond the base install (`icalendar`,
> `PyMuPDF`, `python-docx`, …) — see *Extra Python dependencies* in
> [`dataset/README.md`](../dataset/README.md).
