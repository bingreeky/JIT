<div align="center">

# `harness_factory/`

**The seed bank of hand-written harnesses**

[Main README](../README.md) · [`jit/`](../jit/) · [`scripts/`](../scripts/) · [`benchmark/`](../benchmark/) · [`dataset/`](../dataset/)

</div>

---

Complete agent architectures, hand-written under one shared protocol and one
shared kernel. They are natural baselines for a JIT run: *this* is what one
fixed harness achieves, versus a harness written per task.

Harness names follow the paper's seed-bank inventory, so a row in the catalogue
below reads the same as a row in the paper's seed-harness table.

```bash
python -m scripts.run_seed_harness --bench xbench --harness plan_and_execute
```

```bash
bash scripts/run_seed_harness.sh officebench hiagent --max-samples 5
```

No meta model is involved — the harness is installed verbatim and executed.
Scoring, aggregation and resume are the same code the JIT pipeline uses, so the
two `summary.json` files are directly comparable.

**Contents** — [The catalogue](#the-catalogue) · [What is in here](#what-is-in-here) ·
[Writing your own](#writing-your-own) · [Notes on the numbers](#notes-on-the-numbers) ·
[Output](#output)

---

## The catalogue

| Harness | The idea |
|---|---|
| `plan_and_execute` | Linear ReAct. Emits an ordered 3–7 step roadmap up front, then works it. The minimal baseline. |
| `flash_searcher` | A planning call decomposes the task into a DAG of subtasks; execution follows dependency order. |
| `agentfold` | DAG planning plus AgentFold-style context folding, so long trajectories stay inside the window. |
| `resum` | Linear ReAct with ReSum-style token-budgeted summarisation of the trajectory. |
| `hiagent` | Flat ReAct, no explicit plan; leans on a structured working memory instead. |
| `memobrain` | Marker-based ReAct with a dependency-aware reasoning memory. |
| `deepagent` | Flat, plan-free tool use over a marker protocol rather than JSON tool calls. |
| `gam` | DAG planning coupled with Generative Agent Memory. |
| `roma` | Recursive decomposition — for tasks with several semi-independent goals. |
| `aggagent` | Two stages: explore broadly, then adjudicate the findings into an answer. |
| `oagent` | Runs several independent solution paths and votes. |

Full design write-ups — architecture and per-module behaviour — are in
[`descriptions/`](descriptions/).

> [!IMPORTANT]
> These same files are what the JIT meta model is shown as reference material in
> its generation prompt, so editing one changes both the documentation and the
> generation prompt.

`harnesses/` itself is reference material too, but only in the optional code
reference mode (`--harness-refs code`), which shows the meta model the full
source of three of the eleven, drawn at random, instead of the descriptions —
see [`jit/README.md`](../jit/README.md#reference-material-in-the-generation-prompt).

List what is actually installed:

```bash
python -m scripts.run_seed_harness --bench xbench --list-harnesses
```

---

## What is in here

```text
harness_factory/
├── harnesses/<name>/       the code — five files per architecture (below)
└── descriptions/<name>.md  the design write-up, verbatim what the meta model is shown
```

A run never executes these files in place: the runner copies the five files into
`scripts/workspaces/<workspace>/` and imports them from there, exactly as it does
for a harness the meta model just wrote. That is what makes the two paths
comparable. (`load_harness` will also import a harness straight out of
`harnesses/<name>/` when you load one by hand for debugging.)

Each architecture is five files plus a description:

```text
harness_factory/harnesses/plan_and_execute/
├── memory.py         MemoryStrategy      what the agent remembers, and how it is rebuilt per step
├── planning.py       PlanningStrategy    whether there is a plan, and how it is revised
├── action.py         ActionStrategy      how a step becomes a tool call or a final answer
├── tool_policy.py    ToolPolicyStrategy  which tools are offered at each step
├── prompt.yaml                           the prompts these modules render
└── description.yaml                      metadata
```

---

## Writing your own

Drop a directory with those five files under `harness_factory/harnesses/<name>/`,
exporting `MemoryStrategy`, `PlanningStrategy`, `ActionStrategy` and
`ToolPolicyStrategy` (protocols in `scripts/kernel/protocols.py`). It is
immediately runnable:

```bash
python -m scripts.run_seed_harness --bench xbench --harness my_harness
```

Each class is constructed as `Cls(prompts=<parsed prompt.yaml>)`, with a
fallback to `Cls()` for components that do not accept it.

---

## Notes on the numbers

- **No repairs.** `max_repairs` is fixed at 0. A seed harness is the
  baseline *as written*; letting a model patch it mid-run would stop the number
  from describing the harness.
- **`--rollouts 1` by default.** With the execution model at temperature 0 a
  fixed harness is close to deterministic. Raise it for a variance estimate;
  `summary.json → per_case_metrics` then reports `pass@k` and best-of-k.
- **Same evaluators, same aggregation.** Whatever grades a JIT run grades this
  one.

---

## Output

```text
runs/<bench>_<harness>_<ts>/
├── summary.json                    metrics + which harness ran
├── scores.jsonl                    one line per (case, rollout)
└── cases/<qid>/rollout_<r>/
    ├── report.json                 full trajectory, tokens, errors
    └── harness/                    the five files that ran (a copy, for the record)
```

Resume works exactly as in the JIT pipeline: re-run the same command and only
infrastructure failures are retried.
