<div align="center">

# `jit/`

**The meta agent, and the best-of-N pipeline around it**

[Main README](../README.md) · [`scripts/`](../scripts/) · [`harness_factory/`](../harness_factory/) · [`benchmark/`](../benchmark/) · [`dataset/`](../dataset/)

</div>

---

This is the core of JIT: the code that makes a model write an agent framework for
the task in front of it, and picks which of N attempts to run.

```text
jit/
├── prompt.yaml      the generation / repair / review prompts
├── meta_agent.py    the loop: generate → validate → repair, with a review panel
├── harness_ops.py   parse the model's five tagged blocks, render tool info
├── schemas.py       the request / result / validation-record dataclasses
├── candidates.py    rebuild each generated harness as a (prompt, completion) pair
└── selector.py      best-of-N: logprob or judge
```

The runner that drives all of it is [`scripts/run_jit.py`](../scripts/run_jit.py);
the shared evaluation engine it runs on top of is
[`scripts/eval/`](../scripts/eval/).

**Contents** — [Running it](#running-it) · [The pipeline](#the-pipeline) ·
[Reference material](#reference-material-in-the-generation-prompt) ·
[Serving the meta model](#serving-the-meta-model) · [Output](#output) ·
[Resuming](#resuming) · [Useful flags](#useful-flags)

---

## Running it

The meta model writes the agent framework; one command takes it from there.

```bash
python -m scripts.run_jit --bench xbench \
    --meta-model jit-ckpt70 --meta-base http://localhost:8000/v1
```

Or the shell wrapper, which reads the same settings from the environment:

```bash
META_API_BASE=http://127.0.0.1:8000/v1 META_MODEL=jit-ckpt70 \
  bash scripts/run_jit.sh xbench
```

---

## The pipeline

### 1 · Generate

> `--rollouts 3` harnesses per case, at `--meta-temperature 1.0`

The meta model is shown the task, the available tools, and reference material
(see [*Reference material*](#reference-material-in-the-generation-prompt) below),
and emits five files inside tagged blocks:

| Tagged block | Becomes |
|---|---|
| `<<<PYTHON_MEMORY>>> … <<<END_PYTHON_MEMORY>>>` | `memory.py` |
| `<<<PYTHON_PLANNING>>> … <<<END_PYTHON_PLANNING>>>` | `planning.py` |
| `<<<PYTHON_ACTION>>> … <<<END_PYTHON_ACTION>>>` | `action.py` |
| `<<<PYTHON_TOOL_POLICY>>> … <<<END_PYTHON_TOOL_POLICY>>>` | `tool_policy.py` |
| `<<<YAML>>> … <<<END_YAML>>>` | `prompt.yaml` |

Nothing is executed and nothing is scored in this phase — it only produces
candidates. Each lands in
`generate/cases/<question_id>/rollout_<r>/{report.json,harness/}`.

> [!IMPORTANT]
> Temperature matters: at 0 the N rollouts are the same harness and best-of-N is
> a no-op. **1.0 is the default and the published setting.**

### 2 · Select — the model picks its own favourite


| `--selector` | How | Requires |
|---|---|---|
| `logprob` | Teacher-force each candidate through the meta model; rank by the summed log-probability of its completion tokens. | `prompt_logprobs` on `/v1/completions` (vLLM ≥ 0.9, SGLang) **and** `--tokenizer` |
| `judge` | Show the task and the N designs to a model; it replies `{"best": i, "reason": …}`. | plain `/v1/chat/completions` — works with any provider |
| `auto` *(default)* | Probe for `prompt_logprobs`; use `logprob` if available and a tokenizer was given, else `judge`. Prints which route it took. | — |

Both apply the same free static gate first: prefer candidates that actually
emitted all five blocks. A case with no scorable candidate falls back to rollout
0 and is counted as a fallback in the log — it is never dropped, because a
missing case would silently shrink the eval set.

`select/selection.json` records the pick, the rule that produced it, and the
per-candidate scores.

```bash
python -m scripts.run_jit --bench xbench \
    --meta-model jit-ckpt70 --meta-base http://localhost:8000/v1 \
    --selector logprob --tokenizer /path/to/checkpoint-70-merged
```

> [!TIP]
> Scoring is prefill-heavy. Either give it its own endpoint
> (`--selector-base http://host:8001/v1`) so generation throughput is unaffected,
> or cap the prefill chunk on the shared server — `serve_meta_model.sh` sets
> `--max-num-batched-tokens 4096` for exactly this reason.

### 3 · Execute — run the selected harness, score it

The chosen harness is installed verbatim and executed against its case by the
execution model, then scored by the benchmark's evaluator.

The meta model is **not** called again unless the run raises. On an exception,
`--max-repairs` regeneration attempts are allowed; a harness that merely scores
*low* is never repaired, since that would be optimising against the benchmark.

---

## Reference material in the generation prompt

Two renderings of the prompt's `### 3.` section exist.

### `desc` (default) — descriptions, no code

Natural-language design descriptions of all 11 seed harnesses, **no source
code**. The catalogue is read from
[`harness_factory/descriptions/`](../harness_factory/descriptions/) (one
`.md` per harness); point elsewhere with `META_REF_DESC_DIR`. ~13k tokens.

### `code` — three seed harnesses, full source

Three of the same eleven harnesses, drawn at random per (case, rollout), shown
as complete source: the `description.yaml` prose plus all five files under the
same `<<<TAG>>>` protocol the model must emit. 

```bash
python -m scripts.run_jit --bench xbench --harness-refs code   # --harness-refs-k 3
```

In `code` mode the fit-to-task transforms are deliberately **not** applied: the
prompt keeps `prompt.yaml`'s own `### 3. Agent harness examples:` header and its
originality/innovation wording, which is what pairs with code references.

> [!WARNING]
> Two costs to know about:
>
> - **~25k–46k tokens** of reference material instead of ~13k (it depends which
>   three are drawn). Still inside the 163840-token window `--max-model-len`
>   serves, alongside a 64k generation, but the margin is thinner.


### Choosing the mode

CLI beats YAML beats env, and the default is `desc`:

```bash
# per run
--harness-refs code --harness-refs-k 3
```

```yaml
# in the benchmark YAML
meta_references:
  mode: code   # desc | code
  k: 3
```

```bash
# when driving jit/ as a library
JIT_META_REF_CODE=1
JIT_META_REF_CODE_K=3
JIT_META_REF_SEED=42       # fix the draw; unset = a fresh draw per rollout
```

The resolved mode is echoed in the run banner, recorded in `summary.json`
(`harness_refs`), and each case's `report.json` carries `reference_mode` plus
the `reference_harnesses` that were drawn.

---

## Serving the meta model

Any OpenAI-compatible endpoint works. For a local checkpoint:

```bash
MODEL=/path/to/checkpoint-70-merged TP=4 bash scripts/serve_meta_model.sh
# then: --meta-base http://<host>:8000/v1 --meta-model jit
```

Two settings in that script are not cosmetic:

- **`--max-model-len 163840`.** The prompt carries the reference catalogue, the
  full tool descriptions and the task; the response is a whole 5-file harness.
  A 32k window truncates harnesses, which surfaces as "did not output all
  required files". `--harness-refs code` roughly triples the reference block
  (~46k tokens worst case) and still fits.
- **No `--reasoning-parser qwen3`.** It splits the assistant turn at the *first*
  `</think>`, and a generated harness legitimately contains a literal
  `</think>` (e.g. `re.sub(r'<think>.*?</think>', '', x)`, used to strip the
  executor's own reasoning). The parser then splits inside the generated code
  and drops `memory.py` / `planning.py` into `reasoning_content`.

---

## Output

```text
runs/<bench>_jit_bo3_<ts>/
├── summary.json                    headline metrics + how the run was configured
├── generate/
│   ├── cases/<qid>/rollout_<r>/
│   │   ├── report.json             prompts, responses, token counts
│   │   └── harness/                the five generated files
│   └── scores.jsonl, summary.json
├── select/
│   ├── candidates.jsonl            per-candidate scoring record
│   └── selection.json              the pick per case, the rule, and the scores
└── execute/
    ├── cases/<qid>/rollout_0/      report + the harness that actually ran
    └── scores.jsonl, summary.json  the numbers you report
```

> [!TIP]
> Read `summary.json → result.per_rollout_metrics.overall` for the headline, and
> `selected_rollout_distribution` to sanity-check the selector: a distribution
> pinned to `{"0": N}` usually means selection silently fell back — check the
> `fallbacks` count in the log.

---

## Resuming

Every phase is resume-idempotent. Re-run the identical command and it reloads
finished units and re-runs only what is missing or failed for **infrastructure**
reasons (dead endpoint, network flake, full disk — the marker list is in
`scripts/eval/metrics.py`). 

To reuse earlier phases explicitly:

```bash
# same candidates, different selector
python -m scripts.run_jit --bench xbench --output runs/prev --skip-generate
```

```bash
# same candidates and same picks, different execution model
python -m scripts.run_jit --bench xbench --output runs/prev \
    --skip-generate --skip-select --exec-model some-other-model
```

The second form is the controlled comparison: identical harnesses, one variable
changed.

---

## Useful flags

| Flag | Why |
|---|---|
| `--max-samples 5` | Smoke-test a benchmark before committing to the full sweep. |
| `--cases id1,id2` | Re-run specific cases. |
| `--workers-gen` / `--workers-exec` | Generation queues against the meta server's `max_num_seqs`; execution is one OS-level workspace per case. Tune them separately. |
| `--attempts` | Passes to retry while infrastructure failures remain (default 3). |
| `--max-steps` | Override the benchmark's step budget. |
| `--no-resume` | Ignore existing artifacts and redo everything. |

Full list: `python -m scripts.run_jit --help`.
