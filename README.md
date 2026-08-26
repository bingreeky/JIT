<div align="center">

<img src="assets/jit-agent-logo.png" alt="JIT-Agent" width="380">

**Scaling Harness Intelligence via *Just-in-Time* Harness Evolution**

[![GitHub](https://img.shields.io/badge/GitHub-bingreeky%2FJIT-181717?style=flat-square&logo=github)](https://github.com/bingreeky/JIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-JIT--Agent-ffbd45?style=flat-square)](https://huggingface.co/JIT-Agent)
[![License](https://img.shields.io/badge/License-see%20LICENSE-blue?style=flat-square)](LICENSE)

</div>


## What is JIT-Agent?

**JIT-Agent is a compact meta-agent that writes your agent harness on the fly.** Instead of
precompiling one general-purpose scaffold and hoping it transfers, JIT-Agent takes a task
spec, a protocol, a tool/skill registry, and a few retrieved prior harnesses, and emits an
**executable, task-specific harness that wraps any off-the-shelf agentic LLM** —
***Model-as-a-Harness***.

<div align="center">
<img src="assets/method_overview.png" alt="Overview of JIT-Agent" width="100%">
</div>

<div align="center"><sub><b>Overview of JIT-Agent.</b> Given a task, JIT-Agent composes a
problem-specific agent harness by selecting and instantiating four modules: memory,
planning, action, and capability. Different task structures therefore induce distinct
executable protocols and state organizations.</sub></div>

Every harness is factored into **four modules — memory, planning, action, capability
orchestration** — implemented against the shared interfaces in
[**HarnessFactory**](harness_factory/), so generation means **emitting structured code
rather than free-form agent programs**. As traces and feedback come back, JIT-Agent revises
the harness and updates the archive: **harnesses keep improving at test time while the
generator itself stays frozen.**

<div align="center">
<img src="assets/leaderboard.png" alt="JIT-Agent leaderboard across four representative agent benchmarks" width="100%">
</div>

**Results.** The resulting **JIT-Agent-27B** lifts a wide range of backbone agents across
deep research, daily work, planning, and workspace tasks.

> **Building the scaffold turns out to be a trainable, transferable axis of agent
> intelligence — orthogonal to scaling the base model.**

---

## Repository layout

| Directory | What it holds |
|---|---|
| [`jit/`](jit/) | The meta agent: generation / repair prompts, best-of-N selection |
| [`scripts/`](scripts/) | The agent kernel, tools, models, evaluation engine, and the two runners |
| [`harness_factory/`](harness_factory/) | Hand-written harness implementations and their design write-ups |
| [`benchmark/`](benchmark/) | One adapter, config and evaluator per benchmark |
| [`dataset/`](dataset/) | The benchmark data itself |

Each directory has its own README with the details.

## Setup

**1. Environment** (Python 3.11)

```bash
conda env create -f environment.yml && conda activate jit
```

or, in an existing environment: `pip install -r requirements.txt`. Serving a local meta
model (vLLM/SGLang + torch) is deliberately not included — the pipeline only ever talks
HTTP to it.

**2. Credentials**

```bash
cp .env.example .env   # then fill it in
```

Anything already exported in the shell wins over `.env`, and every model role can also be
overridden per run on the command line.

| Group | Keys | Used for |
|---|---|---|
| Execution model | `OPENAI_API_BASE`, `OPENAI_API_KEY`, `EXEC_MODEL` | runs the generated harness's agent loop |
| Judge model | `JUDGE_MODEL`, optional `JUDGE_API_*` | grades produced artifacts (falls back to the execution endpoint) |
| Meta model | `META_MODEL`, `META_API_BASE`, `META_API_KEY`, `META_TOKENIZER` | writes the harness (JIT pipeline only) |
| Tools | `SERPER_API_KEY`, `JINA_API_KEY` | `web_search` / `crawl_page` |

**3. Data**

Small datasets ship in the repo; anything large is a documented download.

```bash
python scripts/check_datasets.py        # present / partial / missing, per benchmark
bash scripts/fetch_datasets.sh travel   # one benchmark ("all" ≈ 1 GB)
```

## Usage

**Serve the meta model** — any OpenAI-compatible endpoint works; for a local checkpoint:

```bash
MODEL=/path/to/jit-checkpoint TP=4 bash scripts/serve_meta_model.sh
```

**Run the JIT pipeline** — generate N harnesses per case, let the model pick one, execute
it, score it:

```bash
python -m scripts.run_jit --bench xbench \
    --meta-model jit --meta-base http://localhost:8000/v1 --max-samples 5
```

Defaults reproduce the published setup: 3 candidates per case at temperature 1.0, selected
by the meta model itself (never by benchmark score). Drop `--max-samples` for the full
sweep; add `--selector logprob --tokenizer /path/to/checkpoint` to reproduce the published
selection exactly. `bash scripts/run_jit.sh xbench` is the env-var wrapper around the same
runner.

**Run a fixed harness baseline** — same evaluation path, no meta model, so the two
`summary.json` files are directly comparable:

```bash
python -m scripts.run_seed_harness --bench xbench --harness plan_and_execute
python -m scripts.run_seed_harness --bench xbench --list-harnesses
```

**Benchmarks** — `xbench`, `deepsearchqa`, `agentif`, `officebench`, `odyssey`,
`shopping`, `travel`.

**Output** — everything lands under `runs/<bench>_<...>_<timestamp>/`:

```
summary.json    headline metrics + how the run was configured
generate/       the N candidate harnesses per case, with prompts and responses
select/         the pick per case, the rule that produced it, per-candidate scores
execute/        the harness that actually ran, its trajectory, and the numbers you report
```

Every phase is resume-idempotent: re-run the identical command and only units that failed
for *infrastructure* reasons are retried — a genuine 0-score is a result and is never
re-run. `--skip-generate` / `--skip-select` reuse earlier phases, which is how you change
one variable (selector, execution model) over identical harnesses.

Full flag list: `python -m scripts.run_jit --help`.

## License

See [LICENSE](LICENSE).
