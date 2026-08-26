#!/usr/bin/env python3
"""JIT best-of-N: generate N harnesses, let the model pick one, run it, score it.

One command evaluates one benchmark end to end::

    python -m scripts.run_jit --bench xbench \
        --meta-model jit-ckpt70 --meta-base http://localhost:8000/v1

Three phases, all resumable -- re-running the same command picks up where it
stopped and only redoes units that failed for infrastructure reasons:

  A. GENERATE  the meta model writes ``--rollouts`` harnesses per case at
     ``--meta-temperature`` (3 @ T=1 by default). No execution, no scoring: this
     phase only produces candidates.

  B. SELECT    the model picks its own favourite of the N candidates for each
     case -- by completion log-probability where the endpoint supports it, else
     by asking a model to judge (see ``jit/selector.py``). No benchmark
     score is consulted; using one would be leakage.

  C. EXECUTE   the selected harness is installed and run against its case by
     the execution model, then scored by the benchmark's evaluator. Scores are
     aggregated into ``summary.json``.

Every model role -- meta, execution, judge, selector -- is an OpenAI-compatible
chat-completions endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark import registry  # noqa: E402
from jit import candidates as candidates_mod  # noqa: E402
from jit import selector  # noqa: E402
from jit.meta_agent import resolve_reference_k, resolve_reference_mode  # noqa: E402
from scripts.eval import runner  # noqa: E402
from scripts.eval.config import (  # noqa: E402
    build_run_config,
    load_dotenv,
    make_meta_model_config,
)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts.run_jit",
        description="JIT best-of-N harness generation + selection + evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bench", required=True, choices=registry.names(), help="Benchmark to run.")
    p.add_argument("--config", default="", help="Override the benchmark YAML.")
    p.add_argument("--dataset-path", default="", help="Override benchmark.dataset_path.")

    meta = p.add_argument_group("meta model (writes the harness)")
    meta.add_argument("--meta-model", default=os.getenv("META_MODEL", "jit"),
                      help="Served model name of the JIT model.")
    meta.add_argument("--meta-base", default=os.getenv("META_API_BASE", "http://localhost:8000/v1"),
                      help="OpenAI-compatible base URL of the JIT model.")
    meta.add_argument("--meta-key", default=os.getenv("META_API_KEY", "EMPTY"), help="API key.")
    meta.add_argument("--meta-max-tokens", type=int, default=64000,
                      help="A harness is 5 files plus reasoning; do not shrink this.")
    meta.add_argument("--harness-refs", default=None, choices=["desc", "code"],
                      help="Reference material in the generation prompt. desc: no-code design "
                           "catalogue of all 11 reference harnesses. code: full source of "
                           "--harness-refs-k reference harnesses drawn at random. "
                           "Default: meta_references.mode in the benchmark YAML, "
                           "else JIT_META_REF_CODE=1, else desc.")
    meta.add_argument("--harness-refs-k", type=int, default=None,
                      help="How many seed harnesses --harness-refs code shows. Default: 3.")
    meta.add_argument("--meta-temperature", type=float, default=1.0,
                      help="Sampling temperature for generation (best-of-N needs > 0).")

    ex = p.add_argument_group("execution + judge models")
    ex.add_argument("--exec-model", default="", help="Override the execution model id.")
    ex.add_argument("--exec-base", default="", help="Override the execution api_base.")
    ex.add_argument("--exec-key", default="", help="Override the execution api_key.")
    ex.add_argument("--judge-model", default="", help="Override the judge model id.")
    ex.add_argument("--judge-base", default="", help="Override the judge api_base.")
    ex.add_argument("--judge-key", default="", help="Override the judge api_key.")

    sel = p.add_argument_group("best-of-N selection")
    sel.add_argument("--rollouts", type=int, default=3, help="Candidate harnesses per case.")
    sel.add_argument("--selector", default="auto", choices=["auto", "logprob", "judge"],
                     help="auto probes for prompt_logprobs and falls back to judge.")
    sel.add_argument("--tokenizer", default=os.getenv("META_TOKENIZER", ""),
                     help="Model dir / HF repo id for the logprob selector's tokenizer.")
    sel.add_argument("--selector-base", default="",
                     help="Score on a second endpoint (keeps generation throughput up).")
    sel.add_argument("--selector-judge-model", default="",
                     help="Judge-selector model id (default: the meta model).")
    sel.add_argument("--selector-concurrency", type=int, default=3)
    sel.add_argument("--max-model-len", type=int, default=163840)

    run = p.add_argument_group("run control")
    run.add_argument("--max-steps", type=int, default=None, help="Default: per-benchmark.")
    run.add_argument("--max-repairs", type=int, default=5,
                     help="Repair attempts during execution when a run errors out.")
    run.add_argument("--workers-gen", type=int, default=5, help="Concurrent generation units.")
    run.add_argument("--workers-exec", type=int, default=10, help="Concurrent execution units.")
    run.add_argument("--max-samples", type=int, default=None, help="Cap the number of cases.")
    run.add_argument("--cases", default="", help="Comma-separated question_id list.")
    run.add_argument("--attempts", type=int, default=3,
                     help="Re-run passes that ended with infrastructure failures.")
    run.add_argument("--output", default="", help="Default: runs/<bench>_<timestamp>.")
    run.add_argument("--skip-generate", action="store_true",
                     help="Reuse an existing generate/ directory.")
    run.add_argument("--skip-select", action="store_true",
                     help="Reuse an existing select/selection.json.")
    run.add_argument("--no-resume", action="store_true", help="Re-run every unit.")
    return p


def _apply_bench_env(spec: registry.BenchmarkSpec) -> None:
    """Export the environment the benchmark harness requires."""
    for key, value in spec.env.items():
        os.environ.setdefault(key, value)


def _preflight(spec: registry.BenchmarkSpec, config: Dict[str, Any]) -> None:
    """Warn loudly about missing prerequisites instead of scoring garbage."""
    dataset_path = (config.get("benchmark", {}) or {}).get("dataset_path", "")
    resolved = Path(dataset_path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / dataset_path
    if not resolved.exists():
        raise SystemExit(
            f"dataset not found: {resolved}\n"
            f"See dataset/README.md for how to fetch '{spec.key}'."
        )
    missing = [need for need in spec.needs if need.startswith("dataset/")
               and not (REPO_ROOT / need).exists()]
    missing += [need for need in spec.needs
                if not need.startswith("dataset/") and not os.getenv(need.split()[0])]
    if missing:
        print(
            "[preflight] WARNING -- these prerequisites look unmet, results may be "
            "meaningless:\n  " + "\n  ".join(missing) + "\n"
            "  (see dataset/README.md)",
            flush=True,
        )


def _run_phase_with_retries(
    config: Dict[str, Any], opts: runner.RunOptions, attempts: int, phase: str
) -> Dict[str, Any]:
    """Run a phase, retrying only while infrastructure failures remain.

    A genuine 0-score is a result and is kept. A dead endpoint, a network flake
    or a full disk is not, and resume re-runs exactly those units.
    """
    summary: Dict[str, Any] = {}
    for attempt in range(1, max(1, attempts) + 1):
        print(f"\n=== [{phase}] attempt {attempt}/{attempts} {datetime.now().isoformat()}",
              flush=True)
        summary = runner.run(config, opts)
        remaining = runner.count_infra_failures(opts.output)
        print(f"=== [{phase}] attempt {attempt} done; infra-failed units: {remaining}", flush=True)
        if remaining == 0:
            break
        if attempt < attempts:
            # Later attempts must resume, or the retry would redo everything.
            opts.no_resume = False
            time.sleep(20)
        else:
            print(f"=== [{phase}] out of attempts; {remaining} units still failing (kept)",
                  flush=True)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv()

    spec = registry.get(args.bench)
    _apply_bench_env(spec)

    # Benchmark YAMLs use repo-relative dataset paths.
    os.chdir(REPO_ROOT)

    max_steps = args.max_steps if args.max_steps is not None else spec.max_steps
    meta_model = make_meta_model_config(
        args.meta_model, args.meta_base, args.meta_key,
        max_tokens=args.meta_max_tokens, temperature=args.meta_temperature,
    )
    config = build_run_config(
        args.config or spec.config,
        meta_model=meta_model,
        exec_model=args.exec_model,
        exec_base=args.exec_base,
        exec_key=args.exec_key,
        judge_model=args.judge_model,
        judge_base=args.judge_base,
        judge_key=args.judge_key,
        dataset_path=args.dataset_path or spec.dataset_path,
        max_steps=max_steps,
        harness_refs=args.harness_refs or "",
        harness_refs_k=args.harness_refs_k,
    )
    _preflight(spec, config)

    if args.output:
        output_root = Path(args.output)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "runs" / f"{args.bench}_jit_bo{args.rollouts}_{stamp}"
    output_root = output_root.resolve()
    gen_dir, sel_dir, exec_dir = (output_root / "generate", output_root / "select",
                                  output_root / "execute")
    for directory in (gen_dir, sel_dir, exec_dir):
        directory.mkdir(parents=True, exist_ok=True)

    case_list: Optional[List[str]] = (
        [c.strip() for c in args.cases.split(",") if c.strip()] if args.cases else None
    )

    refs_mode = resolve_reference_mode(config)
    refs_label = (
        f"code (source of {resolve_reference_k(config)} sampled seed harnesses)"
        if refs_mode == "code"
        else "desc (no-code catalogue of all 11 seed harnesses)"
    )

    print("=" * 78, flush=True)
    print(
        f"[jit] bench={args.bench} ({spec.notes})\n"
        f"[jit] meta   = {args.meta_model} @ {args.meta_base}"
        f"  (T={args.meta_temperature})\n"
        f"[jit] exec   = {config['model'].get('model_id')} @ {config['model'].get('api_base')}\n"
        f"[jit] judge  = {(config['benchmark'].get('judge') or {}).get('model_id', '-')}\n"
        f"[jit] refs   = {refs_label}\n"
        f"[jit] best-of-{args.rollouts}, selector={args.selector}, max_steps={max_steps}\n"
        f"[jit] output = {output_root}",
        flush=True,
    )
    print("=" * 78, flush=True)

    common = dict(
        group_field=spec.group_field,
        metrics_profile=spec.metrics,
        max_steps=max_steps,
        max_samples=args.max_samples,
        cases=case_list,
        no_resume=args.no_resume,
    )

    # ---------------- Phase A: generate N candidate harnesses ---------------- #
    if not args.skip_generate:
        gen_opts = runner.RunOptions(
            output=gen_dir,
            rollouts=args.rollouts,
            parallel_workers=args.workers_gen,
            generate_only=True,
            # Generation never executes anything, so repairs are meaningless here.
            max_repairs=0,
            label=f"generate/bo{args.rollouts}@T{args.meta_temperature}",
            **common,
        )
        _run_phase_with_retries(config, gen_opts, args.attempts, "A/generate")
    else:
        print("[jit] phase A skipped (--skip-generate)", flush=True)

    # ---------------- Phase B: the model picks its favourite ----------------- #
    selection_path = sel_dir / "selection.json"
    if args.skip_select and selection_path.is_file():
        selection = selector.load_selection(str(selection_path))
        print(f"[jit] phase B skipped; reusing {selection_path} ({len(selection)} cases)",
              flush=True)
    else:
        rows = candidates_mod.extract(gen_dir)
        if not rows:
            raise SystemExit(f"no candidates found under {gen_dir}/cases -- did phase A run?")
        print(f"[jit] phase B: {len(rows)} candidates from {gen_dir}", flush=True)
        result = selector.select(
            rows,
            strategy=args.selector,
            api_base=args.selector_base or args.meta_base,
            model=args.meta_model,
            api_key=args.meta_key,
            tokenizer_dir=args.tokenizer,
            judge_model=args.selector_judge_model,
            max_model_len=args.max_model_len,
            concurrency=args.selector_concurrency,
        )
        selector.write_candidate_records(rows, str(sel_dir / "candidates.jsonl"))
        selection_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        selection = {str(k): int(v) for k, v in result["selected"].items()}
        print(f"[jit] phase B: wrote {selection_path} (strategy={result['strategy']})", flush=True)

    # ---------------- Phase C: run the selected harness --------------------- #
    exec_opts = runner.RunOptions(
        output=exec_dir,
        # One execution of one chosen harness -- the best-of-N happened upstream.
        rollouts=1,
        parallel_workers=args.workers_exec,
        harnesses_from=str(gen_dir),
        selection=selection,
        max_repairs=args.max_repairs,
        # Repair a crash, but never "repair" a harness that merely scored low:
        # that would be optimising against the benchmark.
        repair_only_on_error=True,
        regenerate_on_error=True,
        label=f"execute/selected-of-{args.rollouts}",
        **common,
    )
    exec_summary = _run_phase_with_retries(config, exec_opts, args.attempts, "C/execute")

    # ---------------- Final summary ------------------------------------------ #
    picks: Dict[str, int] = {}
    for rollout in selection.values():
        picks[str(rollout)] = picks.get(str(rollout), 0) + 1
    summary = {
        "benchmark": args.bench,
        "pipeline": "jit-best-of-n",
        "rollouts": args.rollouts,
        "meta_model": args.meta_model,
        "meta_base": args.meta_base,
        "meta_temperature": args.meta_temperature,
        "harness_refs": refs_mode,
        "selector": args.selector,
        "execution_model": config["model"].get("model_id", ""),
        "judge_model": (config["benchmark"].get("judge") or {}).get("model_id", ""),
        "selected_rollout_distribution": dict(sorted(picks.items())),
        "generate_dir": str(gen_dir),
        "select_file": str(selection_path),
        "execute_dir": str(exec_dir),
        "result": exec_summary,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    overall = ((exec_summary.get("per_rollout_metrics") or {}).get("overall")) or {}
    print("\n" + "=" * 78, flush=True)
    print(f"[jit] {args.bench}: n={overall.get('count', 0)} "
          f"avg_score={overall.get('avg_score', 0)} "
          f"pass_rate={overall.get('pass_rate', overall.get('case_pass_rate', '-'))}", flush=True)
    print(f"[jit] picked rollout distribution: {dict(sorted(picks.items()))}", flush=True)
    print(f"[jit] summary: {output_root / 'summary.json'}", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
