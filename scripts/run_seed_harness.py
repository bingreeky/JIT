#!/usr/bin/env python3
"""Run one hand-written seed harness from the harness factory on a benchmark.

These are fixed agent architectures spanning ReAct variants, hierarchical
planners, memory-centric designs, and ensembles. Running one is the natural
baseline for a JIT run: *this* is what a single fixed framework achieves,
versus a framework written per task.

    python -m scripts.run_seed_harness --bench xbench --harness plan_and_execute

No meta model is involved. The harness is installed verbatim, executed by the
execution model, and scored by the same evaluator and the same aggregation code
the JIT pipeline uses -- so the two ``summary.json`` files are directly
comparable.

The harness sources live in ``harness_factory/harnesses/<name>/``;
``harness_factory/descriptions/`` holds the natural-language design
write-up of each one, which is also what the JIT model is shown as reference
material in its generation prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark import registry  # noqa: E402
from scripts.eval import runner  # noqa: E402
from scripts.eval.config import build_run_config, load_dotenv  # noqa: E402

HARNESS_ROOT = REPO_ROOT / "harness_factory" / "harnesses"

# The seed bank, in the order the JIT generation prompt lists them.
SEED_HARNESSES = [
    "plan_and_execute",
    "flash_searcher",
    "agentfold",
    "resum",
    "hiagent",
    "memobrain",
    "deepagent",
    "gam",
    "roma",
    "aggagent",
    "oagent",
]


def available_harnesses() -> List[str]:
    """Every harness on disk that has the five required files."""
    found = []
    for path in sorted(HARNESS_ROOT.iterdir()):
        if not path.is_dir() or path.name.startswith("__"):
            continue
        if all((path / f).is_file() for f in runner.HARNESS_FILES):
            found.append(path.name)
    return found


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts.run_seed_harness",
        description="Run a fixed seed harness on a benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bench", required=True, choices=registry.names())
    p.add_argument("--harness", default="plan_and_execute",
                   help=f"One of: {', '.join(SEED_HARNESSES)}")
    p.add_argument("--list-harnesses", action="store_true",
                   help="Print the available harnesses and exit.")
    p.add_argument("--config", default="", help="Override the benchmark YAML.")
    p.add_argument("--dataset-path", default="", help="Override benchmark.dataset_path.")

    ex = p.add_argument_group("execution + judge models")
    ex.add_argument("--exec-model", default="", help="Override the execution model id.")
    ex.add_argument("--exec-base", default="", help="Override the execution api_base.")
    ex.add_argument("--exec-key", default="", help="Override the execution api_key.")
    ex.add_argument("--judge-model", default="", help="Override the judge model id.")
    ex.add_argument("--judge-base", default="", help="Override the judge api_base.")
    ex.add_argument("--judge-key", default="", help="Override the judge api_key.")

    run = p.add_argument_group("run control")
    run.add_argument("--rollouts", type=int, default=1,
                     help="Repeat each case N times (a fixed harness is deterministic "
                          "at temperature 0, so 1 is usually right).")
    run.add_argument("--max-steps", type=int, default=None, help="Default: per-benchmark.")
    run.add_argument("--workers", type=int, default=10, help="Concurrent cases.")
    run.add_argument("--max-samples", type=int, default=None, help="Cap the number of cases.")
    run.add_argument("--cases", default="", help="Comma-separated question_id list.")
    run.add_argument("--attempts", type=int, default=3,
                     help="Re-run passes that ended with infrastructure failures.")
    run.add_argument("--output", default="", help="Default: runs/<bench>_<harness>_<ts>.")
    run.add_argument("--no-resume", action="store_true", help="Re-run every unit.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.list_harnesses:
        for name in available_harnesses():
            print(f"  {name}")
        return 0

    load_dotenv()
    spec = registry.get(args.bench)
    for key, value in spec.env.items():
        os.environ.setdefault(key, value)

    harness_dir = HARNESS_ROOT / args.harness
    missing = [f for f in runner.HARNESS_FILES if not (harness_dir / f).is_file()]
    if missing:
        raise SystemExit(
            f"harness '{args.harness}' is incomplete at {harness_dir} "
            f"(missing: {', '.join(missing)}).\n"
            f"Available: {', '.join(available_harnesses())}"
        )

    os.chdir(REPO_ROOT)
    max_steps = args.max_steps if args.max_steps is not None else spec.max_steps
    config = build_run_config(
        args.config or spec.config,
        # A preset harness is never generated or repaired, so the meta model is
        # never called. The block exists only to satisfy the agent constructor.
        meta_model={"model_id": "unused", "api_base": "", "api_key": "EMPTY"},
        exec_model=args.exec_model,
        exec_base=args.exec_base,
        exec_key=args.exec_key,
        judge_model=args.judge_model,
        judge_base=args.judge_base,
        judge_key=args.judge_key,
        dataset_path=args.dataset_path or spec.dataset_path,
        max_steps=max_steps,
    )

    dataset_path = Path(config["benchmark"].get("dataset_path", ""))
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path
    if not dataset_path.exists():
        raise SystemExit(
            f"dataset not found: {dataset_path}\n"
            f"See dataset/README.md for how to fetch '{spec.key}'."
        )

    if args.output:
        output_root = Path(args.output)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "runs" / f"{args.bench}_{args.harness}_{stamp}"
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cases: Optional[List[str]] = (
        [c.strip() for c in args.cases.split(",") if c.strip()] if args.cases else None
    )

    print("=" * 78, flush=True)
    print(
        f"[seed] bench   = {args.bench} ({spec.notes})\n"
        f"[seed] harness = {args.harness}  ({harness_dir})\n"
        f"[seed] exec    = {config['model'].get('model_id')} @ {config['model'].get('api_base')}\n"
        f"[seed] judge   = {(config['benchmark'].get('judge') or {}).get('model_id', '-')}\n"
        f"[seed] output  = {output_root}",
        flush=True,
    )
    print("=" * 78, flush=True)

    opts = runner.RunOptions(
        output=output_root,
        group_field=spec.group_field,
        metrics_profile=spec.metrics,
        max_steps=max_steps,
        # A fixed harness is the baseline as written: never repair it, or the
        # number stops describing the harness.
        max_repairs=0,
        rollouts=args.rollouts,
        parallel_workers=args.workers,
        preset_harness=str(harness_dir),
        max_samples=args.max_samples,
        cases=cases,
        no_resume=args.no_resume,
        label=f"seed/{args.harness}",
    )

    summary: Dict = {}
    for attempt in range(1, max(1, args.attempts) + 1):
        print(f"\n=== attempt {attempt}/{args.attempts} {datetime.now().isoformat()}", flush=True)
        summary = runner.run(config, opts)
        remaining = runner.count_infra_failures(output_root)
        print(f"=== attempt {attempt} done; infra-failed units: {remaining}", flush=True)
        if remaining == 0:
            break
        opts.no_resume = False

    summary["pipeline"] = "seed-harness"
    summary["harness"] = args.harness
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    overall = ((summary.get("per_rollout_metrics") or {}).get("overall")) or {}
    print("\n" + "=" * 78, flush=True)
    print(f"[seed] {args.bench} / {args.harness}: n={overall.get('count', 0)} "
          f"avg_score={overall.get('avg_score', 0)} "
          f"pass_rate={overall.get('pass_rate', overall.get('case_pass_rate', '-'))}", flush=True)
    print(f"[seed] summary: {output_root / 'summary.json'}", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
