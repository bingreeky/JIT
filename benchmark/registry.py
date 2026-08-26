"""The benchmark suite this project evaluates on.

One entry per benchmark, holding everything the runners need to know that is
*not* already in the YAML: which metric family the evaluator speaks, which item
field to bucket results by, the full-dataset case count, and any environment
variables the benchmark's own harness requires.

Adding a benchmark = adding a YAML under ``benchmark/config/`` plus one
entry here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class BenchmarkSpec:
    """Everything the runners need beyond the benchmark YAML itself."""

    key: str
    """Short name used on the command line (``--bench <key>``)."""

    config: str
    """Benchmark YAML, relative to the repo root."""

    group_field: str
    """Item field results are bucketed by in the summary (``by_group``)."""

    max_steps: int
    """Step budget for one harness execution."""

    expected_cases: int
    """Case count of the full dataset -- progress reporting only."""

    metrics: str = "generic"
    """Metric family: ``generic`` | ``travel`` | ``shopping`` (see metrics.py)."""

    dataset_path: str = ""
    """Overrides ``benchmark.dataset_path`` when non-empty."""

    env: Dict[str, str] = field(default_factory=dict)
    """Extra environment variables exported before the benchmark harness runs."""

    needs: List[str] = field(default_factory=list)
    """Human-readable external prerequisites, surfaced in preflight warnings."""

    notes: str = ""


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "xbench": BenchmarkSpec(
        key="xbench",
        config="benchmark/config/xbench.yaml",
        group_field="group",
        max_steps=40,
        expected_cases=100,
        needs=["SERPER_API_KEY", "JINA_API_KEY"],
        notes="xbench DeepSearch-2505; web search + crawl tools.",
    ),
    "deepsearchqa": BenchmarkSpec(
        key="deepsearchqa",
        config="benchmark/config/deepsearchqa.yaml",
        group_field="problem_category",
        max_steps=40,
        expected_cases=900,
        needs=["SERPER_API_KEY", "JINA_API_KEY"],
        notes="DeepSearchQA (DSQA-full); web search + crawl tools.",
    ),
    "agentif": BenchmarkSpec(
        key="agentif",
        config="benchmark/config/agentif_oneday.yaml",
        group_field="group",
        max_steps=40,
        expected_cases=104,
        env={
            # A text-only execution model cannot see the HTML-render
            # screenshots the stock judge payload carries; without this the
            # judge endpoint rejects those cases outright.
            "AGENTIF_JUDGE_TEXT_ONLY": "1",
            # agentif_oneday_eval defaults llm_max_tokens=600000, which most
            # providers reject outright -- every judge call then 400s and the
            # case looks like a harness failure.
            "LLM_MAX_TOKENS": "32000",
        },
        needs=["SERPER_API_KEY", "JINA_API_KEY", "dataset/agentif_oneday/Attachments/"],
        notes="AgentIF-OneDay, full 104-case set; text-only judge payload.",
    ),
    "officebench": BenchmarkSpec(
        key="officebench",
        config="benchmark/config/officebench.yaml",
        group_field="num_app_tag",
        max_steps=40,
        expected_cases=295,
        notes="OfficeBench; deterministic evaluator, no judge model.",
    ),
    "odyssey": BenchmarkSpec(
        key="odyssey",
        config="benchmark/config/odysseybench.yaml",
        group_field="track",
        max_steps=50,
        expected_cases=300,
        notes="OdysseyBench (plus subset, raw_chat memory -- no embedding server).",
    ),
    "shopping": BenchmarkSpec(
        key="shopping",
        config="benchmark/config/deepplanning_shopping.yaml",
        # Shopping items carry the difficulty level, so results bucket by it.
        group_field="_level",
        max_steps=40,
        expected_cases=120,
        metrics="shopping",
        notes="DeepPlanning-Shopping (levels 1/2/3); programmatic cart scoring.",
    ),
    "travel": BenchmarkSpec(
        key="travel",
        config="benchmark/config/deepplanning_travel.yaml",
        group_field="group",
        max_steps=40,
        expected_cases=120,
        metrics="travel",
        needs=["dataset/deepplanning_travel/database/"],
        notes="DeepPlanning-Travel (zh); judge converts plan text -> JSON, "
        "then a deterministic constraint evaluator scores it.",
    ),
}


def get(key: str) -> BenchmarkSpec:
    """Look up a benchmark by its ``--bench`` key."""
    if key not in BENCHMARKS:
        raise SystemExit(
            f"unknown benchmark '{key}'. Available: {', '.join(sorted(BENCHMARKS))}"
        )
    return BENCHMARKS[key]


def names() -> List[str]:
    return sorted(BENCHMARKS)
