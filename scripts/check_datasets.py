#!/usr/bin/env python3
"""Report which benchmark data is present, missing or incomplete.

    python scripts/check_datasets.py            # every benchmark
    python scripts/check_datasets.py travel agentif

Exits non-zero when anything required is missing, so it doubles as a CI gate.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "dataset"


@dataclass
class Check:
    key: str
    dataset: str
    committed: List[str] = field(default_factory=list)
    """Paths that ship with the repo -- missing means a broken checkout."""

    downloaded: List[str] = field(default_factory=list)
    """Paths you must fetch yourself (large assets)."""

    hint: str = ""


CHECKS = [
    Check("xbench", "xbench", ["xbench/DeepSearch-2505.csv"]),
    Check("deepsearchqa", "deepsearchqa", ["deepsearchqa/DSQA-full.csv"]),
    Check(
        "agentif",
        "agentif_oneday",
        ["agentif_oneday/data.jsonl"],
        ["agentif_oneday/Attachments"],
        hint="bash scripts/fetch_datasets.sh agentif  (250 MB)",
    ),
    Check("officebench", "officebench", ["officebench/data.jsonl", "officebench/tasks"]),
    Check("odyssey", "odysseybench", ["odysseybench/tasks"]),
    Check(
        "shopping",
        "deepplanning_shopping",
        [
            "deepplanning_shopping/data",
            "deepplanning_shopping/database_level1",
            "deepplanning_shopping/database_level2",
            "deepplanning_shopping/database_level3",
        ],
    ),
    Check(
        "travel",
        "deepplanning_travel",
        ["deepplanning_travel/data", "deepplanning_travel/evaluation"],
        ["deepplanning_travel/database"],
        hint="bash scripts/fetch_datasets.sh travel  (748 MB)",
    ),
]


def _exists(relative: str) -> bool:
    path = DATA_ROOT / relative
    if not path.exists():
        return False
    if path.is_dir():
        return any(path.iterdir())
    return path.stat().st_size > 0


def main(argv: List[str]) -> int:
    wanted = set(argv[1:])
    checks = [c for c in CHECKS if not wanted or c.key in wanted]
    if wanted and not checks:
        print(f"unknown benchmark(s): {', '.join(sorted(wanted))}")
        return 2

    print(f"{'BENCH':<14} {'STATUS':<12} DETAIL")
    print("-" * 78)
    failures = 0
    for check in checks:
        missing_committed = [p for p in check.committed if not _exists(p)]
        missing_download = [p for p in check.downloaded if not _exists(p)]

        if missing_committed:
            status, detail = "BROKEN", f"missing from checkout: {', '.join(missing_committed)}"
            failures += 1
        elif missing_download:
            status, detail = "PARTIAL", f"needs download: {', '.join(missing_download)}"
            failures += 1
        else:
            status, detail = "ok", check.dataset
        print(f"{check.key:<14} {status:<12} {detail}")
        if (missing_committed or missing_download) and check.hint:
            print(f"{'':<14} {'':<12} -> {check.hint}")

    print("-" * 78)
    if failures:
        print(f"{failures} benchmark(s) not runnable. See dataset/README.md.")
    else:
        print("all checked benchmarks are ready.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
