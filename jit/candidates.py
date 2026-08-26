"""Rebuild each generated harness as a (prompt, completion) pair.

Best-of-N selection ranks *what the meta model actually produced*, so each
candidate has to be reconstructed the way the model emitted it:

  * take the FINAL design turn of the run (``generate`` / ``regenerate`` /
    ``review_regenerate``) -- a repair turn is not a design;
  * strip the leading empty ``</think>`` stub (training convention);
  * replace each ``<<<PYTHON_*>>>`` / ``<<<YAML>>>`` block with the canonical
    file that was actually written to disk, so the completion matches the
    harness that would be executed rather than a possibly-truncated echo.

Nothing is dropped here: leaky, non-empty-think and incomplete candidates are
kept and flagged (``n_sub`` counts how many of the five blocks were emitted),
because the selector must rank the real pool.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DESIGN_STAGES = {"generate", "review_regenerate", "regenerate"}

BLOCKS: Tuple[Tuple[str, str, str], ...] = (
    ("<<<PYTHON_MEMORY>>>", "<<<END_PYTHON_MEMORY>>>", "memory.py"),
    ("<<<PYTHON_PLANNING>>>", "<<<END_PYTHON_PLANNING>>>", "planning.py"),
    ("<<<PYTHON_ACTION>>>", "<<<END_PYTHON_ACTION>>>", "action.py"),
    ("<<<PYTHON_TOOL_POLICY>>>", "<<<END_PYTHON_TOOL_POLICY>>>", "tool_policy.py"),
    ("<<<YAML>>>", "<<<END_YAML>>>", "prompt.yaml"),
)
N_BLOCKS = len(BLOCKS)


def substitute_harness(response: str, harness_dir: Path) -> Tuple[str, int]:
    """Swap each tagged block for the file on disk; return (text, n_replaced)."""
    out, replaced = response, 0
    for open_tag, close_tag, filename in BLOCKS:
        path = harness_dir / filename
        if not path.is_file():
            continue
        code = path.read_text(encoding="utf-8", errors="replace").strip("\n")
        pattern = re.compile(re.escape(open_tag) + r".*?" + re.escape(close_tag), re.DOTALL)
        new_block = f"{open_tag}\n{code}\n{close_tag}"
        out, n = pattern.subn(lambda _m, _b=new_block: _b, out, count=1)
        replaced += n
    return out, replaced


def strip_empty_think(response: str) -> Tuple[str, bool]:
    """Drop a leading empty ``</think>`` stub; flag a non-empty thinking trace."""
    idx = response.find("</think>")
    if idx == -1:
        return response.lstrip("\n"), False
    if response[:idx].strip():
        return response, True  # real reasoning before the tag: keep it as-is
    return response[idx + len("</think>") :].lstrip("\n"), False


def load_candidate(report_path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """Rebuild one candidate from a ``--generate-only`` report.json."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None, "report_unreadable"

    trajectory = [t for t in (report.get("meta_agent_trajectory") or []) if isinstance(t, dict)]
    designs = [t for t in trajectory if t.get("stage") in DESIGN_STAGES]
    if not designs:
        return None, "no_design_turn"

    prompt = designs[0].get("prompt") or {}
    if not isinstance(prompt, dict):
        return None, "prompt_not_dict"
    system_prompt, user_prompt = prompt.get("system_prompt"), prompt.get("user_prompt")
    response = designs[-1].get("response")
    if not (system_prompt and user_prompt and isinstance(response, str) and response.strip()):
        return None, "missing_prompt_or_response"

    response, nonempty_think = strip_empty_think(response)
    completion, n_sub = substitute_harness(response, report_path.parent / "harness")
    return {
        "system": system_prompt,
        "user": user_prompt,
        "completion": completion,
        "n_sub": n_sub,
        "final_stage": designs[-1].get("stage"),
        "nonempty_think": nonempty_think,
    }, ""


def extract(gen_out: Path, cases: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Collect every candidate under a ``--generate-only`` output directory.

    Returns one row per (case, rollout). Rows that could not be rebuilt carry
    an ``error`` field instead of a completion; the selector treats them as
    unscorable rather than dropping the case.
    """
    gen_out = Path(gen_out)
    wanted = set(cases) if cases else None

    rows: List[Dict[str, Any]] = []
    for report_path in sorted(gen_out.glob("cases/*/rollout_*/report.json")):
        case = report_path.parent.parent.name
        if wanted is not None and case not in wanted:
            continue
        match = re.fullmatch(r"rollout_(\d+)", report_path.parent.name)
        if not match:
            continue
        row: Dict[str, Any] = {"case": case, "rollout": int(match.group(1))}
        candidate, why = load_candidate(report_path)
        if candidate is None:
            row["error"] = why
        else:
            row.update(candidate)
        rows.append(row)
    return rows
