"""Score extraction and aggregation, one profile per evaluator contract.

The benchmarks in this suite expose three different evaluator payloads:

``generic``   xbench / deepsearchqa / agentif / officebench / odyssey --
              ``{score, is_pass, actual_score, max_score, percentage,
              criteria_results}``.
``travel``    DeepPlanning-Travel -- ``{score, composite_score,
              commonsense_weighted_score, personalized_score, case_acc}``.
``shopping``  DeepPlanning-Shopping -- ``{score, case_score, match_rate}``.

Every profile flattens a run into a record carrying a common core (``score``,
``case_acc``, ``passed``, token counts, cost, errors) plus its own extras, so
one runner and one summary format cover all of them.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List

# Substrings marking an *infrastructure* failure (server down, network flake,
# context overflow, full disk) rather than a genuine model/harness failure.
# Units that failed this way are re-run on resume; honest 0-scores are kept.
INFRA_MARKERS = (
    "no space left on device",
    "errno 28",
    "connection error",
    "network error",
    "max retries",
    "maximum context length",
    "apiconnection",
    "service unavailable",
    "bad gateway",
    "502",
    "503",
    "504",
    # Keep engine markers SPECIFIC: a bare "engine" also matches genuine
    # harness bugs (e.g. "should_replan_engine") and would loop them forever.
    "engine is dead",
    "engine loop has died",
    "enginedeaderror",
    "enginecore",
    "remoteprotocol",
    "econnrefused",
    "timed out",
    "read timeout",
    "budget exceeded",
)


def is_infra_failure(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in INFRA_MARKERS)


def should_rerun(report: Dict[str, Any] | None, record: Dict[str, Any]) -> bool:
    """Decide whether a previously saved (case, rollout) must be re-run."""
    if not report:
        return True
    if not report.get("validation_records") and not report.get("meta_agent_trajectory"):
        return True
    return is_infra_failure(
        f"{record.get('run_error', '')} {record.get('validation_error', '')}"
    )


def _last_validation_record(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The FINAL validation record -- the harness the run actually ends with.

    Not the best round seen along the way: a later repair can regress, and the
    number we report has to be the one the pipeline would ship.
    """
    return records[-1] if records else {}


def _common_fields(
    item: Dict[str, Any],
    item_index: int,
    rollout: int,
    group: str,
    report: Dict[str, Any],
    report_path: str,
    error: str,
) -> Dict[str, Any]:
    records = report.get("validation_records", []) or []
    last = _last_validation_record(records)
    return {
        "item_index": item_index,
        "rollout": rollout,
        "group": group,
        "question": (item.get("question", "") or "")[:500],
        "passed": bool(last.get("passed")),
        "generation_success": bool(report.get("generation_success", False)),
        "evaluation_result": last.get("evaluation_result", {}) or {},
        # run diagnostics
        "rounds": len(records),
        "number_of_regenerations": int(report.get("number_of_regenerations", 0) or 0),
        "number_of_review_regenerations": int(
            report.get("number_of_review_regenerations", 0) or 0
        ),
        "steps_used": int(last.get("steps_used", 0) or 0),
        "input_token_count": int(last.get("input_token_count", 0) or 0),
        "output_token_count": int(last.get("output_token_count", 0) or 0),
        "total_token_count": int(last.get("total_token_count", 0) or 0),
        "validation_error": str(last.get("error", "") or ""),
        "run_error": error,
        "report_path": report_path,
    }


# --------------------------------------------------------------------------- #
# generic
# --------------------------------------------------------------------------- #
def _summarize_generic(base: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = base["evaluation_result"]
    is_pass = bool(evaluation.get("is_pass", base["passed"]))
    base.update(
        {
            "score": float(evaluation.get("score", 0.0) or 0.0),
            "is_pass": is_pass,
            "actual_score": float(evaluation.get("actual_score", 0.0) or 0.0),
            "max_score": float(evaluation.get("max_score", 0.0) or 0.0),
            "percentage": float(evaluation.get("percentage", 0.0) or 0.0),
            # binary case accuracy: 1 iff every rubric criterion was satisfied
            "case_acc": 1.0 if is_pass else 0.0,
        }
    )
    return base


def _bucket_generic(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"count": 0}
    return {
        "count": n,
        "avg_score": round(sum(r["score"] for r in rows) / n, 4),
        "pass_rate": round(sum(r["case_acc"] for r in rows) / n, 4),
        "passed": sum(1 for r in rows if r["is_pass"]),
        "generation_success": sum(1 for r in rows if r["generation_success"]),
        "errors": sum(1 for r in rows if r["run_error"] or r["validation_error"]),
    }


# --------------------------------------------------------------------------- #
# travel
# --------------------------------------------------------------------------- #
def _summarize_travel(base: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = base["evaluation_result"]
    composite = float(
        evaluation.get("composite_score", evaluation.get("score", 0.0)) or 0.0
    )
    base.update(
        {
            "score": float(evaluation.get("score", 0.0) or 0.0),
            "composite_score": composite,
            "commonsense_weighted_score": float(
                evaluation.get("commonsense_weighted_score", 0.0) or 0.0
            ),
            "personalized_score": float(evaluation.get("personalized_score", 0.0) or 0.0),
            # binary: commonsense AND personalized both perfect
            "case_acc": float(evaluation.get("case_acc", 0.0) or 0.0),
            "is_pass": float(evaluation.get("case_acc", 0.0) or 0.0) >= 1.0,
        }
    )
    return base


def _bucket_travel(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"count": 0}
    return {
        "count": n,
        "avg_score": round(sum(r["score"] for r in rows) / n, 4),
        "avg_composite_score": round(sum(r["composite_score"] for r in rows) / n, 4),
        "avg_commonsense_score": round(
            sum(r["commonsense_weighted_score"] for r in rows) / n, 4
        ),
        "avg_personalized_score": round(
            sum(r["personalized_score"] for r in rows) / n, 4
        ),
        "case_acc_rate": round(sum(r["case_acc"] for r in rows) / n, 4),
        "passed": sum(1 for r in rows if r["passed"]),
        "generation_success": sum(1 for r in rows if r["generation_success"]),
        "errors": sum(1 for r in rows if r["run_error"] or r["validation_error"]),
    }


# --------------------------------------------------------------------------- #
# shopping
# --------------------------------------------------------------------------- #
def _summarize_shopping(base: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = base["evaluation_result"]
    case_score = float(evaluation.get("case_score", 0.0) or 0.0)
    base.update(
        {
            "score": float(evaluation.get("score", 0.0) or 0.0),
            # strict all-or-nothing per-case pass
            "case_score": case_score,
            "match_rate": float(evaluation.get("match_rate", 0.0) or 0.0),
            "case_acc": case_score,
            "is_pass": case_score >= 1.0,
        }
    )
    return base


def _bucket_shopping(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"count": 0}
    return {
        "count": n,
        "avg_score": round(sum(r["score"] for r in rows) / n, 4),
        "avg_match_rate": round(sum(r["match_rate"] for r in rows) / n, 4),
        "case_pass_rate": round(sum(r["case_score"] for r in rows) / n, 4),
        "passed": sum(1 for r in rows if r["passed"]),
        "generation_success": sum(1 for r in rows if r["generation_success"]),
        "errors": sum(1 for r in rows if r["run_error"] or r["validation_error"]),
    }


PROFILES: Dict[str, Dict[str, Callable]] = {
    "generic": {"summarize": _summarize_generic, "bucket": _bucket_generic},
    "travel": {"summarize": _summarize_travel, "bucket": _bucket_travel},
    "shopping": {"summarize": _summarize_shopping, "bucket": _bucket_shopping},
}


def summarize_case(
    profile: str,
    item: Dict[str, Any],
    item_index: int,
    rollout: int,
    question_id: str,
    group: str,
    report: Dict[str, Any],
    report_path: str,
    error: str = "",
) -> Dict[str, Any]:
    """Flatten one (case, rollout)'s meta-agent report into a score record."""
    base = _common_fields(item, item_index, rollout, group, report, report_path, error)
    # `question_id` must be the same id the artifact dirs and the best-of-N
    # selection use, INDEX-FALLBACK included: datasets that carry no id field
    # (xbench, deepsearchqa) would otherwise collapse every case into one key,
    # silently reducing per-case pass@k to a single "task".
    base["question_id"] = question_id
    return PROFILES[profile]["summarize"](base)


def empty_record(
    item_index: int, rollout: int, question_id: str, group: str, error: str
) -> Dict[str, Any]:
    """A zero record for a unit that blew up before producing any report."""
    return {
        "item_index": item_index,
        "rollout": rollout,
        "question_id": question_id,
        "group": group,
        "score": 0.0,
        "is_pass": False,
        "case_acc": 0.0,
        "passed": False,
        "generation_success": False,
        "evaluation_result": {},
        "rounds": 0,
        "number_of_regenerations": 0,
        "number_of_review_regenerations": 0,
        "steps_used": 0,
        "input_token_count": 0,
        "output_token_count": 0,
        "total_token_count": 0,
        "validation_error": "",
        "run_error": error,
        "report_path": "",
    }


def aggregate_per_rollout(profile: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate treating every rollout as an independent sample."""
    bucket = PROFILES[profile]["bucket"]
    by_group = {
        f"group_{g}": bucket([r for r in records if str(r.get("group", "")) == g])
        for g in sorted({str(r.get("group", "")) for r in records})
    }
    return {"overall": bucket(records), "by_group": by_group}


def aggregate_per_case(profile: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate over each task's rollouts: best-of-k, mean, and pass@k.

    ``pass@k``          fraction of tasks where ANY rollout passes.
    ``avg_best_score``  mean over tasks of the task's best rollout score.
    ``avg_mean_score``  mean over tasks of the task's mean-rollout score.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[str(r.get("question_id", ""))].append(r)

    # travel reports on composite_score; the others on score.
    score_key = "composite_score" if profile == "travel" else "score"

    def _bucket(qids: List[str]) -> Dict[str, Any]:
        n = len(qids)
        if n == 0:
            return {"tasks": 0}
        best, mean, any_pass = [], [], 0
        for qid in qids:
            rows = groups[qid]
            scores = [float(x.get(score_key, 0.0) or 0.0) for x in rows]
            best.append(max(scores) if scores else 0.0)
            mean.append(sum(scores) / len(scores) if scores else 0.0)
            if any(x.get("is_pass") for x in rows):
                any_pass += 1
        return {
            "tasks": n,
            "rollouts_per_task": round(sum(len(groups[q]) for q in qids) / n, 2),
            "pass@k": round(any_pass / n, 4),
            "score_key": score_key,
            "avg_best_score": round(sum(best) / n, 4),
            "avg_mean_score": round(sum(mean) / n, 4),
        }

    all_qids = list(groups.keys())
    by_group = {}
    for g in sorted({str(groups[q][0].get("group", "")) for q in all_qids}):
        qids = [q for q in all_qids if str(groups[q][0].get("group", "")) == g]
        by_group[f"group_{g}"] = _bucket(qids)

    return {"overall": _bucket(all_qids), "by_group": by_group}
