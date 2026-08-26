"""XBench evaluation helpers."""

from __future__ import annotations

from typing import Any, Dict

from ..judge import judge_xbench


def evaluate_xbench_prediction(
    prediction: str,
    ground_truth: str,
    *,
    question: str = "",
    judge_model: str,
    judge_api_base: str = "",
    judge_api_key: str = "",
) -> Dict[str, Any]:
    """Run the XBench judge."""
    score, extracted, explanation = judge_xbench(
        question,
        ground_truth,
        prediction,
        model=judge_model,
        api_base=judge_api_base,
        api_key=judge_api_key,
    )

    return {
        "score": float(score),
        "extracted_answer": extracted,
        "explanation": explanation,
    }
