"""DeepSearchQA evaluation helpers."""

from __future__ import annotations

from typing import Any, Dict

from ..judge import judge_deepsearchqa_item


def evaluate_deepsearchqa_prediction(
    prediction: str,
    ground_truth: str,
    *,
    question: str = "",
    answer_type: str = "Single Answer",
    judge_model: str,
    judge_api_base: str = "",
    judge_api_key: str = "",
) -> Dict[str, Any]:
    """Evaluate a DeepSearchQA prediction with the benchmark judge."""
    result = judge_deepsearchqa_item(
        question,
        ground_truth,
        prediction,
        answer_type=answer_type,
        model=judge_model,
        api_base=judge_api_base,
        api_key=judge_api_key,
    )

    return {
        "precision": result.get("precision", 0.0),
        "recall": result.get("recall", 0.0),
        "f1": result.get("f1", 0.0),
        "judgement": result.get("judgement", "error"),
        "score": result.get("f1", 0.0),
    }
