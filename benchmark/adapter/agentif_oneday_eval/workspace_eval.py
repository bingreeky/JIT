"""
Workspace-oriented AgentIF-OneDay evaluation helpers.

This bridges the standalone llm_as_judge scorer with this project's
benchmark workflow, where answer files live in a per-task workspace
instead of Attachments/{agent_name}/.
"""

import asyncio
import os
from typing import Any, Dict, Iterable, List, Optional

from .config import get_settings
from .data_loader import Answer, Question, ScoreCriterion
from .llm.client.openrouter import (
    ContextLengthExceededError,
    PayloadTooLargeError,
)
from .scorer import AsyncScorer, ScoringTask


def collect_workspace_output_paths(workspace: str) -> List[str]:
    """Collect user-created files from the task workspace top level only."""
    if not workspace or not os.path.isdir(workspace):
        return []

    output_paths: List[str] = []
    for filename in sorted(os.listdir(workspace)):
        if filename.startswith(".") or filename.startswith("_script_"):
            continue
        path = os.path.join(workspace, filename)
        if os.path.isfile(path):
            output_paths.append(path)

    return output_paths


async def evaluate_workspace_task_async(
    *,
    question_id: str,
    title: str,
    description: str,
    prediction: str,
    score_criteria: Iterable[Dict[str, Any]],
    question_attachment_paths: Optional[List[str]] = None,
    reference_answer_description: str = "",
    reference_attachment_paths: Optional[List[str]] = None,
    answer_attachment_paths: Optional[List[str]] = None,
    judge_model: str = "qwen/qwen3.5-122b-a10b",
    judge_api_base: str = "",
    judge_api_key: str = "",
    max_concurrent: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one workspace task using the migrated official scorer."""
    judge_model = judge_model
    settings = get_settings()
    _apply_judge_overrides(
        settings=settings,
        judge_api_base=judge_api_base,
        judge_api_key=judge_api_key,
    )
    scorer = AsyncScorer(
        model_name=judge_model,
        max_concurrent=max_concurrent or settings.llm_max_concurrent,
        max_retries=max_retries or settings.llm_max_retries,
        attachment_base_path="",
    )

    criteria = [
        ScoreCriterion(
            content=criterion.get("content", ""),
            score=criterion.get("score", 0),
            criterion_id=criterion.get("criterion_id") or str(index),
        )
        for index, criterion in enumerate(score_criteria, 1)
    ]

    task = ScoringTask(
        question=Question(
            question_id=question_id,
            title=title,
            description=description,
            score_criteria=criteria,
            reference_answer_description=reference_answer_description,
        ),
        answer=Answer(
            question_id=question_id,
            agent_name="workspace_agent",
            content={"text": prediction},
        ),
        model_name=judge_model,
    )

    question_attachments = await _parse_text_attachments(
        scorer, question_attachment_paths or []
    )
    answer_text_attachments, answer_image_attachments = await _parse_answer_attachments(
        scorer, answer_attachment_paths or []
    )
    reference_answer_attachments = await _parse_text_attachments(
        scorer, reference_attachment_paths or []
    )

    use_grounding = False
    if judge_model.startswith("gemini-") and settings.enable_google_search_grounding:
        use_grounding = True

    try:
        results = await scorer._do_score_with_llm(
            task=task,
            question_attachments=question_attachments,
            answer_text=prediction,
            answer_text_attachments=answer_text_attachments,
            answer_image_attachments=answer_image_attachments,
            reference_answer_attachments=reference_answer_attachments,
            score_criteria=[
                {
                    "criterion_id": criterion.criterion_id or str(index),
                    "content": criterion.content,
                    "score": criterion.score,
                }
                for index, criterion in enumerate(criteria, 1)
            ],
            use_grounding=use_grounding,
        )
    except (PayloadTooLargeError, ContextLengthExceededError):
        if not settings.web_search_fallback_enabled:
            raise
        results = await scorer._do_score_with_fallback(
            task=task,
            question_attachments=question_attachments,
            answer_text=prediction,
            answer_text_attachments=answer_text_attachments,
            answer_image_attachments=answer_image_attachments,
            reference_answer_attachments=reference_answer_attachments,
            score_criteria=[
                {
                    "criterion_id": criterion.criterion_id or str(index),
                    "content": criterion.content,
                    "score": criterion.score,
                }
                for index, criterion in enumerate(criteria, 1)
            ],
        )

    max_score = sum(max(criterion.score, 0) for criterion in criteria)
    actual_score = sum(
        result.criterion_score for result in results if result.satisfied
    )
    actual_score = max(0, actual_score)
    percentage = (actual_score / max_score * 100) if max_score > 0 else 0.0

    return {
        "score": round(percentage / 100, 4),
        "actual_score": actual_score,
        "max_score": max_score,
        "percentage": round(percentage, 2),
        "criteria_results": [
            {
                "criterion_id": criterion.criterion_id or str(index),
                "satisfied": result.satisfied,
                "reasoning": result.reasoning,
            }
            for index, (criterion, result) in enumerate(zip(criteria, results), 1)
        ],
    }


def evaluate_workspace_task(**kwargs) -> Dict[str, Any]:
    """Synchronous wrapper for benchmark adapters."""
    return asyncio.run(evaluate_workspace_task_async(**kwargs))


def _apply_judge_overrides(
    *,
    settings: Any,
    judge_api_base: str,
    judge_api_key: str,
) -> None:
    if judge_api_key:
        settings.openrouter_api_key = judge_api_key
    if judge_api_base:
        settings.openrouter_base_url = judge_api_base


async def _parse_text_attachments(
    scorer: AsyncScorer,
    attachment_paths: List[str],
) -> List[Dict[str, str]]:
    attachments: List[Dict[str, str]] = []
    for path in attachment_paths:
        try:
            content = await scorer.attachment_parser.parse_single_attachment(path)
            attachments.append(
                {"filename": os.path.basename(path), "content": content}
            )
        except Exception as exc:
            attachments.append(
                {
                    "filename": os.path.basename(path),
                    "content": f"[Attachment parsing failed: {exc}]",
                }
            )
    return attachments


async def _parse_answer_attachments(
    scorer: AsyncScorer,
    attachment_paths: List[str],
):
    text_attachments: List[Dict[str, str]] = []
    image_attachments = []

    # AGENTIF_JUDGE_TEXT_ONLY=1: judge endpoint has no vision — parse every
    # answer attachment as text (HTML via html2text instead of a rendered
    # screenshot) and send no image parts, which a text-only API would reject.
    if os.environ.get("AGENTIF_JUDGE_TEXT_ONLY", "0") == "1":
        return await _parse_text_attachments(scorer, attachment_paths), []

    for path in attachment_paths:
        try:
            result = await scorer.attachment_parser.parse_attachment_with_screenshots(
                path
            )
            if result.text_content:
                text_attachments.append(
                    {
                        "filename": os.path.basename(path),
                        "content": result.text_content,
                    }
                )
            for screenshot in result.screenshots:
                image_attachments.append(
                    {
                        "filename": screenshot.get(
                            "filename", os.path.basename(path)
                        ),
                        "mime_type": screenshot.get("mime_type", "image/png"),
                        "base64_data": screenshot.get("base64_data", ""),
                    }
                )
        except Exception as exc:
            text_attachments.append(
                {
                    "filename": os.path.basename(path),
                    "content": f"[Attachment parsing failed: {exc}]",
                }
            )

    from .llm.schemas.types import ImageAttachment

    return text_attachments, [
        ImageAttachment(**image_attachment)
        for image_attachment in image_attachments
        if image_attachment["base64_data"]
    ]
