"""
LLM Auto Scoring Script
"""
__version__ = "1.0.0"

from .config import get_settings, Settings
from .logging_config import setup_logging, get_logger
from .data_loader import DataLoader, AttachmentLoader, Question, Answer, ScoreResult
from .scorer import AsyncScorer, ScoringProgress
from .workspace_eval import (
    collect_workspace_output_paths,
    evaluate_workspace_task,
    evaluate_workspace_task_async,
)

__all__ = [
    "get_settings",
    "Settings",
    "setup_logging",
    "get_logger",
    "DataLoader",
    "AttachmentLoader",
    "Question",
    "Answer",
    "ScoreResult",
    "AsyncScorer",
    "ScoringProgress",
    "collect_workspace_output_paths",
    "evaluate_workspace_task",
    "evaluate_workspace_task_async",
]
