"""
DeepSearchQA benchmark adapter.

DeepSearchQA is a 900-prompt factuality benchmark from Google DeepMind
for evaluating deep research agents across 17 domains.

Dataset format (CSV):
    problem, problem_category, answer, answer_type

answer_type is either "Single Answer" or "Set Answer".
- Single Answer: binary match (F1 = 0 or 1)
- Set Answer: compute precision, recall, F1 over item sets

Evaluation uses LLM-as-a-Judge (Gemini) for semantic matching.

Dataset source: https://huggingface.co/datasets/google/deepsearchqa
"""

import csv
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .base import BenchmarkAdapter
from .deepsearchqa_eval import evaluate_deepsearchqa_prediction

logger = logging.getLogger(__name__)


class DeepSearchQAAdapter(BenchmarkAdapter):
    """Adapter for the DeepSearchQA benchmark.

    Args:
        judge_model: Model name for the LLM judge.
    """

    def __init__(
        self,
        judge_model: str = "qwen/qwen3.5-122b-a10b",
        judge_api_base: str = "",
        judge_api_key: str = "",
        workspace_base: str = "",
    ):
        self._judge_model = judge_model
        self._judge_api_base = judge_api_base
        self._judge_api_key = judge_api_key
        self._workspace_base = workspace_base or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            ".runtime",
            "workspace",
        )

    # ── Loading ───────────────────────────────────────────────────────

    def load_dataset(self, path: str) -> List[dict]:
        """Load DeepSearchQA from CSV or JSONL.

        Supports:
        - Direct CSV file (DSQA-full.csv from HuggingFace)
        - JSONL file (if pre-converted)
        - Directory containing either format
        """
        items: List[dict] = []

        if os.path.isdir(path):
            # Find data file in directory
            for ext in [".csv", ".jsonl"]:
                candidates = [f for f in os.listdir(path) if f.endswith(ext)]
                if candidates:
                    path = os.path.join(path, candidates[0])
                    break
            else:
                raise FileNotFoundError(
                    f"No CSV or JSONL files found in {path}"
                )

        if path.endswith(".csv"):
            items = self._load_csv(path)
        elif path.endswith(".jsonl"):
            items = self._load_jsonl(path)
        else:
            # Try CSV first, then JSONL
            try:
                items = self._load_csv(path)
            except Exception:
                items = self._load_jsonl(path)

        # Normalise field names
        for item in items:
            if "question" not in item:
                item["question"] = item.get("problem", "")
            if "answer" not in item and "answer" in item:
                pass  # already exists
            item.setdefault("answer_type", "Single Answer")

        logger.info(
            f"Loaded {len(items)} DeepSearchQA items from {path} "
            f"({sum(1 for i in items if i.get('answer_type') == 'Set Answer')} set-answer, "
            f"{sum(1 for i in items if i.get('answer_type') == 'Single Answer')} single-answer)"
        )
        return items

    @staticmethod
    def _load_csv(path: str) -> List[dict]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(dict(row))
        return items

    @staticmethod
    def _load_jsonl(path: str) -> List[dict]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    # ── Task formatting ───────────────────────────────────────────────

    def format_task(self, item: dict) -> str:
        """Format a DeepSearchQA problem as an agent task."""
        problem = item.get("problem", item.get("question", ""))
        answer_type = item.get("answer_type", "Single Answer")

        question_id = str(item.get("question_id", "")).strip()
        if not question_id:
            question_id = hashlib.sha1(problem.encode("utf-8")).hexdigest()[:12]

        workspace = os.path.join(self._workspace_base, "deepsearchqa", question_id)
        self._reset_workspace(workspace)
        item["_workspace"] = workspace

        task = (
            f"{problem}\n\n"
            "## Workspace\n"
            f"You are currently in the task workspace: `{workspace}`.\n"
            "When using execute_code, the current working directory is this workspace.\n"
            "If you create or download files, save them in this workspace "
            f"(for example: `{workspace}/notes.txt`).\n"
            "Use relative paths when reading/writing files, such as `notes.txt`.\n"
            "Do not assume paths are relative to the repository root."
        )

        # Add hints about expected answer format
        if answer_type == "Set Answer":
            task += (
                "\n\nNote: This question may require listing multiple items. "
                "Please provide a comprehensive answer including all relevant items."
            )

        return task

    def get_workspace(self, item: dict) -> str:
        return item.get("_workspace", "")

    # ── Tools ─────────────────────────────────────────────────────────

    def get_tools(self) -> List[str]:
        """DeepSearchQA toolset with web + file-capable workspace tools."""
        return [
            "web_search",
            "crawl_page",
            "execute_code",
            "inspect_file_as_text",
            "inspect_file_as_image",
            "final_answer",
        ]

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> dict:
        """LLM-based evaluation with Precision/Recall/F1 for set answers.

        For Single Answer: F1 equals exact match accuracy (0 or 1).
        For Set Answer: F1 penalises both under-retrieval and hallucinations.
        """
        question = kwargs.get("question", kwargs.get("problem", ""))
        answer_type = kwargs.get("answer_type", "Single Answer")
        return evaluate_deepsearchqa_prediction(
            prediction,
            ground_truth,
            question=question,
            answer_type=answer_type,
            judge_model=self._judge_model,
            judge_api_base=self._judge_api_base,
            judge_api_key=self._judge_api_key,
        )
