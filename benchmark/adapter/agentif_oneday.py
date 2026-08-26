"""
AgentIF-OneDay benchmark adapter.

This adapter now delegates scoring to the migrated official
llm_as_judge implementation under benchmark.adapter.agentif_oneday_eval.
"""

import json
import logging
import os
import shutil
from typing import List

from .base import BenchmarkAdapter

logger = logging.getLogger(__name__)


def _collect_workspace_output_paths(workspace: str) -> List[str]:
    """Collect user-created files from the workspace output directory recursively."""
    if not workspace:
        return []

    output_dir = os.path.join(workspace, "output")
    if not os.path.isdir(output_dir):
        return []

    output_paths: List[str] = []
    for root, dirnames, filenames in os.walk(output_dir):
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if not dirname.startswith(".") and not dirname.startswith("_script_")
        )
        for filename in sorted(filenames):
            if filename.startswith(".") or filename.startswith("_script_"):
                continue
            path = os.path.join(root, filename)
            if os.path.isfile(path):
                output_paths.append(path)
    return output_paths


class AgentIFOneDayAdapter(BenchmarkAdapter):
    """Adapter for the AgentIF-OneDay benchmark."""

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
        self._data_dir = ""

    def load_dataset(self, path: str) -> List[dict]:
        """Load AgentIF-OneDay dataset from the data directory."""
        self._data_dir = path
        data_file = os.path.join(path, "data.jsonl")

        if not os.path.isfile(data_file):
            raise FileNotFoundError(f"data.jsonl not found at: {data_file}")

        items = []
        with open(data_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))

        for item in items:
            item["question"] = item.get("description", "")
            item["answer"] = item.get("reference_answer_description", "")
            item["_attachment_paths"] = self._resolve_paths(
                item.get("attachment_filenames", []), "Questions"
            )
            item["_reference_paths"] = self._resolve_paths(
                item.get("reference_answer_attachment_filenames", []),
                "Reference_answer",
            )

        logger.info("Loaded %s AgentIF-OneDay tasks from %s", len(items), data_file)
        return items

    def _resolve_paths(self, filenames: List[str], subdir: str) -> List[str]:
        paths = []
        base_dir = os.path.join(self._data_dir, "Attachments", subdir)
        for filename in filenames:
            path = os.path.join(base_dir, filename)
            if os.path.exists(path):
                paths.append(path)
            else:
                logger.debug("Attachment not found: %s", path)
        return paths

    def format_task(self, item: dict) -> str:
        """Format task with title, description, input files, and workspace info."""
        question_id = item.get("question_id", "unknown")

        workspace = os.path.join(self._workspace_base, "agentif", question_id)
        self._reset_workspace(workspace)
        os.makedirs(os.path.join(workspace, "output"), exist_ok=True)
        item["_workspace"] = workspace

        workspace_input_paths = self._prepare_workspace_inputs(
            item.get("_attachment_paths", []), workspace
        )
        item["_workspace_input_paths"] = workspace_input_paths

        parts = [
            f"# Task: {item.get('title', '')}",
            "",
            item.get("description", ""),
        ]

        if workspace_input_paths:
            parts.append("\n## Input Files")
            parts.append(
                "The following input files have been copied into your task workspace. "
                "When using execute_code, your current working directory is this workspace, "
                "so use the relative paths shown below to find these files. "
                "Use 'inspect_file_as_text' or 'inspect_file_as_image' to read them."
            )
            for rel_path in workspace_input_paths:
                actual_name = os.path.basename(rel_path)
                parts.append(f"- **{actual_name}**: `{rel_path}`")

        relative_path_hint = (
            f"Prefer workspace-relative paths such as `scratch_notes.md`, "
            f"`tmp_table.csv`, `output/report.md`, "
            f"`output/chart.png`, or "
            f"`{workspace_input_paths[0]}` for reading copied inputs.\n"
            if workspace_input_paths
            else "Prefer workspace-relative paths such as `scratch_notes.md`, `tmp_table.csv`, `output/report.md` or `output/chart.png`.\n"
        )
        parts.append("\n## Output Directory")
        parts.append(
            f"Your task workspace is: `{workspace}`\n"
            f"When using execute_code, the current working directory is this workspace.\n"
            f"Store intermediate artifacts in the workspace top level "
            f"(e.g., `{workspace}/scratch_notes.md`, `{workspace}/tmp_table.csv`).\n"
            f"Store final deliverables in `output/` under this workspace "
            f"(e.g., `{workspace}/output/final_report.md`, `{workspace}/output/final_chart.png`). "
            f"{relative_path_hint}"
            f"Do not assume paths are relative to the repository root.\n"
            f"After creating files, describe what you created in your final answer."
        )

        return "\n".join(parts)

    def _prepare_workspace_inputs(
        self, attachment_paths: List[str], workspace: str
    ) -> List[str]:
        if not attachment_paths:
            return []

        inputs_dir = os.path.join(workspace, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)

        relative_paths = []
        for src_path in attachment_paths:
            basename = os.path.basename(src_path)
            actual_name = basename.split(" + attachment + ", 1)[-1]
            dest_path = os.path.join(inputs_dir, actual_name)
            shutil.copy2(src_path, dest_path)
            relative_paths.append(os.path.join("inputs", actual_name))

        return relative_paths

    def get_workspace(self, item: dict) -> str:
        return item.get("_workspace", "")

    def get_tools(self) -> List[str]:
        return [
            "web_search",
            "crawl_page",
            "execute_code",
            "inspect_file_as_text",
            "inspect_file_as_image",
            "final_answer",
        ]

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> dict:
        """Evaluate using the migrated official llm_as_judge pipeline."""
        from .agentif_oneday_eval.workspace_eval import evaluate_workspace_task

        item = kwargs.get("item", {})
        criteria = item.get("score_criteria", [])

        if not criteria:
            logger.warning("No scoring criteria for %s", item.get("question_id", "?"))
            return {
                "question_id": item.get("question_id", "unknown"),
                "method": self._judge_model,
                "criteria_results": [],
            }

        for index, criterion in enumerate(criteria, 1):
            if "criterion_id" not in criterion:
                criterion["criterion_id"] = f"criterion_{index}"

        return evaluate_workspace_task(
            question_id=item.get("question_id", "unknown"),
            title=item.get("title", ""),
            description=item.get("description", ""),
            prediction=str(prediction),
            score_criteria=criteria,
            question_attachment_paths=item.get("_attachment_paths", []),
            reference_answer_description=item.get(
                "reference_answer_description", ""
            ),
            reference_attachment_paths=item.get("_reference_paths", []),
            answer_attachment_paths=_collect_workspace_output_paths(
                item.get("_workspace", "")
            ),
            judge_model=self._judge_model,
            judge_api_base=self._judge_api_base,
            judge_api_key=self._judge_api_key,
        )
