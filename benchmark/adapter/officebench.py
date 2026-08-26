"""OfficeBench benchmark adapter (workspace mode, no Docker)."""

import copy
import json
import logging
import os
import shutil
from typing import Any, Dict, List

from .base import BenchmarkAdapter
from .officebench_eval.officebench_eval import evaluate_officebench_task

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class OfficeBenchAdapter(BenchmarkAdapter):
    """Adapter for OfficeBench tasks with original app/action semantics."""

    def __init__(self, workspace_base: str = ""):
        self._workspace_base = workspace_base or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            ".runtime",
            "workspace",
            "officebench",
        )
        self._dataset_root = ""

    def load_dataset(self, path: str) -> List[dict]:
        data_file = path
        if os.path.isdir(path):
            data_file = os.path.join(path, "data.jsonl")
            self._dataset_root = path
        else:
            self._dataset_root = os.path.dirname(os.path.abspath(path))

        if not os.path.isfile(data_file):
            raise FileNotFoundError(f"OfficeBench data.jsonl not found: {data_file}")

        items: List[dict] = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))

        for item in items:
            item.setdefault("question", item.get("task", ""))
            item.setdefault("answer", "")

        logger.info("Loaded %s OfficeBench items from %s", len(items), data_file)
        return items

    def _prepare_workspace_testbed(self, item: Dict[str, Any]) -> str:
        task_id = item.get("task_id", "unknown")
        subtask_id = item.get("subtask_id", "0")
        workspace = os.path.join(self._workspace_base, task_id, subtask_id)
        testbed_dir = os.path.join(workspace, "testbed")

        self._reset_workspace(workspace)

        testbed_template_rel = item.get("testbed_template_dir", "")
        testbed_template_abs = os.path.join(self._dataset_root, testbed_template_rel)
        if testbed_template_rel and os.path.isdir(testbed_template_abs):
            shutil.copytree(testbed_template_abs, testbed_dir)
        else:
            logger.warning(
                "OfficeBench testbed template missing for task %s/%s: %s; creating an empty testbed instead",
                task_id,
                subtask_id,
                testbed_template_abs,
            )
            os.makedirs(testbed_dir, exist_ok=True)
            for dirname in ("data", "emails", "calendar"):
                os.makedirs(os.path.join(testbed_dir, dirname), exist_ok=True)

        item["_workspace"] = workspace
        item["_testbed_dir"] = testbed_dir
        return workspace

    def _get_original_output_dir(self, item: Dict[str, Any]) -> str:
        task_id = item.get("task_id", "unknown")
        subtask_id = item.get("subtask_id", "0")
        return os.path.join(self._workspace_base, task_id, subtask_id, "output")

    def _sync_testbed_to_original_output(self, item: Dict[str, Any]) -> str:
        testbed_dir = item.get("_testbed_dir", "")
        if not testbed_dir or not os.path.isdir(testbed_dir):
            return ""

        output_dir = self._get_original_output_dir(item)
        output_testbed_dir = os.path.join(output_dir, "testbed")

        if os.path.exists(output_testbed_dir):
            shutil.rmtree(output_testbed_dir)
        os.makedirs(output_dir, exist_ok=True)
        shutil.copytree(testbed_dir, output_testbed_dir)
        item["_output_testbed_dir"] = output_testbed_dir
        return output_testbed_dir

    def format_task(self, item: dict) -> str:
        """Short task description for harness generation.

        Includes tool usage conventions, evaluation hints, and coding guidelines
        so the meta-agent can generate harness code that correctly calls
        officebench_action and focuses on file-level artifact manipulation.
        """
        workspace = self._prepare_workspace_testbed(item)
        testbed_dir = item.get("_testbed_dir", os.path.join(workspace, "testbed"))

        username = item.get("username", "")
        date = item.get("date", "")
        weekday = item.get("weekday", "")
        time = item.get("time", "")
        question = item.get("question", "")

        return (
            f"Today is {date} ({weekday}). The current time is {time}. "
            f"You are an AI assistant for user {username}.\n"
            "You are solving an OfficeBench task.\n\n"
            f"Task: {question}\n\n"
            "You have access to two tools: `officebench_action` (for app-based operations "
            "like shell, calendar, email, excel, word, pdf, etc.) and `final_answer`.\n"
            "Every action must be called through `officebench_action` using JSON of the form "
            '`{"app": "...", "action": "...", "args": {...}}`.\n'
            'Example: `{"app": "shell", "action": "command", "args": {"command": "ls data"}}`.\n'
            "Use `officebench_action` in an app-based way: first switch to the target app, "
            "then call actions from that app.\n\n"
            "Important: OfficeBench tasks are primarily evaluated by checking the final state "
            "of files and artifacts inside the testbed, not by judging the text of final_answer. "
            "The agent you design should complete the task by creating or modifying files, "
            "spreadsheets, documents, emails, calendars, or other task artifacts in the testbed "
            "before calling final_answer.\n\n"
            "Input files are typically under `data/`, `emails/`, and `calendar/` inside "
            "the testbed. All outputs must be written inside this testbed directory structure.\n\n"
            "Coding guidelines for the generated Python modules:\n"
            "- When rendering prompt templates that contain JSON examples (with `{` and `}`), "
            "use Jinja2 (via `jinja2.Template`) rather than Python's `str.format()`, "
            "because JSON braces will clash with `.format()` placeholders and cause KeyError.\n"
            "- Write raw Python source code directly into each output file. "
            "Do NOT wrap the code in Markdown fences (```python ... ```).\n"
            "- Ensure all imports (e.g. StepRecord, PlanState, SummaryState, MemoryView) "
            "are explicitly imported at the top of each module.\n\n"
            f"Task testbed root: {testbed_dir}\n"
            "All inputs are under this testbed. Prefer relative paths. "
            "Do not read or write outside this testbed directory structure."
        )

    def get_runtime_task(self, item: dict) -> str:
        """Full task description for agent execution (includes tool usage conventions)."""
        workspace = item.get("_workspace", "")
        testbed_dir = item.get("_testbed_dir", "")

        username = item.get("username", "")
        date = item.get("date", "")
        weekday = item.get("weekday", "")
        time = item.get("time", "")
        question = item.get("question", "")

        if not workspace or not testbed_dir:
            workspace = self._prepare_workspace_testbed(item)
            testbed_dir = item.get("_testbed_dir", os.path.join(workspace, "testbed"))

        return (
            f"Today is {date} ({weekday}). The current time is {time}. "
            f"You are an AI assistant for user {username}.\n"
            "You are solving an OfficeBench task.\n\n"
            f"Task: {question}\n\n"
            "You only have two exposed tools in this environment: `officebench_action` and `final_answer`.\n"
            "Every OfficeBench action must be called through `officebench_action` using JSON of the form "
            '`{"app": "...", "action": "...", "args": {...}}`.\n'
            'Example: `{"app": "shell", "action": "command", "args": {"command": "ls data"}}`.\n\n'
            "Important: OfficeBench tasks are primarily evaluated by checking the final state of files and other artifacts inside the testbed, "
            "not by judging the text of your final_answer.\n"
            "This means your answer should usually be carried out by creating or modifying files, spreadsheets, documents, emails, calendars, "
            "or other task artifacts in the testbed.\n"
            "Before using final_answer, you should try to complete the task through concrete changes in the testbed whenever the task allows it.\n"
            f"Your per-task workspace is: {workspace}\n"
            f"Task testbed root is: {testbed_dir}\n"
            "You can find files for your task in `data/` (the local workspace equivalent of upstream `/testbed/data`). "
            "If you do not know the filenames, use the shell app to list that directory.\n"
            "When you pass file paths to apps, prefer relative paths such as `data/...`, `emails/...`, and `calendar/...`.\n"
            "When you use shell commands, prefer relative paths like `ls data`, `find data -name '*.xlsx'`, or `cp data/file.xlsx data/temp.xlsx`.\n"
            "Do not assume `/testbed/...` is a reliable shell path. If needed, use the explicit task testbed root shown above.\n"
            "If the task asks for a textual answer file, use "
            '`{"app": "system", "action": "finish_task", "args": {"answer": "<answer text>"}}` '
            "to write `data/answer.txt`, then call `final_answer`.\n"
            "Input files are usually under `data`, `emails`, and `calendar` inside the task testbed.\n"
            "All outputs must be written inside this testbed directory structure.\n"
            "IMPORTANT (output location): Unless the task explicitly states a different path, every NEW "
            "file you produce (Word/Excel/PDF/text/image files, new subdirectories, etc.) MUST be created "
            "under the `data/` directory using a relative path such as `data/<filename>` "
            "(e.g. `data/abstract.docx`, `data/new_dir/file1.docx`) — the SAME directory the input files "
            "live in. Never write new output files to the testbed root. (Calendar and email artifacts are "
            "handled by the calendar/email apps and are stored under `calendar/` and `emails/` automatically.)\n"
            "Use final_answer only after you have finished the necessary modifications or creations in the testbed."
        )

    def get_workspace(self, item: dict) -> str:
        return item.get("_workspace", "")

    def get_tools(self) -> List[str]:
        return [
            "officebench_action",
            "final_answer",
        ]

    def _resolve_evaluation_path(self, item: Dict[str, Any], path: str) -> str:
        if not path or os.path.isabs(path):
            return path

        normalized = str(path).replace("\\", "/")
        task_dir = str(
            item.get("task_dir")
            or os.path.join("tasks", str(item.get("task_id", "")))
        )
        task_root = os.path.abspath(os.path.join(self._dataset_root, task_dir))

        reference_prefix = "../../../../reference/"
        if normalized.startswith(reference_prefix):
            suffix = normalized[len(reference_prefix):]
            return os.path.relpath(
                os.path.join(task_root, "reference", *suffix.split("/")),
                PROJECT_ROOT,
            )

        cache_prefix = "../../../../cache/"
        if normalized.startswith(cache_prefix):
            suffix = normalized[len(cache_prefix):]
            parts = suffix.split("/")
            if "testbed" in parts:
                testbed_index = parts.index("testbed")
                rest = parts[testbed_index + 1:]
                return os.path.relpath(
                    os.path.join(task_root, "testbed", *rest),
                    PROJECT_ROOT,
                )

        return path

    def _resolve_evaluation_items(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        evaluation_items = copy.deepcopy(item.get("evaluation", []) or [])
        for eval_item in evaluation_items:
            args = eval_item.get("args")
            if not isinstance(args, dict):
                continue
            for key, value in list(args.items()):
                if isinstance(value, str):
                    args[key] = self._resolve_evaluation_path(item, value)
        return evaluation_items

    def evaluate(self, prediction: str, ground_truth: Any, **kwargs) -> dict:
        item = kwargs.get("item", {})
        testbed_dir = self._sync_testbed_to_original_output(item) or item.get("_testbed_dir", "")
        evaluation_items = self._resolve_evaluation_items(item)
        if not testbed_dir:
            return {
                "score": 0.0,
                "is_pass": False,
                "actual_score": 0,
                "max_score": 1,
                "percentage": 0.0,
                "criteria_results": [],
            }
        result = evaluate_officebench_task(testbed_dir, evaluation_items)
        if item.get("_output_testbed_dir"):
            result["output_testbed_dir"] = item["_output_testbed_dir"]
        return result
