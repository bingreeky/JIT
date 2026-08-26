"""OdysseyBench benchmark adapter (workspace mode, no Docker)."""

from __future__ import annotations

import copy
import glob
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Union

from .base import BenchmarkAdapter
from .odysseybench_eval import evaluate_odysseybench_task
from scripts.tools.odysseybench_tools import (
    ODYSSEYBENCH_APP_INTRO_BLOCK,
    ODYSSEYBENCH_SWITCH_APP_GUIDANCE,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class OdysseyBenchAdapter(BenchmarkAdapter):
    """Adapter for OdysseyBench+ and OdysseyBench-Neo tasks in local workspace mode."""

    def __init__(
        self,
        workspace_base: str = "",
        memory_mode: str = "raw_chat",
        rag_mode: str = "dialogutterance",
        top_k: int = 5,
        embedding_model: Union[str, Dict[str, Any]] = "qwen/qwen3-embedding-8b",
        embedding_api_base: str = "",
        embedding_api_key: str = "",
        embedding_api_key_env: str = "OPENROUTER_API_KEY",
        override_task_with_query: bool = True,
        subset: str = "both",
    ):
        self._workspace_base = workspace_base or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            ".runtime",
            "workspace",
            "odysseybench",
        )
        self._dataset_root = ""
        self.memory_mode = str(memory_mode or "raw_chat").strip().lower()
        self.rag_mode = str(rag_mode or "dialogutterance").strip().lower()
        self.top_k = max(1, int(top_k))
        subset_val = str(subset or "both").strip().lower()
        if subset_val not in ("plus", "neo", "both"):
            raise ValueError(f"subset must be 'plus', 'neo', or 'both', got '{subset}'")
        self.subset = subset_val
        embedding_cfg: Dict[str, Any] = {}
        if isinstance(embedding_model, dict):
            embedding_cfg = embedding_model
            self.embedding_model = str(
                embedding_cfg.get("model_id")
                or embedding_cfg.get("model")
                or "qwen/qwen3-embedding-8b"
            )
        else:
            self.embedding_model = str(embedding_model or "qwen/qwen3-embedding-8b")

        self.embedding_api_base = (
            str(embedding_cfg.get("api_base", "")) if embedding_cfg else ""
        ) or embedding_api_base or os.getenv("OPENAI_API_BASE") or os.getenv("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"
        self.embedding_api_key = (
            (str(embedding_cfg.get("api_key", "")) if embedding_cfg else "")
            or embedding_api_key
            or os.getenv(embedding_api_key_env, "")
            or os.getenv("OPENROUTER_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
        )
        self.override_task_with_query = bool(override_task_with_query)

    @staticmethod
    def _task_sort_key(task_id: str) -> tuple[int, int]:
        first, second = task_id.split("-")
        return int(first), int(second)

    @staticmethod
    def _subtask_sort_key(path: str) -> int:
        return int(os.path.splitext(os.path.basename(path))[0])

    def load_dataset(self, path: str) -> List[dict]:
        dataset_root = os.path.abspath(path)
        tasks_root = os.path.join(dataset_root, "tasks")
        if not os.path.isdir(tasks_root):
            raise FileNotFoundError(f"OdysseyBench tasks directory not found: {tasks_root}")

        self._dataset_root = dataset_root
        items: List[dict] = []

        task_ids = sorted(
            [entry for entry in os.listdir(tasks_root) if "-" in entry and os.path.isdir(os.path.join(tasks_root, entry))],
            key=self._task_sort_key,
        )

        for task_id in task_ids:
            task_dir = os.path.join(tasks_root, task_id)
            tracks = ("plus", "neo") if self.subset == "both" else (self.subset,)
            for track in tracks:
                subtasks_dir = os.path.join(task_dir, f"subtasks_{track}")
                if not os.path.isdir(subtasks_dir):
                    continue

                for config_path in sorted(glob.glob(os.path.join(subtasks_dir, "*.json")), key=self._subtask_sort_key):
                    with open(config_path, "r", encoding="utf-8") as f:
                        item = json.load(f)

                    subtask_id = os.path.splitext(os.path.basename(config_path))[0]
                    chat_history_path = os.path.join(
                        task_dir,
                        f"chat_histories_{track}",
                        f"{subtask_id}_day_session.json",
                    )

                    item["task_id"] = task_id
                    item["subtask_id"] = subtask_id
                    item["track"] = track
                    item["config_path"] = config_path
                    item["chat_history_path"] = chat_history_path
                    item["testbed_template_dir"] = os.path.join(task_dir, "testbed")
                    item["question"] = item.get("query_sentence") or item.get("task", "")
                    item["answer"] = ""
                    item["question_id"] = f"{task_id}:{track}:{subtask_id}"
                    items.append(item)

        logger.info("Loaded %s OdysseyBench items from %s", len(items), dataset_root)
        return items

    def _prepare_workspace_testbed(self, item: Dict[str, Any]) -> str:
        task_id = item.get("task_id", "unknown")
        track = item.get("track", "plus")
        subtask_id = item.get("subtask_id", "0")
        workspace = os.path.join(self._workspace_base, track, task_id, subtask_id)
        testbed_dir = os.path.join(workspace, "testbed")

        self._reset_workspace(workspace)

        testbed_template_dir = item.get("testbed_template_dir", "")
        if testbed_template_dir and os.path.isdir(testbed_template_dir):
            shutil.copytree(testbed_template_dir, testbed_dir)
        else:
            logger.warning(
                "OdysseyBench testbed template missing for task %s/%s/%s: %s; creating an empty testbed instead",
                track,
                task_id,
                subtask_id,
                testbed_template_dir,
            )
            os.makedirs(testbed_dir, exist_ok=True)
            for dirname in ("data", "emails", "calendar"):
                os.makedirs(os.path.join(testbed_dir, dirname), exist_ok=True)

        item["_workspace"] = workspace
        item["_testbed_dir"] = testbed_dir
        return workspace

    def _get_output_dir(self, item: Dict[str, Any]) -> str:
        return os.path.join(item.get("_workspace", ""), "output")

    def _sync_testbed_to_output(self, item: Dict[str, Any]) -> str:
        testbed_dir = item.get("_testbed_dir", "")
        if not testbed_dir or not os.path.isdir(testbed_dir):
            return ""

        output_dir = self._get_output_dir(item)
        output_testbed_dir = os.path.join(output_dir, "testbed")
        if os.path.exists(output_testbed_dir):
            shutil.rmtree(output_testbed_dir)
        os.makedirs(output_dir, exist_ok=True)
        shutil.copytree(testbed_dir, output_testbed_dir)
        item["_output_testbed_dir"] = output_testbed_dir
        return output_testbed_dir

    @staticmethod
    def _format_memory(chat_history_path: str) -> str:
        if not os.path.isfile(chat_history_path):
            return ""

        with open(chat_history_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list) and payload:
            payload = payload[0]
        if not isinstance(payload, dict):
            return json.dumps(payload, ensure_ascii=False, indent=2)

        conversation = payload.get("conversation", {})
        if not isinstance(conversation, dict):
            return json.dumps(payload, ensure_ascii=False, indent=2)

        session_numbers = sorted(
            int(key.split("_")[-1])
            for key in conversation
            if key.startswith("session_") and not key.endswith("_date_time")
        )
        sessions: List[str] = []
        for session_num in session_numbers:
            dialog_lines = []
            for dialog in conversation.get(f"session_{session_num}", []):
                speaker = dialog.get("speaker", "Unknown")
                text = dialog.get("text", "")
                dialog_lines.append(f'{speaker} said, "{text}"')
            if dialog_lines:
                sessions.append("\n".join(dialog_lines))
        return "\n\n".join(sessions)

    @staticmethod
    def _load_chat_payload(chat_history_path: str) -> Any:
        if not os.path.isfile(chat_history_path):
            return None
        with open(chat_history_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _extract_dialog_sessions(chat_payload: Any) -> List[str]:
        if isinstance(chat_payload, list) and chat_payload:
            chat_payload = chat_payload[0]
        if not isinstance(chat_payload, dict):
            return []

        conversation = chat_payload.get("conversation", {})
        if not isinstance(conversation, dict):
            return []

        session_numbers = sorted(
            int(key.split("_")[-1])
            for key in conversation
            if key.startswith("session_") and not key.endswith("_date_time")
        )
        sessions: List[str] = []
        for session_num in session_numbers:
            dialogs = []
            for dialog in conversation.get(f"session_{session_num}", []):
                speaker = dialog.get("speaker", "Unknown")
                text = dialog.get("text", "")
                dialogs.append(f'{speaker} said, "{text}"')
            if dialogs:
                sessions.append("\n".join(dialogs))
        return sessions

    @staticmethod
    def _extract_dialog_utterances(chat_payload: Any) -> List[str]:
        utterances: List[str] = []
        for session in OdysseyBenchAdapter._extract_dialog_sessions(chat_payload):
            utterances.extend(line for line in session.split("\n") if line.strip())
        return utterances

    @staticmethod
    def _extract_clean_memory(item: Dict[str, Any]) -> str:
        clean_memory = item.get("ground_truth_memory")
        if clean_memory:
            return json.dumps(clean_memory, ensure_ascii=False, indent=2)
        return ""

    @staticmethod
    def _summary_from_session(session: str) -> str:
        lines = [line.strip() for line in session.split("\n") if line.strip()]
        if not lines:
            return ""
        if len(lines) <= 4:
            return "\n".join(lines)
        return "\n".join(lines[:2] + lines[-2:])

    @staticmethod
    def _summary_chunks_from_sessions(sessions: List[str], chunk_size: int = 2) -> List[str]:
        if not sessions:
            return []
        chunks: List[str] = []
        for idx in range(0, len(sessions), chunk_size):
            chunk = sessions[idx: idx + chunk_size]
            lines: List[str] = []
            for session in chunk:
                lines.extend(line.strip() for line in session.split("\n") if line.strip())
            if lines:
                if len(lines) > 8:
                    lines = lines[:4] + lines[-4:]
                chunks.append("\n".join(lines))
        return chunks

    def _get_embedding_client(self):
        if not self.embedding_api_key:
            return None
        try:
            from openai import OpenAI
        except Exception:
            return None
        return OpenAI(api_key=self.embedding_api_key, base_url=self.embedding_api_base)

    def _embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not texts:
            return None
        client = self._get_embedding_client()
        if client is None:
            return None
        try:
            response = client.embeddings.create(
                model=self.embedding_model,
                input=texts,
                encoding_format="float",
            )
            return [row.embedding for row in response.data]
        except Exception as exc:
            logger.warning("OdysseyBench embedding retrieval failed: %s", exc)
            return None

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a <= 1e-12 or norm_b <= 1e-12:
            return 0.0
        return dot / (norm_a * norm_b)

    def _select_benchmark_memory(self, item: Dict[str, Any]) -> str:
        mode = self.memory_mode
        chat_payload = self._load_chat_payload(item.get("chat_history_path", ""))
        query = item.get("query_sentence") or item.get("question", "")

        if mode == "no":
            return ""

        if mode == "clean":
            clean_memory = self._extract_clean_memory(item)
            if clean_memory:
                return clean_memory
            return self._format_memory(item.get("chat_history_path", ""))

        if mode == "raw_chat":
            return self._format_memory(item.get("chat_history_path", ""))

        if mode != "use_rag":
            logger.warning("Unknown OdysseyBench memory_mode '%s', falling back to raw_chat", mode)
            return self._format_memory(item.get("chat_history_path", ""))

        if self.rag_mode == "dialogsession":
            units = self._extract_dialog_sessions(chat_payload)
        elif self.rag_mode == "dialogutterance":
            units = self._extract_dialog_utterances(chat_payload)
        elif self.rag_mode == "summarysession":
            units = [
                self._summary_from_session(session)
                for session in self._extract_dialog_sessions(chat_payload)
                if session.strip()
            ]
        elif self.rag_mode == "summarychunk":
            units = self._summary_chunks_from_sessions(self._extract_dialog_sessions(chat_payload))
        else:
            logger.warning("Unknown OdysseyBench rag_mode '%s', falling back to dialogutterance", self.rag_mode)
            units = self._extract_dialog_utterances(chat_payload)

        units = [unit for unit in units if unit.strip()]
        if not units:
            return ""

        embeddings = self._embed_texts([query] + units)
        if not embeddings or len(embeddings) != len(units) + 1:
            logger.warning(
                "OdysseyBench RAG memory falling back to raw_chat because embeddings are unavailable. "
                "model=%s api_base=%s",
                self.embedding_model,
                self.embedding_api_base,
            )
            return self._format_memory(item.get("chat_history_path", ""))

        query_embedding = embeddings[0]
        scored_units = [
            (self._cosine_similarity(query_embedding, unit_embedding), unit)
            for unit_embedding, unit in zip(embeddings[1:], units)
        ]
        scored_units.sort(key=lambda row: row[0], reverse=True)
        selected_units = [unit for _, unit in scored_units[: self.top_k]]
        return "\n\n".join(selected_units)

    @staticmethod
    def _evaluation_uses_answer_file(evaluation_items: List[Dict[str, Any]]) -> bool:
        candidate_keys = {"file", "result_file", "output_file", "expected_file"}
        for item in evaluation_items:
            args = item.get("args", {})
            for key in candidate_keys:
                value = str(args.get(key, "")).replace("\\", "/").lower()
                if value.endswith("answer.txt") or value == "answer.txt":
                    return True
        return False

    @staticmethod
    def _write_prediction_to_answer_file(testbed_dir: str, prediction: str) -> None:
        answer_path = os.path.join(testbed_dir, "data", "answer.txt")
        os.makedirs(os.path.dirname(answer_path), exist_ok=True)
        if os.path.exists(answer_path):
            with open(answer_path, "r", encoding="utf-8") as f:
                if f.read().strip():
                    return
        with open(answer_path, "w", encoding="utf-8") as f:
            f.write(str(prediction or ""))

    def _format_task_common(self, item: dict) -> dict:
        """Return common fields for task formatting."""
        workspace = self._prepare_workspace_testbed(item)
        testbed_dir = item.get("_testbed_dir", os.path.join(workspace, "testbed"))
        return {
            "workspace": workspace,
            "testbed_dir": testbed_dir,
            "username": item.get("username", ""),
            "date": item.get("date", ""),
            "weekday": item.get("weekday", ""),
            "time": item.get("time", ""),
            "track": item.get("track", "plus"),
            "query_sentence": item.get("query_sentence") or item.get("question", ""),
            "task_description": item.get("task", ""),
        }

    def format_task(self, item: dict) -> str:
        """Short task description for harness generation (no chat memory, no execution protocol)."""
        c = self._format_task_common(item)
        primary_task = c["query_sentence"] if self.override_task_with_query and c["query_sentence"] else c["task_description"]

        return (
            f"Today is {c['date']} ({c['weekday']}). The current time is {c['time']}. "
            f"You are an AI assistant for user {c['username']}.\n"
            f"You are solving an OdysseyBench-{c['track']} task in local workspace mode.\n\n"
            f"Task: {primary_task}\n\n"
            "You have access to two tools: `odysseybench_action` (for app-based operations like "
            "shell, calendar, email, word, excel, pdf, etc.) and `final_answer`.\n"
            "Use `odysseybench_action` in an app-based way: first switch to the target app, "
            "then call actions from that app.\n\n"
            f"Task testbed root: {c['testbed_dir']}\n"
            "All inputs are under this testbed. Prefer relative paths. "
            "Do not read or write outside this testbed directory structure."
        )

    def get_runtime_task(self, item: dict) -> str:
        """Full task description for agent execution (includes chat memory and execution protocol)."""
        c = self._format_task_common(item)
        query_sentence = c["query_sentence"]
        task_description = c["task_description"]
        track = c["track"]
        workspace = c["workspace"]
        testbed_dir = c["testbed_dir"]

        evaluation_items = item.get("evaluation", [])
        answer_file_expected = self._evaluation_uses_answer_file(evaluation_items)
        memory = self._select_benchmark_memory(item)
        if not memory and self.memory_mode != "no":
            memory = "[No benchmark chat memory available after applying the selected memory mode.]"
        primary_task = query_sentence if self.override_task_with_query and query_sentence else task_description

        memory_block = ""
        if memory:
            memory_block = (
                f"Benchmark chat memory mode: {self.memory_mode}"
                + (f" ({self.rag_mode}, top_k={self.top_k})" if self.memory_mode == "use_rag" else "")
                + "\n"
                "Relevant benchmark memory:\n"
                f"{memory}\n\n"
            )

        if answer_file_expected:
            completion_guidance = (
                "5. This task is evaluated through an answer file. Use odysseybench_action with "
                '`{"app": "system", "action": "finish_task", "args": {"answer": "<your answer>"}}` '
                "to write the answer file inside the current task testbed, then call final_answer with the same answer to end the run.\n"
                "6. If the task also requires artifact changes beyond answer.txt, finish those changes first before ending the run.\n\n"
            )
        else:
            completion_guidance = (
                "5. Do not assume that writing answer.txt is sufficient. If the task requires updating "
                "a calendar, email, spreadsheet, document, PDF, or another artifact under /testbed, make those changes first.\n"
                "6. Only use system.finish_task / answer.txt when the task explicitly asks for a textual answer or the workflow clearly requires it.\n\n"
            )

        return (
            f"Today is {c['date']} ({c['weekday']}). The current time is {c['time']}. "
            f"You are an AI assistant for user {c['username']}.\n"
            f"You are solving an OdysseyBench-{track} task in local workspace mode.\n\n"
            "You can help solve the task step by step.\n"
            "You can interact with an operation system and use apps to solve the task.\n"
            "You must follow the instructions and use the given json format to call APIs.\n"
            "You can only generate one action at a time.\n"
            f"{ODYSSEYBENCH_SWITCH_APP_GUIDANCE}\n"
            f"{ODYSSEYBENCH_APP_INTRO_BLOCK}\n\n"
            f"Primary task to solve: {primary_task}\n"
            f"Primary query sentence: {query_sentence}\n"
            f"Original task description: {task_description}\n\n"
            f"{memory_block}"
            "These tasks are usually solved by creating or modifying concrete artifacts in the testbed, "
            "such as calendar events, emails, spreadsheets, documents, PDFs, or answer files, rather than by giving a natural-language reply alone.\n\n"
            "Important:\n"
            "1. The query sentence may omit details that must be recovered from benchmark chat memory when such memory is provided.\n"
            "2. You can find files for your task in the task testbed data directory. If you don't know the filenames, switch to the shell app and list `data`, `emails`, or `calendar` using relative paths from the current shell working directory.\n"
            "3. Safety: You may only manipulate or create files and folders under the `/testbed` directory for this task. Do not access unrelated data outside the task testbed.\n"
            "4. OdysseyBench is evaluated mainly by the final state of files and artifacts inside the testbed.\n"
            f"{completion_guidance}"
            f"Your per-task workspace is: {workspace}\n"
            f"Task testbed root is: {testbed_dir}\n"
            "When using app actions, relative paths are resolved inside the task testbed.\n"
            "When using shell commands, the shell starts inside the task testbed, so prefer relative paths like `data/...`, `emails/...`, and `calendar/...`.\n"
            "Input files are typically under testbed/data, testbed/emails, and testbed/calendar.\n"
            "IMPORTANT (output location): Unless the task explicitly states a different path, every NEW "
            "file you produce (Word/Excel/PDF/text/image files, new subdirectories, etc.) MUST be created "
            "under the `data/` directory using a relative path such as `data/<filename>` "
            "(e.g. `data/sorted_score.xlsx`, `data/new_dir/file1.docx`) — the SAME directory the input files "
            "live in. Never write new output files to the testbed root. (Calendar and email artifacts are "
            "handled by the calendar/email apps and are stored under `calendar/` and `emails/` automatically.)\n"
            "Do not read or write outside this testbed directory structure."
        )

    def get_workspace(self, item: dict) -> str:
        return item.get("_workspace", "")

    def get_tools(self) -> List[str]:
        return ["odysseybench_action", "final_answer"]

    def _resolve_evaluation_path(self, item: Dict[str, Any], path: str) -> str:
        if not path or os.path.isabs(path):
            return path

        normalized = str(path).replace("\\", "/")
        task_root = os.path.join(self._dataset_root, "tasks", str(item.get("task_id", "")))

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
            cache_path = os.path.join(task_root, "cache", *suffix.split("/"))
            if os.path.exists(cache_path):
                return os.path.relpath(cache_path, PROJECT_ROOT)

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
        testbed_dir = item.get("_testbed_dir", "")
        evaluation_items = self._resolve_evaluation_items(item)
        if self._evaluation_uses_answer_file(evaluation_items) and testbed_dir:
            self._write_prediction_to_answer_file(testbed_dir, prediction)

        synced_testbed_dir = self._sync_testbed_to_output(item) or testbed_dir
        if not synced_testbed_dir:
            return {
                "score": 0.0,
                "is_pass": False,
                "actual_score": 0,
                "max_score": 1,
                "percentage": 0.0,
                "criteria_results": [],
            }

        result = evaluate_odysseybench_task(synced_testbed_dir, evaluation_items)
        if item.get("_output_testbed_dir"):
            result["output_testbed_dir"] = item["_output_testbed_dir"]
        return result
