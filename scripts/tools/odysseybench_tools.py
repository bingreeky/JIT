"""OdysseyBench tools adapted for local workspace mode."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from .base import Tool
from .officebench_tools import _OfficeBenchExecutor


APP_INTROS: Dict[str, str] = {
    "calculator": "calculator: an app to evaluate mathematical expressions.",
    "calendar": "calendar: an app to manage daily events on calendar.",
    "email": "email: an app to manage emails, such as sending and reading emails.",
    "excel": "excel: an app to manipulate excel files, including reading, writing, etc.",
    "llm": "llm: an app to interact with the large language model to answer questions, generate text, etc.",
    "ocr": "ocr: an app to recognize text from images.",
    "pdf": "pdf: an app to manipulate pdf files, including format conversion and file reading.",
    "shell": "shell: an app to run shell commands in the system.",
    "system": "system: used when you want to switch between apps.",
    "word": "word: an app to manipulate word files, including reading, writing, converting, etc.",
}

APP_INSTRUCTIONS: Dict[str, List[str]] = {
    "calculator": [
        'Command to perform function: calculate:\ncalculate a mathematical expression written in python syntax. Any math module functions are allowed.\n{"app": "calculator", "action": "calculate", "args": {"expression": [MATH_EXPRESSION_TO_EVALUATE]}}'
    ],
    "calendar": [
        'Command to perform function: create_event:\ncreate a new event to a user\'s calendar where the time format is \'%Y-%m-%d %H:%M:%S\':{"app": "calendar", "action": "create_event", "args": {"user": [USER_NAME], "summary": [EVENT_SUMMARY], "time_start": [EVENT_START_TIME], "time_end": [EVENT_END_TIME]}}',
        'Command to perform function: delete_event:\ndelete an event from a user\'s calendar given the event summary:{"app": "calendar", "action": "delete_event", "args": {"user": [USER_NAME], "summary": [EVENT_SUMMARY]}}',
        'Command to perform function: list_events:\nlist all events from a user\'s calendar: {"app": "calendar", "action": "list_events", "args": {"username": [USER_NAME]}}',
    ],
    "email": [
        'Command to perform function: list_emails:\nList emails for a given username: {"app": "email", "action": "list_emails", "args": {"username": [USER_NAME]}}',
        'Command to perform function: read_email:\nRead a user\'s email by the given Email ID: {"app": "email", "action": "read_email", "args": {"username": [USERNAME], "email_id": [EMAIL_ID]}}',
        'Command to perform function: send_email:\nSend an email to a recipient: {"app": "email", "action": "send_email", "args": {"sender": [SENDER], "recipient": [RECIPIENT], "subject": [SUBJECT], "content": [CONTENT]}}',
    ],
    "excel": [
        'Command to perform function: create_new_file:\ncreate a new excel file: {"app": "excel", "action": "create_new_file", "args": {"file_path": [THE_PATH_TO_THE_NEW_EXCEL_FILE]}}',
        'Command to perform function: read_file:\nread an excel file: {"app": "excel", "action": "read_file", "args": {"file_path": [THE_PATH_TO_THE_EXCEL_FILE]}}',
        'Command to perform function: set_cell:\nwrite text to a cell in the excel file: {"app": "excel", "action": "set_cell", "args": {"file_path": [THE_PATH_TO_THE_EXCEL_FILE], "row_idx": [THE_ROW_INDEX], "column_idx": [THE_COLUMN_INDEX], "text": [THE_TEXT_TO_WRITE]}}',
        'Command to perform function: delete_cell:\ndelete a cell in the excel file: {"app": "excel", "action": "delete_cell", "args": {"file_path": [THE_PATH_TO_THE_EXCEL_FILE], "row_idx": [THE_ROW_INDEX], "column_idx": [THE_COLUMN_INDEX]}}',
        'Command to perform function: convert_to_pdf:\nconvert an excel document to a pdf: {"app": "excel", "action": "convert_to_pdf", "args": {"excel_file_path": [THE_PATH_TO_THE_EXCEL_FILE], "pdf_file_path": [THE_PATH_TO_THE_PDF_FILE]}}',
    ],
    "llm": [
        'Command to perform function: complete_text:\ncomplete text with an LLM: {"app": "llm", "action": "complete_text", "args": {"prompt": [PROMPT]}}'
    ],
    "ocr": [
        'Command to perform function: recognize_file:\nrecognize the text from an image file: {"app": "ocr", "action": "recognize_file", "args": {"file_path": [THE_PATH_TO_THE_IMAGE_FILE]}}'
    ],
    "pdf": [
        'Command to perform function: convert_to_pdf:\nconvert an image file to a pdf file: {"app": "pdf", "action": "image_convert_to_pdf", "args": {"image_file_path": [THE_PATH_TO_THE_IMAGE_FILE], "pdf_file_path": [THE_PATH_TO_THE_PDF_FILE]}}',
        'Command to perform function: convert_to_image:\nconvert a pdf file to an image file: {"app": "pdf", "action": "convert_to_image", "args": {"pdf_file_path": [THE_PATH_TO_THE_PDF_FILE], "image_file_path": [THE_PATH_TO_THE_IMAGE_FILE]}}',
        'Command to perform function: read_file:\nread a pdf file: {"app": "pdf", "action": "read_file", "args": {"pdf_file_path": [THE_PATH_TO_THE_PDF_FILE]}}',
        'Command to perform function: convert_to_word:\nconvert a pdf file to a word file: {"app": "pdf", "action": "convert_to_word", "args": {"pdf_file_path": [THE_PATH_TO_THE_PDF_FILE], "word_file_path": [THE_PATH_TO_THE_WORD_FILE]}}',
    ],
    "shell": [
        'Command to perform function: command:\nrun a shell command. Because the shell already starts inside the task testbed, prefer relative paths such as `data`, `emails`, and `calendar` instead of absolute `/testbed/...` paths.\n{"app": "shell", "action": "command", "args": {"command": [THE_COMMAND_YOU_WISH_TO_RUN]}}'
    ],
    "system": [
        'Command to perform function: switch_app:\nchoose an app from the available apps: {"app": "system", "action": "switch_app", "args": {"target_app": [THE_APP_YOU_CHOOSE]}}',
        'Command to perform function: copy:\nYou can copy text by calling `copy` with 1 argument.\n1. text: the text you want to copy.\n{"app": "system", "action": "copy", "args": {"text": ...}}',
        'Command to perform function: paste:\nYou can paste text that previous get copied to the clipboard by calling `paste` with 0 argument.\n{"app": "system", "action": "paste", "args": {}}',
        'Command to perform function: finish_task:\nUse this only when the task needs a textual answer file. The executor accepts a single `answer` field and writes it to `data/answer.txt` inside the current task testbed.\n{"app": "system", "action": "finish_task", "args": {"answer": "None"}}',
    ],
    "word": [
        'Command to perform function: create_new_file:\ncreate a new word file: {"app": "word", "action": "create_new_file", "args": {"file_path": [THE_PATH_TO_THE_NEW_WORD_FILE]}}',
        'Command to perform function: read_file:\nread the content of the word file: {"app": "word", "action": "read_file", "args": {"file_path": [THE_PATH_TO_THE_WORD_FILE]}}',
        'Command to perform function: write_to_file:\nwrite text to a word file: {"app": "word", "action": "write_to_file", "args": {"file_path": [THE_PATH_TO_THE_WORD_FILE], "contents": [THE_CONTENTS_YOU_WISH_TO_WRITE]}}',
        'Command to perform function: convert_to_pdf:\nconvert a word document to a pdf: {"app": "word", "action": "convert_to_pdf", "args": {"word_file_path": [THE_PATH_TO_THE_WORD_FILE], "pdf_file_path": [THE_PATH_TO_THE_PDF_FILE]}}',
    ],
}


def _format_app_introduction_block() -> str:
    lines = ["You have following apps installed in the system:"]
    for app_name in (
        "calculator",
        "calendar",
        "excel",
        "ocr",
        "pdf",
        "shell",
        "word",
        "email",
        "llm",
    ):
        lines.append(f" - {APP_INTROS[app_name]}")
    lines.append(" - system: used to switch between apps and to finish the task.")
    return "\n".join(lines)


def _format_current_app_instruction(app_name: str) -> str:
    instructions = APP_INSTRUCTIONS.get(app_name, [])
    if not instructions:
        return f"## How to use the {app_name} app:\n\nNo detailed instruction available."
    return f"## How to use the {app_name} app:\n\n" + "\n\n".join(instructions)


ODYSSEYBENCH_TOOL_DESCRIPTION = (
    "Execute a single OdysseyBench action in local workspace mode. "
    "Call format: {app: string, action: string, args: dict}. "
    "This tool follows the OdysseyBench app-based interaction style: first use system.switch_app to choose the current app, then call actions from that app. "
    "When you switch apps, the observation will include the detailed instruction for the selected app. "
    "Relative file paths inside args are resolved under /testbed; absolute /testbed/... paths are remapped to the current task workspace. "
    "For shell.command, prefer relative paths such as data/, emails/, and calendar/ because the shell already starts inside the task testbed. "
    "The tool always returns a single string observation.\n\n"
    + _format_app_introduction_block()
)

ODYSSEYBENCH_APP_INTRO_BLOCK = _format_app_introduction_block()

ODYSSEYBENCH_SWITCH_APP_GUIDANCE = (
    "Use `odysseybench_action` in an app-based way: first call "
    '`{"app": "system", "action": "switch_app", "args": {"target_app": "<app>"}}` '
    "to enter the right app, then call actions from that app. "
    "After switching, the observation will include detailed instructions for the selected app."
)


class _OdysseyBenchExecutor(_OfficeBenchExecutor):
    def __init__(self):
        super().__init__()
        self._action_handlers[("calculator", "calculate")] = self._calculator_calculate

    @staticmethod
    def _calculator_calculate(args: Dict[str, Any]) -> str:
        expression = str(args.get("expression", "")).strip()
        if not expression:
            return "OBSERVATION: Error calculating expression: empty expression"

        math_functions = {
            name: getattr(math, name)
            for name in dir(math)
            if not name.startswith("_") and callable(getattr(math, name))
        }
        math_constants = {
            name: getattr(math, name)
            for name in dir(math)
            if not name.startswith("_") and not callable(getattr(math, name))
        }
        safe_dict = {
            **math_functions,
            **math_constants,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "sorted": sorted,
            "len": len,
            "round": round,
        }

        expression = expression.replace("^", "**")
        blocked_tokens = (
            "import",
            "exec",
            "eval",
            "getattr",
            "setattr",
            "delattr",
            "compile",
            "open",
            "__",
            "globals",
            "locals",
            "subprocess",
        )
        if any(token in expression for token in blocked_tokens):
            return "OBSERVATION: Error calculating expression: Potentially unsafe operations detected in expression"

        try:
            result = eval(expression, {"__builtins__": {}}, safe_dict)  # noqa: S307
            return f"OBSERVATION: {result}"
        except Exception as exc:
            return f"OBSERVATION: Error calculating expression: {exc}"

    def execute(self, app: str, action: str, args: Dict[str, Any]) -> str:
        app = str(app)
        action = str(action)
        args = self._minor_action_fix(args or {})

        if action == "switch_app":
            app = "system"

        if app == "system" and action == "switch_app":
            target = str(args.get("target_app", "")).strip()
            if not target:
                return "Error: 'target_app' is required for switch_app."
            self.current_app = target
            available = sorted(APP_INSTRUCTIONS.get(target, []))
            if available:
                return (
                    f"Successfully switched to app: {target}. "
                    f"Available actions for {target} are shown below.\n"
                    f"{_format_current_app_instruction(target)}"
                )
            return f"Successfully switched to app: {target}"

        return super().execute(app=app, action=action, args=args)


class OdysseyBenchActionTool(Tool):
    name = "odysseybench_action"
    description = ODYSSEYBENCH_TOOL_DESCRIPTION
    inputs = {
        "app": {
            "type": "string",
            "description": "OdysseyBench app name, e.g. system/excel/word/pdf/email/calendar/shell/ocr/llm/calculator.",
        },
        "action": {
            "type": "string",
            "description": "Action name under the app.",
        },
        "args": {
            "type": "dict",
            "description": "Action arguments as a dictionary.",
        },
    }
    output_type = "string"

    def __init__(self, model=None):
        super().__init__()
        self._executor = _OdysseyBenchExecutor()

    @property
    def workspace(self) -> str:
        return self._executor.workspace

    def set_workspace(self, path: str) -> None:
        self._executor.set_workspace(path)

    def forward(self, app: str, action: str, args: Dict[str, Any]) -> str:
        if not isinstance(args, dict):
            return "Error: 'args' must be a dict."
        return self._executor.execute(app=app, action=action, args=args)


__all__ = [
    "ODYSSEYBENCH_APP_INTRO_BLOCK",
    "ODYSSEYBENCH_SWITCH_APP_GUIDANCE",
    "OdysseyBenchActionTool",
]
