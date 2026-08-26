"""
Code execution tool for running Python and Bash code in a sandboxed environment.

Uses a dedicated conda environment and workspace directory for isolation.
"""

import logging
import os
import subprocess
import sys
import tempfile
import uuid
from typing import Optional

from .base import Tool

logger = logging.getLogger(__name__)

STDOUT_CHAR_LIMIT = int(os.getenv("EXECUTE_CODE_STDOUT_LIMIT", "30000"))
STDERR_CHAR_LIMIT = int(os.getenv("EXECUTE_CODE_STDERR_LIMIT", "10000"))


def _clip_stream(text: str, limit: int) -> str:
    """Clip an execute_code stream to `limit` chars, keeping head and tail."""
    if not text or len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    dropped = len(text) - limit
    return (
        f"{text[:head]}\n\n"
        f"[... {dropped} chars truncated ({len(text)} total); "
        f"print less or write to a file and inspect it ...]\n\n"
        f"{text[-tail:]}"
    )


def _resolve_sandbox_python() -> str:
    """Locate the Python interpreter for the code sandbox.

    The ``agent_sandbox`` conda env is not always at ``~/miniconda3/envs``: on
    some deployments the env is elsewhere or absent entirely. Probe the known
    locations in order -- ``AGENT_SANDBOX_PYTHON`` first, so any layout can be
    pointed at explicitly -- and fall back to the current interpreter so
    ``execute_code`` degrades gracefully instead of hard-failing with
    "No such file or directory".
    """
    candidates = []
    env_override = os.environ.get("AGENT_SANDBOX_PYTHON", "").strip()
    if env_override:
        candidates.append(env_override)
    candidates.append(
        os.path.join(
            os.path.expanduser("~"),
            "miniconda3", "envs", "agent_sandbox", "bin", "python",
        )
    )
    candidates.append(
        os.path.join(os.path.expanduser("~"), "anaconda3", "envs",
                     "agent_sandbox", "bin", "python")
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    logger.warning(
        "agent_sandbox python not found in %s; falling back to current "
        "interpreter %s",
        candidates, sys.executable,
    )
    return sys.executable

DISALLOWED_COMMANDS = [
    "rm -rf /",
    "rm -rf /*",
    "sudo ",
    "shutdown",
    "reboot",
    "chmod 777 /",
    "chown ",
    "> /dev/",
    "mkfs",
    "dd if=",
    "mount ",
    "umount ",
    "kill -9 1",
    ":(){ :|:& };:",
]


def _check_safety(code: str) -> Optional[str]:
    """Check code against blocklist. Returns error message if blocked, else None."""
    code_lower = code.lower()
    for pattern in DISALLOWED_COMMANDS:
        if pattern.lower() in code_lower:
            return f"Blocked: code contains disallowed pattern '{pattern}'"
    return None


class ExecuteCodeTool(Tool):
    """Execute Python or Bash code in a sandboxed environment.

    Code runs in a dedicated conda environment with its own workspace directory.
    Files created by the code are persisted in the workspace.
    """

    name = "execute_code"
    description = (
        "Execute Python or Bash code in a sandboxed environment. "
        "Use this to run computations, create files (Excel, charts, HTML, PDF, etc.), "
        "process data, or perform any programmatic task. "
        "The code runs in a workspace directory where created files persist. "
        "Available Python packages include: openpyxl, python-pptx, matplotlib, pandas, "
        "pdfminer, reportlab, fpdf2, Pillow, beautifulsoup4, numpy, scipy, sympy, requests."
    )
    inputs = {
        "code": {
            "type": "string",
            "description": (
                "The code to execute. For Python, write a complete script. "
                "For Bash, write shell commands."
            ),
        },
        "code_type": {
            "type": "string",
            "description": "The type of code to execute: 'python' or 'bash'. Default: 'python'.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        sandbox_dir: str = "",
        conda_python: str = "",
        timeout: int = 120,
    ):
        """
        Args:
            sandbox_dir: Base directory for workspace. If empty, uses .runtime/workspace/.
            conda_python: Path to Python binary in sandbox conda env.
            timeout: Maximum execution time in seconds.
        """
        super().__init__()
        self._sandbox_base = sandbox_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            ".runtime",
            "workspace",
        )
        self._conda_python = conda_python or _resolve_sandbox_python()
        if not os.path.exists(self._conda_python):
            logger.warning(
                "configured conda_python %s does not exist; falling back to %s",
                self._conda_python, sys.executable,
            )
            self._conda_python = sys.executable
        self._timeout = timeout

        self._run_id = str(uuid.uuid4())[:8]
        self._workspace = os.path.join(self._sandbox_base, self._run_id)

    @property
    def workspace(self) -> str:
        """Return the current workspace directory path."""
        return self._workspace

    def set_workspace(self, path: str) -> None:
        """Override the workspace directory (e.g., for per-task workspace)."""
        self._workspace = path

    def forward(self, code: str, code_type: Optional[str] = None) -> str:
        """Execute code and return stdout + stderr."""
        code_type = (code_type or "python").strip().lower()

        safety_error = _check_safety(code)
        if safety_error:
            return f"Error: {safety_error}"

        os.makedirs(self._workspace, exist_ok=True)

        env = os.environ.copy()
        conda_bin = os.path.dirname(self._conda_python)
        env["PATH"] = conda_bin + ":" + env.get("PATH", "")

        try:
            if code_type == "python":
                return self._exec_python(code, env)
            elif code_type == "bash":
                return self._exec_bash(code, env)
            else:
                return f"Error: unsupported code_type '{code_type}'. Use 'python' or 'bash'."
        except subprocess.TimeoutExpired:
            return f"Error: code execution timed out after {self._timeout} seconds."
        except Exception as e:
            return f"Error executing code: {str(e)}"

    def _exec_python(self, code: str, env: dict) -> str:
        """Execute Python code via temp file."""
        script_path = os.path.join(self._workspace, f"_script_{uuid.uuid4().hex[:8]}.py")
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            result = subprocess.run(
                [self._conda_python, script_path],
                cwd=self._workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )

            output_parts = []
            if result.stdout:
                output_parts.append(_clip_stream(result.stdout, STDOUT_CHAR_LIMIT))
            if result.stderr:
                output_parts.append(
                    f"[stderr]: {_clip_stream(result.stderr, STDERR_CHAR_LIMIT)}"
                )
            if result.returncode != 0:
                output_parts.append(f"[exit code]: {result.returncode}")

            output = "\n".join(output_parts).strip()
            if not output:
                output = "(no output)"

            try:
                files = [
                    f for f in os.listdir(self._workspace)
                    if not f.startswith("_script_") and os.path.isfile(
                        os.path.join(self._workspace, f)
                    )
                ]
                if files:
                    output += f"\n\n[Files in workspace]: {', '.join(sorted(files))}"
            except Exception:
                pass

            return output
        finally:
            try:
                os.remove(script_path)
            except Exception:
                pass

    def _exec_bash(self, code: str, env: dict) -> str:
        """Execute Bash code."""
        result = subprocess.run(
            ["bash", "-c", code],
            cwd=self._workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=self._timeout,
        )

        output_parts = []
        if result.stdout:
            output_parts.append(_clip_stream(result.stdout, STDOUT_CHAR_LIMIT))
        if result.stderr:
            output_parts.append(
                f"[stderr]: {_clip_stream(result.stderr, STDERR_CHAR_LIMIT)}"
            )
        if result.returncode != 0:
            output_parts.append(f"[exit code]: {result.returncode}")

        return "\n".join(output_parts).strip() or "(no output)"
