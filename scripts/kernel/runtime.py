"""
AgentRuntime: the fixed orchestrator that wires 4 modules together and launches execution.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from jinja2 import Template, StrictUndefined

from scripts.models.openai_server import OpenAIServerModel
from scripts.tools.registry import ToolRegistry
from .loader import load_harness
from .monitoring import AgentLogger, LogLevel, YELLOW_HEX
from .types import (
    MemoryView, PlanState, RunResult, RuntimeContext, StepRecord,
    SummaryState, TaskInput, ToolSelection,
)

logger = logging.getLogger(__name__)


class ModelCallBudgetExceeded(BaseException):
    """Raised when a runtime model-call budget must escape harness catches."""


class BudgetedModel:
    """Thin callable wrapper that enforces a hard model-call budget."""

    def __init__(self, model: Any, max_calls: int):
        self._model = model
        self._max_calls = max(0, int(max_calls))
        self._calls = 0

    @property
    def calls_used(self) -> int:
        return self._calls

    @property
    def max_calls(self) -> int:
        return self._max_calls

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._calls >= self._max_calls:
            raise ModelCallBudgetExceeded("Budget exceeded")
        self._calls += 1
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def populate_template(template: str, variables: Dict[str, Any]) -> str:
    """Render a Jinja2 template string with variables."""
    compiled_template = Template(template, undefined=StrictUndefined)
    return compiled_template.render(**variables)


class AgentRuntime:
    """The fixed orchestrator. Wires the 4 modules together and runs tasks.

    This is the ONLY class that knows about all modules. It creates a
    RuntimeContext and delegates execution to the Action module.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dict with keys:
                - harness: harness name (str)
                - model: {model_id, api_base, api_key, ...}
                - tools: list of tool names
                - execution: {max_steps, concurrency}
        """
        self.config = config

        # 1. Load model
        model_cfg = config.get("model", {})
        custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
        self.model = OpenAIServerModel(
            model_id=model_cfg.get("model_id", os.getenv("DEFAULT_MODEL", "gpt-4o")),
            api_base=model_cfg.get("api_base", os.getenv("OPENAI_API_BASE")),
            api_key=model_cfg.get("api_key", os.getenv("OPENAI_API_KEY")),
            temperature=model_cfg.get("temperature"),
            custom_role_conversions=custom_role_conversions,
            max_tokens=model_cfg.get("max_tokens", 16000),
        )

        # 2. Load tools into registry
        self.tool_registry = ToolRegistry()
        tool_names = config.get("tools", ["web_search", "crawl_page", "final_answer"])
        self.tool_registry.register_defaults(tool_names, model=self.model, config=config)

        # 3. Load harness (the 4 modules)
        harness_name = config.get("harness", "react_baseline")
        harness = load_harness(harness_name)
        self.memory = harness["memory"]
        self.planning = harness["planning"]
        self.action = harness["action"]
        self.tool_policy = harness["tool_policy"]
        self.harness_prompts = harness.get("prompts", {})

        # 4. Execution config (read early — needed by tool_policy and trace setup)
        exec_cfg = config.get("execution", {})

        # 6. Init tool policy with full catalog
        enable_skills = exec_cfg.get("enable_skills", False)
        self.tool_policy.initialize(self.tool_registry.get_all(), enable_skills=enable_skills)

        # With EXEC_NATIVE_TOOLCALL=1 the send layer needs the tool schemas in order to
        # use OpenAI native function calling. A harness calls ctx.model(messages) bare
        # and never passes tools_to_call_from, so hand the registry to the model here
        # and let models/openai_server.py decide whether to use it. The default path is
        # unaffected.
        self.model._native_tool_registry = self.tool_registry.get_all()

        # 6b. Inject model into memory if it supports summarization (e.g., ReSumMemory)
        if hasattr(self.memory, 'set_model'):
            self.memory.set_model(self.model)
        self.logger = AgentLogger(level=LogLevel.INFO)
        self.max_steps = exec_cfg.get("max_steps", 40)

        # 7. Trace saving (optional)
        self.trace_dir = exec_cfg.get("trace_dir", "")
        self._run_counter = 0

    def set_tool_workspace(
        self, workspace: str, tool_names: Optional[List[str]] = None
    ) -> List[str]:
        """Try to set workspace for registered tools that support it.

        Args:
            workspace: Workspace directory to apply.
            tool_names: Optional subset of tool names to update. Defaults to all
                registered tools.

        Returns:
            List of tool names that accepted the workspace update.
        """
        if not workspace:
            return []

        candidate_names = tool_names or list(self.tool_registry.get_all().keys())
        updated_tools: List[str] = []

        for tool_name in candidate_names:
            try:
                tool = self.tool_registry.get(tool_name)
            except KeyError:
                continue

            if not hasattr(tool, "set_workspace"):
                continue

            try:
                tool.set_workspace(workspace)
                updated_tools.append(tool_name)
            except Exception as exc:
                logger.debug(
                    "Failed to set workspace for tool '%s': %s",
                    tool_name,
                    exc,
                )

        return updated_tools

    def run(self, task: str, images: Optional[List[str]] = None,
            task_tools: Optional[Dict] = None) -> RunResult:
        """Execute one task. This is the main public API.

        Args:
            task: Task description string.
            images: Optional list of image paths.
            task_tools: Optional dict of per-task domain tools to inject
                       temporarily (e.g., DeepPlanning shopping/travel tools).
        Returns:
            RunResult with answer and trajectory.
        """
        # Inject per-task domain tools (e.g., DeepPlanning shopping tools)
        if task_tools:
            self.tool_registry.register_batch(task_tools)
            # Re-initialize tool policy to include the new tools
            exec_cfg = self.config.get("execution", {})
            enable_skills = exec_cfg.get("enable_skills", False)
            self.tool_policy.initialize(
                self.tool_registry.get_all(), enable_skills=enable_skills
            )
            # The native-tool-call schema snapshot has to be refreshed along with it:
            # __init__ attached a COPY of get_all(), holding only the base tools from
            # the config (for travel/shopping that is execute_code + final_answer).
            # Without this refresh, the domain tools injected per case (travel has nine
            # query tools) would never reach the request's `tools` field, leaving the
            # model execute_code as its only callable -- which degenerates into page
            # after page of execute_code poking around the filesystem.
            self.model._native_tool_registry = self.tool_registry.get_all()

        try:
            if hasattr(self.model, "reset_token_counters"):
                self.model.reset_token_counters()

            exec_cfg = self.config.get("execution", {})
            model_for_run = self.model
            model_call_budget = exec_cfg.get("model_call_budget", None)
            if model_call_budget is not None:
                model_for_run = BudgetedModel(self.model, int(model_call_budget))
                if hasattr(self.memory, "set_model"):
                    self.memory.set_model(model_for_run)

            # Build RuntimeContext
            ctx = RuntimeContext(
                memory=self.memory,
                planning=self.planning,
                tool_policy=self.tool_policy,
                model=model_for_run,
                execute_tool=self._execute_tool,
                get_tool_schemas=self._get_tool_schemas,
                logger=self.logger,
                prompt_templates=self.harness_prompts,
                max_steps=self.max_steps,
            )

            # Initialize memory
            system_prompt = self._build_system_prompt()
            task_input = TaskInput(task=task, task_images=images or [])
            self.memory.initialize(system_prompt, task_input)

            # Log task
            self.logger.log_task(
                task,
                subtitle=f"harness={self.config.get('harness', 'unknown')}",
            )

            # Delegate to Action module — it owns the loop
            result = self.action.run(task, ctx)

            metadata = dict(getattr(result, "metadata", {}) or {})
            if isinstance(model_for_run, BudgetedModel):
                metadata["model_call_budget"] = model_for_run.max_calls
                metadata["model_calls_used"] = model_for_run.calls_used
            model_totals = {}
            if hasattr(self.model, "get_total_token_counts"):
                model_totals = self.model.get_total_token_counts() or {}
            input_count = int(model_totals.get("input_token_count", 0))
            output_count = int(model_totals.get("output_token_count", 0))
            metadata["input_token_count"] = input_count
            metadata["output_token_count"] = output_count
            metadata["total_token_count"] = input_count + output_count
            result.metadata = metadata

            # Save full execution trace if trace_dir is configured
            if self.trace_dir:
                self._save_trace(task, result)

            return result
        finally:
            # Clean up per-task tools
            if task_tools:
                self.tool_registry.unregister_batch(task_tools.keys())
                # Re-initialize tool policy without per-task tools
                exec_cfg = self.config.get("execution", {})
                enable_skills = exec_cfg.get("enable_skills", False)
                self.tool_policy.initialize(
                    self.tool_registry.get_all(), enable_skills=enable_skills
                )
                # Same as above, in reverse: roll the native schema snapshot back to
                # the base tools so one case's domain tools -- bound to that case's
                # database directory -- cannot leak into the next one.
                self.model._native_tool_registry = self.tool_registry.get_all()

    def _execute_tool(self, tool_name: str, arguments: Any) -> str:
        """Execute a tool by name. Called by Action via ctx."""
        try:
            tool = self.tool_registry.get(tool_name)
            if isinstance(arguments, dict):
                return str(tool(**arguments))
            else:
                return str(tool(arguments))
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"

    def _get_tool_schemas(self, tools: Optional[Dict] = None) -> str:
        """Generate JSON schema string for tools."""
        return self.tool_registry.format_schemas(tools)

    def _build_system_prompt(self) -> str:
        """Build harness system prompt from template + tool descriptions + skills."""
        system_prompt_template = self.harness_prompts.get("system_prompt", "")
        if not system_prompt_template:
            harness_name = self.config.get("harness", "unknown")
            raise RuntimeError(
                f"Missing 'system_prompt' in harness prompt.yaml for '{harness_name}'."
            )

        skills_prompt = ""
        if hasattr(self.tool_policy, 'get_skills_prompt'):
            skills_prompt = self.tool_policy.get_skills_prompt()
        return populate_template(
            system_prompt_template,
            variables={
                "tools": self.tool_registry.get_all(),
                "skills_prompt": skills_prompt
            },
        )

    def _save_trace(self, task: str, result: RunResult) -> None:
        """Save full execution trace to trace_dir as a JSONL file.

        Each run produces one JSON object per line containing:
        - task: the input task string
        - timestamp: when the run completed
        - config: harness, model, tools used
        - Full RunResult with raw LLM messages (via full_dict())
        """
        try:
            os.makedirs(self.trace_dir, exist_ok=True)
            self._run_counter += 1
            ts = int(time.time())
            filename = f"trace_{ts}_{self._run_counter:04d}.json"
            filepath = os.path.join(self.trace_dir, filename)

            trace = {
                "task": task[:2000],  # Truncate very long tasks
                "timestamp": time.time(),
                "config": {
                    "harness": self.config.get("harness", ""),
                    "model": self.config.get("model", {}).get("model_id", ""),
                    "tools": self.config.get("tools", []),
                    "max_steps": self.max_steps,
                },
                **result.full_dict(),
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(trace, f, ensure_ascii=False, default=str, indent=2)

            logger.info(f"Saved execution trace to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save execution trace: {e}")
