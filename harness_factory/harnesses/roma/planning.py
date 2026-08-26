"""
ROMAPlanning: Atomizer + Planner for recursive task decomposition.

Adapted from ROMA (Recursive Orchestrated Multi-Agent).

Two-phase planning:
  1. Atomizer: decides if a task is atomic (execute directly) or complex (decompose)
  2. Planner: decomposes complex tasks into subtasks with a dependency DAG

The decomposition result is stored in PlanState.metadata for the action module.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

import json_repair

from scripts.kernel.protocols import BasePlanning
from scripts.kernel.types import (
    Directive, MemoryView, PlanState, StepRecord, SummaryState,
)

# We store decomposition metadata in plan_reasoning as JSON
# since PlanState doesn't have a metadata field.
# Action module parses this with parse_plan_metadata().


logger = logging.getLogger(__name__)


class PlanningStrategy(BasePlanning):
    """Atomizer + Planner: decide complexity, then decompose if needed.

    init_plan() performs two LLM calls:
      1. Atomizer: "Is this task atomic or complex?"
      2. Planner (if complex): "Decompose into subtasks with dependencies."

    Results stored in PlanState.metadata:
      - is_atomic: bool
      - subtasks: List[{goal, index}]  (if not atomic)
      - dependencies: Dict[str, List[str]]  (if not atomic)
    """

    def __init__(self, prompts=None):
        self.prompts = prompts or {}
        self._task = ""
        self._plan_metadata: Dict[str, Any] = {}

    def init_plan(
        self,
        task: str,
        memory_view: MemoryView,
        tool_schemas: str,
        model: Callable,
    ) -> PlanState:
        self._task = task

        # Phase 1: Atomizer
        is_atomic = self._atomize(task, tool_schemas, model)

        if is_atomic:
            self._plan_metadata = {"is_atomic": True}
            logger.info("ROMAPlanning: atomizer decided ATOMIC — execute directly")
            return PlanState(
                plan="Task is atomic. Execute directly.",
                plan_reasoning=json.dumps({"is_atomic": True}),
            )

        # Phase 2: Planner (decompose)
        subtasks, dependencies = self._plan(task, tool_schemas, model)

        if not subtasks:
            # Planner failed to decompose — fallback to atomic
            logger.warning("ROMAPlanning: planner returned no subtasks, falling back to atomic")
            self._plan_metadata = {"is_atomic": True}
            return PlanState(
                plan="Planner could not decompose. Execute directly.",
                plan_reasoning=json.dumps({"is_atomic": True}),
            )

        self._plan_metadata = {
            "is_atomic": False,
            "subtasks": subtasks,
            "dependencies": dependencies,
        }

        # Format plan text
        plan_lines = [f"Decomposed into {len(subtasks)} subtasks:"]
        for st in subtasks:
            deps = dependencies.get(str(st["index"]), [])
            dep_str = f" (depends on: {deps})" if deps else ""
            plan_lines.append(f"  [{st['index']}] {st['goal']}{dep_str}")
        plan_text = "\n".join(plan_lines)

        logger.info(
            f"ROMAPlanning: planner decomposed into {len(subtasks)} subtasks "
            f"with dependency graph: {dependencies}"
        )

        return PlanState(
            plan=plan_text,
            plan_reasoning=json.dumps(self._plan_metadata),
        )

    def should_replan(self, step_number: int, step: StepRecord) -> bool:
        return False

    def update_plan(
        self,
        task: str,
        step_number: int,
        memory_view: MemoryView,
        model: Callable,
    ) -> SummaryState:
        return SummaryState(summary="No replanning needed.")

    def get_directive(self) -> Directive:
        return Directive(text=self._task, metadata=self._plan_metadata)

    # ── Private: Atomizer ──

    def _atomize(self, task: str, tool_schemas: str, model: Callable) -> bool:
        """Call LLM to decide if task is atomic or needs decomposition."""
        prompt_template = self.prompts.get("planning", {}).get(
            "atomizer_prompt", ""
        )
        if not prompt_template:
            # Fallback built-in prompt
            prompt_template = (
                "You are a task complexity analyzer. Given a task, decide whether it can be "
                "solved directly with available tools (ATOMIC), or needs to be broken into subtasks (COMPLEX).\n\n"
                "Task: {task}\n\n"
                "Available tools:\n{tool_schemas}\n\n"
                "Respond with ONLY a JSON object:\n"
                '{{"is_atomic": true}}  if the task can be solved directly\n'
                '{{"is_atomic": false}} if the task needs decomposition into subtasks\n\n'
                "Consider a task ATOMIC if it can be answered by a straightforward sequence of "
                "tool calls (e.g., search + read + answer). Consider it COMPLEX if it has multiple "
                "independent sub-questions or requires gathering different types of information separately."
            )

        prompt_text = prompt_template.replace("{task}", task).replace(
            "{tool_schemas}", tool_schemas
        )
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ]

        try:
            response = model(messages)
            content = response.content or ""
            parsed = json_repair.loads(content)
            if isinstance(parsed, dict):
                result = parsed.get("is_atomic", True)
                logger.info(f"ROMAPlanning: atomizer result is_atomic={result}")
                return bool(result)
        except Exception as e:
            logger.error(f"ROMAPlanning: atomizer failed: {e}, defaulting to atomic")

        return True  # Default: atomic (safe fallback)

    # ── Private: Planner ──

    def _plan(
        self, task: str, tool_schemas: str, model: Callable,
    ) -> tuple:
        """Call LLM to decompose task into subtasks with dependencies.

        Returns:
            (subtasks, dependencies) where:
              subtasks: List[{"goal": str, "index": int}]
              dependencies: Dict[str, List[str]]  e.g. {"1": ["0"], "2": ["0","1"]}
        """
        prompt_template = self.prompts.get("planning", {}).get(
            "planner_prompt", ""
        )
        if not prompt_template:
            prompt_template = (
                "You are a task decomposition planner. Break down the given task into "
                "smaller, independently solvable subtasks.\n\n"
                "Task: {task}\n\n"
                "Available tools:\n{tool_schemas}\n\n"
                "Rules:\n"
                "- Create 2-5 subtasks that together cover the full task.\n"
                "- Each subtask should be a self-contained goal achievable with the available tools.\n"
                "- Specify dependencies: which subtasks must complete before others can start.\n"
                "- Independent subtasks can run in parallel.\n\n"
                "Respond with ONLY a JSON object:\n"
                '{{\n'
                '  "subtasks": [\n'
                '    {{"goal": "First subtask description", "index": 0}},\n'
                '    {{"goal": "Second subtask description", "index": 1}},\n'
                '    {{"goal": "Third subtask description", "index": 2}}\n'
                '  ],\n'
                '  "dependencies": {{\n'
                '    "0": [],\n'
                '    "1": ["0"],\n'
                '    "2": ["0", "1"]\n'
                '  }}\n'
                '}}\n\n'
                "The dependencies dict maps subtask index (as string) to list of indices it depends on.\n"
                "Subtasks with empty dependency lists can start immediately."
            )

        prompt_text = prompt_template.replace("{task}", task).replace(
            "{tool_schemas}", tool_schemas
        )
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ]

        try:
            response = model(messages)
            content = response.content or ""
            parsed = json_repair.loads(content)

            if not isinstance(parsed, dict):
                return [], {}

            subtasks = parsed.get("subtasks", [])
            dependencies = parsed.get("dependencies", {})

            # Validate subtasks
            valid_subtasks = []
            for st in subtasks:
                if isinstance(st, dict) and "goal" in st:
                    idx = st.get("index", len(valid_subtasks))
                    valid_subtasks.append({"goal": st["goal"], "index": idx})

            # Validate dependencies (ensure all indices exist)
            valid_indices = {str(st["index"]) for st in valid_subtasks}
            valid_deps = {}
            for k, v in dependencies.items():
                if k in valid_indices:
                    valid_deps[k] = [
                        d for d in v
                        if isinstance(d, str) and d in valid_indices and d != k
                    ]

            logger.info(
                f"ROMAPlanning: planner produced {len(valid_subtasks)} subtasks"
            )
            return valid_subtasks, valid_deps

        except Exception as e:
            logger.error(f"ROMAPlanning: planner failed: {e}")
            return [], {}
