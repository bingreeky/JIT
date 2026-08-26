"""
FlashDAGPlanning: Flash-Searcher-style DAG planning with parallel goals
and sequential fallback paths.

Generates a plan where tasks are decomposed into 1-5 independent goals,
each with 1-5 distinct execution paths. Goals are executed in parallel,
while paths within each goal are tried sequentially as fallbacks.

Periodically analyzes progress and suggests next parallel sub-paths.
"""

import logging
from typing import Callable

from scripts.kernel.protocols import BasePlanning
from scripts.kernel.types import (
    Directive, MemoryView, PlanState, StepRecord, SummaryState,
)
from scripts.models.base import MessageRole


logger = logging.getLogger(__name__)


# ── DAG Planning Prompts ──



class PlanningStrategy(BasePlanning):
    """DAG planning with parallel goals and sequential fallback paths.

    Based on Flash-Searcher's planning system:
    - init_plan: Decompose task into parallel goals with paths
    - should_replan: Check every summary_interval steps
    - update_plan: Analyze goal completion and suggest next paths
    - get_directive: Return full plan text
    """

    def __init__(self, prompts=None, summary_interval: int = 8):
        self.prompts = prompts or {}
        self._summary_interval = summary_interval
        self._current_plan = ""
        self._task = ""

    def init_plan(
        self,
        task: str,
        memory_view: MemoryView,
        tool_schemas: str,
        model: Callable,
    ) -> PlanState:
        self._task = task
        plan_system = self.prompts["planning"]["initial_plan"].format(
            task=task,
            tool_schemas=tool_schemas[:2000],
        )

        # Build messages
        planning_messages = memory_view.messages.copy()
        planning_messages = planning_messages[1:]
        planning_messages.append({
            "role": MessageRole.USER,
            "content": [{"type": "text", "text": plan_system}],
        })

        response = model(planning_messages)
        plan_text = response.content or ""
        plan_reasoning = getattr(response, 'reasoning_content', '') or ""

        self._current_plan = plan_text

        return PlanState(
            plan=plan_text,
            plan_think="",
            plan_reasoning=plan_reasoning,
            model_input_messages=planning_messages,
        )

    def should_replan(self, step_number: int, step: StepRecord) -> bool:
        return (
            self._summary_interval > 0
            and step_number > 0
            and step_number % self._summary_interval == 0
        )

    def update_plan(
        self,
        task: str,
        step_number: int,
        memory_view: MemoryView,
        model: Callable,
    ) -> SummaryState:
        pre = self.prompts["summary"]["update_pre_messages"]
        post = self.prompts["summary"]["update_post_messages"]

        summary_messages = memory_view.messages.copy()
        summary_messages = summary_messages[1:]
        summary_messages.append({
            "role": MessageRole.USER,
            "content": [{"type": "text", "text": pre}],
        })
        summary_messages.append({
            "role": MessageRole.USER,
            "content": [{"type": "text", "text": post}],
        })

        response = model(summary_messages)
        summary_text = response.content or ""
        summary_reasoning = getattr(response, 'reasoning_content', '') or ""

        # Update plan with summary insights
        self._current_plan = summary_text

        return SummaryState(
            summary=summary_text,
            summary_reasoning=summary_reasoning,
            model_input_messages=summary_messages,
        )

    def get_directive(self) -> Directive:
        return Directive(text=self._current_plan)