"""
LinearPlanning: generates an initial plan using Flash-Searcher's DAG planning prompt,
then periodically summarizes progress and updates the plan.

This directly reuses Flash-Searcher's planning_step and summary_step logic.
"""

import logging
from typing import Callable

from scripts.kernel.protocols import BasePlanning
from scripts.kernel.types import (
    Directive, MemoryView, PlanState, StepRecord, SummaryState,
)
from scripts.models.base import MessageRole


logger = logging.getLogger(__name__)


# ── Linear Planning Prompts ──



class PlanningStrategy(BasePlanning):
    """Flash-Searcher-style planning with periodic adaptation.

    Uses the planning prompt to decompose tasks into parallel goals with paths,
    and the summary prompt to analyze progress every N steps.
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
        planning_messages = memory_view.messages.copy()
        planning_messages = planning_messages[1:]
        planning_messages.append({
            "role": MessageRole.USER,
            "content": [{
                "type": "text",
                "text": self.prompts["planning"]["initial_plan"].format(
                    task=task,
                    tool_schemas=tool_schemas[:2000],
                ),
            }],
        })

        # Call LLM
        response = model(planning_messages)

        # Extract plan from response
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
        summary_messages = memory_view.messages.copy()
        summary_messages = summary_messages[1:]
        summary_messages.append({
            "role": MessageRole.USER,
            "content": [{
                "type": "text",
                "text": self.prompts["summary"]["update_messages"].format(step_number=step_number),
            }],
        })

        response = model(summary_messages)
        summary_text = response.content or ""
        summary_reasoning = getattr(response, 'reasoning_content', '') or ""

        return SummaryState(
            summary=summary_text,
            summary_reasoning=summary_reasoning,
            model_input_messages=summary_messages,
        )

    def get_directive(self) -> Directive:
        return Directive(text=self._current_plan)