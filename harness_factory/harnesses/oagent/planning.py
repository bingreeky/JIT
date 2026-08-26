"""
NoPlanning: no explicit plan generation. The task itself is the directive.

Suitable for simple tasks, low-step tasks, or strong-model baselines.
"""

from typing import Callable, Optional

from scripts.kernel.protocols import BasePlanning
from scripts.kernel.types import (
    Directive, MemoryView, PlanState, StepRecord, SummaryState,
)

class PlanningStrategy(BasePlanning):
    """No explicit planning — just echo the task as the directive."""

    def __init__(self, prompts=None):
        self.prompts = prompts or {}
        self._task = ""

    def init_plan(
        self,
        task: str,
        memory_view: MemoryView,
        tool_schemas: str,
        model: Callable,
    ) -> PlanState:
        self._task = task
        return PlanState(plan="No explicit plan. Proceed directly with the task.")

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
        return Directive(text=self._task)