"""
FullHistoryMemory: keeps all steps in memory without compression.

Equivalent to Flash-Searcher's AgentMemory behavior.
Suitable for short tasks (< 15 steps) where context window is not a concern.
"""

from typing import Dict, List, Optional, Any, Union

from scripts.kernel.protocols import BaseMemory
from scripts.kernel.types import (
    MemoryView, Message, PlanState, StepRecord, SummaryState, TaskInput,
)
from scripts.models.base import MessageRole


class MemoryStrategy(BaseMemory):
    """Keeps all steps as-is. No compression, no summarization."""

    def __init__(self, prompts=None):
        self.prompts = prompts or {}

    def initialize(self, system_prompt: str, task: TaskInput) -> None:
        self._system_prompt = system_prompt
        self._task = task
        self._steps: List[Union[StepRecord, PlanState, SummaryState]] = []

    def build_context(self, plan: Optional[PlanState] = None) -> MemoryView:
        """Concatenate system prompt + task + all steps into messages."""
        messages: List[Message] = []

        # System prompt
        messages.append(
            Message(
                role=MessageRole.SYSTEM,
                content=[{"type": "text", "text": self._system_prompt}],
            )
        )

        # Task
        messages.extend(self._task.to_messages())

        # All steps in order
        for step in self._steps:
            messages.extend(step.to_messages())

        return MemoryView(messages=messages)

    def update(self, step: StepRecord) -> None:
        self._steps.append(step)

    def update_plan(self, plan: PlanState) -> None:
        self._steps.append(plan)

    def update_summary(self, summary: SummaryState) -> None:
        self._steps.append(summary)

    def get_all_steps(self) -> List[StepRecord]:
        return [s for s in self._steps if isinstance(s, StepRecord)]
