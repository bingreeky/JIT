"""
AggMemory: coordinator memory for the aggagent harness.

The outer memory does NOT accumulate per-step history like a normal
ReAct memory. Its responsibilities are:
  1. Store the system prompt and task so the action module can extract
     them to initialize per-rollout memories.
  2. Expose build_context() that returns minimal system+task context
     (used by the kernel before handing control to the action module).
  3. Accept the merged trajectory written back by the action module
     after all rollouts and the aggregator finish (for get_all_steps /
     trajectory logging).

Per-rollout memories (FullHistoryMemory instances) are created and
managed entirely inside the action module.
"""

import logging
from typing import List, Optional, Union

from scripts.kernel.protocols import BaseMemory
from scripts.kernel.types import (
    MemoryView, Message, PlanState, StepRecord, SummaryState, TaskInput,
)
from scripts.models.base import MessageRole


logger = logging.getLogger(__name__)


class MemoryStrategy(BaseMemory):
    """Coordinator memory for AggAgentAction.

    Holds the system prompt and task for rollout worker initialization.
    Does NOT accumulate per-step history — each rollout's independent
    FullHistoryMemory instance handles that inside the action module.
    """

    def __init__(self, prompts=None):
        self.prompts = prompts or {}
        self._system_prompt: str = ""
        self._task: Optional[TaskInput] = None
        self._merged_steps: List[StepRecord] = []

    # ── Public accessors used by the action module ──

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def task(self) -> Optional[TaskInput]:
        return self._task

    # ── BaseMemory protocol ──

    def initialize(self, system_prompt: str, task: TaskInput) -> None:
        self._system_prompt = system_prompt
        self._task = task
        self._merged_steps = []
        logger.info("AggMemory: initialized (coordinator)")

    def build_context(self, plan: Optional[PlanState] = None) -> MemoryView:
        """Return minimal context: system prompt + task."""
        messages: List[Message] = [
            Message(
                role=MessageRole.SYSTEM,
                content=[{"type": "text", "text": self._system_prompt}],
            ),
        ]
        if self._task:
            messages.extend(self._task.to_messages())
        return MemoryView(messages=messages)

    def update(self, step: StepRecord) -> None:
        """Accept merged steps written back by the action module."""
        self._merged_steps.append(step)

    def update_plan(self, plan: PlanState) -> None:
        pass  # aggagent does not use outer-level plans

    def update_summary(self, summary: SummaryState) -> None:
        pass  # aggagent does not use outer-level summaries

    def get_all_steps(self) -> List[StepRecord]:
        return list(self._merged_steps)


class FullHistoryMemory(BaseMemory):
    """Per-rollout memory: keeps all steps as-is, no compression.

    Used internally by the action module for each of the K rollout
    workers. Not loaded by the harness loader.
    """

    def __init__(self, prompts=None):
        self.prompts = prompts or {}
        self._system_prompt: str = ""
        self._task: Optional[TaskInput] = None
        self._steps: List[Union[StepRecord, PlanState, SummaryState]] = []

    def initialize(self, system_prompt: str, task: TaskInput) -> None:
        self._system_prompt = system_prompt
        self._task = task
        self._steps = []

    def build_context(self, plan: Optional[PlanState] = None) -> MemoryView:
        messages: List[Message] = [
            Message(
                role=MessageRole.SYSTEM,
                content=[{"type": "text", "text": self._system_prompt}],
            ),
        ]
        messages.extend(self._task.to_messages())
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
