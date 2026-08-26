"""
OAgentMemory: coordinator memory adapted for multi-agent ensemble action.

In ensemble mode the outer memory never accumulates a step-by-step history
the way a normal ReAct memory does.  Its only responsibilities are:

  1. Store the system prompt and task so the action module can extract them
     to initialize per-expert memories.
  2. Expose a build_context() that returns the minimal system+task context
     (used by the kernel before handing control to the action module).
  3. Accept the merged trajectory written back by the action module after
     all experts finish (for get_all_steps() / result logging).

Per-expert memories are created and managed entirely inside the action
module — this memory is NOT involved in the per-step loop.
"""

from typing import List, Optional, Union

from scripts.kernel.protocols import BaseMemory
from scripts.kernel.types import (
    MemoryView, Message, PlanState, StepRecord, SummaryState, TaskInput,
)
from scripts.models.base import MessageRole

class MemoryStrategy(BaseMemory):
    """Coordinator memory for OAgentAction.

    Holds the system prompt and task for expert initialization.
    Does NOT accumulate per-step history — that is handled by each
    expert's independent memory instance inside the action module.
    """

    def __init__(self, prompts=None):
        self.prompts = prompts or {}
        self._system_prompt: str = ""
        self._task: Optional[TaskInput] = None
        self._merged_steps: List[StepRecord] = []

    # ── Public accessors for the action module ──

    @property
    def system_prompt(self) -> str:
        """The raw system prompt string.  Used by the action module to
        construct per-expert system prompts without parsing messages."""
        return self._system_prompt

    @property
    def task(self) -> Optional[TaskInput]:
        return self._task

    # ── BaseMemory protocol ──

    def initialize(self, system_prompt: str, task: TaskInput) -> None:
        self._system_prompt = system_prompt
        self._task = task
        self._merged_steps = []

    def build_context(self, plan: Optional[PlanState] = None) -> MemoryView:
        """Minimal context: system prompt + task."""
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
        pass  # ensemble does not use outer-level plans

    def update_summary(self, summary: SummaryState) -> None:
        pass  # ensemble does not use outer-level summaries

    def get_all_steps(self) -> List[StepRecord]:
        return list(self._merged_steps)


class FullHistoryMemory(BaseMemory):
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
