"""
ROMAMemory: FullHistoryMemory with artifact store for subtask result passing.

Adapted from ROMA's ContextStore + ArtifactRegistry.

The artifact store allows subtasks to register results that dependent
subtasks and the aggregator can retrieve. This enables structured
information flow across the decomposition tree.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from scripts.kernel.protocols import BaseMemory
from scripts.kernel.types import (
    MemoryView, Message, PlanState, StepRecord, SummaryState, TaskInput,
)
from scripts.models.base import MessageRole


logger = logging.getLogger(__name__)


class ArtifactStore:
    """In-memory store for subtask results and artifacts.

    Mirrors ROMA's ContextStore: stores results keyed by subtask index,
    supports dependency-aware context retrieval.
    """

    def __init__(self):
        self._results: Dict[str, str] = {}      # index -> result text
        self._goals: Dict[str, str] = {}         # index -> subtask goal
        self._artifacts: Dict[str, Any] = {}     # key -> arbitrary data

    def store_result(self, index: str, goal: str, result: str) -> None:
        """Store a subtask's result."""
        self._results[index] = result
        self._goals[index] = goal
        logger.debug(f"ArtifactStore: stored result for subtask [{index}] ({len(result)} chars)")

    def get_result(self, index: str) -> Optional[str]:
        """Retrieve a subtask's result."""
        return self._results.get(index)

    def get_dependency_context(self, dep_indices: List[str]) -> str:
        """Build context string from dependency results.

        Args:
            dep_indices: List of subtask indices whose results are needed.
        Returns:
            Formatted string with dependency goals and results.
        """
        if not dep_indices:
            return ""

        parts = []
        for idx in dep_indices:
            goal = self._goals.get(idx, f"Subtask {idx}")
            result = self._results.get(idx, "(not yet available)")
            parts.append(
                f"[Subtask {idx}] {goal}\n"
                f"Result: {result}"
            )
        return "\n\n".join(parts)

    def get_all_results(self) -> str:
        """Build context string with ALL subtask results (for aggregator)."""
        if not self._results:
            return ""

        parts = []
        for idx in sorted(self._results.keys(), key=lambda x: int(x) if x.isdigit() else x):
            goal = self._goals.get(idx, f"Subtask {idx}")
            result = self._results[idx]
            parts.append(
                f"[Subtask {idx}] {goal}\n"
                f"Result: {result}"
            )
        return "\n\n".join(parts)

    def store_artifact(self, key: str, value: Any) -> None:
        """Store an arbitrary artifact."""
        self._artifacts[key] = value

    def get_artifact(self, key: str) -> Optional[Any]:
        """Retrieve an artifact."""
        return self._artifacts.get(key)

    def clear(self) -> None:
        """Reset all stored data."""
        self._results.clear()
        self._goals.clear()
        self._artifacts.clear()


class MemoryStrategy(BaseMemory):
    """FullHistory memory with integrated ArtifactStore.

    Keeps all steps as-is (no compression). The ArtifactStore tracks
    subtask results for dependency resolution and aggregation.
    """

    def __init__(self, prompts=None):
        self.prompts = prompts or {}
        self._system_prompt: str = ""
        self._task: Optional[TaskInput] = None
        self._steps: List[Union[StepRecord, PlanState, SummaryState]] = []
        self.artifact_store = ArtifactStore()

    def initialize(self, system_prompt: str, task: TaskInput) -> None:
        self._system_prompt = system_prompt
        self._task = task
        self._steps = []
        self.artifact_store.clear()
        logger.info("ROMAMemory: initialized with artifact store")

    def build_context(self, plan: Optional[PlanState] = None) -> MemoryView:
        """Concatenate system prompt + task + all steps into messages."""
        messages: List[Message] = []

        messages.append(
            Message(
                role=MessageRole.SYSTEM,
                content=[{"type": "text", "text": self._system_prompt}],
            )
        )

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
