"""
HierarchicalMemory: value-based step sampling for context management.

Adapted from HiAgent (https://github.com/HiAgent).

Core idea: store ALL steps internally, but when building context for
the next LLM call, select only the most informative steps based on a
value function combining:
  - w1: Boundary weights (first/last steps are important)
  - w2: Observation novelty (high change between consecutive steps)
  - w3: Reward signal (disabled by default; our benchmarks lack per-step rewards)

Steps not selected are represented as "[Omitted]" placeholders so the
LLM knows history was pruned.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np

from scripts.kernel.protocols import BaseMemory
from scripts.kernel.token_counter import count_tokens_messages
from scripts.kernel.types import (
    MemoryView, Message, PlanState, StepRecord, SummaryState, TaskInput,
)
from scripts.models.base import MessageRole


logger = logging.getLogger(__name__)


# ── Value calculation helpers (from HiAgent ours_agent.py) ──

def _calculate_similarity(obs1: str, obs2: str) -> float:
    """TF-IDF cosine similarity between two observation strings."""
    if not obs1.strip() or not obs2.strip():
        return 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer().fit_transform([obs1, obs2])
        vectors = vectorizer.toarray()
        return float(cosine_similarity(vectors)[0, 1])
    except Exception as e:
        # Fallback: trivial character overlap ratio
        logger.debug(f"HierarchicalMemory: TF-IDF similarity failed ({e}), using word overlap fallback")
        set1, set2 = set(obs1.split()), set(obs2.split())
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / max(len(set1 | set2), 1)


def _calculate_values(
    observations: List[str],
    rewards: List[float],
    w1: float = 1.0,
    w2: float = 1.0,
    w3: float = 0.0,
) -> np.ndarray:
    """Compute per-step information value.

    Mirrors HiAgent's calculate_value():
      - w1: Boundary weights via inverted Gaussian (edges > middle)
      - w2: Observation change rate (1 - cosine_similarity)
      - w3: Reward delta between consecutive steps

    Args:
        observations: List of observation strings, one per step.
        rewards: List of reward floats, one per step.
        w1, w2, w3: Component weights. w3 defaults to 0 because our
            benchmarks lack per-step reward signals.
    Returns:
        np.ndarray of value scores, same length as observations.
    """
    n = len(observations)
    if n == 0:
        return np.array([])

    values = np.zeros(n)

    # w1: Boundary weights — Gaussian centered at middle, inverted so
    # first/last steps have highest weight.
    mu = (n - 1) / 2
    sigma = max(n / 6, 1e-6)
    boundary_weights = 1 - np.exp(-((np.arange(n) - mu) ** 2) / (2 * sigma ** 2))
    values += w1 * boundary_weights

    # w2 + w3: Observation novelty and reward change (pairwise)
    for i in range(n - 1):
        if w2 > 0:
            similarity = _calculate_similarity(observations[i], observations[i + 1])
            change_rate = 1 - similarity
            values[i] += w2 * change_rate
        if w3 > 0:
            reward_change = rewards[i + 1] - rewards[i]
            values[i] += w3 * reward_change

    return values


class MemoryStrategy(BaseMemory):
    """Hierarchical working memory with value-based step sampling.

    Stores all steps internally. On build_context(), selects the top-K
    most informative steps (by boundary weight + observation novelty),
    preserving first and last steps. Omitted steps are shown as
    placeholders so the LLM is aware of pruning.

    PlanState and SummaryState items are NEVER pruned — they are
    structural and always included in context.
    """

    def __init__(
        self,
        prompts=None,
        memory_size: int = 15,
        max_context_tokens: int = 31768,
        w1: float = 1.0,
        w2: float = 1.0,
        w3: float = 0.0,
    ):
        """
        Args:
            prompts: Harness prompt templates (from prompt.yaml).
            memory_size: Target number of StepRecord entries to keep
                in the working context. If total steps <= memory_size,
                all are kept (no sampling).
            max_context_tokens: Token budget. If the sampled context
                exceeds this, memory_size is iteratively reduced.
            w1: Weight for boundary importance.
            w2: Weight for observation novelty.
            w3: Weight for reward change (0 by default — no per-step
                reward in our benchmarks).
        """
        self.prompts = prompts or {}
        self._memory_size = memory_size
        self._max_context_tokens = max_context_tokens
        self._w1 = w1
        self._w2 = w2
        self._w3 = w3

        self._system_prompt: str = ""
        self._task: Optional[TaskInput] = None
        self._all_items: List[Union[StepRecord, PlanState, SummaryState]] = []

    def initialize(self, system_prompt: str, task: TaskInput) -> None:
        self._system_prompt = system_prompt
        self._task = task
        self._all_items = []
        logger.info(
            f"HierarchicalMemory: initialized "
            f"(memory_size={self._memory_size}, "
            f"max_tokens={self._max_context_tokens}, "
            f"w1={self._w1}, w2={self._w2}, w3={self._w3})"
        )

    def build_context(self, plan: Optional[PlanState] = None) -> MemoryView:
        """Build context with value-based step sampling.

        Steps:
        1. Separate StepRecords (samplable) from PlanState/SummaryState (always kept).
        2. If len(steps) <= memory_size, keep all (no sampling needed).
        3. Otherwise, compute values, keep first+last, select top-(K-2) from middle.
        4. Rebuild messages in original order, with omitted placeholders.
        5. If token count exceeds budget, reduce K and retry.
        """
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

        # Nothing else to add if no items yet
        if not self._all_items:
            return MemoryView(messages=messages)

        # Separate step records from structural items, preserving order
        indexed_items = []  # (original_index, item)
        step_indices = []   # indices into indexed_items that are StepRecords
        for i, item in enumerate(self._all_items):
            indexed_items.append((i, item))
            if isinstance(item, StepRecord):
                step_indices.append(i)

        steps_only = [self._all_items[i] for i in step_indices]
        n_steps = len(steps_only)

        # If no step records yet (only plan/summary items), build without sampling
        if n_steps == 0:
            result = list(messages)
            for _, item in indexed_items:
                result.extend(item.to_messages())
            return MemoryView(messages=result)

        # Determine effective sampling size
        sampling_n = min(self._memory_size, n_steps)

        # Build messages with current sampling_n, reducing if over token budget
        while sampling_n >= 1:
            candidate_messages = self._build_sampled_messages(
                messages, indexed_items, step_indices, steps_only, sampling_n,
            )
            token_count = count_tokens_messages(candidate_messages)
            if token_count <= self._max_context_tokens:
                if sampling_n < n_steps:
                    logger.info(
                        f"HierarchicalMemory: sampled {sampling_n}/{n_steps} steps "
                        f"({n_steps - sampling_n} omitted), "
                        f"token_count={token_count}"
                    )
                return MemoryView(
                    messages=candidate_messages,
                    metadata={
                        "total_steps": n_steps,
                        "sampled_steps": sampling_n,
                        "token_count": token_count,
                    },
                )
            # Over budget — reduce sampling
            sampling_n -= 1
            logger.info(
                f"HierarchicalMemory: token count {token_count} exceeds "
                f"budget {self._max_context_tokens}, reducing sampling to {sampling_n}"
            )

        # Fallback: even 1 step is over budget. Return just system+task.
        logger.warning(
            "HierarchicalMemory: cannot fit any steps within token budget"
        )
        return MemoryView(messages=messages)

    def _build_sampled_messages(
        self,
        base_messages: List[Message],
        indexed_items: List[tuple],
        step_indices: List[int],
        steps_only: List[StepRecord],
        sampling_n: int,
    ) -> List[Message]:
        """Build full message list with value-sampled steps.

        Args:
            base_messages: [system, task] messages to prepend.
            indexed_items: [(original_index, item)] for all items.
            step_indices: Original indices of StepRecord items.
            steps_only: List of StepRecord objects in order.
            sampling_n: Number of steps to keep.
        Returns:
            Complete message list with sampled steps and omitted placeholders.
        """
        n_steps = len(steps_only)

        # Determine which step indices to keep
        if n_steps <= sampling_n:
            # Keep all steps
            kept_step_set = set(step_indices)
        else:
            kept_step_set = self._select_steps(
                steps_only, step_indices, sampling_n,
            )

        # Rebuild messages in original order
        result = list(base_messages)
        for orig_idx, item in indexed_items:
            if isinstance(item, StepRecord):
                if orig_idx in kept_step_set:
                    result.extend(item.to_messages())
                else:
                    # Omitted placeholder
                    result.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=[{
                                "type": "text",
                                "text": (
                                    f"[Step {item.step_number}] "
                                    f"Omitted: action-observation pair"
                                ),
                            }],
                        )
                    )
            else:
                # PlanState / SummaryState — always included
                result.extend(item.to_messages())

        return result

    def _select_steps(
        self,
        steps_only: List[StepRecord],
        step_indices: List[int],
        sampling_n: int,
    ) -> set:
        """Select which steps to keep using value-based sampling.

        Always preserves first and last step. Selects top-(K-2) from
        middle steps based on computed values.

        Returns:
            Set of original indices (into self._all_items) of kept steps.
        """
        n = len(steps_only)
        if n <= sampling_n:
            return set(step_indices)

        # Always keep first and last
        kept = {step_indices[0], step_indices[-1]}

        if sampling_n <= 2 or n <= 2:
            return kept

        # Compute values for middle steps
        middle_steps = steps_only[1:-1]
        middle_indices = step_indices[1:-1]

        observations = [s.observations or "" for s in middle_steps]
        rewards = [0.0] * len(middle_steps)  # No per-step reward
        values = _calculate_values(
            observations, rewards,
            w1=self._w1, w2=self._w2, w3=self._w3,
        )

        # Select top-(sampling_n - 2) from middle
        m = sampling_n - 2
        if m >= len(middle_steps):
            kept.update(middle_indices)
        else:
            top_positions = np.argsort(values)[-m:]
            for pos in top_positions:
                kept.add(middle_indices[pos])

        kept_step_numbers = sorted(
            self._all_items[idx].step_number
            for idx in kept
            if isinstance(self._all_items[idx], StepRecord)
        )
        logger.info(
            f"HierarchicalMemory: value-based selection kept steps "
            f"{kept_step_numbers} out of {n} total"
        )

        return kept

    def update(self, step: StepRecord) -> None:
        self._all_items.append(step)
        n_steps = sum(1 for item in self._all_items if isinstance(item, StepRecord))
        logger.debug(
            f"HierarchicalMemory: added step {step.step_number} "
            f"(total steps: {n_steps}, memory_size: {self._memory_size})"
        )

    def update_plan(self, plan: PlanState) -> None:
        self._all_items.append(plan)

    def update_summary(self, summary: SummaryState) -> None:
        self._all_items.append(summary)

    def get_all_steps(self) -> List[StepRecord]:
        return [s for s in self._all_items if isinstance(s, StepRecord)]
