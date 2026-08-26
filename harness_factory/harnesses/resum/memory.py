"""
ReSumMemory: periodic summarization-based memory management.

Faithfully reproduces the ReSum codebase (https://github.com/ReSum).
Primary trigger: token-count-based (summarize when context reaches 90%
of max_context_tokens). After summarization, the working context is
a clean restart: system_prompt + summary_observation. No trailing steps.

The one allowed architectural difference: the original uses a separate
vLLM-served ReSum Tool server; we call the main LLM directly.
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union

from scripts.kernel.protocols import BaseMemory
from scripts.kernel.token_counter import count_tokens_messages
from scripts.kernel.types import (
    MemoryView, Message, PlanState, StepRecord, SummaryState, TaskInput,
)
from scripts.models.base import MessageRole


logger = logging.getLogger(__name__)




class MemoryStrategy(BaseMemory):
    """Token-count-triggered summarization memory.

    Faithfully mirrors ReSum's MultiTurnReactAgent._run() logic:
    - Primary trigger: token_count >= max_context_tokens * 0.9
    - Fallback trigger: every summary_interval rounds (default 1000, ~never)
    - After summarization: clean restart [system_prompt, summary_observation]
    - Summary stored WITH <summary> tags (matching original)
    """

    def __init__(
        self,
        prompts=None,
        max_context_tokens: int = 31768,  # Original: MAX_CONTEXT * 1024 - 1000
        summary_interval: int = 4,     # Original: default 1000 (rare fallback)
    ):
        """
        Args:
            max_context_tokens: Token budget for working context.
                Original default: 32 * 1024 - 1000 = 31768.
            summary_interval: Fallback — summarize every N rounds regardless.
                Original default: 1000 (effectively a safety-net).
        """
        self.prompts = prompts or {}
        self._max_context_tokens = max_context_tokens
        self._summary_interval = summary_interval

        self._system_prompt: str = ""
        self._task: Optional[TaskInput] = None
        self._model: Optional[Callable] = None

        # Working messages — mirrors original's `messages` list.
        # After summarization, this is reset to [system, summary_observation].
        self._messages: List[Message] = []

        # Full trajectory — mirrors original's `full_trajectory`.
        # Never reset; accumulates everything for trajectory logging.
        self._full_trajectory: List[Union[StepRecord, PlanState, SummaryState]] = []

        # Summarization state
        self._last_summary: Optional[str] = None  # None = no summary yet
        self._summary_count: int = 0
        self._round: int = 0  # Tracks rounds for the interval fallback

    def set_model(self, model: Callable) -> None:
        """Set the LLM callable for summarization."""
        self._model = model

    def initialize(self, system_prompt: str, task: TaskInput) -> None:
        self._system_prompt = system_prompt
        self._task = task
        self._full_trajectory = []
        self._last_summary = None
        self._summary_count = 0
        self._round = 0

        # Initialize working messages: [system, user_question]
        # (mirrors original react_agent.py lines 89-92)
        self._messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=[{"type": "text", "text": self._system_prompt}],
            ),
            Message(
                role=MessageRole.USER,
                content=[{"type": "text", "text": task.task}],
            ),
        ]

    def build_context(self, plan: Optional[PlanState] = None) -> MemoryView:
        """Return the current working messages directly.

        After summarization, this is [system_prompt, summary_observation].
        Before summarization / between summaries, this grows with each step.
        """
        return MemoryView(
            messages=list(self._messages),
            metadata={
                "summary_count": self._summary_count,
                "token_count": count_tokens_messages(self._messages),
            },
        )

    def update(self, step: StepRecord) -> None:
        """Append step messages to working context, then check summarization."""
        self._full_trajectory.append(step)
        self._round += 1

        # Append the step's messages to working context
        # (mirrors original: append assistant content, then tool response)
        step_msgs = step.to_messages()
        self._messages.extend(step_msgs)

        # Check if summarization should fire
        if self._should_summarize():
            self._do_summarize()

    def update_plan(self, plan: PlanState) -> None:
        self._full_trajectory.append(plan)
        # Append plan messages to working context
        self._messages.extend(plan.to_messages())

    def update_summary(self, summary: SummaryState) -> None:
        self._full_trajectory.append(summary)
        # Append summary messages to working context
        self._messages.extend(summary.to_messages())

    def get_all_steps(self) -> List[StepRecord]:
        return [s for s in self._full_trajectory if isinstance(s, StepRecord)]

    # ── Private: Summarization trigger ──

    def _should_summarize(self) -> bool:
        """Token-count-based trigger (primary) + interval fallback.

        Mirrors original react_agent.py line 135:
          should_summarize = (
              (RESUM and token_count >= max_tokens * 0.9)
              or round % summary_iteration == 0
          ) and num_llm_calls_available
        """
        if self._model is None:
            return False

        # Primary trigger: token count >= 90% of budget
        token_count = count_tokens_messages(self._messages)
        if token_count >= self._max_context_tokens * 0.9:
            logger.info(
                f"ReSumMemory: token trigger fired "
                f"({token_count} >= {int(self._max_context_tokens * 0.9)})"
            )
            return True

        # Fallback trigger: every summary_interval rounds
        if (self._summary_interval > 0
                and self._round > 0
                and self._round % self._summary_interval == 0):
            logger.info(f"ReSumMemory: interval trigger fired (round {self._round})")
            return True

        return False

    def _do_summarize(self) -> None:
        """Perform LLM-based summarization and reset working messages.

        Mirrors original react_agent.py lines 136-156 and summary_utils.py.
        """
        if self._model is None:
            return

        question = self._task.task if self._task else ""

        # Build history: messages[2:] (skip system + question/summary_observation)
        # Mirrors original react_agent.py line 137: recent_messages = messages[2:].copy()
        recent_messages = self._messages[2:]

        # Convert to string: "\n".join([str(msg) for msg in recent_messages])
        # Mirrors original summary_utils.py line 51
        recent_history_str = "\n".join(str(msg) for msg in recent_messages)

        # Choose prompt
        if not self._last_summary:
            prompt_text = self.prompts["memory"]["query_summary_prompt"].replace(
                "{question}", question
            ).replace(
                "{recent_history_messages}", recent_history_str
            )
        else:
            prompt_text = self.prompts["memory"]["query_summary_prompt_last"].replace(
                "{question}", question
            ).replace(
                "{recent_history_messages}", recent_history_str
            ).replace(
                "{last_summary}", self._last_summary
            )

        # Call LLM — no system message (matches original: single user message)
        # Original summary_utils.py line 16: messages = [{"role":"user","content":query}]
        summary_messages = [
            {
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": prompt_text}],
            },
        ]

        try:
            response = self._model(summary_messages)
            content = response.content or ""

            # Post-process: strip <think> tags, extract <summary>, re-wrap
            # Mirrors original summary_utils.py lines 32-39
            if content:
                # Strip <think>...</think> blocks
                pattern = r'<think>.*?</think>'
                content = re.sub(pattern, '', content, flags=re.DOTALL).strip()

                # Extract content between <summary> tags
                try:
                    content = content.split("<summary>")[1].split("</summary>")[0]
                except (IndexError, ValueError):
                    pass  # Use full content if tags not found

                # Always re-wrap in <summary> tags (matches original line 39)
                summary_response = "<summary>" + content + "</summary>"
            else:
                summary_response = ""

            # Only update state if summary is non-empty
            # Mirrors original react_agent.py line 146: if summary_response:
            if summary_response:
                self._last_summary = summary_response
                self._summary_count += 1

                # Build summary observation text
                # Mirrors original react_agent.py lines 148-149
                new_observation = (
                    "Question: " + question
                    + "\nBelow is a summary of the previous conversation. "
                    "This summary condenses key information from earlier steps, "
                    "so please consider it carefully. Assess whether the summary "
                    "provides enough information to answer the question and use it "
                    "as the basis for further reasoning and information gathering "
                    "to answer the question.\n"
                    + "Summary: " + summary_response + "\n"
                )

                # Reset working messages: [system, summary_observation]
                # Mirrors original react_agent.py lines 150-153
                self._messages = [
                    Message(
                        role=MessageRole.SYSTEM,
                        content=[{"type": "text", "text": self._system_prompt}],
                    ),
                    Message(
                        role=MessageRole.USER,
                        content=[{"type": "text", "text": new_observation}],
                    ),
                ]

                new_token_count = count_tokens_messages(self._messages)
                logger.info(
                    f"ReSumMemory: summary #{self._summary_count} "
                    f"(len={len(summary_response)}), "
                    f"context reset to {new_token_count} tokens"
                )
            else:
                logger.warning("ReSumMemory: summarization returned empty, keeping current context")

        except Exception as e:
            logger.error(f"ReSumMemory: summarization failed: {e}")
            # On failure, keep current messages unchanged (matches original)
