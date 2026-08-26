"""
AgentFoldMemory: LLM-guided step compression via structured JSON fields.

Based on the AgentFold codebase. The LLM is instructed to include
compression instructions in its output, specifying which steps to fold together.
Memory maintains a step_list with {start, end, content} entries that can
represent individual or compressed step ranges.

The compression is triggered by the agent's own output (not by the memory
module itself), making it a cooperative compression scheme.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from scripts.kernel.protocols import BaseMemory
from scripts.kernel.types import (
    MemoryView, Message, PlanState, StepRecord, SummaryState, TaskInput,
)
from scripts.models.base import MessageRole
logger = logging.getLogger(__name__)


class MemoryStrategy(BaseMemory):
    """Step-based history folding memory.

    Maintains a step_list where each entry is:
        {'start': int, 'end': int, 'content': str}

    - Single steps have start == end
    - Compressed ranges have start < end with summarized content
    - The most recent step is always shown uncompressed
    - Older steps get [Compressed Step N] or [Compressed Step N to M] headers

    Compression is triggered externally via apply_compression() when the
    action module detects compression instructions in the LLM response.
    """

    def __init__(self, prompts=None):
        self.prompts = prompts or {}
        self._system_prompt: str = ""
        self._task: Optional[TaskInput] = None
        self._step_list: List[Dict[str, Any]] = []
        self._raw_steps: List[Union[StepRecord, PlanState, SummaryState]] = []
        self._next_step_id: int = 0

    def initialize(self, system_prompt: str, task: TaskInput) -> None:
        self._system_prompt = system_prompt
        self._task = task
        self._step_list = []
        self._raw_steps = []
        self._next_step_id = 0

    def build_context(self, plan: Optional[PlanState] = None) -> MemoryView:
        """Build context using the folded step list."""
        messages: List[Message] = []

        # System prompt with compression instructions
        system_text = self._system_prompt
        messages.append(
            Message(
                role=MessageRole.SYSTEM,
                content=[{"type": "text", "text": system_text}],
            )
        )

        # Task
        messages.extend(self._task.to_messages())

        # Format folded history as a single context block
        if self._step_list:
            previous_steps = self._format_previous_steps()
            messages.extend(previous_steps)
            # messages.append(
            #     Message(
            #         role=MessageRole.USER,
            #         content=[{
            #             "type": "text",
            #             "text": f"### Previous Steps\n{previous_steps}",
            #         }],
            #     )
            # )

        return MemoryView(messages=messages)

    def update(self, step: StepRecord) -> None:
        """Add a new step to the step list."""
        self._raw_steps.append(step)

        # Build step content string
        tools_str = ""
        if step.tool_calls:
            parts = []
            for tc in step.tool_calls:
                parts.append(f"Tool: {tc.name}, Args: {tc.arguments}")
            tools_str = "\n".join(parts)

        content_parts = []
        if step.action_think:
            content_parts.append(f"**Motivation:** {step.action_think}")
        if tools_str:
            content_parts.append(f"**Tool call:** {tools_str}")
        if step.observations:
            # Truncate long observations
            obs = step.observations
            content_parts.append(f"**Tool response:** {obs}")

        step_content = "\n".join(content_parts) if content_parts else "No output"

        self._step_list.append({
            "start": self._next_step_id,
            "end": self._next_step_id,
            "content": step_content,
        })
        self._next_step_id += 1

    def update_plan(self, plan: PlanState) -> None:
        self._raw_steps.append(plan)
        self._step_list.append({
            "start": self._next_step_id,
            "end": self._next_step_id,
            "content": f"**Plan:** {plan.plan[:1000]}",
        })
        self._next_step_id += 1

    def update_summary(self, summary: SummaryState) -> None:
        self._raw_steps.append(summary)
        self._step_list.append({
            "start": self._next_step_id,
            "end": self._next_step_id,
            "content": f"**Summary:** {summary.summary[:1000]}",
        })
        self._next_step_id += 1

    def get_all_steps(self) -> List[StepRecord]:
        return [s for s in self._raw_steps if isinstance(s, StepRecord)]

    def apply_compression(self, compress_range: List[int], compress_text: str) -> None:
        """Apply compression to a range of steps.

        Called by the action module when it detects compression instructions in the
        LLM response.

        Args:
            compress_range: List of step IDs to compress (e.g., [0, 1, 2]).
            compress_text: LLM-generated compressed summary of those steps.
        """
        if not compress_range:
            return

        start_step = compress_range[0]
        end_step = compress_range[-1]

        new_compressed = {
            "start": start_step,
            "end": end_step,
            "content": compress_text,
        }

        # Remove old steps in the compression range
        self._step_list = [
            step for step in self._step_list
            if not (start_step <= step["start"] <= end_step)
        ]

        # Add the new compressed step
        self._step_list.append(new_compressed)

        # Sort by start ID
        self._step_list.sort(key=lambda x: x["start"])

        logger.info(
            f"AgentFoldMemory: compressed steps {start_step}-{end_step} "
            f"({len(compress_range)} steps → 1 entry, len={len(compress_text)})"
        )

    def _format_previous_steps(self) -> str:
        """Format the step list with appropriate headers.

        - Latest step gets [Step N] header (uncompressed)
        - Older single steps get [Compressed Step N]
        - Range entries get [Compressed Step N to M]
        """
        if not self._step_list:
            return "EMPTY"

        self._step_list.sort(key=lambda x: x["start"])
        max_id = max(item["start"] for item in self._step_list)

        parts = []
        for item in self._step_list:
            start_id = item["start"]
            end_id = item["end"]
            content = item["content"]

            if start_id == end_id:
                if start_id == max_id:
                    header = f"[Step {start_id}]"
                else:
                    header = f"[Compressed Step {start_id}]"
            else:
                header = f"[Compressed Step {start_id} to {end_id}]"

            parts.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=[{
                        "type": "text",
                        "text": f"**{header}**\n{content}",
                    }],
                )
            )

            # parts.append(f"**{header}**\n{content}")

        # return "\n\n".join(parts)
        return parts
