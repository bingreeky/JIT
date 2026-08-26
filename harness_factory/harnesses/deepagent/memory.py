"""
DeepAgentMemory: three-tier brain-inspired memory with thought folding.

Adapted from DeepAgent (https://github.com/DeepAgent).

Core idea: when the agent's reasoning becomes too long or stuck, it can
trigger a "thought fold" which generates three structured memories:
  - Episode Memory: key events, milestones, progress summary
  - Working Memory: current goal, challenges, next actions
  - Tool Memory: tool usage patterns, success rates, derived rules

After folding, the context resets to [system_prompt + memories + task],
allowing fresh reasoning guided by compressed experience.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union

from scripts.kernel.protocols import BaseMemory
from scripts.kernel.types import (
    MemoryView, Message, PlanState, StepRecord, SummaryState, TaskInput,
)
from scripts.models.base import MessageRole


logger = logging.getLogger(__name__)


class MemoryStrategy(BaseMemory):
    """Three-tier folding memory with episodic, working, and tool memories.

    Stores all steps. When fold() is called by the action module:
    1. Generates three memory types via LLM (episodic, working, tool)
    2. Resets working context to [system_with_memories, task]
    3. Subsequent steps accumulate on top of the reset context

    PlanState and SummaryState items are tracked but not included
    in fold reasoning (they are structural).
    """

    def __init__(
        self,
        prompts=None,
        max_folds: int = 3,
    ):
        self.prompts = prompts or {}
        self._max_folds = max_folds

        self._system_prompt: str = ""
        self._task: Optional[TaskInput] = None
        self._model: Optional[Callable] = None

        # Full history (never reset, for trajectory logging)
        self._all_items: List[Union[StepRecord, PlanState, SummaryState]] = []

        # Fold state
        self._fold_count: int = 0
        self._fold_memories: Optional[Dict[str, str]] = None  # {episodic, working, tool}
        self._pre_fold_items_count: int = 0  # items before last fold

        # Interaction tracking (for fold prompts)
        self._interactions: List[Dict[str, Any]] = []

    def set_model(self, model: Callable) -> None:
        """Set the LLM callable for memory generation during folding."""
        self._model = model

    def initialize(self, system_prompt: str, task: TaskInput) -> None:
        self._system_prompt = system_prompt
        self._task = task
        self._all_items = []
        self._fold_count = 0
        self._fold_memories = None
        self._pre_fold_items_count = 0
        self._interactions = []
        logger.info(
            f"DeepAgentMemory: initialized (max_folds={self._max_folds})"
        )

    def build_context(self, plan: Optional[PlanState] = None) -> MemoryView:
        """Build context, incorporating fold memories if present."""
        messages: List[Message] = []

        # System prompt (with fold memories injected if available)
        system_text = self._system_prompt
        if self._fold_memories:
            memory_block = self._format_fold_memories()
            system_text = system_text + "\n\n" + memory_block

        messages.append(
            Message(
                role=MessageRole.SYSTEM,
                content=[{"type": "text", "text": system_text}],
            )
        )

        # Task
        messages.extend(self._task.to_messages())

        # Steps: only those AFTER the last fold
        post_fold_items = self._all_items[self._pre_fold_items_count:]
        for item in post_fold_items:
            messages.extend(item.to_messages())

        return MemoryView(
            messages=messages,
            metadata={
                "fold_count": self._fold_count,
                "total_items": len(self._all_items),
                "post_fold_items": len(post_fold_items),
            },
        )

    def can_fold(self) -> bool:
        """Check if folding is still allowed."""
        return self._fold_count < self._max_folds

    def fold(self, model: Callable, task: str) -> bool:
        """Generate three-tier memories and reset context.

        Called by the action module when <fold_thought> is detected.

        Args:
            model: LLM callable for memory generation.
            task: Task description for context.
        Returns:
            True if fold succeeded, False otherwise.
        """
        if not self.can_fold():
            logger.warning(
                f"DeepAgentMemory: fold rejected "
                f"(count={self._fold_count} >= max={self._max_folds})"
            )
            return False

        use_model = model or self._model
        if use_model is None:
            logger.error("DeepAgentMemory: no model available for folding")
            return False

        # Build reasoning history text from all items since last fold
        post_fold_items = self._all_items[self._pre_fold_items_count:]
        reasoning_text = self._build_reasoning_text(post_fold_items)

        # Build tool call history for tool memory
        tool_call_history = self._build_tool_call_history()

        # Build available tools description
        available_tools = ""  # Will be empty; tools are in system prompt

        logger.info(
            f"DeepAgentMemory: generating fold #{self._fold_count + 1} "
            f"({len(post_fold_items)} items to compress)"
        )

        # Generate three memories sequentially
        try:
            episodic = self._generate_episode_memory(
                use_model, task, reasoning_text, available_tools,
            )
            working = self._generate_working_memory(
                use_model, task, reasoning_text, available_tools,
            )
            tool_mem = self._generate_tool_memory(
                use_model, task, reasoning_text, tool_call_history, available_tools,
            )

            self._fold_memories = {
                "episodic": episodic,
                "working": working,
                "tool": tool_mem,
            }
            self._pre_fold_items_count = len(self._all_items)
            self._fold_count += 1

            # Record fold in interactions
            self._interactions.append({
                "type": "thought_folding",
                "episode_memory": episodic,
                "working_memory": working,
                "tool_memory": tool_mem,
            })

            logger.info(
                f"DeepAgentMemory: fold #{self._fold_count} complete "
                f"(episodic={len(episodic)} chars, "
                f"working={len(working)} chars, "
                f"tool={len(tool_mem)} chars)"
            )
            return True

        except Exception as e:
            logger.error(f"DeepAgentMemory: fold failed: {e}")
            return False

    def update(self, step: StepRecord) -> None:
        self._all_items.append(step)

        # Track interactions for fold prompts
        if step.tool_calls:
            for tc in step.tool_calls:
                self._interactions.append({
                    "type": "tool_call",
                    "tool_call_query": json.dumps(tc.dict()),
                    "tool_response": step.observations or "",
                })

        logger.debug(
            f"DeepAgentMemory: added step {step.step_number} "
            f"(total: {len(self._all_items)}, since fold: "
            f"{len(self._all_items) - self._pre_fold_items_count})"
        )

    def update_plan(self, plan: PlanState) -> None:
        self._all_items.append(plan)

    def update_summary(self, summary: SummaryState) -> None:
        self._all_items.append(summary)

    def get_all_steps(self) -> List[StepRecord]:
        return [s for s in self._all_items if isinstance(s, StepRecord)]

    # ── Private: Memory generation ──

    def _format_fold_memories(self) -> str:
        """Format fold memories as a text block for system prompt injection."""
        if not self._fold_memories:
            return ""
        return (
            "Memory of previous folded thoughts:\n\n"
            "Episode Memory:\n"
            f"{self._fold_memories['episodic']}\n\n"
            "Working Memory:\n"
            f"{self._fold_memories['working']}\n\n"
            "Tool Memory:\n"
            f"{self._fold_memories['tool']}"
        )

    def _build_reasoning_text(
        self, items: List[Union[StepRecord, PlanState, SummaryState]]
    ) -> str:
        """Build a text representation of reasoning history for fold prompts."""
        parts = []
        for item in items:
            if isinstance(item, StepRecord):
                if item.action_think:
                    parts.append(f"Think: {item.action_think}")
                for tc in item.tool_calls:
                    parts.append(
                        f"Tool call: {tc.name}({json.dumps(tc.arguments)})"
                    )
                if item.observations:
                    parts.append(f"Observation: {item.observations[:2000]}")
                if item.error:
                    parts.append(f"Error: {item.error}")
            elif isinstance(item, PlanState):
                parts.append(f"Plan: {item.plan}")
            elif isinstance(item, SummaryState):
                parts.append(f"Summary: {item.summary}")
        return "\n".join(parts)

    def _build_tool_call_history(self) -> str:
        """Build tool call history for tool memory generation."""
        history = []
        for interaction in self._interactions:
            if interaction.get("type") == "tool_call":
                history.append({
                    "tool_call": interaction.get("tool_call_query", ""),
                    "tool_response": interaction.get("tool_response", "")[:1000],
                })
        return json.dumps(history, indent=2, ensure_ascii=False) if history else "No tool calls yet."

    def _generate_episode_memory(
        self, model: Callable, task: str, reasoning: str, tools: str,
    ) -> str:
        """Generate episodic memory via LLM. Mirrors DeepAgent's get_episode_memory_instruction."""
        prompt = f"""You are a memory compression assistant. Your task is to summarize the key events and decisions in the agent's reasoning process into structured episode memory.

Task:
{task}

Full reasoning history:
{reasoning}

Instructions:
1. Identify major milestones, subgoal completions, and strategic decisions
2. Extract only the most critical events that provide experience for long-term goals
3. Output in this JSON format:
```json
{{
  "task_description": "A general summary of what the reasoning history has been doing and the overall goals it has been striving for.",
  "key_events": [
    {{
      "step": "step number",
      "description": "A detailed description of the specific action taken, decision made, or milestone achieved at this step.",
      "outcome": "A detailed account of the direct result or feedback received from this action."
    }}
  ],
  "current_progress": "A general summary of the current progress, including what has been completed and what is left."
}}
```

Now generate the episode memory. Directly output the JSON format episode memory."""

        messages = [
            {
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        try:
            response = model(messages)
            return response.content or ""
        except Exception as e:
            logger.error(f"DeepAgentMemory: episode memory generation failed: {e}")
            return "{}"

    def _generate_working_memory(
        self, model: Callable, task: str, reasoning: str, tools: str,
    ) -> str:
        """Generate working memory via LLM. Mirrors DeepAgent's get_working_memory_instruction."""
        prompt = f"""You are a working memory manager. Create a concise snapshot of the agent's CURRENT working state.

Task:
{task}

Full reasoning history:
{reasoning}

Instructions:
1. Extract ONLY immediate goals, current challenges, and next steps
2. Ignore completed/historical information
3. Output in this JSON format:
```json
{{
  "immediate_goal": "A clear summary of the current subgoal you are actively working toward.",
  "current_challenges": "A concise summary of the main obstacles or difficulties presently encountered.",
  "next_actions": [
    {{
      "type": "tool_call/planning/decision",
      "description": "The next concrete action to advance the task."
    }}
  ]
}}
```

Now generate the current working memory. Directly output the JSON format working memory."""

        messages = [
            {
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        try:
            response = model(messages)
            return response.content or ""
        except Exception as e:
            logger.error(f"DeepAgentMemory: working memory generation failed: {e}")
            return "{}"

    def _generate_tool_memory(
        self, model: Callable, task: str, reasoning: str,
        tool_call_history: str, tools: str,
    ) -> str:
        """Generate tool memory via LLM. Mirrors DeepAgent's get_tool_memory_instruction."""
        prompt = f"""You are a tool experience recorder. Synthesize tool usage patterns into structured knowledge.

Task:
{task}

Full reasoning history:
{reasoning}

Tool Call History (in chronological order):
{tool_call_history}

Instructions:
1. Analyze successful/unsuccessful tool patterns
2. Extract metadata about each tool's effective parameters, failure modes, and response structures
3. Output in this JSON format:
```json
{{
  "tools_used": [
    {{
      "tool_name": "string",
      "success_rate": "float",
      "effective_parameters": ["param1", "param2"],
      "common_errors": ["error_type1"],
      "response_pattern": "description of typical output",
      "experience": "Summary of experience using this tool."
    }}
  ],
  "derived_rules": [
    "When X condition occurs, prefer tool Y"
  ]
}}
```

Now generate the tool memory. Directly output the JSON format tool memory."""

        messages = [
            {
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        try:
            response = model(messages)
            return response.content or ""
        except Exception as e:
            logger.error(f"DeepAgentMemory: tool memory generation failed: {e}")
            return "{}"
