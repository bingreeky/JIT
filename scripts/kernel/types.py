"""
Shared data types for the modular agent framework.

These are the FIXED protocol types that all modules communicate through.
Adapted from Flash-Searcher's memory.py step types.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

from scripts.models.base import MessageRole


# ── Message type (same as Flash-Searcher) ──

class Message(TypedDict):
    role: str
    content: str | list[dict]


# ── Tool call record ──

@dataclass
class ToolCall:
    """A single tool invocation record."""
    name: str
    arguments: Any
    id: str = ""

    def dict(self):
        return {"name": self.name, "arguments": self.arguments}


# ── Serialization helpers ──

def _serialize_messages(messages: List[Any]) -> List[dict]:
    """Serialize a list of Message/dict objects for JSON output."""
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append(msg)
        else:
            # TypedDict or other mapping-like
            try:
                result.append(dict(msg))
            except (TypeError, ValueError):
                result.append({"raw": str(msg)})
    return result


def _serialize_model_output(output: Any) -> Any:
    """Serialize model output (ChatMessage or similar) for JSON output."""
    if output is None:
        return None
    if isinstance(output, (str, int, float, bool)):
        return output
    if isinstance(output, dict):
        return output
    # ChatMessage-like objects: extract key fields
    result = {}
    for attr in ("content", "reasoning_content", "role", "tool_calls"):
        if hasattr(output, attr):
            val = getattr(output, attr)
            if val is not None:
                result[attr] = str(val) if not isinstance(val, (str, list, dict)) else val
    return result if result else str(output)


# ── Step record (= Flash-Searcher ActionStep) ──

@dataclass
class StepRecord:
    """One action step in the execution trace."""
    step_number: int = 0
    tool_calls: List[ToolCall] = field(default_factory=list)
    observations: str = ""
    observations_images: List[str] = field(default_factory=list)
    action_output: Any = None
    action_think: str = ""
    action_reasoning: str = ""
    error: Optional[Exception] = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    model_input_messages: Optional[List[Message]] = None
    model_output_messages: Any = None
    # Token usage tracking
    input_token_count: int = 0   # Tokens in the model input (prompt)
    output_token_count: int = 0  # Tokens in the model output (completion)
    total_token_count: int = 0   # input + output

    def to_messages(self, summary_mode: bool = False) -> List[Message]:
        """Convert to LLM message format (same logic as Flash-Searcher ActionStep)."""
        messages = []

        if self.tool_calls:
            tool_output = {"tools": [tc.dict() for tc in self.tool_calls]}
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=[{
                        "type": "text",
                        "text": "Calling tools:\n" + str(tool_output),
                    }],
                )
            )

        if self.observations:
            messages.append(
                Message(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[{
                        "type": "text",
                        "text": f"Tool calling observation:\n{self.observations}",
                    }],
                )
            )

        if self.error is not None:
            error_message = (
                "Error:\n" + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! "
                "If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                Message(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        return messages

    def dict(self):
        """Compact serialization for benchmarks and evaluation."""
        return {
            "step_number": self.step_number,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "observations": self.observations,
            "action_output": self.action_output,
            "action_think": self.action_think,
            "action_reasoning": self.action_reasoning,
            "error": str(self.error) if self.error else None,
            "duration": self.duration,
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "total_token_count": self.total_token_count,
        }

    def full_dict(self):
        """Full serialization including raw LLM messages, for trace saving."""
        d = self.dict()
        d["start_time"] = self.start_time
        d["end_time"] = self.end_time
        d["observations_images"] = self.observations_images
        d["model_input_messages"] = (
            _serialize_messages(self.model_input_messages)
            if self.model_input_messages else None
        )
        d["model_output_messages"] = (
            _serialize_model_output(self.model_output_messages)
            if self.model_output_messages else None
        )
        return d


# ── Plan state (= Flash-Searcher PlanningStep) ──

@dataclass
class PlanState:
    """Output of the planning module."""
    plan: str = ""
    plan_think: str = ""
    plan_reasoning: str = ""
    model_input_messages: Optional[List[Message]] = None

    def to_messages(self, summary_mode: bool = False) -> List[Message]:
        messages = []
        messages.append(
            Message(
                role=MessageRole.USER,
                content=[{"type": "text", "text": "Now, begin your planning analysis for this task!"}],
            )
        )
        messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": f"[PLAN]:\n{self.plan.strip()}"}],
            )
        )
        return messages


# ── Summary state (= Flash-Searcher SummaryStep) ──

@dataclass
class SummaryState:
    """Output of summary/adaptation."""
    summary: str = ""
    summary_reasoning: str = ""
    model_input_messages: Optional[List[Message]] = None

    def to_messages(self, summary_mode: bool = False) -> List[Message]:
        messages = []
        messages.append(
            Message(
                role=MessageRole.USER,
                content=[{"type": "text", "text": "Now, summarize and analysis the task completion status and provide recommendations for next steps!"}],
            )
        )
        messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": f"[SUMMARY]:\n{self.summary.strip()}"}],
            )
        )
        return messages


# ── Task input (= Flash-Searcher TaskStep) ──

@dataclass
class TaskInput:
    """Initial task description."""
    task: str = ""
    task_images: List[str] = field(default_factory=list)

    def to_messages(self, **kwargs) -> List[Message]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        return [Message(role=MessageRole.USER, content=content)]


# ── Memory view (output of memory.build_context) ──

@dataclass
class MemoryView:
    """The working context that Memory produces for the next LLM call."""
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Tool selection (output of tool_policy.select_tools) ──

@dataclass
class ToolSelection:
    """Which tools are available for the current step."""
    tools: Dict[str, Any] = field(default_factory=dict)
    tool_schemas_json: str = ""
    skills_prompt: str = ""  # Domain-knowledge skills for prompt injection


# ── Directive (output of planning.get_directive) ──

@dataclass
class Directive:
    """Current sub-goal instruction from Planning."""
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Run result (final output of Action) ──

@dataclass
class RunResult:
    """Final output of an agent run.

    Two-layer trajectory design:
    - trajectory: flat List[StepRecord] for backward compatibility.
      Benchmarks and evaluation use this (via .dict() on each step).
    - sub_runs: hierarchical nesting for complex Actions (ensemble, tree
      search, multi-session). Each sub_run is a full RunResult with its
      own trajectory. Simple Actions (ReAct) leave this empty.
    - metadata: action-specific data that doesn't fit into steps, e.g.
      critic response for ensemble, compression events for AgentFold.
    """
    answer: Any = None
    trajectory: List[StepRecord] = field(default_factory=list)
    terminated_reason: str = ""  # "final_answer" | "max_steps" | "error"
    metadata: Dict[str, Any] = field(default_factory=dict)
    sub_runs: List[Any] = field(default_factory=list)  # List['RunResult']

    def dict(self):
        """Compact serialization (trajectory uses StepRecord.dict())."""
        return {
            "answer": self.answer,
            "trajectory": [s.dict() for s in self.trajectory],
            "terminated_reason": self.terminated_reason,
            "metadata": self.metadata,
            "sub_runs": [sr.dict() for sr in self.sub_runs] if self.sub_runs else [],
        }

    def full_dict(self):
        """Full serialization for trace saving (includes raw LLM messages)."""
        return {
            "answer": self.answer,
            "trajectory": [s.full_dict() for s in self.trajectory],
            "terminated_reason": self.terminated_reason,
            "metadata": self.metadata,
            "sub_runs": [sr.full_dict() for sr in self.sub_runs] if self.sub_runs else [],
        }


# ── Runtime context (passed to Action module) ──

@dataclass
class RuntimeContext:
    """Services that the Action module can call during execution.

    This bridges the fixed kernel and the pluggable Action.
    """
    # Module references (will be set by kernel)
    memory: Any = None           # BaseMemory
    planning: Any = None         # BasePlanning
    tool_policy: Any = None      # BaseToolPolicy

    # Kernel services
    model: Any = None            # Callable: messages -> ChatMessage
    execute_tool: Any = None     # Callable: (tool_name, arguments) -> str
    get_tool_schemas: Any = None # Callable: (tools_dict) -> str
    logger: Any = None           # AgentLogger
    prompt_templates: Dict = field(default_factory=dict)

    # Configuration
    max_steps: int = 40
