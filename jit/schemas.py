"""Schemas for meta agent inputs/outputs."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class MetaAgentRequest:
    """User-facing request for one meta-agent run."""

    benchmark_adapter: Any
    item: Dict[str, Any]
    tools: List[str] = field(default_factory=list)
    max_steps: int = 7
    max_repairs: int = 3
    repair_only_on_error: bool = False
    # When True, on a failed round regenerate a fresh harness from scratch
    # instead of repairing the existing one. max_repairs then bounds the number
    # of regeneration attempts. Default False preserves the repair behavior.
    regenerate_on_error: bool = False
    benchmark_config: Dict[str, Any] = field(default_factory=dict)
    # Best-of-n selection pipelines: stop right after harness generation and
    # return (no validation/repair round is run, no executor calls are made).
    generate_only: bool = False
    # Best-of-n selection pipelines: path to a previously generated harness
    # whose files are installed into the workspace instead of generating one;
    # the validate/repair loop then runs unchanged.
    preset_harness_dir: str = ""


@dataclass
class ToolExecutionRecord:
    """Single tool execution record for observability."""

    step: int
    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_latency_sec: float
    success: bool
    output_preview: str


@dataclass
class ValidationRecord:
    """Validation summary for one small-sample run."""

    passed: bool
    benchmark: str
    error: str = ""
    evaluation_result: Dict[str, Any] = field(default_factory=dict)
    api_latency_sec: float = 0.0
    tool_latency_sec: float = 0.0
    input_token_count: int = 0
    output_token_count: int = 0
    total_token_count: int = 0
    model: str = ""
    tool_set: List[str] = field(default_factory=list)
    steps_used: int = 0
    trajectories: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class MetaAgentResult:
    """Top-level run result."""

    generation_success: bool
    model: str = ""
    validation_records: List[ValidationRecord] = field(default_factory=list)
    meta_agent_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    # Number of times the harness was regenerated from scratch (regenerate_on_error path).
    number_of_regenerations: int = 0
    # Number of regenerations triggered by the pre-flight self-review gate
    # (meta_review config): reviewer judged the harness unqualified.
    number_of_review_regenerations: int = 0
    # Which reference material the generation prompt carried ("desc" = the
    # no-code design catalogue of all 11 seed harnesses, "code" = full source
    # of the sampled harnesses below), and, in code mode, what was sampled.
    reference_mode: str = "desc"
    reference_harnesses: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict suitable for JSON serialization."""
        data = asdict(self)
        return data
