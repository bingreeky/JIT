"""
Abstract base classes (protocols) for the four pluggable modules.

These define the FIXED interface that every harness implementation must satisfy.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .types import (
    Directive, MemoryView, PlanState, RunResult, RuntimeContext,
    StepRecord, SummaryState, TaskInput, ToolSelection,
)


class BaseMemory(ABC):
    """Inside-trial working memory manager.

    Responsibilities:
    1. Initialize internal state at the start of a run
    2. Build working context (MemoryView) for the next LLM call
    3. Update internal state after each step completes
    """

    @abstractmethod
    def initialize(self, system_prompt: str, task: TaskInput) -> None:
        """Called once at the start of each run.

        Args:
            system_prompt: The system prompt string.
            task: The initial task input.
        """
        ...

    @abstractmethod
    def build_context(self, plan: Optional[PlanState] = None) -> MemoryView:
        """Build the working context for the next LLM call.

        Converts internal memory state into a list of messages.

        Args:
            plan: Current plan state (optional, for plan-aware memory).
        Returns:
            MemoryView with messages ready for LLM.
        """
        ...

    @abstractmethod
    def update(self, step: StepRecord) -> None:
        """Called after each action step completes.

        Store, compress, summarize, or fold the step into memory.

        Args:
            step: The completed step record.
        """
        ...

    @abstractmethod
    def update_plan(self, plan: PlanState) -> None:
        """Called when a new plan or plan update is generated."""
        ...

    @abstractmethod
    def update_summary(self, summary: SummaryState) -> None:
        """Called when a summary/adaptation is generated."""
        ...

    def get_all_steps(self) -> List[StepRecord]:
        """Return all raw steps (for trajectory logging)."""
        return []


class BasePlanning(ABC):
    """Task structuring and directive generation.

    Responsibilities:
    1. Generate an initial plan from the task
    2. Decide if replanning is needed based on execution state
    3. Update the plan when needed
    4. Provide a directive for the current step
    5. Provide planning guidance for system-prompt injection
    """

    @abstractmethod
    def init_plan(
        self,
        task: str,
        memory_view: MemoryView,
        tool_schemas: str,
        model: Callable,
    ) -> PlanState:
        """Generate initial plan at the start of a run.

        Args:
            task: Task description.
            memory_view: Current memory context.
            tool_schemas: JSON string of available tool schemas.
            model: LLM callable (messages -> ChatMessage).
        Returns:
            PlanState with the generated plan.
        """
        ...

    @abstractmethod
    def should_replan(self, step_number: int, step: StepRecord) -> bool:
        """Decide whether replanning is needed after this step.

        Args:
            step_number: Current step index.
            step: The just-completed step.
        Returns:
            True if replanning should happen.
        """
        ...

    @abstractmethod
    def update_plan(
        self,
        task: str,
        step_number: int,
        memory_view: MemoryView,
        model: Callable,
    ) -> SummaryState:
        """Update/adapt the plan based on execution progress.

        Args:
            task: Original task.
            step_number: Current step index.
            memory_view: Current memory context.
            model: LLM callable.
        Returns:
            SummaryState with progress summary and updated directive.
        """
        ...

    @abstractmethod
    def get_directive(self) -> Directive:
        """Get the current directive for the action module.

        Returns:
            Directive with the current sub-goal text.
        """
        ...

class BaseAction(ABC):
    """Run-level execution protocol.

    The Action module OWNS the main loop. The kernel calls action.run()
    and action orchestrates everything else via the RuntimeContext.

    Responsibilities:
    1. Execute the full run loop
    2. Provide action guidance for system-prompt injection
    """

    @abstractmethod
    def run(self, task: str, ctx: RuntimeContext) -> RunResult:
        """Execute the full agent run.

        Args:
            task: Task description.
            ctx: RuntimeContext providing access to all kernel services.
        Returns:
            RunResult with final answer and trajectory.
        """
        ...


class BaseToolPolicy(ABC):
    """Per-step tool selection.

    Responsibilities:
    1. Initialize with the full tool catalog
    2. Select available tools for the current step
    3. (Optional) Manage domain-knowledge skills for prompt injection
    """

    @abstractmethod
    def initialize(self, tool_catalog: Dict[str, Any], enable_skills: bool = False) -> None:
        """Called once with the full set of registered tools.

        Args:
            tool_catalog: {tool_name: Tool} dict of all available tools.
            enable_skills: Whether to load domain-knowledge skills into the prompt.
                Only benchmarks that need skills (e.g., AgentIF-OneDay)
                should set this to True. Defaults to False.
        """
        ...

    @abstractmethod
    def select_tools(
        self,
        task: str,
        step_number: int,
        memory_view: MemoryView,
        plan: Optional[PlanState] = None,
    ) -> ToolSelection:
        """Select which tools to expose for this step.

        Args:
            task: Current task.
            step_number: Current step index.
            memory_view: Current memory context.
            plan: Current plan state.
        Returns:
            ToolSelection with filtered tools and their JSON schemas.
        """
        ...

    def get_skills_prompt(self) -> str:
        """Return skill instructions to inject into the system prompt.

        Default implementation returns empty string (no skills).
        Override to load and format domain-knowledge skills.
        """
        return ""
