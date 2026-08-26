"""
ROMAAction: recursive role-based execution with decomposition and aggregation.

Adapted from ROMA (Recursive Orchestrated Multi-Agent).

Flow:
  1. Atomizer + Planner (via planning module) decide decomposition
  2. If atomic → standard ReAct execution
  3. If decomposed → execute subtasks in dependency order (parallel up to 2)
     → each subtask can itself be decomposed (up to max_depth=3)
     → aggregate all results via Aggregator LLM call
  4. Return final answer
"""

import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import json_repair
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from scripts.kernel.protocols import BaseAction
from scripts.kernel.token_counter import count_tokens_messages, count_tokens_text
from scripts.kernel.types import (
    MemoryView, Message, PlanState, RunResult, RuntimeContext,
    StepRecord, SummaryState, TaskInput, ToolCall,
)
from scripts.kernel.monitoring import LogLevel, YELLOW_HEX
from scripts.models.base import MessageRole

from .memory import ArtifactStore, MemoryStrategy


logger = logging.getLogger(__name__)


def _parse_plan_metadata(plan: PlanState) -> Dict[str, Any]:
    """Extract ROMA decomposition metadata from PlanState.plan_reasoning."""
    try:
        return json.loads(plan.plan_reasoning) if plan.plan_reasoning else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _populate_template(template: str, variables: dict) -> str:
    from jinja2 import Template, StrictUndefined
    return Template(template, undefined=StrictUndefined).render(**variables)


def _topological_order(
    subtasks: List[Dict], dependencies: Dict[str, List[str]],
) -> List[List[Dict]]:
    """Compute topological layers for parallel execution.

    Returns list of layers, where each layer contains subtasks whose
    dependencies are all in previous layers. Subtasks within a layer
    can run in parallel.
    """
    idx_to_task = {str(st["index"]): st for st in subtasks}
    all_indices = set(idx_to_task.keys())

    # Build in-degree map
    in_degree = {idx: 0 for idx in all_indices}
    for idx in all_indices:
        deps = dependencies.get(idx, [])
        in_degree[idx] = len(deps)

    completed: Set[str] = set()
    layers: List[List[Dict]] = []

    while completed != all_indices:
        # Find all ready tasks (in-degree 0 among remaining)
        ready = [
            idx for idx in all_indices - completed
            if all(d in completed for d in dependencies.get(idx, []))
        ]

        if not ready:
            # Cycle detected or bad deps — force remaining
            remaining = list(all_indices - completed)
            logger.warning(
                f"ROMAAction: dependency cycle detected, forcing {remaining}"
            )
            layers.append([idx_to_task[idx] for idx in remaining])
            break

        layers.append([idx_to_task[idx] for idx in sorted(ready)])
        completed.update(ready)

    return layers


class ActionStrategy(BaseAction):
    """ROMA recursive execution: atomize → plan → execute/recurse → aggregate.

    Supports:
    - Recursive decomposition up to max_depth=3
    - Parallel execution of independent subtasks (max 2 concurrent)
    - Artifact store for dependency result passing
    - Aggregator for synthesizing subtask results
    """

    MAX_DEPTH = 3
    MAX_PARALLEL = 2

    def __init__(self, prompts=None):
        self.prompts = prompts or {}

    def run(self, task: str, ctx: RuntimeContext) -> RunResult:
        return self._run_recursive(task, ctx, depth=0)

    def _run_recursive(
        self,
        task: str,
        ctx: RuntimeContext,
        depth: int,
        dependency_context: str = "",
    ) -> RunResult:
        """Execute a task, potentially decomposing recursively.

        Args:
            task: Task description.
            ctx: RuntimeContext.
            depth: Current recursion depth.
            dependency_context: Results from dependency subtasks (if any).
        """
        trajectory: List[StepRecord] = []
        sub_runs: List[RunResult] = []

        # ── Step 0: Planning (Atomizer + Planner) ──
        memory_view = ctx.memory.build_context()
        tool_selection = ctx.tool_policy.select_tools(task, 0, memory_view)

        # At max depth, force atomic execution
        if depth >= self.MAX_DEPTH:
            logger.info(
                f"ROMAAction: depth {depth} >= max {self.MAX_DEPTH}, "
                f"forcing atomic execution"
            )
            plan = PlanState(
                plan="Max depth reached. Execute directly.",
                plan_reasoning=json.dumps({"is_atomic": True}),
            )
        else:
            plan = ctx.planning.init_plan(
                task, memory_view, tool_selection.tool_schemas_json, ctx.model,
            )

        ctx.memory.update_plan(plan)

        plan_meta = _parse_plan_metadata(plan)
        is_atomic = plan_meta.get("is_atomic", True)

        if depth == 0:
            ctx.logger.log_markdown(
                plan.plan, title="ROMA Plan", level=LogLevel.INFO,
            )

        # ── Atomic execution: standard ReAct loop ──
        if is_atomic:
            return self._execute_atomic(
                task, ctx, trajectory, dependency_context,
            )

        # ── Decomposed execution: subtasks → aggregate ──
        subtasks = plan_meta.get("subtasks", [])
        dependencies = plan_meta.get("dependencies", {})

        if not subtasks:
            return self._execute_atomic(task, ctx, trajectory, dependency_context)

        ctx.logger.log_rule(
            f"ROMA Decomposition (depth={depth}): {len(subtasks)} subtasks",
            level=LogLevel.INFO,
        )

        # Create artifact store for this decomposition level
        artifact_store = ArtifactStore()

        # Compute execution layers (topological order)
        layers = _topological_order(subtasks, dependencies)

        # Execute layers
        steps_per_subtask = max(ctx.max_steps // max(len(subtasks), 1), 5)

        for layer_idx, layer in enumerate(layers):
            ctx.logger.log_rule(
                f"Layer {layer_idx + 1}/{len(layers)}: "
                f"{len(layer)} subtask(s)",
                level=LogLevel.INFO,
            )

            if len(layer) == 1 or self.MAX_PARALLEL <= 1:
                # Sequential execution
                for st in layer:
                    result = self._execute_subtask(
                        st, ctx, depth, dependencies,
                        artifact_store, steps_per_subtask,
                    )
                    sub_runs.append(result)
            else:
                # Parallel execution (up to MAX_PARALLEL)
                results = self._execute_subtasks_parallel(
                    layer, ctx, depth, dependencies,
                    artifact_store, steps_per_subtask,
                )
                sub_runs.extend(results)

        # ── Aggregation ──
        ctx.logger.log_rule("ROMA Aggregation", level=LogLevel.INFO)
        final_answer = self._aggregate(
            task, ctx, artifact_store, trajectory, subtasks=subtasks,
        )

        return RunResult(
            answer=final_answer,
            trajectory=trajectory,
            terminated_reason="final_answer",
            sub_runs=sub_runs,
            metadata={
                "roma_depth": depth,
                "num_subtasks": len(subtasks),
                "decomposition": plan.plan,
            },
        )

    # ── Subtask execution ──

    def _execute_subtask(
        self,
        subtask: Dict,
        ctx: RuntimeContext,
        depth: int,
        dependencies: Dict[str, List[str]],
        artifact_store: ArtifactStore,
        max_steps: int,
    ) -> RunResult:
        """Execute a single subtask, potentially recursing."""
        idx = str(subtask["index"])
        goal = subtask["goal"]
        dep_indices = dependencies.get(idx, [])

        # Build dependency context
        dep_context = artifact_store.get_dependency_context(dep_indices)

        ctx.logger.log(
            Panel(Text(f"Subtask [{idx}]: {goal}")),
            level=LogLevel.INFO,
        )
        if dep_context:
            logger.info(
                f"ROMAAction: subtask [{idx}] has {len(dep_indices)} "
                f"dependencies: {dep_indices}"
            )

        # Create independent memory for subtask
        subtask_memory = MemoryStrategy(prompts=self.prompts)
        subtask_task = (
            f"{goal}\n\nContext from previous subtasks:\n{dep_context}"
            if dep_context else goal
        )

        # Build system prompt for subtask
        system_prompt = self._build_subtask_system_prompt(ctx)
        subtask_memory.initialize(system_prompt, TaskInput(task=subtask_task))

        # Create sub-context with limited steps
        sub_ctx = RuntimeContext(
            memory=subtask_memory,
            planning=ctx.planning,
            tool_policy=ctx.tool_policy,
            model=ctx.model,
            execute_tool=ctx.execute_tool,
            get_tool_schemas=ctx.get_tool_schemas,
            logger=ctx.logger,
            prompt_templates=ctx.prompt_templates,
            max_steps=max_steps,
        )

        # Recursive call at depth+1
        result = self._run_recursive(
            subtask_task, sub_ctx, depth + 1, dep_context,
        )

        # Store result in artifact store
        answer_str = str(result.answer or "")
        artifact_store.store_result(idx, goal, answer_str)

        ctx.logger.log(
            Text(
                f"Subtask [{idx}] done: {answer_str[:150]}...",
                style=f"bold {YELLOW_HEX}",
            ),
            level=LogLevel.INFO,
        )

        return result

    def _execute_subtasks_parallel(
        self,
        layer: List[Dict],
        ctx: RuntimeContext,
        depth: int,
        dependencies: Dict[str, List[str]],
        artifact_store: ArtifactStore,
        max_steps: int,
    ) -> List[RunResult]:
        """Execute subtasks in parallel (up to MAX_PARALLEL)."""
        logger.info(
            f"ROMAAction: executing {len(layer)} subtasks in parallel "
            f"(max {self.MAX_PARALLEL})"
        )
        results = [None] * len(layer)

        with ThreadPoolExecutor(max_workers=self.MAX_PARALLEL) as pool:
            futures = {}
            for i, st in enumerate(layer):
                future = pool.submit(
                    self._execute_subtask,
                    st, ctx, depth, dependencies,
                    artifact_store, max_steps,
                )
                futures[future] = i

            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    logger.error(
                        f"ROMAAction: parallel subtask {i} failed: {e}"
                    )
                    results[i] = RunResult(
                        answer=f"Error: {e}",
                        terminated_reason="error",
                    )

        return [r for r in results if r is not None]

    # ── Atomic execution (ReAct loop) ──

    def _execute_atomic(
        self,
        task: str,
        ctx: RuntimeContext,
        trajectory: List[StepRecord],
        dependency_context: str = "",
    ) -> RunResult:
        """Standard ReAct execution for an atomic task."""
        final_answer = None
        step_number = 1

        while final_answer is None and step_number <= ctx.max_steps:
            step_start_time = time.time()
            ctx.logger.log_rule(f"Step {step_number}", level=LogLevel.INFO)

            memory_view = ctx.memory.build_context()
            tool_selection = ctx.tool_policy.select_tools(
                task, step_number, memory_view,
            )

            # Build step prompt
            step_prompt = _populate_template(
                self.prompts["step"]["pre_messages"],
                variables={
                    "tool_functions_json": tool_selection.tool_schemas_json,
                    "task": task,
                },
            )

            messages = memory_view.messages + [{
                "role": "user",
                "content": [{"type": "text", "text": step_prompt}],
            }]

            # Call LLM
            try:
                response = ctx.model(messages)
            except Exception as e:
                logger.error(f"Model call failed at step {step_number}: {e}")
                step = StepRecord(
                    step_number=step_number, error=e,
                    start_time=step_start_time, end_time=time.time(),
                )
                step.duration = step.end_time - step.start_time
                ctx.memory.update(step)
                trajectory.append(step)
                step_number += 1
                continue

            input_tokens = count_tokens_messages(messages)
            output_tokens = count_tokens_text(
                getattr(response, 'content', '') or ''
            )

            step = StepRecord(
                step_number=step_number,
                model_input_messages=messages,
                model_output_messages=response,
                start_time=step_start_time,
                input_token_count=input_tokens,
                output_token_count=output_tokens,
                total_token_count=input_tokens + output_tokens,
            )

            try:
                tool_calls, think, final_answer = self._parse_and_execute(
                    response, step, ctx, task,
                )
            except Exception as e:
                logger.error(f"Step execution error: {e}")
                step.error = e

            step.end_time = time.time()
            step.duration = step.end_time - step.start_time
            ctx.memory.update(step)
            trajectory.append(step)
            step_number += 1

        if final_answer is None:
            final_answer = self._force_final_answer(task, ctx, trajectory, step_number)

        return RunResult(
            answer=final_answer,
            trajectory=trajectory,
            terminated_reason="final_answer" if step_number <= ctx.max_steps else "max_steps",
        )

    # ── Aggregation ──

    def _aggregate(
        self,
        original_task: str,
        ctx: RuntimeContext,
        artifact_store: ArtifactStore,
        trajectory: List[StepRecord],
        subtasks: Optional[List[Dict]] = None,
    ) -> Any:
        """Call Aggregator LLM to synthesize subtask results."""
        step_start_time = time.time()
        all_results = artifact_store.get_all_results()

        agg_prompt_template = self.prompts.get("planning", {}).get(
            "aggregator_prompt", ""
        )
        if not agg_prompt_template:
            agg_prompt_template = (
                "You are a result synthesizer. Given the original task and results "
                "from multiple subtasks, produce a comprehensive final answer.\n\n"
                "Original task: {task}\n\n"
                "Subtask results:\n{results}\n\n"
                "Synthesize these results into a single, complete answer to the original task. "
                "Your answer should be concise and directly address the question.\n\n"
                "Respond with ONLY a JSON object:\n"
                '{{"answer": "your comprehensive final answer here"}}'
            )

        prompt_text = agg_prompt_template.replace(
            "{task}", original_task
        ).replace(
            "{results}", all_results
        )

        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ]

        try:
            response = ctx.model(messages)
            content = response.content or ""

            # Build decomposition summary for trajectory readability
            decomp_summary = "ROMA Aggregation"
            if subtasks:
                subtask_lines = []
                for st in subtasks:
                    idx = str(st["index"])
                    goal = st["goal"]
                    result_preview = (artifact_store.get_result(idx) or "")[:100]
                    subtask_lines.append(
                        f"  [{idx}] {goal} -> {result_preview}..."
                    )
                decomp_summary = (
                    f"ROMA Aggregation of {len(subtasks)} subtasks:\n"
                    + "\n".join(subtask_lines)
                )

            # Record aggregation step
            input_tokens = count_tokens_messages(messages)
            output_tokens = count_tokens_text(content)
            step = StepRecord(
                step_number=len(trajectory) + 1,
                model_input_messages=messages,
                model_output_messages=response,
                start_time=step_start_time,
                end_time=time.time(),
                action_think=decomp_summary,
                action_reasoning="ROMA Aggregation",
                input_token_count=input_tokens,
                output_token_count=output_tokens,
                total_token_count=input_tokens + output_tokens,
            )
            step.duration = step.end_time - step.start_time

            # Parse answer
            answer = content
            try:
                parsed = json_repair.loads(content)
                if isinstance(parsed, dict) and "answer" in parsed:
                    answer = parsed["answer"]
            except Exception:
                pass

            step.action_output = answer
            step.observations = str(answer)
            trajectory.append(step)

            ctx.logger.log(
                Text(f"Aggregated answer: {str(answer)[:200]}", style=f"bold {YELLOW_HEX}"),
                level=LogLevel.INFO,
            )
            logger.info(
                f"ROMAAction: aggregation complete ({len(str(answer))} chars)"
            )

            return answer

        except Exception as e:
            logger.error(f"ROMAAction: aggregation failed: {e}")
            # Fallback: return concatenated results
            return all_results or f"Aggregation failed: {e}"

    # ── Helpers ──

    def _build_subtask_system_prompt(self, ctx: RuntimeContext) -> str:
        """Build system prompt for subtask execution."""
        system_template = self.prompts.get("system_prompt", "")
        if not system_template:
            return "You are an expert assistant that solves tasks using tools."

        tools = {}
        try:
            tool_selection = ctx.tool_policy.select_tools("", 0, MemoryView())
            tools = tool_selection.tools
        except Exception:
            pass

        skills_prompt = ""
        if hasattr(ctx.tool_policy, 'get_skills_prompt'):
            skills_prompt = ctx.tool_policy.get_skills_prompt()

        return _populate_template(
            system_template,
            variables={"tools": tools, "skills_prompt": skills_prompt},
        )

    def _parse_and_execute(
        self,
        response: Any,
        step: StepRecord,
        ctx: RuntimeContext,
        task: str,
    ) -> Tuple[List[ToolCall], str, Optional[Any]]:
        """Parse LLM response and execute tool calls (standard ReAct)."""
        try:
            content_dict = json_repair.loads(response.content)
        except Exception:
            content_dict = {}

        if isinstance(content_dict, list):
            if content_dict and isinstance(content_dict[0], dict) and "tools" in content_dict[0]:
                answer_data = content_dict[0]["tools"]
                think = content_dict[0].get("think", "")
            else:
                answer_data = content_dict
                think = ""
        elif isinstance(content_dict, dict):
            answer_data = content_dict.get("tools", None)
            think = content_dict.get("think", "")
        else:
            answer_data = None
            think = ""

        step.action_think = think
        step.action_reasoning = getattr(response, 'reasoning_content', '') or ""

        if think:
            ctx.logger.log(
                Panel(Text(f"Think: {think[:500]}...")),
                level=LogLevel.INFO,
            )

        if isinstance(answer_data, list):
            tool_calls_list = answer_data
        elif isinstance(answer_data, dict):
            tool_calls_list = [answer_data]
        else:
            tool_calls_list = []

        ctx.logger.log(
            Panel(Text(f"Tool calls: {len(tool_calls_list)}")),
            level=LogLevel.INFO,
        )

        observations = []
        final_answer = None

        for tc_data in tool_calls_list:
            if not isinstance(tc_data, dict):
                continue

            tool_name = tc_data.get("name", "")
            tool_arguments = tc_data.get("arguments", {})
            tool_call_id = tc_data.get("id", "")

            tc = ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)
            step.tool_calls.append(tc)

            ctx.logger.log(
                Panel(Text(f"Calling: '{tool_name}' with {tool_arguments}")),
                level=LogLevel.INFO,
            )

            if tool_name == "final_answer":
                if isinstance(tool_arguments, dict):
                    final_answer = tool_arguments.get("answer", tool_arguments)
                else:
                    final_answer = tool_arguments
                ctx.logger.log(
                    Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                    level=LogLevel.INFO,
                )
                step.action_output = final_answer
                observations.append(str(final_answer))
                break

            try:
                observation = ctx.execute_tool(tool_name, tool_arguments)
            except Exception as e:
                observation = f"Error executing '{tool_name}': {str(e)}"
                logger.error(observation)

            updated_info = str(observation).strip()
            observations.append(
                f"Results for tool call '{tool_name}' with arguments "
                f"'{tool_arguments}':\n{updated_info}"
            )
            safe_info = escape(updated_info[:500])
            ctx.logger.log(
                f"Observations: {safe_info}...",
                level=LogLevel.INFO,
            )

        step.observations = "\n\n".join(observations) if observations else "No observations"
        return step.tool_calls, think, final_answer

    def _force_final_answer(
        self,
        task: str,
        ctx: RuntimeContext,
        trajectory: Optional[List[StepRecord]] = None,
        step_number: int = 0,
    ) -> Any:
        """Generate final answer when max steps reached."""
        ctx.logger.log_rule("Forcing final answer (max steps reached)")
        step_start_time = time.time()

        memory_view = ctx.memory.build_context()
        messages = memory_view.messages.copy()

        final_prompts = self.prompts.get("final_answer", {})
        pre = final_prompts.get("pre_messages", "An agent reached max steps.")
        post = final_prompts.get("post_messages", "Provide a brief answer.")

        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": pre}],
        })

        post_text = (
            _populate_template(post, variables={"task": task})
            if "{{task}}" in post else post
        )
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": post_text}],
        })

        try:
            response = ctx.model(messages)
            content = response.content or ""
            answer = content

            try:
                parsed = json_repair.loads(content)
                if isinstance(parsed, dict) and "answer" in parsed:
                    answer = parsed["answer"]
            except Exception:
                pass

            input_tokens = count_tokens_messages(messages)
            output_tokens = count_tokens_text(content)
            step = StepRecord(
                step_number=step_number,
                model_input_messages=messages,
                model_output_messages=response,
                start_time=step_start_time,
                end_time=time.time(),
                action_output=answer,
                action_reasoning="Forced final answer (max steps reached)",
                observations=str(answer),
                input_token_count=input_tokens,
                output_token_count=output_tokens,
                total_token_count=input_tokens + output_tokens,
            )
            step.duration = step.end_time - step.start_time
            if trajectory is not None:
                trajectory.append(step)
            return answer

        except Exception as e:
            logger.error(f"Failed to generate final answer: {e}")
            step = StepRecord(
                step_number=step_number,
                model_input_messages=messages,
                start_time=step_start_time, end_time=time.time(),
                error=e,
                action_reasoning="Forced final answer — failed",
            )
            step.duration = step.end_time - step.start_time
            if trajectory is not None:
                trajectory.append(step)
            return f"Error generating final answer: {e}"
