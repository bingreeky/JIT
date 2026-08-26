"""
OAgentAction: parallel multi-path execution with LLM-based voting.

Inspired by JoyAgent's multi-expert architecture. Runs N independent "expert"
ReAct paths (PE-Workers + ReAct-Workers), each with fresh independent memory,
then uses an LLM "critic" call to evaluate all expert answers and produce
a final synthesized answer.

Design decisions:
- Each expert gets independent FullHistoryMemory (no cross-contamination)
- Experts run sequentially (share API rate limits)
- PE-Worker uses a plan-and-execute prompt; ReAct-Worker uses standard ReAct
- Critic LLM call evaluates all answers and picks/synthesizes the best
"""

import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import json_repair
from jinja2 import StrictUndefined, Template
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from scripts.kernel.token_counter import count_tokens_messages, count_tokens_text
from scripts.kernel.protocols import BaseAction
from scripts.kernel.types import (
    RunResult, RuntimeContext, StepRecord, TaskInput, ToolCall,
)
from .memory import FullHistoryMemory
from .planning import PlanningStrategy as NoPlanning
from scripts.kernel.monitoring import LogLevel, YELLOW_HEX


logger = logging.getLogger(__name__)


def _populate_template(template: str, variables: dict) -> str:
    return Template(template, undefined=StrictUndefined).render(**variables)


# ── Expert system prompts ──



class ActionStrategy(BaseAction):
    """Multi-path ensemble execution with LLM critic voting.

    Runs N expert paths (PE-Workers + ReAct-Workers), each with independent
    memory and different system prompts. After all paths complete, a critic
    LLM call evaluates all answers and produces the final result.
    """

    def __init__(self, prompts=None, num_react_workers: int = 2, num_pe_workers: int = 1):
        self.prompts = prompts or {}
        self._num_react = num_react_workers
        self._num_pe = num_pe_workers
    def get_action_prompt_template(self) -> str:
        return self.prompts["step"]["pre_messages"]

    def run(self, task: str, ctx: RuntimeContext) -> RunResult:
        expert_results: List[Tuple[str, RunResult]] = []
        total_experts = self._num_pe + self._num_react
        steps_per_expert = max(ctx.max_steps // total_experts, 5)
        workspace = self._resolve_workspace(ctx)
        baseline_dir = self._snapshot_workspace_baseline(workspace) if workspace else ""
        expert_counter = 0

        ctx.logger.log_rule(
            f"OAgent ensemble vote: {total_experts} experts "
            f"({self._num_pe} PE + {self._num_react} ReAct), "
            f"{steps_per_expert} steps each",
            level=LogLevel.INFO,
        )

        # 1. Run PE-Worker paths
        for i in range(self._num_pe):
            expert_counter += 1
            if workspace:
                self._reset_workspace_for_expert(workspace, baseline_dir)
            ctx.logger.log_rule(f"PE-Worker {i+1}", level=LogLevel.INFO)
            result = self._run_expert(task, ctx, "pe", steps_per_expert, i)
            expert_results.append((f"PE-Worker-{i+1}", result))
            if workspace:
                self._archive_expert_output(workspace, expert_counter)
            ctx.logger.log(
                f"PE-Worker {i+1} answer: {result.answer}",
                level=LogLevel.INFO,
            )

        # 2. Run ReAct-Worker paths
        for i in range(self._num_react):
            expert_counter += 1
            if workspace:
                self._reset_workspace_for_expert(workspace, baseline_dir)
            ctx.logger.log_rule(f"ReAct-Worker {i+1}", level=LogLevel.INFO)
            result = self._run_expert(task, ctx, "react", steps_per_expert, i)
            expert_results.append((f"ReAct-Worker-{i+1}", result))
            if workspace:
                self._archive_expert_output(workspace, expert_counter)
            ctx.logger.log(
                f"ReAct-Worker {i+1} answer: {result.answer}",
                level=LogLevel.INFO,
            )

        # 3. Critic vote
        ctx.logger.log_rule("Critic Voting", level=LogLevel.INFO)
        critic_result = self._critic_vote(task, expert_results, ctx)
        final_answer = critic_result["answer"]
        selected_expert = critic_result["selected_expert"]
        selected_expert_index = self._find_expert_index(selected_expert, expert_results)
        if workspace:
            self._restore_selected_output(
                workspace,
                selected_expert_index,
                total_experts,
                baseline_dir,
            )
        ctx.logger.log(
            f"Critic verdict: {final_answer}",
            level=LogLevel.INFO,
        )

        # 4. Combine trajectories from all experts
        # Flat trajectory: merged, tagged by expert name (backward compatible)
        all_trajectories: List[StepRecord] = []
        for name, result in expert_results:
            for step in result.trajectory:
                # Tag steps with expert name for debugging
                step.action_reasoning = f"[{name}] " + (step.action_reasoning or "")
                all_trajectories.append(step)

        # Hierarchical sub_runs: each expert's full RunResult preserved
        sub_runs = []
        for name, result in expert_results:
            result.metadata["expert_name"] = name
            sub_runs.append(result)

        return RunResult(
            answer=final_answer,
            trajectory=all_trajectories,
            terminated_reason="oagent",
            metadata={
                "num_experts": total_experts,
                "num_pe_workers": self._num_pe,
                "num_react_workers": self._num_react,
                "critic_answer": str(final_answer)[:2000],
                "selected_expert": selected_expert,
                "selected_expert_index": selected_expert_index,
                "expert_answers": {
                    name: str(result.answer)[:1000]
                    for name, result in expert_results
                },
            },
            sub_runs=sub_runs,
        )
    
    def _run_react_loop(self, task: str, ctx: RuntimeContext) -> RunResult:
        trajectory: List[StepRecord] = []
        final_answer = None
        step_number = 0

        # ── Step 0: Planning ──
        memory_view = ctx.memory.build_context()
        tool_selection = ctx.tool_policy.select_tools(task, step_number, memory_view)
        plan = ctx.planning.init_plan(
            task, memory_view, tool_selection.tool_schemas_json, ctx.model,
        )
        ctx.memory.update_plan(plan)
        ctx.logger.log_markdown(plan.plan, title="Initial Plan", level=LogLevel.INFO)
        step_number += 1

        # ── Main loop ──
        while final_answer is None and step_number <= ctx.max_steps:
            step_start_time = time.time()

            # Check if replanning needed
            if ctx.planning.should_replan(step_number, trajectory[-1] if trajectory else StepRecord()):
                memory_view = ctx.memory.build_context()
                summary = ctx.planning.update_plan(task, step_number, memory_view, ctx.model)
                ctx.memory.update_summary(summary)
                ctx.logger.log_markdown(summary.summary, title="Plan Update", level=LogLevel.INFO)
                step_number += 1

            ctx.logger.log_rule(f"Step {step_number}", level=LogLevel.INFO)

            # Build context for this step
            memory_view = ctx.memory.build_context(plan)
            tool_selection = ctx.tool_policy.select_tools(task, step_number, memory_view, plan)

            # Build step prompt
            step_prompt = _populate_template(
                self.get_action_prompt_template(),
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
                    step_number=step_number,
                    error=e,
                    start_time=step_start_time,
                    end_time=time.time(),
                )
                step.duration = step.end_time - step.start_time
                ctx.memory.update(step)
                trajectory.append(step)
                step_number += 1
                continue

            # Hook: process raw LLM response before tool parsing
            # (subclasses can override to intercept e.g. <compress> tags)
            response = self._process_response(response, ctx)

            # Record token usage
            input_tokens = count_tokens_messages(messages)
            output_tokens = count_tokens_text(
                getattr(response, 'content', '') or ''
            )

            # Parse response
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
            step.duration = step.end_time - step_start_time

            # Update memory
            ctx.memory.update(step)
            trajectory.append(step)
            step_number += 1

        # If max steps reached without answer, force final answer
        if final_answer is None:
            final_answer = self._force_final_answer(task, ctx, trajectory, step_number)

        return RunResult(
            answer=final_answer,
            trajectory=trajectory,
            terminated_reason="final_answer" if step_number <= ctx.max_steps else "max_steps",
        )

    def _process_response(self, response: Any, ctx: RuntimeContext) -> Any:
        return response

    def _parse_and_execute(
        self,
        response: Any,
        step: StepRecord,
        ctx: RuntimeContext,
        task: str,
    ) -> Tuple[List[ToolCall], str, Optional[Any]]:
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
        step.action_reasoning = getattr(response, "reasoning_content", "") or ""

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
                f"Results for tool call '{tool_name}' with arguments '{tool_arguments}':\n{updated_info}"
            )
            safe_updated_info = escape(updated_info)
            ctx.logger.log(
                f"Observations: {safe_updated_info}...",
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
        ctx.logger.log_rule("Forcing final answer (max steps reached)")
        step_start_time = time.time()

        memory_view = ctx.memory.build_context()
        messages = memory_view.messages.copy()

        final_prompts = self.prompts.get("final_answer", {})
        pre = final_prompts.get("pre_messages", "An agent tried to answer a user query but reached max steps.")
        post = final_prompts.get("post_messages", "Based on the above, provide a brief answer.")

        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": pre}],
        })

        post_text = _populate_template(post, variables={"task": task}) if "{{task}}" in post else post
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

            step = StepRecord(
                step_number=step_number,
                model_input_messages=messages,
                model_output_messages=response,
                start_time=step_start_time,
                end_time=time.time(),
                action_output=answer,
                action_reasoning="Forced final answer (max steps reached)",
                observations=str(answer),
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
                start_time=step_start_time,
                end_time=time.time(),
                error=e,
                action_reasoning="Forced final answer (max steps reached) - failed",
            )
            step.duration = step.end_time - step.start_time
            if trajectory is not None:
                trajectory.append(step)

            return f"Error generating final answer: {e}"

    def _run_expert(
        self,
        task: str,
        ctx: RuntimeContext,
        expert_type: str,
        max_steps: int,
        index: int,
    ) -> RunResult:
        """Run one expert path with independent memory."""
        # Create fresh memory for this expert
        expert_memory = FullHistoryMemory(prompts=self.prompts)

        # Build expert-specific system prompt
        if expert_type == "pe":
            expert_system = self.prompts["action"]["pe_worker_system_prompt"]
        else:
            expert_system = self.prompts["action"]["react_worker_system_prompt"]

        # Combine with the base system prompt from the original context
        base_system = self._extract_base_system(ctx)
        full_system = f"{base_system}\n\n{expert_system}"

        # Initialize expert memory
        expert_memory.initialize(
            full_system,
            TaskInput(task=task),
        )

        # Create expert context with independent memory and NoPlanning
        expert_planning = NoPlanning(prompts=self.prompts)
        expert_ctx = RuntimeContext(
            memory=expert_memory,
            planning=expert_planning,
            tool_policy=ctx.tool_policy,
            model=ctx.model,
            execute_tool=ctx.execute_tool,
            get_tool_schemas=ctx.get_tool_schemas,
            logger=ctx.logger,
            prompt_templates=ctx.prompt_templates,
            max_steps=max_steps,
        )

        return self._run_react_loop(task, expert_ctx)

    def _critic_vote(
        self,
        task: str,
        expert_results: List[Tuple[str, RunResult]],
        ctx: RuntimeContext,
    ) -> Dict[str, Any]:
        """LLM-based voting across all expert answers."""
        # Format expert answers with their evidence
        expert_sections = []
        for name, result in expert_results:
            answer = result.answer or "No answer produced"
            # Collect key observations from trajectory
            evidence_parts = []
            for step in result.trajectory:
                if step.observations and step.observations != "No observations":
                    # Take first 500 chars of each observation
                    obs_summary = step.observations[:500]
                    if step.tool_calls:
                        tool_names = ", ".join(tc.name for tc in step.tool_calls if tc.name != "final_answer")
                        if tool_names:
                            evidence_parts.append(f"  Tool({tool_names}): {obs_summary}")
                    else:
                        evidence_parts.append(f"  Observation: {obs_summary}")

            evidence_str = "\n".join(evidence_parts[:5])  # Max 5 evidence items
            expert_sections.append(
                f"### {name}\n"
                f"**Answer:** {answer}\n"
                f"**Evidence:**\n{evidence_str if evidence_str else '  (No evidence collected)'}"
            )

        expert_answers_text = "\n\n".join(expert_sections)

        # Build critic prompt
        critic_text = self.prompts["action"]["critic_prompt"].format(
            task=task,
            expert_answers=expert_answers_text,
        )

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": critic_text}],
        }]

        try:
            response = ctx.model(messages)
            verdict = (response.content or "").strip()
            parsed = json_repair.loads(verdict)
            if isinstance(parsed, dict):
                answer = parsed.get("answer", "")
                selected_expert = str(parsed.get("selected_expert", "")).strip()
                if answer:
                    return {
                        "answer": answer,
                        "selected_expert": self._match_selected_expert(
                            selected_expert, expert_results, answer
                        ),
                    }
        except Exception as e:
            logger.error(f"Critic vote failed: {e}")
            verdict = ""

        if expert_results:
            fallback_name, fallback_result = expert_results[0]
            fallback_answer = fallback_result.answer
            return {
                "answer": fallback_answer if fallback_answer is not None else verdict,
                "selected_expert": self._match_selected_expert(
                    verdict, expert_results, fallback_answer
                ) or fallback_name,
            }
        return {
            "answer": verdict or "Error in critic voting",
            "selected_expert": "",
        }

    def _extract_base_system(self, ctx: RuntimeContext) -> str:
        """Extract the base system prompt from the original context.

        The system prompt contains tool descriptions and general instructions.
        We reuse it for all experts.
        """
        # Build a temporary memory view to get the system prompt
        memory_view = ctx.memory.build_context()
        for msg in memory_view.messages:
            role = msg.get("role", "")
            if role == "system":
                content = msg.get("content", [])
                if isinstance(content, list) and content:
                    return content[0].get("text", "")
                elif isinstance(content, str):
                    return content
        return ""

    def _resolve_workspace(self, ctx: RuntimeContext) -> str:
        """Best-effort workspace discovery from workspace-aware tools."""
        catalog = getattr(ctx.tool_policy, "_catalog", {}) or {}
        for tool in catalog.values():
            workspace = getattr(tool, "workspace", "")
            if workspace:
                return workspace

        return ""

    def _snapshot_workspace_baseline(self, workspace: str) -> str:
        """Save the initial inputs and root-level files for expert resets."""
        if not workspace:
            return ""

        os.makedirs(workspace, exist_ok=True)
        baseline_dir = os.path.join(workspace, ".oagent_baseline")
        baseline_root = os.path.join(baseline_dir, "root")

        if os.path.isdir(baseline_dir):
            shutil.rmtree(baseline_dir)
        os.makedirs(baseline_root, exist_ok=True)

        for name in os.listdir(workspace):
            if name == ".oagent_baseline":
                continue

            source_path = os.path.join(workspace, name)
            target_path = os.path.join(baseline_root, name)

            if os.path.isfile(source_path):
                shutil.copy2(source_path, target_path)
            elif os.path.isdir(source_path) and name in {"input", "inputs", "output"}:
                shutil.copytree(source_path, target_path)

        return baseline_dir

    def _reset_workspace_for_expert(self, workspace: str, baseline_dir: str) -> None:
        """Restore a clean per-expert workspace while preserving archived outputs."""
        if not workspace:
            return

        for name in os.listdir(workspace):
            if name == ".oagent_baseline":
                continue
            if name.startswith("output") and name != "output":
                continue

            path = os.path.join(workspace, name)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path) and name in {"input", "inputs", "output"}:
                shutil.rmtree(path)

        baseline_root = os.path.join(baseline_dir, "root")
        if not os.path.isdir(baseline_root):
            return

        for name in os.listdir(baseline_root):
            source_path = os.path.join(baseline_root, name)
            target_path = os.path.join(workspace, name)
            if os.path.isfile(source_path):
                shutil.copy2(source_path, target_path)
            elif os.path.isdir(source_path):
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)

    def _archive_expert_output(self, workspace: str, expert_index: int) -> None:
        """Move one expert's output directory to output{i}."""
        output_dir = os.path.join(workspace, "output")
        archived_dir = os.path.join(workspace, f"output{expert_index}")

        if os.path.isdir(archived_dir):
            shutil.rmtree(archived_dir)

        if os.path.isdir(output_dir):
            shutil.move(output_dir, archived_dir)
        else:
            os.makedirs(archived_dir, exist_ok=True)

        self._cleanup_workspace_root(workspace, preserve={"output", f"output{expert_index}"})

    def _restore_selected_output(
        self,
        workspace: str,
        selected_expert_index: int,
        total_experts: int,
        baseline_dir: str,
    ) -> None:
        """Restore the selected expert output as output/ and remove other temp dirs."""
        if not workspace:
            return

        output_dir = os.path.join(workspace, "output")
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

        selected_dir = os.path.join(workspace, f"output{selected_expert_index}")
        if os.path.isdir(selected_dir):
            shutil.move(selected_dir, output_dir)
        else:
            os.makedirs(output_dir, exist_ok=True)

        for i in range(1, total_experts + 1):
            if i == selected_expert_index:
                continue
            candidate = os.path.join(workspace, f"output{i}")
            if os.path.isdir(candidate):
                shutil.rmtree(candidate)

        for name in ("input", "inputs"):
            candidate = os.path.join(workspace, name)
            if os.path.isdir(candidate):
                shutil.rmtree(candidate)

        self._cleanup_workspace_root(
            workspace,
            preserve={"output", ".oagent_baseline"},
        )

        if baseline_dir and os.path.isdir(baseline_dir):
            shutil.rmtree(baseline_dir)

    def _cleanup_workspace_root(self, workspace: str, preserve: Optional[set] = None) -> None:
        """Delete root-level files and transient input/output dirs in workspace."""
        preserve = preserve or set()
        for name in os.listdir(workspace):
            if name in preserve:
                continue

            path = os.path.join(workspace, name)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path) and name in {"input", "inputs", "output"}:
                shutil.rmtree(path)

    def _match_selected_expert(
        self,
        selected_expert: str,
        expert_results: List[Tuple[str, RunResult]],
        answer: Any = None,
    ) -> str:
        """Normalize critic expert choice, falling back to answer matching."""
        for name, result in expert_results:
            if selected_expert == name:
                return name

        answer_text = str(answer).strip() if answer is not None else ""
        for name, result in expert_results:
            result_answer = str(result.answer).strip() if result.answer is not None else ""
            if selected_expert and selected_expert in name:
                return name
            if answer_text and result_answer and answer_text == result_answer:
                return name

        return expert_results[0][0] if expert_results else ""

    def _find_expert_index(
        self,
        selected_expert: str,
        expert_results: List[Tuple[str, RunResult]],
    ) -> int:
        for index, (name, _) in enumerate(expert_results, start=1):
            if name == selected_expert:
                return index
        return 1
