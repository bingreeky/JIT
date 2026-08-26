"""
ReActAction: single/multi-tool-per-step execution protocol.

This is the primary action module, directly mirroring Flash-Searcher's
ToolCallingAgent._run() and step() logic. It OWNS the main agent loop.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import json_repair
from rich.markup import escape

from scripts.kernel.token_counter import count_tokens_messages, count_tokens_text
from rich.panel import Panel
from rich.text import Text

from scripts.kernel.protocols import BaseAction
from scripts.kernel.types import (
    RunResult, RuntimeContext, StepRecord, ToolCall,
)
from scripts.kernel.monitoring import LogLevel, YELLOW_HEX


logger = logging.getLogger(__name__)






def _populate_template(template: str, variables: dict) -> str:
    from jinja2 import Template, StrictUndefined
    return Template(template, undefined=StrictUndefined).render(**variables)


class ActionStrategy(BaseAction):
    """ReAct execution protocol with multi-tool support per step.

    Mirrors Flash-Searcher's ToolCallingAgent:
    - Step 0: Planning
    - Steps 1-N: Action/Observation loop
    - Every N steps: Summary/Adaptation
    - Final: force answer if max_steps reached
    """

    def __init__(self, prompts=None):
        self.prompts = prompts or {}
    def get_action_prompt_template(self) -> str:
        """Return the per-step action prompt template."""
        return self.prompts["step"]["pre_messages"]

    def run(self, task: str, ctx: RuntimeContext) -> RunResult:
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
        """Hook for subclasses to intercept raw LLM response before tool parsing.

        Override this to detect inline tags (e.g., <compress>), apply
        transformations, or trigger memory operations. The returned response
        object will be passed to _parse_and_execute().

        Default: no-op (returns response unchanged).
        """
        return response

    def _parse_and_execute(
        self,
        response: Any,
        step: StepRecord,
        ctx: RuntimeContext,
        task: str,
    ) -> Tuple[List[ToolCall], str, Optional[Any]]:
        """Parse LLM response and execute tool calls.

        Returns:
            (tool_calls, think_text, final_answer_or_None)
        """
        # Parse JSON response
        try:
            content_dict = json_repair.loads(response.content)
        except Exception:
            content_dict = {}

        # Extract think and tools
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

        # Normalize tool calls list
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

        # Execute each tool call
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
                #f"Observations: {updated_info[:300]}...",
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
        """Generate final answer when max steps are reached.

        Creates a StepRecord for this forced LLM call and appends it
        to *trajectory* so the interaction is visible in traces.
        """
        ctx.logger.log_rule("Forcing final answer (max steps reached)")
        step_start_time = time.time()

        memory_view = ctx.memory.build_context()
        messages = memory_view.messages.copy()

        # Add final answer prompts
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

            # Try to parse as JSON with think/answer
            try:
                parsed = json_repair.loads(content)
                if isinstance(parsed, dict) and "answer" in parsed:
                    answer = parsed["answer"]
            except Exception:
                pass

            # Record this forced-answer step
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

            # Record the error step too
            step = StepRecord(
                step_number=step_number,
                model_input_messages=messages,
                start_time=step_start_time,
                end_time=time.time(),
                error=e,
                action_reasoning="Forced final answer (max steps reached) — failed",
            )
            step.duration = step.end_time - step.start_time
            if trajectory is not None:
                trajectory.append(step)

            return f"Error generating final answer: {e}"
