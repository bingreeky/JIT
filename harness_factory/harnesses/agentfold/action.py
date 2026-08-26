"""
ReActFoldAction: ReAct execution protocol with AgentFold compression support.

Implements the full ReAct loop directly while adding compression support for
AgentFold-style memory modules. Falls back to standard ReAct behavior when
used with non-fold memory modules.
"""

import logging
import time
from typing import Any, List, Optional, Tuple

import json_repair
from jinja2 import StrictUndefined, Template
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from scripts.kernel.token_counter import count_tokens_messages, count_tokens_text
from scripts.kernel.protocols import BaseAction
from scripts.kernel.types import RunResult, RuntimeContext, StepRecord, ToolCall
from scripts.kernel.monitoring import LogLevel, YELLOW_HEX


logger = logging.getLogger(__name__)



def _populate_template(template: str, variables: dict) -> str:
    return Template(template, undefined=StrictUndefined).render(**variables)




def _extract_compress_info(text: str) -> Optional[dict]:
    """Extract compression info from the top-level JSON object."""
    try:
        content = json_repair.loads(text)
    except Exception:
        return None

    if isinstance(content, dict):
        compress = content.get("compress")
        if isinstance(compress, dict):
            return compress

    if isinstance(content, list) and content and isinstance(content[0], dict):
        compress = content[0].get("compress")
        if isinstance(compress, dict):
            return compress

    return None


class ActionStrategy(BaseAction):
    """ReAct with AgentFold compression support.

    Implements the full ReAct loop directly and extends it via
    _process_response() to:
    1. Detect compression instructions in LLM responses
    2. Apply compression to AgentFoldMemory if available
    3. Preserve the JSON response for normal tool call parsing
    """

    def __init__(self, prompts=None):
        self.prompts = prompts or {}
    
    def get_action_prompt_template(self) -> str:
        return self.prompts["step"]["pre_messages"]

    def run(self, task: str, ctx: RuntimeContext) -> RunResult:
        trajectory: List[StepRecord] = []
        final_answer = None
        step_number = 0

        memory_view = ctx.memory.build_context()
        tool_selection = ctx.tool_policy.select_tools(task, step_number, memory_view)
        plan = ctx.planning.init_plan(
            task, memory_view, tool_selection.tool_schemas_json, ctx.model,
        )
        ctx.memory.update_plan(plan)
        ctx.logger.log_markdown(plan.plan, title="Initial Plan", level=LogLevel.INFO)
        step_number += 1

        while final_answer is None and step_number <= ctx.max_steps:
            step_start_time = time.time()

            if ctx.planning.should_replan(step_number, trajectory[-1] if trajectory else StepRecord()):
                memory_view = ctx.memory.build_context()
                summary = ctx.planning.update_plan(task, step_number, memory_view, ctx.model)
                ctx.memory.update_summary(summary)
                ctx.logger.log_markdown(summary.summary, title="Plan Update", level=LogLevel.INFO)
                step_number += 1

            ctx.logger.log_rule(f"Step {step_number}", level=LogLevel.INFO)

            memory_view = ctx.memory.build_context(plan)
            tool_selection = ctx.tool_policy.select_tools(task, step_number, memory_view, plan)

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

            response = self._process_response(response, ctx)

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
                _, _, final_answer = self._parse_and_execute(
                    response, step, ctx, task,
                )
            except Exception as e:
                logger.error(f"Step execution error: {e}")
                step.error = e

            step.end_time = time.time()
            step.duration = step.end_time - step_start_time

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

    def _process_response(self, response: Any, ctx: RuntimeContext) -> Any:
        """Intercept LLM response to handle top-level JSON compression fields."""
        raw_content = response.content or ""

        compress_info = _extract_compress_info(raw_content)
        if compress_info and hasattr(ctx.memory, 'apply_compression'):
            compress_range = compress_info.get("compress_range", [])
            compress_text = compress_info.get("compress_text", "")
            if compress_range and compress_text:
                ctx.memory.apply_compression(compress_range, compress_text)
                ctx.logger.log(
                    f"Applied compression to steps {compress_range}",
                    level=LogLevel.INFO,
                )

        return response

    def _parse_and_execute(
        self,
        response: Any,
        step: StepRecord,
        ctx: RuntimeContext,
        task: str,
    ) -> Tuple[List[ToolCall], str, Optional[Any]]:
        """Parse LLM response and execute tool calls."""
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
        """Generate final answer when max steps are reached."""
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
                action_reasoning="Forced final answer (max steps reached) — failed",
            )
            step.duration = step.end_time - step.start_time
            if trajectory is not None:
                trajectory.append(step)

            return f"Error generating final answer: {e}"
