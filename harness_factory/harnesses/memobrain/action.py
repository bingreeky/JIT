"""
MemoBrain ActionStrategy: marker-based ReAct loop with passive
graph-memory hooks.

Adapted from MemoBrain (https://github.com/TommyChien/MemoBrain)
examples/react_with_memory.py. The original driver is a LangGraph
StateGraph with four nodes (planning / tool_call / check_limits /
finalize); here those four nodes collapse into a single while-loop in
`run()`. The timing of memorize / recall is preserved EXACTLY:

  - Per-turn ordering (unchanged from original):
      1. recall_if_needed        (passive; fires iff tokens > budget)
      2. ctx.model(messages)     (the one agent LLM decision per turn)
      3. parse markers:
           <tool_call>   -> ctx.execute_tool(...) -> memorize_pair
           <answer>      -> final answer, terminate
           otherwise     -> check tokens; if > max_tokens, force-answer
      4. ctx.memory.update(step) (protocol; trajectory logging only)

  - The agent's ACTIVE decisions are the marker choice only; everything
    memory-related is passively driven by loop position and token
    budget, exactly as in the original.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import json_repair
from rich.panel import Panel
from rich.text import Text

from scripts.kernel.protocols import BaseAction
from scripts.kernel.token_counter import count_tokens_messages, count_tokens_text
from scripts.kernel.monitoring import LogLevel, YELLOW_HEX
from scripts.kernel.types import (
    Message, RunResult, RuntimeContext, StepRecord, ToolCall,
)
from scripts.models.base import MessageRole


logger = logging.getLogger(__name__)


# ── Markers (verbatim from MemoBrain SYSTEM_PROMPT / driver) ──
BEGIN_TOOL_CALL = "<tool_call>"
END_TOOL_CALL = "</tool_call>"
BEGIN_ANSWER = "<answer>"
END_ANSWER = "</answer>"
BEGIN_TOOL_RESPONSE = "<tool_response>"

# Stop sequences used by the original driver (utils.call_server_async).
STOP_SEQUENCES = ["\n<tool_response>", "<tool_response>"]


class ActionStrategy(BaseAction):
    """MemoBrain marker-driven ReAct with passive graph-memory hooks."""

    def __init__(
        self,
        prompts: Optional[Dict[str, Any]] = None,
        max_memory_size: int = 32 * 1024,
        max_tokens: int = 105 * 1024,
        max_llm_call_per_run: int = 200,
    ) -> None:
        self.prompts = prompts or {}
        self._max_memory_size = max_memory_size
        self._max_tokens = max_tokens
        self._max_llm_call_per_run = max_llm_call_per_run

    def run(self, task: str, ctx: RuntimeContext) -> RunResult:
        # Initialize no-op planning (MemoBrain has no explicit plan state).
        memory_view = ctx.memory.build_context()
        tool_selection = ctx.tool_policy.select_tools(task, 0, memory_view)
        plan = ctx.planning.init_plan(
            task, memory_view, tool_selection.tool_schemas_json, ctx.model,
        )
        ctx.memory.update_plan(plan)

        trajectory: List[StepRecord] = []
        final_answer: Optional[Any] = None
        terminated_reason = "final_answer"
        step_number = 0
        llm_calls_remaining = self._max_llm_call_per_run

        max_steps = min(ctx.max_steps, self._max_llm_call_per_run)

        while final_answer is None and step_number < max_steps and llm_calls_remaining > 0:
            step_number += 1
            step_start = time.time()
            ctx.logger.log_rule(f"Step {step_number}", level=LogLevel.INFO)

            # ── 1) Passive recall (token-budget triggered) ──
            try:
                ctx.memory.recall_if_needed(self._max_memory_size)
            except Exception as exc:
                logger.warning(f"MemoBrainAction: recall_if_needed failed: {exc}")

            # ── 2) Planning LLM call ──
            memory_view = ctx.memory.build_context()
            messages = memory_view.messages

            try:
                response = ctx.model(messages, stop_sequences=STOP_SEQUENCES)
                llm_calls_remaining -= 1
            except Exception as exc:
                logger.error(f"MemoBrainAction: model call failed at step {step_number}: {exc}")
                step = StepRecord(
                    step_number=step_number,
                    model_input_messages=messages,
                    start_time=step_start,
                    end_time=time.time(),
                    error=exc,
                )
                step.duration = step.end_time - step.start_time
                ctx.memory.update(step)
                trajectory.append(step)
                continue

            content = (response.content or "").strip()

            # Original driver strips anything after a <tool_response> tail
            # that the LLM hallucinated despite the stop sequences.
            if BEGIN_TOOL_RESPONSE in content:
                content = content[: content.find(BEGIN_TOOL_RESPONSE)].rstrip()

            input_tokens = count_tokens_messages(messages)
            output_tokens = count_tokens_text(content)

            step = StepRecord(
                step_number=step_number,
                model_input_messages=messages,
                model_output_messages=response,
                start_time=step_start,
                action_reasoning=getattr(response, "reasoning_content", "") or "",
                input_token_count=input_tokens,
                output_token_count=output_tokens,
                total_token_count=input_tokens + output_tokens,
            )

            # Extract think text (everything before the first marker, for logging).
            think_text = content
            for marker in (BEGIN_TOOL_CALL, BEGIN_ANSWER):
                if marker in think_text:
                    think_text = think_text[: think_text.index(marker)]
            step.action_think = think_text.strip()
            if step.action_think:
                ctx.logger.log(
                    Panel(Text(f"Think: {step.action_think[:500]}")),
                    level=LogLevel.INFO,
                )

            # Append raw assistant message to memory (mirrors original driver:
            # assistant content gets added to state["messages"] in planning_node).
            ctx.memory.append_assistant(content)

            # ── 3) Parse markers ──
            has_tool_call = BEGIN_TOOL_CALL in content and END_TOOL_CALL in content
            has_answer = BEGIN_ANSWER in content and END_ANSWER in content

            if has_answer and not has_tool_call:
                # Agent chose to answer.
                final_answer = self._extract_answer(content)
                step.action_output = final_answer
                step.observations = str(final_answer)
                ctx.logger.log(
                    Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                    level=LogLevel.INFO,
                )

            elif has_tool_call:
                tool_name, tool_args = self._parse_tool_call(content)
                if tool_name is None:
                    observation = (
                        "Error: Tool call is not a valid JSON. Tool call must "
                        'contain a valid "name" and "arguments" field.'
                    )
                elif tool_name == "final_answer":
                    # JIT convention: `final_answer` tool terminates the run.
                    # Treat equivalent to an <answer>...</answer> marker so
                    # we don't burn an extra turn.
                    tc = ToolCall(name=tool_name, arguments=tool_args)
                    step.tool_calls.append(tc)
                    if isinstance(tool_args, dict):
                        final_answer = tool_args.get("answer", tool_args)
                    else:
                        final_answer = tool_args
                    step.action_output = final_answer
                    step.observations = str(final_answer)
                    ctx.logger.log(
                        Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                        level=LogLevel.INFO,
                    )
                    step.end_time = time.time()
                    step.duration = step.end_time - step.start_time
                    ctx.memory.update(step)
                    trajectory.append(step)
                    break
                else:
                    tc = ToolCall(name=tool_name, arguments=tool_args)
                    step.tool_calls.append(tc)
                    ctx.logger.log(
                        Panel(Text(f"Calling: '{tool_name}' with {tool_args}")),
                        level=LogLevel.INFO,
                    )
                    try:
                        observation = ctx.execute_tool(tool_name, tool_args)
                    except Exception as exc:
                        observation = f"Error executing '{tool_name}': {exc}"
                        logger.error(observation)

                if tool_name != "final_answer":
                    step.observations = str(observation)
                    safe_obs = str(observation)[:500].replace("\n", " ")
                    ctx.logger.log(f"Observations: {safe_obs}...", level=LogLevel.INFO)

                    # ── Passive memorize (verbatim original driver timing) ──
                    tool_response_text = f"<tool_response>\n{observation}\n</tool_response>"
                    try:
                        ctx.memory.memorize_pair(content, tool_response_text)
                    except Exception as exc:
                        logger.warning(f"MemoBrainAction: memorize_pair failed: {exc}")

            else:
                # "Continue" path: LLM produced no marker. Original driver's
                # check_limits_node logic — if hard token ceiling breached,
                # force-answer; otherwise fall through and try again.
                if ctx.memory.current_tokens() > self._max_tokens:
                    ctx.logger.log_rule(
                        f"Force answer (tokens > {self._max_tokens})",
                        level=LogLevel.INFO,
                    )
                    final_answer = self._force_answer(
                        ctx, task, step, llm_calls_remaining, reason="max_tokens",
                    )
                    llm_calls_remaining -= 1
                    terminated_reason = "max_tokens"
                else:
                    step.action_reasoning = (
                        (step.action_reasoning or "")
                        + " [no marker; continuing to next turn]"
                    ).strip()

            step.end_time = time.time()
            step.duration = step.end_time - step.start_time
            ctx.memory.update(step)
            trajectory.append(step)

        # Exit without an answer → force one from accumulated context.
        if final_answer is None:
            ctx.logger.log_rule(
                "Force answer (budget exhausted)", level=LogLevel.INFO,
            )
            fallback_step = StepRecord(
                step_number=step_number + 1,
                start_time=time.time(),
            )
            final_answer = self._force_answer(
                ctx, task, fallback_step, llm_calls_remaining,
                reason="budget_exhausted",
            )
            fallback_step.end_time = time.time()
            fallback_step.duration = fallback_step.end_time - fallback_step.start_time
            ctx.memory.update(fallback_step)
            trajectory.append(fallback_step)
            terminated_reason = (
                "max_steps" if step_number >= max_steps else "max_llm_calls"
            )

        return RunResult(
            answer=final_answer,
            trajectory=trajectory,
            terminated_reason=terminated_reason,
        )

    # ─────────────────────────────────────────────────────────────────
    # Parsing / extraction
    # ─────────────────────────────────────────────────────────────────

    def _parse_tool_call(
        self, content: str,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Parse <tool_call>{...}</tool_call> from content.

        Returns (tool_name, tool_args) or (None, None).

        NOTE: The original MemoBrain examples/tools.py also has a
        special double-segment branch for `PythonInterpreter` (empty
        arguments + a `<code>...</code>` block). That branch was tied
        to the original's hardcoded SYSTEM_PROMPT which taught the LLM
        that exact double-segment protocol. In JIT the tools block is
        Jinja-rendered from `ctx.tool_policy` — the LLM only sees the
        tool names that are actually registered and never learns the
        `<code>` protocol. So the branch is dead code here; worse, its
        heuristic (`"python" in tool_call_str.lower()`) can false-fire
        when a legitimate tool's arguments merely contain the substring
        "python" (e.g. `web_search("python tutorial")`). Removed.
        """
        if BEGIN_TOOL_CALL not in content or END_TOOL_CALL not in content:
            return None, None
        tool_call_str = content.split(BEGIN_TOOL_CALL)[1].split(END_TOOL_CALL)[0]

        try:
            tool_call = json_repair.loads(tool_call_str)
        except Exception:
            return None, None

        if not isinstance(tool_call, dict):
            return None, None
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})
        if not tool_name:
            return None, None
        return tool_name, tool_args

    def _extract_answer(self, content: str) -> str:
        """Extract text between <answer>...</answer>."""
        try:
            return content.split(BEGIN_ANSWER)[1].split(END_ANSWER)[0].strip()
        except Exception:
            return content.strip()

    # ─────────────────────────────────────────────────────────────────
    # Force-answer paths
    # ─────────────────────────────────────────────────────────────────

    def _force_answer(
        self,
        ctx: RuntimeContext,
        task: str,
        step: StepRecord,
        llm_calls_remaining: int,
        reason: str,
    ) -> str:
        """Force a final answer out of the current context.

        For `reason == "max_tokens"` this mirrors check_limits_node: inject
        a "you reached max context, answer now" prompt and make ONE more
        LLM call expecting <answer>...</answer>.

        For `reason == "budget_exhausted"` we use the harness's
        final_answer pre/post prompts (same style as other harnesses).
        """
        memory_view = ctx.memory.build_context()
        messages = list(memory_view.messages)

        if reason == "max_tokens":
            limit_message = self.prompts.get("force_answer", {}).get(
                "limit_message",
                "You have now reached the maximum context length. "
                "Provide the best answer in <answer>...</answer>.",
            )
            # Original check_limits_node (examples/react_with_memory.py:190-194)
            # operates on `all_messages.copy()` — the limit message is
            # only injected into the one-shot LLM call and is NEVER
            # persisted back into memory. Mirror that here: mutate only
            # the local `messages` list; `ctx.memory.messages` stays
            # clean.
            limit_content = [{"type": "text", "text": limit_message}]
            if messages and messages[-1]["role"] == MessageRole.ASSISTANT.value:
                messages[-1] = Message(
                    role=messages[-1]["role"],
                    content=limit_content,
                )
            else:
                messages.append(Message(
                    role=MessageRole.USER,
                    content=limit_content,
                ))
        else:
            pre = self.prompts.get("final_answer", {}).get(
                "pre_messages",
                "An agent exhausted its budget; based on its memory, provide an answer.",
            )
            post_raw = self.prompts.get("final_answer", {}).get(
                "post_messages",
                "Provide the answer to the task enclosed in <answer></answer>:\n{{task}}",
            )
            post = post_raw.replace("{{task}}", task)
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": pre}]}
            )
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": post}]}
            )

        try:
            response = ctx.model(messages, stop_sequences=STOP_SEQUENCES)
            content = (response.content or "").strip()
        except Exception as exc:
            logger.error(f"MemoBrainAction: force_answer model call failed: {exc}")
            step.error = exc
            return f"Error generating final answer: {exc}"

        input_tokens = count_tokens_messages(messages)
        output_tokens = count_tokens_text(content)
        step.model_input_messages = messages
        step.model_output_messages = response
        step.input_token_count = input_tokens
        step.output_token_count = output_tokens
        step.total_token_count = input_tokens + output_tokens
        step.action_reasoning = f"Forced final answer ({reason})"

        if BEGIN_ANSWER in content and END_ANSWER in content:
            answer = self._extract_answer(content)
        else:
            answer = content

        step.action_output = answer
        step.observations = str(answer)
        return answer
