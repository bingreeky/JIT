"""
DeepAgentAction: marker-based execution protocol with thought folding and tool search.

Adapted from DeepAgent. Key differences from standard ReAct:
  - Marker-based parsing: <tool_call>, <tool_search>, <fold_thought>
  - Single tool call per step (model stops at first marker)
  - Duplicate tool call / search prevention
  - Tool search: LLM queries for relevant tools from catalog
  - Long response analysis: LLM summarizes verbose tool outputs (>5000 chars)
  - Thought folding: delegates to memory.fold() when detected
  - Final answer: either via final_answer tool or marker-free text with \\boxed{}
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import json_repair
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from scripts.kernel.protocols import BaseAction
from scripts.kernel.token_counter import count_tokens_messages, count_tokens_text
from scripts.kernel.types import (
    RunResult, RuntimeContext, StepRecord, ToolCall,
)
from scripts.kernel.monitoring import LogLevel, YELLOW_HEX


logger = logging.getLogger(__name__)

# ── DeepAgent markers ──
BEGIN_TOOL_SEARCH = "<tool_search>"
END_TOOL_SEARCH = "</tool_search>"
BEGIN_TOOL_SEARCH_RESULT = "<tool_search_result>"
END_TOOL_SEARCH_RESULT = "</tool_search_result>"

BEGIN_TOOL_CALL = "<tool_call>"
END_TOOL_CALL = "</tool_call>"
BEGIN_TOOL_RESPONSE = "<tool_call_result>"
END_TOOL_RESPONSE = "</tool_call_result>"

FOLD_THOUGHT = "<fold_thought>"


def _populate_template(template: str, variables: dict) -> str:
    from jinja2 import Template, StrictUndefined
    return Template(template, undefined=StrictUndefined).render(**variables)


def _extract_between(text: str, start_marker: str, end_marker: str) -> Optional[str]:
    """Extract content between the FIRST occurrence of markers."""
    pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _truncate_at_first_marker(content: str) -> str:
    """Truncate model output at the end of the first complete marker block.

    Gemini and some other models generate full multi-turn "simulations"
    in a single response (tool_call + fake result + next tool_call...).
    We must only process the FIRST action and discard the rest.
    """
    # Find the first complete marker block end
    first_end = len(content)
    for end_marker in [END_TOOL_CALL, END_TOOL_SEARCH, FOLD_THOUGHT]:
        idx = content.find(end_marker)
        if idx != -1:
            end_pos = idx + len(end_marker)
            first_end = min(first_end, end_pos)

    if first_end < len(content):
        truncated = content[:first_end]
        logger.info(
            f"DeepAgentAction: truncated response at first marker "
            f"({first_end}/{len(content)} chars)"
        )
        return truncated
    return content


def _extract_boxed(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format."""
    pattern = r'\\boxed\{(.*?)\}'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


class ActionStrategy(BaseAction):
    """DeepAgent execution protocol with marker-based tool calls and thought folding.

    Main loop:
    1. Call LLM with current context
    2. Parse response for markers:
       a. <tool_call>...</tool_call> -> execute tool
       b. <tool_search>...</tool_search> -> search tool catalog
       c. <fold_thought> -> trigger memory folding
       d. No markers -> extract final answer
    3. Record step, update memory
    4. Check action/step limits
    """

    def __init__(self, prompts=None):
        self.prompts = prompts or {}
        self._executed_tool_calls: Set[str] = set()
        self._executed_searches: Set[str] = set()
        self._response_analysis_threshold: int = 5000

    def run(self, task: str, ctx: RuntimeContext) -> RunResult:
        trajectory: List[StepRecord] = []
        final_answer = None
        step_number = 0

        # Reset duplicate tracking per run
        self._executed_tool_calls = set()
        self._executed_searches = set()

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
            ctx.logger.log_rule(f"Step {step_number}", level=LogLevel.INFO)

            # Build context
            memory_view = ctx.memory.build_context()
            tool_selection = ctx.tool_policy.select_tools(task, step_number, memory_view)

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

            content = response.content or ""

            # Truncate at first marker to prevent multi-turn simulation
            content = _truncate_at_first_marker(content)

            input_tokens = count_tokens_messages(messages)
            output_tokens = count_tokens_text(content)

            # Create step record
            step = StepRecord(
                step_number=step_number,
                model_input_messages=messages,
                model_output_messages=response,
                start_time=step_start_time,
                input_token_count=input_tokens,
                output_token_count=output_tokens,
                total_token_count=input_tokens + output_tokens,
            )

            # Extract reasoning text (everything before first marker)
            think_text = content
            for marker in [BEGIN_TOOL_CALL, BEGIN_TOOL_SEARCH, FOLD_THOUGHT]:
                if marker in think_text:
                    think_text = think_text[:think_text.index(marker)]
            step.action_think = think_text.strip()
            step.action_reasoning = getattr(response, 'reasoning_content', '') or ""

            if step.action_think:
                ctx.logger.log(
                    Panel(Text(f"Think: {step.action_think[:500]}...")),
                    level=LogLevel.INFO,
                )

            # ── Parse markers ──
            try:
                if BEGIN_TOOL_CALL in content and END_TOOL_CALL in content:
                    final_answer = self._handle_tool_call(
                        content, step, ctx, task, tool_selection,
                    )

                elif BEGIN_TOOL_SEARCH in content and END_TOOL_SEARCH in content:
                    self._handle_tool_search(
                        content, step, ctx, task, tool_selection,
                    )

                elif FOLD_THOUGHT in content:
                    self._handle_fold(step, ctx, task)

                else:
                    # No markers -> extract answer
                    final_answer = self._extract_final_answer(content, task)
                    if final_answer:
                        step.action_output = final_answer
                        step.observations = str(final_answer)
                        ctx.logger.log(
                            Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                            level=LogLevel.INFO,
                        )

            except Exception as e:
                logger.error(f"Step execution error: {e}")
                step.error = e

            step.end_time = time.time()
            step.duration = step.end_time - step.start_time

            ctx.memory.update(step)
            trajectory.append(step)
            step_number += 1

        # Force final answer if needed
        if final_answer is None:
            final_answer = self._force_final_answer(task, ctx, trajectory, step_number)

        return RunResult(
            answer=final_answer,
            trajectory=trajectory,
            terminated_reason="final_answer" if step_number <= ctx.max_steps else "max_steps",
        )

    # ── Marker handlers ──

    def _handle_tool_call(
        self,
        content: str,
        step: StepRecord,
        ctx: RuntimeContext,
        task: str,
        tool_selection: Any,
    ) -> Optional[Any]:
        """Handle <tool_call>...</tool_call> marker. Returns final_answer if final_answer tool called."""
        tool_call_raw = _extract_between(content, BEGIN_TOOL_CALL, END_TOOL_CALL)
        if not tool_call_raw:
            step.observations = "Failed to parse tool call content."
            return None

        # Parse JSON
        try:
            tool_call_dict = json_repair.loads(tool_call_raw)
        except Exception:
            step.observations = f"Failed to parse tool call JSON: {tool_call_raw[:200]}"
            return None

        tool_name = tool_call_dict.get("name", "")
        tool_arguments = tool_call_dict.get("arguments", {})
        call_key = json.dumps(tool_call_dict, sort_keys=True)

        tc = ToolCall(name=tool_name, arguments=tool_arguments)
        step.tool_calls.append(tc)

        ctx.logger.log(
            Panel(Text(f"Tool calls: 1")),
            level=LogLevel.INFO,
        )
        ctx.logger.log(
            Panel(Text(f"Calling: '{tool_name}' with {tool_arguments}")),
            level=LogLevel.INFO,
        )

        # Handle final_answer
        if tool_name == "final_answer":
            if isinstance(tool_arguments, dict):
                answer = tool_arguments.get("answer", tool_arguments)
            else:
                answer = tool_arguments
            ctx.logger.log(
                Text(f"Final answer: {answer}", style=f"bold {YELLOW_HEX}"),
                level=LogLevel.INFO,
            )
            step.action_output = answer
            step.observations = str(answer)
            return answer

        # Duplicate check
        if call_key in self._executed_tool_calls:
            observation = "You have already called this tool with the same arguments. Try a different approach."
            ctx.logger.log(f"Duplicate tool call detected: {tool_name}", level=LogLevel.INFO)
        else:
            self._executed_tool_calls.add(call_key)
            try:
                observation = ctx.execute_tool(tool_name, tool_arguments)
            except Exception as e:
                observation = f"Error executing '{tool_name}': {str(e)}"
                logger.error(observation)

            # Analyze long responses
            if len(str(observation)) > self._response_analysis_threshold:
                observation = self._analyze_tool_response(
                    ctx, str(observation), tool_name, tool_arguments, task,
                )

        # Store raw observation — do NOT wrap in marker tags.
        # The markers belong in the prompt protocol, not in stored history.
        # Wrapping them causes LLMs (especially Gemini) to hallucinate
        # markers in their responses, leading to MALFORMED_FUNCTION_CALL errors.
        step.observations = str(observation)
        safe_obs = escape(str(observation)[:500])
        ctx.logger.log(f"Observations: {safe_obs}...", level=LogLevel.INFO)
        return None

    def _handle_tool_search(
        self,
        content: str,
        step: StepRecord,
        ctx: RuntimeContext,
        task: str,
        tool_selection: Any,
    ) -> None:
        """Handle <tool_search>...</tool_search> marker.

        Searches the available tool catalog by keyword matching and returns
        matching tool descriptions.
        """
        search_query = _extract_between(content, BEGIN_TOOL_SEARCH, END_TOOL_SEARCH)
        if not search_query or len(search_query) <= 3:
            step.observations = "Tool search query too short or empty."
            return

        ctx.logger.log(
            Panel(Text(f"Tool search: '{search_query}'")),
            level=LogLevel.INFO,
        )

        # Duplicate check
        if search_query in self._executed_searches:
            result_text = "You have already searched for this query. Try a different search or use the tools already found."
            ctx.logger.log(f"Duplicate tool search: {search_query}", level=LogLevel.INFO)
        else:
            self._executed_searches.add(search_query)

            # Search tool catalog by keyword matching
            matched_tools = self._search_tool_catalog(
                search_query, tool_selection.tools,
            )
            if matched_tools:
                result_text = json.dumps(matched_tools, indent=2, ensure_ascii=False)
            else:
                result_text = (
                    "No matching tools found. Available tools: "
                    + ", ".join(tool_selection.tools.keys())
                )

        # Store raw result — no marker wrapping (same rationale as _handle_tool_call)
        step.observations = f"Tool search results:\n{result_text}"
        safe_result = escape(result_text[:500])
        ctx.logger.log(f"Tool search results: {safe_result}...", level=LogLevel.INFO)

    def _handle_fold(
        self,
        step: StepRecord,
        ctx: RuntimeContext,
        task: str,
    ) -> None:
        """Handle <fold_thought> marker."""
        ctx.logger.log_rule("Thought Folding", level=LogLevel.INFO)

        if not hasattr(ctx.memory, 'can_fold') or not ctx.memory.can_fold():
            step.observations = (
                "You have reached the maximum number of allowed thought folds. "
                "Further thought folding is not permitted. Please continue reasoning."
            )
            ctx.logger.log(
                "Fold rejected: max folds reached", level=LogLevel.INFO,
            )
            return

        success = ctx.memory.fold(ctx.model, task)
        if success:
            step.observations = (
                "Thought folding complete. Your previous reasoning has been "
                "compressed into structured memories (episodic, working, tool). "
                "You can now continue with fresh reasoning guided by these memories."
            )
            ctx.logger.log(
                "Fold complete: context reset with compressed memories",
                level=LogLevel.INFO,
            )
        else:
            step.observations = (
                "Thought folding failed. Please continue reasoning with "
                "the current context."
            )
            logger.error("Fold failed during execution")

    # ── Helper methods ──

    def _search_tool_catalog(
        self, query: str, tools: Dict[str, Any],
    ) -> List[Dict]:
        """Search tool catalog by keyword matching against name and description.

        Returns tool schemas for matched tools.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        matched = []

        for tool in tools.values():
            name_lower = tool.name.lower()
            desc_lower = tool.description.lower()
            text = name_lower + " " + desc_lower

            # Match if any query word appears in name/description
            if any(word in text for word in query_words) or query_lower in text:
                tool_schema = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "properties": {k: v for k, v in tool.inputs.items()},
                        "required": [
                            k for k, v in tool.inputs.items()
                            if not v.get("nullable", False)
                        ],
                    },
                }
                matched.append(tool_schema)

        # If no keyword match, return all tools (better than nothing)
        if not matched:
            for tool in tools.values():
                tool_schema = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "properties": {k: v for k, v in tool.inputs.items()},
                        "required": [
                            k for k, v in tool.inputs.items()
                            if not v.get("nullable", False)
                        ],
                    },
                }
                matched.append(tool_schema)

        return matched

    def _analyze_tool_response(
        self,
        ctx: RuntimeContext,
        observation: str,
        tool_name: str,
        tool_arguments: Any,
        task: str,
    ) -> str:
        """Use LLM to analyze and compress a long tool response.

        Mirrors DeepAgent's run_tool_response_analysis().
        """
        logger.info(
            f"DeepAgentAction: analyzing long response "
            f"({len(observation)} chars) from {tool_name}"
        )
        prompt = (
            f"You are an assistant analyzing a tool's response. "
            f"The tool '{tool_name}' was called with arguments: {tool_arguments}\n"
            f"The original task is: {task}\n\n"
            f"Tool response (may be very long):\n{observation[:8000]}\n\n"
            f"Please extract and summarize ONLY the information relevant to the task. "
            f"Be concise but preserve all important details, data, and facts."
        )
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]
        try:
            response = ctx.model(messages)
            result = response.content or observation[:3000]
            logger.info(
                f"DeepAgentAction: compressed response to {len(result)} chars"
            )
            return result
        except Exception as e:
            logger.error(f"DeepAgentAction: response analysis failed: {e}")
            return observation[:3000] + "\n... [truncated]"

    def _extract_final_answer(self, content: str, task: str) -> Optional[str]:
        """Extract final answer from marker-free text.

        Checks for \\boxed{} format first, then falls back to full text.
        """
        # Check for \boxed{} format
        boxed = _extract_boxed(content)
        if boxed:
            return boxed

        # If content is substantial, treat it as the answer
        clean = content.strip()
        if len(clean) > 10:
            return clean

        return None

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
        pre = final_prompts.get(
            "pre_messages",
            "An agent tried to answer a user query but reached max steps.",
        )
        post = final_prompts.get(
            "post_messages",
            "Based on the above, provide a brief answer.",
        )

        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": pre}],
        })

        post_text = (
            _populate_template(post, variables={"task": task})
            if "{{task}}" in post
            else post
        )
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": post_text}],
        })

        try:
            response = ctx.model(messages)
            content = response.content or ""
            answer = content

            # Try JSON parse
            try:
                parsed = json_repair.loads(content)
                if isinstance(parsed, dict) and "answer" in parsed:
                    answer = parsed["answer"]
            except Exception:
                pass

            # Try \boxed{}
            boxed = _extract_boxed(content)
            if boxed:
                answer = boxed

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
                start_time=step_start_time,
                end_time=time.time(),
                error=e,
                action_reasoning="Forced final answer — failed",
            )
            step.duration = step.end_time - step.start_time
            if trajectory is not None:
                trajectory.append(step)
            return f"Error generating final answer: {e}"
