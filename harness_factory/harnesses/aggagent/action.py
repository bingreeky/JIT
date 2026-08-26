"""
AggAgentAction: K parallel rollouts + agentic aggregation.

Adapted from AggAgent (https://github.com/princeton-pli/AggAgent).

Flow:
  1. Run K independent ReAct rollout workers on the same task. Each
     worker has its own FullHistoryMemory + NoPlanning; they share the
     tool catalog from ctx.tool_policy.
  2. Convert each rollout's StepRecord trajectory into an OpenAI-style
     message list (the format the aggregator expects).
  3. Run an aggregator loop: an LLM-driven agent with 4 internal
     meta-tools over the K trajectories:
       - get_solution(trajectory_id?)
       - search_trajectory(trajectory_id, query, role?, k?)
       - get_segment(trajectory_id, start_step, end_step)
       - finish(solution, reason)
  4. Return RunResult with the synthesized answer, merged trajectories,
     and full sub_runs.

Differences from the original (approved):
  - Rollouts run sequentially (same as oagent).
  - Aggregator uses our JSON-in-content tool-calling protocol
    ({"think":..., "tools":[...]}) instead of native OpenAI FC.
  - Only short-answer task type supported: finish() expects
    <explanation>...</explanation><answer>...</answer> in solution.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import json_repair
from jinja2 import StrictUndefined, Template
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text
from rouge_score import rouge_scorer as _rouge_scorer

from scripts.kernel.token_counter import (
    count_tokens_messages, count_tokens_text,
)
from scripts.kernel.protocols import BaseAction
from scripts.kernel.types import (
    RunResult, RuntimeContext, StepRecord, TaskInput, ToolCall,
)
from scripts.kernel.monitoring import LogLevel, YELLOW_HEX
from scripts.models.base import MessageRole

from .memory import FullHistoryMemory
from .planning import PlanningStrategy as NoPlanning


logger = logging.getLogger(__name__)

# ROUGE-L scorer singleton (matches original aggagent/tools.py)
_ROUGE_SCORER = _rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


def _populate_template(template: str, variables: dict) -> str:
    return Template(template, undefined=StrictUndefined).render(**variables)


# ════════════════════════════════════════════════════════════════════
# META-TOOL HELPERS (faithful ports of aggagent/tools.py)
# ════════════════════════════════════════════════════════════════════

def _get_text_content(message: dict, key: str = "content") -> str:
    """Extract text content from a message (handles str or list formats)."""
    value = message.get(key, "")
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        if isinstance(value[0], dict):
            text = value[0].get("text") or ""
            if key == "content":
                name = message.get("name")
                recipient = message.get("recipient")
                if name:
                    text = f"[Tool Response: {name}]\n{text}"
                elif recipient:
                    text = f"[Tool Call: {recipient}]\n{text}"
            return text
    return ""


def _rouge_l_recall(query: str, text: str) -> float:
    """ROUGE-L recall of query against text."""
    if not query or not text:
        return 0.0
    return _ROUGE_SCORER.score(query, text)['rougeL'].recall


def _truncate_text(text: str, max_words: int = 150) -> str:
    """Truncate text to first n words (faithful to original)."""
    if not text:
        return ""
    count = 0
    for m in re.finditer(r'\S+', text):
        count += 1
        if count == max_words:
            return text[:m.end()] + '\n[... truncated]'
    return text


def _approx_tokens_from_messages(messages: list, chars_per_token: float = 4.0) -> int:
    """Approximate token count for a list of messages.
    Matches original aggagent/tools.py::_count_tokens_approx."""
    total_chars = 0
    for msg in messages:
        for key in ["role", "reasoning_content", "reasoning", "content"]:
            value = msg.get(key)
            if isinstance(value, str):
                total_chars += len(value)
        tool_calls = msg.get("tool_calls")
        if tool_calls is not None:
            total_chars += len(json.dumps(tool_calls, ensure_ascii=False))
    return int(total_chars / chars_per_token)


def _format_trajectory_metadata(trajectories: List[List[dict]]) -> str:
    """Per-trajectory one-liner for the aggregator user prompt.
    Faithful port of aggagent/tools.py::format_metadata."""
    blocks = []
    for i, traj in enumerate(trajectories):
        num_steps = len(traj)
        approx_tokens = _approx_tokens_from_messages(traj)

        tool_counts: Dict[str, int] = {}
        for msg in traj:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tc = msg["tool_calls"][0]
                func = tc.get("function", {})
                if not isinstance(func, dict):
                    try:
                        func = func.to_dict()
                    except Exception:
                        func = {}
                name = func.get("name")
                if name:
                    tool_counts[name] = tool_counts.get(name, 0) + 1

        tool_str = (
            ", ".join(f"{n}x{c}" for n, c in sorted(tool_counts.items()))
            if tool_counts else "none"
        )
        blocks.append(
            f"Trajectory {i + 1}: {num_steps} steps, ~{approx_tokens:,} tokens "
            f"| tools: {tool_str}"
        )
    return "\n\n".join(blocks)


# ════════════════════════════════════════════════════════════════════
# META-TOOLS: 4 internal tools the aggregator uses over trajectories
# ════════════════════════════════════════════════════════════════════

def _meta_get_solution(params: dict, trajectories: List[List[dict]]) -> Any:
    """Retrieve the final content from trajectories' last step.
    Returns list of {trajectory_id, content} entries."""
    trajectory_id = params.get("trajectory_id") if isinstance(params, dict) else None

    n = len(trajectories)
    if trajectory_id is not None:
        if not isinstance(trajectory_id, int) or trajectory_id < 1 or trajectory_id > n:
            return f"[get_solution] 'trajectory_id' must be 1-{n}"
        trajs = [(trajectory_id - 1, trajectories[trajectory_id - 1])]
    else:
        trajs = list(enumerate(trajectories))

    results = []
    for i, traj in trajs:
        content = (_get_text_content(traj[-1]) if traj else None) or ""
        results.append({"trajectory_id": i + 1, "content": content})
    return results


def _meta_search_trajectory(params: dict, trajectories: List[List[dict]]) -> Any:
    """ROUGE-L ranked keyword search within a single trajectory.
    Faithful port of SearchTrajectoriesTool.call."""
    if not isinstance(params, dict):
        return "[search_trajectory] Invalid request: params must be an object."
    query = params.get("query")
    if not query:
        return "[search_trajectory] 'query' is required."
    trajectory_id = params.get("trajectory_id")
    if trajectory_id is None:
        return "[search_trajectory] 'trajectory_id' is required"

    n = len(trajectories)
    if not isinstance(trajectory_id, int) or trajectory_id < 1 or trajectory_id > n:
        return f"[search_trajectory] 'trajectory_id' must be 1-{n}"

    max_results = min(int(params.get("k", 5) or 5), 10)
    role_filter = params.get("role", None)
    traj = trajectories[trajectory_id - 1]

    scored = []
    for step_idx, step in enumerate(traj):
        if role_filter is not None and step.get("role", "") != role_filter:
            continue
        content = _get_text_content(step) or ""
        reasoning_content = (
            _get_text_content(step, "reasoning_content")
            or _get_text_content(step, "reasoning")
            or ""
        )
        tool_calls_str = (
            json.dumps(step.get("tool_calls"), ensure_ascii=False)
            if step.get("tool_calls") else ""
        )
        score = max(
            _rouge_l_recall(query, content),
            _rouge_l_recall(query, reasoning_content),
            _rouge_l_recall(query, tool_calls_str),
        )
        if score > 0:
            scored.append((score, step_idx, step))

    scored.sort(key=lambda x: -x[0])

    matches = []
    for score, step_idx, step in scored[:max_results]:
        content = _get_text_content(step)
        reasoning_content = (
            _get_text_content(step, "reasoning_content")
            or _get_text_content(step, "reasoning")
        )
        tool_calls = step.get("tool_calls")
        match_entry = {
            "trajectory_id": trajectory_id,
            "step": step_idx + 1,
            "role": step.get("role", ""),
            "score": round(score, 3),
        }
        if content:
            match_entry["content"] = _truncate_text(content)
        if reasoning_content:
            match_entry["reasoning"] = _truncate_text(reasoning_content)
        if tool_calls:
            match_entry["tool_calls"] = tool_calls
        matches.append(match_entry)

    if not matches:
        role_msg = f" (role={role_filter})" if role_filter else ""
        return (
            f"[search_trajectory] No matches found for '{query}'"
            f"{role_msg} in trajectory {trajectory_id}"
        )
    return matches


def _meta_get_segment(params: dict, trajectories: List[List[dict]]) -> Any:
    """Read a contiguous range of steps (max 5).
    Faithful port of GetSegmentTool.call."""
    if not isinstance(params, dict):
        return "[get_segment] Invalid request: params must be an object."
    try:
        trajectory_id = params["trajectory_id"]
        start_step = params["start_step"]
        end_step = params["end_step"]
    except Exception:
        return (
            "[get_segment] Invalid request: must contain "
            "'trajectory_id', 'start_step', 'end_step'."
        )

    n_traj = len(trajectories)
    if (not isinstance(trajectory_id, int)) or trajectory_id < 1 or trajectory_id > n_traj:
        return f"[get_segment] 'trajectory_id' must be 1-{n_traj}"
    traj = trajectories[trajectory_id - 1]
    n = len(traj)
    try:
        start_step = int(start_step)
        end_step = int(end_step)
    except Exception:
        return "[get_segment] 'start_step' and 'end_step' must be integers."
    start_step = max(1, min(start_step, n))
    end_step = max(1, min(end_step, n))
    if start_step > end_step:
        start_step = end_step
    if end_step - start_step > 4:
        end_step = start_step + 4
    start_0 = start_step - 1
    end_0 = end_step - 1

    result = []
    for step_idx in range(start_0, end_0 + 1):
        step = traj[step_idx]
        entry = {"step": step_idx + 1, "role": step.get("role", "")}
        content = _get_text_content(step)
        reasoning = (
            _get_text_content(step, "reasoning_content")
            or _get_text_content(step, "reasoning")
        )
        tool_calls = step.get("tool_calls")
        if content:
            entry["content"] = _truncate_text(content, 600)
        if reasoning:
            entry["reasoning"] = _truncate_text(reasoning, 600)
        if tool_calls:
            entry["tool_calls"] = tool_calls
        result.append(entry)
    return result


def _meta_finish(params: dict) -> Any:
    """Validate the final synthesized solution (short-answer variant only).
    Requires <explanation>...</explanation><answer>...</answer> in solution."""
    if not isinstance(params, dict):
        return "[finish] Invalid request: params must be an object."
    required = ["solution", "reason"]
    missing = [f for f in required if f not in params]
    if missing:
        return f"[finish] Invalid request: missing field(s): {', '.join(missing)}"
    solution = params.get("solution", "") or ""
    explanation_match = re.search(r'<explanation>(.*?)</explanation>', solution, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)
    if not explanation_match or not explanation_match.group(1).strip():
        return (
            "[finish] Invalid solution format: missing or empty "
            "<explanation>...</explanation> section."
        )
    if not answer_match or not answer_match.group(1).strip():
        return (
            "[finish] Invalid solution format: missing or empty "
            "<answer>...</answer> section."
        )
    return {
        "solution": solution,
        "reason": params.get("reason", ""),
        "extracted_answer": answer_match.group(1).strip(),
    }


# Meta-tool JSON schemas (injected into the aggregator step prompt)
_META_TOOL_SCHEMAS = [
    {
        "name": "get_solution",
        "description": (
            "Retrieves the final content from trajectories' last step. "
            "Returns a list of {trajectory_id, content} entries. "
            "Omit trajectory_id to retrieve all."
        ),
        "parameters": {
            "properties": {
                "trajectory_id": {
                    "type": "integer",
                    "description": "Trajectory index. Omit to retrieve all.",
                    "nullable": True,
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_trajectory",
        "description": (
            "Searches for keywords or phrases within a single trajectory. "
            "Returns top matching steps ranked by ROUGE-L recall."
        ),
        "parameters": {
            "properties": {
                "trajectory_id": {
                    "type": "integer",
                    "description": "Trajectory index to search within.",
                },
                "query": {
                    "type": "string",
                    "description": "Search term or phrase.",
                },
                "role": {
                    "type": "string",
                    "description": (
                        "Optional role filter: 'tool' (actual environment "
                        "observations) or 'assistant'. Omit to search all steps."
                    ),
                    "nullable": True,
                },
                "k": {
                    "type": "integer",
                    "description": "Max matches to return (default 5, max 10).",
                    "nullable": True,
                },
            },
            "required": ["trajectory_id", "query"],
        },
    },
    {
        "name": "get_segment",
        "description": (
            "Reads the full content of a contiguous range of steps from a "
            "trajectory (max 5 steps). Use after search_trajectory to inspect "
            "a step in full with surrounding context."
        ),
        "parameters": {
            "properties": {
                "trajectory_id": {
                    "type": "integer",
                    "description": "Trajectory index.",
                },
                "start_step": {
                    "type": "integer",
                    "description": "Start step (inclusive, 1-indexed).",
                },
                "end_step": {
                    "type": "integer",
                    "description": (
                        "End step (inclusive). end_step - start_step <= 4."
                    ),
                },
            },
            "required": ["trajectory_id", "start_step", "end_step"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Submits the final synthesized solution. 'solution' must contain "
            "<explanation>...</explanation><answer>...</answer>."
        ),
        "parameters": {
            "properties": {
                "solution": {
                    "type": "string",
                    "description": (
                        "Self-contained solution string with exactly two XML "
                        "sections: <explanation>detailed reasoning</explanation>"
                        "<answer>the exact answer</answer>. Do not reference "
                        "trajectories, get_solution, or agents."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Meta-reasoning explaining how you evaluated trajectories "
                        "and resolved conflicts."
                    ),
                },
            },
            "required": ["solution", "reason"],
        },
    },
]


def _format_meta_tool_schemas() -> str:
    return json.dumps(_META_TOOL_SCHEMAS, indent=2, ensure_ascii=False)


# ════════════════════════════════════════════════════════════════════
# ACTION STRATEGY
# ════════════════════════════════════════════════════════════════════

class ActionStrategy(BaseAction):
    """K rollouts + agentic aggregation.

    Constructor parameters:
      num_rollouts: number of independent rollout workers (default 4)
      max_aggregator_iterations: cap on aggregator tool-calling loop (default 100)
      max_aggregator_context_tokens: when aggregator context exceeds this,
        inject the final_message turn and force a finish-only call
        (default 100000, matching original paper)
    """

    def __init__(
        self,
        prompts=None,
        num_rollouts: int = 4,
        max_aggregator_iterations: int = 100,
        max_aggregator_context_tokens: int = 100_000,
    ):
        self.prompts = prompts or {}
        self._num_rollouts = num_rollouts
        self._max_agg_iters = max_aggregator_iterations
        self._max_agg_tokens = max_aggregator_context_tokens

    # ── Public entry point ──────────────────────────────────────────

    def run(self, task: str, ctx: RuntimeContext) -> RunResult:
        ctx.logger.log_rule(
            f"AggAgent: {self._num_rollouts} rollouts + agentic aggregation",
            level=LogLevel.INFO,
        )
        logger.info(
            f"AggAgentAction: starting with K={self._num_rollouts}, "
            f"max_agg_iters={self._max_agg_iters}, "
            f"max_agg_tokens={self._max_agg_tokens}"
        )

        steps_per_rollout = max(ctx.max_steps // max(self._num_rollouts, 1), 5)

        # ── Phase 1: K rollouts ──
        rollout_results: List[RunResult] = []
        for i in range(self._num_rollouts):
            ctx.logger.log_rule(f"Rollout Worker {i+1}/{self._num_rollouts}", level=LogLevel.INFO)
            result = self._run_rollout(task, ctx, steps_per_rollout, i + 1)
            rollout_results.append(result)
            ctx.logger.log(
                f"Rollout {i+1} answer: {str(result.answer)[:200]}",
                level=LogLevel.INFO,
            )

        # ── Phase 2: convert trajectories to OpenAI-style messages ──
        trajectories_as_messages = [
            self._trajectory_to_message_list(task, r) for r in rollout_results
        ]

        # ── Phase 3: aggregator loop ──
        ctx.logger.log_rule("Aggregator", level=LogLevel.INFO)
        agg_output = self._run_aggregator(task, trajectories_as_messages, ctx)
        final_answer = agg_output.get("extracted_answer") or agg_output.get("solution") or ""
        agg_reason = agg_output.get("reason", "")
        agg_steps = agg_output.get("steps", [])
        agg_terminated = agg_output.get("terminated_reason", "aggregator_done")

        ctx.logger.log(
            Text(
                f"Aggregator final answer: {str(final_answer)[:300]}",
                style=f"bold {YELLOW_HEX}",
            ),
            level=LogLevel.INFO,
        )

        # ── Phase 4: assemble merged trajectory ──
        all_trajectories: List[StepRecord] = []
        for i, r in enumerate(rollout_results, start=1):
            for step in r.trajectory:
                step.action_reasoning = (
                    f"[Rollout-{i}] " + (step.action_reasoning or "")
                )
                all_trajectories.append(step)
        for step in agg_steps:
            step.action_reasoning = (
                "[Aggregator] " + (step.action_reasoning or "")
            )
            all_trajectories.append(step)

        sub_runs = []
        for i, r in enumerate(rollout_results, start=1):
            r.metadata["worker_name"] = f"Rollout-{i}"
            sub_runs.append(r)

        return RunResult(
            answer=final_answer,
            trajectory=all_trajectories,
            terminated_reason="aggagent_done",
            metadata={
                "num_rollouts": self._num_rollouts,
                "rollout_answers": {
                    f"Rollout-{i+1}": str(r.answer)[:1000]
                    for i, r in enumerate(rollout_results)
                },
                "aggregator_terminated_reason": agg_terminated,
                "aggregator_reason": str(agg_reason)[:4000],
                "aggregator_solution": str(agg_output.get("solution", ""))[:4000],
                "aggregator_iterations": agg_output.get("iterations", 0),
            },
            sub_runs=sub_runs,
        )

    # ── Phase 1: rollout worker (standard ReAct loop) ────────────────

    def _run_rollout(
        self,
        task: str,
        ctx: RuntimeContext,
        max_steps: int,
        index: int,
    ) -> RunResult:
        """Run one rollout worker with independent memory and standard ReAct."""
        rollout_memory = FullHistoryMemory(prompts=self.prompts)

        # Extract base system prompt from outer memory's build_context
        base_system = self._extract_base_system(ctx)
        rollout_system = self.prompts.get("rollout_worker", {}).get(
            "system_prompt_suffix", ""
        )
        full_system = (
            f"{base_system}\n\n{rollout_system}" if rollout_system else base_system
        )

        rollout_memory.initialize(full_system, TaskInput(task=task))

        rollout_planning = NoPlanning(prompts=self.prompts)
        rollout_ctx = RuntimeContext(
            memory=rollout_memory,
            planning=rollout_planning,
            tool_policy=ctx.tool_policy,
            model=ctx.model,
            execute_tool=ctx.execute_tool,
            get_tool_schemas=ctx.get_tool_schemas,
            logger=ctx.logger,
            prompt_templates=ctx.prompt_templates,
            max_steps=max_steps,
        )

        return self._react_loop(task, rollout_ctx)

    def _react_loop(self, task: str, ctx: RuntimeContext) -> RunResult:
        """Standard ReAct loop used by each rollout worker."""
        trajectory: List[StepRecord] = []
        final_answer = None
        step_number = 0

        # Step 0: Planning (NoPlanning — just initializes directive)
        memory_view = ctx.memory.build_context()
        tool_selection = ctx.tool_policy.select_tools(task, step_number, memory_view)
        plan = ctx.planning.init_plan(
            task, memory_view, tool_selection.tool_schemas_json, ctx.model,
        )
        ctx.memory.update_plan(plan)
        ctx.logger.log_markdown(plan.plan, title="Initial Plan", level=LogLevel.INFO)
        step_number += 1

        step_template = self.prompts["step"]["pre_messages"]

        while final_answer is None and step_number <= ctx.max_steps:
            step_start_time = time.time()
            ctx.logger.log_rule(f"Step {step_number}", level=LogLevel.INFO)

            memory_view = ctx.memory.build_context(plan)
            tool_selection = ctx.tool_policy.select_tools(
                task, step_number, memory_view, plan,
            )

            step_prompt = _populate_template(
                step_template,
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
                tool_calls, think, final_answer = self._parse_and_execute_tools(
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
            terminated_reason=(
                "final_answer" if step_number <= ctx.max_steps else "max_steps"
            ),
        )

    def _parse_and_execute_tools(
        self,
        response: Any,
        step: StepRecord,
        ctx: RuntimeContext,
        task: str,
    ) -> Tuple[List[ToolCall], str, Optional[Any]]:
        """Standard JSON tool-call parsing, identical to simple_react pattern."""
        try:
            content_dict = json_repair.loads(response.content)
        except Exception:
            content_dict = {}

        if isinstance(content_dict, list):
            if (content_dict and isinstance(content_dict[0], dict)
                    and "tools" in content_dict[0]):
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
        """Force a final answer when max_steps reached (rollout worker fallback)."""
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
                start_time=step_start_time,
                end_time=time.time(),
                error=e,
                action_reasoning="Forced final answer — failed",
            )
            step.duration = step.end_time - step.start_time
            if trajectory is not None:
                trajectory.append(step)
            return f"Error generating final answer: {e}"

    def _extract_base_system(self, ctx: RuntimeContext) -> str:
        """Extract base system prompt from outer memory's build_context."""
        memory_view = ctx.memory.build_context()
        for msg in memory_view.messages:
            if msg.get("role") == MessageRole.SYSTEM or msg.get("role") == "system":
                content = msg.get("content", [])
                if isinstance(content, list) and content:
                    return content[0].get("text", "")
                elif isinstance(content, str):
                    return content
        return ""

    # ── Phase 2: trajectory → OpenAI message list ────────────────────

    def _trajectory_to_message_list(
        self, task: str, run_result: RunResult,
    ) -> List[dict]:
        """Convert a StepRecord trajectory into an OpenAI-style message list.

        Format matches the aggregator's expected trajectory input:
          [{role: system, ...}, {role: user, ...}, {role: assistant, ..., tool_calls: [...]}, {role: tool, ...}, ...]
        """
        messages: List[dict] = []
        messages.append({
            "role": "user",
            "content": task,
        })

        for step in run_result.trajectory:
            # assistant message with reasoning and tool_calls (if any)
            assistant_msg = {
                "role": "assistant",
                "content": step.action_output if step.action_output else "",
            }
            reasoning = step.action_reasoning or step.action_think or ""
            if reasoning:
                assistant_msg["reasoning"] = reasoning
            if step.tool_calls:
                tc_list = []
                for tc in step.tool_calls:
                    tc_list.append({
                        "id": tc.id or f"call_{len(tc_list)}",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(
                                tc.arguments, ensure_ascii=False,
                            ) if not isinstance(tc.arguments, str)
                            else tc.arguments,
                        },
                    })
                assistant_msg["tool_calls"] = tc_list
            messages.append(assistant_msg)

            # tool response messages (one per tool call, observations lumped)
            if step.tool_calls and step.observations:
                for tc in step.tool_calls:
                    if tc.name == "final_answer":
                        continue
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id or "",
                        "name": tc.name,
                        "content": step.observations,
                    })

        # Terminal assistant message with the final answer
        if run_result.answer is not None:
            messages.append({
                "role": "assistant",
                "content": str(run_result.answer),
            })
        return messages

    # ── Phase 3: aggregator loop ─────────────────────────────────────

    def _run_aggregator(
        self,
        task: str,
        trajectories: List[List[dict]],
        ctx: RuntimeContext,
    ) -> Dict[str, Any]:
        """Run the aggregator loop with 4 internal meta-tools.

        Returns dict with keys:
          - solution: raw solution string (with XML sections)
          - extracted_answer: the text inside <answer>...</answer>
          - reason: meta-reasoning
          - steps: List[StepRecord] of aggregator iterations
          - iterations: int
          - terminated_reason: str
        """
        metadata = _format_trajectory_metadata(trajectories)
        traj_n = len(trajectories)

        agg_prompts = self.prompts.get("aggregator", {})
        system_prompt = agg_prompts.get("system_prompt", "")
        user_prompt_template = agg_prompts.get("user_prompt", "")
        step_prompt_template = agg_prompts.get("step_prompt", "")
        final_message = agg_prompts.get(
            "final_message",
            "You have now reached the maximum context length. Stop making "
            "tool calls and call 'finish' with your best answer based on "
            "all information above.",
        )

        user_prompt = user_prompt_template.replace("{question}", task) \
            .replace("{metadata}", metadata) \
            .replace("{traj_N}", str(traj_n))

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]

        steps: List[StepRecord] = []
        iteration = 0
        terminated_reason = "max_iterations"
        result: Optional[Dict[str, Any]] = None
        context_limit_reached = False
        meta_tool_schemas_json = _format_meta_tool_schemas()

        while iteration < self._max_agg_iters:
            iteration += 1
            step_start_time = time.time()
            ctx.logger.log_rule(
                f"Aggregator iter {iteration}", level=LogLevel.INFO,
            )

            # Build step prompt that asks for the next meta-tool call.
            step_prompt = step_prompt_template.replace(
                "{meta_tools_json}", meta_tool_schemas_json,
            ).replace("{question}", task)
            call_messages = messages + [{
                "role": "user",
                "content": [{"type": "text", "text": step_prompt}],
            }]

            # Force-finish branch when context overflow
            force_finish = False
            if _approx_tokens_from_messages(messages) > self._max_agg_tokens:
                context_limit_reached = True
                ctx.logger.log(
                    "Aggregator context budget exceeded — forcing finish",
                    level=LogLevel.INFO,
                )
                force_finish_msg = {
                    "role": "user",
                    "content": [{"type": "text", "text": final_message}],
                }
                messages.append(force_finish_msg)
                call_messages = messages + [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": (
                            step_prompt
                            + "\n\nYou MUST call the 'finish' tool now."
                        ),
                    }],
                }]
                force_finish = True

            try:
                response = ctx.model(call_messages)
            except Exception as e:
                logger.error(f"Aggregator model call failed at iter {iteration}: {e}")
                step = StepRecord(
                    step_number=iteration,
                    error=e,
                    start_time=step_start_time,
                    end_time=time.time(),
                    action_reasoning=f"[Aggregator iter {iteration}] model error",
                )
                step.duration = step.end_time - step.start_time
                steps.append(step)
                continue

            content = response.content or ""
            input_tokens = count_tokens_messages(call_messages)
            output_tokens = count_tokens_text(content)

            # Parse JSON {"think":..., "tools":[...]}
            try:
                parsed = json_repair.loads(content)
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                think = parsed.get("think", "") or ""
                tools_field = parsed.get("tools", None)
            elif isinstance(parsed, list):
                think = ""
                tools_field = parsed
            else:
                think = ""
                tools_field = None

            if isinstance(tools_field, dict):
                tool_calls_list = [tools_field]
            elif isinstance(tools_field, list):
                tool_calls_list = [t for t in tools_field if isinstance(t, dict)]
            else:
                tool_calls_list = []

            step = StepRecord(
                step_number=iteration,
                model_input_messages=call_messages,
                model_output_messages=response,
                start_time=step_start_time,
                end_time=time.time(),
                action_think=think,
                action_reasoning=f"[Aggregator iter {iteration}]",
                input_token_count=input_tokens,
                output_token_count=output_tokens,
                total_token_count=input_tokens + output_tokens,
            )
            step.duration = step.end_time - step.start_time

            if think:
                ctx.logger.log(
                    Panel(Text(f"Think: {think[:400]}...")),
                    level=LogLevel.INFO,
                )

            # Persist the assistant message into the aggregator history
            assistant_history_msg = {
                "role": "assistant",
                "content": [{"type": "text", "text": content}],
            }
            messages.append(assistant_history_msg)

            if not tool_calls_list:
                # No valid tool call produced. If we were already forcing
                # finish, break out with failure. Otherwise nudge and retry.
                ctx.logger.log(
                    "Aggregator produced no tool call — nudging.",
                    level=LogLevel.INFO,
                )
                if force_finish:
                    terminated_reason = "no_finish_after_force"
                    steps.append(step)
                    break
                nudge = (
                    "Your previous response did not contain a valid tool call. "
                    "Output ONLY a JSON object of the form "
                    '{"think": "...", "tools": [{"name": "...", "arguments": {...}}]} '
                    "using one of the 4 meta-tools."
                )
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": nudge}],
                })
                steps.append(step)
                continue

            # Execute first meta-tool call (match original which takes tool_calls[:1])
            tc = tool_calls_list[0]
            tool_name = str(tc.get("name", ""))
            tool_args = tc.get("arguments", {}) or {}
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {}

            step.tool_calls.append(ToolCall(
                name=tool_name, arguments=tool_args, id=tc.get("id", ""),
            ))
            ctx.logger.log(
                Panel(Text(
                    f"Meta-tool: '{tool_name}' with {json.dumps(tool_args)[:400]}"
                )),
                level=LogLevel.INFO,
            )

            # Dispatch
            try:
                if tool_name == "get_solution":
                    observation = _meta_get_solution(tool_args, trajectories)
                elif tool_name == "search_trajectory":
                    observation = _meta_search_trajectory(tool_args, trajectories)
                elif tool_name == "get_segment":
                    observation = _meta_get_segment(tool_args, trajectories)
                elif tool_name == "finish":
                    observation = _meta_finish(tool_args)
                else:
                    observation = (
                        f"Error: unknown meta-tool '{tool_name}'. "
                        "Allowed: get_solution, search_trajectory, "
                        "get_segment, finish."
                    )
            except Exception as e:
                logger.error(f"Meta-tool '{tool_name}' execution failed: {e}")
                observation = f"Error executing '{tool_name}': {str(e)}"

            # finish termination
            if tool_name == "finish" and isinstance(observation, dict):
                step.action_output = observation.get(
                    "extracted_answer",
                    observation.get("solution", ""),
                )
                step.observations = json.dumps(observation, ensure_ascii=False)[:2000]
                steps.append(step)
                result = observation
                terminated_reason = "finish"
                ctx.logger.log(
                    "Aggregator finished successfully.",
                    level=LogLevel.INFO,
                )
                break

            # Regular tool response appended to aggregator history
            obs_str = (
                json.dumps(observation, ensure_ascii=False)
                if not isinstance(observation, str) else observation
            )
            step.observations = obs_str[:4000]
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        f"Tool response for '{tool_name}':\n{obs_str[:8000]}"
                    ),
                }],
            })
            safe_obs = escape(obs_str[:400])
            ctx.logger.log(
                f"Meta-tool response: {safe_obs}...",
                level=LogLevel.INFO,
            )
            steps.append(step)

            if force_finish:
                # We forced finish but the model called a non-finish tool.
                # Abort — it didn't follow instructions.
                terminated_reason = "no_finish_after_force"
                break

        if result is None:
            # Aggregator failed to emit a valid finish. Fall back to the
            # first rollout's final answer if we have one, otherwise return
            # an empty answer.
            result = {
                "solution": "",
                "reason": "Aggregator did not produce a valid finish call.",
                "extracted_answer": "",
            }

        result["steps"] = steps
        result["iterations"] = iteration
        result["terminated_reason"] = terminated_reason
        result["context_limit_reached"] = context_limit_reached
        logger.info(
            f"AggAgentAction aggregator done: iterations={iteration}, "
            f"terminated={terminated_reason}, "
            f"has_solution={bool(result.get('solution'))}"
        )
        return result
