"""
MemoBrain MemoryStrategy: dependency-aware reasoning-graph memory.

Adapted from MemoBrain (https://github.com/TommyChien/MemoBrain):
  - src/memobrain.py (the MemoBrain class: memorize / recall / _organize)
  - src/problem_tree.py (ReasoningGraph + ReasoningNode + Edge; inlined
    below so the harness holds only the four canonical modules)

The reasoning-graph data structure and the two LLM-driven operations
(memorize / flush+fold) are preserved in spirit; only the LLM plumbing
is swapped from AsyncOpenAI to the kernel's synchronous `ctx.model`
(injected via set_model).

Execution semantics (preserved from original):
  - memorize is PASSIVELY triggered by the Action module after every
    completed tool call, with the exact (assistant, tool_response) pair
    that just occurred. Not under the agent's decision.
  - recall is PASSIVELY triggered at the start of each planning turn
    whenever current-context tokens exceed `max_memory_size`. Not under
    the agent's decision.
  - The agent's only ACTIVE decisions are the marker-based outputs
    parsed by Action: <tool_call>, <answer>, or "continue" fallthrough.
"""

from __future__ import annotations

import datetime
import itertools
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import tiktoken

from scripts.kernel.protocols import BaseMemory
from scripts.kernel.types import (
    MemoryView, Message, PlanState, StepRecord, SummaryState, TaskInput,
)
from scripts.models.base import MessageRole


logger = logging.getLogger(__name__)

# Token encoder used by the original MemoBrain (utils.py).
_ENCODING = tiktoken.get_encoding("o200k_base")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _strip_markdown_fence(content: str) -> str:
    """Strip a surrounding ```json ... ``` / ``` ... ``` fence if present.

    Many post-training-free models (Gemini, GPT-4o, etc.) wrap JSON in
    markdown fences despite explicit instructions. The original
    MemoBrain was trained on raw JSON and never hit this; we normalise
    defensively before json.loads to keep memorize/recall useful.
    """
    s = content.strip()
    if s.startswith("```"):
        newline = s.find("\n")
        if newline != -1:
            s = s[newline + 1:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _num_tokens_from_messages(
    messages: List[Dict[str, Any]],
    tokens_per_message: int = 3,
    tokens_per_name: int = 1,
) -> int:
    """Token counter ported verbatim from MemoBrain examples/utils.py."""
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if not isinstance(value, str):
                value = str(value)
            num_tokens += len(_ENCODING.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


# ─────────────────────────────────────────────────────────────────────
# Reasoning graph (inlined verbatim from MemoBrain src/problem_tree.py)
# ─────────────────────────────────────────────────────────────────────


NodeKind = Literal[
    "task",
    "subtask",
    "evidence",
    "summary",
]


@dataclass
class ReasoningNode:
    node_id: int
    kind: NodeKind
    thought: str
    related_turn_ids: List[int]
    active: Union[bool, str] = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "related_turn_ids": self.related_turn_ids,
            "thought": self.thought,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningNode":
        return cls(
            node_id=data["node_id"],
            kind=data["kind"],
            related_turn_ids=data.get("related_turn_ids", []),
            thought=data["thought"],
            active=data.get("active", True),
        )


@dataclass
class Edge:
    src: str
    dst: str
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        return cls(
            src=data["src"],
            dst=data["dst"],
            rationale=data.get("rationale", ""),
        )


class ReasoningGraph:
    def __init__(self):
        self.nodes: Dict[int, ReasoningNode] = {}
        self.edges: List[Edge] = []
        self._id_counter = itertools.count(1)

    def add_node(
        self,
        kind: NodeKind,
        thought: str,
        related_turn_ids: List[int] = None,
    ) -> ReasoningNode:
        node = ReasoningNode(
            node_id=next(self._id_counter),
            kind=kind,
            thought=thought,
            related_turn_ids=related_turn_ids if related_turn_ids is not None else [],
        )
        self.nodes[node.node_id] = node
        return node

    def add_edge(
        self,
        src: int,
        dst: int,
        rationale: str = "",
    ) -> Edge:
        if src not in self.nodes or dst not in self.nodes:
            raise ValueError(f"Unknown node id in edge: {src} -> {dst}")
        edge = Edge(src=src, dst=dst, rationale=rationale)
        self.edges.append(edge)
        return edge

    def fold_nodes(
        self,
        span_node_ids: List[int],
        thought: str,
        rationale: str = "",
    ) -> ReasoningNode:
        related_turn_ids = []
        for nid in span_node_ids:
            if nid in self.nodes.keys():
                related_turn_ids.extend(self.nodes[nid].related_turn_ids)
        related_turn_ids = list(set(related_turn_ids))
        related_turn_ids.sort()

        summary_node = self.add_node(
            kind="summary",
            thought=thought,
            related_turn_ids=related_turn_ids,
        )

        span_set = set(span_node_ids)
        new_edges: List[Edge] = []

        for e in self.edges:
            if e.src in span_set and e.dst in span_set:
                continue
            elif e.src not in span_set and e.dst in span_set:
                new_edges.append(Edge(src=e.src, dst=summary_node.node_id, rationale=rationale))
            elif e.src in span_set and e.dst not in span_set:
                new_edges.append(Edge(src=summary_node.node_id, dst=e.dst, rationale=e.rationale))
            else:
                new_edges.append(e)

        self.edges = new_edges

        for nid in span_set:
            if nid in self.nodes.keys():
                self.nodes[nid].active = False

        return summary_node

    def flush_node(self, node_id: int) -> None:
        if node_id in self.nodes.keys():
            self.nodes[node_id].active = "Flushed"
            return
        raise ValueError(f"Node {node_id} not found in graph")

    def apply_patch(self, patch_json: Dict[str, Any], related_turn_ids: List[int] = None):
        tempid2realid = {}
        for node in patch_json["add_nodes"]:
            real_node_id = self.add_node(
                kind=node["kind"],
                thought=node["thought"],
                related_turn_ids=related_turn_ids if related_turn_ids is not None else [],
            ).node_id
            tempid2realid[node["tmp_id"]] = real_node_id

        def extract_numbers_from_string(s: str) -> list:
            return [int(num) for num in re.findall(r'\d+', s)]

        for edge in patch_json["add_edges"]:
            src = str(edge["src"])
            dst = str(edge["dst"])

            if src in tempid2realid:
                src = tempid2realid[src]
            else:
                try:
                    src = int(src)
                except ValueError:
                    src = extract_numbers_from_string(src)[0]
            if dst in tempid2realid:
                dst = tempid2realid[dst]
            else:
                dst = int(dst)

            self.add_edge(
                src=src,
                dst=dst,
                rationale=edge.get("rationale", ""),
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningGraph":
        g = cls()
        g.nodes = {int(nid): ReasoningNode.from_dict(nd) for nid, nd in data["nodes"].items()}
        g.edges = [Edge.from_dict(ed) for ed in data["edges"]]
        max_id = max(g.nodes.keys())
        g._id_counter = itertools.count(max_id + 1)
        return g

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def get_leaf_node_ids(self) -> List[int]:
        out_degree = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            if edge.src in out_degree:
                out_degree[edge.src] += 1
        leaf_ids = [nid for nid, node in self.nodes.items() if out_degree.get(nid, 0) == 0 and node.active]
        return leaf_ids

    def pretty_print(self, mode: str = "full") -> str:
        lines: List[str] = []

        children = defaultdict(list)
        in_degrees = {nid: 0 for nid in self.nodes if self.nodes[nid].active}
        for edge in self.edges:
            if self.nodes.get(edge.src, None) and self.nodes.get(edge.dst, None):
                if self.nodes[edge.src].active and self.nodes[edge.dst].active:
                    children[edge.src].append((edge.dst, edge.rationale))
                    in_degrees[edge.dst] += 1
        roots = [nid for nid, deg in in_degrees.items() if deg == 0 and self.nodes[nid].active]
        visited = set()

        def walk(node_id, indent: str):
            if node_id in visited:
                return
            visited.add(node_id)
            n = self.nodes[node_id]
            if isinstance(n.active, str):
                status = n.active
            else:
                status = "Active" if n.active else "Inactive"

            if mode == "full":
                lines.append(f"{indent}- Node {node_id}: [{n.kind}] [{status}] {n.thought}")
            else:
                lines.append(f"{indent}- Node {node_id}: [{n.kind}] [{status}]")

            for dst, rationale in children.get(node_id, []):
                dst_node = self.nodes.get(dst)
                if dst_node and dst not in visited:
                    edge_info = f"--[->] Node {dst} [Rationale: {rationale}]"
                else:
                    continue
                lines.append(f"{indent}    {edge_info}")
                walk(dst, indent + "        ")

        if not roots:
            lines.append("No roots (possibly empty graph)")
        else:
            for root in roots:
                walk(root, "")

        shown_nodes = set(visited)
        hidden_nodes = [nid for nid in self.nodes if self.nodes[nid].active and nid not in shown_nodes]
        if hidden_nodes:
            lines.append("\nIsolated active nodes (not connected to roots):")
            for nid in hidden_nodes:
                n = self.nodes[nid]
                mark = []
                mark_str = f" ({', '.join(mark)})" if mark else ""
                lines.append(f"- {nid}{mark_str}: [{n.kind}] {n.thought}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# MemoryStrategy
# ─────────────────────────────────────────────────────────────────────


class MemoryStrategy(BaseMemory):
    """Reasoning-graph memory with flush+fold compaction.

    Maintains two synchronized state slots (matching the original):
      - `self.messages`: OpenAI-style `List[Dict]` (string content) —
        the raw chat history the agent LLM sees.
      - `self.graph`: ReasoningGraph — dependency graph whose active
        nodes determine how `self.messages` is reorganized when recall
        fires.
    """

    def __init__(
        self,
        prompts: Optional[Dict[str, Any]] = None,
        max_memory_size: int = 32 * 1024,
        max_retries: int = 3,
    ) -> None:
        self.prompts = prompts or {}
        self._max_memory_size = max_memory_size
        self._max_retries = max_retries

        self.graph: ReasoningGraph = ReasoningGraph()
        self.messages: List[Dict[str, Any]] = []

        self._model: Optional[Callable] = None
        self._system_prompt: str = ""
        self._task: Optional[TaskInput] = None
        self._all_steps: List[Union[StepRecord, PlanState, SummaryState]] = []

    # ─────────────────────────────────────────────────────────────────
    # Kernel integration
    # ─────────────────────────────────────────────────────────────────

    def set_model(self, model: Callable) -> None:
        """Injected by AgentRuntime — used for memorize / recall LLM calls."""
        self._model = model

    def initialize(self, system_prompt: str, task: TaskInput) -> None:
        # Inject today's date into the SYSTEM_PROMPT placeholder (matches
        # MemoBrain's original behaviour: SYSTEM_PROMPT + today_date()).
        today = datetime.date.today().strftime("%Y-%m-%d")
        final_system = system_prompt.replace("__CURRENT_DATE__", today)

        self._system_prompt = final_system
        self._task = task

        # Fresh graph + initial message pair, exactly like the original
        # driver (examples/react_with_memory.py, planning_node first-turn).
        self.graph = ReasoningGraph()
        self.messages = [
            {"role": "system", "content": final_system},
            {"role": "user", "content": task.task},
        ]
        # Seed task node with related_turn_ids=[0,1] — verbatim original.
        self.graph.add_node(
            kind="task",
            thought=f"Begin to solve the task: {task.task}",
            related_turn_ids=[0, 1],
        )
        self._all_steps = []
        logger.info(
            f"MemoBrainMemory: initialized (max_memory_size={self._max_memory_size})"
        )

    def build_context(self, plan: Optional[PlanState] = None) -> MemoryView:
        """Expose `self.messages` in JIT MessageView format for ctx.model."""
        return MemoryView(messages=[self._to_jit_message(m) for m in self.messages])

    def update(self, step: StepRecord) -> None:
        """Protocol hook: record step for trajectory logging.

        NOTE: this does NOT call memorize. Memorize is triggered
        explicitly by the Action module via `memorize_pair()` right
        after each successful tool call, matching the original driver's
        tool_call_node semantics.
        """
        self._all_steps.append(step)

    def update_plan(self, plan: PlanState) -> None:
        self._all_steps.append(plan)

    def update_summary(self, summary: SummaryState) -> None:
        self._all_steps.append(summary)

    def get_all_steps(self) -> List[StepRecord]:
        return [s for s in self._all_steps if isinstance(s, StepRecord)]

    # ─────────────────────────────────────────────────────────────────
    # MemoBrain-native API (called by ActionStrategy)
    # ─────────────────────────────────────────────────────────────────

    def current_tokens(self) -> int:
        """Current token count of `self.messages` (o200k_base)."""
        return _num_tokens_from_messages(self.messages)

    def append_assistant(self, content: str) -> None:
        """Append a raw assistant message to `self.messages`.

        Called by Action right after the LLM responds (before any tool
        call is executed) so that `self.messages` stays aligned with
        the chat the next planning turn will see.
        """
        self.messages.append({"role": "assistant", "content": content.strip()})

    def memorize_pair(
        self,
        assistant_content: str,
        tool_response_text: str,
    ) -> None:
        """PASSIVE: patch the reasoning graph with the latest (assistant,
        tool_response) pair.

        Called by Action after each completed tool call. The assistant
        message is assumed to already be in `self.messages` (appended by
        `append_assistant` before tool execution); this method appends
        the tool_response and runs the memorize LLM call over the pair.

        Verbatim algorithm ported from MemoBrain.memorize:
          1. Record start_idx = index of the just-appended assistant.
             Append the tool_response (so assistant is at start_idx,
             tool_response at start_idx+1 — matching the original's
             `self.messages.extend([assistant, tool_response])`).
          2. Group into (assistant, user) pairs from start_idx.
          3. For each pair: LLM-generate a patch, apply to graph.
        """
        start_idx = len(self.messages) - 1  # index of the assistant we just appended
        self.messages.append({"role": "user", "content": tool_response_text})

        grouped = self._group_pairs(start_idx)
        logger.info(f"MemoBrainMemory: {len(grouped)} pair(s) to memorize")

        for pair in grouped:
            patch_json = self._generate_patch(pair)
            if patch_json is None:
                # D5 lenient: skip this memorize round on LLM / JSON failure.
                logger.warning("MemoBrainMemory: skipping memorize for this pair")
                start_idx += 2
                continue
            try:
                self.graph.apply_patch(patch_json, [start_idx, start_idx + 1])
            except Exception as exc:
                logger.warning(f"MemoBrainMemory: apply_patch failed: {exc}")
            start_idx += 2

    def recall_if_needed(self, max_memory_size: Optional[int] = None) -> bool:
        """PASSIVE: if current context exceeds budget, run flush+fold.

        Called by Action at the START of each planning turn, matching
        the original driver (examples/react_with_memory.py planning_node
        lines 67-74).

        Returns True if recall ran, False otherwise.
        """
        budget = max_memory_size if max_memory_size is not None else self._max_memory_size
        tok = self.current_tokens()
        if tok <= budget:
            return False
        logger.info(
            f"MemoBrainMemory: token count {tok} > budget {budget}; running recall"
        )
        self._recall()
        post_tok = self.current_tokens()
        if post_tok > budget:
            logger.warning(
                f"MemoBrainMemory: token count after recall ({post_tok}) still > budget"
            )
        return True

    # ─────────────────────────────────────────────────────────────────
    # Internal: ported from MemoBrain.memorize / .recall /
    # ._generate_patch / ._organize / ._group_pairs.
    # ─────────────────────────────────────────────────────────────────

    def _group_pairs(self, start_idx: int) -> List[List[Dict[str, Any]]]:
        grouped: List[List[Dict[str, Any]]] = []
        temp: List[Dict[str, Any]] = []
        for msg in self.messages[start_idx:]:
            if msg["role"] in ("user", "assistant"):
                temp.append(msg)
                if len(temp) == 2:
                    grouped.append(temp)
                    temp = []
        return grouped

    def _generate_patch(self, pair: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Run the MemoBrain memorize LLM call. Returns patch JSON or None on failure."""
        if self._model is None:
            logger.warning("MemoBrainMemory: model not injected; cannot memorize")
            return None

        sys_prompt = self._get_memorize_sys_prompt()
        round_info = json.dumps(pair, ensure_ascii=False)
        graph_str = self.graph.pretty_print()
        current_message = f"CURRENT_INTERACTION:\n{round_info}\n\n{graph_str}"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": current_message},
        ]
        jit_messages = [self._to_jit_message(m) for m in messages]

        content = self._call_model_with_retries(jit_messages, label="memorize")
        if content is None:
            return None

        # D5 lenient: log + skip on JSON parse failure.
        try:
            return json.loads(_strip_markdown_fence(content))
        except Exception as exc:
            logger.warning(
                f"MemoBrainMemory: memorize JSON parse failed: {exc}; "
                f"raw content (first 200 chars): {content[:200]!r}"
            )
            return None

    def _recall(self) -> None:
        """Run the MemoBrain flush+fold LLM call and reorganize self.messages."""
        if self._model is None:
            logger.warning("MemoBrainMemory: model not injected; cannot recall")
            return

        sys_prompt = self._get_flush_fold_sys_prompt()
        graph_str = self.graph.pretty_print()
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"CURRENT_GRAPH:\n{graph_str}"},
        ]
        jit_messages = [self._to_jit_message(m) for m in messages]

        content = self._call_model_with_retries(jit_messages, label="recall")
        if content is None:
            return

        # D5 lenient: log + skip.
        try:
            result = json.loads(_strip_markdown_fence(content))
        except Exception as exc:
            logger.warning(
                f"MemoBrainMemory: recall JSON parse failed: {exc}; "
                f"raw content (first 200 chars): {content[:200]!r}"
            )
            return

        for flush_op in result.get("flush_ops", []):
            try:
                self.graph.flush_node(flush_op["id"])
            except Exception as exc:
                logger.warning(f"MemoBrainMemory: flush_node failed: {exc}")

        for fold_op in result.get("fold_ops", []):
            try:
                self.graph.fold_nodes(
                    fold_op["ids"],
                    json.dumps(fold_op.get("notes", [])),
                    fold_op.get("rationale", ""),
                )
            except Exception as exc:
                logger.warning(f"MemoBrainMemory: fold_nodes failed: {exc}")

        # Rebuild the chat-history view from the graph.
        self.messages = self._organize()

    def _organize(self) -> List[Dict[str, Any]]:
        """Rebuild message list from current graph state.

        Verbatim port of MemoBrain._organize(). Protected indices:
        first 3 and last 4 messages are always preserved.
        """
        ops_list: List[Dict[str, Any]] = []
        active_ids: List[int] = []
        summary_dict: Dict[int, Any] = {}
        total_messages = len(self.messages)
        protected_indices = set()
        protected_indices.update(range(min(3, total_messages)))
        if total_messages > 3:
            protected_indices.update(range(max(3, total_messages - 4), total_messages))

        for node in self.graph.nodes.values():
            is_active = True if node.active is True else False
            kind = node.kind
            related_turn_ids = node.related_turn_ids
            thought = node.thought

            if kind == "summary":
                if related_turn_ids:
                    last_tid = max(related_turn_ids)
                    summary_dict[last_tid] = (
                        json.loads(thought) if isinstance(thought, str) else thought
                    )
            elif not is_active:
                if isinstance(thought, str):
                    try:
                        t = json.loads(thought)
                    except Exception:
                        t = []
                else:
                    t = thought
                for idx, tid in enumerate(related_turn_ids):
                    if t and len(t) > idx:
                        ops_list.append({"turn_id": tid, "new_message": t[idx]})
            elif is_active and kind:
                for tid in related_turn_ids:
                    active_ids.append(tid)

        summary_turn_ids_to_remove = set()
        for node in self.graph.nodes.values():
            if node.kind == "summary":
                for tid in node.related_turn_ids:
                    if tid not in protected_indices:
                        summary_turn_ids_to_remove.add(tid)

        replace_dict: Dict[int, Any] = {}
        for change in ops_list:
            tid = change["turn_id"]
            if (
                tid not in active_ids
                and tid not in protected_indices
                and tid not in summary_turn_ids_to_remove
            ):
                replace_dict[tid] = change["new_message"]

        result_messages: List[Dict[str, Any]] = []

        for idx, original_msg in enumerate(self.messages):
            if idx in summary_dict and idx not in protected_indices:
                summary_msgs = summary_dict[idx]
                result_messages.extend(summary_msgs)

            if idx in summary_turn_ids_to_remove:
                continue

            if idx in replace_dict:
                result_messages.append(replace_dict[idx])
            else:
                result_messages.append(original_msg)

        return result_messages

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _call_model_with_retries(
        self, jit_messages: List[Message], label: str
    ) -> Optional[str]:
        """Call ctx.model with retries on transient failure; return content or None."""
        for attempt in range(self._max_retries):
            try:
                response = self._model(jit_messages)
                content = getattr(response, "content", "") or ""
                if not content.strip():
                    logger.warning(
                        f"MemoBrainMemory: {label} attempt {attempt+1} returned empty content"
                    )
                    if attempt < self._max_retries - 1:
                        time.sleep(3)
                        continue
                    return None
                return content
            except Exception as exc:
                logger.warning(
                    f"MemoBrainMemory: {label} attempt {attempt+1}/{self._max_retries} failed: {exc}"
                )
                if attempt < self._max_retries - 1:
                    time.sleep(3)
                else:
                    return None
        return None

    def _get_memorize_sys_prompt(self) -> str:
        return (
            self.prompts.get("memory", {}).get("memorize_sys_prompt")
            or ""
        )

    def _get_flush_fold_sys_prompt(self) -> str:
        return (
            self.prompts.get("memory", {}).get("flush_fold_sys_prompt")
            or ""
        )

    @staticmethod
    def _to_jit_message(msg: Dict[str, Any]) -> Message:
        """Convert OpenAI-style `{role, content:str}` → JIT Message."""
        role = msg["role"]
        if role == "system":
            jit_role = MessageRole.SYSTEM
        elif role == "assistant":
            jit_role = MessageRole.ASSISTANT
        else:
            jit_role = MessageRole.USER
        content = msg.get("content", "")
        if isinstance(content, str):
            jit_content: Any = [{"type": "text", "text": content}]
        else:
            jit_content = content
        return Message(role=jit_role, content=jit_content)
