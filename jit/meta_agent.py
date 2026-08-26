"""Deterministic meta-agent loop: generate -> validate -> repair(retry)."""

from __future__ import annotations

import importlib.resources
import json
import os
import random
import re
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import StrictUndefined, Template
import yaml

from scripts.kernel.token_counter import count_tokens_text
from scripts.kernel.runtime import AgentRuntime, ModelCallBudgetExceeded
from scripts.models.base import MessageRole
from scripts.models.openai_server import OpenAIServerModel
from scripts.tools.base import Tool

from .schemas import (
    MetaAgentRequest,
    MetaAgentResult,
    ValidationRecord,
)
from .harness_ops import (
    HARNESS_BANK_DIR,
    WORKSPACE_DIR,
    WORKSPACE_FILES,
    WORKSPACE_PROMPT_FILE,
    _build_tools_info,
    _make_reference_txt,
    _parse_harness_response,
)


def _load_meta_prompts() -> Dict[str, str]:
    """Load meta-agent prompts from prompt.yaml in this package."""
    prompt_path = importlib.resources.files(__package__).joinpath("prompt.yaml")
    prompt = yaml.safe_load(prompt_path.read_text(encoding="utf-8")) or {}
    if not isinstance(prompt, dict):
        raise RuntimeError(f"Prompt config for {__package__} must be a mapping.")
    return prompt


PROMPTS = _load_meta_prompts()
VALIDATION_RUNTIME_TIMEOUT_SECONDS = 60 * 60

# The originality/innovation wording of prompt.yaml swapped for fit-to-task
# wording -- byte-identical port of build_merged_desc.TRANSFORMS, the list the
# SFT dataset sft_train/data_merged_desc was built with. Each rule must fire
# exactly once or we raise: a miss means prompt.yaml drifted from what the model
# was trained on.
_DESC_DEL_MARK = "Your design should go beyond simple variants of the basic ReAct architecture"
_DESC_TRANSFORMS = [
    ("replace_str",
     "design a **fully original and diverse** agent framework",
     "design an agent framework"),
    ("delete", _DESC_DEL_MARK),
    ("delete", "We encourage you to introduce innovations across all four core modules"),
    ("replace_line",
     "the Action module is a highly flexible space for innovation",
     "  In particular, the Action module is a highly flexible space: you may invoke "
     "LLM-based processes to support advanced functionalities, such as Decompose tasks "
     "and invoke sub-agents or performing memory compression. Please fully exercise "
     "your capabilities by combining LLMs with other tools to build a task-adaptive "
     "action flow that fits the task at hand."),
    ("replace_line",
     "we expect your Action module design to be original and uniquely tailored",
     "  Importantly, we expect your Action module design to be tailored to the task at hand."),
    ("delete", "Please place particular emphasis on the innovations in your Action flow"),
    ("replace_str",
     "If any innovation in your design relies on LLM assistance",
     "If any mechanism in your design relies on LLM assistance"),
    ("replace_line",
     "demonstrates clear originality",
     "  After completing your design, ensure that it satisfies the task requirements. "
     "You should explicitly state in your reasoning how your design aligns with the "
     "task’s needs."),
]


# The default rendering ("desc" reference mode): the "### 3. Agent harness
# examples:" section of prompt.yaml is replaced by a NO-code catalog of
# natural-language design descriptions for all 11 reference harnesses, the
# section header is renamed to "### 3. Reference harness designs:", and the
# transforms above are applied. No harness source code is shown to the model.
# Any drift (missing anchor, non-unique transform target) raises.
# Ships with the repo: harness_factory/descriptions/*.md (one per harness);
# point the catalog elsewhere with META_REF_DESC_DIR.
_DESC_DIR_DEFAULT = str(
    Path(__file__).resolve().parents[1] / "harness_factory" / "descriptions"
)
_DESC_HARNESS_ORDER = [
    "plan_and_execute",
    "flash_searcher",
    "agentfold",
    "resum",
    "hiagent",
    "memobrain",
    "deepagent",
    "gam",
    "roma",
    "aggagent",
    "oagent",
]
_DESC_CATALOG_INTRO = """\
Below are design descriptions of existing agent harnesses. NO code is given: \
each entry describes the harness's overall architecture and how each of its \
components (memory.py, planning.py, action.py, tool_policy.py, prompt.yaml) is \
implemented. Use them as design references -- you may adopt, combine, or adapt \
any of the described mechanisms, or design different ones, whichever best fits \
the task at hand. You must write all code yourself from scratch, strictly \
following the shared data structures and output requirements defined above.
"""
_DESC_OLD_HEADER = "### 3. Agent harness examples:\n"
_DESC_NEW_HEADER = "### 3. Reference harness designs:\n"
_DESC_CATALOG_CACHE: str | None = None


def _render_desc_catalog() -> str:
    """Byte-identical port of build_merged_desc.render_desc_catalog."""
    global _DESC_CATALOG_CACHE
    if _DESC_CATALOG_CACHE is None:
        desc_dir = Path(os.environ.get("META_REF_DESC_DIR", _DESC_DIR_DEFAULT))
        parts = [_DESC_CATALOG_INTRO]
        for name in _DESC_HARNESS_ORDER:
            p = desc_dir / f"{name}.md"
            txt = p.read_text(encoding="utf-8").strip()
            if not txt.startswith(f"### Harness: {name}"):
                raise ValueError(f"{p} does not start with expected header")
            parts.append(txt)
        _DESC_CATALOG_CACHE = "\n\n".join(parts)
    return _DESC_CATALOG_CACHE


def _apply_desc_transforms(sp: str) -> str:
    """Byte-identical port of build_merged_desc.swap_refs' post-swap steps:
    rename the section header, then apply the shared instruction transforms."""
    if sp.count(_DESC_OLD_HEADER) != 1:
        raise ValueError(
            f"desc header target not unique: {_DESC_OLD_HEADER!r} x{sp.count(_DESC_OLD_HEADER)}"
        )
    sp = sp.replace(_DESC_OLD_HEADER, _DESC_NEW_HEADER)
    for rule in _DESC_TRANSFORMS:
        kind = rule[0]
        if kind == "replace_str":
            _, old, new = rule
            if sp.count(old) != 1:
                raise ValueError(f"replace_str target not unique: {old[:60]!r} x{sp.count(old)}")
            sp = sp.replace(old, new)
        else:
            mark = rule[1]
            lines = sp.splitlines(keepends=True)
            hits = [i for i, l in enumerate(lines) if mark in l]
            if len(hits) != 1:
                raise ValueError(f"line target not unique: {mark[:60]!r} x{len(hits)}")
            if kind == "delete":
                del lines[hits[0]]
            else:
                nl = "\n" if lines[hits[0]].endswith("\n") else ""
                lines[hits[0]] = rule[2] + nl
            sp = "".join(lines)
    return sp


# --------------------------------------------------------------------------- #
# Reference mode: which reference material the generation prompt carries
# --------------------------------------------------------------------------- #
# "desc" (default) renders the no-code catalog above. "code" instead shows the
# full source of three (REFERENCE_K_DEFAULT) reference harnesses drawn at random.
# In code mode the desc transforms are deliberately NOT applied: the
# prompt keeps prompt.yaml's original "### 3. Agent harness examples:" header
# and its originality/innovation wording, which is what pairs with code
# references.
#
# Code mode is primarily meant for teacher models and data generation. It grows
# the reference block from ~13k to ~25-46k tokens depending on
# which three are drawn, which still fits the 163840-token window
# scripts/serve_meta_model.sh serves alongside a 64k generation.
#
# Off by default. Turn it on for a run with ``--harness-refs code``
# (scripts/run_jit.py), a ``meta_references:`` block in the benchmark YAML, or
# ``JIT_META_REF_CODE=1`` when driving jit/ as a library. Precedence:
# config < env < CLI (the CLI writes the config block).
REFERENCE_MODE_DESC = "desc"
REFERENCE_MODE_CODE = "code"
REFERENCE_MODES = (REFERENCE_MODE_DESC, REFERENCE_MODE_CODE)
REFERENCE_K_DEFAULT = 3


def resolve_reference_mode(config: Dict[str, Any]) -> str:
    """Resolve the reference mode for one run: config, else env, else desc."""
    mode = str(((config or {}).get("meta_references") or {}).get("mode", "") or "").strip().lower()
    if not mode:
        mode = (
            REFERENCE_MODE_CODE
            if os.environ.get("JIT_META_REF_CODE", "") == "1"
            else REFERENCE_MODE_DESC
        )
    if mode not in REFERENCE_MODES:
        raise ValueError(
            f"Unknown reference mode {mode!r}; expected one of {list(REFERENCE_MODES)}"
        )
    return mode


def resolve_reference_k(config: Dict[str, Any]) -> int:
    """How many seed harnesses code mode shows. Config, else env, else 3."""
    raw = ((config or {}).get("meta_references") or {}).get("k", "")
    if str(raw).strip() == "":
        raw = os.environ.get("JIT_META_REF_CODE_K", "")
    if str(raw).strip() == "":
        return REFERENCE_K_DEFAULT
    k = int(raw)
    if k <= 0:
        raise ValueError(f"meta_references.k must be positive, got {k}")
    return k


def _pick_reference_harnesses(k: int) -> List[str]:
    """Draw k seed harnesses to show as source code.

    The pool is the same eleven references covered by the description catalog.
    Unseeded by default, so every case and rollout draws its own references; set
    JIT_META_REF_SEED for a fixed draw.
    """
    pool = [name for name in _DESC_HARNESS_ORDER if (HARNESS_BANK_DIR / name).is_dir()]
    if not pool:
        raise ValueError(f"No reference harnesses found under {HARNESS_BANK_DIR}")
    seed = os.environ.get("JIT_META_REF_SEED", "").strip()
    rng = random.Random(seed) if seed else random
    return rng.sample(pool, k=min(k, len(pool)))


def _normalize_tool_set(tools: List[str]) -> List[str]:
    """Deduplicate tools while preserving original order."""
    seen = set()
    normalized: List[str] = []
    for tool_name in tools:
        key = str(tool_name).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _extract_model_call_usage(model: Any) -> Dict[str, int]:
    """Read last model call token usage from model state."""
    token_counts = {}
    if hasattr(model, "get_token_counts"):
        try:
            token_counts = model.get_token_counts() or {}
        except Exception:
            token_counts = {}

    input_token_count = int(token_counts.get("input_token_count", 0) or 0)
    output_token_count = int(token_counts.get("output_token_count", 0) or 0)
    return {
        "input_token_count": input_token_count,
        "output_token_count": output_token_count,
        "total_token_count": input_token_count + output_token_count,
    }


def _response_truncated(response: Any, output_token_count: int = 0, max_tokens: int = 0) -> bool:
    """Detect whether the model's output was cut off at the token limit.

    Primary signal: the OpenAI/vLLM ``finish_reason == "length"`` on the raw
    response. Fallback (when finish_reason is unavailable): the recorded output
    token count reached the configured ``max_tokens`` ceiling.
    """
    try:
        raw = getattr(response, "raw", None)
        choices = getattr(raw, "choices", None) or []
        if choices:
            finish_reason = getattr(choices[0], "finish_reason", None)
            if finish_reason == "length":
                return True
            if finish_reason in ("stop", "tool_calls"):
                return False
    except Exception:
        pass
    if max_tokens and output_token_count and output_token_count >= max_tokens:
        return True
    return False


def _full_response_text(response: Any) -> str:
    """Reconstruct the model's complete output (reasoning + content).

    The meta-model is served behind vLLM ``--reasoning-parser qwen3``, which
    splits the assistant turn at the FIRST ``</think>`` token: everything before
    it goes into ``reasoning_content`` and only the remainder lands in
    ``content``. The harness the model writes legitimately contains literal
    ``</think>`` substrings (e.g. ``re.sub(r'<think>.*?</think>', '', x, ...)``
    used to strip the executor's reasoning). When the model does not emit its own
    closing ``</think>`` first, the parser splits *inside* the generated code,
    dropping the natural-language analysis and the opening ``<<<PYTHON_MEMORY>>>``
    tag into ``reasoning_content`` — which we would otherwise discard, producing
    the spurious "did not output all required files" failure.

    Stitch the two halves back together, re-inserting the single ``</think>``
    delimiter the parser consumed, so the harness parser always sees the model's
    true, complete output regardless of where the split happened. When there is no
    reasoning split, ``content`` already holds the full output and is returned as-is.
    """
    content = response.content or ""
    reasoning = getattr(response, "reasoning_content", "") or ""
    if not reasoning:
        return content
    return f"{reasoning}</think>{content}"


def _parse_expert_vote(text: str) -> tuple[bool, bool]:
    """Extract one review-panel expert's pass/fail vote from its response.

    Returns (passed, parsed). Prefer matches after the last </think> (the
    reasoning may rehearse trial votes) and take the LAST occurrence. Fall back
    to a FINAL VOTE: PASS/FAIL line, then a Chinese 合格 / 不合格 ("qualified" /
    "not qualified") verdict on the last non-empty line -- a reviewer answering in
    Chinese is matched by those two literals, nothing else reads them. Unparseable responses fail open (passed=True) so a sloppy
    expert never burns regeneration budget; parsed=False flags them in the
    trajectory.
    """
    tail = text.rsplit("</think>", 1)[-1]
    for haystack in (tail, text):
        matches = re.findall(r'"vote"\s*:\s*"(pass|fail)"', haystack, re.IGNORECASE)
        if matches:
            return matches[-1].lower() == "pass", True
        matches = re.findall(
            r"FINAL\s*(?:VOTE|VERDICT)\s*[::]\s*(PASS|FAIL)", haystack, re.IGNORECASE
        )
        if matches:
            return matches[-1].upper() == "PASS", True
        lines = [line.strip() for line in haystack.strip().splitlines() if line.strip()]
        if lines:
            if "不合格" in lines[-1]:
                return False, True
            if "合格" in lines[-1]:
                return True, True
    return True, False


_REVIEW_RISK_KEYS = (
    "phase_transition_risk",
    "tool_access_risk",
    "termination_risk",
    "syntax_risk",
)


def _parse_review_verdict(text: str) -> tuple[bool, bool]:
    """Extract the single-reviewer's qualified/unqualified verdict (legacy mode).

    Returns (qualified, parsed). Strict rule, enforced in code: if ANY of the
    four risk ratings is "low" or "high", the harness is unqualified; all-"none"
    qualifies. Prefer matches after the last </think> and take the LAST
    occurrence of each field. Fall back to the "qualified" boolean, then a plain
    FINAL VERDICT: PASS/FAIL line. Unparseable responses fail open
    (qualified=True); parsed=False flags them in the trajectory.
    """
    tail = text.rsplit("</think>", 1)[-1]
    for haystack in (tail, text):
        ratings = {}
        for key, level in re.findall(
            r'"(%s)"\s*:\s*"(none|low|high)"' % "|".join(_REVIEW_RISK_KEYS),
            haystack,
            re.IGNORECASE,
        ):
            ratings[key.lower()] = level.lower()  # last occurrence wins
        if ratings:
            return all(level == "none" for level in ratings.values()), True
        matches = re.findall(r'"qualified"\s*:\s*(true|false)', haystack, re.IGNORECASE)
        if matches:
            return matches[-1].lower() == "true", True
        matches = re.findall(r"FINAL\s*VERDICT\s*[::]\s*(PASS|FAIL)", haystack, re.IGNORECASE)
        if matches:
            return matches[-1].upper() == "PASS", True
    return True, False


def _condense_expert_feedback(feedback: str, limit: int = 500) -> str:
    """One-line digest of an expert's visible analysis for the stage log.

    Drops the trailing single-line JSON vote (redundant with the votes summary
    in the verdict line), collapses whitespace, and truncates to `limit` chars.
    """
    lines = feedback.strip().splitlines()
    if lines and re.search(r'"vote"\s*:', lines[-1]):
        lines = lines[:-1]
    text = " ".join(" ".join(lines).split())
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text or "(no visible analysis)"


def _run_with_timeout(func, timeout_seconds: int, *args, **kwargs):
    """Run a callable with timeout and raise TimeoutError('Run timed out') when exceeded."""
    if timeout_seconds <= 0:
        return func(*args, **kwargs)

    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.getsignal(signal.SIGALRM)

        def _timeout_handler(signum, frame):
            raise TimeoutError("Run timed out")

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
        try:
            return func(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=float(timeout_seconds))
    except FuturesTimeoutError as exc:
        future.cancel()
        raise TimeoutError("Run timed out") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def populate_template(template: str, variables: Dict[str, Any]) -> str:
    """Render a Jinja2 template string with variables."""
    compiled_template = Template(template, undefined=StrictUndefined)
    return compiled_template.render(**variables)


class MetaReActAgent:
    """Meta-agent with fixed orchestration (no tool-selection ReAct)."""

    _REPAIR_ERROR_MESSAGE_TOKEN_BUDGET = 6000
    _REPAIR_USER_PROMPT_TOKEN_BUDGET = 24000
    _REPAIR_LAST_STEP_PREVIEW_CHARS = 2200
    _REPAIR_RECENT_STEP_PREVIEW_CHARS = 900
    _REPAIR_MEMORY_MESSAGE_PREVIEW_CHARS = 700

    def __init__(self, model, config: Dict[str, Any], workspace_name: str = "harness_workspace"):
        model_cfg = model
        # Use explicit meta-model credentials only; fallback to empty strings.
        api_key = str(model_cfg.get("api_key", "") or "")
        api_base = str(model_cfg.get("api_base", "") or "")
        self.model = OpenAIServerModel(
            model_id=model_cfg.get("model_id", "gpt-4o-mini"),
            api_base=api_base,
            api_key=api_key,
            temperature=model_cfg.get("temperature"),
            max_tokens=model_cfg.get("max_tokens", 64000),
        )
        self.config = config
        self.workspace_name = workspace_name
        self.workspace_dir = WORKSPACE_DIR.parent / workspace_name
        self._last_task_description = ""
        self._repair_history: List[str] = []
        self.reference_mode = resolve_reference_mode(config)
        # Drawn once per agent -- i.e. once per (case, rollout) -- so the first
        # generation, any review regeneration and any repair regeneration all
        # see the same references.
        self.reference_harnesses: List[str] = (
            _pick_reference_harnesses(resolve_reference_k(config))
            if self.reference_mode == REFERENCE_MODE_CODE
            else []
        )
    
    def _make_current_harness_txt(self) -> str:
        """Serialize current harness_workspace files in tagged text format."""
        tag_to_file = {
            "PYTHON_MEMORY": "memory.py",
            "PYTHON_PLANNING": "planning.py",
            "PYTHON_ACTION": "action.py",
            "PYTHON_TOOL_POLICY": "tool_policy.py",
            "YAML": WORKSPACE_PROMPT_FILE,
        }

        parts: List[str] = [f"### Harness: {self.workspace_name}"]
        has_any_content = False

        for tag_name, filename in tag_to_file.items():
            file_path = self.workspace_dir / filename
            if not file_path.exists():
                continue
            content = file_path.read_text(encoding="utf-8").rstrip("\n")
            parts.append(f"<<<{tag_name}>>>\n{content}\n<<<END_{tag_name}>>>")
            has_any_content = True

        return "\n\n".join(parts) if has_any_content else "N/A"

    @staticmethod
    def _truncate_text(value: Any, max_chars: int) -> str:
        text = str(value or "").strip()
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "... [truncated]"

    @classmethod
    def _stringify_for_prompt(cls, value: Any, max_chars: int) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return cls._truncate_text(value, max_chars)
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
        return cls._truncate_text(text, max_chars)

    @classmethod
    def _extract_message_text(cls, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                    elif "text" in block:
                        parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(p for p in parts if p)
        if isinstance(content, dict):
            if content.get("type") == "text":
                return str(content.get("text", ""))
            return cls._stringify_for_prompt(content, 2000)
        return str(content)

    @classmethod
    def _summarize_tool_calls(cls, tool_calls: Any) -> List[Dict[str, Any]]:
        if not isinstance(tool_calls, list):
            return []
        summarized: List[Dict[str, Any]] = []
        for call in tool_calls[:5]:
            if not isinstance(call, dict):
                summarized.append(
                    {"name": "unknown", "arguments_preview": cls._truncate_text(call, 240)}
                )
                continue
            summarized.append(
                {
                    "name": str(call.get("name", "") or "unknown"),
                    "arguments_preview": cls._stringify_for_prompt(
                        call.get("arguments", {}), 350
                    ),
                }
            )
        return summarized

    def _summarize_memory_from_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        model_input = step.get("model_input_messages")
        if not isinstance(model_input, list) or not model_input:
            return {"message_count": 0, "tail_messages": []}

        tail = model_input[-6:]
        compact_tail: List[Dict[str, str]] = []
        for message in tail:
            if not isinstance(message, dict):
                compact_tail.append(
                    {
                        "role": "unknown",
                        "content_preview": self._truncate_text(
                            message, self._REPAIR_MEMORY_MESSAGE_PREVIEW_CHARS
                        ),
                    }
                )
                continue
            role = str(message.get("role", "unknown"))
            content_text = self._extract_message_text(message.get("content"))
            compact_tail.append(
                {
                    "role": role,
                    "content_preview": self._truncate_text(
                        content_text, self._REPAIR_MEMORY_MESSAGE_PREVIEW_CHARS
                    ),
                }
            )
        return {
            "message_count": len(model_input),
            "tail_messages": compact_tail,
        }

    def _summarize_step(
        self,
        step: Dict[str, Any],
        observation_limit: int,
        output_limit: int,
    ) -> Dict[str, Any]:
        return {
            "step_number": int(step.get("step_number", -1) or -1),
            "tool_calls": self._summarize_tool_calls(step.get("tool_calls")),
            "observations_preview": self._truncate_text(
                step.get("observations", ""), observation_limit
            ),
            "action_output_preview": self._stringify_for_prompt(
                step.get("action_output"), output_limit
            ),
            "error": self._truncate_text(step.get("error", ""), 800),
            "token_usage": {
                "input": int(step.get("input_token_count", 0) or 0),
                "output": int(step.get("output_token_count", 0) or 0),
                "total": int(step.get("total_token_count", 0) or 0),
            },
        }

    def _extract_step_trajectory(self, failed_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        step_like = [
            case
            for case in (failed_cases or [])
            if isinstance(case, dict) and "step_number" in case
        ]
        if step_like:
            return step_like

        if failed_cases and isinstance(failed_cases[0], dict):
            nested = failed_cases[0].get("trajectory")
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return []

    @staticmethod
    def _truncate_to_token_budget(text: str, token_budget: int) -> str:
        clean = str(text or "").strip()
        if token_budget <= 0:
            return ""
        if count_tokens_text(clean) <= token_budget:
            return clean

        suffix = "\n\n[truncated for repair token budget]"
        lo, hi = 0, len(clean)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = clean[:mid].rstrip() + suffix
            if count_tokens_text(candidate) <= token_budget:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return best or suffix.strip()

    def _build_repair_error_message(
        self,
        error_summary: str,
        failed_cases: List[Dict[str, Any]],
        evaluation_result: Dict[str, Any],
        validation_error: str,
    ) -> str:
        steps = self._extract_step_trajectory(failed_cases)
        last_step = steps[-1] if steps else {}
        recent_steps = steps[-3:] if steps else []

        compact_eval = {}
        if isinstance(evaluation_result, dict):
            for key in ("score", "is_pass", "reason", "error", "details"):
                if key in evaluation_result:
                    compact_eval[key] = self._stringify_for_prompt(
                        evaluation_result.get(key), 1200
                    )

        failure_type = "validation_failed"
        if error_summary == "harness_generation_incomplete":
            failure_type = "harness_generation_incomplete"
        elif error_summary not in {"validation_failed", "harness_generation_incomplete"}:
            failure_type = "validation_exception"

        payload: Dict[str, Any] = {
            "failure_info": {
                "failure_type": failure_type,
                "error_summary": self._truncate_text(error_summary, 1200),
                "validation_error": self._truncate_text(validation_error, 2000),
                "total_steps": len(steps),
                "evaluation_summary": compact_eval,
            },
            "memory_summary": self._summarize_memory_from_step(last_step) if last_step else {},
            "last_step_snapshot": (
                self._summarize_step(
                    last_step,
                    observation_limit=self._REPAIR_LAST_STEP_PREVIEW_CHARS,
                    output_limit=1200,
                )
                if last_step
                else {}
            ),
            "recent_steps": [
                self._summarize_step(
                    step,
                    observation_limit=self._REPAIR_RECENT_STEP_PREVIEW_CHARS,
                    output_limit=500,
                )
                for step in recent_steps
            ],
        }

        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if count_tokens_text(text) <= self._REPAIR_ERROR_MESSAGE_TOKEN_BUDGET:
            return text

        payload["recent_steps"] = payload["recent_steps"][-1:] if payload["recent_steps"] else []
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if count_tokens_text(text) <= self._REPAIR_ERROR_MESSAGE_TOKEN_BUDGET:
            return text

        payload["memory_summary"] = {"note": "memory summary trimmed to fit token budget"}
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if count_tokens_text(text) <= self._REPAIR_ERROR_MESSAGE_TOKEN_BUDGET:
            return text

        return self._truncate_to_token_budget(
            text, self._REPAIR_ERROR_MESSAGE_TOKEN_BUDGET
        )

    def _generate_harness(
        self,
        task_description: str,
        tools: List[str],
        benchmark_adapter: Any = None,
        item: Dict[str, Any] | None = None,
        review_feedback: str = "",
    ) -> Dict[str, Any]:
        generate_system_prompt = PROMPTS.get("generate_system_prompt", "")
        generate_user_prompt_tpl = PROMPTS.get("generate_user_prompt", "")

        code_refs = self.reference_mode == REFERENCE_MODE_CODE
        references = (
            _make_reference_txt(self.reference_harnesses)
            if code_refs
            else _render_desc_catalog()
        )
        tools_info = _build_tools_info(
            tools,
            self.config,
            self.model,
            benchmark_adapter=benchmark_adapter,
            item=item,
        )

        generate_harness_prompt = populate_template(
            generate_system_prompt,
            {
                "EXAMPLES_PLACEHOLDER": references,
                "TOOLS_INFO_PLACEHOLDER": tools_info,
            },
        )
        if not code_refs:
            # Code mode keeps prompt.yaml's own wording; see REFERENCE_MODE_CODE.
            generate_harness_prompt = _apply_desc_transforms(generate_harness_prompt)
        generate_user_prompt = populate_template(
            generate_user_prompt_tpl,
            {"task_description": task_description},
        )
        if review_feedback:
            generate_user_prompt += (
                "\n\nIMPORTANT: A previous harness you generated for this task was "
                "rejected by an expert review panel (each expert audits one failure "
                "mode: getting stuck in action loops, tool invocation/parsing/"
                "exposure defects, early or max-step-only termination, prompt "
                "parsing/rendering crashes, over-complex architecture for a simple "
                "task). Generate a NEW harness that fixes every issue raised by "
                "the failing experts below:\n"
                + self._truncate_text(review_feedback, 6000)
            )

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [{"type": "text", "text": generate_harness_prompt}],
            },
            {
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": generate_user_prompt}],
            },
        ]

        response = self.model(messages)
        call_usage = _extract_model_call_usage(self.model)
        llm_response = _full_response_text(response)
        sections = _parse_harness_response(llm_response)

        updated_files: List[str] = []
        for filename, content in sections.items():
            if not isinstance(content, str) or not content.strip():
                continue
            target = self.workspace_dir / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            updated_files.append(filename)

        expected = set(WORKSPACE_FILES + [WORKSPACE_PROMPT_FILE])
        output_token_count = int(call_usage.get("output_token_count", 0))
        model_max_tokens = int((getattr(self.model, "kwargs", {}) or {}).get("max_tokens", 0) or 0)
        return {
            "success": expected.issubset(set(updated_files)),
            "truncated": _response_truncated(response, output_token_count, model_max_tokens),
            "llm_prompt": {
                "system_prompt": generate_harness_prompt,
                "user_prompt": generate_user_prompt,
            },
            "llm_response": llm_response,
            "input_token_count": int(call_usage.get("input_token_count", 0)),
            "output_token_count": output_token_count,
            "total_token_count": int(call_usage.get("total_token_count", 0)),
        }

    def _static_harness_checks(self) -> str:
        """Deterministic syntax report fed to the reviewer as ground truth."""
        findings: List[str] = []
        for filename in WORKSPACE_FILES:
            file_path = self.workspace_dir / filename
            if not file_path.exists():
                findings.append(f"{filename}: FILE MISSING")
                continue
            try:
                compile(file_path.read_text(encoding="utf-8"), filename, "exec")
            except SyntaxError as exc:
                findings.append(f"{filename}: SyntaxError: {exc}")
        prompt_path = self.workspace_dir / WORKSPACE_PROMPT_FILE
        if not prompt_path.exists():
            findings.append(f"{WORKSPACE_PROMPT_FILE}: FILE MISSING")
        else:
            try:
                parsed = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
                if not isinstance(parsed, dict):
                    findings.append(
                        f"{WORKSPACE_PROMPT_FILE}: parsed to "
                        f"{type(parsed).__name__}, expected a mapping"
                    )
                else:
                    required = ("system_prompt", "planning", "summary", "final_answer", "step")
                    missing = [key for key in required if key not in parsed]
                    if missing:
                        findings.append(
                            f"{WORKSPACE_PROMPT_FILE}: missing required keys: {missing}"
                        )
            except yaml.YAMLError as exc:
                findings.append(f"{WORKSPACE_PROMPT_FILE}: YAML parse error: {exc}")
        if findings:
            return "\n".join(findings)
        return "No static errors detected (all Python modules compile; prompt.yaml parses with required keys)."

    def _review_harness_expert(
        self,
        expert: Dict[str, str],
        system_tpl: str,
        user_tpl: str,
        task_description: str,
        tools_info: str,
        harness_text: str,
        static_check_report: str,
    ) -> Dict[str, Any]:
        """One panel expert: single-charter review, thinking first, then a vote."""
        expert_key = str(expert["key"])
        expert_name = str(expert.get("name", expert_key))
        system_prompt = populate_template(
            system_tpl,
            {
                "expert_key": expert_key,
                "expert_name": expert_name,
                "expert_charter": str(expert["charter"]),
            },
        )
        user_prompt = populate_template(
            user_tpl,
            {
                "expert_name": expert_name,
                "task_description": task_description,
                "tools_info": tools_info,
                "harness_text": harness_text,
                "static_check_report": static_check_report,
            },
        )

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]

        response = self.model(messages)
        call_usage = _extract_model_call_usage(self.model)
        llm_response = _full_response_text(response)
        vote_pass, vote_parsed = _parse_expert_vote(llm_response)
        # Feedback for a retry = the expert's visible analysis (post-reasoning
        # content), which quotes the concrete defects to fix.
        feedback = llm_response.rsplit("</think>", 1)[-1].strip()
        return {
            "expert": expert_key,
            "expert_name": expert_name,
            "vote_pass": vote_pass,
            "vote_parsed": vote_parsed,
            "feedback": feedback,
            "llm_prompt": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
            "llm_response": llm_response,
            "input_token_count": int(call_usage.get("input_token_count", 0)),
            "output_token_count": int(call_usage.get("output_token_count", 0)),
            "total_token_count": int(call_usage.get("total_token_count", 0)),
        }

    def _review_harness(
        self,
        task_description: str,
        tools: List[str],
        benchmark_adapter: Any = None,
        item: Dict[str, Any] | None = None,
        stage_log=None,
    ) -> Dict[str, Any]:
        """Dispatch harness review by mode (config meta_review.mode).

        "panel" (default): a 5-expert voting panel (any fail -> regenerate).
        "single": the legacy single reviewer that rates four risk dimensions and
        qualifies only when all are "none".
        Both return the same shape: {qualified, verdict_parsed, votes,
        feedback, expert_events} — so the caller's loop is mode-agnostic.
        """
        mode = str((self.config.get("meta_review") or {}).get("mode", "panel")).lower()
        if mode == "single":
            return self._review_harness_single(
                task_description, tools, benchmark_adapter=benchmark_adapter, item=item
            )
        return self._review_harness_panel(
            task_description,
            tools,
            benchmark_adapter=benchmark_adapter,
            item=item,
            stage_log=stage_log,
        )

    def _review_harness_single(
        self,
        task_description: str,
        tools: List[str],
        benchmark_adapter: Any = None,
        item: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Legacy single-reviewer gate: one call rates four risk dimensions.

        Review axes: phase-transition dead ends, tool accessibility (incl.
        final_answer reachability), early/never termination, and syntax. The
        reviewer outputs its thinking then a single-line JSON verdict; strict
        rule (any dimension low/high -> unqualified) enforced in code. Returns
        the panel-compatible shape with a single synthetic "reviewer" event.
        """
        review_system_prompt = PROMPTS.get("review_system_prompt", "")
        review_user_prompt_tpl = PROMPTS.get("review_user_prompt", "")
        if not review_system_prompt or not review_user_prompt_tpl:
            raise RuntimeError(
                "review_system_prompt / review_user_prompt missing from prompt.yaml"
            )

        tools_info = _build_tools_info(
            tools,
            self.config,
            self.model,
            benchmark_adapter=benchmark_adapter,
            item=item,
        )
        review_user_prompt = populate_template(
            review_user_prompt_tpl,
            {
                "task_description": task_description,
                "tools_info": tools_info,
                "harness_text": self._make_current_harness_txt(),
                "static_check_report": self._static_harness_checks(),
            },
        )

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [{"type": "text", "text": review_system_prompt}],
            },
            {
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": review_user_prompt}],
            },
        ]

        response = self.model(messages)
        call_usage = _extract_model_call_usage(self.model)
        llm_response = _full_response_text(response)
        qualified, verdict_parsed = _parse_review_verdict(llm_response)
        feedback = llm_response.rsplit("</think>", 1)[-1].strip()
        event = {
            "expert": "reviewer",
            "expert_name": "Reviewer",
            "vote_pass": qualified,
            "vote_parsed": verdict_parsed,
            "feedback": feedback,
            "llm_prompt": {
                "system_prompt": review_system_prompt,
                "user_prompt": review_user_prompt,
            },
            "llm_response": llm_response,
            "input_token_count": int(call_usage.get("input_token_count", 0)),
            "output_token_count": int(call_usage.get("output_token_count", 0)),
            "total_token_count": int(call_usage.get("total_token_count", 0)),
        }
        return {
            "qualified": qualified,
            "verdict_parsed": verdict_parsed,
            "votes": {"reviewer": "pass" if qualified else "fail"},
            "feedback": "" if qualified else feedback,
            "expert_events": [event],
        }

    def _review_harness_panel(
        self,
        task_description: str,
        tools: List[str],
        benchmark_adapter: Any = None,
        item: Dict[str, Any] | None = None,
        stage_log=None,
    ) -> Dict[str, Any]:
        """Gate the just-generated harness through a panel of single-charter experts.

        Each expert audits exactly one failure mode (action loops, tool
        invocation/parsing/exposure, early or max-step-only termination, prompt
        parsing, over-complex architecture), outputs its thinking, then votes
        pass/fail. The harness is qualified ONLY when every voting expert
        passes; a single fail vote triggers regeneration. An expert whose call
        errors counts as pass (fail-open, consistent with the parse fallback);
        if EVERY expert call errors, raise so the caller accepts unreviewed.
        """
        log = stage_log or (lambda message: None)
        system_tpl = PROMPTS.get("review_expert_system_prompt", "")
        user_tpl = PROMPTS.get("review_expert_user_prompt", "")
        experts = [
            expert
            for expert in (PROMPTS.get("review_experts") or [])
            if isinstance(expert, dict) and expert.get("key") and expert.get("charter")
        ]
        if not experts or not system_tpl or not user_tpl:
            raise RuntimeError(
                "review_experts / review_expert_*_prompt missing from prompt.yaml"
            )

        tools_info = _build_tools_info(
            tools,
            self.config,
            self.model,
            benchmark_adapter=benchmark_adapter,
            item=item,
        )
        harness_text = self._make_current_harness_txt()
        static_check_report = self._static_harness_checks()

        expert_events: List[Dict[str, Any]] = []
        failing_feedback: List[str] = []
        votes: Dict[str, str] = {}
        all_parsed = True
        for expert in experts:
            expert_key = str(expert["key"])
            try:
                event = self._review_harness_expert(
                    expert,
                    system_tpl,
                    user_tpl,
                    task_description,
                    tools_info,
                    harness_text,
                    static_check_report,
                )
            except Exception as exc:  # noqa: BLE001 - one expert must never kill the panel
                log(
                    f"Review expert '{expert_key}' call failed ({exc}); "
                    "counting as pass (fail-open)"
                )
                votes[expert_key] = "error"
                continue
            expert_events.append(event)
            votes[expert_key] = "pass" if event["vote_pass"] else "fail"
            if not event["vote_parsed"]:
                all_parsed = False
            if not event["vote_pass"]:
                failing_feedback.append(
                    f"[{event['expert_name']}] "
                    + self._truncate_text(str(event.get("feedback", "")), 1200)
                )

        if not expert_events:
            raise RuntimeError("every review expert call failed")

        return {
            "qualified": not failing_feedback,
            "verdict_parsed": all_parsed,
            "votes": votes,
            "feedback": "\n\n".join(failing_feedback),
            "expert_events": expert_events,
        }

    def _generate_harness_reviewed(
        self,
        task_description: str,
        tools: List[str],
        benchmark_adapter: Any = None,
        item: Dict[str, Any] | None = None,
        stage: str = "generate",
        stage_log=None,
    ) -> tuple[List[Dict[str, Any]], bool, int]:
        """Generate a harness, then gate it through the expert-panel review loop.

        A panel of single-charter experts each votes pass/fail; while ANY expert
        votes fail (and retry budget remains), regenerate from scratch with the
        failing experts' feedback attached. Returns (events, harness_ok,
        review_regenerations); every event carries a "stage" key
        ("generate"/"regenerate", "review" — one per expert, "review_regenerate").
        """
        log = stage_log or (lambda message: None)
        events: List[Dict[str, Any]] = []

        generation_event = self._generate_harness(
            task_description=task_description,
            tools=tools,
            benchmark_adapter=benchmark_adapter,
            item=item,
        )
        generation_event["stage"] = stage
        events.append(generation_event)
        harness_ok = bool(generation_event.get("success", False))

        review_cfg = self.config.get("meta_review") or {}
        if not review_cfg.get("enabled", False):
            return events, harness_ok, 0

        max_review_retries = max(0, int(review_cfg.get("max_retries", 2)))
        review_regenerations = 0
        while harness_ok:
            try:
                panel = self._review_harness(
                    task_description,
                    tools,
                    benchmark_adapter=benchmark_adapter,
                    item=item,
                    stage_log=log,
                )
            except Exception as exc:  # noqa: BLE001 - review must never kill the case
                log(f"Harness review panel failed ({exc}); accepting harness unreviewed")
                break
            for expert_event in panel["expert_events"]:
                expert_event["stage"] = "review"
                events.append(expert_event)
                log(
                    "Harness review [%s] vote=%s: %s"
                    % (
                        expert_event.get("expert", "?"),
                        "pass" if expert_event.get("vote_pass", True) else "fail",
                        _condense_expert_feedback(str(expert_event.get("feedback", ""))),
                    )
                )
            votes_txt = ", ".join(f"{key}={vote}" for key, vote in panel["votes"].items())
            if panel.get("qualified", True):
                log(
                    f"Harness review panel verdict: QUALIFIED ({votes_txt})"
                    + ("" if panel.get("verdict_parsed") else " (some votes unparsed, fail-open)")
                )
                break
            if review_regenerations >= max_review_retries:
                log(
                    f"Harness review panel verdict: NOT QUALIFIED ({votes_txt}), but review "
                    f"retry budget ({max_review_retries}) exhausted; keeping last harness"
                )
                break
            review_regenerations += 1
            log(
                f"Harness review panel verdict: NOT QUALIFIED ({votes_txt}); regenerating "
                f"with the failing experts' feedback "
                f"(review regeneration #{review_regenerations}/{max_review_retries})"
            )
            generation_event = self._generate_harness(
                task_description=task_description,
                tools=tools,
                benchmark_adapter=benchmark_adapter,
                item=item,
                review_feedback=str(panel.get("feedback", "")),
            )
            generation_event["stage"] = "review_regenerate"
            events.append(generation_event)
            harness_ok = bool(generation_event.get("success", False))

        return events, harness_ok, review_regenerations

    def _append_meta_events(
        self,
        events: List[Dict[str, Any]],
        trajectory: List[Dict[str, Any]],
    ) -> None:
        """Append meta-model call events to the trajectory."""
        for event in events:
            input_token_count = int(event.get("input_token_count", 0))
            output_token_count = int(event.get("output_token_count", 0))
            entry = {
                "prompt": event.get("llm_prompt", {}),
                "response": event.get("llm_response", ""),
                "input_token_count": input_token_count,
                "output_token_count": output_token_count,
                "total_token_count": int(event.get("total_token_count", 0)),
                "stage": event.get("stage", ""),
            }
            if event.get("stage") == "review":
                entry["review_expert"] = str(event.get("expert", ""))
                entry["review_vote"] = "pass" if event.get("vote_pass", True) else "fail"
                entry["review_vote_parsed"] = bool(event.get("vote_parsed", False))
            trajectory.append(entry)

    def _validate_harness(
        self,
        benchmark_adapter,
        item: Dict[str, Any],
        max_steps: int,
    ) -> Dict[str, Any]:
        """Run harness_workspace on one benchmark item and return detailed validation."""
        run_cfg = json.loads(json.dumps(self.config))
        run_cfg["harness"] = self.workspace_name
        run_cfg.setdefault("execution", {})["max_steps"] = int(max_steps)
        run_cfg.setdefault("execution", {})["model_call_budget"] = max(0, 3*int(max_steps))

        try:
            runtime = AgentRuntime(run_cfg)
            if hasattr(benchmark_adapter, "get_runtime_task"):
                task = benchmark_adapter.get_runtime_task(item)
            else:
                task = benchmark_adapter.format_task(item)
            if hasattr(benchmark_adapter, "get_workspace"):
                workspace = benchmark_adapter.get_workspace(item)
                if workspace:
                    runtime.set_tool_workspace(workspace)

            task_tools = None
            if hasattr(benchmark_adapter, "get_task_tools"):
                task_tools = benchmark_adapter.get_task_tools(item)

            Tool.reset_total_latency()
            run_started = time.perf_counter()
            result = _run_with_timeout(
                runtime.run,
                VALIDATION_RUNTIME_TIMEOUT_SECONDS,
                task,
                task_tools=task_tools,
            )
            run_latency = time.perf_counter() - run_started
            tool_latency = Tool.get_total_latency()
            api_latency = max(0.0, run_latency - tool_latency)
            trajectory = [step.full_dict() for step in result.trajectory]
            score = benchmark_adapter.evaluate(
                str(result.answer),
                item.get("answer", ""),
                question=item.get("question", item.get("problem", "")),
                answer_type=item.get("answer_type", ""),
                item=item,
                trajectory=[step.dict() for step in result.trajectory],
            )
            score_value = float(score.get("score", 0.0))
            is_pass = bool(score.get("is_pass", score_value > 0.0))
            result_meta = getattr(result, "metadata", {}) or {}
            input_token_count = int(result_meta.get("input_token_count", 0) or 0)
            output_token_count = int(result_meta.get("output_token_count", 0) or 0)
            total_token_count = int(result_meta.get("total_token_count", input_token_count + output_token_count) or 0)
            return {
                "passed": is_pass,
                "steps": len(trajectory),
                "api_latency": api_latency,
                "tool_latency": tool_latency,
                "input_token_count": input_token_count,
                "output_token_count": output_token_count,
                "total_token_count": total_token_count,
                "evaluation": score,
                "trajectory": trajectory,
            }
        except ModelCallBudgetExceeded:
            return {
                "passed": False,
                "steps": 0,
                "api_latency": 0.0,
                "tool_latency": Tool.get_total_latency(),
                "input_token_count": 0,
                "output_token_count": 0,
                "total_token_count": 0,
                "error": "Budget exceeded",
                "trajectory": [],
            }
        except Exception as exc:
            tb_exc = traceback.TracebackException.from_exception(exc)
            last_frame = tb_exc.stack[-1] if tb_exc.stack else None
            error_file = str(last_frame.filename) if last_frame else ""
            error_line = int(last_frame.lineno) if last_frame else 0
            error_function = str(last_frame.name) if last_frame else ""
            error_traceback = "".join(tb_exc.format()).strip()
            error_with_location = str(exc)
            if error_file and error_line:
                if error_function:
                    error_with_location = f"{str(exc)} (at {error_file}:{error_line} in {error_function})"
                else:
                    error_with_location = f"{str(exc)} (at {error_file}:{error_line})"
            if error_traceback:
                error_with_location = f"{error_with_location}\n{error_traceback}"

            return {
                "passed": False,
                "steps": 0,
                "api_latency": 0.0,
                "tool_latency": 0.0,
                "input_token_count": 0,
                "output_token_count": 0,
                "total_token_count": 0,
                "error": error_with_location,
                "trajectory": [],
            }

    def _repair_harness(
        self,
        error_summary: str,
        failed_cases: List[Dict[str, Any]],
        evaluation_result: Dict[str, Any] | None = None,
        validation_error: str = "",
    ) -> Dict[str, Any]:
        """Repair harness files based on failure summary and failed trajectories."""
        repair_system_prompt = PROMPTS.get("repair_system_prompt", "")
        repair_user_prompt_tpl = PROMPTS.get("repair_user_prompt", "")

        error_message = self._build_repair_error_message(
            error_summary=error_summary,
            failed_cases=failed_cases,
            evaluation_result=evaluation_result or {},
            validation_error=validation_error,
        )

        repair_user_prompt = populate_template(
            repair_user_prompt_tpl,
            {
                "task_description": self._last_task_description,
                "error_code_description": self._make_current_harness_txt(),
                "error_message": error_message,
                "history_mistakes": "\n".join(self._repair_history[:]),
            },
        )
        if count_tokens_text(repair_user_prompt) > self._REPAIR_USER_PROMPT_TOKEN_BUDGET:
            compressed_error = self._truncate_to_token_budget(
                error_message,
                max(1000, self._REPAIR_ERROR_MESSAGE_TOKEN_BUDGET // 2),
            )
            repair_user_prompt = populate_template(
                repair_user_prompt_tpl,
                {
                    "task_description": self._last_task_description,
                    "error_code_description": self._make_current_harness_txt(),
                    "error_message": compressed_error,
                    "history_mistakes": "\n".join(self._repair_history[:]),
                },
            )

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [{"type": "text", "text": repair_system_prompt}],
            },
            {
                "role": MessageRole.USER,
                "content": [{"type": "text", "text": repair_user_prompt}],
            },
        ]

        response = self.model(messages)
        call_usage = _extract_model_call_usage(self.model)
        llm_response = _full_response_text(response)
        sections = _parse_harness_response(llm_response)

        updated_files: List[str] = []
        for filename, content in sections.items():
            if not isinstance(content, str) or not content.strip():
                continue
            target = self.workspace_dir / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            updated_files.append(filename)

        if updated_files:
            round_error = error_summary
            if failed_cases:
                round_error = str(failed_cases[0].get("error", "")).strip() or error_summary

            updated_files_sorted = sorted(updated_files)
            history_parts: List[str] = [f"Current round error: {round_error}"]
            for filename in updated_files_sorted:
                code = str(sections.get(filename, "")).rstrip("\n")
                history_parts.append(f"Updated file this round: {filename}")
                history_parts.append(
                    f"Updated file code:\n<<<FILE_CODE>>>\n{code}\n<<<END_FILE_CODE>>>"
                )
            self._repair_history.append("\n".join(history_parts))
        
        output_token_count = int(call_usage.get("output_token_count", 0))
        model_max_tokens = int((getattr(self.model, "kwargs", {}) or {}).get("max_tokens", 0) or 0)
        return {
            "success": bool(updated_files),
            "truncated": _response_truncated(response, output_token_count, model_max_tokens),
            "llm_prompt": {
                "system_prompt": repair_system_prompt,
                "user_prompt": repair_user_prompt,
            },
            "llm_response": llm_response,
            "input_token_count": int(call_usage.get("input_token_count", 0)),
            "output_token_count": output_token_count,
            "total_token_count": int(call_usage.get("total_token_count", 0)),
        }

    def run(self, request: MetaAgentRequest) -> MetaAgentResult:
        """Run deterministic generate-once -> validate(single item) -> repair loop."""
        def _stage_log(message: str) -> None:
            """Print progress logs to stderr so stdout artifacts stay clean."""
            print(f"[meta-agent] {message}", file=sys.stderr, flush=True)

        self._repair_history = []
        _stage_log("Start run")

        runtime_model_name = str(self.config.get("model", {}).get("model_id", "") or "")
        meta_agent_model_name = str(getattr(self.model, "model_id", "") or "")
        tool_set = _normalize_tool_set(request.tools)

        task_description = str(request.benchmark_adapter.format_task(request.item)).strip()
        self._last_task_description = task_description

        benchmark_name = request.benchmark_config.get("name", "")

        validation_records: List[ValidationRecord] = []
        generation_success = False
        meta_agent_trajectory: List[Dict[str, Any]] = []
        number_of_regenerations = 0
        number_of_review_regenerations = 0

        total_rounds = max(1, int(request.max_repairs) + 1)
        _stage_log(f"Prepared task, total rounds: {total_rounds}")

        if request.preset_harness_dir:
            # Best-of-n selection pipelines: install a previously generated
            # harness instead of generating one; validate/repair runs as usual.
            preset = Path(request.preset_harness_dir)
            expected = WORKSPACE_FILES + [WORKSPACE_PROMPT_FILE]
            copied = 0
            for fname in expected:
                src = preset / fname
                if src.is_file():
                    (self.workspace_dir / fname).write_text(
                        src.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                    copied += 1
            harness_ok = copied == len(expected)
            generation_events = []
            _stage_log(
                f"[Round 1/{total_rounds}] Installed preset harness "
                f"({copied}/{len(expected)} files) from {preset}"
            )
        else:
            _stage_log("[Round 1/{0}] Generating harness".format(total_rounds))
            generation_events, harness_ok, review_regens = self._generate_harness_reviewed(
                task_description=task_description,
                tools=request.tools,
                benchmark_adapter=request.benchmark_adapter,
                item=request.item,
                stage="generate",
                stage_log=_stage_log,
            )
            number_of_review_regenerations += review_regens
            self._append_meta_events(generation_events, meta_agent_trajectory)
            _stage_log(
                "[Round 1/{0}] Harness generation {1}".format(
                    total_rounds, "succeeded" if harness_ok else "failed"
                )
            )

        if request.generate_only:
            # Best-of-n selection pipelines: return right after generation.
            # generation_success here means "all harness files were emitted".
            _stage_log("generate_only=True: skipping validation/repair loop")
            return MetaAgentResult(
                generation_success=harness_ok,
                model=meta_agent_model_name,
                validation_records=[],
                meta_agent_trajectory=meta_agent_trajectory,
                number_of_regenerations=0,
                number_of_review_regenerations=number_of_review_regenerations,
                reference_mode=self.reference_mode,
                reference_harnesses=list(self.reference_harnesses),
            )

        for round_idx in range(total_rounds):
            round_no = round_idx + 1
            validation_error = ""

            if not harness_ok:
                failed = {
                    "id": request.item.get("question_id", request.item.get("id", "")),
                    "passed": False,
                    "steps": 0,
                    "error": "harness generation did not output all required files",
                    "trajectory": [],
                }
                validation_record = ValidationRecord(
                    passed=False,
                    benchmark=benchmark_name,
                    error="harness generation did not output all required files",
                    evaluation_result={},
                    api_latency_sec=0.0,
                    tool_latency_sec=0.0,
                    input_token_count=0,
                    output_token_count=0,
                    total_token_count=0,
                    model=runtime_model_name,
                    tool_set=tool_set,
                    steps_used=0,
                    trajectories=[failed],
                )
            else:
                # Validate only the single provided benchmark item.
                _stage_log(f"[Round {round_no}/{total_rounds}] Validating harness")
                one_result = self._validate_harness(
                    request.benchmark_adapter,
                    request.item,
                    request.max_steps,
                )
                validation_error = str(one_result.get("error", "")).strip()
                validation_record = ValidationRecord(
                    passed=bool(one_result.get("passed", False)),
                    benchmark=benchmark_name,
                    error=str(one_result.get("error", "") or ""),
                    evaluation_result=one_result.get("evaluation", {}),
                    api_latency_sec=float(one_result.get("api_latency", 0.0)),
                    tool_latency_sec=float(one_result.get("tool_latency", 0.0)),
                    input_token_count=int(one_result.get("input_token_count", 0)),
                    output_token_count=int(one_result.get("output_token_count", 0)),
                    total_token_count=int(one_result.get("total_token_count", 0)),
                    model=runtime_model_name,
                    tool_set=tool_set,
                    steps_used=int(one_result.get("steps", 0)),
                    trajectories=one_result.get("trajectory", []),
                )

            validation_records.append(validation_record)
            _stage_log(
                f"[Round {round_no}/{total_rounds}] Validation "
                f"{'passed' if validation_record.passed else 'failed'}"
            )
            if validation_record.error:
                _stage_log(
                    f"[Round {round_no}/{total_rounds}] Validation error: "
                    f"{validation_record.error}"
                )
            if validation_record.passed:
                generation_success = True
                _stage_log(f"[Round {round_no}/{total_rounds}] Finished successfully")
                break

            if round_idx >= total_rounds - 1:
                _stage_log(f"[Round {round_no}/{total_rounds}] Reached max rounds, stopping")
                break

            if request.repair_only_on_error and harness_ok and not validation_error:
                _stage_log(
                    f"[Round {round_no}/{total_rounds}] Validation did not pass but no error "
                    f"occurred; repair_only_on_error=True, skipping repair"
                )
                break

            error_summary = "validation_failed"
            if not harness_ok:
                error_summary = "harness_generation_incomplete"
            elif validation_error:
                error_summary = validation_error

            failed_cases = validation_record.trajectories or []
            if not failed_cases and error_summary == "validation_failed":
                failed_cases = [{"error": "validation failed but no failed case detail provided"}]

            if request.regenerate_on_error:
                # Regenerate a fresh harness from scratch instead of repairing
                # the failed one. _generate_harness overwrites all workspace files.
                number_of_regenerations += 1
                _stage_log(
                    f"[Round {round_no}/{total_rounds}] Error encountered; regenerating "
                    f"harness from scratch (regeneration #{number_of_regenerations})"
                )
                action_events, harness_ok, review_regens = self._generate_harness_reviewed(
                    task_description=task_description,
                    tools=request.tools,
                    benchmark_adapter=request.benchmark_adapter,
                    item=request.item,
                    stage="regenerate",
                    stage_log=_stage_log,
                )
                number_of_review_regenerations += review_regens
                action_stage = "regenerate"
            else:
                _stage_log(f"[Round {round_no}/{total_rounds}] Repairing harness")
                action_event = self._repair_harness(
                    error_summary=error_summary,
                    failed_cases=failed_cases,
                    evaluation_result=validation_record.evaluation_result,
                    validation_error=validation_record.error,
                )
                action_event["stage"] = "repair"
                action_events = [action_event]
                harness_ok = bool(action_event.get("success", False))
                action_stage = "repair"

            self._append_meta_events(action_events, meta_agent_trajectory)
            if harness_ok:
                _stage_log(f"[Round {round_no}/{total_rounds}] {action_stage} completed")
            else:
                # The action did not produce a complete harness (commonly the
                # output was truncated at the token limit before all files were
                # emitted). Do NOT stop here: fall through so the loop uses the
                # remaining repair/regeneration rounds (max_repairs) instead of
                # giving up after a single failed attempt.
                last_generation_event = next(
                    (e for e in reversed(action_events) if e.get("stage") != "review"),
                    {},
                )
                truncated = bool(last_generation_event.get("truncated", False))
                reason = (
                    "output truncated at token limit"
                    if truncated
                    else "did not output all required files"
                )
                _stage_log(
                    f"[Round {round_no}/{total_rounds}] {action_stage} incomplete "
                    f"({reason}); continuing to next round to use remaining repairs"
                )

        _stage_log(
            "Run completed: "
            + ("generation_success=True" if generation_success else "generation_success=False")
        )

        return MetaAgentResult(
            generation_success=generation_success,
            model=meta_agent_model_name,
            validation_records=validation_records,
            meta_agent_trajectory=meta_agent_trajectory,
            number_of_regenerations=number_of_regenerations,
            number_of_review_regenerations=number_of_review_regenerations,
            reference_mode=self.reference_mode,
            reference_harnesses=list(self.reference_harnesses),
        )
