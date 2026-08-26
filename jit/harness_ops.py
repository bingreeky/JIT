"""Utilities for meta-agent harness generation/repair/validation."""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from scripts.tools.registry import ToolRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_ROOT / "scripts" / "workspaces" / "harness_workspace"
WORKSPACE_FILES = ["memory.py", "planning.py", "action.py", "tool_policy.py"]
WORKSPACE_PROMPT_FILE = "prompt.yaml"
# The seed bank the code reference mode samples from (see jit/meta_agent.py);
# the descriptions/ dir next to it feeds the default desc mode.
HARNESS_BANK_DIR = PROJECT_ROOT / "harness_factory" / "harnesses"

TAG_TO_FILE = {
    "PYTHON_MEMORY": "memory.py",
    "PYTHON_PLANNING": "planning.py",
    "PYTHON_ACTION": "action.py",
    "PYTHON_TOOL_POLICY": "tool_policy.py",
}
SECTION_TAG_TO_FILE = {
    **TAG_TO_FILE,
    "YAML": WORKSPACE_PROMPT_FILE,
}
FILE_TO_PART_NAME = {
    "memory.py": "memory",
    "planning.py": "planning",
    "action.py": "action",
    "tool_policy.py": "tool_policy",
    WORKSPACE_PROMPT_FILE: "yaml",
}

def _load_benchmark_adapter(name: str, **kwargs):
    """Load a benchmark adapter by name.

    The single source of truth for which benchmarks exist: every entry point
    resolves adapters through here, so a new benchmark is registered in
    exactly one place.
    """
    benchmarks = {
        "xbench": "benchmark.adapter.xbench.XBenchAdapter",
        "deepsearchqa": "benchmark.adapter.deepsearchqa.DeepSearchQAAdapter",
        "agentif_oneday": "benchmark.adapter.agentif_oneday.AgentIFOneDayAdapter",
        "officebench": "benchmark.adapter.officebench.OfficeBenchAdapter",
        "odysseybench": "benchmark.adapter.odysseybench.OdysseyBenchAdapter",
        "deepplanning_shopping": "benchmark.adapter.deepplanning.DeepPlanningShoppingAdapter",
        "deepplanning_travel": "benchmark.adapter.deepplanning.DeepPlanningTravelAdapter",
    }
    if name not in benchmarks:
        raise ValueError(f"Unknown benchmark '{name}'.")

    module_path, class_name = benchmarks[name].rsplit(".", 1)
    import importlib
    import inspect

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    init_sig = inspect.signature(cls.__init__)
    params = init_sig.parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_var_kwargs:
        return cls(**kwargs)

    accepted = {
        pname
        for pname, p in params.items()
        if pname != "self"
        and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**filtered_kwargs)


def _parse_harness_response(
    text: str,
) -> Dict[str, Any]:
    """Parse model output and return only five harness parts."""
    def _extract_tagged_block(text: str, tag: str) -> Optional[str]:
        """Extract the LAST block between <<<TAG>>> and <<<END_TAG>>>.

        Take the last match, not the first. The meta-model is a thinking model:
        its reasoning trace first sketches a <<<TAG>>>...<<<END_TAG>>> skeleton
        (with literal ``...`` placeholders) before writing the real harness, so
        the genuine block is always the one that appears last. Matching the last
        occurrence lets us serve the model WITHOUT ``--reasoning-parser qwen3``
        (which would otherwise split the answer at an in-code ``</think>`` and
        drop the leading files) and parse the full raw output directly.
        """
        pattern = re.compile(rf"<<<{re.escape(tag)}>>>(.*?)<<<END_{re.escape(tag)}>>>", re.DOTALL)
        matches = list(pattern.finditer(text))
        if not matches:
            return None
        block = matches[-1].group(1).strip("\n")
        return f"{block}\n" if block.strip() else None
    
    sections: Dict[str, str] = {}
    for tag_name, filename in SECTION_TAG_TO_FILE.items():
        block = _extract_tagged_block(text, tag_name)
        if block:
            sections[filename] = block
    return sections

def _make_reference_txt(reference_harnesses: List[str]) -> str:
    """Render seed harnesses as reference material for the generation prompt.

    One block per harness: the prose from its ``description.yaml`` followed by
    all five source files wrapped in the same ``<<<TAG>>>`` protocol the model
    must emit, so reference material and expected output share one format.
    Only the code reference mode uses this; the default desc mode renders a
    no-code catalogue instead (see jit/meta_agent.py).
    """
    if not reference_harnesses:
        return "N/A"

    harness_texts: List[str] = []
    for harness_name in reference_harnesses:
        harness_dir = HARNESS_BANK_DIR / harness_name
        if not harness_dir.is_dir():
            continue

        parts: List[str] = [f"### Harness: {harness_name}"]
        has_any_content = False
        description_path = harness_dir / "description.yaml"

        if description_path.exists():
            try:
                description_data = yaml.safe_load(
                    description_path.read_text(encoding="utf-8")
                ) or {}
                description = description_data.get("description")
                if description:
                    parts.append(f"Description: {str(description).strip()}")
            except Exception:
                pass

        for tag_name, filename in SECTION_TAG_TO_FILE.items():
            file_path = harness_dir / filename
            if not file_path.exists():
                continue

            content = file_path.read_text(encoding="utf-8").rstrip("\n")
            parts.append(f"<<<{tag_name}>>>\n{content}\n<<<END_{tag_name}>>>")
            has_any_content = True

        if has_any_content:
            harness_texts.append("\n\n".join(parts))

    return "\n\n".join(harness_texts) if harness_texts else "N/A"


def _format_tool_info(name: str, tool: Any) -> str:
    """Serialize one tool into a concise text line for prompt injection."""
    description = getattr(tool, "description", "") or "No description provided."
    inputs = getattr(tool, "inputs", {}) or {}
    if not inputs:
        return f"{name}: {description}"

    arg_parts: List[str] = []
    for arg_name, arg_meta in inputs.items():
        arg_type = str(arg_meta.get("type", "any"))
        is_optional = bool(arg_meta.get("nullable", False))
        optional_suffix = " optional" if is_optional else ""
        arg_desc = str(arg_meta.get("description", "")).strip()
        if arg_desc:
            arg_parts.append(
                f"{arg_name}<{arg_type}{optional_suffix}>: {arg_desc}"
            )
        else:
            arg_parts.append(f"{arg_name}<{arg_type}{optional_suffix}>")

    return f"{name}: {description} Inputs: {'; '.join(arg_parts)}"


def _build_tools_info(
    tool_names: List[str],
    config: Dict[str, Any],
    model: Any,
    benchmark_adapter: Any = None,
    item: Optional[Dict[str, Any]] = None,
) -> str:
    """Build tool description text for both static and per-task dynamic tools."""
    ordered_names: List[str] = []
    for name in tool_names:
        if name not in ordered_names:
            ordered_names.append(name)

    if benchmark_adapter and hasattr(benchmark_adapter, "get_tools"):
        try:
            benchmark_tool_names = benchmark_adapter.get_tools() or []
            for name in benchmark_tool_names:
                if name not in ordered_names:
                    ordered_names.append(name)
        except Exception:
            pass

    registry = ToolRegistry()
    if ordered_names:
        registry.register_defaults(ordered_names, model=model, config=config)

    if benchmark_adapter and item is not None and hasattr(benchmark_adapter, "get_task_tools"):
        try:
            task_tools = benchmark_adapter.get_task_tools(item) or {}
            if task_tools:
                registry.register_batch(task_tools)
                for name in task_tools:
                    if name not in ordered_names:
                        ordered_names.append(name)
        except Exception:
            pass

    lines: List[str] = []
    for name in ordered_names:
        try:
            tool = registry.get(name)
            lines.append(_format_tool_info(name, tool))
        except Exception:
            continue
    return "\n".join(lines) if lines else "N/A"
