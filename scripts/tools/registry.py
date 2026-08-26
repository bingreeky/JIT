"""
Tool registry for managing available tools.
"""

import json
import logging
from copy import deepcopy
from typing import Dict, List, Optional

from .base import Tool, FinalAnswerTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing tools by name."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def get_all(self) -> Dict[str, Tool]:
        return dict(self._tools)

    def register_batch(self, tools: Dict[str, Tool]) -> None:
        """Register multiple tools at once (used for per-task domain tools)."""
        for name, tool in tools.items():
            self._tools[name] = tool

    def unregister_batch(self, names) -> None:
        """Unregister multiple tools by name (cleanup after per-task tools)."""
        for name in names:
            self._tools.pop(name, None)

    def format_schemas(self, tools: Optional[Dict[str, Tool]] = None) -> str:
        """Generate JSON schema string for tools (like Flash-Searcher reformulate_tool_fuctions)."""
        tools = tools or self._tools
        schemas = []
        for tool in tools.values():
            required = []
            properties = deepcopy(tool.inputs)
            for key, value in properties.items():
                if value["type"] == "any":
                    value["type"] = "string"
                if not ("nullable" in value and value["nullable"]):
                    required.append(key)
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "properties": properties,
                    "required": required,
                }
            })
        return json.dumps(schemas, indent=2, ensure_ascii=False)

    def register_defaults(self, tool_names: List[str], model=None, config: Optional[dict] = None) -> None:
        """Register standard tools by name.

        Args:
            tool_names: List of tool names to register.
            model: LLM callable (for tools that need summarization).
            config: Optional config dict (for sandbox paths, API keys, etc.).
        """
        from .search_tools import WebSearchTool, CrawlPageTool, WikiSearchTool
        from .code_tools import ExecuteCodeTool
        from .officebench_tools import OfficeBenchActionTool
        from .odysseybench_tools import OdysseyBenchActionTool
        
        TextInspectorTool = None
        VisualInspectorTool = None
        mm_import_error: Optional[Exception] = None
        if any(name in {"inspect_file_as_text", "inspect_file_as_image"} for name in tool_names):
            try:
                from .mm_tools import TextInspectorTool as _TextInspectorTool, VisualInspectorTool as _VisualInspectorTool
                TextInspectorTool = _TextInspectorTool
                VisualInspectorTool = _VisualInspectorTool
            except Exception as exc:  # pragma: no cover - dependency/environment specific
                mm_import_error = exc

        import os

        cfg = config or {}
        sandbox_dir = cfg.get("sandbox_dir", "")
        conda_python = cfg.get("conda_python", "")

        def _raise_mm_dependency_error(tool_name: str):
            detail = f": {mm_import_error}" if mm_import_error else ""
            raise RuntimeError(
                f"Tool '{tool_name}' requires optional multimodal dependencies that are not available{detail}"
            )

        TOOL_MAP = {
            "web_search": lambda: WebSearchTool(),
            "crawl_page": lambda: CrawlPageTool(model=model),
            "wiki_search": lambda: WikiSearchTool(),
            "execute_code": lambda: ExecuteCodeTool(
                sandbox_dir=sandbox_dir,
                conda_python=conda_python,
                timeout=cfg.get("code_timeout", 120),
            ),
            "inspect_file_as_text": lambda: TextInspectorTool(model=model) if TextInspectorTool else _raise_mm_dependency_error("inspect_file_as_text"),
            "inspect_file_as_image": lambda: VisualInspectorTool(
                model=model,
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
            ) if VisualInspectorTool else _raise_mm_dependency_error("inspect_file_as_image"),
            "final_answer": lambda: FinalAnswerTool(),
            "officebench_action": lambda: OfficeBenchActionTool(model=model),
            "odysseybench_action": lambda: OdysseyBenchActionTool(model=model),
        }

        for name in tool_names:
            if name in TOOL_MAP:
                self.register(TOOL_MAP[name]())
            else:
                logger.warning(f"Unknown tool name: '{name}', skipping.")

        # Always register final_answer
        if "final_answer" not in self._tools:
            self.register(FinalAnswerTool())
