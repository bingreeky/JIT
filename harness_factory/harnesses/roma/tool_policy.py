"""
AllToolsPolicy: exposes all available tools at every step.
"""

import json
import logging
from copy import deepcopy
from typing import Any, Dict, Optional

from scripts.kernel.protocols import BaseToolPolicy
from scripts.kernel.types import MemoryView, PlanState, ToolSelection
from scripts.tools.skills import SkillRegistry


logger = logging.getLogger(__name__)


class ToolPolicyStrategy(BaseToolPolicy):
    """Always expose the full tool catalog and all loaded skills."""

    def __init__(self, prompts=None):
        self.prompts = prompts or {}

    def initialize(self, tool_catalog: Dict[str, Any], enable_skills: bool = False) -> None:
        self._catalog = tool_catalog
        self._schemas_json = self._format_schemas(tool_catalog)

        self._skills_prompt = ""
        if enable_skills:
            self._skill_registry = SkillRegistry()
            self._skill_registry.load_all()
            self._skills_prompt = self._skill_registry.format_skills_prompt()
            skill_count = len(self._skill_registry.get_all())
            if skill_count > 0:
                logger.info(f"AllToolsPolicy: loaded {skill_count} skill(s)")
        else:
            self._skill_registry = None

    def select_tools(
        self,
        task: str,
        step_number: int,
        memory_view: MemoryView,
        plan: Optional[PlanState] = None,
    ) -> ToolSelection:
        return ToolSelection(
            tools=self._catalog,
            tool_schemas_json=self._schemas_json,
            skills_prompt=self._skills_prompt,
        )

    def get_skills_prompt(self) -> str:
        return self._skills_prompt

    def _format_schemas(self, tools: Dict[str, Any]) -> str:
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
