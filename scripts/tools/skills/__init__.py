"""
Skill Registry: loads and manages domain-knowledge skills (.md files).

Skills are stored as SKILL.md files under skills/<skill_name>/.
They provide domain expertise that gets injected into the agent's prompt,
adjacent to tool descriptions but conceptually distinct — skills tell the
agent *how* to approach problems, tools give it *what* to call.

Usage:
    registry = SkillRegistry()
    skills = registry.load_all()
    prompt_text = registry.format_skills_prompt()
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SkillInfo:
    """Metadata and content of a loaded skill."""
    name: str           # Skill directory name (e.g., "skill_office_process")
    description: str    # First paragraph of the SKILL.md (for display/logging)
    content: str        # Full markdown content


class SkillRegistry:
    """Discovers and loads skills from the skills/ directory.

    Each skill is a directory containing a SKILL.md file.
    """

    def __init__(self, skills_dir: Optional[str] = None):
        """
        Args:
            skills_dir: Root directory containing skill folders.
                        Defaults to the `skills/` dir next to this file.
        """
        self._skills_dir = skills_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
        )
        self._skills: Dict[str, SkillInfo] = {}

    def load_all(self) -> Dict[str, SkillInfo]:
        """Scan skills directory and load all SKILL.md files."""
        self._skills = {}

        if not os.path.isdir(self._skills_dir):
            logger.warning(f"Skills directory not found: {self._skills_dir}")
            return self._skills

        for entry in sorted(os.listdir(self._skills_dir)):
            skill_dir = os.path.join(self._skills_dir, entry)
            if not os.path.isdir(skill_dir):
                continue

            skill_file = os.path.join(skill_dir, "SKILL.md")
            if not os.path.isfile(skill_file):
                continue

            try:
                with open(skill_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if not content:
                    continue

                # Extract first paragraph as description
                paragraphs = content.split("\n\n")
                description = ""
                for para in paragraphs:
                    stripped = para.strip()
                    if stripped and not stripped.startswith("#"):
                        description = stripped[:200]
                        break

                skill = SkillInfo(
                    name=entry,
                    description=description,
                    content=content,
                )
                self._skills[entry] = skill
                logger.info(f"Loaded skill: {entry}")
            except Exception as e:
                logger.warning(f"Failed to load skill {entry}: {e}")

        return self._skills

    def get_all(self) -> Dict[str, SkillInfo]:
        """Return all loaded skills."""
        return dict(self._skills)

    def get(self, name: str) -> Optional[SkillInfo]:
        """Get a specific skill by name."""
        return self._skills.get(name)

    def format_skills_prompt(self) -> str:
        """Format all loaded skills into a prompt string for injection.

        Returns a markdown-formatted string with all skill contents,
        or empty string if no skills are loaded.
        """
        if not self._skills:
            return ""

        parts = []
        for name, skill in sorted(self._skills.items()):
            parts.append(skill.content)

        return "\n\n---\n\n".join(parts)
