"""
Agent logger for runtime logging.
Adapted from Flash-Searcher FlashOAgents/monitoring.py
"""

import json
from enum import IntEnum
from typing import List, Optional

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text


YELLOW_HEX = "#d4b702"


class LogLevel(IntEnum):
    OFF = -1
    ERROR = 0
    INFO = 1
    DEBUG = 2


class AgentLogger:
    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level
        self.console = Console()

    def error(self, *args, **kwargs) -> None:
        self.log(*args, level=LogLevel.ERROR, **kwargs)

    def log(self, *args, level=LogLevel.INFO, **kwargs) -> None:
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        if level <= self.level:
            self.console.print(*args, **kwargs)

    def log_markdown(self, content: str, title: Optional[str] = None, level=LogLevel.INFO, style=YELLOW_HEX) -> None:
        content = str(content)
        markdown_content = Syntax(
            content, lexer="markdown", theme="github-dark", word_wrap=True,
        )
        if title:
            self.log(
                Group(
                    Rule("[bold italic]" + title, align="left", style=style),
                    markdown_content,
                ),
                level=level,
            )
        else:
            self.log(markdown_content, level=level)

    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        self.log(
            Rule("[bold]" + title, characters="━", style=YELLOW_HEX),
            level=LogLevel.INFO,
        )

    def log_task(self, content: str, subtitle: str, title: Optional[str] = None, level: int = LogLevel.INFO) -> None:
        self.log(
            Panel(
                f"\n[bold]{content}\n",
                title="[bold]New run" + (f" - {title}" if title else ""),
                subtitle=subtitle,
                border_style=YELLOW_HEX,
                subtitle_align="left",
            ),
            level=level,
        )
