"""
Benchmark adapter base class.

Each benchmark defines how to load data, format tasks, provide tools, and evaluate.
"""

from abc import ABC, abstractmethod
import os
import shutil
from typing import Any, Dict, List


class BenchmarkAdapter(ABC):
    """Adapts a specific benchmark's data format and evaluation."""

    @abstractmethod
    def load_dataset(self, path: str) -> List[dict]:
        """Load dataset items. Each item must have at least 'question' and 'answer'."""
        ...

    @abstractmethod
    def format_task(self, item: dict) -> str:
        """Convert a dataset item into the task string for the agent."""
        ...

    @abstractmethod
    def get_tools(self) -> List[str]:
        """Return list of tool names needed for this benchmark."""
        ...

    @abstractmethod
    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> dict:
        """Evaluate a single prediction. Returns {"score": float, ...}.

        kwargs may include benchmark-specific fields like 'question',
        'answer_type', etc. that some judges need for evaluation.
        """
        ...

    def get_file_description(self, item: dict) -> str:
        """For multimodal benchmarks: describe attached files."""
        return ""

    def _reset_workspace(self, workspace: str) -> None:
        """Ensure a task workspace exists and is empty."""
        if not workspace:
            return
        if os.path.isdir(workspace):
            shutil.rmtree(workspace)
        os.makedirs(workspace, exist_ok=True)
