"""
XBench benchmark adapter.

XBench uses an encrypted CSV format (XOR cipher + base64 encoding).
Tasks are multimodal research questions, and evaluation uses a
Chinese-language LLM judge.

CSV fields (after decryption):
    id, prompt, answer

The dataset is encrypted with a canary key.
"""

import base64
import csv
import io
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .base import BenchmarkAdapter
from .xbench_eval import evaluate_xbench_prediction

logger = logging.getLogger(__name__)


def xor_decrypt(encrypted_b64: str, key: str) -> str:
    """Decrypt XOR-encrypted base64-encoded text."""
    try:
        encrypted_bytes = base64.b64decode(encrypted_b64)
        key_bytes = key.encode("utf-8")
        decrypted = bytes(
            b ^ key_bytes[i % len(key_bytes)]
            for i, b in enumerate(encrypted_bytes)
        )
        return decrypted.decode("utf-8")
    except Exception as e:
        logger.warning(f"XOR decryption failed: {e}")
        return encrypted_b64


class XBenchAdapter(BenchmarkAdapter):
    """Adapter for the XBench benchmark.

    Args:
        judge_model: Model name for the LLM judge (Chinese judge).
        canary: The XOR decryption key for the encrypted CSV.
    """

    def __init__(
        self,
        judge_model: str = "qwen/qwen3.5-122b-a10b",
        judge_api_base: str = "",
        judge_api_key: str = "",
        canary: Optional[str] = None,
    ):
        self._judge_model = judge_model
        self._judge_api_base = judge_api_base
        self._judge_api_key = judge_api_key
        self._canary = canary

    # ── Loading ───────────────────────────────────────────────────────

    def load_dataset(self, path: str) -> List[dict]:
        """Load XBench from encrypted CSV file.

        Tries to auto-detect the canary key from the CSV header or
        a companion file.
        """
        if os.path.isdir(path):
            # Find .csv file in directory
            csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {path}")
            path = os.path.join(path, csv_files[0])

        items: List[dict] = []
        canary = self._canary

        # Try to read canary from companion file
        if not canary:
            canary_file = os.path.join(os.path.dirname(path), "canary.txt")
            if os.path.exists(canary_file):
                with open(canary_file, "r") as f:
                    canary = f.read().strip()

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = dict(row)

                # Auto-detect canary from first row if not set
                if not canary and "canary" in item:
                    canary = item["canary"]

                # Decrypt encrypted fields
                if canary:
                    if "prompt" in item:
                        item["prompt"] = xor_decrypt(item["prompt"], canary)
                    if "answer" in item:
                        item["answer"] = xor_decrypt(item["answer"], canary)

                # Normalise field names
                item["question"] = item.get("prompt", item.get("question", ""))
                if "answer" not in item:
                    item["answer"] = ""

                items.append(item)

        logger.info(
            f"Loaded {len(items)} XBench items from {path}"
            + (f" (canary: {canary[:8]}...)" if canary else " (no encryption)")
        )
        return items

    # ── Task formatting ───────────────────────────────────────────────

    def format_task(self, item: dict) -> str:
        """Format XBench task."""
        return item.get("question", item.get("prompt", ""))

    # ── Tools ─────────────────────────────────────────────────────────

    def get_tools(self) -> List[str]:
        """XBench uses multimodal tools like GAIA."""
        return [
            "web_search",
            "crawl_page",
            "inspect_file_as_text",
            "inspect_file_as_image",
            "final_answer",
        ]

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> dict:
        """Chinese-language LLM judge evaluation.

        Uses a two-phase approach:
        1. Direct regex match (skip LLM if exact match).
        2. LLM judge with Chinese prompt.
        """
        question = kwargs.get("question", "")
        return evaluate_xbench_prediction(
            prediction,
            ground_truth,
            question=question,
            judge_model=self._judge_model,
            judge_api_base=self._judge_api_base,
            judge_api_key=self._judge_api_key,
        )
