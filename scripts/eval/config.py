"""Repo paths, `.env` loading, YAML config wiring.

The three model roles in this project are all reached over the **OpenAI
chat-completions protocol**, and each is configured independently:

    meta model  -- generates the harness (the JIT model, e.g. the ckpt-70 DPO
                   checkpoint served by vLLM/SGLang, or any hosted model).
    exec model  -- runs the generated harness's agent loop.
    judge model -- grades the produced artifacts, where the benchmark needs one.

Nothing here is vLLM-specific: a "meta model" is just a base URL, an API key
and a model id.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# One placeholder/.env implementation for the whole repo.
from scripts.env_config import (  # noqa: E402
    load_dotenv as _load_dotenv,
    load_yaml_config as _load_yaml_config,
    resolve_env_placeholders,
)

__all__ = [
    "REPO_ROOT",
    "load_dotenv",
    "resolve_env_placeholders",
    "load_yaml_config",
    "build_run_config",
    "make_meta_model_config",
]


def load_dotenv(path: Optional[os.PathLike] = None) -> None:
    """Load the repo's `.env` (or a specific file) into the environment."""
    _load_dotenv(path or REPO_ROOT / ".env")


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Read a benchmark YAML (relative paths resolve against the repo root)."""
    try:
        return _load_yaml_config(config_path, root=REPO_ROOT)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc


def build_run_config(
    config_path: str,
    *,
    meta_model: Optional[Dict[str, Any]] = None,
    exec_model: str = "",
    exec_base: str = "",
    exec_key: str = "",
    judge_model: str = "",
    judge_base: str = "",
    judge_key: str = "",
    dataset_path: str = "",
    max_steps: Optional[int] = None,
    harness_refs: str = "",
    harness_refs_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Load a benchmark YAML and apply the CLI overrides for all three roles.

    The YAML only ever describes the *execution* model (``model:``) and the
    *judge* (``benchmark.judge``). The meta model is injected here so the two
    can never be confused -- in particular so the meta model's ``max_tokens``
    (a whole harness, 64k) never leaks into the execution model and vice versa.
    """
    config = load_yaml_config(config_path)

    exec_cfg = dict(config.get("model", {}) or {})
    if exec_model:
        exec_cfg["model_id"] = exec_model
    if exec_base:
        exec_cfg["api_base"] = exec_base
    if exec_key:
        exec_cfg["api_key"] = exec_key
    config["model"] = exec_cfg

    if meta_model is not None:
        config["meta_model"] = dict(meta_model)

    benchmark_cfg = config.setdefault("benchmark", {})
    if dataset_path:
        benchmark_cfg["dataset_path"] = dataset_path
    if judge_model or judge_base or judge_key:
        judge_cfg = dict(benchmark_cfg.get("judge", {}) or {})
        if judge_model:
            judge_cfg["model_id"] = judge_model
        if judge_base:
            judge_cfg["api_base"] = judge_base
        if judge_key:
            judge_cfg["api_key"] = judge_key
        benchmark_cfg["judge"] = judge_cfg

    if max_steps is not None:
        config.setdefault("execution", {})["max_steps"] = int(max_steps)

    # Reference material in the generation prompt (jit/meta_agent.py). Only
    # written when the CLI asked for a mode, so a ``meta_references:`` block in
    # the YAML -- or JIT_META_REF_CODE -- still decides when the flag is absent.
    if harness_refs or harness_refs_k is not None:
        refs_cfg = dict(config.get("meta_references", {}) or {})
        if harness_refs:
            refs_cfg["mode"] = harness_refs
        if harness_refs_k is not None:
            refs_cfg["k"] = int(harness_refs_k)
        config["meta_references"] = refs_cfg

    return config


def make_meta_model_config(
    model_id: str,
    api_base: str,
    api_key: str,
    max_tokens: int = 64000,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """A meta-model block: any OpenAI-compatible chat-completions endpoint."""
    return {
        "model_id": model_id,
        "api_base": api_base,
        "api_key": api_key or "EMPTY",
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
