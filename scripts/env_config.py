"""`.env` loading and `${VAR}` expansion for config files.

One implementation, used by every entry point in the repo (``scripts/run_jit.py``,
``scripts/run_seed_harness.py``, ``scripts.eval``) so a placeholder means the same
thing everywhere.

Supported forms::

    ${VAR}                 -> value, or "" when unset
    ${VAR:-fallback}       -> value, or the literal fallback when unset/empty
    ${VAR:-${OTHER}}       -> nested; resolved innermost-first

Expansion runs to a fixed point (bounded), so a fallback may itself contain a
placeholder -- which is how the benchmark YAMLs let a judge endpoint default to
the execution endpoint without repeating credentials.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

# The fallback body excludes braces so the INNERMOST placeholder matches first.
ENV_PLACEHOLDER_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^{}]*))?\}")

_MAX_EXPANSION_PASSES = 8


def load_dotenv(path: os.PathLike | str) -> None:
    """Load `KEY=value` lines from a .env file without clobbering real env vars."""
    env_path = Path(path)
    if env_path.is_dir():
        env_path = env_path / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _expand_str(text: str) -> str:
    for _ in range(_MAX_EXPANSION_PASSES):
        expanded = ENV_PLACEHOLDER_PATTERN.sub(
            lambda m: os.environ.get(m.group(1))
            or (m.group(2) if m.group(2) is not None else ""),
            text,
        )
        if expanded == text:
            return expanded
        text = expanded
    return text


def resolve_env_placeholders(value: Any) -> Any:
    """Recursively expand placeholders across a nested config structure."""
    if isinstance(value, str):
        return _expand_str(value)
    if isinstance(value, dict):
        return {k: resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_env_placeholders(v) for v in value]
    return value


def load_yaml_config(path: os.PathLike | str, root: Optional[os.PathLike | str] = None) -> dict:
    """Read a YAML config and expand its placeholders.

    A relative ``path`` is resolved against ``root`` when given, else against
    the current working directory.
    """
    import yaml

    config_path = Path(path)
    if not config_path.is_absolute() and root is not None:
        config_path = Path(root) / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        return resolve_env_placeholders(yaml.safe_load(fh) or {})
