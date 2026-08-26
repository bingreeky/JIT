"""
Low-level token counting utility for the modular agent framework.

Provides a singleton-ish TokenCounter that uses tiktoken to estimate
token counts for messages and text. This is a fundamental kernel service
used by memory modules (e.g., ReSumMemory) and recorded in StepRecord
for trajectory analysis.

Uses cl100k_base encoding (GPT-4 / GPT-3.5-turbo family) as the default,
which also provides a reasonable approximation for other models.
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Module-level lazy singleton
_encoder = None


def _get_encoder():
    """Lazy-load tiktoken encoder (singleton)."""
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning(
                "tiktoken not installed; token counting will use char/4 estimate. "
                "Install with: pip install tiktoken"
            )
    return _encoder


def count_tokens_text(text: str) -> int:
    """Count tokens in a plain text string.

    Falls back to len(text)//4 if tiktoken is unavailable.
    """
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return len(text) // 4


def count_tokens_messages(messages: List[Union[Dict, Any]]) -> int:
    """Count tokens across a list of chat messages.

    Each message dict should have 'role' and 'content' keys.
    Content can be a string or a list of {"type":"text","text":"..."} dicts.

    Accounts for per-message overhead (~4 tokens each for role/separator tokens).
    """
    enc = _get_encoder()
    total = 0

    for msg in messages:
        # Per-message overhead (role, separators)
        total += 4

        if isinstance(msg, dict):
            content = msg.get("content", "")
        elif hasattr(msg, "content"):
            content = getattr(msg, "content", "")
        else:
            content = str(msg)

        # Content can be string or list of content blocks
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            text = "\n".join(parts)
        else:
            text = str(content)

        if enc is not None:
            total += len(enc.encode(text))
        else:
            total += len(text) // 4

    # Final assistant reply priming
    total += 2
    return total
