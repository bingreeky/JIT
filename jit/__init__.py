"""JIT: the meta agent that writes a harness per task.

  * ``meta_agent``   the generate -> validate -> repair loop and its review panel.
  * ``prompt.yaml``  the generation / repair / review prompts.
  * ``harness_ops``  harness text parsing, tool-info rendering, adapter lookup.
  * ``schemas``      the request / result / validation dataclasses.
  * ``candidates``   rebuild each generated harness as a (prompt, completion) pair.
  * ``selector``     best-of-N: let the meta model pick its own favourite.
"""

from .meta_agent import MetaReActAgent
from .schemas import MetaAgentRequest

__all__ = ["MetaReActAgent", "MetaAgentRequest"]
