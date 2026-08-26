"""What a benchmark is: its run configuration and its interfaces.

  * ``config/<name>.yaml``  what one run needs: exec model, tools, steps, judge.
  * ``registry.py``         what the runner needs on top of the YAML.
  * ``adapter/``            the code: one adapter + one vendored evaluator each.

The data these point at lives in ``dataset/``.
"""

from .adapter import BenchmarkAdapter

__all__ = ["BenchmarkAdapter"]
