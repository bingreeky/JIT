"""Shared evaluation engine.

Everything under this package is used by BOTH runners:

  * ``scripts/run_jit.py``        -- the JIT meta-agent best-of-N pipeline
                                     (generate N harnesses -> select -> execute).
  * ``scripts/run_seed_harness.py``  -- run one hand-written seed harness.

Both paths share the same benchmark registry (``benchmark.registry``), the same
config wiring, the same per-unit runner (resume, artifacts, thread pool) and the
same score aggregation, so their ``summary.json`` files are directly comparable.
"""

__all__ = ["config", "metrics", "runner"]
