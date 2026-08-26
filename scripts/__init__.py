"""Shared code and entry points.

  * ``kernel``      the agent runtime: harness loader, step loop, types.
  * ``models``      OpenAI-compatible chat clients (meta / exec / judge roles).
  * ``tools``       every tool a harness can call, plus the registry.
  * ``eval``        the shared evaluation engine (config, runner, metrics).
  * ``workspaces``  transient: the harness under execution is installed here.

The two runners are ``scripts/run_jit.py`` (JIT best-of-N) and
``scripts/run_seed_harness.py`` (one fixed seed harness).
"""
