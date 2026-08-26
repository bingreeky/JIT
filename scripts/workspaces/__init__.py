"""Execution workspaces -- transient, one per (case, rollout).

The runner installs the harness it is about to execute into
``scripts/workspaces/<workspace_name>/`` (five files: memory.py, planning.py,
action.py, tool_policy.py, prompt.yaml) and imports it from there, whether the
harness was written by the meta model or copied from
``harness_factory/harnesses/``. Directories here are run artifacts and are
removed when the unit finishes; nothing but this file is committed.
"""
