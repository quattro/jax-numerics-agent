# Project Engineering Checklist

Use this for general software engineering hygiene around numerical/JAX projects.

## API stability
- [ ] Public API signatures and return structures are stable and documented.
- [ ] Public solver-style APIs return structured results (e.g., `Solution` with `value`, `result`, and optional `stats`/`state`).
- [ ] Solver APIs prefer mature Lineax/Optimistix entry points before introducing bespoke solver kernels.
- [ ] `RESULTS` enum members have actionable, user-facing messages for non-success outcomes.
- [ ] Enum compatibility is preserved with deprecation windows (no silent renames/removals).
- [ ] Enum evolution is backward-compatible (additive extensions and deprecation aliases/warnings before removal).
- [ ] Any breaking change includes a deprecation period or migration guide.
- [ ] Error semantics are consistent (`throw=False` paths tested).
- [ ] `throw=True` escalation paths are explicit and use `result.error_if` where applicable.

## Documentation
- [ ] Public APIs and Modules have docstrings.
- [ ] Docstring format is consistent with repo style (e.g., `**Arguments:**` / `**Returns:**`).
- [ ] Determinism limits and failure modes are documented.
- [ ] `throw` behavior is documented for batching (`vmap`) and backend-specific runtime error caveats where relevant.
- [ ] Examples cover common usage patterns and failure modes.

## Dependencies
- [ ] Project metadata and tool configuration are centralized in `pyproject.toml` unless there is a documented exception.
- [ ] Core dependency constraints include minimum versions and explicit exclusions for known-bad releases when needed.

## CI / Quality gates
- [ ] Format/lint/typecheck/tests run in CI.
- [ ] New features include tests (including JIT/AD/vmap where relevant).
- [ ] Bug fixes include regression tests.

## Reproducibility
- [ ] PRNGKeys are explicit and deterministic in tests/examples.
- [ ] Device/backend nondeterminism is documented if it affects results.

## Serialization / checkpoints
- [ ] Persisted state is versioned.
- [ ] Load paths validate version compatibility or provide migration notes.

## CLI
- [ ] CLI entrypoints are thin wrappers over library functions and avoid import-time side effects.
- [ ] CLI exposes reproducibility controls where relevant (seed/dtype/device and version reporting).

## Logging / diagnostics
- [ ] Diagnostic output is gated behind flags.
- [ ] No `print` in traced/JIT paths; use `jax.debug.print` sparingly.
