# Project Engineering Checklist

Use this for general software engineering hygiene around numerical/JAX projects.

## API stability
- [ ] Public API signatures and return structures are stable and documented.
- [ ] Any breaking change includes a deprecation period or migration guide.
- [ ] Error semantics are consistent (`throw=False` paths tested).

## Documentation
- [ ] Public APIs and Modules have docstrings.
- [ ] Docstring format is consistent with repo style (e.g., `**Arguments:**` / `**Returns:**`).
- [ ] Examples cover common usage patterns and failure modes.

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

## Logging / diagnostics
- [ ] Diagnostic output is gated behind flags.
- [ ] No `print` in traced/JIT paths; use `jax.debug.print` sparingly.
