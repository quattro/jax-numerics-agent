---
name: project-engineering
description: Use for software engineering practices around JAX/Equinox numerical codebases; covers API stability, documentation, pyproject, typing, CLI, and CI.
metadata:
  short-description: Project engineering for JAX/Equinox
---

# Project Engineering (JAX/Equinox Scientific Computing)

Guidance for software engineering practices around numerical/JAX codebases.

## Public API stability

### Rule: Keep API structure stable and explicit
- Do: Preserve return shapes/fields and document result codes/statuses.
- Don’t: Change return structures or error semantics without deprecation.
- Why: Scientific workflows rely on stable interfaces.
- Example:
```python
# Return a structured result: value + result + stats/state
return Solution(value=out, result=result, stats=stats, state=state)
```
- Allowed break: Major version bump with migration guide.

### Rule: Prefer composition over subclassing for concrete Modules
- Do: Wrap and delegate when customizing behavior.
- Don’t: Subclass or override concrete modules.
- Why: Aligns with Equinox abstract‑or‑final pattern and avoids override ambiguity.

## Documentation

### Rule: Match repository docstring style
- Do: Use markdown sections like `**Arguments:**` / `**Returns:**` when the repo uses that style.
- Don’t: Mix multiple docstring styles in the same module.
- Why: Consistency improves usability and tooling.
- Example:
```python
def solve(...):
    """**Arguments:**
    - `x`: input

    **Returns:**
    - solution
    """
```
- Allowed break: Internal‑only helpers.

### Rule: Document determinism and failure modes
- Do: Explain nondeterminism across devices/backends and list expected failure results.
- Don’t: Leave runtime behavior implicit.
- Why: Reproducibility depends on clear constraints.

## pyproject.toml (project configuration)

### Rule: Centralize project metadata and tooling in `pyproject.toml`
- Do: Declare Python version, dependencies, optional extras, and tool configs in one place.
- Don’t: Scatter configuration across multiple files without justification.
- Why: Single source of truth reduces drift.
- Example (minimal structure):
```toml
[project]
name = "my-lib"
requires-python = ">=3.10"

[project.optional-dependencies]
test = ["pytest", "pytest-xdist"]

[tool.ruff]
line-length = 88

[tool.pyright]
typeCheckingMode = "strict"
```
- Allowed break: Legacy repos that already standardize on separate config files.

### Rule: Pin minimum versions for core scientific deps
- Do: Set lower bounds for JAX/Equinox/jaxtyping and note compatibility.
- Don’t: Allow unconstrained upgrades that silently change semantics.
- Why: Numerical stability and API behavior change across versions.

## Type checking

### Rule: Enforce type checks in CI
- Do: Use pyright/mypy and keep annotations on public APIs.
- Don’t: Ship untyped public functions or ignore type checker errors.
- Why: Prevents misuse of PyTrees, shapes, and configs.
- Example:
```python
from jaxtyping import Array, PyTree

def f(x: PyTree[Array]) -> PyTree[Array]:
    ...
```
- Allowed break: Small private helpers where typing adds no value.

### Rule: Model solver state with generics when it varies
- Do: Use `TypeVar`/`Generic` in ABCs for solver state.
- Don’t: Use `Any` for state in public interfaces.

## CLI building

### Rule: Keep CLI thin and side‑effect free
- Do: Parse args in `main()`, call into library functions, and return exit codes.
- Don’t: Execute JAX code at import time or use global state.
- Why: Keeps CLI deterministic and testable.
- Example:
```python
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    return run(seed=args.seed)

if __name__ == "__main__":
    raise SystemExit(main())
```
- Allowed break: Tiny scripts not shipped as part of the library.

### Rule: Make CLI outputs reproducible
- Do: Expose seed/dtype/device flags and log versions.
- Don’t: Hide randomness or backend choices.

## CI / Quality gates

### Rule: Enforce formatting, linting, type checks, and tests
- Do: Run format/lint/typecheck/tests in CI.
- Don’t: Merge changes with failing checks.
- Why: Numerical regressions are subtle and expensive.

### Rule: Add regression tests for bug fixes
- Do: Write a failing test first, then fix.
- Don’t: Merge fixes without test coverage.

## Serialization / checkpoints

### Rule: Version persisted state
- Do: Store a version tag and validate on load.
- Don’t: Persist opaque state without compatibility notes.
- Why: Long‑running experiments need stable restore paths.
