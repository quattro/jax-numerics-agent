# AGENTS.md

Contract for coding agents working on JAX/Equinox scientific computing.

## Definitions (strict)
- JIT boundary: the public API function wrapped with `eqx.filter_jit`/`jax.jit`.
- Static data: non-array metadata that must be compile-time constant across calls; declare via `eqx.field(static=True)` or `eqx.partition`.
- Dynamic data: array leaves that change per-call and flow through JIT.
- PyTree stability: identical treedef and leaf shapes/dtypes across iterations and across calls; only values may change.
- Abstract module: `eqx.Module` containing `abc.abstractmethod` or `eqx.AbstractVar`; not instantiable.
- Final module: concrete `eqx.Module` with no further subclassing or overrides.
- Deterministic: same inputs and PRNGKey produce the same outputs under `jit/vmap/scan`.

## Non-negotiable rules (DO / DON'T)

**DO**
- Thread PRNGKeys explicitly through every stochastic function.
- Separate static config from dynamic data before JIT and keep it stable.
- Keep PyTree structure and shapes constant inside iterative loops.
- Use `lax.scan/while_loop/cond` for control flow under JIT.
- Validate critical assumptions (shape, dtype, finiteness) and surface failures predictably.
- Use `eqx.Module` for ABCs, with `AbstractVar` for abstract attributes.
- Define all fields and `__init__` in one class; concrete classes are final.

**DON'T**
- Don’t capture JAX arrays in closures that cross `jit` or custom-AD boundaries.
- Don’t use global RNG state or create keys inside jitted code.
- Don’t mutate or replace PyTrees in ways that change structure mid-loop.
- Don’t use Python `if/for/while` inside JIT when the condition depends on arrays.
- Don’t raise Python exceptions inside JIT; use `eqx.error_if` or result codes.
- Don’t subclass or override methods of a concrete module; use composition.
- Don’t use `super()` in module hierarchies; follow abstract-or-final.
- Don’t use `hasattr` for optional attributes; declare them on the ABC.

## Strong guidelines
- JIT public APIs once; avoid nested JITs in hot loops.
- Batch with `eqx.filter_vmap` and explicit `in_axes` for PyTrees and Modules.
- Prefer operator-based linear algebra over materializing large matrices.
- Use `jax.eval_shape`/`ShapeDtypeStruct` for structure validation.
- Choose a dtype policy up front; cast once to inexact types.
- Prefer x64 for stiff or ill-conditioned problems when accuracy matters.
- Use `__check_init__` (Equinox) to enforce invariants early.
- When ABC state varies, use `TypeVar`/`Generic[...]` to type solver state.

## Project engineering rules (software development)

**DO**
- Keep public API signatures stable; document any breaking change explicitly.
- Return structured results consistently (value + status/result + optional stats).
- Use `throw=False` modes where failures are expected and test them.
- Add docstrings for public APIs and Modules; use markdown sections like `**Arguments:**`/`**Returns:**` if the repo uses that style.
- Maintain CI gates: format, lint, typecheck, tests; do not merge with failing checks.
- Add regression tests for bug fixes and for AD/batching behavior.
- Record serialization/checkpoint formats and version any persisted state.
- Note determinism limits across devices/backends in docs when relevant.

**DON'T**
- Don’t change return structure or error semantics without a deprecation period.
- Don’t introduce silent behavior changes (e.g., new defaults) without docs and tests.
- Don’t add logging in hot JIT paths; gate diagnostics behind flags.
- Don’t serialize raw modules without documenting version/compat constraints.

## Performance reasoning
- Minimize retracing by isolating static arguments and keeping PyTree structures stable.
- Avoid shape-changing branches; use `lax.cond` and preserve output structure.
- If compile time dominates, stage large subgraphs (e.g., noinline-style wrappers).

## Numerics reasoning
- Guard divisions and norms; avoid NaNs/inf and define stable JVPs if needed.
- Prefer explicit `rtol/atol` and scaling choices; document the chosen norm.
- Detect nonfinite values early and propagate result codes explicitly.

## Shape semantics
- Treat shapes/dtypes as part of the API contract.
- Any change in PyTree structure is a breaking change and must be explicit.
