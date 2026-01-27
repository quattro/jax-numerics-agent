---
name: jax-equinox-numerics
description: Use for any JAX + Equinox numerics project; repo-agnostic patterns plus a checklist to align with local style.
metadata:
  short-description: JAX/Equinox numerics
---

# JAX + Equinox Best Practices (from equinox/lineax/optimistix/diffrax patterns)

This playbook distills recurring patterns from the scanned repos into reusable rules
for general scientific/numerical computing in JAX + Equinox.

## Definitions (strict)
- JIT boundary: the public API function wrapped with `eqx.filter_jit`/`jax.jit`.
- Static data: non-array metadata/config that must be compile-time constant across calls.
- Dynamic data: array leaves that vary per call and flow through JIT.
- PyTree stability: identical treedef and leaf shapes/dtypes across iterations and calls.
- Abstract module: `eqx.Module` with `abc.abstractmethod` or `eqx.AbstractVar`.
- Final module: concrete `eqx.Module` with no further overrides or subclassing.

## Global DO / DON'T

**DO**
- Thread PRNGKeys explicitly and deterministically.
- Separate static vs dynamic data before JIT.
- Use `lax.scan/while_loop/cond` for traced control flow.
- Validate structure and finiteness early.
- Use operator-based linear algebra when possible.

**DON'T**
- Don’t capture JAX arrays in closures that cross JIT or custom AD.
- Don’t mutate PyTree structure inside loops.
- Don’t rely on Python control flow under JIT.
- Don’t subclass or override concrete modules.
- Don’t use global RNG state.

## JIT boundaries and static vs dynamic arguments

### Rule: JIT at the public boundary and keep configuration static
- Do: Apply `eqx.filter_jit` at the API boundary and mark non-array config as static.
- Don’t: JIT inside hot loops or pass config as dynamic arrays.
- Why: Prevents retracing and keeps compile graphs stable.
- Example:
```python
import equinox as eqx
import jax.numpy as jnp

class Solver(eqx.Module):
    rtol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)

@eqx.filter_jit
def solve(fn, y0, args, solver: Solver):
    return fn(y0, args)
```
- Allowed break: Debug/prototype code or Python-only inputs.

### Rule: Closure-convert functions that cross JIT/AD boundaries
- Do: Use `eqx.filter_closure_convert` for functions passed into operators/solvers.
- Don’t: Capture model parameters in a closure while passing only `y` to the solver.
- Why: Avoids closed-over tracers and keeps metadata in the PyTree.
- Example:
```python
import equinox as eqx
import lineax as lx

fn = eqx.filter_closure_convert(fn, x, args)
operator = lx.JacobianLinearOperator(fn, x, args)
```
- Allowed break: Pure Python closures with no JAX arrays (e.g., constants).

## Equinox Modules, ABCs, and generics

### Rule: Use `eqx.Module` for ABCs and follow abstract-or-final
- Do: Define abstract methods/attributes on `eqx.Module` with `abc.abstractmethod` and `eqx.AbstractVar`.
- Don’t: Override concrete methods or split fields across a class hierarchy.
- Why: Enforces readable initialization and eliminates override ambiguity.
- Example:
```python
import abc
import equinox as eqx
from jaxtyping import Array, PyTree

class AbstractSolver(eqx.Module):
    rtol: eqx.AbstractVar[float]

    @abc.abstractmethod
    def step(self, y: PyTree[Array], state): ...

class MySolver(AbstractSolver):
    rtol: float

    def step(self, y, state):
        ...
```
- Allowed break: Non-Equinox code or one-off prototypes.

### Rule: Use `TypeVar`/`Generic` to express state types when needed
- Do: Parameterize solver ABCs with `Generic[_State]` if state differs by solver.
- Don’t: Use `Any` for state or change state structure across iterations.
- Why: Makes state shape explicit and improves type checking.
- Example:
```python
from typing import Generic, TypeVar
import equinox as eqx
from jaxtyping import PyTree, Array

_State = TypeVar("_State")

class AbstractIterativeSolver(eqx.Module, Generic[_State]):
    @abc.abstractmethod
    def init(self, y: PyTree[Array]) -> _State: ...

    @abc.abstractmethod
    def step(self, y: PyTree[Array], state: _State) -> tuple[PyTree[Array], _State]: ...
```
- Allowed break: Single-solver scripts where state is fixed and local.

## PyTree stability and recompilation avoidance

### Rule: Partition dynamic vs static state and keep static unchanged
- Do: `eqx.partition` state, update dynamic leaves, and assert static stability.
- Don’t: Replace buffers with different shapes or change treedef mid-loop.
- Why: JIT assumes stable structure; changes trigger recompilation or errors.
- Example:
```python
import equinox as eqx

state = solver.init(...)
state_dyn, state_static = eqx.partition(state, eqx.is_array)

new_state = solver.step(...)
new_dyn, new_static = eqx.partition(new_state, eqx.is_array)
assert eqx.tree_equal(state_static, new_static) is True
state = eqx.combine(state_static, new_dyn)
```
- Allowed break: Only outside JIT with an explicit re-jit boundary.

### Rule: Use ShapeDtypeStruct for structure checks
- Do: Compare structures via `jax.eval_shape`/`ShapeDtypeStruct`.
- Don’t: Infer structure from runtime values inside JIT.
- Why: Structure checks must not execute numerics.
- Example:
```python
import jax
import jax.tree_util as jtu

struct = jax.eval_shape(lambda: y)
assert jtu.tree_structure(struct) == jtu.tree_structure(expected_struct)
```
- Allowed break: Small non-jitted utilities.

### Rule: Preserve static outputs in control flow
- Do: Wrap static outputs with `eqxi.Static` (or helper) inside `lax.cond`.
- Don’t: Return Python objects from one branch and arrays from another.
- Why: `lax.cond` requires consistent output structure.
- Example:
```python
import equinox as eqx
import equinox.internal as eqxi
import jax.lax as lax

def filter_cond(pred, true_fun, false_fun, *ops):
    dyn, stat = eqx.partition(ops, eqx.is_array)
    def _wrap(fn):
        def inner(dyn_ops):
            out = fn(*eqx.combine(dyn_ops, stat))
            dyn_out, stat_out = eqx.partition(out, eqx.is_array)
            return dyn_out, eqxi.Static(stat_out)
        return inner
    dyn_out, stat_out = lax.cond(pred, _wrap(true_fun), _wrap(false_fun), dyn)
    return eqx.combine(dyn_out, stat_out.value)
```
- Allowed break: Non-jitted control flow.

## vmap / scan / lax usage patterns

### Rule: Use `lax.scan` for fixed-length loops and `lax.while_loop` for data-dependent loops
- Do: Replace Python loops under JIT with `lax.scan`/`lax.while_loop`.
- Don’t: Use Python `for/while` when loop bounds depend on arrays.
- Why: Preserves traceability and avoids repeated tracing.
- Example:
```python
import jax.lax as lax

carry, ys = lax.scan(body_fn, init_carry, xs)
final = lax.while_loop(cond_fn, body_fn, init_state)
```
- Allowed break: Small loops outside JIT.

### Rule: Use `eqx.filter_vmap` with explicit `in_axes`
- Do: Batch solvers with explicit `in_axes` that keep static args static.
- Don’t: Accidentally batch config/static fields.
- Why: Prevents wrong broadcasting and retracing.
- Example:
```python
import equinox as eqx

batched_solve = eqx.filter_vmap(solve, in_axes=(None, 0, None))
```
- Allowed break: When the entire module is intentionally batched.

### Rule: Avoid compiling heavy functions twice inside branches
- Do: Use `lax.scan` or a shared subroutine when evaluating heavy functions multiple times.
- Don’t: Duplicate large calls in separate branches of `lax.cond`.
- Why: Reduces compile time and graph size.
- Example:
```python
import jax.lax as lax

(init_out, _), _ = lax.scan(step_fn, init_carry, jnp.arange(2))
```
- Allowed break: Trivial functions where clarity dominates.

## PRNG discipline

### Rule: Thread keys explicitly; split and fold_in deterministically
- Do: Accept and return PRNGKeys; use `split` and `fold_in` for steps/times.
- Don’t: Create keys inside jitted code or rely on global RNG state.
- Why: Ensures determinism under JIT/vmap/scan.
- Example:
```python
import jax.random as jr

key = jr.PRNGKey(0)
key, subkey = jr.split(key)
key = jr.fold_in(key, step_index)
```
- Allowed break: Standalone scripts/benchmarks (still prefer explicit keys).

### Rule: Split keys by PyTree structure when sampling PyTree outputs
- Do: `split_by_tree(key, shape_tree)` and map over leaves.
- Don’t: Use `split(key, n)` with implicit leaf ordering.
- Why: Keeps randomness aligned with structure and stable under refactors.
- Example:
```python
keys = split_by_tree(key, output_shape_tree)
out = jax.tree_util.tree_map(sample_leaf, keys, output_shape_tree)
```
- Allowed break: Flat arrays with stable leaf ordering.

## Numerical stability and dtype policy

### Rule: Cast once to an inexact dtype and follow a clear dtype policy
- Do: Normalize dtype at the boundary and keep it consistent.
- Don’t: Mix Python scalars or ints into hot loops.
- Why: Prevents accidental integer math and dtype promotion surprises.
- Example:
```python
import jax.numpy as jnp

x = jnp.asarray(x)
if not jnp.issubdtype(x.dtype, jnp.inexact):
    x = x.astype(jnp.float32)
```
- Allowed break: Safe scalar literals in trivial helpers.

### Rule: Guard divisions and norms; define stable JVPs
- Do: Use `jnp.where` to avoid divide-by-zero; define stable custom JVPs for norms.
- Don’t: Divide by quantities that can be zero or near-zero without guards.
- Why: Avoids NaNs and unstable gradients.
- Example:
```python
den = jnp.where(den == 0, 1.0, den)
out = num / den
```
- Allowed break: When invariants guarantee nonzero denominators (and you validate them).

### Rule: Validate and surface nonfinite values early
- Do: Use `eqx.error_if` or explicit result codes for nonfinite outputs.
- Don’t: Let NaNs/inf silently propagate.
- Why: Makes failures debuggable and reproducible.
- Example:
```python
import equinox as eqx

x = eqx.error_if(x, ~jnp.isfinite(x), "nonfinite")
```
- Allowed break: Exploratory runs where failures are intentionally ignored.

## Custom JVP/VJP patterns

### Rule: Use custom JVP/VJP for implicit or iterative methods
- Do: Implement custom JVP/VJP when autodiff is unstable or too costly.
- Don’t: Rely on default AD for ill-conditioned solves.
- Why: Improves stability and performance.
- Example:
```python
import jax

@jax.custom_jvp
def stable_norm(x):
    return jnp.sqrt(jnp.sum(x * x))

@stable_norm.defjvp
def _jvp(primals, tangents):
    (x,), (tx,) = primals, tangents
    y = stable_norm(x)
    return y, jnp.where(y == 0, 0.0, jnp.vdot(x, tx) / y)
```
- Allowed break: Small, well-behaved problems where default AD is fine.

### Rule: For custom primitives, implement abstract_eval + JVP + transpose
- Do: Provide full AD rules and guard nondiff data.
- Don’t: Register a primitive without transpose/JVP or with leaked tracers.
- Why: Missing rules break AD and batching.
- Example:
```python
# Pseudocode: define impl, abstract_eval, jvp, transpose
# and use materialised zeros for missing tangents.
```
- Allowed break: Research prototypes that intentionally skip AD.

## Performance instrumentation and testing

### Rule: Test JVP/VJP and batching under JIT with finite differences
- Do: Validate AD using finite differences and test vmapped solves.
- Don’t: Only test primal outputs for iterative solvers.
- Why: Catches AD regressions and batching bugs early.
- Example:
```python
out, t_out = eqx.filter_jit(eqx.filter_jvp(f))((x,), (tx,))
# compare with finite_difference_jvp(...)
```
- Allowed break: Minimal smoke tests for early prototypes.

### Rule: Use debug hooks and structured errors for diagnostics
- Do: Use `eqx.error_if`, `jax.debug.print`, and `EQX_ON_ERROR=breakpoint`.
- Don’t: Use Python `print` inside jitted code.
- Why: JIT debugging requires structured error points.
- Example:
```python
jax.debug.print("residual: {}", res)
```
- Allowed break: Non-jitted scripts.

### Rule: Reduce compile-time cost with static choices when possible
- Do: Use static flags or staged subroutines to keep compile graphs small.
- Don’t: Leave everything dynamic when compile time dominates.
- Why: Reduces compilation overhead for large graphs.
- Example:
```python
import equinox as eqx

class Config(eqx.Module):
    use_fast_path: bool = eqx.field(static=True)

@eqx.filter_jit
def run(x, cfg: Config):
    return x if cfg.use_fast_path else x + 1
```
- Allowed break: When adaptive behavior is essential and compile time is acceptable.

## See also: project engineering

For API stability, CI gates, documentation style, type checking, CLI patterns, and
serialization guidance, see `skills/project_engineering.md`.
