# Numerics / AD / Testing Checklist

Use this for solver implementation and verification.

## Dtype and stability
- [ ] Inputs are cast once to an inexact dtype at the boundary.
- [ ] Dtype policy (x32/x64) is explicit and documented.
- [ ] Divisions and norms are guarded against zero/inf.
- [ ] Nonfinite values are detected early and surfaced predictably.

## PRNG discipline
- [ ] PRNGKeys are explicit inputs/outputs to stochastic functions.
- [ ] Keys are split/folded deterministically (step/time included).
- [ ] PyTree-shaped randomness uses split-by-tree helpers.

## Custom AD
- [ ] Custom JVP/VJP is used for implicit/iterative methods where needed.
- [ ] Custom primitives include abstract_eval + JVP + transpose.
- [ ] Nondifferentiable config/state is guarded (`eqx.nondifferentiable`).

## Testing
- [ ] JIT + vmap + grad are exercised in tests.
- [ ] JVPs are checked against finite differences where feasible.
- [ ] Batching invariants hold (vmapped vs unbatched consistency).
- [ ] Failure modes (max_steps, nonfinite, ill-conditioned) are tested.

## Debugging
- [ ] `eqx.error_if` is used for runtime errors inside JIT.
- [ ] `jax.debug.print` is used for traced diagnostics.
- [ ] `EQX_ON_ERROR=breakpoint` is documented for runtime failures.
