# Linear Algebra Checklist

Use this for linear solves, least squares, Jacobian operators, and preconditioning.

## Operator vs matrix
- [ ] Use operator-based representations when matrices are large or implicit.
- [ ] Materialize matrices only when necessary and justified by performance.
- [ ] Keep operator tags accurate (e.g., symmetric/PSD); incorrect tags are unsafe.

## Shape/structure
- [ ] Vector and operator structures match (PyTree shape/dtype compatible).
- [ ] Use `jax.eval_shape`/`ShapeDtypeStruct` to validate structure early.
- [ ] Keep operator input/output structures static.

## Solver configuration
- [ ] Choose solver based on conditioning and matrix structure.
- [ ] Handle under/overdetermined systems explicitly (least-squares/min-norm).
- [ ] Expose tolerances (`rtol`, `atol`) and max-iteration controls.

## Numerical stability
- [ ] Guard ill-conditioned solves with scaling or regularization.
- [ ] Prefer stable norm/conditioning checks over raw inverse operations.
- [ ] Detect nonfinite outputs and propagate result codes.

## AD considerations
- [ ] Use custom JVP/VJP when default gradients are unstable.
- [ ] Mark solver state/options as nondifferentiable.
- [ ] Ensure transpose/adjoint operations are implemented where required.

## Testing
- [ ] Compare against known solutions for small systems.
- [ ] Test singular/ill-conditioned cases and failure paths.
- [ ] Validate JVPs for linear solves when AD is supported.
