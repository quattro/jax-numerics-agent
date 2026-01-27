# JIT / Static / PyTree Checklist

Use this before performance tuning or when encountering retracing.

## JIT boundaries
- [ ] JIT is applied once at the public API boundary.
- [ ] No nested JITs inside inner loops or solver steps.
- [ ] Large subgraphs are staged explicitly if compile time dominates.

## Static vs dynamic
- [ ] All non-array config is static (fields or partitioned).
- [ ] No Python objects or containers flow as dynamic inputs.
- [ ] `eqx.filter_closure_convert` is used for functions passed across JIT/AD boundaries.

## PyTree stability
- [ ] State treedef and leaf shapes/dtypes are unchanged across iterations.
- [ ] `jax.eval_shape`/`ShapeDtypeStruct` is used to validate structure.
- [ ] `eqx.partition` + `eqx.combine` are used to keep static leaves fixed.

## Control flow
- [ ] No Python `if/for/while` branches depend on arrays inside JIT.
- [ ] `lax.cond` branches return identical PyTree structure.
- [ ] `lax.scan` carries only fixed-shape data.
