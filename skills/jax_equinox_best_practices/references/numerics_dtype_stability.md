# Numerical Stability and Dtype Policy

Detailed rules for dtype normalization, divide/norm guards, and nonfinite handling.

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
import jax.numpy as jnp

x = eqx.error_if(x, ~jnp.isfinite(x), "nonfinite")
```
- Allowed break: Exploratory runs where failures are intentionally ignored.
