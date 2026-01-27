import jax
import jax.numpy as jnp


@jax.custom_jvp
def stable_norm(x):
    return jnp.sqrt(jnp.sum(x * x))


@stable_norm.defjvp
def _stable_norm_jvp(primals, tangents):
    (x,), (tx,) = primals, tangents
    y = stable_norm(x)
    denom = jnp.where(y == 0, 1.0, y)
    return y, jnp.where(y == 0, 0.0, jnp.vdot(x, tx) / denom)
