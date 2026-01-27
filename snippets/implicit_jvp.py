import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx


# Global functions only: no closed-over JAX arrays.

def fn_primal(inputs):
    _, a = inputs  # First element unused; we solve for root such that root^2 = a
    root = jnp.sqrt(a)
    residual = root * root - a
    return root, residual


def fn_rewrite(root, residual, inputs):
    # Rewrite whose Jacobian w.r.t. root is used by implicit JVP
    _, a = inputs
    return root * root - a


inputs = (jnp.array(1.0), jnp.array(4.0))
root, residual = optx.implicit_jvp(
    fn_primal,
    fn_rewrite,
    inputs,
    tags=frozenset(),
    linear_solver=lx.AutoLinearSolver(well_posed=True),
)
