# Snippets

Minimal, ready-to-paste examples aligned with the JAX/Equinox guidance in this repo.

## Index
- `jit_boundary.py` — public JIT boundary with static config.
- `partition_static_state.py` — keep static state fixed across iterations.
- `filter_vmap_batching.py` — batched compute with `eqx.filter_vmap`.
- `prng_split_by_tree.py` — deterministic key splitting by PyTree structure.
- `custom_jvp_norm.py` — stable custom JVP for a norm.
- `abc_module_pattern.py` — abstract-or-final module with `AbstractVar` and generics.
- `filter_cond_static.py` — safe `lax.cond` with static outputs.
- `linear_operator_pattern.py` — Jacobian operator usage without materializing matrices.
- `implicit_jvp.py` — implicit-function JVP pattern.
- `test_jvp_finite_difference.py` — JVP vs finite-difference test template.
- `cli_skeleton.py` — minimal CLI pattern.
- `pyproject_minimal.toml` — minimal pyproject configuration.
