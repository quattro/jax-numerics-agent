# Checklists

Focused checklists for JAX/Equinox numerics and software engineering.

## Index
- `jax_equinox_design_checklist.md` — API/Module architecture and ABC design.
- `jit_static_pytree_checklist.md` — JIT boundaries, static/dynamic data, PyTree stability.
- `numerics_ad_testing_checklist.md` — dtype, stability, PRNG, custom AD, testing.
- `linear_algebra_checklist.md` — operator-based linear algebra and solver concerns.
- `testing_checklist.md` — testing scope and coverage expectations.
- `project_engineering_checklist.md` — API stability, docs, CI, reproducibility, serialization.

## Selection guide

| Common task | Recommended checklist set | Why |
|---|---|---|
| Designing a new solver API/module | `jax_equinox_design_checklist.md`, `jit_static_pytree_checklist.md`, `project_engineering_checklist.md` | Covers architecture, traced execution constraints, and API/engineering contracts together. |
| Fixing retracing, JIT shape errors, or control-flow trace issues | `jit_static_pytree_checklist.md`, `jax_equinox_design_checklist.md` | Targets static/dynamic partitioning, PyTree stability, and control-flow structure invariants. |
| Implementing linear solves or nonlinear root/least-squares/minimise paths | `linear_algebra_checklist.md`, `numerics_ad_testing_checklist.md`, `testing_checklist.md` | Combines solver correctness, numerical/AD discipline, and required verification coverage. |
| Changing AD behavior or custom JVP/VJP/primitive rules | `numerics_ad_testing_checklist.md`, `testing_checklist.md`, `linear_algebra_checklist.md` (if solves are involved) | Ensures gradient stability and test depth, including solver-specific AD pitfalls. |
| Pre-merge review for numerics features | `project_engineering_checklist.md`, `testing_checklist.md`, plus one domain checklist above | Enforces release hygiene while retaining domain-specific correctness checks. |
