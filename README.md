# JAX Numerics Agent

Practical guidance and artifacts for building reliable JAX/Equinox scientific
computing code: design rules, best practices, checklists, and copy‑paste snippets.

## What this is
- A compact “front desk” for JAX/Equinox numerics conventions and engineering rules.
- A synthesis of patterns from Equinox, Lineax, Optimistix, and Diffrax.
- A toolkit to help humans or coding agents produce stable, performant, and testable
  numerical code.

## What this is not
- A library or framework.
- A replacement for upstream docs. This repo points at patterns and codifies rules.

## How to use
- Read `AGENTS.md` first for bootstrap guidance and source-of-truth pointers.
- Use the skills files for deeper guidance.
- Use checklists for reviews and pre‑merge sanity checks.
- Use snippets as minimal, correct starting points.

## Install for Codex (global)
If you want these skills available in every project, install them once into your
Codex home directory (no project copies needed):

```bash
./scripts/install_skills.sh            # installs to ~/.codex/skills
./scripts/install_skills.sh /path/to/codex/home
./scripts/install_skills_with_assets.sh            # installs skills + assets
./scripts/install_skills_with_assets.sh /path/to/codex/home
```

## Project structure
- `AGENTS.md` — bootstrap entrypoint and source-of-truth pointers.
- `skills/jax_equinox_best_practices/SKILL.md` — numerics‑focused patterns (JIT, PyTrees, AD, RNG, stability).
- `skills/project_engineering/SKILL.md` — API stability, docs, pyproject, typing, CLI, CI, serialization.
- `checklists/` — targeted checklists for design, JIT/static, numerics/AD/testing, linear algebra, engineering.
- `snippets/` — ready‑to‑paste code templates.
- `sources/` — scanned source codebases used to derive rules (read‑only).

## Suggested entry points
- New to the repo: `AGENTS.md`
- Implementing a solver or optimizer: `skills/jax_equinox_best_practices/SKILL.md`
- Tightening engineering/CI: `skills/project_engineering/SKILL.md`
- Reviewing a PR: `checklists/README.md`
- Starting code: `snippets/README.md`

## Contributing
- Keep guidance actionable and minimal.
- Prefer DO/DON’T rules, then examples.
- If you change a skill rule, update the corresponding checklist and companion snippet references.
- Keep `skills/*/SKILL.md` references to `checklists/*` and `snippets/*` valid.
- Run `./scripts/validate_skill_links.sh` before opening a PR (CI also enforces this).
