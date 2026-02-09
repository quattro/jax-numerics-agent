# Contributing

Thanks for improving this repository.

## Scope

This project is a guidance package for coding agents and developers. It is not
a runtime library. Keep changes focused on skills, checklists, snippets, and
supporting maintenance scripts.

## Workflow

1. Read `AGENTS.md`.
2. Edit the canonical source files:
   - `skills/jax_equinox_best_practices/SKILL.md`
   - `skills/project_engineering/SKILL.md`
3. Keep companion links valid across:
   - `checklists/*`
   - `snippets/*`
   - `skills/jax_equinox_best_practices/references/*`
4. Run validation:
   - `./scripts/validate_skill_links.sh`
   - `./scripts/check_skill_entrypoint_line_budget.sh 500`

## Content guidelines

- Keep guidance actionable and specific.
- Prefer direct DO and DON'T rules with short rationale.
- Keep skill entrypoints concise and move depth into `references/` files.
- When changing a rule, update linked checklists/snippets in the same change.

## Optional local checks

- Validate global install behavior:
  - `./scripts/install_skills.sh /tmp/codex-home`
  - `./scripts/install_skills_with_assets.sh /tmp/codex-home`

## Pull requests

- Explain what changed and why.
- Include any validation command outputs in the PR description.
- Keep unrelated edits out of the same PR.
