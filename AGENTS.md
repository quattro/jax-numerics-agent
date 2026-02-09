# AGENTS.md

Bootstrap contract for coding agents in this repository.

## Scope

This repo is a guidance package (skills, checklists, snippets), not a runtime library.

## Source of truth

Canonical rules live in:
- `skills/jax_equinox_best_practices/SKILL.md`
- `skills/project_engineering/SKILL.md`

If this file conflicts with either skill file, the skill files take precedence.

## Agent workflow

- Load the relevant skill file(s) first.
- For numerics deep-dives, load topic docs from `skills/jax_equinox_best_practices/references/` as directed by the numerics `SKILL.md`.
- Follow companion checklist/snippet references declared in each skill.
- Keep `checklists/*`, `snippets/*`, and `references/*` links in skill docs valid.
- Run `./scripts/validate_skill_links.sh` after editing skills/checklists/snippets.

## Global install note

When installed via `scripts/install_skills_with_assets.sh`, assets live under:
- `~/.codex/skills/assets/checklists/`
- `~/.codex/skills/assets/snippets/`

## Maintenance

Keep this file minimal. Do not duplicate rule catalogs here; update the skill files.
