# Scripts

Utility scripts for installing skills globally.

## Validate skill links
Validates that `checklists/*`, `snippets/*`, and `references/*` links in markdown
files under `skills/` point to files that exist in this repo. Relative links are
resolved from the current markdown file's directory.

```bash
./scripts/validate_skill_links.sh
```

## Enforce SKILL.md line budget
Checks top-level `skills/*/SKILL.md` files against a maximum line count (default: 500).

```bash
./scripts/check_skill_entrypoint_line_budget.sh
./scripts/check_skill_entrypoint_line_budget.sh 500
```

## Install skills only
Installs skill files into the Codex home directory.

```bash
./scripts/install_skills.sh            # installs to ~/.codex/skills
./scripts/install_skills.sh /path/to/codex/home
```

## Install skills + assets
Installs skills plus checklists/snippets into a portable assets folder.

```bash
./scripts/install_skills_with_assets.sh            # installs to ~/.codex/skills
./scripts/install_skills_with_assets.sh /path/to/codex/home
```

Assets are placed in:
- `<CODEX_HOME>/skills/assets/checklists/`
- `<CODEX_HOME>/skills/assets/snippets/`

The script validates layout before installing:
- `skills/*/SKILL.md` must exist for each top-level skill directory.
- `checklists/README.md` and at least one checklist `.md` must exist.
- `snippets/README.md` and at least one snippet file must exist.
