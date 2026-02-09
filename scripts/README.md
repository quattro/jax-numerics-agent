# Scripts

Utility scripts for installing skills globally.

## Validate skill links
Validates that `checklists/*` and `snippets/*` references in `skills/*/SKILL.md` point to files that exist in this repo.

```bash
./validate_skill_links.sh
```

## Install skills only
Installs skill files into the Codex home directory.

```bash
./install_skills.sh            # installs to ~/.codex/skills
./install_skills.sh /path/to/codex/home
```

## Install skills + assets
Installs skills plus checklists/snippets into a portable assets folder.

```bash
./install_skills_with_assets.sh            # installs to ~/.codex/skills
./install_skills_with_assets.sh /path/to/codex/home
```

Assets are placed in:
- `<CODEX_HOME>/skills/assets/checklists/`
- `<CODEX_HOME>/skills/assets/snippets/`

The script validates layout before installing:
- `skills/*/SKILL.md` must exist for each top-level skill directory.
- `checklists/README.md` and at least one checklist `.md` must exist.
- `snippets/README.md` and at least one snippet file must exist.
