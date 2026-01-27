# Scripts

Utility scripts for installing skills globally.

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
