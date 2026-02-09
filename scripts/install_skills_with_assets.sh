#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_HOME="${1:-$HOME/.codex}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

validate_skills_layout() {
  local skills_dir="$ROOT_DIR/skills"
  [[ -d "$skills_dir" ]] || die "Missing skills directory: $skills_dir"

  local skill_dirs=()
  while IFS= read -r skill_dir; do
    skill_dirs+=("$skill_dir")
  done < <(find "$skills_dir" -mindepth 1 -maxdepth 1 -type d | sort)
  [[ ${#skill_dirs[@]} -gt 0 ]] || die "No skill directories found in: $skills_dir"

  local missing=0
  for skill_dir in "${skill_dirs[@]}"; do
    if [[ ! -f "$skill_dir/SKILL.md" ]]; then
      echo "ERROR: Missing SKILL.md in skill directory: $skill_dir" >&2
      missing=1
    fi
  done
  [[ $missing -eq 0 ]] || die "Invalid skills layout. Each top-level skill directory must contain SKILL.md."
}

validate_assets_layout() {
  local checklists_dir="$ROOT_DIR/checklists"
  local snippets_dir="$ROOT_DIR/snippets"

  [[ -d "$checklists_dir" ]] || die "Missing checklists directory: $checklists_dir"
  [[ -d "$snippets_dir" ]] || die "Missing snippets directory: $snippets_dir"
  [[ -f "$checklists_dir/README.md" ]] || die "Missing $checklists_dir/README.md"
  [[ -f "$snippets_dir/README.md" ]] || die "Missing $snippets_dir/README.md"

  [[ -n "$(find "$checklists_dir" -mindepth 1 -type f -name '*.md' -print -quit)" ]] || die "No checklist markdown files found in: $checklists_dir"
  [[ -n "$(find "$snippets_dir" -mindepth 1 -type f -print -quit)" ]] || die "No snippet files found in: $snippets_dir"
}

command -v rsync >/dev/null 2>&1 || die "rsync is required but was not found on PATH."
validate_skills_layout
validate_assets_layout

mkdir -p "$CODEX_HOME/skills"
rsync -a "$ROOT_DIR/skills/" "$CODEX_HOME/skills/"

mkdir -p "$CODEX_HOME/skills/assets"
rsync -a "$ROOT_DIR/checklists/" "$CODEX_HOME/skills/assets/checklists/"
rsync -a "$ROOT_DIR/snippets/" "$CODEX_HOME/skills/assets/snippets/"

cat <<MSG
Installed skills into: $CODEX_HOME/skills
Installed assets into: $CODEX_HOME/skills/assets
MSG
