#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILLS_DIR="$ROOT_DIR/skills"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "$SKILLS_DIR" ]] || die "Missing skills directory: $SKILLS_DIR"

skill_files=()
while IFS= read -r skill_file; do
  skill_files+=("$skill_file")
done < <(find "$SKILLS_DIR" -mindepth 2 -maxdepth 2 -type f -name "SKILL.md" | sort)

[[ ${#skill_files[@]} -gt 0 ]] || die "No SKILL.md files found under: $SKILLS_DIR"

missing=0
checked=0

echo "Validating checklist/snippet references in skills..."

for skill_file in "${skill_files[@]}"; do
  rel_skill="${skill_file#$ROOT_DIR/}"
  echo "- $rel_skill"

  refs=()
  while IFS= read -r ref; do
    refs+=("$ref")
  done < <(grep -oE 'checklists/[A-Za-z0-9._/-]+\.md|snippets/[A-Za-z0-9._/-]+\.[A-Za-z0-9]+' "$skill_file" | sort -u || true)

  if [[ ${#refs[@]} -eq 0 ]]; then
    echo "  INFO: no checklist/snippet references found."
    continue
  fi

  for ref in "${refs[@]}"; do
    checked=$((checked + 1))
    if [[ -f "$ROOT_DIR/$ref" ]]; then
      echo "  OK: $ref"
    else
      echo "  MISSING: $ref"
      missing=1
    fi
  done
done

if [[ $missing -ne 0 ]]; then
  die "Found missing checklist/snippet references."
fi

echo "Validation passed ($checked references checked)."
