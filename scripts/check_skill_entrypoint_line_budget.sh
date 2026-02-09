#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILLS_DIR="$ROOT_DIR/skills"
LINE_BUDGET="${1:-500}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "$SKILLS_DIR" ]] || die "Missing skills directory: $SKILLS_DIR"
[[ "$LINE_BUDGET" =~ ^[0-9]+$ ]] || die "Line budget must be an integer. Got: $LINE_BUDGET"

entrypoint_files=()
while IFS= read -r file; do
  entrypoint_files+=("$file")
done < <(find "$SKILLS_DIR" -mindepth 2 -maxdepth 2 -type f -name "SKILL.md" | sort)

[[ ${#entrypoint_files[@]} -gt 0 ]] || die "No skill entrypoint files found under: $SKILLS_DIR"

failed=0
checked=0

echo "Checking SKILL.md line budget (max $LINE_BUDGET lines)..."

for file in "${entrypoint_files[@]}"; do
  rel_file="${file#$ROOT_DIR/}"
  line_count="$(wc -l < "$file" | tr -d ' ')"
  checked=$((checked + 1))

  if (( line_count > LINE_BUDGET )); then
    echo "  FAIL: $rel_file ($line_count lines; limit $LINE_BUDGET)"
    failed=1
  else
    echo "  OK: $rel_file ($line_count lines)"
  fi
done

if [[ $failed -ne 0 ]]; then
  die "One or more SKILL.md files exceed the line budget."
fi

echo "Line-budget check passed ($checked files checked)."
