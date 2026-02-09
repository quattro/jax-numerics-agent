#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILLS_DIR="$ROOT_DIR/skills"
REF_PATTERN='checklists/[A-Za-z0-9._/-]+\.md|snippets/[A-Za-z0-9._/-]+\.[A-Za-z0-9]+|references/[A-Za-z0-9._/-]+\.md'

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "$SKILLS_DIR" ]] || die "Missing skills directory: $SKILLS_DIR"

resolve_ref_path() {
  local source_file="$1"
  local ref="$2"
  local source_dir
  source_dir="$(dirname "$source_file")"

  if [[ -f "$ROOT_DIR/$ref" ]]; then
    printf '%s\n' "$ROOT_DIR/$ref"
    return 0
  fi

  if [[ -f "$source_dir/$ref" ]]; then
    printf '%s\n' "$source_dir/$ref"
    return 0
  fi

  return 1
}

doc_files=()
while IFS= read -r doc_file; do
  doc_files+=("$doc_file")
done < <(find "$SKILLS_DIR" -type f -name "*.md" | sort)

[[ ${#doc_files[@]} -gt 0 ]] || die "No markdown files found under: $SKILLS_DIR"

missing=0
checked=0

echo "Validating checklist/snippet/reference links in skills..."

for doc_file in "${doc_files[@]}"; do
  rel_doc="${doc_file#$ROOT_DIR/}"
  echo "- $rel_doc"

  refs=()
  while IFS= read -r ref; do
    refs+=("$ref")
  done < <(grep -oE "$REF_PATTERN" "$doc_file" | sort -u || true)

  if [[ ${#refs[@]} -eq 0 ]]; then
    echo "  INFO: no checklist/snippet/reference links found."
    continue
  fi

  for ref in "${refs[@]}"; do
    checked=$((checked + 1))
    if resolve_ref_path "$doc_file" "$ref" >/dev/null; then
      echo "  OK: $ref"
    else
      echo "  MISSING: $ref"
      missing=1
    fi
  done
done

if [[ $missing -ne 0 ]]; then
  die "Found missing checklist/snippet/reference links."
fi

echo "Validation passed ($checked references checked)."
