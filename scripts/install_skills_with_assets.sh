#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_HOME="${1:-$HOME/.codex}"

mkdir -p "$CODEX_HOME/skills"
rsync -a "$ROOT_DIR/skills/" "$CODEX_HOME/skills/"

mkdir -p "$CODEX_HOME/skills/assets"
rsync -a "$ROOT_DIR/checklists/" "$CODEX_HOME/skills/assets/checklists/"
rsync -a "$ROOT_DIR/snippets/" "$CODEX_HOME/skills/assets/snippets/"

cat <<MSG
Installed skills into: $CODEX_HOME/skills
Installed assets into: $CODEX_HOME/skills/assets
MSG
