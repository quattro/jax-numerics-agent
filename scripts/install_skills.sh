#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_HOME="${1:-$HOME/.codex}"

mkdir -p "$CODEX_HOME/skills"
rsync -a "$ROOT_DIR/skills/" "$CODEX_HOME/skills/"

echo "Installed skills into: $CODEX_HOME/skills"
