#!/usr/bin/env bash
# Install the Nebius Scientific AI skills for Claude Code.
#
#   bash install.sh                 install for your user (~/.claude/skills)
#   bash install.sh ./myproject     install into one project (.claude/skills)
#
# Two things go in:
#   1. the official Nebius scientific-ai bundle, 51 skills, one per model, with
#      the scientific parameters and references
#   2. nebius-scientific-ai, the client this kit adds, which runs the jobs from
#      your laptop
#
# It registers no MCP server and does not touch .mcp.json. forge.py reaches the
# platform over plain HTTPS with the key from your credentials file, so nothing
# else has to be connected to run models.
#
# Nothing here reads or writes your credentials file.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/nebius-scientific-ai"
BUNDLE="$(ls "$HERE"/scientific-ai-skills-*.tar.gz 2>/dev/null | head -1 || true)"

if [ $# -ge 1 ]; then
  mkdir -p "$1"
  DEST="$(cd "$1" && pwd)/.claude/skills"
else
  DEST="$HOME/.claude/skills"
fi

if [ ! -f "$SRC/SKILL.md" ]; then
  echo "error: cannot find $SRC/SKILL.md" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is not installed" >&2
  exit 1
fi

mkdir -p "$DEST"

# 1. the official bundle, if it shipped with this kit
if [ -n "$BUNDLE" ]; then
  if [ -f "$HERE/SHA256SUMS" ]; then
    if command -v shasum >/dev/null 2>&1; then
      ( cd "$HERE" && shasum -a 256 -c SHA256SUMS >/dev/null 2>&1 ) \
        && echo "Bundle checksum ok" \
        || { echo "error: bundle checksum does not match SHA256SUMS" >&2; exit 1; }
    fi
  fi
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  tar -xzf "$BUNDLE" -C "$TMP"
  ( cd "$TMP/scientific-ai" \
    && python3 bundle.py verify >/dev/null \
    && python3 bundle.py install --dest "$DEST" >/dev/null )
  echo "Installed the official scientific-ai bundle (51 skills) in $DEST"
else
  echo "Note: no scientific-ai bundle found next to this script, skipping it."
fi

# 2. the client this kit adds
rm -rf "$DEST/nebius-scientific-ai"
cp -R "$SRC" "$DEST/nebius-scientific-ai"
rm -rf "$DEST/nebius-scientific-ai/__pycache__"
chmod +x "$DEST/nebius-scientific-ai/forge.py"
echo "Installed nebius-scientific-ai in $DEST"

echo
echo "Next:"
echo "  1. export NEBIUS_CREDS=/path/to/credentials.private.json"
echo "  2. start claude in your project folder"
echo "  3. ask: list the Nebius models I can use"
