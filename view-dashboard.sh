#!/usr/bin/env bash
# View the Team 16 HER2 3D dashboard, served live from the VM.
# Usage:  ./view-dashboard.sh        (Ctrl-C to disconnect)
# Needs:  the shared VM key at ~/.ssh/team-16  (override: TEAM16_KEY=/path ./view-dashboard.sh)
set -euo pipefail

VM=hackathon@195.242.31.250
KEY="${TEAM16_KEY:-$HOME/.ssh/team-16}"
LPORT="${LPORT:-8899}"                 # local port in your browser
RPORT=$(( 8900 + RANDOM % 95 ))        # your own port on the VM, so teammates don't clash
URL="http://localhost:${LPORT}/dashboard.html"

[ -f "$KEY" ] || { echo "VM key not found at $KEY (set TEAM16_KEY=/path/to/team-16)"; exit 1; }

echo "Connecting to the VM and starting the dashboard (VM port $RPORT -> your localhost:$LPORT)…"
( for _ in $(seq 1 30); do
    if curl -fsS -o /dev/null "$URL" 2>/dev/null; then
      echo "Dashboard ready: $URL"
      (xdg-open "$URL" 2>/dev/null || open "$URL" 2>/dev/null) || echo "Open it manually: $URL"
      break
    fi
    sleep 1
  done ) &

# Server lifetime is tied to this SSH connection: close it (Ctrl-C) and the VM server stops.
exec ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -L "${LPORT}:127.0.0.1:${RPORT}" "$VM" \
  "cd ~/team16-ai-x-antibodies/viz && exec python3 -m http.server ${RPORT} --bind 127.0.0.1"
