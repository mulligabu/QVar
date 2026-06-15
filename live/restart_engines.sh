#!/usr/bin/env bash
# DESTRUCTIVE fresh-reset of the live forward engines: $1000 bankroll, cold model.
# Old DB + model state are ARCHIVED (not deleted) per data dir. Pass KEEP_MODEL=1 to
# keep the warmed model state. For a resume-mode (re)launch use boot_engines.sh instead.
#
# ⚠️ ENGINE CONFIGS LIVE IN live/engines.conf (single source of truth, shared with
# boot_engines.sh). This script archives + relaunches every engine listed there, inside
# the tmux session 'crypto' (one window per engine) so the result is identical to a boot.
set -u
cd "$(dirname "$0")/.."
REPO="$(pwd)"
CONF="$REPO/live/engines.conf"
SESSION="crypto"
TS=$(date -u +%Y%m%dT%H%M%SZ)

echo ">> stopping existing engines + tmux session"
tmux kill-session -t "$SESSION" 2>/dev/null
pkill -f "live.forward_engine" 2>/dev/null
sleep 2

while IFS='|' read -r name dir flags; do
  case "$name" in ''|\#*) continue;; esac
  arch="$dir/archive_$TS"; mkdir -p "$arch"
  for f in forward_engine.db forward_engine.db-wal forward_engine.db-shm status.json; do
    [ -e "$dir/$f" ] && mv "$dir/$f" "$arch/" 2>/dev/null
  done
  if [ "${KEEP_MODEL:-0}" = "1" ]; then
    [ -e "$dir/model_state.json" ] && cp "$dir/model_state.json" "$arch/"   # keep live, back up
  else
    [ -e "$dir/model_state.json" ] && mv "$dir/model_state.json" "$arch/"   # cold start
  fi
  mv "$dir"/preTWEAK_*.db "$arch/" 2>/dev/null
  echo "   archived $dir -> $arch"
done < "$CONF"

echo ">> relaunching via boot_engines.sh (reads $CONF)"
bash "$REPO/live/boot_engines.sh"

sleep 4
echo ">> running engines:"
pgrep -af "live.forward_engine" | sed 's/.*python/  python/'
