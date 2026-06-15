#!/usr/bin/env bash
# Boot/auto-start the live forward engines in a detached tmux session 'crypto'.
#
# ⚠️ ENGINE CONFIGS LIVE IN live/engines.conf (single source of truth) — one tmux window
# per conf line, plus the market-data service and the dashboard. Add/retire/re-band engines
# by editing the conf, NEVER by hand-typing flags into tmux (a hand-typed flag that differs
# from the conf is exactly what the config-drift guard exists to catch).
#
# RESUMES the existing DB + model_state on every launch (equity + online model are
# continuous across reboots — forward_engine reloads settlement_totals() and
# model_state.json). It does NOT archive / cold-start to $1000 — that destructive reset
# is restart_engines.sh's job ONLY. Do not point this at archiving logic.
#
# Idempotent: if the 'crypto' session already exists, it leaves it alone and exits 0, so
# it is safe to run from systemd on boot AND by hand. Invoked by crypto-engines.service.
set -u
REPO="/home/user/crypto_v2"
PY="$REPO/.venv/bin/python"
SESSION="crypto"
CONF="$REPO/live/engines.conf"
cd "$REPO" || exit 1

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already running; nothing to do."
  exit 0
fi

COMMON="--assets BTC,ETH,SOL,XRP,DOGE,BNB,HYPE --minutes 0 --interval 8 --anchor on"

started=""
first=1
while IFS='|' read -r name dir flags; do
  case "$name" in ''|\#*) continue;; esac
  mkdir -p "$REPO/$dir"
  cmd="cd $REPO && exec $PY -u -m live.forward_engine $COMMON --name $name --data-dir $dir $flags 2>&1 | tee -a $dir/engine.log"
  if [ "$first" = 1 ]; then
    tmux new-session -d -s "$SESSION" -n "$name" "$cmd"
    first=0
  else
    tmux new-window -t "$SESSION" -n "$name" "$cmd"
  fi
  started="$started $name"
done < "$CONF"

if [ "$first" = 1 ]; then
  echo "ERROR: no engines parsed from $CONF" >&2
  exit 1
fi

# MARKET-DATA service — ONE process owns all external market I/O (WebSocket-first:
# Polymarket CLOB WSS books, Coinbase/Binance spot WS, Chainlink round sequence via
# batched RPC + AnswerUpdated logs -> authoritative strike table, shared 60s discovery)
# republished over data/md.sock (UDS pub/sub) + data/md.db snapshot. Engines consume it
# via live/md_client adapters (per-lane --md flag in engines.conf) with AUTOMATIC direct-
# REST fallback: killing this window degrades engines to polling, it never stops them.
tmux new-window -t "$SESSION" -n md \
  "cd $REPO && exec $PY -u -m live.md_service \
     --assets BTC,ETH,SOL,XRP,DOGE,BNB,HYPE 2>&1 | tee -a data/md_service.log"

# DASHBOARD — http://127.0.0.1:8011 (one tab per engine view)
tmux new-window -t "$SESSION" -n dash \
  "cd $REPO && exec $PY -u -m live.dashboard_server 2>&1 | tee -a data/dashboard.log"

echo "started tmux session '$SESSION' (engines:$started + md + dash)."
