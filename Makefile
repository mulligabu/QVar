# Build/quality gates for the live forward-prediction engine.
# `make check` is the pre-merge gate: tests + lint + types.
PY := .venv/bin/python

.PHONY: test lint type check fix status engines-status

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check live/ tests/ research/ tools/

type:
	$(PY) -m mypy

check: test lint type

fix:
	$(PY) -m ruff check live/ tests/ --fix

# quick operational views (read-only; safe while engines run)
status:
	@for d in data/forward_live*/; do \
	  echo "== $$d"; \
	  $(PY) -c "import json;s=json.load(open('$$d/status.json'));p=s['performance'];print(' updated',s['updated']);print(' equity',p['equity'],'settled',p['settled'],'winrate',p['winrate'],'fills',p['fills'],'no_fills',p['no_fills'])" 2>/dev/null || echo "  (no status.json)"; \
	done

engines-status:
	@tmux list-windows -t crypto 2>/dev/null || echo "tmux session 'crypto' not running"
	@pgrep -af "live.forward_engine" | sed 's/.*python/  python/' || true
