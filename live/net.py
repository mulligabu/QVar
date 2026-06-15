"""Pooled HTTP for the live engine: keep-alive connections + health counters.

Every feed/venue call used to pay a fresh TCP+TLS handshake (55-280ms measured)
through one-shot urllib. This module keeps one persistent connection per
(thread, host) — worker threads from the engine's fan-out pool each get their
own, so no locking on the wire — and records per-host latency/error counters
that the engine surfaces in status.json (the API error budget).

Stdlib only. Fail-safe: any transport weirdness falls back to one-shot urllib
so a keep-alive edge case can never be worse than the old behaviour.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque

UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept": "application/json",
}

_local = threading.local()
_stats_lock = threading.Lock()
# host -> [calls, errors, total_ms, deque(recent ms), last_error_str]
_stats: dict[str, list] = defaultdict(lambda: [0, 0, 0.0, deque(maxlen=50), ""])


def _record(host: str, ms: float, err: str | None):
    with _stats_lock:
        s = _stats[host]
        s[0] += 1
        s[2] += ms
        s[3].append(ms)
        if err:
            s[1] += 1
            s[4] = err[:120]


def stats() -> dict:
    """Per-host API health snapshot for status.json: call count, error count,
    mean + recent p95 latency, last error."""
    out = {}
    with _stats_lock:
        for host, (n, errs, total, recent, last_err) in _stats.items():
            rec = sorted(recent)
            out[host] = {
                "calls": n, "errors": errs,
                "err_rate": round(errs / n, 4) if n else 0.0,
                "mean_ms": round(total / n, 1) if n else None,
                "p95_ms": round(rec[int(0.95 * (len(rec) - 1))], 1) if rec else None,
                "last_error": last_err or None,
            }
    return out


def _conn_for(scheme: str, host: str, timeout: float) -> http.client.HTTPConnection:
    pool = getattr(_local, "pool", None)
    if pool is None:
        pool = _local.pool = {}
    key = f"{scheme}://{host}"
    conn = pool.get(key)
    if conn is None:
        cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        conn = cls(host, timeout=timeout)
        pool[key] = conn
    conn.timeout = timeout  # respect the tightest caller without reconnecting
    return conn


def _drop_conn(scheme: str, host: str):
    pool = getattr(_local, "pool", None)
    if pool:
        conn = pool.pop(f"{scheme}://{host}", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _urllib_fallback(url: str, data: bytes | None, timeout: float, headers: dict):
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_json(url: str, data: bytes | None = None, timeout: float = 4.0,
             headers: dict | None = None):
    """GET (or POST when `data` given) returning parsed JSON, over a per-thread
    keep-alive connection. One transparent retry on a fresh connection (stale
    keep-alive sockets die between cycles), then a one-shot urllib fallback.
    Raises on HTTP >= 400 like urllib did, so existing callers are unchanged."""
    u = urllib.parse.urlsplit(url)
    hdrs = {**UA, **(headers or {})}
    if data is not None:
        hdrs.setdefault("Content-Type", "application/json")
    path = (u.path or "/") + (f"?{u.query}" if u.query else "")
    method = "POST" if data is not None else "GET"
    t0 = time.monotonic()
    err = None
    try:
        for attempt in (0, 1):
            conn = _conn_for(u.scheme, u.netloc, timeout)
            try:
                conn.request(method, path, body=data, headers=hdrs)
                resp = conn.getresponse()
                body = resp.read()  # always drain so the connection stays reusable
                if resp.status >= 400:
                    raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, None)
                return json.loads(body)
            except urllib.error.HTTPError:
                raise  # real HTTP status error — do not retry (HTTPError IS an OSError)
            except (http.client.HTTPException, ConnectionError, BrokenPipeError, OSError) as e:
                # transport-level: stale keep-alive socket — retry once on a fresh conn
                _drop_conn(u.scheme, u.netloc)
                if attempt == 1:
                    err = f"{type(e).__name__}: {e}"
                    return _urllib_fallback(url, data, timeout, hdrs)
            except Exception:
                _drop_conn(u.scheme, u.netloc)
                raise
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        _record(u.netloc, (time.monotonic() - t0) * 1000.0, err)
