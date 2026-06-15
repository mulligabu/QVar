"""Dashboard control endpoints: GET /control.json, POST toggles, trip clear."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import live.dashboard_server as ds
from live.control import load_control, read_trip, trip


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "DATA_ROOT", tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ds.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", tmp_path
    srv.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def _post(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def test_control_json_serves_state_and_aggregates(server):
    url, _root = server
    c = _get(f"{url}/control.json")
    assert c["control"]["max_loss"]["enabled"] is True
    assert c["tripped"] is None and c["halt_file"] is False
    assert "agg" in c and c["agg"]["lanes"] == []


def test_post_set_toggles_and_adjusts(server):
    url, root = server
    _post(f"{url}/control/set", {"max_loss": {"limit": 75.0},
                                 "exposure_cap": {"enabled": False, "max_total_frac": 0.2}})
    c = load_control(root)
    assert c["max_loss"]["limit"] == 75.0 and c["max_loss"]["enabled"] is True
    assert c["exposure_cap"]["enabled"] is False
    assert c["exposure_cap"]["max_total_frac"] == 0.2


def test_post_estop_stamps_reason_and_clears_on_resume(server):
    url, root = server
    _post(f"{url}/control/set", {"estop": True})
    c = load_control(root)
    assert c["estop"] is True
    assert c["estop_reason"] == "operator (dashboard)" and c["estop_ts"]
    _post(f"{url}/control/set", {"estop": False})
    c = load_control(root)
    assert c["estop"] is False and c["estop_reason"] is None


def test_post_clear_trip(server):
    url, root = server
    trip(root, "max-loss test", -200.0, 150.0)
    assert _get(f"{url}/control.json")["tripped"]["day_pnl"] == -200.0
    _post(f"{url}/control/clear-trip")
    assert read_trip(root) is None
