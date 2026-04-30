import json

import pytest

from paperforge import sandbox


def _write(tmp_path, code):
    (tmp_path / "main.py").write_text(code)


def test_success_captures_output_and_files(tmp_path):
    _write(tmp_path, "import json; print('hi'); json.dump({'m': 1.5}, open('metrics.json','w'))")
    r = sandbox.run(["python", "main.py"], tmp_path, timeout_s=30, backend="subprocess")
    assert r.ok and r.stdout.strip() == "hi" and r.backend == "subprocess"
    assert json.loads((tmp_path / "metrics.json").read_text()) == {"m": 1.5}


def test_failure_reports_returncode_and_stderr(tmp_path):
    _write(tmp_path, "raise SystemExit('boom')")
    r = sandbox.run(["python", "main.py"], tmp_path, timeout_s=30, backend="subprocess")
    assert not r.ok and r.returncode == 1 and "boom" in r.stderr


def test_timeout_kills_process(tmp_path):
    _write(tmp_path, "import time\nwhile True: time.sleep(0.1)")
    r = sandbox.run(["python", "main.py"], tmp_path, timeout_s=1, backend="subprocess")
    assert r.timed_out and not r.ok and r.returncode is None
    assert r.duration_s < 10


def test_network_is_blocked(tmp_path):
    _write(tmp_path, "import urllib.request\nurllib.request.urlopen('http://example.com', timeout=5)")
    r = sandbox.run(["python", "main.py"], tmp_path, timeout_s=30, backend="subprocess")
    assert not r.ok and "network access is disabled" in r.stderr


def test_env_is_scrubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    _write(tmp_path, "import os; print(os.environ.get('ANTHROPIC_API_KEY'))")
    r = sandbox.run(["python", "main.py"], tmp_path, timeout_s=30, backend="subprocess")
    assert r.stdout.strip() == "None"


def test_unknown_backend(tmp_path):
    with pytest.raises(ValueError):
        sandbox.run(["python", "-V"], tmp_path, backend="vm")


@pytest.mark.skipif(not sandbox.docker_available(), reason="docker sandbox image not built")
def test_docker_backend_has_no_network(tmp_path):
    _write(tmp_path, "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)")
    r = sandbox.run(["python", "main.py"], tmp_path, timeout_s=120, backend="docker")
    assert not r.ok
