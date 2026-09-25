import json

from paperforge.cli import main


def test_reimplement_scripted_and_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "forge.toml").write_text('[reimplement]\nsandbox = "subprocess"\n')
    assert main(["references"]) == 0
    assert main(["reimplement", "0909.4061", "--backend", "scripted", "--day", "2026-09-25"]) == 0
    res = json.loads((tmp_path / "site/papers/0909.4061/results.json").read_text())
    assert res["backend"] == "scripted" and res["status"] == "reproduced"
    assert (tmp_path / "site/daily/2026-09-25.html").exists()
    assert "0909.4061" in (tmp_path / "site/leaderboard.html").read_text()
    assert main(["report"]) == 0 and (tmp_path / "site/.nojekyll").exists()
