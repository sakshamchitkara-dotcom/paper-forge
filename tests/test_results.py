from paperforge.results import build_results, compare, decide_status
from paperforge.sandbox import RunResult

CLAIMS = [
    {"name": "acc", "claimed": 0.9, "tolerance": 0.02},
    {"name": "ratio", "claimed": 1.0, "kind": "at_most"},
    {"name": "beats_sgd", "kind": "qualitative", "claimed": True},
]


def _run(ok=True, timed_out=False):
    return RunResult(["python", "x"], 0 if ok else 1, "", "Traceback\nValueError: bad", 1.0, timed_out, "subprocess")


def test_compare_kinds():
    got = {c["name"]: c["verdict"] for c in compare(CLAIMS, {"acc": 0.91, "ratio": 0.4, "beats_sgd": True})}
    assert got == {"acc": "match", "ratio": "match", "beats_sgd": "match"}
    got = {c["name"]: c["verdict"] for c in compare(CLAIMS, {"acc": 0.5, "ratio": 1.2})}
    assert got == {"acc": "mismatch", "ratio": "mismatch", "beats_sgd": "missing"}
    assert compare([{"name": "x", "claimed": 10.0}], {"x": 10.9})[0]["verdict"] == "match"  # default 10% rel
    assert compare([{"name": "x", "claimed": 1.0}], {"x": float("nan")})[0]["verdict"] == "mismatch"
    assert compare([{"name": "x", "claimed": 1.0}], {"x": True})[0]["verdict"] == "mismatch"


def test_status_rules():
    match = [{"verdict": "match"}]
    assert decide_status(False, {"a": 1}, match, True) == "failed"
    assert decide_status(True, None, [], True) == "failed"
    assert decide_status(True, {"a": 1}, [], True) == "unverified"
    assert decide_status(True, {"a": 1}, match, True) == "reproduced"
    assert decide_status(True, {"a": 1}, match, False) == "partial"  # failing tests never "reproduced"
    assert decide_status(True, {"a": 1}, match + [{"verdict": "mismatch"}], True) == "partial"
    assert decide_status(True, {"a": 1}, [{"verdict": "missing"}], True) == "not_reproduced"


def test_build_results_never_claims_success_without_metrics(tmp_path):
    paper = {"arxiv_id": "1", "title": "t"}
    r = build_results(paper=paper, backend="scripted", claims=CLAIMS, experiment=_run(), tests=_run(),
                      metrics_path=tmp_path / "metrics.json", attempts=1, sandbox="subprocess")
    assert r["status"] == "failed" and r["error"] == "no metrics written to metrics.json"
    (tmp_path / "metrics.json").write_text('{"acc": 0.9, "ratio": 0.5, "beats_sgd": true}')
    r = build_results(paper=paper, backend="scripted", claims=CLAIMS, experiment=_run(ok=False), tests=None,
                      metrics_path=tmp_path / "metrics.json", attempts=3, sandbox="subprocess")
    assert r["status"] == "failed" and r["error"] == "ValueError: bad"  # stale metrics ignored
    r = build_results(paper=paper, backend="scripted", claims=CLAIMS, experiment=_run(), tests=_run(),
                      metrics_path=tmp_path / "metrics.json", attempts=1, sandbox="subprocess")
    assert r["status"] == "reproduced" and r["paper"]["url"] == "https://arxiv.org/abs/1"
