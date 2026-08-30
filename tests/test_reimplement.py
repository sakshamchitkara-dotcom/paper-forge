import json
from types import SimpleNamespace

import pytest

from paperforge.reimplement import _safe_write, claims_from_analysis, list_references, run_claude, run_scripted

CFG = {"sandbox": "subprocess", "timeout_s": 120, "max_iterations": 3}


def test_references_are_labeled_and_complete():
    refs = list_references()
    assert set(refs) == {"1206.1901", "0909.4061", "1412.6980", "2010.09649"}
    for r in refs.values():
        assert "Hand-written reference implementation" in r["reference_note"]
        assert (r["dir"] / "claims.json").exists() and (r["dir"] / "fidelity.md").exists()
        assert (r["dir"] / "impl" / "experiment.py").exists()
        for c in json.loads((r["dir"] / "claims.json").read_text()):
            assert c["source"], "every claim cites where it comes from"


@pytest.mark.parametrize("arxiv_id", ["0909.4061", "1206.1901", "1412.6980", "2010.09649"])
def test_scripted_end_to_end(tmp_path, arxiv_id):
    res = run_scripted(arxiv_id, tmp_path, CFG)
    assert res["tests_passed"] is True and res["metrics"]
    assert res["status"] in {"reproduced", "partial", "not_reproduced"}
    assert (tmp_path / "implementation" / "metrics.json").exists()
    assert (tmp_path / "implementation" / "run.log").exists()
    assert res["fidelity_notes"]


def test_scripted_unknown_paper(tmp_path):
    with pytest.raises(KeyError):
        run_scripted("9999.99999", tmp_path, CFG)


def test_safe_write_rejects_escapes(tmp_path):
    for bad in ("../x.py", "/etc/passwd", "a/../../b.py", ".hidden"):
        with pytest.raises(ValueError):
            _safe_write(tmp_path, [{"path": bad, "content": ""}])
    _safe_write(tmp_path, [{"path": "pkg/mod.py", "content": "x = 1"}])
    assert (tmp_path / "pkg" / "mod.py").read_text() == "x = 1"


def test_claims_from_analysis_unique_keys():
    a = {"source": "claude", "claimed_metrics": [
        {"name": "Test Accuracy", "value": 91.0, "context": "Table 1"},
        {"name": "test accuracy", "value": 88.0, "context": "Table 2"}]}
    assert [c["name"] for c in claims_from_analysis(a)] == ["test_accuracy", "test_accuracy_2"]


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class FakeStreamClient:
    """Returns queued file sets as if Claude produced them."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = self

    def stream(self, **kw):
        self.calls.append(kw)
        text = json.dumps(self.replies.pop(0))
        return _Stream(SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)]))


BROKEN = {"files": [{"path": "experiment.py", "content": "raise RuntimeError('oops')"}], "deviations": []}
FIXED = {"files": [
    {"path": "method.py", "content": "def f():\n    return 0.9\n"},
    {"path": "test_method.py", "content": "from method import f\n\ndef test_f():\n    assert f() == 0.9\n"},
    {"path": "experiment.py", "content": "import json\nfrom method import f\njson.dump({'accuracy': f()}, open('metrics.json','w'))\n"},
], "deviations": ["synthetic data"]}


def test_claude_loop_retries_on_failure_then_succeeds(tmp_path):
    client = FakeStreamClient([BROKEN, FIXED])
    paper = {"arxiv_id": "2609.1", "title": "T", "abstract": "A"}
    analysis = {"source": "claude", "claimed_metrics": [{"name": "accuracy", "value": 0.9, "context": "Tab 1"}]}
    res = run_claude(paper, analysis, tmp_path, CFG, {"model": "claude-opus-5-5", "effort": "high"}, client)
    assert res["status"] == "reproduced" and res["attempts"] == 2
    assert len(client.calls) == 2
    feedback = client.calls[1]["messages"][-1]["content"]
    assert "oops" in feedback  # stderr was fed back
    assert client.calls[0]["output_config"]["format"]["type"] == "json_schema"
    assert "- synthetic data" in res["fidelity_notes"]


def test_claude_loop_gives_up_and_reports_failed(tmp_path):
    client = FakeStreamClient([BROKEN, BROKEN, BROKEN])
    res = run_claude({"arxiv_id": "1", "title": "T"}, {"claimed_metrics": []}, tmp_path, CFG,
                     {"model": "claude-opus-5-5"}, client)
    assert res["status"] == "failed" and res["attempts"] == 3 and "oops" in res["error"]
