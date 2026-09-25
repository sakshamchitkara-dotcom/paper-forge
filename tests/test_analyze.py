import json
from types import SimpleNamespace

from paperforge.analyze import ANALYSIS_SCHEMA, analyze, heuristic_analysis
from paperforge.arxiv import parse_feed
from paperforge.config import DEFAULTS
from paperforge.rubric import score_paper


def _setup(feed_xml):
    p = parse_feed(feed_xml)[0]
    p.abstract += " It reaches 97.5% accuracy on MNIST."
    r = score_paper(p, DEFAULTS["screening"]["weights"], []).to_dict()
    return p, r


def test_heuristic_matches_schema(feed_xml, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    p, r = _setup(feed_xml)
    out = analyze(p, "We minimise $m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t$ here.", r, DEFAULTS)
    assert out["source"] == "heuristic"
    assert set(ANALYSIS_SCHEMA["required"]) <= set(out)
    assert out["claimed_metrics"][0]["value"] == 97.5 and out["claimed_metrics"][0]["unit"] == "%"
    assert "MNIST" in out["datasets"]
    assert out["key_equations"] == ["m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t"]


class FakeClaude:
    def __init__(self, payload=None, stop="end_turn", exc=None):
        self.payload, self.stop, self.exc, self.calls = payload, stop, exc, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(stop_reason=self.stop,
                               content=[SimpleNamespace(type="text", text=json.dumps(self.payload))])


def test_claude_path_uses_structured_output(feed_xml):
    p, r = _setup(feed_xml)
    payload = heuristic_analysis(p, "", r)
    payload.pop("source")
    payload["method_summary"] = "from claude"
    fake = FakeClaude(payload)
    out = analyze(p, "x" * 70000, r, DEFAULTS, client=fake)
    assert out["method_summary"] == "from claude"
    assert out["source"] == "claude:claude-opus-5-5"
    kw = fake.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["format"]["schema"] is ANALYSIS_SCHEMA
    assert "thinking" not in kw  # Opus 5.5: thinking is always adaptive; effort is the control
    assert any("first 60000" in x for x in out["risks"])  # truncation is disclosed


def test_claude_refusal_or_error_falls_back(feed_xml):
    p, r = _setup(feed_xml)
    out = analyze(p, "", r, DEFAULTS, client=FakeClaude({}, stop="refusal"))
    assert out["source"] == "heuristic" and any("failed" in x for x in out["risks"])
    out = analyze(p, "", r, DEFAULTS, client=FakeClaude(exc=ConnectionError("down")))
    assert out["source"] == "heuristic"
