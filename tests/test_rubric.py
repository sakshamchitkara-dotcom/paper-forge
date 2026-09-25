from paperforge.arxiv import parse_feed
from paperforge.config import DEFAULTS
from paperforge.rubric import score_paper

W = DEFAULTS["screening"]["weights"]
KW = DEFAULTS["screening"]["interest_keywords"]


def test_small_method_outranks_giant_pretraining(feed_xml):
    adam, giant, _ = parse_feed(feed_xml)
    a = score_paper(adam, W, KW)
    g = score_paper(giant, W, KW)
    assert a.total > g.total
    assert any("billion" in e for e in g.criteria["compute"].evidence)
    assert any("GPU" in e for e in g.criteria["compute"].evidence)
    assert any("web-scale" in e for e in g.criteria["data"].evidence)
    assert any("simplicity" in e for e in a.criteria["clarity"].evidence)


def test_code_and_fulltext_signals(feed_xml):
    adam = parse_feed(feed_xml)[0]
    base = score_paper(adam, W, KW)
    adam.code_url = "https://github.com/x/y"
    boosted = score_paper(adam, W, KW, fulltext="Algorithm 1: Adam, our proposed algorithm")
    assert boosted.criteria["code"].score == 1.0
    assert boosted.criteria["clarity"].score > base.criteria["clarity"].score
    assert boosted.total > base.total


def test_scores_are_bounded_and_serialisable(feed_xml):
    for p in parse_feed(feed_xml):
        d = score_paper(p, W, KW).to_dict()
        assert 0 <= d["total"] <= 100
        assert all(0 <= c["score"] <= 1 for c in d["criteria"].values())
