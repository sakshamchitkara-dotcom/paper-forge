from datetime import datetime, timezone

from paperforge.arxiv import ArxivClient
from paperforge.config import DEFAULTS, _merge
from paperforge.screen import screen
from paperforge.store import Store


def test_screen_end_to_end_offline(feed_xml, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr("paperforge.arxiv.datetime",
                        type("D", (datetime,), {"now": staticmethod(lambda tz=None: datetime(2026, 9, 25, tzinfo=timezone.utc))}))
    cfg = _merge(DEFAULTS, {"sources": {"categories": ["cs.LG"]}, "screening": {"top_k": 1}})
    client = ArxivClient(delay_s=0, opener=lambda url: feed_xml)
    hf = [{"paper": {"id": "1412.6980", "upvotes": 3, "githubRepo": None}}]
    store = Store(":memory:")
    rows = screen(cfg, store, "2026-09-25", arxiv=client,
                  fulltext_getter=lambda url: b"<p>Algorithm 1</p>" * 500, hf_getter=lambda url: hf)
    assert [r["arxiv_id"] for r in rows] == ["1412.6980", "2609.00001"]  # old paper filtered out
    assert rows[0]["analysis"]["source"] == "heuristic"  # top-1 analyzed
    assert rows[1]["analysis"] is None
    assert rows[0]["paper"]["hf_upvotes"] == 3
    # second run the same day: nothing new
    assert screen(cfg, store, "2026-09-26", arxiv=client, fulltext_getter=lambda u: b"",
                  hf_getter=lambda url: []) == []
