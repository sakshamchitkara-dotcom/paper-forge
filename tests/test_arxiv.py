from datetime import datetime, timezone

from paperforge.arxiv import ArxivClient, parse_feed


def test_parse_feed_fields(feed_xml):
    papers = parse_feed(feed_xml)
    assert [p.arxiv_id for p in papers] == ["1412.6980", "2609.00001", "2601.00002"]
    adam = papers[0]
    assert adam.version == "v9"
    assert adam.title == "Adam: A Method for Stochastic Optimization"  # whitespace collapsed
    assert adam.authors == ["Diederik P. Kingma", "Jimmy Ba"]
    assert adam.comment.startswith("Published as a conference paper")
    assert adam.pdf_url == "https://arxiv.org/pdf/1412.6980v9"
    big = papers[1]
    assert big.categories == ["cs.CL", "cs.LG"]
    assert big.doi == "10.1234/example"
    assert big.journal_ref.startswith("Journal of Examples")


def test_parse_feed_skips_error_entries():
    xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/api/errors#incorrect_id_format_for_x</id>
      <title>Error</title><summary>incorrect id format</summary></entry></feed>"""
    assert parse_feed(xml) == []


def test_client_rate_limit_and_lookback(feed_xml, monkeypatch):
    urls, sleeps = [], []
    monkeypatch.setattr("paperforge.arxiv.time.sleep", sleeps.append)
    client = ArxivClient(delay_s=3.0, opener=lambda u: urls.append(u) or feed_xml)
    now = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
    got = client.recent("cs.LG", max_results=10, lookback_days=2, now=now)
    assert [p.arxiv_id for p in got] == ["1412.6980", "2609.00001"]
    client.recent("stat.ML", now=now)
    assert "search_query=cat%3Acs.LG" in urls[0]
    assert "sortBy=submittedDate" in urls[0]
    assert len(sleeps) == 1 and 2.5 < sleeps[0] <= 3.0  # second call waited ~3s


def test_client_retries_transient_http_errors(feed_xml, monkeypatch):
    import urllib.error

    monkeypatch.setattr("paperforge.arxiv.time.sleep", lambda s: None)
    calls = []

    def flaky(url):
        calls.append(url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(url, 406, "Not Acceptable", {}, None)
        return feed_xml

    assert len(ArxivClient(opener=flaky).query("cat:cs.LG")) == 3
    assert len(calls) == 3


def test_client_does_not_retry_client_errors(monkeypatch):
    import urllib.error

    import pytest

    def bad(url):
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        ArxivClient(opener=bad, delay_s=0).query("x")


def test_retry_honours_retry_after_header(feed_xml, monkeypatch):
    import urllib.error

    sleeps, calls = [], []
    monkeypatch.setattr("paperforge.arxiv.time.sleep", sleeps.append)

    def limited(url):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 503, "busy", {"Retry-After": "20"}, None)
        return feed_xml

    ArxivClient(opener=limited, delay_s=3.0).query("cat:cs.LG")
    assert sleeps and 19 < sleeps[-1] <= 20  # waited what arXiv asked, not the 6s backoff


def test_user_agent_carries_version_and_contact(monkeypatch):
    from paperforge import __version__
    from paperforge.arxiv import user_agent

    monkeypatch.delenv("FORGE_CONTACT", raising=False)
    assert user_agent().startswith(f"paper-forge/{__version__} (+https://github.com/")
    monkeypatch.setenv("FORGE_CONTACT", "ops@example.org")
    assert user_agent().endswith("; mailto:ops@example.org)")


def test_parse_listing_keeps_first_announcements_only():
    from pathlib import Path

    from paperforge.arxiv import parse_listing

    papers = parse_listing((Path(__file__).parent / "fixtures" / "arxiv_rss.xml").read_text())
    assert [p.arxiv_id for p in papers] == ["2609.28553"]  # v3 cross-list and replacement dropped
    p = papers[0]
    assert p.title.startswith("SMILESGNN") and p.version == "v1"
    assert p.abstract.startswith("Drug toxicity prediction")  # announce preamble stripped
    assert p.authors[:2] == ["Quang Minh Nguyen", "Thuy Quynh Nguyen"] and len(p.authors) == 6
    assert p.categories == ["cs.LG", "cs.AI"]
    assert p.doi == "10.1109/MAPR72750.2026.11685822" and p.journal_ref.startswith("2026 International")
    assert p.published.startswith("2026-09-25") and p.sources == ["arxiv-listing"]
