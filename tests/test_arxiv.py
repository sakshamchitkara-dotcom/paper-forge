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
