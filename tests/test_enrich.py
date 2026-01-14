from paperforge.enrich import apply_hf, fetch_citations, fetch_hf_daily, find_code_link
from paperforge.models import Paper


def _p(i, abstract="", comment=""):
    return Paper(arxiv_id=i, title="t", abstract=abstract, authors=["a"], categories=["cs.LG"],
                 published="2026-09-24T00:00:00Z", comment=comment, sources=["arxiv"])


def test_hf_daily_parsing_and_apply():
    payload = [{"paper": {"id": "2609.1", "upvotes": 7, "githubRepo": "https://github.com/x/y"}},
               {"paper": {"id": "2609.2", "upvotes": 1, "githubRepo": None}}]
    hf = fetch_hf_daily(getter=lambda url: payload)
    papers = [_p("2609.1"), _p("2609.2", comment="Code: https://github.com/me/repo."), _p("2609.3")]
    apply_hf(papers, hf)
    assert papers[0].hf_upvotes == 7 and papers[0].code_url == "https://github.com/x/y"
    assert "huggingface" in papers[0].sources
    assert papers[1].code_url == "https://github.com/me/repo"  # falls back to comment link
    assert papers[2].hf_upvotes is None and papers[2].code_url == ""


def test_hf_failure_is_non_fatal():
    def boom(url):
        raise OSError("offline")
    assert fetch_hf_daily(getter=boom) == {}


def test_find_code_link():
    assert find_code_link(_p("1", abstract="see https://github.com/a/b-c for code")) == "https://github.com/a/b-c"


def test_citations_batch():
    papers = [_p("1"), _p("2")]
    fetch_citations(papers, poster=lambda body: [{"citationCount": 5}, None])
    assert papers[0].citations == 5 and papers[1].citations is None
