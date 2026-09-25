from paperforge.fulltext import fetch_fulltext, html_to_text
from paperforge.models import Paper

HTML = """<html><head><style>.x{}</style><script>var a=1;</script></head><body>
<nav>menu</nav><h2>1 Method</h2><p>We minimise
<math alttext="f(\\theta)"><mi>f</mi></math> with Algorithm 1.</p></body></html>"""


def test_html_to_text_keeps_math_drops_chrome():
    text = html_to_text(HTML)
    assert "menu" not in text and "var a" not in text
    assert "1 Method" in text
    assert "$f(\\theta)$" in text and "Algorithm 1" in text


def test_fetch_falls_back_to_pdf(monkeypatch):
    p = Paper("1234.5678", "t", "a", ["x"], ["cs.LG"], "2026-01-01")
    monkeypatch.setattr("paperforge.fulltext.pdf_to_text", lambda b: "pdf body")
    calls = []

    def getter(url):
        calls.append(url)
        if "/html/" in url:
            return b"<html><body>stub</body></html>"
        return b"%PDF"

    assert fetch_fulltext(p, getter=getter) == ("pdf body", "pdf")
    assert calls == [p.html_url, p.pdf_url]
