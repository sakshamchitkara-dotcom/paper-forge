"""Fetch paper full text: arXiv HTML (LaTeXML) first, PDF via pypdf as fallback."""

from __future__ import annotations

import io
import logging
from html.parser import HTMLParser

from . import http
from .models import Paper

log = logging.getLogger(__name__)
_SKIP = {"script", "style", "nav", "header", "footer", "button"}
_BLOCK = {"p", "div", "section", "h1", "h2", "h3", "h4", "li", "tr", "br", "figcaption", "table"}


class _TextExtractor(HTMLParser):
    """Collect visible text; render <math> as its LaTeX alttext so equations survive."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self._math = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        elif tag == "math":
            self._math += 1
            alt = dict(attrs).get("alttext")
            if alt and not self._skip:
                self.parts.append(f" ${alt}$ ")
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP and self._skip:
            self._skip -= 1
        elif tag == "math" and self._math:
            self._math -= 1

    def handle_data(self, data):
        if not self._skip and not self._math:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    ex = _TextExtractor()
    ex.feed(html)
    lines = (" ".join(line.split()) for line in "".join(ex.parts).splitlines())
    return "\n".join(line for line in lines if line)


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def fetch_fulltext(paper: Paper, getter=http.get) -> tuple[str, str]:
    """Return (text, source) where source is "html", "pdf" or "" if both failed."""
    try:
        text = html_to_text(getter(paper.html_url).decode("utf-8", "replace"))
        if len(text) > 2000:  # arXiv serves a short stub page when HTML conversion failed
            return text, "html"
    except Exception as e:
        log.info("no arXiv HTML for %s: %s", paper.arxiv_id, e)
    try:
        return pdf_to_text(getter(paper.pdf_url)), "pdf"
    except Exception as e:
        log.warning("could not fetch PDF for %s: %s", paper.arxiv_id, e)
    return "", ""
