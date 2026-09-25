"""arXiv API client (export.arxiv.org/api/query) and Atom feed parser.

arXiv asks API users to wait 3 seconds between calls; `ArxivClient` enforces that.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from .models import Paper

log = logging.getLogger(__name__)
API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "paper-forge/0.1 (+https://github.com/sakshamchitkara-dotcom/paper-forge)"
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_ID_RE = re.compile(r"arxiv\.org/abs/(?P<id>.+?)(?P<ver>v\d+)?$")


def _text(el, path: str) -> str:
    found = el.find(path, NS)
    return " ".join(found.text.split()) if found is not None and found.text else ""


def parse_feed(xml_text: str) -> list[Paper]:
    """Parse an arXiv Atom feed into Papers. Error entries (bad queries) are skipped."""
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("a:entry", NS):
        m = _ID_RE.search(_text(entry, "a:id"))
        if not m:  # arXiv reports query errors as an entry with an api/errors id
            continue
        papers.append(
            Paper(
                arxiv_id=m.group("id"),
                version=m.group("ver") or "v1",
                title=_text(entry, "a:title"),
                abstract=_text(entry, "a:summary"),
                authors=[_text(a, "a:name") for a in entry.findall("a:author", NS)],
                categories=[c.get("term", "") for c in entry.findall("a:category", NS)],
                published=_text(entry, "a:published"),
                updated=_text(entry, "a:updated"),
                comment=_text(entry, "arxiv:comment"),
                journal_ref=_text(entry, "arxiv:journal_ref"),
                doi=_text(entry, "arxiv:doi"),
                sources=["arxiv"],
            )
        )
    return papers


class ArxivClient:
    def __init__(self, delay_s: float = 3.0, timeout_s: float = 30.0, opener=None, retries: int = 3):
        self.delay_s = delay_s
        self.retries = retries
        self.timeout_s = timeout_s
        self._last = 0.0
        self._open = opener or self._urlopen

    def _urlopen(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/atom+xml"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return r.read().decode("utf-8")

    def _get(self, params: dict) -> str:
        url = API_URL + "?" + urllib.parse.urlencode(params)
        for attempt in range(self.retries + 1):
            wait = self.delay_s * (2 ** attempt) - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                return self._open(url)
            except urllib.error.HTTPError as e:
                # arXiv intermittently answers 406/429/503 under load; back off and retry
                if e.code not in (406, 429, 500, 502, 503) or attempt == self.retries:
                    raise
                log.info("arXiv HTTP %s, retrying (%d/%d)", e.code, attempt + 1, self.retries)
            finally:
                self._last = time.monotonic()
        raise AssertionError("unreachable")

    def query(self, search_query: str = "", id_list: list[str] | None = None,
              max_results: int = 50, start: int = 0) -> list[Paper]:
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        if id_list:
            params["id_list"] = ",".join(id_list)
        return parse_feed(self._get(params))

    def recent(self, category: str, max_results: int = 50, lookback_days: int = 2,
               now: datetime | None = None) -> list[Paper]:
        """Newest submissions in a category, limited to the last `lookback_days`."""
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=lookback_days)
        papers = self.query(f"cat:{category}", max_results=max_results)
        return [p for p in papers if _parse_ts(p.published) >= cutoff]


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
