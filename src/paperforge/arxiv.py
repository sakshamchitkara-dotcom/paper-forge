"""arXiv API client (export.arxiv.org/api/query), daily listing fallback
(rss.arxiv.org), and their Atom feed parsers.

arXiv asks API users to wait 3 seconds between calls; `ArxivClient` enforces that
for every request it makes, including the listing fallback.
"""

from __future__ import annotations

import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from . import __version__
from .models import Paper

log = logging.getLogger(__name__)
API_URL = "https://export.arxiv.org/api/query"
LISTING_URL = "https://rss.arxiv.org/atom/{category}"
REPO_URL = "https://github.com/sakshamchitkara-dotcom/paper-forge"


def user_agent(contact: str | None = None) -> str:
    """arXiv asks automated clients to identify themselves with a way to reach the operator.

    Set FORGE_CONTACT (an email address) so arXiv can reach whoever runs this instance
    instead of the project repo.
    """
    contact = contact if contact is not None else os.environ.get("FORGE_CONTACT", "").strip()
    return f"paper-forge/{__version__} (+{REPO_URL}" + (f"; mailto:{contact})" if contact else ")")

NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "dc": "http://purl.org/dc/elements/1.1/",
}
_ID_RE = re.compile(r"arxiv\.org/abs/(?P<id>.+?)(?P<ver>v\d+)?$")
_OAI_ID_RE = re.compile(r"oai:arXiv\.org:(?P<id>.+?)(?P<ver>v\d+)$")
_ABSTRACT_RE = re.compile(r"^arXiv:\S+\s+Announce Type:\s*\S+\s*Abstract:\s*", re.S)


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


def parse_listing(xml_text: str) -> list[Paper]:
    """Parse a rss.arxiv.org Atom listing (one category's latest announcement).

    Keeps first announcements only (`new` and `cross`, version v1): replacements are
    old papers, and their `published` would be the announcement date, not the
    submission date the citation needs.
    """
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("a:entry", NS):
        m = _OAI_ID_RE.search(_text(entry, "a:id"))
        kind = _text(entry, "arxiv:announce_type")
        if not m or kind not in ("new", "cross") or m.group("ver") != "v1":
            continue
        creators = _text(entry, "dc:creator")
        papers.append(
            Paper(
                arxiv_id=m.group("id"),
                version="v1",
                title=_text(entry, "a:title"),
                abstract=_ABSTRACT_RE.sub("", _text(entry, "a:summary")),
                authors=[a.strip() for a in creators.split(",") if a.strip()],
                categories=[c.get("term", "") for c in entry.findall("a:category", NS)],
                published=_text(entry, "a:published"),
                updated=_text(entry, "a:updated"),
                journal_ref=_text(entry, "arxiv:journal_reference"),
                doi=_text(entry, "arxiv:DOI"),
                sources=["arxiv-listing"],
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
        req = urllib.request.Request(url, headers={"User-Agent": user_agent(), "Accept": "application/atom+xml"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return r.read().decode("utf-8")

    def _get(self, params: dict) -> str:
        return self._fetch(API_URL + "?" + urllib.parse.urlencode(params))

    def _fetch(self, url: str) -> str:
        retry_after = 0.0
        for attempt in range(self.retries + 1):
            wait = max(self.delay_s * (2 ** attempt), retry_after) - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                return self._open(url)
            except urllib.error.HTTPError as e:
                # arXiv intermittently answers 406/429/503 under load; back off and retry
                if e.code not in (406, 429, 500, 502, 503) or attempt == self.retries:
                    raise
                retry_after = _retry_after(e)
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


    def listing(self, category: str) -> list[Paper]:
        """First announcements in `category` from the daily listing feed.

        Fallback for when the query API keeps refusing (arXiv's API intermittently
        answers 406 under load while the listing feed stays up).
        """
        return parse_listing(self._fetch(LISTING_URL.format(category=urllib.parse.quote(category))))


def _retry_after(e: urllib.error.HTTPError, cap: float = 120.0) -> float:
    """Seconds from a numeric Retry-After header (0 if absent or an HTTP date), capped."""
    try:
        return min(float((e.headers or {}).get("Retry-After", 0)), cap)
    except (TypeError, ValueError):
        return 0.0


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
