"""Optional enrichment: Hugging Face daily papers (code links, upvotes) and
Semantic Scholar (citation counts).

Papers with Code's API is gone (paperswithcode.com now redirects to Hugging Face
trending), so code availability comes from HF's `githubRepo` field and from GitHub
links in the arXiv abstract/comment. Every source here is best-effort: a failure
logs a warning and leaves the paper un-enriched.
"""

from __future__ import annotations

import logging
import os
import re
import time

from . import http
from .models import Paper

log = logging.getLogger(__name__)

HF_DAILY = "https://huggingface.co/api/daily_papers"
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
_GITHUB_RE = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+", re.I)


def find_code_link(p: Paper) -> str:
    m = _GITHUB_RE.search(f"{p.abstract} {p.comment}")
    return m.group(0).rstrip(".") if m else ""


def fetch_hf_daily(limit: int = 100, getter=http.get_json) -> dict[str, dict]:
    """arXiv id -> {"upvotes", "github"} for today's Hugging Face daily papers."""
    try:
        items = getter(f"{HF_DAILY}?limit={limit}")
    except Exception as e:  # network/API failure is non-fatal
        log.warning("huggingface daily papers unavailable: %s", e)
        return {}
    out = {}
    for it in items:
        paper = it.get("paper", {})
        if paper.get("id"):
            out[paper["id"]] = {"upvotes": paper.get("upvotes"), "github": paper.get("githubRepo") or ""}
    return out


def hf_only_ids(hf: dict[str, dict], known: set[str]) -> list[str]:
    """HF-featured arXiv ids not already seen via the category listings."""
    return [i for i in hf if i not in known]


def apply_hf(papers: list[Paper], hf: dict[str, dict]) -> None:
    for p in papers:
        info = hf.get(p.arxiv_id)
        if info:
            p.hf_upvotes = info["upvotes"]
            p.code_url = p.code_url or info["github"]
            if "huggingface" not in p.sources:
                p.sources.append("huggingface")
        p.code_url = p.code_url or find_code_link(p)


def fetch_citations(papers: list[Paper], poster=None, retries: int = 2) -> None:
    """Fill `citations` via Semantic Scholar's batch endpoint (one request for all)."""
    if not papers:
        return
    import json
    import urllib.request

    headers = {"Content-Type": "application/json", "User-Agent": http.USER_AGENT}
    if key := os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = key

    def _post(body: bytes):
        req = urllib.request.Request(f"{S2_BATCH}?fields=citationCount", data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    poster = poster or _post
    body = json.dumps({"ids": [f"ARXIV:{p.arxiv_id}" for p in papers[:500]]}).encode()
    for attempt in range(retries + 1):
        try:
            rows = poster(body)
            break
        except Exception as e:
            log.warning("semantic scholar attempt %d failed: %s", attempt + 1, e)
            time.sleep(3 * (attempt + 1))
    else:
        return
    for p, row in zip(papers, rows):
        if row:
            p.citations = row.get("citationCount")
            if "semantic_scholar" not in p.sources:
                p.sources.append("semantic_scholar")
