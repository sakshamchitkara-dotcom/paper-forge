"""Tiny HTTP helper shared by enrichment and full-text fetchers."""

from __future__ import annotations

import json
import urllib.request

from .arxiv import user_agent


def get(url: str, timeout: float = 30.0, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent(), **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_json(url: str, timeout: float = 30.0, headers: dict | None = None):
    return json.loads(get(url, timeout, headers))
