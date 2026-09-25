"""Daily screening pipeline: fetch -> enrich -> dedupe -> score -> analyze top-k -> store."""

from __future__ import annotations

import logging
import time

from . import enrich
from .analyze import analyze
from .arxiv import ArxivClient
from .fulltext import fetch_fulltext
from .models import Paper
from .rubric import score_paper
from .store import Store

log = logging.getLogger(__name__)


def collect(cfg: dict, client: ArxivClient, hf_getter=None) -> list[Paper]:
    src = cfg["sources"]
    papers: dict[str, Paper] = {}
    for cat in src["categories"]:
        try:
            for p in client.recent(cat, src["max_results_per_category"], src["lookback_days"]):
                papers.setdefault(p.arxiv_id, p)
        except OSError as e:  # one failing category should not sink the whole day
            log.warning("arxiv %s failed: %s", cat, e)
        log.info("arxiv %s: %d papers so far", cat, len(papers))

    if src.get("huggingface"):
        hf = enrich.fetch_hf_daily(**({"getter": hf_getter} if hf_getter else {}))
        extra = enrich.hf_only_ids(hf, set(papers))
        if extra:  # HF-featured papers outside today's listings, if in our categories
            for p in client.query(id_list=extra[:50], max_results=len(extra[:50])):
                if set(p.categories) & set(src["categories"]):
                    papers.setdefault(p.arxiv_id, p)
        enrich.apply_hf(list(papers.values()), hf)
    else:
        enrich.apply_hf(list(papers.values()), {})
    if src.get("semantic_scholar"):
        enrich.fetch_citations(list(papers.values()))
    return list(papers.values())


def screen(cfg: dict, store: Store, day: str, arxiv: ArxivClient | None = None,
           llm_client=None, fulltext_getter=None, hf_getter=None) -> list[dict]:
    arxiv = arxiv or ArxivClient()
    sc = cfg["screening"]
    papers = collect(cfg, arxiv, hf_getter)
    new_ids = set(store.unseen([p.arxiv_id for p in papers]))
    fresh = [p for p in papers if p.arxiv_id in new_ids]
    log.info("%d collected, %d new after dedupe", len(papers), len(fresh))

    scored = sorted(((score_paper(p, sc["weights"], sc["interest_keywords"]), p) for p in fresh),
                    key=lambda t: t[0].total, reverse=True)
    for rank, (s, p) in enumerate(scored):
        analysis = None
        if rank < sc["top_k"]:
            text = ""
            if sc["fetch_fulltext"]:
                kw = {"getter": fulltext_getter} if fulltext_getter else {}
                text, how = fetch_fulltext(p, **kw)
                log.info("fulltext %s: %s (%d chars)", p.arxiv_id, how or "none", len(text))
                if not fulltext_getter:
                    time.sleep(1.0)  # be polite to arxiv.org between downloads
                s = score_paper(p, sc["weights"], sc["interest_keywords"], fulltext=text)
            analysis = analyze(p, text, s.to_dict(), cfg, client=llm_client)
        store.add_screened(day, p.to_dict(), s.to_dict(), analysis)
    return store.screened_on(day)
