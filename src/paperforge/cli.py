"""`forge` command line."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from .config import load_config
from .store import Store

log = logging.getLogger("paperforge")


def cmd_screen(args, cfg) -> int:
    from .screen import screen

    if args.categories:
        cfg["sources"]["categories"] = args.categories.split(",")
    if args.max_results:
        cfg["sources"]["max_results_per_category"] = args.max_results
    if args.top_k is not None:
        cfg["screening"]["top_k"] = args.top_k
    rows = screen(cfg, Store(cfg["paths"]["db"]), args.day)
    for r in rows[:args.show]:
        src = (r["analysis"] or {}).get("source", "-")
        print(f"{r['score']:5.1f}  {r['arxiv_id']:<12} [{src}] {r['title'][:90]}")
    print(f"{len(rows)} new papers screened for {args.day}", file=sys.stderr)
    return 0


def _paper_for(arxiv_id: str, store: Store, cfg: dict) -> tuple[dict, dict]:
    """Paper dict + analysis for the claude backend, from the store or fetched fresh."""
    row = store.get(arxiv_id)
    if row and row["analysis"]:
        return row["paper"], row["analysis"]
    from .analyze import analyze
    from .arxiv import ArxivClient
    from .fulltext import fetch_fulltext
    from .rubric import score_paper

    papers = ArxivClient().query(id_list=[arxiv_id], max_results=1)
    if not papers:
        raise SystemExit(f"arXiv has no paper {arxiv_id}")
    p = papers[0]
    text, _ = fetch_fulltext(p)
    sc = cfg["screening"]
    rubric = score_paper(p, sc["weights"], sc["interest_keywords"], fulltext=text).to_dict()
    return p.to_dict(), analyze(p, text, rubric, cfg)


def reimplement_one(arxiv_id: str, backend: str, cfg: dict, store: Store, day: str) -> dict:
    from .reimplement import list_references, run_claude, run_scripted
    from .report import write_paper_folder

    out = Path(cfg["paths"]["site"]) / "papers" / arxiv_id.replace("/", "_")
    rcfg = cfg["reimplement"]
    if backend == "scripted":
        paper = {k: v for k, v in list_references()[arxiv_id].items() if k != "dir"}
        res = run_scripted(arxiv_id, out, rcfg)
    elif backend == "claude":
        paper, analysis = _paper_for(arxiv_id, store, cfg)
        res = run_claude(paper, analysis, out, rcfg, cfg["llm"])
    else:
        raise SystemExit(f"unknown backend {backend!r}")
    write_paper_folder(out, paper, res)
    store.record_run(arxiv_id, day, backend, res["status"], res)
    print(f"{res['status']:<15} {arxiv_id:<12} {backend:<9} {paper['title'][:70]}")
    return res


def cmd_reimplement(args, cfg) -> int:
    cfg["reimplement"]["sandbox"] = args.sandbox or cfg["reimplement"]["sandbox"]
    store = Store(cfg["paths"]["db"])
    for arxiv_id in args.arxiv_ids:
        reimplement_one(arxiv_id, args.backend or cfg["reimplement"]["backend"], cfg, store, args.day)
    build_site(cfg, store)
    return 0


def cmd_references(args, cfg) -> int:
    from .reimplement import list_references

    for i, r in list_references().items():
        print(f"{i:<12} {r['title']}  [hand-written reference]")
    return 0


def build_site(cfg: dict, store: Store, day: str | None = None) -> Path:
    from .report import write_daily, write_index, write_leaderboard

    site = Path(cfg["paths"]["site"])
    runs = store.latest_runs()
    days = sorted(set(store.days()) | {r["day"] for r in runs} | ({day} if day else set()), reverse=True)
    for d in days:
        write_daily(site, d, store.screened_on(d), [r for r in runs if r["day"] == d])
    write_leaderboard(site, runs)
    write_index(site, days, runs)
    (site / ".nojekyll").touch()
    return site


def cmd_report(args, cfg) -> int:
    site = build_site(cfg, Store(cfg["paths"]["db"]), args.day)
    print(f"site written to {site}/index.html")
    return 0


def cmd_daily(args, cfg) -> int:
    from .screen import screen

    store = Store(cfg["paths"]["db"])
    rows = screen(cfg, store, args.day)
    print(f"screened {len(rows)} new papers", file=sys.stderr)
    rcfg = cfg["reimplement"]
    if args.reimplement or rcfg["enabled"]:
        backend = args.backend or rcfg["backend"]
        if backend == "scripted":
            from .reimplement import list_references
            targets = list(list_references())  # demo: hand-written references, not today's papers
        else:
            targets = [r["arxiv_id"] for r in rows
                       if r["analysis"] and r["analysis"]["laptop_feasible"]][:rcfg["max_papers"]]
        for arxiv_id in targets:
            reimplement_one(arxiv_id, backend, cfg, store, args.day)
    site = build_site(cfg, store, args.day)
    print(f"report: {site}/daily/{args.day}.html")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="forge", description=__doc__)
    ap.add_argument("-c", "--config", help="path to forge.toml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("screen", help="fetch and score today's papers")
    s.add_argument("--day", default=date.today().isoformat())
    s.add_argument("--categories", help="comma-separated arXiv categories (overrides config)")
    s.add_argument("--max-results", type=int)
    s.add_argument("--top-k", type=int)
    s.add_argument("--show", type=int, default=15)
    s.set_defaults(func=cmd_screen)

    r = sub.add_parser("reimplement", help="reimplement papers and run them in the sandbox")
    r.add_argument("arxiv_ids", nargs="+")
    r.add_argument("--backend", choices=["scripted", "claude"])
    r.add_argument("--sandbox", choices=["auto", "docker", "subprocess"])
    r.add_argument("--day", default=date.today().isoformat())
    r.set_defaults(func=cmd_reimplement)

    d = sub.add_parser("daily", help="screen, optionally reimplement, and build the report site")
    d.add_argument("--day", default=date.today().isoformat())
    d.add_argument("--reimplement", action="store_true", help="opt in to reimplementation (claude backend costs tokens)")
    d.add_argument("--backend", choices=["scripted", "claude"])
    d.set_defaults(func=cmd_daily)

    rep = sub.add_parser("report", help="rebuild the static site from the database")
    rep.add_argument("--day")
    rep.set_defaults(func=cmd_report)

    ref = sub.add_parser("references", help="list hand-written reference implementations")
    ref.set_defaults(func=cmd_references)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args, load_config(args.config))


if __name__ == "__main__":
    sys.exit(main())
