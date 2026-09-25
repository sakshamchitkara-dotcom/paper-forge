"""`forge` command line."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

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
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args, load_config(args.config))


if __name__ == "__main__":
    sys.exit(main())
