"""Configuration: a TOML file merged over built-in defaults."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

DEFAULTS: dict = {
    "sources": {
        "categories": ["cs.LG", "cs.CL", "cs.IR", "stat.ML"],
        "max_results_per_category": 50,
        "lookback_days": 2,
        "huggingface": True,
        "semantic_scholar": False,  # unauthenticated S2 is heavily rate limited
    },
    "screening": {
        "top_k": 10,
        "fetch_fulltext": True,
        "fulltext_char_limit": 60000,
        "interest_keywords": [
            "optimizer", "sampling", "retrieval", "sketch", "kernel",
            "bandit", "monte carlo", "variational", "regularization", "ranking",
        ],
        "weights": {
            "compute": 3.0,
            "data": 2.0,
            "clarity": 2.0,
            "code": 1.0,
            "interest": 1.5,
        },
    },
    "llm": {
        "model": "claude-opus-5-5",
        "effort": "high",
    },
    "reimplement": {
        "enabled": False,  # opt-in: generating code costs API tokens
        "backend": "scripted",  # "scripted" (offline references) or "claude"
        "max_papers": 1,
        "max_iterations": 4,
        "timeout_s": 300,
        "sandbox": "auto",  # "auto" | "docker" | "subprocess"
    },
    "paths": {
        "db": "data/forge.db",
        "work": "work",
        "site": "site",
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path("forge.toml")
        if not path.exists():
            return copy.deepcopy(DEFAULTS)
    with open(path, "rb") as f:
        return _merge(DEFAULTS, tomllib.load(f))
