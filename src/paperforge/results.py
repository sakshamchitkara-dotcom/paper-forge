"""Compare reproduced metrics with claimed ones and decide an honest status.

The status is computed here, by code, from files the experiment actually wrote.
Nothing an implementation (or an LLM) says about itself can mark a paper as
reproduced.

Statuses:
  reproduced      experiment ran, tests passed, every claim matched
  partial         experiment ran and some (not all) claims matched, or tests failed
  not_reproduced  experiment ran, no claim matched
  unverified      experiment ran but there were no comparable claims
  failed          experiment crashed, timed out, or produced no metrics
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("reproduced", "partial", "not_reproduced", "unverified", "failed")


def _check(claim: dict, value) -> tuple[str, float | None]:
    kind = claim.get("kind", "numeric")
    if value is None:
        return "missing", None
    if kind == "qualitative":
        return ("match" if value is True else "mismatch"), None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "mismatch", None
    target = claim["claimed"]
    if kind == "at_most":
        return ("match" if value <= target else "mismatch"), None
    if kind == "at_least":
        return ("match" if value >= target else "mismatch"), None
    tol = claim.get("tolerance")
    if tol is None:
        tol = abs(target) * claim.get("rel_tolerance", 0.1)
    return ("match" if abs(value - target) <= tol else "mismatch"), tol


def compare(claims: list[dict], metrics: dict) -> list[dict]:
    out = []
    for c in claims:
        value = metrics.get(c["name"])
        verdict, tol = _check(c, value)
        out.append({"name": c["name"], "kind": c.get("kind", "numeric"), "claimed": c.get("claimed"),
                    "reproduced": value, "tolerance": tol, "verdict": verdict,
                    "source": c.get("source", "")})
    return out


def load_metrics(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data else None


def decide_status(ran_ok: bool, metrics: dict | None, comparisons: list[dict], tests_ok: bool | None) -> str:
    if not ran_ok or metrics is None:
        return "failed"
    if not comparisons:
        return "unverified"
    matched = sum(c["verdict"] == "match" for c in comparisons)
    if matched == len(comparisons):
        return "reproduced" if tests_ok is not False else "partial"
    return "partial" if matched else "not_reproduced"


def build_results(*, paper: dict, backend: str, claims: list[dict], experiment, tests,
                  metrics_path: Path, attempts: int, sandbox: str, notes: list[str] | None = None) -> dict:
    """`experiment`/`tests` are sandbox.RunResult (tests may be None if there are none)."""
    ran_ok = experiment is not None and experiment.ok
    metrics = load_metrics(metrics_path) if ran_ok else None
    comparisons = compare(claims, metrics or {}) if metrics else []
    tests_ok = None if tests is None else tests.ok
    return {
        "paper": {"arxiv_id": paper["arxiv_id"], "title": paper["title"],
                  "url": f"https://arxiv.org/abs/{paper['arxiv_id']}"},
        "backend": backend,
        "status": decide_status(ran_ok, metrics, comparisons, tests_ok),
        "tests_passed": tests_ok,
        "metrics": metrics,
        "comparisons": comparisons,
        "attempts": attempts,
        "sandbox": sandbox,
        "runtime_s": experiment.duration_s if experiment else None,
        "error": None if ran_ok else _error(experiment, metrics_path),
        "notes": notes or [],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _error(experiment, metrics_path: Path) -> str:
    if experiment is None:
        return "experiment never ran"
    if experiment.timed_out:
        return f"timed out after {experiment.duration_s}s"
    if experiment.returncode != 0:
        return (experiment.stderr.strip().splitlines() or ["non-zero exit"])[-1][:500]
    return f"no metrics written to {Path(metrics_path).name}"
