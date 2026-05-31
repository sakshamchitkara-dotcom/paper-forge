"""Reimplementation backends.

- scripted: copies one of the hand-written reference implementations shipped in
  `paperforge/references/` (clearly labeled as such) and runs it. Offline, no key.
- claude: an agent loop that asks Claude for a small self-contained repo, runs its
  tests and experiment in the sandbox, feeds failures back, and stops after
  `max_iterations`.

Either way the status is decided by `results.build_results` from what the sandboxed
run actually produced.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path, PurePosixPath

from . import sandbox
from .results import build_results

log = logging.getLogger(__name__)
REFERENCES_DIR = Path(__file__).parent / "references"
TEST_CMD = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
EXPERIMENT_CMD = ["python", "experiment.py"]


def list_references() -> dict[str, dict]:
    """arxiv_id -> reference metadata (+ `dir`)."""
    out = {}
    for d in sorted(REFERENCES_DIR.iterdir()):
        if (d / "paper.json").exists():
            meta = json.loads((d / "paper.json").read_text())
            out[meta["arxiv_id"]] = {**meta, "dir": d}
    return out


def execute(workdir: Path, *, paper: dict, claims: list[dict], backend: str, cfg: dict,
            attempts: int = 1, notes: list[str] | None = None) -> tuple[dict, object, object]:
    """Run tests + experiment for one implementation directory and build results."""
    metrics_path = workdir / "metrics.json"
    metrics_path.unlink(missing_ok=True)  # never grade a stale file
    box = sandbox.resolve_backend(cfg["sandbox"])
    has_tests = any(workdir.glob("test_*.py"))
    tests = sandbox.run(TEST_CMD, workdir, cfg["timeout_s"], box) if has_tests else None
    exp = sandbox.run(EXPERIMENT_CMD, workdir, cfg["timeout_s"], box)
    (workdir / "run.log").write_text(
        (f"$ {' '.join(TEST_CMD)}\n{tests.stdout}\n{tests.stderr}\n" if tests else "(no tests)\n")
        + f"$ {' '.join(EXPERIMENT_CMD)}\n{exp.stdout}\n{exp.stderr}\n")
    res = build_results(paper=paper, backend=backend, claims=claims, experiment=exp, tests=tests,
                        metrics_path=metrics_path, attempts=attempts, sandbox=box, notes=notes)
    return res, tests, exp


def run_scripted(arxiv_id: str, out_dir: Path, cfg: dict) -> dict:
    refs = list_references()
    if arxiv_id not in refs:
        raise KeyError(f"no hand-written reference for {arxiv_id}; available: {', '.join(refs)}")
    ref = refs[arxiv_id]
    impl = out_dir / "implementation"
    if impl.exists():
        shutil.rmtree(impl)
    shutil.copytree(ref["dir"] / "impl", impl)
    claims = json.loads((ref["dir"] / "claims.json").read_text())
    res, _, _ = execute(impl, paper=ref, claims=claims, backend="scripted", cfg=cfg,
                        notes=[ref["reference_note"]])
    res["claims"] = claims
    res["fidelity_notes"] = (ref["dir"] / "fidelity.md").read_text().strip().splitlines()
    return res


# --------------------------------------------------------------------------- claude

FILES_SCHEMA = {
    "type": "object",
    "properties": {
        "files": {"type": "array", "items": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"], "additionalProperties": False}},
        "deviations": {"type": "array", "items": {"type": "string"},
                       "description": "every way this implementation differs from the paper"},
    },
    "required": ["files", "deviations"],
    "additionalProperties": False,
}

AGENT_SYSTEM = """You write small, faithful reference implementations of machine-learning papers.

Produce a complete, self-contained Python project as a list of files:
- one or more modules implementing the method, written from the paper's description (do not \
reproduce the authors' code verbatim);
- test_*.py with pytest unit tests of the core algorithm (fast, deterministic);
- experiment.py that runs a scaled-down version of the paper's main experiment and writes \
metrics.json (a flat JSON object) in the current directory.

Hard constraints: Python 3.10+, numpy only (torch CPU only if unavoidable, and import it lazily); \
no network access at runtime, so generate synthetic data or use data you construct in code; \
fixed random seeds; total runtime well under {timeout} seconds on one CPU core; never write the \
claimed numbers into the code or metrics - metrics must be measured.

List every deviation from the paper (scaled-down sizes, substituted datasets, unspecified \
hyperparameters you chose) in `deviations`. When you get run feedback, return the full corrected \
file set, not a diff."""

_SAFE_PATH = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")


def _safe_write(workdir: Path, files: list[dict], max_files: int = 20, max_bytes: int = 200_000) -> None:
    """Write model-generated files, refusing anything that escapes `workdir`."""
    if len(files) > max_files:
        raise ValueError(f"too many files ({len(files)})")
    for f in files:
        p = PurePosixPath(f["path"])
        if not _SAFE_PATH.match(f["path"]) or p.is_absolute() or ".." in p.parts:
            raise ValueError(f"unsafe path {f['path']!r}")
        if len(f["content"].encode()) > max_bytes:
            raise ValueError(f"file too large: {f['path']}")
        dest = workdir / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f["content"])


def claims_from_analysis(analysis: dict) -> list[dict]:
    claims, used = [], set()
    for m in analysis.get("claimed_metrics", []):
        key = re.sub(r"[^a-z0-9]+", "_", m["name"].lower()).strip("_") or "metric"
        base, i = key, 2
        while key in used:
            key, i = f"{base}_{i}", i + 1
        used.add(key)
        claims.append({"name": key, "kind": "numeric", "claimed": m["value"], "rel_tolerance": 0.1,
                       "source": f"{m['name']} ({m.get('context', '')}), extracted by {analysis.get('source')}"})
    return claims


def _feedback(res: dict, tests, exp, claims: list[dict]) -> str:
    parts = [f"Run result: status={res['status']}."]
    if tests is not None and not tests.ok:
        parts.append(f"pytest failed:\n{tests.stdout[-4000:]}\n{tests.stderr[-2000:]}")
    if not exp.ok:
        parts.append(f"experiment.py failed ({res['error']}):\n{exp.stderr[-4000:]}")
    elif res["metrics"] is None:
        parts.append("experiment.py exited 0 but metrics.json is missing or empty.")
    else:
        missing = [c["name"] for c in res["comparisons"] if c["verdict"] == "missing"]
        if missing:
            parts.append(f"metrics.json lacks keys: {missing}. Measure them if the scaled-down setting allows.")
    parts.append("Fix the problems and return the complete file set.")
    return "\n\n".join(parts)


def _needs_retry(res: dict) -> bool:
    if res["status"] == "failed" or res["tests_passed"] is False:
        return True
    return any(c["verdict"] == "missing" for c in res["comparisons"])


def run_claude(paper: dict, analysis: dict, out_dir: Path, cfg: dict, llm: dict, client=None) -> dict:
    import anthropic

    client = client or anthropic.Anthropic()
    impl = out_dir / "implementation"
    claims = claims_from_analysis(analysis)
    brief = {k: analysis.get(k) for k in ("method_summary", "key_equations", "datasets",
                                          "claimed_metrics", "reproduction_plan", "risks")}
    keys = ", ".join(c["name"] for c in claims) or "(none extracted: choose sensible metric names)"
    messages = [{"role": "user", "content":
                 f"Paper: {paper['title']} (arXiv:{paper['arxiv_id']})\nAbstract: {paper.get('abstract', '')}\n\n"
                 f"Analysis:\n{json.dumps(brief, indent=2)}\n\n"
                 f"metrics.json must use exactly these keys where measurable: {keys}"}]
    res, deviations = None, []
    for attempt in range(1, cfg["max_iterations"] + 1):
        with client.messages.stream(
            model=llm["model"], max_tokens=64000,
            system=AGENT_SYSTEM.format(timeout=cfg["timeout_s"]),
            output_config={"effort": llm.get("effort", "high"),
                           "format": {"type": "json_schema", "schema": FILES_SCHEMA}},
            messages=messages,
        ) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason in ("refusal", "max_tokens"):
            log.warning("generation stopped: %s", msg.stop_reason)
            break
        out = json.loads(next(b.text for b in msg.content if b.type == "text"))
        deviations = out["deviations"]
        if impl.exists():
            shutil.rmtree(impl)
        impl.mkdir(parents=True)
        try:
            _safe_write(impl, out["files"])
        except ValueError as e:
            messages += [{"role": "assistant", "content": msg.content},
                         {"role": "user", "content": f"Rejected file set: {e}. Use simple relative paths."}]
            continue
        res, tests, exp = execute(impl, paper=paper, claims=claims, backend="claude", cfg=cfg, attempts=attempt,
                                  notes=[f"Generated by {llm['model']} (attempt {attempt}); not reviewed by a human."])
        log.info("attempt %d: %s", attempt, res["status"])
        if not _needs_retry(res):
            break
        messages += [{"role": "assistant", "content": msg.content},
                     {"role": "user", "content": _feedback(res, tests, exp, claims)}]
    if res is None:  # nothing ever ran
        res = build_results(paper=paper, backend="claude", claims=claims, experiment=None, tests=None,
                            metrics_path=impl / "metrics.json", attempts=cfg["max_iterations"], sandbox="-",
                            notes=["No runnable implementation was produced."])
    res["claims"] = claims
    res["fidelity_notes"] = [f"- {d}" for d in deviations] + [
        "- Claimed values were extracted automatically from the paper text; compare against the paper before citing.",
        "- Numeric claims use a default 10% relative tolerance; scaled-down runs can legitimately miss full-scale numbers."]
    return res
