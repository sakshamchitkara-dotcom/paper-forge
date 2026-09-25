"""Structured paper analysis: Claude when credentials exist, heuristics otherwise.

Both paths return the same dict shape (ANALYSIS_SCHEMA plus a `source` field) so
downstream code never cares which one ran. The report always shows the source.
"""

from __future__ import annotations

import json
import logging
import os
import re

from .models import Paper

log = logging.getLogger(__name__)

_METRIC = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "value": {"type": "number"},
        "unit": {"type": "string"},
        "higher_is_better": {"type": "boolean"},
        "context": {"type": "string", "description": "dataset/setting and where in the paper"},
    },
    "required": ["name", "value", "unit", "higher_is_better", "context"],
    "additionalProperties": False,
}
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "method_summary": {"type": "string"},
        "key_equations": {"type": "array", "items": {"type": "string"}},
        "datasets": {"type": "array", "items": {"type": "string"}},
        "claimed_metrics": {"type": "array", "items": _METRIC},
        "compute_estimate": {"type": "string"},
        "laptop_feasible": {"type": "boolean"},
        "feasibility_score": {"type": "integer", "description": "0 (impossible on a laptop) .. 10 (trivial)"},
        "reproduction_plan": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["method_summary", "key_equations", "datasets", "claimed_metrics", "compute_estimate",
                 "laptop_feasible", "feasibility_score", "reproduction_plan", "risks"],
    "additionalProperties": False,
}

SYSTEM = """You screen new machine-learning papers for whether one engineer can faithfully \
reproduce the core result on a laptop CPU in under an hour of compute, using numpy or \
CPU-only PyTorch and synthetic or small public datasets.

Fill the JSON schema from the paper text you are given. Rules:
- Only report numbers that literally appear in the text as claimed_metrics; put the table/section in `context`. Never invent or round numbers.
- key_equations: the few equations needed to implement the method, in LaTeX.
- reproduction_plan: concrete, minimal steps; say which parts must be scaled down and what substitutes for unavailable data.
- If the text is truncated or missing, say so in `risks` and lower feasibility_score accordingly."""

KNOWN_DATASETS = ["MNIST", "Fashion-MNIST", "CIFAR-10", "CIFAR-100", "ImageNet", "SVHN", "UCI",
                  "Iris", "MovieLens", "MS MARCO", "BEIR", "GLUE", "SQuAD", "WikiText", "Penn Treebank",
                  "Common Crawl", "IMDB", "AG News", "OpenML", "synthetic"]
_METRIC_RE = re.compile(
    r"(?P<name>accuracy|F1|BLEU|ROUGE-?L?|PSNR|AUC|nDCG@?\d*|MRR|recall@?\d*|perplexity|error rate)"
    r"[^.\d]{0,40}?(?P<value>\d+(?:\.\d+)?)\s?(?P<unit>%|dB)?", re.I)


def has_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def heuristic_analysis(paper: Paper, fulltext: str, rubric: dict) -> dict:
    text = f"{paper.abstract}\n{fulltext}"
    sentences = re.split(r"(?<=[.!?])\s+", paper.abstract)
    eqs = sorted({m for m in re.findall(r"\$([^$]{8,200})\$", fulltext) if "=" in m}, key=len, reverse=True)
    metrics = []
    for m in _METRIC_RE.finditer(paper.abstract):
        name = m.group("name")
        metrics.append({"name": name, "value": float(m.group("value")), "unit": m.group("unit") or "",
                        "higher_is_better": name.lower() not in ("perplexity", "error rate"),
                        "context": "abstract (regex extraction, unverified)"})
    crit = rubric["criteria"]
    feasible = crit["compute"]["score"] >= 0.5 and crit["data"]["score"] >= 0.4
    return {
        "method_summary": " ".join(sentences[:2]),
        "key_equations": eqs[:5],
        "datasets": [d for d in KNOWN_DATASETS if re.search(rf"\b{re.escape(d)}\b", text, re.I)],
        "claimed_metrics": metrics[:6],
        "compute_estimate": "unknown (heuristic: " + ("light" if crit["compute"]["score"] >= 0.5 else "heavy") + ")",
        "laptop_feasible": feasible,
        "feasibility_score": round(10 * (crit["compute"]["score"] + crit["data"]["score"] + crit["clarity"]["score"]) / 3),
        "reproduction_plan": [
            "Implement the method from the key equations / algorithm box.",
            "Substitute a synthetic or small public dataset matching the paper's setting.",
            "Reproduce the headline comparison at reduced scale and report deviations.",
        ],
        "risks": ["Heuristic analysis only (no LLM): summary, equations and metrics are regex extractions."],
        "source": "heuristic",
    }


def claude_analysis(paper: Paper, fulltext: str, cfg: dict, client=None, char_limit: int = 60000) -> dict:
    import anthropic

    client = client or anthropic.Anthropic()
    truncated = len(fulltext) > char_limit
    body = fulltext[:char_limit] if fulltext else "(full text unavailable; abstract only)"
    prompt = (f"Title: {paper.title}\narXiv: {paper.arxiv_id}\nAuthors: {', '.join(paper.authors)}\n"
              f"Abstract: {paper.abstract}\n\nFull text"
              f"{f' (truncated to the first {char_limit} characters)' if truncated else ''}:\n{body}")
    resp = client.messages.create(
        model=cfg["model"],
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": cfg.get("effort", "high"),
                       "format": {"type": "json_schema", "schema": ANALYSIS_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"analysis stopped: {resp.stop_reason}")
    data = json.loads(next(b.text for b in resp.content if b.type == "text"))
    if truncated:
        data["risks"].append(f"Analyzer saw only the first {char_limit} of {len(fulltext)} characters.")
    data["source"] = f"claude:{cfg['model']}"
    return data


def analyze(paper: Paper, fulltext: str, rubric: dict, cfg: dict, client=None) -> dict:
    """Claude analysis if possible, otherwise (or on any API failure) heuristics."""
    if client is not None or has_credentials():
        try:
            return claude_analysis(paper, fulltext, cfg["llm"], client,
                                   cfg["screening"]["fulltext_char_limit"])
        except Exception as e:
            log.warning("claude analysis failed for %s, using heuristics: %s", paper.arxiv_id, e)
            out = heuristic_analysis(paper, fulltext, rubric)
            out["risks"].append(f"Claude analysis failed: {type(e).__name__}")
            return out
    return heuristic_analysis(paper, fulltext, rubric)
