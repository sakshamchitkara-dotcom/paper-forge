"""Transparent reproducibility / interest rubric.

Each criterion scores 0..1 from regex signals and records *why* (the evidence list),
so every number in the daily report can be traced back to text in the paper. The
total is the weighted mean scaled to 0..100. Weights live in config
(`screening.weights`).

ponytail: regex heuristics, not understanding. They rank today's listings well
enough to pick candidates; the Claude analysis (when a key is set) is the
second opinion.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import Paper

# (pattern, delta, label). Deltas are added to the criterion's base score.
COMPUTE = [
    (r"\b\d{2,4}\s?[bB]\b|\bbillion\b|\btrillion\b", -0.5, "billion-scale model/data"),
    (r"GPU[- ]hours|\bA100s?\b|\bH100s?\b|\bTPUs?\b|\d+\s?GPUs", -0.5, "large GPU/TPU budget"),
    (r"\bpre-?train(ing|ed)?\b", -0.25, "pretraining"),
    (r"\bLLMs?\b|large language model|foundation model", -0.2, "LLM-based"),
    (r"\bdiffusion\b|\bvideo generation\b", -0.15, "diffusion/video"),
    (r"closed[- ]form|\bconvex\b|linear[- ]time|\bO\(n( log n)?\)", 0.2, "cheap/closed-form"),
    (r"\btoy\b|\bsynthetic\b|\bsimulat(ed|ion)s?\b", 0.15, "toy/synthetic experiments"),
    (r"\bCPU\b|\blaptop\b|lightweight|\bnumpy\b", 0.15, "CPU-friendly"),
]
DATA = [
    (r"\bMNIST\b|CIFAR-?10\b|\bUCI\b|\bIris\b|\bsynthetic\b|\btoy\b|\bsimulated\b", 0.35, "small/synthetic data"),
    (r"ImageNet|Common ?Crawl|web-scale|LAION|\bT tokens\b|trillion tokens", -0.4, "web-scale data"),
    (r"proprietary|in-house|private dataset|internal data", -0.4, "non-public data"),
    (r"\bbenchmarks?\b", 0.05, "public benchmarks"),
]
CLARITY = [
    (r"\bAlgorithm \d\b|pseudo-?code", 0.3, "explicit algorithm/pseudocode"),
    (r"update rule|closed[- ]form|\bwe (propose|introduce) (a|an) (simple|new|novel)", 0.2, "concrete method"),
    (r"\btheorem\b|\bwe prove\b|\bconvergence\b|\bbound\b", 0.15, "formal statement"),
    (r"\bsimple\b|straightforward|easy to implement", 0.15, "claims simplicity"),
    (r"\bsurvey\b|position paper|\bperspective\b|\bwe argue\b", -0.4, "survey/position (no method)"),
    (r"\bagentic\b|\bagents?\b.*\btools?\b|framework", -0.1, "system/framework (many moving parts)"),
]


@dataclass
class Criterion:
    score: float
    weight: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class Score:
    total: float
    criteria: dict[str, Criterion]

    def to_dict(self) -> dict:
        return {"total": round(self.total, 1), "criteria": {
            k: {"score": round(c.score, 3), "weight": c.weight, "evidence": c.evidence}
            for k, c in self.criteria.items()}}


def _apply(text: str, base: float, rules) -> tuple[float, list[str]]:
    s, ev = base, []
    for pattern, delta, label in rules:
        m = re.search(pattern, text, re.I)
        if m:
            s += delta
            ev.append(f"{'+' if delta > 0 else '-'} {label} (\"{m.group(0)}\")")
    return min(1.0, max(0.0, s)), ev


def score_paper(paper: Paper, weights: dict, keywords: list[str], fulltext: str = "",
                now: datetime | None = None) -> Score:
    text = f"{paper.title}\n{paper.abstract}\n{paper.comment}"
    rich = text + "\n" + fulltext[:20000]
    c: dict[str, Criterion] = {}

    s, ev = _apply(text, 0.5, COMPUTE)
    c["compute"] = Criterion(s, weights["compute"], ev)
    s, ev = _apply(text, 0.4, DATA)
    c["data"] = Criterion(s, weights["data"], ev)
    # positive signals may come from the full text; "this is a survey/framework" only from
    # title/abstract, since nearly every full text mentions surveys in related work
    s, ev = _apply(rich, 0.4, [r for r in CLARITY if r[1] > 0])
    s2, ev2 = _apply(text, s, [r for r in CLARITY if r[1] < 0])
    c["clarity"] = Criterion(s2, weights["clarity"], ev + ev2)

    if paper.code_url:
        c["code"] = Criterion(1.0, weights["code"], [f"+ code: {paper.code_url}"])
    else:
        c["code"] = Criterion(0.0, weights["code"], ["no public code found"])

    ev, s = [], 0.0
    hits = [k for k in keywords if k.lower() in text.lower()]
    if hits:
        s += min(0.6, 0.2 * len(hits))
        ev.append(f"+ keywords: {', '.join(hits)}")
    if paper.hf_upvotes:
        s += min(0.3, 0.1 * math.log2(1 + paper.hf_upvotes))
        ev.append(f"+ {paper.hf_upvotes} HF upvotes")
    if paper.citations:
        now = now or datetime.now(timezone.utc)
        days = max(1.0, (now - datetime.fromisoformat(paper.published.replace("Z", "+00:00"))).days)
        velocity = paper.citations / days * 30
        s += min(0.3, 0.1 * math.log2(1 + velocity))
        ev.append(f"+ {paper.citations} citations ({velocity:.1f}/month)")
    c["interest"] = Criterion(min(1.0, s), weights["interest"], ev)

    wsum = sum(x.weight for x in c.values()) or 1.0
    total = 100 * sum(x.score * x.weight for x in c.values()) / wsum
    return Score(total, c)
