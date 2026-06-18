import json

from paperforge.report import write_daily, write_index, write_leaderboard, write_paper_folder

PAPER = {"arxiv_id": "1206.1901", "title": "MCMC <using> Hamiltonian dynamics", "authors": ["Radford M. Neal"],
         "published": "2012-06-09"}
RESULTS = {
    "paper": {"arxiv_id": "1206.1901", "title": PAPER["title"], "url": "https://arxiv.org/abs/1206.1901"},
    "backend": "scripted", "status": "partial", "tests_passed": True, "metrics": {"a": 0.14},
    "comparisons": [
        {"name": "a", "kind": "numeric", "claimed": 0.13, "reproduced": 0.14, "tolerance": 0.03, "verdict": "match", "source": "Sec 3.3"},
        {"name": "b", "kind": "qualitative", "claimed": True, "reproduced": None, "tolerance": None, "verdict": "missing", "source": "Fig 7"}],
    "attempts": 1, "sandbox": "subprocess", "runtime_s": 3.2, "error": None,
    "notes": ["Hand-written reference implementation."], "fidelity_notes": ["- seed fixed"],
}
SCREENED = [{
    "arxiv_id": "2609.1", "title": "Tiny <b>method</b>", "score": 72.5,
    "rubric": {"total": 72.5, "criteria": {k: {"score": 0.5, "weight": 1, "evidence": [f"+ {k} signal"]}
                                           for k in ("compute", "data", "clarity", "code", "interest")}},
    "analysis": {"source": "heuristic", "method_summary": "We do a thing.", "feasibility_score": 7,
                 "laptop_feasible": True, "claimed_metrics": [{"name": "accuracy", "value": 91.0, "unit": "%", "context": "abstract"}]},
}, {"arxiv_id": "2609.2", "title": "Giant model", "score": 20.0,
    "rubric": {"total": 20.0, "criteria": {k: {"score": 0.1, "weight": 1, "evidence": []}
                                           for k in ("compute", "data", "clarity", "code", "interest")}},
    "analysis": None}]
RUNS = [{"arxiv_id": "1206.1901", "day": "2026-09-25", "backend": "scripted", "status": "partial", "results": RESULTS}]


def test_paper_folder_cites_links_and_discloses(tmp_path):
    out = tmp_path / "papers" / "1206.1901"
    write_paper_folder(out, PAPER, RESULTS)
    readme = (out / "README.md").read_text()
    assert "https://arxiv.org/abs/1206.1901" in readme
    assert "Radford M. Neal (2012)" in readme
    assert "Partially reproduced" in readme and "hand-written reference" in readme
    assert "| `b` | qualitative | True | — |" in readme and "**missing**" in readme
    assert "- seed fixed" in readme
    assert json.loads((out / "results.json").read_text())["status"] == "partial"
    html = (out / "index.html").read_text()
    assert "MCMC &lt;using&gt;" in html and "<using>" not in html  # escaped


def test_daily_leaderboard_index(tmp_path):
    path = write_daily(tmp_path, "2026-09-25", SCREENED, RUNS)
    html = path.read_text()
    assert "2 new papers screened" in html and "1 analyzed" in html
    assert "Tiny &lt;b&gt;method&lt;/b&gt;" in html
    assert "Partially reproduced" in html and "1/2" in html
    md = (tmp_path / "daily" / "2026-09-25.md").read_text()
    assert "Claimed: accuracy 91.0% (abstract)" in md and "+ compute signal" in md
    write_leaderboard(tmp_path, RUNS)
    assert "papers/1206.1901/index.html" in (tmp_path / "leaderboard.html").read_text()
    write_index(tmp_path, ["2026-09-25"], RUNS)
    idx = (tmp_path / "index.html").read_text()
    assert "daily/2026-09-25.html" in idx and "prefers-color-scheme:dark" in idx


def test_daily_without_runs_says_opt_in(tmp_path):
    md_path = write_daily(tmp_path, "2026-09-26", SCREENED[1:], [])
    assert "No reproduction runs today" in md_path.read_text()
    assert "opt-in" in (tmp_path / "daily" / "2026-09-26.md").read_text()
