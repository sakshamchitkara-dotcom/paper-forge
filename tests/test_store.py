from paperforge.store import Store


def _paper(i):
    return {"arxiv_id": i, "title": f"paper {i}"}


def test_dedupe_across_days(tmp_path):
    db = tmp_path / "f.db"
    s = Store(db)
    s.add_screened("2026-09-24", _paper("1"), {"total": 50.0}, None)
    s.add_screened("2026-09-24", _paper("2"), {"total": 70.0}, {"method_summary": "x"})
    s = Store(db)  # reopen: state persists
    assert s.unseen(["1", "2", "3"]) == ["3"]
    s.add_screened("2026-09-25", _paper("1"), {"total": 99.0}, None)  # ignored: already seen
    assert [p["arxiv_id"] for p in s.screened_on("2026-09-24")] == ["2", "1"]
    assert s.screened_on("2026-09-25") == []
    assert s.get("2")["analysis"] == {"method_summary": "x"}
    assert s.days() == ["2026-09-24"]


def test_latest_run_per_paper_and_backend():
    s = Store(":memory:")
    s.record_run("1", "2026-09-24", "scripted", "failed", {"n": 1})
    s.record_run("1", "2026-09-25", "scripted", "reproduced", {"n": 2})
    s.record_run("1", "2026-09-25", "claude", "partial", {"n": 3})
    runs = s.latest_runs()
    assert {(r["backend"], r["status"]) for r in runs} == {("scripted", "reproduced"), ("claude", "partial")}
    assert all(isinstance(r["results"], dict) for r in runs)
