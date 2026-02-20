"""SQLite state: which papers were already screened (dedupe across days) and
every reproduction attempt (for the leaderboard)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id   TEXT PRIMARY KEY,
    first_seen TEXT NOT NULL,          -- YYYY-MM-DD of the screen that found it
    title      TEXT NOT NULL,
    score      REAL NOT NULL,
    paper      TEXT NOT NULL,          -- Paper.to_dict() JSON
    rubric     TEXT NOT NULL,          -- Score.to_dict() JSON
    analysis   TEXT                    -- analysis JSON, may be NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id  TEXT NOT NULL,
    day       TEXT NOT NULL,
    backend   TEXT NOT NULL,
    status    TEXT NOT NULL,
    results   TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def unseen(self, ids: list[str]) -> list[str]:
        if not ids:
            return []
        q = f"SELECT arxiv_id FROM papers WHERE arxiv_id IN ({','.join('?' * len(ids))})"
        seen = {r[0] for r in self.db.execute(q, ids)}
        return [i for i in ids if i not in seen]

    def add_screened(self, day: str, paper: dict, rubric: dict, analysis: dict | None) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO papers VALUES (?,?,?,?,?,?,?)",
                (paper["arxiv_id"], day, paper["title"], rubric["total"], json.dumps(paper),
                 json.dumps(rubric), json.dumps(analysis) if analysis else None),
            )

    def screened_on(self, day: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM papers WHERE first_seen = ? ORDER BY score DESC", (day,))
        return [self._row(r) for r in rows]

    def get(self, arxiv_id: str) -> dict | None:
        r = self.db.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
        return self._row(r) if r else None

    def days(self) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT DISTINCT first_seen FROM papers ORDER BY first_seen DESC")]

    def record_run(self, arxiv_id: str, day: str, backend: str, status: str, results: dict) -> None:
        with self.db:
            self.db.execute("INSERT INTO runs (arxiv_id, day, backend, status, results) VALUES (?,?,?,?,?)",
                            (arxiv_id, day, backend, status, json.dumps(results)))

    def latest_runs(self) -> list[dict]:
        """Most recent run per (paper, backend)."""
        rows = self.db.execute(
            "SELECT * FROM runs WHERE id IN (SELECT MAX(id) FROM runs GROUP BY arxiv_id, backend) "
            "ORDER BY day DESC, arxiv_id")
        return [{**dict(r), "results": json.loads(r["results"])} for r in rows]

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in ("paper", "rubric", "analysis"):
            d[k] = json.loads(d[k]) if d[k] else None
        return d
