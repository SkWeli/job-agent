"""Long-term memory, part 1: the job tracker (SQLite).

Everything the agent has ever seen, scored, drafted or sent survives between runs, so it never
re-reads or re-scores a job, and it can answer "what did I apply to last week?".
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import Job

STATUSES = ["new", "filtered", "assessed", "drafted", "emailed", "applied", "interview", "offer",
            "rejected", "skipped"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, source TEXT, title TEXT, company TEXT, url TEXT, location TEXT,
    salary TEXT, posted_at TEXT, description TEXT, extra TEXT,
    status TEXT DEFAULT 'new', filter_reason TEXT,
    score INTEGER, verdict TEXT, reasons TEXT, gaps TEXT,
    draft_dir TEXT, notes TEXT, first_seen TEXT, updated TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Tracker:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def add(self, job: Job, status: str = "new", filter_reason: str | None = None) -> bool:
        if self.get(job.uid):
            return False
        self.conn.execute(
            "INSERT INTO jobs (id, source, title, company, url, location, salary, posted_at, description, extra,"
            " status, filter_reason, first_seen, updated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (job.uid, job.source, job.title, job.company, job.url, job.location, job.salary, job.posted_at,
             job.description, json.dumps(job.extra), status, filter_reason, now(), now()))
        self.conn.commit()
        return True

    def get(self, job_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()

    def update(self, job_id: str, **fields) -> None:
        fields["updated"] = now()
        cols = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))
        self.conn.commit()

    def query(self, statuses: list[str] | None = None, min_score: int = 0, limit: int = 20) -> list[sqlite3.Row]:
        sql, args = "SELECT * FROM jobs WHERE COALESCE(score, 0) >= ?", [min_score]
        if statuses:
            sql += f" AND status IN ({','.join('?' * len(statuses))})"
            args += statuses
        sql += " ORDER BY COALESCE(score, 0) DESC, first_seen DESC LIMIT ?"
        return self.conn.execute(sql, (*args, limit)).fetchall()

    def counts(self) -> dict:
        return {r["status"]: r["n"] for r in
                self.conn.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status")}

    @staticmethod
    def to_job(row) -> Job:
        return Job(source=row["source"], title=row["title"], company=row["company"], url=row["url"],
                   location=row["location"] or "", description=row["description"] or "",
                   posted_at=row["posted_at"] or "", salary=row["salary"] or "",
                   extra=json.loads(row["extra"] or "{}"))
