"""SQLite storage. Two layers of dedupe:
1. `id` primary key — the same ATS posting seen on a later run.
2. `dedupe_key` UNIQUE — same company + normalized title (e.g. one role posted per city).
`INSERT OR IGNORE` lets SQLite enforce both, so we never race a separate "exists?" query.
"""
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.models import Job

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    dedupe_key      TEXT NOT NULL UNIQUE,
    company         TEXT NOT NULL,
    title           TEXT NOT NULL,
    location        TEXT,
    remote          INTEGER NOT NULL,
    url             TEXT,
    description     TEXT,
    source          TEXT NOT NULL,
    posted_at       TEXT,
    employment_type TEXT,
    role_hint       TEXT,
    status          TEXT NOT NULL DEFAULT 'new',   -- new -> scored -> tailored -> applied / skipped
    first_seen_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


def normalize_title(title: str) -> str:
    """'Applied AI Engineer (NYC)' and 'Applied AI Engineer' -> 'applied ai engineer'."""
    t = title.lower()
    t = re.sub(r"[\(\[].*?[\)\]]", " ", t)    # drop parentheticals, usually location/team
    t = re.sub(r"\bsr\.?\b", "senior", t)
    t = re.sub(r"[^a-z0-9+/ ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def dedupe_key(job: Job) -> str:
    company = re.sub(r"[^a-z0-9]", "", job.company.lower())
    return f"{company}|{normalize_title(job.title)}"


class Store:
    def __init__(self, path: str | Path = "data/jobs.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def save(self, jobs: list[Job], role_hints: dict[str, str] | None = None) -> list[Job]:
        """Insert jobs, skipping duplicates. Returns only the jobs that were actually new."""
        role_hints = role_hints or {}
        now = datetime.now(timezone.utc).isoformat()
        new = []
        with self.conn:  # one transaction for the whole batch
            for job in jobs:
                cur = self.conn.execute(
                    """INSERT OR IGNORE INTO jobs
                       (id, dedupe_key, company, title, location, remote, url, description,
                        source, posted_at, employment_type, role_hint, first_seen_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (job.id, dedupe_key(job), job.company, job.title, job.location, int(job.remote),
                     job.url, job.description, job.source,
                     job.posted_at.isoformat() if job.posted_at else None,
                     job.employment_type, role_hints.get(job.id), now),
                )
                if cur.rowcount:
                    new.append(job)
        return new

    def count(self, status: str | None = None) -> int:
        if status:
            return self.conn.execute("SELECT COUNT(*) FROM jobs WHERE status=?", (status,)).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    def set_status(self, job_id: str, status: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))

    def close(self) -> None:
        self.conn.close()
