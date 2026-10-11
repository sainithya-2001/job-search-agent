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

-- One row per scored job. fit_score etc. are the FINAL values (strong tier if escalated, else fast tier);
-- fast_score is kept so we can see how often and how much escalation changes the answer.
CREATE TABLE IF NOT EXISTS scores (
    job_id             TEXT PRIMARY KEY REFERENCES jobs(id),
    fit_score          INTEGER NOT NULL,
    fast_score         INTEGER NOT NULL,
    escalated          INTEGER NOT NULL,
    model              TEXT NOT NULL,          -- model that produced the final score
    role_family        TEXT,
    matched_skills     TEXT,                   -- JSON list
    gaps               TEXT,                   -- JSON list
    sponsorship_signal TEXT,                   -- yes / no / unclear (from the posting text)
    sponsorship_quote  TEXT,
    uscis_status       TEXT,                   -- sponsor / zero_approvals / not_found / no_data
    uscis_approvals    INTEGER,
    e_verify           TEXT,                   -- yes / no / unknown (from config.yaml, filled by hand)
    is_staffing_firm   INTEGER,
    seniority_ok       INTEGER,
    one_line_reason    TEXT,
    input_tokens       INTEGER,
    output_tokens      INTEGER,
    cost_usd           REAL,
    scored_at          TEXT NOT NULL
);

-- Application log, one row per job, upserted by the agent's log_application tool. Same columns as the
-- Phase 5 Google Sheet, so sheets.py only has to sync this table. The agent may write status 'to_review' or
-- 'skipped' only; 'applied' is set after Nithya submits and confirms (Phase 5), never by the agent.
CREATE TABLE IF NOT EXISTS applications (
    job_id         TEXT PRIMARY KEY REFERENCES jobs(id),
    date_found     TEXT,
    company        TEXT,
    role           TEXT,
    location       TEXT,
    url            TEXT,
    fit_score      INTEGER,
    sponsorship    TEXT,
    matched_skills TEXT,
    gaps           TEXT,
    resume_file    TEXT,
    status         TEXT NOT NULL,
    date_applied   TEXT,
    notes          TEXT,
    updated_at     TEXT NOT NULL,
    review_outcome TEXT,                       -- approved_unchanged / approved_edited / abandoned / submit_failed
    edited_fields  TEXT,                       -- JSON list of pre-filled questions Nithya changed
    shadow_qualified INTEGER,                  -- would it have passed every auto-submit gate (shadow mode)?
    shadow_blockers  TEXT                      -- JSON list of failed gates
);

-- Phase 5 review queue: what the form asks and how much the answer bank covers, read before Nithya opens it.
CREATE TABLE IF NOT EXISTS apply_queue (
    job_id           TEXT PRIMARY KEY REFERENCES jobs(id),
    ats              TEXT NOT NULL,
    form_url         TEXT NOT NULL,
    questions        TEXT NOT NULL,            -- JSON list of form questions
    matches          TEXT NOT NULL,            -- JSON list of answer-bank match results
    captcha          TEXT,                     -- JSON list of CAPTCHA markers found
    gates            TEXT,                     -- JSON list of shadow gate results
    shadow_qualified INTEGER NOT NULL,
    answered         INTEGER NOT NULL,
    total            INTEGER NOT NULL,
    queued_at        TEXT NOT NULL
);
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
        # Lightweight migration: add columns introduced after a table was first created.
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(scores)")}
        if "e_verify" not in cols:
            with self.conn:
                self.conn.execute("ALTER TABLE scores ADD COLUMN e_verify TEXT DEFAULT 'unknown'")
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(applications)")}
        for col in ("review_outcome TEXT", "edited_fields TEXT", "shadow_qualified INTEGER", "shadow_blockers TEXT"):
            if col.split()[0] not in cols:
                with self.conn:
                    self.conn.execute(f"ALTER TABLE applications ADD COLUMN {col}")

    def jobs_by_status(self, status: str) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM jobs WHERE status=?", (status,)).fetchall()

    def scored_rows(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT j.*, s.* FROM scores s JOIN jobs j ON j.id = s.job_id ORDER BY s.fit_score DESC").fetchall()

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

    def preview_new(self, jobs: list[Job]) -> list[Job]:
        """Dry-run twin of save(): which of these jobs WOULD be inserted (same two dedupe rules), writing nothing."""
        seen_ids, seen_keys, new = set(), set(), []
        for job in jobs:
            key = dedupe_key(job)
            if job.id in seen_ids or key in seen_keys:
                continue
            seen_ids.add(job.id); seen_keys.add(key)
            exists = self.conn.execute("SELECT 1 FROM jobs WHERE id=? OR dedupe_key=?", (job.id, key)).fetchone()
            if not exists:
                new.append(job)
        return new

    def upsert_application(self, row: dict) -> None:
        row = {**row, "updated_at": datetime.now(timezone.utc).isoformat()}
        cols, marks = ", ".join(row), ", ".join("?" for _ in row)
        updates = ", ".join(f"{c}=excluded.{c}" for c in row if c != "job_id")
        with self.conn:
            self.conn.execute(f"INSERT INTO applications ({cols}) VALUES ({marks}) "
                              f"ON CONFLICT(job_id) DO UPDATE SET {updates}", tuple(row.values()))

    def application(self, job_id: str):
        return self.conn.execute("SELECT * FROM applications WHERE job_id=?", (job_id,)).fetchone()

    def applications(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM applications ORDER BY updated_at").fetchall()

    def save_queue(self, row: dict) -> None:
        cols, marks = ", ".join(row), ", ".join("?" for _ in row)
        with self.conn:
            self.conn.execute(f"INSERT OR REPLACE INTO apply_queue ({cols}) VALUES ({marks})", tuple(row.values()))

    def queue_row(self, job_id: str):
        return self.conn.execute("SELECT * FROM apply_queue WHERE job_id=?", (job_id,)).fetchone()

    def count(self, status: str | None = None) -> int:
        if status:
            return self.conn.execute("SELECT COUNT(*) FROM jobs WHERE status=?", (status,)).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    def set_status(self, job_id: str, status: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))

    def unscored_jobs(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM jobs WHERE status='new' AND id NOT IN (SELECT job_id FROM scores)").fetchall()

    def save_score(self, row: dict) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self.conn:
            self.conn.execute(f"INSERT OR REPLACE INTO scores ({cols}) VALUES ({marks})", tuple(row.values()))
            # Only new jobs move to 'scored'; re-scoring never moves a tailored/applied job backwards.
            self.conn.execute("UPDATE jobs SET status='scored' WHERE id=? AND status='new'", (row["job_id"],))

    def close(self) -> None:
        self.conn.close()
