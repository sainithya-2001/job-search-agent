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

-- Application log, one row per job, upserted by the agent's log_application tool. src/notion.py syncs this
-- table to the Notion "Applications" database. The agent may write status 'to_review' or
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

-- ATS keywords extracted from each posting (src/keywords.py), cached so extraction is paid for once per job.
CREATE TABLE IF NOT EXISTS job_keywords (
    job_id       TEXT PRIMARY KEY REFERENCES jobs(id),
    keywords     TEXT NOT NULL,              -- JSON list of {text, kind, importance}
    model        TEXT,
    extracted_at TEXT NOT NULL
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

-- Every JSearch API request (free plan: 200/month, hard limit). A row is written BEFORE the request is sent, so a
-- crash mid-request is still counted; status/results are filled in afterwards.
CREATE TABLE IF NOT EXISTS jsearch_requests (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    sent_at          TEXT NOT NULL,            -- UTC ISO timestamp
    query            TEXT NOT NULL,
    location         TEXT,
    page             INTEGER NOT NULL,
    status           INTEGER,                  -- HTTP status (NULL = no response)
    results          INTEGER,                  -- jobs returned by this page
    quota_remaining  INTEGER,                  -- RapidAPI x-ratelimit-requests-remaining, when sent
    quota_limit      INTEGER                   -- RapidAPI x-ratelimit-requests-limit
);

-- Notion sync state per application (src/notion.py): what was last written to / read from the Notion page, so a
-- Status change made in Notion can be told apart from one made here.
CREATE TABLE IF NOT EXISTS notion_sync (
    job_id     TEXT PRIMARY KEY REFERENCES applications(job_id),
    page_id    TEXT NOT NULL,
    status     TEXT NOT NULL,                  -- Status on the page after the last sync
    pushed     TEXT NOT NULL,                  -- hash of the properties last written from jobs.db
    synced_at  TEXT NOT NULL
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
        job_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(jobs)")}
        if "tailored_at" not in job_cols:
            with self.conn:     # when the current tailored resume was made (UTC ISO); NULL for older ones
                self.conn.execute("ALTER TABLE jobs ADD COLUMN tailored_at TEXT")
        if "apply_direct" not in job_cols:
            with self.conn:     # JSearch: 1 = url is the employer's own link, 0 = only a job-board link
                self.conn.execute("ALTER TABLE jobs ADD COLUMN apply_direct INTEGER")
                self.conn.execute("ALTER TABLE jobs ADD COLUMN apply_via TEXT")
        if "link_check" not in job_cols:
            with self.conn:     # JSearch: result of the pre-tailoring employer-link check (linkcheck.py)
                self.conn.execute("ALTER TABLE jobs ADD COLUMN link_check TEXT")
                self.conn.execute("ALTER TABLE jobs ADD COLUMN link_checked_at TEXT")

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
                        source, posted_at, employment_type, role_hint, first_seen_at, apply_direct, apply_via)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (job.id, dedupe_key(job), job.company, job.title, job.location, int(job.remote),
                     job.url, job.description, job.source,
                     job.posted_at.isoformat() if job.posted_at else None,
                     job.employment_type, role_hints.get(job.id), now,
                     None if job.apply_direct is None else int(job.apply_direct), job.apply_via),
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

    # ---- JSearch request accounting ----
    def jsearch_used(self, since: str) -> int:
        """Requests sent at or after `since` (UTC ISO), failed ones included."""
        return self.conn.execute("SELECT COUNT(*) FROM jsearch_requests WHERE sent_at >= ?", (since,)).fetchone()[0]

    def jsearch_last_quota(self, since: str) -> int | None:
        """Most recent quota_remaining RapidAPI reported since `since`, if any."""
        row = self.conn.execute("SELECT quota_remaining FROM jsearch_requests WHERE sent_at >= ? AND "
                                "quota_remaining IS NOT NULL ORDER BY id DESC LIMIT 1", (since,)).fetchone()
        return row[0] if row else None

    def jsearch_sent(self, query: str, location: str, page: int, sent_at: str) -> int:
        with self.conn:
            return self.conn.execute("INSERT INTO jsearch_requests (sent_at, query, location, page) VALUES (?,?,?,?)",
                                     (sent_at, query, location, page)).lastrowid

    def jsearch_done(self, row_id: int, status: int | None, results: int | None,
                     quota_remaining: int | None, quota_limit: int | None) -> None:
        with self.conn:
            self.conn.execute("UPDATE jsearch_requests SET status=?, results=?, quota_remaining=?, quota_limit=? "
                              "WHERE id=?", (status, results, quota_remaining, quota_limit, row_id))

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

    def save_link_check(self, job_id: str, ok: bool, note: str) -> None:
        """ok=False also sets status 'unverified': skipped, never tailored."""
        with self.conn:
            self.conn.execute("UPDATE jobs SET link_check=?, link_checked_at=?" + ("" if ok else ", status='unverified'")
                              + " WHERE id=?", (("verified: " if ok else "unverified: ") + note,
                                                datetime.now(timezone.utc).isoformat(), job_id))

    def mark_closed(self, job_id: str, note: str) -> None:
        """The posting no longer accepts applications: status 'closed' (out of ranking, tailoring, the review
        queue, and the digest). An application row that isn't applied/skipped is closed too; nothing is deleted."""
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute("UPDATE jobs SET status='closed', link_check=?, link_checked_at=? WHERE id=?",
                              (note if note.startswith("closed") else f"closed: {note}", now, job_id))
            self.conn.execute("UPDATE applications SET status='closed', updated_at=?, "
                              "notes=COALESCE(notes || ' | ', '') || ? WHERE job_id=? AND status NOT IN "
                              "('applied', 'skipped', 'interviewing', 'offer', 'rejected')", (now, f"Posting closed: {note}", job_id))

    def mark_tailored(self, job_id: str, when: str | None = None) -> None:
        with self.conn:
            self.conn.execute("UPDATE jobs SET status='tailored', tailored_at=? WHERE id=?",
                              (when or datetime.now(timezone.utc).isoformat(), job_id))

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
