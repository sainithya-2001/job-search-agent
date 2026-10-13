"""Dated backups of jobs.db: data/backups/jobs_YYYY-MM-DD.db, one per day (a second run that day replaces it),
keeping the last backup.keep_days days (config.yaml). Uses SQLite's online backup API, so the copy is consistent
even while the database is open."""
from __future__ import annotations

import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path

NAME = re.compile(r"^jobs_(\d{4}-\d{2}-\d{2})\.db$")


def backup_db(db_path: Path, dest: Path, keep_days: int = 14, today: date | None = None) -> tuple[Path, list[Path]]:
    """Write today's backup and delete ones older than keep_days days. -> (backup path, deleted paths)"""
    today = today or date.today()
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"jobs_{today.isoformat()}.db"
    tmp = out.with_suffix(".db.tmp")
    src, dst = sqlite3.connect(db_path), sqlite3.connect(tmp)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    tmp.replace(out)
    oldest = today - timedelta(days=keep_days - 1)          # keep_days=14: today and the 13 days before
    deleted = []
    for p in dest.iterdir():
        m = NAME.match(p.name)
        if m and date.fromisoformat(m.group(1)) < oldest:
            p.unlink()
            deleted.append(p)
    return out, deleted
