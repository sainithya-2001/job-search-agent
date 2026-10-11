"""Sync the local applications table to Google Sheets (one row per job, upserted by job_id).

The SQLite table is the source of truth; the sheet is a view of it. Setup (once):
  1. Google Cloud console: create a service account, enable the Google Sheets API, download its JSON key to
     data/google_service_account.json (gitignored), or set GOOGLE_SERVICE_ACCOUNT_FILE in .env.
  2. Create a sheet, share it with the service account's email as Editor, put its id in .env as GOOGLE_SHEET_ID.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KEY = ROOT / "data" / "google_service_account.json"
WORKSHEET = "Applications"
# Spec columns, plus job_id (the upsert key) and review_outcome.
COLUMNS = ["job_id", "date_found", "company", "role", "location", "url", "fit_score", "sponsorship",
           "matched_skills", "gaps", "resume_file", "status", "date_applied", "review_outcome", "notes"]


class SheetsNotConfigured(RuntimeError):
    pass


def _cell(row, col: str) -> str:
    v = row[col] if col in row.keys() else None
    if col == "notes":                       # fold the list of edited answers into notes
        edited = json.loads(row["edited_fields"] or "[]") if "edited_fields" in row.keys() else []
        parts = ([f"edited: {'; '.join(edited)}"] if edited else []) + ([str(v)] if v else [])
        return " | ".join(parts)[:500]
    if v is None:
        return ""
    if col in ("matched_skills", "gaps") and isinstance(v, str) and v.startswith("["):
        return ", ".join(json.loads(v))
    return str(v)


def rows_for(applications) -> list[list[str]]:
    return [[_cell(r, c) for c in COLUMNS] for r in applications]


def upsert(ws, rows: list[list[str]]) -> tuple[int, int]:
    """Write rows into a gspread-like worksheet: update rows whose job_id exists, append the rest.
    Returns (updated, appended)."""
    values = ws.get_all_values()
    if not values or values[0] != COLUMNS:
        ws.update([COLUMNS], "A1")
        values = [COLUMNS] + values[1:] if values else [COLUMNS]
    where = {r[0]: i + 1 for i, r in enumerate(values) if i > 0 and r}     # job_id -> 1-based sheet row
    end_col = chr(ord("A") + len(COLUMNS) - 1)
    updates, appends = [], []
    for row in rows:
        if row[0] in where:
            n = where[row[0]]
            updates.append({"range": f"A{n}:{end_col}{n}", "values": [row]})
        else:
            appends.append(row)
    if updates:
        ws.batch_update(updates)
    if appends:
        ws.append_rows(appends, value_input_option="RAW")
    return len(updates), len(appends)


def open_worksheet():
    import gspread
    from src.llm import load_env

    load_env(ROOT / ".env")
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    key = Path(os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE") or DEFAULT_KEY)
    if not sheet_id or not key.exists():
        missing = [n for n, ok in (("GOOGLE_SHEET_ID in .env", sheet_id), (f"service account key {key.name}", key.exists()))
                   if not ok]
        raise SheetsNotConfigured("Sheets sync skipped: missing " + " and ".join(missing))
    sh = gspread.service_account(filename=str(key)).open_by_key(sheet_id)
    try:
        return sh.worksheet(WORKSHEET)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(WORKSHEET, rows=1000, cols=len(COLUMNS))


def sync(store) -> str:
    ws = open_worksheet()
    updated, appended = upsert(ws, rows_for(store.applications()))
    return f"Sheets: {updated} row(s) updated, {appended} added"


def try_sync(store) -> str:
    """Sync if configured; otherwise say why not. Never raises (the local table is the source of truth)."""
    try:
        return sync(store)
    except SheetsNotConfigured as e:
        return str(e)
    except Exception as e:      # noqa: BLE001
        return f"Sheets sync failed ({type(e).__name__}: {e}); the local table is unaffected — retry with `sheets-sync`"
