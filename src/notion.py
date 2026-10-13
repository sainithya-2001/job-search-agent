"""Sync the applications table to a Notion database, one page per job, upserted by job_id.

jobs.db is the source of truth. The one thing that flows back is Status: if Nithya changes a job's Status in
Notion, the next sync applies it to jobs.db (through apply.change_status, so "applied" sets the date and "closed"
closes the job, exactly like the CLI commands). Every other column is written from jobs.db.

How a sync works (sync()):
  1. Find the "Applications" database under the parent page (NOTION_PARENT_PAGE_ID in .env); create it if missing.
  2. Pull: for each page whose Status differs from what the last sync left there (table notion_sync), Notion
     changed it. If jobs.db hasn't changed that job's status since, the Notion value is applied to jobs.db. If
     both changed, jobs.db wins and the conflict is reported. A Status that isn't one of apply.STATUSES is ignored.
  3. Push: create pages for jobs not in Notion yet; update a page only when its jobs.db values changed since the
     last push (or its Status differs), so pages that didn't change cost no API calls.
"Days since applied" is a Notion formula on Date applied, so it stays current without a sync.

Setup (once): create an internal integration at notion.so/profile/integrations, put its secret in .env as
NOTION_TOKEN, and connect it to the parent page (page menu > Connections). Plain httpx, no Notion SDK.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.apply import STATUSES, change_status
from src.store import Store

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.notion.com/v1"
VERSION = "2022-06-28"
DB_TITLE = "Applications"
STATUS_COLORS = {"queued": "gray", "applied": "blue", "interviewing": "purple", "offer": "green",
                 "rejected": "red", "skipped": "brown", "closed": "default"}
DAYS_FORMULA = ('if(empty(prop("Date applied")), toNumber(""), '
                'dateBetween(now(), prop("Date applied"), "days"))')
TEXT_LIMIT = 2000                                # Notion's max characters per rich_text item


class NotionNotConfigured(RuntimeError):
    pass


def schema() -> dict:
    return {
        "Company": {"title": {}},
        "Role": {"rich_text": {}},
        "Family": {"select": {}},
        "Fit": {"number": {"format": "number"}},
        "Status": {"select": {"options": [{"name": s, "color": STATUS_COLORS[s]} for s in STATUSES]}},
        "Date applied": {"date": {}},
        "Days since applied": {"formula": {"expression": DAYS_FORMULA}},
        "Resume file": {"rich_text": {}},
        "Job link": {"url": {}},
        "Notes": {"rich_text": {}},
        "Job ID": {"rich_text": {}},                 # the upsert key
    }


def notion_status(db_status: str) -> str:
    """jobs.db status -> Notion Status option. 'to_review' (tailored, form not read yet) shows as queued."""
    return db_status if db_status in STATUSES else "queued"


def _text(s) -> list:
    s = str(s or "")[:TEXT_LIMIT]
    return [{"type": "text", "text": {"content": s}}] if s else []


def _notes(row) -> str:
    edited = json.loads(row["edited_fields"] or "[]") if row["edited_fields"] else []
    return " | ".join(([f"edited: {'; '.join(edited)}"] if edited else []) + ([row["notes"]] if row["notes"] else []))


def properties(row) -> dict:
    """Notion page properties for one row of rows() (an application joined with its score)."""
    return {
        "Company": {"title": _text(row["company"])},
        "Role": {"rich_text": _text(row["role"])},
        "Family": {"select": {"name": row["role_family"]} if row["role_family"] else None},
        "Fit": {"number": row["fit_score"]},
        "Status": {"select": {"name": notion_status(row["status"])}},
        "Date applied": {"date": {"start": row["date_applied"]} if row["date_applied"] else None},
        "Resume file": {"rich_text": _text(row["resume_file"])},
        "Job link": {"url": row["url"] or None},
        "Notes": {"rich_text": _text(_notes(row))},
        "Job ID": {"rich_text": _text(row["job_id"])},
    }


def digest(props: dict) -> str:
    return hashlib.sha256(json.dumps(props, sort_keys=True).encode()).hexdigest()[:16]


def rows(store: Store) -> list:
    return store.conn.execute(
        "SELECT a.*, s.role_family FROM applications a LEFT JOIN scores s ON s.job_id = a.job_id "
        "ORDER BY a.updated_at").fetchall()


def page_id_from(url_or_id: str) -> str:
    """A Notion page URL or id -> the 32-hex id."""
    found = re.findall(r"[0-9a-f]{32}", url_or_id.replace("-", "").lower())
    if not found:
        raise NotionNotConfigured(f"not a Notion page id or URL: {url_or_id!r}")
    return found[-1]


# ---- HTTP client ------------------------------------------------------------------------------------------------
class NotionClient:
    """The few Notion REST calls the sync needs, with retry on rate limits (429) and 5xx."""

    def __init__(self, token: str, http: httpx.Client | None = None, retries: int = 4, pause: float = 0.34):
        self.http = http or httpx.Client(base_url=API, timeout=30, headers={
            "Authorization": f"Bearer {token}", "Notion-Version": VERSION, "Content-Type": "application/json"})
        self.retries, self.pause = retries, pause           # pause between calls: Notion allows ~3 requests/s

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        for attempt in range(self.retries + 1):
            try:
                r = self.http.request(method, path, json=body)
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == self.retries:
                    r.raise_for_status()
                time.sleep(float(r.headers.get("Retry-After") or 2 ** attempt))
                continue
            if r.status_code >= 400:
                info = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                raise RuntimeError(f"Notion {method} {path}: HTTP {r.status_code} {info.get('code', '')}: "
                                   f"{info.get('message', r.text[:200])}")
            time.sleep(self.pause)
            return r.json()
        raise AssertionError("unreachable")

    def database(self, parent_id: str) -> tuple[str, bool]:
        """The id of the "Applications" database under the parent page, creating it if needed. -> (id, created)"""
        cursor = None
        while True:
            q = "?page_size=100" + (f"&start_cursor={cursor}" if cursor else "")
            res = self._call("GET", f"/blocks/{parent_id}/children{q}")
            for b in res["results"]:
                if b["type"] == "child_database" and b["child_database"]["title"] == DB_TITLE:
                    return b["id"], False
            if not res.get("has_more"):
                break
            cursor = res["next_cursor"]
        db = self._call("POST", "/databases", {"parent": {"type": "page_id", "page_id": parent_id},
                                               "title": _text(DB_TITLE), "properties": schema()})
        return db["id"], True

    def pages(self, db_id: str) -> dict[str, tuple[str, str | None]]:
        """job_id -> (page_id, Status name) for every live page in the database."""
        out, cursor = {}, None
        while True:
            body = {"page_size": 100} | ({"start_cursor": cursor} if cursor else {})
            res = self._call("POST", f"/databases/{db_id}/query", body)
            for p in res["results"]:
                props = p["properties"]
                job_id = "".join(t["plain_text"] for t in props.get("Job ID", {}).get("rich_text", []))
                status = (props.get("Status", {}).get("select") or {}).get("name")
                if job_id:
                    out[job_id] = (p["id"], status)
            if not res.get("has_more"):
                return out
            cursor = res["next_cursor"]

    def create(self, db_id: str, props: dict) -> str:
        return self._call("POST", "/pages", {"parent": {"database_id": db_id}, "properties": props})["id"]

    def update(self, page_id: str, props: dict) -> None:
        self._call("PATCH", f"/pages/{page_id}", {"properties": props})


# ---- sync -------------------------------------------------------------------------------------------------------
@dataclass
class SyncResult:
    database_created: bool = False
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    pulled: list[tuple[str, str, str]] = field(default_factory=list)       # (job_id, old status, new status)
    conflicts: list[str] = field(default_factory=list)

    def summary(self) -> str:
        s = (f"Notion: {self.created} added, {self.updated} updated, {self.unchanged} unchanged"
             + (" (created the Applications database)" if self.database_created else ""))
        if self.pulled:
            s += "; pulled from Notion: " + ", ".join(f"{j} {a} -> {b}" for j, a, b in self.pulled)
        if self.conflicts:
            s += "; kept jobs.db for: " + "; ".join(self.conflicts)
        return s


def sync(store: Store, client, parent_id: str) -> SyncResult:
    res = SyncResult()
    db_id, res.database_created = client.database(parent_id)
    pages = client.pages(db_id)
    state = {r["job_id"]: r for r in store.conn.execute("SELECT * FROM notion_sync")}

    # 1. pull Status changes made in Notion
    for job_id, (page_id, n_status) in pages.items():
        st, app = state.get(job_id), store.application(job_id)
        if st is None or app is None or n_status is None or n_status == st["status"]:
            continue
        here = notion_status(app["status"])
        if n_status not in STATUSES:
            res.conflicts.append(f"{job_id}: Notion Status {n_status!r} is not one of {', '.join(STATUSES)}")
        elif here != st["status"]:
            res.conflicts.append(f"{job_id}: changed in both (Notion {n_status}, jobs.db {here})")
        else:
            try:
                change_status(store, job_id, n_status, source="notion")
                res.pulled.append((job_id, here, n_status))
            except ValueError as e:
                res.conflicts.append(f"{job_id}: {e}")

    # 2. push jobs.db -> Notion
    now = datetime.now(timezone.utc).isoformat()
    for row in rows(store):
        props = properties(row)
        h, status = digest(props), props["Status"]["select"]["name"]
        page = pages.get(row["job_id"])
        st = state.get(row["job_id"])
        if page is None:
            page_id = client.create(db_id, props)
            res.created += 1
        elif st is None or st["pushed"] != h or page[1] != status or st["page_id"] != page[0]:
            page_id = page[0]
            client.update(page_id, props)
            res.updated += 1
        else:
            res.unchanged += 1
            continue
        with store.conn:
            store.conn.execute("INSERT OR REPLACE INTO notion_sync (job_id, page_id, status, pushed, synced_at) "
                               "VALUES (?, ?, ?, ?, ?)", (row["job_id"], page_id, status, h, now))
    return res


def configured() -> tuple[str, str]:
    from src.llm import load_env

    load_env(ROOT / ".env")
    token, parent = os.environ.get("NOTION_TOKEN"), os.environ.get("NOTION_PARENT_PAGE_ID")
    missing = [n for n, v in (("NOTION_TOKEN", token), ("NOTION_PARENT_PAGE_ID", parent)) if not v]
    if missing:
        raise NotionNotConfigured("Notion sync skipped: missing " + " and ".join(missing) + " in .env")
    return token, page_id_from(parent)


def try_sync(store: Store) -> str:
    """Sync if configured; otherwise say why not. Never raises (jobs.db is the source of truth)."""
    try:
        token, parent = configured()
        return sync(store, NotionClient(token), parent).summary()
    except NotionNotConfigured as e:
        return str(e)
    except Exception as e:      # noqa: BLE001
        return f"Notion sync failed ({type(e).__name__}: {e}); jobs.db is unaffected. Retry with `notion-sync`"
