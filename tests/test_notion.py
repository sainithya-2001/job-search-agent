"""Notion sync (fake client and mocked HTTP, no network), the status command's logic, and jobs.db backups."""
import sqlite3
from datetime import date, timedelta

import httpx
import pytest

from src import notion as N
from src.apply import change_status, confirm
from src.backup import backup_db
from src.models import Job
from src.store import Store


class FakeNotion:
    """In-memory stand-in for NotionClient: pages keyed by page id, holding the properties written to them."""

    def __init__(self):
        self.db, self.pages_by_id, self.calls = None, {}, []

    def database(self, parent_id):
        created = self.db is None
        self.db = self.db or "db1"
        return self.db, created

    def pages(self, db_id):
        return {"".join(t["text"]["content"] for t in p["Job ID"]["rich_text"]):
                (pid, (p["Status"]["select"] or {}).get("name")) for pid, p in self.pages_by_id.items()}

    def create(self, db_id, props):
        pid = f"p{len(self.pages_by_id) + 1}"
        self.pages_by_id[pid] = props
        self.calls.append(("create", pid))
        return pid

    def update(self, page_id, props):
        self.pages_by_id[page_id] = props
        self.calls.append(("update", page_id))

    def set_status(self, job_id, status):                  # what Nithya does in the Notion UI
        pid = self.pages(self.db)[job_id][0]
        self.pages_by_id[pid]["Status"] = {"select": {"name": status}}

    def status(self, job_id):
        return self.pages(self.db)[job_id][1]


@pytest.fixture
def db(tmp_path):
    store = Store(tmp_path / "jobs.db")
    store.save([Job(id=f"j{i}", company=f"Co{i}", title="AI Engineer", location="Atlanta, GA", remote=False,
                    url=f"https://job-boards.greenhouse.io/co{i}/jobs/{i}", description="d", source="greenhouse")
                for i in (1, 2)])
    store.save_score({"job_id": "j1", "fit_score": 72, "fast_score": 70, "escalated": 1, "model": "m",
                      "role_family": "ai_engineer", "scored_at": "2026-10-12T00:00:00+00:00"})
    store.upsert_application({"job_id": "j1", "company": "Co1", "role": "AI Engineer", "url": "https://x/1",
                              "fit_score": 72, "status": "to_review", "resume_file": "Co1_AI_Engineer.pdf"})
    store.upsert_application({"job_id": "j2", "company": "Co2", "role": "AI Engineer", "status": "queued"})
    yield store
    store.close()


def test_first_sync_creates_database_and_pages_then_nothing_to_do(db):
    fake = FakeNotion()
    r = N.sync(db, fake, "parent")
    assert r.database_created and (r.created, r.updated) == (2, 0)
    props = fake.pages_by_id["p1"]
    assert props["Company"]["title"][0]["text"]["content"] == "Co1"
    assert props["Status"]["select"]["name"] == "queued"            # to_review shows as queued
    assert props["Family"]["select"]["name"] == "ai_engineer" and props["Fit"]["number"] == 72
    assert props["Job link"]["url"] == "https://x/1" and props["Date applied"]["date"] is None
    fake.calls.clear()
    r = N.sync(db, fake, "parent")
    assert (r.created, r.updated, r.unchanged) == (0, 0, 2) and not fake.calls and not r.database_created


def test_change_in_jobs_db_updates_only_that_page(db):
    fake = FakeNotion()
    N.sync(db, fake, "parent")
    fake.calls.clear()
    confirm(db, "j1", applied=True)
    r = N.sync(db, fake, "parent")
    assert fake.calls == [("update", "p1")] and r.updated == 1
    assert fake.status("j1") == "applied"
    assert fake.pages_by_id["p1"]["Date applied"]["date"]["start"] == date.today().isoformat()


def test_status_changed_in_notion_is_pulled_into_jobs_db(db):
    fake = FakeNotion()
    N.sync(db, fake, "parent")
    fake.set_status("j2", "interviewing")
    r = N.sync(db, fake, "parent")
    assert r.pulled == [("j2", "queued", "interviewing")] and not r.conflicts
    assert db.application("j2")["status"] == "interviewing" and fake.status("j2") == "interviewing"
    assert N.sync(db, fake, "parent").unchanged == 2                 # settled: no ping-pong


def test_applied_and_closed_from_notion_behave_like_the_commands(db):
    fake = FakeNotion()
    N.sync(db, fake, "parent")
    fake.set_status("j1", "applied")
    fake.set_status("j2", "closed")
    N.sync(db, fake, "parent")
    assert db.application("j1")["date_applied"] == date.today().isoformat()
    assert db.conn.execute("SELECT status FROM jobs WHERE id='j2'").fetchone()[0] == "closed"
    assert fake.status("j2") == "closed" and fake.pages_by_id["p1"]["Date applied"]["date"]


def test_changed_in_both_places_keeps_jobs_db(db):
    fake = FakeNotion()
    N.sync(db, fake, "parent")
    fake.set_status("j2", "rejected")
    confirm(db, "j2", applied=False)                                  # skipped here meanwhile
    r = N.sync(db, fake, "parent")
    assert not r.pulled and "changed in both" in r.conflicts[0]
    assert db.application("j2")["status"] == "skipped" and fake.status("j2") == "skipped"


def test_unknown_notion_status_is_ignored_and_overwritten(db):
    fake = FakeNotion()
    N.sync(db, fake, "parent")
    fake.set_status("j2", "ghosted")
    r = N.sync(db, fake, "parent")
    assert "not one of" in r.conflicts[0] and db.application("j2")["status"] == "queued"
    assert fake.status("j2") == "queued"


def test_page_deleted_in_notion_is_recreated(db):
    fake = FakeNotion()
    N.sync(db, fake, "parent")
    del fake.pages_by_id["p1"]
    r = N.sync(db, fake, "parent")
    assert r.created == 1 and "j1" in fake.pages(fake.db)


def test_change_status_rules(db):
    confirm(db, "j1", applied=True)
    db.upsert_application({"job_id": "j1", "status": "applied", "date_applied": "2026-10-01"})
    assert change_status(db, "j1", "interviewing") == "interviewing"
    assert change_status(db, "j1", "applied") == "applied"
    assert db.application("j1")["date_applied"] == "2026-10-01"      # back from interviewing keeps the date
    with pytest.raises(ValueError):
        change_status(db, "j1", "ghosted")
    with pytest.raises(ValueError):
        change_status(db, "nope", "offer")
    db.mark_closed("j1", "posting gone")                               # closing never overwrites a later stage
    assert db.application("j1")["status"] == "applied"


def test_page_id_and_not_configured(monkeypatch):
    assert N.page_id_from("https://app.notion.com/p/Job-Applications-0123456789abcdef0123456789abcdef?source=x") \
        == "0123456789abcdef0123456789abcdef"
    monkeypatch.setattr("src.llm.load_env", lambda *a, **k: None)
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    monkeypatch.delenv("NOTION_PARENT_PAGE_ID", raising=False)
    assert N.try_sync(None).startswith("Notion sync skipped: missing NOTION_TOKEN and NOTION_PARENT_PAGE_ID")


def test_client_retries_rate_limits_and_reports_errors(monkeypatch):
    monkeypatch.setattr(N.time, "sleep", lambda s: None)
    seen = []

    def handler(req):
        seen.append(req.url.path)
        if len(seen) == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        if req.url.path.endswith("/pages/bad"):
            return httpx.Response(400, json={"code": "validation_error", "message": "nope"})
        return httpx.Response(200, json={"id": "p9"})

    c = N.NotionClient("t", http=httpx.Client(base_url=N.API, transport=httpx.MockTransport(handler)), pause=0)
    assert c.create("db", {}) == "p9" and seen == ["/v1/pages", "/v1/pages"]
    with pytest.raises(RuntimeError, match="validation_error"):
        c.update("bad", {})


def test_backup_is_a_readable_copy_and_keeps_14_days(tmp_path):
    src = tmp_path / "jobs.db"
    Store(src).close()
    dest = tmp_path / "backups"
    dest.mkdir()
    today = date(2026, 10, 12)
    for d in range(1, 20):
        (dest / f"jobs_{today - timedelta(days=d)}.db").write_bytes(b"")
    (dest / "notes.txt").write_text("keep me")
    out, deleted = backup_db(src, dest, 14, today)
    assert out.name == "jobs_2026-10-12.db"
    assert sqlite3.connect(out).execute("SELECT count(*) FROM jobs").fetchone() == (0,)
    names = sorted(p.name for p in dest.iterdir())
    assert len(deleted) == 6 and "notes.txt" in names
    assert names[0] == f"jobs_{today - timedelta(days=13)}.db" and len(names) == 15    # 14 days + notes.txt
