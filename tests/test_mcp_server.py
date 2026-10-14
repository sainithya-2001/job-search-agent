"""MCP server: the five tools answer from a temporary database, never write it, and never reach the LLM,
tailoring, form-filling or apply code. Driven through the SDK's in-process client, like a real MCP client."""
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anyio
import pytest
from mcp import Client

from main import load_config
from src.models import Job
from src.sources import FetchResult
from src.store import Store

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.now(timezone.utc)
TOOLS = {"get_digest", "list_ranked_jobs", "get_job", "get_application_status", "run_search_dry_run"}


def job(n: int, title="AI Engineer", days_old=1) -> Job:
    return Job(id=f"greenhouse:acme{n}:{n}", company=f"Acme{n}", title=title, location="Remote - US", remote=True,
               url=f"https://job-boards.greenhouse.io/acme{n}/jobs/{n}", description="Build LLM apps. " * 200,
               source="greenhouse", posted_at=NOW - timedelta(days=days_old))


def score(store: Store, job_id: str, fit: int) -> None:
    store.save_score({"job_id": job_id, "fit_score": fit, "fast_score": fit, "escalated": 0, "model": "fake",
                      "role_family": "ai_engineer", "matched_skills": json.dumps(["Python"]),
                      "gaps": json.dumps(["Go"]), "sponsorship_signal": "unclear", "is_staffing_firm": 0,
                      "seniority_ok": 1, "one_line_reason": f"reason {job_id}", "cost_usd": 0.001,
                      "scored_at": NOW.isoformat()})
    store.set_status(job_id, "scored")


@pytest.fixture
def env(tmp_path):
    db, out = tmp_path / "jobs.db", tmp_path / "output"
    out.mkdir()
    s = Store(db)
    s.save([job(1), job(2), job(3), job(4), job(5, days_old=30)])
    for jid, fit in [("greenhouse:acme1:1", 80), ("greenhouse:acme2:2", 60), ("greenhouse:acme3:3", 70),
                     ("greenhouse:acme4:4", 90), ("greenhouse:acme5:5", 85)]:
        score(s, jid, fit)
    s.set_status("greenhouse:acme1:1", "tailored")
    s.upsert_application({"job_id": "greenhouse:acme1:1", "company": "Acme1", "role": "AI Engineer",
                          "status": "queued", "date_found": "2026-10-01", "fit_score": 80})
    s.upsert_application({"job_id": "greenhouse:acme4:4", "company": "Acme4", "role": "AI Engineer",
                          "status": "applied", "date_found": "2026-10-01", "date_applied": "2026-10-02",
                          "fit_score": 90})
    s.save_queue({"job_id": "greenhouse:acme1:1", "ats": "greenhouse", "form_url": "https://x/form",
                  "questions": "[]", "matches": json.dumps([{"key": "email", "value": "secret-answer@example.com"}]),
                  "captcha": json.dumps(["recaptcha"]),
                  "gates": json.dumps([{"name": "fit_score", "passed": True, "detail": "80"}]),
                  "shadow_qualified": 0, "answered": 9, "total": 10, "queued_at": NOW.isoformat()})
    s.close()
    (out / "digest_2026-10-11.md").write_text("# Job digest 2026-10-11\n", encoding="utf-8")
    (out / "digest_2026-10-12.md").write_text("# Job digest 2026-10-12\n", encoding="utf-8")
    cfg = load_config()
    from src.mcp_server import build_server
    return {"server": build_server(db, out, config=lambda: cfg), "db": db}


def call(server, name: str, args: dict | None = None):
    async def go():
        async with Client(server) as c:
            return await c.call_tool(name, args or {})
    return anyio.run(go)


def payload(result):
    assert not result.is_error, result.content
    return json.loads(result.content[0].text) if result.content[0].text.lstrip().startswith(("{", "[")) \
        else result.content[0].text


def ranked(result) -> list[dict]:
    # A list return value is sent as structured content {"result": [...]}; the text blocks hold one item each.
    return result.structured_content["result"]


def test_exactly_five_read_only_tools(env):
    async def go():
        async with Client(env["server"]) as c:
            return (await c.list_tools()).tools
    tools = anyio.run(go)
    assert {t.name for t in tools} == TOOLS
    assert all(t.annotations.read_only_hint and not t.annotations.destructive_hint for t in tools)


def test_list_ranked_jobs_matches_pipeline_ranking(env):
    rows = ranked(call(env["server"], "list_ranked_jobs", {"limit": 10}))
    ids = [r["job_id"] for r in rows]
    # applied (acme4) and stale (acme5, 30 days old) are left out; the rest ranked by fit
    assert ids == ["greenhouse:acme1:1", "greenhouse:acme3:3", "greenhouse:acme2:2"]
    assert rows[0]["rank"] == 1 and rows[0]["application_status"] == "queued" and rows[0]["fresh"] is True
    assert len(ranked(call(env["server"], "list_ranked_jobs", {"limit": 1}))) == 1


def test_get_job_has_score_and_queue_but_no_answer_values(env):
    r = call(env["server"], "get_job", {"job_id": "greenhouse:acme1:1"})
    d = payload(r)
    assert d["score"]["fit_score"] == 80 and d["score"]["gaps"] == ["Go"]
    assert d["application"]["status"] == "queued"
    assert d["review_queue"]["answered"] == 9 and d["review_queue"]["captcha"] == ["recaptcha"]
    assert len(d["description"]) <= 1501                       # truncated
    assert "secret-answer@example.com" not in r.content[0].text  # pre-filled answers never leave the server


def test_get_job_unknown_id_is_an_error(env):
    assert call(env["server"], "get_job", {"job_id": "nope"}).is_error


def test_get_application_status_counts(env):
    d = payload(call(env["server"], "get_application_status"))
    assert d["counts"] == {"queued": 1, "applied": 1}
    assert {a["job_id"] for a in d["applications"]} == {"greenhouse:acme1:1", "greenhouse:acme4:4"}


def test_get_digest_reads_file_latest_by_default(env):
    assert payload(call(env["server"], "get_digest", {"date": "2026-10-11"})).startswith("# Job digest 2026-10-11")
    assert payload(call(env["server"], "get_digest")).startswith("# Job digest 2026-10-12")
    assert call(env["server"], "get_digest", {"date": "yesterday"}).is_error


def test_get_digest_builds_in_memory_when_no_file(env, monkeypatch):
    import src.resume
    sample = src.resume.load_master(ROOT / "data" / "master_resume.example.yaml")
    monkeypatch.setattr(src.resume, "load_master", lambda path: sample)
    text = payload(call(env["server"], "get_digest", {"date": "2026-10-01"}))
    assert text.startswith("# Job digest 2026-10-01")


def test_dry_run_search_saves_nothing_and_never_calls_jsearch(env, monkeypatch):
    import src.pipeline
    import src.sources.jsearch

    async def fake_fetch_all(companies, concurrency=10):
        return FetchResult(jobs=[job(1), job(9, title="Applied AI Engineer"), job(10, title="Frontend Engineer")],
                           per_board={"greenhouse:acme": 3})

    async def no_jsearch(*a, **k):
        raise AssertionError("JSearch must never be called by the MCP server")
    monkeypatch.setattr(src.pipeline, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(src.sources.jsearch, "fetch", no_jsearch)
    monkeypatch.setenv("JSEARCH_API_KEY", "fake-key")       # even with a key present
    d = payload(call(env["server"], "run_search_dry_run"))
    assert d["fetched"] == 3 and d["passed_filters"] == 2
    assert [j["job_id"] for j in d["new_jobs"]] == ["greenhouse:acme9:9"]   # acme1 is already stored
    conn = sqlite3.connect(env["db"])
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 5


def test_no_tool_writes_or_reaches_llm_tailor_forms_or_apply(env, monkeypatch):
    """Every spending / acting entry point raises; all tools still succeed and the database bytes don't change."""
    import src.apply
    import src.forms
    import src.llm
    import src.pipeline
    import src.tailor

    def forbidden(*a, **k):
        raise AssertionError("the MCP server must not reach this")
    for mod, names in [(src.llm, ["get_llm"]), (src.tailor, ["tailor_job"]),
                       (src.pipeline, ["tailor_and_publish"]), (src.forms, ["fill", "safe_click", "hand_over"]),
                       (src.apply, ["queue_jobs", "review", "confirm", "change_status", "close"])]:
        for n in names:
            monkeypatch.setattr(mod, n, forbidden)
    monkeypatch.setattr(src.llm.AnthropicProvider, "__init__", forbidden)
    monkeypatch.setattr(src.llm.GeminiProvider, "__init__", forbidden)

    async def fake_fetch_all(companies, concurrency=10):
        return FetchResult(jobs=[job(9)], per_board={"greenhouse:acme": 1})
    monkeypatch.setattr(src.pipeline, "fetch_all", fake_fetch_all)

    before = hashlib.sha256(env["db"].read_bytes()).hexdigest()
    for name, args in [("get_digest", {}), ("list_ranked_jobs", {"limit": 50}),
                       ("get_job", {"job_id": "greenhouse:acme1:1"}), ("get_application_status", {}),
                       ("run_search_dry_run", {})]:
        assert not call(env["server"], name, args).is_error, name
    assert hashlib.sha256(env["db"].read_bytes()).hexdigest() == before


def test_read_only_store_refuses_writes(env):
    s = Store.read_only(env["db"])
    with pytest.raises(sqlite3.OperationalError):
        s.set_status("greenhouse:acme1:1", "closed")
    s.close()
    with pytest.raises(FileNotFoundError):
        Store.read_only(env["db"].parent / "missing.db")
