"""Daily digest: Tier A (tailored today, capped at 5), Tier B (next 10 untailored at 50+), run stats from logs."""
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from main import load_config
from src.digest import build, run_stats, tier_a, tier_b
from src.models import Job
from src.store import Store

NOW = datetime(2026, 10, 12, 18, tzinfo=timezone.utc)
TODAY = NOW.astimezone().date()
MASTER = {"approved_summaries": [
    {"id": "sum_ai", "text": "x", "use_for": {"role_families": ["ai_engineer"]}},
    {"id": "sum_analytics", "text": "x", "use_for": {"role_families": ["analyst_backup"]}}]}


@pytest.fixture
def cfg():
    c = load_config()
    c["digest"] = {"tier_a_max": 5, "tier_b_max": 10, "tier_b_min_score": 50, "queue_after_run": False}
    return c


def add(store, i, fit, family="ai_engineer", posted_days=5, status="scored", tailored_at=None):
    j = Job(id=f"j{i}", company=f"Co{i}", title=f"AI Engineer {i}", location="Atlanta, GA", remote=False,
            url=f"https://job-boards.greenhouse.io/co{i}/jobs/{i}", description="d", source="greenhouse",
            posted_at=NOW - timedelta(days=posted_days))
    store.save([j], {j.id: family})
    store.save_score({"job_id": j.id, "fit_score": fit, "fast_score": fit, "escalated": 0, "model": "m",
                      "role_family": family, "matched_skills": "[]", "gaps": "[]", "sponsorship_signal": "unclear",
                      "uscis_status": "no_data", "scored_at": "2026-10-12", "one_line_reason": f"reason {i}"})
    if status == "tailored":
        store.mark_tailored(j.id, tailored_at or NOW.isoformat())
    elif status != "scored":
        store.set_status(j.id, status)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "jobs.db")
    yield s
    s.close()


def test_tier_a_is_today_only_ranked_and_capped_at_5(store, cfg):
    for i in range(7):
        add(store, i, 66 + i, status="tailored")
    add(store, 9, 90, status="tailored", tailored_at=(NOW - timedelta(days=2)).isoformat())   # not today
    rows, total = tier_a(store, cfg, TODAY)
    assert total == 7 and len(rows) == 5
    assert [r["fit_score"] for r in rows] == [72, 71, 70, 69, 68]
    md = build(store, cfg, MASTER, TODAY, NOW)
    assert "+ 2 more tailored today" in md and "Co9" not in md.split("## Tier B")[0]


def test_tier_b_next_10_untailored_50_plus_with_summary(store, cfg):
    add(store, 0, 70, status="tailored")
    add(store, 1, 49)                                       # under 50
    add(store, 2, 60, family="program_product_manager")     # no approved summary in MASTER
    add(store, 3, 62, family="analyst_backup")
    add(store, 4, 70, posted_days=1)                        # eligible, would be tailored next run
    add(store, 5, 55, status="excluded")
    for i in range(10, 22):
        add(store, i, 50 + (i % 5))
    rows, hidden = tier_b(store, cfg, MASTER, NOW)
    assert hidden == 0
    ids = [r["id"] for r in rows]
    assert len(rows) == 10 and not {"j0", "j1", "j5"} & set(ids)
    assert ids[:3] == ["j4", "j3", "j2"]
    by = {r["id"]: r for r in rows}
    assert by["j3"]["summary"] == "sum_analytics (analyst_backup)" and by["j3"]["held_by"] == "below 65"
    assert by["j2"]["summary"].startswith("none approved") and by["j2"]["held_by"] == "no approved summary"
    assert by["j4"]["held_by"] == "eligible: next run"


def test_applied_or_skipped_jobs_are_not_in_tier_b(store, cfg):
    add(store, 1, 60)
    store.upsert_application({"job_id": "j1", "status": "skipped"})
    assert tier_b(store, cfg, MASTER, NOW) == ([], 0)


def test_run_stats_sum_today_logs(tmp_path):
    def log(name, records):
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    log(f"run_{TODAY:%Y%m%d}_090000.jsonl", [
        {"kind": "model_turn", "cost_usd": 0.001},
        {"kind": "tool_call", "name": "search_jobs", "ok": True, "cost_usd": 0,
         "result": {"fetched": 8000, "passed_filters": 60, "new_jobs": 4, "jsearch_requests": 7}},
        {"kind": "tool_call", "name": "score_jobs", "ok": True, "cost_usd": 0.01, "result": {"scored_count": 4}},
        {"kind": "tool_call", "name": "tailor_resume", "ok": True, "cost_usd": 0.1, "result": {"status": "PASS"}},
        {"kind": "tool_call", "name": "tailor_resume", "ok": True, "cost_usd": 0.1, "result": {"status": "FAIL"}}])
    log("run_20200101_090000.jsonl", [{"kind": "model_turn", "cost_usd": 9}])          # another day
    (r,) = run_stats(TODAY, tmp_path)
    assert (r["searched"], r["passed"], r["new"], r["scored"], r["tailored"], r["jsearch_requests"]) == \
        (8000, 60, 4, 4, 1, 7)
    assert r["cost"] == pytest.approx(0.211)


def test_stale_jobs_hidden_from_tier_b_but_tailored_ones_kept(store, cfg):
    add(store, 1, 64, posted_days=20)                                  # scored, too old -> hidden
    add(store, 2, 55, posted_days=13)                                  # within 14 days -> shown
    add(store, 3, 70, posted_days=40, status="tailored")               # tailored today, old posting -> kept
    rows, hidden = tier_b(store, cfg, MASTER, NOW)
    assert [r["id"] for r in rows] == ["j2"] and hidden == 1
    a_rows, _ = tier_a(store, cfg, TODAY)
    assert [r["id"] for r in a_rows] == ["j3"]
    md = build(store, cfg, MASTER, TODAY, NOW)
    assert "1 job(s) scoring 50+ hidden: posted more than 14 days ago." in md


def test_hide_stale_keeps_pipeline_jobs_and_undated(cfg):
    from src.scorer import hide_stale
    old = (NOW - timedelta(days=30)).isoformat()
    rows = [{"id": "a", "status": "scored", "posted_at": old},
            {"id": "b", "status": "tailored", "posted_at": old},
            {"id": "c", "status": "scored", "posted_at": old},          # queued/applied by application status
            {"id": "d", "status": "scored", "posted_at": None},         # no date: kept
            {"id": "e", "status": "scored", "posted_at": (NOW - timedelta(days=14)).isoformat()}]
    kept, hidden = hide_stale(rows, cfg, NOW, {"c": "applied"})
    assert [r["id"] for r in kept] == ["b", "c", "d", "e"] and hidden == 1


def test_stale_jobs_are_not_tailor_candidates(store, cfg):
    from src.agent import Toolbox
    add(store, 1, 80, posted_days=30)
    add(store, 2, 70, posted_days=2)
    tb = Toolbox(cfg, store, llm=None)
    tb._master = MASTER
    assert [c["job_id"] for c in tb.tailor_candidates()] == ["j2"]


def add_jsearch(store, i, fit, direct, url, posted_days=2, status="scored"):
    j = Job(id=f"jsearch:{i}", company=f"JS{i}", title=f"Forward Deployed Engineer {i}", location="Remote - US",
            remote=True, url=url, description="d", source="jsearch", posted_at=NOW - timedelta(days=posted_days),
            apply_direct=direct, apply_via="test")
    store.save([j], {j.id: "forward_deployed"})
    store.save_score({"job_id": j.id, "fit_score": fit, "fast_score": fit, "escalated": 0, "model": "m",
                      "role_family": "forward_deployed", "matched_skills": "[]", "gaps": "[]",
                      "sponsorship_signal": "unclear", "uscis_status": "no_data", "scored_at": "2026-10-12",
                      "one_line_reason": "r"})
    if status == "tailored":
        store.mark_tailored(j.id, NOW.isoformat())


FDE_MASTER = {"approved_summaries": [{"id": "sum_ai", "text": "x", "use_for": {"role_families": ["forward_deployed"]}}]}


def test_jsearch_without_direct_link_is_never_tailored_or_shown(store, cfg):
    from src.agent import Toolbox
    add_jsearch(store, 1, 80, False, "https://www.talent.com/view?id=1")
    add_jsearch(store, 2, 70, True, "https://acme.wd5.myworkdayjobs.com/en-US/careers/job/9")
    add_jsearch(store, 3, 72, True, "https://job-boards.greenhouse.io/acme/jobs/3")
    tb = Toolbox(cfg, store, llm=None)
    tb._master = FDE_MASTER
    assert [c["job_id"] for c in tb.tailor_candidates()] == ["jsearch:3", "jsearch:2"]
    rows, _ = tier_b(store, cfg, FDE_MASTER, NOW)
    assert [r["id"] for r in rows] == ["jsearch:3", "jsearch:2"]


def test_unverified_link_is_skipped_not_failed_and_listed(store, cfg, monkeypatch):
    import src.linkcheck as lc
    from src.agent import Limits, RunState, Toolbox
    from src.linkcheck import LinkCheck
    add_jsearch(store, 3, 72, True, "https://job-boards.greenhouse.io/acme/jobs/3")
    add_jsearch(store, 2, 70, True, "https://acme.wd5.myworkdayjobs.com/en-US/careers/job/9")
    monkeypatch.setattr(lc, "verify_posting", lambda url, title, cfg: LinkCheck(False, "employer link returned HTTP 404"))
    tb = Toolbox(cfg, store, llm=None)
    tb._master = FDE_MASTER
    state = RunState(Limits.from_cfg(cfg))
    out = tb.tailor_resume(state, "jsearch:3")              # no LLM: the check fails before any paid step
    assert out.result["status"] == "skipped_unverified" and out.cost == 0 and state.tailored == []
    assert out.result["next_candidates"][0]["job_id"] == "jsearch:2"
    row = store.conn.execute("SELECT status, link_check FROM jobs WHERE id='jsearch:3'").fetchone()
    assert row["status"] == "unverified" and row["link_check"] == "unverified: employer link returned HTTP 404"
    by = {r["id"]: r for r in tier_b(store, cfg, FDE_MASTER, NOW)[0]}
    assert by["jsearch:3"]["held_by"] == "unverified: employer link returned HTTP 404"


def test_tier_a_marks_non_prefillable_employer_links(store, cfg):
    add_jsearch(store, 2, 70, True, "https://acme.wd5.myworkdayjobs.com/en-US/careers/job/9", status="tailored")
    add_jsearch(store, 3, 72, True, "https://job-boards.greenhouse.io/acme/jobs/3", status="tailored")
    md = build(store, cfg, MASTER, TODAY, NOW)
    tier_a_md = md.split("## Tier B")[0]
    assert "apply manually on employer site" in tier_a_md.split("JS2")[1].splitlines()[0]
    assert "not queued yet" in tier_a_md.split("JS3")[1].splitlines()[0]


# ---- link check (no network: httpx.get is faked; the rendered fallback is turned off) ----------------------------
def _fake_get(status=200, html="", final="https://careers.acme.ai/jobs/9"):
    import httpx

    def get(url, **kw):
        return httpx.Response(status, text=html, request=httpx.Request("GET", final))
    return get


@pytest.mark.parametrize("status, html, final, ok, note", [
    (200, "<h1>Senior Forward Deployed Engineer</h1><p>Apply</p>", "https://careers.acme.ai/jobs/9", True,
     "loads and shows the same title"),
    (404, "not found", "https://careers.acme.ai/jobs/9", False, "closed: posting returns HTTP 404"),
    (403, "forbidden", "https://careers.acme.ai/jobs/9", False, "employer link returned HTTP 403"),
    (200, "<h1>Senior Forward Deployed Engineer</h1><p>This job is no longer accepting applications.</p>",
     "https://careers.acme.ai/jobs/9", False, 'closed: page says "no longer accepting applications"'),
    (200, "<h1>Senior Forward Deployed Engineer</h1>", "https://www.linkedin.com/jobs/view/9", False,
     "employer link redirects to a job board (linkedin.com)"),
    (200, "<h1>Data Analyst</h1><script>Senior Forward Deployed Engineer</script>", "https://careers.acme.ai/jobs/9",
     False, "page loads but the title isn't in its HTML"),
])
def test_verify_posting(cfg, monkeypatch, status, html, final, ok, note):
    import httpx
    from src.linkcheck import verify_posting
    monkeypatch.setattr(httpx, "get", _fake_get(status, html, final))
    r = verify_posting("https://careers.acme.ai/jobs/9", "Senior Forward Deployed Engineer (10569)", cfg["jsearch"],
                       render=False)
    assert (r.ok, r.note) == (ok, note)


# ---- closed postings -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, found", [
    ("Sorry, this position has been filled.", "position has been filled"),
    ("We are no longer accepting applications for this role", "no longer accepting applications"),
    ("Job Closed", "Job Closed"),
    ("This job has expired", "This job has expired"),
    ("The job you are looking for is no longer available", "The job you are looking for is no longer"),
    ("This role is no longer available.", "role is no longer available"),
    ("Apply now! We are hiring across teams.", None),
    ("Our customers close deals; the role closes the loop on feedback.", None),
])
def test_closed_reason(text, found):
    from src.linkcheck import closed_reason
    assert closed_reason(text) == found


@pytest.mark.parametrize("status, html, final, note", [
    (404, "", "https://jobs.lever.co/acme/1", "closed: posting returns HTTP 404"),
    (200, "<h1>Jobs at Acme</h1>", "https://job-boards.greenhouse.io/acme?error=true",
     "closed: Greenhouse redirects to the job board (posting removed)"),
    (200, "<h1>AI Engineer</h1><p>Position filled</p>", "https://jobs.lever.co/acme/1",
     'closed: page says "Position filled"'),
    (200, "<h1>AI Engineer</h1><p>Apply</p>", "https://jobs.lever.co/acme/1", None),
    (500, "oops", "https://jobs.lever.co/acme/1", None),      # no clear sign of closure: not blocked
])
def test_posting_closed(monkeypatch, status, html, final, note):
    import httpx
    from src.linkcheck import posting_closed
    monkeypatch.setattr(httpx, "get", _fake_get(status, html, final))
    r = posting_closed("https://jobs.lever.co/acme/1", "AI Engineer", render=False)
    assert (r.note if r else None) == note and (r is None or r.closed)


def test_closed_posting_is_skipped_before_tailoring_and_leaves_digest(store, cfg, monkeypatch):
    import src.linkcheck as lc
    from src.agent import Limits, RunState, Toolbox
    from src.linkcheck import LinkCheck
    add(store, 1, 80)
    add(store, 2, 70)
    monkeypatch.setattr(lc, "check_before_tailoring",
                        lambda job, cfg: LinkCheck(False, 'closed: page says "Position filled"', closed=True)
                        if job["id"] == "j1" else None)
    tb = Toolbox(cfg, store, llm=None)
    tb._master = MASTER
    state = RunState(Limits.from_cfg(cfg))
    out = tb.tailor_resume(state, "j1")
    assert out.result["status"] == "skipped_closed" and out.cost == 0 and state.tailored == []
    assert out.result["next_candidates"][0]["job_id"] == "j2"
    assert store.conn.execute("SELECT status FROM jobs WHERE id='j1'").fetchone()[0] == "closed"
    assert "j1" not in [r["id"] for r in tier_b(store, cfg, MASTER, NOW)[0]]


def test_manual_close_removes_tailored_job_from_digest(store, cfg):
    from src.apply import close
    add(store, 1, 80, status="tailored")
    store.upsert_application({"job_id": "j1", "status": "to_review", "notes": "Fit 80."})
    close(store, "j1", "posting no longer accepting applications")
    assert tier_a(store, cfg, TODAY) == ([], 0)
    app = store.application("j1")
    assert app["status"] == "closed" and app["notes"].endswith("Posting closed: posting no longer accepting applications")


# ---- contract labels ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("etype, title, desc, label", [
    ("CONTRACTOR", "AI Engineer", "", "contract"),
    ("Full-time", "AI Engineer (Contract)", "", "contract"),
    ("Full-time", "AI Engineer", "This is a 6-month contract role.", "contract"),
    ("Full-time", "AI Engineer", "Paid as 1099.", "contract"),
    ("Contract", "AI Engineer", "Contract-to-hire after 6 months.", "contract-to-hire"),
    ("Full-time", "AI Engineer", "You will review vendor contracts and contract terms.", None),
    (None, "AI Engineer", "", None),
])
def test_contract_label(etype, title, desc, label):
    from src.digest import contract_label
    assert contract_label({"employment_type": etype, "title": title, "description": desc}) == label


def test_contract_roles_labelled_not_ranked_lower(store, cfg):
    add(store, 1, 60)
    add(store, 2, 61)
    store.conn.execute("UPDATE jobs SET employment_type='Contract' WHERE id='j2'")
    store.conn.commit()
    rows, _ = tier_b(store, cfg, MASTER, NOW)
    assert [r["id"] for r in rows] == ["j2", "j1"]                  # fit order unchanged
    md = build(store, cfg, MASTER, TODAY, NOW)
    assert "AI Engineer 2 [contract]" in md and "AI Engineer 1 [" not in md
