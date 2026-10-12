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
            url=f"https://example.com/{i}", description="d", source="greenhouse",
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
    rows = tier_b(store, cfg, MASTER)
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
    assert tier_b(store, cfg, MASTER) == []


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
