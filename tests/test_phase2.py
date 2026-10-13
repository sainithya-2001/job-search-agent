import typing
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from main import load_config
from src.llm import LLMProvider, LLMResponse, cost_usd
from src.models import RoleFamily, ScoredJob, SponsorshipSignal
from src.resume import ResumeError, load_master, profile_text
from src.scorer import estimate, location_rank, score_jobs, select_jobs
from src.sponsorship import SponsorshipDB, normalize_employer
from src.store import Store


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def job(i, company="Acme", role="ai_engineer", location="San Francisco, CA", posted="2026-10-01T00:00:00+00:00"):
    return {"id": f"j{i}", "company": company, "title": f"AI Engineer {i}", "location": location, "remote": 0,
            "description": "Build LLM apps with Python.", "role_hint": role, "posted_at": posted}


def scored(score):
    return ScoredJob(fit_score=score, role_family="ai_engineer", matched_skills=["Python"], gaps=[],
                     sponsorship_signal=SponsorshipSignal(signal="unclear"), is_staffing_firm=False,
                     seniority_ok=True, one_line_reason="ok")


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, fast_scores, strong_score=90):
        super().__init__({"fast": "claude-haiku-5-5", "strong": "claude-sonnet-5-5"})
        self.fast_scores, self.strong_score, self.calls = list(fast_scores), strong_score, []

    def generate_json(self, prompt, schema, *, tier="fast", system=None):
        self.calls.append(tier)
        s = self.fast_scores.pop(0) if tier == "fast" else self.strong_score
        return LLMResponse(scored(s), self.models[tier], 5000, 1000, 0.1, 1)

    def _call(self, *a):  # pragma: no cover — generate_json is overridden
        raise NotImplementedError

    def _retry_hint(self, exc):  # pragma: no cover
        return None


class FakeStore:
    def __init__(self):
        self.rows = []

    def save_score(self, row):
        self.rows.append(row)


class NoUSCIS:
    def lookup(self, company, aliases=None, e_verify=None):
        return {"status": "no_data", "e_verify": e_verify or "unknown"}


# --- config / model consistency -----------------------------------------------

def test_role_families_match_config(cfg):
    assert set(typing.get_args(RoleFamily)) == {f["name"] for f in cfg["filters"]["role_families"]} | {"other"}


def test_headline_labels_cover_every_family(cfg):
    labels = cfg["tailoring"]["headline_role_labels"]
    assert set(labels) == {f["name"] for f in cfg["filters"]["role_families"]}
    forbidden = [w.lower() for w in cfg["tailoring"]["headline_forbidden_words"]]
    assert not any(w in label.lower() for ls in labels.values() for label in ls for w in forbidden)


def test_scored_job_rejects_out_of_range():
    with pytest.raises(ValueError):
        ScoredJob(**{**scored(50).model_dump(), "fit_score": 140})


# --- selection -------------------------------------------------------------------

def test_location_rank(cfg):
    tiers = cfg["preferences"]["location_priority"]
    assert [location_rank(l, tiers) for l in ["Atlanta, GA", "Menlo Park, CA", "Austin, TX", "New York, NY"]] == [0, 1, 2, 3]


def test_select_orders_by_family_then_location_and_caps_company(cfg):
    rows = [job(1, "A", "analyst_backup", "Atlanta, GA"),
            job(2, "B", "ai_engineer", "New York, NY"),
            job(3, "C", "ai_engineer", "Atlanta, GA"),
            *[job(10 + k, "D", "ai_engineer", "Austin, TX") for k in range(5)]]
    picked = select_jobs(rows, cfg, limit=10)
    assert [r["id"] for r in picked[:2]] == ["j3", "j10"]            # ai_engineer: Atlanta before Texas
    assert sum(r["company"] == "D" for r in picked) == cfg["scoring"]["max_per_company"]
    assert picked[-1]["id"] == "j1"                                   # backup family last


# --- scoring flow ----------------------------------------------------------------

def test_escalates_only_at_or_above_threshold(cfg):
    cfg["scoring"]["escalate_threshold"] = 60           # pinned: the live value changes with evals
    llm, store = FakeLLM([59, 60, 85]), FakeStore()
    jobs = [job(1), job(2), job(3)]
    est = estimate(jobs, "system", llm, cfg)
    res = score_jobs(jobs, "system", llm, cfg, store, NoUSCIS(), est)
    assert llm.calls == ["fast", "fast", "strong", "fast", "strong"]
    assert [r["escalated"] for r in store.rows] == [0, 1, 1]
    assert [r["fit_score"] for r in store.rows] == [59, 90, 90] and store.rows[1]["fast_score"] == 60
    expected = cost_usd("claude-haiku-5-5", 5000, 1000, cfg["llm"]["pricing"]) * 3 \
        + cost_usd("claude-sonnet-5-5", 5000, 1000, cfg["llm"]["pricing"]) * 2
    assert res.spent == pytest.approx(expected)


def test_budget_cap_stops_run(cfg):
    tight = {**cfg, "scoring": {**cfg["scoring"], "max_run_cost_usd": 0.02}}
    llm, store = FakeLLM([90] * 10), FakeStore()
    jobs = [job(i) for i in range(10)]
    res = score_jobs(jobs, "system", llm, tight, store, NoUSCIS(), estimate(jobs, "system", llm, tight))
    assert res.stopped_early and res.spent <= 0.02 + 1e-9
    assert len(store.rows) < 10


def test_estimate_bounds(cfg):
    llm = FakeLLM([])
    est = estimate([job(1), job(2)], "x" * 7000, llm, cfg)
    assert est.fast_cost < est.expected < est.worst


# --- resume loader -----------------------------------------------------------------

def test_loader_refuses_unresolved(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"conflicts": [{"id": "C1", "status": "unresolved"}], "missing": []}))
    with pytest.raises(ResumeError, match="C1"):
        load_master(p)


def test_real_profile_has_no_contact_or_superseded_claims():
    try:
        text = profile_text(load_master())
    except ResumeError as e:  # master resume is gitignored; skip where it doesn't exist (e.g. CI)
        pytest.skip(str(e))
    for banned in ("@", "221-8780", "5+ years", "patient records"):
        assert banned not in text
    assert "4+ years" in text


# --- sponsorship ------------------------------------------------------------------

@pytest.mark.parametrize("raw,norm", [("Anthropic, PBC", "anthropic"), ("Scale AI, Inc.", "scale ai"),
                                      ("Databricks Inc", "databricks"), ("AT&T Services, L.L.C.", "at and t services")])
def test_normalize_employer(raw, norm):
    assert normalize_employer(raw) == norm


def test_uscis_utf16_tab_export_and_exact_match(tmp_path):
    content = ("Fiscal Year\tEmployer (Petitioner) Name\tNew Employment Approval\tContinuation Approval\n"
               "2025\tANTHROPIC, PBC\t120\t30\n"
               "2025\tHEXAWARE TECHNOLOGIES INC\t500\t1,200\n")
    (tmp_path / "fy2025.csv").write_bytes(content.encode("utf-16"))
    db = SponsorshipDB(tmp_path)
    assert db.lookup("Anthropic")["approvals"] == 150
    assert db.lookup("Hex")["status"] == "not_found"          # no prefix matching
    assert SponsorshipDB(tmp_path / "empty").lookup("Anthropic")["status"] == "no_data"


def test_e_verify_defaults_unknown(tmp_path):
    assert SponsorshipDB(tmp_path).lookup("Anthropic")["e_verify"] == "unknown"
    assert SponsorshipDB(tmp_path).lookup("Anthropic", e_verify="yes")["e_verify"] == "yes"


# --- tiers, mix, ranking -------------------------------------------------------------

def test_every_company_has_tier_and_e_verify(cfg):
    for c in cfg["companies"]:
        assert c["tier"] in ("startup", "growth", "big_tech") and c["e_verify"] in ("yes", "no", "unknown"), c
    assert {c["name"] for c in cfg["companies"] if c["tier"] == "big_tech"} >= \
        {"OpenAI", "Anthropic", "Scale AI", "CoreWeave", "Snowflake", "Datadog"}


def test_us_remote_ties_with_atlanta(cfg):
    tiers = cfg["preferences"]["location_priority"]
    assert location_rank("Remote - US", tiers) == location_rank("Atlanta, GA", tiers) == 0
    assert location_rank("United States", tiers, remote=True) == 0      # ATS remote flag + US
    assert location_rank("San Francisco, CA", tiers, remote=True) == 1  # remote flag but tied to a CA city


def test_result_sort_fit_then_tier_then_location(cfg):
    from src.scorer import result_sort_key
    rows = [{"fit_score": 70, "company": "OpenAI", "location": "Atlanta, GA", "remote": 0},       # big_tech
            {"fit_score": 70, "company": "Rogo", "location": "New York, NY", "remote": 0},        # startup, NY
            {"fit_score": 70, "company": "LangChain", "location": "Remote - US", "remote": 1},   # startup, US-remote
            {"fit_score": 80, "company": "Datadog", "location": "Austin, TX", "remote": 0}]
    order = [r["company"] for r in sorted(rows, key=lambda r: result_sort_key(r, cfg))]
    assert order == ["Datadog", "LangChain", "Rogo", "OpenAI"]


def test_result_sort_sponsorship_after_fit_before_tier(cfg):
    from src.scorer import result_sort_key
    rows = [{"fit_score": 70, "company": "LangChain", "location": "Remote - US", "remote": 1, "sponsorship_signal": "no"},
            {"fit_score": 70, "company": "OpenAI", "location": "Atlanta, GA", "remote": 0, "sponsorship_signal": "yes"},
            {"fit_score": 70, "company": "Rogo", "location": "New York, NY", "remote": 0, "sponsorship_signal": None},
            {"fit_score": 71, "company": "Hex", "location": "Austin, TX", "remote": 0, "sponsorship_signal": "no"}]
    order = [r["company"] for r in sorted(rows, key=lambda r: result_sort_key(r, cfg))]
    # fit first; then yes > unclear (missing counts as unclear) > no, even against a better tier
    assert order == ["Hex", "OpenAI", "Rogo", "LangChain"]


def _t(name, fit, signal):
    return {"id": name, "company": "Rogo", "location": "New York, NY", "remote": 0, "fit_score": fit,
            "sponsorship_signal": signal}


@pytest.mark.parametrize("rows,expected", [
    ([_t("a", 70, "unclear"), _t("b", 67, "yes")], ["b", "a"]),          # within 3 points: yes first
    ([_t("a", 70, "unclear"), _t("b", 66, "yes")], ["a", "b"]),          # 4 points apart: fit wins
    ([_t("a", 70, "no"), _t("b", 69, "unclear")], ["a", "b"]),           # only "yes" gets the margin
    ([_t("a", 75, "unclear"), _t("b", 73, "yes"), _t("c", 72, "yes")], ["b", "c", "a"]),
    ([_t("a", 80, "unclear"), _t("b", 78, "unclear"), _t("c", 76, "yes")], ["a", "c", "b"]),
])
def test_tailor_order_prefers_sponsorship_within_3_points(cfg, rows, expected):
    from src.scorer import tailor_order
    assert [r["id"] for r in tailor_order(rows, cfg)] == expected


NOW = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)


def _f(name, fit, days_old, signal="unclear"):
    posted = None if days_old is None else (NOW - timedelta(days=days_old)).isoformat()
    return {**_t(name, fit, signal), "posted_at": posted}


def test_freshness_is_0_to_3_days_and_no_date_is_not_fresh(cfg):
    from src.scorer import is_fresh
    assert is_fresh(_f("a", 60, 0), cfg, NOW) and is_fresh(_f("a", 60, 3), cfg, NOW)
    assert not is_fresh(_f("a", 60, 4), cfg, NOW) and not is_fresh(_f("a", 60, None), cfg, NOW)


@pytest.mark.parametrize("rows,expected", [
    ([_f("a", 75, 10), _f("b", 70, 1)], ["b", "a"]),                 # fresh within 5 points goes first
    ([_f("a", 76, 10), _f("b", 70, 1)], ["a", "b"]),                 # 6 points apart: fit wins
    ([_f("a", 72, None), _f("b", 70, 2)], ["b", "a"]),               # no date counts as not fresh
    ([_f("a", 70, 10, "yes"), _f("b", 70, 10, "unclear"), _f("c", 70, 1, "no")], ["c", "a", "b"]),  # fit, fresh, sponsor
])
def test_rank_results_freshness_preference(cfg, rows, expected):
    from src.scorer import rank_results
    assert [r["id"] for r in rank_results(rows, cfg, NOW)] == expected


def test_ranking_ignores_sponsorship_margin_but_tailoring_uses_it(cfg):
    from src.scorer import rank_results, tailor_order
    rows = [_f("a", 70, 10, "unclear"), _f("b", 68, 10, "yes")]
    assert [r["id"] for r in rank_results(rows, cfg, NOW)] == ["a", "b"]
    assert [r["id"] for r in tailor_order(rows, cfg, NOW)] == ["b", "a"]


@pytest.mark.parametrize("rows,expected", [
    # fresh beats older-but-sponsoring: sponsorship can't move an old job ahead of a fresh one
    ([_f("a", 70, 10), _f("b", 67, 1), _f("c", 66, 10, "yes")], ["b", "a", "c"]),
    # both fresh: sponsorship margin applies between them
    ([_f("a", 70, 1), _f("b", 68, 2, "yes")], ["b", "a"]),
    # old yes within 3 of an old job, but a fresh job is ahead of both
    ([_f("a", 80, 1), _f("b", 75, 10), _f("c", 73, 10, "yes")], ["a", "c", "b"]),
])
def test_tailor_order_freshness_outranks_sponsorship(cfg, rows, expected):
    from src.scorer import tailor_order
    assert [r["id"] for r in tailor_order(rows, cfg, NOW)] == expected


def test_select_mix_buckets_tiers_and_shortfall(cfg):
    from src.scorer import select_mix
    rows = [job(1, "Rogo", "ai_solutions"), job(2, "Sierra", "forward_deployed"), job(3, "OpenAI", "ai_solutions"),
            job(4, "LangChain", "ai_eval"), job(5, "Hex", "analyst_backup"), job(6, "Modal", "ai_engineer")]
    picked, notes = select_mix(rows, cfg)
    ids = {r["id"] for r in picked}
    assert "j3" not in ids                                   # big_tech excluded from the mix
    assert ids == {"j1", "j2", "j4", "j5", "j6"}
    assert len(notes) == 4 and all("only" in n for n in notes)   # every bucket short of 5


# --- skills gap ---------------------------------------------------------------------------

def test_skills_gap_ignores_urls_company_name_and_departments():
    from src.skills_gap import build_report
    rows = [
        {"company": "Anthropic", "gaps": "[]", "description":
            "Requirements:\n- Kubernetes and TypeScript\n- Partner with Legal and Security\n"
            "See https://writer.com/legal/notice. Anthropic recruiters only contact you from @anthropic.com"},
        {"company": "Acme", "gaps": '["dbt"]', "description": "Qualifications: Python, Kubernetes, law firms experience"},
    ]
    _, table = build_report(rows, resume_text="Python, SQL", top_n=50)
    by = {r["skill"]: r for r in table}
    assert by["Kubernetes"]["jobs"] == 2 and by["Kubernetes"]["in_resume"] is False
    assert by["Python"]["in_resume"] is True
    assert by["Legal domain"]["jobs"] == 1                 # only "law firms", not the Legal department or the URL
    assert "Anthropic / Claude API" not in by              # the company's own name doesn't count


# --- JSearch ----------------------------------------------------------------------------

def test_jsearch_parse_keeps_only_direct_employer_links(cfg):
    from src.sources import jsearch
    payload = {"data": {"cursor": None, "jobs": [
        {"job_id": "a1", "employer_name": "Acme AI", "job_title": "AI Engineer", "job_city": "Atlanta",
         "job_state": "GA", "job_country": "US", "job_is_remote": False, "job_apply_link": "https://acme.ai/jobs/1",
         "job_publisher": "Acme AI", "employer_website": "https://www.acme.ai",
         "job_description": "d", "job_posted_at_datetime_utc": "2026-10-01T00:00:00Z", "job_employment_type": "FULLTIME"},
        {"job_id": "a2", "employer_name": "Best Staffing LLC", "job_title": "AI Engineer", "job_description": "d"},
        {"job_id": "a3", "employer_name": "Boston Consulting Group", "job_title": "AI Consultant",
         "job_is_remote": True, "job_country": "US", "job_description": "d",
         "job_apply_link": "https://www.linkedin.com/jobs/view/3", "job_publisher": "LinkedIn"},
        {"job_id": "a4", "employer_name": "Enduring Ventures", "job_title": "AI FDE", "job_description": "d",
         "job_publisher": "Talent.com", "job_apply_link": "https://www.talent.com/view?id=1"},
        {"job_id": "a5", "employer_name": "Jobleads-US", "job_title": "AI FDE", "job_description": "d"},
        {"job_id": "a6", "employer_name": "Beta Labs", "job_title": "AI Engineer", "job_is_remote": True,
         "job_country": "US", "job_description": "d", "job_apply_link": "https://www.indeed.com/viewjob?jk=6",
         "apply_options": [{"apply_link": "https://betalabs.wd1.myworkdayjobs.com/External/job/6"}]},
    ]}}
    jobs, dropped = jsearch.parse(payload, cfg["jsearch"])
    assert dropped == {"staffing/excluded employer": 2, "repost/aggregator site": 1, "no direct employer link": 1}
    assert [j.company for j in jobs] == ["Acme AI", "Beta Labs"]
    assert jobs[0].location == "Atlanta, GA, US" and jobs[0].id == "jsearch:a1"
    assert jobs[0].apply_via == "employer careers site (acme.ai)" and jobs[0].url == "https://acme.ai/jobs/1"
    assert jobs[1].url == "https://betalabs.wd1.myworkdayjobs.com/External/job/6" and jobs[1].remote
    assert all(j.apply_direct for j in jobs)


@pytest.mark.parametrize("raw, link, how", [
    # Greenhouse/Lever/Ashby board under the employer's name, even when listed after a job board
    ({"employer_name": "Acme", "job_apply_link": "https://www.linkedin.com/jobs/view/1",
      "apply_options": [{"publisher": "LinkedIn", "apply_link": "https://www.linkedin.com/jobs/view/1"},
                        {"publisher": "Acme", "apply_link": "https://job-boards.greenhouse.io/acme/jobs/9"}]},
     "https://job-boards.greenhouse.io/acme/jobs/9", "employer board (greenhouse/lever/ashby)"),
    # another ATS under the employer's name
    ({"employer_name": "Extreme Networks, Inc.", "job_apply_link": "https://www.indeed.com/viewjob?jk=1",
      "apply_options": [{"apply_link": "https://extremenetworks.wd5.myworkdayjobs.com/External/job/9"}]},
     "https://extremenetworks.wd5.myworkdayjobs.com/External/job/9",
     "employer board (extremenetworks.wd5.myworkdayjobs.com)"),
    ({"employer_name": "Acme", "apply_options": [{"apply_link": "https://jobs.smartrecruiters.com/Acme/123-ai"}]},
     "https://jobs.smartrecruiters.com/Acme/123-ai", "employer board (jobs.smartrecruiters.com)"),
    # an ATS board under SOMEONE ELSE's name (e.g. a staffing firm reposting) does not count
    ({"employer_name": "Acme", "apply_options": [{"apply_link": "https://job-boards.greenhouse.io/talentco/jobs/9"}]},
     None, "no direct employer link"),
    # the employer's own careers site
    ({"employer_name": "Acme", "employer_website": "https://acme.ai",
      "apply_options": [{"apply_link": "https://careers.acme.ai/9"}]},
     "https://careers.acme.ai/9", "employer careers site (careers.acme.ai)"),
    # is_direct alone no longer counts
    ({"employer_name": "Acme", "job_apply_link": "https://www.glassdoor.com/job/1",
      "apply_options": [{"apply_link": "https://apply.someportal.com/9", "is_direct": True}]},
     None, "no direct employer link"),
    ({"employer_name": "Acme", "job_apply_link": "https://www.dice.com/job-detail/1"}, None, "no direct employer link"),
])
def test_jsearch_direct_employer_link(cfg, raw, link, how):
    from src.sources import jsearch
    assert jsearch.direct_link(raw, cfg["jsearch"]) == (link, how)


def test_jsearch_request_excludes_aggregator_publishers(cfg, tmp_path):
    js = cfg["jsearch"]
    (_, _, _, st), seen = _js_run(js, Store(tmp_path / "t.db"))
    pubs = seen[0]["exclude_job_publishers"].split(",")
    assert {"Talent.com", "Dice", "BeBee", "JobLeads"} <= set(pubs)


def test_stored_jsearch_urls_are_classified(cfg):
    from src.sources.jsearch import classify_stored_url
    js = cfg["jsearch"]
    assert classify_stored_url("https://www.talent.com/view?id=1", "Enduring Ventures", js)[0] is False
    assert classify_stored_url("https://bebee.com/us/jobs/x", "Full Stack LLC", js)[0] is False
    assert classify_stored_url("https://jobs.ashbyhq.com/acme/1", "Acme", js) == (True, "employer board (jobs.ashbyhq.com)")
    assert classify_stored_url("https://jobs.ashbyhq.com/other/1", "Acme", js)[0] is False
    assert classify_stored_url("https://careers.acme.ai/1", "Acme", js)[0] is True
    assert classify_stored_url("https://careers.someoneelse.com/1", "Acme", js)[0] is False


def test_jsearch_config_fits_the_free_plan(cfg):
    from src.sources import jsearch
    js = cfg["jsearch"]
    pairs = jsearch.requests_planned(js)
    assert len(pairs) == 6 <= js["max_requests_per_day"] == 7          # 3 broad queries x (US + remote)
    assert {loc["name"] for _, loc in pairs} == {"United States", "US remote"}
    assert js["date_posted"] == "3days"
    assert js["monthly_soft_limit"] == 160 < js["monthly_hard_limit"] == 200


def _js_run(js, store, pages_full=(), status=200, headers=None, now=None):
    """Run jsearch.fetch against a fake API. pages_full: (query_prefix, page) pairs that return a full page."""
    import asyncio
    import httpx
    from src.sources import jsearch
    seen = []

    def handler(request):
        p = dict(request.url.params)
        seen.append(p)
        full = any(p["query"].startswith(q) and int(p["page"]) == pg for q, pg in pages_full)
        data = [{"job_id": f"{len(seen)}-{i}", "employer_name": "Acme AI", "job_title": "AI Engineer",
                 "job_apply_link": f"https://job-boards.greenhouse.io/acmeai/jobs/{len(seen)}{i}",
                 "job_description": "d"} for i in range(10 if full else 3)]
        body = {"data": {"jobs": data, "cursor": f"c{len(seen)}" if full else None}} if status == 200 else {
            "message": "Endpoint '/search' does not exist"}
        return httpx.Response(status, json=body, headers=headers or {})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await jsearch.fetch(c, js, "k", store, now=now)
    return asyncio.run(go()), seen


def test_jsearch_sends_one_page_per_request_and_records_each(cfg, tmp_path):
    js = cfg["jsearch"]
    store = Store(tmp_path / "t.db")
    (jobs, _, errors, st), seen = _js_run(js, store)
    assert st["sent"] == 6 and not errors and len(jobs) == 18          # no full pages -> no extra pages
    assert [p["query"] for p in seen[:2]] == [
        "AI engineer OR solutions engineer OR forward deployed engineer in United States",
        "AI engineer OR solutions engineer OR forward deployed engineer"]
    assert seen[1].get("work_from_home") == "true" and "work_from_home" not in seen[0]
    assert all(p["num_pages"] == "1" and p["date_posted"] == "3days" for p in seen)
    rows = store.conn.execute("SELECT status, results FROM jsearch_requests").fetchall()
    assert len(rows) == 6 and all(r["status"] == 200 and r["results"] == 3 for r in rows)


def test_jsearch_extra_page_only_within_daily_budget(cfg, tmp_path):
    js = cfg["jsearch"]
    store = Store(tmp_path / "t.db")
    full = [("AI engineer", 1), ("data analyst", 1), ("AI engineer", 2)]   # 4 pairs would want page 2
    (_, _, _, st), seen = _js_run(js, store, pages_full=full)
    assert st["sent"] == 7 and st["stopped"].startswith("budget reached: daily budget")
    assert [int(p["page"]) for p in seen] == [1] * 6 + [2]            # base pages first, then one extra
    assert "cursor" not in seen[0] and seen[6]["cursor"] == "c1"      # page 2 continues page 1's cursor
    # Same day again: nothing left.
    (_, _, _, st2), seen2 = _js_run(js, store)
    assert st2["sent"] == 0 and seen2 == []


def test_jsearch_monthly_soft_limit_and_provider_count(cfg, tmp_path):
    from datetime import datetime, timezone
    from src.sources import jsearch
    js = cfg["jsearch"]
    store = Store(tmp_path / "t.db")
    now = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
    for d in range(1, 20):            # 158 requests earlier this month, none today
        for _ in range(8 if d < 19 else 14):
            store.jsearch_sent("q", "US", 1, datetime(2026, 10, d, 9, tzinfo=timezone.utc).isoformat())
    assert store.jsearch_used("2026-10-01T00:00:00+00:00") == 158
    b = jsearch.budget(js, store, now)
    assert b.allowed == 2 and "monthly limit" in b.reason
    (_, _, _, st), _ = _js_run(js, store, now=now)
    assert st["sent"] == 2 and jsearch.budget(js, store, now).allowed == 0
    # Next month starts fresh.
    assert jsearch.budget(js, store, datetime(2026, 11, 2, tzinfo=timezone.utc)).allowed == 7

    # RapidAPI's own count wins when it is lower (e.g. requests made in the playground).
    store2 = Store(tmp_path / "t2.db")
    rid = store2.jsearch_sent("q", "US", 1, now.isoformat())
    store2.jsearch_done(rid, 200, 3, 43, 200)                  # provider: 157 used
    b = jsearch.budget(js, store2, now)
    assert b.allowed == 3 and "RapidAPI reports 43" in b.reason


@pytest.mark.parametrize("status", [404, 429, 403])
def test_jsearch_stops_after_first_client_error(cfg, tmp_path, status):
    store = Store(tmp_path / "t.db")
    (jobs, _, errors, st), seen = _js_run(cfg["jsearch"], store, status=status)
    assert len(seen) == 1 and st["sent"] == 1 and f"HTTP {status}" in errors[0] and jobs == []
    assert store.conn.execute("SELECT status FROM jsearch_requests").fetchone()[0] == status


def test_jsearch_unexpected_shape_is_an_error_not_zero_jobs():
    from src.sources import jsearch
    with pytest.raises(jsearch.JSearchFormatError):
        jsearch.jobs_of({"data": []})          # the retired /search shape


def test_restricted_remote_ranks_after_preferred_and_unrestricted_remote(cfg):
    from src.scorer import job_location_rank
    rank = lambda loc, desc="": job_location_rank({"location": loc, "remote": 1, "description": desc}, cfg)
    assert rank("Remote - US") == 0 and rank("Remote - TX") == 2
    assert rank("Remote - Colorado") == 3
    assert rank("Remote - US", "You must reside in Colorado or Utah.") == 3     # looks US-remote, but restricted
