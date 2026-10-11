import typing

import pytest
import yaml

from main import load_config
from src.llm import LLMProvider, LLMResponse, cost_usd
from src.models import RoleFamily, ScoredJob, SponsorshipSignal
from src.resume import ResumeError, load_master, profile_text
from src.scorer import estimate, location_rank, score_jobs, select_jobs
from src.sponsorship import SponsorshipDB, normalize_employer


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
    def lookup(self, company, aliases=None):
        return {"status": "no_data"}


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
