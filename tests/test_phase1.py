from datetime import timezone

import pytest

from main import load_config
from src.filters import JobFilter
from src.models import Job
from src.sources import ashby, greenhouse, lever
from src.sources.common import html_to_text, parse_datetime
from src.store import Store, normalize_title


def make_job(**kw) -> Job:
    base = dict(id="greenhouse:acme:1", company="Acme", title="AI Engineer", location="San Francisco, CA",
                remote=False, url="https://x", description="Build LLM apps.", source="greenhouse")
    return Job(**{**base, **kw})


@pytest.fixture(scope="module")
def cfg_filters():
    return load_config()["filters"]


@pytest.fixture(scope="module")
def jf(cfg_filters):
    return JobFilter(cfg_filters)


# --- filters ---------------------------------------------------------------

@pytest.mark.parametrize("title,family", [
    ("Applied AI Engineer", "ai_engineer"),
    ("GenAI Engineer", "ai_engineer"),
    ("Solutions Architect, AI", "ai_solutions"),
    ("Software Engineer, LLM Platform", "ai_engineer"),
    ("Solutions Engineer", "ai_solutions"),
    ("Forward Deployed Engineer", "forward_deployed"),
    ("Deployment Strategist", "forward_deployed"),
    ("AI Strategist", "ai_consultant"),
    ("Model Evaluation Specialist", "ai_eval"),
    ("Analytics Engineer", "analyst_backup"),
    ("Account Executive", None),
    ("Recruiter", None),
])
def test_role_family(jf, title, family):
    assert jf.role_family(title) == family


@pytest.mark.parametrize("title", ["Staff AI Engineer", "Principal Solutions Architect",
                                   "Engineering Manager, LLM", "AI Engineer Intern", "Tech Lead, Applied AI"])
def test_too_senior_or_not_ic(jf, title):
    assert not jf.check(make_job(title=title)).keep


@pytest.mark.parametrize("title,keep", [
    ("Backend Solutions Engineer", False),
    ("Solutions Architect, Infrastructure", False),
    ("Frontend Forward Deployed Engineer", False),
    ("iOS Solutions Engineer", False),
    ("Solutions Engineer, Embedded", False),
    ("Backend Engineer, LLM", True),
    ("Backend Engineer, LLM Platform", False),          # "platform" is excluded even with AI terms
    ("Software Engineer, AI Infrastructure", False),    # so is "infrastructure"
    ("Senior Applied AI Engineer", True),     # Senior titles stay in
])
def test_specialty_excluded_unless_ai(jf, title, keep):
    assert jf.check(make_job(title=title)).keep is keep


@pytest.mark.parametrize("desc", [
    "Must be a U.S. citizen.",
    "This role is open to U.S. citizens only.",
    "Only open to US citizens.",
    "U.S. citizenship is required.",
    "Active TS/SCI clearance required.",
    "Must be a permanent resident or citizen; this is required.",
    "This is a C2C engagement.",
    "W2 through vendor only.",
])
def test_exclusion_phrases(jf, desc):
    r = jf.check(make_job(description=desc))
    assert not r.keep and r.reason.startswith("phrase")


@pytest.mark.parametrize("desc", [
    # Since 2026-10-11 sponsorship statements no longer exclude a job.
    "We are unable to sponsor visas for this role.",
    "Candidates must be authorized to work without sponsorship.",
    "Must be authorized to work in the U.S. without the need for current or future employer sponsorship.",
    "No visa sponsorship is available for this position.",
    "We sponsor visas and support H-1B transfers.",
])
def test_sponsorship_statements_are_kept(jf, desc):
    assert jf.check(make_job(description=desc)).keep


def test_recheck_restores_jobs_that_now_pass(tmp_path, cfg_filters):
    from src.pipeline import recheck_existing
    from src.store import Store

    store = Store(tmp_path / "jobs.db")
    ok = make_job(description="We are unable to sponsor visas for this role.")
    bad = make_job(title="AI Engineer II", description="Must be a U.S. citizen.").model_copy(update={"id": "gh:x:2"})
    store.save([ok, bad])
    store.set_status(ok.id, "excluded"); store.set_status(bad.id, "excluded")
    cfg = {"filters": cfg_filters, "companies": [{"name": ok.company, "hq": "US"}]}
    assert recheck_existing(cfg, store)["restored (now passes filters)"] == 1
    assert [r["id"] for r in store.jobs_by_status("new")] == [ok.id]
    store.close()


def test_contract_employment_type(jf):
    assert not jf.check(make_job(employment_type="Contract")).keep


@pytest.mark.parametrize("loc,remote,desc,ok", [
    ("San Francisco, CA", False, "", True),
    ("Remote - US", True, "", True),
    ("Remote (United States)", True, "", True),
    ("US-based remote", True, "", True),
    ("Remote - Atlanta, GA", True, "", True),
    ("London, UK", False, "", False),
    ("London; New York", False, "", True),             # any US location is enough
    ("Remote in Canada", True, "", False),             # "in" must not read as Indiana
    ("Remote - EMEA", True, "", False),
    ("Remote - LATAM", True, "", False),
    ("Remote - Colorado", True, "", False),            # remote restricted to a non-GA/CA/TX state
    ("Remote (CA, NY)", True, "", True),               # CA allowed
    ("Remote - TX", True, "", True),
    ("Remote", True, "", False),                       # bare remote, country unknown
    ("Remote", True, "This role is remote within the United States.", True),
    ("Remote - US", True, "Candidates must reside in Colorado or Utah.", False),
    ("Remote - US", True, "You must live in Georgia or Florida.", True),
    ("US-CA-Menlo Park", False, "", True),
    ("Toronto, ON", False, "", False),
    ("", True, "", False),
    ("", False, "", False),
])
def test_location_rules(jf, loc, remote, desc, ok):
    assert (jf.location_reason(make_job(location=loc, remote=remote, description=desc)) is None) is ok


@pytest.mark.parametrize("title,family", [
    ("AI Evaluation Engineer", "ai_eval"), ("Model Evaluation Specialist", "ai_eval"),
    ("LLM Evaluation Analyst", "ai_eval"), ("Evals Engineer", "ai_eval"), ("AI Quality Analyst", "ai_eval"),
    ("Senior Software Engineer, Evaluation Engine", None),     # plain "evaluation" no longer counts
    ("Senior Research Scientist, Model Evaluation", None),     # research scientist excluded from ai_eval
    ("Backend Engineer, Evals", None),                         # backend excluded from ai_eval
])
def test_tightened_ai_eval(jf, title, family):
    assert jf.role_family(title) == family


def test_bare_remote_kept_for_us_hq_company(cfg_filters):
    us = JobFilter(cfg_filters, us_hq_companies={"Acme"})
    assert us.location_reason(make_job(company="Acme", location="Remote", remote=True)) is None
    assert us.location_reason(make_job(company="Other", location="Remote", remote=True)) is not None
    assert us.location_reason(make_job(company="Acme", location="Remote - EMEA", remote=True)) is not None


def test_max_posting_age(cfg_filters):
    from datetime import datetime, timezone
    now = datetime(2026, 10, 11, tzinfo=timezone.utc)
    f = JobFilter(cfg_filters, now=now)
    assert f.check(make_job(posted_at=datetime(2026, 8, 20, tzinfo=timezone.utc))).keep          # 52 days
    old = f.check(make_job(posted_at=datetime(2026, 8, 1, tzinfo=timezone.utc)))                   # 71 days
    assert not old.keep and "60 days" in old.reason
    assert f.check(make_job(posted_at=None)).keep                                                   # no date: kept


@pytest.mark.parametrize("title", ["Frontend AI Engineer", "Full-Stack Engineer, AI", "AI Platform Engineer",
                                   "AI Infrastructure Engineer", "AI Security Engineer", "SRE, LLM", "DevOps AI Engineer",
                                   "Systems Engineer, Enterprise AI", "Staff Solutions Engineer", "Director, AI Strategy",
                                   "Network Operations Engineer, AI Networking", "Research Engineer, AI for Chip Design",
                                   "AI Hardware Engineer"])
def test_excluded_specialties_even_with_ai(jf, title):
    assert not jf.check(make_job(title=title)).keep


def test_senior_kept(jf):
    assert jf.check(make_job(title="Senior Solutions Engineer")).keep


@pytest.mark.parametrize("desc,excluded", [
    ("This position requires access to export controlled information.", True),
    ("Subject to ITAR regulations.", True),
    ("Subject to EAR.", True),
    ("Applicant must be a U.S. person.", True),
    ("Must be a permanent resident.", True),
    ("Green card holders only.", True),
    ("We offer green card sponsorship after one year.", False),
    ("We support permanent residency applications.", False),
    ("Each year we hear from customers.", False),          # "ear"/"year" must not trigger EAR
])
def test_export_and_residency_phrases(jf, desc, excluded):
    assert (not jf.check(make_job(description=desc)).keep) is excluded


# --- store / dedupe --------------------------------------------------------

def test_normalize_title():
    assert normalize_title("Sr. Applied AI Engineer (NYC)") == "senior applied ai engineer"


def test_store_dedupes_by_id_and_by_company_title(tmp_path):
    store = Store(tmp_path / "t.db")
    a = make_job(id="greenhouse:acme:1")
    same_role_other_city = make_job(id="greenhouse:acme:2", title="AI Engineer (Seattle)", location="Seattle, WA")
    different_role = make_job(id="greenhouse:acme:3", title="Solutions Engineer")

    assert len(store.save([a, same_role_other_city, different_role])) == 2
    assert store.save([a]) == []          # second run: nothing new
    assert store.count() == 2


# --- source parsing --------------------------------------------------------

def test_html_to_text_unescapes_greenhouse():
    assert html_to_text("&lt;p&gt;Hello &amp;amp; welcome&lt;/p&gt;&lt;li&gt;Python&lt;/li&gt;") == "Hello & welcome\nPython"


def test_parse_datetime_formats():
    assert parse_datetime("2026-10-01T12:00:00Z").tzinfo == timezone.utc
    assert parse_datetime(1_759_320_000_000).year == 2025
    assert parse_datetime("garbage") is None


def test_greenhouse_parse():
    payload = {"jobs": [{"id": 9, "title": " AI Engineer ", "location": {"name": "Remote - US"},
                         "absolute_url": "u", "content": "&lt;p&gt;Hi&lt;/p&gt;", "first_published": "2026-10-01T00:00:00Z"}]}
    j = greenhouse.parse("Acme", "acme", payload)[0]
    assert (j.id, j.title, j.remote, j.description) == ("greenhouse:acme:9", "AI Engineer", True, "Hi")


def test_lever_parse():
    payload = [{"id": "x", "text": "FDE", "hostedUrl": "u", "createdAt": 1_759_320_000_000, "workplaceType": "remote",
                "categories": {"allLocations": ["NYC", "SF"], "commitment": "Full-time"},
                "descriptionPlain": "Intro", "lists": [{"text": "Reqs", "content": "<li>Python</li>"}]}]
    j = lever.parse("Acme", "acme", payload)[0]
    assert j.location == "NYC; SF" and j.remote and "Python" in j.description and j.employment_type == "Full-time"


def test_ashby_parse_skips_unlisted():
    payload = {"jobs": [
        {"id": "a", "title": "AI Engineer", "location": "SF", "secondaryLocations": [{"location": "NYC"}],
         "isRemote": False, "jobUrl": "u", "descriptionPlain": "d", "publishedAt": "2026-10-01T00:00:00Z",
         "employmentType": "FullTime", "isListed": True},
        {"id": "b", "title": "Hidden", "isListed": False},
    ]}
    jobs = ashby.parse("Acme", "acme", payload)
    assert len(jobs) == 1 and jobs[0].location == "SF; NYC"
