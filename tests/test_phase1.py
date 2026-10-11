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
def jf():
    return JobFilter(load_config()["filters"])


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
    ("Backend Engineer, LLM Platform", True),
    ("Software Engineer, AI Infrastructure", True),
    ("Senior Applied AI Engineer", True),     # Senior titles stay in
])
def test_specialty_excluded_unless_ai(jf, title, keep):
    assert jf.check(make_job(title=title)).keep is keep


@pytest.mark.parametrize("desc", [
    "We are unable to sponsor visas for this role.",
    "Candidates must be authorized to work without sponsorship.",
    "Must be a U.S. citizen.",
    "Active TS/SCI clearance required.",
    "This is a C2C engagement.",
    "W2 through vendor only.",
])
def test_exclusion_phrases(jf, desc):
    r = jf.check(make_job(description=desc))
    assert not r.keep and r.reason.startswith("phrase")


def test_sponsorship_positive_is_kept(jf):
    assert jf.check(make_job(description="We sponsor visas and support H-1B transfers.")).keep


def test_contract_employment_type(jf):
    assert not jf.check(make_job(employment_type="Contract")).keep


@pytest.mark.parametrize("loc,remote,ok", [
    ("San Francisco, CA", False, True),
    ("Remote - US", True, True),
    ("Remote", True, True),
    ("London, UK", False, False),
    ("London; New York", False, True),       # any US location is enough
    ("Remote in Canada", True, False),       # "in" must not read as Indiana
    ("Toronto, ON", False, False),
    ("", True, True),
    ("", False, False),
])
def test_us_location(loc, remote, ok):
    assert JobFilter.is_us_location(loc, remote) is ok


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
