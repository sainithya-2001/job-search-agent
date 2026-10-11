"""JSearch (RapidAPI) aggregator: jsearch.p.rapidapi.com/search. Optional; needs JSEARCH_API_KEY in .env.

Unlike the ATS boards, JSearch returns jobs from any employer, including staffing agencies, so employers whose
names match `jsearch.exclude_employers` in config.yaml are dropped here. The LLM scorer's is_staffing_firm flag
is the second line of defense. Each query costs one API request (free plans are a few hundred per month).
"""
import os
import re

import httpx

from src.models import Job
from src.sources.common import parse_datetime

URL = "https://jsearch.p.rapidapi.com/search"
HOST = "jsearch.p.rapidapi.com"


def api_key() -> str | None:
    return os.environ.get("JSEARCH_API_KEY") or None


def parse(payload: dict, exclude_employers: list[str] | None = None) -> tuple[list[Job], int]:
    """-> (jobs, number dropped as staffing/excluded employers)"""
    patterns = [re.compile(p, re.I) for p in exclude_employers or []]
    jobs, dropped = [], 0
    for raw in payload.get("data") or []:
        employer = (raw.get("employer_name") or "").strip()
        if not employer or any(p.search(employer) for p in patterns):
            dropped += 1
            continue
        location = ", ".join(x for x in (raw.get("job_city"), raw.get("job_state"), raw.get("job_country")) if x)
        remote = bool(raw.get("job_is_remote"))
        if remote and "remote" not in location.lower():
            location = f"Remote - {location}" if location else "Remote"
        jobs.append(Job(
            id=f"jsearch:{raw['job_id']}",
            company=employer,
            title=(raw.get("job_title") or "").strip(),
            location=location,
            remote=remote,
            url=apply_url(raw),
            description=raw.get("job_description") or "",
            source="jsearch",
            # When the aggregator says the job was posted (it can't tell a re-post from a new post).
            posted_at=parse_datetime(raw.get("job_posted_at_datetime_utc")),
            employment_type=raw.get("job_employment_type"),
        ))
    return jobs, dropped


ATS_HOSTS = ("greenhouse.io", "lever.co", "ashbyhq.com")


def apply_url(raw: dict) -> str:
    """Prefer a direct Greenhouse/Lever/Ashby apply link (the forms Phase 5 can pre-fill) over the aggregator's
    default link, which is often LinkedIn or Indeed (never automated)."""
    for opt in raw.get("apply_options") or []:
        link = opt.get("apply_link") or ""
        if any(h in link for h in ATS_HOSTS):
            return link
    return raw.get("job_apply_link") or ""


def requests_planned(cfg: dict) -> list[tuple[str, dict]]:
    """Every (query, location) pair = one API request."""
    locations = cfg.get("locations") or [{"name": "US", "query": ""}]
    return [(q, loc) for q in cfg["queries"] for loc in locations]


async def fetch(client: httpx.AsyncClient, cfg: dict, key: str) -> tuple[list[Job], int, list[str]]:
    """Run every configured query in every configured location. -> (jobs, dropped, errors)"""
    jobs, dropped, errors = [], 0, []
    headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": HOST}
    for query, loc in requests_planned(cfg):
        text = f"{query} {loc.get('query', '')}".strip()
        params = {"query": text, "page": 1, "num_pages": cfg.get("pages_per_query", 1),
                  "country": "us", "date_posted": cfg.get("date_posted", "week")}
        if loc.get("remote"):
            params["work_from_home"] = "true"
        try:
            resp = await client.get(URL, params=params, headers=headers)
            resp.raise_for_status()
            got, d = parse(resp.json(), cfg.get("exclude_employers"))
            jobs += got
            dropped += d
        except Exception as e:  # noqa: BLE001 — one failed query shouldn't sink the run
            errors.append(f"jsearch '{text}': {type(e).__name__}: {e}")
    return jobs, dropped, errors
