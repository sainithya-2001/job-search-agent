"""JSearch (RapidAPI) aggregator: jsearch.p.rapidapi.com/search-v2. Optional; needs JSEARCH_API_KEY in .env.

The old /search endpoint was retired (404 "Endpoint '/search' does not exist"). /search-v2 returns jobs under
data.jobs and pages with a cursor (data.cursor, null on the last page); each page of up to 10 jobs is one request.

Unlike the ATS boards, JSearch returns jobs from any employer, including staffing agencies, so employers whose
names match `jsearch.exclude_employers` in config.yaml are dropped here. The LLM scorer's is_staffing_firm flag
is the second line of defense. Each query costs one API request (free plans are a few hundred per month).
"""
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from src.models import Job
from src.sources.common import parse_datetime

URL = "https://jsearch.p.rapidapi.com/search-v2"
HOST = "jsearch.p.rapidapi.com"


def api_key() -> str | None:
    return os.environ.get("JSEARCH_API_KEY") or None


class JSearchFormatError(ValueError):
    pass


def jobs_of(payload: dict) -> list[dict]:
    """/search-v2 shape: {"data": {"jobs": [...], "cursor": ...}}. Anything else is an error, never "no jobs"."""
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        raise JSearchFormatError(f"unexpected response shape: data is {type(data).__name__}")
    return data["jobs"]


def next_cursor(payload: dict) -> str | None:
    return (payload.get("data") or {}).get("cursor") or None


def parse(payload: dict, exclude_employers: list[str] | None = None) -> tuple[list[Job], int]:
    """-> (jobs, number dropped as staffing/excluded employers)"""
    patterns = [re.compile(p, re.I) for p in exclude_employers or []]
    jobs, dropped = [], 0
    for raw in jobs_of(payload):
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
    """The first page of every (query, location) pair: one API request each. Extra pages come later, only if the
    budget allows."""
    locations = cfg.get("locations") or [{"name": "US", "query": ""}]
    return [(q, loc) for q in cfg["queries"] for loc in locations]


# ---- request budget ------------------------------------------------------------------------------------------
# The free plan is a HARD 200 requests/month. Every request is written to the jsearch_requests table before it is
# sent, and this budget is recomputed from that table each run:
#   per day   <= jsearch.max_requests_per_day   (7)
#   per month <= jsearch.monthly_soft_limit     (160: JSearch stops for the month here)
#   per month <  jsearch.monthly_hard_limit     (200: never reached, the soft limit comes first)
# RapidAPI also reports the provider's own count (x-ratelimit-requests-remaining). When it is known for this month
# it is trusted too, so requests made outside this app (e.g. the RapidAPI playground) still count.

@dataclass
class Budget:
    allowed: int               # requests this run may send
    used_today: int
    used_month: int
    provider_remaining: int | None
    reason: str                # what limits `allowed`


def _starts(now: datetime) -> tuple[str, str]:
    day = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return day.isoformat(), day.replace(day=1).isoformat()


def budget(cfg: dict, store, now: datetime | None = None) -> Budget:
    now = now or datetime.now(timezone.utc)
    day_start, month_start = _starts(now)
    used_today, used_month = store.jsearch_used(day_start), store.jsearch_used(month_start)
    daily, soft, hard = cfg["max_requests_per_day"], cfg["monthly_soft_limit"], cfg["monthly_hard_limit"]
    if soft > hard:
        raise ValueError("jsearch.monthly_soft_limit must not exceed monthly_hard_limit")
    limits = {f"daily budget ({used_today}/{daily} used today)": daily - used_today,
              f"monthly limit ({used_month}/{soft} used this month)": soft - used_month}
    remaining = store.jsearch_last_quota(month_start)
    if remaining is not None:      # provider's count: keep (hard - soft) requests of headroom on THEIR count too
        limits[f"RapidAPI reports {remaining} of {hard} left"] = remaining - (hard - soft)
    reason, allowed = min(limits.items(), key=lambda kv: kv[1])
    return Budget(max(0, allowed), used_today, used_month, remaining, reason)


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


async def fetch(client: httpx.AsyncClient, cfg: dict, key: str, store,
                now: datetime | None = None) -> tuple[list[Job], int, list[str], dict]:
    """First page of every (query, location) in priority order, then page 2, 3, ... of the pairs whose previous
    page came back full, while the budget lasts. One request = one page (num_pages is always 1, so the provider
    never bills a request as several). -> (jobs, dropped, errors, stats)"""
    b = budget(cfg, store, now)
    jobs, dropped, errors = [], 0, []
    stats = {"budget": b, "sent": 0, "pages": Counter(), "stopped": None}
    headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": HOST}
    max_pages, full_page = cfg.get("max_pages_per_query", 1), cfg.get("full_page_size", 10)

    pending = [(q, loc, 1, None) for q, loc in requests_planned(cfg)]
    while pending:
        if stats["sent"] >= b.allowed:
            stats["stopped"] = f"budget reached: {b.reason}"
            break
        query, loc, page, cursor = pending.pop(0)
        text = f"{query} {loc.get('query', '')}".strip()
        params = {"query": text, "page": page, "num_pages": 1, "country": "us",
                  "date_posted": cfg.get("date_posted", "3days")}
        if loc.get("remote"):
            params["work_from_home"] = "true"
        if cursor:
            params["cursor"] = cursor
        row_id = store.jsearch_sent(text, loc.get("name"), page, (now or datetime.now(timezone.utc)).isoformat())
        stats["sent"] += 1
        status = results = remaining = limit = None
        try:
            resp = await client.get(URL, params=params, headers=headers)
            status = resp.status_code
            remaining = _int(resp.headers.get("x-ratelimit-requests-remaining"))
            limit = _int(resp.headers.get("x-ratelimit-requests-limit"))
            if 400 <= status < 500:   # bad key, wrong endpoint/params, or quota: every further request fails too
                errors.append(f"jsearch '{text}' p{page}: HTTP {status} {resp.text[:150]!r}; "
                              f"JSearch stopped for this run")
                stats["stopped"] = f"HTTP {status}"
                break
            resp.raise_for_status()
            payload = resp.json()
            try:
                results = len(jobs_of(payload))
            except JSearchFormatError as e:   # the API changed shape: stop rather than spend more requests
                errors.append(f"jsearch '{text}' p{page}: {e}; JSearch stopped for this run")
                stats["stopped"] = "unexpected response shape"
                break
            got, d = parse(payload, cfg.get("exclude_employers"))
            jobs += got
            dropped += d
            stats["pages"][page] += 1
            more = next_cursor(payload)
            if more and results >= full_page and page < max_pages:   # a full page with a cursor: there is more
                pending.append((query, loc, page + 1, more))
        except Exception as e:  # noqa: BLE001 — one failed query shouldn't sink the run
            errors.append(f"jsearch '{text}' p{page}: {type(e).__name__}: {e}")
        finally:
            store.jsearch_done(row_id, status, results, remaining, limit)
    return jobs, dropped, errors, stats
