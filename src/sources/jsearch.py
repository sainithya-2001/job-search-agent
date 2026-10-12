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
from urllib.parse import urlparse

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


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def host_of(url: str | None) -> str:
    return (urlparse(url or "").hostname or "").lower().removeprefix("www.")


def _host_in(host: str, hosts) -> bool:
    return any(host == h or host.endswith("." + h) for h in hosts)


def is_excluded_publisher(raw: dict, cfg: dict) -> bool:
    """Repost/aggregator sites (config jsearch.exclude_publishers): by publisher name or by the main link's host."""
    pub = _norm(raw.get("job_publisher"))
    if pub and any(_norm(n) and _norm(n) in pub for n in cfg.get("exclude_publishers", [])):
        return True
    return _host_in(host_of(raw.get("job_apply_link")), cfg.get("aggregator_hosts", []))


def direct_link(raw: dict, cfg: dict) -> tuple[str | None, str]:
    """The employer's own apply link, if JSearch gives one. -> (link or None, how it was recognized)

    In order: a Greenhouse/Lever/Ashby link (review mode can pre-fill those); another employer ATS
    (config jsearch.employer_ats_hosts, e.g. Workday); a link on the employer's own website domain; any option
    JSearch flags is_direct, unless it is on a job board (LinkedIn, Indeed, aggregators: never "direct").
    """
    opts = [{"apply_link": raw.get("job_apply_link"), "publisher": raw.get("job_publisher"),
             "is_direct": raw.get("job_apply_is_direct")}] + list(raw.get("apply_options") or [])
    links = [(o.get("apply_link") or "", bool(o.get("is_direct"))) for o in opts if o.get("apply_link")]
    boards = list(cfg.get("aggregator_hosts", [])) + list(cfg.get("job_board_hosts", []))
    for link, _ in links:
        if _host_in(host_of(link), ATS_HOSTS):
            return link, "greenhouse/lever/ashby"
    for link, _ in links:
        if _host_in(host_of(link), cfg.get("employer_ats_hosts", [])):
            return link, f"employer ATS ({host_of(link)})"
    site = host_of(raw.get("employer_website"))
    if site and not _host_in(site, boards):
        root = ".".join(site.split(".")[-2:])
        for link, _ in links:
            if _host_in(host_of(link), [root]):
                return link, "employer website"
    for link, direct in links:
        if direct and not _host_in(host_of(link), boards):
            return link, f"is_direct ({host_of(link)})"
    return None, "no direct employer link"


def classify_stored_url(url: str | None, cfg: dict) -> tuple[bool, str]:
    """For JSearch rows saved before direct links were tracked: judge the saved url alone (its other apply
    options weren't kept). A job-board/aggregator host = not direct."""
    host = host_of(url)
    if _host_in(host, ATS_HOSTS):
        return True, "greenhouse/lever/ashby"
    if not host or _host_in(host, list(cfg.get("aggregator_hosts", [])) + list(cfg.get("job_board_hosts", []))):
        return False, f"no direct employer link (saved link: {host or 'none'})"
    return True, f"employer link (saved link: {host})"


def parse(payload: dict, cfg: dict) -> tuple[list[Job], Counter]:
    """-> (jobs, dropped counts: 'staffing/excluded employer', 'repost/aggregator site')"""
    patterns = [re.compile(p, re.I) for p in cfg.get("exclude_employers") or []]
    jobs, dropped = [], Counter()
    for raw in jobs_of(payload):
        employer = (raw.get("employer_name") or "").strip()
        if not employer or any(p.search(employer) for p in patterns):
            dropped["staffing/excluded employer"] += 1
            continue
        if is_excluded_publisher(raw, cfg):
            dropped["repost/aggregator site"] += 1
            continue
        location = ", ".join(x for x in (raw.get("job_city"), raw.get("job_state"), raw.get("job_country")) if x)
        remote = bool(raw.get("job_is_remote"))
        if remote and "remote" not in location.lower():
            location = f"Remote - {location}" if location else "Remote"
        link, how = direct_link(raw, cfg)
        jobs.append(Job(
            id=f"jsearch:{raw['job_id']}",
            company=employer,
            title=(raw.get("job_title") or "").strip(),
            location=location,
            remote=remote,
            # The employer's link when there is one; otherwise the publisher's, kept only for reference.
            url=link or raw.get("job_apply_link") or "",
            description=raw.get("job_description") or "",
            source="jsearch",
            # When the aggregator says the job was posted (it can't tell a re-post from a new post).
            posted_at=parse_datetime(raw.get("job_posted_at_datetime_utc")),
            employment_type=raw.get("job_employment_type"),
            apply_direct=link is not None,
            apply_via=how,
        ))
    return jobs, dropped


ATS_HOSTS = ("greenhouse.io", "lever.co", "ashbyhq.com")


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
                now: datetime | None = None) -> tuple[list[Job], Counter, list[str], dict]:
    """First page of every (query, location) in priority order, then page 2, 3, ... of the pairs whose previous
    page came back full, while the budget lasts. One request = one page (num_pages is always 1, so the provider
    never bills a request as several). -> (jobs, dropped, errors, stats)"""
    b = budget(cfg, store, now)
    jobs, dropped, errors = [], Counter(), []
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
        if cfg.get("exclude_publishers"):          # server-side; parse() re-checks in case names differ
            params["exclude_job_publishers"] = ",".join(cfg["exclude_publishers"])
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
            got, d = parse(payload, cfg)
            jobs += got
            dropped.update(d)
            stats["pages"][page] += 1
            more = next_cursor(payload)
            if more and results >= full_page and page < max_pages:   # a full page with a cursor: there is more
                pending.append((query, loc, page + 1, more))
        except Exception as e:  # noqa: BLE001 — one failed query shouldn't sink the run
            errors.append(f"jsearch '{text}' p{page}: {type(e).__name__}: {e}")
        finally:
            store.jsearch_done(row_id, status, results, remaining, limit)
    return jobs, dropped, errors, stats
