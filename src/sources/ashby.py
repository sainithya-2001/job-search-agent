"""Ashby public job board API: api.ashbyhq.com/posting-api/job-board/{slug}"""
import httpx

from src.models import Job
from src.sources.common import html_to_text, parse_datetime

URL = "https://api.ashbyhq.com/posting-api/job-board/{slug}"


def parse(company: str, slug: str, payload: dict) -> list[Job]:
    jobs = []
    for raw in payload.get("jobs", []):
        if raw.get("isListed") is False:
            continue
        locations = [raw.get("location") or ""]
        locations += [s.get("location", "") for s in raw.get("secondaryLocations") or []]
        location = "; ".join(loc for loc in locations if loc)
        jobs.append(Job(
            id=f"ashby:{slug}:{raw['id']}",
            company=company,
            title=raw.get("title", "").strip(),
            location=location,
            remote=bool(raw.get("isRemote")) or raw.get("workplaceType") == "Remote",
            url=raw.get("jobUrl", ""),
            description=raw.get("descriptionPlain") or html_to_text(raw.get("descriptionHtml")),
            source="ashby",
            posted_at=parse_datetime(raw.get("publishedAt")),
            employment_type=raw.get("employmentType"),
        ))
    return jobs


async def fetch(client: httpx.AsyncClient, company: str, slug: str) -> list[Job]:
    resp = await client.get(URL.format(slug=slug))
    resp.raise_for_status()
    return parse(company, slug, resp.json())
