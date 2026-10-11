"""Greenhouse public job board API: boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"""
import httpx

from src.models import Job
from src.sources.common import html_to_text, parse_datetime

URL = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"


def parse(company: str, slug: str, payload: dict) -> list[Job]:
    jobs = []
    for raw in payload.get("jobs", []):
        location = (raw.get("location") or {}).get("name") or ""
        jobs.append(Job(
            id=f"greenhouse:{slug}:{raw['id']}",
            company=company,
            title=raw.get("title", "").strip(),
            location=location,
            # Greenhouse has no structured remote flag; the location string is the only signal.
            remote="remote" in location.lower(),
            url=raw.get("absolute_url", ""),
            description=html_to_text(raw.get("content")),
            source="greenhouse",
            posted_at=parse_datetime(raw.get("first_published") or raw.get("updated_at")),
        ))
    return jobs


async def fetch(client: httpx.AsyncClient, company: str, slug: str) -> list[Job]:
    resp = await client.get(URL.format(slug=slug))
    resp.raise_for_status()
    return parse(company, slug, resp.json())
